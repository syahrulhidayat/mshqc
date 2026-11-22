# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""Pengoptimal orbital Matrix-Free Steihaug-Toint Trust-Region SOSCF (RHF & UHF)."""

from typing import Tuple
import numpy as np
from .._mshqc import kernels as K


def solve_tr_soscf_rhf(
    w_vo: np.ndarray,
    eps_occ: np.ndarray,
    eps_vir: np.ndarray,
    B_oo: np.ndarray,
    B_ov: np.ndarray,
    B_vv: np.ndarray,
    trust_radius: float = 0.25,
    max_cg: int = 15,
) -> np.ndarray:
    """Preconditioned Steihaug-CG untuk langkah rotasi orbital RHF memakai MO-DF Hessian."""
    nv, no = w_vo.shape
    grad_norm = float(np.linalg.norm(w_vo))
    if grad_norm == 0.0:
        return np.zeros_like(w_vo)

    level_shift = 0.05 if grad_norm > 0.1 else 0.005
    J_ia = np.sum(B_ov ** 2, axis=1).reshape(no, nv).T
    H_diag = 4.0 * np.maximum(np.abs(eps_vir[:, None] - eps_occ[None, :]), 1e-4) + 8.0 * J_ia + level_shift

    x = np.zeros_like(w_vo)
    r = -w_vo.copy()
    z = r / H_diag
    p = z.copy()
    rz = float(np.sum(r * z))
    tol = min(1e-4, 0.1 * grad_norm)

    for _ in range(max_cg):
        if float(np.linalg.norm(r)) < tol:
            break
        Hp = K.apply_mo_df_hessian_rhf(np.ascontiguousarray(p), eps_occ, eps_vir, B_oo, B_ov, B_vv)
        pHp = float(np.sum(p * Hp))
        if pHp <= 1e-12:
            x = -w_vo / H_diag
            break
        alpha = rz / pHp
        x_next = x + alpha * p
        if float(np.linalg.norm(x_next)) >= trust_radius:
            x = x_next * (trust_radius / float(np.linalg.norm(x_next)))
            break
        x = x_next
        r = r - alpha * Hp
        z = r / H_diag
        rz_new = float(np.sum(r * z))
        p = z + (rz_new / max(rz, 1e-30)) * p
        rz = rz_new

    nx = float(np.linalg.norm(x))
    if nx > trust_radius:
        x *= trust_radius / nx
    return x


def solve_tr_soscf_uhf(
    w_vo_a: np.ndarray,
    w_vo_b: np.ndarray,
    eps_oa: np.ndarray,
    eps_va: np.ndarray,
    eps_ob: np.ndarray,
    eps_vb: np.ndarray,
    B_oo_a: np.ndarray,
    B_ov_a: np.ndarray,
    B_vv_a: np.ndarray,
    B_oo_b: np.ndarray,
    B_ov_b: np.ndarray,
    B_vv_b: np.ndarray,
    trust_radius: float = 0.25,
    max_cg: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Preconditioned Steihaug-CG terkopel (alpha + beta) untuk langkah TR-SOSCF UHF."""
    va, na = w_vo_a.shape
    vb, nb = w_vo_b.shape
    grad_norm = float(np.sqrt(np.sum(w_vo_a ** 2) + np.sum(w_vo_b ** 2)))
    if grad_norm == 0.0:
        return np.zeros_like(w_vo_a), np.zeros_like(w_vo_b)

    level_shift = 0.05 if grad_norm > 0.1 else 0.005
    J_a = np.sum(B_ov_a ** 2, axis=1).reshape(na, va).T if (na > 0 and va > 0) else np.zeros((va, na))
    J_b = np.sum(B_ov_b ** 2, axis=1).reshape(nb, vb).T if (nb > 0 and vb > 0) else np.zeros((vb, nb))
    H_da = 2.0 * np.maximum(np.abs(eps_va[:, None] - eps_oa[None, :]), 1e-4) + 4.0 * J_a + level_shift
    H_db = 2.0 * np.maximum(np.abs(eps_vb[:, None] - eps_ob[None, :]), 1e-4) + 4.0 * J_b + level_shift

    xa, xb = np.zeros_like(w_vo_a), np.zeros_like(w_vo_b)
    ra, rb = -w_vo_a.copy(), -w_vo_b.copy()
    za, zb = ra / H_da, rb / H_db
    pa, pb = za.copy(), zb.copy()
    rz = float(np.sum(ra * za) + np.sum(rb * zb))
    tol = min(1e-4, 0.1 * grad_norm)

    for _ in range(max_cg):
        if float(np.sqrt(np.sum(ra ** 2) + np.sum(rb ** 2))) < tol:
            break
        Hpa, Hpb = K.apply_mo_df_hessian_uhf(
            np.ascontiguousarray(pa), np.ascontiguousarray(pb),
            eps_oa, eps_va, eps_ob, eps_vb,
            B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b,
        )
        pHp = float(np.sum(pa * Hpa) + np.sum(pb * Hpb))
        if pHp <= 1e-12:
            xa, xb = -w_vo_a / H_da, -w_vo_b / H_db
            break
        alpha = rz / pHp
        xa_next, xb_next = xa + alpha * pa, xb + alpha * pb
        n_next = float(np.sqrt(np.sum(xa_next ** 2) + np.sum(xb_next ** 2)))
        if n_next >= trust_radius:
            scale = trust_radius / n_next
            xa, xb = xa_next * scale, xb_next * scale
            break
        xa, xb = xa_next, xb_next
        ra, rb = ra - alpha * Hpa, rb - alpha * Hpb
        za, zb = ra / H_da, rb / H_db
        rz_new = float(np.sum(ra * za) + np.sum(rb * zb))
        beta = rz_new / max(rz, 1e-30)
        pa, pb = za + beta * pa, zb + beta * pb
        rz = rz_new

    nx = float(np.sqrt(np.sum(xa ** 2) + np.sum(xb ** 2)))
    if nx > trust_radius:
        scale = trust_radius / nx
        xa, xb = xa * scale, xb * scale
    return xa, xb
