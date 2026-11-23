# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
kappa-Regularized Orbital-Optimized MP2 (kappa-OOMP2, Lee & Head-Gordon 2018) untuk RHF & UHF.
Meredam divergensi celah HOMO-LUMO kecil melalui faktor f(Delta) = 1 - exp(-kappa * Delta),
dengan gradien orbital analitik eksak (termasuk limit L'Hopital aman-degenerasi) di atas
kernel C++20 `mshqc.kernels` tanpa menambah 1 baris pun kode C++.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple
import numpy as np

from .._mshqc import kernels as K
from .omp_hybrid import HybridOMPResult, optimize_orbitals_rhf, optimize_orbitals_uhf
from .tensors import t2_from_4d


def _kappa_factors(delta: np.ndarray, kappa: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mengembalikan (g(Delta), -g'(Delta)) untuk g(Delta) = (1 - exp(-kappa*Delta))^2 / Delta
    secara stabil numerik hingga Delta -> 0 (limit analitik: g -> kappa^2*Delta, -g' -> -kappa^2).
    """
    k = float(kappa)
    if k <= 0.0:
        raise ValueError("Parameter regularisasi kappa harus > 0.")

    x = k * delta
    small = np.abs(delta) < 1.0e-7
    safe_d = np.where(small, 1.0, delta)
    safe_x = np.where(small, 0.0, x)

    f = -np.expm1(-safe_x)
    exp_neg = np.exp(-safe_x)

    g_reg = (f * f) / safe_d
    neg_gp_reg = (f * f) / (safe_d * safe_d) - (2.0 * k * exp_neg * f) / safe_d

    g_lim = (k * k) * delta * (1.0 - x)
    neg_gp_lim = -(k * k) * (1.0 - 2.0 * x)

    g = np.where(small, g_lim, g_reg)
    neg_gp = np.where(small, neg_gp_lim, neg_gp_reg)
    return g, neg_gp


def _assemble_subspace_1rdm(
    A_mat: np.ndarray,
    deg_mat: np.ndarray,
    eps: np.ndarray,
) -> np.ndarray:
    """Merakit 1-RDM subruang (OO atau VV) kontinu melintasi orbital degenerasi."""
    diff = eps[:, None] - eps[None, :]
    near_deg = np.abs(diff) <= 1.0e-7
    safe_diff = np.where(near_deg, 1.0, diff)
    gamma = np.where(near_deg, deg_mat, (A_mat - A_mat.T) / safe_diff)
    return 0.5 * (gamma + gamma.T)


def _kappa_omp2_rhf_callback(kappa: float, c_os: float, c_ss: float):
    c_same = c_os + c_ss

    def corr_fn(B_oo, B_ov, B_vv, eps_o, eps_v):
        no, nv = eps_o.size, eps_v.size
        if no == 0 or nv == 0:
            return 0.0, 0.0, np.zeros((no, no)), np.zeros((nv, nv)), np.zeros((nv, no))

        V = (B_ov @ B_ov.T).reshape(no, nv, no, nv).transpose(0, 2, 1, 3)
        V_tilde = c_same * V - c_ss * V.transpose(0, 1, 3, 2)

        Delta = (
            eps_v[None, None, :, None]
            + eps_v[None, None, None, :]
            - eps_o[:, None, None, None]
            - eps_o[None, :, None, None]
        )
        g_val, neg_gp = _kappa_factors(Delta, kappa)

        e_corr = -float(np.sum(V * V_tilde * g_val))
        T_eff = -V * g_val
        T_tilde_eff = -V_tilde * g_val

        A_oo = np.einsum("ijab,kjab->ik", T_tilde_eff, V, optimize=True)
        deg_oo = -np.einsum("ijab,kjab,ijab->ik", V, V_tilde, neg_gp, optimize=True)
        g_oo = _assemble_subspace_1rdm(A_oo, deg_oo, eps_o)

        A_vv = np.einsum("ijab,ijcb->ac", T_tilde_eff, V, optimize=True)
        deg_vv = np.einsum("ijab,ijcb,ijab->ac", V, V_tilde, neg_gp, optimize=True)
        g_vv = _assemble_subspace_1rdm(A_vv, deg_vv, eps_v)

        T2_mat = t2_from_4d(T_eff)
        Z_vo = K.eval_omp2_zvec_rhf_df(T2_mat, B_oo, B_ov, B_vv, no, nv, c_os=c_os, c_ss=c_ss)
        return e_corr, 0.0, g_oo, g_vv, Z_vo

    return corr_fn


def _kappa_omp2_uhf_callback(kappa: float, c_os: float, c_ss: float):
    def corr_fn(B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b, eps_oa, eps_va, eps_ob, eps_vb):
        na, va = eps_oa.size, eps_va.size
        nb, vb = eps_ob.size, eps_vb.size

        e_aa = e_bb = e_ab = 0.0
        A_oo_a = np.zeros((na, na), dtype=np.float64)
        deg_oo_a = np.zeros((na, na), dtype=np.float64)
        A_vv_a = np.zeros((va, va), dtype=np.float64)
        deg_vv_a = np.zeros((va, va), dtype=np.float64)

        A_oo_b = np.zeros((nb, nb), dtype=np.float64)
        deg_oo_b = np.zeros((nb, nb), dtype=np.float64)
        A_vv_b = np.zeros((vb, vb), dtype=np.float64)
        deg_vv_b = np.zeros((vb, vb), dtype=np.float64)

        T2_aa_4d = np.zeros((na, na, va, va), dtype=np.float64)
        T2_bb_4d = np.zeros((nb, nb, vb, vb), dtype=np.float64)
        T2_ab_4d = np.zeros((na, nb, va, vb), dtype=np.float64)

        if na > 1 and va > 0 and c_ss != 0.0:
            V_aa = (B_ov_a @ B_ov_a.T).reshape(na, va, na, va).transpose(0, 2, 1, 3)
            V_aa = V_aa - V_aa.transpose(0, 1, 3, 2)
            D_aa = (
                eps_va[None, None, :, None] + eps_va[None, None, None, :]
                - eps_oa[:, None, None, None] - eps_oa[None, :, None, None]
            )
            g_aa, neg_gp_aa = _kappa_factors(D_aa, kappa)
            e_aa = -0.25 * float(np.sum((V_aa * V_aa) * g_aa))
            T2_aa_4d = -V_aa * g_aa

            A_oo_a += 0.5 * c_ss * np.einsum("ijab,kjab->ik", T2_aa_4d, V_aa, optimize=True)
            deg_oo_a -= 0.5 * c_ss * np.einsum("ijab,kjab,ijab->ik", V_aa, V_aa, neg_gp_aa, optimize=True)
            A_vv_a += 0.5 * c_ss * np.einsum("ijab,ijcb->ac", T2_aa_4d, V_aa, optimize=True)
            deg_vv_a += 0.5 * c_ss * np.einsum("ijab,ijcb,ijab->ac", V_aa, V_aa, neg_gp_aa, optimize=True)

        if nb > 1 and vb > 0 and c_ss != 0.0:
            V_bb = (B_ov_b @ B_ov_b.T).reshape(nb, vb, nb, vb).transpose(0, 2, 1, 3)
            V_bb = V_bb - V_bb.transpose(0, 1, 3, 2)
            D_bb = (
                eps_vb[None, None, :, None] + eps_vb[None, None, None, :]
                - eps_ob[:, None, None, None] - eps_ob[None, :, None, None]
            )
            g_bb, neg_gp_bb = _kappa_factors(D_bb, kappa)
            e_bb = -0.25 * float(np.sum((V_bb * V_bb) * g_bb))
            T2_bb_4d = -V_bb * g_bb

            A_oo_b += 0.5 * c_ss * np.einsum("ijab,kjab->ik", T2_bb_4d, V_bb, optimize=True)
            deg_oo_b -= 0.5 * c_ss * np.einsum("ijab,kjab,ijab->ik", V_bb, V_bb, neg_gp_bb, optimize=True)
            A_vv_b += 0.5 * c_ss * np.einsum("ijab,ijcb->ac", T2_bb_4d, V_bb, optimize=True)
            deg_vv_b += 0.5 * c_ss * np.einsum("ijab,ijcb,ijab->ac", V_bb, V_bb, neg_gp_bb, optimize=True)

        if na > 0 and nb > 0 and va > 0 and vb > 0 and c_os != 0.0:
            V_ab = (B_ov_a @ B_ov_b.T).reshape(na, va, nb, vb).transpose(0, 2, 1, 3)
            D_ab = (
                eps_va[None, None, :, None] + eps_vb[None, None, None, :]
                - eps_oa[:, None, None, None] - eps_ob[None, :, None, None]
            )
            g_ab, neg_gp_ab = _kappa_factors(D_ab, kappa)
            e_ab = -float(np.sum((V_ab * V_ab) * g_ab))
            T2_ab_4d = -V_ab * g_ab

            A_oo_a += c_os * np.einsum("ijab,kjab->ik", T2_ab_4d, V_ab, optimize=True)
            deg_oo_a -= c_os * np.einsum("ijab,kjab,ijab->ik", V_ab, V_ab, neg_gp_ab, optimize=True)
            A_vv_a += c_os * np.einsum("ijab,ijcb->ac", T2_ab_4d, V_ab, optimize=True)
            deg_vv_a += c_os * np.einsum("ijab,ijcb,ijab->ac", V_ab, V_ab, neg_gp_ab, optimize=True)

            A_oo_b += c_os * np.einsum("ijab,ikab->jk", T2_ab_4d, V_ab, optimize=True)
            deg_oo_b -= c_os * np.einsum("ijab,ikab,ijab->jk", V_ab, V_ab, neg_gp_ab, optimize=True)
            A_vv_b += c_os * np.einsum("ijab,ijac->bc", T2_ab_4d, V_ab, optimize=True)
            deg_vv_b += c_os * np.einsum("ijab,ijac,ijab->bc", V_ab, V_ab, neg_gp_ab, optimize=True)

        g_oo_a = _assemble_subspace_1rdm(A_oo_a, deg_oo_a, eps_oa) if na > 0 else A_oo_a
        g_vv_a = _assemble_subspace_1rdm(A_vv_a, deg_vv_a, eps_va) if va > 0 else A_vv_a
        g_oo_b = _assemble_subspace_1rdm(A_oo_b, deg_oo_b, eps_ob) if nb > 0 else A_oo_b
        g_vv_b = _assemble_subspace_1rdm(A_vv_b, deg_vv_b, eps_vb) if vb > 0 else A_vv_b

        Za, Zb = K.eval_omp2_zvec_uhf_df(
            t2_from_4d(T2_aa_4d), t2_from_4d(T2_bb_4d), t2_from_4d(T2_ab_4d),
            B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b,
            na, va, nb, vb, c_os=c_os, c_ss=c_ss,
        )
        e_corr = c_ss * (e_aa + e_bb) + c_os * e_ab
        return e_corr, 0.0, g_oo_a, g_vv_a, g_oo_b, g_vv_b, Za, Zb

    return corr_fn


def run_df_kappa_omp2_rhf(
    H_core: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    C0: np.ndarray,
    n_occ: int,
    kappa: float = 1.45,
    c_os: float = 1.0,
    c_ss: float = 1.0,
    grad_tol: float = 1e-5,
    energy_tol: float = 1e-9,
    max_iter: int = 40,
    n_frozen: int = 0,
    callback: Optional[Callable[[dict], None]] = None,
) -> HybridOMPResult:
    """Jalankan DF/Cholesky kappa-OOMP2 RHF (default kappa = 1.45 Eh^-1)."""
    return optimize_orbitals_rhf(
        H_core, v_nn, B_ao, C0, n_occ,
        corr_fn=_kappa_omp2_rhf_callback(kappa=kappa, c_os=c_os, c_ss=c_ss),
        method_name=f"kappa-omp2({kappa:.2f})",
        grad_tol=grad_tol, energy_tol=energy_tol, max_iter=max_iter,
        n_frozen=n_frozen, callback=callback,
    )


def run_df_kappa_omp2_uhf(
    H_core: np.ndarray,
    S_ovlp: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    C0_a: np.ndarray,
    C0_b: np.ndarray,
    na: int,
    nb: int,
    kappa: float = 1.45,
    c_os: float = 1.0,
    c_ss: float = 1.0,
    grad_tol: float = 1e-5,
    energy_tol: float = 1e-9,
    max_iter: int = 40,
    n_frozen: int = 0,
    callback: Optional[Callable[[dict], None]] = None,
) -> HybridOMPResult:
    """Jalankan DF/Cholesky kappa-OOMP2 UHF (radikal kulit terbuka, default kappa = 1.45 Eh^-1)."""
    return optimize_orbitals_uhf(
        H_core, S_ovlp, v_nn, B_ao, C0_a, C0_b, na, nb,
        corr_fn=_kappa_omp2_uhf_callback(kappa=kappa, c_os=c_os, c_ss=c_ss),
        method_name=f"kappa-omp2({kappa:.2f})",
        grad_tol=grad_tol, energy_tol=energy_tol, max_iter=max_iter,
        n_frozen=n_frozen, callback=callback,
    )
