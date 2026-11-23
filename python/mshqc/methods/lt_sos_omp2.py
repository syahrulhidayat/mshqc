# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
O(N^4) Zero-T2 Laplace-Transformed SOS-OMP2 (RHF & UHF, all-electron & frozen-core).

Menghilangkan alokasi tensor amplitudo 4-indeks T2 O(O^2 V^2) dan menurunkan kompleksitas
komputasi dari O(N^5) menjadi O(n_q * O * V * N_aux^2) ~ O(N^4) melalui kuadratur Laplace
berbobot non-negatif (NNLS sparse minimax) yang mencocokkan 1/Delta dan 1/Delta^2 sekaligus.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple
import numpy as np
from scipy.optimize import nnls

from .kappa_omp2 import _assemble_subspace_1rdm
from .omp_hybrid import HybridOMPResult, optimize_orbitals_rhf, optimize_orbitals_uhf


def build_laplace_quadrature(
    delta_min: float,
    delta_max: float,
    n_grid: int = 220,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Membangun pasangan kuadratur Laplace (weights, nodes) positif untuk rentang [delta_min, delta_max]
    sehingga sum_q w_q exp(-t_q x) ~= 1/x DAN sum_q (w_q t_q) exp(-t_q x) ~= 1/x^2 secara simultan.
    Solusi NNLS secara otomatis bersifat sparse (hanya ~10-18 titik aktif, akurasi relatif < 1e-8).
    """
    d_min = max(float(delta_min), 1.0e-6)
    d_max = max(float(delta_max), d_min * 1.05)
    R = d_max / d_min

    # Diskretisasi log-spaced pada rentang ternormalisasi x in [0.90, 1.10 * R]
    x = np.geomspace(0.90, 1.10 * R, max(160, int(n_grid)))
    t_cand = np.geomspace(1.0e-3 / R, 36.0, max(180, int(n_grid)))

    E = np.exp(-np.outer(x, t_cand))
    A_val = x[:, None] * E
    b_val = np.ones_like(x)

    # Baris turunan berbobot memastikan 1/x^2 (untuk diagonal/degenerasi 1-RDM) sama akuratnya
    der_w = 0.25
    A_der = der_w * (x[:, None] ** 2) * (t_cand[None, :] * E)
    b_der = np.full_like(x, der_w)

    w_raw, _ = nnls(np.vstack((A_val, A_der)), np.concatenate((b_val, b_der)))
    keep = w_raw > 1.0e-14 * float(np.max(w_raw))

    weights = w_raw[keep] / d_min
    nodes = t_cand[keep] / d_min
    return np.ascontiguousarray(weights, dtype=np.float64), np.ascontiguousarray(nodes, dtype=np.float64)


def _assemble_lt_1rdm_and_zvec(
    B_oo: np.ndarray,
    B_ov: np.ndarray,
    B_vv: np.ndarray,
    X_conn: np.ndarray,
    X_deriv: np.ndarray,
    eps_o: np.ndarray,
    eps_v: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rakit (g_oo, g_vv, Z_vo) murni via Level-3 BLAS DGEMM berorde O(N^4) tanpa tensor 4D."""
    no, nv = eps_o.size, eps_v.size
    n_aux = B_ov.shape[1]
    if no == 0 or nv == 0 or n_aux == 0:
        return np.zeros((no, no)), np.zeros((nv, nv)), np.zeros((nv, no))

    Bov_3d = B_ov.T.reshape(n_aux, no, nv)
    Xc_3d = X_conn.T.reshape(n_aux, no, nv)
    Xd_3d = X_deriv.T.reshape(n_aux, no, nv)

    # 1-RDM subruang OO: O(no^2 * nv * n_aux)
    Bov_o = Bov_3d.transpose(1, 0, 2).reshape(no, n_aux * nv)
    Xc_o = Xc_3d.transpose(1, 0, 2).reshape(no, n_aux * nv)
    Xd_o = Xd_3d.transpose(1, 0, 2).reshape(no, n_aux * nv)
    A_oo = Xc_o @ Bov_o.T
    deg_oo = Xd_o @ Bov_o.T
    g_oo = _assemble_subspace_1rdm(A_oo, deg_oo, eps_o)

    # 1-RDM subruang VV: O(nv^2 * no * n_aux)
    Bov_v = Bov_3d.transpose(2, 0, 1).reshape(nv, n_aux * no)
    Xc_v = Xc_3d.transpose(2, 0, 1).reshape(nv, n_aux * no)
    Xd_v = Xd_3d.transpose(2, 0, 1).reshape(nv, n_aux * no)
    A_vv = Xc_v @ Bov_v.T
    deg_vv = -(Xd_v @ Bov_v.T)
    g_vv = _assemble_subspace_1rdm(A_vv, deg_vv, eps_v)

    # Z-vektor terhubung via 2 BLAS-3 DGEMM: O(n_aux * nv^2 * no + n_aux * nv * no^2)
    Bvv_3d = B_vv.T.reshape(n_aux, nv, nv)
    Boo_3d = B_oo.T.reshape(n_aux, no, no)
    Z1 = Bvv_3d.transpose(1, 0, 2).reshape(nv, n_aux * nv) @ Xc_3d.transpose(0, 2, 1).reshape(n_aux * nv, no)
    Z2 = Xc_v @ Boo_3d.reshape(n_aux * no, no)
    return g_oo, g_vv, Z1 - Z2


def _lt_sos_omp2_rhf_callback(c_os: float = 1.3):
    def corr_fn(B_oo, B_ov, B_vv, eps_o, eps_v):
        no, nv = eps_o.size, eps_v.size
        n_aux = B_ov.shape[1]
        if no == 0 or nv == 0 or n_aux == 0:
            return 0.0, 0.0, np.zeros((no, no)), np.zeros((nv, nv)), np.zeros((nv, no))

        d_ov = (eps_v[None, :] - eps_o[:, None]).reshape(-1)
        d_min = max(2.0 * float(np.min(d_ov)), 1.0e-5)
        d_max = max(2.0 * float(np.max(d_ov)), d_min * 1.1)
        weights, nodes = build_laplace_quadrature(d_min, d_max)

        e_sum = 0.0
        X_conn = np.zeros_like(B_ov)
        X_deriv = np.zeros_like(B_ov)

        for wq, tq in zip(weights, nodes):
            exp_half = np.exp(-0.5 * tq * d_ov)
            B_w = B_ov * exp_half[:, None]
            X_q = B_w.T @ B_w
            e_sum += wq * float(np.sum(X_q * X_q))

            Y_q = (B_w @ X_q) * exp_half[:, None]
            X_conn -= (c_os * wq) * Y_q
            X_deriv -= (c_os * wq * tq) * Y_q

        e_corr = -c_os * e_sum
        g_oo, g_vv, Z_vo = _assemble_lt_1rdm_and_zvec(B_oo, B_ov, B_vv, X_conn, X_deriv, eps_o, eps_v)
        return e_corr, 0.0, g_oo, g_vv, Z_vo

    return corr_fn


def _lt_sos_omp2_uhf_callback(c_os: float = 1.3):
    def corr_fn(B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b, eps_oa, eps_va, eps_ob, eps_vb):
        na, va = eps_oa.size, eps_va.size
        nb, vb = eps_ob.size, eps_vb.size
        n_aux = B_ov_a.shape[1]
        if na == 0 or va == 0 or nb == 0 or vb == 0 or n_aux == 0:
            return (
                0.0, 0.0,
                np.zeros((na, na)), np.zeros((va, va)),
                np.zeros((nb, nb)), np.zeros((vb, vb)),
                np.zeros((va, na)), np.zeros((vb, nb)),
            )

        da = (eps_va[None, :] - eps_oa[:, None]).reshape(-1)
        db = (eps_vb[None, :] - eps_ob[:, None]).reshape(-1)
        d_min = max(float(np.min(da)) + float(np.min(db)), 1.0e-5)
        d_max = max(float(np.max(da)) + float(np.max(db)), d_min * 1.1)
        weights, nodes = build_laplace_quadrature(d_min, d_max)

        e_sum = 0.0
        Xc_a = np.zeros_like(B_ov_a)
        Xd_a = np.zeros_like(B_ov_a)
        Xc_b = np.zeros_like(B_ov_b)
        Xd_b = np.zeros_like(B_ov_b)

        for wq, tq in zip(weights, nodes):
            ea_half = np.exp(-0.5 * tq * da)
            eb_half = np.exp(-0.5 * tq * db)
            Ba_w = B_ov_a * ea_half[:, None]
            Bb_w = B_ov_b * eb_half[:, None]

            Xa_q = Ba_w.T @ Ba_w
            Xb_q = Bb_w.T @ Bb_w
            e_sum += wq * float(np.sum(Xa_q * Xb_q))

            Ya_q = (Ba_w @ Xb_q) * ea_half[:, None]
            Yb_q = (Bb_w @ Xa_q) * eb_half[:, None]
            Xc_a -= (c_os * wq) * Ya_q
            Xd_a -= (c_os * wq * tq) * Ya_q
            Xc_b -= (c_os * wq) * Yb_q
            Xd_b -= (c_os * wq * tq) * Yb_q

        e_corr = -c_os * e_sum
        g_oo_a, g_vv_a, Za = _assemble_lt_1rdm_and_zvec(B_oo_a, B_ov_a, B_vv_a, Xc_a, Xd_a, eps_oa, eps_va)
        g_oo_b, g_vv_b, Zb = _assemble_lt_1rdm_and_zvec(B_oo_b, B_ov_b, B_vv_b, Xc_b, Xd_b, eps_ob, eps_vb)
        return e_corr, 0.0, g_oo_a, g_vv_a, g_oo_b, g_vv_b, Za, Zb

    return corr_fn


def run_df_lt_sos_omp2_rhf(
    H_core: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    C0: np.ndarray,
    n_occ: int,
    c_os: float = 1.3,
    grad_tol: float = 1e-5,
    energy_tol: float = 1e-9,
    max_iter: int = 40,
    n_frozen: int = 0,
    callback: Optional[Callable[[dict], None]] = None,
) -> HybridOMPResult:
    """Jalankan O(N^4) Zero-T2 LT-SOS-OMP2 RHF (default c_os = 1.3)."""
    return optimize_orbitals_rhf(
        H_core, v_nn, B_ao, C0, n_occ,
        corr_fn=_lt_sos_omp2_rhf_callback(c_os=c_os),
        method_name="lt-sos-omp2",
        grad_tol=grad_tol, energy_tol=energy_tol, max_iter=max_iter,
        n_frozen=n_frozen, callback=callback,
    )


def run_df_lt_sos_omp2_uhf(
    H_core: np.ndarray,
    S_ovlp: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    C0_a: np.ndarray,
    C0_b: np.ndarray,
    na: int,
    nb: int,
    c_os: float = 1.3,
    grad_tol: float = 1e-5,
    energy_tol: float = 1e-9,
    max_iter: int = 40,
    n_frozen: int = 0,
    callback: Optional[Callable[[dict], None]] = None,
) -> HybridOMPResult:
    """Jalankan O(N^4) Zero-T2 LT-SOS-OMP2 UHF (radikal kulit terbuka, default c_os = 1.3)."""
    return optimize_orbitals_uhf(
        H_core, S_ovlp, v_nn, B_ao, C0_a, C0_b, na, nb,
        corr_fn=_lt_sos_omp2_uhf_callback(c_os=c_os),
        method_name="lt-sos-omp2",
        grad_tol=grad_tol, energy_tol=energy_tol, max_iter=max_iter,
        n_frozen=n_frozen, callback=callback,
    )
