# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
Gradien nuklir analitik Thin-Core di Python untuk:
  - DF-RHF, DF-UHF, DF-ROHF
  - DF-OMP2, DF-SCS-OMP2, DF-SOS-OMP2 (RHF & UHF)
Seluruh logika densitas 1-RDM/2-RDM ditulis bersih di NumPy/BLAS, sedangkan kontraksi
turunan integral `libcint` didelegasikan langsung ke C++20 `K.eval_df_nuclear_gradient`.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from .._mshqc import kernels as K
from .omp_hybrid import HybridOMPResult, run_df_omp_rhf, run_df_omp_uhf
from .scf_hybrid import HybridSCFResult, build_sad_guess, run_df_rhf, run_df_rohf, run_df_uhf
from .tensors import t2_to_4d


def _assemble_scf_gamma_3c(
    B_ao: np.ndarray,
    P_eff_ao: np.ndarray,
    C_occ_a: np.ndarray,
    nbf: int,
    is_rhf: bool,
    C_occ_b: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bangun densitas 2-elektron terkopel Gamma_3c (nbf^2 x n_aux) dan gamma_2c (n_aux x n_aux)."""
    n_aux = B_ao.shape[1]
    c_vec = K.contract_ao_sym(B_ao, P_eff_ao, nbf)
    Gamma_3c = P_eff_ao.reshape(-1, 1) * c_vec[None, :]

    na = C_occ_a.shape[1]
    if na > 0:
        B_oo_a = K.transform_3c_mo(B_ao, C_occ_a, C_occ_a, nbf)
        B_oo_a_3d = B_oo_a.T.reshape(n_aux, na, na)
        K_3c_a = ((C_occ_a @ B_oo_a_3d) @ C_occ_a.T).reshape(n_aux, nbf * nbf).T
        Gamma_3c -= (2.0 if is_rhf else 1.0) * K_3c_a

    if not is_rhf and C_occ_b is not None and C_occ_b.shape[1] > 0:
        nb = C_occ_b.shape[1]
        B_oo_b = K.transform_3c_mo(B_ao, C_occ_b, C_occ_b, nbf)
        B_oo_b_3d = B_oo_b.T.reshape(n_aux, nb, nb)
        K_3c_b = ((C_occ_b @ B_oo_b_3d) @ C_occ_b.T).reshape(n_aux, nbf * nbf).T
        Gamma_3c -= K_3c_b

    Gamma_3c = np.ascontiguousarray(Gamma_3c, dtype=np.float64)
    gamma_2c = K.compute_gamma2c(Gamma_3c, B_ao, nbf)
    return Gamma_3c, gamma_2c


def eval_df_scf_gradient_rhf(
    mol,
    primary_basis,
    aux_basis,
    scf_res: HybridSCFResult,
    cutoff: float = 1.0e-12,
) -> np.ndarray:
    """Evaluasi gradien nuklir analitik DF-RHF (natoms x 3, Ha/Bohr)."""
    nbf = int(primary_basis.n_basis_functions())
    na = int(scf_res.n_occ_alpha)
    C_occ = np.ascontiguousarray(scf_res.C_alpha[:, :na], dtype=np.float64)
    eps_o = np.asarray(scf_res.orbital_energies_alpha[:na], dtype=np.float64)

    P_eff = np.ascontiguousarray(2.0 * (C_occ @ C_occ.T), dtype=np.float64)
    W_eff = np.ascontiguousarray(2.0 * ((C_occ * eps_o[None, :]) @ C_occ.T), dtype=np.float64)

    Gamma_3c, gamma_2c = _assemble_scf_gamma_3c(scf_res.L_mat, P_eff, C_occ, nbf, is_rhf=True)
    return K.eval_df_nuclear_gradient(
        mol, primary_basis, aux_basis, P_eff, W_eff, Gamma_3c, gamma_2c, cutoff
    )


def eval_df_scf_gradient_uhf(
    mol,
    primary_basis,
    aux_basis,
    scf_res: HybridSCFResult,
    cutoff: float = 1.0e-12,
) -> np.ndarray:
    """Evaluasi gradien nuklir analitik DF-UHF (natoms x 3, Ha/Bohr)."""
    nbf = int(primary_basis.n_basis_functions())
    na, nb = int(scf_res.n_occ_alpha), int(scf_res.n_occ_beta)
    Ca_o = np.ascontiguousarray(scf_res.C_alpha[:, :na], dtype=np.float64)
    Cb_o = np.ascontiguousarray(scf_res.C_beta[:, :nb], dtype=np.float64)
    eps_oa = np.asarray(scf_res.orbital_energies_alpha[:na], dtype=np.float64)
    eps_ob = np.asarray(scf_res.orbital_energies_beta[:nb], dtype=np.float64)

    P_eff = Ca_o @ Ca_o.T if na > 0 else np.zeros((nbf, nbf), dtype=np.float64)
    W_eff = (Ca_o * eps_oa[None, :]) @ Ca_o.T if na > 0 else np.zeros((nbf, nbf), dtype=np.float64)
    if nb > 0:
        P_eff = P_eff + (Cb_o @ Cb_o.T)
        W_eff = W_eff + ((Cb_o * eps_ob[None, :]) @ Cb_o.T)

    P_eff = np.ascontiguousarray(P_eff, dtype=np.float64)
    W_eff = np.ascontiguousarray(W_eff, dtype=np.float64)

    Gamma_3c, gamma_2c = _assemble_scf_gamma_3c(
        scf_res.L_mat, P_eff, Ca_o, nbf, is_rhf=False, C_occ_b=Cb_o
    )
    return K.eval_df_nuclear_gradient(
        mol, primary_basis, aux_basis, P_eff, W_eff, Gamma_3c, gamma_2c, cutoff
    )


def eval_df_scf_gradient_rohf(
    mol,
    primary_basis,
    aux_basis,
    scf_res: HybridSCFResult,
    H_core: Optional[np.ndarray] = None,
    cutoff: float = 1.0e-12,
) -> np.ndarray:
    """Evaluasi gradien nuklir analitik DF-ROHF (natoms x 3, Ha/Bohr)."""
    nbf = int(primary_basis.n_basis_functions())
    na, nb = int(scf_res.n_occ_alpha), int(scf_res.n_occ_beta)
    C = np.ascontiguousarray(scf_res.C_alpha, dtype=np.float64)
    Ca_o = np.ascontiguousarray(C[:, :na])
    Cb_o = np.ascontiguousarray(C[:, :nb])

    if scf_res.F_alpha is not None and scf_res.F_beta is not None:
        Fa, Fb = scf_res.F_alpha, scf_res.F_beta
    else:
        if H_core is None:
            import mshqc
            ints = mshqc.IntegralEngine(mol, primary_basis)
            H_core = ints.compute_kinetic() + ints.compute_nuclear()
        J, Ka, Kb = K.build_jk_3c_uhf(scf_res.L_mat, Ca_o, Cb_o, nbf)
        Fa, Fb = H_core + J - Ka, H_core + J - Kb

    Pa = Ca_o @ Ca_o.T if na > 0 else np.zeros((nbf, nbf), dtype=np.float64)
    Pb = Cb_o @ Cb_o.T if nb > 0 else np.zeros((nbf, nbf), dtype=np.float64)
    P_eff = np.ascontiguousarray(Pa + Pb, dtype=np.float64)

    Fa_mo = C.T @ Fa @ C
    Fb_mo = C.T @ Fb @ C
    n_c = nb
    n_o = na - nb
    W_mo = np.zeros((nbf, nbf), dtype=np.float64)
    if n_c > 0:
        W_mo[:n_c, :n_c] = Fa_mo[:n_c, :n_c] + Fb_mo[:n_c, :n_c]
    if n_o > 0:
        W_mo[n_c:na, n_c:na] = Fa_mo[n_c:na, n_c:na]
    if n_c > 0 and n_o > 0:
        W_co = Fa_mo[:n_c, n_c:na] + 0.5 * Fb_mo[:n_c, n_c:na]
        W_mo[:n_c, n_c:na] = W_co
        W_mo[n_c:na, :n_c] = W_co.T
    W_eff = np.ascontiguousarray(C @ W_mo @ C.T, dtype=np.float64)

    Gamma_3c, gamma_2c = _assemble_scf_gamma_3c(
        scf_res.L_mat, P_eff, Ca_o, nbf, is_rhf=False, C_occ_b=Cb_o
    )
    return K.eval_df_nuclear_gradient(
        mol, primary_basis, aux_basis, P_eff, W_eff, Gamma_3c, gamma_2c, cutoff
    )


def _build_omp2_f_blocks(
    F_HF_mo: np.ndarray,
    G_gamma_mo: np.ndarray,
    g_oo: np.ndarray,
    g_vv: np.ndarray,
    B_oo_3d: np.ndarray,
    B_ov_3d: np.ndarray,
    B_vv_3d: np.ndarray,
    X_3d: np.ndarray,
    no: int,
) -> np.ndarray:
    """Bangun matriks Lagrangian MO tergeneralisasi simetris (F_1e_sep + F_conn) untuk DF-OMP2."""
    nbf = F_HF_mo.shape[0]
    F_sep = np.zeros((nbf, nbf), dtype=np.float64)
    F_sep[:no, :no] = F_HF_mo[:no, :no] + F_HF_mo[:no, :no] @ g_oo + G_gamma_mo[:no, :no]
    F_sep[no:, no:] = F_HF_mo[no:, no:] @ g_vv
    F_sep[no:, :no] = F_HF_mo[no:, :no] + F_HF_mo[no:, :no] @ g_oo + G_gamma_mo[no:, :no]
    F_sep[:no, no:] = F_HF_mo[:no, no:] @ g_vv

    F_conn = np.zeros((nbf, nbf), dtype=np.float64)
    F_conn[:no, :no] = np.einsum("Pia,Pja->ij", B_ov_3d, X_3d, optimize=True)
    F_conn[no:, no:] = np.einsum("Pia,Pib->ab", B_ov_3d, X_3d, optimize=True)
    F_conn[no:, :no] = np.einsum("Pab,Pib->ai", B_vv_3d, X_3d, optimize=True)
    F_conn[:no, no:] = np.einsum("Pij,Pja->ia", B_oo_3d, X_3d, optimize=True)

    F_sep = 0.5 * (F_sep + F_sep.T)
    F_conn = 0.5 * (F_conn + F_conn.T)
    return F_sep + F_conn


def eval_df_omp2_gradient_rhf(
    mol,
    primary_basis,
    aux_basis,
    H_core: np.ndarray,
    B_ao: np.ndarray,
    omp_res: HybridOMPResult,
    c_os: float = 1.0,
    c_ss: float = 1.0,
    cutoff: float = 1.0e-12,
) -> np.ndarray:
    """
    Evaluasi gradien nuklir analitik DF-OMP2 / SCS-OMP2 / SOS-OMP2 (RHF, all-electron).
    Identik secara matematis dengan `OMP2::build_df_gradient_data()` di C++.
    """
    if getattr(omp_res, "n_frozen", 0) > 0:
        raise ValueError("Gradien analitik DF-OMP2 mensyaratkan all-electron (n_frozen == 0).")

    nbf = int(primary_basis.n_basis_functions())
    no = int(omp_res.n_occ_alpha)
    nv = nbf - no
    n_aux = B_ao.shape[1]

    C_in = np.ascontiguousarray(omp_res.C_alpha, dtype=np.float64)
    C_occ_in = np.ascontiguousarray(C_in[:, :no])
    J, K_mat = K.build_jk_3c_rhf(B_ao, C_occ_in, nbf)
    F_ao = H_core + J - K_mat

    C, eps, _ = K.semicanonicalize_subspaces(C_in, F_ao, no)
    C_o = np.ascontiguousarray(C[:, :no])
    C_v = np.ascontiguousarray(C[:, no:])
    B_oo, B_ov, B_vv = K.transform_3c_mo_blocks(B_ao, C_o, C_v, nbf)

    _, _, T2 = K.eval_rmp2_df(B_ov, eps[:no], eps[no:])
    g_oo, g_vv = K.eval_omp2_1rdm_rhf(T2, no, nv, c_os=c_os, c_ss=c_ss)

    # Amplitudo efektif T_eff -> X_a (no*nv x n_aux) dalam 1 BLAS DGEMM
    T2_4d = t2_to_4d(T2, no, no, nv, nv)
    T_eff = (c_os + c_ss) * T2_4d - c_ss * T2_4d.transpose(0, 1, 3, 2)
    X_mat = T_eff.transpose(0, 2, 1, 3).reshape(no * nv, no * nv) @ B_ov

    B_oo_3d = B_oo.T.reshape(n_aux, no, no)
    B_ov_3d = B_ov.T.reshape(n_aux, no, nv)
    B_vo_3d = B_ov_3d.transpose(0, 2, 1)
    B_vv_3d = B_vv.T.reshape(n_aux, nv, nv)
    X_3d = X_mat.T.reshape(n_aux, no, nv)
    X_vo_3d = X_3d.transpose(0, 2, 1)

    P_ref_a = C_o @ C_o.T
    P_corr_a = C_o @ g_oo @ C_o.T + C_v @ g_vv @ C_v.T
    P_ref_tot = 2.0 * P_ref_a
    P_corr_tot = 2.0 * P_corr_a
    P_eff_ao = np.ascontiguousarray(P_ref_tot + P_corr_tot, dtype=np.float64)

    G_gamma_ao = 2.0 * K.build_coulomb_3c(B_ao, P_corr_a, nbf) - K.build_exchange_3c_sym(B_ao, P_corr_a, nbf)
    F_tot_mo = _build_omp2_f_blocks(
        C.T @ F_ao @ C, C.T @ G_gamma_ao @ C, g_oo, g_vv, B_oo_3d, B_ov_3d, B_vv_3d, X_3d, no
    )
    W_eff_ao = np.ascontiguousarray(2.0 * (C @ F_tot_mo @ C.T), dtype=np.float64)

    # Perakitan Gamma_3c secara chunked (ramah RAM)
    c_ref = K.contract_ao_sym(B_ao, P_ref_tot, nbf)
    c_corr = K.contract_ao_sym(B_ao, P_corr_tot, nbf)
    Gamma_3c = np.empty((nbf * nbf, n_aux), dtype=np.float64)

    chunk = 64
    for p0 in range(0, n_aux, chunk):
        p1 = min(n_aux, p0 + chunk)
        bs = p1 - p0
        G_blk = (
            P_eff_ao[None, :, :] * c_ref[p0:p1, None, None]
            + P_ref_tot[None, :, :] * c_corr[p0:p1, None, None]
        )
        M_ref = (C_o @ B_oo_3d[p0:p1]) @ C_o.T
        M_corr = (C_o @ (g_oo @ B_oo_3d[p0:p1])) @ C_o.T + (C_v @ (g_vv @ B_vo_3d[p0:p1])) @ C_o.T
        G_blk -= 2.0 * (M_ref + M_corr + M_corr.swapaxes(1, 2))
        Conn = (C_v @ X_vo_3d[p0:p1]) @ C_o.T
        G_blk += 2.0 * (Conn + Conn.swapaxes(1, 2))
        Gamma_3c[:, p0:p1] = G_blk.reshape(bs, nbf * nbf).T

    gamma_2c = K.compute_gamma2c(Gamma_3c, B_ao, nbf)
    return K.eval_df_nuclear_gradient(
        mol, primary_basis, aux_basis, P_eff_ao, W_eff_ao, Gamma_3c, gamma_2c, cutoff
    )


def eval_df_omp2_gradient_uhf(
    mol,
    primary_basis,
    aux_basis,
    H_core: np.ndarray,
    B_ao: np.ndarray,
    omp_res: HybridOMPResult,
    c_os: float = 1.0,
    c_ss: float = 1.0,
    cutoff: float = 1.0e-12,
) -> np.ndarray:
    """
    Evaluasi gradien nuklir analitik DF-OMP2 / SCS-OMP2 / SOS-OMP2 (UHF, all-electron).
    Identik secara matematis dengan `OMP2::build_df_gradient_data()` di C++.
    """
    if getattr(omp_res, "n_frozen", 0) > 0:
        raise ValueError("Gradien analitik DF-OMP2 mensyaratkan all-electron (n_frozen == 0).")

    nbf = int(primary_basis.n_basis_functions())
    na, nb = int(omp_res.n_occ_alpha), int(omp_res.n_occ_beta)
    va, vb = nbf - na, nbf - nb
    n_aux = B_ao.shape[1]
    has_b = (nb > 0 and vb > 0)

    Ca_in = np.ascontiguousarray(omp_res.C_alpha, dtype=np.float64)
    Cb_in = np.ascontiguousarray(omp_res.C_beta, dtype=np.float64)
    J, Ka, Kb = K.build_jk_3c_uhf(B_ao, np.ascontiguousarray(Ca_in[:, :na]), np.ascontiguousarray(Cb_in[:, :nb]), nbf)
    Fa_ao = H_core + J - Ka
    Fb_ao = H_core + J - Kb

    Ca, eps_a, _ = K.semicanonicalize_subspaces(Ca_in, Fa_ao, na)
    Cb, eps_b, _ = K.semicanonicalize_subspaces(Cb_in, Fb_ao, nb)
    Ca_o, Ca_v = np.ascontiguousarray(Ca[:, :na]), np.ascontiguousarray(Ca[:, na:])
    Cb_o, Cb_v = np.ascontiguousarray(Cb[:, :nb]), np.ascontiguousarray(Cb[:, nb:])

    Boo_a, Bov_a, Bvv_a, Boo_b, Bov_b, Bvv_b = K.transform_3c_mo_blocks_uhf(
        B_ao, Ca_o, Ca_v, Cb_o, Cb_v, nbf
    )
    _, _, _, T2_aa, T2_bb, T2_ab = K.eval_ump2_df(
        Bov_a, Bov_b, eps_a[:na], eps_a[na:], eps_b[:nb], eps_b[nb:]
    )
    g_oo_a, g_vv_a, g_oo_b, g_vv_b = K.eval_omp2_1rdm_uhf(
        T2_aa, T2_bb, T2_ab, na, va, nb, vb, c_os=c_os, c_ss=c_ss
    )

    # Kontraksi 2-RDM terhubung X_a & X_b via BLAS DGEMM
    X_a = np.zeros((na * va, n_aux), dtype=np.float64)
    if na > 1 and va > 0:
        Taa_4d = t2_to_4d(T2_aa, na, na, va, va)
        X_a += c_ss * (Taa_4d.transpose(0, 2, 1, 3).reshape(na * va, na * va) @ Bov_a)
    if has_b and na > 0 and va > 0:
        Tab_4d = t2_to_4d(T2_ab, na, nb, va, vb)
        X_a += c_os * (Tab_4d.transpose(0, 2, 1, 3).reshape(na * va, nb * vb) @ Bov_b)

    X_b = np.zeros((nb * vb, n_aux), dtype=np.float64)
    if has_b:
        if nb > 1:
            Tbb_4d = t2_to_4d(T2_bb, nb, nb, vb, vb)
            X_b += c_ss * (Tbb_4d.transpose(0, 2, 1, 3).reshape(nb * vb, nb * vb) @ Bov_b)
        if na > 0 and va > 0:
            X_b += c_os * (Tab_4d.transpose(1, 3, 0, 2).reshape(nb * vb, na * va) @ Bov_a)

    Boo_a_3d = Boo_a.T.reshape(n_aux, na, na)
    Bov_a_3d = Bov_a.T.reshape(n_aux, na, va)
    Bvo_a_3d = Bov_a_3d.transpose(0, 2, 1)
    Bvv_a_3d = Bvv_a.T.reshape(n_aux, va, va)
    Xa_3d = X_a.T.reshape(n_aux, na, va)
    Xa_vo_3d = Xa_3d.transpose(0, 2, 1)

    P_ref_a = Ca_o @ Ca_o.T if na > 0 else np.zeros((nbf, nbf), dtype=np.float64)
    P_ref_b = Cb_o @ Cb_o.T if nb > 0 else np.zeros((nbf, nbf), dtype=np.float64)
    P_corr_a = Ca_o @ g_oo_a @ Ca_o.T + Ca_v @ g_vv_a @ Ca_v.T if (na > 0 and va > 0) else np.zeros((nbf, nbf))
    P_corr_b = Cb_o @ g_oo_b @ Cb_o.T + Cb_v @ g_vv_b @ Cb_v.T if has_b else np.zeros((nbf, nbf))

    P_ref_tot = P_ref_a + P_ref_b
    P_corr_tot = P_corr_a + P_corr_b
    P_eff_ao = np.ascontiguousarray(P_ref_tot + P_corr_tot, dtype=np.float64)

    J_corr = K.build_coulomb_3c(B_ao, P_corr_tot, nbf)
    Ga_gamma_ao = J_corr - K.build_exchange_3c_sym(B_ao, P_corr_a, nbf)
    Gb_gamma_ao = J_corr - K.build_exchange_3c_sym(B_ao, P_corr_b, nbf)

    Fa_tot_mo = _build_omp2_f_blocks(
        Ca.T @ Fa_ao @ Ca, Ca.T @ Ga_gamma_ao @ Ca, g_oo_a, g_vv_a, Boo_a_3d, Bov_a_3d, Bvv_a_3d, Xa_3d, na
    )
    W_eff_ao = Ca @ Fa_tot_mo @ Ca.T

    if has_b:
        Boo_b_3d = Boo_b.T.reshape(n_aux, nb, nb)
        Bov_b_3d = Bov_b.T.reshape(n_aux, nb, vb)
        Bvo_b_3d = Bov_b_3d.transpose(0, 2, 1)
        Bvv_b_3d = Bvv_b.T.reshape(n_aux, vb, vb)
        Xb_3d = X_b.T.reshape(n_aux, nb, vb)
        Xb_vo_3d = Xb_3d.transpose(0, 2, 1)
        Fb_tot_mo = _build_omp2_f_blocks(
            Cb.T @ Fb_ao @ Cb, Cb.T @ Gb_gamma_ao @ Cb, g_oo_b, g_vv_b, Boo_b_3d, Bov_b_3d, Bvv_b_3d, Xb_3d, nb
        )
        W_eff_ao = W_eff_ao + (Cb @ Fb_tot_mo @ Cb.T)

    W_eff_ao = np.ascontiguousarray(W_eff_ao, dtype=np.float64)

    c_ref = K.contract_ao_sym(B_ao, P_ref_tot, nbf)
    c_corr = K.contract_ao_sym(B_ao, P_corr_tot, nbf)
    Gamma_3c = np.empty((nbf * nbf, n_aux), dtype=np.float64)

    chunk = 64
    for p0 in range(0, n_aux, chunk):
        p1 = min(n_aux, p0 + chunk)
        bs = p1 - p0
        G_blk = (
            P_eff_ao[None, :, :] * c_ref[p0:p1, None, None]
            + P_ref_tot[None, :, :] * c_corr[p0:p1, None, None]
        )
        M_ref_a = (Ca_o @ Boo_a_3d[p0:p1]) @ Ca_o.T
        M_corr_a = (Ca_o @ (g_oo_a @ Boo_a_3d[p0:p1])) @ Ca_o.T + (Ca_v @ (g_vv_a @ Bvo_a_3d[p0:p1])) @ Ca_o.T
        G_blk -= M_ref_a + M_corr_a + M_corr_a.swapaxes(1, 2)
        Conn_a = (Ca_v @ Xa_vo_3d[p0:p1]) @ Ca_o.T
        G_blk += Conn_a + Conn_a.swapaxes(1, 2)

        if has_b:
            M_ref_b = (Cb_o @ Boo_b_3d[p0:p1]) @ Cb_o.T
            M_corr_b = (Cb_o @ (g_oo_b @ Boo_b_3d[p0:p1])) @ Cb_o.T + (Cb_v @ (g_vv_b @ Bvo_b_3d[p0:p1])) @ Cb_o.T
            G_blk -= M_ref_b + M_corr_b + M_corr_b.swapaxes(1, 2)
            Conn_b = (Cb_v @ Xb_vo_3d[p0:p1]) @ Cb_o.T
            G_blk += Conn_b + Conn_b.swapaxes(1, 2)

        Gamma_3c[:, p0:p1] = G_blk.reshape(bs, nbf * nbf).T

    gamma_2c = K.compute_gamma2c(Gamma_3c, B_ao, nbf)
    return K.eval_df_nuclear_gradient(
        mol, primary_basis, aux_basis, P_eff_ao, W_eff_ao, Gamma_3c, gamma_2c, cutoff
    )


def compute_df_gradient(
    mol,
    method: str = "omp2",
    basis_name: str = "cc-pVDZ",
    aux_basis_name: Optional[str] = None,
    reference: str = "auto",
    threshold: float = 1.0e-12,
    orb_grad_tol: float = 1.0e-10,
    basis_dir: str = "data/basis",
    warm_start: Optional[dict] = None,
) -> Tuple[float, np.ndarray]:
    """
    One-liner Thin-Core gradien nuklir analitik untuk DF-RHF, DF-UHF, DF-ROHF,
    DF-OMP2, DF-SCS-OMP2, dan DF-SOS-OMP2 (100% bebas legacy C++).
    """
    import mshqc

    m = method.strip().lower().replace("_", "-")
    if m.startswith("df-"):
        m = m[3:]

    aux_name = aux_basis_name or f"{basis_name}-RI"
    basis = mshqc.BasisSet(basis_name, mol, basis_dir)
    aux_basis = mshqc.BasisSet(aux_name, mol, basis_dir)
    ints = mshqc.IntegralEngine(mol, basis)

    H_core = ints.compute_kinetic() + ints.compute_nuclear()
    S_ovlp = ints.compute_overlap()
    v_nn = float(mol.nuclear_repulsion_energy())
    B_ao = K.compute_df_ao_factor(mol, basis, aux_basis, threshold, False, False)

    n_elec = int(mol.n_electrons())
    mult = int(mol.multiplicity())
    na = (n_elec + mult - 1) // 2
    nb = n_elec - na

    ref = reference.lower()
    if ref in ("", "auto"):
        if m in ("rhf", "uhf", "rohf"):
            ref = m
        else:
            ref = "rhf" if mult == 1 else "uhf"

    P_sad = build_sad_guess(mol, basis)
    ws = warm_start if warm_start is not None else {}
    c_init_a = ws.get("scf_C_alpha")
    c_init_b = ws.get("scf_C_beta")

    if ref == "rhf":
        scf_res = run_df_rhf(
            H_core, S_ovlp, v_nn, B_ao, na,
            energy_tol=threshold, density_tol=threshold,
            C_init=c_init_a, P_init=None if c_init_a is not None else P_sad,
        )
    elif ref == "rohf":
        scf_res = run_df_rohf(
            H_core, S_ovlp, v_nn, B_ao, na, nb,
            energy_tol=threshold, density_tol=threshold,
            C_init=c_init_a, P_init=None if c_init_a is not None else P_sad,
        )
    else:
        scf_res = run_df_uhf(
            H_core, S_ovlp, v_nn, B_ao, na, nb,
            energy_tol=threshold, density_tol=threshold,
            C_init_a=c_init_a, C_init_b=c_init_b,
            P_init=None if (c_init_a is not None and c_init_b is not None) else P_sad,
        )

    if warm_start is not None:
        warm_start["scf_C_alpha"] = scf_res.C_alpha
        warm_start["scf_C_beta"] = scf_res.C_beta

    if m in ("scf", "hf", "rhf", "uhf", "rohf"):
        if ref == "rhf":
            return scf_res.energy_total, eval_df_scf_gradient_rhf(mol, basis, aux_basis, scf_res, cutoff=threshold)
        if ref == "rohf":
            return scf_res.energy_total, eval_df_scf_gradient_rohf(
                mol, basis, aux_basis, scf_res, H_core=H_core, cutoff=threshold
            )
        return scf_res.energy_total, eval_df_scf_gradient_uhf(mol, basis, aux_basis, scf_res, cutoff=threshold)

    omp_params = {
        "omp2": (1.0, 1.0),
        "scs-omp2": (1.2, 1.0 / 3.0),
        "sos-omp2": (1.3, 0.0),
    }
    if m not in omp_params:
        raise ValueError(f"Gradien analitik DF belum mendukung metode: {method}")

    c_os, c_ss = omp_params[m]
    if ref == "rhf":
        omp_res = run_df_omp_rhf(
            H_core, v_nn, B_ao, scf_res.C_alpha, na,
            method_name=m, c_os=c_os, c_ss=c_ss,
            grad_tol=orb_grad_tol, energy_tol=threshold, max_iter=80,
        )
        grad = eval_df_omp2_gradient_rhf(
            mol, basis, aux_basis, H_core, B_ao, omp_res, c_os=c_os, c_ss=c_ss, cutoff=threshold
        )
        return omp_res.energy_total, grad

    omp_res = run_df_omp_uhf(
        H_core, S_ovlp, v_nn, B_ao, scf_res.C_alpha, scf_res.C_beta, na, nb,
        method_name=m, c_os=c_os, c_ss=c_ss,
        grad_tol=orb_grad_tol, energy_tol=threshold, max_iter=80,
    )
    grad = eval_df_omp2_gradient_uhf(
        mol, basis, aux_basis, H_core, B_ao, omp_res, c_os=c_os, c_ss=c_ss, cutoff=threshold
    )
    return omp_res.energy_total, grad


def compute_df_scf_gradient(
    mol,
    basis_name: str = "cc-pVDZ",
    aux_basis_name: Optional[str] = None,
    reference: str = "auto",
    threshold: float = 1.0e-12,
    basis_dir: str = "data/basis",
) -> Tuple[float, np.ndarray]:
    """Wrapper kompatibel Tahap 3 untuk gradien analitik DF-SCF (RHF/UHF/ROHF)."""
    return compute_df_gradient(
        mol, method="scf", basis_name=basis_name, aux_basis_name=aux_basis_name,
        reference=reference, threshold=threshold, basis_dir=basis_dir,
    )
