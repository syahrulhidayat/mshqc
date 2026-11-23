# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
Gradien nuklir analitik Thin-Core di Python (mendukung All-Electron & Strict Frozen-Core) untuk:
  - DF-RHF, DF-UHF, DF-ROHF
  - DF-OMP2, DF-SCS-OMP2, DF-SOS-OMP2, DF-LT-SOS-OMP2, dan DF-kappa-OOMP2 (RHF & UHF)
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from .._mshqc import kernels as K
from .kappa_omp2 import (
    _eval_kappa_omp2_rhf_data,
    _eval_kappa_omp2_uhf_data,
    run_df_kappa_omp2_rhf,
    run_df_kappa_omp2_uhf,
)
from .omp_hybrid import HybridOMPResult, run_df_omp_rhf, run_df_omp_uhf, semicanonicalize_subspaces_frozen
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


def _build_omp2_f_raw(
    F_HF_mo: np.ndarray,
    G_gamma_mo: np.ndarray,
    g_oo: np.ndarray,
    g_vv: np.ndarray,
    B_oo_3d: np.ndarray,
    B_ov_3d: np.ndarray,
    B_vv_3d: np.ndarray,
    X_3d: np.ndarray,
    no: int,
    nf: int = 0,
    B_fa_3d: Optional[np.ndarray] = None,
    B_fi_3d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Bangun matriks Lagrangian MO mentah F_raw[p, q] = (1/4) dE/dU_{pq} (RHF) atau (1/2) dE/dU_{pq} (UHF),
    termasuk blok frozen-core F_conn[:nf, nf:no] dan F_conn[:nf, no:].
    """
    nbf = F_HF_mo.shape[0]
    F_sep = np.zeros((nbf, nbf), dtype=np.float64)
    F_sep[:no, :no] = F_HF_mo[:no, :no] + G_gamma_mo[:no, :no]
    F_sep[:no, nf:no] += F_HF_mo[:no, nf:no] @ g_oo
    F_sep[no:, no:] = F_HF_mo[no:, no:] @ g_vv
    F_sep[no:, :no] = F_HF_mo[no:, :no] + G_gamma_mo[no:, :no]
    F_sep[no:, nf:no] += F_HF_mo[no:, nf:no] @ g_oo
    F_sep[:no, no:] = F_HF_mo[:no, no:] @ g_vv

    F_conn = np.zeros((nbf, nbf), dtype=np.float64)
    F_conn[nf:no, nf:no] = np.einsum("Pia,Pja->ij", B_ov_3d, X_3d, optimize=True)
    F_conn[no:, no:] = np.einsum("Pia,Pib->ab", B_ov_3d, X_3d, optimize=True)
    F_conn[no:, nf:no] = np.einsum("Pab,Pib->ai", B_vv_3d, X_3d, optimize=True)
    F_conn[nf:no, no:] = np.einsum("Pij,Pja->ia", B_oo_3d, X_3d, optimize=True)
    if nf > 0 and B_fa_3d is not None and B_fi_3d is not None:
        F_conn[:nf, nf:no] = np.einsum("Pfa,Pia->fi", B_fa_3d, X_3d, optimize=True)
        F_conn[:nf, no:] = np.einsum("Pfi,Pia->fa", B_fi_3d, X_3d, optimize=True)

    return F_sep + F_conn


def _solve_rhf_fc_zvec(
    B_ao: np.ndarray,
    S_ovlp: np.ndarray,
    C0: np.ndarray,
    eps0: np.ndarray,
    C_omp: np.ndarray,
    g_pf_omp: np.ndarray,
    no: int,
    nf: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Handy-Schaefer Z-vector eksak untuk Strict Frozen-Core RHF-OMP2 / kappa-OOMP2:
      1. Transformasi gradien non-variasional ke basis kanonik RHF: b0 = Q_nf @ g_pf @ Q_f^T.
      2. Solusi closed-form inti-aktif: Z0_{i0, f} = -b0_{i0, f} / (eps0_{i0} - eps0_f).
      3. Solusi CPHF RHF penuh (A_RHF * Z0_{vo} = -(b0_{vo} + G[P_ca]_{vo})).
    Mengembalikan (Z0_sym_mo, Pz_alpha_ao, dF0_sym_mo).
    """
    nbf = C0.shape[0]
    n_act = no - nf
    nv = nbf - no

    Q_f = C0[:, :nf].T @ S_ovlp @ C_omp[:, :nf]
    Q_nf = C0[:, nf:].T @ S_ovlp @ C_omp[:, nf:]
    b0 = Q_nf @ g_pf_omp @ Q_f.T  # shape: (n_act + nv, nf)

    # 1. Closed-form core-active Z-vector (P_RHF invarian thd rotasi core-active)
    Z0_ca = -b0[:n_act, :] / (eps0[nf:no, None] - eps0[None, :nf])
    P_ca_a = C0[:, nf:no] @ Z0_ca @ C0[:, :nf].T + C0[:, :nf] @ Z0_ca.T @ C0[:, nf:no].T
    G_ca_ao = 2.0 * K.build_coulomb_3c(B_ao, P_ca_a, nbf) - K.build_exchange_3c_sym(B_ao, P_ca_a, nbf)

    # 2. RHS CPHF RHF pada seluruh ruang virtual x occupied (nv x no)
    rhs_vo = -(C0[:, no:].T @ G_ca_ao @ C0[:, :no])
    rhs_vo[:, :nf] -= b0[n_act:, :]

    delta_vo = eps0[no:, None] - eps0[None, :no]
    C0_o, C0_v = np.ascontiguousarray(C0[:, :no]), np.ascontiguousarray(C0[:, no:])

    def matvec(z_flat: np.ndarray) -> np.ndarray:
        Z = z_flat.reshape(nv, no)
        Pz = C0_v @ Z @ C0_o.T + C0_o @ Z.T @ C0_v.T
        Gz = 2.0 * K.build_coulomb_3c(B_ao, Pz, nbf) - K.build_exchange_3c_sym(B_ao, Pz, nbf)
        return (delta_vo * Z + C0_v.T @ Gz @ C0_o).reshape(-1)

    b = rhs_vo.reshape(-1)
    precond = 1.0 / np.maximum(np.abs(delta_vo.reshape(-1)), 1.0e-3)
    x = np.zeros_like(b)
    r = b.copy()
    z = precond * r
    p = z.copy()
    rz = float(np.dot(r, z))
    for _ in range(100):
        if float(np.linalg.norm(r)) < 1.0e-13:
            break
        Ap = matvec(p)
        alpha = rz / max(float(np.dot(p, Ap)), 1.0e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        z = precond * r
        rz_new = float(np.dot(r, z))
        p = z + (rz_new / max(rz, 1.0e-30)) * p
        rz = rz_new
    Z0_vo = x.reshape(nv, no)

    # 3. Rakit Z0_sym (nbf x nbf), Pz_alpha_ao, dan dF0_sym (kontribusi ke W_eff)
    Z0_sym = np.zeros((nbf, nbf), dtype=np.float64)
    Z0_sym[no:, :no] = Z0_vo
    Z0_sym[:no, no:] = Z0_vo.T
    Z0_sym[nf:no, :nf] = Z0_ca
    Z0_sym[:nf, nf:no] = Z0_ca.T
    Pz_a = np.ascontiguousarray(C0 @ Z0_sym @ C0.T, dtype=np.float64)

    dF_orb = 0.5 * (eps0[:, None] + eps0[None, :]) * Z0_sym
    Gz_tot_ao = 2.0 * K.build_coulomb_3c(B_ao, Pz_a, nbf) - K.build_exchange_3c_sym(B_ao, Pz_a, nbf)
    dF_dens_raw = np.zeros((nbf, nbf), dtype=np.float64)
    dF_dens_raw[:, :no] = C0.T @ Gz_tot_ao @ C0_o
    dF0_sym = dF_orb + 0.5 * (dF_dens_raw + dF_dens_raw.T)
    return Z0_sym, Pz_a, dF0_sym


def _solve_uhf_fc_zvec(
    B_ao: np.ndarray,
    S_ovlp: np.ndarray,
    Ca0: np.ndarray,
    Cb0: np.ndarray,
    ea0: np.ndarray,
    eb0: np.ndarray,
    Ca_omp: np.ndarray,
    Cb_omp: np.ndarray,
    ga_pf_omp: np.ndarray,
    gb_pf_omp: np.ndarray,
    na: int,
    nb: int,
    nf: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Handy-Schaefer Z-vector terkopel (alpha + beta) untuk Strict Frozen-Core UHF-OMP2 / kappa-OOMP2."""
    nbf = Ca0.shape[0]
    na_act, nb_act = na - nf, nb - nf
    va, vb = nbf - na, nbf - nb

    Qa_f = Ca0[:, :nf].T @ S_ovlp @ Ca_omp[:, :nf]
    Qa_nf = Ca0[:, nf:].T @ S_ovlp @ Ca_omp[:, nf:]
    ba0 = Qa_nf @ ga_pf_omp @ Qa_f.T

    Qb_f = Cb0[:, :nf].T @ S_ovlp @ Cb_omp[:, :nf]
    Qb_nf = Cb0[:, nf:].T @ S_ovlp @ Cb_omp[:, nf:]
    bb0 = Qb_nf @ gb_pf_omp @ Qb_f.T

    Za0_ca = -ba0[:na_act, :] / (ea0[nf:na, None] - ea0[None, :nf]) if na_act > 0 else np.zeros((0, nf))
    Zb0_ca = -bb0[:nb_act, :] / (eb0[nf:nb, None] - eb0[None, :nf]) if nb_act > 0 else np.zeros((0, nf))

    Pca_a = Ca0[:, nf:na] @ Za0_ca @ Ca0[:, :nf].T + Ca0[:, :nf] @ Za0_ca.T @ Ca0[:, nf:na].T if na_act > 0 else np.zeros((nbf, nbf))
    Pca_b = Cb0[:, nf:nb] @ Zb0_ca @ Cb0[:, :nf].T + Cb0[:, :nf] @ Zb0_ca.T @ Cb0[:, nf:nb].T if nb_act > 0 else np.zeros((nbf, nbf))
    J_ca = K.build_coulomb_3c(B_ao, Pca_a + Pca_b, nbf)
    Ga_ca = J_ca - K.build_exchange_3c_sym(B_ao, Pca_a, nbf)
    Gb_ca = J_ca - K.build_exchange_3c_sym(B_ao, Pca_b, nbf)

    Ca0_o, Ca0_v = np.ascontiguousarray(Ca0[:, :na]), np.ascontiguousarray(Ca0[:, na:])
    Cb0_o, Cb0_v = np.ascontiguousarray(Cb0[:, :nb]), np.ascontiguousarray(Cb0[:, nb:])

    rhs_a = -(Ca0_v.T @ Ga_ca @ Ca0_o)
    rhs_a[:, :nf] -= ba0[na_act:, :]
    rhs_b = -(Cb0_v.T @ Gb_ca @ Cb0_o)
    rhs_b[:, :nf] -= bb0[nb_act:, :]

    da_vo = ea0[na:, None] - ea0[None, :na]
    db_vo = eb0[nb:, None] - eb0[None, :nb]
    dim_a = va * na

    def matvec(vec: np.ndarray) -> np.ndarray:
        Za = vec[:dim_a].reshape(va, na)
        Zb = vec[dim_a:].reshape(vb, nb)
        Pza = Ca0_v @ Za @ Ca0_o.T + Ca0_o @ Za.T @ Ca0_v.T
        Pzb = Cb0_v @ Zb @ Cb0_o.T + Cb0_o @ Zb.T @ Cb0_v.T
        Jz = K.build_coulomb_3c(B_ao, Pza + Pzb, nbf)
        Gza = Jz - K.build_exchange_3c_sym(B_ao, Pza, nbf)
        Gzb = Jz - K.build_exchange_3c_sym(B_ao, Pzb, nbf)
        oa = da_vo * Za + Ca0_v.T @ Gza @ Ca0_o
        ob = db_vo * Zb + Cb0_v.T @ Gzb @ Cb0_o
        return np.concatenate((oa.reshape(-1), ob.reshape(-1)))

    b = np.concatenate((rhs_a.reshape(-1), rhs_b.reshape(-1)))
    precond = 1.0 / np.maximum(np.abs(np.concatenate((da_vo.reshape(-1), db_vo.reshape(-1)))), 1.0e-3)
    x = np.zeros_like(b)
    r = b.copy()
    z = precond * r
    p = z.copy()
    rz = float(np.dot(r, z))
    for _ in range(100):
        if float(np.linalg.norm(r)) < 1.0e-13:
            break
        Ap = matvec(p)
        alpha = rz / max(float(np.dot(p, Ap)), 1.0e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        z = precond * r
        rz_new = float(np.dot(r, z))
        p = z + (rz_new / max(rz, 1.0e-30)) * p
        rz = rz_new

    Za0_vo = x[:dim_a].reshape(va, na)
    Zb0_vo = x[dim_a:].reshape(vb, nb)

    Za0_sym = np.zeros((nbf, nbf), dtype=np.float64)
    Za0_sym[na:, :na] = Za0_vo
    Za0_sym[:na, na:] = Za0_vo.T
    Za0_sym[nf:na, :nf] = Za0_ca
    Za0_sym[:nf, nf:na] = Za0_ca.T

    Zb0_sym = np.zeros((nbf, nbf), dtype=np.float64)
    Zb0_sym[nb:, :nb] = Zb0_vo
    Zb0_sym[:nb, nb:] = Zb0_vo.T
    Zb0_sym[nf:nb, :nf] = Zb0_ca
    Zb0_sym[:nf, nf:nb] = Zb0_ca.T

    Pz_a = np.ascontiguousarray(Ca0 @ Za0_sym @ Ca0.T, dtype=np.float64)
    Pz_b = np.ascontiguousarray(Cb0 @ Zb0_sym @ Cb0.T, dtype=np.float64)

    Jz_tot = K.build_coulomb_3c(B_ao, Pz_a + Pz_b, nbf)
    Gza_tot = Jz_tot - K.build_exchange_3c_sym(B_ao, Pz_a, nbf)
    Gzb_tot = Jz_tot - K.build_exchange_3c_sym(B_ao, Pz_b, nbf)

    dFa_dens = np.zeros((nbf, nbf), dtype=np.float64)
    dFa_dens[:, :na] = Ca0.T @ Gza_tot @ Ca0_o
    dFa0_sym = 0.5 * (ea0[:, None] + ea0[None, :]) * Za0_sym + 0.5 * (dFa_dens + dFa_dens.T)

    dFb_dens = np.zeros((nbf, nbf), dtype=np.float64)
    dFb_dens[:, :nb] = Cb0.T @ Gzb_tot @ Cb0_o
    dFb0_sym = 0.5 * (eb0[:, None] + eb0[None, :]) * Zb0_sym + 0.5 * (dFb_dens + dFb_dens.T)
    return Za0_sym, Zb0_sym, Pz_a, Pz_b, dFa0_sym, dFb0_sym


def _finalize_omp2_family_grad_rhf(
    mol,
    primary_basis,
    aux_basis,
    H_core: np.ndarray,
    B_ao: np.ndarray,
    C: np.ndarray,
    F_ao: np.ndarray,
    B_oo: np.ndarray,
    B_ov: np.ndarray,
    B_vv: np.ndarray,
    g_oo: np.ndarray,
    g_vv: np.ndarray,
    X_mat: np.ndarray,
    no: int,
    nf: int,
    cutoff: float,
    scf_res: Optional[HybridSCFResult] = None,
    S_ovlp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Perakitan (P_eff, W_eff, Gamma_3c, gamma_2c) -> gradien nuklir RHF (AE & Frozen-Core)."""
    nbf = C.shape[0]
    n_act = no - nf
    nv = nbf - no
    n_aux = B_ao.shape[1]
    C_o = np.ascontiguousarray(C[:, :no])
    C_frz = np.ascontiguousarray(C[:, :nf])
    C_act = np.ascontiguousarray(C[:, nf:no])
    C_v = np.ascontiguousarray(C[:, no:])

    B_oo_3d = B_oo.T.reshape(n_aux, n_act, n_act)
    B_ov_3d = B_ov.T.reshape(n_aux, n_act, nv)
    B_vo_3d = B_ov_3d.transpose(0, 2, 1)
    B_vv_3d = B_vv.T.reshape(n_aux, nv, nv)
    X_3d = X_mat.T.reshape(n_aux, n_act, nv)
    X_vo_3d = X_3d.transpose(0, 2, 1)

    B_fa_3d = K.transform_3c_mo(B_ao, C_frz, C_v, nbf).T.reshape(n_aux, nf, nv) if (nf > 0 and nv > 0) else None
    B_fi_3d = K.transform_3c_mo(B_ao, C_frz, C_act, nbf).T.reshape(n_aux, nf, n_act) if (nf > 0 and n_act > 0) else None

    P_ref_a = C_o @ C_o.T
    P_corr_a = C_act @ g_oo @ C_act.T + C_v @ g_vv @ C_v.T
    F_HF_mo = C.T @ F_ao @ C
    G_gamma_ao = 2.0 * K.build_coulomb_3c(B_ao, P_corr_a, nbf) - K.build_exchange_3c_sym(B_ao, P_corr_a, nbf)
    G_gamma_mo = C.T @ G_gamma_ao @ C

    F_raw = _build_omp2_f_raw(
        F_HF_mo, G_gamma_mo, g_oo, g_vv, B_oo_3d, B_ov_3d, B_vv_3d, X_3d, no, nf=nf,
        B_fa_3d=B_fa_3d, B_fi_3d=B_fi_3d,
    )
    F_tot_mo = 0.5 * (F_raw + F_raw.T)

    P_ref_tot = 2.0 * P_ref_a
    P_corr_tot = 2.0 * P_corr_a
    P_eff_ao = P_ref_tot + P_corr_tot
    W_eff_ao = 2.0 * (C @ F_tot_mo @ C.T)

    # Frozen-core Z-vector relaxation in canonical RHF basis C0
    has_fc = nf > 0
    C0 = C0_o = Z0_sym = P0_ref_tot = Pz_tot = B0_all_o_3d = None
    if has_fc:
        import mshqc
        if S_ovlp is None:
            S_ovlp = mshqc.IntegralEngine(mol, primary_basis).compute_overlap()
        if scf_res is None:
            scf_res = run_df_rhf(
                H_core, S_ovlp, float(mol.nuclear_repulsion_energy()), B_ao, no,
                energy_tol=cutoff, density_tol=cutoff, C_init=C,
            )
        C0 = np.ascontiguousarray(scf_res.C_alpha, dtype=np.float64)
        C0_o = np.ascontiguousarray(C0[:, :no])
        eps0 = np.asarray(scf_res.orbital_energies_alpha, dtype=np.float64)
        g_pf_omp = F_raw[nf:, :nf] - F_raw[:nf, nf:].T
        Z0_sym, Pz_a, dF0_sym = _solve_rhf_fc_zvec(B_ao, S_ovlp, C0, eps0, C, g_pf_omp, no, nf)
        Pz_tot = 2.0 * Pz_a
        P0_ref_tot = 2.0 * (C0_o @ C0_o.T)
        P_eff_ao = P_eff_ao + Pz_tot
        W_eff_ao = W_eff_ao + 2.0 * (C0 @ dF0_sym @ C0.T)
        B0_all_o_3d = K.transform_3c_mo(B_ao, C0, C0_o, nbf).T.reshape(n_aux, nbf, no)

    P_eff_ao = np.ascontiguousarray(P_eff_ao, dtype=np.float64)
    W_eff_ao = np.ascontiguousarray(W_eff_ao, dtype=np.float64)

    c_ref = K.contract_ao_sym(B_ao, P_ref_tot, nbf)
    c_corr = K.contract_ao_sym(B_ao, P_corr_tot, nbf)
    c_rhf0 = K.contract_ao_sym(B_ao, P0_ref_tot, nbf) if has_fc else None
    c_pz = K.contract_ao_sym(B_ao, Pz_tot, nbf) if has_fc else None

    B_oo_all_3d = K.transform_3c_mo(B_ao, C_o, C_o, nbf).T.reshape(n_aux, no, no)
    B_ao_all_3d = B_oo_all_3d[:, nf:no, :]
    B_vo_all_3d = K.transform_3c_mo(B_ao, C_v, C_o, nbf).T.reshape(n_aux, nv, no)
    Gamma_3c = np.empty((nbf * nbf, n_aux), dtype=np.float64)

    chunk = 64
    for p0 in range(0, n_aux, chunk):
        p1 = min(n_aux, p0 + chunk)
        bs = p1 - p0
        G_blk = (
            (P_ref_tot + P_corr_tot)[None, :, :] * c_ref[p0:p1, None, None]
            + P_ref_tot[None, :, :] * c_corr[p0:p1, None, None]
        )
        M_ref = (C_o @ B_oo_all_3d[p0:p1]) @ C_o.T
        M_corr = (C_act @ (g_oo @ B_ao_all_3d[p0:p1])) @ C_o.T + (C_v @ (g_vv @ B_vo_all_3d[p0:p1])) @ C_o.T
        G_blk -= 2.0 * (M_ref + M_corr + M_corr.swapaxes(1, 2))
        Conn = (C_v @ X_vo_3d[p0:p1]) @ C_act.T
        G_blk += 2.0 * (Conn + Conn.swapaxes(1, 2))

        if has_fc:
            G_blk += (
                Pz_tot[None, :, :] * c_rhf0[p0:p1, None, None]
                + P0_ref_tot[None, :, :] * c_pz[p0:p1, None, None]
            )
            M_z = (C0 @ (Z0_sym @ B0_all_o_3d[p0:p1])) @ C0_o.T
            G_blk -= 2.0 * (M_z + M_z.swapaxes(1, 2))

        Gamma_3c[:, p0:p1] = G_blk.reshape(bs, nbf * nbf).T

    gamma_2c = K.compute_gamma2c(Gamma_3c, B_ao, nbf)
    return K.eval_df_nuclear_gradient(
        mol, primary_basis, aux_basis, P_eff_ao, W_eff_ao, Gamma_3c, gamma_2c, cutoff
    )


def _finalize_omp2_family_grad_uhf(
    mol,
    primary_basis,
    aux_basis,
    H_core: np.ndarray,
    B_ao: np.ndarray,
    Ca: np.ndarray,
    Cb: np.ndarray,
    Fa_ao: np.ndarray,
    Fb_ao: np.ndarray,
    Boo_a: np.ndarray,
    Bov_a: np.ndarray,
    Bvv_a: np.ndarray,
    Boo_b: np.ndarray,
    Bov_b: np.ndarray,
    Bvv_b: np.ndarray,
    g_oo_a: np.ndarray,
    g_vv_a: np.ndarray,
    g_oo_b: np.ndarray,
    g_vv_b: np.ndarray,
    X_a: np.ndarray,
    X_b: np.ndarray,
    na: int,
    nb: int,
    nf: int,
    cutoff: float,
    scf_res: Optional[HybridSCFResult] = None,
    S_ovlp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Perakitan (P_eff, W_eff, Gamma_3c, gamma_2c) -> gradien nuklir UHF (AE & Frozen-Core)."""
    nbf = Ca.shape[0]
    na_act, nb_act = na - nf, nb - nf
    va, vb = nbf - na, nbf - nb
    n_aux = B_ao.shape[1]
    has_b = (nb_act > 0 and vb > 0)
    Ca_o, Ca_frz, Ca_act, Ca_v = (
        np.ascontiguousarray(Ca[:, :na]),
        np.ascontiguousarray(Ca[:, :nf]),
        np.ascontiguousarray(Ca[:, nf:na]),
        np.ascontiguousarray(Ca[:, na:]),
    )
    Cb_o, Cb_frz, Cb_act, Cb_v = (
        np.ascontiguousarray(Cb[:, :nb]),
        np.ascontiguousarray(Cb[:, :nf]),
        np.ascontiguousarray(Cb[:, nf:nb]),
        np.ascontiguousarray(Cb[:, nb:]),
    )

    Boo_a_3d = Boo_a.T.reshape(n_aux, na_act, na_act)
    Bov_a_3d = Bov_a.T.reshape(n_aux, na_act, va)
    Bvo_a_3d = Bov_a_3d.transpose(0, 2, 1)
    Bvv_a_3d = Bvv_a.T.reshape(n_aux, va, va)
    Xa_3d = X_a.T.reshape(n_aux, na_act, va)
    Xa_vo_3d = Xa_3d.transpose(0, 2, 1)

    Bfa_a_3d = K.transform_3c_mo(B_ao, Ca_frz, Ca_v, nbf).T.reshape(n_aux, nf, va) if (nf > 0 and va > 0) else None
    Bfi_a_3d = K.transform_3c_mo(B_ao, Ca_frz, Ca_act, nbf).T.reshape(n_aux, nf, na_act) if (nf > 0 and na_act > 0) else None

    P_ref_a = Ca_o @ Ca_o.T if na > 0 else np.zeros((nbf, nbf), dtype=np.float64)
    P_ref_b = Cb_o @ Cb_o.T if nb > 0 else np.zeros((nbf, nbf), dtype=np.float64)
    P_corr_a = Ca_act @ g_oo_a @ Ca_act.T + Ca_v @ g_vv_a @ Ca_v.T if (na_act > 0 and va > 0) else np.zeros((nbf, nbf))
    P_corr_b = Cb_act @ g_oo_b @ Cb_act.T + Cb_v @ g_vv_b @ Cb_v.T if has_b else np.zeros((nbf, nbf))

    Fa_HF_mo = Ca.T @ Fa_ao @ Ca
    Fb_HF_mo = Cb.T @ Fb_ao @ Cb
    J_corr = K.build_coulomb_3c(B_ao, P_corr_a + P_corr_b, nbf)
    Ga_gamma_mo = Ca.T @ (J_corr - K.build_exchange_3c_sym(B_ao, P_corr_a, nbf)) @ Ca
    Gb_gamma_mo = Cb.T @ (J_corr - K.build_exchange_3c_sym(B_ao, P_corr_b, nbf)) @ Cb

    Fa_raw = _build_omp2_f_raw(
        Fa_HF_mo, Ga_gamma_mo, g_oo_a, g_vv_a, Boo_a_3d, Bov_a_3d, Bvv_a_3d, Xa_3d, na, nf=nf,
        B_fa_3d=Bfa_a_3d, B_fi_3d=Bfi_a_3d,
    )
    W_eff_ao = Ca @ (0.5 * (Fa_raw + Fa_raw.T)) @ Ca.T

    Fb_raw = np.zeros((nbf, nbf), dtype=np.float64)
    if nb > 0:
        if has_b:
            Boo_b_3d = Boo_b.T.reshape(n_aux, nb_act, nb_act)
            Bov_b_3d = Bov_b.T.reshape(n_aux, nb_act, vb)
            Bvo_b_3d = Bov_b_3d.transpose(0, 2, 1)
            Bvv_b_3d = Bvv_b.T.reshape(n_aux, vb, vb)
            Xb_3d = X_b.T.reshape(n_aux, nb_act, vb)
            Xb_vo_3d = Xb_3d.transpose(0, 2, 1)
            Bfa_b_3d = K.transform_3c_mo(B_ao, Cb_frz, Cb_v, nbf).T.reshape(n_aux, nf, vb) if (nf > 0 and vb > 0) else None
            Bfi_b_3d = K.transform_3c_mo(B_ao, Cb_frz, Cb_act, nbf).T.reshape(n_aux, nf, nb_act) if (nf > 0 and nb_act > 0) else None
            Fb_raw = _build_omp2_f_raw(
                Fb_HF_mo, Gb_gamma_mo, g_oo_b, g_vv_b, Boo_b_3d, Bov_b_3d, Bvv_b_3d, Xb_3d, nb, nf=nf,
                B_fa_3d=Bfa_b_3d, B_fi_3d=Bfi_b_3d,
            )
        else:
            Fb_raw[:, :nb] = Fb_HF_mo[:, :nb] + Gb_gamma_mo[:, :nb]
        W_eff_ao = W_eff_ao + Cb @ (0.5 * (Fb_raw + Fb_raw.T)) @ Cb.T

    P_ref_tot = P_ref_a + P_ref_b
    P_corr_tot = P_corr_a + P_corr_b
    P_eff_ao = P_ref_tot + P_corr_tot

    has_fc = nf > 0
    Ca0 = Cb0 = Ca0_o = Cb0_o = Za0_sym = Zb0_sym = P0_ref_tot = Pz_tot = B0_a_3d = B0_b_3d = None
    if has_fc:
        import mshqc
        if S_ovlp is None:
            S_ovlp = mshqc.IntegralEngine(mol, primary_basis).compute_overlap()
        if scf_res is None:
            scf_res = run_df_uhf(
                H_core, S_ovlp, float(mol.nuclear_repulsion_energy()), B_ao, na, nb,
                energy_tol=cutoff, density_tol=cutoff, C_init_a=Ca, C_init_b=Cb,
            )
        Ca0 = np.ascontiguousarray(scf_res.C_alpha, dtype=np.float64)
        Cb0 = np.ascontiguousarray(scf_res.C_beta, dtype=np.float64)
        Ca0_o, Cb0_o = np.ascontiguousarray(Ca0[:, :na]), np.ascontiguousarray(Cb0[:, :nb])
        ea0 = np.asarray(scf_res.orbital_energies_alpha, dtype=np.float64)
        eb0 = np.asarray(scf_res.orbital_energies_beta, dtype=np.float64)
        ga_pf = Fa_raw[nf:, :nf] - Fa_raw[:nf, nf:].T
        gb_pf = Fb_raw[nf:, :nf] - Fb_raw[:nf, nf:].T
        Za0_sym, Zb0_sym, Pz_a, Pz_b, dFa0_sym, dFb0_sym = _solve_uhf_fc_zvec(
            B_ao, S_ovlp, Ca0, Cb0, ea0, eb0, Ca, Cb, ga_pf, gb_pf, na, nb, nf
        )
        Pz_tot = Pz_a + Pz_b
        P0_ref_tot = (Ca0_o @ Ca0_o.T) + (Cb0_o @ Cb0_o.T)
        P_eff_ao = P_eff_ao + Pz_tot
        W_eff_ao = W_eff_ao + (Ca0 @ dFa0_sym @ Ca0.T) + (Cb0 @ dFb0_sym @ Cb0.T)
        B0_a_3d = K.transform_3c_mo(B_ao, Ca0, Ca0_o, nbf).T.reshape(n_aux, nbf, na)
        B0_b_3d = K.transform_3c_mo(B_ao, Cb0, Cb0_o, nbf).T.reshape(n_aux, nbf, nb)

    P_eff_ao = np.ascontiguousarray(P_eff_ao, dtype=np.float64)
    W_eff_ao = np.ascontiguousarray(W_eff_ao, dtype=np.float64)

    c_ref = K.contract_ao_sym(B_ao, P_ref_tot, nbf)
    c_corr = K.contract_ao_sym(B_ao, P_corr_tot, nbf)
    c_uhf0 = K.contract_ao_sym(B_ao, P0_ref_tot, nbf) if has_fc else None
    c_pz = K.contract_ao_sym(B_ao, Pz_tot, nbf) if has_fc else None

    Boo_a_all_3d = K.transform_3c_mo(B_ao, Ca_o, Ca_o, nbf).T.reshape(n_aux, na, na)
    Bao_a_all_3d = Boo_a_all_3d[:, nf:na, :]
    Bvo_a_all_3d = K.transform_3c_mo(B_ao, Ca_v, Ca_o, nbf).T.reshape(n_aux, va, na)
    Boo_b_all_3d = K.transform_3c_mo(B_ao, Cb_o, Cb_o, nbf).T.reshape(n_aux, nb, nb) if nb > 0 else None
    Bao_b_all_3d = Boo_b_all_3d[:, nf:nb, :] if has_b else None
    Bvo_b_all_3d = K.transform_3c_mo(B_ao, Cb_v, Cb_o, nbf).T.reshape(n_aux, vb, nb) if has_b else None
    Gamma_3c = np.empty((nbf * nbf, n_aux), dtype=np.float64)

    chunk = 64
    for p0 in range(0, n_aux, chunk):
        p1 = min(n_aux, p0 + chunk)
        bs = p1 - p0
        G_blk = (
            (P_ref_tot + P_corr_tot)[None, :, :] * c_ref[p0:p1, None, None]
            + P_ref_tot[None, :, :] * c_corr[p0:p1, None, None]
        )
        M_ref_a = (Ca_o @ Boo_a_all_3d[p0:p1]) @ Ca_o.T
        M_corr_a = (Ca_act @ (g_oo_a @ Bao_a_all_3d[p0:p1])) @ Ca_o.T + (Ca_v @ (g_vv_a @ Bvo_a_all_3d[p0:p1])) @ Ca_o.T
        G_blk -= M_ref_a + M_corr_a + M_corr_a.swapaxes(1, 2)
        Conn_a = (Ca_v @ Xa_vo_3d[p0:p1]) @ Ca_act.T
        G_blk += Conn_a + Conn_a.swapaxes(1, 2)

        if nb > 0:
            M_ref_b = (Cb_o @ Boo_b_all_3d[p0:p1]) @ Cb_o.T
            G_blk -= M_ref_b
            if has_b:
                M_corr_b = (Cb_act @ (g_oo_b @ Bao_b_all_3d[p0:p1])) @ Cb_o.T + (Cb_v @ (g_vv_b @ Bvo_b_all_3d[p0:p1])) @ Cb_o.T
                G_blk -= M_corr_b + M_corr_b.swapaxes(1, 2)
                Conn_b = (Cb_v @ Xb_vo_3d[p0:p1]) @ Cb_act.T
                G_blk += Conn_b + Conn_b.swapaxes(1, 2)

        if has_fc:
            G_blk += (
                Pz_tot[None, :, :] * c_uhf0[p0:p1, None, None]
                + P0_ref_tot[None, :, :] * c_pz[p0:p1, None, None]
            )
            M_za = (Ca0 @ (Za0_sym @ B0_a_3d[p0:p1])) @ Ca0_o.T
            M_zb = (Cb0 @ (Zb0_sym @ B0_b_3d[p0:p1])) @ Cb0_o.T
            G_blk -= (M_za + M_za.swapaxes(1, 2)) + (M_zb + M_zb.swapaxes(1, 2))

        Gamma_3c[:, p0:p1] = G_blk.reshape(bs, nbf * nbf).T

    gamma_2c = K.compute_gamma2c(Gamma_3c, B_ao, nbf)
    return K.eval_df_nuclear_gradient(
        mol, primary_basis, aux_basis, P_eff_ao, W_eff_ao, Gamma_3c, gamma_2c, cutoff
    )


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
    scf_res: Optional[HybridSCFResult] = None,
    S_ovlp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Evaluasi gradien nuklir analitik DF-OMP2 / SCS-OMP2 / SOS-OMP2 (RHF, AE & Frozen-Core)."""
    nbf = int(primary_basis.n_basis_functions())
    no = int(omp_res.n_occ_alpha)
    nf = max(0, min(int(getattr(omp_res, "n_frozen", 0)), no))
    n_act = no - nf
    nv = nbf - no

    C_in = np.ascontiguousarray(omp_res.C_alpha, dtype=np.float64)
    J, K_mat = K.build_jk_3c_rhf(B_ao, np.ascontiguousarray(C_in[:, :no]), nbf)
    F_ao = H_core + J - K_mat
    C, eps, _ = semicanonicalize_subspaces_frozen(C_in, F_ao, no, nf)
    B_oo, B_ov, B_vv = K.transform_3c_mo_blocks(
        B_ao, np.ascontiguousarray(C[:, nf:no]), np.ascontiguousarray(C[:, no:]), nbf
    )

    _, _, T2 = K.eval_rmp2_df(B_ov, eps[nf:no], eps[no:])
    g_oo, g_vv = K.eval_omp2_1rdm_rhf(T2, n_act, nv, c_os=c_os, c_ss=c_ss)
    T2_4d = t2_to_4d(T2, n_act, n_act, nv, nv)
    T_eff = (c_os + c_ss) * T2_4d - c_ss * T2_4d.transpose(0, 1, 3, 2)
    X_mat = T_eff.transpose(0, 2, 1, 3).reshape(n_act * nv, n_act * nv) @ B_ov

    return _finalize_omp2_family_grad_rhf(
        mol, primary_basis, aux_basis, H_core, B_ao, C, F_ao, B_oo, B_ov, B_vv,
        g_oo, g_vv, X_mat, no, nf, cutoff, scf_res=scf_res, S_ovlp=S_ovlp,
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
    scf_res: Optional[HybridSCFResult] = None,
    S_ovlp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Evaluasi gradien nuklir analitik DF-OMP2 / SCS-OMP2 / SOS-OMP2 (UHF, AE & Frozen-Core)."""
    nbf = int(primary_basis.n_basis_functions())
    na, nb = int(omp_res.n_occ_alpha), int(omp_res.n_occ_beta)
    nf = max(0, min(int(getattr(omp_res, "n_frozen", 0)), nb))
    na_act, nb_act = na - nf, nb - nf
    va, vb = nbf - na, nbf - nb
    n_aux = B_ao.shape[1]
    has_b = (nb_act > 0 and vb > 0)

    Ca_in = np.ascontiguousarray(omp_res.C_alpha, dtype=np.float64)
    Cb_in = np.ascontiguousarray(omp_res.C_beta, dtype=np.float64)
    J, Ka, Kb = K.build_jk_3c_uhf(B_ao, np.ascontiguousarray(Ca_in[:, :na]), np.ascontiguousarray(Cb_in[:, :nb]), nbf)
    Fa_ao, Fb_ao = H_core + J - Ka, H_core + J - Kb

    Ca, eps_a, _ = semicanonicalize_subspaces_frozen(Ca_in, Fa_ao, na, nf)
    Cb, eps_b, _ = semicanonicalize_subspaces_frozen(Cb_in, Fb_ao, nb, nf)
    Boo_a, Bov_a, Bvv_a, Boo_b, Bov_b, Bvv_b = K.transform_3c_mo_blocks_uhf(
        B_ao,
        np.ascontiguousarray(Ca[:, nf:na]), np.ascontiguousarray(Ca[:, na:]),
        np.ascontiguousarray(Cb[:, nf:nb]), np.ascontiguousarray(Cb[:, nb:]),
        nbf,
    )
    _, _, _, T2_aa, T2_bb, T2_ab = K.eval_ump2_df(
        Bov_a, Bov_b, eps_a[nf:na], eps_a[na:], eps_b[nf:nb], eps_b[nb:]
    )
    g_oo_a, g_vv_a, g_oo_b, g_vv_b = K.eval_omp2_1rdm_uhf(
        T2_aa, T2_bb, T2_ab, na_act, va, nb_act, vb, c_os=c_os, c_ss=c_ss
    )

    X_a = np.zeros((na_act * va, n_aux), dtype=np.float64)
    if na_act > 1 and va > 0:
        Taa_4d = t2_to_4d(T2_aa, na_act, na_act, va, va)
        X_a += c_ss * (Taa_4d.transpose(0, 2, 1, 3).reshape(na_act * va, na_act * va) @ Bov_a)
    if has_b and na_act > 0 and va > 0:
        Tab_4d = t2_to_4d(T2_ab, na_act, nb_act, va, vb)
        X_a += c_os * (Tab_4d.transpose(0, 2, 1, 3).reshape(na_act * va, nb_act * vb) @ Bov_b)

    X_b = np.zeros((nb_act * vb, n_aux), dtype=np.float64)
    if has_b:
        if nb_act > 1:
            Tbb_4d = t2_to_4d(T2_bb, nb_act, nb_act, vb, vb)
            X_b += c_ss * (Tbb_4d.transpose(0, 2, 1, 3).reshape(nb_act * vb, nb_act * vb) @ Bov_b)
        if na_act > 0 and va > 0:
            X_b += c_os * (Tab_4d.transpose(1, 3, 0, 2).reshape(nb_act * vb, na_act * va) @ Bov_a)

    return _finalize_omp2_family_grad_uhf(
        mol, primary_basis, aux_basis, H_core, B_ao, Ca, Cb, Fa_ao, Fb_ao,
        Boo_a, Bov_a, Bvv_a, Boo_b, Bov_b, Bvv_b,
        g_oo_a, g_vv_a, g_oo_b, g_vv_b, X_a, X_b, na, nb, nf, cutoff,
        scf_res=scf_res, S_ovlp=S_ovlp,
    )


def eval_df_kappa_omp2_gradient_rhf(
    mol,
    primary_basis,
    aux_basis,
    H_core: np.ndarray,
    B_ao: np.ndarray,
    omp_res: HybridOMPResult,
    kappa: float = 1.45,
    c_os: float = 1.0,
    c_ss: float = 1.0,
    cutoff: float = 1.0e-12,
    scf_res: Optional[HybridSCFResult] = None,
    S_ovlp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Evaluasi gradien nuklir analitik eksak DF-kappa-OOMP2 (RHF, AE & Frozen-Core)."""
    nbf = int(primary_basis.n_basis_functions())
    no = int(omp_res.n_occ_alpha)
    nf = max(0, min(int(getattr(omp_res, "n_frozen", 0)), no))
    n_act = no - nf
    nv = nbf - no

    C_in = np.ascontiguousarray(omp_res.C_alpha, dtype=np.float64)
    J, K_mat = K.build_jk_3c_rhf(B_ao, np.ascontiguousarray(C_in[:, :no]), nbf)
    F_ao = H_core + J - K_mat
    C, eps, _ = semicanonicalize_subspaces_frozen(C_in, F_ao, no, nf)
    B_oo, B_ov, B_vv = K.transform_3c_mo_blocks(
        B_ao, np.ascontiguousarray(C[:, nf:no]), np.ascontiguousarray(C[:, no:]), nbf
    )

    _, g_oo, g_vv, _, T_tilde_eff = _eval_kappa_omp2_rhf_data(
        B_ov, eps[nf:no], eps[no:], kappa=kappa, c_os=c_os, c_ss=c_ss
    )
    X_mat = T_tilde_eff.transpose(0, 2, 1, 3).reshape(n_act * nv, n_act * nv) @ B_ov
    return _finalize_omp2_family_grad_rhf(
        mol, primary_basis, aux_basis, H_core, B_ao, C, F_ao, B_oo, B_ov, B_vv,
        g_oo, g_vv, X_mat, no, nf, cutoff, scf_res=scf_res, S_ovlp=S_ovlp,
    )


def eval_df_kappa_omp2_gradient_uhf(
    mol,
    primary_basis,
    aux_basis,
    H_core: np.ndarray,
    B_ao: np.ndarray,
    omp_res: HybridOMPResult,
    kappa: float = 1.45,
    c_os: float = 1.0,
    c_ss: float = 1.0,
    cutoff: float = 1.0e-12,
    scf_res: Optional[HybridSCFResult] = None,
    S_ovlp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Evaluasi gradien nuklir analitik eksak DF-kappa-OOMP2 (UHF, AE & Frozen-Core)."""
    nbf = int(primary_basis.n_basis_functions())
    na, nb = int(omp_res.n_occ_alpha), int(omp_res.n_occ_beta)
    nf = max(0, min(int(getattr(omp_res, "n_frozen", 0)), nb))
    na_act, nb_act = na - nf, nb - nf
    va, vb = nbf - na, nbf - nb
    n_aux = B_ao.shape[1]
    has_b = (nb_act > 0 and vb > 0)

    Ca_in = np.ascontiguousarray(omp_res.C_alpha, dtype=np.float64)
    Cb_in = np.ascontiguousarray(omp_res.C_beta, dtype=np.float64)
    J, Ka, Kb = K.build_jk_3c_uhf(B_ao, np.ascontiguousarray(Ca_in[:, :na]), np.ascontiguousarray(Cb_in[:, :nb]), nbf)
    Fa_ao, Fb_ao = H_core + J - Ka, H_core + J - Kb

    Ca, eps_a, _ = semicanonicalize_subspaces_frozen(Ca_in, Fa_ao, na, nf)
    Cb, eps_b, _ = semicanonicalize_subspaces_frozen(Cb_in, Fb_ao, nb, nf)
    Boo_a, Bov_a, Bvv_a, Boo_b, Bov_b, Bvv_b = K.transform_3c_mo_blocks_uhf(
        B_ao,
        np.ascontiguousarray(Ca[:, nf:na]), np.ascontiguousarray(Ca[:, na:]),
        np.ascontiguousarray(Cb[:, nf:nb]), np.ascontiguousarray(Cb[:, nb:]),
        nbf,
    )

    _, g_oo_a, g_vv_a, g_oo_b, g_vv_b, Taa_4d, Tbb_4d, Tab_4d = _eval_kappa_omp2_uhf_data(
        Bov_a, Bov_b, eps_a[nf:na], eps_a[na:], eps_b[nf:nb], eps_b[nb:], kappa=kappa, c_os=c_os, c_ss=c_ss
    )

    X_a = np.zeros((na_act * va, n_aux), dtype=np.float64)
    if na_act > 1 and va > 0 and c_ss != 0.0:
        X_a += c_ss * (Taa_4d.transpose(0, 2, 1, 3).reshape(na_act * va, na_act * va) @ Bov_a)
    if has_b and na_act > 0 and va > 0 and c_os != 0.0:
        X_a += c_os * (Tab_4d.transpose(0, 2, 1, 3).reshape(na_act * va, nb_act * vb) @ Bov_b)

    X_b = np.zeros((nb_act * vb, n_aux), dtype=np.float64)
    if has_b:
        if nb_act > 1 and c_ss != 0.0:
            X_b += c_ss * (Tbb_4d.transpose(0, 2, 1, 3).reshape(nb_act * vb, nb_act * vb) @ Bov_b)
        if na_act > 0 and va > 0 and c_os != 0.0:
            X_b += c_os * (Tab_4d.transpose(1, 3, 0, 2).reshape(nb_act * vb, na_act * va) @ Bov_a)

    return _finalize_omp2_family_grad_uhf(
        mol, primary_basis, aux_basis, H_core, B_ao, Ca, Cb, Fa_ao, Fb_ao,
        Boo_a, Bov_a, Bvv_a, Boo_b, Bov_b, Bvv_b,
        g_oo_a, g_vv_a, g_oo_b, g_vv_b, X_a, X_b, na, nb, nf, cutoff,
        scf_res=scf_res, S_ovlp=S_ovlp,
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
    kappa: float = 1.45,
    freeze_core: bool = False,
    n_frozen: int = 0,
) -> Tuple[float, np.ndarray]:
    """One-liner Thin-Core gradien nuklir analitik DF (mendukung AE & Frozen-Core)."""
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
    nf = int(n_frozen) if n_frozen > 0 else (int(mshqc.standard_frozen_core(mol)) if freeze_core else 0)

    ref = reference.lower()
    if ref in ("", "auto"):
        ref = m if m in ("rhf", "uhf", "rohf") else ("rhf" if mult == 1 else "uhf")

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

    if m in ("kappa-omp2", "k-omp2", "kappa-oomp2", "koomp2"):
        if ref == "rhf":
            omp_res = run_df_kappa_omp2_rhf(
                H_core, v_nn, B_ao, scf_res.C_alpha, na, kappa=kappa,
                grad_tol=orb_grad_tol, energy_tol=threshold, max_iter=80, n_frozen=nf,
            )
            grad = eval_df_kappa_omp2_gradient_rhf(
                mol, basis, aux_basis, H_core, B_ao, omp_res, kappa=kappa,
                cutoff=threshold, scf_res=scf_res, S_ovlp=S_ovlp,
            )
            return omp_res.energy_total, grad
        omp_res = run_df_kappa_omp2_uhf(
            H_core, S_ovlp, v_nn, B_ao, scf_res.C_alpha, scf_res.C_beta, na, nb, kappa=kappa,
            grad_tol=orb_grad_tol, energy_tol=threshold, max_iter=80, n_frozen=nf,
        )
        grad = eval_df_kappa_omp2_gradient_uhf(
            mol, basis, aux_basis, H_core, B_ao, omp_res, kappa=kappa,
            cutoff=threshold, scf_res=scf_res, S_ovlp=S_ovlp,
        )
        return omp_res.energy_total, grad

    omp_params = {
        "omp2": (1.0, 1.0),
        "scs-omp2": (1.2, 1.0 / 3.0),
        "sos-omp2": (1.3, 0.0),
        "lt-sos-omp2": (1.3, 0.0),
    }
    if m not in omp_params:
        raise ValueError(f"Gradien analitik DF belum mendukung metode: {method}")

    c_os, c_ss = omp_params[m]
    if ref == "rhf":
        omp_res = run_df_omp_rhf(
            H_core, v_nn, B_ao, scf_res.C_alpha, na,
            method_name=m, c_os=c_os, c_ss=c_ss,
            grad_tol=orb_grad_tol, energy_tol=threshold, max_iter=80, n_frozen=nf,
        )
        grad = eval_df_omp2_gradient_rhf(
            mol, basis, aux_basis, H_core, B_ao, omp_res, c_os=c_os, c_ss=c_ss,
            cutoff=threshold, scf_res=scf_res, S_ovlp=S_ovlp,
        )
        return omp_res.energy_total, grad

    omp_res = run_df_omp_uhf(
        H_core, S_ovlp, v_nn, B_ao, scf_res.C_alpha, scf_res.C_beta, na, nb,
        method_name=m, c_os=c_os, c_ss=c_ss,
        grad_tol=orb_grad_tol, energy_tol=threshold, max_iter=80, n_frozen=nf,
    )
    grad = eval_df_omp2_gradient_uhf(
        mol, basis, aux_basis, H_core, B_ao, omp_res, c_os=c_os, c_ss=c_ss,
        cutoff=threshold, scf_res=scf_res, S_ovlp=S_ovlp,
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
