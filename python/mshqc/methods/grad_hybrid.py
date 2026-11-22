# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
Gradien nuklir analitik DF-RHF & DF-UHF yang sangat terbaca di Python,
hanya mendelegasikan kontraksi integral turunan `libcint` ke C++20 `K.eval_df_nuclear_gradient`.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from .._mshqc import kernels as K
from .scf_hybrid import HybridSCFResult, build_sad_guess, run_df_rhf, run_df_uhf


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

    Gamma_3c, gamma_2c = _assemble_scf_gamma_3c(
        scf_res.L_mat, P_eff, C_occ, nbf, is_rhf=True
    )
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


def compute_df_scf_gradient(
    mol,
    basis_name: str = "cc-pVDZ",
    aux_basis_name: Optional[str] = None,
    reference: str = "auto",
    threshold: float = 1.0e-12,
    basis_dir: str = "data/basis",
) -> Tuple[float, np.ndarray]:
    """One-liner Thin-Core untuk menghitung (E_SCF, grad_Ha_Bohr) tanpa solver legacy C++."""
    import mshqc

    aux_name = aux_basis_name or f"{basis_name}-RI"
    basis = mshqc.BasisSet(basis_name, mol, basis_dir)
    aux_basis = mshqc.BasisSet(aux_name, mol, basis_dir)
    ints = mshqc.IntegralEngine(mol, basis)

    H_core = ints.compute_kinetic() + ints.compute_nuclear()
    S_ovlp = ints.compute_overlap()
    v_nn = float(mol.nuclear_repulsion_energy())
    B_ao = K.compute_df_ao_factor(mol, basis, aux_basis, threshold, False, False)
    P_sad = build_sad_guess(mol, basis)

    n_elec = int(mol.n_electrons())
    mult = int(mol.multiplicity())
    na = (n_elec + mult - 1) // 2
    nb = n_elec - na
    ref = reference.lower()
    if ref == "auto":
        ref = "rhf" if mult == 1 else "uhf"

    if ref == "rhf":
        scf_res = run_df_rhf(
            H_core, S_ovlp, v_nn, B_ao, na,
            energy_tol=threshold, density_tol=threshold, P_init=P_sad,
        )
        grad = eval_df_scf_gradient_rhf(mol, basis, aux_basis, scf_res, cutoff=threshold)
        return scf_res.energy_total, grad

    scf_res = run_df_uhf(
        H_core, S_ovlp, v_nn, B_ao, na, nb,
        energy_tol=threshold, density_tol=threshold, P_init=P_sad,
    )
    grad = eval_df_scf_gradient_uhf(mol, basis, aux_basis, scf_res, cutoff=threshold)
    return scf_res.energy_total, grad
