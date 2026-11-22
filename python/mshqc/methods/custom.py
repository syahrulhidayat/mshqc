# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
Plugin hook generik (RHF & UHF) untuk merakit metode Orbital-Optimized kustom
dalam ~10-20 baris Python di atas C++20 `mshqc.kernels` + TR-SOSCF Steihaug-CG.
"""

from typing import Callable, Tuple
import numpy as np

from .._mshqc import kernels as K
from .omp_hybrid import (
    HybridOMPResult,
    _assemble_orbital_grad_rhf,
    _assemble_orbital_grad_uhf,
    _compute_s2_uhf,
    _rotate_orbitals,
    _update_trust_radius,
)
from .tr_soscf import solve_tr_soscf_rhf, solve_tr_soscf_uhf

# RHF callback: (B_oo, B_ov, B_vv, eps_o, eps_v) -> (e_corr, gamma_oo, gamma_vv, Z_vo)
CorrCallbackRHF = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[float, np.ndarray, np.ndarray, np.ndarray],
]

# UHF callback: (B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b, eps_oa, eps_va, eps_ob, eps_vb)
#            -> (e_corr, g_oo_a, g_vv_a, g_oo_b, g_vv_b, Za_vo, Zb_vo)
CorrCallbackUHF = Callable[
    [
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    ],
    Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]


def optimize_orbitals_rhf(
    H_core: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    C0: np.ndarray,
    n_occ: int,
    corr_fn: CorrCallbackRHF,
    method_name: str = "custom-omp",
    grad_tol: float = 1e-5,
    energy_tol: float = 1e-9,
    max_iter: int = 40,
) -> HybridOMPResult:
    """Mesin optimisasi orbital generik RHF berbasis C++20 `mshqc.kernels`."""
    H = np.ascontiguousarray(H_core, dtype=np.float64)
    B = np.ascontiguousarray(B_ao, dtype=np.float64)
    C = np.array(C0, dtype=np.float64, copy=True)
    nbf = H.shape[0]
    e_prev = 0.0
    trust_radius = 0.25

    for it in range(max_iter):
        C_occ = np.ascontiguousarray(C[:, :n_occ])
        J, K_mat = K.build_jk_3c_rhf(B, C_occ, nbf)
        F_ao = H + J - K_mat
        e_ref = float(np.sum((C_occ @ C_occ.T) * (H + F_ao)) + v_nn)

        C, eps, F_vo = K.semicanonicalize_subspaces(C, F_ao, n_occ)
        C_occ = np.ascontiguousarray(C[:, :n_occ])
        C_vir = np.ascontiguousarray(C[:, n_occ:])
        eps_o, eps_v = eps[:n_occ], eps[n_occ:]

        B_oo = K.transform_3c_mo(B, C_occ, C_occ, nbf)
        B_ov = K.transform_3c_mo(B, C_occ, C_vir, nbf)
        B_vv = K.transform_3c_mo(B, C_vir, C_vir, nbf)

        e_corr, g_oo, g_vv, Z_vo = corr_fn(B_oo, B_ov, B_vv, eps_o, eps_v)
        e_tot = e_ref + float(e_corr)
        w_vo = _assemble_orbital_grad_rhf(B, C_occ, C_vir, F_vo, g_oo, g_vv, Z_vo)
        w_rms = float(np.sqrt(np.mean(w_vo ** 2)))

        if w_rms <= grad_tol and (it == 0 or abs(e_tot - e_prev) <= energy_tol):
            return HybridOMPResult(
                method=method_name, reference="rhf",
                energy_scf=e_ref, energy_mp2_corr=float(e_corr), energy_mp3_corr=0.0,
                energy_total=e_tot, converged=True, iterations=it, grad_rms=w_rms, s2=0.0,
                C_alpha=C, C_beta=C.copy(),
                orbital_energies_alpha=eps, orbital_energies_beta=eps.copy(),
                n_occ_alpha=n_occ, n_occ_beta=n_occ,
            )

        trust_radius = _update_trust_radius(trust_radius, it, e_tot, e_prev)
        e_prev = e_tot
        kappa = solve_tr_soscf_rhf(w_vo, eps_o, eps_v, B_oo, B_ov, B_vv, trust_radius=trust_radius)
        C = _rotate_orbitals(C, kappa, n_occ)

    return HybridOMPResult(
        method=method_name, reference="rhf",
        energy_scf=e_ref, energy_mp2_corr=float(e_corr), energy_mp3_corr=0.0,
        energy_total=e_tot, converged=False, iterations=max_iter, grad_rms=w_rms, s2=0.0,
        C_alpha=C, C_beta=C.copy(),
        orbital_energies_alpha=eps, orbital_energies_beta=eps.copy(),
        n_occ_alpha=n_occ, n_occ_beta=n_occ,
    )


def optimize_orbitals_uhf(
    H_core: np.ndarray,
    S_ovlp: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    C0_a: np.ndarray,
    C0_b: np.ndarray,
    na: int,
    nb: int,
    corr_fn: CorrCallbackUHF,
    method_name: str = "custom-uhf-omp",
    grad_tol: float = 1e-5,
    energy_tol: float = 1e-9,
    max_iter: int = 40,
) -> HybridOMPResult:
    """Mesin optimisasi orbital generik UHF (radikal kulit terbuka) berbasis C++20 `mshqc.kernels`."""
    H = np.ascontiguousarray(H_core, dtype=np.float64)
    S = np.ascontiguousarray(S_ovlp, dtype=np.float64)
    B = np.ascontiguousarray(B_ao, dtype=np.float64)
    Ca = np.array(C0_a, dtype=np.float64, copy=True)
    Cb = np.array(C0_b, dtype=np.float64, copy=True)
    nbf = H.shape[0]
    n_params = max(1, (nbf - na) * na + (nbf - nb) * nb)
    e_prev = 0.0
    trust_radius = 0.25

    for it in range(max_iter):
        Ca_o = np.ascontiguousarray(Ca[:, :na])
        Cb_o = np.ascontiguousarray(Cb[:, :nb])
        J, Ka, Kb = K.build_jk_3c_uhf(B, Ca_o, Cb_o, nbf)
        Fa_ao = H + J - Ka
        Fb_ao = H + J - Kb
        e_ref = float(
            0.5 * (np.sum((Ca_o @ Ca_o.T) * (H + Fa_ao)) + np.sum((Cb_o @ Cb_o.T) * (H + Fb_ao)))
            + v_nn
        )

        Ca, eps_a, Fvo_a = K.semicanonicalize_subspaces(Ca, Fa_ao, na)
        Cb, eps_b, Fvo_b = K.semicanonicalize_subspaces(Cb, Fb_ao, nb)
        Ca_o, Ca_v = np.ascontiguousarray(Ca[:, :na]), np.ascontiguousarray(Ca[:, na:])
        Cb_o, Cb_v = np.ascontiguousarray(Cb[:, :nb]), np.ascontiguousarray(Cb[:, nb:])
        eps_oa, eps_va = eps_a[:na], eps_a[na:]
        eps_ob, eps_vb = eps_b[:nb], eps_b[nb:]

        B_oo_a = K.transform_3c_mo(B, Ca_o, Ca_o, nbf)
        B_ov_a = K.transform_3c_mo(B, Ca_o, Ca_v, nbf)
        B_vv_a = K.transform_3c_mo(B, Ca_v, Ca_v, nbf)
        B_oo_b = K.transform_3c_mo(B, Cb_o, Cb_o, nbf)
        B_ov_b = K.transform_3c_mo(B, Cb_o, Cb_v, nbf)
        B_vv_b = K.transform_3c_mo(B, Cb_v, Cb_v, nbf)

        e_corr, g_oo_a, g_vv_a, g_oo_b, g_vv_b, Za, Zb = corr_fn(
            B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b, eps_oa, eps_va, eps_ob, eps_vb
        )
        e_tot = e_ref + float(e_corr)
        wa_vo, wb_vo = _assemble_orbital_grad_uhf(
            B, Ca_o, Ca_v, Cb_o, Cb_v, Fvo_a, Fvo_b, g_oo_a, g_vv_a, g_oo_b, g_vv_b, Za, Zb
        )
        w_rms = float(np.sqrt((np.sum(wa_vo ** 2) + np.sum(wb_vo ** 2)) / n_params))

        if w_rms <= grad_tol and (it == 0 or abs(e_tot - e_prev) <= energy_tol):
            s2 = _compute_s2_uhf(Ca, Cb, S, na, nb)
            return HybridOMPResult(
                method=method_name, reference="uhf",
                energy_scf=e_ref, energy_mp2_corr=float(e_corr), energy_mp3_corr=0.0,
                energy_total=e_tot, converged=True, iterations=it, grad_rms=w_rms, s2=s2,
                C_alpha=Ca, C_beta=Cb,
                orbital_energies_alpha=eps_a, orbital_energies_beta=eps_b,
                n_occ_alpha=na, n_occ_beta=nb,
            )

        trust_radius = _update_trust_radius(trust_radius, it, e_tot, e_prev)
        e_prev = e_tot
        ka, kb = solve_tr_soscf_uhf(
            wa_vo, wb_vo, eps_oa, eps_va, eps_ob, eps_vb,
            B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b,
            trust_radius=trust_radius,
        )
        Ca = _rotate_orbitals(Ca, ka, na)
        if nb > 0:
            Cb = _rotate_orbitals(Cb, kb, nb)

    s2 = _compute_s2_uhf(Ca, Cb, S, na, nb)
    return HybridOMPResult(
        method=method_name, reference="uhf",
        energy_scf=e_ref, energy_mp2_corr=float(e_corr), energy_mp3_corr=0.0,
        energy_total=e_tot, converged=False, iterations=max_iter, grad_rms=w_rms, s2=s2,
        C_alpha=Ca, C_beta=Cb,
        orbital_energies_alpha=eps_a, orbital_energies_beta=eps_b,
        n_occ_alpha=na, n_occ_beta=nb,
    )
