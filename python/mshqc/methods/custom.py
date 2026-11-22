# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
Plugin hook untuk merakit metode Orbital-Optimized kustom dalam ~15 baris Python
di atas mesin C++20 `mshqc.kernels` + pengoptimal TR-SOSCF Steihaug-CG.
"""

from typing import Callable, Tuple
import numpy as np

from .._mshqc import kernels as K
from .omp_hybrid import HybridOMPResult, _rotate_orbitals
from .tr_soscf import solve_tr_soscf_rhf


# Tipe callback: corr_fn(B_oo, B_ov, B_vv, eps_o, eps_v) -> (e_corr, gamma_oo, gamma_vv, Z_vo)
CorrCallbackRHF = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[float, np.ndarray, np.ndarray, np.ndarray],
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
    """
    Mesin optimisasi orbital generik (RHF) berbasis C++20 `mshqc.kernels`.
    Peneliti cukup memasok fungsi `corr_fn(B_oo, B_ov, B_vv, eps_o, eps_v)` yang
    mengembalikan `(e_corr, gamma_oo, gamma_vv, Z_vo)`.
    """
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
        P_ref = 2.0 * (C_occ @ C_occ.T)
        e_ref = float(0.5 * np.sum(P_ref * (H + F_ao)) + v_nn)

        C, eps, F_vo = K.semicanonicalize_subspaces(C, F_ao, n_occ)
        C_occ = np.ascontiguousarray(C[:, :n_occ])
        C_vir = np.ascontiguousarray(C[:, n_occ:])
        eps_o, eps_v = eps[:n_occ], eps[n_occ:]

        B_oo = K.transform_3c_mo(B, C_occ, C_occ, nbf)
        B_ov = K.transform_3c_mo(B, C_occ, C_vir, nbf)
        B_vv = K.transform_3c_mo(B, C_vir, C_vir, nbf)

        e_corr, g_oo, g_vv, Z_vo = corr_fn(B_oo, B_ov, B_vv, eps_o, eps_v)
        e_tot = e_ref + float(e_corr)

        P_corr_a = C_occ @ g_oo @ C_occ.T + C_vir @ g_vv @ C_vir.T
        G_ao = 2.0 * K.build_coulomb_3c(B, P_corr_a, nbf) - K.build_exchange_3c_sym(B, P_corr_a, nbf)
        G_vo = C_vir.T @ G_ao @ C_occ
        L_sep = F_vo @ g_oo - g_vv @ F_vo
        w_vo = 4.0 * (F_vo + G_vo + L_sep + Z_vo)
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

        if it > 0 and e_tot > e_prev + 1e-7:
            trust_radius = max(0.05, 0.5 * trust_radius)
        elif it > 0:
            trust_radius = min(0.30, 1.25 * trust_radius)
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
