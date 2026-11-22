# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
Orkestrasi Python bersih berbasis C++20 `mshqc.kernels` untuk keluarga metode
Orbital-Optimized Perturbation Theory (RHF & UHF):
  - OMP2, SCS-OMP2 (c_os=1.2, c_ss=1/3), SOS-OMP2 (c_os=1.3, c_ss=0.0)
  - OMP2.5 (lam_mp3=0.5), OMP3 (lam_mp3=1.0)
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.linalg import expm

from .._mshqc import kernels as K
from .tr_soscf import solve_tr_soscf_rhf, solve_tr_soscf_uhf


@dataclass
class HybridOMPResult:
    method: str
    reference: str
    energy_scf: float
    energy_mp2_corr: float
    energy_mp3_corr: float
    energy_total: float
    converged: bool
    iterations: int
    grad_rms: float
    s2: float
    C_alpha: np.ndarray
    C_beta: np.ndarray
    orbital_energies_alpha: np.ndarray
    orbital_energies_beta: np.ndarray
    n_occ_alpha: int
    n_occ_beta: int

    @property
    def e_total(self) -> float:
        return self.energy_total

    @property
    def e_hf(self) -> float:
        return self.energy_scf

    @property
    def e_mp2(self) -> float:
        return self.energy_mp2_corr

    @property
    def e_mp3(self) -> float:
        return self.energy_mp3_corr

    @property
    def e_corr_total(self) -> float:
        return self.energy_mp2_corr + self.energy_mp3_corr


def _rotate_orbitals(C: np.ndarray, kappa_vo: np.ndarray, n_occ: int) -> np.ndarray:
    nbf = C.shape[0]
    K_rot = np.zeros((nbf, nbf), dtype=np.float64)
    K_rot[n_occ:, :n_occ] = kappa_vo
    K_rot[:n_occ, n_occ:] = -kappa_vo.T
    return C @ expm(K_rot)


def _compute_s2_uhf(C_a: np.ndarray, C_b: np.ndarray, S: np.ndarray, na: int, nb: int) -> float:
    sz = 0.5 * (na - nb)
    if nb == 0:
        return float(sz * (sz + 1.0))
    ovlp = C_a[:, :na].T @ S @ C_b[:, :nb]
    return float(sz * (sz + 1.0) + nb - np.sum(ovlp ** 2))


def _update_trust_radius(trust_radius: float, it: int, e_tot: float, e_prev: float) -> float:
    if it == 0:
        return trust_radius
    if e_tot > e_prev + 1e-7:
        return max(0.05, 0.5 * trust_radius)
    return min(0.30, 1.25 * trust_radius)


def _assemble_orbital_grad_rhf(
    B_ao: np.ndarray,
    C_occ: np.ndarray,
    C_vir: np.ndarray,
    F_vo: np.ndarray,
    g_oo: np.ndarray,
    g_vv: np.ndarray,
    Z_vo: np.ndarray,
) -> np.ndarray:
    nbf = C_occ.shape[0]
    P_corr_a = C_occ @ g_oo @ C_occ.T + C_vir @ g_vv @ C_vir.T
    G_ao = 2.0 * K.build_coulomb_3c(B_ao, P_corr_a, nbf) - K.build_exchange_3c_sym(B_ao, P_corr_a, nbf)
    G_vo = C_vir.T @ G_ao @ C_occ
    L_sep = F_vo @ g_oo - g_vv @ F_vo
    return 4.0 * (F_vo + G_vo + L_sep + Z_vo)


def _assemble_orbital_grad_uhf(
    B_ao: np.ndarray,
    Ca_o: np.ndarray,
    Ca_v: np.ndarray,
    Cb_o: np.ndarray,
    Cb_v: np.ndarray,
    Fvo_a: np.ndarray,
    Fvo_b: np.ndarray,
    g_oo_a: np.ndarray,
    g_vv_a: np.ndarray,
    g_oo_b: np.ndarray,
    g_vv_b: np.ndarray,
    Za: np.ndarray,
    Zb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    nbf = Ca_o.shape[0]
    P_corr_a = Ca_o @ g_oo_a @ Ca_o.T + Ca_v @ g_vv_a @ Ca_v.T
    P_corr_b = Cb_o @ g_oo_b @ Cb_o.T + Cb_v @ g_vv_b @ Cb_v.T
    J_corr = K.build_coulomb_3c(B_ao, P_corr_a + P_corr_b, nbf)
    Ga_ao = J_corr - K.build_exchange_3c_sym(B_ao, P_corr_a, nbf)
    Gb_ao = J_corr - K.build_exchange_3c_sym(B_ao, P_corr_b, nbf)

    wa_vo = 2.0 * (Fvo_a + Ca_v.T @ Ga_ao @ Ca_o + (Fvo_a @ g_oo_a - g_vv_a @ Fvo_a) + Za)
    wb_vo = 2.0 * (Fvo_b + Cb_v.T @ Gb_ao @ Cb_o + (Fvo_b @ g_oo_b - g_vv_b @ Fvo_b) + Zb)
    return wa_vo, wb_vo


def eval_omp_rhf_point(
    H_core: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    C: np.ndarray,
    n_occ: int,
    c_os: float = 1.0,
    c_ss: float = 1.0,
    lam_mp3: float = 0.0,
    need_grad: bool = True,
):
    """Evaluasi 1 titik energi & gradien analitik keluarga DF-OMP RHF (stateless)."""
    nbf = H_core.shape[0]
    n_vir = nbf - n_occ

    C_occ = np.ascontiguousarray(C[:, :n_occ])
    J, K_mat = K.build_jk_3c_rhf(B_ao, C_occ, nbf)
    F_ao = H_core + J - K_mat
    P_ref = 2.0 * (C_occ @ C_occ.T)
    e_ref = float(0.5 * np.sum(P_ref * (H_core + F_ao)) + v_nn)

    C_sc, eps, F_vo = K.semicanonicalize_subspaces(C, F_ao, n_occ)
    C_occ = np.ascontiguousarray(C_sc[:, :n_occ])
    C_vir = np.ascontiguousarray(C_sc[:, n_occ:])
    eps_o, eps_v = eps[:n_occ], eps[n_occ:]

    B_oo = K.transform_3c_mo(B_ao, C_occ, C_occ, nbf)
    B_ov = K.transform_3c_mo(B_ao, C_occ, C_vir, nbf)
    B_vv = K.transform_3c_mo(B_ao, C_vir, C_vir, nbf)

    e_os, e_ss, T2 = K.eval_rmp2_df(B_ov, eps_o, eps_v)
    e_mp2 = c_os * e_os + c_ss * e_ss

    e_mp3, L2 = 0.0, None
    if lam_mp3 != 0.0:
        if c_os != 1.0 or c_ss != 1.0:
            raise ValueError("Spin-scaling (c_os, c_ss) bersama lam_mp3 != 0 belum diaktifkan di RHF")
        e_mp3_raw, L2 = K.eval_rmp3_df(T2, B_oo, B_ov, B_vv, eps_o, eps_v)
        e_mp3 = lam_mp3 * e_mp3_raw

    e_tot = e_ref + e_mp2 + e_mp3
    if not need_grad:
        return e_tot, e_ref, e_mp2, e_mp3, C_sc, eps, None, None

    if lam_mp3 == 0.0:
        g_oo, g_vv = K.eval_omp2_1rdm_rhf(T2, n_occ, n_vir, c_os=c_os, c_ss=c_ss)
        Z_vo = K.eval_omp2_zvec_rhf_df(T2, B_oo, B_ov, B_vv, n_occ, n_vir, c_os=c_os, c_ss=c_ss)
    elif lam_mp3 == 1.0:
        g_oo, g_vv = K.eval_omp3_1rdm_rhf(T2, L2, n_occ, n_vir)
        Z_vo = K.eval_omp3_zvec_rhf_df(T2, L2, B_oo, B_ov, B_vv, n_occ, n_vir)
    else:
        # Linearitas eksak 1-RDM terhadap L2 menghemat 1x panggilan 1-RDM penuh untuk OMP2.5
        g_oo, g_vv = K.eval_omp3_1rdm_rhf(T2, np.ascontiguousarray(lam_mp3 * L2), n_occ, n_vir)
        Z2 = K.eval_omp2_zvec_rhf_df(T2, B_oo, B_ov, B_vv, n_occ, n_vir)
        Z3 = K.eval_omp3_zvec_rhf_df(T2, L2, B_oo, B_ov, B_vv, n_occ, n_vir)
        Z_vo = (1.0 - lam_mp3) * Z2 + lam_mp3 * Z3

    w_vo = _assemble_orbital_grad_rhf(B_ao, C_occ, C_vir, F_vo, g_oo, g_vv, Z_vo)
    return e_tot, e_ref, e_mp2, e_mp3, C_sc, eps, w_vo, (eps_o, eps_v, B_oo, B_ov, B_vv)


def run_df_omp_rhf(
    H_core: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    C0: np.ndarray,
    n_occ: int,
    method_name: str = "omp2",
    c_os: float = 1.0,
    c_ss: float = 1.0,
    lam_mp3: float = 0.0,
    grad_tol: float = 1e-5,
    energy_tol: float = 1e-9,
    max_iter: int = 40,
) -> HybridOMPResult:
    """Jalankan optimisasi orbital DF-OMP RHF (OMP2, SCS-OMP2, SOS-OMP2, OMP2.5, OMP3)."""
    C = np.array(C0, dtype=np.float64, copy=True)
    e_prev = 0.0
    trust_radius = 0.25

    for it in range(max_iter):
        e_tot, e_ref, e_mp2, e_mp3, C, eps, w_vo, aux = eval_omp_rhf_point(
            H_core, v_nn, B_ao, C, n_occ, c_os=c_os, c_ss=c_ss, lam_mp3=lam_mp3, need_grad=True
        )
        w_rms = float(np.sqrt(np.mean(w_vo ** 2)))
        if w_rms <= grad_tol and (it == 0 or abs(e_tot - e_prev) <= energy_tol):
            return HybridOMPResult(
                method=method_name, reference="rhf",
                energy_scf=e_ref, energy_mp2_corr=e_mp2, energy_mp3_corr=e_mp3,
                energy_total=e_tot, converged=True, iterations=it, grad_rms=w_rms, s2=0.0,
                C_alpha=C, C_beta=C.copy(),
                orbital_energies_alpha=eps, orbital_energies_beta=eps.copy(),
                n_occ_alpha=n_occ, n_occ_beta=n_occ,
            )

        trust_radius = _update_trust_radius(trust_radius, it, e_tot, e_prev)
        e_prev = e_tot
        eps_o, eps_v, B_oo, B_ov, B_vv = aux
        kappa = solve_tr_soscf_rhf(w_vo, eps_o, eps_v, B_oo, B_ov, B_vv, trust_radius=trust_radius)
        C = _rotate_orbitals(C, kappa, n_occ)

    return HybridOMPResult(
        method=method_name, reference="rhf",
        energy_scf=e_ref, energy_mp2_corr=e_mp2, energy_mp3_corr=e_mp3,
        energy_total=e_tot, converged=False, iterations=max_iter, grad_rms=w_rms, s2=0.0,
        C_alpha=C, C_beta=C.copy(),
        orbital_energies_alpha=eps, orbital_energies_beta=eps.copy(),
        n_occ_alpha=n_occ, n_occ_beta=n_occ,
    )


def eval_omp_uhf_point(
    H_core: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    C_a: np.ndarray,
    C_b: np.ndarray,
    na: int,
    nb: int,
    c_os: float = 1.0,
    c_ss: float = 1.0,
    lam_mp3: float = 0.0,
    need_grad: bool = True,
):
    """Evaluasi 1 titik energi & gradien analitik keluarga DF-OMP UHF (stateless)."""
    nbf = H_core.shape[0]
    va = nbf - na
    vb = nbf - nb

    Ca_o = np.ascontiguousarray(C_a[:, :na])
    Cb_o = np.ascontiguousarray(C_b[:, :nb])
    J, Ka, Kb = K.build_jk_3c_uhf(B_ao, Ca_o, Cb_o, nbf)
    Fa_ao = H_core + J - Ka
    Fb_ao = H_core + J - Kb
    Pa_ref = Ca_o @ Ca_o.T
    Pb_ref = Cb_o @ Cb_o.T
    e_ref = float(0.5 * (np.sum(Pa_ref * (H_core + Fa_ao)) + np.sum(Pb_ref * (H_core + Fb_ao))) + v_nn)

    Ca_sc, eps_a, Fvo_a = K.semicanonicalize_subspaces(C_a, Fa_ao, na)
    Cb_sc, eps_b, Fvo_b = K.semicanonicalize_subspaces(C_b, Fb_ao, nb)
    Ca_o, Ca_v = np.ascontiguousarray(Ca_sc[:, :na]), np.ascontiguousarray(Ca_sc[:, na:])
    Cb_o, Cb_v = np.ascontiguousarray(Cb_sc[:, :nb]), np.ascontiguousarray(Cb_sc[:, nb:])
    eps_oa, eps_va = eps_a[:na], eps_a[na:]
    eps_ob, eps_vb = eps_b[:nb], eps_b[nb:]

    B_oo_a = K.transform_3c_mo(B_ao, Ca_o, Ca_o, nbf)
    B_ov_a = K.transform_3c_mo(B_ao, Ca_o, Ca_v, nbf)
    B_vv_a = K.transform_3c_mo(B_ao, Ca_v, Ca_v, nbf)
    B_oo_b = K.transform_3c_mo(B_ao, Cb_o, Cb_o, nbf)
    B_ov_b = K.transform_3c_mo(B_ao, Cb_o, Cb_v, nbf)
    B_vv_b = K.transform_3c_mo(B_ao, Cb_v, Cb_v, nbf)

    e_aa, e_bb, e_ab, T2_aa, T2_bb, T2_ab = K.eval_ump2_df(
        B_ov_a, B_ov_b, eps_oa, eps_va, eps_ob, eps_vb
    )
    e_mp2 = c_ss * (e_aa + e_bb) + c_os * e_ab

    e_mp3 = 0.0
    L2_aa = L2_bb = L2_ab = None
    if lam_mp3 != 0.0:
        if c_os != 1.0 or c_ss != 1.0:
            raise ValueError("Spin-scaling (c_os, c_ss) bersama lam_mp3 != 0 belum diaktifkan di UHF")
        e3_aa, e3_bb, e3_ab, L2_aa, L2_bb, L2_ab = K.eval_ump3_df(
            T2_aa, T2_bb, T2_ab,
            B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b,
            eps_oa, eps_va, eps_ob, eps_vb,
        )
        e_mp3 = lam_mp3 * (e3_aa + e3_bb + e3_ab)

    e_tot = e_ref + e_mp2 + e_mp3
    if not need_grad:
        return e_tot, e_ref, e_mp2, e_mp3, Ca_sc, Cb_sc, eps_a, eps_b, None, None, None

    if lam_mp3 == 0.0:
        g_oo_a, g_vv_a, g_oo_b, g_vv_b = K.eval_omp2_1rdm_uhf(
            T2_aa, T2_bb, T2_ab, na, va, nb, vb, c_os=c_os, c_ss=c_ss
        )
        Za, Zb = K.eval_omp2_zvec_uhf_df(
            T2_aa, T2_bb, T2_ab,
            B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b,
            na, va, nb, vb, c_os=c_os, c_ss=c_ss,
        )
    elif lam_mp3 == 1.0:
        g_oo_a, g_vv_a, g_oo_b, g_vv_b = K.eval_omp3_1rdm_uhf(
            T2_aa, T2_bb, T2_ab, L2_aa, L2_bb, L2_ab, na, va, nb, vb
        )
        Za, Zb = K.eval_omp3_zvec_uhf_df(
            T2_aa, T2_bb, T2_ab, L2_aa, L2_bb, L2_ab,
            B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b,
            na, va, nb, vb,
        )
    else:
        g_oo_a, g_vv_a, g_oo_b, g_vv_b = K.eval_omp3_1rdm_uhf(
            T2_aa, T2_bb, T2_ab,
            np.ascontiguousarray(lam_mp3 * L2_aa),
            np.ascontiguousarray(lam_mp3 * L2_bb),
            np.ascontiguousarray(lam_mp3 * L2_ab),
            na, va, nb, vb,
        )
        Z2a, Z2b = K.eval_omp2_zvec_uhf_df(
            T2_aa, T2_bb, T2_ab, B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b, na, va, nb, vb
        )
        Z3a, Z3b = K.eval_omp3_zvec_uhf_df(
            T2_aa, T2_bb, T2_ab, L2_aa, L2_bb, L2_ab,
            B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b, na, va, nb, vb,
        )
        w2, w3 = 1.0 - lam_mp3, lam_mp3
        Za, Zb = w2 * Z2a + w3 * Z3a, w2 * Z2b + w3 * Z3b

    wa_vo, wb_vo = _assemble_orbital_grad_uhf(
        B_ao, Ca_o, Ca_v, Cb_o, Cb_v, Fvo_a, Fvo_b, g_oo_a, g_vv_a, g_oo_b, g_vv_b, Za, Zb
    )
    aux = (eps_oa, eps_va, eps_ob, eps_vb, B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b)
    return e_tot, e_ref, e_mp2, e_mp3, Ca_sc, Cb_sc, eps_a, eps_b, wa_vo, wb_vo, aux


def run_df_omp_uhf(
    H_core: np.ndarray,
    S_ovlp: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    C0_a: np.ndarray,
    C0_b: np.ndarray,
    na: int,
    nb: int,
    method_name: str = "omp2",
    c_os: float = 1.0,
    c_ss: float = 1.0,
    lam_mp3: float = 0.0,
    grad_tol: float = 1e-5,
    energy_tol: float = 1e-9,
    max_iter: int = 40,
) -> HybridOMPResult:
    """Jalankan optimisasi orbital DF-OMP UHF (radikal kulit terbuka) + evaluasi <S^2>."""
    Ca = np.array(C0_a, dtype=np.float64, copy=True)
    Cb = np.array(C0_b, dtype=np.float64, copy=True)
    e_prev = 0.0
    trust_radius = 0.25
    n_params = max(1, (H_core.shape[0] - na) * na + (H_core.shape[0] - nb) * nb)

    for it in range(max_iter):
        e_tot, e_ref, e_mp2, e_mp3, Ca, Cb, eps_a, eps_b, wa_vo, wb_vo, aux = eval_omp_uhf_point(
            H_core, v_nn, B_ao, Ca, Cb, na, nb,
            c_os=c_os, c_ss=c_ss, lam_mp3=lam_mp3, need_grad=True,
        )
        w_rms = float(np.sqrt((np.sum(wa_vo ** 2) + np.sum(wb_vo ** 2)) / n_params))
        if w_rms <= grad_tol and (it == 0 or abs(e_tot - e_prev) <= energy_tol):
            s2 = _compute_s2_uhf(Ca, Cb, S_ovlp, na, nb)
            return HybridOMPResult(
                method=method_name, reference="uhf",
                energy_scf=e_ref, energy_mp2_corr=e_mp2, energy_mp3_corr=e_mp3,
                energy_total=e_tot, converged=True, iterations=it, grad_rms=w_rms, s2=s2,
                C_alpha=Ca, C_beta=Cb,
                orbital_energies_alpha=eps_a, orbital_energies_beta=eps_b,
                n_occ_alpha=na, n_occ_beta=nb,
            )

        trust_radius = _update_trust_radius(trust_radius, it, e_tot, e_prev)
        e_prev = e_tot
        ka, kb = solve_tr_soscf_uhf(wa_vo, wb_vo, *aux, trust_radius=trust_radius)
        Ca = _rotate_orbitals(Ca, ka, na)
        if nb > 0:
            Cb = _rotate_orbitals(Cb, kb, nb)

    s2 = _compute_s2_uhf(Ca, Cb, S_ovlp, na, nb)
    return HybridOMPResult(
        method=method_name, reference="uhf",
        energy_scf=e_ref, energy_mp2_corr=e_mp2, energy_mp3_corr=e_mp3,
        energy_total=e_tot, converged=False, iterations=max_iter, grad_rms=w_rms, s2=s2,
        C_alpha=Ca, C_beta=Cb,
        orbital_energies_alpha=eps_a, orbital_energies_beta=eps_b,
        n_occ_alpha=na, n_occ_beta=nb,
    )
