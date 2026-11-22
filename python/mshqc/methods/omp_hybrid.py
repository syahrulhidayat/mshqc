# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
Orkestrasi Python bersih berbasis C++20 `mshqc.kernels` untuk keluarga metode
Orbital-Optimized Perturbation Theory (RHF & UHF, all-electron & strict frozen-core):
  - OMP2, SCS-OMP2 (c_os=1.2, c_ss=1/3), SOS-OMP2 (c_os=1.3, c_ss=0.0)
  - OMP2.5 (lam_mp3=0.5), OMP3 (lam_mp3=1.0)
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.linalg import eigh, expm

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
    n_frozen: int = 0

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


def semicanonicalize_subspaces_frozen(
    C: np.ndarray,
    F_ao: np.ndarray,
    n_occ: int,
    n_frozen: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Semikanonikalisasi subruang (frozen / active-occ / virtual) yang identik
    dengan OMP2::pseudocanonicalize() di C++. Mengembalikan (C_sc, eps, F_vo_act).
    """
    nf = max(0, min(int(n_frozen), int(n_occ)))
    if nf == 0:
        return K.semicanonicalize_subspaces(C, F_ao, n_occ)

    nbf = C.shape[0]
    n_act = n_occ - nf
    n_vir = nbf - n_occ
    C_out = C.copy()
    eps = np.zeros(nbf, dtype=np.float64)

    C_frz = C[:, :nf]
    F_ff = C_frz.T @ F_ao @ C_frz
    eps_f, U_f = eigh(0.5 * (F_ff + F_ff.T))
    C_out[:, :nf] = C_frz @ U_f
    eps[:nf] = eps_f

    if n_act > 0:
        C_act = C[:, nf:n_occ]
        F_aa = C_act.T @ F_ao @ C_act
        eps_a, U_a = eigh(0.5 * (F_aa + F_aa.T))
        C_out[:, nf:n_occ] = C_act @ U_a
        eps[nf:n_occ] = eps_a

    if n_vir > 0:
        C_vir = C[:, n_occ:]
        F_vv = C_vir.T @ F_ao @ C_vir
        eps_v, U_v = eigh(0.5 * (F_vv + F_vv.T))
        C_out[:, n_occ:] = C_vir @ U_v
        eps[n_occ:] = eps_v

    F_vo_act = (
        C_out[:, n_occ:].T @ F_ao @ C_out[:, nf:n_occ]
        if (n_act > 0 and n_vir > 0)
        else np.zeros((n_vir, n_act), dtype=np.float64)
    )
    return C_out, eps, F_vo_act


def _rotate_orbitals(
    C: np.ndarray,
    kappa_vo: np.ndarray,
    n_occ: int,
    n_frozen: int = 0,
) -> np.ndarray:
    """Rotasi uniter eksponensial hanya pada subruang aktif+virtual (menjaga inti beku mutlak)."""
    nf = max(0, min(int(n_frozen), int(n_occ)))
    nbf = C.shape[0]
    n_act = n_occ - nf
    if n_act <= 0 or kappa_vo.size == 0:
        return C
    n_sub = nbf - nf
    K_rot = np.zeros((n_sub, n_sub), dtype=np.float64)
    K_rot[n_act:, :n_act] = kappa_vo
    K_rot[:n_act, n_act:] = -kappa_vo.T
    if nf == 0:
        return C @ expm(K_rot)
    C_out = C.copy()
    C_out[:, nf:] = C[:, nf:] @ expm(K_rot)
    return C_out


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
    C_act: np.ndarray,
    C_vir: np.ndarray,
    F_vo: np.ndarray,
    g_oo: np.ndarray,
    g_vv: np.ndarray,
    Z_vo: np.ndarray,
) -> np.ndarray:
    nbf = C_act.shape[0]
    P_corr_a = C_act @ g_oo @ C_act.T + C_vir @ g_vv @ C_vir.T
    G_ao = 2.0 * K.build_coulomb_3c(B_ao, P_corr_a, nbf) - K.build_exchange_3c_sym(B_ao, P_corr_a, nbf)
    G_vo = C_vir.T @ G_ao @ C_act
    L_sep = F_vo @ g_oo - g_vv @ F_vo
    return 4.0 * (F_vo + G_vo + L_sep + Z_vo)


def _assemble_orbital_grad_uhf(
    B_ao: np.ndarray,
    Ca_act: np.ndarray,
    Ca_v: np.ndarray,
    Cb_act: np.ndarray,
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
    nbf = Ca_act.shape[0]
    na_act = Ca_act.shape[1]
    nb_act = Cb_act.shape[1]

    P_corr_a = (
        Ca_act @ g_oo_a @ Ca_act.T + Ca_v @ g_vv_a @ Ca_v.T
        if (na_act > 0 and Ca_v.shape[1] > 0)
        else np.zeros((nbf, nbf), dtype=np.float64)
    )
    P_corr_b = (
        Cb_act @ g_oo_b @ Cb_act.T + Cb_v @ g_vv_b @ Cb_v.T
        if (nb_act > 0 and Cb_v.shape[1] > 0)
        else np.zeros((nbf, nbf), dtype=np.float64)
    )
    J_corr = K.build_coulomb_3c(B_ao, P_corr_a + P_corr_b, nbf)
    Ga_ao = J_corr - K.build_exchange_3c_sym(B_ao, P_corr_a, nbf)
    Gb_ao = J_corr - K.build_exchange_3c_sym(B_ao, P_corr_b, nbf)

    wa_vo = (
        2.0 * (Fvo_a + Ca_v.T @ Ga_ao @ Ca_act + (Fvo_a @ g_oo_a - g_vv_a @ Fvo_a) + Za)
        if na_act > 0
        else np.zeros_like(Fvo_a)
    )
    wb_vo = (
        2.0 * (Fvo_b + Cb_v.T @ Gb_ao @ Cb_act + (Fvo_b @ g_oo_b - g_vv_b @ Fvo_b) + Zb)
        if nb_act > 0
        else np.zeros_like(Fvo_b)
    )
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
    n_frozen: int = 0,
):
    """Evaluasi 1 titik energi & gradien analitik keluarga DF-OMP RHF (stateless, fused MO transform)."""
    nbf = H_core.shape[0]
    n_vir = nbf - n_occ
    nf = max(0, min(int(n_frozen), int(n_occ)))
    n_act = n_occ - nf

    C_occ_all = np.ascontiguousarray(C[:, :n_occ])
    J, K_mat = K.build_jk_3c_rhf(B_ao, C_occ_all, nbf)
    F_ao = H_core + J - K_mat
    P_ref = 2.0 * (C_occ_all @ C_occ_all.T)
    e_ref = float(0.5 * np.sum(P_ref * (H_core + F_ao)) + v_nn)

    C_sc, eps, F_vo = semicanonicalize_subspaces_frozen(C, F_ao, n_occ, nf)
    if n_act <= 0 or n_vir <= 0:
        w_empty = np.zeros((n_vir, n_act), dtype=np.float64)
        return e_ref, e_ref, 0.0, 0.0, C_sc, eps, w_empty, None

    C_act = np.ascontiguousarray(C_sc[:, nf:n_occ])
    C_vir = np.ascontiguousarray(C_sc[:, n_occ:])
    eps_o, eps_v = eps[nf:n_occ], eps[n_occ:]

    # 1x AO unpack terpadu untuk seluruh blok (B_oo, B_ov, B_vv)
    B_oo, B_ov, B_vv = K.transform_3c_mo_blocks(B_ao, C_act, C_vir, nbf)

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
        g_oo, g_vv = K.eval_omp2_1rdm_rhf(T2, n_act, n_vir, c_os=c_os, c_ss=c_ss)
        Z_vo = K.eval_omp2_zvec_rhf_df(T2, B_oo, B_ov, B_vv, n_act, n_vir, c_os=c_os, c_ss=c_ss)
    elif lam_mp3 == 1.0:
        g_oo, g_vv = K.eval_omp3_1rdm_rhf(T2, L2, n_act, n_vir)
        Z_vo = K.eval_omp3_zvec_rhf_df(T2, L2, B_oo, B_ov, B_vv, n_act, n_vir)
    else:
        g_oo, g_vv = K.eval_omp3_1rdm_rhf(T2, np.ascontiguousarray(lam_mp3 * L2), n_act, n_vir)
        Z2 = K.eval_omp2_zvec_rhf_df(T2, B_oo, B_ov, B_vv, n_act, n_vir)
        Z3 = K.eval_omp3_zvec_rhf_df(T2, L2, B_oo, B_ov, B_vv, n_act, n_vir)
        Z_vo = (1.0 - lam_mp3) * Z2 + lam_mp3 * Z3

    w_vo = _assemble_orbital_grad_rhf(B_ao, C_act, C_vir, F_vo, g_oo, g_vv, Z_vo)
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
    n_frozen: int = 0,
) -> HybridOMPResult:
    """Jalankan optimisasi orbital DF-OMP RHF (OMP2, SCS-OMP2, SOS-OMP2, OMP2.5, OMP3)."""
    C = np.array(C0, dtype=np.float64, copy=True)
    nf = max(0, min(int(n_frozen), int(n_occ)))
    e_prev = 0.0
    trust_radius = 0.25

    for it in range(max_iter):
        e_tot, e_ref, e_mp2, e_mp3, C, eps, w_vo, aux = eval_omp_rhf_point(
            H_core, v_nn, B_ao, C, n_occ,
            c_os=c_os, c_ss=c_ss, lam_mp3=lam_mp3, need_grad=True, n_frozen=nf,
        )
        w_rms = float(np.sqrt(np.mean(w_vo ** 2))) if w_vo.size > 0 else 0.0
        if w_rms <= grad_tol and (it == 0 or abs(e_tot - e_prev) <= energy_tol):
            return HybridOMPResult(
                method=method_name, reference="rhf",
                energy_scf=e_ref, energy_mp2_corr=e_mp2, energy_mp3_corr=e_mp3,
                energy_total=e_tot, converged=True, iterations=it, grad_rms=w_rms, s2=0.0,
                C_alpha=C, C_beta=C.copy(),
                orbital_energies_alpha=eps, orbital_energies_beta=eps.copy(),
                n_occ_alpha=n_occ, n_occ_beta=n_occ, n_frozen=nf,
            )

        trust_radius = _update_trust_radius(trust_radius, it, e_tot, e_prev)
        e_prev = e_tot
        eps_o, eps_v, B_oo, B_ov, B_vv = aux
        kappa = solve_tr_soscf_rhf(w_vo, eps_o, eps_v, B_oo, B_ov, B_vv, trust_radius=trust_radius)
        C = _rotate_orbitals(C, kappa, n_occ, n_frozen=nf)

    return HybridOMPResult(
        method=method_name, reference="rhf",
        energy_scf=e_ref, energy_mp2_corr=e_mp2, energy_mp3_corr=e_mp3,
        energy_total=e_tot, converged=False, iterations=max_iter, grad_rms=w_rms, s2=0.0,
        C_alpha=C, C_beta=C.copy(),
        orbital_energies_alpha=eps, orbital_energies_beta=eps.copy(),
        n_occ_alpha=n_occ, n_occ_beta=n_occ, n_frozen=nf,
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
    n_frozen: int = 0,
):
    """Evaluasi 1 titik energi & gradien analitik keluarga DF-OMP UHF (stateless, fused MO transform)."""
    nbf = H_core.shape[0]
    va = nbf - na
    vb = nbf - nb
    nf_a = max(0, min(int(n_frozen), int(na)))
    nf_b = max(0, min(int(n_frozen), int(nb)))
    na_act = na - nf_a
    nb_act = nb - nf_b

    Ca_all = np.ascontiguousarray(C_a[:, :na])
    Cb_all = np.ascontiguousarray(C_b[:, :nb])
    J, Ka, Kb = K.build_jk_3c_uhf(B_ao, Ca_all, Cb_all, nbf)
    Fa_ao = H_core + J - Ka
    Fb_ao = H_core + J - Kb
    Pa_ref = Ca_all @ Ca_all.T if na > 0 else np.zeros((nbf, nbf), dtype=np.float64)
    Pb_ref = Cb_all @ Cb_all.T if nb > 0 else np.zeros((nbf, nbf), dtype=np.float64)
    e_ref = float(0.5 * (np.sum(Pa_ref * (H_core + Fa_ao)) + np.sum(Pb_ref * (H_core + Fb_ao))) + v_nn)

    Ca_sc, eps_a, Fvo_a = semicanonicalize_subspaces_frozen(C_a, Fa_ao, na, nf_a)
    Cb_sc, eps_b, Fvo_b = semicanonicalize_subspaces_frozen(C_b, Fb_ao, nb, nf_b)

    Ca_act = np.ascontiguousarray(Ca_sc[:, nf_a:na])
    Ca_v = np.ascontiguousarray(Ca_sc[:, na:])
    Cb_act = np.ascontiguousarray(Cb_sc[:, nf_b:nb])
    Cb_v = np.ascontiguousarray(Cb_sc[:, nb:])
    eps_oa, eps_va = eps_a[nf_a:na], eps_a[na:]
    eps_ob, eps_vb = eps_b[nf_b:nb], eps_b[nb:]

    # 1x AO unpack terpadu untuk seluruh 6 blok spin alpha + beta
    B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b = K.transform_3c_mo_blocks_uhf(
        B_ao, Ca_act, Ca_v, Cb_act, Cb_v, nbf
    )

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
            T2_aa, T2_bb, T2_ab, na_act, va, nb_act, vb, c_os=c_os, c_ss=c_ss
        )
        Za, Zb = K.eval_omp2_zvec_uhf_df(
            T2_aa, T2_bb, T2_ab,
            B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b,
            na_act, va, nb_act, vb, c_os=c_os, c_ss=c_ss,
        )
    elif lam_mp3 == 1.0:
        g_oo_a, g_vv_a, g_oo_b, g_vv_b = K.eval_omp3_1rdm_uhf(
            T2_aa, T2_bb, T2_ab, L2_aa, L2_bb, L2_ab, na_act, va, nb_act, vb
        )
        Za, Zb = K.eval_omp3_zvec_uhf_df(
            T2_aa, T2_bb, T2_ab, L2_aa, L2_bb, L2_ab,
            B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b,
            na_act, va, nb_act, vb,
        )
    else:
        g_oo_a, g_vv_a, g_oo_b, g_vv_b = K.eval_omp3_1rdm_uhf(
            T2_aa, T2_bb, T2_ab,
            np.ascontiguousarray(lam_mp3 * L2_aa),
            np.ascontiguousarray(lam_mp3 * L2_bb),
            np.ascontiguousarray(lam_mp3 * L2_ab),
            na_act, va, nb_act, vb,
        )
        Z2a, Z2b = K.eval_omp2_zvec_uhf_df(
            T2_aa, T2_bb, T2_ab,
            B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b,
            na_act, va, nb_act, vb,
        )
        Z3a, Z3b = K.eval_omp3_zvec_uhf_df(
            T2_aa, T2_bb, T2_ab, L2_aa, L2_bb, L2_ab,
            B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b,
            na_act, va, nb_act, vb,
        )
        w2, w3 = 1.0 - lam_mp3, lam_mp3
        Za, Zb = w2 * Z2a + w3 * Z3a, w2 * Z2b + w3 * Z3b

    wa_vo, wb_vo = _assemble_orbital_grad_uhf(
        B_ao, Ca_act, Ca_v, Cb_act, Cb_v, Fvo_a, Fvo_b, g_oo_a, g_vv_a, g_oo_b, g_vv_b, Za, Zb
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
    n_frozen: int = 0,
) -> HybridOMPResult:
    """Jalankan optimisasi orbital DF-OMP UHF (radikal kulit terbuka) + evaluasi <S^2>."""
    Ca = np.array(C0_a, dtype=np.float64, copy=True)
    Cb = np.array(C0_b, dtype=np.float64, copy=True)
    nf_a = max(0, min(int(n_frozen), int(na)))
    nf_b = max(0, min(int(n_frozen), int(nb)))
    e_prev = 0.0
    trust_radius = 0.25
    n_params = max(1, (H_core.shape[0] - na) * (na - nf_a) + (H_core.shape[0] - nb) * (nb - nf_b))

    for it in range(max_iter):
        e_tot, e_ref, e_mp2, e_mp3, Ca, Cb, eps_a, eps_b, wa_vo, wb_vo, aux = eval_omp_uhf_point(
            H_core, v_nn, B_ao, Ca, Cb, na, nb,
            c_os=c_os, c_ss=c_ss, lam_mp3=lam_mp3, need_grad=True, n_frozen=n_frozen,
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
                n_occ_alpha=na, n_occ_beta=nb, n_frozen=nf_a,
            )

        trust_radius = _update_trust_radius(trust_radius, it, e_tot, e_prev)
        e_prev = e_tot
        ka, kb = solve_tr_soscf_uhf(wa_vo, wb_vo, *aux, trust_radius=trust_radius)
        Ca = _rotate_orbitals(Ca, ka, na, n_frozen=nf_a)
        if nb - nf_b > 0:
            Cb = _rotate_orbitals(Cb, kb, nb, n_frozen=nf_b)

    s2 = _compute_s2_uhf(Ca, Cb, S_ovlp, na, nb)
    return HybridOMPResult(
        method=method_name, reference="uhf",
        energy_scf=e_ref, energy_mp2_corr=e_mp2, energy_mp3_corr=e_mp3,
        energy_total=e_tot, converged=False, iterations=max_iter, grad_rms=w_rms, s2=s2,
        C_alpha=Ca, C_beta=Cb,
        orbital_energies_alpha=eps_a, orbital_energies_beta=eps_b,
        n_occ_alpha=na, n_occ_beta=nb, n_frozen=nf_a,
    )
