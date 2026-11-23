# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
Orkestrasi Python tipis untuk MP2 & MP3 kanonik/semikanonik (RHF, UHF, ROHF; DF & Cholesky,
mendukung strict frozen-core `n_frozen`) di atas kernel C++20 `mshqc.kernels`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .._mshqc import kernels as K
from .omp_hybrid import semicanonicalize_subspaces_frozen


@dataclass
class HybridMP2Result:
    reference: str
    energy_scf: float
    energy_mp2_ss: float
    energy_mp2_os: float
    energy_mp2_corr: float
    energy_total: float
    converged: bool
    iterations: int
    C_alpha: np.ndarray
    C_beta: np.ndarray
    orbital_energies_alpha: np.ndarray
    orbital_energies_beta: np.ndarray
    n_occ_alpha: int
    n_occ_beta: int
    n_virt_alpha: int
    n_virt_beta: int
    n_frozen: int = 0
    energy_mp2_singles: float = 0.0

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
    def e_corr_total(self) -> float:
        return self.energy_mp2_corr

    def __str__(self) -> str:
        return (
            f"<MP2 ({self.reference.upper()}) | E_total = {self.energy_total:.10f} Ha | "
            f"E_corr = {self.energy_mp2_corr:+.10f} Ha>"
        )


@dataclass
class HybridMP3Result:
    reference: str
    e_hf: float
    e_mp2: float
    e_mp3: float
    e3_aa: float
    e3_bb: float
    e3_ab: float
    e_corr_total: float
    e_total: float
    converged: bool
    iterations: int
    C_alpha: np.ndarray
    C_beta: np.ndarray
    n_occ_alpha: int
    n_occ_beta: int
    n_frozen: int = 0

    @property
    def energy_total(self) -> float:
        return self.e_total

    @property
    def energy_scf(self) -> float:
        return self.e_hf

    @property
    def energy_mp2_corr(self) -> float:
        return self.e_mp2

    @property
    def energy_mp3_corr(self) -> float:
        return self.e_mp3

    def __str__(self) -> str:
        return (
            f"<MP3 ({self.reference.upper()}) | E_total = {self.e_total:.10f} Ha | "
            f"E_corr = {self.e_corr_total:+.10f} Ha>"
        )


def _clamp_frozen(n_frozen: int, n_occ: int) -> int:
    return max(0, min(int(n_frozen), int(n_occ)))


def _compute_romp2_singles(
    F_vo_a: np.ndarray,
    F_vo_b: np.ndarray,
    eps_oa: np.ndarray,
    eps_va: np.ndarray,
    eps_ob: np.ndarray,
    eps_vb: np.ndarray,
) -> float:
    """Suku eksitasi tunggal non-Brillouin ROMP2: sum_{i,a} |F_ia|^2 / (eps_i - eps_a)."""
    e_s = 0.0
    if F_vo_a.size > 0:
        denom_a = eps_oa[None, :] - eps_va[:, None]
        mask_a = np.abs(denom_a) > 1.0e-12
        e_s += float(np.sum(np.where(mask_a, (F_vo_a ** 2) / np.where(mask_a, denom_a, 1.0), 0.0)))
    if F_vo_b.size > 0:
        denom_b = eps_ob[None, :] - eps_vb[:, None]
        mask_b = np.abs(denom_b) > 1.0e-12
        e_s += float(np.sum(np.where(mask_b, (F_vo_b ** 2) / np.where(mask_b, denom_b, 1.0), 0.0)))
    return e_s


def run_df_mp2_rhf(
    H_core: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    C0: np.ndarray,
    n_occ: int,
    n_frozen: int = 0,
    c_os: float = 1.0,
    c_ss: float = 1.0,
) -> HybridMP2Result:
    """DF/Cholesky-RMP2 (termasuk SCS/SOS-MP2) berbasis C++20 `mshqc.kernels`."""
    H = np.ascontiguousarray(H_core, dtype=np.float64)
    B = np.ascontiguousarray(B_ao, dtype=np.float64)
    C = np.ascontiguousarray(C0, dtype=np.float64)
    nbf = H.shape[0]
    n_vir = nbf - n_occ
    nf = _clamp_frozen(n_frozen, n_occ)

    C_occ_all = np.ascontiguousarray(C[:, :n_occ])
    J, K_mat = K.build_jk_3c_rhf(B, C_occ_all, nbf)
    F_ao = H + J - K_mat
    e_scf = float(np.sum((C_occ_all @ C_occ_all.T) * (H + F_ao)) + v_nn)

    C_sc, eps, _ = semicanonicalize_subspaces_frozen(C, F_ao, n_occ, nf)
    n_act = n_occ - nf
    if n_act <= 0 or n_vir <= 0:
        e_os, e_ss = 0.0, 0.0
    else:
        C_act = np.ascontiguousarray(C_sc[:, nf:n_occ])
        C_vir = np.ascontiguousarray(C_sc[:, n_occ:])
        B_ov = K.transform_3c_mo(B, C_act, C_vir, nbf)
        e_os, e_ss, _ = K.eval_rmp2_df(B_ov, eps[nf:n_occ], eps[n_occ:])

    e_corr = c_os * e_os + c_ss * e_ss
    return HybridMP2Result(
        reference="rhf",
        energy_scf=e_scf,
        energy_mp2_ss=c_ss * e_ss,
        energy_mp2_os=c_os * e_os,
        energy_mp2_corr=e_corr,
        energy_total=e_scf + e_corr,
        converged=True,
        iterations=1,
        C_alpha=C_sc,
        C_beta=C_sc.copy(),
        orbital_energies_alpha=eps,
        orbital_energies_beta=eps.copy(),
        n_occ_alpha=n_occ,
        n_occ_beta=n_occ,
        n_virt_alpha=n_vir,
        n_virt_beta=n_vir,
        n_frozen=nf,
    )


def run_df_mp2_uhf(
    H_core: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    C0_a: np.ndarray,
    C0_b: np.ndarray,
    na: int,
    nb: int,
    n_frozen: int = 0,
    c_os: float = 1.0,
    c_ss: float = 1.0,
    include_singles: bool = False,
    reference: str = "uhf",
) -> HybridMP2Result:
    """DF/Cholesky-UMP2 & Semicanonical ROMP2 berbasis C++20 `mshqc.kernels`."""
    H = np.ascontiguousarray(H_core, dtype=np.float64)
    B = np.ascontiguousarray(B_ao, dtype=np.float64)
    Ca = np.ascontiguousarray(C0_a, dtype=np.float64)
    Cb = np.ascontiguousarray(C0_b, dtype=np.float64)
    nbf = H.shape[0]
    va, vb = nbf - na, nbf - nb
    nf_a = _clamp_frozen(n_frozen, na)
    nf_b = _clamp_frozen(n_frozen, nb)

    Ca_all = np.ascontiguousarray(Ca[:, :na])
    Cb_all = np.ascontiguousarray(Cb[:, :nb])
    J, Ka, Kb = K.build_jk_3c_uhf(B, Ca_all, Cb_all, nbf)
    Fa = H + J - Ka
    Fb = H + J - Kb
    Pa = Ca_all @ Ca_all.T if na > 0 else np.zeros((nbf, nbf), dtype=np.float64)
    Pb = Cb_all @ Cb_all.T if nb > 0 else np.zeros((nbf, nbf), dtype=np.float64)
    e_scf = float(0.5 * (np.sum(Pa * (H + Fa)) + np.sum(Pb * (H + Fb))) + v_nn)

    # [INJECTED_STAGE6_ROMP_FC_PARITY]
    # ROMP2/ROMP3 standar (Psi4 & C++ ROMP2::semicanonicalize) mendiagonalisasi
    # seluruh blok occupied sebelum memotong orbital inti beku n_frozen.
    sc_nf_a = 0 if include_singles else nf_a
    sc_nf_b = 0 if include_singles else nf_b
    Ca_sc, eps_a, Fvo_a_raw = semicanonicalize_subspaces_frozen(Ca, Fa, na, sc_nf_a)
    Cb_sc, eps_b, Fvo_b_raw = semicanonicalize_subspaces_frozen(Cb, Fb, nb, sc_nf_b)
    Fvo_a = Fvo_a_raw[:, nf_a:] if include_singles else Fvo_a_raw
    Fvo_b = Fvo_b_raw[:, nf_b:] if include_singles else Fvo_b_raw

    e_singles = (
        _compute_romp2_singles(
            Fvo_a, Fvo_b, eps_a[nf_a:na], eps_a[na:], eps_b[nf_b:nb], eps_b[nb:]
        )
        if include_singles
        else 0.0
    )

    na_act = na - nf_a
    nb_act = nb - nf_b
    n_aux = B.shape[1]
    B_ov_a = (
        K.transform_3c_mo(B, np.ascontiguousarray(Ca_sc[:, nf_a:na]), np.ascontiguousarray(Ca_sc[:, na:]), nbf)
        if (na_act > 0 and va > 0)
        else np.zeros((0, n_aux), dtype=np.float64)
    )
    B_ov_b = (
        K.transform_3c_mo(B, np.ascontiguousarray(Cb_sc[:, nf_b:nb]), np.ascontiguousarray(Cb_sc[:, nb:]), nbf)
        if (nb_act > 0 and vb > 0)
        else np.zeros((0, n_aux), dtype=np.float64)
    )

    e_aa, e_bb, e_ab, _, _, _ = K.eval_ump2_df(
        B_ov_a, B_ov_b, eps_a[nf_a:na], eps_a[na:], eps_b[nf_b:nb], eps_b[nb:]
    )
    e_ss = c_ss * (e_aa + e_bb)
    e_os = c_os * e_ab
    e_corr = e_singles + e_ss + e_os
    return HybridMP2Result(
        reference=reference,
        energy_scf=e_scf,
        energy_mp2_ss=e_ss,
        energy_mp2_os=e_os,
        energy_mp2_corr=e_corr,
        energy_total=e_scf + e_corr,
        converged=True,
        iterations=1,
        C_alpha=Ca_sc,
        C_beta=Cb_sc,
        orbital_energies_alpha=eps_a,
        orbital_energies_beta=eps_b,
        n_occ_alpha=na,
        n_occ_beta=nb,
        n_virt_alpha=va,
        n_virt_beta=vb,
        n_frozen=nf_a,
        energy_mp2_singles=e_singles,
    )


def run_df_romp2(
    H_core: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    C_rohf: np.ndarray,
    na: int,
    nb: int,
    n_frozen: int = 0,
) -> HybridMP2Result:
    """DF/Cholesky-ROMP2 semikanonik (non-Brillouin singles + doubles)."""
    return run_df_mp2_uhf(
        H_core, v_nn, B_ao, C_rohf, C_rohf, na, nb,
        n_frozen=n_frozen, include_singles=True, reference="rohf",
    )


def run_df_mp3_rhf(
    H_core: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    C0: np.ndarray,
    n_occ: int,
    n_frozen: int = 0,
    eri4=None,
) -> HybridMP3Result:
    """DF/Cholesky-RMP3 bebas O(V^4) memakai 1x transformasi MO terpadu."""
    H = np.ascontiguousarray(H_core, dtype=np.float64)
    B = np.ascontiguousarray(B_ao, dtype=np.float64)
    C = np.ascontiguousarray(C0, dtype=np.float64)
    nbf = H.shape[0]
    n_vir = nbf - n_occ
    nf = _clamp_frozen(n_frozen, n_occ)

    C_occ_all = np.ascontiguousarray(C[:, :n_occ])
    J, K_mat = K.build_jk_4d_rhf(eri4, C_occ_all) if eri4 is not None else K.build_jk_3c_rhf(B, C_occ_all, nbf)
    F_ao = H + J - K_mat
    e_scf = float(np.sum((C_occ_all @ C_occ_all.T) * (H + F_ao)) + v_nn)

    C_sc, eps, _ = semicanonicalize_subspaces_frozen(C, F_ao, n_occ, nf)
    n_act = n_occ - nf
    if n_act <= 0 or n_vir <= 0:
        e_mp2, e_mp3 = 0.0, 0.0
    else:
        C_act = np.ascontiguousarray(C_sc[:, nf:n_occ])
        C_vir = np.ascontiguousarray(C_sc[:, n_occ:])
        eps_o, eps_v = eps[nf:n_occ], eps[n_occ:]
        B_oo, B_ov, B_vv = K.transform_3c_mo_blocks(B, C_act, C_vir, nbf)
        e_os, e_ss, T2 = K.eval_rmp2_df(B_ov, eps_o, eps_v)
        e_mp2 = e_os + e_ss
        if eri4 is not None:
            W_pp, _ = K.eval_pp_ladder_and_zvec_rhf_exact(eri4, T2, C_act, C_vir, False)
            e_mp3, _ = K.eval_rmp3_df(T2, B_oo, B_ov, B_vv, eps_o, eps_v, W_pp)
        else:
            e_mp3, _ = K.eval_rmp3_df(T2, B_oo, B_ov, B_vv, eps_o, eps_v)

    e_corr = e_mp2 + e_mp3
    return HybridMP3Result(
        reference="rhf",
        e_hf=e_scf,
        e_mp2=e_mp2,
        e_mp3=e_mp3,
        e3_aa=0.0,
        e3_bb=0.0,
        e3_ab=e_mp3,
        e_corr_total=e_corr,
        e_total=e_scf + e_corr,
        converged=True,
        iterations=1,
        C_alpha=C_sc,
        C_beta=C_sc.copy(),
        n_occ_alpha=n_occ,
        n_occ_beta=n_occ,
        n_frozen=nf,
    )


def run_df_mp3_uhf(
    H_core: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    C0_a: np.ndarray,
    C0_b: np.ndarray,
    na: int,
    nb: int,
    n_frozen: int = 0,
    include_singles: bool = False,
    reference: str = "uhf",
    eri4=None,
) -> HybridMP3Result:
    """DF/Cholesky-UMP3 & Semicanonical ROMP3 bebas O(V^4) memakai 1x transformasi MO terpadu."""
    H = np.ascontiguousarray(H_core, dtype=np.float64)
    B = np.ascontiguousarray(B_ao, dtype=np.float64)
    Ca = np.ascontiguousarray(C0_a, dtype=np.float64)
    Cb = np.ascontiguousarray(C0_b, dtype=np.float64)
    nbf = H.shape[0]
    nf_a = _clamp_frozen(n_frozen, na)
    nf_b = _clamp_frozen(n_frozen, nb)

    Ca_all = np.ascontiguousarray(Ca[:, :na])
    Cb_all = np.ascontiguousarray(Cb[:, :nb])
    J, Ka, Kb = K.build_jk_4d_uhf(eri4, Ca_all, Cb_all) if eri4 is not None else K.build_jk_3c_uhf(B, Ca_all, Cb_all, nbf)
    Fa = H + J - Ka
    Fb = H + J - Kb
    Pa = Ca_all @ Ca_all.T if na > 0 else np.zeros((nbf, nbf), dtype=np.float64)
    Pb = Cb_all @ Cb_all.T if nb > 0 else np.zeros((nbf, nbf), dtype=np.float64)
    e_scf = float(0.5 * (np.sum(Pa * (H + Fa)) + np.sum(Pb * (H + Fb))) + v_nn)

    # [INJECTED_STAGE6_ROMP_FC_PARITY]
    sc_nf_a = 0 if include_singles else nf_a
    sc_nf_b = 0 if include_singles else nf_b
    Ca_sc, eps_a, Fvo_a_raw = semicanonicalize_subspaces_frozen(Ca, Fa, na, sc_nf_a)
    Cb_sc, eps_b, Fvo_b_raw = semicanonicalize_subspaces_frozen(Cb, Fb, nb, sc_nf_b)
    Fvo_a = Fvo_a_raw[:, nf_a:] if include_singles else Fvo_a_raw
    Fvo_b = Fvo_b_raw[:, nf_b:] if include_singles else Fvo_b_raw

    Ca_act = np.ascontiguousarray(Ca_sc[:, nf_a:na])
    Ca_v = np.ascontiguousarray(Ca_sc[:, na:])
    Cb_act = np.ascontiguousarray(Cb_sc[:, nf_b:nb])
    Cb_v = np.ascontiguousarray(Cb_sc[:, nb:])
    eps_oa, eps_va = eps_a[nf_a:na], eps_a[na:]
    eps_ob, eps_vb = eps_b[nf_b:nb], eps_b[nb:]

    e_singles = (
        _compute_romp2_singles(Fvo_a, Fvo_b, eps_oa, eps_va, eps_ob, eps_vb)
        if include_singles
        else 0.0
    )

    B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b = K.transform_3c_mo_blocks_uhf(
        B, Ca_act, Ca_v, Cb_act, Cb_v, nbf
    )
    e_aa, e_bb, e_ab, T2_aa, T2_bb, T2_ab = K.eval_ump2_df(
        B_ov_a, B_ov_b, eps_oa, eps_va, eps_ob, eps_vb
    )
    e_mp2 = e_singles + e_aa + e_bb + e_ab
    if eri4 is not None:
        W_aa, W_bb, W_ab, _, _ = K.eval_pp_ladder_and_zvec_uhf_exact(
            eri4, T2_aa, T2_bb, T2_ab, Ca_act, Ca_v, Cb_act, Cb_v, False
        )
        e3_aa, e3_bb, e3_ab, _, _, _ = K.eval_ump3_df(
            T2_aa, T2_bb, T2_ab,
            B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b,
            eps_oa, eps_va, eps_ob, eps_vb,
            W_aa, W_bb, W_ab,
        )
    else:
        e3_aa, e3_bb, e3_ab, _, _, _ = K.eval_ump3_df(
            T2_aa, T2_bb, T2_ab,
            B_oo_a, B_ov_a, B_vv_a, B_oo_b, B_ov_b, B_vv_b,
            eps_oa, eps_va, eps_ob, eps_vb,
        )
    e_mp3 = e3_aa + e3_bb + e3_ab
    e_corr = e_mp2 + e_mp3
    return HybridMP3Result(
        reference=reference,
        e_hf=e_scf,
        e_mp2=e_mp2,
        e_mp3=e_mp3,
        e3_aa=e3_aa,
        e3_bb=e3_bb,
        e3_ab=e3_ab,
        e_corr_total=e_corr,
        e_total=e_scf + e_corr,
        converged=True,
        iterations=1,
        C_alpha=Ca_sc,
        C_beta=Cb_sc,
        n_occ_alpha=na,
        n_occ_beta=nb,
        n_frozen=nf_a,
    )


def run_df_romp3(
    H_core: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    C_rohf: np.ndarray,
    na: int,
    nb: int,
    n_frozen: int = 0,
    eri4=None,
) -> HybridMP3Result:
    """DF/Cholesky/Exact-ROMP3 semikanonik (identik dengan ROMP3::compute() C++)."""
    return run_df_mp3_uhf(
        H_core, v_nn, B_ao, C_rohf, C_rohf, na, nb,
        n_frozen=n_frozen, include_singles=True, reference="rohf", eri4=eri4,
    )


def run_exact_mp2_rhf(
    H_core: np.ndarray,
    v_nn: float,
    eri4,
    C0: np.ndarray,
    n_occ: int,
    n_frozen: int = 0,
    c_os: float = 1.0,
    c_ss: float = 1.0,
) -> HybridMP2Result:
    """Direct Exact 4-Index RMP2 menggunakan TBLIS O(O N^4) `eval_rmp2_exact`."""
    H = np.ascontiguousarray(H_core, dtype=np.float64)
    C = np.ascontiguousarray(C0, dtype=np.float64)
    nbf = H.shape[0]
    n_vir = nbf - n_occ
    nf = _clamp_frozen(n_frozen, n_occ)

    C_occ_all = np.ascontiguousarray(C[:, :n_occ])
    J, K_mat = K.build_jk_4d_rhf(eri4, C_occ_all)
    F_ao = H + J - K_mat
    e_scf = float(np.sum((C_occ_all @ C_occ_all.T) * (H + F_ao)) + v_nn)

    C_sc, eps, _ = semicanonicalize_subspaces_frozen(C, F_ao, n_occ, nf)
    n_act = n_occ - nf
    if n_act <= 0 or n_vir <= 0:
        e_os, e_ss = 0.0, 0.0
    else:
        e_os, e_ss = K.eval_rmp2_exact(
            eri4,
            np.ascontiguousarray(C_sc[:, nf:n_occ]),
            np.ascontiguousarray(C_sc[:, n_occ:]),
            eps[nf:n_occ], eps[n_occ:],
        )
    e_corr = c_os * e_os + c_ss * e_ss
    return HybridMP2Result(
        reference="rhf", energy_scf=e_scf,
        energy_mp2_ss=c_ss * e_ss, energy_mp2_os=c_os * e_os,
        energy_mp2_corr=e_corr, energy_total=e_scf + e_corr,
        converged=True, iterations=1,
        C_alpha=C_sc, C_beta=C_sc.copy(),
        orbital_energies_alpha=eps, orbital_energies_beta=eps.copy(),
        n_occ_alpha=n_occ, n_occ_beta=n_occ,
        n_virt_alpha=n_vir, n_virt_beta=n_vir, n_frozen=nf,
    )


def run_exact_mp2_uhf(
    H_core: np.ndarray,
    v_nn: float,
    eri4,
    C0_a: np.ndarray,
    C0_b: np.ndarray,
    na: int,
    nb: int,
    n_frozen: int = 0,
    c_os: float = 1.0,
    c_ss: float = 1.0,
) -> HybridMP2Result:
    """Direct Exact 4-Index UMP2 menggunakan TBLIS O(O N^4) `eval_ump2_exact`."""
    H = np.ascontiguousarray(H_core, dtype=np.float64)
    Ca = np.ascontiguousarray(C0_a, dtype=np.float64)
    Cb = np.ascontiguousarray(C0_b, dtype=np.float64)
    nbf = H.shape[0]
    va, vb = nbf - na, nbf - nb
    nf_a, nf_b = _clamp_frozen(n_frozen, na), _clamp_frozen(n_frozen, nb)

    Ca_o, Cb_o = np.ascontiguousarray(Ca[:, :na]), np.ascontiguousarray(Cb[:, :nb])
    J, Ka, Kb = K.build_jk_4d_uhf(eri4, Ca_o, Cb_o)
    Fa, Fb = H + J - Ka, H + J - Kb
    Pa = Ca_o @ Ca_o.T if na > 0 else np.zeros((nbf, nbf))
    Pb = Cb_o @ Cb_o.T if nb > 0 else np.zeros((nbf, nbf))
    e_scf = float(0.5 * (np.sum(Pa * (H + Fa)) + np.sum(Pb * (H + Fb))) + v_nn)

    Ca_sc, ea, _ = semicanonicalize_subspaces_frozen(Ca, Fa, na, nf_a)
    Cb_sc, eb, _ = semicanonicalize_subspaces_frozen(Cb, Fb, nb, nf_b)
    e_aa, e_bb, e_ab = K.eval_ump2_exact(
        eri4,
        np.ascontiguousarray(Ca_sc[:, nf_a:na]), np.ascontiguousarray(Ca_sc[:, na:]),
        np.ascontiguousarray(Cb_sc[:, nf_b:nb]), np.ascontiguousarray(Cb_sc[:, nb:]),
        ea[nf_a:na], ea[na:], eb[nf_b:nb], eb[nb:],
    )
    e_ss = c_ss * (e_aa + e_bb)
    e_os = c_os * e_ab
    e_corr = e_os + e_ss
    return HybridMP2Result(
        reference="uhf", energy_scf=e_scf,
        energy_mp2_ss=e_ss, energy_mp2_os=e_os,
        energy_mp2_corr=e_corr, energy_total=e_scf + e_corr,
        converged=True, iterations=1,
        C_alpha=Ca_sc, C_beta=Cb_sc,
        orbital_energies_alpha=ea, orbital_energies_beta=eb,
        n_occ_alpha=na, n_occ_beta=nb,
        n_virt_alpha=va, n_virt_beta=vb, n_frozen=nf_a,
    )
