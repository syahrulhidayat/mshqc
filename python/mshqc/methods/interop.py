# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""Adapter interoperabilitas berbasis NumPy/PySCF (tanpa wajib membuat mshqc.Molecule)."""

from typing import Optional, Tuple, Union
import numpy as np
from .mp_hybrid import (
    HybridMP2Result,
    HybridMP3Result,
    run_df_mp2_rhf,
    run_df_mp2_uhf,
    run_df_mp3_rhf,
    run_df_mp3_uhf,
    run_df_romp2,
    run_df_romp3,
)
from .omp_hybrid import HybridOMPResult, run_df_omp_rhf, run_df_omp_uhf
from .kappa_omp2 import run_df_kappa_omp2_rhf, run_df_kappa_omp2_uhf
from .lt_sos_omp2 import run_df_lt_sos_omp2_rhf, run_df_lt_sos_omp2_uhf


def _resolve_omp_params(method: str) -> Tuple[str, float, float, float]:
    m = method.strip().lower().replace("_", "-")
    if m == "omp2":
        return ("omp2", 1.0, 1.0, 0.0)
    if m == "scs-omp2":
        return ("scs-omp2", 1.2, 1.0 / 3.0, 0.0)
    if m == "sos-omp2":
        return ("sos-omp2", 1.3, 0.0, 0.0)
    if m in ("omp2.5", "omp25"):
        return ("omp2.5", 1.0, 1.0, 0.5)
    if m == "omp3":
        return ("omp3", 1.0, 1.0, 1.0)
    raise ValueError(f"Metode hibrida tidak dikenali: {method}")


def run_from_arrays(
    H_core: np.ndarray,
    S_ovlp: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    C_alpha: np.ndarray,
    n_occ_alpha: int,
    C_beta: Optional[np.ndarray] = None,
    n_occ_beta: Optional[int] = None,
    method: str = "omp2",
    grad_tol: float = 1e-5,
    energy_tol: float = 1e-9,
    max_iter: int = 40,
    n_frozen: int = 0,
    kappa: float = 1.45,
) -> Union[HybridOMPResult, HybridMP2Result, HybridMP3Result]:
    """
    Menjalankan MP2 / ROMP2 / MP3 / ROMP3 / OMP2 / SCS-OMP2 / SOS-OMP2 / OMP2.5 / OMP3
    langsung dari array NumPy (packed triangular maupun full, mendukung strict n_frozen).
    """
    m_low = method.strip().lower().replace("_", "-")
    H = np.ascontiguousarray(H_core, dtype=np.float64)
    S = np.ascontiguousarray(S_ovlp, dtype=np.float64)
    B = np.ascontiguousarray(B_ao, dtype=np.float64)
    Ca = np.ascontiguousarray(C_alpha, dtype=np.float64)
    na = int(n_occ_alpha)
    nf = int(n_frozen)

    if m_low == "romp2":
        nb = int(na if n_occ_beta is None else n_occ_beta)
        return run_df_romp2(H, float(v_nn), B, Ca, na, nb, n_frozen=nf)

    if m_low == "romp3":
        nb = int(na if n_occ_beta is None else n_occ_beta)
        return run_df_romp3(H, float(v_nn), B, Ca, na, nb, n_frozen=nf)

    if m_low in ("mp2", "rmp2", "ump2"):
        if C_beta is None:
            return run_df_mp2_rhf(H, float(v_nn), B, Ca, na, n_frozen=nf)
        Cb = np.ascontiguousarray(C_beta, dtype=np.float64)
        nb = int(na if n_occ_beta is None else n_occ_beta)
        return run_df_mp2_uhf(H, float(v_nn), B, Ca, Cb, na, nb, n_frozen=nf)

    if m_low in ("mp3", "rmp3", "ump3"):
        if C_beta is None:
            return run_df_mp3_rhf(H, float(v_nn), B, Ca, na, n_frozen=nf)
        Cb = np.ascontiguousarray(C_beta, dtype=np.float64)
        nb = int(na if n_occ_beta is None else n_occ_beta)
        return run_df_mp3_uhf(H, float(v_nn), B, Ca, Cb, na, nb, n_frozen=nf)

    if m_low in ("lt-sos-omp2", "lt-sos-oomp2"):
        if C_beta is None:
            return run_df_lt_sos_omp2_rhf(
                H, float(v_nn), B, Ca, na,
                grad_tol=grad_tol, energy_tol=energy_tol, max_iter=max_iter, n_frozen=nf,
            )
        Cb = np.ascontiguousarray(C_beta, dtype=np.float64)
        nb = int(na if n_occ_beta is None else n_occ_beta)
        return run_df_lt_sos_omp2_uhf(
            H, S, float(v_nn), B, Ca, Cb, na, nb,
            grad_tol=grad_tol, energy_tol=energy_tol, max_iter=max_iter, n_frozen=nf,
        )

    if m_low in ("kappa-omp2", "k-omp2", "kappa-oomp2", "koomp2"):
        if C_beta is None:
            return run_df_kappa_omp2_rhf(
                H, float(v_nn), B, Ca, na, kappa=float(kappa),
                grad_tol=grad_tol, energy_tol=energy_tol, max_iter=max_iter, n_frozen=nf,
            )
        Cb = np.ascontiguousarray(C_beta, dtype=np.float64)
        nb = int(na if n_occ_beta is None else n_occ_beta)
        return run_df_kappa_omp2_uhf(
            H, S, float(v_nn), B, Ca, Cb, na, nb, kappa=float(kappa),
            grad_tol=grad_tol, energy_tol=energy_tol, max_iter=max_iter, n_frozen=nf,
        )

    m_name, c_os, c_ss, lam_mp3 = _resolve_omp_params(m_low)
    if C_beta is None:
        return run_df_omp_rhf(
            H, float(v_nn), B, Ca, na,
            method_name=m_name, c_os=c_os, c_ss=c_ss, lam_mp3=lam_mp3,
            grad_tol=grad_tol, energy_tol=energy_tol, max_iter=max_iter, n_frozen=nf,
        )

    Cb = np.ascontiguousarray(C_beta, dtype=np.float64)
    nb = int(na if n_occ_beta is None else n_occ_beta)
    return run_df_omp_uhf(
        H, S, float(v_nn), B, Ca, Cb, na, nb,
        method_name=m_name, c_os=c_os, c_ss=c_ss, lam_mp3=lam_mp3,
        grad_tol=grad_tol, energy_tol=energy_tol, max_iter=max_iter, n_frozen=nf,
    )


def run_from_pyscf(
    mf,
    method: str = "omp3",
    grad_tol: float = 1e-5,
    energy_tol: float = 1e-9,
    max_iter: int = 40,
    n_frozen: int = 0,
) -> Union[HybridOMPResult, HybridMP2Result, HybridMP3Result]:
    """Plug-and-play adapter untuk objek PySCF `RHF`/`UHF` bermode `.density_fit()`."""
    if not hasattr(mf, "with_df") or mf.with_df is None:
        raise ValueError("Objek PySCF mf harus menjalankan `.density_fit()` terlebih dahulu.")

    H_core = mf.get_hcore()
    S_ovlp = mf.get_ovlp()
    v_nn = float(mf.energy_nuc())
    cderi = mf.with_df._cderi
    if cderi is None:
        mf.with_df.build()
        cderi = mf.with_df._cderi
    B_ao = np.ascontiguousarray(cderi.T, dtype=np.float64)

    mo_coeff = mf.mo_coeff
    if isinstance(mo_coeff, np.ndarray) and mo_coeff.ndim == 2:
        na = int(np.count_nonzero(mf.mo_occ > 0))
        return run_from_arrays(
            H_core, S_ovlp, v_nn, B_ao, mo_coeff, na,
            method=method, grad_tol=grad_tol, energy_tol=energy_tol,
            max_iter=max_iter, n_frozen=n_frozen,
        )

    Ca, Cb = mo_coeff[0], mo_coeff[1]
    na = int(np.count_nonzero(mf.mo_occ[0] > 0))
    nb = int(np.count_nonzero(mf.mo_occ[1] > 0))
    return run_from_arrays(
        H_core, S_ovlp, v_nn, B_ao, Ca, na, C_beta=Cb, n_occ_beta=nb,
        method=method, grad_tol=grad_tol, energy_tol=energy_tol,
        max_iter=max_iter, n_frozen=n_frozen,
    )
