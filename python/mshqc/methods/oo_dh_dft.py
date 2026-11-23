# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
Orbital-Optimized Double-Hybrid DFT (`OO-B2PLYP` & `OO-DSD-PBEP86`) di atas
kisi pertukaran-korelasi `pyscf.dft.numint` + kernel DF-OMP2 & TR-SOSCF `mshqc` (0 C++ baru).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from ..backends import active_kernels as K  
from .omp_hybrid import semicanonicalize_subspaces_frozen, _rotate_orbitals, _update_trust_radius
from .tr_soscf import solve_tr_soscf_rhf

_DH_PRESETS: Dict[str, Tuple[str, float, float, float, float]] = {
    # name: (xc_code, a_x_hf, a_c_dft, c_os_mp2, c_ss_mp2)
    "oo-b2plyp": ("0.47*B88, 0.73*LYP", 0.53, 0.73, 0.27, 0.27),
    "oo-dsd-pbep86": ("0.31*PBE, 0.44*P86", 0.69, 0.44, 0.53, 0.22),
}


@dataclass
class OODoubleHybridResult:
    functional: str
    energy_hf_like: float
    energy_xc_dft: float
    energy_mp2_corr: float
    energy_total: float
    converged: bool
    iterations: int
    grad_rms: float
    C_alpha: np.ndarray
    orbital_energies_alpha: np.ndarray
    n_occ_alpha: int


def run_oo_double_hybrid_rhf(
    pyscf_mol,
    aux_basis: str = "cc-pVDZ-RI",
    functional: str = "oo-b2plyp",
    grad_tol: float = 1.0e-5,
    energy_tol: float = 1.0e-8,
    max_iter: int = 35,
    grid_level: int = 3,
    C0: Optional[np.ndarray] = None,
) -> OODoubleHybridResult:
    """
    Jalankan Orbital-Optimized Double-Hybrid DFT (OO-B2PLYP / OO-DSD-PBEP86) untuk sistem RHF
    dengan mengevaluasi V_xc pada kisi PySCF dan korelasi/Z-vektor/TR-SOSCF pada `mshqc.kernels`.
    """
    from pyscf import dft, gto, scf
    from .interop import _extract_pyscf_df_factor

    key = functional.strip().lower().replace("_", "-")
    if key not in _DH_PRESETS:
        raise ValueError(f"Fungsional OO-DH tidak dikenali: {functional}. Pilihan: {list(_DH_PRESETS)}")
    xc_code, a_x, _a_c, c_os, c_ss = _DH_PRESETS[key]

    mf = scf.RHF(pyscf_mol).density_fit(auxbasis=aux_basis)
    if C0 is None:
        mf.verbose = 0
        mf.run()
        C = np.array(mf.mo_coeff, dtype=np.float64, copy=True)
    else:
        C = np.array(C0, dtype=np.float64, copy=True)

    H_core = np.asarray(mf.get_hcore(), dtype=np.float64)
    v_nn = float(pyscf_mol.energy_nuc())
    n_occ = int(pyscf_mol.nelectron) // 2
    nbf = C.shape[0]
    n_vir = nbf - n_occ

    aux_mol = gto.Mole()
    aux_mol.atom = pyscf_mol.atom
    aux_mol.basis = aux_basis
    aux_mol.unit = pyscf_mol.unit
    aux_mol.build(verbose=0)
    B_ao = _extract_pyscf_df_factor(pyscf_mol, aux_mol)

    grids = dft.Grids(pyscf_mol)
    grids.level = int(grid_level)
    grids.build(with_non0tab=True)
    ni = dft.numint.NumInt()

    e_prev = 0.0
    trust_radius = 0.25

    for it in range(max_iter):
        C_occ = np.ascontiguousarray(C[:, :n_occ])
        P_ref = 2.0 * (C_occ @ C_occ.T)
        J, K_mat = K.build_jk_3c_rhf(B_ao, C_occ, nbf)
        _, e_xc, v_xc = ni.nr_rks(pyscf_mol, grids, xc_code, P_ref)

        F_ks = H_core + J - a_x * K_mat + v_xc
        e_ref = float(0.5 * np.sum(P_ref * (2.0 * H_core + J - a_x * K_mat)) + e_xc + v_nn)

        C, eps, F_vo = semicanonicalize_subspaces_frozen(C, F_ks, n_occ, 0)
        C_occ = np.ascontiguousarray(C[:, :n_occ])
        C_vir = np.ascontiguousarray(C[:, n_occ:])
        eps_o, eps_v = eps[:n_occ], eps[n_occ:]

        B_oo, B_ov, B_vv = K.transform_3c_mo_blocks(B_ao, C_occ, C_vir, nbf)
        e_os, e_ss, T2 = K.eval_rmp2_df(B_ov, eps_o, eps_v)
        e_mp2 = c_os * e_os + c_ss * e_ss
        g_oo, g_vv = K.eval_omp2_1rdm_rhf(T2, n_occ, n_vir, c_os=c_os, c_ss=c_ss)
        Z_vo = K.eval_omp2_zvec_rhf_df(T2, B_oo, B_ov, B_vv, n_occ, n_vir, c_os=c_os, c_ss=c_ss)

        e_tot = e_ref + e_mp2
        P_corr_a = C_occ @ g_oo @ C_occ.T + C_vir @ g_vv @ C_vir.T
        G_ao = 2.0 * K.build_coulomb_3c(B_ao, P_corr_a, nbf) - a_x * K.build_exchange_3c_sym(B_ao, P_corr_a, nbf)
        w_vo = 4.0 * (F_vo + C_vir.T @ G_ao @ C_occ + (F_vo @ g_oo - g_vv @ F_vo) + Z_vo)
        w_rms = float(np.sqrt(np.mean(w_vo ** 2))) if w_vo.size > 0 else 0.0

        if w_rms <= grad_tol and (it == 0 or abs(e_tot - e_prev) <= energy_tol):
            return OODoubleHybridResult(
                functional=key.upper(),
                energy_hf_like=e_ref - float(e_xc),
                energy_xc_dft=float(e_xc),
                energy_mp2_corr=float(e_mp2),
                energy_total=e_tot,
                converged=True,
                iterations=it,
                grad_rms=w_rms,
                C_alpha=C,
                orbital_energies_alpha=eps,
                n_occ_alpha=n_occ,
            )

        trust_radius = _update_trust_radius(trust_radius, it, e_tot, e_prev)
        e_prev = e_tot
        kappa = solve_tr_soscf_rhf(w_vo, eps_o, eps_v, B_oo, B_ov, B_vv, trust_radius=trust_radius)
        C = _rotate_orbitals(C, kappa, n_occ, n_frozen=0)

    return OODoubleHybridResult(
        functional=key.upper(),
        energy_hf_like=e_ref - float(e_xc),
        energy_xc_dft=float(e_xc),
        energy_mp2_corr=float(e_mp2),
        energy_total=e_tot,
        converged=False,
        iterations=max_iter,
        grad_rms=w_rms,
        C_alpha=C,
        orbital_energies_alpha=eps,
        n_occ_alpha=n_occ,
    )
