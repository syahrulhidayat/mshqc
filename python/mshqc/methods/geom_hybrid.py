# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
Optimisasi Geometri & Frekuensi Vibrasi Harmonik + Termokimia RRHO Thin-Core (100% Python
di atas `UniversalGeomOptimizer` & kernel gradien analitik `grad_hybrid.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..geom_opt import (
    BOHR_TO_ANGSTROM,
    UniversalGeomOptimizer,
    build_internal_vibrational_subspace,
)
from .grad_hybrid import compute_df_gradient


_HARTREE_TO_KCAL = 627.5094740631
_HARTREE_TO_J = 4.3597447222071e-18
_BOHR_TO_M = 5.29177210903e-11
_AMU_TO_KG = 1.66053906660e-27
_C_CMS = 2.99792458e10
_PLANCK_SI = 6.62607015e-34
_KB_SI = 1.380649e-23

_EXACT_MASSES: Dict[int, float] = {
    1: 1.007825032, 2: 4.002603254, 3: 7.016003437, 4: 9.01218307,
    5: 11.00930536, 6: 12.00000000, 7: 14.003074004, 8: 15.994914619,
    9: 18.998403163, 10: 19.99244018, 11: 22.98976928, 12: 23.98504170,
    13: 26.98153853, 14: 27.976926535, 15: 30.973761998, 16: 31.972071174,
    17: 34.96885268, 18: 39.96238312, 19: 38.96370649, 20: 39.96259086,
    25: 54.93804391, 26: 55.93493633, 29: 62.92959772, 30: 63.92914201,
    35: 78.9183376, 36: 83.91149773,
}

_Z_TO_SYM: Dict[int, str] = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
    19: "K", 20: "Ca", 35: "Br", 36: "Kr",
}


@dataclass
class ThinCoreGeomResult:
    converged: bool
    iterations: int
    method: str
    eri_method: str
    basis_name: str
    initial_energy: float
    final_energy: float
    final_g_rms: float
    final_g_max: float
    wall_time_sec: float
    coordinates_bohr: np.ndarray
    coordinates_angstrom: np.ndarray
    gradient_ha_bohr: np.ndarray
    energy_history: List[float]
    g_rms_history: List[float]
    bond_lengths_angstrom: Dict[str, float]


@dataclass
class ThinCoreVibResult:
    frequencies_cm1: np.ndarray
    normal_modes: np.ndarray
    hessian_ha_bohr2: np.ndarray
    electronic_energy_ha: float
    zpe_hartree: float
    zpe_kcal_mol: float
    thermal_energy_ha: float
    enthalpy_ha: float
    gibbs_free_energy_ha: float
    temperature_k: float
    n_imaginary: int


def _isotopic_mass(z: int) -> float:
    return _EXACT_MASSES.get(z, float(2.4 * z))


def _clone_mol_at(mol, coords_bohr: np.ndarray):
    import mshqc

    m = mshqc.Molecule(int(mol.charge()), int(mol.multiplicity()))
    for i in range(int(mol.n_atoms())):
        m.add_atom(
            int(mol.atom(i).atomic_number),
            float(coords_bohr[i, 0]),
            float(coords_bohr[i, 1]),
            float(coords_bohr[i, 2]),
        )
    return m


def _supports_analytic_grad(method: str, eri_method: str) -> bool:
    m = method.strip().lower().replace("_", "-")
    if m.startswith("df-"):
        m = m[3:]
        eri = "df"
    else:
        eri = eri_method.strip().lower()
    return eri in ("df", "ri") and m in ("scf", "hf", "rhf", "uhf", "rohf", "omp2", "scs-omp2", "sos-omp2", "lt-sos-omp2", "kappa-omp2", "k-omp2", "kappa-oomp2")


def _eval_single_point_thin(
    mol,
    method: str,
    eri_method: str,
    basis_name: str,
    aux_basis_name: Optional[str],
    reference: str,
    energy_thresh: float,
    orb_grad_thresh: float,
    basis_dir: str,
) -> float:
    import mshqc

    m = method.strip().lower().replace("_", "-")
    if m.startswith("df-"):
        m, eri_method = m[3:], "df"
    elif m.startswith("cholesky-"):
        m, eri_method = m[9:], "cholesky"

    ref = None if reference.lower() in ("", "auto") else reference.lower()
    calc = mshqc.MSHQCCalculator(
        mol, basis_name=basis_name, eri_method=eri_method,
        aux_basis_name=aux_basis_name, reference=ref,
        threshold=energy_thresh, basis_dir=basis_dir, use_symmetry=False,
    )
    scf_res = calc.run_scf(use_hybrid=True)
    if m in ("scf", "hf", "rhf", "uhf", "rohf"):
        return float(scf_res.energy_total)
    if m in ("mp2", "rmp2", "ump2", "romp2"):
        return float(calc.run_mp2(scf_res, use_hybrid=True).energy_total)
    if m in ("mp3", "rmp3", "ump3", "romp3"):
        return float(calc.run_mp3(scf_res, use_hybrid=True).energy_total)
    if m == "omp2":
        return float(calc.run_omp2(scf_res, grad_thresh=orb_grad_thresh, max_iter=80, use_hybrid=True).energy_total)
    if m == "omp3":
        return float(calc.run_omp3(scf_res, grad_thresh=orb_grad_thresh, max_iter=80, use_hybrid=True).energy_total)
    if m == "scs-omp2":
        return float(calc.run_scs_omp2(scf_res, grad_thresh=orb_grad_thresh, max_iter=80).energy_total)
    if m == "sos-omp2":
        return float(calc.run_sos_omp2(scf_res, grad_thresh=orb_grad_thresh, max_iter=80).energy_total)
    if m in ("omp2.5", "omp25"):
        return float(calc.run_omp25(scf_res, grad_thresh=orb_grad_thresh, max_iter=80).energy_total)
    if m in ("kappa-omp2", "k-omp2", "kappa-oomp2"):
        return float(calc.run_kappa_omp2(scf_res, grad_thresh=orb_grad_thresh, max_iter=80).energy_total)
    if m == "lt-sos-omp2":
        return float(calc.run_lt_sos_omp2(scf_res, grad_thresh=orb_grad_thresh, max_iter=80).energy_total)
    raise ValueError(f"Metode tidak dikenali di Thin-Core optimizer: {method}")


def optimize_geometry_thin_core(
    mol,
    method: str = "omp2",
    eri_mode: str = "df",
    basis: str = "cc-pVDZ",
    aux_basis: Optional[str] = None,
    reference: str = "auto",
    max_iter: int = 40,
    g_rms_tol: float = 1.0e-6,
    g_max_tol: float = 3.0e-6,
    dx_rms_tol: float = 4.0e-6,
    dx_max_tol: float = 1.2e-5,
    energy_thresh: float = 1.0e-12,
    orb_grad_thresh: float = 1.0e-10,
    fd_step_bohr: float = 5.0e-4,
    use_analytic_gradient: bool = True,
    basis_dir: str = "data/basis",
) -> ThinCoreGeomResult:
    """Optimisasi geometri molekul in-place memakai Thin-Core Python + C++20 kernels."""
    t0 = time.perf_counter()
    natoms = int(mol.n_atoms())
    x0 = np.array([[mol.atom(i).x, mol.atom(i).y, mol.atom(i).z] for i in range(natoms)], dtype=np.float64)
    ws: dict = {}

    use_ana = use_analytic_gradient and _supports_analytic_grad(method, eri_mode)
    grad_fn = None
    if use_ana:
        def grad_fn(coords_bohr: np.ndarray) -> Tuple[float, np.ndarray]:
            m_cur = _clone_mol_at(mol, coords_bohr)
            return compute_df_gradient(
                m_cur, method=method, basis_name=basis, aux_basis_name=aux_basis,
                reference=reference, threshold=energy_thresh, orb_grad_tol=orb_grad_thresh,
                basis_dir=basis_dir, warm_start=ws,
            )

    def energy_fn(coords_bohr: np.ndarray) -> float:
        m_cur = _clone_mol_at(mol, coords_bohr)
        return _eval_single_point_thin(
            m_cur, method, eri_mode, basis, aux_basis, reference,
            energy_thresh, orb_grad_thresh, basis_dir,
        )

    reg_method = method.strip().lower().replace("_", "-")
    if reg_method.startswith("df-"):
        reg_method = reg_method[3:]
    reg_key = reg_method if reg_method in UniversalGeomOptimizer.METHOD_REGISTRY else "omp2"

    opt = UniversalGeomOptimizer(
        energy_fn=energy_fn,
        grad_fn=grad_fn,
        method=reg_key,
        integral_mode=eri_mode,
        fd_step_bohr=fd_step_bohr,
        max_iter=max_iter,
        grad_rms_tol=g_rms_tol,
        grad_max_tol=g_max_tol,
        step_rms_tol=dx_rms_tol,
        step_max_tol=dx_max_tol,
        energy_tol=max(1.0e-9, energy_thresh * 100.0),
    )
    res = opt.run(x0)

    for i in range(natoms):
        mol.set_atom_position(
            i,
            float(res.final_coords_bohr[i, 0]),
            float(res.final_coords_bohr[i, 1]),
            float(res.final_coords_bohr[i, 2]),
        )

    bonds: Dict[str, float] = {}
    for i in range(natoms):
        si = _Z_TO_SYM.get(int(mol.atom(i).atomic_number), f"Z{mol.atom(i).atomic_number}")
        for j in range(i + 1, natoms):
            sj = _Z_TO_SYM.get(int(mol.atom(j).atomic_number), f"Z{mol.atom(j).atomic_number}")
            d_ang = float(np.linalg.norm(res.final_coords_angstrom[i] - res.final_coords_angstrom[j]))
            if d_ang < 3.0 or natoms <= 4:
                bonds[f"{si}{i+1}-{sj}{j+1}"] = d_ang

    return ThinCoreGeomResult(
        converged=res.converged,
        iterations=res.iterations,
        method=method,
        eri_method=eri_mode,
        basis_name=basis,
        initial_energy=res.initial_energy,
        final_energy=res.final_energy,
        final_g_rms=res.grad_rms,
        final_g_max=res.grad_max,
        wall_time_sec=time.perf_counter() - t0,
        coordinates_bohr=res.final_coords_bohr,
        coordinates_angstrom=res.final_coords_angstrom,
        gradient_ha_bohr=res.final_gradient_bohr,
        energy_history=res.energy_history,
        g_rms_history=res.grad_rms_history,
        bond_lengths_angstrom=bonds,
    )


def compute_frequencies_thin_core(
    mol,
    method: str = "omp2",
    eri_mode: str = "df",
    basis: str = "cc-pVDZ",
    aux_basis: Optional[str] = None,
    reference: str = "auto",
    temperature_k: float = 298.15,
    step_bohr: float = 2.0e-3,
    energy_thresh: float = 1.0e-12,
    orb_grad_thresh: float = 1.0e-10,
    basis_dir: str = "data/basis",
) -> ThinCoreVibResult:
    """Frekuensi vibrasi harmonik O(N) via FD gradien analitik + termokimia RRHO."""
    natoms = int(mol.n_atoms())
    ndof = 3 * natoms
    x0 = np.array([[mol.atom(i).x, mol.atom(i).y, mol.atom(i).z] for i in range(natoms)], dtype=np.float64)
    masses = np.array([_isotopic_mass(int(mol.atom(i).atomic_number)) for i in range(natoms)], dtype=np.float64)

    ws: dict = {}
    hess = np.zeros((ndof, ndof), dtype=np.float64)
    n_indep = 3 * max(0, natoms - 1)

    if _supports_analytic_grad(method, eri_mode):
        e0, _ = compute_df_gradient(
            mol, method=method, basis_name=basis, aux_basis_name=aux_basis,
            reference=reference, threshold=energy_thresh, orb_grad_tol=orb_grad_thresh,
            basis_dir=basis_dir, warm_start=ws,
        )
        inv_2h = 1.0 / (2.0 * step_bohr)
        for k in range(n_indep):
            xp = x0.reshape(-1).copy()
            xm = x0.reshape(-1).copy()
            xp[k] += step_bohr
            xm[k] -= step_bohr
            _, gp = compute_df_gradient(
                _clone_mol_at(mol, xp.reshape(natoms, 3)),
                method=method, basis_name=basis, aux_basis_name=aux_basis,
                reference=reference, threshold=energy_thresh, orb_grad_tol=orb_grad_thresh,
                basis_dir=basis_dir, warm_start=dict(ws),
            )
            _, gm = compute_df_gradient(
                _clone_mol_at(mol, xm.reshape(natoms, 3)),
                method=method, basis_name=basis, aux_basis_name=aux_basis,
                reference=reference, threshold=energy_thresh, orb_grad_tol=orb_grad_thresh,
                basis_dir=basis_dir, warm_start=dict(ws),
            )
            hess[:, k] = (gp.reshape(-1) - gm.reshape(-1)) * inv_2h
        for xyz in range(3):
            hess[:, n_indep + xyz] = -np.sum(hess[:, xyz:n_indep:3], axis=1)
    else:
        e0 = _eval_single_point_thin(
            mol, method, eri_mode, basis, aux_basis, reference, energy_thresh, orb_grad_thresh, basis_dir
        )
        inv_h2 = 1.0 / (step_bohr * step_bohr)
        ep_1d = np.zeros(n_indep)
        em_1d = np.zeros(n_indep)
        for k in range(n_indep):
            xp, xm = x0.reshape(-1).copy(), x0.reshape(-1).copy()
            xp[k] += step_bohr
            xm[k] -= step_bohr
            ep_1d[k] = _eval_single_point_thin(
                _clone_mol_at(mol, xp.reshape(natoms, 3)), method, eri_mode, basis, aux_basis,
                reference, energy_thresh, orb_grad_thresh, basis_dir,
            )
            em_1d[k] = _eval_single_point_thin(
                _clone_mol_at(mol, xm.reshape(natoms, 3)), method, eri_mode, basis, aux_basis,
                reference, energy_thresh, orb_grad_thresh, basis_dir,
            )
            hess[k, k] = (ep_1d[k] - 2.0 * e0 + em_1d[k]) * inv_h2
        for k in range(n_indep):
            for l in range(k + 1, n_indep):
                xpp, xmm = x0.reshape(-1).copy(), x0.reshape(-1).copy()
                xpp[k] += step_bohr
                xpp[l] += step_bohr
                xmm[k] -= step_bohr
                xmm[l] -= step_bohr
                epp = _eval_single_point_thin(
                    _clone_mol_at(mol, xpp.reshape(natoms, 3)), method, eri_mode, basis, aux_basis,
                    reference, energy_thresh, orb_grad_thresh, basis_dir,
                )
                emm = _eval_single_point_thin(
                    _clone_mol_at(mol, xmm.reshape(natoms, 3)), method, eri_mode, basis, aux_basis,
                    reference, energy_thresh, orb_grad_thresh, basis_dir,
                )
                h_kl = 0.5 * (epp + emm - ep_1d[k] - em_1d[k] - ep_1d[l] - em_1d[l] + 2.0 * e0) * inv_h2
                hess[k, l] = hess[l, k] = h_kl
        for xyz in range(3):
            s_col = -np.sum(hess[:n_indep, xyz:n_indep:3], axis=1)
            hess[:n_indep, n_indep + xyz] = s_col
            hess[n_indep + xyz, :n_indep] = s_col
        for x1 in range(3):
            for x2 in range(3):
                hess[n_indep + x1, n_indep + x2] = -np.sum(hess[n_indep + x1, x2:n_indep:3])

    hess = 0.5 * (hess + hess.T)
    V_int = build_internal_vibrational_subspace(x0, masses)
    n_vib = V_int.shape[1]
    n_tr = ndof - n_vib

    sqrt_m = np.repeat(np.sqrt(masses), 3)
    F_mw = hess / np.outer(sqrt_m, sqrt_m)
    F_int = V_int.T @ F_mw @ V_int
    evals, evecs = np.linalg.eigh(0.5 * (F_int + F_int.T))
    normal_modes = V_int @ evecs

    force_conv = _HARTREE_TO_J / (_BOHR_TO_M * _BOHR_TO_M * _AMU_TO_KG)
    to_cm1 = 1.0 / (2.0 * np.pi * _C_CMS)

    freqs = np.zeros(n_vib, dtype=np.float64)
    n_imag = 0
    zpe_j = 0.0
    u_vib_j = 0.0
    s_vib_jk = 0.0

    for k, lam in enumerate(evals):
        if lam < 0.0:
            freqs[k] = -float(np.sqrt(-lam * force_conv)) * to_cm1
            n_imag += 1
        else:
            f_cm = float(np.sqrt(lam * force_conv)) * to_cm1
            freqs[k] = f_cm
            hv = _PLANCK_SI * f_cm * _C_CMS
            zpe_j += 0.5 * hv
            if temperature_k > 1.0e-3 and f_cm > 1.0:
                xv = hv / (_KB_SI * temperature_k)
                if xv < 100.0:
                    u_vib_j += hv / (np.exp(xv) - 1.0)
                    s_vib_jk += _KB_SI * (xv / (np.exp(xv) - 1.0) - np.log(1.0 - np.exp(-xv)))

    zpe_ha = zpe_j / _HARTREE_TO_J
    kt_ha = (_KB_SI * temperature_k) / _HARTREE_TO_J
    is_linear = (n_tr == 5)
    u_tr_ha = (2.5 if is_linear else 3.0) * kt_ha
    thermal_ha = e0 + zpe_ha + u_tr_ha + (u_vib_j / _HARTREE_TO_J)
    enthalpy_ha = thermal_ha + kt_ha
    s_elec_jk = _KB_SI * float(np.log(max(1, int(mol.multiplicity()))))
    s_tr_jk = _KB_SI * (18.5 if is_linear else 21.0)
    gibbs_ha = enthalpy_ha - temperature_k * (s_tr_jk + s_vib_jk + s_elec_jk) / _HARTREE_TO_J

    return ThinCoreVibResult(
        frequencies_cm1=freqs,
        normal_modes=normal_modes,
        hessian_ha_bohr2=hess,
        electronic_energy_ha=e0,
        zpe_hartree=zpe_ha,
        zpe_kcal_mol=zpe_ha * _HARTREE_TO_KCAL,
        thermal_energy_ha=thermal_ha,
        enthalpy_ha=enthalpy_ha,
        gibbs_free_energy_ha=gibbs_ha,
        temperature_k=temperature_k,
        n_imaginary=n_imag,
    )


# [INJECTED_THIN_CORE_STAGE9_PES_SCAN]
@dataclass
class ThinCorePESPoint:
    r_angstrom: float
    r_bohr: float
    energy_ha: float


def scan_bond_pes_thin_core(
    mol,
    atom_i: int,
    atom_j: int,
    distances_angstrom: List[float],
    method: str = "omp2",
    eri_mode: str = "df",
    basis: str = "cc-pVDZ",
    aux_basis: Optional[str] = None,
    reference: str = "auto",
    energy_thresh: float = 1.0e-10,
    orb_grad_thresh: float = 1.0e-7,
    basis_dir: str = "data/basis",
) -> List[ThinCorePESPoint]:
    """Pemindaian 1D Potential Energy Surface (PES) antar-ikatan di mode Thin-Core."""
    nat = int(mol.n_atoms())
    if not (0 <= atom_i < nat and 0 <= atom_j < nat and atom_i != atom_j):
        raise ValueError(f"Indeks atom tidak valid: atom_i={atom_i}, atom_j={atom_j}, nat={nat}")

    x0 = np.array([[mol.atom(k).x, mol.atom(k).y, mol.atom(k).z] for k in range(nat)], dtype=np.float64)
    vec = x0[atom_j] - x0[atom_i]
    norm = float(np.linalg.norm(vec))
    axis = (vec / norm) if norm > 1.0e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

    pts: List[ThinCorePESPoint] = []
    for r_ang in distances_angstrom:
        r_bohr = float(r_ang) / BOHR_TO_ANGSTROM
        coords = x0.copy()
        coords[atom_j] = coords[atom_i] + r_bohr * axis
        m_cur = _clone_mol_at(mol, coords)
        e_val = _eval_single_point_thin(
            m_cur, method, eri_mode, basis, aux_basis, reference,
            energy_thresh, orb_grad_thresh, basis_dir,
        )
        pts.append(ThinCorePESPoint(r_angstrom=float(r_ang), r_bohr=r_bohr, energy_ha=float(e_val)))
    return pts
