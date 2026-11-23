# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
Analisis fungsi gelombang & properti molekuler Thin-Core (0 baris C++ baru):
  - Natural Orbitals & NOONs (Natural Orbital Occupation Numbers) + Head-Gordon N_U
  - Muatan populasi atomik Mulliken & Lowdin
  - Ekspor format standar `.molden` (kompatibel Avogadro, VMD, Jmol, Multiwfn)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.linalg import eigh


_Z_TO_SYM: Dict[int, str] = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
    19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe",
    27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se",
    35: "Br", 36: "Kr", 53: "I", 54: "Xe",
}
_L_TO_MOLDEN = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}


@dataclass
class NaturalOrbitalResult:
    occupations: np.ndarray
    orbitals_ao: np.ndarray
    n_electrons: float
    n_unpaired_head_gordon: float
    n_unpaired_linear: float
    biradical_index_y0: float


@dataclass
class PopulationResult:
    symbols: List[str]
    mulliken_charges: np.ndarray
    lowdin_charges: np.ndarray
    mulliken_populations: np.ndarray
    lowdin_populations: np.ndarray


def _overlap_sqrt_and_inv_sqrt(S: np.ndarray, tol: float = 1.0e-10) -> Tuple[np.ndarray, np.ndarray]:
    evals, evecs = eigh(0.5 * (S + S.T))
    keep = evals > tol
    U = evecs[:, keep]
    ev = evals[keep]
    S_half = (U * np.sqrt(ev)[None, :]) @ U.T
    S_inv_half = (U * (1.0 / np.sqrt(ev))[None, :]) @ U.T
    return S_half, S_inv_half


def build_1rdm_ao(
    C_alpha: np.ndarray,
    n_occ_alpha: int,
    C_beta: Optional[np.ndarray] = None,
    n_occ_beta: Optional[int] = None,
    g_oo_alpha: Optional[np.ndarray] = None,
    g_vv_alpha: Optional[np.ndarray] = None,
    g_oo_beta: Optional[np.ndarray] = None,
    g_vv_beta: Optional[np.ndarray] = None,
    n_frozen: int = 0,
) -> np.ndarray:
    """Merakit matriks densitas 1-elektron total AO P_tot (SCF maupun berkorelasi OMP2/OMP3)."""
    Ca = np.asarray(C_alpha, dtype=np.float64)
    nbf = Ca.shape[0]
    na = int(n_occ_alpha)
    nf = max(0, min(int(n_frozen), na))

    Pa = Ca[:, :na] @ Ca[:, :na].T if na > 0 else np.zeros((nbf, nbf), dtype=np.float64)
    if g_oo_alpha is not None and g_vv_alpha is not None:
        Pa = Pa + Ca[:, nf:na] @ g_oo_alpha @ Ca[:, nf:na].T + Ca[:, na:] @ g_vv_alpha @ Ca[:, na:].T

    if C_beta is None:
        return np.ascontiguousarray(2.0 * Pa, dtype=np.float64)

    Cb = np.asarray(C_beta, dtype=np.float64)
    nb = na if n_occ_beta is None else int(n_occ_beta)
    nfb = max(0, min(int(n_frozen), nb))
    Pb = Cb[:, :nb] @ Cb[:, :nb].T if nb > 0 else np.zeros((nbf, nbf), dtype=np.float64)
    if g_oo_beta is not None and g_vv_beta is not None:
        Pb = Pb + Cb[:, nfb:nb] @ g_oo_beta @ Cb[:, nfb:nb].T + Cb[:, nb:] @ g_vv_beta @ Cb[:, nb:].T
    return np.ascontiguousarray(Pa + Pb, dtype=np.float64)


def scf_density_ao(res: Any) -> np.ndarray:
    """Ekstrak matriks densitas total AO P_tot langsung dari HybridSCFResult / HybridOMPResult."""
    ref = str(getattr(res, "reference", "rhf")).lower()
    if ref == "rhf":
        return build_1rdm_ao(res.C_alpha, int(res.n_occ_alpha))
    return build_1rdm_ao(res.C_alpha, int(res.n_occ_alpha), res.C_beta, int(res.n_occ_beta))


def correlated_density_ao(res: Any) -> np.ndarray:
    """
    Ekstrak matriks densitas total AO P_tot langsung dari HybridSCFResult maupun
    HybridOMPResult (memakai g_oo_*, g_vv_*, dan n_frozen tersimpan tanpa hitung ulang).
    """
    ref = str(getattr(res, "reference", "rhf")).lower()
    na = int(getattr(res, "n_occ_alpha", 0))
    nb = int(getattr(res, "n_occ_beta", na))
    nf = int(getattr(res, "n_frozen", 0))
    g_oo_a = getattr(res, "g_oo_alpha", None)
    g_vv_a = getattr(res, "g_vv_alpha", None)
    g_oo_b = getattr(res, "g_oo_beta", None)
    g_vv_b = getattr(res, "g_vv_beta", None)
    if ref == "rhf":
        return build_1rdm_ao(res.C_alpha, na, g_oo_alpha=g_oo_a, g_vv_alpha=g_vv_a, n_frozen=nf)
    return build_1rdm_ao(
        res.C_alpha, na, res.C_beta, nb,
        g_oo_alpha=g_oo_a, g_vv_alpha=g_vv_a,
        g_oo_beta=g_oo_b, g_vv_beta=g_vv_b,
        n_frozen=nf,
    )


def compute_natural_orbitals(P_ao: np.ndarray, S_ovlp: np.ndarray) -> NaturalOrbitalResult:
    """Menghitung Natural Orbitals, NOONs, indeks elektron tak berpasangan Head-Gordon, dan y_0."""
    P = 0.5 * (np.asarray(P_ao, dtype=np.float64) + np.asarray(P_ao, dtype=np.float64).T)
    S = 0.5 * (np.asarray(S_ovlp, dtype=np.float64) + np.asarray(S_ovlp, dtype=np.float64).T)
    S_half, S_inv_half = _overlap_sqrt_and_inv_sqrt(S)

    D_orth = S_half @ P @ S_half
    evals, evecs = eigh(0.5 * (D_orth + D_orth.T))
    order = np.argsort(evals)[::-1]
    noons = evals[order]
    C_no = np.ascontiguousarray(S_inv_half @ evecs[:, order], dtype=np.float64)

    clipped = np.clip(noons, 0.0, 2.0)
    n_u_hg = float(np.sum(clipped * (2.0 - clipped)))
    n_u_lin = float(np.sum(np.minimum(clipped, 2.0 - clipped)))
    n_elec = float(np.sum(noons))

    # Indeks biradikal Yamaguchi y_0 = n_LUNO (untuk sistem singlet)
    n_half = int(round(n_elec / 2.0))
    y0 = float(clipped[n_half]) if 0 <= n_half < clipped.size else 0.0

    return NaturalOrbitalResult(
        occupations=noons,
        orbitals_ao=C_no,
        n_electrons=n_elec,
        n_unpaired_head_gordon=n_u_hg,
        n_unpaired_linear=n_u_lin,
        biradical_index_y0=y0,
    )


def mulliken_and_lowdin_charges(mol, basis, S_ovlp: np.ndarray, P_ao: np.ndarray) -> PopulationResult:
    """Menghitung populasi dan muatan parsial atomik Mulliken & Lowdin."""
    nat = int(mol.n_atoms())
    S = 0.5 * (np.asarray(S_ovlp, dtype=np.float64) + np.asarray(S_ovlp, dtype=np.float64).T)
    P = 0.5 * (np.asarray(P_ao, dtype=np.float64) + np.asarray(P_ao, dtype=np.float64).T)
    S_half, _ = _overlap_sqrt_and_inv_sqrt(S)

    diag_mull = np.diag(P @ S)
    diag_low = np.diag(S_half @ P @ S_half)

    mull_pop = np.zeros(nat, dtype=np.float64)
    low_pop = np.zeros(nat, dtype=np.float64)
    off = 0
    for s_idx in range(int(basis.n_shells())):
        sh = basis.shell(s_idx)
        center = int(sh.center)
        nf = int(sh.n_functions())
        mull_pop[center] += float(np.sum(diag_mull[off : off + nf]))
        low_pop[center] += float(np.sum(diag_low[off : off + nf]))
        off += nf

    z_vec = np.array([int(mol.atom(i).atomic_number) for i in range(nat)], dtype=np.float64)
    syms = [_Z_TO_SYM.get(int(z), f"X{int(z)}") for z in z_vec]
    return PopulationResult(
        symbols=syms,
        mulliken_charges=z_vec - mull_pop,
        lowdin_charges=z_vec - low_pop,
        mulliken_populations=mull_pop,
        lowdin_populations=low_pop,
    )


def write_molden(
    path: Union[str, Path],
    mol,
    basis,
    C_alpha: np.ndarray,
    eps_alpha: np.ndarray,
    occ_alpha: np.ndarray,
    C_beta: Optional[np.ndarray] = None,
    eps_beta: Optional[np.ndarray] = None,
    occ_beta: Optional[np.ndarray] = None,
) -> Path:
    """Mengekspor geometri, basis GTO, dan orbital MO/Natural ke format standar `.molden`."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = ["[Molden Format]", "[Title]", "Generated by mshqc v2.0", "[Atoms] AU"]

    nat = int(mol.n_atoms())
    for i in range(nat):
        atom = mol.atom(i)
        z = int(atom.atomic_number)
        sym = _Z_TO_SYM.get(z, f"X{z}")
        lines.append(f"{sym:<3s} {i + 1:4d} {z:4d} {atom.x:16.10f} {atom.y:16.10f} {atom.z:16.10f}")

    lines.append("[GTO]")
    shells_by_atom: List[List[Any]] = [[] for _ in range(nat)]
    max_l = 0
    for s_idx in range(int(basis.n_shells())):
        sh = basis.shell(s_idx)
        shells_by_atom[int(sh.center)].append(sh)
        max_l = max(max_l, int(sh.l()))

    for a_idx in range(nat):
        lines.append(f"{a_idx + 1:4d} 0")
        for sh in shells_by_atom[a_idx]:
            l_sym = _L_TO_MOLDEN.get(int(sh.l()), "g")
            prims = sh.primitives
            lines.append(f" {l_sym} {len(prims):4d} 1.00")
            for p in prims:
                lines.append(f"  {float(p.exponent):18.10e} {float(p.coefficient):18.10e}")
        lines.append("")

    if max_l >= 2:
        lines.append("[5D]")
    if max_l >= 3:
        lines.append("[7F]")
    if max_l >= 4:
        lines.append("[9G]")

    lines.append("[MO]")

    def _append_spin(C_mat: np.ndarray, eps_vec: np.ndarray, occ_vec: np.ndarray, spin_label: str) -> None:
        nbf_m, nmo_m = C_mat.shape
        for mo_i in range(nmo_m):
            lines.append(" Sym= A")
            lines.append(f" Ene= {float(eps_vec[mo_i]):.10f}")
            lines.append(f" Spin= {spin_label}")
            lines.append(f" Occup= {float(occ_vec[mo_i]):.6f}")
            for bf_i in range(nbf_m):
                lines.append(f" {bf_i + 1:4d} {float(C_mat[bf_i, mo_i]):16.10f}")

    _append_spin(np.asarray(C_alpha), np.asarray(eps_alpha), np.asarray(occ_alpha), "Alpha")
    if C_beta is not None and eps_beta is not None and occ_beta is not None:
        _append_spin(np.asarray(C_beta), np.asarray(eps_beta), np.asarray(occ_beta), "Beta")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
