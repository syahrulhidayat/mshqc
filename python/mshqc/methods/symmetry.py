# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
Lightweight Pure-Python Abelian Point-Group Symmetry (C1, Ci, Cs, C2, C2v, C2h, D2, D2h).
Dirancang untuk arsitektur Thin-Core mshqc v2.0 (0 baris C++ baru):
  - Deteksi subgrup Abelian maksimal (axis-agnostic) & penyelarasan sumbu utama kanonik (Mulliken)
  - Konstruksi operator transformasi AO (s, p, d, f harmonik sferis konvensi libcint)
  - Proyeksi matriks Fock/densitas invarian simetri & pelabelan irrep orbital MO (Ag, B1u, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

# 8 operator Abelian diagonal pada sumbu (x, y, z): tanda (sx, sy, sz)
_ABELIAN_OPS: Dict[str, Tuple[int, int, int]] = {
    "E": (1, 1, 1),
    "C2z": (-1, -1, 1),
    "C2y": (-1, 1, -1),
    "C2x": (1, -1, -1),
    "i": (-1, -1, -1),
    "sxy": (1, 1, -1),
    "sxz": (1, -1, 1),
    "syz": (-1, 1, 1),
}

# Tabel karakter kanonik sesuai urutan tuple operasi tiap grup
_CHAR_TABLES: Dict[str, Dict[str, Tuple[int, ...]]] = {
    "C1": {"A": (1,)},
    "Ci": {"Ag": (1, 1), "Au": (1, -1)},
    "Cs": {"A'": (1, 1), "A''": (1, -1)},
    "C2": {"A": (1, 1), "B": (1, -1)},
    "C2v": {
        "A1": (1, 1, 1, 1),
        "A2": (1, 1, -1, -1),
        "B1": (1, -1, 1, -1),
        "B2": (1, -1, -1, 1),
    },
    "C2h": {
        "Ag": (1, 1, 1, 1),
        "Bg": (1, -1, 1, -1),
        "Au": (1, 1, -1, -1),
        "Bu": (1, -1, -1, 1),
    },
    "D2": {
        "A": (1, 1, 1, 1),
        "B1": (1, 1, -1, -1),
        "B2": (1, -1, 1, -1),
        "B3": (1, -1, -1, 1),
    },
    "D2h": {
        "Ag": (1, 1, 1, 1, 1, 1, 1, 1),
        "B1g": (1, 1, -1, -1, 1, 1, -1, -1),
        "B2g": (1, -1, 1, -1, 1, -1, 1, -1),
        "B3g": (1, -1, -1, 1, 1, -1, -1, 1),
        "Au": (1, 1, 1, 1, -1, -1, -1, -1),
        "B1u": (1, 1, -1, -1, -1, -1, 1, 1),
        "B2u": (1, -1, 1, -1, -1, 1, -1, 1),
        "B3u": (1, -1, -1, 1, -1, 1, 1, -1),
    },
}


@dataclass
class AbelianSymmetryInfo:
    point_group: str
    operations: List[str]
    atom_perms: Dict[str, np.ndarray]
    op_signs: Dict[str, Tuple[int, int, int]]


def _mol_arrays(mol) -> Tuple[np.ndarray, np.ndarray]:
    nat = int(mol.n_atoms())
    z = np.array([int(mol.atom(i).atomic_number) for i in range(nat)], dtype=np.int64)
    coords = np.array(
        [[mol.atom(i).x, mol.atom(i).y, mol.atom(i).z] for i in range(nat)],
        dtype=np.float64,
    )
    return z, coords


def _find_perm(
    z: np.ndarray, coords: np.ndarray, signs: Tuple[int, int, int], tol: float
) -> Optional[np.ndarray]:
    nat = z.size
    transformed = coords * np.asarray(signs, dtype=np.float64)[None, :]
    perm = np.full(nat, -1, dtype=np.int64)
    for i in range(nat):
        same_z = np.flatnonzero(z == z[i])
        dists = np.linalg.norm(coords[same_z] - transformed[i], axis=1)
        j_local = int(np.argmin(dists))
        if dists[j_local] > tol:
            return None
        perm[i] = int(same_z[j_local])
    if len(set(perm.tolist())) != nat:
        return None
    return perm


def _detect_valid_ops(z: np.ndarray, coords: np.ndarray, tol: float) -> Dict[str, np.ndarray]:
    valid: Dict[str, np.ndarray] = {}
    for name, sgn in _ABELIAN_OPS.items():
        perm = _find_perm(z, coords, sgn, tol)
        if perm is not None:
            valid[name] = perm
    return valid


def _lock_degenerate_axes(rel: np.ndarray, evals: np.ndarray, evecs: np.ndarray, tol: float = 1.0e-5) -> np.ndarray:
    """Kunci rotasi bebas pada subruang inersia degenerasi (symmetric/spherical top) ke atom referensi."""
    V = evecs.copy()
    scale = max(1.0, float(np.max(np.abs(evals))))
    d01 = abs(float(evals[1] - evals[0])) <= tol * scale
    d12 = abs(float(evals[2] - evals[1])) <= tol * scale

    if d01 and not d12:
        # Subruang degenerasi di kolom (0, 1), sumbu unik di kolom 2 (mis. C6H6, NH3)
        u3 = V[:, 2]
        for r in rel:
            proj = r - float(np.dot(r, u3)) * u3
            nrm = float(np.linalg.norm(proj))
            if nrm > 1.0e-6:
                u1 = proj / nrm
                u2 = np.cross(u3, u1)
                V = np.column_stack((u1, u2, u3))
                break
    elif d12 and not d01:
        # Subruang degenerasi di kolom (1, 2), sumbu unik di kolom 0 (mis. molekul linear/prolate)
        u1 = V[:, 0]
        for r in rel:
            proj = r - float(np.dot(r, u1)) * u1
            nrm = float(np.linalg.norm(proj))
            if nrm > 1.0e-6:
                u2 = proj / nrm
                u3 = np.cross(u1, u2)
                V = np.column_stack((u1, u2, u3))
                break
    if np.linalg.det(V) < 0.0:
        V[:, 2] *= -1.0
    return V


def align_principal_axes(mol, tol: float = 1.0e-8):
    """
    Kembalikan salinan Molecule yang terpusat di COM, sejajar sumbu inersia utama,
    dan diorientasikan ke konvensi Mulliken kanonik (sumbu C2 utama -> z, bidang Cs -> xy).
    """
    import mshqc

    z, coords = _mol_arrays(mol)
    masses = z.astype(np.float64)
    com = np.sum(coords * masses[:, None], axis=0) / max(1.0, float(np.sum(masses)))
    rel = coords - com

    inertia = np.zeros((3, 3), dtype=np.float64)
    for m, r in zip(masses, rel):
        r2 = float(np.dot(r, r))
        inertia += m * (r2 * np.eye(3) - np.outer(r, r))

    evals, evecs = np.linalg.eigh(0.5 * (inertia + inertia.T))
    evecs = _lock_degenerate_axes(rel, evals, evecs)
    aligned = rel @ evecs
    aligned[np.abs(aligned) < tol] = 0.0

    # Konvensi kanonik (permutasi siklik tangan-kanan, det = +1):
    # - Jika ada tepat satu sumbu C2 (C2v/C2h/C2), pindahkan ke sumbu z (C2z)
    # - Jika hanya Cs (1 bidang cermin), pindahkan normal bidang ke sumbu z (sxy)
    valid = _detect_valid_ops(z, aligned, tol=1.0e-4)
    c2_axes = [ax for ax in ("C2z", "C2y", "C2x") if ax in valid]
    if len(c2_axes) == 1:
        if c2_axes[0] == "C2x":
            aligned = aligned[:, [1, 2, 0]]
        elif c2_axes[0] == "C2y":
            aligned = aligned[:, [2, 0, 1]]
    elif len(c2_axes) == 0:
        if "sxy" not in valid:
            if "sxz" in valid:
                aligned = aligned[:, [2, 0, 1]]
            elif "syz" in valid:
                aligned = aligned[:, [1, 2, 0]]

    aligned[np.abs(aligned) < tol] = 0.0

    out = mshqc.Molecule(int(mol.charge()), int(mol.multiplicity()))
    for zi, xyz in zip(z, aligned):
        out.add_atom(int(zi), float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return out


def detect_abelian_symmetry(mol, tol: float = 1.0e-4) -> AbelianSymmetryInfo:
    """Deteksi subgrup Abelian maksimal secara axis-agnostic (orientasi x, y, maupun z)."""
    z, coords = _mol_arrays(mol)
    valid_perms = _detect_valid_ops(z, coords, tol)
    ops = set(valid_perms.keys())

    if {"C2z", "C2y", "C2x", "i"}.issubset(ops):
        pg = "D2h"
        req_ops = ["E", "C2z", "C2y", "C2x", "i", "sxy", "sxz", "syz"]
    elif {"C2z", "C2y", "C2x"}.issubset(ops):
        pg = "D2"
        req_ops = ["E", "C2z", "C2y", "C2x"]
    elif {"C2z", "sxz", "syz"}.issubset(ops):
        pg = "C2v"
        req_ops = ["E", "C2z", "sxz", "syz"]
    elif {"C2y", "sxy", "syz"}.issubset(ops):
        pg = "C2v"
        req_ops = ["E", "C2y", "sxy", "syz"]
    elif {"C2x", "sxy", "sxz"}.issubset(ops):
        pg = "C2v"
        req_ops = ["E", "C2x", "sxy", "sxz"]
    elif {"C2z", "i", "sxy"}.issubset(ops):
        pg = "C2h"
        req_ops = ["E", "C2z", "i", "sxy"]
    elif {"C2y", "i", "sxz"}.issubset(ops):
        pg = "C2h"
        req_ops = ["E", "C2y", "i", "sxz"]
    elif {"C2x", "i", "syz"}.issubset(ops):
        pg = "C2h"
        req_ops = ["E", "C2x", "i", "syz"]
    elif "C2z" in ops:
        pg = "C2"
        req_ops = ["E", "C2z"]
    elif "C2y" in ops:
        pg = "C2"
        req_ops = ["E", "C2y"]
    elif "C2x" in ops:
        pg = "C2"
        req_ops = ["E", "C2x"]
    elif "sxy" in ops:
        pg = "Cs"
        req_ops = ["E", "sxy"]
    elif "sxz" in ops:
        pg = "Cs"
        req_ops = ["E", "sxz"]
    elif "syz" in ops:
        pg = "Cs"
        req_ops = ["E", "syz"]
    elif "i" in ops:
        pg = "Ci"
        req_ops = ["E", "i"]
    else:
        pg = "C1"
        req_ops = ["E"]

    return AbelianSymmetryInfo(
        point_group=pg,
        operations=req_ops,
        atom_perms={op: valid_perms[op] for op in req_ops},
        op_signs={op: _ABELIAN_OPS[op] for op in req_ops},
    )


def _sph_signs(l_val: int, sx: int, sy: int, sz: int) -> List[int]:
    """Tanda refleksi/rotasi diagonal untuk harmonik sferis riil urutan libcint (l=0..3)."""
    if l_val == 0:
        return [1]
    if l_val == 1:
        # libcint spherical p: px, py, pz
        return [sx, sy, sz]
    if l_val == 2:
        # libcint spherical d (m=-2..+2): xy, yz, z^2, xz, x^2-y^2
        return [sx * sy, sy * sz, 1, sx * sz, 1]
    if l_val == 3:
        # libcint spherical f (m=-3..+3): y(3x^2-y^2), xyz, yz^2, z^3, xz^2, z(x^2-y^2), x(x^2-3y^2)
        return [sy, sx * sy * sz, sy, sz, sx, sz, sx]
    return [1] * (2 * l_val + 1)


def build_ao_symmetry_operators(mol, basis, sym: AbelianSymmetryInfo) -> Dict[str, np.ndarray]:
    """Bangun matriks representasi AO R_g (nbf x nbf) untuk setiap operasi Abelian g."""
    nbf = int(basis.n_basis_functions())
    nat = int(mol.n_atoms())
    atom_shells: List[List[Tuple[int, int, int]]] = [[] for _ in range(nat)]

    offset = 0
    for sh_idx in range(int(basis.n_shells())):
        sh = basis.shell(sh_idx)
        center = int(sh.center)
        l_val = int(sh.l()) if hasattr(sh, "l") else int(sh.angular_momentum.value)
        nf = int(sh.n_functions()) if hasattr(sh, "n_functions") else (2 * l_val + 1)
        atom_shells[center].append((offset, l_val, nf))
        offset += nf

    mats: Dict[str, np.ndarray] = {}
    for op in sym.operations:
        R = np.zeros((nbf, nbf), dtype=np.float64)
        perm = sym.atom_perms[op]
        sx, sy, sz = sym.op_signs[op]
        for atom_i in range(nat):
            atom_j = int(perm[atom_i])
            for (off_i, l_i, nf_i), (off_j, l_j, nf_j) in zip(atom_shells[atom_i], atom_shells[atom_j]):
                if l_i != l_j or nf_i != nf_j:
                    raise ValueError("Basis set pada atom ekuivalen simetri tidak cocok.")
                signs = _sph_signs(l_i, sx, sy, sz)
                for m in range(nf_i):
                    R[off_j + m, off_i + m] = float(signs[m])
        mats[op] = R
    return mats


def symmetrize_ao_matrix(M: np.ndarray, op_mats: Dict[str, np.ndarray]) -> np.ndarray:
    """Proyeksikan matriks AO M ke subruang invarian simetri: (1/|G|) sum_g R_g M R_g^T."""
    acc = np.zeros_like(M, dtype=np.float64)
    for R in op_mats.values():
        acc += R @ M @ R.T
    out = acc / float(len(op_mats))
    return 0.5 * (out + out.T)


def label_mo_irreps(
    mol,
    basis,
    C: np.ndarray,
    S: np.ndarray,
    sym: Optional[AbelianSymmetryInfo] = None,
    tol: float = 0.15,
) -> List[str]:
    """Beri label irrep Abelian pada setiap kolom orbital MO di C."""
    info = sym if sym is not None else detect_abelian_symmetry(mol)
    table = _CHAR_TABLES[info.point_group]
    if info.point_group == "C1":
        return ["A"] * C.shape[1]

    op_mats = build_ao_symmetry_operators(mol, basis, info)
    nmo = C.shape[1]
    SC = S @ C
    chars = np.zeros((nmo, len(info.operations)), dtype=np.float64)
    for k, op in enumerate(info.operations):
        R = op_mats[op]
        chars[:, k] = np.einsum("mp,mn,np->p", SC, R, C, optimize=True)

    labels: List[str] = []
    for p in range(nmo):
        best_irrep = "mix"
        best_err = 1.0e9
        for irrep_name, vec in table.items():
            err = float(np.max(np.abs(chars[p] - np.asarray(vec, dtype=np.float64))))
            if err < best_err:
                best_err = err
                best_irrep = irrep_name
        labels.append(best_irrep if best_err <= tol else "mix")
    return labels
