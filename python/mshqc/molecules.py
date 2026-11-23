# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
Antarmuka ringkas pembangun atom & molekul untuk mshqc:
  - `mshqc.atom("O")`          : atom tunggal di titik asal dengan multiplisitas ground-state Hund
  - `mshqc.molecule("H2O")`    : molekul preset, nama atom, string XYZ (`;` / multiline), atau file `.xyz`
  - `mshqc.list_molecules()`   : daftar nama molekul preset yang tersedia
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import mshqc

SYMBOL_TO_Z: Dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
    "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
    "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
    "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92, "Np": 93, "Pu": 94,
    "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101, "No": 102,
    "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110,
    "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118,
}

# Multiplisitas ground-state atom netral (aturan Hund, Z = 1..54)
GROUND_STATE_MULTIPLICITY: Dict[int, int] = {
    1: 2, 2: 1,
    3: 2, 4: 1, 5: 2, 6: 3, 7: 4, 8: 3, 9: 2, 10: 1,
    11: 2, 12: 1, 13: 2, 14: 3, 15: 4, 16: 3, 17: 2, 18: 1,
    19: 2, 20: 1, 21: 2, 22: 3, 23: 4, 24: 7, 25: 6, 26: 5, 27: 4, 28: 3, 29: 2, 30: 1,
    31: 2, 32: 3, 33: 4, 34: 3, 35: 2, 36: 1,
    37: 2, 38: 1, 39: 2, 40: 3, 41: 6, 42: 7, 43: 6, 44: 5, 45: 4, 46: 1, 47: 2, 48: 1,
    49: 2, 50: 3, 51: 4, 52: 3, 53: 2, 54: 1,
}

# Geometri referensi standar (koordinat dalam Angstrom): (charge, multiplicity, [(sym, x, y, z), ...])
PRESET_MOLECULES: Dict[str, Tuple[int, int, List[Tuple[str, float, float, float]]]] = {
    "H2": (0, 1, [("H", 0.0, 0.0, -0.37), ("H", 0.0, 0.0, 0.37)]),
    "LiH": (0, 1, [("Li", 0.0, 0.0, -0.7975), ("H", 0.0, 0.0, 0.7975)]),
    "HF": (0, 1, [("F", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.9168)]),
    "HCl": (0, 1, [("Cl", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.2746)]),
    "N2": (0, 1, [("N", 0.0, 0.0, -0.5488), ("N", 0.0, 0.0, 0.5488)]),
    "CO": (0, 1, [("C", 0.0, 0.0, -0.5642), ("O", 0.0, 0.0, 0.5642)]),
    "O2": (0, 3, [("O", 0.0, 0.0, -0.6035), ("O", 0.0, 0.0, 0.6035)]),
    "H2O": (0, 1, [("O", 0.0, 0.0, 0.1173), ("H", 0.0, 0.7572, -0.4692), ("H", 0.0, -0.7572, -0.4692)]),
    "H2S": (0, 1, [("S", 0.0, 0.0, 0.1030), ("H", 0.0, 0.9616, -0.8240), ("H", 0.0, -0.9616, -0.8240)]),
    "NH3": (0, 1, [
        ("N", 0.0, 0.0, 0.1165), ("H", 0.0, 0.9397, -0.2718),
        ("H", 0.8138, -0.46985, -0.2718), ("H", -0.8138, -0.46985, -0.2718),
    ]),
    "CH4": (0, 1, [
        ("C", 0.0, 0.0, 0.0),
        ("H", 0.629118, 0.629118, 0.629118),
        ("H", -0.629118, -0.629118, 0.629118),
        ("H", -0.629118, 0.629118, -0.629118),
        ("H", 0.629118, -0.629118, -0.629118),
    ]),
    "OH-": (-1, 1, [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.9640)]),
    "H3O+": (1, 1, [
        ("O", 0.0, 0.0, 0.0650), ("H", 0.0, 0.9570, -0.2100),
        ("H", 0.8288, -0.4785, -0.2100), ("H", -0.8288, -0.4785, -0.2100),
    ]),
    "OH": (0, 2, [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.9697)]),
    "SH": (0, 2, [("S", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.3409)]),
    "NH": (0, 3, [("N", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.0362)]),
    "NH2": (0, 2, [("N", 0.0, 0.0, 0.1415), ("H", 0.0, 0.8015, -0.5060), ("H", 0.0, -0.8015, -0.5060)]),
    "CH2_triplet": (0, 3, [("C", 0.0, 0.0, 0.1050), ("H", 0.0, 0.9920, -0.3150), ("H", 0.0, -0.9920, -0.3150)]),
    "CH3": (0, 2, [
        ("C", 0.0, 0.0, 0.0), ("H", 0.0, 1.0790, 0.0),
        ("H", 0.934442, -0.5395, 0.0), ("H", -0.934442, -0.5395, 0.0),
    ]),
    "C6H6": (0, 1, [
        ("C",  1.3965,  0.0000, 0.0), ("C",  0.6982,  1.2094, 0.0),
        ("C", -0.6982,  1.2094, 0.0), ("C", -1.3965,  0.0000, 0.0),
        ("C", -0.6982, -1.2094, 0.0), ("C",  0.6982, -1.2094, 0.0),
        ("H",  2.4825,  0.0000, 0.0), ("H",  1.2412,  2.1499, 0.0),
        ("H", -1.2412,  2.1499, 0.0), ("H", -2.4825,  0.0000, 0.0),
        ("H", -1.2412, -2.1499, 0.0), ("H",  1.2412, -2.1499, 0.0),
    ]),
}
_PRESET_LOOKUP: Dict[str, str] = {k.lower(): k for k in PRESET_MOLECULES}


def _resolve_atomic_number(sym_or_z: Union[str, int]) -> int:
    if isinstance(sym_or_z, int):
        return sym_or_z
    tok = sym_or_z.strip()
    if tok.isdigit():
        return int(tok)
    cap = tok.capitalize()
    if cap not in SYMBOL_TO_Z:
        raise ValueError(f"Simbol unsur tidak dikenali: {sym_or_z!r}")
    return SYMBOL_TO_Z[cap]


def _default_atom_multiplicity(z: int, charge: int) -> int:
    if charge == 0 and z in GROUND_STATE_MULTIPLICITY:
        return GROUND_STATE_MULTIPLICITY[z]
    n_elec = z - charge
    return 2 if (n_elec % 2 == 1) else 1


def list_molecules() -> List[str]:
    """Kembalikan daftar nama molekul preset bawaan."""
    return sorted(PRESET_MOLECULES.keys())


def atom(
    symbol_or_z: Union[str, int],
    charge: int = 0,
    multiplicity: Optional[int] = None,
) -> mshqc.Molecule:
    """Bangun atom tunggal di (0, 0, 0) dengan multiplisitas ground-state otomatis."""
    z = _resolve_atomic_number(symbol_or_z)
    mult = _default_atom_multiplicity(z, int(charge)) if multiplicity is None else int(multiplicity)
    mol = mshqc.Molecule(int(charge), mult)
    mol.add_atom(z, 0.0, 0.0, 0.0)
    return mol


def _parse_xyz_text(text: str, scale: float) -> List[Tuple[int, float, float, float]]:
    entries: List[Tuple[int, float, float, float]] = []
    raw_lines = [seg.strip() for line in text.strip().splitlines() for seg in line.split(";")]
    for line in raw_lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 4:
            try:
                z = _resolve_atomic_number(parts[0])
                entries.append((z, float(parts[1]) * scale, float(parts[2]) * scale, float(parts[3]) * scale))
            except ValueError:
                continue
    if not entries:
        raise ValueError(f"Tidak dapat mem-parse geometri XYZ dari input: {text!r}")
    return entries


def molecule(
    spec: Union[mshqc.Molecule, str, int, List[Tuple]],
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
    units: str = "angstrom",
) -> mshqc.Molecule:
    """
    Bangun `mshqc.Molecule` secara ringkas dari salah satu format:
      - nama molekul preset: `"H2O"`, `"OH-"`, `"O2"`, `"CH4"`, `"C6H6"`, ...
      - simbol/nomor atom  : `"O"`, `"C"`, `"Mn"`, `8` (otomatis ground-state Hund)
      - path file `.xyz`   : `"water.xyz"`
      - string XYZ         : `"O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"`
      - list tuple         : `[("O", 0, 0, 0.1173), ...]`
    """
    if isinstance(spec, mshqc.Molecule):
        return spec
    if isinstance(spec, int):
        return atom(spec, charge=0 if charge is None else charge, multiplicity=multiplicity)

    scale = mshqc.angstrom_to_bohr(1.0) if units.lower().startswith("ang") else 1.0

    if isinstance(spec, str):
        s = spec.strip()
        if "\n" not in s and ";" not in s and s.lower().endswith(".xyz") and Path(s).is_file():
            lines = Path(s).read_text(encoding="utf-8").splitlines()
            body = "\n".join(lines[2:]) if (len(lines) >= 3 and lines[0].strip().isdigit()) else "\n".join(lines)
            atoms_bohr = _parse_xyz_text(body, mshqc.angstrom_to_bohr(1.0))
            q = 0 if charge is None else int(charge)
            n_elec = sum(a[0] for a in atoms_bohr) - q
            m = (2 if (n_elec % 2 == 1) else 1) if multiplicity is None else int(multiplicity)
            mol = mshqc.Molecule(q, m)
            for z, x, y, zc in atoms_bohr:
                mol.add_atom(z, x, y, zc)
            return mol

        if "\n" not in s and ";" not in s and len(s.split()) == 1:
            low = s.lower()
            if low in _PRESET_LOOKUP:
                def_q, def_m, coords = PRESET_MOLECULES[_PRESET_LOOKUP[low]]
                q = def_q if charge is None else int(charge)
                if multiplicity is not None:
                    m = int(multiplicity)
                elif q == def_q:
                    m = def_m
                else:
                    n_elec = sum(_resolve_atomic_number(a[0]) for a in coords) - q
                    m = 2 if (n_elec % 2 == 1) else 1
                ang2b = mshqc.angstrom_to_bohr(1.0)
                mol = mshqc.Molecule(q, m)
                for sym, x, y, zc in coords:
                    mol.add_atom(_resolve_atomic_number(sym), x * ang2b, y * ang2b, zc * ang2b)
                return mol
            if s.isdigit() or s.capitalize() in SYMBOL_TO_Z:
                return atom(s, charge=0 if charge is None else charge, multiplicity=multiplicity)
            raise ValueError(
                f"Molekul/atom {spec!r} tidak dikenali. Preset tersedia: {', '.join(list_molecules())}"
            )

        atoms_bohr = _parse_xyz_text(s, scale)
    else:
        atoms_bohr = [
            (_resolve_atomic_number(it[0]), float(it[1]) * scale, float(it[2]) * scale, float(it[3]) * scale)
            for it in spec
        ]

    q = 0 if charge is None else int(charge)
    if multiplicity is not None:
        m = int(multiplicity)
    elif len(atoms_bohr) == 1:
        m = _default_atom_multiplicity(atoms_bohr[0][0], q)
    else:
        n_elec = sum(a[0] for a in atoms_bohr) - q
        m = 2 if (n_elec % 2 == 1) else 1

    mol = mshqc.Molecule(q, m)
    for z, x, y, zc in atoms_bohr:
        mol.add_atom(z, x, y, zc)
    return mol


__all__ = [
    "SYMBOL_TO_Z",
    "GROUND_STATE_MULTIPLICITY",
    "PRESET_MOLECULES",
    "atom",
    "molecule",
    "list_molecules",
]
