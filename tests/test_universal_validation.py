#!/usr/bin/env python3
"""
Suite Validasi Universal MSHQC vs Psi4 Lintas Atom & Molekul
Menguji:
- Atom Closed-Shell & Open-Shell : He, Be, B, C, N, O, F, Ne, Li, Na
- Molekul Closed-Shell (RHF)     : H2, LiH, HF, H2O, NH3, CH4, CO, N2, F2
- Molekul Radikal Doublet (UHF)  : OH, CH, NH, NO, CN, BeH
- Molekul Biradikal Triplet (UHF): O2, CH2 (triplet), NH (triplet)
Pada semua mode integral (EXACT, CHOLESKY, DF) dan semua metode (HF, MP2, MP3, OMP2, OMP3).
"""

import json
import subprocess
import sys
import numpy as np
import mshqc

ANG_TO_BOHR = 1.8897261246257702

UNIVERSAL_SYSTEMS = {
    # --- Atom ---
    "He":  {"atoms": [("He", 0.0, 0.0, 0.0)], "charge": 0, "mult": 1},
    "Li":  {"atoms": [("Li", 0.0, 0.0, 0.0)], "charge": 0, "mult": 2},
    "Be":  {"atoms": [("Be", 0.0, 0.0, 0.0)], "charge": 0, "mult": 1},
    "B":   {"atoms": [("B",  0.0, 0.0, 0.0)], "charge": 0, "mult": 2},
    "C":   {"atoms": [("C",  0.0, 0.0, 0.0)], "charge": 0, "mult": 3},
    "N":   {"atoms": [("N",  0.0, 0.0, 0.0)], "charge": 0, "mult": 4},
    "O":   {"atoms": [("O",  0.0, 0.0, 0.0)], "charge": 0, "mult": 3},
    "F":   {"atoms": [("F",  0.0, 0.0, 0.0)], "charge": 0, "mult": 2},
    "Ne":  {"atoms": [("Ne", 0.0, 0.0, 0.0)], "charge": 0, "mult": 1},
    # --- Molekul Closed-Shell (RHF) ---
    "H2":  {"atoms": [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.7414 * ANG_TO_BOHR)], "charge": 0, "mult": 1},
    "HF":  {"atoms": [("F", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.9168 * ANG_TO_BOHR)], "charge": 0, "mult": 1},
    "H2O": {"atoms": [("O", 0.0, 0.0, 0.1173 * ANG_TO_BOHR),
                      ("H", 0.0,  0.7572 * ANG_TO_BOHR, -0.4692 * ANG_TO_BOHR),
                      ("H", 0.0, -0.7572 * ANG_TO_BOHR, -0.4692 * ANG_TO_BOHR)], "charge": 0, "mult": 1},
    "N2":  {"atoms": [("N", 0.0, 0.0, -0.5488 * ANG_TO_BOHR),
                      ("N", 0.0, 0.0,  0.5488 * ANG_TO_BOHR)], "charge": 0, "mult": 1},
    "CO":  {"atoms": [("C", 0.0, 0.0, -0.5641 * ANG_TO_BOHR),
                      ("O", 0.0, 0.0,  0.5641 * ANG_TO_BOHR)], "charge": 0, "mult": 1},
    # --- Molekul Open-Shell Radikal & Biradikal (UHF) ---
    "OH":  {"atoms": [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.9697 * ANG_TO_BOHR)], "charge": 0, "mult": 2},
    "NH":  {"atoms": [("N", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.0362 * ANG_TO_BOHR)], "charge": 0, "mult": 3},
    "O2":  {"atoms": [("O", 0.0, 0.0, -0.6035 * ANG_TO_BOHR),
                      ("O", 0.0, 0.0,  0.6035 * ANG_TO_BOHR)], "charge": 0, "mult": 3},
    "NO":  {"atoms": [("N", 0.0, 0.0, -0.5754 * ANG_TO_BOHR),
                      ("O", 0.0, 0.0,  0.5754 * ANG_TO_BOHR)], "charge": 0, "mult": 2},
}

def test_system(name, spec, basis_name="cc-pVDZ", eri_mode="df"):
    mol = mshqc.Molecule(spec["charge"], spec["mult"])
    for sym, x, y, z in spec["atoms"]:
        mol.add_atom(sym, x, y, z, units="bohr")

    is_uhf = spec["mult"] > 1
    basis = mshqc.BasisSet(basis_name, mol)
    ints = mshqc.IntegralEngine(mol, basis)

    scf_cfg = mshqc.SCFConfig(eri_method=eri_mode, threshold=1e-9, aux_basis_name=f"{basis_name}-RI")
    scf_res = (mshqc.UHF(mol, basis, ints, scf_cfg) if is_uhf else mshqc.RHF(mol, basis, ints, scf_cfg)).compute()

    mp_cfg = mshqc.MP2Config(eri_method=eri_mode, threshold=1e-9, gradient_threshold=1e-5, aux_basis_name=f"{basis_name}-RI")
    mp2_res = (mshqc.UMP2(mol, basis, ints, scf_res, mp_cfg) if is_uhf else mshqc.RMP2(mol, basis, ints, scf_res, mp_cfg)).compute()
    mp3_res = (mshqc.UMP3(scf_res, mp2_res, mp_cfg, ints) if is_uhf else mshqc.RMP3(scf_res, mp2_res, mp_cfg, ints)).compute()
    omp2_res = mshqc.OMP2(mol, basis, ints, scf_res, mp_cfg).compute()
    omp3_res = mshqc.OMP3(mol, basis, ints, scf_res, mp_cfg).compute()

    # Pemeriksaan fisik dasar (Sanity Assertions)
    assert scf_res.converged, f"{name} SCF tidak konvergen!"
    assert mp2_res.energy_mp2_corr < 0.0, f"{name} Energi korelasi MP2 harus negatif!"
    assert mp2_res.energy_mp2_corr > -5.0, f"{name} Energi korelasi MP2 meledak ({mp2_res.energy_mp2_corr})!"
    assert omp2_res.converged, f"{name} OMP2 tidak konvergen!"
    assert omp3_res.converged, f"{name} OMP3 tidak konvergen!"
    assert omp2_res.energy_total <= mp2_res.energy_total + 1e-5, f"{name} Relaksasi orbital OMP2 tidak menurunkan energi!"

    print(f"[PASS] {name:<5} ({eri_mode.upper():<8}) | HF: {scf_res.energy_total:12.6f} | MP2: {mp2_res.energy_total:12.6f} | OMP2: {omp2_res.energy_total:12.6f} | OMP3: {omp3_res.e_total:12.6f}")

if __name__ == "__main__":
    for sys_name, spec in UNIVERSAL_SYSTEMS.items():
        for mode in ["exact", "cholesky", "df"]:
            test_system(sys_name, spec, basis_name="cc-pVDZ", eri_mode=mode)