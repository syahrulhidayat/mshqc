#!/usr/bin/env python3
"""
Benchmark Validasi & Performa Optimisasi Geometri + Frekuensi Vibrasi:
mshqc (DF-OMP2 / DF-OMP3) vs Psi4 (dfocc OMP2 / OMP3)
"""

import time
import numpy as np
import mshqc
from mshqc.geometry import optimize_geometry, compute_frequencies, scan_pes

BOHR_TO_ANGSTROM = 0.529177210903


def build_oh_radical(r_angstrom: float = 1.02) -> "mshqc.Molecule":
    """Membangun radikal hidroksil (.OH, doublet, charge=0, mult=2) dengan jarak awal belum rileks."""
    r_bohr = r_angstrom / BOHR_TO_ANGSTROM
    mol = mshqc.Molecule(0, 2)
    mol.add_atom(8, 0.0, 0.0, 0.0)
    mol.add_atom(1, 0.0, 0.0, r_bohr)
    return mol


def run_mshqc_benchmark(method: str = "omp2", basis: str = "cc-pVDZ", eri_mode: str = "df"):
    mol = build_oh_radical(r_angstrom=1.02)
    print(f"\n>>> Menjalankan mshqc mol.optimize_geometry(method='{method}', eri_mode='{eri_mode}')...")
    if hasattr(mol, "optimize_geometry"):
        opt_res = mol.optimize_geometry(method=method, basis=basis, eri_mode=eri_mode)
    else:
        opt_res = optimize_geometry(mol, method=method, basis=basis, eri_mode=eri_mode)

    freq_res = compute_frequencies(mol, method=method, basis=basis, eri_mode=eri_mode)
    return opt_res, freq_res


def run_psi4_reference(method: str = "omp2", basis: str = "cc-pVDZ"):
    try:
        import psi4
    except ImportError:
        print("[INFO] Psi4 tidak terinstal di environment ini; melewati pembanding Psi4.")
        return None

    psi4.core.be_quiet()
    psi4_mol = psi4.geometry("""
    0 2
    O  0.00000000  0.00000000  0.00000000
    H  0.00000000  0.00000000  1.02000000
    symmetry c1
    """)
    psi4.set_options({
        "basis": basis,
        "reference": "uhf",
        "scf_type": "df",
        "orb_opt": "true",
        "e_convergence": 1e-10,
        "d_convergence": 1e-8,
        "g_convergence": "gau_tight",
    })
    t0 = time.perf_counter()
    e_psi4 = psi4.optimize(method, molecule=psi4_mol)
    dt = time.perf_counter() - t0
    coords_b = np.asarray(psi4_mol.geometry())
    r_eq_ang = float(np.linalg.norm(coords_b[1] - coords_b[0])) * BOHR_TO_ANGSTROM
    return {"energy_ha": float(e_psi4), "r_eq_angstrom": r_eq_ang, "time_s": dt}


if __name__ == "__main__":
    for m in ("omp2", "omp3"):
        opt_mshqc, freq_mshqc = run_mshqc_benchmark(method=m, basis="cc-pVDZ", eri_mode="df")
        psi4_ref = run_psi4_reference(method=m, basis="cc-pVDZ")
        if psi4_ref is not None:
            r_mshqc = list(opt_mshqc.bond_lengths_angstrom.values())[0]
            de_uh = (opt_mshqc.final_energy - psi4_ref["energy_ha"]) * 1e6
            dr_ma = (r_mshqc - psi4_ref["r_eq_angstrom"]) * 1e3
            print(f"\n[KOMPARASI {m.upper()} vs PSI4]")
            print(f"  mshqc R_eq : {r_mshqc:.6f} A | Psi4 R_eq : {psi4_ref['r_eq_angstrom']:.6f} A | |dR| = {abs(dr_ma):.3f} mA")
            print(f"  mshqc E_opt: {opt_mshqc.final_energy:.10f} Ha | Psi4 E_opt: {psi4_ref['energy_ha']:.10f} Ha | dE = {de_uh:+.2f} uHa")
