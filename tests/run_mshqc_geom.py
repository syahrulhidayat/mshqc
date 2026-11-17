#!/usr/bin/env python3
"""
Runner Mandiri MSHQC untuk Benchmark Optimisasi Geometri & Vibrasi Multi-Sistem
Output: mshqc_geom_raw.json, mshqc_geom_test.txt
"""

import os
import sys
import time
import json
import math
from pathlib import Path
import numpy as np
import mshqc

CPU_AVAIL = os.cpu_count() or 4
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", str(min(4, CPU_AVAIL))))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["TBLIS_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

GEOM_SYSTEMS = {
    "H2": {
        "charge": 0, "mult": 1, "ref": "rhf", "basis": "cc-pVTZ",
        "atoms": [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.80)],
        "bond_key": "H1-H2", "is_triatomic": False,
        "methods": ["rhf", "rmp2", "rmp3", "omp2", "omp3"],
    },
    "HF": {
        "charge": 0, "mult": 1, "ref": "rhf", "basis": "cc-pVDZ",
        "atoms": [("F", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.97)],
        "bond_key": "F1-H2", "is_triatomic": False,
        "methods": ["rhf", "rmp2", "rmp3", "omp2", "omp3"],
    },
    "OH": {
        "charge": 0, "mult": 2, "ref": "uhf", "basis": "cc-pVDZ",
        "atoms": [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.03)],
        "bond_key": "O1-H2", "is_triatomic": False,
        "methods": ["uhf", "ump2", "ump3", "omp2", "omp3"],
    },
    "NH": {
        "charge": 0, "mult": 3, "ref": "uhf", "basis": "cc-pVDZ",
        "atoms": [("N", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.08)],
        "bond_key": "N1-H2", "is_triatomic": False,
        "methods": ["uhf", "ump2", "ump3", "omp2", "omp3"],
    },
    "H2O": {
        "charge": 0, "mult": 1, "ref": "rhf", "basis": "cc-pVDZ",
        "atoms": [
            ("O", 0.0,  0.0000,  0.1250),
            ("H", 0.0,  0.7800, -0.4900),
            ("H", 0.0, -0.7800, -0.4900),
        ],
        "bond_key": "O1-H2", "is_triatomic": True,
        "methods": ["rhf", "rmp2", "omp2", "omp3"],
    },
}


def compute_angle_deg(coords_ang: np.ndarray) -> float:
    if coords_ang.shape[0] < 3:
        return 0.0
    v1 = coords_ang[1] - coords_ang[0]
    v2 = coords_ang[2] - coords_ang[0]
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main():
    db = {}
    lines = []

    hdr = (
        f"| {'System':<6} | {'Basis':<7} | {'Method':<8} | {'R_eq (A)':<10} | {'Angle (deg)':<11} | "
        f"{'E_opt (Ha)':<16} | {'Max Freq (cm-1)':<15} | {'ZPE (kcal/mol)':<14} | {'Iter':<4} | {'Time (s)':<8} |"
    )
    sep = "-" * len(hdr)
    print("\n" + "=" * len(hdr), flush=True)
    print(" MSHQC GEOMETRY OPTIMIZATION & HARMONIC VIBRATIONAL BENCHMARK SUITE (DF MODE)", flush=True)
    print("=" * len(hdr), flush=True)
    print(sep, flush=True)
    print(hdr, flush=True)
    print(sep, flush=True)
    lines.extend([sep, hdr, sep])

    for sys_name, spec in GEOM_SYSTEMS.items():
        basis = spec["basis"]
        for m in spec["methods"]:
            mol = mshqc.Molecule(charge=spec["charge"], multiplicity=spec["mult"])
            for sym, x, y, z in spec["atoms"]:
                mol.add_atom(sym, x, y, z, units="angstrom")

            t0 = time.perf_counter()
            opt_res = mol.optimize_geometry(
                method=m,
                eri_mode="df",
                basis=basis,
                reference=spec["ref"],
                max_iter=40,
                g_rms_tol=1.0e-6,
                fd_step_bohr=5.0e-4,
                verbose=False,
            )
            t_opt = time.perf_counter() - t0

            coords_ang = np.asarray(opt_res.coordinates_angstrom)
            r_eq = float(np.linalg.norm(coords_ang[1] - coords_ang[0]))
            angle_eq = compute_angle_deg(coords_ang) if spec["is_triatomic"] else 0.0

            vib_res = mol.frequencies(
                method=m,
                eri_mode="df",
                basis=basis,
                reference=spec["ref"],
                verbose=False,
            )
            freqs = [float(f) for f in np.asarray(vib_res.frequencies_cm1) if f > 50.0]
            max_freq = max(freqs) if freqs else 0.0
            zpe_kcal = float(vib_res.zpe_kcal_mol)

            key = f"{sys_name}|{m.upper()}"
            db[key] = {
                "system": sys_name,
                "basis": basis,
                "method": m.upper(),
                "ref": spec["ref"].upper(),
                "r_eq": r_eq,
                "angle_eq": angle_eq,
                "e_opt": float(opt_res.final_energy),
                "g_rms": float(opt_res.final_g_rms),
                "iterations": int(opt_res.iterations),
                "converged": bool(opt_res.converged),
                "freqs_cm1": freqs,
                "max_freq_cm1": max_freq,
                "zpe_kcal": zpe_kcal,
                "time_opt_s": t_opt,
            }

            ang_str = f"{angle_eq:.4f}" if spec["is_triatomic"] else "   -   "
            row_str = (
                f"| {sys_name:<6} | {basis:<7} | {m.upper():<8} | {r_eq:<10.6f} | {ang_str:<11} | "
                f"{opt_res.final_energy:<16.10f} | {max_freq:<15.2f} | {zpe_kcal:<14.4f} | "
                f"{opt_res.iterations:<4d} | {t_opt:<8.3f} |"
            )
            print(row_str, flush=True)
            lines.append(row_str)

    print(sep + "\n", flush=True)
    lines.append(sep)

    Path("mshqc_geom_raw.json").write_text(json.dumps(db, indent=2), encoding="utf-8")
    Path("mshqc_geom_test.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
