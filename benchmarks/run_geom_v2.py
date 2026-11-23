#!/usr/bin/env python3
"""
mshqc v2.0 Pure Thin-Core Geometry Optimization & Vibrational Benchmark Runner (DF Mode).
Mencakup RHF/UHF, MP2, OMP2, kappa-OOMP2 (gradien analitik) dan OMP3.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import numpy as np

CPU_AVAIL = os.cpu_count() or 4
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", str(min(4, CPU_AVAIL))))
os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("TBLIS_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

import mshqc

GEOM_SYSTEMS = {
    "H2":  {"charge": 0, "mult": 1, "ref": "rhf", "basis": "cc-pVTZ", "atoms": [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.80)], "is_triatomic": False, "methods": ["rhf", "omp2", "kappa-omp2", "omp3"]},
    "HF":  {"charge": 0, "mult": 1, "ref": "rhf", "basis": "cc-pVDZ", "atoms": [("F", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.97)], "is_triatomic": False, "methods": ["rhf", "omp2", "kappa-omp2", "omp3"]},
    "OH":  {"charge": 0, "mult": 2, "ref": "uhf", "basis": "cc-pVDZ", "atoms": [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.03)], "is_triatomic": False, "methods": ["uhf", "omp2", "kappa-omp2", "omp3"]},
    "NH":  {"charge": 0, "mult": 3, "ref": "uhf", "basis": "cc-pVDZ", "atoms": [("N", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.08)], "is_triatomic": False, "methods": ["uhf", "omp2", "kappa-omp2", "omp3"]},
    "H2O": {"charge": 0, "mult": 1, "ref": "rhf", "basis": "cc-pVDZ", "atoms": [("O", 0.0, 0.0, 0.125), ("H", 0.0, 0.780, -0.490), ("H", 0.0, -0.780, -0.490)], "is_triatomic": True, "methods": ["rhf", "omp2", "kappa-omp2", "omp3"]},
}


def compute_angle_deg(coords_ang: np.ndarray) -> float:
    if coords_ang.shape[0] < 3:
        return 0.0
    v1 = coords_ang[1] - coords_ang[0]
    v2 = coords_ang[2] - coords_ang[0]
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main() -> None:
    parser = argparse.ArgumentParser(description="mshqc v2.0 Geometry & Vibrations Benchmark")
    parser.add_argument("--out", default="mshqc_geom_v2_raw.json")
    parser.add_argument("--quick", action="store_true", help="Hanya H2 & H2O (omp2, kappa-omp2)")
    args = parser.parse_args()

    db = {}
    hdr = (
        f"| {'System':<6} | {'Basis':<7} | {'Method':<10} | {'R_eq (A)':<10} | {'Angle (deg)':<11} | "
        f"{'E_opt (Ha)':<16} | {'Max Freq (cm-1)':<15} | {'ZPE (kcal/mol)':<14} | {'Iter':<4} | {'Time (s)':<8} |"
    )
    sep = "-" * len(hdr)
    print("\n" + "=" * len(hdr), flush=True)
    print(f" MSHQC v{mshqc.__version__} GEOMETRY & HARMONIC VIBRATIONS BENCHMARK (DF MODE)", flush=True)
    print("=" * len(hdr), flush=True)
    print(sep + "\n" + hdr + "\n" + sep, flush=True)

    targets = {"H2": GEOM_SYSTEMS["H2"], "H2O": GEOM_SYSTEMS["H2O"]} if args.quick else GEOM_SYSTEMS
    for sys_name, spec in targets.items():
        basis = spec["basis"]
        methods = ["omp2", "kappa-omp2"] if args.quick else spec["methods"]
        for m in methods:
            mol = mshqc.Molecule(charge=spec["charge"], multiplicity=spec["mult"])
            for sym, x, y, z in spec["atoms"]:
                mol.add_atom(sym, x, y, z, units="angstrom")

            t0 = time.perf_counter()
            opt = mol.optimize_geometry(
                method=m, eri_mode="df", basis=basis, reference=spec["ref"],
                max_iter=40, g_convergence="gau_verytight",
            )
            t_opt = time.perf_counter() - t0
            coords_ang = np.asarray(opt.coordinates_angstrom)
            r_eq = float(np.linalg.norm(coords_ang[1] - coords_ang[0]))
            ang = compute_angle_deg(coords_ang) if spec["is_triatomic"] else 0.0

            vib = mol.frequencies(method=m, eri_mode="df", basis=basis, reference=spec["ref"])
            freqs = [float(f) for f in np.asarray(vib.frequencies_cm1) if f > 50.0]
            max_f = max(freqs) if freqs else 0.0
            zpe = float(vib.zpe_kcal_mol)

            db[f"{sys_name}|{m.upper()}"] = {
                "system": sys_name, "basis": basis, "method": m.upper(),
                "r_eq": r_eq, "angle_eq": ang, "e_opt": float(opt.final_energy),
                "iterations": int(opt.iterations), "converged": bool(opt.converged),
                "freqs_cm1": freqs, "max_freq_cm1": max_f, "zpe_kcal": zpe, "time_opt_s": t_opt,
            }
            ang_s = f"{ang:11.4f}" if ang > 0 else f"{'-':>11}"
            print(
                f"| {sys_name:<6} | {basis:<7} | {m.upper():<10} | {r_eq:<10.6f} | {ang_s} | "
                f"{opt.final_energy:<16.10f} | {max_f:<15.2f} | {zpe:<14.4f} | {opt.iterations:<4d} | {t_opt:<8.3f} |",
                flush=True,
            )

    print(sep, flush=True)
    Path(args.out).write_text(json.dumps(db, indent=2), encoding="utf-8")
    print(f"[SELESAI] Data geometri v2.0 disimpan ke: {args.out}", flush=True)


if __name__ == "__main__":
    main()
