#!/usr/bin/env python3
"""
mshqc v2.0 Pure Thin-Core Heavy Atoms (Z=18..36) & Large Molecules Benchmark Runner.
Membandingkan DF-RHF, DF-OMP2, O(N^4) Zero-T2 LT-SOS-OMP2, dan DF-OMP3.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

CPU_AVAIL = os.cpu_count() or 4
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", str(min(4, CPU_AVAIL))))
os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("TBLIS_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

import mshqc

BENZENE = [
    ("C", 1.3960, 0.0000, 0.0), ("C", 0.6980, 1.2090, 0.0),
    ("C", -0.6980, 1.2090, 0.0), ("C", -1.3960, 0.0000, 0.0),
    ("C", -0.6980, -1.2090, 0.0), ("C", 0.6980, -1.2090, 0.0),
    ("H", 2.4820, 0.0000, 0.0), ("H", 1.2410, 2.1495, 0.0),
    ("H", -1.2410, 2.1495, 0.0), ("H", -2.4820, 0.0000, 0.0),
    ("H", -1.2410, -2.1495, 0.0), ("H", 1.2410, -2.1495, 0.0),
]
NAPHTHALENE = [
    ("C", 0.0000, 0.7100, 0.0), ("C", 0.0000, -0.7100, 0.0),
    ("C", 1.2430, 1.4040, 0.0), ("C", 2.4310, 0.7080, 0.0),
    ("C", 2.4310, -0.7080, 0.0), ("C", 1.2430, -1.4040, 0.0),
    ("C", -1.2430, 1.4040, 0.0), ("C", -2.4310, 0.7080, 0.0),
    ("C", -2.4310, -0.7080, 0.0), ("C", -1.2430, -1.4040, 0.0),
    ("H", 1.2450, 2.4900, 0.0), ("H", 3.3740, 1.2460, 0.0),
    ("H", 3.3740, -1.2460, 0.0), ("H", 1.2450, -2.4900, 0.0),
    ("H", -1.2450, 2.4900, 0.0), ("H", -3.3740, 1.2460, 0.0),
    ("H", -3.3740, -1.2460, 0.0), ("H", -1.2450, -2.4900, 0.0),
]
CASES = [
    ("Ar (Z=18, FC)",      [("Ar", 0.0, 0.0, 0.0)], 0, 1, "cc-pVTZ", True, True),
    ("HBr (Z=35, FC)",     [("Br", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.414)], 0, 1, "cc-pVDZ", True, True),
    ("Kr (Z=36, FC)",      [("Kr", 0.0, 0.0, 0.0)], 0, 1, "cc-pVTZ", True, True),
    ("C6H6 (114 BF, FC)",  BENZENE, 0, 1, "cc-pVDZ", True, True),
    ("C10H8 (180 BF, FC)", NAPHTHALENE, 0, 1, "cc-pVDZ", True, True),
    ("C6H6 (264 BF, FC)",  BENZENE, 0, 1, "cc-pVTZ", True, False),
]


def peak_rss_mb() -> float:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/status", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        return float(line.split()[1]) / 1024.0
        except OSError:
            pass
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss) / (1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="mshqc v2.0 Heavy & Large Molecules Benchmark")
    parser.add_argument("--out", default="mshqc_heavy_large_v2_raw.json")
    parser.add_argument("--quick", action="store_true", help="Hanya Ar (Z=18) & Benzene 114 BF")
    args = parser.parse_args()

    selected = [CASES[0], CASES[3]] if args.quick else CASES
    hdr = (
        f"| {'Sistem':<19} | {'Basis':<7} | {'Nbf':>4} | {'FC':>2} | "
        f"{'E_RHF (Ha)':>14} | {'t_RHF':>6} | {'t_OMP2':>7} | {'t_LT-SOS':>8} | {'t_OMP3':>7} | {'RSS(MB)':>7} |"
    )
    print("\n" + "=" * len(hdr), flush=True)
    print(f" MSHQC v{mshqc.__version__} HEAVY & LARGE BENCHMARK (THREADS={N_THREADS})", flush=True)
    print("=" * len(hdr) + "\n" + hdr + "\n" + "-" * len(hdr), flush=True)

    rows = []
    for label, atoms, q, mult, basis, fc, do_omp3 in selected:
        calc = mshqc.MSHQCCalculator(
            atoms, basis_name=basis, eri_method="df", charge=q, multiplicity=mult,
            units="angstrom", threshold=1e-9, freeze_core=fc,
        )
        nbf = int(calc.basis.n_basis_functions())
        t0 = time.perf_counter()
        scf_r = calc.run_scf()
        t_scf = time.perf_counter() - t0

        t0 = time.perf_counter()
        omp2_r = calc.run_omp2(scf_r, grad_thresh=1e-5)
        t_o2 = time.perf_counter() - t0

        t0 = time.perf_counter()
        lt_r = calc.run_lt_sos_omp2(scf_r, grad_thresh=1e-5)
        t_lt = time.perf_counter() - t0

        t_o3, e_o3 = None, None
        if do_omp3:
            t0 = time.perf_counter()
            omp3_r = calc.run_omp3(scf_r, grad_thresh=1e-5)
            t_o3 = time.perf_counter() - t0
            e_o3 = float(omp3_r.energy_total)

        rss = peak_rss_mb()
        rows.append({
            "label": label, "basis": basis, "nbf": nbf, "n_frozen": calc.n_frozen,
            "e_scf": float(scf_r.energy_total), "e_omp2": float(omp2_r.energy_total),
            "e_lt_sos": float(lt_r.energy_total), "e_omp3": e_o3,
            "t_scf": t_scf, "t_omp2": t_o2, "t_lt_sos": t_lt, "t_omp3": t_o3, "peak_mb": rss,
        })
        o3_s = f"{t_o3:>6.2f}s" if t_o3 is not None else f"{'skip':>7}"
        print(
            f"| {label:<19} | {basis:<7} | {nbf:>4d} | {calc.n_frozen:>2d} | "
            f"{scf_r.energy_total:>14.8f} | {t_scf:>5.2f}s | {t_o2:>6.2f}s | {t_lt:>7.2f}s | {o3_s} | {rss:>7.1f} |",
            flush=True,
        )

    print("-" * len(hdr), flush=True)
    Path(args.out).write_text(json.dumps({"engine": f"mshqc-v{mshqc.__version__}", "cases": rows}, indent=2), encoding="utf-8")
    print(f"[SELESAI] Data heavy/large v2.0 disimpan ke: {args.out}", flush=True)


if __name__ == "__main__":
    main()
