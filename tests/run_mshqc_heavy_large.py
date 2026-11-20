#!/usr/bin/env python3
"""
tests/run_mshqc_heavy_large.py
------------------------------
Benchmark & Validasi mshqc untuk:
- Atom berat hingga Z=36 (Ar, HBr, Kr) all-electron & freeze_core=True
- Molekul menengah-besar (C6H6 cc-pVDZ 114 BF, C10H8 cc-pVDZ 180 BF, C6H6 cc-pVTZ 264 BF)
- Smoke-test mshqc.run() one-liner (memastikan PetiteList 4D tidak jalan pada mode DF)
Menghasilkan: mshqc_heavy_large_raw.json & mshqc_heavy_large_summary.txt
"""

import os
import sys
import json
import time
import resource
from datetime import datetime, timezone

CPU_AVAIL = os.cpu_count() or 4
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", str(min(4, CPU_AVAIL))))
os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("TBLIS_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

import mshqc


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
    if sys.platform == "darwin":
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


def make_mol(atoms, charge=0, mult=1) -> mshqc.Molecule:
    mol = mshqc.Molecule(charge, mult)
    for s, x, y, z in atoms:
        mol.add_atom(s, x, y, z, units="angstrom")
    return mol


BENZENE = [
    ("C",  1.3960,  0.0000, 0.0), ("C",  0.6980,  1.2090, 0.0),
    ("C", -0.6980,  1.2090, 0.0), ("C", -1.3960,  0.0000, 0.0),
    ("C", -0.6980, -1.2090, 0.0), ("C",  0.6980, -1.2090, 0.0),
    ("H",  2.4820,  0.0000, 0.0), ("H",  1.2410,  2.1495, 0.0),
    ("H", -1.2410,  2.1495, 0.0), ("H", -2.4820,  0.0000, 0.0),
    ("H", -1.2410, -2.1495, 0.0), ("H",  1.2410, -2.1495, 0.0),
]

NAPHTHALENE = [
    ("C",  0.0000,  0.7100, 0.0), ("C",  0.0000, -0.7100, 0.0),
    ("C",  1.2430,  1.4040, 0.0), ("C",  2.4310,  0.7080, 0.0),
    ("C",  2.4310, -0.7080, 0.0), ("C",  1.2430, -1.4040, 0.0),
    ("C", -1.2430,  1.4040, 0.0), ("C", -2.4310,  0.7080, 0.0),
    ("C", -2.4310, -0.7080, 0.0), ("C", -1.2430, -1.4040, 0.0),
    ("H",  1.2450,  2.4900, 0.0), ("H",  3.3740,  1.2460, 0.0),
    ("H",  3.3740, -1.2460, 0.0), ("H",  1.2450, -2.4900, 0.0),
    ("H", -1.2450,  2.4900, 0.0), ("H", -3.3740,  1.2460, 0.0),
    ("H", -3.3740, -1.2460, 0.0), ("H", -1.2450, -2.4900, 0.0),
]

CASES = [
    # (label, atoms, charge, mult, basis, freeze_core, run_omp3)
    ("Ar (Z=18, FC)",       [("Ar", 0.0, 0.0, 0.0)], 0, 1, "cc-pVTZ", True,  True),
    ("HBr (Z=35, FC)",      [("Br", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.414)], 0, 1, "cc-pVDZ", True,  True),
    ("Kr (Z=36, AE)",       [("Kr", 0.0, 0.0, 0.0)], 0, 1, "cc-pVDZ", False, True),
    ("Kr (Z=36, FC)",       [("Kr", 0.0, 0.0, 0.0)], 0, 1, "cc-pVTZ", True,  True),
    ("C6H6 (114 BF, AE)",   BENZENE, 0, 1, "cc-pVDZ", False, True),
    ("C6H6 (114 BF, FC)",   BENZENE, 0, 1, "cc-pVDZ", True,  True),
    ("C10H8 (180 BF, FC)",  NAPHTHALENE, 0, 1, "cc-pVDZ", True,  True),
    ("C6H6 (264 BF, FC)",   BENZENE, 0, 1, "cc-pVTZ", True,  False),
]


def format_final_report(results: list, t_smoke: float, e_smoke: float, t_suite_total: float) -> str:
    lines = []
    sep = "=" * 124
    lines.append(sep)
    lines.append(f"RINGKASAN AKHIR BENCHMARK HEAVY ATOMS & LARGE MOLECULES (mshqc | {N_THREADS} threads)")
    lines.append(f"Timestamp (UTC)   : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Smoke-test DF     : PASS dalam {t_smoke:.3f} s (E_MP2 = {e_smoke:.8f} Ha)")
    lines.append(f"Total Waktu Suite : {t_suite_total:.3f} s ({t_suite_total / 60.0:.2f} menit)")
    lines.append(sep)

    hdr = (
        f"| {'Sistem':<19} | {'Basis':<7} | {'Nbf':>4} | {'FC':>2} | "
        f"{'E_RHF (Ha)':>15} | {'E_OMP2 (Ha)':>15} | {'E_OMP3 (Ha)':>15} | "
        f"{'t_RHF':>6} | {'t_OMP2':>7} | {'t_OMP3':>7} | {'t_Tot':>7} | {'RSS(MB)':>7} |"
    )
    lines.append(hdr)
    lines.append("|" + "-" * (len(hdr) - 2) + "|")

    sum_rhf = 0.0
    sum_omp2 = 0.0
    sum_omp3 = 0.0
    sum_case = 0.0
    max_rss = 0.0

    for r in results:
        e_omp3_str = f"{r['e_omp3']:>15.8f}" if r["e_omp3"] is not None else f"{'skip':>15}"
        t_omp3_str = f"{r['t_omp3']:>6.2f}s" if r["t_omp3"] is not None else f"{'skip':>7}"

        sum_rhf += r["t_scf"]
        sum_omp2 += r["t_omp2"]
        if r["t_omp3"] is not None:
            sum_omp3 += r["t_omp3"]
        sum_case += r["t_total"]
        max_rss = max(max_rss, r["peak_mb"])

        lines.append(
            f"| {r['label']:<19} | {r['basis']:<7} | {r['nbf']:>4d} | {r['n_frozen']:>2d} | "
            f"{r['e_scf']:>15.8f} | {r['e_omp2']:>15.8f} | {e_omp3_str} | "
            f"{r['t_scf']:>5.2f}s | {r['t_omp2']:>6.2f}s | {t_omp3_str} | "
            f"{r['t_total']:>6.2f}s | {r['peak_mb']:>7.1f} |"
        )

    lines.append("|" + "-" * (len(hdr) - 2) + "|")
    lines.append(
        f"| {'TOTAL / PEAK':<19} | {'-':<7} | {'-':>4} | {'-':>2} | "
        f"{'-':>15} | {'-':>15} | {'-':>15} | "
        f"{sum_rhf:>5.2f}s | {sum_omp2:>6.2f}s | {sum_omp3:>6.2f}s | "
        f"{sum_case:>6.2f}s | {max_rss:>7.1f} |"
    )
    lines.append(sep)
    return "\n".join(lines)


def main():
    t_suite_start = time.perf_counter()

    # 1. Smoke-test mshqc.run() one-liner untuk memastikan PetiteList 4D tidak macet di DF
    t0 = time.perf_counter()
    r_one = mshqc.run(
        BENZENE,
        method=["scf", "mp2"],
        basis="cc-pVDZ",
        eri_method="df",
        units="angstrom",
        freeze_core=True,
        use_symmetry=False,
    )
    t_smoke = time.perf_counter() - t0
    e_smoke = float(r_one["mp2"]["energy"])
    print(f"[PASS] mshqc.run() Benzene DF (FC) selesai dalam {t_smoke:.3f}s | E_MP2={e_smoke:.8f} Ha\n", flush=True)

    hdr = (
        f"| {'Sistem':<19} | {'Nbf':>4} | {'FC':>2} | "
        f"{'RHF(s)':>7} | {'OMP2(s)':>8} | {'OMP3(s)':>8} | {'Total(s)':>8} | {'PeakMB':>7} |"
    )
    print(hdr)
    print("-" * len(hdr))

    results = []
    for label, atoms, q, mult, basis_name, fc, do_omp3 in CASES:
        t_case_start = time.perf_counter()
        aux = f"{basis_name}-RI"

        t_init0 = time.perf_counter()
        mol = make_mol(atoms, q, mult)
        basis = mshqc.BasisSet(basis_name, mol)
        ints = mshqc.IntegralEngine(mol, basis)
        nbf = int(basis.n_basis_functions())
        nf = int(mshqc.standard_frozen_core(mol)) if fc else 0
        t_init = time.perf_counter() - t_init0

        scf_cfg = mshqc.SCFConfig(eri_method="df", threshold=1e-9, aux_basis_name=aux, print_level=0)
        mp2_cfg = mshqc.MP2Config(
            eri_method="df", threshold=1e-9, gradient_threshold=1e-5, aux_basis_name=aux, print_level=0
        )
        mp2_cfg.freeze_core = fc
        mp2_cfg.n_frozen = nf

        t0 = time.perf_counter()
        scf_res = mshqc.RHF(mol, basis, ints, scf_cfg).compute()
        t_rhf = time.perf_counter() - t0

        t0 = time.perf_counter()
        omp2_res = mshqc.OMP2(mol, basis, ints, scf_res, mp2_cfg).compute()
        t_omp2 = time.perf_counter() - t0

        t_omp3 = None
        e_omp3 = None
        if do_omp3:
            t0 = time.perf_counter()
            omp3_res = mshqc.OMP3(mol, basis, ints, scf_res, mp2_cfg).compute()
            t_omp3 = time.perf_counter() - t0
            e_omp3 = float(omp3_res.energy_total)

        t_case_total = time.perf_counter() - t_case_start
        rss = peak_rss_mb()
        omp3_str = f"{t_omp3:>8.3f}" if do_omp3 else "    skip"
        print(
            f"| {label:<19} | {nbf:>4d} | {nf:>2d} | "
            f"{t_rhf:>7.3f} | {t_omp2:>8.3f} | {omp3_str} | {t_case_total:>8.3f} | {rss:>7.1f} |",
            flush=True,
        )

        results.append({
            "label": label,
            "basis": basis_name,
            "aux_basis": aux,
            "freeze_core": fc,
            "run_omp3": do_omp3,
            "nbf": nbf,
            "n_frozen": nf,
            "t_init": t_init,
            "t_scf": t_rhf,
            "t_omp2": t_omp2,
            "t_omp3": t_omp3,
            "t_total": t_case_total,
            "e_scf": float(scf_res.energy_total),
            "e_omp2": float(omp2_res.energy_total),
            "e_omp3": e_omp3,
            "peak_mb": rss,
        })

    t_suite_total = time.perf_counter() - t_suite_start
    final_table = format_final_report(results, t_smoke, e_smoke, t_suite_total)
    print("\n" + final_table + "\n", flush=True)

    payload = {
        "engine": "mshqc",
        "n_threads": N_THREADS,
        "total_wall_time_s": t_suite_total,
        "smoke_test": {"t_s": t_smoke, "e_mp2": e_smoke},
        "cases": results,
    }
    with open("mshqc_heavy_large_raw.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open("mshqc_heavy_large_summary.txt", "w", encoding="utf-8") as f:
        f.write(final_table + "\n")

    print("Saved mshqc_heavy_large_raw.json & mshqc_heavy_large_summary.txt")


if __name__ == "__main__":
    main()