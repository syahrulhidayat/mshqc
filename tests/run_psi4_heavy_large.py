#!/usr/bin/env python3
"""
tests/run_psi4_heavy_large.py
-----------------------------
Benchmark Referensi Psi4 (DF-RHF / DF-OMP2 / DF-OMP3):
- Atom berat hingga Z=36 (Ar, HBr, Kr) all-electron & freeze_core=True
- Molekul menengah-besar (C6H6 cc-pVDZ 114 BF, C10H8 cc-pVDZ 188 BF, C6H6 cc-pVTZ 264 BF)
Menghasilkan: psi4_heavy_large_raw.json
"""

import os
import sys
import json
import time
import resource

CPU_AVAIL = os.cpu_count() or 4
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", str(min(4, CPU_AVAIL))))
os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import psi4


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
    ("C10H8 (188 BF, FC)",  NAPHTHALENE, 0, 1, "cc-pVDZ", True,  True),
    ("C6H6 (264 BF, FC)",   BENZENE, 0, 1, "cc-pVTZ", True,  False),
]


def set_case_options(basis_name: str, aux: str, fc: bool) -> None:
    psi4.core.clean_options()
    psi4.set_options({
        "basis": basis_name,
        "df_basis_scf": aux,
        "df_basis_mp2": aux,
        "df_basis_cc": aux,
        "reference": "rhf",
        "scf_type": "df",
        "mp2_type": "df",
        "mp_type": "df",
        "cc_type": "df",
        "freeze_core": "true" if fc else "false",
        "e_convergence": 1e-9,
        "d_convergence": 1e-9,
        "r_convergence": 1e-6,
        "rms_mograd_convergence": 1e-5,
        "max_mograd_convergence": 1e-4,
    })


def main():
    psi4.set_num_threads(N_THREADS)
    psi4.set_memory("6 GB")
    psi4.core.set_output_file("/dev/null", False)

    hdr = f"| {'Sistem':<19} | {'Nbf':>4} | {'FC':>2} | {'SCF(s)':>7} | {'OMP2(s)':>8} | {'OMP3(s)':>8} | {'PeakMB':>7} |"
    print(hdr)
    print("-" * len(hdr))

    results = []
    for label, atoms, q, mult, basis_name, fc, do_omp3 in CASES:
        aux = f"{basis_name}-RI"
        psi4.core.clean()
        set_case_options(basis_name, aux, fc)

        g_lines = [f"{q} {mult}", "units angstrom", "symmetry c1"] + [
            f"{s} {x} {y} {z}" for s, x, y, z in atoms
        ]
        p4_mol = psi4.geometry("\n".join(g_lines))

        # 1. Standalone RHF SCF timing & metadata
        t0 = time.perf_counter()
        e_scf, p4_scf = psi4.energy("scf", molecule=p4_mol, return_wfn=True)
        t_scf = time.perf_counter() - t0
        e_scf = float(e_scf)
        nbf = int(p4_scf.basisset().nbf())
        nf = int(p4_scf.nfrzc())
        del p4_scf

        # 2. OMP2 (biarkan run_dfocc menjalankan SCF post_scf=True sendiri)
        psi4.core.clean()
        set_case_options(basis_name, aux, fc)
        t0 = time.perf_counter()
        e_omp2 = float(psi4.energy("omp2", molecule=p4_mol))
        t_omp2_total = time.perf_counter() - t0
        t_omp2 = max(0.0, t_omp2_total - t_scf)

        # 3. OMP3 (opsional)
        t_omp3 = None
        t_omp3_total = None
        e_omp3 = None
        if do_omp3:
            psi4.core.clean()
            set_case_options(basis_name, aux, fc)
            t0 = time.perf_counter()
            e_omp3 = float(psi4.energy("omp3", molecule=p4_mol))
            t_omp3_total = time.perf_counter() - t0
            t_omp3 = max(0.0, t_omp3_total - t_scf)

        psi4.core.clean()
        rss = peak_rss_mb()
        omp3_str = f"{t_omp3:>8.3f}" if do_omp3 else "    skip"
        print(
            f"| {label:<19} | {nbf:>4d} | {nf:>2d} | {t_scf:>7.3f} | {t_omp2:>8.3f} | {omp3_str} | {rss:>7.1f} |",
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
            "t_scf": t_scf,
            "t_omp2": t_omp2,
            "t_omp2_total": t_omp2_total,
            "t_omp3": t_omp3,
            "t_omp3_total": t_omp3_total,
            "e_scf": e_scf,
            "e_omp2": e_omp2,
            "e_omp3": e_omp3,
            "peak_mb": rss,
        })

    payload = {
        "engine": "psi4",
        "version": str(psi4.__version__),
        "n_threads": N_THREADS,
        "cases": results,
    }
    with open("psi4_heavy_large_raw.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("\nSaved psi4_heavy_large_raw.json")


if __name__ == "__main__":
    main()