#!/usr/bin/env python3
"""
tests/run_mshqc_heavy_large.py
------------------------------
Benchmark & Validasi mshqc untuk:
- Atom berat hingga Z=36 (Ar, HBr, Kr) all-electron & freeze_core=True
- Molekul menengah-besar (C6H6 cc-pVDZ 114 BF, C10H8 cc-pVDZ 188 BF, C6H6 cc-pVTZ 264 BF)
- Smoke-test mshqc.run() one-liner (memastikan PetiteList 4D tidak jalan pada mode DF)
Menghasilkan: mshqc_heavy_large_raw.json
"""

import os
import sys
import json
import time
import resource

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
    ("C10H8 (188 BF, FC)",  NAPHTHALENE, 0, 1, "cc-pVDZ", True,  True),
    ("C6H6 (264 BF, FC)",   BENZENE, 0, 1, "cc-pVTZ", True,  False),
]


def main():
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
    print(f"[PASS] mshqc.run() Benzene DF (FC) selesai dalam {t_smoke:.3f}s | E_MP2={e_smoke:.8f} Ha\n")

    hdr = f"| {'Sistem':<19} | {'Nbf':>4} | {'FC':>2} | {'RHF(s)':>7} | {'OMP2(s)':>8} | {'OMP3(s)':>8} | {'PeakMB':>7} |"
    print(hdr)
    print("-" * len(hdr))

    results = []
    for label, atoms, q, mult, basis_name, fc, do_omp3 in CASES:
        aux = f"{basis_name}-RI"
        mol = make_mol(atoms, q, mult)
        basis = mshqc.BasisSet(basis_name, mol)
        ints = mshqc.IntegralEngine(mol, basis)
        nbf = int(basis.n_basis_functions())
        nf = int(mshqc.standard_frozen_core(mol)) if fc else 0

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

        rss = peak_rss_mb()
        omp3_str = f"{t_omp3:>8.3f}" if do_omp3 else "    skip"
        print(
            f"| {label:<19} | {nbf:>4d} | {nf:>2d} | {t_rhf:>7.3f} | {t_omp2:>8.3f} | {omp3_str} | {rss:>7.1f} |",
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
            "t_scf": t_rhf,
            "t_omp2": t_omp2,
            "t_omp3": t_omp3,
            "e_scf": float(scf_res.energy_total),
            "e_omp2": float(omp2_res.energy_total),
            "e_omp3": e_omp3,
            "peak_mb": rss,
        })

    payload = {
        "engine": "mshqc",
        "n_threads": N_THREADS,
        "smoke_test": {"t_s": t_smoke, "e_mp2": e_smoke},
        "cases": results,
    }
    with open("mshqc_heavy_large_raw.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("\nSaved mshqc_heavy_large_raw.json")


if __name__ == "__main__":
    main()