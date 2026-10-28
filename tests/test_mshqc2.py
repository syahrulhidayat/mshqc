#!/usr/bin/env python3
import time
import mshqc

SYSTEMS = [
    ("Neon",    [("Ne", 0.0, 0.0,  0.0000)], 0, 1, "R"),
    ("H2O",     [("O",  0.0, 0.0,  0.1173),
                 ("H",  0.0, 0.7572, -0.4692),
                 ("H",  0.0, -0.7572, -0.4692)], 0, 1, "R"),
    ("Lithium", [("Li", 0.0, 0.0,  0.0000)], 0, 2, "U"),
    ("Carbon",  [("C",  0.0, 0.0,  0.0000)], 0, 3, "U"),
    ("Oxygen",  [("O",  0.0, 0.0,  0.0000)], 0, 3, "U"),
]

ERI_MODES = ["exact", "cholesky", "df"]


def run_benchmark():
    print("==========================================================================================================", flush=True)
    print("  MSHQC BENCHMARK (APPLE-TO-APPLE) | COMPARISON: EXACT vs CHOLESKY vs DF (HF, MP2, MP3, OMP2, OMP3)", flush=True)
    print("  Base: cc-pVTZ | Aux Base: cc-pVTZ-RI | Chol Tol: 1e-09 | E_conv: 1e-09 | Grad_conv: 1e-05", flush=True)
    print("==========================================================================================================", flush=True)

    results = []

    for sys_name, atoms, chg, mult, ref in SYSTEMS:
        mol = mshqc.Molecule(charge=chg, multiplicity=mult)
        for sym, x, y, z in atoms:
            mol.add_atom(sym, x, y, z, units="angstrom")

        basis     = mshqc.BasisSet("cc-pVTZ", mol)
        integrals = mshqc.IntegralEngine(mol, basis)

        scf_cls = mshqc.RHF if ref == "R" else mshqc.UHF
        mp2_cls = mshqc.RMP2 if ref == "R" else mshqc.UMP2
        mp3_cls = mshqc.RMP3 if ref == "R" else mshqc.UMP3

        for mode in ERI_MODES:
            print(f"Running: {sys_name:<9}-> {ref}HF, {ref}MP2, {ref}MP3, OMP2, & OMP3 (Integral: {mode.upper()})", flush=True)

            scf_conf = mshqc.SCFConfig(eri_method=mode, threshold=1e-9, max_iterations=200)
            mp2_conf = mshqc.MP2Config(eri_method=mode, threshold=1e-9, gradient_threshold=1e-5, max_iterations=50)

            # 1. HF (RHF / UHF)
            t0 = time.perf_counter()
            scf_res = scf_cls(mol, basis, integrals, scf_conf).compute()
            results.append((sys_name, mode.upper(), f"{ref}HF", scf_res.energy_total, time.perf_counter() - t0))

            # 2. MP2 (RMP2 / UMP2)
            t0 = time.perf_counter()
            mp2_res = mp2_cls(mol, basis, integrals, scf_res, mp2_conf).compute()
            results.append((sys_name, mode.upper(), f"{ref}MP2", mp2_res.energy_total, time.perf_counter() - t0))

            # 3. MP3 (RMP3 / UMP3)
            t0 = time.perf_counter()
            mp3_res = mp3_cls(scf_res, mp2_res, mp2_conf, integrals).compute()
            results.append((sys_name, mode.upper(), f"{ref}MP3", mp3_res.energy_total, time.perf_counter() - t0))

            # 4. OMP2
            t0 = time.perf_counter()
            omp2_res = mshqc.OMP2(mol, basis, integrals, scf_res, mp2_conf).compute()
            results.append((sys_name, mode.upper(), "OMP2", omp2_res.energy_total, time.perf_counter() - t0))

            # 5. OMP3
            t0 = time.perf_counter()
            omp3_res = mshqc.OMP3(mol, basis, integrals, scf_res, mp2_conf).compute()
            results.append((sys_name, mode.upper(), "OMP3", omp3_res.energy_total, time.perf_counter() - t0))

    print("\n===============================================================================", flush=True)
    print("                     MSHQC BENCHMARK FINAL RESULTS", flush=True)
    print("===============================================================================", flush=True)
    print("-" * 67, flush=True)
    print(f"| {'System':<8} | {'ERI Mode':<10} | {'Method':<8} | {'Energy (Ha)':<16} | {'Time (s)'} |", flush=True)
    print("-" * 67, flush=True)
    for r in results:
        print(f"| {r[0]:<8} | {r[1]:<10} | {r[2]:<8} | {r[3]:<16.8f} | {r[4]:<8.3f} |", flush=True)
    print("-" * 67, flush=True)


if __name__ == "__main__":
    run_benchmark()