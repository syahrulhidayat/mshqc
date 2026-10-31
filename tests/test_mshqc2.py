#!/usr/bin/env python3
import os
import sys
import time
import ctypes
from contextlib import contextmanager

try:
    os.sched_setaffinity(0, set(range(os.cpu_count() or 4)))
except Exception:
    pass

CPU_AVAIL = os.cpu_count() or 4
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", str(min(6, CPU_AVAIL))))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["TBLIS_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import mshqc

try:
    _LIBC = ctypes.CDLL(None)
except Exception:
    _LIBC = None

BASIS_NAME = "cc-pVTZ"
AUX_BASIS  = "cc-pVTZ-RI"
THRESHOLD  = 1e-9
CHOLESKY_THRESH = 1e-9
GRAD_THRESH = 1e-5

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


@contextmanager
def quiet_c_stdout():
    """Meredam output stdout C++ selama pengukuran waktu agar setara dengan psi4.core.be_quiet()."""
    sys.stdout.flush()
    if _LIBC is not None:
        _LIBC.fflush(None)
    old_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        os.close(devnull_fd)
        yield
    finally:
        sys.stdout.flush()
        if _LIBC is not None:
            _LIBC.fflush(None)
        os.dup2(old_fd, 1)
        os.close(old_fd)


def run_benchmark():
    print("==========================================================================================================", flush=True)
    print(f"  MSHQC BENCHMARK (APPLE-TO-APPLE C1) | THREADS: {N_THREADS} | EXACT vs CHOLESKY vs DF (HF, MP2, MP3, OMP2, OMP3)", flush=True)
    print(f"  Base: {BASIS_NAME} | Aux Base: {AUX_BASIS} | Chol Tol: {CHOLESKY_THRESH} | E_conv: {THRESHOLD} | Grad_conv: {GRAD_THRESH}", flush=True)
    print("==========================================================================================================", flush=True)

    results = []

    for sys_name, atoms, chg, mult, ref in SYSTEMS:
        mol = mshqc.Molecule(charge=chg, multiplicity=mult)
        for sym, x, y, z in atoms:
            mol.add_atom(sym, x, y, z, units="angstrom")

        basis     = mshqc.BasisSet(BASIS_NAME, mol)
        integrals = mshqc.IntegralEngine(mol, basis)

        scf_cls = mshqc.RHF if ref == "R" else mshqc.UHF
        mp2_cls = mshqc.RMP2 if ref == "R" else mshqc.UMP2
        mp3_cls = mshqc.RMP3 if ref == "R" else mshqc.UMP3

        for mode in ERI_MODES:
            print(f"Running: {sys_name:<9}-> {ref}HF, {ref}MP2, {ref}MP3, OMP2, & OMP3 (Integral: {mode.upper()})", flush=True)

            scf_conf = mshqc.SCFConfig(
                eri_method=mode,
                threshold=THRESHOLD,
                aux_basis_name=AUX_BASIS,
                max_iterations=200,
                print_level=0,
            )
            scf_conf.cholesky_threshold = CHOLESKY_THRESH
            scf_conf.df_threshold = THRESHOLD

            mp2_conf = mshqc.MP2Config(
                eri_method=mode,
                threshold=THRESHOLD,
                gradient_threshold=GRAD_THRESH,
                aux_basis_name=AUX_BASIS,
                max_iterations=50,
                print_level=0,
            )
            if hasattr(mp2_conf, "cholesky_threshold"):
                mp2_conf.cholesky_threshold = CHOLESKY_THRESH
            if hasattr(mp2_conf, "df_threshold"):
                mp2_conf.df_threshold = THRESHOLD

            # 1. HF (RHF / UHF)
            with quiet_c_stdout():
                t0 = time.perf_counter()
                scf_res = scf_cls(mol, basis, integrals, scf_conf).compute()
                t_scf = time.perf_counter() - t0
            results.append((sys_name, mode.upper(), f"{ref}HF", scf_res.energy_total, t_scf))

            # 2. MP2 (RMP2 / UMP2)
            with quiet_c_stdout():
                t0 = time.perf_counter()
                mp2_res = mp2_cls(mol, basis, integrals, scf_res, mp2_conf).compute()
                t_mp2 = time.perf_counter() - t0
            results.append((sys_name, mode.upper(), f"{ref}MP2", mp2_res.energy_total, t_mp2))

            # 3. MP3 (RMP3 / UMP3)
            with quiet_c_stdout():
                t0 = time.perf_counter()
                mp3_res = mp3_cls(scf_res, mp2_res, mp2_conf, integrals).compute()
                t_mp3 = time.perf_counter() - t0
            results.append((sys_name, mode.upper(), f"{ref}MP3", mp3_res.energy_total, t_mp3))

            # 4. OMP2
            with quiet_c_stdout():
                t0 = time.perf_counter()
                omp2_res = mshqc.OMP2(mol, basis, integrals, scf_res, mp2_conf).compute()
                t_omp2 = time.perf_counter() - t0
            results.append((sys_name, mode.upper(), "OMP2", omp2_res.energy_total, t_omp2))

            # 5. OMP3
            with quiet_c_stdout():
                t0 = time.perf_counter()
                omp3_res = mshqc.OMP3(mol, basis, integrals, scf_res, mp2_conf).compute()
                t_omp3 = time.perf_counter() - t0
            results.append((sys_name, mode.upper(), "OMP3", omp3_res.energy_total, t_omp3))

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