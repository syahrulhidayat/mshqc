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

cpu_avail = os.cpu_count() or 4
n_threads = int(os.environ.get("OMP_NUM_THREADS", str(min(6, cpu_avail))))
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = str(n_threads)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["TBLIS_NUM_THREADS"] = str(n_threads)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import mshqc

ANG_TO_BOHR = 1.8897261246257702

SYSTEMS = [
    ("Neon",    [(10, 0.0, 0.0, 0.0)], 0, 1, "R"),
    ("H2O",     [(8,  0.0, 0.0,  0.117300 * ANG_TO_BOHR),
                 (1,  0.0,  0.757200 * ANG_TO_BOHR, -0.469200 * ANG_TO_BOHR),
                 (1,  0.0, -0.757200 * ANG_TO_BOHR, -0.469200 * ANG_TO_BOHR)], 0, 1, "R"),
    ("Lithium", [(3,  0.0, 0.0, 0.0)], 0, 2, "U"),
    ("Carbon",  [(6,  0.0, 0.0, 0.0)], 0, 3, "U"),
    ("Oxygen",  [(8,  0.0, 0.0, 0.0)], 0, 3, "U"),
]

try:
    libc = ctypes.CDLL(None)
except Exception:
    libc = None


def flush_all():
    sys.stdout.flush()
    if libc is not None:
        try:
            libc.fflush(None)
        except Exception:
            pass


@contextmanager
def suppress_stdout():
    """Membungkam std::cout C++ saat timing agar setara dengan psi4.core.be_quiet()."""
    flush_all()
    old_stdout_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        yield
    finally:
        flush_all()
        os.dup2(old_stdout_fd, 1)
        os.close(devnull_fd)
        os.close(old_stdout_fd)


def make_scf_config(mode):
    cfg = mshqc.SCFConfig()
    cfg.scf_type = "incore"
    cfg.max_iterations = 200
    cfg.energy_threshold = 1e-9
    cfg.density_threshold = 1e-9
    cfg.df_threshold = 1e-9
    cfg.print_level = 0
    if mode == "df":
        cfg.eri_method = "df"
        cfg.use_df = True
        cfg.aux_basis_name = "cc-pvtz-ri"
    elif mode == "cholesky":
        cfg.eri_method = "cholesky"
        cfg.cholesky_threshold = 1e-9
    else:
        cfg.eri_method = "exact"
    return cfg


def make_mp2_config(mode):
    cfg = mshqc.MP2Config()
    cfg.energy_threshold = 1e-9
    cfg.gradient_threshold = 1e-5
    cfg.max_iterations = 50
    cfg.print_level = 0
    if mode == "df":
        cfg.eri_method = "df"
        cfg.use_df = True
        cfg.aux_basis_name = "cc-pvtz-ri"
    elif mode == "cholesky":
        cfg.eri_method = "cholesky"
        cfg.cholesky_threshold = 1e-9
    else:
        cfg.eri_method = "exact"
    return cfg


def run_benchmark():
    print("==========================================================================================================", flush=True)
    print("  MSHQC BENCHMARK (APPLE-TO-APPLE) | COMPARISON: EXACT vs CHOLESKY vs DF (HF, MP2, MP3, OMP2, OMP3)", flush=True)
    print("  Base: cc-pVTZ | Aux Base: cc-pVTZ-RI | Chol Tol: 1e-09 | E_conv: 1e-09 | Grad_conv: 1e-05", flush=True)
    print("==========================================================================================================", flush=True)

    modes = ["exact", "cholesky", "df"]
    results = []

    for sys_name, atoms, chg, mult, ref_type in SYSTEMS:
        for mode in modes:
            print(f"Running: {sys_name:<9}-> {ref_type}HF, {ref_type}MP2, {ref_type}MP3, OMP2, & OMP3 (Integral: {mode.upper()})", flush=True)

            mol = mshqc.Molecule(chg, mult)
            for z, x, y, zc in atoms:
                mol.add_atom(z, x, y, zc)

            with suppress_stdout():
                basis = mshqc.BasisSet("cc-pVTZ", mol)
                integrals = mshqc.IntegralEngine(mol, basis)
                scf_conf = make_scf_config(mode)
                mp2_conf = make_mp2_config(mode)

                # 1. HF
                t0 = time.perf_counter()
                if ref_type == "R":
                    scf_res = mshqc.RHF(mol, basis, integrals, scf_conf).compute()
                else:
                    n_elec = mol.n_electrons()
                    n_unp = mol.multiplicity() - 1
                    na = (n_elec + n_unp) // 2
                    nb = (n_elec - n_unp) // 2
                    scf_res = mshqc.UHF(mol, basis, integrals, na, nb, scf_conf).compute()
                t_hf = time.perf_counter() - t0
                results.append((sys_name, mode.upper(), f"{ref_type}HF", scf_res.energy_total, t_hf))

                # 2. MP2
                t0 = time.perf_counter()
                if ref_type == "R":
                    mp2_res = mshqc.RMP2(mol, basis, integrals, scf_res, mp2_conf).compute()
                else:
                    mp2_res = mshqc.UMP2(mol, basis, integrals, scf_res, mp2_conf).compute()
                t_mp2 = time.perf_counter() - t0
                results.append((sys_name, mode.upper(), f"{ref_type}MP2", mp2_res.energy_total, t_mp2))

                # 3. MP3
                t0 = time.perf_counter()
                if ref_type == "R":
                    mp3_res = mshqc.RMP3(scf_res, mp2_res, mp2_conf, integrals).compute()
                else:
                    mp3_res = mshqc.UMP3(scf_res, mp2_res, mp2_conf, integrals).compute()
                t_mp3 = time.perf_counter() - t0
                e_mp3 = mp3_res.e_total if hasattr(mp3_res, "e_total") else mp3_res.energy_total
                results.append((sys_name, mode.upper(), f"{ref_type}MP3", e_mp3, t_mp3))

                # 4. OMP2
                t0 = time.perf_counter()
                omp2_res = mshqc.OMP2(mol, basis, integrals, scf_res, mp2_conf).compute()
                t_omp2 = time.perf_counter() - t0
                results.append((sys_name, mode.upper(), "OMP2", omp2_res.energy_total, t_omp2))

                # 5. OMP3
                t0 = time.perf_counter()
                omp3_res = mshqc.OMP3(mol, basis, integrals, scf_res, mp2_conf).compute()
                t_omp3 = time.perf_counter() - t0
                e_omp3 = omp3_res.e_total if hasattr(omp3_res, "e_total") else omp3_res.energy_total
                results.append((sys_name, mode.upper(), "OMP3", e_omp3, t_omp3))

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