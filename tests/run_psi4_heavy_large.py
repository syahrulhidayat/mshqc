#!/usr/bin/env python3
"""
tests/run_psi4_heavy_large.py
-----------------------------
Benchmark Referensi Psi4 (DF-RHF / DF-OMP2 / DF-OMP3):
- Atom berat hingga Z=36 (Ar, HBr, Kr) all-electron & freeze_core=True
- Molekul menengah-besar (C6H6 cc-pVDZ 114 BF, C10H8 cc-pVDZ 180 BF, C6H6 cc-pVTZ 264 BF)
Menghasilkan: psi4_heavy_large_raw.json
"""

import os
import gc
import sys
import json
import time
import ctypes
import shutil
import tempfile
import resource
from pathlib import Path

CPU_AVAIL = os.cpu_count() or 4
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", str(min(4, CPU_AVAIL))))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

SHM_DIR = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
if "PSI_SCRATCH" not in os.environ:
    os.environ["PSI_SCRATCH"] = SHM_DIR

import psi4


def _load_blas_setters():
    setters = []
    lib_names = ("libmkl_rt.so", "libmkl_rt.so.2", "libopenblas.so", "libopenblas.so.0")
    candidates = []
    for d in (os.path.join(sys.prefix, "lib"), os.environ.get("CONDA_PREFIX", "") and os.path.join(os.environ["CONDA_PREFIX"], "lib")):
        if d and os.path.isdir(d):
            for name in lib_names:
                full_p = os.path.join(d, name)
                if os.path.exists(full_p):
                    candidates.append(full_p)
    candidates.extend(lib_names)
    candidates.append(None)

    seen_ptrs = set()
    for lib_name in candidates:
        try:
            lib = ctypes.CDLL(lib_name)
        except Exception:
            continue
        for fn_name in ("MKL_Set_Num_Threads", "MKL_Domain_Set_Num_Threads", "openblas_set_num_threads"):
            if hasattr(lib, fn_name):
                fn = getattr(lib, fn_name)
                ptr = ctypes.cast(fn, ctypes.c_void_p).value
                if ptr not in seen_ptrs:
                    seen_ptrs.add(ptr)
                    if fn_name == "MKL_Domain_Set_Num_Threads":
                        fn.argtypes = [ctypes.c_int, ctypes.c_int]
                        fn.restype = ctypes.c_int
                        setters.append(lambda n, f=fn: f(int(n), 0))
                    else:
                        fn.argtypes = [ctypes.c_int]
                        fn.restype = None
                        setters.append(fn)
    return setters


_BLAS_SETTERS = _load_blas_setters()


def set_blas_threads(num_t: int) -> None:
    for fn in _BLAS_SETTERS:
        try:
            fn(int(num_t))
        except Exception:
            pass


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


def snapshot_scf_wfn(wfn) -> dict:
    return {
        "e": float(wfn.energy()),
        "Ca": wfn.Ca().clone(),
        "Da": wfn.Da().clone(),
        "Fa": wfn.Fa().clone(),
        "ea": wfn.epsilon_a().clone(),
    }


def restore_scf_wfn(wfn, snap: dict) -> None:
    wfn.set_energy(snap["e"])
    wfn.Ca().copy(snap["Ca"])
    wfn.Da().copy(snap["Da"])
    wfn.Fa().copy(snap["Fa"])
    wfn.epsilon_a().copy(snap["ea"])


def tail_file(path: str, n_lines: int = 35) -> str:
    p = Path(path)
    if not p.exists():
        return "(no psi4 output file)"
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


def run_dfocc_with_fallback(
    method: str,
    label: str,
    p4_scf,
    scf_snap: dict,
    p4_mol,
    scratch_dir: str,
):
    """
    Menjalankan DF-OMP2 / DF-OMP3 dengan beberapa strategi konvergensi dfocc
    dan menangkap potongan log iterasi terakhir jika seluruh strategi gagal.
    """
    if method == "omp2":
        strategies = [
            ("approx_diag+diis", {"hess_type": "approx_diag", "do_diis": True, "mo_step_max": 0.5, "mo_maxiter": 200}),
            ("approx_diag+damped", {"hess_type": "approx_diag", "do_diis": False, "mo_step_max": 0.1, "mo_maxiter": 250}),
        ]
    else:
        strategies = [
            ("approx_diag+nodiis", {"hess_type": "approx_diag", "do_diis": False, "mo_step_max": 0.5, "mo_maxiter": 200}),
            ("approx_diag+damped", {"hess_type": "approx_diag", "do_diis": False, "mo_step_max": 0.1, "mo_maxiter": 250}),
            ("msd+damped", {"opt_method": "MSD", "hess_type": "approx_diag", "do_diis": False, "mo_step_max": 0.1, "mo_maxiter": 250}),
            ("approx_diag_hf+diis", {"opt_method": "ORB_RESP", "hess_type": "approx_diag_hf", "do_diis": True, "mo_step_max": 0.2, "mo_maxiter": 200}),
        ]

    log_path = os.path.join(scratch_dir, f"{method}_debug.out")
    last_err = None
    last_tail = ""

    for strat_name, extra_opts in strategies:
        restore_scf_wfn(p4_scf, scf_snap)
        psi4.core.clean()
        psi4.core.set_output_file(log_path, False)
        try:
            psi4.set_options(extra_opts)
            set_blas_threads(1)
            t0 = time.perf_counter()
            e_val = float(psi4.energy(method, ref_wfn=p4_scf, molecule=p4_mol))
            dt = time.perf_counter() - t0
            psi4.core.flush_outfile()
            psi4.core.set_output_file("/dev/null", False)
            psi4.core.clean()
            return e_val, dt, strat_name, None
        except Exception as exc:
            psi4.core.flush_outfile()
            psi4.core.set_output_file("/dev/null", False)
            last_err = f"[{strat_name}] {exc}"
            last_tail = tail_file(log_path, 35)
            psi4.core.clean()

    print(
        f"\n[PSI4 DIAG] {label} | {method.upper()} GAGAL pada semua strategi.\n"
        f"Error terakhir: {last_err}\n"
        f"--- 35 baris terakhir output Psi4 ---\n{last_tail}\n"
        f"------------------------------------\n",
        flush=True,
    )
    return None, None, None, last_err


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


def main():
    psi4.set_num_threads(N_THREADS)
    psi4.set_memory("6 GB")
    psi4.core.set_output_file("/dev/null", False)

    hdr = f"| {'Sistem':<19} | {'Nbf':>4} | {'FC':>2} | {'SCF(s)':>7} | {'OMP2(s)':>8} | {'OMP3(s)':>8} | {'PeakMB':>7} |"
    print(hdr)
    print("-" * len(hdr))

    base_scratch = os.environ.get("PSI_SCRATCH", SHM_DIR)
    os.makedirs(base_scratch, exist_ok=True)

    results = []
    for label, atoms, q, mult, basis_name, fc, do_omp3 in CASES:
        aux = f"{basis_name}-RI"
        scratch_dir = tempfile.mkdtemp(prefix="psi4_hl_", dir=base_scratch)
        try:
            psi4.core.IOManager.shared_object().set_default_path(scratch_dir)
            psi4.core.clean()
            psi4.core.clean_options()

            g_lines = [
                f"{q} {mult}",
                "units angstrom",
                "symmetry c1",
                "no_reorient",
                "no_com",
            ] + [f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, x, y, z in atoms]
            p4_mol = psi4.geometry("\n".join(g_lines))

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
                "hess_type": "approx_diag",
                "freeze_core": "true" if fc else "false",
                "e_convergence": 1e-9,
                "d_convergence": 1e-9,
                "r_convergence": 1e-6,
                "rms_mograd_convergence": 1e-5,
                "max_mograd_convergence": 1e-4,
                "maxiter": 200,
                "mo_maxiter": 200,
                "cachelevel": 2,
                "read_scf_3index": False,
            })

            # 1. DF-RHF SCF
            set_blas_threads(1)
            t0 = time.perf_counter()
            e_scf, p4_scf = psi4.energy("scf", molecule=p4_mol, return_wfn=True)
            t_scf = time.perf_counter() - t0
            e_scf = float(e_scf)
            nbf = int(p4_scf.basisset().nbf())
            nf = int(p4_scf.nfrzc())
            scf_snap = snapshot_scf_wfn(p4_scf)

            # Wajib agar dfocc tidak mencari Unit 97 (PSIF_DFSCF_BJ) saat memakai ref_wfn
            psi4.set_options({"scf_type": "pk", "read_scf_3index": False})

            # 2. DF-OMP2
            e_omp2, t_omp2, omp2_strat, omp2_err = run_dfocc_with_fallback(
                "omp2", label, p4_scf, scf_snap, p4_mol, scratch_dir
            )

            # 3. DF-OMP3 (opsional)
            t_omp3 = None
            e_omp3 = None
            omp3_strat = None
            omp3_err = None
            if do_omp3:
                e_omp3, t_omp3, omp3_strat, omp3_err = run_dfocc_with_fallback(
                    "omp3", label, p4_scf, scf_snap, p4_mol, scratch_dir
                )

            del scf_snap, p4_scf, p4_mol
            gc.collect()

            rss = peak_rss_mb()
            omp2_str = f"{t_omp2:>8.3f}" if t_omp2 is not None else "    FAIL"
            omp3_str = (
                f"{t_omp3:>8.3f}"
                if (do_omp3 and t_omp3 is not None)
                else ("    FAIL" if do_omp3 else "    skip")
            )
            print(
                f"| {label:<19} | {nbf:>4d} | {nf:>2d} | {t_scf:>7.3f} | {omp2_str} | {omp3_str} | {rss:>7.1f} |",
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
                "t_omp3": t_omp3,
                "e_scf": e_scf,
                "e_omp2": e_omp2,
                "e_omp3": e_omp3,
                "omp2_strategy": omp2_strat,
                "omp3_strategy": omp3_strat,
                "omp2_error": omp2_err,
                "omp3_error": omp3_err,
                "peak_mb": rss,
            })
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    payload = {
        "engine": "psi4",
        "version": str(psi4.__version__),
        "n_threads": N_THREADS,
        "hess_type": "approx_diag",
        "read_scf_3index": False,
        "cases": results,
    }
    with open("psi4_heavy_large_raw.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("\nSaved psi4_heavy_large_raw.json")


if __name__ == "__main__":
    main()