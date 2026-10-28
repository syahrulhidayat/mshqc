#!/usr/bin/env python3
"""
Runner Mandiri Psi4 untuk Benchmark Bab IV (Max Performance & Isolated Heap)
Output: psi4_bab4_raw.json
"""

import os
import re
import sys
import time
import json
import ctypes
import shutil
import tempfile
import argparse
import subprocess

try:
    os.sched_setaffinity(0, set(range(os.cpu_count() or 4)))
except Exception:
    pass

CPU_AVAIL = os.cpu_count() or 4
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", str(min(6, CPU_AVAIL))))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

SHM_DIR = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
os.environ["PSI_SCRATCH"] = SHM_DIR

import numpy as np
import psi4

psi4.set_num_threads(N_THREADS)
psi4.set_memory("8 GB")
psi4.core.be_quiet()


def _load_blas_setters():
    setters = []
    for lib_name in ("libmkl_rt.so", "libmkl_rt.so.2", "libopenblas.so", "libopenblas.so.0", None):
        try:
            lib = ctypes.CDLL(lib_name)
        except Exception:
            continue
        for fn_name in ("MKL_Set_Num_Threads", "openblas_set_num_threads"):
            if hasattr(lib, fn_name):
                fn = getattr(lib, fn_name)
                fn.argtypes = [ctypes.c_int]
                fn.restype = None
                setters.append(fn)
    return setters


_BLAS_SETTERS = _load_blas_setters()


def set_blas_threads(num_t: int):
    for fn in _BLAS_SETTERS:
        try:
            fn(int(num_t))
        except Exception:
            pass


ANG_TO_BOHR = 1.8897261246257702
Z_TO_SYM = {1: "H", 3: "Li", 6: "C", 8: "O", 10: "Ne"}

SYSTEMS = {
    "Neon":    {"atoms": [(10, 0.0, 0.0, 0.0)], "charge": 0, "mult": 1, "is_uhf": False},
    "H2O":     {"atoms": [(8, 0.0, 0.0, 0.117300 * ANG_TO_BOHR), (1, 0.0, 0.757200 * ANG_TO_BOHR, -0.469200 * ANG_TO_BOHR), (1, 0.0, -0.757200 * ANG_TO_BOHR, -0.469200 * ANG_TO_BOHR)], "charge": 0, "mult": 1, "is_uhf": False},
    "Lithium": {"atoms": [(3, 0.0, 0.0, 0.0)], "charge": 0, "mult": 2, "is_uhf": True},
    "Carbon":  {"atoms": [(6, 0.0, 0.0, 0.0)], "charge": 0, "mult": 3, "is_uhf": True},
    "Oxygen":  {"atoms": [(8, 0.0, 0.0, 0.0)], "charge": 0, "mult": 3, "is_uhf": True},
    "OH":      {"atoms": [(8, 0.0, 0.0, 0.0), (1, 0.0, 0.0, 0.9697 * ANG_TO_BOHR)], "charge": 0, "mult": 2, "is_uhf": True},
    "O2":      {"atoms": [(8, 0.0, 0.0, -0.6035 * ANG_TO_BOHR), (8, 0.0, 0.0, 0.6035 * ANG_TO_BOHR)], "charge": 0, "mult": 3, "is_uhf": True},
}

ALL_SYSTEM_ORDER = ["Neon", "H2O", "Lithium", "Carbon", "Oxygen", "OH", "O2"]
ERI_MODE_ORDER = ["EXACT", "CHOLESKY", "DF"]

RAM_CASES = [
    ("Neon",   "cc-pvtz", "cc-pvtz-ri", "EXACT",    "OMP3"),
    ("Neon",   "cc-pvtz", "cc-pvtz-ri", "CHOLESKY", "OMP3"),
    ("Neon",   "cc-pvtz", "cc-pvtz-ri", "DF",       "OMP3"),
    ("Neon",   "cc-pvqz", "cc-pvqz-ri", "DF",       "OMP3"),
    ("H2O",    "cc-pvdz", "cc-pvdz-ri", "EXACT",    "OMP3"),
    ("H2O",    "cc-pvtz", "cc-pvtz-ri", "EXACT",    "OMP3"),
    ("H2O",    "cc-pvtz", "cc-pvtz-ri", "CHOLESKY", "OMP3"),
    ("H2O",    "cc-pvtz", "cc-pvtz-ri", "DF",       "OMP2"),
    ("H2O",    "cc-pvtz", "cc-pvtz-ri", "DF",       "OMP3"),
    ("H2O",    "cc-pvqz", "cc-pvqz-ri", "DF",       "OMP3"),
    ("Oxygen", "cc-pvtz", "cc-pvtz-ri", "EXACT",    "OMP3"),
    ("Oxygen", "cc-pvtz", "cc-pvtz-ri", "CHOLESKY", "OMP3"),
    ("Oxygen", "cc-pvtz", "cc-pvtz-ri", "DF",       "OMP3"),
    ("O2",     "cc-pvtz", "cc-pvtz-ri", "DF",       "OMP3"),
]


def extract_json_safe(output_str):
    for line in reversed(output_str.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except Exception:
                pass
    return {}


def snapshot_wfn(wfn):
    snap = {"e": float(wfn.energy()), "Ca": wfn.Ca().clone(), "Da": wfn.Da().clone(), "Fa": wfn.Fa().clone(), "ea": wfn.epsilon_a().clone()}
    if wfn.Cb() is not None:
        snap.update({"Cb": wfn.Cb().clone(), "Db": wfn.Db().clone(), "Fb": wfn.Fb().clone(), "eb": wfn.epsilon_b().clone()})
    return snap


def restore_wfn(wfn, snap):
    wfn.set_energy(snap["e"])
    for k, g in (("Ca", wfn.Ca), ("Da", wfn.Da), ("Fa", wfn.Fa), ("ea", wfn.epsilon_a)):
        g().copy(snap[k])
    if "Cb" in snap and wfn.Cb() is not None:
        for k, g in (("Cb", wfn.Cb), ("Db", wfn.Db), ("Fb", wfn.Fb), ("eb", wfn.epsilon_b)):
            g().copy(snap[k])


def calc_s2_wfn(wfn, is_closed: bool):
    if is_closed:
        return 0.0
    na, nb = wfn.nalpha(), wfn.nbeta()
    if na == nb:
        return 0.0
    sz = 0.5 * (na - nb)
    s_ideal = sz * (sz + 1.0)
    if na == 0 or nb == 0 or wfn.Cb() is None:
        return float(s_ideal)
    try:
        Ca = np.asarray(wfn.Ca_subset("AO", "OCC"))
        Cb = np.asarray(wfn.Cb_subset("AO", "OCC"))
        S  = np.asarray(wfn.mintshelper().ao_overlap())
        return float(s_ideal + nb - np.sum((Ca.T @ S @ Cb) ** 2))
    except Exception:
        return float(s_ideal)


def parse_occ_iterations(log_text):
    iters = 0
    for line in log_text.splitlines():
        m = re.match(r"^\s*(\d+)\s+-?\d+\.\d+\s+-?\d+\.\d+", line)
        if m:
            iters = max(iters, int(m.group(1)))
    return iters


PSI4_RAM_WORKER = r"""
import sys, gc, json, os, ctypes, tempfile, shutil
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
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

base_shm = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
scratch_sub = tempfile.mkdtemp(prefix="psi4_ram_", dir=base_shm)
os.environ["PSI_SCRATCH"] = scratch_sub

import psi4

Z_TO_SYM = {1: "H", 3: "Li", 6: "C", 8: "O", 10: "Ne"}

def _load_blas_setters():
    setters = []
    for lib_name in ("libmkl_rt.so", "libmkl_rt.so.2", "libopenblas.so", "libopenblas.so.0", None):
        try:
            lib = ctypes.CDLL(lib_name)
        except Exception:
            continue
        for fn_name in ("MKL_Set_Num_Threads", "openblas_set_num_threads"):
            if hasattr(lib, fn_name):
                fn = getattr(lib, fn_name)
                fn.argtypes = [ctypes.c_int]
                fn.restype = None
                setters.append(fn)
    return setters

_BLAS_SETTERS = _load_blas_setters()

def set_blas_threads(num_t):
    for fn in _BLAS_SETTERS:
        try:
            fn(int(num_t))
        except Exception:
            pass

def get_rss_and_hwm_mb():
    rss, hwm = 0.0, 0.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"): rss = int(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"): hwm = int(line.split()[1]) / 1024.0
    return rss, hwm

def get_scratch_mb(folder):
    total_b = 0
    if os.path.isdir(folder):
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath):
                try:
                    total_b += os.path.getsize(fpath)
                except Exception:
                    pass
    return total_b / (1024.0 * 1024.0)

try:
    psi4.set_num_threads(n_threads)
    psi4.set_memory("8 GB")
    psi4.core.be_quiet()

    mol_w = psi4.geometry("0 1\nH 0.0 0.0 0.0\nH 0.0 0.0 1.4\nunits bohr\nsymmetry c1")
    bas_w = psi4.core.BasisSet.build(mol_w, "ORBITAL", "cc-pVDZ")
    mints_w = psi4.core.MintsHelper(bas_w); mints_w.ao_overlap()
    del mints_w, bas_w, mol_w
    psi4.core.clean(); gc.collect()

    base_rss, base_hwm = get_rss_and_hwm_mb()
    payload = json.loads(sys.argv[1])
    mode, sys_name = payload["mode"].lower(), payload["sys"]

    lines = [f"{payload['charge']} {payload['mult']}", "units bohr"]
    if mode in ["df", "cholesky"] or sys_name == "OH":
        lines.append("symmetry c1")
    lines += [f"{Z_TO_SYM[z]} {x:.12f} {y:.12f} {zc:.12f}" for z, x, y, zc in payload["atoms"]]
    mol = psi4.geometry("\n".join(lines))

    scf_t = "pk" if mode == "exact" else ("cd" if mode == "cholesky" else "df")
    corr_t = "conv" if mode == "exact" else ("cd" if mode == "cholesky" else "df")

    opts = {
        "basis": payload["basis"],
        "reference": "uhf" if payload["is_uhf"] else "rhf",
        "e_convergence": 1e-9,
        "d_convergence": 1e-9,
        "scf_type": scf_t,
        "mp2_type": corr_t,
        "mp_type": corr_t,
        "cc_type": corr_t,
        "freeze_core": "false",
        "r_convergence": 1e-6,
        "rms_mograd_convergence": 1e-5,
        "max_mograd_convergence": 1e-4,
        "cachelevel": 2,
        "read_scf_3index": False,
    }
    if mode == "df":
        opts.update({"df_basis_scf": payload["aux"], "df_basis_mp2": payload["aux"], "df_basis_cc": payload["aux"]})
    elif mode == "cholesky":
        opts["cholesky_tolerance"] = 1e-9

    psi4.set_options(opts)
    set_blas_threads(1)
    _, scf_wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
    if mode in ["cholesky", "df"]:
        psi4.set_options({"scf_type": "pk", "read_scf_3index": False})

    set_blas_threads(n_threads)
    psi4.energy(payload["method"].lower(), ref_wfn=scf_wfn, molecule=mol)
    set_blas_threads(1)

    curr_rss, peak_hwm = get_rss_and_hwm_mb()
    scratch_ram = get_scratch_mb(scratch_sub)
    psi4.core.clean()

    total_peak = peak_hwm + scratch_ram
    net_working = max(peak_hwm - base_hwm, curr_rss - base_rss, 0.10) + scratch_ram
    print(json.dumps({"base_hwm": base_hwm, "peak_hwm": total_peak, "net_ram": net_working}))
finally:
    shutil.rmtree(scratch_sub, ignore_errors=True)
"""


def compute_psi4_suite():
    results = {}
    tmp_log = os.path.join(SHM_DIR, f"psi4_bab4_{os.getpid()}.out")

    print("==========================================================================================================", flush=True)
    print(f"  PSI4 BAB IV BENCHMARK RUNNER (THREADS: {N_THREADS}) | cc-pVTZ / cc-pVTZ-RI", flush=True)
    print("==========================================================================================================", flush=True)

    for sys_name in ALL_SYSTEM_ORDER:
        info = SYSTEMS[sys_name]
        chg, mult, is_uhf = info["charge"], info["mult"], info["is_uhf"]
        is_closed = not is_uhf
        ref_lbl = "U" if is_uhf else "R"

        for mode_upper in ERI_MODE_ORDER:
            eri_mode = mode_upper.lower()
            print(f"  [PSI4]  Running: {sys_name:<8} | Mode: {mode_upper:<9} -> {ref_lbl}HF, {ref_lbl}MP2, {ref_lbl}MP3, OMP2, OMP3", flush=True)

            psi4.core.clean()
            psi4.core.clean_options()
            psi4.core.be_quiet()

            lines = [f"{chg} {mult}", "units bohr"]
            if eri_mode in ["df", "cholesky"] or sys_name == "OH":
                lines.append("symmetry c1")
            lines += [f"{Z_TO_SYM[z]} {x:.12f} {y:.12f} {zc:.12f}" for z, x, y, zc in info["atoms"]]
            mol = psi4.geometry("\n".join(lines))

            scf_type = "pk" if eri_mode == "exact" else ("cd" if eri_mode == "cholesky" else "df")
            corr_type = "conv" if eri_mode == "exact" else ("cd" if eri_mode == "cholesky" else "df")

            opts = {
                "basis": "cc-pVTZ",
                "reference": "uhf" if is_uhf else "rhf",
                "e_convergence": 1e-9,
                "d_convergence": 1e-9,
                "scf_type": scf_type,
                "mp2_type": corr_type,
                "mp_type": corr_type,
                "cc_type": corr_type,
                "freeze_core": "false",
                "r_convergence": 1e-6,
                "rms_mograd_convergence": 1e-5,
                "max_mograd_convergence": 1e-4,
                "maxiter": 200,
                "mo_maxiter": 50,
                "cachelevel": 2,
                "read_scf_3index": False,
            }
            if eri_mode == "df":
                opts.update({"df_basis_scf": "cc-pVTZ-RI", "df_basis_mp2": "cc-pVTZ-RI", "df_basis_cc": "cc-pVTZ-RI"})
            elif eri_mode == "cholesky":
                opts["cholesky_tolerance"] = 1e-9

            psi4.set_options(opts)
            key = f"{sys_name}|{mode_upper}"
            entry = {}

            set_blas_threads(1)
            try:
                t0 = time.perf_counter()
                e_scf, scf_wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
                t_scf = time.perf_counter() - t0
                entry["SCF"] = {"e": float(e_scf), "t": float(t_scf), "s2": calc_s2_wfn(scf_wfn, is_closed)}
            except Exception as ex:
                results[key] = {"error": str(ex)}
                psi4.core.clean()
                continue

            hf_snap = snapshot_wfn(scf_wfn)
            if eri_mode in ["cholesky", "df"]:
                psi4.set_options({"scf_type": "pk", "read_scf_3index": False})

            set_blas_threads(N_THREADS)
            need_iter_log = is_uhf and eri_mode in ["exact", "df"]
            try:
                for m_key, m_call in (("MP2", "mp2"), ("MP3", "mp3")):
                    try:
                        t0 = time.perf_counter()
                        e_val = psi4.energy(m_call, ref_wfn=scf_wfn, molecule=mol)
                        t_val = time.perf_counter() - t0
                        entry[m_key] = {"e": float(e_val), "t": float(t_val)}
                    except Exception:
                        pass
                    psi4.core.clean()
                    restore_wfn(scf_wfn, hf_snap)

                for m_key, m_call in (("OMP2", "omp2"), ("OMP3", "omp3")):
                    try:
                        if need_iter_log:
                            psi4.core.set_output_file(tmp_log, False)
                        t0 = time.perf_counter()
                        e_val, opt_wfn = psi4.energy(m_call, ref_wfn=scf_wfn, molecule=mol, return_wfn=True)
                        t_val = time.perf_counter() - t0
                        it_cnt = 0
                        if need_iter_log:
                            psi4.core.flush_outfile()
                            psi4.core.be_quiet()
                            log_txt = open(tmp_log, "r", errors="replace").read() if os.path.exists(tmp_log) else ""
                            it_cnt = parse_occ_iterations(log_txt)
                        s2_val = calc_s2_wfn(opt_wfn, is_closed)
                        entry[m_key] = {"e": float(e_val), "t": float(t_val), "iter": it_cnt, "s2": s2_val}
                    except Exception:
                        psi4.core.be_quiet()
                    psi4.core.clean()
                    if m_key == "OMP2":
                        restore_wfn(scf_wfn, hf_snap)
            finally:
                set_blas_threads(1)

            results[key] = entry

    if os.path.exists(tmp_log):
        try:
            os.remove(tmp_log)
        except Exception:
            pass

    return results


def compute_ram_benchmark():
    ram_results = {}
    print("  [PSI4]  Mengukur Peak RSS (VmHWM) + Scratch /dev/shm Terkalibrasi (Tabel 6)...", flush=True)
    for sys_name, b_name, aux_name, mode, method in RAM_CASES:
        key = f"{sys_name}|{b_name}|{mode}|{method}"
        payload = json.dumps({"sys": sys_name, "basis": b_name, "aux": aux_name, "mode": mode, "method": method, **SYSTEMS[sys_name]})
        try:
            res_p = extract_json_safe(subprocess.check_output([sys.executable, "-c", PSI4_RAM_WORKER, payload], text=True))
        except Exception:
            res_p = {"peak_hwm": 0.0, "net_ram": 0.0}
        ram_results[key] = res_p
    return ram_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Psi4 Bab IV Standalone Benchmark Runner")
    parser.add_argument("--out", default="psi4_bab4_raw.json", help="Path output berkas JSON Psi4")
    args = parser.parse_args()

    t_start = time.perf_counter()
    ref_suite = compute_psi4_suite()
    ram_bench = compute_ram_benchmark()
    wall_time = time.perf_counter() - t_start

    output_payload = {
        "engine": "psi4",
        "threads": N_THREADS,
        "wall_time_s": wall_time,
        "ref_suite": ref_suite,
        "ram_bench": ram_bench,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[SELESAI] Data benchmark Psi4 Bab IV disimpan ke: {args.out} ({wall_time:.2f} s)", flush=True)