#!/usr/bin/env python3
"""
Psi4 Reference Benchmark Runner untuk Bab IV v2 (Head-to-Head vs mshqc v2.0.0).
Skema JSON identik dengan benchmarks/run_bab4_v2.py (ref_suite & ram_bench).

Mencakup:
  1. 150 konfigurasi inti Bab IV (7 sistem x 3 mode ERI x 11 metode:
     RHF/UHF, ROHF, ROMP2, ROMP3 semikanonik, RMP2/UMP2, RMP3/UMP3, OMP2, OMP3)
     + metode ekstensi native Psi4: OMP2.5.
  2. Pengukuran Peak RSS (VmHWM) + scratch /dev/shm & Net Working RAM terkalibrasi
     pada 14 kasus RAM_CASES di subproses worker terisolasi.

Usage:
  python benchmarks/run_psi4_bab4_v2.py
  python benchmarks/run_psi4_bab4_v2.py --quick --skip-ram
  python benchmarks/run_psi4_bab4_v2.py --exact-symmetry c1
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    os.sched_setaffinity(0, set(range(os.cpu_count() or 4)))
except Exception:
    pass

CPU_AVAIL = os.cpu_count() or 4
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", str(min(4, CPU_AVAIL))))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

SHM_DIR = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
os.environ.setdefault("PSI_SCRATCH", SHM_DIR)

import numpy as np
import psi4

psi4.set_num_threads(N_THREADS)
psi4.set_memory("1800 MB" if os.name == "nt" else "4 GB")
psi4.core.set_output_file("NUL" if os.name == "nt" else "/dev/null", False)

ANG_TO_BOHR = 1.8897261246257702

SYSTEMS: Dict[str, Dict[str, Any]] = {
    "Neon": {"atoms": [("Ne", 0.0, 0.0, 0.0)], "charge": 0, "mult": 1, "is_uhf": False},
    "H2O": {"atoms": [("O", 0.0, 0.0, 0.1173), ("H", 0.0, 0.7572, -0.4692), ("H", 0.0, -0.7572, -0.4692)], "charge": 0, "mult": 1, "is_uhf": False},
    "Lithium": {"atoms": [("Li", 0.0, 0.0, 0.0)], "charge": 0, "mult": 2, "is_uhf": True},
    "Carbon": {"atoms": [("C", 0.0, 0.0, 0.0)], "charge": 0, "mult": 3, "is_uhf": True},
    "Oxygen": {"atoms": [("O", 0.0, 0.0, 0.0)], "charge": 0, "mult": 3, "is_uhf": True},
    "OH": {"atoms": [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.9697)], "charge": 0, "mult": 2, "is_uhf": True},
    "O2": {"atoms": [("O", 0.0, 0.0, -0.6035), ("O", 0.0, 0.0, 0.6035)], "charge": 0, "mult": 3, "is_uhf": True},
}

ALL_SYSTEM_ORDER = ["Neon", "H2O", "Lithium", "Carbon", "Oxygen", "OH", "O2"]
CORE5_ORDER = ["Neon", "H2O", "Lithium", "Carbon", "Oxygen"]
ERI_MODE_ORDER = ["EXACT", "CHOLESKY", "DF"]

RAM_CASES = [
    ("Neon", "cc-pvtz", "cc-pvtz-ri", "EXACT", "OMP3"),
    ("Neon", "cc-pvtz", "cc-pvtz-ri", "CHOLESKY", "OMP3"),
    ("Neon", "cc-pvtz", "cc-pvtz-ri", "DF", "OMP3"),
    ("Neon", "cc-pvqz", "cc-pvqz-ri", "DF", "OMP3"),
    ("H2O", "cc-pvdz", "cc-pvdz-ri", "EXACT", "OMP3"),
    ("H2O", "cc-pvtz", "cc-pvtz-ri", "EXACT", "OMP3"),
    ("H2O", "cc-pvtz", "cc-pvtz-ri", "CHOLESKY", "OMP3"),
    ("H2O", "cc-pvtz", "cc-pvtz-ri", "DF", "OMP2"),
    ("H2O", "cc-pvtz", "cc-pvtz-ri", "DF", "OMP3"),
    ("H2O", "cc-pvqz", "cc-pvqz-ri", "DF", "OMP3"),
    ("Oxygen", "cc-pvtz", "cc-pvtz-ri", "EXACT", "OMP3"),
    ("Oxygen", "cc-pvtz", "cc-pvtz-ri", "CHOLESKY", "OMP3"),
    ("Oxygen", "cc-pvtz", "cc-pvtz-ri", "DF", "OMP3"),
    ("O2", "cc-pvtz", "cc-pvtz-ri", "DF", "OMP3"),
]


class TeeLogger:
    def __init__(self, log_path: str):
        self.terminal = sys.__stdout__
        self.log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, msg: str) -> None:
        self.terminal.write(msg)
        self.terminal.flush()
        self.log_file.write(msg)
        self.log_file.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()


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
        for fn_name in ("MKL_Set_Num_Threads", "openblas_set_num_threads"):
            if hasattr(lib, fn_name):
                fn = getattr(lib, fn_name)
                ptr = ctypes.cast(fn, ctypes.c_void_p).value
                if ptr not in seen_ptrs:
                    seen_ptrs.add(ptr)
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


def get_rss_and_hwm_mb() -> Tuple[float, float]:
    if sys.platform.startswith("linux"):
        rss, hwm = 0.0, 0.0
        try:
            with open("/proc/self/status", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = float(line.split()[1]) / 1024.0
                    elif line.startswith("VmHWM:"):
                        hwm = float(line.split()[1]) / 1024.0
            return rss, hwm
        except OSError:
            return 0.0, 0.0
    if sys.platform == "darwin":
        import resource
        hwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
        return hwm, hwm
    return 0.0, 0.0


def get_scratch_mb(folder: str) -> float:
    if not os.path.isdir(folder):
        return 0.0
    total_b = sum(
        os.path.getsize(os.path.join(folder, f))
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    )
    return total_b / (1024.0 * 1024.0)


def build_psi4_mol(sys_name: str, mode_lower: str, exact_symmetry: str = "auto"):
    info = SYSTEMS[sys_name]
    lines = [f"{info['charge']} {info['mult']}", "units bohr"]
    # Psi4 dfocc mewajibkan C1 pada CHOLESKY dan DF; pada EXACT bisa auto (Abelian) atau c1
    if mode_lower in ("cholesky", "df") or exact_symmetry == "c1":
        lines.append("symmetry c1")
    for sym, x, y, z in info["atoms"]:
        lines.append(f"{sym} {x * ANG_TO_BOHR:.12f} {y * ANG_TO_BOHR:.12f} {z * ANG_TO_BOHR:.12f}")
    return psi4.geometry("\n".join(lines))


def configure_psi4(mode_lower: str, is_uhf: bool, basis: str = "cc-pVTZ", aux: str = "cc-pVTZ-RI") -> None:
    scf_t = "pk" if mode_lower == "exact" else ("cd" if mode_lower == "cholesky" else "df")
    corr_t = "conv" if mode_lower == "exact" else ("cd" if mode_lower == "cholesky" else "df")
    opts: Dict[str, Any] = {
        "basis": basis,
        "reference": "uhf" if is_uhf else "rhf",
        "e_convergence": 1e-9,
        "d_convergence": 1e-9,
        "scf_type": scf_t,
        "mp2_type": corr_t,
        "mp_type": corr_t,
        "cc_type": corr_t,
        "freeze_core": "false",
        "hess_type": "approx_diag",
        "r_convergence": 1e-6,
        "rms_mograd_convergence": 1e-5,
        "max_mograd_convergence": 1e-4,
        "maxiter": 200,
        "mo_maxiter": 50,
        "cachelevel": 2,
        "read_scf_3index": False,
    }
    if mode_lower == "df":
        opts.update({"df_basis_scf": aux, "df_basis_mp2": aux, "df_basis_cc": aux})
    elif mode_lower == "cholesky":
        opts["cholesky_tolerance"] = 1e-9
    psi4.set_options(opts)


def snapshot_wfn(wfn) -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "e": float(wfn.energy()),
        "Ca": wfn.Ca().clone(),
        "Da": wfn.Da().clone(),
        "Fa": wfn.Fa().clone(),
        "ea": wfn.epsilon_a().clone(),
    }
    if wfn.Cb() is not None:
        snap.update({
            "Cb": wfn.Cb().clone(),
            "Db": wfn.Db().clone(),
            "Fb": wfn.Fb().clone(),
            "eb": wfn.epsilon_b().clone(),
        })
    return snap


def restore_wfn(wfn, snap: Dict[str, Any]) -> None:
    wfn.set_energy(snap["e"])
    for k, g in (("Ca", wfn.Ca), ("Da", wfn.Da), ("Fa", wfn.Fa), ("ea", wfn.epsilon_a)):
        g().copy(snap[k])
    if "Cb" in snap and wfn.Cb() is not None:
        for k, g in (("Cb", wfn.Cb), ("Db", wfn.Db), ("Fb", wfn.Fb), ("eb", wfn.epsilon_b)):
            g().copy(snap[k])


def semicanonicalize_rohf_wfn(wfn) -> float:
    """Rotasi orbital ROHF ke basis semikanonik (per-irrep) dan hitung E_singles (MP2)."""
    e_singles = 0.0
    nalphapi = wfn.nalphapi()
    nbetapi = wfn.nbetapi()
    for h in range(wfn.nirrep()):
        nso = wfn.Ca().nph[h].shape[0]
        if nso == 0:
            continue

        na = nalphapi[h]
        va = nso - na
        Ca_h = np.array(wfn.Ca().nph[h], copy=True)
        Fa_h = np.asarray(wfn.Fa().nph[h])
        if na > 0:
            Foo_a = Ca_h[:, :na].T @ Fa_h @ Ca_h[:, :na]
            eo_a, Uo_a = np.linalg.eigh(0.5 * (Foo_a + Foo_a.T))
            Ca_h[:, :na] = Ca_h[:, :na] @ Uo_a
            wfn.epsilon_a().nph[h][:na] = eo_a
        if va > 0:
            Fvv_a = Ca_h[:, na:].T @ Fa_h @ Ca_h[:, na:]
            ev_a, Uv_a = np.linalg.eigh(0.5 * (Fvv_a + Fvv_a.T))
            Ca_h[:, na:] = Ca_h[:, na:] @ Uv_a
            wfn.epsilon_a().nph[h][na:] = ev_a
        wfn.Ca().nph[h][:, :] = Ca_h
        if na > 0 and va > 0:
            Fov_a = Ca_h[:, :na].T @ Fa_h @ Ca_h[:, na:]
            denom_a = wfn.epsilon_a().nph[h][:na, None] - wfn.epsilon_a().nph[h][None, na:]
            e_singles += float(np.sum((Fov_a ** 2) / denom_a))

        nb = nbetapi[h]
        vb = nso - nb
        Cb_h = np.array(wfn.Cb().nph[h], copy=True)
        Fb_h = np.asarray(wfn.Fb().nph[h])
        if nb > 0:
            Foo_b = Cb_h[:, :nb].T @ Fb_h @ Cb_h[:, :nb]
            eo_b, Uo_b = np.linalg.eigh(0.5 * (Foo_b + Foo_b.T))
            Cb_h[:, :nb] = Cb_h[:, :nb] @ Uo_b
            wfn.epsilon_b().nph[h][:nb] = eo_b
        if vb > 0:
            Fvv_b = Cb_h[:, nb:].T @ Fb_h @ Cb_h[:, nb:]
            ev_b, Uv_b = np.linalg.eigh(0.5 * (Fvv_b + Fvv_b.T))
            Cb_h[:, nb:] = Cb_h[:, nb:] @ Uv_b
            wfn.epsilon_b().nph[h][nb:] = ev_b
        wfn.Cb().nph[h][:, :] = Cb_h
        if nb > 0 and vb > 0:
            Fov_b = Cb_h[:, :nb].T @ Fb_h @ Cb_h[:, nb:]
            denom_b = wfn.epsilon_b().nph[h][:nb, None] - wfn.epsilon_b().nph[h][None, nb:]
            e_singles += float(np.sum((Fov_b ** 2) / denom_b))

    return e_singles


def calc_s2_wfn(wfn, is_uhf: bool) -> Tuple[float, float]:
    if not is_uhf:
        return 0.0, 0.0
    na, nb = wfn.nalpha(), wfn.nbeta()
    sz = 0.5 * (na - nb)
    s_ideal = float(sz * (sz + 1.0))
    if na == nb or na == 0 or nb == 0 or wfn.Cb() is None:
        return s_ideal, s_ideal
    try:
        Ca = np.asarray(wfn.Ca_subset("AO", "OCC"))
        Cb = np.asarray(wfn.Cb_subset("AO", "OCC"))
        S = np.asarray(wfn.mintshelper().ao_overlap())
        return s_ideal, float(s_ideal + nb - np.sum((Ca.T @ S @ Cb) ** 2))
    except Exception:
        return s_ideal, s_ideal


_ITER_RE = re.compile(r"^\s*(\d+)\s+-?\d+\.\d+\s+[-+]?\d+\.\d+(?:[eE][-+]?\d+)?")


def parse_occ_iterations(log_text: str) -> int:
    iters = 0
    for line in log_text.splitlines():
        m = _ITER_RE.match(line)
        if m:
            iters = max(iters, int(m.group(1)))
    return iters


def run_ram_worker(payload_str: str) -> None:
    scratch_sub = tempfile.mkdtemp(prefix="psi4_ram_v2_", dir=SHM_DIR)
    os.environ["PSI_SCRATCH"] = scratch_sub
    psi4.core.IOManager.shared_object().set_default_path(scratch_sub + "/")
    try:
        mol_w = psi4.geometry("0 1\nHe 0.0 0.0 0.0\nunits bohr\nsymmetry c1")
        bas_w = psi4.core.BasisSet.build(mol_w, "ORBITAL", "cc-pVDZ")
        psi4.core.MintsHelper(bas_w).ao_overlap()
        del bas_w, mol_w
        psi4.core.clean()
        gc.collect()

        base_rss, base_hwm = get_rss_and_hwm_mb()
        p = json.loads(payload_str)
        sys_name, mode_lower = p["sys"], p["mode"].lower()
        is_uhf = SYSTEMS[sys_name]["is_uhf"]

        mol = build_psi4_mol(sys_name, mode_lower, exact_symmetry=p.get("exact_symmetry", "auto"))
        configure_psi4(mode_lower, is_uhf, basis=p["basis"], aux=p["aux"])

        set_blas_threads(1)
        _, scf_wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
        if mode_lower in ("cholesky", "df"):
            psi4.set_options({"scf_type": "pk", "read_scf_3index": False})

        set_blas_threads(N_THREADS if mode_lower == "exact" else 1)
        psi4.energy(p["method"].lower(), ref_wfn=scf_wfn, molecule=mol)
        set_blas_threads(1)

        curr_rss, peak_hwm = get_rss_and_hwm_mb()
        scratch_ram = get_scratch_mb(scratch_sub)
        psi4.core.clean()

        total_peak = peak_hwm + scratch_ram
        net_working = max(peak_hwm - base_hwm, curr_rss - base_rss, 0.10) + scratch_ram
        print(
            json.dumps({
                "base_hwm": base_hwm,
                "scratch_mb": scratch_ram,
                "peak_hwm": total_peak,
                "net_ram": net_working,
            }),
            flush=True,
        )
    finally:
        shutil.rmtree(scratch_sub, ignore_errors=True)


def compute_psi4_suite(
    sys_list: List[str],
    modes: List[str],
    basis_name: str,
    include_omp25: bool = True,
    exact_symmetry: str = "auto",
) -> Dict[str, Any]:
    aux_name = f"{basis_name}-RI"
    results: Dict[str, Any] = {}
    tmp_log = os.path.join(SHM_DIR, f"psi4_bab4_v2_{os.getpid()}.out")

    print("\n" + "=" * 110, flush=True)
    print(
        f" [PSI4 LIVE TEST 1/2] PSI4 v{psi4.__version__} SOLVER SUITE "
        f"(THREADS: {N_THREADS} | {basis_name} / {aux_name} | EXACT_SYM={exact_symmetry.upper()})",
        flush=True,
    )
    print("=" * 110, flush=True)
    print(
        f"| {'Sistem':<8} | {'Mode ERI':<9} | {'Metode':<11} | {'E_Psi4 (Ha)':<16} | "
        f"{'Waktu (s)':<9} | {'<S^2>':<10} | {'Ideal':<6} | {'Iter':<5} | {'Status':<6} |",
        flush=True,
    )
    print(f"|{'-'*10}|{'-'*11}|{'-'*13}|{'-'*18}|{'-'*11}|{'-'*12}|{'-'*8}|{'-'*7}|{'-'*8}|", flush=True)

    tot_solver_s = 0.0
    for sys_name in sys_list:
        is_uhf = SYSTEMS[sys_name]["is_uhf"]
        ref_lbl = "U" if is_uhf else "R"

        for mode_upper in modes:
            mode_lower = mode_upper.lower()
            psi4.core.clean()
            psi4.core.clean_options()
            psi4.core.set_output_file("NUL" if os.name == "nt" else "/dev/null", False)

            mol = build_psi4_mol(sys_name, mode_lower, exact_symmetry=exact_symmetry)
            configure_psi4(mode_lower, is_uhf, basis=basis_name, aux=aux_name)

            key = f"{sys_name}|{mode_upper}"
            entry: Dict[str, Any] = {}

            set_blas_threads(1)
            try:
                t0 = time.perf_counter()
                e_scf, scf_wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
                t_scf = time.perf_counter() - t0
                s_ideal, s2_scf = calc_s2_wfn(scf_wfn, is_uhf)
                entry["s_ideal"] = s_ideal
                entry["SCF"] = {"e": float(e_scf), "t": float(t_scf), "s2": s2_scf}
                tot_solver_s += t_scf
                print(
                    f"| {sys_name:<8} | {mode_upper:<9} | {ref_lbl+'HF':<11} | {float(e_scf):<16.8f} | "
                    f"{t_scf:<9.3f} | {s2_scf:<10.6f} | {s_ideal:<6.2f} | {'-':<5} | {'OK':<6} |",
                    flush=True,
                )
            except Exception as ex:
                results[key] = {"error": str(ex)}
                psi4.core.clean()
                continue

            hf_snap = snapshot_wfn(scf_wfn)

            if is_uhf:
                try:
                    psi4.set_options({"reference": "rohf"})
                    t0 = time.perf_counter()
                    e_rohf, rohf_wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
                    t_rohf = time.perf_counter() - t0
                    _, s2_rohf = calc_s2_wfn(rohf_wfn, True)
                    entry["ROHF"] = {"e": float(e_rohf), "t": float(t_rohf), "s2": s2_rohf}
                    tot_solver_s += t_rohf
                    print(
                        f"| {sys_name:<8} | {mode_upper:<9} | {'ROHF':<11} | {float(e_rohf):<16.8f} | "
                        f"{t_rohf:<9.3f} | {s2_rohf:<10.6f} | {s_ideal:<6.2f} | {'-':<5} | {'OK':<6} |",
                        flush=True,
                    )

                    rohf_snap = snapshot_wfn(rohf_wfn)
                    if mode_lower in ("cholesky", "df"):
                        psi4.set_options({"scf_type": "pk", "read_scf_3index": False})

                    set_blas_threads(1)
                    for ro_key, ro_call in (("ROMP2", "mp2"), ("ROMP3", "mp3")):
                        try:
                            t0 = time.perf_counter()
                            if ro_key == "ROMP2":
                                psi4.set_options({"reference": "rohf"})
                                e_ro_val = float(psi4.energy(ro_call, ref_wfn=rohf_wfn, molecule=mol))
                            else:
                                e_singles_ro = semicanonicalize_rohf_wfn(rohf_wfn)
                                psi4.set_options({"reference": "uhf"})
                                e_ro_val = float(psi4.energy(ro_call, ref_wfn=rohf_wfn, molecule=mol)) + e_singles_ro
                            t_ro_val = time.perf_counter() - t0
                            entry[ro_key] = {"e": e_ro_val, "t": t_ro_val}
                            tot_solver_s += t_ro_val
                            print(
                                f"| {sys_name:<8} | {mode_upper:<9} | {ro_key:<11} | {e_ro_val:<16.8f} | "
                                f"{t_ro_val:<9.3f} | {'-':<10} | {s_ideal:<6.2f} | {'-':<5} | {'OK':<6} |",
                                flush=True,
                            )
                        except Exception as ex:
                            print(
                                f"| {sys_name:<8} | {mode_upper:<9} | {ro_key:<11} | {'ERROR':<16} | "
                                f"{0.0:<9.3f} | {'-':<10} | {s_ideal:<6.2f} | {'-':<5} | {'FAIL':<6} | ({ex})",
                                flush=True,
                            )
                        psi4.core.clean()
                        restore_wfn(rohf_wfn, rohf_snap)
                finally:
                    set_blas_threads(1)
                    psi4.core.clean()
                    configure_psi4(mode_lower, is_uhf, basis=basis_name, aux=aux_name)
                    restore_wfn(scf_wfn, hf_snap)

            if mode_lower in ("cholesky", "df"):
                psi4.set_options({"scf_type": "pk", "read_scf_3index": False})

            set_blas_threads(N_THREADS if mode_lower == "exact" else 1)
            try:
                for m_key, m_call, m_lbl in (("MP2", "mp2", f"{ref_lbl}MP2"), ("MP3", "mp3", f"{ref_lbl}MP3")):
                    try:
                        t0 = time.perf_counter()
                        e_val = float(psi4.energy(m_call, ref_wfn=scf_wfn, molecule=mol))
                        t_val = time.perf_counter() - t0
                        entry[m_key] = {"e": e_val, "t": t_val}
                        tot_solver_s += t_val
                        print(
                            f"| {sys_name:<8} | {mode_upper:<9} | {m_lbl:<11} | {e_val:<16.8f} | "
                            f"{t_val:<9.3f} | {'-':<10} | {s_ideal:<6.2f} | {'-':<5} | {'OK':<6} |",
                            flush=True,
                        )
                    except Exception as ex:
                        print(
                            f"| {sys_name:<8} | {mode_upper:<9} | {m_lbl:<11} | {'ERROR':<16} | "
                            f"{0.0:<9.3f} | {'-':<10} | {s_ideal:<6.2f} | {'-':<5} | {'FAIL':<6} | ({ex})",
                            flush=True,
                        )
                    psi4.core.clean()
                    restore_wfn(scf_wfn, hf_snap)

                oo_methods = [("OMP2", "omp2")]
                if include_omp25:
                    oo_methods.append(("OMP2.5", "omp2.5"))
                oo_methods.append(("OMP3", "omp3"))

                for m_key, m_call in oo_methods:
                    try:
                        psi4.core.set_output_file(tmp_log, False)
                        t0 = time.perf_counter()
                        e_val, opt_wfn = psi4.energy(m_call, ref_wfn=scf_wfn, molecule=mol, return_wfn=True)
                        t_val = time.perf_counter() - t0

                        psi4.core.flush_outfile()
                        psi4.core.set_output_file("NUL" if os.name == "nt" else "/dev/null", False)
                        log_txt = Path(tmp_log).read_text(errors="replace") if os.path.exists(tmp_log) else ""
                        it_cnt = parse_occ_iterations(log_txt)

                        _, s2_val = calc_s2_wfn(opt_wfn, is_uhf)
                        entry[m_key] = {"e": float(e_val), "t": float(t_val), "iter": it_cnt, "s2": s2_val}
                        tot_solver_s += t_val
                        it_disp = str(it_cnt) if it_cnt > 0 else "-"
                        print(
                            f"| {sys_name:<8} | {mode_upper:<9} | {m_key:<11} | {float(e_val):<16.8f} | "
                            f"{t_val:<9.3f} | {s2_val:<10.6f} | {s_ideal:<6.2f} | {it_disp:<5} | {'OK':<6} |",
                            flush=True,
                        )
                    except Exception as ex:
                        psi4.core.set_output_file("NUL" if os.name == "nt" else "/dev/null", False)
                        print(
                            f"| {sys_name:<8} | {mode_upper:<9} | {m_key:<11} | {'ERROR':<16} | "
                            f"{0.0:<9.3f} | {'-':<10} | {s_ideal:<6.2f} | {'-':<5} | {'FAIL':<6} | ({ex})",
                            flush=True,
                        )
                    psi4.core.clean()
                    restore_wfn(scf_wfn, hf_snap)
            finally:
                set_blas_threads(1)

            results[key] = entry
            gc.collect()

    if os.path.exists(tmp_log):
        try:
            os.remove(tmp_log)
        except Exception:
            pass

    print("=" * 110, flush=True)
    print(f" Total Waktu Solver Psi4: {tot_solver_s:.3f} s", flush=True)
    return results


def compute_ram_benchmark(
    cases: List[Tuple[str, str, str, str, str]],
    exact_symmetry: str = "auto",
) -> Dict[str, Any]:
    ram_results: Dict[str, Any] = {}
    script_path = os.path.abspath(__file__)
    print("\n" + "=" * 112, flush=True)
    print(" [PSI4 LIVE TEST 2/2] PENGUKURAN PEAK RSS (VmHWM) + SCRATCH /dev/shm & NET WORKING RAM (PSI4)", flush=True)
    print("=" * 112, flush=True)
    print(
        f"| {'Sistem':<8} | {'Basis':<8} | {'Mode ERI':<9} | {'Metode':<6} | "
        f"{'Base HWM(MB)':<13} | {'Scratch(MB)':<12} | {'Peak Tot(MB)':<13} | {'Net RAM(MB)':<12} |",
        flush=True,
    )
    print(f"|{'-'*10}|{'-'*10}|{'-'*11}|{'-'*8}|{'-'*15}|{'-'*14}|{'-'*15}|{'-'*14}|", flush=True)

    for sys_name, b_name, aux_name, mode, method in cases:
        key = f"{sys_name}|{b_name}|{mode}|{method}"
        payload = json.dumps({
            "sys": sys_name,
            "basis": b_name,
            "aux": aux_name,
            "mode": mode,
            "method": method,
            "exact_symmetry": exact_symmetry,
        })
        res_p = {"base_hwm": 0.0, "scratch_mb": 0.0, "peak_hwm": 0.0, "net_ram": 0.0}
        try:
            out = subprocess.check_output([sys.executable, script_path, "--ram-worker", payload], text=True)
            for line in reversed(out.splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    res_p = json.loads(line)
                    break
        except Exception:
            pass
        ram_results[key] = res_p
        print(
            f"| {sys_name:<8} | {b_name:<8} | {mode:<9} | {method:<6} | "
            f"{res_p.get('base_hwm', 0.0):<13.2f} | {res_p.get('scratch_mb', 0.0):<12.2f} | "
            f"{res_p.get('peak_hwm', 0.0):<13.2f} | {res_p.get('net_ram', 0.0):<12.2f} |",
            flush=True,
        )
    print("=" * 112, flush=True)
    return ram_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Psi4 Bab IV v2 Reference Benchmark Runner")
    parser.add_argument("--out", default="psi4_bab4_v2_raw.json")
    parser.add_argument("--log", default="13_psi4_bab4_v2_log.txt")
    parser.add_argument("--core5", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-ram", action="store_true")
    parser.add_argument("--no-omp25", action="store_true")
    parser.add_argument("--exact-symmetry", choices=["auto", "c1"], default="auto")
    parser.add_argument("--ram-worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.ram_worker is not None:
        run_ram_worker(args.ram_worker)
        sys.exit(0)

    sys_list = ["H2O", "OH"] if args.quick else (CORE5_ORDER if args.core5 else ALL_SYSTEM_ORDER)
    modes = ["DF"] if args.quick else ERI_MODE_ORDER
    basis_name = "cc-pVDZ" if args.quick else "cc-pVTZ"
    ram_cases = [("H2O", "cc-pvdz", "cc-pvdz-ri", "DF", "OMP3")] if args.quick else RAM_CASES

    logger = TeeLogger(args.log)
    sys.stdout = logger
    try:
        t0 = time.perf_counter()
        ref_suite = compute_psi4_suite(
            sys_list, modes, basis_name,
            include_omp25=(not args.no_omp25),
            exact_symmetry=args.exact_symmetry,
        )
        ram_bench = {} if args.skip_ram else compute_ram_benchmark(ram_cases, exact_symmetry=args.exact_symmetry)
        wall_s = time.perf_counter() - t0

        Path(args.out).write_text(
            json.dumps({
                "engine": f"psi4-v{psi4.__version__}",
                "psi4_version": str(psi4.__version__),
                "exact_symmetry": args.exact_symmetry,
                "python_exe": sys.executable,
                "threads": N_THREADS,
                "wall_time_s": wall_s,
                "ref_suite": ref_suite,
                "ram_bench": ram_bench,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"\n[SELESAI] Data benchmark Psi4 Bab IV v2 disimpan ke: {args.out} ({wall_s:.2f} s)", flush=True)
    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()
