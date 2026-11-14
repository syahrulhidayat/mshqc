#!/usr/bin/env python3
"""
Runner Mandiri Psi4 untuk Benchmark Bab IV (Simplified, Zero-Redundancy & Real-Time UI)
Output: psi4_bab4_raw.json, psi4_bab4_runner.log, psi4_bab4_test.txt
"""

import os
import re
import gc
import sys
import time
import json
import ctypes
import shutil
import tempfile
import argparse
import subprocess
from pathlib import Path

try:
    os.sched_setaffinity(0, set(range(os.cpu_count() or 4)))
except Exception:
    pass

CPU_AVAIL = os.cpu_count() or 4
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", str(min(6, CPU_AVAIL))))
os.environ.update({
    "OMP_NUM_THREADS": str(N_THREADS),
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_MAX_ACTIVE_LEVELS": "1",
    "HDF5_USE_FILE_LOCKING": "FALSE",
})

SHM_DIR = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
if "PSI_SCRATCH" not in os.environ:
    os.environ["PSI_SCRATCH"] = SHM_DIR

import numpy as np
import psi4

psi4.set_num_threads(N_THREADS)
psi4.set_memory("1800 MB" if os.name == "nt" else "4 GB")
psi4.core.set_output_file("NUL" if os.name == "nt" else "/dev/null", False)

ANG_TO_BOHR = 1.8897261246257702

SYSTEMS = {
    "Neon":    {"atoms": [("Ne", 0.0, 0.0,  0.0000)], "charge": 0, "mult": 1, "is_uhf": False},
    "H2O":     {"atoms": [("O",  0.0, 0.0,  0.1173), ("H", 0.0, 0.7572, -0.4692), ("H", 0.0, -0.7572, -0.4692)], "charge": 0, "mult": 1, "is_uhf": False},
    "Lithium": {"atoms": [("Li", 0.0, 0.0,  0.0000)], "charge": 0, "mult": 2, "is_uhf": True},
    "Carbon":  {"atoms": [("C",  0.0, 0.0,  0.0000)], "charge": 0, "mult": 3, "is_uhf": True},
    "Oxygen":  {"atoms": [("O",  0.0, 0.0,  0.0000)], "charge": 0, "mult": 3, "is_uhf": True},
    "OH":      {"atoms": [("O",  0.0, 0.0,  0.0000), ("H", 0.0, 0.0, 0.9697)], "charge": 0, "mult": 2, "is_uhf": True},
    "O2":      {"atoms": [("O",  0.0, 0.0, -0.6035), ("O", 0.0, 0.0, 0.6035)], "charge": 0, "mult": 3, "is_uhf": True},
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


class TeeRecorder:
    def __init__(self):
        self.terminal = sys.stdout
        self.chunks = []

    def write(self, message: str):
        self.terminal.write(message)
        self.terminal.flush()
        self.chunks.append(message)

    def flush(self):
        self.terminal.flush()

    def get_text(self) -> str:
        return "".join(self.chunks)


def atomic_write_text(path_str: str, content: str):
    tmp_path = f"{path_str}.tmp.{os.getpid()}"
    Path(tmp_path).write_text(content, encoding="utf-8")
    os.replace(tmp_path, path_str)


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


def get_rss_and_hwm_mb() -> tuple[float, float]:
    """Mengukur (Current RSS MB, Peak HWM MB) secara terpadu di Linux, macOS, dan Windows."""
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

    elif sys.platform == "darwin":
        class MachTaskBasicInfo(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("resident_size", ctypes.c_uint64),
                ("resident_size_max", ctypes.c_uint64),
                ("user_time_sec", ctypes.c_int32),
                ("user_time_usec", ctypes.c_int32),
                ("system_time_sec", ctypes.c_int32),
                ("system_time_usec", ctypes.c_int32),
                ("policy", ctypes.c_int32),
                ("suspend_count", ctypes.c_int32),
            ]
        try:
            libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            info = MachTaskBasicInfo()
            count = ctypes.c_uint32(12)
            kr = libc.task_info(
                libc.mach_task_self(), 20, ctypes.byref(info), ctypes.byref(count)
            )
            if kr == 0:
                rss_mb = info.resident_size / (1024.0 * 1024.0)
                hwm_mb = info.resident_size_max / (1024.0 * 1024.0)
                return rss_mb, max(hwm_mb, rss_mb)
        except Exception:
            import resource
            hwm_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
            return hwm_mb, hwm_mb

    elif sys.platform == "win32":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        try:
            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                rss_mb = pmc.WorkingSetSize / (1024.0 * 1024.0)
                hwm_mb = pmc.PeakWorkingSetSize / (1024.0 * 1024.0)
                return rss_mb, max(hwm_mb, rss_mb)
        except Exception:
            pass

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


def build_psi4_mol(sys_name: str, mode_lower: str):
    info = SYSTEMS[sys_name]
    lines = [f"{info['charge']} {info['mult']}", "units bohr"]
    if mode_lower in ("df", "cholesky") or sys_name == "OH":
        lines.append("symmetry c1")
    for sym, x, y, z in info["atoms"]:
        lines.append(f"{sym} {x * ANG_TO_BOHR:.12f} {y * ANG_TO_BOHR:.12f} {z * ANG_TO_BOHR:.12f}")
    return psi4.geometry("\n".join(lines))


def configure_psi4(mode_lower: str, is_uhf: bool, basis: str = "cc-pVTZ", aux: str = "cc-pVTZ-RI"):
    scf_t = "pk" if mode_lower == "exact" else ("cd" if mode_lower == "cholesky" else "df")
    corr_t = "conv" if mode_lower == "exact" else ("cd" if mode_lower == "cholesky" else "df")
    opts = {
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

def semicanonicalize_rohf_wfn(wfn) -> float:
    """Rotasi orbital ROHF ke basis semikanonik (per-irrep) dan hitung E_singles (MP2)."""
    e_singles = 0.0
    nalphapi = wfn.nalphapi()
    nbetapi = wfn.nbetapi()
    for h in range(wfn.nirrep()):
        nso = wfn.Ca().nph[h].shape[0]
        if nso == 0:
            continue

        # Kanal spin Alpha (DOCC + SOCC vs VIR)
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

        # Kanal spin Beta (DOCC vs SOCC + VIR)
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

def calc_s2_wfn(wfn, is_uhf: bool):
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


def run_ram_worker(payload_str: str):
    scratch_sub = tempfile.mkdtemp(prefix="psi4_ram_", dir=SHM_DIR)
    os.environ["PSI_SCRATCH"] = scratch_sub
    psi4.core.IOManager.shared_object().set_default_path(scratch_sub + "/")
    try:
        mol_w = psi4.geometry("0 1\nH 0.0 0.0 0.0\nH 0.0 0.0 1.4\nunits bohr\nsymmetry c1")
        bas_w = psi4.core.BasisSet.build(mol_w, "ORBITAL", "cc-pVDZ")
        psi4.core.MintsHelper(bas_w).ao_overlap()
        del bas_w, mol_w
        psi4.core.clean()
        gc.collect()

        base_rss, base_hwm = get_rss_and_hwm_mb()
        p = json.loads(payload_str)
        sys_name, mode_lower = p["sys"], p["mode"].lower()
        is_uhf = SYSTEMS[sys_name]["is_uhf"]

        mol = build_psi4_mol(sys_name, mode_lower)
        configure_psi4(mode_lower, is_uhf, basis=p["basis"], aux=p["aux"])

        set_blas_threads(1)
        _, scf_wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
        if mode_lower in ("cholesky", "df"):
            psi4.set_options({"scf_type": "pk", "read_scf_3index": False})

        set_blas_threads(1)
        psi4.energy(p["method"].lower(), ref_wfn=scf_wfn, molecule=mol)
        set_blas_threads(1)

        curr_rss, peak_hwm = get_rss_and_hwm_mb()
        scratch_ram = get_scratch_mb(scratch_sub)
        psi4.core.clean()

        total_peak = peak_hwm + scratch_ram
        net_working = max(peak_hwm - base_hwm, curr_rss - base_rss, 0.10) + scratch_ram
        print(json.dumps({"base_hwm": base_hwm, "scratch_mb": scratch_ram, "peak_hwm": total_peak, "net_ram": net_working}), flush=True)
    finally:
        shutil.rmtree(scratch_sub, ignore_errors=True)


def compute_psi4_suite():
    results = {}
    tmp_log = os.path.join(SHM_DIR, f"psi4_bab4_{os.getpid()}.out")

    print("\n" + "=" * 106, flush=True)
    print(f"  [LIVE TEST 1/2] PSI4 SOLVER SUITE (THREADS: {N_THREADS} | cc-pVTZ / cc-pVTZ-RI)", flush=True)
    print("=" * 106, flush=True)
    print(f"| {'Sistem':<8} | {'Mode ERI':<9} | {'Metode':<6} | {'E_Psi4 (Ha)':<16} | {'Waktu (s)':<9} | {'<S^2>':<10} | {'Ideal':<6} | {'Iter':<6} | {'Status':<6} |", flush=True)
    print(f"|{'-'*10}|{'-'*11}|{'-'*8}|{'-'*18}|{'-'*11}|{'-'*12}|{'-'*8}|{'-'*8}|{'-'*8}|", flush=True)

    tot_solver_s = 0.0
    for sys_name in ALL_SYSTEM_ORDER:
        is_uhf = SYSTEMS[sys_name]["is_uhf"]
        ref_lbl = "U" if is_uhf else "R"

        for mode_upper in ERI_MODE_ORDER:
            mode_lower = mode_upper.lower()
            psi4.core.clean()
            psi4.core.clean_options()
            psi4.core.set_output_file("NUL" if os.name == "nt" else "/dev/null", False)

            mol = build_psi4_mol(sys_name, mode_lower)
            configure_psi4(mode_lower, is_uhf)

            key = f"{sys_name}|{mode_upper}"
            entry = {}

            set_blas_threads(1)
            try:
                t0 = time.perf_counter()
                e_scf, scf_wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
                t_scf = time.perf_counter() - t0
                s_ideal, s2_scf = calc_s2_wfn(scf_wfn, is_uhf)
                entry["SCF"] = {"e": float(e_scf), "t": float(t_scf), "s2": s2_scf}
                tot_solver_s += t_scf
                print(f"| {sys_name:<8} | {mode_upper:<9} | {ref_lbl+'HF':<6} | {float(e_scf):<16.8f} | {t_scf:<9.3f} | {s2_scf:<10.6f} | {s_ideal:<6.2f} | {'-':<6} | {'OK':<6} |", flush=True)
            except Exception as ex:
                results[key] = {"error": str(ex)}
                psi4.core.clean()
                continue

            hf_snap = snapshot_wfn(scf_wfn)

            # Evaluasi ROHF, ROMP2, dan ROMP3 untuk sistem kulit terbuka (tanpa mengganggu scf_wfn UHF)
            if is_uhf:
                try:
                    psi4.set_options({"reference": "rohf"})
                    t0 = time.perf_counter()
                    e_rohf, rohf_wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
                    t_rohf = time.perf_counter() - t0
                    _, s2_rohf = calc_s2_wfn(rohf_wfn, True)
                    entry["ROHF"] = {"e": float(e_rohf), "t": float(t_rohf), "s2": s2_rohf}
                    tot_solver_s += t_rohf
                    print(f"| {sys_name:<8} | {mode_upper:<9} | {'ROHF':<6} | {float(e_rohf):<16.8f} | {t_rohf:<9.3f} | {s2_rohf:<10.6f} | {s_ideal:<6.2f} | {'-':<6} | {'OK':<6} |", flush=True)

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
                            print(f"| {sys_name:<8} | {mode_upper:<9} | {ro_key:<6} | {e_ro_val:<16.8f} | {t_ro_val:<9.3f} | {'-':<10} | {s_ideal:<6.2f} | {'-':<6} | {'OK':<6} |", flush=True)
                        except Exception as ex:
                            print(f"| {sys_name:<8} | {mode_upper:<9} | {ro_key:<6} | {'ERROR':<16} | {0.0:<9.3f} | {'-':<10} | {s_ideal:<6.2f} | {'-':<6} | {'FAIL':<6} | ({ex})", flush=True)
                        psi4.core.clean()
                        restore_wfn(rohf_wfn, rohf_snap)
                    set_blas_threads(1)
                except Exception:
                    pass
                finally:
                    set_blas_threads(1)
                    psi4.core.clean()
                    configure_psi4(mode_lower, is_uhf)
                    restore_wfn(scf_wfn, hf_snap)

            if mode_lower in ("cholesky", "df"):
                psi4.set_options({"scf_type": "pk", "read_scf_3index": False})

            set_blas_threads(1)
            try:
                for m_key, m_call, m_lbl in (("MP2", "mp2", f"{ref_lbl}MP2"), ("MP3", "mp3", f"{ref_lbl}MP3")):
                    try:
                        t0 = time.perf_counter()
                        e_val = float(psi4.energy(m_call, ref_wfn=scf_wfn, molecule=mol))
                        t_val = time.perf_counter() - t0
                        entry[m_key] = {"e": e_val, "t": t_val}
                        tot_solver_s += t_val
                        print(f"| {sys_name:<8} | {mode_upper:<9} | {m_lbl:<6} | {e_val:<16.8f} | {t_val:<9.3f} | {'-':<10} | {s_ideal:<6.2f} | {'-':<6} | {'OK':<6} |", flush=True)
                    except Exception:
                        pass
                    psi4.core.clean()
                    restore_wfn(scf_wfn, hf_snap)

                for m_key, m_call in (("OMP2", "omp2"), ("OMP3", "omp3")):
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
                        print(f"| {sys_name:<8} | {mode_upper:<9} | {m_key:<6} | {float(e_val):<16.8f} | {t_val:<9.3f} | {s2_val:<10.6f} | {s_ideal:<6.2f} | {it_disp:<6} | {'OK':<6} |", flush=True)
                    except Exception:
                        psi4.core.set_output_file("NUL" if os.name == "nt" else "/dev/null", False)
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

    print("=" * 106, flush=True)
    print(f"  Total Waktu Solver Psi4: {tot_solver_s:.3f} s", flush=True)
    return results

def compute_ram_benchmark():
    ram_results = {}
    script_path = os.path.abspath(__file__)
    print("\n" + "=" * 112, flush=True)
    print("  [LIVE TEST 2/2] PENGUKURAN PEAK RSS (VmHWM) + SCRATCH /dev/shm & NET WORKING RAM (PSI4)", flush=True)
    print("=" * 112, flush=True)
    print(f"| {'Sistem':<8} | {'Basis':<8} | {'Mode ERI':<9} | {'Metode':<6} | {'Base HWM(MB)':<13} | {'Scratch(MB)':<12} | {'Peak Tot(MB)':<13} | {'Net RAM(MB)':<12} |", flush=True)
    print(f"|{'-'*10}|{'-'*10}|{'-'*11}|{'-'*8}|{'-'*15}|{'-'*14}|{'-'*15}|{'-'*14}|", flush=True)

    for sys_name, b_name, aux_name, mode, method in RAM_CASES:
        key = f"{sys_name}|{b_name}|{mode}|{method}"
        payload = json.dumps({"sys": sys_name, "basis": b_name, "aux": aux_name, "mode": mode, "method": method})
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Psi4 Bab IV Standalone Benchmark Runner (Real-Time UI)")
    parser.add_argument("--out", default="psi4_bab4_raw.json", help="Path output berkas JSON Psi4")
    parser.add_argument("--log", default="psi4_bab4_runner.log", help="Path output berkas log teks real-time")
    parser.add_argument("--ram-worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.ram_worker is not None:
        run_ram_worker(args.ram_worker)
        sys.exit(0)

    recorder = TeeRecorder()
    sys.stdout = recorder
    try:
        t_start = time.perf_counter()
        ref_suite = compute_psi4_suite()
        ram_bench = compute_ram_benchmark()
        wall_time = time.perf_counter() - t_start

        output_payload = {
            "engine": "psi4",
            "python_exe": sys.executable,
            "threads": N_THREADS,
            "wall_time_s": wall_time,
            "ref_suite": ref_suite,
            "ram_bench": ram_bench,
        }

        atomic_write_text(args.out, json.dumps(output_payload, indent=2))
        print(f"\n[SELESAI] Data benchmark Psi4 Bab IV disimpan ke: {args.out} & {args.log} ({wall_time:.2f} s)", flush=True)
    finally:
        sys.stdout = recorder.terminal

    log_content = recorder.get_text()
    atomic_write_text(args.log, log_content)
    atomic_write_text("psi4_bab4_test.txt", log_content)