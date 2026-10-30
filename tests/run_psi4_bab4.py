#!/usr/bin/env python3
"""
Runner Mandiri Psi4 untuk Benchmark Bab IV (Full Abelian Symmetry on EXACT, DIIS Enabled & Real-Time UI)
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
    "MKL_DYNAMIC": "FALSE",
    "OMP_MAX_ACTIVE_LEVELS": "1",
    "HDF5_USE_FILE_LOCKING": "FALSE",
})

SHM_DIR = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
if "PSI_SCRATCH" not in os.environ:
    os.environ["PSI_SCRATCH"] = SHM_DIR

import numpy as np
import psi4

psi4.set_num_threads(N_THREADS)
psi4.set_memory("8 GB")
psi4.core.be_quiet()

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
    """Menampilkan output secara real-time ke terminal dan merekamnya untuk disimpan ke berkas."""
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
    lib_names = ("libmkl_rt.so", "libmkl_rt.so.2", "libopenblas.so", "libopenblas.so.0")
    candidates = []

    env_lib_dirs = [
        os.path.join(sys.prefix, "lib"),
        os.environ.get("CONDA_PREFIX", "") and os.path.join(os.environ["CONDA_PREFIX"], "lib"),
    ]
    for d in env_lib_dirs:
        if d and os.path.isdir(d):
            for name in lib_names:
                full_p = os.path.join(d, name)
                if os.path.exists(full_p):
                    candidates.append(full_p)

    if os.path.exists("/proc/self/maps"):
        try:
            with open("/proc/self/maps", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6 and parts[5].startswith("/"):
                        path = parts[5]
                        base = os.path.basename(path)
                        if ("mkl_rt" in base or "openblas" in base) and path not in candidates:
                            candidates.append(path)
        except Exception:
            pass

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


def set_blas_threads(num_t: int):
    for fn in _BLAS_SETTERS:
        try:
            fn(int(num_t))
        except Exception:
            pass


def get_rss_and_hwm_mb():
    rss, hwm = 0.0, 0.0
    with open("/proc/self/status", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"):
                hwm = int(line.split()[1]) / 1024.0
    return rss, hwm


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
    """
    Membangun molekul Psi4:
    - Mode EXACT menggunakan simetri grup titik Abelian penuh (D2h, C2v, Cs) seperti pada benc.py.
    - Mode DF & CHOLESKY menggunakan symmetry c1 karena modul dfocc Psi4 mewajibkan C1.
    """
    info = SYSTEMS[sys_name]
    lines = [
        f"{info['charge']} {info['mult']}",
        "units bohr",
    ]
    if mode_lower in ("df", "cholesky"):
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
        "freeze_core": False,
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
    snap = {
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


def restore_wfn(wfn, snap):
    wfn.set_energy(snap["e"])
    wfn.Ca().copy(snap["Ca"])
    wfn.Da().copy(snap["Da"])
    wfn.Fa().copy(snap["Fa"])
    wfn.epsilon_a().copy(snap["ea"])
    if "Cb" in snap and wfn.Cb() is not None:
        wfn.Cb().copy(snap["Cb"])
        wfn.Db().copy(snap["Db"])
        wfn.Fb().copy(snap["Fb"])
        wfn.epsilon_b().copy(snap["eb"])


def calc_s2_wfn(wfn, is_uhf: bool):
    """
    Menghitung <S^2> secara eksak baik untuk simetri C1 maupun simetri Abelian multi-irrep (D2h/C2v).
    Menjumlahkan kontribusi tumpang-tindih orbital terisi per blok irrep h.
    """
    if not is_uhf:
        return 0.0, 0.0
    na, nb = wfn.nalpha(), wfn.nbeta()
    sz = 0.5 * (na - nb)
    s_ideal = float(sz * (sz + 1.0))
    if na == nb or na == 0 or nb == 0 or wfn.Cb() is None:
        return s_ideal, s_ideal
    try:
        nalphapi = wfn.nalphapi()
        nbetapi = wfn.nbetapi()
        nirrep = wfn.nirrep()
        Ca_blocks = wfn.Ca().nph
        Cb_blocks = wfn.Cb().nph
        S_blocks = wfn.S().nph

        overlap_sq_sum = 0.0
        for h in range(nirrep):
            oa_h = nalphapi[h]
            ob_h = nbetapi[h]
            if oa_h == 0 or ob_h == 0:
                continue
            Ca_occ = np.asarray(Ca_blocks[h], dtype=np.float64)[:, :oa_h]
            Cb_occ = np.asarray(Cb_blocks[h], dtype=np.float64)[:, :ob_h]
            S_h = np.asarray(S_blocks[h], dtype=np.float64)
            S_ab = Ca_occ.T @ S_h @ Cb_occ
            overlap_sq_sum += float(np.sum(S_ab ** 2))

        return s_ideal, float(s_ideal + nb - overlap_sq_sum)
    except Exception:
        return s_ideal, s_ideal


def parse_occ_iterations(log_text: str) -> int:
    iters = 0
    for line in log_text.splitlines():
        m = re.match(r"^\s*(\d+)\s+-?\d+\.\d+\s+[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", line)
        if m:
            iters = max(iters, int(m.group(1)))
    return iters


def run_ram_worker(payload_str: str):
    """Worker terisolasi untuk mengukur Peak RSS + Scratch /dev/shm Psi4 tanpa duplikasi kode."""
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

        corr_blas_threads = N_THREADS if mode_lower == "exact" else 1
        set_blas_threads(corr_blas_threads)
        psi4.set_options({"do_diis": True})
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
    print(f"  [LIVE TEST 1/2] PSI4 105-RUN SOLVER SUITE (THREADS: {N_THREADS} | cc-pVTZ / cc-pVTZ-RI)", flush=True)
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
            psi4.core.be_quiet()

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
            if mode_lower in ("cholesky", "df"):
                psi4.set_options({"scf_type": "pk", "read_scf_3index": False})

            # occ/libdpd (EXACT) serial di outer-loop -> BLAS multi-thread.
            # dfocc (CHOLESKY/DF) paralel OpenMP di outer-loop -> BLAS 1 thread.
            corr_blas_threads = N_THREADS if mode_lower == "exact" else 1
            set_blas_threads(corr_blas_threads)
            try:
                for m_key, m_call, m_lbl in (("MP2", "mp2", f"{ref_lbl}MP2"), ("MP3", "mp3", f"{ref_lbl}MP3")):
                    try:
                        set_blas_threads(corr_blas_threads)
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
                        set_blas_threads(corr_blas_threads)
                        psi4.set_options({"do_diis": True})

                        if os.path.exists(tmp_log):
                            os.remove(tmp_log)
                        psi4.core.set_output_file(tmp_log, False)

                        t0 = time.perf_counter()
                        e_val, opt_wfn = psi4.energy(m_call, ref_wfn=scf_wfn, molecule=mol, return_wfn=True)
                        t_val = time.perf_counter() - t0

                        psi4.core.flush_outfile()
                        psi4.core.be_quiet()
                        log_txt = Path(tmp_log).read_text(errors="replace") if os.path.exists(tmp_log) else ""
                        it_cnt = parse_occ_iterations(log_txt)

                        _, s2_val = calc_s2_wfn(opt_wfn, is_uhf)
                        entry[m_key] = {"e": float(e_val), "t": float(t_val), "iter": it_cnt, "s2": s2_val}
                        tot_solver_s += t_val
                        it_disp = str(it_cnt) if it_cnt > 0 else "-"
                        print(f"| {sys_name:<8} | {mode_upper:<9} | {m_key:<6} | {float(e_val):<16.8f} | {t_val:<9.3f} | {s2_val:<10.6f} | {s_ideal:<6.2f} | {it_disp:<6} | {'OK':<6} |", flush=True)
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

    print("=" * 106, flush=True)
    print(f"  Total Waktu Solver Psi4 (105 Run): {tot_solver_s:.3f} s", flush=True)
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