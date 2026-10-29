#!/usr/bin/env python3
"""
MSHQC Bab IV Benchmark Runner (Clean User-Friendly API & Real-Time Console UI)
Output: mshqc_bab4_raw.json, mshqc_bab4_runner.log, mshqc_bab4_test.txt
"""

import os
import gc
import sys
import time
import json
import ctypes
import argparse
import subprocess
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import mshqc

try:
    _LIBC = ctypes.CDLL(None)
except Exception:
    _LIBC = None

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
TRACE_CASES = {("H2O", "DF"), ("Oxygen", "DF")}

TABLE1_CASES = [
    ("Neon", "cc-pvdz", "cc-pvdz-ri"), ("Neon", "cc-pvtz", "cc-pvtz-ri"), ("Neon", "cc-pvqz", "cc-pvqz-ri"),
    ("H2O", "cc-pvdz", "cc-pvdz-ri"), ("H2O", "cc-pvtz", "cc-pvtz-ri"), ("H2O", "cc-pvqz", "cc-pvqz-ri"),
    ("Lithium", "cc-pvtz", "cc-pvtz-ri"), ("Carbon", "cc-pvtz", "cc-pvtz-ri"),
    ("Oxygen", "cc-pvtz", "cc-pvtz-ri"), ("OH", "cc-pvtz", "cc-pvtz-ri"), ("O2", "cc-pvtz", "cc-pvtz-ri"),
]

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


class TeeLogger:
    """Menampilkan output secara real-time ke terminal sekaligus menyimpannya ke berkas log."""
    def __init__(self, log_path: str):
        self.terminal = sys.__stdout__
        self.log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        self.chunks = []

    def write(self, message: str):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()
        self.chunks.append(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def fileno(self):
        return self.terminal.fileno()

    def get_text(self) -> str:
        return "".join(self.chunks)

    def close(self):
        self.log_file.close()


def atomic_write_text(path_str: str, content: str):
    tmp_path = f"{path_str}.tmp.{os.getpid()}"
    Path(tmp_path).write_text(content, encoding="utf-8")
    os.replace(tmp_path, path_str)


@contextmanager
def capture_c_stdout():
    sys.stdout.flush()
    if _LIBC is not None:
        _LIBC.fflush(None)
    old_fd = os.dup(1)
    r_fd, w_fd = os.pipe()
    buf = []
    try:
        os.dup2(w_fd, 1)
        os.close(w_fd)
        yield buf
    finally:
        sys.stdout.flush()
        if _LIBC is not None:
            _LIBC.fflush(None)
        os.dup2(old_fd, 1)
        os.close(old_fd)
        chunks = []
        while True:
            data = os.read(r_fd, 4096)
            if not data:
                break
            chunks.append(data.decode("utf-8", errors="replace"))
        os.close(r_fd)
        buf.append("".join(chunks))


def get_rss_and_hwm_mb():
    rss, hwm = 0.0, 0.0
    with open("/proc/self/status", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"):
                hwm = int(line.split()[1]) / 1024.0
    return rss, hwm


def build_mol(sys_name: str) -> mshqc.Molecule:
    info = SYSTEMS[sys_name]
    mol = mshqc.Molecule(charge=info["charge"], multiplicity=info["mult"])
    for sym, x, y, z in info["atoms"]:
        mol.add_atom(sym, x, y, z, units="angstrom")
    return mol


def compute_spin_s2(ints, res, na: int, nb: int):
    if na == nb:
        return 0.0, 0.0
    sz = 0.5 * (na - nb)
    s_ideal = float(sz * (sz + 1.0))
    if nb == 0:
        return s_ideal, s_ideal
    S_ao = np.asarray(ints.compute_overlap())
    Ca_occ = np.asarray(res.C_alpha)[:, :na]
    Cb_occ = np.asarray(res.C_beta)[:, :nb]
    return s_ideal, float(s_ideal + nb - np.sum((Ca_occ.T @ S_ao @ Cb_occ) ** 2))


def run_ram_worker(payload_str: str):
    """Worker subproses terisolasi untuk mengukur VmHWM & Net Working RAM tanpa redundansi kode."""
    mol_w = mshqc.Molecule(0, 1)
    mol_w.add_atom("He", 0.0, 0.0, 0.0)
    bas_w = mshqc.BasisSet("cc-pvdz", mol_w)
    mshqc.IntegralEngine(mol_w, bas_w).compute_overlap()
    del bas_w, mol_w
    gc.collect()

    base_rss, base_hwm = get_rss_and_hwm_mb()
    p = json.loads(payload_str)
    mode_lower = p["mode"].lower()

    mol = build_mol(p["sys"])
    basis = mshqc.BasisSet(p["basis"], mol)
    ints = mshqc.IntegralEngine(mol, basis)

    scf_conf = mshqc.SCFConfig(eri_method=mode_lower, threshold=1e-9, aux_basis_name=p["aux"])
    mp2_conf = mshqc.MP2Config(eri_method=mode_lower, threshold=1e-9, gradient_threshold=1e-5, aux_basis_name=p["aux"])

    scf_cls = mshqc.UHF if SYSTEMS[p["sys"]]["is_uhf"] else mshqc.RHF
    with capture_c_stdout():
        scf_res = scf_cls(mol, basis, ints, scf_conf).compute()
    gc.collect()

    omp_cls = mshqc.OMP2 if p["method"] == "OMP2" else mshqc.OMP3
    with capture_c_stdout():
        omp_cls(mol, basis, ints, scf_res, mp2_conf).compute()

    curr_rss, peak_hwm = get_rss_and_hwm_mb()
    net_ram = max(peak_hwm - base_hwm, curr_rss - base_rss, 0.10)
    print(json.dumps({"base_hwm": base_hwm, "peak_hwm": peak_hwm, "net_ram": net_ram}), flush=True)


def compute_mshqc_suite():
    ref_suite, traces = {}, {}
    print("\n" + "=" * 106, flush=True)
    print("  [LIVE TEST 1/5] MSHQC 105-RUN SOLVER SUITE (HF, MP2, MP3, OMP2, OMP3 | cc-pVTZ / cc-pVTZ-RI)", flush=True)
    print("=" * 106, flush=True)
    print(f"| {'Sistem':<8} | {'Mode ERI':<9} | {'Metode':<6} | {'E_mshqc (Ha)':<16} | {'Waktu (s)':<9} | {'<S^2>':<10} | {'Ideal':<6} | {'Iter':<6} | {'Status':<6} |", flush=True)
    print(f"|{'-'*10}|{'-'*11}|{'-'*8}|{'-'*18}|{'-'*11}|{'-'*12}|{'-'*8}|{'-'*8}|{'-'*8}|", flush=True)

    tot_solver_s = 0.0
    for sys_name in ALL_SYSTEM_ORDER:
        is_uhf = SYSTEMS[sys_name]["is_uhf"]
        ref_lbl = "U" if is_uhf else "R"
        mol = build_mol(sys_name)
        basis = mshqc.BasisSet("cc-pVTZ", mol)
        ints = mshqc.IntegralEngine(mol, basis)

        scf_cls = mshqc.UHF if is_uhf else mshqc.RHF
        mp2_cls = mshqc.UMP2 if is_uhf else mshqc.RMP2
        mp3_cls = mshqc.UMP3 if is_uhf else mshqc.RMP3

        for mode in ERI_MODE_ORDER:
            # PENTING: C++ mshqc mengharuskan eri_method dalam huruf kecil ("exact", "cholesky", "df")
            mode_lower = mode.lower()
            scf_conf = mshqc.SCFConfig(
                eri_method=mode_lower, threshold=1e-9, max_iterations=200, aux_basis_name="cc-pVTZ-RI"
            )
            mp2_conf = mshqc.MP2Config(
                eri_method=mode_lower, threshold=1e-9, gradient_threshold=1e-5, max_iterations=50, aux_basis_name="cc-pVTZ-RI"
            )

            with capture_c_stdout():
                t0 = time.perf_counter()
                scf_res = scf_cls(mol, basis, ints, scf_conf).compute()
                t_scf = time.perf_counter() - t0
            na, nb = scf_res.n_occ_alpha, scf_res.n_occ_beta
            s_ideal, s2_uhf = compute_spin_s2(ints, scf_res, na, nb)
            print(f"| {sys_name:<8} | {mode:<9} | {ref_lbl+'HF':<6} | {scf_res.energy_total:<16.8f} | {t_scf:<9.3f} | {s2_uhf:<10.6f} | {s_ideal:<6.2f} | {'-':<6} | {'OK':<6} |", flush=True)

            with capture_c_stdout():
                t0 = time.perf_counter()
                res_mp2 = mp2_cls(mol, basis, ints, scf_res, mp2_conf).compute()
                t_mp2 = time.perf_counter() - t0
            print(f"| {sys_name:<8} | {mode:<9} | {ref_lbl+'MP2':<6} | {res_mp2.energy_total:<16.8f} | {t_mp2:<9.3f} | {'-':<10} | {s_ideal:<6.2f} | {'-':<6} | {'OK':<6} |", flush=True)

            with capture_c_stdout():
                t0 = time.perf_counter()
                res_mp3 = mp3_cls(scf_res, res_mp2, mp2_conf, ints).compute()
                t_mp3 = time.perf_counter() - t0
            print(f"| {sys_name:<8} | {mode:<9} | {ref_lbl+'MP3':<6} | {res_mp3.energy_total:<16.8f} | {t_mp3:<9.3f} | {'-':<10} | {s_ideal:<6.2f} | {'-':<6} | {'OK':<6} |", flush=True)

            need_trace = (sys_name, mode) in TRACE_CASES
            omp_conf = mshqc.MP2Config(
                eri_method=mode_lower, threshold=1e-9, gradient_threshold=1e-5,
                max_iterations=50, aux_basis_name="cc-pVTZ-RI", print_level=1 if need_trace else 0
            )

            with capture_c_stdout() as buf_o2:
                t0 = time.perf_counter()
                res_omp2 = mshqc.OMP2(mol, basis, ints, scf_res, omp_conf).compute()
                t_omp2 = time.perf_counter() - t0
            with capture_c_stdout() as buf_o3:
                t0 = time.perf_counter()
                res_omp3 = mshqc.OMP3(mol, basis, ints, scf_res, omp_conf).compute()
                t_omp3 = time.perf_counter() - t0

            if need_trace:
                traces[f"{sys_name}|{mode}"] = buf_o2[0] + buf_o3[0]

            _, s2_omp2 = compute_spin_s2(ints, res_omp2, na, nb)
            _, s2_omp3 = compute_spin_s2(ints, res_omp3, na, nb)
            print(f"| {sys_name:<8} | {mode:<9} | {'OMP2':<6} | {res_omp2.energy_total:<16.8f} | {t_omp2:<9.3f} | {s2_omp2:<10.6f} | {s_ideal:<6.2f} | {int(res_omp2.iterations):<6} | {'OK':<6} |", flush=True)
            print(f"| {sys_name:<8} | {mode:<9} | {'OMP3':<6} | {res_omp3.energy_total:<16.8f} | {t_omp3:<9.3f} | {s2_omp3:<10.6f} | {s_ideal:<6.2f} | {int(res_omp3.iterations):<6} | {'OK':<6} |", flush=True)

            tot_solver_s += (t_scf + t_mp2 + t_mp3 + t_omp2 + t_omp3)
            ref_suite[f"{sys_name}|{mode}"] = {
                "s_ideal": s_ideal,
                "SCF":  {"e": float(scf_res.energy_total),  "t": t_scf,  "s2": s2_uhf},
                "MP2":  {"e": float(res_mp2.energy_total),  "t": t_mp2},
                "MP3":  {"e": float(res_mp3.energy_total),  "t": t_mp3},
                "OMP2": {"e": float(res_omp2.energy_total), "t": t_omp2, "iter": int(res_omp2.iterations), "s2": s2_omp2},
                "OMP3": {"e": float(res_omp3.energy_total), "t": t_omp3, "iter": int(res_omp3.iterations), "s2": s2_omp3},
            }
            gc.collect()

    print("=" * 106, flush=True)
    print(f"  Total Waktu Solver MSHQC (105 Run): {tot_solver_s:.3f} s", flush=True)
    return ref_suite, traces


def compute_table1_and_table2():
    table1_rows, table2_rows = [], []
    mb = 1024.0 * 1024.0

    print("\n" + "=" * 112, flush=True)
    print("  [LIVE TEST 2/5] TABEL KOMPRESI TENSOR ERI AO & VEKTOR CHOLESKY AKTUAL (MSHQC, tau_CD = 1e-9)", flush=True)
    print("=" * 112, flush=True)
    print(f"| {'Sistem':<8} | {'Basis':<8} | {'N_bf':<5} | {'N_aux(DF)':<9} | {'M_chol':<7} | {'Exact (MB)':<11} | {'Chol (MB)':<10} | {'DF (MB)':<9} | {'Exact/Chol':<10} | {'Exact/DF':<9} |", flush=True)
    print(f"|{'-'*10}|{'-'*10}|{'-'*7}|{'-'*11}|{'-'*9}|{'-'*13}|{'-'*12}|{'-'*11}|{'-'*12}|{'-'*11}|", flush=True)

    for sys_name, b_name, aux_name in TABLE1_CASES:
        mol = build_mol(sys_name)
        basis = mshqc.BasisSet(b_name, mol)
        aux = mshqc.BasisSet(aux_name, mol)
        nbf, naux = basis.n_basis_functions(), aux.n_basis_functions()
        with capture_c_stdout():
            m_chol = mshqc.CholeskyERI(mol, basis, 1e-9).n_vectors()

        mem_exact = (8.0 * (nbf ** 4)) / mb
        mem_chol = (8.0 * (nbf ** 2) * m_chol) / mb
        mem_df = (8.0 * (nbf ** 2) * naux) / mb
        r_chol = mem_exact / mem_chol if m_chol > 0 else 0.0
        r_df = mem_exact / mem_df

        row1 = {
            "sys": sys_name, "basis": b_name, "nbf": nbf, "naux": naux, "m_chol": m_chol,
            "mem_exact": mem_exact, "mem_chol": mem_chol, "mem_df": mem_df,
            "r_chol": r_chol, "r_df": r_df,
        }
        table1_rows.append(row1)
        print(
            f"| {sys_name:<8} | {b_name:<8} | {nbf:<5} | {naux:<9} | {m_chol:<7} | "
            f"{mem_exact:<11.3f} | {mem_chol:<10.3f} | {mem_df:<9.3f} | "
            f"{r_chol:<8.2f} x | {r_df:<7.2f} x |",
            flush=True,
        )

        if sys_name not in ("Lithium", "Carbon"):
            n_elec, mult = mol.n_electrons(), mol.multiplicity()
            oa, ob = (n_elec + mult - 1) // 2, (n_elec - mult + 1) // 2
            va, vb = nbf - oa, nbf - ob
            if not SYSTEMS[sys_name]["is_uhf"]:
                mem_conv_v4 = 2.0 * 8.0 * (va ** 4) / mb
                n_pairs = oa * (oa + 1) // 2
                mem_mshqc = (8.0 * ((oa*oa + oa*va + va*va) * naux + 2.0 * oa*oa*va*va + 2.0 * n_pairs*va*va + oa*oa*oa*va)) / mb
            else:
                mem_conv_v4 = 2.0 * 8.0 * ((va ** 4) + (vb ** 4) + (va * va * vb * vb)) / mb
                mem_mshqc = (
                    8.0 * (
                        ((oa*oa + oa*va + va*va) + (ob*ob + ob*vb + vb*vb)) * naux
                        + 2.0 * ((oa*oa*va*va) + (ob*ob*vb*vb) + (oa*ob*va*vb))
                        + 2.0 * ((oa*ob*va*vb) + (oa*(oa-1)//2)*va*va)
                        + ((oa*oa*oa*va) + (ob*ob*ob*vb) + (oa*ob*oa*vb) + (oa*ob*ob*va))
                    )
                ) / mb
            table2_rows.append({
                "sys": sys_name, "basis": b_name, "oa": oa, "va": va, "naux": naux,
                "mem_conv_v4": mem_conv_v4, "mem_mshqc": mem_mshqc, "ratio": mem_conv_v4 / mem_mshqc,
            })

    print("\n" + "=" * 116, flush=True)
    print("  [LIVE TEST 3/5] TABEL ELIMINASI HAMBATAN MEMORI O(V^4) PADA OMP3: KONVENSIONAL vs MSHQC (DF MODE)", flush=True)
    print("=" * 116, flush=True)
    print(f"| {'Sistem':<7} | {'Basis':<8} | {'(O, V, N_aux)':<15} | {'Konvensional V_vvvv + Gamma_vvvv (MB)':<37} | {'MSHQC B_3c + T2 + Y_ooov (MB)':<30} | {'Reduksi Memori MO':<17} |", flush=True)
    print(f"|{'-'*9}|{'-'*10}|{'-'*17}|{'-'*39}|{'-'*32}|{'-'*19}|", flush=True)
    for r in table2_rows:
        ov_str = f"({r['oa']},{r['va']},{r['naux']})"
        print(
            f"| {r['sys']:<7} | {r['basis']:<8} | {ov_str:<15} | "
            f"{r['mem_conv_v4']:<37.3f} | {r['mem_mshqc']:<30.3f} | {r['ratio']:<15.2f} x |",
            flush=True,
        )

    return table1_rows, table2_rows


def compute_fd_logs(traces):
    print("\n" + "=" * 108, flush=True)
    print("  [LIVE TEST 4/5] JEJAK ITERASI TR-SOSCF & VERIFIKASI GRADIEN FINITE-DIFFERENCE (cc-pVTZ)", flush=True)
    print("=" * 108, flush=True)
    for sys_name, mode in [("H2O", "DF"), ("Oxygen", "DF")]:
        print(f"\n>>> Log Iterasi Lengkap TR-SOSCF: {sys_name} (cc-pVTZ, Mode={mode}) <<<", flush=True)
        sys.stdout.write(traces.get(f"{sys_name}|{mode}", ""))
        sys.stdout.flush()

    fd_logs = {}
    print("\n>>> Uji Verifikasi Finite-Difference (C++ print_level=3) <<<", flush=True)
    for sys_name in ["Neon", "H2O", "Carbon"]:
        mol = build_mol(sys_name)
        basis = mshqc.BasisSet("cc-pVTZ", mol)
        ints = mshqc.IntegralEngine(mol, basis)
        scf_cls = mshqc.UHF if SYSTEMS[sys_name]["is_uhf"] else mshqc.RHF
        with capture_c_stdout():
            scf_res = scf_cls(mol, basis, ints, mshqc.SCFConfig(eri_method="exact", threshold=1e-9)).compute()
        cfg_fd = mshqc.MP2Config(eri_method="exact", print_level=3, max_iterations=0)
        with capture_c_stdout() as buf:
            mshqc.OMP2(mol, basis, ints, scf_res, cfg_fd).compute()
            mshqc.OMP3(mol, basis, ints, scf_res, cfg_fd).compute()
        fd_logs[sys_name] = buf[0]
        print(f"\n[{sys_name}] Menguji FD C++ internal:", flush=True)
        sys.stdout.write(fd_logs[sys_name])
        sys.stdout.flush()
    return fd_logs


def compute_ram_benchmark():
    ram_results = {}
    script_path = os.path.abspath(__file__)
    print("\n" + "=" * 96, flush=True)
    print("  [LIVE TEST 5/5] PENGUKURAN PEAK RSS (VmHWM) & NET WORKING RAM TERKALIBRASI (MSHQC)", flush=True)
    print("=" * 96, flush=True)
    print(f"| {'Sistem':<8} | {'Basis':<8} | {'Mode ERI':<9} | {'Metode':<6} | {'Base HWM (MB)':<14} | {'Peak HWM (MB)':<14} | {'Net RAM (MB)':<13} |", flush=True)
    print(f"|{'-'*10}|{'-'*10}|{'-'*11}|{'-'*8}|{'-'*16}|{'-'*16}|{'-'*15}|", flush=True)

    for sys_name, b_name, aux_name, mode, method in RAM_CASES:
        key = f"{sys_name}|{b_name}|{mode}|{method}"
        payload = json.dumps({"sys": sys_name, "basis": b_name, "aux": aux_name, "mode": mode, "method": method})
        try:
            out = subprocess.check_output([sys.executable, script_path, "--ram-worker", payload], text=True)
            res = json.loads(out.strip().splitlines()[-1])
        except Exception:
            res = {"base_hwm": 0.0, "peak_hwm": 0.0, "net_ram": 0.0}
        ram_results[key] = res
        print(
            f"| {sys_name:<8} | {b_name:<8} | {mode:<9} | {method:<6} | "
            f"{res.get('base_hwm', 0.0):<14.2f} | {res.get('peak_hwm', 0.0):<14.2f} | {res.get('net_ram', 0.0):<13.2f} |",
            flush=True,
        )
    print("=" * 96, flush=True)
    return ram_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSHQC Bab IV Benchmark Runner (Real-Time UI & Clean API)")
    parser.add_argument("--out", default="mshqc_bab4_raw.json", help="Path output berkas JSON MSHQC")
    parser.add_argument("--log", default="mshqc_bab4_runner.log", help="Path berkas log teks real-time")
    parser.add_argument("--ram-worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.ram_worker is not None:
        run_ram_worker(args.ram_worker)
        sys.exit(0)

    logger = TeeLogger(args.log)
    sys.stdout = logger
    try:
        t_start = time.perf_counter()
        ref_suite, traces = compute_mshqc_suite()
        t1_rows, t2_rows  = compute_table1_and_table2()
        fd_logs           = compute_fd_logs(traces)
        ram_bench         = compute_ram_benchmark()
        wall_time         = time.perf_counter() - t_start

        atomic_write_text(args.out, json.dumps({
            "engine": "mshqc",
            "wall_time_s": wall_time,
            "ref_suite": ref_suite,
            "table1": t1_rows,
            "table2": t2_rows,
            "traces": traces,
            "fd_logs": fd_logs,
            "ram_bench": ram_bench,
        }, indent=2))

        print(f"\n[SELESAI] Data JSON MSHQC disimpan ke: {args.out} | Log Teks: {args.log} ({wall_time:.2f} s)", flush=True)
    finally:
        sys.stdout = logger.terminal
        log_content = logger.get_text()
        logger.close()
        atomic_write_text("mshqc_bab4_test.txt", log_content)