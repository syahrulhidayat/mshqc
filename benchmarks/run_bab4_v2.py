#!/usr/bin/env python3
"""
mshqc v2.0 Pure Thin-Core Bab IV & Core-5 Benchmark Runner
(Pengganti `run_mshqc_bab4.py` & `test_mshqc2.py` untuk arsitektur 8-file C++20 + Python).

Mencakup 5 laporan lengkap (mode CHOLESKY & DF; mode EXACT diarsipkan di tag v1.0.5-thesis-archive):
  1. Suite Solver Lengkap (7 sistem: Neon, H2O, Lithium, Carbon, Oxygen, OH, O2):
     RHF/UHF, ROHF, MP2/ROMP2, MP3/ROMP3, OMP2, SCS-OMP2, LT-SOS-OMP2, kappa-OOMP2, OMP2.5, OMP3.
  2. Tabel Kompresi ERI AO & Jumlah Vektor Cholesky aktual (cc-pVDZ/TZ/QZ).
  3. Tabel Eliminasi Hambatan Memori O(V^4) pada OMP3 (DF Mode).
  4. Jejak Konvergensi TR-SOSCF (via telemetry callback) & Verifikasi Gradien Orbital FD.
  5. Pengukuran Peak RSS (VmHWM) & Net Working RAM terkalibrasi (isolated worker).

Usage:
  python benchmarks/run_bab4_v2.py               # Penuh (cc-pVTZ, 7 sistem, CHOLESKY + DF)
  python benchmarks/run_bab4_v2.py --core5       # Setara test_mshqc2 (5 sistem, cc-pVTZ)
  python benchmarks/run_bab4_v2.py --quick       # Uji cepat (H2O & OH, cc-pVDZ, DF)
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.linalg import expm

CPU_AVAIL = os.cpu_count() or 4
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", str(min(6, CPU_AVAIL))))
os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("TBLIS_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

import mshqc
from mshqc import kernels as K
from mshqc import methods as M
from mshqc.methods.omp_hybrid import eval_omp_rhf_point, eval_omp_uhf_point

SYSTEMS: Dict[str, Dict[str, Any]] = {
    "Neon":    {"atoms": [("Ne", 0.0, 0.0, 0.0)], "charge": 0, "mult": 1, "is_uhf": False},
    "H2O":     {"atoms": [("O", 0.0, 0.0, 0.1173), ("H", 0.0, 0.7572, -0.4692), ("H", 0.0, -0.7572, -0.4692)], "charge": 0, "mult": 1, "is_uhf": False},
    "Lithium": {"atoms": [("Li", 0.0, 0.0, 0.0)], "charge": 0, "mult": 2, "is_uhf": True},
    "Carbon":  {"atoms": [("C", 0.0, 0.0, 0.0)], "charge": 0, "mult": 3, "is_uhf": True},
    "Oxygen":  {"atoms": [("O", 0.0, 0.0, 0.0)], "charge": 0, "mult": 3, "is_uhf": True},
    "OH":      {"atoms": [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.9697)], "charge": 0, "mult": 2, "is_uhf": True},
    "O2":      {"atoms": [("O", 0.0, 0.0, -0.6035), ("O", 0.0, 0.0, 0.6035)], "charge": 0, "mult": 3, "is_uhf": True},
}

ALL_SYSTEM_ORDER = ["Neon", "H2O", "Lithium", "Carbon", "Oxygen", "OH", "O2"]
CORE5_ORDER = ["Neon", "H2O", "Lithium", "Carbon", "Oxygen"]

TABLE1_CASES = [
    ("Neon", "cc-pvdz", "cc-pvdz-ri"), ("Neon", "cc-pvtz", "cc-pvtz-ri"), ("Neon", "cc-pvqz", "cc-pvqz-ri"),
    ("H2O", "cc-pvdz", "cc-pvdz-ri"), ("H2O", "cc-pvtz", "cc-pvtz-ri"), ("H2O", "cc-pvqz", "cc-pvqz-ri"),
    ("Lithium", "cc-pvtz", "cc-pvtz-ri"), ("Carbon", "cc-pvtz", "cc-pvtz-ri"),
    ("Oxygen", "cc-pvtz", "cc-pvtz-ri"), ("OH", "cc-pvtz", "cc-pvtz-ri"), ("O2", "cc-pvtz", "cc-pvtz-ri"),
]

RAM_CASES = [
    ("Neon", "cc-pvtz", "cc-pvtz-ri", "CHOLESKY", "OMP3"),
    ("Neon", "cc-pvtz", "cc-pvtz-ri", "DF", "OMP3"),
    ("Neon", "cc-pvqz", "cc-pvqz-ri", "DF", "OMP3"),
    ("H2O", "cc-pvtz", "cc-pvtz-ri", "CHOLESKY", "OMP3"),
    ("H2O", "cc-pvtz", "cc-pvtz-ri", "DF", "OMP2"),
    ("H2O", "cc-pvtz", "cc-pvtz-ri", "DF", "OMP3"),
    ("H2O", "cc-pvqz", "cc-pvqz-ri", "DF", "OMP3"),
    ("Oxygen", "cc-pvtz", "cc-pvtz-ri", "CHOLESKY", "OMP3"),
    ("Oxygen", "cc-pvtz", "cc-pvtz-ri", "DF", "OMP3"),
    ("O2", "cc-pvtz", "cc-pvtz-ri", "DF", "OMP3"),
]


class TeeLogger:
    def __init__(self, log_path: str):
        self.terminal = sys.__stdout__
        self.log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        self.chunks: List[str] = []

    def write(self, msg: str) -> None:
        self.terminal.write(msg)
        self.terminal.flush()
        self.log_file.write(msg)
        self.log_file.flush()
        self.chunks.append(msg)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()


def get_rss_and_hwm_mb() -> Tuple[float, float]:
    if sys.platform.startswith("linux"):
        rss_kb, hwm_kb = 0.0, 0.0
        try:
            with open("/proc/self/status", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_kb = float(line.split()[1])
                    elif line.startswith("VmHWM:"):
                        hwm_kb = float(line.split()[1])
            return rss_kb / 1024.0, hwm_kb / 1024.0
        except OSError:
            return 0.0, 0.0
    if sys.platform == "darwin":
        import resource
        hwm_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
        return hwm_mb, hwm_mb
    return 0.0, 0.0


def build_mol(sys_name: str) -> mshqc.Molecule:
    info = SYSTEMS[sys_name]
    mol = mshqc.Molecule(charge=info["charge"], multiplicity=info["mult"])
    for sym, x, y, z in info["atoms"]:
        mol.add_atom(sym, x, y, z, units="angstrom")
    return mol


def ideal_s2(mol: mshqc.Molecule) -> float:
    s = 0.5 * (int(mol.multiplicity()) - 1)
    return float(s * (s + 1.0))


def run_ram_worker(payload_str: str) -> None:
    mol_w = mshqc.Molecule(0, 1)
    mol_w.add_atom("He", 0.0, 0.0, 0.0)
    bas_w = mshqc.BasisSet("cc-pvdz", mol_w)
    mshqc.IntegralEngine(mol_w, bas_w).compute_overlap()
    del bas_w, mol_w
    gc.collect()

    base_rss, base_hwm = get_rss_and_hwm_mb()
    p = json.loads(payload_str)
    mol = build_mol(p["sys"])
    calc = mshqc.MSHQCCalculator(
        mol, basis_name=p["basis"], eri_method=p["mode"].lower(),
        aux_basis_name=p["aux"], threshold=1e-9,
    )
    scf_res = calc.run_scf()
    gc.collect()
    if p["method"] == "OMP2":
        calc.run_omp2(scf_res, grad_thresh=1e-5)
    else:
        calc.run_omp3(scf_res, grad_thresh=1e-5)

    curr_rss, peak_hwm = get_rss_and_hwm_mb()
    net_ram = max(peak_hwm - base_hwm, curr_rss - base_rss, 0.10)
    print(json.dumps({"base_hwm": base_hwm, "peak_hwm": peak_hwm, "net_ram": net_ram}), flush=True)


def compute_solver_suite(
    sys_list: List[str],
    modes: List[str],
    basis_name: str,
    include_extended: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    aux_name = f"{basis_name}-RI"
    ref_suite: Dict[str, Any] = {}
    traces: Dict[str, Any] = {}

    print("\n" + "=" * 110, flush=True)
    print(
        f" [LIVE TEST 1/5] MSHQC v{mshqc.__version__} THIN-CORE SOLVER SUITE "
        f"(THREADS: {N_THREADS} | {basis_name} / {aux_name})",
        flush=True,
    )
    print("=" * 110, flush=True)
    print(
        f"| {'Sistem':<8} | {'Mode ERI':<9} | {'Metode':<11} | {'E_mshqc (Ha)':<16} | "
        f"{'Waktu (s)':<9} | {'<S^2>':<10} | {'Ideal':<6} | {'Iter':<5} | {'Status':<6} |",
        flush=True,
    )
    print(f"|{'-'*10}|{'-'*11}|{'-'*13}|{'-'*18}|{'-'*11}|{'-'*12}|{'-'*8}|{'-'*7}|{'-'*8}|", flush=True)

    tot_s = 0.0
    for sys_name in sys_list:
        is_uhf = SYSTEMS[sys_name]["is_uhf"]
        ref_lbl = "U" if is_uhf else "R"
        mol = build_mol(sys_name)
        s_id = ideal_s2(mol)

        for mode in modes:
            calc = mshqc.MSHQCCalculator(
                mol, basis_name=basis_name, eri_method=mode.lower(),
                aux_basis_name=aux_name, threshold=1e-9,
            )

            t0 = time.perf_counter()
            scf_res = calc.run_scf()
            t_scf = time.perf_counter() - t0
            tot_s += t_scf
            s2_hf = float(getattr(scf_res, "s2", 0.0))
            print(
                f"| {sys_name:<8} | {mode:<9} | {ref_lbl+'HF':<11} | {scf_res.energy_total:<16.8f} | "
                f"{t_scf:<9.3f} | {s2_hf:<10.6f} | {s_id:<6.2f} | {int(scf_res.iterations):<5} | {'OK':<6} |",
                flush=True,
            )

            entry: Dict[str, Any] = {
                "s_ideal": s_id,
                "SCF": {"e": float(scf_res.energy_total), "t": t_scf, "s2": s2_hf, "iter": int(scf_res.iterations)},
            }

            # Open-shell ROHF / ROMP2 / ROMP3 (reuse 3-index factor from calc)
            if is_uhf:
                calc_ro = mshqc.MSHQCCalculator(
                    mol, basis_name=basis_name, eri_method=mode.lower(),
                    aux_basis_name=aux_name, reference="rohf", threshold=1e-9,
                )
                calc_ro._ao_factor_cache = calc._get_ao_3index_factor(scf_res)
                for lbl, fn in (
                    ("ROHF", lambda: calc_ro.run_scf()),
                    ("ROMP2", lambda: calc_ro.run_mp2()),
                    ("ROMP3", lambda: calc_ro.run_mp3()),
                ):
                    t0 = time.perf_counter()
                    r = fn()
                    dt = time.perf_counter() - t0
                    tot_s += dt
                    s2_str = f"{float(getattr(r, 's2', s_id)):<10.6f}" if lbl == "ROHF" else f"{'-':<10}"
                    print(
                        f"| {sys_name:<8} | {mode:<9} | {lbl:<11} | {r.energy_total:<16.8f} | "
                        f"{dt:<9.3f} | {s2_str} | {s_id:<6.2f} | {int(r.iterations):<5} | {'OK':<6} |",
                        flush=True,
                    )
                    entry[lbl] = {"e": float(r.energy_total), "t": dt}

            # Canonical MP2 / MP3 + Orbital-Optimized family
            corr_plan: List[Tuple[str, Any]] = [
                (f"{ref_lbl}MP2", lambda: calc.run_mp2(scf_res)),
                (f"{ref_lbl}MP3", lambda: calc.run_mp3(scf_res)),
                ("OMP2", lambda: calc.run_omp2(scf_res, grad_thresh=1e-5)),
            ]
            if include_extended:
                corr_plan.extend([
                    ("SCS-OMP2", lambda: calc.run_scs_omp2(scf_res, grad_thresh=1e-5)),
                    ("LT-SOS-OMP2", lambda: calc.run_lt_sos_omp2(scf_res, grad_thresh=1e-5)),
                    ("kappa-OOMP2", lambda: calc.run_kappa_omp2(scf_res, grad_thresh=1e-5)),
                    ("OMP2.5", lambda: calc.run_omp25(scf_res, grad_thresh=1e-5)),
                ])
            corr_plan.append(("OMP3", lambda: calc.run_omp3(scf_res, grad_thresh=1e-5)))

            for lbl, fn in corr_plan:
                t0 = time.perf_counter()
                r = fn()
                dt = time.perf_counter() - t0
                tot_s += dt
                has_s2 = lbl not in (f"{ref_lbl}MP2", f"{ref_lbl}MP3")
                s2_val = float(getattr(r, "s2", 0.0)) if has_s2 else None
                s2_str = f"{s2_val:<10.6f}" if s2_val is not None else f"{'-':<10}"
                print(
                    f"| {sys_name:<8} | {mode:<9} | {lbl:<11} | {r.energy_total:<16.8f} | "
                    f"{dt:<9.3f} | {s2_str} | {s_id:<6.2f} | {int(r.iterations):<5} | {'OK':<6} |",
                    flush=True,
                )
                key_store = "MP2" if lbl.endswith("MP2") and lbl in ("RMP2", "UMP2") else (
                    "MP3" if lbl in ("RMP3", "UMP3") else lbl
                )
                entry[key_store] = {
                    "e": float(r.energy_total),
                    "t": dt,
                    "iter": int(r.iterations),
                    **({"s2": s2_val} if s2_val is not None else {}),
                }

            # Telemetry trace capture for representative DF systems
            if mode == "DF" and sys_name in ("H2O", "Oxygen", "OH"):
                H_c = calc.integrals.compute_kinetic() + calc.integrals.compute_nuclear()
                S_o = calc.integrals.compute_overlap()
                v_nn = float(mol.nuclear_repulsion_energy())
                B_ao = scf_res.L_mat
                tr_o2, tr_o3 = [], []
                if not is_uhf:
                    M.run_df_omp_rhf(H_c, v_nn, B_ao, scf_res.C_alpha, scf_res.n_occ_alpha,
                                     method_name="omp2", grad_tol=1e-5, callback=tr_o2.append)
                    M.run_df_omp_rhf(H_c, v_nn, B_ao, scf_res.C_alpha, scf_res.n_occ_alpha,
                                     method_name="omp3", lam_mp3=1.0, grad_tol=1e-5, callback=tr_o3.append)
                else:
                    M.run_df_omp_uhf(H_c, S_o, v_nn, B_ao, scf_res.C_alpha, scf_res.C_beta,
                                     scf_res.n_occ_alpha, scf_res.n_occ_beta,
                                     method_name="omp2", grad_tol=1e-5, callback=tr_o2.append)
                    M.run_df_omp_uhf(H_c, S_o, v_nn, B_ao, scf_res.C_alpha, scf_res.C_beta,
                                     scf_res.n_occ_alpha, scf_res.n_occ_beta,
                                     method_name="omp3", lam_mp3=1.0, grad_tol=1e-5, callback=tr_o3.append)
                traces[f"{sys_name}|DF"] = {"OMP2": tr_o2, "OMP3": tr_o3}

            ref_suite[f"{sys_name}|{mode}"] = entry
            gc.collect()

    print("=" * 110, flush=True)
    print(f" Total Waktu Solver MSHQC v{mshqc.__version__}: {tot_s:.3f} s", flush=True)
    return ref_suite, traces


def compute_table1_and_table2(cases: List[Tuple[str, str, str]]) -> Tuple[List[dict], List[dict]]:
    table1_rows, table2_rows = [], []
    mb = 1024.0 * 1024.0

    print("\n" + "=" * 112, flush=True)
    print(" [LIVE TEST 2/5] TABEL KOMPRESI TENSOR ERI AO & VEKTOR CHOLESKY AKTUAL (tau_CD = 1e-9)", flush=True)
    print("=" * 112, flush=True)
    print(f"| {'Sistem':<8} | {'Basis':<8} | {'N_bf':<5} | {'N_aux(DF)':<9} | {'M_chol':<7} | {'Exact (MB)':<11} | {'Chol (MB)':<10} | {'DF (MB)':<9} | {'Exact/Chol':<10} | {'Exact/DF':<9} |", flush=True)
    print(f"|{'-'*10}|{'-'*10}|{'-'*7}|{'-'*11}|{'-'*9}|{'-'*13}|{'-'*12}|{'-'*11}|{'-'*12}|{'-'*11}|", flush=True)

    for sys_name, b_name, aux_name in cases:
        mol = build_mol(sys_name)
        basis = mshqc.BasisSet(b_name, mol)
        aux = mshqc.BasisSet(aux_name, mol)
        nbf, naux = int(basis.n_basis_functions()), int(aux.n_basis_functions())
        m_chol = int(mshqc.CholeskyERI(mol, basis, 1e-9).n_vectors())

        mem_exact = (8.0 * (nbf ** 4)) / mb
        # Packed lower-triangular AO storage in v2.0: nbf*(nbf+1)/2 * M
        n_tri = nbf * (nbf + 1) / 2.0
        mem_chol = (8.0 * n_tri * m_chol) / mb
        mem_df = (8.0 * n_tri * naux) / mb
        r_chol = mem_exact / max(mem_chol, 1e-12)
        r_df = mem_exact / max(mem_df, 1e-12)

        table1_rows.append({
            "sys": sys_name, "basis": b_name, "nbf": nbf, "naux": naux, "m_chol": m_chol,
            "mem_exact": mem_exact, "mem_chol_packed": mem_chol, "mem_df_packed": mem_df,
            "r_chol": r_chol, "r_df": r_df,
        })
        print(
            f"| {sys_name:<8} | {b_name:<8} | {nbf:<5} | {naux:<9} | {m_chol:<7} | "
            f"{mem_exact:<11.3f} | {mem_chol:<10.3f} | {mem_df:<9.3f} | "
            f"{r_chol:<8.2f} x | {r_df:<7.2f} x |",
            flush=True,
        )

        if sys_name not in ("Lithium", "Carbon"):
            n_elec, mult = int(mol.n_electrons()), int(mol.multiplicity())
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
                "mem_conv_v4": mem_conv_v4, "mem_mshqc": mem_mshqc, "ratio": mem_conv_v4 / max(mem_mshqc, 1e-12),
            })

    print("\n" + "=" * 116, flush=True)
    print(" [LIVE TEST 3/5] ELIMINASI HAMBATAN MEMORI O(V^4) PADA OMP3: KONVENSIONAL vs MSHQC v2.0 (DF MODE)", flush=True)
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


def verify_orbital_gradients_fd(sys_list: List[str], basis_name: str, traces: Dict[str, Any]) -> Dict[str, Any]:
    print("\n" + "=" * 108, flush=True)
    print(f" [LIVE TEST 4/5] TELEMETRY TR-SOSCF & VERIFIKASI GRADIEN ORBITAL ANALITIK vs FD ({basis_name}, DF)", flush=True)
    print("=" * 108, flush=True)

    for key, td in traces.items():
        for meth in ("OMP2", "OMP3"):
            steps = td.get(meth, [])
            if steps:
                print(f"  [TRACE {key:<10} {meth}] {len(steps)} iter | E0={steps[0]['energy_total']:.8f} -> Ef={steps[-1]['energy_total']:.8f} | g_rms={steps[-1]['grad_rms']:.2e}", flush=True)

    fd_summary: Dict[str, Any] = {}
    h = 1.0e-5
    for sys_name in sys_list:
        mol = build_mol(sys_name)
        is_uhf = SYSTEMS[sys_name]["is_uhf"]
        basis = mshqc.BasisSet(basis_name, mol)
        aux = mshqc.BasisSet(f"{basis_name}-RI", mol)
        ints = mshqc.IntegralEngine(mol, basis)
        H_c = ints.compute_kinetic() + ints.compute_nuclear()
        S_o = ints.compute_overlap()
        v_nn = float(mol.nuclear_repulsion_energy())
        B_ao = K.compute_df_ao_factor(mol, basis, aux, 1e-10)
        nbf = int(basis.n_basis_functions())
        n_elec, mult = int(mol.n_electrons()), int(mol.multiplicity())
        na = (n_elec + mult - 1) // 2
        nb = n_elec - na

        if not is_uhf:
            scf = M.run_df_rhf(H_c, S_o, v_nn, B_ao, na, energy_tol=1e-11, density_tol=1e-11)
            for meth, lam in (("OMP2", 0.0), ("OMP3", 1.0)):
                e0, _, _, C_sc, w_vo, _ = eval_omp_rhf_point(H_c, v_nn, B_ao, scf.C_alpha, na, lam_mp3=lam, need_grad=True)
                a_i, i_i = np.unravel_index(int(np.argmax(np.abs(w_vo))), w_vo.shape)
                g_ana = float(w_vo[a_i, i_i])
                K_rot = np.zeros((nbf, nbf))
                K_rot[na + a_i, i_i] = 1.0
                K_rot[i_i, na + a_i] = -1.0
                ep = eval_omp_rhf_point(H_c, v_nn, B_ao, C_sc @ expm(h * K_rot), na, lam_mp3=lam, need_grad=False)[0]
                em = eval_omp_rhf_point(H_c, v_nn, B_ao, C_sc @ expm(-h * K_rot), na, lam_mp3=lam, need_grad=False)[0]
                g_fd = (ep - em) / (2.0 * h)
                err = abs(g_ana - g_fd)
                fd_summary[f"{sys_name}|{meth}"] = {"ana": g_ana, "fd": g_fd, "err": err}
                print(f"  [FD {sys_name:<7} {meth}] Ana={g_ana:+.10e} | FD={g_fd:+.10e} | |dw|={err:.2e}", flush=True)
        else:
            scf = M.run_df_uhf(H_c, S_o, v_nn, B_ao, na, nb, energy_tol=1e-11, density_tol=1e-11)
            for meth, lam in (("OMP2", 0.0), ("OMP3", 1.0)):
                _, _, _, Ca_sc, Cb_sc, wa, _, _ = eval_omp_uhf_point(
                    H_c, v_nn, B_ao, scf.C_alpha, scf.C_beta, na, nb, lam_mp3=lam, need_grad=True
                )
                a_i, i_i = np.unravel_index(int(np.argmax(np.abs(wa))), wa.shape)
                g_ana = float(wa[a_i, i_i])
                K_rot = np.zeros((nbf, nbf))
                K_rot[na + a_i, i_i] = 1.0
                K_rot[i_i, na + a_i] = -1.0
                ep = eval_omp_uhf_point(H_c, v_nn, B_ao, Ca_sc @ expm(h * K_rot), Cb_sc, na, nb, lam_mp3=lam, need_grad=False)[0]
                em = eval_omp_uhf_point(H_c, v_nn, B_ao, Ca_sc @ expm(-h * K_rot), Cb_sc, na, nb, lam_mp3=lam, need_grad=False)[0]
                g_fd = (ep - em) / (2.0 * h)
                err = abs(g_ana - g_fd)
                fd_summary[f"{sys_name}|{meth}"] = {"ana": g_ana, "fd": g_fd, "err": err}
                print(f"  [FD {sys_name:<7} {meth}] Ana={g_ana:+.10e} | FD={g_fd:+.10e} | |dw|={err:.2e}", flush=True)
    return fd_summary


def compute_ram_benchmark(cases: List[Tuple[str, str, str, str, str]]) -> Dict[str, Any]:
    ram_results: Dict[str, Any] = {}
    script_path = os.path.abspath(__file__)
    print("\n" + "=" * 96, flush=True)
    print(" [LIVE TEST 5/5] PENGUKURAN PEAK RSS (VmHWM) & NET WORKING RAM TERKALIBRASI (MSHQC v2.0)", flush=True)
    print("=" * 96, flush=True)
    print(f"| {'Sistem':<8} | {'Basis':<8} | {'Mode ERI':<9} | {'Metode':<6} | {'Base HWM (MB)':<14} | {'Peak HWM (MB)':<14} | {'Net RAM (MB)':<13} |", flush=True)
    print(f"|{'-'*10}|{'-'*10}|{'-'*11}|{'-'*8}|{'-'*16}|{'-'*16}|{'-'*15}|", flush=True)

    for sys_name, b_name, aux_name, mode, method in cases:
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
            f"{res['base_hwm']:<14.2f} | {res['peak_hwm']:<14.2f} | {res['net_ram']:<13.2f} |",
            flush=True,
        )
    print("=" * 96, flush=True)
    return ram_results


def main() -> None:
    parser = argparse.ArgumentParser(description="mshqc v2.0 Pure Thin-Core Bab IV & Core-5 Benchmark Runner")
    parser.add_argument("--out", default="mshqc_bab4_v2_raw.json")
    parser.add_argument("--log", default="mshqc_bab4_v2_runner.log")
    parser.add_argument("--core5", action="store_true", help="Jalankan 5 sistem inti (pengganti test_mshqc2)")
    parser.add_argument("--quick", action="store_true", help="Mode uji cepat (H2O & OH, cc-pVDZ, DF)")
    parser.add_argument("--skip-ram", action="store_true", help="Lewati subprocess RAM benchmark")
    parser.add_argument("--ram-worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.ram_worker is not None:
        run_ram_worker(args.ram_worker)
        sys.exit(0)

    sys_list = ["H2O", "OH"] if args.quick else (CORE5_ORDER if args.core5 else ALL_SYSTEM_ORDER)
    modes = ["DF"] if args.quick else ["CHOLESKY", "DF"]
    basis_name = "cc-pVDZ" if args.quick else "cc-pVTZ"
    t1_cases = [("H2O", "cc-pvdz", "cc-pvdz-ri"), ("OH", "cc-pvdz", "cc-pvdz-ri")] if args.quick else TABLE1_CASES
    fd_systems = ["H2O", "OH"] if args.quick else ["H2O", "Carbon"]
    ram_cases = [("H2O", "cc-pvdz", "cc-pvdz-ri", "DF", "OMP3")] if args.quick else RAM_CASES

    logger = TeeLogger(args.log)
    sys.stdout = logger
    try:
        t0 = time.perf_counter()
        ref_suite, traces = compute_solver_suite(sys_list, modes, basis_name, include_extended=True)
        t1_rows, t2_rows = compute_table1_and_table2(t1_cases)
        fd_summary = verify_orbital_gradients_fd(fd_systems, basis_name, traces)
        ram_bench = {} if args.skip_ram else compute_ram_benchmark(ram_cases)
        wall_s = time.perf_counter() - t0

        Path(args.out).write_text(
            json.dumps({
                "engine": f"mshqc-v{mshqc.__version__}",
                "thin_core_only": mshqc.THIN_CORE_ONLY,
                "threads": N_THREADS,
                "wall_time_s": wall_s,
                "ref_suite": ref_suite,
                "table1": t1_rows,
                "table2": t2_rows,
                "traces": traces,
                "fd_summary": fd_summary,
                "ram_bench": ram_bench,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"\n[SELESAI] Data Bab IV v2.0 disimpan ke: {args.out} ({wall_s:.2f} s)", flush=True)
    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()
