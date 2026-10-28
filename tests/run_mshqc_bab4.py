#!/usr/bin/env python3
"""
Simplified MSHQC Bab IV Benchmark Runner (Clean User-Friendly API)
Output: mshqc_bab4_raw.json
"""

import os
import gc
import sys
import time
import json
import argparse
import subprocess
from contextlib import contextmanager
import numpy as np
import mshqc

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


@contextmanager
def capture_c_stdout():
    sys.stdout.flush()
    old_fd = os.dup(1)
    r_fd, w_fd = os.pipe()
    buf = []
    try:
        os.dup2(w_fd, 1)
        os.close(w_fd)
        yield buf
    finally:
        sys.stdout.flush()
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


MSHQC_RAM_WORKER = r"""
import sys, gc, json, mshqc

def get_rss_and_hwm_mb():
    rss, hwm = 0.0, 0.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"): rss = int(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"): hwm = int(line.split()[1]) / 1024.0
    return rss, hwm

mol_w = mshqc.Molecule(0, 1); mol_w.add_atom("He", 0.0, 0.0, 0.0)
bas_w = mshqc.BasisSet("cc-pvdz", mol_w)
mshqc.IntegralEngine(mol_w, bas_w).compute_overlap()
del bas_w, mol_w
gc.collect()

base_rss, base_hwm = get_rss_and_hwm_mb()
p = json.loads(sys.argv[1])

mol = mshqc.Molecule(p["charge"], p["mult"])
for sym, x, y, z in p["atoms"]:
    mol.add_atom(sym, x, y, z, units="angstrom")

basis = mshqc.BasisSet(p["basis"], mol)
ints  = mshqc.IntegralEngine(mol, basis)

scf_conf = mshqc.SCFConfig(eri_method=p["mode"], threshold=1e-9, aux_basis_name=p["aux"])
mp2_conf = mshqc.MP2Config(eri_method=p["mode"], threshold=1e-9, gradient_threshold=1e-5, aux_basis_name=p["aux"])

scf_cls = mshqc.UHF if p["is_uhf"] else mshqc.RHF
scf_res = scf_cls(mol, basis, ints, scf_conf).compute()
gc.collect()

omp_cls = mshqc.OMP2 if p["method"] == "OMP2" else mshqc.OMP3
omp_cls(mol, basis, ints, scf_res, mp2_conf).compute()

curr_rss, peak_hwm = get_rss_and_hwm_mb()
print(json.dumps({"base_hwm": base_hwm, "peak_hwm": peak_hwm, "net_ram": max(peak_hwm - base_hwm, curr_rss - base_rss, 0.10)}))
"""


def compute_mshqc_suite():
    ref_suite, traces = {}, {}
    print("==========================================================================================================", flush=True)
    print("  MSHQC BAB IV BENCHMARK RUNNER | cc-pVTZ / cc-pVTZ-RI", flush=True)
    print("==========================================================================================================", flush=True)

    for sys_name in ALL_SYSTEM_ORDER:
        is_uhf  = SYSTEMS[sys_name]["is_uhf"]
        ref_lbl = "U" if is_uhf else "R"
        mol     = build_mol(sys_name)
        basis   = mshqc.BasisSet("cc-pVTZ", mol)
        ints    = mshqc.IntegralEngine(mol, basis)

        scf_cls = mshqc.UHF if is_uhf else mshqc.RHF
        mp2_cls = mshqc.UMP2 if is_uhf else mshqc.RMP2
        mp3_cls = mshqc.UMP3 if is_uhf else mshqc.RMP3

        for mode in ERI_MODE_ORDER:
            print(f"  [MSHQC] Running: {sys_name:<8} | Mode: {mode:<9} -> {ref_lbl}HF, {ref_lbl}MP2, {ref_lbl}MP3, OMP2, OMP3", flush=True)

            scf_conf = mshqc.SCFConfig(eri_method=mode, threshold=1e-9, max_iterations=200)
            mp2_conf = mshqc.MP2Config(eri_method=mode, threshold=1e-9, gradient_threshold=1e-5, max_iterations=50)

            t0 = time.perf_counter()
            scf_res = scf_cls(mol, basis, ints, scf_conf).compute()
            t_scf = time.perf_counter() - t0

            na, nb = scf_res.n_occ_alpha, scf_res.n_occ_beta
            s_ideal, s2_uhf = compute_spin_s2(ints, scf_res, na, nb)

            t0 = time.perf_counter()
            res_mp2 = mp2_cls(mol, basis, ints, scf_res, mp2_conf).compute()
            t_mp2 = time.perf_counter() - t0

            t0 = time.perf_counter()
            res_mp3 = mp3_cls(scf_res, res_mp2, mp2_conf, ints).compute()
            t_mp3 = time.perf_counter() - t0

            t0 = time.perf_counter()
            res_omp2 = mshqc.OMP2(mol, basis, ints, scf_res, mp2_conf).compute()
            t_omp2 = time.perf_counter() - t0
            _, s2_omp2 = compute_spin_s2(ints, res_omp2, na, nb)

            t0 = time.perf_counter()
            res_omp3 = mshqc.OMP3(mol, basis, ints, scf_res, mp2_conf).compute()
            t_omp3 = time.perf_counter() - t0
            _, s2_omp3 = compute_spin_s2(ints, res_omp3, na, nb)

            if (sys_name, mode) in (("H2O", "DF"), ("Oxygen", "DF")):
                cfg_tr = mshqc.MP2Config(eri_method=mode, threshold=1e-9, gradient_threshold=1e-5, print_level=1)
                with capture_c_stdout() as buf:
                    mshqc.OMP2(mol, basis, ints, scf_res, cfg_tr).compute()
                    mshqc.OMP3(mol, basis, ints, scf_res, cfg_tr).compute()
                traces[f"{sys_name}|{mode}"] = buf[0]

            ref_suite[f"{sys_name}|{mode}"] = {
                "s_ideal": s_ideal,
                "SCF":  {"e": float(scf_res.energy_total),  "t": t_scf,  "s2": s2_uhf},
                "MP2":  {"e": float(res_mp2.energy_total),  "t": t_mp2},
                "MP3":  {"e": float(res_mp3.energy_total),  "t": t_mp3},
                "OMP2": {"e": float(res_omp2.energy_total), "t": t_omp2, "iter": int(res_omp2.iterations), "s2": s2_omp2},
                "OMP3": {"e": float(res_omp3.energy_total), "t": t_omp3, "iter": int(res_omp3.iterations), "s2": s2_omp3},
            }
            gc.collect()

    return ref_suite, traces


def compute_table1_and_table2():
    table1_rows, table2_rows = [], []
    mb = 1024.0 * 1024.0

    for sys_name, b_name, aux_name in TABLE1_CASES:
        mol   = build_mol(sys_name)
        basis = mshqc.BasisSet(b_name, mol)
        aux   = mshqc.BasisSet(aux_name, mol)
        nbf, naux = basis.n_basis_functions(), aux.n_basis_functions()
        m_chol = mshqc.CholeskyERI(mol, basis, 1e-9).n_vectors()

        mem_exact = (8.0 * (nbf ** 4)) / mb
        mem_chol  = (8.0 * (nbf ** 2) * m_chol) / mb
        mem_df    = (8.0 * (nbf ** 2) * naux) / mb
        table1_rows.append({
            "sys": sys_name, "basis": b_name, "nbf": nbf, "naux": naux, "m_chol": m_chol,
            "mem_exact": mem_exact, "mem_chol": mem_chol, "mem_df": mem_df,
            "r_chol": mem_exact / mem_chol if m_chol > 0 else 0.0,
            "r_df": mem_exact / mem_df,
        })

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
                "mem_conv_v4": mem_conv_v4, "mem_mshqc": mem_mshqc, "ratio": mem_conv_v4 / mem_mshqc
            })

    return table1_rows, table2_rows


def compute_fd_logs():
    fd_logs = {}
    for sys_name in ["Neon", "H2O", "Carbon"]:
        mol     = build_mol(sys_name)
        basis   = mshqc.BasisSet("cc-pVTZ", mol)
        ints    = mshqc.IntegralEngine(mol, basis)
        scf_cls = mshqc.UHF if SYSTEMS[sys_name]["is_uhf"] else mshqc.RHF
        scf_res = scf_cls(mol, basis, ints, mshqc.SCFConfig(eri_method="exact", threshold=1e-9)).compute()
        cfg_fd  = mshqc.MP2Config(eri_method="exact", print_level=3, max_iterations=0)
        with capture_c_stdout() as buf:
            mshqc.OMP2(mol, basis, ints, scf_res, cfg_fd).compute()
            mshqc.OMP3(mol, basis, ints, scf_res, cfg_fd).compute()
        fd_logs[sys_name] = buf[0]
    return fd_logs


def compute_ram_benchmark():
    ram_results = {}
    for sys_name, b_name, aux_name, mode, method in RAM_CASES:
        key = f"{sys_name}|{b_name}|{mode}|{method}"
        payload = json.dumps({"sys": sys_name, "basis": b_name, "aux": aux_name, "mode": mode, "method": method, **SYSTEMS[sys_name]})
        try:
            out = subprocess.check_output([sys.executable, "-c", MSHQC_RAM_WORKER, payload], text=True)
            ram_results[key] = json.loads(out.strip().splitlines()[-1])
        except Exception:
            ram_results[key] = {"peak_hwm": 0.0, "net_ram": 0.0}
    return ram_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified MSHQC Bab IV Runner")
    parser.add_argument("--out", default="mshqc_bab4_raw.json")
    args = parser.parse_args()

    t_start = time.perf_counter()
    ref_suite, traces = compute_mshqc_suite()
    t1_rows, t2_rows  = compute_table1_and_table2()
    fd_logs           = compute_fd_logs()
    ram_bench         = compute_ram_benchmark()
    wall_time         = time.perf_counter() - t_start

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "engine": "mshqc",
            "wall_time_s": wall_time,
            "ref_suite": ref_suite,
            "table1": t1_rows,
            "table2": t2_rows,
            "traces": traces,
            "fd_logs": fd_logs,
            "ram_bench": ram_bench,
        }, f, indent=2)

    print(f"\n[SELESAI] Data benchmark MSHQC Bab IV disimpan ke: {args.out} ({wall_time:.2f} s)", flush=True)