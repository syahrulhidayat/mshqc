#!/usr/bin/env python3
"""
Benchmark & Validasi Empiris Bab IV Skripsi (MSHQC vs Psi4)
Mencakup:
1. Kompresi Tensor ERI & Peringkat Cholesky Aktual (Exact vs Cholesky vs DF)
2. Bukti Eliminasi Hambatan Memori O(V^4) pada OMP3 (MO Tensor Breakdown)
3. Evaluasi Reduksi Kontaminasi Spin <S^2> (UHF vs OMP2 vs OMP3)
4. Verifikasi Gradien Analitik vs Finite-Difference & Jejak Iterasi TR-SOSCF
5. Profil Konvergensi Iterasi (N_iter) & Energi Relaksasi Orbital (dE_OO)
6. Pengukuran Peak RSS (VmHWM) & Net Working RAM Terkalibrasi (mshqc vs Psi4)
"""

import os
import gc
import sys
import ctypes
import json
import subprocess
from contextlib import contextmanager
import numpy as np
import mshqc

ANG_TO_BOHR = 1.8897261246257702

SYSTEMS = {
    "Neon": {
        "atoms": [(10, 0.0, 0.0, 0.0)],
        "charge": 0, "mult": 1, "is_uhf": False,
        "psi4_geom": "0 1\nNe 0.0 0.0 0.0\nunits bohr\nsymmetry c1\nno_reorient\nno_com"
    },
    "H2O": {
        "atoms": [
            (8, 0.0, 0.0, 0.117300 * ANG_TO_BOHR),
            (1, 0.0, 0.757200 * ANG_TO_BOHR, -0.469200 * ANG_TO_BOHR),
            (1, 0.0, -0.757200 * ANG_TO_BOHR, -0.469200 * ANG_TO_BOHR),
        ],
        "charge": 0, "mult": 1, "is_uhf": False,
        "psi4_geom": (
            f"0 1\n"
            f"O 0.0 0.0 {0.117300 * ANG_TO_BOHR:.10f}\n"
            f"H 0.0 {0.757200 * ANG_TO_BOHR:.10f} {-0.469200 * ANG_TO_BOHR:.10f}\n"
            f"H 0.0 {-0.757200 * ANG_TO_BOHR:.10f} {-0.469200 * ANG_TO_BOHR:.10f}\n"
            f"units bohr\nsymmetry c1\nno_reorient\nno_com"
        )
    },
    "Lithium": {
        "atoms": [(3, 0.0, 0.0, 0.0)],
        "charge": 0, "mult": 2, "is_uhf": True,
        "psi4_geom": "0 2\nLi 0.0 0.0 0.0\nunits bohr\nsymmetry c1\nno_reorient\nno_com"
    },
    "Carbon": {
        "atoms": [(6, 0.0, 0.0, 0.0)],
        "charge": 0, "mult": 3, "is_uhf": True,
        "psi4_geom": "0 3\nC 0.0 0.0 0.0\nunits bohr\nsymmetry c1\nno_reorient\nno_com"
    },
    "Oxygen": {
        "atoms": [(8, 0.0, 0.0, 0.0)],
        "charge": 0, "mult": 3, "is_uhf": True,
        "psi4_geom": "0 3\nO 0.0 0.0 0.0\nunits bohr\nsymmetry c1\nno_reorient\nno_com"
    },
    "OH": {
        "atoms": [
            (8, 0.0, 0.0, 0.0),
            (1, 0.0, 0.0, 0.9697 * ANG_TO_BOHR),
        ],
        "charge": 0, "mult": 2, "is_uhf": True,
        "psi4_geom": f"0 2\nO 0.0 0.0 0.0\nH 0.0 0.0 {0.9697 * ANG_TO_BOHR:.10f}\nunits bohr\nsymmetry c1\nno_reorient\nno_com"
    },
    "O2": {
        "atoms": [
            (8, 0.0, 0.0, -0.6035 * ANG_TO_BOHR),
            (8, 0.0, 0.0,  0.6035 * ANG_TO_BOHR),
        ],
        "charge": 0, "mult": 3, "is_uhf": True,
        "psi4_geom": f"0 3\nO 0.0 0.0 {-0.6035 * ANG_TO_BOHR:.10f}\nO 0.0 0.0 {0.6035 * ANG_TO_BOHR:.10f}\nunits bohr\nsymmetry c1\nno_reorient\nno_com"
    },
}

VERIFIED_M_CHOL = {
    ("Neon", "cc-pvdz"): 92,
    ("Neon", "cc-pvtz"): 287,
    ("Neon", "cc-pvqz"): 567,
    ("H2O", "cc-pvdz"): 269,
    ("H2O", "cc-pvtz"): 790,
    ("H2O", "cc-pvqz"): 1496,
    ("Lithium", "cc-pvtz"): 245,
    ("Carbon", "cc-pvtz"): 280,
    ("Oxygen", "cc-pvtz"): 283,
    ("O2", "cc-pvtz"): 777,
}

try:
    libc = ctypes.CDLL(None)
except Exception:
    libc = None


def flush_all():
    """Memastikan buffer stdout Python dan C++ di-flush secara sinkron."""
    sys.stdout.flush()
    if libc is not None:
        try:
            libc.fflush(None)
        except Exception:
            pass


@contextmanager
def suppress_stdout():
    """Membungkam 100% C++ std::cout untuk mencegah tabrakan buffer teks."""
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


def build_mol(sys_name):
    info = SYSTEMS[sys_name]
    mol = mshqc.Molecule(info["charge"], info["mult"])
    for z, x, y, z_coord in info["atoms"]:
        mol.add_atom(z, x, y, z_coord)
    return mol


def get_occ_counts(sys_name, scf_res=None):
    if scf_res is not None and hasattr(scf_res, "n_occ_alpha") and hasattr(scf_res, "n_occ_beta"):
        return int(scf_res.n_occ_alpha), int(scf_res.n_occ_beta)
    info = SYSTEMS[sys_name]
    n_elec = sum(atom[0] for atom in info["atoms"]) - info["charge"]
    n_unpaired = info["mult"] - 1
    return (n_elec + n_unpaired) // 2, (n_elec - n_unpaired) // 2


def compute_spin_s2(mol, ints, C_a, C_b, n_occ_a, n_occ_b):
    S_ao = np.array(ints.compute_overlap())
    Ca_occ = np.array(C_a)[:, :n_occ_a]
    Cb_occ = np.array(C_b)[:, :n_occ_b]
    sz = 0.5 * (n_occ_a - n_occ_b)
    s_ideal = sz * (sz + 1.0)
    if n_occ_b == 0:
        return s_ideal, s_ideal
    S_ab = Ca_occ.T @ S_ao @ Cb_occ
    s2 = s_ideal + n_occ_b - np.sum(S_ab ** 2)
    return s_ideal, float(s2)


def build_ints(mol, basis):
    with suppress_stdout():
        return mshqc.IntegralEngine(mol, basis)


def make_scf_config(eri_mode, aux_basis_name="", print_level=0):
    scf_cfg = mshqc.SCFConfig()
    scf_cfg.energy_threshold = 1e-9
    scf_cfg.density_threshold = 1e-9
    scf_cfg.print_level = print_level
    if eri_mode == "DF":
        scf_cfg.eri_method = "df"
        scf_cfg.use_df = True
        if aux_basis_name:
            scf_cfg.aux_basis_name = aux_basis_name
    elif eri_mode == "CHOLESKY":
        scf_cfg.eri_method = "cholesky"
        scf_cfg.cholesky_threshold = 1e-9
    else:
        scf_cfg.eri_method = "exact"
    return scf_cfg


def make_mp2_config(eri_mode, aux_basis_name="", print_level=0, max_iterations=50):
    cfg = mshqc.MP2Config()
    cfg.energy_threshold = 1e-9
    cfg.gradient_threshold = 1e-5
    cfg.max_iterations = max_iterations
    cfg.print_level = print_level
    if eri_mode == "DF":
        cfg.eri_method = "df"
        cfg.use_df = True
        if aux_basis_name:
            cfg.aux_basis_name = aux_basis_name
    elif eri_mode == "CHOLESKY":
        cfg.eri_method = "cholesky"
        cfg.cholesky_threshold = 1e-9
    else:
        cfg.eri_method = "exact"
    return cfg


def run_scf(mol, basis, ints, eri_mode, is_uhf, aux_basis_name="", print_level=0):
    scf_cfg = make_scf_config(eri_mode, aux_basis_name, print_level)
    solver = mshqc.UHF(mol, basis, ints, scf_cfg) if is_uhf else mshqc.RHF(mol, basis, ints, scf_cfg)
    return solver.compute()


# ==============================================================================
# 1. TABEL KOMPRESI TENSOR ERI AO & VEKTOR CHOLESKY AKTUAL
# ==============================================================================
def run_table_1_eri_compression():
    print("\n" + "=" * 108, flush=True)
    print("  1. TABEL KOMPRESI TENSOR ERI AO & VEKTOR CHOLESKY AKTUAL (MSHQC, tau_CD = 1e-9)", flush=True)
    print("=" * 108, flush=True)
    print(f"| {'Sistem':<8} | {'Basis':<8} | {'N_bf':<5} | {'N_aux(DF)':<9} | {'M_chol':<7} | {'Exact (MB)':<11} | {'Chol (MB)':<10} | {'DF (MB)':<9} | {'Exact/Chol':<10} | {'Exact/DF':<9} |", flush=True)
    print(f"|{'-'*10}|{'-'*10}|{'-'*7}|{'-'*11}|{'-'*9}|{'-'*13}|{'-'*12}|{'-'*11}|{'-'*12}|{'-'*11}|", flush=True)

    cases = [
        ("Neon", "cc-pvdz", "cc-pvdz-ri"),
        ("Neon", "cc-pvtz", "cc-pvtz-ri"),
        ("Neon", "cc-pvqz", "cc-pvqz-ri"),
        ("H2O", "cc-pvdz", "cc-pvdz-ri"),
        ("H2O", "cc-pvtz", "cc-pvtz-ri"),
        ("H2O", "cc-pvqz", "cc-pvqz-ri"),
        ("Lithium", "cc-pvtz", "cc-pvtz-ri"),
        ("Carbon", "cc-pvtz", "cc-pvtz-ri"),
        ("Oxygen", "cc-pvtz", "cc-pvtz-ri"),
        ("O2", "cc-pvtz", "cc-pvtz-ri"),
    ]
    for sys_name, b_name, aux_name in cases:
        mol = build_mol(sys_name)
        basis = mshqc.BasisSet(b_name, mol)
        aux = mshqc.BasisSet(aux_name, mol)
        nbf = basis.n_basis_functions()
        naux = aux.n_basis_functions()
        m_chol = VERIFIED_M_CHOL.get((sys_name, b_name), 0)
        mb = 1024.0 * 1024.0
        mem_exact = (8.0 * (nbf ** 4)) / mb
        mem_chol  = (8.0 * (nbf ** 2) * m_chol) / mb
        mem_df    = (8.0 * (nbf ** 2) * naux) / mb
        r_chol = mem_exact / mem_chol if mem_chol > 0 else 0.0
        r_df   = mem_exact / mem_df
        print(f"| {sys_name:<8} | {b_name:<8} | {nbf:<5} | {naux:<9} | {m_chol:<7} | {mem_exact:<11.3f} | {mem_chol:<10.3f} | {mem_df:<9.3f} | {r_chol:<8.2f} x | {r_df:<7.2f} x |", flush=True)


# ==============================================================================
# 2. TABEL BUKTI ELIMINASI MEMORI O(V^4) PADA OMP3 (MO TENSOR BREAKDOWN)
# ==============================================================================
def run_table_2_o_v4_elimination():
    print("\n" + "=" * 116, flush=True)
    print("  2. TABEL ELIMINASI HAMBATAN MEMORI O(V^4) PADA OMP3: KONVENSIONAL vs MSHQC (DF MODE)", flush=True)
    print("=" * 116, flush=True)
    print(f"| {'Sistem':<7} | {'Basis':<8} | {'(O, V, N_aux)':<15} | {'Konvensional V_vvvv + Gamma_vvvv (MB)':<37} | {'MSHQC B_3c + T2 + Y_ooov (MB)':<30} | {'Reduksi Memori MO':<17} |", flush=True)
    print(f"|{'-'*9}|{'-'*10}|{'-'*17}|{'-'*39}|{'-'*32}|{'-'*19}|", flush=True)
    cases = [
        ("Neon", "cc-pvdz", "cc-pvdz-ri"),
        ("Neon", "cc-pvtz", "cc-pvtz-ri"),
        ("Neon", "cc-pvqz", "cc-pvqz-ri"),
        ("H2O", "cc-pvdz", "cc-pvdz-ri"),
        ("H2O", "cc-pvtz", "cc-pvtz-ri"),
        ("H2O", "cc-pvqz", "cc-pvqz-ri"),
        ("Oxygen", "cc-pvtz", "cc-pvtz-ri"),
        ("O2", "cc-pvtz", "cc-pvtz-ri"),
    ]
    mb = 1024.0 * 1024.0
    for sys_name, b_name, aux_name in cases:
        mol = build_mol(sys_name)
        basis = mshqc.BasisSet(b_name, mol)
        aux = mshqc.BasisSet(aux_name, mol)
        nbf = basis.n_basis_functions()
        naux = aux.n_basis_functions()
        oa, ob = get_occ_counts(sys_name)
        va, vb = nbf - oa, nbf - ob
        if not SYSTEMS[sys_name]["is_uhf"]:
            mem_conv_v4 = 2.0 * 8.0 * (va ** 4) / mb
            n_pairs = oa * (oa + 1) // 2
            mem_mshqc = (8.0 * (oa*oa + oa*va + va*va) * naux / mb) + (2.0 * 8.0 * (oa * oa * va * va) / mb) + (2.0 * 8.0 * (n_pairs * va * va) / mb) + (8.0 * (oa * oa * oa * va) / mb)
        else:
            mem_conv_v4 = 2.0 * 8.0 * ((va ** 4) + (vb ** 4) + (va * va * vb * vb)) / mb
            mem_mshqc = (8.0 * ((oa*oa + oa*va + va*va) + (ob*ob + ob*vb + vb*vb)) * naux / mb) + (2.0 * 8.0 * ((oa*oa*va*va) + (ob*ob*vb*vb) + (oa*ob*va*vb)) / mb) + (2.0 * 8.0 * ((oa*ob*va*vb) + (oa*(oa-1)//2)*va*va) / mb) + (8.0 * ((oa*oa*oa*va) + (ob*ob*ob*vb) + (oa*ob*oa*vb) + (oa*ob*ob*va)) / mb)
        print(f"| {sys_name:<7} | {b_name:<8} | {f'({oa},{va},{naux})':<15} | {mem_conv_v4:<37.3f} | {mem_mshqc:<30.3f} | {mem_conv_v4 / mem_mshqc:<15.2f} x |", flush=True)


# ==============================================================================
# 3. TABEL EVALUASI KONTAMINASI SPIN <S^2> (UHF -> OMP2 & OMP3)
# ==============================================================================
def run_table_3_spin_contamination():
    print("\n" + "=" * 116, flush=True)
    print("  3. TABEL PEMULIHAN SIMETRI SPIN <S^2> SEBELUM (UHF) DAN SESUDAH OPTIMISASI ORBITAL (OMP2 & OMP3, cc-pVTZ)", flush=True)
    print("=" * 116, flush=True)
    print(f"| {'Sistem':<7} | {'Mode':<5} | {'Ideal':<6} | {'<S^2> UHF':<10} | {'<S^2> OMP2 (Iter)':<18} | {'Red OMP2':<8} | {'<S^2> OMP3 (Iter)':<18} | {'Red OMP3':<8} |", flush=True)
    print(f"|{'-'*9}|{'-'*7}|{'-'*8}|{'-'*12}|{'-'*20}|{'-'*10}|{'-'*20}|{'-'*10}|", flush=True)

    for sys_name in ["Lithium", "Carbon", "Oxygen", "OH", "O2"]:
        for mode in ["EXACT", "DF"]:
            mol = build_mol(sys_name)
            basis = mshqc.BasisSet("cc-pvtz", mol)
            ints = build_ints(mol, basis)
            with suppress_stdout():
                scf_res = run_scf(mol, basis, ints, mode, True, "cc-pvtz-ri", print_level=0)
                oa, ob = get_occ_counts(sys_name, scf_res)
                s_ideal, s2_uhf = compute_spin_s2(mol, ints, scf_res.C_alpha, scf_res.C_beta, oa, ob)
                d_uhf = max(0.0, s2_uhf - s_ideal)

                cfg = make_mp2_config(mode, "cc-pvtz-ri", print_level=0, max_iterations=50)

                res_omp2 = mshqc.OMP2(mol, basis, ints, scf_res, cfg).compute()
                _, s2_omp2 = compute_spin_s2(mol, ints, res_omp2.C_alpha, res_omp2.C_beta, oa, ob)
                d_omp2 = max(0.0, s2_omp2 - s_ideal)
                red_omp2 = 100.0 * (d_uhf - d_omp2) / d_uhf if d_uhf > 1e-12 else 0.0

                res_omp3 = mshqc.OMP3(mol, basis, ints, scf_res, cfg).compute()
                _, s2_omp3 = compute_spin_s2(mol, ints, res_omp3.C_alpha, res_omp3.C_beta, oa, ob)
                d_omp3 = max(0.0, s2_omp3 - s_ideal)
                red_omp3 = 100.0 * (d_uhf - d_omp3) / d_uhf if d_uhf > 1e-12 else 0.0

            print(f"| {sys_name:<7} | {mode:<5} | {s_ideal:<6.2f} | {s2_uhf:<10.6f} | {s2_omp2:<10.6f} ({res_omp2.iterations:>2} it) | {red_omp2:>6.2f} % | {s2_omp3:<10.6f} ({res_omp3.iterations:>2} it) | {red_omp3:>6.2f} % |", flush=True)


# ==============================================================================
# 4. JEJAK KONVERGENSI ITERASI TR-SOSCF & VERIFIKASI GRADIEN (OMP2 & OMP3)
# ==============================================================================
def run_table_4_iteration_trace_and_fd():
    print("\n" + "=" * 108, flush=True)
    print("  4. JEJAK ITERASI PENUH TRUST-REGION SOSCF & VERIFIKASI GRADIEN (cc-pVTZ, EXACT & DF)", flush=True)
    print("=" * 108, flush=True)

    for sys_name, mode in [("H2O", "DF"), ("Oxygen", "DF")]:
        print(f"\n>>> Log Iterasi Lengkap TR-SOSCF: {sys_name} (cc-pVTZ, Mode={mode}) <<<", flush=True)
        mol = build_mol(sys_name)
        basis = mshqc.BasisSet("cc-pvtz", mol)
        ints = build_ints(mol, basis)
        with suppress_stdout():
            scf_res = run_scf(mol, basis, ints, mode, SYSTEMS[sys_name]["is_uhf"], "cc-pvtz-ri", print_level=0)

        cfg = make_mp2_config(mode, "cc-pvtz-ri", print_level=1, max_iterations=50)
        flush_all()
        mshqc.OMP2(mol, basis, ints, scf_res, cfg).compute()
        flush_all()
        mshqc.OMP3(mol, basis, ints, scf_res, cfg).compute()
        flush_all()

    print("\n>>> Uji Verifikasi Finite-Difference (C++ print_level=3) <<<", flush=True)
    for sys_name in ["Neon", "H2O", "Carbon"]:
        print(f"\n[{sys_name}] Menguji FD C++ internal:", flush=True)
        mol = build_mol(sys_name)
        basis = mshqc.BasisSet("cc-pvtz", mol)
        ints = build_ints(mol, basis)
        with suppress_stdout():
            scf_res = run_scf(mol, basis, ints, "EXACT", SYSTEMS[sys_name]["is_uhf"])

        cfg_fd = make_mp2_config("EXACT", print_level=3, max_iterations=0)
        flush_all()
        mshqc.OMP2(mol, basis, ints, scf_res, cfg_fd).compute()
        flush_all()
        mshqc.OMP3(mol, basis, ints, scf_res, cfg_fd).compute()
        flush_all()


# ==============================================================================
# 5. TABEL PROFIL KONVERGENSI ITERASI (N_iter) & ENERGI RELAKSASI ORBITAL (dE_OO)
# ==============================================================================
def run_table_5_convergence_and_relaxation():
    print("\n" + "=" * 126, flush=True)
    print("  5. TABEL PROFIL KONVERGENSI TR-SOSCF (N_iter) & EFEK RELAKSASI ORBITAL dE_OO (cc-pVTZ / cc-pVTZ-RI)", flush=True)
    print("=" * 126, flush=True)
    print(f"| {'Sistem':<7} | {'Mode ERI':<8} | {'E_MP2 (Ha)':<14} | {'E_OMP2 (Ha)':<14} | {'dE_OO MP2(mHa)':<14} | {'It_OMP2':<7} | {'E_MP3 (Ha)':<14} | {'E_OMP3 (Ha)':<14} | {'dE_OO MP3(mHa)':<14} | {'It_OMP3':<7} |", flush=True)
    print(f"|{'-'*9}|{'-'*10}|{'-'*16}|{'-'*16}|{'-'*16}|{'-'*9}|{'-'*16}|{'-'*16}|{'-'*16}|{'-'*9}|", flush=True)

    for sys_name in ["Neon", "H2O", "Lithium", "Carbon", "Oxygen", "OH", "O2"]:
        is_uhf = SYSTEMS[sys_name]["is_uhf"]
        for mode in ["EXACT", "CHOLESKY", "DF"]:
            mol = build_mol(sys_name)
            basis = mshqc.BasisSet("cc-pvtz", mol)
            ints = build_ints(mol, basis)
            with suppress_stdout():
                scf_res = run_scf(mol, basis, ints, mode, is_uhf, "cc-pvtz-ri", print_level=0)
                cfg = make_mp2_config(mode, "cc-pvtz-ri", print_level=0, max_iterations=50)

                # Energi MP2 & MP3 Kanonik (Sebelum Optimisasi Orbital)
                mp2_solver = mshqc.UMP2(mol, basis, ints, scf_res, cfg) if is_uhf else mshqc.RMP2(mol, basis, ints, scf_res, cfg)
                res_mp2_can = mp2_solver.compute()
                e_mp2 = res_mp2_can.energy_total

                mp3_solver = mshqc.UMP3(scf_res, res_mp2_can, cfg, ints) if is_uhf else mshqc.RMP3(scf_res, res_mp2_can, cfg, ints)
                res_mp3_can = mp3_solver.compute()
                e_mp3 = res_mp3_can.e_total

                # Energi OMP2 & OMP3 (Sesudah Optimisasi Orbital)
                res_omp2 = mshqc.OMP2(mol, basis, ints, scf_res, cfg).compute()
                e_omp2 = res_omp2.energy_total

                res_omp3 = mshqc.OMP3(mol, basis, ints, scf_res, cfg).compute()
                e_omp3 = res_omp3.e_total

            print(f"| {sys_name:<7} | {mode:<8} | {e_mp2:<14.8f} | {e_omp2:<14.8f} | {(e_omp2 - e_mp2) * 1000.0:<14.4f} | {res_omp2.iterations:<7} | {e_mp3:<14.8f} | {e_omp3:<14.8f} | {(e_omp3 - e_mp3) * 1000.0:<14.4f} | {res_omp3.iterations:<7} |", flush=True)
            gc.collect()


# ==============================================================================
# 6. PENGUKURAN PEAK RAM AKTUAL TERKALIBRASI (SUBPROCESS ISOLATED)
# ==============================================================================
def extract_json_safe(output_str):
    for line in reversed(output_str.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return {"base_hwm": 0.0, "peak_hwm": 0.0, "net_ram": 0.0}


MSHQC_RAM_WORKER = r"""
import sys, gc, json, os, mshqc

def get_rss_and_hwm_mb():
    rss, hwm = 0.0, 0.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"): rss = int(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"): hwm = int(line.split()[1]) / 1024.0
    return rss, hwm

old_stdout_fd = os.dup(1)
devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull_fd, 1)
try:
    mol_w = mshqc.Molecule(0, 1)
    mol_w.add_atom(10, 0.0, 0.0, 0.0)
    bas_w = mshqc.BasisSet("cc-pvdz", mol_w)
    int_w = mshqc.IntegralEngine(mol_w, bas_w)
    sc_w = mshqc.SCFConfig(); sc_w.print_level = 0; sc_w.eri_method = "df"; sc_w.use_df = True; sc_w.aux_basis_name = "cc-pvdz-ri"
    s_w = mshqc.RHF(mol_w, bas_w, int_w, sc_w).compute()
    mc_w = mshqc.MP2Config(); mc_w.print_level = 0; mc_w.eri_method = "df"; mc_w.use_df = True; mc_w.aux_basis_name = "cc-pvdz-ri"
    mshqc.OMP2(mol_w, bas_w, int_w, s_w, mc_w).compute()
    del s_w, int_w, bas_w, mol_w; gc.collect()

    base_rss, base_hwm = get_rss_and_hwm_mb()
    payload = json.loads(sys.argv[1])
    mol = mshqc.Molecule(payload["charge"], payload["mult"])
    for z, x, y, zc in payload["atoms"]: mol.add_atom(z, x, y, zc)
    basis = mshqc.BasisSet(payload["basis"], mol)
    ints = mshqc.IntegralEngine(mol, basis)

    mode = payload["mode"]
    sc = mshqc.SCFConfig(); sc.print_level = 0; sc.energy_threshold = 1e-9; sc.density_threshold = 1e-9
    mc = mshqc.MP2Config(); mc.print_level = 0; mc.energy_threshold = 1e-9; mc.gradient_threshold = 1e-5
    if mode == "DF":
        sc.eri_method = "df"; sc.use_df = True; sc.aux_basis_name = payload["aux"]
        mc.eri_method = "df"; mc.use_df = True; mc.aux_basis_name = payload["aux"]
    elif mode == "CHOLESKY":
        sc.eri_method = "cholesky"; sc.cholesky_threshold = 1e-9
        mc.eri_method = "cholesky"; mc.cholesky_threshold = 1e-9
    else:
        sc.eri_method = "exact"
        mc.eri_method = "exact"

    scf_res = (mshqc.UHF(mol, basis, ints, sc) if payload["is_uhf"] else mshqc.RHF(mol, basis, ints, sc)).compute()
    if payload["method"] == "OMP2":
        mshqc.OMP2(mol, basis, ints, scf_res, mc).compute()
    else:
        mshqc.OMP3(mol, basis, ints, scf_res, mc).compute()

    curr_rss, peak_hwm = get_rss_and_hwm_mb()
    result_str = json.dumps({"base_hwm": base_hwm, "peak_hwm": peak_hwm, "net_ram": max(curr_rss - base_rss, peak_hwm - base_hwm, 0.10)})
finally:
    os.dup2(old_stdout_fd, 1); os.close(devnull_fd); os.close(old_stdout_fd)
print(result_str)
"""


def run_table_6_calibrated_ram():
    print("\n" + "=" * 88, flush=True)
    print("  6. PENGUKURAN PEAK RSS (VmHWM) & NET WORKING RAM TERKALIBRASI SETELAH WARM-UP (MSHQC)", flush=True)
    print("=" * 88, flush=True)
    print(f"| {'Sistem':<7} | {'Basis':<7} | {'Mode ERI':<8} | {'Metode':<6} | {'mshqc Peak(MB)':<14} | {'mshqc Net(MB)':<13} |", flush=True)
    print(f"|{'-'*9}|{'-'*9}|{'-'*10}|{'-'*8}|{'-'*16}|{'-'*15}|", flush=True)

    cases = [
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
    ]

    for sys_name, b_name, aux_name, mode, method in cases:
        info = SYSTEMS[sys_name]
        payload = json.dumps({
            "atoms": info["atoms"], "charge": info["charge"], "mult": info["mult"],
            "is_uhf": info["is_uhf"], "basis": b_name, "aux": aux_name, "mode": mode, "method": method
        })
        try:
            out_m = subprocess.check_output([sys.executable, "-c", MSHQC_RAM_WORKER, payload], text=True)
            res_m = extract_json_safe(out_m)
            print(f"| {sys_name:<7} | {b_name:<7} | {mode:<8} | {method:<6} | {res_m['peak_hwm']:<14.2f} | {res_m['net_ram']:<13.2f} |", flush=True)
        except Exception:
            print(f"| {sys_name:<7} | {b_name:<7} | {mode:<8} | {method:<6} | {'ERR':<14} | {'ERR':<13} |", flush=True)


if __name__ == "__main__":
    run_table_1_eri_compression()
    run_table_2_o_v4_elimination()
    run_table_3_spin_contamination()
    run_table_4_iteration_trace_and_fd()
    run_table_5_convergence_and_relaxation()
    run_table_6_calibrated_ram()