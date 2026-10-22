#!/usr/bin/env python3
"""
Benchmark & Validasi Empiris Bab IV Skripsi (MSHQC vs Psi4)
Mencakup:
  1. Kompresi Tensor ERI & Peringkat Cholesky Aktual (Exact vs Cholesky vs DF)
  2. Bukti Eliminasi Hambatan Memori O(V^4) pada OMP3 (MO Tensor Breakdown)
  3. Evaluasi Reduksi Kontaminasi Spin <S^2> (UHF -> OMP2 -> OMP3)
  4. Uji Konsistensi Gradien Analitik vs Finite-Difference (OMP2 & OMP3)
  5. Pengukuran Peak RSS (VmHWM) & Net Working RAM Terkalibrasi (mshqc vs Psi4)
"""

import os
import re
import gc
import sys
import json
import subprocess
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
            (8, 0.000000,  0.000000,  0.117300 * ANG_TO_BOHR),
            (1, 0.000000,  0.757200 * ANG_TO_BOHR, -0.469200 * ANG_TO_BOHR),
            (1, 0.000000, -0.757200 * ANG_TO_BOHR, -0.469200 * ANG_TO_BOHR),
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


def build_mol(sys_name):
    info = SYSTEMS[sys_name]
    mol = mshqc.Molecule(info["charge"], info["mult"])
    for z, x, y, z_coord in info["atoms"]:
        mol.add_atom(z, x, y, z_coord)
    return mol


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


def run_scf(mol, basis, ints, eri_mode, is_uhf, aux_basis_name=None):
    scf_cfg = mshqc.SCFConfig()
    scf_cfg.energy_threshold = 1e-9
    scf_cfg.density_threshold = 1e-9
    scf_cfg.print_level = 0

    if eri_mode == "DF":
        scf_cfg.use_df = True
        scf_cfg.eri_method = "df"
        aux = mshqc.BasisSet(aux_basis_name, mol)
        solver = mshqc.UHF(mol, basis, aux, ints, scf_cfg) if is_uhf else mshqc.RHF(mol, basis, aux, ints, scf_cfg)
    elif eri_mode == "CHOLESKY":
        scf_cfg.use_cholesky = True
        scf_cfg.eri_method = "cholesky"
        scf_cfg.cholesky_threshold = 1e-9
        solver = mshqc.UHF(mol, basis, ints, scf_cfg) if is_uhf else mshqc.RHF(mol, basis, ints, scf_cfg)
    else:
        scf_cfg.eri_method = "exact"
        solver = mshqc.UHF(mol, basis, ints, scf_cfg) if is_uhf else mshqc.RHF(mol, basis, ints, scf_cfg)

    return solver.compute()


# ==============================================================================
# 1. TABEL KOMPRESI TENSOR ERI AO & VEKTOR CHOLESKY AKTUAL
# ==============================================================================
def run_table_1_eri_compression():
    print("\n" + "=" * 108)
    print("  1. TABEL KOMPRESI TENSOR ERI AO & VEKTOR CHOLESKY AKTUAL (MSHQC, tau_CD = 1e-9)")
    print("=" * 108)
    print(f"| {'Sistem':<8} | {'Basis':<8} | {'N_bf':<5} | {'N_aux(DF)':<9} | {'M_chol':<7} | {'Exact (MB)':<11} | {'Chol (MB)':<10} | {'DF (MB)':<9} | {'Exact/Chol':<10} | {'Exact/DF':<9} |")
    print(f"|{'-'*10}|{'-'*10}|{'-'*7}|{'-'*11}|{'-'*9}|{'-'*13}|{'-'*12}|{'-'*11}|{'-'*12}|{'-'*11}|")

    cases = [
        ("Neon",    "cc-pvdz", "cc-pvdz-ri"),
        ("Neon",    "cc-pvtz", "cc-pvtz-ri"),
        ("Neon",    "cc-pvqz", "cc-pvqz-ri"),
        ("H2O",     "cc-pvdz", "cc-pvdz-ri"),
        ("H2O",     "cc-pvtz", "cc-pvtz-ri"),
        ("H2O",     "cc-pvqz", "cc-pvqz-ri"),
        ("Lithium", "cc-pvtz", "cc-pvtz-ri"),
        ("Carbon",  "cc-pvtz", "cc-pvtz-ri"),
        ("Oxygen",  "cc-pvtz", "cc-pvtz-ri"),
        ("O2",      "cc-pvtz", "cc-pvtz-ri"),
    ]

    for sys_name, b_name, aux_name in cases:
        mol = build_mol(sys_name)
        basis = mshqc.BasisSet(b_name, mol)
        aux = mshqc.BasisSet(aux_name, mol)
        ints = mshqc.IntegralEngine(mol, basis)

        nbf = basis.n_basis_functions()
        naux = aux.n_basis_functions()

        scf_chol = run_scf(mol, basis, ints, "CHOLESKY", SYSTEMS[sys_name]["is_uhf"])
        m_chol = np.array(scf_chol.L_mat).shape[1]

        mb = 1024.0 * 1024.0
        mem_exact = (8.0 * (nbf ** 4)) / mb
        mem_chol  = (8.0 * (nbf ** 2) * m_chol) / mb
        mem_df    = (8.0 * (nbf ** 2) * naux) / mb

        r_chol = mem_exact / mem_chol
        r_df   = mem_exact / mem_df

        print(f"| {sys_name:<8} | {b_name:<8} | {nbf:<5} | {naux:<9} | {m_chol:<7} | {mem_exact:<11.3f} | {mem_chol:<10.3f} | {mem_df:<9.3f} | {r_chol:<8.2f} x | {r_df:<7.2f} x |")
        gc.collect()


# ==============================================================================
# 2. TABEL BUKTI ELIMINASI MEMORI O(V^4) PADA OMP3 (MO TENSOR BREAKDOWN)
# ==============================================================================
def run_table_2_o_v4_elimination():
    print("\n" + "=" * 116)
    print("  2. TABEL ELIMINASI HAMBATAN MEMORI O(V^4) PADA OMP3: KONVENSIONAL vs MSHQC (DF MODE)")
    print("=" * 116)
    print(f"| {'Sistem':<7} | {'Basis':<8} | {'(O, V, N_aux)':<15} | {'Konvensional V_vvvv + Gamma_vvvv (MB)':<37} | {'MSHQC B_3c + T2 + Y_ooov (MB)':<30} | {'Reduksi Memori MO':<17} |")
    print(f"|{'-'*9}|{'-'*10}|{'-'*17}|{'-'*39}|{'-'*32}|{'-'*19}|")

    cases = [
        ("Neon",   "cc-pvdz", "cc-pvdz-ri"),
        ("Neon",   "cc-pvtz", "cc-pvtz-ri"),
        ("Neon",   "cc-pvqz", "cc-pvqz-ri"),
        ("H2O",    "cc-pvdz", "cc-pvdz-ri"),
        ("H2O",    "cc-pvtz", "cc-pvtz-ri"),
        ("H2O",    "cc-pvqz", "cc-pvqz-ri"),
        ("Oxygen", "cc-pvtz", "cc-pvtz-ri"),
        ("O2",     "cc-pvtz", "cc-pvtz-ri"),
    ]

    mb = 1024.0 * 1024.0
    for sys_name, b_name, aux_name in cases:
        mol = build_mol(sys_name)
        basis = mshqc.BasisSet(b_name, mol)
        aux = mshqc.BasisSet(aux_name, mol)
        ints = mshqc.IntegralEngine(mol, basis)
        is_uhf = SYSTEMS[sys_name]["is_uhf"]

        scf_res = run_scf(mol, basis, ints, "DF", is_uhf, aux_name)
        nbf = basis.n_basis_functions()
        naux = aux.n_basis_functions()
        oa, ob = scf_res.n_occ_alpha, scf_res.n_occ_beta
        va, vb = nbf - oa, nbf - ob

        # Konvensional: menyimpan V_vvvv dan Gamma_vvvv (4-indeks virtual)
        if not is_uhf:
            mem_conv_v4 = 2.0 * 8.0 * (va ** 4) / mb
            # MSHQC: B_oo + B_ia + B_vv + T2^(1) + T2^(2) + T_pairs + Y_ooov
            n_pairs = oa * (oa + 1) // 2
            mem_b3c   = 8.0 * (oa*oa + oa*va + va*va) * naux / mb
            mem_t2    = 2.0 * 8.0 * (oa * oa * va * va) / mb
            mem_slice = 2.0 * 8.0 * (n_pairs * va * va) / mb
            mem_yooov = 8.0 * (oa * oa * oa * va) / mb
            mem_mshqc = mem_b3c + mem_t2 + mem_slice + mem_yooov
        else:
            # UHF: kanal aa, bb, dan ab untuk V_vvvv dan Gamma_vvvv
            mem_conv_v4 = 2.0 * 8.0 * ((va ** 4) + (vb ** 4) + (va * va * vb * vb)) / mb
            mem_b3c   = 8.0 * ((oa*oa + oa*va + va*va) + (ob*ob + ob*vb + vb*vb)) * naux / mb
            mem_t2    = 2.0 * 8.0 * ((oa*oa*va*va) + (ob*ob*vb*vb) + (oa*ob*va*vb)) / mb
            mem_slice = 2.0 * 8.0 * ((oa*ob*va*vb) + (oa*(oa-1)//2)*va*va) / mb
            mem_yooov = 8.0 * ((oa*oa*oa*va) + (ob*ob*ob*vb) + (oa*ob*oa*vb) + (oa*ob*ob*va)) / mb
            mem_mshqc = mem_b3c + mem_t2 + mem_slice + mem_yooov

        ratio = mem_conv_v4 / mem_mshqc
        dims_str = f"({oa},{va},{naux})"
        print(f"| {sys_name:<7} | {b_name:<8} | {dims_str:<15} | {mem_conv_v4:<37.3f} | {mem_mshqc:<30.3f} | {ratio:<15.2f} x |")
        gc.collect()


# ==============================================================================
# 3. TABEL EVALUASI KONTAMINASI SPIN <S^2> (UHF -> OMP2 -> OMP3)
# ==============================================================================
def run_table_3_spin_contamination():
    print("\n" + "=" * 118)
    print("  3. TABEL PEMULIHAN SIMETRI SPIN <S^2> SEBELUM (UHF) DAN SESUDAH OPTIMISASI ORBITAL (OMP2 & OMP3, cc-pVTZ)")
    print("=" * 118)
    print(f"| {'Sistem':<8} | {'Mode':<6} | {'Ideal':<8} | {'<S^2> UHF':<10} | {'d<S^2> UHF':<10} | {'<S^2> OMP2':<10} | {'Red. OMP2':<9} | {'<S^2> OMP3':<10} | {'Red. OMP3':<9} |")
    print(f"|{'-'*10}|{'-'*8}|{'-'*10}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*11}|{'-'*12}|{'-'*11}|")

    radicals = ["Lithium", "Carbon", "Oxygen", "OH", "O2"]
    for sys_name in radicals:
        for mode in ["EXACT", "DF"]:
            mol = build_mol(sys_name)
            basis = mshqc.BasisSet("cc-pvtz", mol)
            ints = mshqc.IntegralEngine(mol, basis)
            scf_res = run_scf(mol, basis, ints, mode, True, "cc-pvtz-ri")

            s_ideal, s2_uhf = compute_spin_s2(
                mol, ints, scf_res.C_alpha, scf_res.C_beta,
                scf_res.n_occ_alpha, scf_res.n_occ_beta
            )
            d_uhf = max(0.0, s2_uhf - s_ideal)

            cfg = mshqc.MP2Config()
            cfg.eri_method = mode.lower()
            cfg.use_df = (mode == "DF")
            cfg.opt_method = "soscf"
            cfg.energy_threshold = 1e-9
            cfg.gradient_threshold = 1e-5
            cfg.print_level = 0

            omp2 = mshqc.OMP2(mol, basis, ints, scf_res, cfg)
            res_omp2 = omp2.compute()
            _, s2_omp2 = compute_spin_s2(
                mol, ints, res_omp2.C_alpha, res_omp2.C_beta,
                scf_res.n_occ_alpha, scf_res.n_occ_beta
            )
            d_omp2 = max(0.0, s2_omp2 - s_ideal)
            red_omp2 = 100.0 * (d_uhf - d_omp2) / d_uhf if d_uhf > 1e-12 else 0.0

            omp3 = mshqc.OMP3(mol, basis, ints, scf_res, cfg)
            # Panggil OMP2.compute(omp3) untuk mendapatkan MP2Result yang memuat C_alpha & C_beta teroptimasi OMP3
            res_omp3_orb = mshqc.OMP2.compute(omp3)
            _, s2_omp3 = compute_spin_s2(
                mol, ints, res_omp3_orb.C_alpha, res_omp3_orb.C_beta,
                scf_res.n_occ_alpha, scf_res.n_occ_beta
            )
            d_omp3 = max(0.0, s2_omp3 - s_ideal)
            red_omp3 = 100.0 * (d_uhf - d_omp3) / d_uhf if d_uhf > 1e-12 else 0.0

            print(f"| {sys_name:<8} | {mode:<6} | {s_ideal:<8.4f} | {s2_uhf:<10.6f} | {d_uhf:<10.6f} | {s2_omp2:<10.6f} | {red_omp2:<7.2f} % | {s2_omp3:<10.6f} | {red_omp3:<7.2f} % |")
            gc.collect()


# ==============================================================================
# 4. UJI FINITE-DIFFERENCE vs GRADIEN ANALITIK (OMP2 & OMP3)
# ==============================================================================
def run_table_4_finite_difference():
    print("\n" + "=" * 105)
    print("  4. VERIFIKASI GRADIEN ROTASI ORBITAL ANALITIK vs CENTRAL FINITE-DIFFERENCE (cc-pVTZ, EXACT)")
    print("=" * 105)

    for sys_name in ["Neon", "H2O", "Lithium", "Carbon", "Oxygen"]:
        print(f"\n>>> Sistem: {sys_name} <<<")
        mol = build_mol(sys_name)
        basis = mshqc.BasisSet("cc-pvtz", mol)
        ints = mshqc.IntegralEngine(mol, basis)
        scf_res = run_scf(mol, basis, ints, "EXACT", SYSTEMS[sys_name]["is_uhf"])

        cfg = mshqc.MP2Config()
        cfg.eri_method = "exact"
        cfg.opt_method = "soscf"
        cfg.max_iterations = 1
        cfg.print_level = 3

        omp2 = mshqc.OMP2(mol, basis, ints, scf_res, cfg)
        omp2.compute()

        omp3 = mshqc.OMP3(mol, basis, ints, scf_res, cfg)
        omp3.compute_omp3()
        gc.collect()


# ==============================================================================
# 5. PENGUKURAN PEAK RAM AKTUAL TERKALIBRASI (SUBPROCESS ISOLATED)
# ==============================================================================
MSHQC_RAM_WORKER = r"""
import sys, gc, json
import mshqc

ANG_TO_BOHR = 1.8897261246257702

def get_rss_And_hwm_mb():
    rss, hwm = 0.0, 0.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"):
                hwm = int(line.split()[1]) / 1024.0
    return rss, hwm

# 1. Warm-up runtime C++, OpenMP thread pool, TBLIS, dan libcint menggunakan H2 kecil
mol_w = mshqc.Molecule(0, 1)
mol_w.add_atom(1, 0.0, 0.0, 0.0)
mol_w.add_atom(1, 0.0, 0.0, 1.4)
bas_w = mshqc.BasisSet("cc-pvdz", mol_w)
aux_w = mshqc.BasisSet("cc-pvdz-ri", mol_w)
int_w = mshqc.IntegralEngine(mol_w, bas_w)
sc_w = mshqc.SCFConfig(); sc_w.print_level = 0; sc_w.use_df = True; sc_w.eri_method = "df"
s_w = mshqc.RHF(mol_w, bas_w, aux_w, int_w, sc_w).compute()
mc_w = mshqc.MP2Config(); mc_w.print_level = 0; mc_w.use_df = True; mc_w.eri_method = "df"
mshqc.OMP3(mol_w, bas_w, int_w, s_w, mc_w).compute_omp3()
del s_w, int_w, aux_w, bas_w, mol_w
gc.collect()

base_rss, base_hwm = get_rss_And_hwm_mb()

# 2. Jalankan target benchmark sebenarnya
payload = json.loads(sys.argv[1])
mol = mshqc.Molecule(payload["charge"], payload["mult"])
for z, x, y, zc in payload["atoms"]:
    mol.add_atom(z, x, y, zc)

basis = mshqc.BasisSet(payload["basis"], mol)
ints = mshqc.IntegralEngine(mol, basis)
sc = mshqc.SCFConfig()
sc.energy_threshold = 1e-9
sc.density_threshold = 1e-9
sc.print_level = 0

mode = payload["mode"]
is_uhf = payload["is_uhf"]
if mode == "DF":
    sc.use_df = True; sc.eri_method = "df"
    aux = mshqc.BasisSet(payload["aux"], mol)
    scf_res = (mshqc.UHF(mol, basis, aux, ints, sc) if is_uhf else mshqc.RHF(mol, basis, aux, ints, sc)).compute()
elif mode == "CHOLESKY":
    sc.use_cholesky = True; sc.eri_method = "cholesky"; sc.cholesky_threshold = 1e-9
    scf_res = (mshqc.UHF(mol, basis, ints, sc) if is_uhf else mshqc.RHF(mol, basis, ints, sc)).compute()
else:
    sc.eri_method = "exact"
    scf_res = (mshqc.UHF(mol, basis, ints, sc) if is_uhf else mshqc.RHF(mol, basis, ints, sc)).compute()

mc = mshqc.MP2Config()
mc.eri_method = mode.lower()
mc.use_df = (mode == "DF")
mc.opt_method = "soscf"
mc.energy_threshold = 1e-9
mc.gradient_threshold = 1e-5
mc.print_level = 0

if payload["method"] == "OMP2":
    mshqc.OMP2(mol, basis, ints, scf_res, mc).compute()
else:
    mshqc.OMP3(mol, basis, ints, scf_res, mc).compute_omp3()

_, peak_hwm = get_rss_And_hwm_mb()
net_ram = max(0.05, peak_hwm - base_hwm)
print(json.dumps({"base_hwm": base_hwm, "peak_hwm": peak_hwm, "net_ram": net_ram}))
"""

PSI4_RAM_WORKER = r"""
import sys, gc, json
import psi4

psi4.core.be_quiet()
psi4.set_num_threads(4)
psi4.set_memory("4 GB")

def get_rss_And_hwm_mb():
    rss, hwm = 0.0, 0.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"):
                hwm = int(line.split()[1]) / 1024.0
    return rss, hwm

# 1. Warm-up runtime Psi4 & Libint2 menggunakan H2 kecil
psi4.geometry("0 1\nH 0 0 0\nH 0 0 1.4\nunits bohr\nsymmetry c1")
psi4.set_options({"basis": "cc-pvdz", "scf_type": "df", "mp2_type": "df", "cc_type": "df"})
psi4.energy("omp3")
psi4.core.clean()
psi4.core.clean_options()
gc.collect()

base_rss, base_hwm = get_rss_And_hwm_mb()

# 2. Jalankan target benchmark sebenarnya
payload = json.loads(sys.argv[1])
psi4.geometry(payload["psi4_geom"])
mode = payload["mode"]
scf_t = "pk" if mode == "EXACT" else ("cd" if mode == "CHOLESKY" else "df")
cor_t = "conv" if mode == "EXACT" else ("cd" if mode == "CHOLESKY" else "df")

opts = {
    "basis": payload["basis"],
    "reference": "uhf" if payload["is_uhf"] else "rhf",
    "scf_type": scf_t,
    "mp2_type": cor_t,
    "cc_type": cor_t,
    "freeze_core": False,
    "e_convergence": 1e-9,
    "d_convergence": 1e-9,
    "rms_mograd_convergence": 1e-5,
    "max_mograd_convergence": 1e-4,
}
if mode == "CHOLESKY":
    opts["cholesky_tolerance"] = 1e-9
elif mode == "DF":
    opts["df_basis_scf"] = payload["aux"]
    opts["df_basis_mp2"] = payload["aux"]
    opts["df_basis_cc"]  = payload["aux"]

psi4.set_options(opts)
psi4.energy(payload["method"].lower())
_, peak_hwm = get_rss_And_hwm_mb()
psi4.core.clean()
net_ram = max(0.05, peak_hwm - base_hwm)
print(json.dumps({"base_hwm": base_hwm, "peak_hwm": peak_hwm, "net_ram": net_ram}))
"""


def run_table_5_calibrated_ram():
    psi4_py = os.environ.get("PSI4_PY", "")
    has_psi4 = bool(psi4_py and os.path.exists(psi4_py))

    print("\n" + "=" * 122)
    print("  5. PENGUKURAN PEAK RSS (VmHWM) & NET WORKING RAM TERKALIBRASI SETELAH WARM-UP (MSHQC vs PSI4)")
    print("=" * 122)
    print(f"| {'Sistem':<7} | {'Basis':<7} | {'Mode ERI':<8} | {'Metode':<6} | {'mshqc Peak(MB)':<14} | {'mshqc Net(MB)':<13} | {'Psi4 Peak(MB)':<13} | {'Psi4 Net(MB)':<12} | {'Rasio Peak':<10} |")
    print(f"|{'-'*9}|{'-'*9}|{'-'*10}|{'-'*8}|{'-'*16}|{'-'*15}|{'-'*15}|{'-'*14}|{'-'*12}|")

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
            "atoms": info["atoms"],
            "charge": info["charge"],
            "mult": info["mult"],
            "is_uhf": info["is_uhf"],
            "psi4_geom": info["psi4_geom"],
            "basis": b_name,
            "aux": aux_name,
            "mode": mode,
            "method": method,
        })

        out_m = subprocess.check_output([sys.executable, "-c", MSHQC_RAM_WORKER, payload], text=True)
        res_m = json.loads(out_m.strip().splitlines()[-1])

        if has_psi4:
            out_p = subprocess.check_output([psi4_py, "-c", PSI4_RAM_WORKER, payload], text=True)
            res_p = json.loads(out_p.strip().splitlines()[-1])
            p_peak = f"{res_p['peak_hwm']:.2f}"
            p_net  = f"{res_p['net_ram']:.2f}"
            ratio  = f"{res_p['peak_hwm'] / res_m['peak_hwm']:.2f} x"
        else:
            p_peak, p_net, ratio = "N/A", "N/A", "N/A"

        print(f"| {sys_name:<7} | {b_name:<7} | {mode:<8} | {method:<6} | {res_m['peak_hwm']:<14.2f} | {res_m['net_ram']:<13.2f} | {p_peak:<13} | {p_net:<12} | {ratio:<10} |")


if __name__ == "__main__":
    run_table_1_eri_compression()
    run_table_2_o_v4_elimination()
    run_table_3_spin_contamination()
    run_table_4_finite_difference()
    run_table_5_calibrated_ram()