#!/usr/bin/env python3
"""
Skrip Pengujian Tambahan Bab IV Skripsi (mshqc vs Psi4):
1. Pengukuran Memori RAM Aktual (Peak RSS / VmHWM) & Kompresi Cholesky Nyata
2. Penskalaan Memori & Waktu terhadap Ukuran Basis (cc-pVDZ -> cc-pVTZ -> cc-pVQZ)
3. Evaluasi Kontaminasi Spin <S^2> Sebelum (UHF) & Sesudah Optimisasi Orbital (OMP2/OMP3)
4. Ekstraksi Tabel Uji Gradien Analitik vs Central Finite-Difference (print_level=3)
5. Perbandingan Konvergensi Matrix-Free Trust-Region SOSCF vs L-BFGS
"""

import os
import re
import sys
import json
import time
import subprocess
import numpy as np

ANG_TO_BOHR = 1.88972612462577

SYSTEMS = [
    ("Neon",    [("Ne", 0.0, 0.0, 0.0)], 0, 1, "rhf"),
    ("H2O",     [("O",  0.0, 0.0, 0.1173),
                 ("H",  0.0, 0.7572, -0.4692),
                 ("H",  0.0, -0.7572, -0.4692)], 0, 1, "rhf"),
    ("Lithium", [("Li", 0.0, 0.0, 0.0)], 0, 2, "uhf"),
    ("Carbon",  [("C",  0.0, 0.0, 0.0)], 0, 3, "uhf"),
    ("Oxygen",  [("O",  0.0, 0.0, 0.0)], 0, 3, "uhf"),
    # Tambahan molekul radikal kulit terbuka untuk uji kontaminasi spin <S^2>
    ("O2",      [("O",  0.0, 0.0,  0.6035),
                 ("O",  0.0, 0.0, -0.6035)], 0, 3, "uhf"),
    ("OH",      [("O",  0.0, 0.0,  0.0000),
                 ("H",  0.0, 0.0,  0.9697)], 0, 2, "uhf"),
]


def get_linux_memory_mb():
    """Membaca VmRSS (RAM saat ini) dan VmHWM (Peak RAM) dari kernel Linux (MB)."""
    vm_rss, vm_hwm = 0.0, 0.0
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    vm_rss = float(line.split()[1]) / 1024.0
                elif line.startswith("VmHWM:"):
                    vm_hwm = float(line.split()[1]) / 1024.0
    except Exception:
        import resource
        vm_hwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        vm_rss = vm_hwm
    return vm_rss, vm_hwm


def compute_s2_from_mo(C_alpha, C_beta, S_mat, n_alpha, n_beta):
    """Menghitung <S^2> eksak sesuai Persamaan (4.1) dan UHF::compute_s2 di scf.cc."""
    sz = 0.5 * (n_alpha - n_beta)
    exact_s2 = sz * (sz + 1.0)
    if n_alpha == 0 or n_beta == 0:
        return exact_s2, exact_s2, 0.0
    Ca_occ = np.asarray(C_alpha)[:, :n_alpha]
    Cb_occ = np.asarray(C_beta)[:, :n_beta]
    S_np = np.asarray(S_mat)
    overlap_ab = Ca_occ.T @ S_np @ Cb_occ
    s2_val = exact_s2 + n_beta - np.sum(overlap_ab ** 2)
    return exact_s2, float(s2_val), float(s2_val - exact_s2)


# ==============================================================================
# MODE WORKER SUBPROCESS: Mengukur Peak RSS (VmHWM) Murni per Perhitungan
# ==============================================================================
def run_single_memory_worker(engine, sys_name, basis_name, eri_mode, method):
    sys_dict = {s[0]: s for s in SYSTEMS}
    _, atoms, chg, mult, ref = sys_dict[sys_name]

    if engine == "mshqc":
        import mshqc
        rss_base, _ = get_linux_memory_mb()
        mol = mshqc.Molecule(charge=chg, multiplicity=mult)
        for sym, x, y, z in atoms:
            mol.add_atom(sym, x, y, z, units="angstrom")
        basis = mshqc.BasisSet(basis_name, mol)
        integrals = mshqc.IntegralEngine(mol, basis)

        scf_conf = mshqc.SCFConfig(eri_method=eri_mode, threshold=1e-9)
        t0 = time.perf_counter()
        scf_res = (mshqc.RHF(mol, basis, integrals, scf_conf).compute()
                   if ref == "rhf" else
                   mshqc.UHF(mol, basis, integrals, scf_conf).compute())

        mp2_conf = mshqc.MP2Config(eri_method=eri_mode, threshold=1e-9, gradient_threshold=1e-5)
        if method == "SCF":
            e_tot = scf_res.energy_total
        elif method == "MP2":
            res = (mshqc.RMP2(mol, basis, integrals, scf_res, mp2_conf).compute()
                   if ref == "rhf" else
                   mshqc.UMP2(mol, basis, integrals, scf_res, mp2_conf).compute())
            e_tot = res.energy_total
        elif method == "MP3":
            mp2_res = (mshqc.RMP2(mol, basis, integrals, scf_res, mp2_conf).compute()
                       if ref == "rhf" else
                       mshqc.UMP2(mol, basis, integrals, scf_res, mp2_conf).compute())
            res = (mshqc.RMP3(scf_res, mp2_res, mp2_conf, integrals).compute()
                   if ref == "rhf" else
                   mshqc.UMP3(scf_res, mp2_res, mp2_conf, integrals).compute())
            e_tot = res.e_total
        elif method == "OMP2":
            res = mshqc.OMP2(mol, basis, integrals, scf_res, mp2_conf).compute()
            e_tot = res.energy_total
        elif method == "OMP3":
            res = mshqc.OMP3(mol, basis, integrals, scf_res, mp2_conf).compute()
            e_tot = res.e_total
        elapsed = time.perf_counter() - t0
        rss_after, hwm_peak = get_linux_memory_mb()
        print(json.dumps({
            "energy": e_tot, "time": elapsed,
            "rss_base_mb": rss_base, "rss_after_mb": rss_after,
            "peak_hwm_mb": hwm_peak, "delta_peak_mb": max(0.0, hwm_peak - rss_base)
        }))

    elif engine == "psi4":
        import psi4
        psi4.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
        psi4.set_memory("8 GB")
        psi4.core.be_quiet()
        rss_base, _ = get_linux_memory_mb()

        lines = [f"{chg} {mult}", "units angstrom"]
        if eri_mode in ["df", "cholesky"]:
            lines.append("symmetry c1")
        for sym, x, y, z in atoms:
            lines.append(f"{sym} {x:.10f} {y:.10f} {z:.10f}")
        mol = psi4.geometry("\n".join(lines))

        scf_type = "pk" if eri_mode == "exact" else ("cd" if eri_mode == "cholesky" else "df")
        corr_type = "conv" if eri_mode == "exact" else ("cd" if eri_mode == "cholesky" else "df")
        aux_basis = f"{basis_name}-RI"

        opts = {
            "basis": basis_name, "reference": ref,
            "e_convergence": 1e-9, "d_convergence": 1e-9,
            "scf_type": scf_type, "mp2_type": corr_type,
            "mp_type": corr_type, "cc_type": corr_type,
            "freeze_core": "false", "r_convergence": 1e-5,
            "rms_mograd_convergence": 1e-5, "max_mograd_convergence": 1e-4,
            "read_scf_3index": False,
        }
        if eri_mode == "df":
            opts.update({"df_basis_scf": aux_basis, "df_basis_mp2": aux_basis, "df_basis_cc": aux_basis})
        elif eri_mode == "cholesky":
            opts["cholesky_tolerance"] = 1e-9
        psi4.set_options(opts)

        t0 = time.perf_counter()
        e_scf, scf_wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
        if eri_mode in ["cholesky", "df"]:
            psi4.set_options({"scf_type": "pk", "read_scf_3index": False})

        if method == "SCF":
            e_tot = e_scf
        else:
            e_tot = psi4.energy(method.lower(), ref_wfn=scf_wfn, molecule=mol)
        elapsed = time.perf_counter() - t0
        rss_after, hwm_peak = get_linux_memory_mb()
        print(json.dumps({
            "energy": float(e_tot), "time": elapsed,
            "rss_base_mb": rss_base, "rss_after_mb": rss_after,
            "peak_hwm_mb": hwm_peak, "delta_peak_mb": max(0.0, hwm_peak - rss_base)
        }))


def spawn_mem_probe(python_exe, engine, sys_name, basis_name, eri_mode, method):
    """Menjalankan satu perhitungan pada subprocess terpisah agar Peak RSS akurat."""
    cmd = [python_exe, __file__, "--worker", engine, sys_name, basis_name, eri_mode, method]
    out = subprocess.check_output(cmd, text=True)
    for line in reversed(out.strip().splitlines()):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"Gagal membaca JSON dari worker: {out}")


def run_mshqc_suite():
    import mshqc

    print("\n" + "=" * 100)
    print("  1. TABEL KOMPRESI TENSOR ERI & VEKTOR CHOLESKY AKTUAL (MSHQC)")
    print("=" * 100)
    print(f"| {'Sistem':<8} | {'Basis':<8} | {'N_bf':<5} | {'N_aux (DF)':<10} | {'M_chol':<8} | {'Mem Exact (MB)':<14} | {'Mem Chol (MB)':<13} | {'Mem DF (MB)':<11} | {'Rasio Exact/Chol':<16} | {'Rasio Exact/DF':<14} |")
    print("|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 7 + "|" + "-" * 12 + "|" + "-" * 10 + "|" + "-" * 16 + "|" + "-" * 15 + "|" + "-" * 13 + "|" + "-" * 18 + "|" + "-" * 16 + "|")

    scaling_cases = [
        ("Neon", "cc-pVDZ"), ("Neon", "cc-pVTZ"), ("Neon", "cc-pVQZ"),
        ("H2O",  "cc-pVDZ"), ("H2O",  "cc-pVTZ"), ("H2O",  "cc-pVQZ"),
        ("Lithium", "cc-pVTZ"), ("Carbon", "cc-pVTZ"), ("Oxygen", "cc-pVTZ"),
        ("O2", "cc-pVTZ"),
    ]
    sys_dict = {s[0]: s for s in SYSTEMS}

    for sys_name, b_name in scaling_cases:
        _, atoms, chg, mult, _ = sys_dict[sys_name]
        mol = mshqc.Molecule(charge=chg, multiplicity=mult)
        for sym, x, y, z in atoms:
            mol.add_atom(sym, x, y, z, units="angstrom")
        basis = mshqc.BasisSet(b_name, mol)
        aux_basis = mshqc.BasisSet(f"{b_name}-RI", mol)

        n_bf = basis.n_basis_functions()
        n_aux = aux_basis.n_basis_functions()

        chol = mshqc.CholeskyERI(mol, basis, 1e-9)
        m_chol = chol.n_vectors()

        mem_exact_mb = (8.0 * (n_bf ** 4)) / (1024.0 ** 2)
        mem_chol_mb  = (8.0 * m_chol * (n_bf ** 2)) / (1024.0 ** 2)
        mem_df_mb    = (8.0 * n_aux * (n_bf ** 2)) / (1024.0 ** 2)

        print(f"| {sys_name:<8} | {b_name:<8} | {n_bf:<5d} | {n_aux:<10d} | {m_chol:<8d} | "
              f"{mem_exact_mb:<14.3f} | {mem_chol_mb:<13.3f} | {mem_df_mb:<11.3f} | "
              f"{mem_exact_mb / max(mem_chol_mb, 1e-9):<15.2f}x | {mem_exact_mb / max(mem_df_mb, 1e-9):<13.2f}x |")

    print("\n" + "=" * 100)
    print("  2. TABEL EVALUASI KONTAMINASI SPIN <S^2> SEBELUM (UHF) & SESUDAH (OMP2) OPTIMISASI ORBITAL")
    print("=" * 100)
    print(f"| {'Sistem':<8} | {'Mode ERI':<9} | {'S(S+1) Ideal':<12} | {'<S^2> UHF':<12} | {'d<S^2> UHF':<12} | {'<S^2> OMP2':<12} | {'d<S^2> OMP2':<12} | {'Reduksi (%)':<11} |")
    print("|" + "-" * 10 + "|" + "-" * 11 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 13 + "|")

    for sys_name in ["Lithium", "Carbon", "Oxygen", "OH", "O2"]:
        _, atoms, chg, mult, _ = sys_dict[sys_name]
        for mode in ["exact", "df"]:
            mol = mshqc.Molecule(charge=chg, multiplicity=mult)
            for sym, x, y, z in atoms:
                mol.add_atom(sym, x, y, z, units="angstrom")
            basis = mshqc.BasisSet("cc-pVTZ", mol)
            integrals = mshqc.IntegralEngine(mol, basis)
            S_mat = integrals.compute_overlap()

            scf_conf = mshqc.SCFConfig(eri_method=mode, threshold=1e-9)
            uhf_res = mshqc.UHF(mol, basis, integrals, scf_conf).compute()

            mp2_conf = mshqc.MP2Config(eri_method=mode, threshold=1e-9, gradient_threshold=1e-5)
            omp2_res = mshqc.OMP2(mol, basis, integrals, uhf_res, mp2_conf).compute()

            s2_ideal, s2_uhf, ds2_uhf = compute_s2_from_mo(
                uhf_res.C_alpha, uhf_res.C_beta, S_mat, uhf_res.n_occ_alpha, uhf_res.n_occ_beta
            )
            _, s2_omp2, ds2_omp2 = compute_s2_from_mo(
                omp2_res.C_alpha, omp2_res.C_beta, S_mat, uhf_res.n_occ_alpha, uhf_res.n_occ_beta
            )
            red_pct = 100.0 * (ds2_uhf - ds2_omp2) / ds2_uhf if ds2_uhf > 1e-10 else 0.0
            print(f"| {sys_name:<8} | {mode.upper():<9} | {s2_ideal:<12.6f} | {s2_uhf:<12.6f} | "
                  f"{ds2_uhf:<12.6f} | {s2_omp2:<12.6f} | {ds2_omp2:<12.6f} | {red_pct:<10.2f}% |")

    print("\n" + "=" * 100)
    print("  3. TABEL PERBANDINGAN PENGOPTIMAL ORBITAL: TRUST-REGION SOSCF vs L-BFGS (OMP3, DF, cc-pVTZ)")
    print("=" * 100)
    print(f"| {'Sistem':<8} | {'Iter SOSCF':<10} | {'Waktu SOSCF (s)':<15} | {'Energi SOSCF (Ha)':<18} | {'Iter L-BFGS':<11} | {'Waktu L-BFGS (s)':<16} | {'Energi L-BFGS (Ha)':<18} |")
    print("|" + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 17 + "|" + "-" * 20 + "|" + "-" * 13 + "|" + "-" * 18 + "|" + "-" * 20 + "|")

    for sys_name in ["Neon", "H2O", "Lithium", "Carbon", "Oxygen"]:
        _, atoms, chg, mult, ref = sys_dict[sys_name]
        mol = mshqc.Molecule(charge=chg, multiplicity=mult)
        for sym, x, y, z in atoms:
            mol.add_atom(sym, x, y, z, units="angstrom")
        basis = mshqc.BasisSet("cc-pVTZ", mol)
        integrals = mshqc.IntegralEngine(mol, basis)
        scf_conf = mshqc.SCFConfig(eri_method="df", threshold=1e-9)
        scf_res = (mshqc.RHF(mol, basis, integrals, scf_conf).compute()
                   if ref == "rhf" else
                   mshqc.UHF(mol, basis, integrals, scf_conf).compute())

        conf_soscf = mshqc.MP2Config(eri_method="df", threshold=1e-9, gradient_threshold=1e-5, opt_method="soscf")
        t0 = time.perf_counter()
        res_soscf = mshqc.OMP3(mol, basis, integrals, scf_res, conf_soscf).compute()
        t_soscf = time.perf_counter() - t0

        scf_res2 = (mshqc.RHF(mol, basis, integrals, scf_conf).compute()
                    if ref == "rhf" else
                    mshqc.UHF(mol, basis, integrals, scf_conf).compute())
        conf_lbfgs = mshqc.MP2Config(eri_method="df", threshold=1e-9, gradient_threshold=1e-5, opt_method="lbfgs")
        t0 = time.perf_counter()
        res_lbfgs = mshqc.OMP3(mol, basis, integrals, scf_res2, conf_lbfgs).compute()
        t_lbfgs = time.perf_counter() - t0

        print(f"| {sys_name:<8} | {res_soscf.iterations:<10d} | {t_soscf:<15.4f} | {res_soscf.e_total:<18.8f} | "
              f"{res_lbfgs.iterations:<11d} | {t_lbfgs:<16.4f} | {res_lbfgs.e_total:<18.8f} |")

    print("\n" + "=" * 100)
    print("  4. UJI FINITE-DIFFERENCE vs GRADIEN ANALITIK (print_level=3)")
    print("=" * 100)
    for sys_name in ["Neon", "H2O", "Lithium", "Carbon", "Oxygen"]:
        _, atoms, chg, mult, ref = sys_dict[sys_name]
        mol = mshqc.Molecule(charge=chg, multiplicity=mult)
        for sym, x, y, z in atoms:
            mol.add_atom(sym, x, y, z, units="angstrom")
        basis = mshqc.BasisSet("cc-pVTZ", mol)
        integrals = mshqc.IntegralEngine(mol, basis)
        scf_conf = mshqc.SCFConfig(eri_method="exact", threshold=1e-9)
        scf_res = (mshqc.RHF(mol, basis, integrals, scf_conf).compute()
                   if ref == "rhf" else
                   mshqc.UHF(mol, basis, integrals, scf_conf).compute())
        print(f"\n>>> Sistem: {sys_name} (OMP2 & OMP3 Finite-Difference Check) <<<")
        fd_conf = mshqc.MP2Config(eri_method="exact", threshold=1e-9, gradient_threshold=1e-5, print_level=3, max_iterations=1)
        mshqc.OMP2(mol, basis, integrals, scf_res, fd_conf).compute()
        scf_res_b = (mshqc.RHF(mol, basis, integrals, scf_conf).compute()
                     if ref == "rhf" else
                     mshqc.UHF(mol, basis, integrals, scf_conf).compute())
        mshqc.OMP3(mol, basis, integrals, scf_res_b, fd_conf).compute()


def run_peak_ram_comparison(mshqc_py, psi4_py=None):
    print("\n" + "=" * 100)
    print("  5. PENGUKURAN PEAK RAM AKTUAL (VmHWM / Peak RSS dalam MB) PER PROSES")
    print("=" * 100)
    cases = [
        ("H2O", "cc-pVDZ", "exact", "OMP3"),
        ("H2O", "cc-pVTZ", "exact", "OMP3"),
        ("H2O", "cc-pVTZ", "cholesky", "OMP3"),
        ("H2O", "cc-pVTZ", "df", "OMP3"),
        ("H2O", "cc-pVQZ", "df", "OMP3"),
        ("Neon", "cc-pVTZ", "exact", "OMP3"),
        ("Neon", "cc-pVTZ", "df", "OMP3"),
        ("Neon", "cc-pVQZ", "df", "OMP3"),
        ("Oxygen", "cc-pVTZ", "exact", "OMP3"),
        ("Oxygen", "cc-pVTZ", "cholesky", "OMP3"),
        ("Oxygen", "cc-pVTZ", "df", "OMP3"),
    ]
    print(f"| {'Sistem':<8} | {'Basis':<8} | {'Mode ERI':<9} | {'Metode':<6} | {'mshqc Peak RSS (MB)':<20} | {'mshqc Delta RAM (MB)':<21} | {'Psi4 Peak RSS (MB)':<19} | {'Psi4 Delta RAM (MB)':<20} |")
    print("|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 11 + "|" + "-" * 8 + "|" + "-" * 22 + "|" + "-" * 23 + "|" + "-" * 21 + "|" + "-" * 22 + "|")

    for sys_name, b_name, eri_mode, method in cases:
        m_res = spawn_mem_probe(mshqc_py, "mshqc", sys_name, b_name, eri_mode, method)
        if psi4_py:
            try:
                p_res = spawn_mem_probe(psi4_py, "psi4", sys_name, b_name, eri_mode, method)
                p_peak = f"{p_res['peak_hwm_mb']:.2f}"
                p_delta = f"{p_res['delta_peak_mb']:.2f}"
            except Exception:
                p_peak, p_delta = "N/A", "N/A"
        else:
            p_peak, p_delta = "-", "-"

        print(f"| {sys_name:<8} | {b_name:<8} | {eri_mode.upper():<9} | {method:<6} | "
              f"{m_res['peak_hwm_mb']:<20.2f} | {m_res['delta_peak_mb']:<21.2f} | "
              f"{p_peak:<19} | {p_delta:<20} |")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _, _, eng, s_name, b_name, e_mode, meth = sys.argv
        run_single_memory_worker(eng, s_name, b_name, e_mode, meth)
    else:
        run_mshqc_suite()
        psi4_python = os.environ.get("PSI4_PYTHON", None)
        run_peak_ram_comparison(sys.executable, psi4_python)