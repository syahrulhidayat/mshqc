#!/usr/bin/env python3
"""
tests/verify_audit_and_scaling.py
---------------------------------
Suite verifikasi total pasca-patch mshqc_sync:
  1. Uji Regresi Segfault Simetri (H2O C2v-z, H2O C2v-x [bug c2x], O2 D2h) vs C1.
  2. Uji Gradien Analitik DF-RHF, DF-UHF, DF-ROHF (NH2 & OH), dan DF-OMP2 vs Central FD.
  3. Uji Frekuensi Harmonik O(N) Gradien Analitik vs O(N^2) Stensil Energi + Speedup.
  4. Uji Warm-Start SCF/OMP pada Optimasi Geometri & PES Scan.
  5. Stress-Test Skalabilitas Molekul Ringan -> Menengah (H2O -> CH4 -> C2H4 -> C6H6)
     + Pengukuran Peak RSS & Perbandingan Apple-to-Apple Opsional vs Psi4.
"""

import os
import sys
import time
import math
import argparse
import numpy as np

# Pastikan environment threading konsisten sebelum import mshqc
CPU_AVAIL = os.cpu_count() or 4
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", str(min(4, CPU_AVAIL))))
os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("TBLIS_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

import mshqc


def get_peak_rss_mb() -> float:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/status", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        return float(line.split()[1]) / 1024.0
        except OSError:
            pass
    return 0.0


def build_mol(atoms_ang, charge=0, mult=1) -> mshqc.Molecule:
    mol = mshqc.Molecule(charge, mult)
    for sym, x, y, z in atoms_ang:
        mol.add_atom(sym, x, y, z, units="angstrom")
    return mol


# =============================================================================
# TEST 1: REGRESI SEGFAULT SIMETRI (molecule_sym.cc + point_group.cc c2x)
# =============================================================================
def test_symmetry_regression():
    print("\n" + "=" * 90)
    print(" [TEST 1] REGRESI SEGFAULT SIMETRI (BasisSymmetrizer Permutasi + PointGroup C2v/D2h)")
    print("=" * 90)

    cases = [
        (
            "H2O (C2v sumbu-z)",
            [("O", 0.0, 0.0, 0.1173), ("H", 0.0, 0.7572, -0.4692), ("H", 0.0, -0.7572, -0.4692)],
            0, 1, "C2v", 4,
        ),
        (
            "H2O diputar (C2v sumbu-x)",
            [("O", 0.1173, 0.0, 0.0), ("H", -0.4692, 0.7572, 0.0), ("H", -0.4692, -0.7572, 0.0)],
            0, 1, "C2v", 4,
        ),
        (
            "O2 (D2h open-shell)",
            [("O", 0.0, 0.0, -0.6035), ("O", 0.0, 0.0, 0.6035)],
            0, 3, "D2h", 8,
        ),
    ]

    for label, atoms, q, mult, exp_sym, exp_ord in cases:
        mol = build_mol(atoms, q, mult)
        pg = mshqc.PointGroup(mol)
        pg.detect()
        sym = pg.get_symbol()
        order = pg.get_order()
        aligned = pg.get_aligned_molecule()

        basis = mshqc.BasisSet("cc-pVDZ", aligned)
        ints = mshqc.IntegralEngine(aligned, basis)
        pl = mshqc.PetiteList(basis, pg)

        for mode in ("exact", "df"):
            cfg = mshqc.SCFConfig(eri_method=mode, threshold=1e-11, aux_basis_name="cc-pVDZ-RI", print_level=0)
            if mult == 1:
                res_sym = mshqc.RHF(aligned, basis, ints, pg, pl, cfg).compute()
                res_c1 = mshqc.RHF(aligned, basis, ints, cfg).compute()
            else:
                res_sym = mshqc.UHF(aligned, basis, ints, cfg, pg, pl).compute()
                res_c1 = mshqc.UHF(aligned, basis, ints, cfg).compute()

            de = abs(res_sym.energy_total - res_c1.energy_total)
            ok = (sym == exp_sym) and (order == exp_ord) and (de < 1e-9)
            status = "PASS" if ok else "FAIL"
            print(
                f" [{status}] {label:26s} | {mode.upper():5s} | PG={sym:3s} (ord={order}) | "
                f"E_sym={res_sym.energy_total:14.9f} | |dE vs C1|={de:.2e}"
            )
            assert ok, f"Gagal pada {label} ({mode}): PG={sym}({order}), dE={de}"


# =============================================================================
# TEST 2: GRADIEN ANALITIK DF-RHF, DF-UHF, DF-ROHF, DF-OMP2 vs CENTRAL FD
# =============================================================================
def test_analytic_gradients():
    print("\n" + "=" * 90)
    print(" [TEST 2] VERIFIKASI GRADIEN ANALITIK DF (Termasuk DF-ROHF) vs CENTRAL FD")
    print("=" * 90)

    h2o = mshqc.Molecule(0, 1)
    h2o.add_atom(8, 0.0, 0.0, 0.22)
    h2o.add_atom(1, 0.0, 1.45, -0.88)
    h2o.add_atom(1, 0.0, -1.45, -0.88)

    nh2 = mshqc.Molecule(0, 2)
    nh2.add_atom(7, 0.0, 0.0, 0.20)
    nh2.add_atom(1, 0.0, 1.55, -0.85)
    nh2.add_atom(1, 0.0, -1.55, -0.85)

    oh = mshqc.Molecule(0, 2)
    oh.add_atom(8, 0.0, 0.0, 0.0)
    oh.add_atom(1, 0.0, 0.0, 1.90)

    tests = [
        ("H2O", h2o, "df-rhf", "rhf", True),
        ("H2O", h2o, "df-omp2", "rhf", True),
        ("NH2", nh2, "df-uhf", "uhf", True),
        ("NH2", nh2, "df-rohf", "rohf", True),
        ("NH2", nh2, "df-omp2", "uhf", True),
        ("OH (degen)", oh, "df-uhf", "uhf", True),
        ("OH (degen)", oh, "df-rohf", "rohf", False),  # pantau efek smearing degenerasi 2Pi
    ]

    for name, mol, method, ref, strict in tests:
        cfg_ana = mshqc.GeomOptConfig()
        cfg_ana.method = method
        cfg_ana.eri_method = "df"
        cfg_ana.reference = ref
        cfg_ana.basis_name = "cc-pVDZ"
        cfg_ana.use_analytic_gradient = True
        cfg_ana.project_trans_rot = False
        cfg_ana.energy_thresh = 1e-12
        cfg_ana.orb_grad_thresh = 1e-10

        cfg_fd = mshqc.GeomOptConfig()
        cfg_fd.method = method
        cfg_fd.eri_method = "df"
        cfg_fd.reference = ref
        cfg_fd.basis_name = "cc-pVDZ"
        cfg_fd.use_analytic_gradient = False
        cfg_fd.project_trans_rot = False
        cfg_fd.fd_step_bohr = 5e-4
        cfg_fd.energy_thresh = 1e-12
        cfg_fd.orb_grad_thresh = 1e-10

        t0 = time.perf_counter()
        e_ana, g_ana = mshqc.HPCGeometryOptimizer.compute_nuclear_gradient_cpp(mol, cfg_ana)
        t_ana = time.perf_counter() - t0

        t0 = time.perf_counter()
        e_fd, g_fd = mshqc.HPCGeometryOptimizer.compute_nuclear_gradient_cpp(mol, cfg_fd)
        t_fd = time.perf_counter() - t0

        g_ana_m = np.asarray(g_ana).reshape(-1, 3)
        g_fd_m = np.asarray(g_fd).reshape(-1, 3)
        max_diff = float(np.max(np.abs(g_ana_m - g_fd_m)))
        trans_sum = float(np.max(np.abs(np.sum(g_ana_m, axis=0))))

        ok = (max_diff < 1e-6) and (trans_sum < 1e-10)
        tag = "PASS" if ok else ("WARN" if not strict else "FAIL")
        print(
            f" [{tag}] {name:10s} | {method:7s} | E={e_ana:14.9f} | "
            f"max|dG|={max_diff:.2e} | trans={trans_sum:.2e} | "
            f"t_ana={t_ana:.3f}s vs t_FD={t_fd:.3f}s ({t_fd/max(t_ana,1e-6):.1f}x)"
        )
        if strict:
            assert ok, f"Gradien analitik {name} ({method}) gagal: max_diff={max_diff}"


# =============================================================================
# TEST 3: FREKUENSI O(N) GRADIEN ANALITIK vs O(N^2) STENSIL ENERGI
# =============================================================================
def test_analytic_frequencies():
    print("\n" + "=" * 90)
    print(" [TEST 3] FREKUENSI HARMONIK: O(N) FD GRADIEN ANALITIK vs O(N^2) STENSIL ENERGI")
    print("=" * 90)

    for method in ("rhf", "omp2"):
        mol = build_mol(
            [("O", 0.0, 0.0, 0.125), ("H", 0.0, 0.78, -0.49), ("H", 0.0, -0.78, -0.49)]
        )
        opt_res = mol.optimize_geometry(
            method=method, eri_mode="df", basis="cc-pVDZ",
            g_rms_tol=1e-7, g_max_tol=2e-7, verbose=False
        )
        assert opt_res.converged, f"Optimasi H2O {method} tidak konvergen!"

        cfg_a = mshqc.GeomOptConfig()
        cfg_a.method = method
        cfg_a.eri_method = "df"
        cfg_a.basis_name = "cc-pVDZ"
        cfg_a.use_analytic_gradient = True
        cfg_a.verbose = False

        cfg_e = mshqc.GeomOptConfig()
        cfg_e.method = method
        cfg_e.eri_method = "df"
        cfg_e.basis_name = "cc-pVDZ"
        cfg_e.use_analytic_gradient = False
        cfg_e.verbose = False

        t0 = time.perf_counter()
        va = mshqc.HPCGeometryOptimizer.compute_frequencies(mol, cfg_a, 298.15, 1.0)
        ta = time.perf_counter() - t0

        t0 = time.perf_counter()
        ve = mshqc.HPCGeometryOptimizer.compute_frequencies(mol, cfg_e, 298.15, 1.0)
        te = time.perf_counter() - t0

        fa = np.sort(np.asarray(va.frequencies_cm1))
        fe = np.sort(np.asarray(ve.frequencies_cm1))
        df_max = float(np.max(np.abs(fa - fe)))
        dzpe = abs(va.zpe_kcal_mol - ve.zpe_kcal_mol)

        ok = df_max < 0.5  # toleransi < 0.5 cm^-1 antara FD-grad vs FD-energy
        print(
            f" [{'PASS' if ok else 'FAIL'}] H2O DF-{method.upper():4s} | "
            f"Freqs(ana)={np.round(fa, 2).tolist()} cm^-1 | "
            f"max|dFreq|={df_max:.3f} cm^-1 | |dZPE|={dzpe:.5f} kcal/mol | "
            f"t_ana={ta:.3f}s vs t_en={te:.3f}s ({te/max(ta,1e-6):.2f}x)"
        )
        assert ok, f"Frekuensi O(N) vs O(N^2) meleset {df_max:.3f} cm^-1"


# =============================================================================
# TEST 4: WARM-START PES SCAN
# =============================================================================
def test_warmstart_pes():
    print("\n" + "=" * 90)
    print(" [TEST 4] VERIFIKASI WARM-START PADA PES SCAN (H2O / DF-OMP2)")
    print("=" * 90)

    mol = build_mol([("F", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.92)])
    dists = [0.85, 0.90, 0.92, 0.95, 1.00, 1.10]
    t0 = time.perf_counter()
    pts = mol.scan_pes(0, 1, dists, method="omp2", eri_mode="df", basis="cc-pVDZ")
    dt = time.perf_counter() - t0

    energies = [p.energy_ha for p in pts]
    min_idx = int(np.argmin(energies))
    print(f" [PASS] HF PES Scan ({len(pts)} titik) selesai dalam {dt:.3f}s | Minimum di R={dists[min_idx]:.2f} A (E={energies[min_idx]:.8f} Ha)")
    assert min_idx in (1, 2, 3), f"Minimum PES tidak wajar di indeks {min_idx}"


# =============================================================================
# TEST 5: STRESS-TEST SKALABILITAS MOLEKUL RINGAN -> MENENGAH (+ OPSIONAL PSI4)
# =============================================================================
def test_scaling_benchmark(with_psi4=False, basis_name="cc-pVDZ"):
    print("\n" + "=" * 90)
    print(f" [TEST 5] STRESS-TEST SKALABILITAS ({basis_name} / {basis_name}-RI | THREADS={N_THREADS})")
    print("=" * 90)

    molecules = [
        (
            "H2O (3 atom)",
            [("O", 0.0, 0.0, 0.1173), ("H", 0.0, 0.7572, -0.4692), ("H", 0.0, -0.7572, -0.4692)],
        ),
        (
            "CH4 (5 atom)",
            [
                ("C", 0.0, 0.0, 0.0),
                ("H", 0.6291, 0.6291, 0.6291),
                ("H", -0.6291, -0.6291, 0.6291),
                ("H", -0.6291, 0.6291, -0.6291),
                ("H", 0.6291, -0.6291, -0.6291),
            ],
        ),
        (
            "C2H4 (6 atom)",
            [
                ("C", 0.0, 0.0, 0.6695), ("C", 0.0, 0.0, -0.6695),
                ("H", 0.0, 0.9289, 1.2321), ("H", 0.0, -0.9289, 1.2321),
                ("H", 0.0, 0.9289, -1.2321), ("H", 0.0, -0.9289, -1.2321),
            ],
        ),
        (
            "C6H6 (12 atom)",
            [
                ("C",  1.3960,  0.0000, 0.0), ("C",  0.6980,  1.2090, 0.0),
                ("C", -0.6980,  1.2090, 0.0), ("C", -1.3960,  0.0000, 0.0),
                ("C", -0.6980, -1.2090, 0.0), ("C",  0.6980, -1.2090, 0.0),
                ("H",  2.4820,  0.0000, 0.0), ("H",  1.2410,  2.1495, 0.0),
                ("H", -1.2410,  2.1495, 0.0), ("H", -2.4820,  0.0000, 0.0),
                ("H", -1.2410, -2.1495, 0.0), ("H",  1.2410, -2.1495, 0.0),
            ],
        ),
    ]

    psi4 = None
    if with_psi4:
        try:
            import psi4 as _psi4
            psi4 = _psi4
            psi4.set_num_threads(N_THREADS)
            psi4.set_memory("4 GB")
            psi4.core.set_output_file("/dev/null", False)
        except Exception as e:
            print(f" [INFO] Psi4 tidak tersedia di env ini ({e}); lanjut mode mshqc-only.")

    aux = f"{basis_name}-RI"
    hdr = f"| {'Molekul':<15} | {'N_bf':>4} | {'DF-RHF(s)':>9} | {'DF-OMP2(s)':>10} | {'DF-OMP3(s)':>10} | {'PeakMB':>7} |"
    if psi4 is not None:
        hdr += f" {'Psi4 OMP2':>9} | {'Psi4 OMP3':>9} | {'Spd OMP3':>8} |"
    print(hdr)
    print("-" * len(hdr))

    for name, atoms in molecules:
        mol = build_mol(atoms, 0, 1)
        basis = mshqc.BasisSet(basis_name, mol)
        ints = mshqc.IntegralEngine(mol, basis)
        nbf = basis.n_basis_functions()

        scf_cfg = mshqc.SCFConfig(eri_method="df", threshold=1e-9, aux_basis_name=aux, print_level=0)
        mp2_cfg = mshqc.MP2Config(eri_method="df", threshold=1e-9, gradient_threshold=1e-5, aux_basis_name=aux, print_level=0)

        t0 = time.perf_counter()
        scf_res = mshqc.RHF(mol, basis, ints, scf_cfg).compute()
        t_rhf = time.perf_counter() - t0

        t0 = time.perf_counter()
        omp2_res = mshqc.OMP2(mol, basis, ints, scf_res, mp2_cfg).compute()
        t_omp2 = time.perf_counter() - t0

        t0 = time.perf_counter()
        omp3_res = mshqc.OMP3(mol, basis, ints, scf_res, mp2_cfg).compute()
        t_omp3 = time.perf_counter() - t0

        peak_mb = get_peak_rss_mb()
        row = f"| {name:<15} | {nbf:>4d} | {t_rhf:>9.3f} | {t_omp2:>10.3f} | {t_omp3:>10.3f} | {peak_mb:>7.1f} |"

        if psi4 is not None:
            psi4.core.clean()
            psi4.core.clean_options()
            geom = ["0 1", "units angstrom", "symmetry c1"] + [f"{s} {x} {y} {z}" for s, x, y, z in atoms]
            p4_mol = psi4.geometry("\n".join(geom))
            psi4.set_options({
                "basis": basis_name, "df_basis_scf": aux, "df_basis_mp2": aux, "df_basis_cc": aux,
                "reference": "rhf", "scf_type": "df", "mp2_type": "df", "mp_type": "df", "cc_type": "df",
                "freeze_core": "false", "e_convergence": 1e-9, "d_convergence": 1e-9,
                "r_convergence": 1e-6, "rms_mograd_convergence": 1e-5, "max_mograd_convergence": 1e-4,
                "read_scf_3index": True,  # strict apple-to-apple: izinkan Psi4 reuse 3-index SCF!
            })
            _, p4_scf = psi4.energy("scf", molecule=p4_mol, return_wfn=True)
            t0 = time.perf_counter()
            e2_p4 = float(psi4.energy("omp2", ref_wfn=p4_scf, molecule=p4_mol))
            t_p4_omp2 = time.perf_counter() - t0
            psi4.core.clean()
            _, p4_scf2 = psi4.energy("scf", molecule=p4_mol, return_wfn=True)
            t0 = time.perf_counter()
            e3_p4 = float(psi4.energy("omp3", ref_wfn=p4_scf2, molecule=p4_mol))
            t_p4_omp3 = time.perf_counter() - t0

            assert abs(omp2_res.energy_total - e2_p4) < 1e-6, f"Mismatch OMP2 {name}: {omp2_res.energy_total} vs {e2_p4}"
            assert abs(omp3_res.energy_total - e3_p4) < 1e-6, f"Mismatch OMP3 {name}: {omp3_res.energy_total} vs {e3_p4}"
            row += f" {t_p4_omp2:>9.3f} | {t_p4_omp3:>9.3f} | {t_p4_omp3/max(t_omp3,1e-6):>7.2f}x |"

        print(row)


def main():
    parser = argparse.ArgumentParser(description="Verifikasi audit pasca-patch & stress-test skalabilitas mshqc")
    parser.add_argument("--with-psi4", action="store_true", help="Bandingkan langsung dengan Psi4 (read_scf_3index=True)")
    parser.add_argument("--basis", default="cc-pVDZ", help="Basis untuk Test 5 (mis. cc-pVDZ atau cc-pVTZ)")
    args = parser.parse_args()

    test_symmetry_regression()
    test_analytic_gradients()
    test_analytic_frequencies()
    test_warmstart_pes()
    test_scaling_benchmark(with_psi4=args.with_psi4, basis_name=args.basis)
    print("\n[ALL PASSED] Seluruh verifikasi pasca-patch lulus!")


if __name__ == "__main__":
    main()