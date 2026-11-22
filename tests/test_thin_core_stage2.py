#!/usr/bin/env python3
"""
Validasi Tahap 2 Thin-Core (Fused Kernels, Batched BLAS-3, Strict Frozen-Core, Canonical MP2/MP3):
1. Fused `transform_3c_mo_blocks` (RHF & UHF) cocok < 1e-14 terhadap `transform_3c_mo`
2. Canonical DF-RMP2/RMP3 & DF-UMP2/UMP3 hibrida cocok < 1e-8 Ha vs legacy C++
3. Strict Frozen-Core DF-OMP2 & DF-OMP3 (H2O nf=1, OH radikal nf=1) cocok < 1e-8 Ha vs legacy C++
4. Benchmark Head-to-Head Benzene C6H6 (114 BF, cc-pVDZ, freeze_core=True):
   Thin-Core Hybrid Python vs Full Legacy C++
"""
import time
import numpy as np
import mshqc
from mshqc import kernels as K
from mshqc import methods as M


def _e(res) -> float:
    return float(getattr(res, "energy_total", getattr(res, "e_total")))


def test_fused_3c_mo_blocks_parity():
    mol = mshqc.Molecule(charge=0, multiplicity=2)
    mol.add_atom("O", 0.0, 0.0, 0.0, units="angstrom")
    mol.add_atom("H", 0.0, 0.0, 0.9697, units="angstrom")
    basis = mshqc.BasisSet("cc-pVDZ", mol)
    ints = mshqc.IntegralEngine(mol, basis)
    scf_res = mshqc.UHF(mol, basis, ints, mshqc.SCFConfig(eri_method="df", threshold=1e-10)).compute()

    nbf = basis.n_basis_functions()
    na, nb = scf_res.n_occ_alpha, scf_res.n_occ_beta
    B_ao = scf_res.L_mat
    Ca_o, Ca_v = np.ascontiguousarray(scf_res.C_alpha[:, :na]), np.ascontiguousarray(scf_res.C_alpha[:, na:])
    Cb_o, Cb_v = np.ascontiguousarray(scf_res.C_beta[:, :nb]), np.ascontiguousarray(scf_res.C_beta[:, nb:])

    Boo, Bov, Bvv = K.transform_3c_mo_blocks(B_ao, Ca_o, Ca_v, nbf)
    assert np.allclose(Boo, K.transform_3c_mo(B_ao, Ca_o, Ca_o, nbf), atol=1e-14)
    assert np.allclose(Bov, K.transform_3c_mo(B_ao, Ca_o, Ca_v, nbf), atol=1e-14)
    assert np.allclose(Bvv, K.transform_3c_mo(B_ao, Ca_v, Ca_v, nbf), atol=1e-14)

    Boo_a, Bov_a, Bvv_a, Boo_b, Bov_b, Bvv_b = K.transform_3c_mo_blocks_uhf(
        B_ao, Ca_o, Ca_v, Cb_o, Cb_v, nbf
    )
    assert np.allclose(Boo_a, Boo, atol=1e-14)
    assert np.allclose(Bov_a, Bov, atol=1e-14)
    assert np.allclose(Bvv_a, Bvv, atol=1e-14)
    assert np.allclose(Boo_b, K.transform_3c_mo(B_ao, Cb_o, Cb_o, nbf), atol=1e-14)
    assert np.allclose(Bov_b, K.transform_3c_mo(B_ao, Cb_o, Cb_v, nbf), atol=1e-14)
    assert np.allclose(Bvv_b, K.transform_3c_mo(B_ao, Cb_v, Cb_v, nbf), atol=1e-14)
    print("[FUSED MO ] RHF & UHF 1-pass AO->MO blocks match reference < 1e-14.")


def test_frozen_core_rhf_and_uhf_parity():
    # 1. Closed-shell H2O (n_frozen=1)
    h2o = "O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692"
    calc_r = mshqc.MSHQCCalculator(
        h2o, basis_name="cc-pVDZ", eri_method="df", units="angstrom",
        threshold=1e-10, freeze_core=True, use_symmetry=False,
    )
    assert calc_r.n_frozen == 1
    scf_r = calc_r.run_scf(use_hybrid=False)
    for name, fn in (
        ("RMP2(FC)", lambda h: calc_r.run_mp2(scf_r, use_hybrid=h)),
        ("RMP3(FC)", lambda h: calc_r.run_mp3(scf_r, use_hybrid=h)),
        ("OMP2(FC)", lambda h: calc_r.run_omp2(scf_r, grad_thresh=1e-6, use_hybrid=h)),
        ("OMP3(FC)", lambda h: calc_r.run_omp3(scf_r, grad_thresh=1e-6, use_hybrid=h)),
    ):
        r_cpp = fn(False)
        r_hyb = fn(True)
        de = abs(_e(r_hyb) - _e(r_cpp))
        print(f"[H2O {name:8s}] Hybrid={_e(r_hyb):.10f} | C++={_e(r_cpp):.10f} | |dE|={de:.2e}")
        assert de < 1e-8, f"Mismatch {name}: {de}"

    # 2. Open-shell OH radical (n_frozen=1)
    oh = "O 0.0 0.0 0.0\nH 0.0 0.0 0.9697"
    calc_u = mshqc.MSHQCCalculator(
        oh, basis_name="cc-pVDZ", eri_method="df", units="angstrom",
        multiplicity=2, reference="uhf", threshold=1e-10,
        freeze_core=True, use_symmetry=False,
    )
    assert calc_u.n_frozen == 1
    scf_u = calc_u.run_scf(use_hybrid=False)
    for name, fn in (
        ("UMP2(FC)", lambda h: calc_u.run_mp2(scf_u, use_hybrid=h)),
        ("UMP3(FC)", lambda h: calc_u.run_mp3(scf_u, use_hybrid=h)),
        ("UOMP2(FC)", lambda h: calc_u.run_omp2(scf_u, grad_thresh=1e-6, use_hybrid=h)),
        ("UOMP3(FC)", lambda h: calc_u.run_omp3(scf_u, grad_thresh=1e-6, use_hybrid=h)),
    ):
        r_cpp = fn(False)
        r_hyb = fn(True)
        de = abs(_e(r_hyb) - _e(r_cpp))
        print(f"[OH  {name:8s}] Hybrid={_e(r_hyb):.10f} | C++={_e(r_cpp):.10f} | |dE|={de:.2e}")
        assert de < 1e-8, f"Mismatch {name}: {de}"


def test_benzene_114bf_frozen_core_speed_and_parity():
    benzene = [
        ("C", 1.3960, 0.0000, 0.0), ("C", 0.6980, 1.2090, 0.0),
        ("C", -0.6980, 1.2090, 0.0), ("C", -1.3960, 0.0000, 0.0),
        ("C", -0.6980, -1.2090, 0.0), ("C", 0.6980, -1.2090, 0.0),
        ("H", 2.4820, 0.0000, 0.0), ("H", 1.2410, 2.1495, 0.0),
        ("H", -1.2410, 2.1495, 0.0), ("H", -2.4820, 0.0000, 0.0),
        ("H", -1.2410, -2.1495, 0.0), ("H", 1.2410, -2.1495, 0.0),
    ]
    calc = mshqc.MSHQCCalculator(
        benzene, basis_name="cc-pVDZ", eri_method="df", units="angstrom",
        threshold=1e-9, freeze_core=True, use_symmetry=False,
    )
    assert calc.basis.n_basis_functions() == 114 and calc.n_frozen == 6

    t0 = time.perf_counter()
    scf_cpp = calc.run_scf(use_hybrid=False)
    t_scf_cpp = time.perf_counter() - t0

    H = calc.integrals.compute_kinetic() + calc.integrals.compute_nuclear()
    S = calc.integrals.compute_overlap()
    v_nn = calc.molecule.nuclear_repulsion_energy()

    t1 = time.perf_counter()
    scf_hyb = M.run_df_rhf(
        H, S, v_nn, scf_cpp.L_mat, scf_cpp.n_occ_alpha,
        energy_tol=1e-9, density_tol=1e-9,
    )
    t_scf_hyb = time.perf_counter() - t1

    rows = [
        ("RHF*", _e(scf_hyb), _e(scf_cpp), abs(_e(scf_hyb) - _e(scf_cpp)),
         t_scf_hyb, t_scf_cpp, t_scf_cpp / max(t_scf_hyb, 1e-6)),
    ]
    assert rows[0][3] < 1e-8

    mp2_cpp_holder = {}
    for label, fn_cpp, fn_hyb in (
        (
            "RMP2",
            lambda: mp2_cpp_holder.setdefault("r", calc.run_mp2(scf_cpp, use_hybrid=False)),
            lambda: calc.run_mp2(scf_cpp, use_hybrid=True),
        ),
        (
            "RMP3",
            lambda: calc.run_mp3(scf_cpp, mp2_cpp_holder["r"], use_hybrid=False),
            lambda: calc.run_mp3(scf_cpp, use_hybrid=True),
        ),
        (
            "OMP2",
            lambda: calc.run_omp2(scf_cpp, grad_thresh=1e-5, use_hybrid=False),
            lambda: calc.run_omp2(scf_cpp, grad_thresh=1e-5, use_hybrid=True),
        ),
        (
            "OMP3",
            lambda: calc.run_omp3(scf_cpp, grad_thresh=1e-5, use_hybrid=False),
            lambda: calc.run_omp3(scf_cpp, grad_thresh=1e-5, use_hybrid=True),
        ),
    ):
        tc0 = time.perf_counter()
        r_c = fn_cpp()
        tc = time.perf_counter() - tc0

        th0 = time.perf_counter()
        r_h = fn_hyb()
        th = time.perf_counter() - th0

        de = abs(_e(r_h) - _e(r_c))
        rows.append((label, _e(r_h), _e(r_c), de, th, tc, tc / max(th, 1e-6)))
        assert de < 1e-7, f"C6H6 {label} mismatch: {de}"

    print("\n=== C6H6 / cc-pVDZ (114 BF, FC=6) THIN-CORE HYBRID vs LEGACY C++ ===")
    print(f"| {'Method':<6} | {'Hybrid (Ha)':>15} | {'C++ (Ha)':>15} | {'|dE| (Ha)':>9} | {'Hyb(s)':>7} | {'C++(s)':>7} | {'Ratio':>6} |")
    print("|" + "-" * 80 + "|")
    for lbl, eh, ec, de, th, tc, ratio in rows:
        print(f"| {lbl:<6} | {eh:>15.9f} | {ec:>15.9f} | {de:>9.2e} | {th:>7.3f} | {tc:>7.3f} | {ratio:>5.2f}x |")
    print("*Catatan: RHF C++ mencakup dekomposisi DF awal; RMP2/RMP3/OMP2/OMP3 berbagi L_mat yang sama.")


if __name__ == "__main__":
    test_fused_3c_mo_blocks_parity()
    test_frozen_core_rhf_and_uhf_parity()
    test_benzene_114bf_frozen_core_speed_and_parity()
    print("[PASS] Semua validasi Tahap 2 Thin-Core lulus!")
