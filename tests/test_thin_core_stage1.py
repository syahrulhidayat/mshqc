#!/usr/bin/env python3
"""
Validasi Tahap 1 Thin-Core Hybrid Migration:
1. DF-RHF hibrida (Python PulayDIIS + C++ build_jk_3c_rhf) vs C++ mshqc.RHF
2. DF-UHF hibrida radikal OH (Python coupled PulayDIIS + C++ build_jk_3c_uhf) vs C++ mshqc.UHF
3. Opt-in routing `use_hybrid` & `MSHQC_USE_HYBRID_SOLVERS=1` pada MSHQCCalculator (RHF & UHF)
"""
import os
import time
import mshqc
from mshqc import methods as M


def _get_energy(res) -> float:
    return float(getattr(res, "energy_total", getattr(res, "e_total")))


def test_hybrid_df_rhf_parity_and_speed():
    mol = mshqc.Molecule(charge=0, multiplicity=1)
    mol.add_atom("O", 0.0, 0.0, 0.1173, units="angstrom")
    mol.add_atom("H", 0.0, 0.7572, -0.4692, units="angstrom")
    mol.add_atom("H", 0.0, -0.7572, -0.4692, units="angstrom")

    basis = mshqc.BasisSet("cc-pVTZ", mol)
    ints = mshqc.IntegralEngine(mol, basis)

    t0 = time.perf_counter()
    scf_cpp = mshqc.RHF(
        mol, basis, ints, mshqc.SCFConfig(eri_method="df", threshold=1e-10)
    ).compute()
    t_cpp = time.perf_counter() - t0

    H = ints.compute_kinetic() + ints.compute_nuclear()
    S = ints.compute_overlap()
    v_nn = mol.nuclear_repulsion_energy()
    B_ao = scf_cpp.L_mat

    t1 = time.perf_counter()
    scf_hyb = M.run_df_rhf(
        H, S, v_nn, B_ao, scf_cpp.n_occ_alpha, energy_tol=1e-10, density_tol=1e-10
    )
    t_hyb = time.perf_counter() - t1

    de = abs(scf_hyb.energy_total - scf_cpp.energy_total)
    print(
        f"[DF-RHF H2O/TZ] Hybrid={scf_hyb.energy_total:.10f} ({scf_hyb.iterations} it, {t_hyb*1e3:.1f} ms) | "
        f"C++={scf_cpp.energy_total:.10f} ({t_cpp*1e3:.1f} ms) | |dE|={de:.2e} Ha"
    )
    assert scf_hyb.converged
    assert de < 1e-9


def test_hybrid_df_uhf_parity_and_spin():
    mol = mshqc.Molecule(charge=0, multiplicity=2)
    mol.add_atom("O", 0.0, 0.0, 0.0, units="angstrom")
    mol.add_atom("H", 0.0, 0.0, 0.9697, units="angstrom")

    basis = mshqc.BasisSet("cc-pVDZ", mol)
    ints = mshqc.IntegralEngine(mol, basis)

    scf_cpp = mshqc.UHF(
        mol, basis, ints, mshqc.SCFConfig(eri_method="df", threshold=1e-10)
    ).compute()

    H = ints.compute_kinetic() + ints.compute_nuclear()
    S = ints.compute_overlap()
    v_nn = mol.nuclear_repulsion_energy()
    B_ao = scf_cpp.L_mat

    scf_hyb = M.run_df_uhf(
        H, S, v_nn, B_ao, scf_cpp.n_occ_alpha, scf_cpp.n_occ_beta,
        energy_tol=1e-10, density_tol=1e-10,
    )
    de = abs(scf_hyb.energy_total - scf_cpp.energy_total)
    print(
        f"[DF-UHF OH/DZ ] Hybrid={scf_hyb.energy_total:.10f} (<S2>={scf_hyb.s2:.6f}, {scf_hyb.iterations} it) | "
        f"C++={scf_cpp.energy_total:.10f} | |dE|={de:.2e} Ha"
    )
    assert scf_hyb.converged
    assert de < 1e-9
    assert abs(scf_hyb.s2 - 0.75) < 0.05


def test_calculator_opt_in_hybrid_routing():
    h2o = "O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692"
    calc = mshqc.MSHQCCalculator(
        h2o, basis_name="cc-pVDZ", eri_method="df", units="angstrom",
        threshold=1e-10, use_symmetry=False,
    )
    scf_cpp = calc.run_scf(use_hybrid=False)
    omp2_cpp = calc.run_omp2(scf_cpp, grad_thresh=1e-6, use_hybrid=False)
    omp3_cpp = calc.run_omp3(scf_cpp, grad_thresh=1e-6, use_hybrid=False)

    omp2_hyb = calc.run_omp2(scf_cpp, grad_thresh=1e-6, use_hybrid=True)
    omp3_hyb = calc.run_omp3(scf_cpp, grad_thresh=1e-6, use_hybrid=True)

    assert isinstance(omp2_hyb, M.HybridOMPResult)
    assert isinstance(omp3_hyb, M.HybridOMPResult)
    assert omp2_hyb.converged and omp3_hyb.converged
    assert abs(omp2_hyb.energy_total - _get_energy(omp2_cpp)) < 1e-8
    assert abs(omp3_hyb.energy_total - _get_energy(omp3_cpp)) < 1e-8

    prev_env = os.environ.get("MSHQC_USE_HYBRID_SOLVERS")
    try:
        os.environ["MSHQC_USE_HYBRID_SOLVERS"] = "1"
        scf_env = calc.run_scf()
        omp2_env = calc.run_omp2(scf_env, grad_thresh=1e-6)
        assert isinstance(scf_env, M.HybridSCFResult)
        assert isinstance(omp2_env, M.HybridOMPResult)
        assert abs(omp2_env.energy_total - _get_energy(omp2_cpp)) < 1e-8
    finally:
        if prev_env is None:
            os.environ.pop("MSHQC_USE_HYBRID_SOLVERS", None)
        else:
            os.environ["MSHQC_USE_HYBRID_SOLVERS"] = prev_env
    print("[ROUTING PASS ] Opt-in use_hybrid & MSHQC_USE_HYBRID_SOLVERS=1 terverifikasi.")


if __name__ == "__main__":
    test_hybrid_df_rhf_parity_and_speed()
    test_hybrid_df_uhf_parity_and_spin()
    test_calculator_opt_in_hybrid_routing()
    print("[PASS] Semua validasi Tahap 1 Thin-Core Hybrid lulus!")
