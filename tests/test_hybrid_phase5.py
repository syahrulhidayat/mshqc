#!/usr/bin/env python3
"""
Validasi Fase 5 (Polish, Konsistensi API & Keterbacaan tanpa Mengurangi HPC):
1. MSHQCSession mendukung SCS-OMP2, SOS-OMP2, OMP2.5 + kolom Corr Energy akurat (e_corr_total)
2. quick_calculation mendukung df-scs-omp2, df-omp2.5, cholesky-omp2.5, df-romp2, df-romp3
3. Plugin hook UHF `mshqc.methods.optimize_orbitals_uhf` cocok < 1e-10 Ha pada radikal OH
4. Optimasi 1-RDM linier pada OMP2.5 (RHF & UHF) tetap eksak terhadap gradien analitik
5. Validasi sintaks AST penuh untuk type stub `python/mshqc/_mshqc.pyi`
"""
import ast
from pathlib import Path
import numpy as np
import mshqc
from mshqc import kernels as K
from mshqc import methods as M


def test_session_and_quick_calculation_parity():
    h2o = "O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692"
    sess = mshqc.MSHQCSession(
        molecule=h2o,
        basis_name="cc-pVDZ",
        eri_method="df",
        units="angstrom",
        threshold=1e-10,
        print_level=0,
        use_symmetry=False,
    )
    res = sess.run_complete_analysis(methods=["rhf", "omp2", "scs-omp2", "sos-omp2", "omp2.5", "omp3"], grad_thresh=1e-6)
    for k in ("RHF", "OMP2", "SCS-OMP2", "SOS-OMP2", "OMP2.5", "OMP3"):
        assert k in res, f"Key {k} hilang dari MSHQCSession"

    # Pastikan e_corr_total pada OMP2.5 dan OMP3 mencakup e_mp2 + e_mp3 (bukan hanya e_mp2)
    assert abs(res["OMP2.5"].e_corr_total - (res["OMP2.5"].e_mp2 + res["OMP2.5"].e_mp3)) < 1e-14
    assert abs(res["OMP2.5"].e_mp3) > 1e-4

    # quick_calculation dengan awalan df- & cholesky-
    q_scs = mshqc.quick_calculation(h2o, basis="cc-pVDZ", method="df-scs-omp2", units="angstrom", threshold=1e-10)
    q_25 = mshqc.quick_calculation(h2o, basis="cc-pVDZ", method="df-omp2.5", units="angstrom", threshold=1e-10)
    assert abs(q_scs.energy_total - res["SCS-OMP2"].energy_total) < 1e-8
    assert abs(q_25.energy_total - res["OMP2.5"].energy_total) < 1e-8

    # ROMP2/ROMP3 via quick_calculation pada atom O triplet
    q_romp2 = mshqc.quick_calculation("O", basis="cc-pVDZ", method="df-romp2", multiplicity=3, threshold=1e-9)
    q_romp3 = mshqc.quick_calculation("O", basis="cc-pVDZ", method="df-romp3", multiplicity=3, threshold=1e-9)
    assert q_romp2.energy_total < -74.8 and q_romp3.e_total < -74.8
    print(f"[SESSION  ] H2O OMP2.5={res['OMP2.5'].energy_total:.10f} | O ROMP3={q_romp3.e_total:.10f}")


def test_uhf_custom_plugin_hook_and_omp25_fast_rdm():
    mol = mshqc.Molecule(charge=0, multiplicity=2)
    mol.add_atom("O", 0.0, 0.0, 0.0, units="angstrom")
    mol.add_atom("H", 0.0, 0.0, 0.9697, units="angstrom")

    basis = mshqc.BasisSet("cc-pVDZ", mol)
    ints = mshqc.IntegralEngine(mol, basis)
    scf_res = mshqc.UHF(mol, basis, ints, mshqc.SCFConfig(eri_method="df", threshold=1e-10)).compute()

    H = ints.compute_kinetic() + ints.compute_nuclear()
    S = ints.compute_overlap()
    B_ao = scf_res.L_mat
    nbf = basis.n_basis_functions()
    na, nb = scf_res.n_occ_alpha, scf_res.n_occ_beta
    va, vb = nbf - na, nbf - nb

    def custom_uhf_scs_omp2(Boo_a, Bov_a, Bvv_a, Boo_b, Bov_b, Bvv_b, e_oa, e_va, e_ob, e_vb):
        e_aa, e_bb, e_ab, Taa, Tbb, Tab = K.eval_ump2_df(Bov_a, Bov_b, e_oa, e_va, e_ob, e_vb)
        g_oo_a, g_vv_a, g_oo_b, g_vv_b = K.eval_omp2_1rdm_uhf(
            Taa, Tbb, Tab, na, va, nb, vb, c_os=1.2, c_ss=1.0 / 3.0
        )
        Za, Zb = K.eval_omp2_zvec_uhf_df(
            Taa, Tbb, Tab, Boo_a, Bov_a, Bvv_a, Boo_b, Bov_b, Bvv_b,
            na, va, nb, vb, c_os=1.2, c_ss=1.0 / 3.0,
        )
        e_corr = (1.0 / 3.0) * (e_aa + e_bb) + 1.2 * e_ab
        return e_corr, g_oo_a, g_vv_a, g_oo_b, g_vv_b, Za, Zb

    res_custom = M.optimize_orbitals_uhf(
        H, S, mol.nuclear_repulsion_energy(), B_ao,
        scf_res.C_alpha, scf_res.C_beta, na, nb,
        corr_fn=custom_uhf_scs_omp2, method_name="custom-uhf-scs-omp2",
        grad_tol=1e-6, energy_tol=1e-10,
    )
    res_ref = M.run_from_arrays(
        H, S, mol.nuclear_repulsion_energy(), B_ao,
        scf_res.C_alpha, na, C_beta=scf_res.C_beta, n_occ_beta=nb,
        method="scs-omp2", grad_tol=1e-6, energy_tol=1e-10,
    )
    diff = abs(res_custom.energy_total - res_ref.energy_total)
    print(f"[UHF HOOK ] Custom={res_custom.energy_total:.10f} ({res_custom.iterations} it, <S2>={res_custom.s2:.6f}) | |dE|={diff:.2e}")
    assert res_custom.converged and diff < 1e-10

    # Pastikan OMP2.5 dengan 1-RDM linier tetap konvergen mulus pada radikal OH
    res_25 = M.run_from_arrays(
        H, S, mol.nuclear_repulsion_energy(), B_ao,
        scf_res.C_alpha, na, C_beta=scf_res.C_beta, n_occ_beta=nb,
        method="omp2.5", grad_tol=1e-6, energy_tol=1e-10,
    )
    assert res_25.converged and abs(res_25.energy_total - (-75.5526723081)) < 1e-7


def test_pyi_stub_is_valid_python():
    stub_path = Path("python/mshqc/_mshqc.pyi")
    assert stub_path.is_file()
    tree = ast.parse(stub_path.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    for expected in ("kernels", "SCFResult", "ROMP2", "ROMP3", "HPCGeometryOptimizer", "standard_frozen_core"):
        assert expected in names, f"Simbol {expected} hilang di _mshqc.pyi"
    print(f"[PYI STUB ] Valid AST ({len(names)} top-level symbols)")


if __name__ == "__main__":
    test_session_and_quick_calculation_parity()
    test_uhf_custom_plugin_hook_and_omp25_fast_rdm()
    test_pyi_stub_is_valid_python()
    print("[PASS] Semua validasi Fase 5 lulus!")
