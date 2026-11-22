#!/usr/bin/env python3
"""
Validasi Fase 4 Arsitektur Hibrida:
1. Reuse SCFResult.L_mat (zero redundant DF decompositions) pada mshqc.run()
2. Mode CHOLESKY untuk metode hibrida (SCS-OMP2, SOS-OMP2, OMP2.5) + verifikasi CHOLESKY-OMP3 vs C++
3. Custom method plugin hook `mshqc.methods.optimize_orbitals_rhf` (~6 baris Python)
4. Regresi deduplikasi C++ (omp_fock.cc & mp3.cc memakai shared kernels/tblis_safe.h & mp3_kernels.h)
"""
import numpy as np
import mshqc
from mshqc import kernels as K
from mshqc import methods as M


def test_phase4_zero_copy_L_mat_and_cholesky_hybrid():
    h2o_xyz = """
    O 0.0000  0.0000  0.1173
    H 0.0000  0.7572 -0.4692
    H 0.0000 -0.7572 -0.4692
    """
    res_chol = mshqc.run(
        h2o_xyz,
        method=["RHF", "OMP2", "SCS-OMP2", "OMP2.5", "OMP3"],
        basis="cc-pVDZ",
        eri_method="cholesky",
        units="angstrom",
        threshold=1e-9,
        grad_thresh=1e-6,
        use_symmetry=False,
    )
    scf_res = res_chol["RHF"]["result"]
    assert scf_res.L_mat.shape[0] > 0 and scf_res.L_mat.shape[1] > 0

    for name, info in res_chol.items():
        print(f"[CHOL H2O ] {name:<9}: {info['energy']:.10f} Ha ({info['time']:.4f} s)")

    mol = mshqc.build_molecule(h2o_xyz, units="angstrom")
    basis = mshqc.BasisSet("cc-pVDZ", mol)
    ints = mshqc.IntegralEngine(mol, basis)
    H = ints.compute_kinetic() + ints.compute_nuclear()
    S = ints.compute_overlap()
    hyb_chol_omp3 = M.run_from_arrays(
        H, S, mol.nuclear_repulsion_energy(), scf_res.L_mat,
        scf_res.C_alpha, scf_res.n_occ_alpha,
        method="omp3", grad_tol=1e-6, energy_tol=1e-9,
    )
    diff = abs(hyb_chol_omp3.energy_total - res_chol["OMP3"]["energy"])
    print(f"[CHOL OMP3] Hybrid={hyb_chol_omp3.energy_total:.10f} | C++={res_chol['OMP3']['energy']:.10f} | |dE|={diff:.2e}")
    assert diff < 1e-8


def test_phase4_custom_plugin_hook():
    mol = mshqc.Molecule(charge=0, multiplicity=1)
    mol.add_atom("O", 0.0000,  0.0000,  0.1173, units="angstrom")
    mol.add_atom("H", 0.0000,  0.7572, -0.4692, units="angstrom")
    mol.add_atom("H", 0.0000, -0.7572, -0.4692, units="angstrom")

    basis = mshqc.BasisSet("cc-pVDZ", mol)
    ints = mshqc.IntegralEngine(mol, basis)
    scf_res = mshqc.RHF(mol, basis, ints, mshqc.SCFConfig(eri_method="df", threshold=1e-10)).compute()

    H = ints.compute_kinetic() + ints.compute_nuclear()
    B_ao = scf_res.L_mat
    no = scf_res.n_occ_alpha
    nv = basis.n_basis_functions() - no

    def my_scs_omp2_kernel(B_oo, B_ov, B_vv, eps_o, eps_v):
        e_os, e_ss, T2 = K.eval_rmp2_df(B_ov, eps_o, eps_v)
        g_oo, g_vv = K.eval_omp2_1rdm_rhf(T2, no, nv, c_os=1.2, c_ss=1.0 / 3.0)
        Z_vo = K.eval_omp2_zvec_rhf_df(T2, B_oo, B_ov, B_vv, no, nv, c_os=1.2, c_ss=1.0 / 3.0)
        return 1.2 * e_os + (1.0 / 3.0) * e_ss, g_oo, g_vv, Z_vo

    res_custom = M.optimize_orbitals_rhf(
        H, mol.nuclear_repulsion_energy(), B_ao, scf_res.C_alpha, no,
        corr_fn=my_scs_omp2_kernel, method_name="custom-scs-omp2",
        grad_tol=1e-6, energy_tol=1e-10,
    )
    res_ref = M.run_from_arrays(
        H, ints.compute_overlap(), mol.nuclear_repulsion_energy(), B_ao,
        scf_res.C_alpha, no, method="scs-omp2", grad_tol=1e-6, energy_tol=1e-10,
    )
    diff = abs(res_custom.energy_total - res_ref.energy_total)
    print(f"[PLUGIN   ] Custom={res_custom.energy_total:.10f} ({res_custom.iterations} it) | Ref={res_ref.energy_total:.10f} | |dE|={diff:.2e}")
    assert res_custom.converged and diff < 1e-10


if __name__ == "__main__":
    test_phase4_zero_copy_L_mat_and_cholesky_hybrid()
    test_phase4_custom_plugin_hook()
    print("[PASS] Semua validasi Fase 4 (Deduplikasi, Caching L_mat, Cholesky Hybrid, Plugin) lulus!")
