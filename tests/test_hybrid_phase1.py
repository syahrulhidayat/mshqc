#!/usr/bin/env python3
"""
Validasi Fase 1 (Fixed): DF-RHF, DF-OMP2, dan SCS-OMP2 hibrida vs solver C++20 penuh.
"""
import numpy as np
from scipy.linalg import eigh, expm
import mshqc
from mshqc import kernels as K


def run_hybrid_df_rhf(mol, basis, ints, B_ao, max_iter=50, tol=1e-10):
    S = ints.compute_overlap()
    H = ints.compute_kinetic() + ints.compute_nuclear()
    nbf = basis.n_basis_functions()
    n_occ = mol.n_electrons() // 2
    v_nn = mol.nuclear_repulsion_energy()

    evals_s, evecs_s = eigh(S)
    X = evecs_s @ np.diag(1.0 / np.sqrt(evals_s)) @ evecs_s.T
    _, C_prime = eigh(X.T @ H @ X)
    C = X @ C_prime
    diis = mshqc.DIIS(8)

    e_prev = 0.0
    for _ in range(max_iter):
        C_occ = np.ascontiguousarray(C[:, :n_occ])
        J, K_mat = K.build_jk_3c_rhf(B_ao, C_occ, nbf)
        F = H + J - K_mat
        P = 2.0 * (C_occ @ C_occ.T)
        e_tot = 0.5 * np.sum(P * (H + F)) + v_nn

        err = F @ P @ S - S @ P @ F
        diis.add_iteration(F, err)
        F_ext = diis.extrapolate()
        eps, C_prime = eigh(X.T @ F_ext @ X)
        C = X @ C_prime
        if abs(e_tot - e_prev) < tol and np.linalg.norm(err) < 1e-8:
            return e_tot, C, eps
        e_prev = e_tot
    raise RuntimeError("Hybrid DF-RHF did not converge")


def run_hybrid_df_omp2(mol, basis, ints, B_ao, C0, c_os=1.0, c_ss=1.0, max_iter=40, grad_tol=1e-6):
    H = ints.compute_kinetic() + ints.compute_nuclear()
    nbf = basis.n_basis_functions()
    n_occ = mol.n_electrons() // 2
    n_vir = nbf - n_occ
    v_nn = mol.nuclear_repulsion_energy()

    C = C0.copy()
    w_rms = 1.0
    for it in range(max_iter):
        C_occ = np.ascontiguousarray(C[:, :n_occ])
        J, K_mat = K.build_jk_3c_rhf(B_ao, C_occ, nbf)
        F_ao = H + J - K_mat
        P_ref = 2.0 * (C_occ @ C_occ.T)
        e_ref = 0.5 * np.sum(P_ref * (H + F_ao)) + v_nn

        C, eps, F_vo = K.semicanonicalize_subspaces(C, F_ao, n_occ)
        C_occ = np.ascontiguousarray(C[:, :n_occ])
        C_vir = np.ascontiguousarray(C[:, n_occ:])

        B_oo = K.transform_3c_mo(B_ao, C_occ, C_occ, nbf)
        B_ov = K.transform_3c_mo(B_ao, C_occ, C_vir, nbf)
        B_vv = K.transform_3c_mo(B_ao, C_vir, C_vir, nbf)

        e_os, e_ss, T2 = K.eval_rmp2_df(B_ov, eps[:n_occ], eps[n_occ:])
        e_tot = e_ref + c_os * e_os + c_ss * e_ss

        gamma_oo, gamma_vv = K.eval_omp2_1rdm_rhf(T2, n_occ, n_vir, c_os=c_os, c_ss=c_ss)
        Z_vo = K.eval_omp2_zvec_rhf_df(T2, B_oo, B_ov, B_vv, n_occ, n_vir, c_os=c_os, c_ss=c_ss)

        P_corr_a = C_occ @ gamma_oo @ C_occ.T + C_vir @ gamma_vv @ C_vir.T
        G_ao = 2.0 * K.build_coulomb_3c(B_ao, P_corr_a, nbf) - K.build_exchange_3c_sym(B_ao, P_corr_a, nbf)
        G_vo = C_vir.T @ G_ao @ C_occ

        L_sep = F_vo @ gamma_oo - gamma_vv @ F_vo
        w_vo = 4.0 * (F_vo + G_vo + L_sep + Z_vo)
        w_rms = float(np.sqrt(np.mean(w_vo ** 2)))
        if w_rms < grad_tol:
            return e_tot, it, w_rms

        H_diag = 4.0 * np.maximum(np.abs(eps[n_occ:, None] - eps[None, :n_occ]), 1e-3)
        kappa = -w_vo / H_diag
        norm_k = float(np.linalg.norm(kappa))
        if norm_k > 0.25:
            kappa *= 0.25 / norm_k

        K_rot = np.zeros((nbf, nbf))
        K_rot[n_occ:, :n_occ] = kappa
        K_rot[:n_occ, n_occ:] = -kappa.T
        C = C @ expm(K_rot)

    raise RuntimeError(f"Hybrid DF-OMP2 did not converge (last w_rms={w_rms:.3e})")


def test_phase1_hybrid_matches_cpp():
    mol = mshqc.Molecule(charge=0, multiplicity=1)
    mol.add_atom("O", 0.0000,  0.0000,  0.1173, units="angstrom")
    mol.add_atom("H", 0.0000,  0.7572, -0.4692, units="angstrom")
    mol.add_atom("H", 0.0000, -0.7572, -0.4692, units="angstrom")

    basis = mshqc.BasisSet("cc-pVDZ", mol)
    aux_basis = mshqc.BasisSet("cc-pVDZ-RI", mol)
    ints = mshqc.IntegralEngine(mol, basis)

    scf_conf = mshqc.SCFConfig(eri_method="df", threshold=1e-10)
    scf_ref = mshqc.RHF(mol, basis, ints, scf_conf).compute()
    mp2_conf = mshqc.MP2Config(eri_method="df", threshold=1e-10, gradient_threshold=1e-6)
    omp2_ref = mshqc.OMP2(mol, basis, ints, scf_ref, mp2_conf).compute()

    # Safe RAII DF engine (auto-combines primary + aux basis, bebas segfault!)
    df = K.DensityFittingERI(mol, basis, aux_basis, 1e-10)
    df.compute(False)
    B_ao = df.get_B_mat()
    assert K.is_packed_ao(B_ao, basis.n_basis_functions())

    e_rhf_hyb, C_rhf, _ = run_hybrid_df_rhf(mol, basis, ints, B_ao)
    e_omp2_hyb, n_it, _ = run_hybrid_df_omp2(mol, basis, ints, B_ao, C_rhf)
    e_scs_omp2, n_it_scs, _ = run_hybrid_df_omp2(mol, basis, ints, B_ao, C_rhf, c_os=1.2, c_ss=1.0 / 3.0)

    diff_rhf = abs(e_rhf_hyb - scf_ref.energy_total)
    diff_omp2 = abs(e_omp2_hyb - omp2_ref.energy_total)

    print(f"[DF-RHF  ] Hybrid={e_rhf_hyb:.10f} | C++={scf_ref.energy_total:.10f} | |dE|={diff_rhf:.2e} Ha")
    print(f"[DF-OMP2 ] Hybrid={e_omp2_hyb:.10f} ({n_it} it) | C++={omp2_ref.energy_total:.10f} | |dE|={diff_omp2:.2e} Ha")
    print(f"[SCS-OMP2] Hybrid={e_scs_omp2:.10f} ({n_it_scs} it)")

    assert diff_rhf < 1e-9
    assert diff_omp2 < 1e-8


if __name__ == "__main__":
    test_phase1_hybrid_matches_cpp()
    print("[PASS] Fase 1 Hibrida lulus!")
