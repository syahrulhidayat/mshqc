#!/usr/bin/env python3
"""
Validasi Fase 2 Arsitektur Hibrida:
- DF-RMP3 hibrida vs C++ mshqc.RMP3
- DF-OMP3 hibrida (dengan MO-DF TR-SOSCF CG) vs C++ mshqc.OMP3
- Metode baru DF-OMP2.5 (3 baris di Python!) + verifikasi gradien analitik vs Finite-Difference
"""
import numpy as np
from scipy.linalg import expm
import mshqc
from mshqc import kernels as K


def solve_tr_soscf_cg(w_vo, eps_occ, eps_vir, B_oo, B_ov, B_vv, trust_radius=0.25, max_cg=15):
    """Preconditioned Steihaug-CG untuk langkah TR-SOSCF memakai apply_mo_df_hessian_rhf."""
    nv, no = w_vo.shape
    J_ia = np.sum(B_ov ** 2, axis=1).reshape(no, nv).T
    H_diag = 4.0 * np.maximum(np.abs(eps_vir[:, None] - eps_occ[None, :]), 1e-4) + 8.0 * J_ia + 0.005

    x = np.zeros_like(w_vo)
    r = -w_vo.copy()
    z = r / H_diag
    p = z.copy()
    rz = float(np.sum(r * z))
    tol = min(1e-4, 0.1 * float(np.linalg.norm(w_vo)))

    for _ in range(max_cg):
        if np.linalg.norm(r) < tol:
            break
        Hp = K.apply_mo_df_hessian_rhf(np.ascontiguousarray(p), eps_occ, eps_vir, B_oo, B_ov, B_vv)
        pHp = float(np.sum(p * Hp))
        if pHp <= 1e-12:
            x = -w_vo / H_diag
            break
        alpha = rz / pHp
        x_next = x + alpha * p
        if np.linalg.norm(x_next) >= trust_radius:
            x = x_next * (trust_radius / float(np.linalg.norm(x_next)))
            break
        x = x_next
        r = r - alpha * Hp
        z = r / H_diag
        rz_new = float(np.sum(r * z))
        p = z + (rz_new / max(rz, 1e-30)) * p
        rz = rz_new

    nx = float(np.linalg.norm(x))
    if nx > trust_radius:
        x *= trust_radius / nx
    return x


def eval_omp_point(mol, basis, ints, B_ao, C, lam_mp3=1.0, need_grad=True):
    """
    Evaluasi energi (dan gradien analitik) untuk keluarga metode DF-OMP(2 + lam_mp3):
      lam_mp3 = 0.0 -> DF-OMP2
      lam_mp3 = 0.5 -> DF-OMP2.5 (metode baru, eksak!)
      lam_mp3 = 1.0 -> DF-OMP3
    """
    H = ints.compute_kinetic() + ints.compute_nuclear()
    nbf = basis.n_basis_functions()
    n_occ = mol.n_electrons() // 2
    n_vir = nbf - n_occ
    v_nn = mol.nuclear_repulsion_energy()

    C_occ = np.ascontiguousarray(C[:, :n_occ])
    J, K_mat = K.build_jk_3c_rhf(B_ao, C_occ, nbf)
    F_ao = H + J - K_mat
    P_ref = 2.0 * (C_occ @ C_occ.T)
    e_ref = 0.5 * np.sum(P_ref * (H + F_ao)) + v_nn

    C_sc, eps, F_vo = K.semicanonicalize_subspaces(C, F_ao, n_occ)
    C_occ = np.ascontiguousarray(C_sc[:, :n_occ])
    C_vir = np.ascontiguousarray(C_sc[:, n_occ:])
    eps_o, eps_v = eps[:n_occ], eps[n_occ:]

    B_oo = K.transform_3c_mo(B_ao, C_occ, C_occ, nbf)
    B_ov = K.transform_3c_mo(B_ao, C_occ, C_vir, nbf)
    B_vv = K.transform_3c_mo(B_ao, C_vir, C_vir, nbf)

    e_os, e_ss, T2 = K.eval_rmp2_df(B_ov, eps_o, eps_v)
    e_mp2 = e_os + e_ss
    e_mp3, L2 = (0.0, None)
    if lam_mp3 != 0.0:
        e_mp3, L2 = K.eval_rmp3_df(T2, B_oo, B_ov, B_vv, eps_o, eps_v)

    e_tot = e_ref + e_mp2 + lam_mp3 * e_mp3
    if not need_grad:
        return e_tot, e_mp2, e_mp3, C_sc, None, None

    if lam_mp3 == 0.0:
        g_oo, g_vv = K.eval_omp2_1rdm_rhf(T2, n_occ, n_vir)
        Z_vo = K.eval_omp2_zvec_rhf_df(T2, B_oo, B_ov, B_vv, n_occ, n_vir)
    elif lam_mp3 == 1.0:
        g_oo, g_vv = K.eval_omp3_1rdm_rhf(T2, L2, n_occ, n_vir)
        Z_vo = K.eval_omp3_zvec_rhf_df(T2, L2, B_oo, B_ov, B_vv, n_occ, n_vir)
    else:
        g2_oo, g2_vv = K.eval_omp2_1rdm_rhf(T2, n_occ, n_vir)
        g3_oo, g3_vv = K.eval_omp3_1rdm_rhf(T2, L2, n_occ, n_vir)
        Z2 = K.eval_omp2_zvec_rhf_df(T2, B_oo, B_ov, B_vv, n_occ, n_vir)
        Z3 = K.eval_omp3_zvec_rhf_df(T2, L2, B_oo, B_ov, B_vv, n_occ, n_vir)
        g_oo = (1.0 - lam_mp3) * g2_oo + lam_mp3 * g3_oo
        g_vv = (1.0 - lam_mp3) * g2_vv + lam_mp3 * g3_vv
        Z_vo = (1.0 - lam_mp3) * Z2 + lam_mp3 * Z3

    P_corr_a = C_occ @ g_oo @ C_occ.T + C_vir @ g_vv @ C_vir.T
    G_ao = 2.0 * K.build_coulomb_3c(B_ao, P_corr_a, nbf) - K.build_exchange_3c_sym(B_ao, P_corr_a, nbf)
    G_vo = C_vir.T @ G_ao @ C_occ
    L_sep = F_vo @ g_oo - g_vv @ F_vo
    w_vo = 4.0 * (F_vo + G_vo + L_sep + Z_vo)
    return e_tot, e_mp2, e_mp3, C_sc, w_vo, (eps_o, eps_v, B_oo, B_ov, B_vv)


def run_hybrid_omp_family(mol, basis, ints, B_ao, C0, lam_mp3=1.0, max_iter=20, grad_tol=1e-6):
    nbf = basis.n_basis_functions()
    n_occ = mol.n_electrons() // 2
    C = C0.copy()
    for it in range(max_iter):
        e_tot, e_mp2, e_mp3, C, w_vo, aux = eval_omp_point(mol, basis, ints, B_ao, C, lam_mp3=lam_mp3, need_grad=True)
        w_rms = float(np.sqrt(np.mean(w_vo ** 2)))
        if w_rms < grad_tol:
            return e_tot, it, w_rms, C
        eps_o, eps_v, B_oo, B_ov, B_vv = aux
        kappa = solve_tr_soscf_cg(w_vo, eps_o, eps_v, B_oo, B_ov, B_vv)
        K_rot = np.zeros((nbf, nbf))
        K_rot[n_occ:, :n_occ] = kappa
        K_rot[:n_occ, n_occ:] = -kappa.T
        C = C @ expm(K_rot)
    raise RuntimeError(f"Hybrid DF-OMP(lam={lam_mp3}) did not converge")


def test_phase2_hybrid_omp3_and_omp25():
    mol = mshqc.Molecule(charge=0, multiplicity=1)
    mol.add_atom("O", 0.0000,  0.0000,  0.1173, units="angstrom")
    mol.add_atom("H", 0.0000,  0.7572, -0.4692, units="angstrom")
    mol.add_atom("H", 0.0000, -0.7572, -0.4692, units="angstrom")

    basis = mshqc.BasisSet("cc-pVDZ", mol)
    aux_basis = mshqc.BasisSet("cc-pVDZ-RI", mol)
    ints = mshqc.IntegralEngine(mol, basis)

    scf_conf = mshqc.SCFConfig(eri_method="df", threshold=1e-10)
    scf_ref = mshqc.RHF(mol, basis, ints, scf_conf).compute()
    mp_conf = mshqc.MP2Config(eri_method="df", threshold=1e-10, gradient_threshold=1e-6)
    rmp3_ref = mshqc.RMP3(mol, basis, ints, scf_ref, mp_conf).compute()
    omp3_ref = mshqc.OMP3(mol, basis, ints, scf_ref, mp_conf).compute()

    # Stateless safe DF factor builder
    B_ao = K.compute_df_ao_factor(mol, basis, aux_basis, 1e-10, False)

    # 1. Validasi DF-RMP3 (iterasi 0)
    e_rmp3_hyb, _, _, C_sc, w_omp25, _ = eval_omp_point(mol, basis, ints, B_ao, scf_ref.C_alpha, lam_mp3=1.0, need_grad=False)
    diff_rmp3 = abs(e_rmp3_hyb - rmp3_ref.e_total)
    print(f"[DF-RMP3  ] Hybrid={e_rmp3_hyb:.10f} | C++={rmp3_ref.e_total:.10f} | |dE|={diff_rmp3:.2e} Ha")
    assert diff_rmp3 < 1e-9

    # 2. Validasi DF-OMP3 dengan MO-DF TR-SOSCF CG
    e_omp3_hyb, it_omp3, w_rms_3, _ = run_hybrid_omp_family(mol, basis, ints, B_ao, scf_ref.C_alpha, lam_mp3=1.0)
    diff_omp3 = abs(e_omp3_hyb - omp3_ref.e_total)
    print(f"[DF-OMP3  ] Hybrid={e_omp3_hyb:.10f} ({it_omp3} it, w={w_rms_3:.2e}) | C++={omp3_ref.e_total:.10f} | |dE|={diff_omp3:.2e} Ha")
    assert diff_omp3 < 1e-8

    # 3. Demonstrasi metode baru DF-OMP2.5 + verifikasi gradien analitik vs Central Finite-Difference
    _, _, _, C_sc25, w_ana_25, _ = eval_omp_point(mol, basis, ints, B_ao, scf_ref.C_alpha, lam_mp3=0.5, need_grad=True)
    a_max, i_max = np.unravel_index(np.argmax(np.abs(w_ana_25)), w_ana_25.shape)
    nbf = basis.n_basis_functions()
    n_occ = mol.n_electrons() // 2
    theta = 1e-5
    K_fd = np.zeros((nbf, nbf))
    K_fd[n_occ + a_max, i_max] = theta
    K_fd[i_max, n_occ + a_max] = -theta
    e_plus, *_ = eval_omp_point(mol, basis, ints, B_ao, C_sc25 @ expm(K_fd), lam_mp3=0.5, need_grad=False)
    e_minus, *_ = eval_omp_point(mol, basis, ints, B_ao, C_sc25 @ expm(-K_fd), lam_mp3=0.5, need_grad=False)
    w_fd = (e_plus - e_minus) / (2.0 * theta)
    grad_err = abs(w_ana_25[a_max, i_max] - w_fd)

    e_omp25, it_25, w_rms_25, _ = run_hybrid_omp_family(mol, basis, ints, B_ao, scf_ref.C_alpha, lam_mp3=0.5)
    print(f"[DF-OMP2.5] Grad FD={w_fd:.10e} | Ana={w_ana_25[a_max, i_max]:.10e} | |dw|={grad_err:.2e} Ha/rad")
    print(f"[DF-OMP2.5] Converged={e_omp25:.10f} Ha ({it_25} it, w={w_rms_25:.2e})")
    assert grad_err < 1e-8


if __name__ == "__main__":
    test_phase2_hybrid_omp3_and_omp25()
    print("[PASS] Semua validasi Fase 2 Hibrida (RMP3, OMP3, OMP2.5 + FD) lulus!")
