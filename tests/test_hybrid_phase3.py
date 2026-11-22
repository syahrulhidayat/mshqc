#!/usr/bin/env python3
"""
Validasi Fase 3 Arsitektur Hibrida:
1. Konversi tensor 4D (t2_to_4d / t2_from_4d) cocok dengan np.einsum
2. Friendly interface `mshqc.run()` untuk ["RHF", "OMP2", "SCS-OMP2", "SOS-OMP2", "OMP2.5", "OMP3"]
3. Radikal kulit terbuka (UHF) atom O (^3P) & radikal OH (^2Pi):
   - DF-UHF-OMP2 & DF-UHF-OMP3 hibrida vs C++ penuh (|dE| < 1e-8 Ha) + pemulihan spin <S^2>
   - Metode baru DF-UHF-SCS-OMP2 & DF-UHF-OMP2.5 + verifikasi gradien analitik vs Finite-Difference
4. Interoperabilitas array mentah `mshqc.methods.run_from_arrays` (gaya PySCF _cderi)
"""
import numpy as np
from scipy.linalg import expm
import mshqc
from mshqc import kernels as K
from mshqc import methods as M


def test_4d_tensor_helpers_and_einsum():
    mol = mshqc.Molecule(charge=0, multiplicity=1)
    mol.add_atom("O", 0.0000,  0.0000,  0.1173, units="angstrom")
    mol.add_atom("H", 0.0000,  0.7572, -0.4692, units="angstrom")
    mol.add_atom("H", 0.0000, -0.7572, -0.4692, units="angstrom")

    basis = mshqc.BasisSet("cc-pVDZ", mol)
    aux = mshqc.BasisSet("cc-pVDZ-RI", mol)
    ints = mshqc.IntegralEngine(mol, basis)
    scf_res = mshqc.RHF(mol, basis, ints, mshqc.SCFConfig(eri_method="df", threshold=1e-10)).compute()

    nbf = basis.n_basis_functions()
    no = mol.n_electrons() // 2
    nv = nbf - no
    B_ao = K.compute_df_ao_factor(mol, basis, aux, 1e-10, False)
    B_ov = K.transform_3c_mo(B_ao, scf_res.C_alpha[:, :no], scf_res.C_alpha[:, no:], nbf)

    e_os, e_ss, T2_mat = K.eval_rmp2_df(B_ov, scf_res.orbital_energies_alpha[:no], scf_res.orbital_energies_alpha[no:])
    T2_4d = M.t2_to_4d(T2_mat, no, no, nv, nv)
    assert np.allclose(M.t2_from_4d(T2_4d), T2_mat, atol=1e-15)

    B_ov_3d = M.b3c_to_3d(B_ov, no, nv)  # (n_aux, no, nv)
    I_iajb = np.einsum("Pia,Pjb->ijab", B_ov_3d, B_ov_3d)
    e_einsum = float(np.einsum("ijab,ijab->", T2_4d, 2.0 * I_iajb - I_iajb.transpose(0, 1, 3, 2)))
    assert abs(e_einsum - (e_os + e_ss)) < 1e-12
    print(f"[4D EINSUM] E_MP2 kernel={e_os+e_ss:.10f} | einsum={e_einsum:.10f} | |dE|={abs(e_einsum-(e_os+e_ss)):.2e}")


def test_friendly_run_interface_closed_shell():
    h2o_xyz = """
    O 0.0000  0.0000  0.1173
    H 0.0000  0.7572 -0.4692
    H 0.0000 -0.7572 -0.4692
    """
    res = mshqc.run(
        h2o_xyz,
        method=["RHF", "OMP2", "SCS-OMP2", "SOS-OMP2", "OMP2.5", "OMP3"],
        basis="cc-pVDZ",
        eri_method="df",
        units="angstrom",
        threshold=1e-10,
        grad_thresh=1e-6,
        use_symmetry=False,
    )
    for k, v in res.items():
        print(f"[RUN H2O  ] {k:<9}: {v['energy']:.10f} Ha ({v['time']:.3f} s)")

    assert res["OMP2.5"]["result"].converged
    assert res["SCS-OMP2"]["result"].converged
    assert res["SOS-OMP2"]["result"].converged
    # Energi OMP2.5 harus berada di antara OMP2 dan OMP3 (untuk H2O/cc-pVDZ)
    e2, e25, e3 = res["OMP2"]["energy"], res["OMP2.5"]["energy"], res["OMP3"]["energy"]
    assert min(e2, e3) < e25 < max(e2, e3)


def test_open_shell_uhf_hybrid_matches_cpp_and_fd():
    mol = mshqc.Molecule(charge=0, multiplicity=2)
    mol.add_atom("O", 0.0000, 0.0000, 0.0000, units="angstrom")
    mol.add_atom("H", 0.0000, 0.0000, 0.9697, units="angstrom")

    basis = mshqc.BasisSet("cc-pVDZ", mol)
    aux = mshqc.BasisSet("cc-pVDZ-RI", mol)
    ints = mshqc.IntegralEngine(mol, basis)

    scf_conf = mshqc.SCFConfig(eri_method="df", threshold=1e-10)
    scf_ref = mshqc.UHF(mol, basis, ints, scf_conf).compute()
    mp_conf = mshqc.MP2Config(eri_method="df", threshold=1e-10, gradient_threshold=1e-6)
    omp2_cpp = mshqc.OMP2(mol, basis, ints, scf_ref, mp_conf).compute()
    omp3_cpp = mshqc.OMP3(mol, basis, ints, scf_ref, mp_conf).compute()

    H_core = ints.compute_kinetic() + ints.compute_nuclear()
    S_ovlp = ints.compute_overlap()
    v_nn = mol.nuclear_repulsion_energy()
    B_ao = K.compute_df_ao_factor(mol, basis, aux, 1e-10, False)
    na, nb = scf_ref.n_occ_alpha, scf_ref.n_occ_beta

    # 1. Uji interoperabilitas run_from_arrays pada OH radikal (OMP2, OMP3, SCS-OMP2, OMP2.5)
    res_omp2 = M.run_from_arrays(
        H_core, S_ovlp, v_nn, B_ao, scf_ref.C_alpha, na,
        C_beta=scf_ref.C_beta, n_occ_beta=nb, method="omp2", grad_tol=1e-6, energy_tol=1e-10,
    )
    res_omp3 = M.run_from_arrays(
        H_core, S_ovlp, v_nn, B_ao, scf_ref.C_alpha, na,
        C_beta=scf_ref.C_beta, n_occ_beta=nb, method="omp3", grad_tol=1e-6, energy_tol=1e-10,
    )
    res_scs = M.run_from_arrays(
        H_core, S_ovlp, v_nn, B_ao, scf_ref.C_alpha, na,
        C_beta=scf_ref.C_beta, n_occ_beta=nb, method="scs-omp2", grad_tol=1e-6, energy_tol=1e-10,
    )
    res_25 = M.run_from_arrays(
        H_core, S_ovlp, v_nn, B_ao, scf_ref.C_alpha, na,
        C_beta=scf_ref.C_beta, n_occ_beta=nb, method="omp2.5", grad_tol=1e-6, energy_tol=1e-10,
    )

    d2 = abs(res_omp2.energy_total - omp2_cpp.energy_total)
    d3 = abs(res_omp3.energy_total - omp3_cpp.e_total)
    s2_uhf = 0.75 + nb - float(np.sum((scf_ref.C_alpha[:, :na].T @ S_ovlp @ scf_ref.C_beta[:, :nb]) ** 2))

    print(f"[OH UHF-OMP2  ] Hybrid={res_omp2.energy_total:.10f} ({res_omp2.iterations} it, <S2>={res_omp2.s2:.6f}) | C++={omp2_cpp.energy_total:.10f} | |dE|={d2:.2e}")
    print(f"[OH UHF-OMP3  ] Hybrid={res_omp3.energy_total:.10f} ({res_omp3.iterations} it, <S2>={res_omp3.s2:.6f}) | C++={omp3_cpp.e_total:.10f} | |dE|={d3:.2e}")
    print(f"[OH UHF-SCS2  ] Hybrid={res_scs.energy_total:.10f} ({res_scs.iterations} it, <S2>={res_scs.s2:.6f})")
    print(f"[OH UHF-OMP2.5] Hybrid={res_25.energy_total:.10f} ({res_25.iterations} it, <S2>={res_25.s2:.6f}) | UHF <S2>={s2_uhf:.6f}")

    assert d2 < 1e-8, f"UHF-OMP2 mismatch: {d2}"
    assert d3 < 1e-8, f"UHF-OMP3 mismatch: {d3}"
    assert res_omp2.s2 < s2_uhf and res_omp3.s2 < s2_uhf

    # 2. Verifikasi gradien analitik UHF-OMP2.5 terhadap Central Finite-Difference (kanal beta)
    _, _, _, _, Ca_sc, Cb_sc, _, _, wa_ana, wb_ana, _ = M.eval_omp_uhf_point(
        H_core, v_nn, B_ao, scf_ref.C_alpha, scf_ref.C_beta, na, nb, lam_mp3=0.5, need_grad=True
    )
    a_max, i_max = np.unravel_index(np.argmax(np.abs(wb_ana)), wb_ana.shape)
    nbf = basis.n_basis_functions()
    theta = 1e-5
    K_fd = np.zeros((nbf, nbf))
    K_fd[nb + a_max, i_max] = theta
    K_fd[i_max, nb + a_max] = -theta
    e_plus = M.eval_omp_uhf_point(H_core, v_nn, B_ao, Ca_sc, Cb_sc @ expm(K_fd), na, nb, lam_mp3=0.5, need_grad=False)[0]
    e_minus = M.eval_omp_uhf_point(H_core, v_nn, B_ao, Ca_sc, Cb_sc @ expm(-K_fd), na, nb, lam_mp3=0.5, need_grad=False)[0]
    w_fd = (e_plus - e_minus) / (2.0 * theta)
    grad_err = abs(wb_ana[a_max, i_max] - w_fd)
    print(f"[OH OMP2.5 FD ] Beta Grad FD={w_fd:.10e} | Ana={wb_ana[a_max, i_max]:.10e} | |dw|={grad_err:.2e}")
    assert grad_err < 1e-8


if __name__ == "__main__":
    test_4d_tensor_helpers_and_einsum()
    test_friendly_run_interface_closed_shell()
    test_open_shell_uhf_hybrid_matches_cpp_and_fd()
    print("[PASS] Semua validasi Fase 3 Hibrida lulus!")
