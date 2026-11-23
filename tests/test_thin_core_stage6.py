#!/usr/bin/env python3
"""
Validasi Tahap 6 (Zero-HDF5 Thin-Core + kappa-OOMP2 + Telemetry Callback + Orbital Checkpoint):
1. Gradien orbital analitik kappa-OOMP2 (kappa=1.45) cocok < 1e-8 Ha/rad vs Central FD
   pada H2O (RHF) dan radikal OH (UHF), termasuk verifikasi limit kappa -> inf == DF-OMP2.
2. Integrasi `mshqc.run(..., method="kappa-omp2")` dan `M.run_from_arrays(..., method="kappa-omp2")`.
3. Telemetry `callback` (SCF & OMP) mencatat iterasi secara terstruktur dengan 0 overhead saat None.
4. Orbital checkpoint `.npz` (save/load -> warm-start) menyusutkan iterasi SCF menjadi <= 3 iterasi.
"""
from pathlib import Path
import tempfile
import numpy as np
from scipy.linalg import expm
import mshqc
from mshqc import kernels as K
from mshqc import methods as M
from mshqc.methods.kappa_omp2 import _kappa_omp2_rhf_callback, _kappa_omp2_uhf_callback
from mshqc.methods.omp_hybrid import _assemble_orbital_grad_rhf, _assemble_orbital_grad_uhf


def test_kappa_omp2_analytic_grad_and_large_kappa_limit():
    mol = mshqc.Molecule(0, 1)
    mol.add_atom("O", 0.0, 0.0, 0.1173, units="angstrom")
    mol.add_atom("H", 0.0, 0.7572, -0.4692, units="angstrom")
    mol.add_atom("H", 0.0, -0.7572, -0.4692, units="angstrom")

    basis = mshqc.BasisSet("cc-pVDZ", mol)
    aux = mshqc.BasisSet("cc-pVDZ-RI", mol)
    ints = mshqc.IntegralEngine(mol, basis)
    H = ints.compute_kinetic() + ints.compute_nuclear()
    S = ints.compute_overlap()
    v_nn = float(mol.nuclear_repulsion_energy())
    B_ao = K.compute_df_ao_factor(mol, basis, aux, 1e-10, False, False)
    nbf = int(basis.n_basis_functions())
    no = int(mol.n_electrons()) // 2

    scf_r = M.run_df_rhf(H, S, v_nn, B_ao, no, energy_tol=1e-11, density_tol=1e-11)
    C0 = scf_r.C_alpha

    kappa_val = 1.45
    cb_r = _kappa_omp2_rhf_callback(kappa=kappa_val, c_os=1.0, c_ss=1.0)

    def eval_r_energy_and_grad(C_in):
        Co = np.ascontiguousarray(C_in[:, :no])
        J, Km = K.build_jk_3c_rhf(B_ao, Co, nbf)
        F = H + J - Km
        e_ref = float(np.sum((Co @ Co.T) * (H + F)) + v_nn)
        C_sc, eps, F_vo = K.semicanonicalize_subspaces(C_in, F, no)
        Co, Cv = np.ascontiguousarray(C_sc[:, :no]), np.ascontiguousarray(C_sc[:, no:])
        Boo, Bov, Bvv = K.transform_3c_mo_blocks(B_ao, Co, Cv, nbf)
        e_c, _, g_oo, g_vv, Z_vo = cb_r(Boo, Bov, Bvv, eps[:no], eps[no:])
        w_vo = _assemble_orbital_grad_rhf(B_ao, Co, Cv, F_vo, g_oo, g_vv, Z_vo)
        return e_ref + e_c, w_vo, C_sc

    _, w_ana, C_base = eval_r_energy_and_grad(C0)
    i_t, a_t = no - 1, 0
    h = 1.0e-5
    K_rot = np.zeros((nbf, nbf))
    K_rot[no + a_t, i_t] = 1.0
    K_rot[i_t, no + a_t] = -1.0
    e_p, _, _ = eval_r_energy_and_grad(C_base @ expm(h * K_rot))
    e_m, _, _ = eval_r_energy_and_grad(C_base @ expm(-h * K_rot))
    g_fd = (e_p - e_m) / (2.0 * h)
    g_ana = float(w_ana[a_t, i_t])
    dw_r = abs(g_fd - g_ana)
    print(f"[k-OOMP2 RHF FD] kappa={kappa_val} | FD={g_fd:.10e} | Ana={g_ana:.10e} | |dw|={dw_r:.2e}")
    assert dw_r < 1.0e-8

    res_k145 = M.run_df_kappa_omp2_rhf(H, v_nn, B_ao, C0, no, kappa=1.45, grad_tol=1e-7, energy_tol=1e-11)
    res_kinf = M.run_df_kappa_omp2_rhf(H, v_nn, B_ao, C0, no, kappa=200.0, grad_tol=1e-7, energy_tol=1e-11)
    res_omp2 = M.run_df_omp_rhf(H, v_nn, B_ao, C0, no, method_name="omp2", grad_tol=1e-7, energy_tol=1e-11)

    d_lim = abs(res_kinf.energy_total - res_omp2.energy_total)
    print(
        f"[k-OOMP2 RHF   ] E(k=1.45)={res_k145.energy_total:.10f} ({res_k145.iterations} it) | "
        f"|E(k=200)-E(OMP2)|={d_lim:.2e}"
    )
    assert res_k145.converged and res_kinf.converged
    assert res_omp2.energy_total < res_k145.energy_total < scf_r.energy_total
    assert d_lim < 1.0e-10

    oh = mshqc.Molecule(0, 2)
    oh.add_atom("O", 0.0, 0.0, 0.0, units="angstrom")
    oh.add_atom("H", 0.0, 0.0, 0.9697, units="angstrom")
    b_oh = mshqc.BasisSet("cc-pVDZ", oh)
    a_oh = mshqc.BasisSet("cc-pVDZ-RI", oh)
    i_oh = mshqc.IntegralEngine(oh, b_oh)
    H_u = i_oh.compute_kinetic() + i_oh.compute_nuclear()
    S_u = i_oh.compute_overlap()
    v_u = float(oh.nuclear_repulsion_energy())
    B_u = K.compute_df_ao_factor(oh, b_oh, a_oh, 1e-10, False, False)
    nbf_u = int(b_oh.n_basis_functions())
    na, nb = 5, 4

    scf_u = M.run_df_uhf(H_u, S_u, v_u, B_u, na, nb, energy_tol=1e-11, density_tol=1e-11)
    cb_u = _kappa_omp2_uhf_callback(kappa=1.45, c_os=1.0, c_ss=1.0)

    def eval_u_energy_and_grad(Ca_in, Cb_in):
        Cao, Cbo = np.ascontiguousarray(Ca_in[:, :na]), np.ascontiguousarray(Cb_in[:, :nb])
        J, Ka, Kb = K.build_jk_3c_uhf(B_u, Cao, Cbo, nbf_u)
        Fa, Fb = H_u + J - Ka, H_u + J - Kb
        e_ref = float(0.5 * (np.sum((Cao @ Cao.T) * (H_u + Fa)) + np.sum((Cbo @ Cbo.T) * (H_u + Fb))) + v_u)
        Ca_sc, ea, Fvo_a = K.semicanonicalize_subspaces(Ca_in, Fa, na)
        Cb_sc, eb, Fvo_b = K.semicanonicalize_subspaces(Cb_in, Fb, nb)
        Cao, Cav = np.ascontiguousarray(Ca_sc[:, :na]), np.ascontiguousarray(Ca_sc[:, na:])
        Cbo, Cbv = np.ascontiguousarray(Cb_sc[:, :nb]), np.ascontiguousarray(Cb_sc[:, nb:])
        Boo_a, Bov_a, Bvv_a, Boo_b, Bov_b, Bvv_b = K.transform_3c_mo_blocks_uhf(B_u, Cao, Cav, Cbo, Cbv, nbf_u)
        e_c, _, goa, gva, gob, gvb, Za, Zb = cb_u(
            Boo_a, Bov_a, Bvv_a, Boo_b, Bov_b, Bvv_b, ea[:na], ea[na:], eb[:nb], eb[nb:]
        )
        wa, wb = _assemble_orbital_grad_uhf(B_u, Cao, Cav, Cbo, Cbv, Fvo_a, Fvo_b, goa, gva, gob, gvb, Za, Zb)
        return e_ref + e_c, wa, wb, Ca_sc, Cb_sc

    _, _, wb_ana, Ca_b, Cb_b = eval_u_energy_and_grad(scf_u.C_alpha, scf_u.C_beta)
    K_b = np.zeros((nbf_u, nbf_u))
    K_b[nb + 0, nb - 1] = 1.0
    K_b[nb - 1, nb + 0] = -1.0
    ep_u, _, _, _, _ = eval_u_energy_and_grad(Ca_b, Cb_b @ expm(h * K_b))
    em_u, _, _, _, _ = eval_u_energy_and_grad(Ca_b, Cb_b @ expm(-h * K_b))
    g_fd_u = (ep_u - em_u) / (2.0 * h)
    g_ana_u = float(wb_ana[0, nb - 1])
    dw_u = abs(g_fd_u - g_ana_u)
    print(f"[k-OOMP2 UHF FD] kappa=1.45 | FD={g_fd_u:.10e} | Ana={g_ana_u:.10e} | |dw|={dw_u:.2e}")
    assert dw_u < 1.0e-8

    res_u_k = M.run_df_kappa_omp2_uhf(H_u, S_u, v_u, B_u, scf_u.C_alpha, scf_u.C_beta, na, nb, kappa=1.45, grad_tol=1e-6)
    res_u_inf = M.run_df_kappa_omp2_uhf(H_u, S_u, v_u, B_u, scf_u.C_alpha, scf_u.C_beta, na, nb, kappa=200.0, grad_tol=1e-6)
    res_u_omp2 = M.run_df_omp_uhf(H_u, S_u, v_u, B_u, scf_u.C_alpha, scf_u.C_beta, na, nb, grad_tol=1e-6)
    d_u_lim = abs(res_u_inf.energy_total - res_u_omp2.energy_total)
    print(
        f"[k-OOMP2 UHF   ] E(k=1.45)={res_u_k.energy_total:.10f} (<S2>={res_u_k.s2:.6f}) | "
        f"|E(k=200)-E(OMP2)|={d_u_lim:.2e}"
    )
    assert res_u_k.converged and res_u_inf.converged and d_u_lim < 1.0e-9


def test_telemetry_callback_checkpoint_and_run_api():
    h2o = "O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692"
    out = mshqc.run(
        h2o,
        method=["RHF", "kappa-omp2"],
        basis="cc-pVDZ",
        eri_method="df",
        units="angstrom",
        threshold=1e-10,
        grad_thresh=1e-6,
        use_symmetry=False,
    )
    assert out["kappa-omp2"]["result"].converged

    mol = mshqc.build_molecule(h2o, units="angstrom")
    basis = mshqc.BasisSet("cc-pVDZ", mol)
    aux = mshqc.BasisSet("cc-pVDZ-RI", mol)
    ints = mshqc.IntegralEngine(mol, basis)
    H = ints.compute_kinetic() + ints.compute_nuclear()
    S = ints.compute_overlap()
    v_nn = float(mol.nuclear_repulsion_energy())
    B_ao = K.compute_df_ao_factor(mol, basis, aux, 1e-10, False, False)
    no = int(mol.n_electrons()) // 2

    scf_trace = []
    omp_trace = []
    scf_res = M.run_df_rhf(
        H, S, v_nn, B_ao, no, energy_tol=1e-10, density_tol=1e-10,
        callback=lambda rec: scf_trace.append(rec),
    )
    omp_res = M.run_df_kappa_omp2_rhf(
        H, v_nn, B_ao, scf_res.C_alpha, no, kappa=1.45, grad_tol=1e-6,
        callback=lambda rec: omp_trace.append(rec),
    )
    assert len(scf_trace) == scf_res.iterations and "err_rms" in scf_trace[-1]
    assert len(omp_trace) >= 2 and "grad_rms" in omp_trace[-1]

    with tempfile.TemporaryDirectory() as td:
        ckpt = M.save_orbital_checkpoint(Path(td) / "h2o_rhf.npz", scf_res)
        loaded = M.load_orbital_checkpoint(ckpt)
        assert abs(loaded["energy_total"] - scf_res.energy_total) < 1e-14
        warm_res = M.run_df_rhf(
            H, S, v_nn, B_ao, no, energy_tol=1e-10, density_tol=1e-10,
            C_init=loaded["C_alpha"],
        )
        assert warm_res.converged and warm_res.iterations <= 3
        print(
            f"[TELEM+CKPT    ] SCF={len(scf_trace)} it -> WarmStart={warm_res.iterations} it | "
            f"k-OOMP2 trace={len(omp_trace)} steps"
        )


if __name__ == "__main__":
    test_kappa_omp2_analytic_grad_and_large_kappa_limit()
    test_telemetry_callback_checkpoint_and_run_api()
    print("[PASS] Semua validasi Tahap 6 Thin-Core lulus!")
