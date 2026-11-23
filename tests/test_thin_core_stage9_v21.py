#!/usr/bin/env python3
"""
Validasi Upgrade v2.1:
1. Kernel C++20 OpenMP `eval_lt_sos_intermediates_rhf_df` & `uhf_df` cocok hingga presisi mesin
   (rtol < 1e-13) terhadap implementasi NumPy murni.
2. `compute_natural_orbitals` (kekekalan elektron, NOONs, Head-Gordon N_U pada H2O).
3. `mulliken_and_lowdin_charges` (kekekalan muatan total < 1e-11, polaritas O- / H+ pada H2O).
4. `write_molden` menghasilkan berkas `.molden` valid ([Molden Format], [Atoms], [GTO], [5D], [MO]).
"""
from pathlib import Path
import tempfile
import numpy as np
import mshqc
from mshqc import kernels as K
from mshqc import methods as M


def test_cpp_openmp_lt_sos_kernel_matches_numpy():
    rng = np.random.default_rng(42)
    no, nv, n_aux = 6, 18, 54
    B_ov = np.ascontiguousarray(rng.standard_normal((no * nv, n_aux)))
    d_ov = np.ascontiguousarray(rng.uniform(0.3, 4.0, size=no * nv))
    weights, nodes = M.build_laplace_quadrature(0.6, 8.0)

    # 1. RHF OpenMP kernel parity
    e_cpp, xc_cpp, xd_cpp = K.eval_lt_sos_intermediates_rhf_df(B_ov, d_ov, weights, nodes, 1.3)
    e_sum = 0.0
    xc_ref = np.zeros_like(B_ov)
    xd_ref = np.zeros_like(B_ov)
    for wq, tq in zip(weights, nodes):
        eh = np.exp(-0.5 * tq * d_ov)
        Bw = B_ov * eh[:, None]
        Xq = Bw.T @ Bw
        e_sum += wq * float(np.sum(Xq * Xq))
        Yq = (Bw @ Xq) * eh[:, None]
        xc_ref -= (1.3 * wq) * Yq
        xd_ref -= (1.3 * wq * tq) * Yq

    e_ref = -1.3 * e_sum
    rel_e = abs(e_cpp - e_ref) / max(1.0, abs(e_ref))
    assert np.isclose(e_cpp, e_ref, rtol=1.0e-13, atol=1.0e-12), f"RHF rel_err={rel_e:.2e}"
    assert np.allclose(xc_cpp, xc_ref, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(xd_cpp, xd_ref, rtol=1.0e-12, atol=1.0e-12)

    # 2. UHF OpenMP kernel parity
    na, va, nb, vb = 5, 14, 4, 15
    Ba = np.ascontiguousarray(rng.standard_normal((na * va, n_aux)))
    Bb = np.ascontiguousarray(rng.standard_normal((nb * vb, n_aux)))
    da = np.ascontiguousarray(rng.uniform(0.3, 3.5, size=na * va))
    db = np.ascontiguousarray(rng.uniform(0.3, 3.5, size=nb * vb))
    eu_cpp, xca, xda, xcb, xdb = K.eval_lt_sos_intermediates_uhf_df(Ba, Bb, da, db, weights, nodes, 1.3)

    eu_sum = 0.0
    xca_r, xda_r = np.zeros_like(Ba), np.zeros_like(Ba)
    xcb_r, xdb_r = np.zeros_like(Bb), np.zeros_like(Bb)
    for wq, tq in zip(weights, nodes):
        ea, eb = np.exp(-0.5 * tq * da), np.exp(-0.5 * tq * db)
        Baw, Bbw = Ba * ea[:, None], Bb * eb[:, None]
        Xa, Xb = Baw.T @ Baw, Bbw.T @ Bbw
        eu_sum += wq * float(np.sum(Xa * Xb))
        Ya, Yb = (Baw @ Xb) * ea[:, None], (Bbw @ Xa) * eb[:, None]
        xca_r -= (1.3 * wq) * Ya
        xda_r -= (1.3 * wq * tq) * Ya
        xcb_r -= (1.3 * wq) * Yb
        xdb_r -= (1.3 * wq * tq) * Yb

    eu_ref = -1.3 * eu_sum
    rel_eu = abs(eu_cpp - eu_ref) / max(1.0, abs(eu_ref))
    assert np.isclose(eu_cpp, eu_ref, rtol=1.0e-13, atol=1.0e-12), f"UHF rel_err={rel_eu:.2e}"
    assert np.allclose(xca, xca_r, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(xda, xda_r, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(xcb, xcb_r, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(xdb, xdb_r, rtol=1.0e-12, atol=1.0e-12)
    print(f"[LT-SOS C++ ] RHF rel={rel_e:.2e} | UHF rel={rel_eu:.2e} (nq={len(weights)})")


def test_natural_orbitals_charges_and_molden_export():
    h2o = mshqc.build_molecule(
        "O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692",
        units="angstrom",
    )
    basis = mshqc.BasisSet("cc-pVDZ", h2o)
    aux = mshqc.BasisSet("cc-pVDZ-RI", h2o)
    ints = mshqc.IntegralEngine(h2o, basis)
    H = ints.compute_kinetic() + ints.compute_nuclear()
    S = ints.compute_overlap()
    v_nn = float(h2o.nuclear_repulsion_energy())
    B_ao = K.compute_df_ao_factor(h2o, basis, aux, 1e-10)
    nbf = int(basis.n_basis_functions())
    no = 5

    scf_res = M.run_df_rhf(H, S, v_nn, B_ao, no, P_init=M.build_sad_guess(h2o, basis))
    P_scf = M.scf_density_ao(scf_res)
    no_scf = M.compute_natural_orbitals(P_scf, S)
    assert abs(no_scf.n_electrons - 10.0) < 1.0e-11
    assert no_scf.n_unpaired_head_gordon < 1.0e-10

    omp_res = M.run_df_omp_rhf(H, v_nn, B_ao, scf_res.C_alpha, no, grad_tol=1e-7)
    B_ov = K.transform_3c_mo(B_ao, np.ascontiguousarray(omp_res.C_alpha[:, :no]), np.ascontiguousarray(omp_res.C_alpha[:, no:]), nbf)
    _, _, T2 = K.eval_rmp2_df(B_ov, omp_res.orbital_energies_alpha[:no], omp_res.orbital_energies_alpha[no:])
    g_oo, g_vv = K.eval_omp2_1rdm_rhf(T2, no, nbf - no)
    P_omp2 = M.build_1rdm_ao(omp_res.C_alpha, no, g_oo_alpha=g_oo, g_vv_alpha=g_vv)
    no_omp2 = M.compute_natural_orbitals(P_omp2, S)
    assert abs(no_omp2.n_electrons - 10.0) < 1.0e-10
    assert no_omp2.n_unpaired_head_gordon > 0.05

    pop = M.mulliken_and_lowdin_charges(h2o, basis, S, P_omp2)
    assert abs(float(np.sum(pop.mulliken_charges))) < 1.0e-11
    assert abs(float(np.sum(pop.lowdin_charges))) < 1.0e-11
    assert pop.lowdin_charges[0] < 0.0 and pop.lowdin_charges[1] > 0.0

    with tempfile.TemporaryDirectory() as td:
        mfile = M.write_molden(
            Path(td) / "h2o_omp2_no.molden",
            h2o, basis, no_omp2.orbitals_ao, np.zeros(nbf), no_omp2.occupations,
        )
        txt = mfile.read_text(encoding="utf-8")
        for section in ("[Molden Format]", "[Atoms] AU", "[GTO]", "[5D]", "[MO]"):
            assert section in txt

    print(
        f"[PROPS H2O  ] N_U(RHF)={no_scf.n_unpaired_head_gordon:.2e} | "
        f"N_U(OMP2)={no_omp2.n_unpaired_head_gordon:.4f} | "
        f"q_O(Lowdin)={pop.lowdin_charges[0]:.4f} | Molden OK"
    )


if __name__ == "__main__":
    test_cpp_openmp_lt_sos_kernel_matches_numpy()
    test_natural_orbitals_charges_and_molden_export()
    print("[PASS] Semua validasi Upgrade v2.1 lulus!")
