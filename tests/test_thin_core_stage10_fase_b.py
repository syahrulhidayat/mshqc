#!/usr/bin/env python3
"""
Validasi Fase B (v2.1 Quantum Chemistry Upgrades — 0 baris C++ baru):
1. `HybridOMPResult` menyimpan blok 1-RDM korelasi (`g_oo_*`, `g_vv_*`) & `M.correlated_density_ao(res)`
   cocok < 1e-13 vs perakitan manual `build_1rdm_ao`.
2. Gradien nuklir analitik Frozen-Core (`n_frozen=1`) untuk `DF-OMP2` & `DF-kappa-OOMP2`
   pada `H2O` (RHF) & `OH` (UHF) cocok < 1e-6 Ha/Bohr vs Central Finite-Difference.
3. Keadaan tereksitasi `run_df_cis_rhf` & `run_df_tdhf_rhf` (singlet & triplet) pada `H2` & `H2O`:
   - Splitting singlet-triplet E(triplet) < E(singlet)
   - Hubungan variasional E(TDHF) <= E(CIS)
   - Ortogonalitas amplitudo eksitasi.
"""
import numpy as np
import mshqc
from mshqc import kernels as K
from mshqc import methods as M


def _central_fd_fc_grad(mol, method: str, basis: str, ref: str, n_frozen: int = 1, h: float = 5.0e-4) -> np.ndarray:
    nat = int(mol.n_atoms())
    x0 = np.array([[mol.atom(i).x, mol.atom(i).y, mol.atom(i).z] for i in range(nat)], dtype=np.float64)
    g_fd = np.zeros((nat, 3), dtype=np.float64)
    for a in range(nat):
        for d in range(3):
            e_vals = []
            for sign in (+1.0, -1.0):
                m_d = mshqc.Molecule(int(mol.charge()), int(mol.multiplicity()))
                for i in range(nat):
                    p = x0[i].copy()
                    if i == a:
                        p[d] += sign * h
                    m_d.add_atom(int(mol.atom(i).atomic_number), float(p[0]), float(p[1]), float(p[2]))
                e_val, _ = M.compute_df_gradient(
                    m_d, method=method, basis_name=basis, reference=ref,
                    threshold=1e-12, orb_grad_tol=1e-10, n_frozen=n_frozen,
                )
                e_vals.append(e_val)
            g_fd[a, d] = (e_vals[0] - e_vals[1]) / (2.0 * h)
    return g_fd


def test_correlated_density_ao_direct_from_omp_result():
    h2o = mshqc.build_molecule("O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692", units="angstrom")
    calc = mshqc.MSHQCCalculator(h2o, basis_name="cc-pVDZ", eri_method="df", threshold=1e-10, freeze_core=True)
    scf_r = calc.run_scf()
    omp_r = calc.run_omp2(scf_r, grad_thresh=1e-7)

    assert omp_r.g_oo_alpha is not None and omp_r.g_vv_alpha is not None
    P_direct = M.correlated_density_ao(omp_r)
    P_manual = M.build_1rdm_ao(
        omp_r.C_alpha, omp_r.n_occ_alpha,
        g_oo_alpha=omp_r.g_oo_alpha, g_vv_alpha=omp_r.g_vv_alpha, n_frozen=omp_r.n_frozen,
    )
    assert float(np.max(np.abs(P_direct - P_manual))) < 1.0e-13
    S = calc.integrals.compute_overlap()
    no_res = M.compute_natural_orbitals(P_direct, S)
    assert abs(no_res.n_electrons - 10.0) < 1.0e-10 and no_res.n_unpaired_head_gordon > 0.04
    print(f"[1RDM DIRECT] H2O FC-OMP2 N_U={no_res.n_unpaired_head_gordon:.4f} | max|dP|<1e-13")


def test_frozen_core_analytic_nuclear_gradients_vs_fd():
    h2o = mshqc.Molecule(0, 1)
    h2o.add_atom(8, 0.0, 0.0, 0.22)
    h2o.add_atom(1, 0.0, 1.45, -0.88)
    h2o.add_atom(1, 0.0, -1.45, -0.88)

    oh = mshqc.Molecule(0, 2)
    oh.add_atom(8, 0.0, 0.0, 0.0)
    oh.add_atom(1, 0.0, 0.0, 1.90)

    for label, mol, method, ref in (
        ("H2O", h2o, "omp2", "rhf"),
        ("H2O", h2o, "kappa-omp2", "rhf"),
        ("OH",  oh,  "omp2", "uhf"),
        ("OH",  oh,  "kappa-omp2", "uhf"),
    ):
        e_ana, g_ana = M.compute_df_gradient(
            mol, method=method, basis_name="cc-pVDZ", reference=ref,
            threshold=1e-12, orb_grad_tol=1e-10, n_frozen=1,
        )
        g_fd = _central_fd_fc_grad(mol, method, "cc-pVDZ", ref, n_frozen=1)
        diff_fd = float(np.max(np.abs(g_ana - g_fd)))
        trans = float(np.max(np.abs(np.sum(g_ana, axis=0))))
        print(f"[FC GRAD {label:3s} {method:10s}] E={e_ana:.10f} | |ana-FD|={diff_fd:.2e} | trans={trans:.2e}")
        assert diff_fd < 1.0e-6 and trans < 1.0e-11


def test_df_cis_and_tdhf_excited_states():
    h2o = mshqc.build_molecule("O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692", units="angstrom")
    basis = mshqc.BasisSet("cc-pVDZ", h2o)
    aux = mshqc.BasisSet("cc-pVDZ-RI", h2o)
    ints = mshqc.IntegralEngine(h2o, basis)
    H = ints.compute_kinetic() + ints.compute_nuclear()
    S = ints.compute_overlap()
    v_nn = float(h2o.nuclear_repulsion_energy())
    B_ao = K.compute_df_ao_factor(h2o, basis, aux, 1e-10)
    scf_r = M.run_df_rhf(H, S, v_nn, B_ao, 5, P_init=M.build_sad_guess(h2o, basis))

    cis_s = M.run_df_cis_rhf(H, v_nn, B_ao, scf_r.C_alpha, 5, n_states=3, triplet=False)
    cis_t = M.run_df_cis_rhf(H, v_nn, B_ao, scf_r.C_alpha, 5, n_states=3, triplet=True)
    tdhf_s = M.run_df_tdhf_rhf(H, v_nn, B_ao, scf_r.C_alpha, 5, n_states=3, triplet=False)

    assert cis_s.converged and cis_t.converged and tdhf_s.converged
    assert cis_t.excitation_energies_ha[0] < cis_s.excitation_energies_ha[0]
    assert tdhf_s.excitation_energies_ha[0] <= cis_s.excitation_energies_ha[0] + 1.0e-10
    # Orthonormal CIS amplitudes
    ovlp_amps = np.einsum("kvi,lvi->kl", cis_s.amplitudes_vo, cis_s.amplitudes_vo)
    assert np.allclose(ovlp_amps, np.eye(3), atol=1.0e-10)
    print(
        f"[EXCITED H2O] T1(CIS)={cis_t.excitation_energies_ev[0]:.3f} eV < "
        f"S1(TDHF)={tdhf_s.excitation_energies_ev[0]:.3f} eV <= "
        f"S1(CIS)={cis_s.excitation_energies_ev[0]:.3f} eV"
    )


if __name__ == "__main__":
    test_correlated_density_ao_direct_from_omp_result()
    test_frozen_core_analytic_nuclear_gradients_vs_fd()
    test_df_cis_and_tdhf_excited_states()
    print("[PASS] Semua validasi Fase B (v2.1) lulus!")
