#!/usr/bin/env python3
"""
Validasi Tahap 7 & 8 (Irreducible C++ Floor + O(N^4) LT-SOS-OMP2 + Analytic DF-kappa-OOMP2 Gradients):
1. Profil C++ Irreducible Thin-Core (`bindings.cc < 190 LOC`, `fock_builder.cc`/`screening.cc` di legacy).
2. O(N^4) Zero-T2 `LT-SOS-OMP2` cocok < 1e-7 Ha terhadap O(N^5) `SOS-OMP2` pada:
   - H2O (RHF), OH radikal (UHF), dan Benzene C6H6 114 BF (RHF, frozen-core=6).
3. Gradien nuklir analitik eksak DF-kappa-OOMP2 (kappa=1.45) cocok < 1e-6 Ha/Bohr vs Central FD
   pada H2O (RHF) dan OH (UHF), serta optimisasi geometri Thin-Core H2O `kappa-omp2`.
"""
from pathlib import Path
import time
import numpy as np
import mshqc
from mshqc import kernels as K
from mshqc import methods as M


def _central_fd_kappa_grad(mol, basis: str, ref: str, kappa: float = 1.45, h: float = 5.0e-4) -> np.ndarray:
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
                    m_d, method="kappa-omp2", basis_name=basis, reference=ref,
                    threshold=1e-12, orb_grad_tol=1e-10, kappa=kappa,
                )
                e_vals.append(e_val)
            g_fd[a, d] = (e_vals[0] - e_vals[1]) / (2.0 * h)
    return g_fd


def test_irreducible_thin_core_boundaries():
    root = Path(__file__).resolve().parent.parent
    bind_loc = len((root / "python" / "bindings.cc").read_text(encoding="utf-8").splitlines())
    print(f"[IRREDUCIBLE] bindings.cc = {bind_loc} LOC | THIN_CORE_ONLY = {mshqc.THIN_CORE_ONLY}")
    assert bind_loc < 190
    if mshqc.THIN_CORE_ONLY:
        assert not hasattr(mshqc, "Screening")
        assert not hasattr(mshqc, "ShellPair")
        assert not hasattr(K, "FockBuilder")
    else:
        assert hasattr(mshqc, "Screening") and hasattr(K, "FockBuilder")


def test_lt_sos_omp2_on4_matches_sos_omp2_rhf_uhf_and_benzene():
    # 1. H2O RHF
    h2o = mshqc.Molecule(0, 1)
    h2o.add_atom("O", 0.0, 0.0, 0.1173, units="angstrom")
    h2o.add_atom("H", 0.0, 0.7572, -0.4692, units="angstrom")
    h2o.add_atom("H", 0.0, -0.7572, -0.4692, units="angstrom")
    b = mshqc.BasisSet("cc-pVDZ", h2o)
    a = mshqc.BasisSet("cc-pVDZ-RI", h2o)
    i = mshqc.IntegralEngine(h2o, b)
    H, S, v = i.compute_kinetic() + i.compute_nuclear(), i.compute_overlap(), float(h2o.nuclear_repulsion_energy())
    B = K.compute_df_ao_factor(h2o, b, a, 1e-10, False, False)
    scf = M.run_df_rhf(H, S, v, B, 5, energy_tol=1e-10, density_tol=1e-10)

    r_ref = M.run_df_omp_rhf(H, v, B, scf.C_alpha, 5, method_name="sos-omp2", c_os=1.3, c_ss=0.0, grad_tol=1e-7)
    r_lt = M.run_df_lt_sos_omp2_rhf(H, v, B, scf.C_alpha, 5, c_os=1.3, grad_tol=1e-7)
    de_h2o = abs(r_lt.energy_total - r_ref.energy_total)
    print(f"[LT-SOS H2O ] E_LT={r_lt.energy_total:.10f} | E_Ref={r_ref.energy_total:.10f} | |dE|={de_h2o:.2e}")
    assert r_lt.converged and de_h2o < 1.0e-7

    # 2. OH UHF
    oh = mshqc.Molecule(0, 2)
    oh.add_atom("O", 0.0, 0.0, 0.0, units="angstrom")
    oh.add_atom("H", 0.0, 0.0, 0.9697, units="angstrom")
    bu, au = mshqc.BasisSet("cc-pVDZ", oh), mshqc.BasisSet("cc-pVDZ-RI", oh)
    iu = mshqc.IntegralEngine(oh, bu)
    Hu, Su, vu = iu.compute_kinetic() + iu.compute_nuclear(), iu.compute_overlap(), float(oh.nuclear_repulsion_energy())
    Bu = K.compute_df_ao_factor(oh, bu, au, 1e-10, False, False)
    scfu = M.run_df_uhf(Hu, Su, vu, Bu, 5, 4, energy_tol=1e-10, density_tol=1e-10)

    u_ref = M.run_df_omp_uhf(Hu, Su, vu, Bu, scfu.C_alpha, scfu.C_beta, 5, 4, method_name="sos-omp2", c_os=1.3, c_ss=0.0, grad_tol=1e-6)
    u_lt = M.run_df_lt_sos_omp2_uhf(Hu, Su, vu, Bu, scfu.C_alpha, scfu.C_beta, 5, 4, c_os=1.3, grad_tol=1e-6)
    de_oh = abs(u_lt.energy_total - u_ref.energy_total)
    print(f"[LT-SOS OH  ] E_LT={u_lt.energy_total:.10f} | E_Ref={u_ref.energy_total:.10f} | |dE|={de_oh:.2e}")
    assert u_lt.converged and de_oh < 1.0e-7

    # 3. Benzene 114 BF (frozen-core=6): verifikasi akurasi & zero-T2 speed
    benzene = [
        ("C", 1.3960, 0.0000, 0.0), ("C", 0.6980, 1.2090, 0.0),
        ("C", -0.6980, 1.2090, 0.0), ("C", -1.3960, 0.0000, 0.0),
        ("C", -0.6980, -1.2090, 0.0), ("C", 0.6980, -1.2090, 0.0),
        ("H", 2.4820, 0.0000, 0.0), ("H", 1.2410, 2.1495, 0.0),
        ("H", -1.2410, 2.1495, 0.0), ("H", -2.4820, 0.0000, 0.0),
        ("H", -1.2410, -2.1495, 0.0), ("H", 1.2410, -2.1495, 0.0),
    ]
    mol_b = mshqc.build_molecule(benzene, units="angstrom")
    bb, ab = mshqc.BasisSet("cc-pVDZ", mol_b), mshqc.BasisSet("cc-pVDZ-RI", mol_b)
    ib = mshqc.IntegralEngine(mol_b, bb)
    Hb, Sb, vb = ib.compute_kinetic() + ib.compute_nuclear(), ib.compute_overlap(), float(mol_b.nuclear_repulsion_energy())
    Bb = K.compute_df_ao_factor(mol_b, bb, ab, 1e-9, False, False)
    P_sad = M.build_sad_guess(mol_b, bb)
    scfb = M.run_df_rhf(Hb, Sb, vb, Bb, 21, energy_tol=1e-9, density_tol=1e-9, P_init=P_sad)

    t0 = time.perf_counter()
    b_ref = M.run_df_omp_rhf(Hb, vb, Bb, scfb.C_alpha, 21, method_name="sos-omp2", c_os=1.3, c_ss=0.0, grad_tol=1e-5, n_frozen=6)
    t_ref = time.perf_counter() - t0

    t1 = time.perf_counter()
    b_lt = M.run_df_lt_sos_omp2_rhf(Hb, vb, Bb, scfb.C_alpha, 21, c_os=1.3, grad_tol=1e-5, n_frozen=6)
    t_lt = time.perf_counter() - t1

    de_bz = abs(b_lt.energy_total - b_ref.energy_total)
    print(
        f"[LT-SOS C6H6] E_LT={b_lt.energy_total:.10f} ({t_lt:.3f}s) | "
        f"O(N^5)={b_ref.energy_total:.10f} ({t_ref:.3f}s) | |dE|={de_bz:.2e}"
    )
    assert b_lt.converged and de_bz < 1.0e-7


def test_kappa_omp2_analytic_nuclear_gradient_and_geom_opt():
    h2o = mshqc.Molecule(0, 1)
    h2o.add_atom(8, 0.0, 0.0, 0.22)
    h2o.add_atom(1, 0.0, 1.45, -0.88)
    h2o.add_atom(1, 0.0, -1.45, -0.88)

    oh = mshqc.Molecule(0, 2)
    oh.add_atom(8, 0.0, 0.0, 0.0)
    oh.add_atom(1, 0.0, 0.0, 1.90)

    for label, mol, ref in (("H2O", h2o, "rhf"), ("OH", oh, "uhf")):
        e_ana, g_ana = M.compute_df_gradient(
            mol, method="kappa-omp2", basis_name="cc-pVDZ", reference=ref,
            threshold=1e-12, orb_grad_tol=1e-10, kappa=1.45,
        )
        g_fd = _central_fd_kappa_grad(mol, "cc-pVDZ", ref, kappa=1.45)
        diff_fd = float(np.max(np.abs(g_ana - g_fd)))
        trans = float(np.max(np.abs(np.sum(g_ana, axis=0))))
        print(f"[GRAD k-OOMP2 {label:3s}] E={e_ana:.10f} | |ana-FD|={diff_fd:.2e} | trans={trans:.2e}")
        assert diff_fd < 1.0e-6 and trans < 1.0e-11

    # End-to-end geometry optimization of H2O using analytic DF-kappa-OOMP2
    m_opt = mshqc.Molecule(0, 1)
    m_opt.add_atom("O", 0.0, 0.0, 0.125, units="angstrom")
    m_opt.add_atom("H", 0.0, 0.780, -0.490, units="angstrom")
    m_opt.add_atom("H", 0.0, -0.780, -0.490, units="angstrom")
    opt = M.optimize_geometry_thin_core(
        m_opt, method="kappa-omp2", eri_mode="df", basis="cc-pVDZ",
        g_rms_tol=1e-6, g_max_tol=2e-6, dx_rms_tol=4e-6, dx_max_tol=6e-6,
    )
    r_oh = opt.bond_lengths_angstrom["O1-H2"]
    print(f"[GEOM k-OOMP2 H2O] R_OH={r_oh:.6f} A | E={opt.final_energy:.10f} Ha ({opt.iterations} it)")
    assert opt.converged and 0.94 < r_oh < 0.99


if __name__ == "__main__":
    test_irreducible_thin_core_boundaries()
    test_lt_sos_omp2_on4_matches_sos_omp2_rhf_uhf_and_benzene()
    test_kappa_omp2_analytic_nuclear_gradient_and_geom_opt()
    print("[PASS] Semua validasi Tahap 7 & 8 lulus!")
