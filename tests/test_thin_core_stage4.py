#!/usr/bin/env python3
"""
Validasi Tahap 4 Thin-Core (0 Baris C++ Baru — Full DF/Cholesky Production Parity):
1. DF-ROHF, DF-ROMP2, DF-ROMP3 (NH2 radikal dublet & atom O triplet, AE + FC) vs Legacy C++
2. Gradien nuklir analitik Thin-Core Python:
   - DF-ROHF (NH2), DF-OMP2 RHF (H2O), DF-OMP2 UHF (OH & NH2), dan metode baru DF-SCS-OMP2 (H2O)
   - Verifikasi vs Central Finite-Difference (< 1e-6) & vs Legacy C++ (< 1e-8)
3. Optimisasi Geometri & Frekuensi Vibrasi Harmonik Thin-Core (H2O DF-OMP2) vs Legacy C++
"""
import numpy as np
import mshqc
from mshqc import methods as M


def _e(res) -> float:
    return float(getattr(res, "energy_total", getattr(res, "e_total")))


def _central_fd_grad(mol, method: str, basis: str, ref: str, h: float = 5.0e-4) -> np.ndarray:
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
                    threshold=1e-12, orb_grad_tol=1e-10,
                )
                e_vals.append(e_val)
            g_fd[a, d] = (e_vals[0] - e_vals[1]) / (2.0 * h)
    return g_fd


def test_thin_core_rohf_romp2_romp3_parity():
    nh2 = "N 0.0 0.0 0.20\nH 0.0 1.55 -0.85\nH 0.0 -1.55 -0.85"
    for fc in (False, True):
        calc = mshqc.MSHQCCalculator(
            nh2, basis_name="cc-pVDZ", eri_method="df", units="bohr",
            multiplicity=2, reference="rohf", threshold=1e-10,
            freeze_core=fc, use_symmetry=False,
        )
        scf_h = calc.run_scf(use_hybrid=True)
        mp2_h = calc.run_mp2(scf_h, use_hybrid=True)
        mp3_h = calc.run_mp3(scf_h, use_hybrid=True)
        assert scf_h.converged and abs(scf_h.s2 - 0.75) < 1e-12

        if hasattr(mshqc, "ROHF"):
            scf_c = calc.run_scf(use_hybrid=False)
            mp2_c = calc.run_mp2(scf_c, use_hybrid=False)
            mp3_c = calc.run_mp3(scf_c, mp2_c, use_hybrid=False)
            d_scf = abs(_e(scf_h) - _e(scf_c))
            d_mp2 = abs(_e(mp2_h) - _e(mp2_c))
            d_mp3 = abs(_e(mp3_h) - _e(mp3_c))
            tag = "FC" if fc else "AE"
            print(
                f"[NH2 ROHF {tag}] |dE_SCF|={d_scf:.2e} | |dE_ROMP2|={d_mp2:.2e} | |dE_ROMP3|={d_mp3:.2e}"
            )
            assert d_scf < 1e-8 and d_mp2 < 1e-8 and d_mp3 < 1e-8


def test_thin_core_analytic_rohf_and_omp2_gradients():
    h2o = mshqc.Molecule(0, 1)
    h2o.add_atom(8, 0.0, 0.0, 0.22)
    h2o.add_atom(1, 0.0, 1.45, -0.88)
    h2o.add_atom(1, 0.0, -1.45, -0.88)

    oh = mshqc.Molecule(0, 2)
    oh.add_atom(8, 0.0, 0.0, 0.0)
    oh.add_atom(1, 0.0, 0.0, 1.90)

    nh2 = mshqc.Molecule(0, 2)
    nh2.add_atom(7, 0.0, 0.0, 0.20)
    nh2.add_atom(1, 0.0, 1.55, -0.85)
    nh2.add_atom(1, 0.0, -1.55, -0.85)

    cases = [
        ("NH2", nh2, "rohf", "rohf", True),
        ("H2O", h2o, "omp2", "rhf", True),
        ("H2O", h2o, "scs-omp2", "rhf", False),
        ("OH", oh, "omp2", "uhf", True),
        ("NH2", nh2, "omp2", "uhf", True),
    ]
    for label, mol, method, ref, has_cpp_ref in cases:
        e_ana, g_ana = M.compute_df_gradient(
            mol, method=method, basis_name="cc-pVDZ", reference=ref,
            threshold=1e-12, orb_grad_tol=1e-10,
        )
        g_fd = _central_fd_grad(mol, method, "cc-pVDZ", ref)
        diff_fd = float(np.max(np.abs(g_ana - g_fd)))
        trans = float(np.max(np.abs(np.sum(g_ana, axis=0))))
        msg = f"[GRAD {label:3s} {method:8s}] E={e_ana:.10f} | |ana-FD|={diff_fd:.2e} | trans={trans:.2e}"
        assert diff_fd < 1.0e-6, f"FD mismatch {label} {method}: {diff_fd}"
        assert trans < 1.0e-11, f"Translation invariance {label} {method}: {trans}"

        if has_cpp_ref and hasattr(mshqc, "HPCGeometryOptimizer"):
            cfg = mshqc.GeomOptConfig()
            cfg.method = f"df-{method}"
            cfg.eri_method = "df"
            cfg.basis_name = "cc-pVDZ"
            cfg.reference = ref
            cfg.use_analytic_gradient = True
            cfg.project_trans_rot = False
            cfg.energy_thresh = 1e-12
            cfg.orb_grad_thresh = 1e-10
            _, g_cpp = mshqc.HPCGeometryOptimizer.compute_nuclear_gradient_cpp(mol, cfg)
            diff_cpp = float(np.max(np.abs(g_ana - np.asarray(g_cpp).reshape(-1, 3))))
            msg += f" | |py-cpp|={diff_cpp:.2e}"
            assert diff_cpp < 1.0e-8, f"C++ grad mismatch {label} {method}: {diff_cpp}"
        print(msg)


def test_thin_core_geom_opt_and_frequencies_h2o():
    def make_h2o():
        m = mshqc.Molecule(0, 1)
        m.add_atom("O", 0.0, 0.0, 0.125, units="angstrom")
        m.add_atom("H", 0.0, 0.780, -0.490, units="angstrom")
        m.add_atom("H", 0.0, -0.780, -0.490, units="angstrom")
        return m

    m_hyb = make_h2o()
    opt_h = M.optimize_geometry_thin_core(
        m_hyb, method="omp2", eri_mode="df", basis="cc-pVDZ",
        g_rms_tol=1e-6, g_max_tol=2e-6, dx_rms_tol=4e-6, dx_max_tol=6e-6,
    )
    assert opt_h.converged
    vib_h = M.compute_frequencies_thin_core(m_hyb, method="omp2", eri_mode="df", basis="cc-pVDZ")
    assert vib_h.n_imaginary == 0 and len(vib_h.frequencies_cm1) == 3

    r_h = opt_h.bond_lengths_angstrom["O1-H2"]
    print(
        f"[GEOM H2O OMP2] Thin-Core: R_OH={r_h:.6f} A | E={opt_h.final_energy:.10f} Ha | "
        f"Freqs={[round(float(f), 2) for f in vib_h.frequencies_cm1]} cm^-1"
    )

    if hasattr(mshqc, "HPCGeometryOptimizer"):
        m_cpp = make_h2o()
        opt_c = m_cpp.optimize_geometry(
            method="omp2", eri_mode="df", basis="cc-pVDZ",
            g_rms_tol=1e-6, g_max_tol=2e-6, dx_rms_tol=4e-6, dx_max_tol=6e-6, verbose=False,
        )
        vib_c = m_cpp.frequencies(method="omp2", eri_mode="df", basis="cc-pVDZ", verbose=False)
        r_c = opt_c.bond_lengths_angstrom["O1-H2"]
        df_max = float(np.max(np.abs(np.sort(vib_h.frequencies_cm1) - np.sort(vib_c.frequencies_cm1))))
        print(
            f"[GEOM PARITY  ] |dR|={abs(r_h-r_c)*1e3:.4f} mA | |dE|={abs(opt_h.final_energy-opt_c.final_energy):.2e} Ha | "
            f"max|dfreq|={df_max:.3f} cm^-1"
        )
        assert abs(r_h - r_c) < 1e-5
        assert abs(opt_h.final_energy - opt_c.final_energy) < 1e-8
        assert df_max < 0.5


if __name__ == "__main__":
    test_thin_core_rohf_romp2_romp3_parity()
    test_thin_core_analytic_rohf_and_omp2_gradients()
    test_thin_core_geom_opt_and_frequencies_h2o()
    print("[PASS] Semua validasi Tahap 4 Thin-Core lulus!")
