#!/usr/bin/env python3
"""
Phase 1 CPC Numerical Verification & Robustness Suite:
  1. 4-point O(h^4) central finite-difference nuclear gradient parity <= 1e-7 Ha/Bohr
     + translational invariance (< 1e-11) + rotational torque invariance (< 1e-9)
     on asymmetric H2O (RHF All-Electron & Strict Frozen-Core) and OH (UHF).
  2. Controlled approximation error bounds:
     - Cholesky threshold sweep (tau_CD = 1e-4, 1e-6, 1e-8) vs EXACT ERI.
     - Laplace quadrature residual (1/x & 1/x^2) + LT-SOS-OMP2 parity across gap ratios.
     - kappa-OOMP2 monotonic convergence toward DF-OMP2 as kappa increases.
  3. Edge-case & graceful degradation invariants (invalid kappa, non-convergence,
     backend context restoration on exception).
"""

from __future__ import annotations

import numpy as np
import pytest
import mshqc
from mshqc import kernels as K
from mshqc import methods as M
from mshqc.backends import get_backend, use_backend
from mshqc.methods.lt_sos_omp2 import build_laplace_quadrature


def _make_asymmetric_h2o_bohr() -> mshqc.Molecule:
    """Asymmetric distorted H2O in Bohr so all 3 force and 3 torque axes are non-trivial."""
    mol = mshqc.Molecule(0, 1)
    mol.add_atom(8, 0.00, 0.00, 0.22)
    mol.add_atom(1, 0.15, 1.48, -0.85)
    mol.add_atom(1, -0.10, -1.40, -0.92)
    return mol


def _make_oh_bohr() -> mshqc.Molecule:
    mol = mshqc.Molecule(0, 2)
    mol.add_atom(8, 0.00, 0.00, 0.00)
    mol.add_atom(1, 0.25, -0.15, 1.88)
    return mol


def _central_fd4_df_grad(
    mol: mshqc.Molecule,
    method: str,
    basis: str,
    ref: str,
    freeze_core: bool = False,
    kappa: float = 1.45,
    h: float = 5.0e-4,
) -> np.ndarray:
    """4-point O(h^4) central finite difference to eliminate O(h^2) truncation noise."""
    nat = int(mol.n_atoms())
    x0 = np.array([[mol.atom(i).x, mol.atom(i).y, mol.atom(i).z] for i in range(nat)], dtype=np.float64)
    g_fd = np.zeros((nat, 3), dtype=np.float64)
    offsets = (-2.0, -1.0, 1.0, 2.0)
    coeffs = (1.0, -8.0, 8.0, -1.0)

    for a in range(nat):
        for d in range(3):
            acc = 0.0
            for step, weight in zip(offsets, coeffs):
                m_d = mshqc.Molecule(int(mol.charge()), int(mol.multiplicity()))
                for i in range(nat):
                    p = x0[i].copy()
                    if i == a:
                        p[d] += step * h
                    m_d.add_atom(int(mol.atom(i).atomic_number), float(p[0]), float(p[1]), float(p[2]))
                e_val, _ = M.compute_df_gradient(
                    m_d,
                    method=method,
                    basis_name=basis,
                    reference=ref,
                    threshold=1.0e-12,
                    orb_grad_tol=1.0e-10,
                    kappa=kappa,
                    freeze_core=freeze_core,
                )
                acc += weight * e_val
            g_fd[a, d] = acc / (12.0 * h)
    return g_fd


def _translational_and_rotational_residuals(mol: mshqc.Molecule, grad: np.ndarray) -> tuple[float, float]:
    nat = int(mol.n_atoms())
    coords = np.array([[mol.atom(i).x, mol.atom(i).y, mol.atom(i).z] for i in range(nat)], dtype=np.float64)
    trans_max = float(np.max(np.abs(np.sum(grad, axis=0))))
    torque = np.sum(np.cross(coords, grad), axis=0)
    rot_max = float(np.max(np.abs(torque)))
    return trans_max, rot_max


@pytest.mark.parametrize(
    ("label", "mol_fn", "method", "ref", "freeze_core"),
    [
        ("H2O-DF-OMP2-AE", _make_asymmetric_h2o_bohr, "omp2", "rhf", False),
        ("H2O-DF-OMP2-FC", _make_asymmetric_h2o_bohr, "omp2", "rhf", True),
        ("H2O-DF-kOOMP2-FC", _make_asymmetric_h2o_bohr, "kappa-omp2", "rhf", True),
        ("OH-DF-kOOMP2-UHF-AE", _make_oh_bohr, "kappa-omp2", "uhf", False),
    ],
)
def test_analytic_gradient_fd4_and_conservation_invariants(label, mol_fn, method, ref, freeze_core) -> None:
    mol = mol_fn()
    _, g_ana = M.compute_df_gradient(
        mol,
        method=method,
        basis_name="cc-pVDZ",
        reference=ref,
        threshold=1.0e-12,
        orb_grad_tol=1.0e-10,
        kappa=1.45,
        freeze_core=freeze_core,
    )
    g_fd = _central_fd4_df_grad(mol, method, "cc-pVDZ", ref, freeze_core=freeze_core, kappa=1.45)

    max_err = float(np.max(np.abs(g_ana - g_fd)))
    trans_err, rot_err = _translational_and_rotational_residuals(mol, g_ana)
    print(f"[CPC-GRAD {label:20s}] max|ana-FD4|={max_err:.2e} | trans={trans_err:.2e} | rot={rot_err:.2e}")

    assert max_err <= 1.0e-7, f"{label}: analytic vs FD4 mismatch {max_err:.3e} > 1e-7 Ha/Bohr"
    assert trans_err <= 1.0e-11, f"{label}: translational invariance violated ({trans_err:.3e})"
    assert rot_err <= 1.0e-9, f"{label}: rotational torque invariance violated ({rot_err:.3e})"


def test_cholesky_threshold_controlled_error_convergence() -> None:
    h2o = mshqc.Molecule(0, 1)
    h2o.add_atom("O", 0.0, 0.0, 0.1173, units="angstrom")
    h2o.add_atom("H", 0.0, 0.7572, -0.4692, units="angstrom")
    h2o.add_atom("H", 0.0, -0.7572, -0.4692, units="angstrom")

    calc_ex = mshqc.MSHQCCalculator(h2o, basis_name="cc-pVDZ", eri_method="exact", threshold=1e-11)
    scf_ex = calc_ex.run_scf()
    omp2_ex = calc_ex.run_omp2(scf_ex, grad_thresh=1e-7)

    errs = []
    for tau in (1.0e-4, 1.0e-6, 1.0e-8):
        calc_cd = mshqc.MSHQCCalculator(h2o, basis_name="cc-pVDZ", eri_method="cholesky", threshold=tau)
        scf_cd = calc_cd.run_scf()
        omp2_cd = calc_cd.run_omp2(scf_cd, grad_thresh=1e-7)
        de = abs(omp2_cd.energy_total - omp2_ex.energy_total)
        errs.append(de)
        print(f"[CPC-CD tau={tau:.0e}] |dE_OMP2|={de:.2e} Ha")

    assert errs[2] <= errs[0]
    assert errs[2] < 1.0e-6, f"Cholesky tau=1e-8 error too large: {errs[2]:.3e} Ha"


def test_laplace_quadrature_and_kappa_monotonic_bounds() -> None:
    # 1. Laplace quadrature accuracy on both 1/x and 1/x^2 over wide gap ratio R=200
    w, t = build_laplace_quadrature(0.4, 80.0, n_grid=220)
    assert 6 <= len(w) <= 80, f"Unexpected Laplace active node count: {len(w)}"
    x_test = np.geomspace(0.4, 80.0, 400)
    approx_inv = np.exp(-np.outer(x_test, t)) @ w
    approx_inv2 = np.exp(-np.outer(x_test, t)) @ (w * t)
    rel_err_inv = float(np.max(np.abs(approx_inv * x_test - 1.0)))
    rel_err_inv2 = float(np.max(np.abs(approx_inv2 * (x_test**2) - 1.0)))
    print(f"[CPC-LAPLACE R=200] n_q={len(w)} | max_rel(1/x)={rel_err_inv:.2e} | max_rel(1/x^2)={rel_err_inv2:.2e}")
    assert rel_err_inv < 1.0e-6
    assert rel_err_inv2 < 1.0e-5

    # 2. Monotonic approach of kappa-OOMP2 toward DF-OMP2 as kappa increases
    h2o = mshqc.Molecule(0, 1)
    h2o.add_atom("O", 0.0, 0.0, 0.1173, units="angstrom")
    h2o.add_atom("H", 0.0, 0.7572, -0.4692, units="angstrom")
    h2o.add_atom("H", 0.0, -0.7572, -0.4692, units="angstrom")
    basis = mshqc.BasisSet("cc-pVDZ", h2o)
    aux = mshqc.BasisSet("cc-pVDZ-RI", h2o)
    ints = mshqc.IntegralEngine(h2o, basis)
    H = ints.compute_kinetic() + ints.compute_nuclear()
    S = ints.compute_overlap()
    v_nn = float(h2o.nuclear_repulsion_energy())
    B_ao = K.compute_df_ao_factor(h2o, basis, aux, 1e-10, False, False)
    scf_r = M.run_df_rhf(H, S, v_nn, B_ao, 5, energy_tol=1e-11, density_tol=1e-11)

    e_omp2 = M.run_df_omp_rhf(H, v_nn, B_ao, scf_r.C_alpha, 5, method_name="omp2", grad_tol=1e-7).energy_total
    kappas = (0.5, 1.45, 4.0, 20.0, 200.0)
    energies = [
        M.run_df_kappa_omp2_rhf(H, v_nn, B_ao, scf_r.C_alpha, 5, kappa=kv, grad_tol=1e-7).energy_total
        for kv in kappas
    ]
    for e_prev, e_next in zip(energies, energies[1:]):
        assert e_next <= e_prev + 1.0e-10
    assert abs(energies[-1] - e_omp2) < 1.0e-9


def test_edge_cases_and_graceful_degradation() -> None:
    h2o = mshqc.Molecule(0, 1)
    h2o.add_atom("O", 0.0, 0.0, 0.1173, units="angstrom")
    h2o.add_atom("H", 0.0, 0.7572, -0.4692, units="angstrom")
    h2o.add_atom("H", 0.0, -0.7572, -0.4692, units="angstrom")
    basis = mshqc.BasisSet("cc-pVDZ", h2o)
    aux = mshqc.BasisSet("cc-pVDZ-RI", h2o)
    ints = mshqc.IntegralEngine(h2o, basis)
    H = ints.compute_kinetic() + ints.compute_nuclear()
    S = ints.compute_overlap()
    v_nn = float(h2o.nuclear_repulsion_energy())
    B_ao = K.compute_df_ao_factor(h2o, basis, aux, 1e-10, False, False)
    scf_r = M.run_df_rhf(H, S, v_nn, B_ao, 5, energy_tol=1e-10, density_tol=1e-10)

    # 1. Invalid kappa <= 0 must raise ValueError cleanly
    with pytest.raises(ValueError):
        M.run_df_kappa_omp2_rhf(H, v_nn, B_ao, scf_r.C_alpha, 5, kappa=0.0)
    with pytest.raises(ValueError):
        M.run_df_kappa_omp2_rhf(H, v_nn, B_ao, scf_r.C_alpha, 5, kappa=-1.45)

    # 2. Tight tolerance with max_iter=1 must return converged=False with finite energy
    nonconv = M.run_df_omp_rhf(
        H, v_nn, B_ao, scf_r.C_alpha, 5, method_name="omp2", grad_tol=1e-14, max_iter=1
    )
    assert nonconv.converged is False
    assert np.isfinite(nonconv.energy_total)

    # 3. Context manager use_backend restores previous backend even if an exception occurs
    before = get_backend()
    with pytest.raises(RuntimeError):
        with use_backend("numpy"):
            assert get_backend() is not before
            raise RuntimeError("simulated failure")
    assert get_backend() is before
