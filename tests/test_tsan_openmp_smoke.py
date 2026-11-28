#!/usr/bin/env python3
"""
Focused C++20 OpenMP ThreadSanitizer (TSan) race-detection smoke suite.

Exercises all major OpenMP-parallelized C++20 kernels (DF-AO assembly, RHF/UHF JK,
3-index MO transform, OMP2/OMP3 contractions, O(N^4) LT-SOS-OMP2 RHF/UHF kernels,
and single-point analytic DF nuclear gradients) without 31x finite-difference loops
so TSan completes deterministically in ~1-2 minutes.
"""

from __future__ import annotations

import numpy as np
import mshqc
from mshqc import kernels as K
from mshqc import methods as M


def test_tsan_openmp_rhf_omp3_lt_sos_and_single_grad() -> None:
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
    scf_r = M.run_df_rhf(H, S, v_nn, B_ao, 5, energy_tol=1e-9, density_tol=1e-9)
    assert scf_r.converged

    lt_r = M.run_df_lt_sos_omp2_rhf(H, v_nn, B_ao, scf_r.C_alpha, 5, c_os=1.3, grad_tol=1e-5)
    assert lt_r.converged and np.isfinite(lt_r.energy_total)

    omp3_r = M.run_df_omp_rhf(H, v_nn, B_ao, scf_r.C_alpha, 5, method_name="omp3", grad_tol=1e-5)
    assert omp3_r.converged and np.isfinite(omp3_r.energy_total)

    e_g, g_ana = M.compute_df_gradient(
        h2o,
        method="omp2",
        basis_name="cc-pVDZ",
        reference="rhf",
        threshold=1e-10,
        orb_grad_tol=1e-6,
    )
    assert np.isfinite(e_g) and g_ana.shape == (3, 3) and np.all(np.isfinite(g_ana))


def test_tsan_openmp_uhf_lt_sos_and_omp3() -> None:
    oh = mshqc.Molecule(0, 2)
    oh.add_atom("O", 0.0, 0.0, 0.0, units="angstrom")
    oh.add_atom("H", 0.0, 0.0, 0.9697, units="angstrom")

    basis = mshqc.BasisSet("cc-pVDZ", oh)
    aux = mshqc.BasisSet("cc-pVDZ-RI", oh)
    ints = mshqc.IntegralEngine(oh, basis)
    H = ints.compute_kinetic() + ints.compute_nuclear()
    S = ints.compute_overlap()
    v_nn = float(oh.nuclear_repulsion_energy())

    B_ao = K.compute_df_ao_factor(oh, basis, aux, 1e-10, False, False)
    scf_u = M.run_df_uhf(H, S, v_nn, B_ao, 5, 4, energy_tol=1e-9, density_tol=1e-9)
    assert scf_u.converged

    lt_u = M.run_df_lt_sos_omp2_uhf(
        H, S, v_nn, B_ao, scf_u.C_alpha, scf_u.C_beta, 5, 4, c_os=1.3, grad_tol=1e-5
    )
    assert lt_u.converged and np.isfinite(lt_u.energy_total)

    omp3_u = M.run_df_omp_uhf(
        H, S, v_nn, B_ao, scf_u.C_alpha, scf_u.C_beta, 5, 4, method_name="omp3", grad_tol=1e-5
    )
    assert omp3_u.converged and np.isfinite(omp3_u.energy_total)
