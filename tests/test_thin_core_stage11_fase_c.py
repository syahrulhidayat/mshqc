#!/usr/bin/env python3
"""
Validasi Fase C (v2.2.0 Scalability & Ecosystem):
1. AdaptiveLaplaceCache menghindari penyelesaian ulang NNLS di setiap iterasi makro LT-SOS-OMP2.
2. Pluggable Kernel Backend Registry (`get_backend("cpp")` vs `get_backend("numpy")` + `use_backend`).
3. OO-Double-Hybrid DFT (`run_oo_double_hybrid_rhf`: OO-B2PLYP & OO-DSD-PBEP86, jika PySCF tersedia).
"""
import numpy as np
import pytest
import mshqc
from mshqc.backends import available_backends, get_backend, register_backend, use_backend
from mshqc import methods as M
from mshqc.methods.lt_sos_omp2 import AdaptiveLaplaceCache, _lt_sos_omp2_rhf_callback
from mshqc.methods.omp_hybrid import optimize_orbitals_rhf


def test_adaptive_laplace_cache_reuses_quadrature():
    h2o = mshqc.build_molecule("O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692", units="angstrom")
    basis = mshqc.BasisSet("cc-pVDZ", h2o)
    aux = mshqc.BasisSet("cc-pVDZ-RI", h2o)
    ints = mshqc.IntegralEngine(h2o, basis)
    H = ints.compute_kinetic() + ints.compute_nuclear()
    S = ints.compute_overlap()
    v_nn = float(h2o.nuclear_repulsion_energy())
    B_ao = get_backend("cpp").compute_df_ao_factor(h2o, basis, aux, 1e-10)
    scf_r = M.run_df_rhf(H, S, v_nn, B_ao, 5, P_init=M.build_sad_guess(h2o, basis))

    cache = AdaptiveLaplaceCache()
    res = optimize_orbitals_rhf(
        H, v_nn, B_ao, scf_r.C_alpha, 5,
        corr_fn=_lt_sos_omp2_rhf_callback(c_os=1.3, cache=cache),
        method_name="lt-sos-omp2", grad_tol=1e-7,
    )
    assert res.converged and res.iterations >= 3
    assert cache.solves == 1, f"NNLS seharusnya hanya dihitung 1x, aktual={cache.solves}"
    print(f"[LAPLACE CACHE] Converged in {res.iterations} iter with only {cache.solves} NNLS solve")


def test_pluggable_backend_registry_cpp_vs_numpy():
    assert set(available_backends()) >= {"cpp", "numpy"}
    h2o = mshqc.build_molecule("O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692", units="angstrom")
    basis = mshqc.BasisSet("cc-pVDZ", h2o)
    aux = mshqc.BasisSet("cc-pVDZ-RI", h2o)
    ints = mshqc.IntegralEngine(h2o, basis)
    H = ints.compute_kinetic() + ints.compute_nuclear()
    S = ints.compute_overlap()
    v_nn = float(h2o.nuclear_repulsion_energy())

    k_cpp = get_backend("cpp")
    B_ao = k_cpp.compute_df_ao_factor(h2o, basis, aux, 1e-10)
    scf_r = M.run_df_rhf(H, S, v_nn, B_ao, 5, P_init=M.build_sad_guess(h2o, basis))
    C = scf_r.C_alpha
    nbf = int(basis.n_basis_functions())

    with use_backend("numpy") as k_np:
        J_c, K_c = k_cpp.build_jk_3c_rhf(B_ao, np.ascontiguousarray(C[:, :5]), nbf)
        J_n, K_n = k_np.build_jk_3c_rhf(B_ao, C[:, :5], nbf)
        assert np.allclose(J_c, J_n, atol=1e-12) and np.allclose(K_c, K_n, atol=1e-12)

        Boo_c, Bov_c, Bvv_c = k_cpp.transform_3c_mo_blocks(
            B_ao, np.ascontiguousarray(C[:, :5]), np.ascontiguousarray(C[:, 5:]), nbf
        )
        Boo_n, Bov_n, Bvv_n = k_np.transform_3c_mo_blocks(B_ao, C[:, :5], C[:, 5:], nbf)
        assert np.allclose(Boo_c, Boo_n, atol=1e-12)
        assert np.allclose(Bov_c, Bov_n, atol=1e-12)
        assert np.allclose(Bvv_c, Bvv_n, atol=1e-12)

        eos_c, ess_c, _ = k_cpp.eval_rmp2_df(Bov_c, scf_r.orbital_energies_alpha[:5], scf_r.orbital_energies_alpha[5:])
        eos_n, ess_n, _ = k_np.eval_rmp2_df(Bov_n, scf_r.orbital_energies_alpha[:5], scf_r.orbital_energies_alpha[5:])
        assert abs(eos_c - eos_n) < 1e-12 and abs(ess_c - ess_n) < 1e-12

    register_backend("custom_alias", k_cpp)
    assert get_backend("custom_alias") is k_cpp
    print("[BACKENDS    ] C++20 <-> Pure-NumPy kernel parity < 1e-12 & custom registry OK")


def test_oo_double_hybrid_dft_if_pyscf_installed():
    pyscf = pytest.importorskip("pyscf")
    mol = pyscf.gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", basis="cc-pVDZ", verbose=0)
    res_b2 = M.run_oo_double_hybrid_rhf(mol, aux_basis="cc-pVDZ-RI", functional="oo-b2plyp", grad_tol=1e-5)
    res_dsd = M.run_oo_double_hybrid_rhf(mol, aux_basis="cc-pVDZ-RI", functional="oo-dsd-pbep86", grad_tol=1e-5)
    assert res_b2.converged and res_dsd.converged
    print(f"[OO-DH DFT   ] OO-B2PLYP={res_b2.energy_total:.8f} Ha | OO-DSD-PBEP86={res_dsd.energy_total:.8f} Ha")


if __name__ == "__main__":
    test_adaptive_laplace_cache_reuses_quadrature()
    test_pluggable_backend_registry_cpp_vs_numpy()
    print("[PASS] Semua validasi Fase C lulus!")
