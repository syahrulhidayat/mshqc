#!/usr/bin/env python3
"""
Validasi Stage 12 (Hybrid-v2 Engine Polish, Dynamic Backends, Pure-Python Abelian Symmetry,
Frozen-Core Frequencies & Optional PyTorch/CUDA Kernel Backend).
"""
from __future__ import annotations

import importlib.util
import numpy as np
import pytest

import mshqc
from mshqc import methods as M
from mshqc.backends import enable_torch_backend, use_backend


def test_dynamic_backend_numpy_end_to_end_scf_and_mp2():
    """Pastikan use_backend('numpy') benar-benar mengeksekusi SCF & MP2 dengan paritas < 1e-11 Ha."""
    h2o = mshqc.molecule("H2O")
    calc = mshqc.MSHQCCalculator(h2o, basis_name="cc-pVDZ", eri_method="df")

    scf_cpp = calc.run_scf()
    mp2_cpp = calc.run_mp2(scf_cpp)

    with use_backend("numpy"):
        scf_np = calc.run_scf()
        mp2_np = calc.run_mp2(scf_np)

    d_scf = abs(scf_cpp.energy_total - scf_np.energy_total)
    d_mp2 = abs(mp2_cpp.energy_total - mp2_np.energy_total)
    assert d_scf < 1.0e-11, f"Selisih SCF cpp vs numpy terlalu besar: {d_scf}"
    assert d_mp2 < 1.0e-11, f"Selisih MP2 cpp vs numpy terlalu besar: {d_mp2}"

    # Pastikan lazy cache H_core & S_ovlp mengembalikan objek array identik
    assert calc._get_h_core() is calc._get_h_core()
    assert calc._get_s_ovlp() is calc._get_s_ovlp()
    print(f"[BACKEND E2E] C++ vs NumPy: dE_SCF={d_scf:.2e} | dE_MP2={d_mp2:.2e}")


def test_pure_python_abelian_symmetry_h2o_and_benzene():
    """Verifikasi deteksi grup Abelian, symmetrize AO, dan pelabelan irrep MO pada H2O (C2v) & C6H6 (D2h)."""
    h2o = M.align_principal_axes(mshqc.molecule("H2O"))
    sym_h2o = M.detect_abelian_symmetry(h2o)
    assert sym_h2o.point_group == "C2v"

    calc = mshqc.MSHQCCalculator(h2o, basis_name="cc-pVDZ", eri_method="df")
    scf_res = calc.run_scf()
    S = calc._get_s_ovlp()
    ops = M.build_ao_symmetry_operators(h2o, calc.basis, sym_h2o)
    S_sym = M.symmetrize_ao_matrix(S, ops)
    assert float(np.max(np.abs(S - S_sym))) < 1.0e-10

    labels = M.label_mo_irreps(h2o, calc.basis, scf_res.C_alpha, S, sym_h2o)
    occ_labels = labels[:5]
    assert "mix" not in occ_labels
    # 5 orbital terisi H2O di C2v terdiri atas: 3 A1, 1 B1, 1 B2 (urutan B1/B2 bergantung konvensi sumbu x/y)
    assert occ_labels.count("A1") == 3
    assert occ_labels.count("B1") + occ_labels.count("B2") == 2

    c6h6 = M.align_principal_axes(mshqc.molecule("C6H6"))
    sym_bz = M.detect_abelian_symmetry(c6h6)
    assert sym_bz.point_group == "D2h"
    print(f"[SYMMETRY   ] H2O={sym_h2o.point_group} Occ={occ_labels} | C6H6={sym_bz.point_group}")


def test_frozen_core_frequencies_and_scan_pes_consistency():
    """Pastikan fix freeze_core pada scan_pes & compute_frequencies_thin_core konsisten (bebas NameError)."""
    h2o = mshqc.molecule("H2O")
    pts = h2o.scan_pes(0, 1, [0.90, 0.96, 1.05], method="omp2", basis="cc-pVDZ", freeze_core=True)
    assert len(pts) == 3
    assert pts[1].energy_ha < pts[0].energy_ha and pts[1].energy_ha < pts[2].energy_ha

    # Uji frekuensi frozen-core pada molekul diatomik kecil (HF) agar cepat & presisi
    hf = mshqc.molecule("HF")
    opt = hf.optimize_geometry(method="omp2", basis="cc-pVDZ", freeze_core=True, g_convergence="gau_tight")
    assert opt.converged
    vib = hf.frequencies(method="omp2", basis="cc-pVDZ", freeze_core=True)
    assert vib.n_imaginary == 0
    assert len(vib.frequencies_cm1) == 1
    assert 3800.0 < float(vib.frequencies_cm1[0]) < 4400.0
    print(f"[FC GEOM/VIB] HF FC-OMP2 R_eq={opt.bond_lengths_angstrom['F1-H2']:.5f} A | freq={vib.frequencies_cm1[0]:.1f} cm^-1")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch tidak terpasang di lingkungan ini")
def test_torch_kernel_backend_parity_and_fallback():
    """Verifikasi TorchKernelBackend (CUDA jika ada, atau CPU FP64) cocok < 1e-11 Ha pada RHF & LT-SOS-OMP2."""
    backend = enable_torch_backend(device="auto", name="torch_test", fallback_to_cpp=True)
    h2o = mshqc.molecule("H2O")
    calc = mshqc.MSHQCCalculator(h2o, basis_name="cc-pVDZ", eri_method="df")

    scf_cpp = calc.run_scf()
    lt_cpp = calc.run_lt_sos_omp2(scf_cpp, grad_thresh=1e-6)

    with use_backend("torch_test"):
        scf_t = calc.run_scf()
        lt_t = calc.run_lt_sos_omp2(scf_t, grad_thresh=1e-6)

    assert abs(scf_cpp.energy_total - scf_t.energy_total) < 1.0e-11
    assert abs(lt_cpp.energy_total - lt_t.energy_total) < 1.0e-10
    print(f"[TORCH BACK ] Device={backend.device} | dE_LT-SOS={abs(lt_cpp.energy_total - lt_t.energy_total):.2e}")


if __name__ == "__main__":
    test_dynamic_backend_numpy_end_to_end_scf_and_mp2()
    test_pure_python_abelian_symmetry_h2o_and_benzene()
    test_frozen_core_frequencies_and_scan_pes_consistency()
    print("[PASS] Stage 12 Polish Tests lulus!")
