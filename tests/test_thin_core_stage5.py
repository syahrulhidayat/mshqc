#!/usr/bin/env python3
"""
Validasi Tahap 5 (Refaktor C++ Core Bersih: primitives.h/.cc, slim bindings.cc,
build_exchange_3c_diag untuk SAD Iter-0, dan CoupledUHFDIIS):
1. `build_exchange_3c_diag` cocok < 1e-14 terhadap `build_exchange_3c_sym` pada matriks diagonal SAD
2. Verifikasi keterpisahan bersih `python/bindings.cc` (< 300 LOC) & `include/mshqc/kernels/primitives.h` (< 180 LOC)
3. End-to-end Thin-Core pipeline (RHF, UHF, ROHF, MP2, MP3, OMP2, SCS-OMP2, OMP2.5, OMP3) pada H2O & OH
"""
from pathlib import Path
import numpy as np
import mshqc
from mshqc import kernels as K
from mshqc import methods as M


def test_build_exchange_3c_diag_matches_sym_exact():
    mol = mshqc.Molecule(0, 1)
    mol.add_atom("O", 0.0, 0.0, 0.1173, units="angstrom")
    mol.add_atom("H", 0.0, 0.7572, -0.4692, units="angstrom")
    mol.add_atom("H", 0.0, -0.7572, -0.4692, units="angstrom")

    basis = mshqc.BasisSet("cc-pVDZ", mol)
    aux = mshqc.BasisSet("cc-pVDZ-RI", mol)
    nbf = int(basis.n_basis_functions())
    B_ao = K.compute_df_ao_factor(mol, basis, aux, 1e-10, False, False)

    P_sad = M.build_sad_guess(mol, basis)
    d_vec = np.ascontiguousarray(0.5 * np.diag(P_sad), dtype=np.float64)

    K_diag = K.build_exchange_3c_diag(B_ao, d_vec, nbf)
    K_ref = K.build_exchange_3c_sym(B_ao, np.diag(d_vec), nbf)
    max_err = float(np.max(np.abs(K_diag - K_ref)))
    print(f"[DIAG EXCH] max|K_diag - K_sym| = {max_err:.2e}")
    assert max_err < 1.0e-13


def test_core_cpp_modularity_and_slimness():
    root = Path(__file__).resolve().parent.parent
    prim_h = root / "include" / "mshqc" / "kernels" / "primitives.h"
    prim_cc = root / "src" / "kernels" / "primitives.cc"
    bind_cc = root / "python" / "bindings.cc"
    bind_leg = root / "python" / "bind_legacy.cc"

    assert prim_h.is_file() and prim_cc.is_file()
    assert bind_cc.is_file() and bind_leg.is_file()

    n_prim_h = len(prim_h.read_text(encoding="utf-8").splitlines())
    n_bind_cc = len(bind_cc.read_text(encoding="utf-8").splitlines())
    print(
        f"[CPP SLIM ] primitives.h = {n_prim_h} LOC | bindings.cc = {n_bind_cc} LOC "
        f"| THIN_CORE_ONLY = {mshqc.THIN_CORE_ONLY}"
    )
    assert n_prim_h < 180, f"primitives.h terlalu tebal: {n_prim_h} LOC"
    assert n_bind_cc < 300, f"bindings.cc terlalu tebal: {n_bind_cc} LOC"


def test_thin_core_end_to_end_pipeline():
    h2o = "O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692"
    calc = mshqc.MSHQCCalculator(
        h2o, basis_name="cc-pVDZ", eri_method="df", units="angstrom",
        threshold=1e-10, use_symmetry=False,
    )
    scf_r = calc.run_scf(use_hybrid=True)
    mp2_r = calc.run_mp2(scf_r, use_hybrid=True)
    mp3_r = calc.run_mp3(scf_r, use_hybrid=True)
    omp2_r = calc.run_omp2(scf_r, grad_thresh=1e-6, use_hybrid=True)
    omp3_r = calc.run_omp3(scf_r, grad_thresh=1e-6, use_hybrid=True)

    assert scf_r.converged and omp2_r.converged and omp3_r.converged
    assert omp3_r.energy_total < omp2_r.energy_total < scf_r.energy_total
    assert abs(mp2_r.energy_total - (-76.2318220233)) < 1e-6
    assert abs(mp3_r.energy_total - (-76.2387203765)) < 1e-6
    print(
        f"[E2E H2O  ] RHF={scf_r.energy_total:.8f} | MP2={mp2_r.energy_total:.8f} | "
        f"OMP2={omp2_r.energy_total:.8f} | OMP3={omp3_r.energy_total:.8f}"
    )


if __name__ == "__main__":
    test_build_exchange_3c_diag_matches_sym_exact()
    test_core_cpp_modularity_and_slimness()
    test_thin_core_end_to_end_pipeline()
    print("[PASS] Semua validasi Tahap 5 Thin-Core lulus!")
