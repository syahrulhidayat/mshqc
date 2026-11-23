#!/usr/bin/env python3
"""
Validasi Final mshqc v2.0 (Pure Zero-Legacy Thin-Core, Stub Parity & Full Top-Level APIs).
"""
import ast
from pathlib import Path
import numpy as np
import mshqc


def test_pure_v2_zero_legacy_repository_layout():
    root = Path(__file__).resolve().parent.parent
    expected_core = {
        "src/core/molecule.cc",
        "src/core/basis.cc",
        "src/ints/integrals.cc",
        "src/integrals/df_eri.cc",
        "src/integrals/cholesky_eri.cc",
        "src/kernels/primitives.cc",
        "src/kernels/mp3_kernels.cc",
        "src/kernels/uhf_kernels.cc",
    }
    actual_cc = {p.relative_to(root).as_posix() for p in (root / "src").rglob("*.cc")}
    assert actual_cc == expected_core, f"Unexpected C++ files in src/: {actual_cc ^ expected_core}"
    assert not (root / "src" / "legacy").exists()
    assert not (root / "python" / "bind_legacy.cc").exists()
    assert mshqc.THIN_CORE_ONLY is True
    assert mshqc.__version__ == "2.0.0"
    print(f"[PURE V2.0  ] Verified strictly {len(actual_cc)} core C++20 files & zero legacy files.")


def test_pyi_stub_covers_all_kernels_symbols():
    root = Path(__file__).resolve().parent.parent
    pyi_path = root / "python" / "mshqc" / "_mshqc.pyi"
    tree = ast.parse(pyi_path.read_text(encoding="utf-8"), filename=str(pyi_path))

    kernels_cls = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "kernels"),
        None,
    )
    assert kernels_cls is not None, "Class kernels tidak ditemukan di _mshqc.pyi"

    stub_syms = {
        n.name for n in kernels_cls.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    runtime_syms = {name for name in dir(mshqc.kernels) if not name.startswith("_")}
    missing = runtime_syms - stub_syms
    print(f"[STUB CHECK ] Runtime kernels={len(runtime_syms)} | Stubbed={len(stub_syms)} | Missing={missing}")
    assert not missing, f"Simbol mshqc.kernels belum ada di _mshqc.pyi: {sorted(missing)}"


def test_top_level_apis_and_molecule_methods():
    h2o = "O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692"

    sess = mshqc.MSHQCSession(h2o, basis_name="cc-pVDZ", units="angstrom", threshold=1e-10)
    res_map = sess.run_complete_analysis(
        methods=["rhf", "mp2", "omp2", "lt-sos-omp2", "kappa-omp2"],
        grad_thresh=1e-6,
    )
    for key in ("RHF", "RMP2", "OMP2", "LT-SOS-OMP2", "KAPPA-OMP2"):
        assert key in res_map and np.isfinite(float(getattr(res_map[key], "energy_total", getattr(res_map[key], "e_total"))))

    r_qk = mshqc.quick_calculation(h2o, basis="cc-pVDZ", method="df-kappa-omp2", units="angstrom")
    r_lt = mshqc.quick_calculation(h2o, basis="cc-pVDZ", method="df-lt-sos-omp2", units="angstrom")
    assert abs(r_qk.energy_total - res_map["KAPPA-OMP2"].energy_total) < 1e-7
    assert abs(r_lt.energy_total - res_map["LT-SOS-OMP2"].energy_total) < 1e-7

    cmp_res = mshqc.compare_methods(
        h2o, basis="cc-pVDZ", methods=["rhf", "omp2", "kappa-omp2"], units="angstrom"
    )
    assert set(cmp_res.keys()) >= {"RHF", "OMP2", "KAPPA-OMP2"}

    h2 = mshqc.Molecule(0, 1)
    h2.add_atom("H", 0.0, 0.0, 0.0, units="angstrom")
    h2.add_atom("H", 0.0, 0.0, 0.78, units="angstrom")
    opt = h2.optimize_geometry(method="omp2", basis="cc-pVDZ", g_convergence="gau_tight")
    assert opt.converged
    pts = h2.scan_pes(0, 1, [0.65, 0.74, 0.90], method="omp2", basis="cc-pVDZ")
    assert len(pts) == 3 and pts[1].energy_ha < pts[0].energy_ha and pts[1].energy_ha < pts[2].energy_ha
    print(
        f"[V2.0 APIs  ] Session/Quick/Compare OK | H2 R_eq={opt.bond_lengths_angstrom['H1-H2']:.5f} A"
    )


if __name__ == "__main__":
    test_pure_v2_zero_legacy_repository_layout()
    test_pyi_stub_covers_all_kernels_symbols()
    test_top_level_apis_and_molecule_methods()
    print("[PASS] Semua validasi Final mshqc v2.0 Pure Thin-Core lulus!")
