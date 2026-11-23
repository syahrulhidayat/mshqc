#!/usr/bin/env python3
"""
Validasi Tahap 9 (Full Top-Level API Parity & Clean `src/legacy/` Separation):
1. Verifikasi struktur repo bersih: 8 file `.cc` inti di `src/{core,ints,integrals,kernels}`
   dan seluruh solver legacy terisolasi rapi di `src/legacy/`.
2. Seluruh API tingkat atas (`mshqc.run`, `MSHQCSession`, `quick_calculation`, `compare_methods`)
   berjalan mulus di build `THIN_CORE_ONLY` maupun `FULL` termasuk `kappa-omp2` & `lt-sos-omp2`.
3. `Molecule.gradient`, `Molecule.optimize_geometry`, `Molecule.frequencies`, dan `Molecule.scan_pes`
   tersedia dan konsisten di kedua profil build.
"""
from pathlib import Path
import numpy as np
import mshqc


def test_clean_src_vs_legacy_repository_layout():
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
    actual_non_legacy = {
        p.relative_to(root).as_posix()
        for p in (root / "src").rglob("*.cc")
        if "src/legacy/" not in p.relative_to(root).as_posix()
    }
    legacy_files = list((root / "src" / "legacy").rglob("*.cc"))
    print(
        f"[REPO LAYOUT] Core C++ files: {len(actual_non_legacy)} | "
        f"Archived in src/legacy/: {len(legacy_files)}"
    )
    assert actual_non_legacy == expected_core, (
        f"Unexpected non-legacy C++ files: {actual_non_legacy ^ expected_core}"
    )
    assert len(legacy_files) >= 15


def test_top_level_apis_work_in_thin_core_and_full():
    h2o = "O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692"

    # 1. MSHQCSession dengan seluruh keluarga metode baru (termasuk kappa-omp2 & lt-sos-omp2)
    sess = mshqc.MSHQCSession(
        h2o, basis_name="cc-pVDZ", eri_method="df", units="angstrom",
        threshold=1e-10, use_symmetry=False,
    )
    res_map = sess.run_complete_analysis(
        methods=["rhf", "mp2", "omp2", "lt-sos-omp2", "kappa-omp2"],
        grad_thresh=1e-6,
    )
    for key in ("RHF", "RMP2", "OMP2", "LT-SOS-OMP2", "KAPPA-OMP2"):
        assert key in res_map and np.isfinite(float(getattr(res_map[key], "energy_total", getattr(res_map[key], "e_total"))))

    # 2. quick_calculation & compare_methods
    r_qk = mshqc.quick_calculation(h2o, basis="cc-pVDZ", method="df-kappa-omp2", units="angstrom")
    r_lt = mshqc.quick_calculation(h2o, basis="cc-pVDZ", method="df-lt-sos-omp2", units="angstrom")
    assert abs(r_qk.energy_total - res_map["KAPPA-OMP2"].energy_total) < 1e-7
    assert abs(r_lt.energy_total - res_map["LT-SOS-OMP2"].energy_total) < 1e-7

    cmp_res = mshqc.compare_methods(
        h2o, basis="cc-pVDZ", methods=["rhf", "omp2", "kappa-omp2"],
        eri_method="df", units="angstrom",
    )
    assert set(cmp_res.keys()) >= {"RHF", "OMP2", "KAPPA-OMP2"}

    # 3. Default eri_method saat THIN_CORE_ONLY tidak melempar AttributeError
    if mshqc.THIN_CORE_ONLY:
        calc_def = mshqc.MSHQCCalculator(h2o, basis_name="cc-pVDZ", units="angstrom")
        assert calc_def.eri_method == "df"
        assert calc_def.run_scf().converged
    print(
        f"[TOP APIs   ] Session/Quick/Compare OK | kappa-OOMP2={r_qk.energy_total:.8f} | "
        f"LT-SOS={r_lt.energy_total:.8f}"
    )


def test_molecule_scan_pes_and_methods_parity():
    h2 = mshqc.Molecule(0, 1)
    h2.add_atom("H", 0.0, 0.0, 0.0, units="angstrom")
    h2.add_atom("H", 0.0, 0.0, 0.74, units="angstrom")

    assert hasattr(h2, "gradient") and hasattr(h2, "optimize_geometry")
    assert hasattr(h2, "frequencies") and hasattr(h2, "scan_pes")

    pts = h2.scan_pes(0, 1, [0.65, 0.74, 0.90], method="omp2", eri_mode="df", basis="cc-pVDZ")
    assert len(pts) == 3
    assert pts[1].energy_ha < pts[0].energy_ha and pts[1].energy_ha < pts[2].energy_ha
    print(
        f"[PES H2 OMP2] r=0.65:{pts[0].energy_ha:.6f} | r=0.74:{pts[1].energy_ha:.6f} | "
        f"r=0.90:{pts[2].energy_ha:.6f} Ha"
    )


if __name__ == "__main__":
    test_clean_src_vs_legacy_repository_layout()
    test_top_level_apis_work_in_thin_core_and_full()
    test_molecule_scan_pes_and_methods_parity()
    print("[PASS] Semua validasi Tahap 9 (Clean Repo & Full Thin-Core API Parity) lulus!")
