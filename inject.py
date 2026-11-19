#!/usr/bin/env python3
"""
Idempotent Python injector untuk menyinkronkan:
1. HPCGeometryOptimizer & parameter analytic gradient di src/python/bindings.cc
2. Ekspor HPCGeometryOptimizer, ROMP2, ROMP3 & forwarding parameter di python/mshqc/__init__.py
3. Re-export HPCGeometryOptimizer di python/mshqc/geom_opt.py
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent


def patch_bindings_cc(bindings_path: Path) -> None:
    if not bindings_path.exists():
        print(f"[SKIP] {bindings_path} tidak ditemukan.")
        return

    src = bindings_path.read_text(encoding="utf-8")
    updated = src

    # 1. Pastikan HPCGeometryOptimizer ter-bind di bagian bawah NB_MODULE
    hpc_opt_block = """    nb::class_<mshqc::gradient::HPCGeometryOptimizer>(m, "HPCGeometryOptimizer")
        .def_static("optimize_molecule",
                    &mshqc::gradient::HPCGeometryOptimizer::optimize_molecule,
                    nb::arg("mol"), nb::arg("config") = mshqc::gradient::GeomOptConfig(),
                    nb::call_guard<nb::gil_scoped_release>())
        .def_static("compute_nuclear_gradient_cpp",
                    &mshqc::gradient::HPCGeometryOptimizer::compute_nuclear_gradient_cpp,
                    nb::arg("mol"), nb::arg("config") = mshqc::gradient::GeomOptConfig(),
                    nb::call_guard<nb::gil_scoped_release>())
        .def_static("compute_frequencies",
                    &mshqc::gradient::HPCGeometryOptimizer::compute_frequencies,
                    nb::arg("mol"),
                    nb::arg("config") = mshqc::gradient::GeomOptConfig(),
                    nb::arg("temperature_k") = 298.15,
                    nb::arg("pressure_atm") = 1.0,
                    nb::call_guard<nb::gil_scoped_release>())
        .def_static("scan_bond_pes",
                    &mshqc::gradient::HPCGeometryOptimizer::scan_bond_pes,
                    nb::arg("mol"), nb::arg("atom_i"), nb::arg("atom_j"),
                    nb::arg("distances_angstrom"),
                    nb::arg("config") = mshqc::gradient::GeomOptConfig(),
                    nb::call_guard<nb::gil_scoped_release>());
"""
    if 'nb::class_<mshqc::gradient::HPCGeometryOptimizer>' not in updated:
        last_brace = updated.rfind("}")
        if last_brace != -1:
            updated = updated[:last_brace] + "\n" + hpc_opt_block + "}\n"

    # 2. Tambahkan use_analytic_gradient ke Molecule.gradient jika belum ada
    old_grad_sig = """        .def("gradient", [](const Molecule& mol,
                            const std::string& method,
                            const std::string& eri_mode,
                            const std::string& basis,
                            const std::string& reference,
                            double fd_step_bohr) {
            mshqc::gradient::GeomOptConfig cfg;
            cfg.method = method;
            cfg.eri_method = eri_mode;
            cfg.basis_name = basis;
            cfg.reference = reference;
            cfg.fd_step_bohr = fd_step_bohr;
            cfg.verbose = false;
            nb::gil_scoped_release release;
            return mshqc::gradient::HPCGeometryOptimizer::compute_nuclear_gradient_cpp(mol, cfg);
        }, nb::arg("method") = "omp2",
           nb::arg("eri_mode") = "df",
           nb::arg("basis") = "cc-pVDZ",
           nb::arg("reference") = "auto",
           nb::arg("fd_step_bohr") = 5.0e-4,
           "Compute nuclear gradient vector dE/dR (Ha/Bohr)")"""

    new_grad_sig = """        .def("gradient", [](const Molecule& mol,
                            const std::string& method,
                            const std::string& eri_mode,
                            const std::string& eri_method,
                            const std::string& basis,
                            const std::string& aux_basis,
                            const std::string& reference,
                            double fd_step_bohr,
                            bool use_analytic_gradient) {
            mshqc::gradient::GeomOptConfig cfg;
            cfg.method = method;
            cfg.eri_method = !eri_method.empty() ? eri_method : eri_mode;
            cfg.basis_name = basis;
            cfg.aux_basis_name = aux_basis;
            cfg.reference = reference;
            cfg.fd_step_bohr = fd_step_bohr;
            cfg.use_analytic_gradient = use_analytic_gradient;
            cfg.verbose = false;
            nb::gil_scoped_release release;
            return mshqc::gradient::HPCGeometryOptimizer::compute_nuclear_gradient_cpp(mol, cfg);
        }, nb::arg("method") = "omp2",
           nb::arg("eri_mode") = "df",
           nb::arg("eri_method") = "",
           nb::arg("basis") = "cc-pVDZ",
           nb::arg("aux_basis") = "",
           nb::arg("reference") = "auto",
           nb::arg("fd_step_bohr") = 5.0e-4,
           nb::arg("use_analytic_gradient") = true,
           "Compute nuclear gradient vector dE/dR (Ha/Bohr)")"""

    if old_grad_sig in updated:
        updated = updated.replace(old_grad_sig, new_grad_sig)

    # 3. Tambahkan use_analytic_gradient ke Molecule.optimize_geometry jika belum ada
    old_opt_tail = """            cfg.fd_step_bohr = fd_step_bohr;
            cfg.verbose = verbose;
            nb::gil_scoped_release release;
            return mshqc::gradient::HPCGeometryOptimizer::optimize_molecule(mol, cfg);
        }, nb::arg("method") = "omp2","""

    if "use_analytic_gradient" not in updated.split('def("optimize_geometry"')[1].split('def("frequencies"')[0]:
        updated = updated.replace(
            "double fd_step_bohr,\n                                     bool verbose) {",
            "double fd_step_bohr,\n                                     bool use_analytic_gradient,\n                                     bool verbose) {",
        )
        updated = updated.replace(
            "cfg.fd_step_bohr = fd_step_bohr;\n            cfg.verbose = verbose;",
            "cfg.fd_step_bohr = fd_step_bohr;\n            cfg.use_analytic_gradient = use_analytic_gradient;\n            cfg.verbose = verbose;",
        )
        updated = updated.replace(
            'nb::arg("fd_step_bohr") = 5.0e-4,\n           nb::arg("verbose") = true,',
            'nb::arg("fd_step_bohr") = 5.0e-4,\n           nb::arg("use_analytic_gradient") = true,\n           nb::arg("verbose") = true,',
        )

    if updated != src:
        bindings_path.write_text(updated, encoding="utf-8")
        print(f"[OK] {bindings_path} berhasil diperbarui.")
    else:
        print(f"[OK] {bindings_path} sudah sinkron.")


def patch_init_py(init_path: Path) -> None:
    if not init_path.exists():
        print(f"[SKIP] {init_path} tidak ditemukan.")
        return

    src = init_path.read_text(encoding="utf-8")
    updated = src

    # 1. Tambahkan HPCGeometryOptimizer ke tuple ekspor __all__
    updated = updated.replace(
        'for _sym in ("GeomOptConfig", "GeomOptResult", "VibThermoResult", "PESScanPoint"):',
        'for _sym in ("GeomOptConfig", "GeomOptResult", "VibThermoResult", "PESScanPoint", "HPCGeometryOptimizer"):',
    )

    # 2. Tambahkan ROMP2 dan ROMP3 ke __all__
    updated = updated.replace(
        '"MP2Config", "MP2Result", "RMP2", "UMP2", "OMP2",',
        '"MP2Config", "MP2Result", "RMP2", "UMP2", "ROMP2", "OMP2",',
    )
    updated = updated.replace(
        '"MP3Result", "RMP3", "UMP3", "OMP3",',
        '"MP3Result", "RMP3", "UMP3", "ROMP3", "OMP3",',
    )

    # 3. Teruskan use_analytic_gradient pada _psi4_compatible_optimize_geometry
    if "use_analytic_gradient: bool = True" not in updated:
        updated = updated.replace(
            "    project_trans_rot: bool = True,\n    verbose: bool = False,",
            "    project_trans_rot: bool = True,\n    use_analytic_gradient: bool = True,\n    verbose: bool = False,",
        )
        updated = updated.replace(
            "        fd_step_bohr=fd_step_bohr,\n        verbose=verbose,\n    )",
            "        fd_step_bohr=fd_step_bohr,\n        use_analytic_gradient=use_analytic_gradient,\n        verbose=verbose,\n    )",
        )

    if updated != src:
        init_path.write_text(updated, encoding="utf-8")
        print(f"[OK] {init_path} berhasil diperbarui.")
    else:
        print(f"[OK] {init_path} sudah sinkron.")


def patch_geom_opt_py(geom_opt_path: Path) -> None:
    if not geom_opt_path.exists():
        print(f"[SKIP] {geom_opt_path} tidak ditemukan.")
        return

    src = geom_opt_path.read_text(encoding="utf-8")
    if "HPCGeometryOptimizer" in src:
        print(f"[OK] {geom_opt_path} sudah mengekspor HPCGeometryOptimizer.")
        return

    inject_snippet = (
        "\ntry:\n"
        "    from ._mshqc import HPCGeometryOptimizer, GeomOptConfig, VibThermoResult, PESScanPoint\n"
        "except ImportError:\n"
        "    HPCGeometryOptimizer = None  # type: ignore\n"
    )
    geom_opt_path.write_text(src.rstrip() + "\n" + inject_snippet, encoding="utf-8")
    print(f"[OK] {geom_opt_path} berhasil disuntikkan re-export HPCGeometryOptimizer.")


if __name__ == "__main__":
    candidates_bindings = [
        ROOT / "src" / "python" / "bindings.cc",
        ROOT / "python" / "bindings.cc",
        ROOT / "bindings.cc",
    ]
    for p in candidates_bindings:
        if p.exists():
            patch_bindings_cc(p)
            break

    patch_init_py(ROOT / "python" / "mshqc" / "__init__.py")
    patch_geom_opt_py(ROOT / "python" / "mshqc" / "geom_opt.py")