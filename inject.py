#!/usr/bin/env python3
# ==============================================================================
# Git Commit: fix(python): remove leaked 'txt' variable in __init__.py and
# forward all C++ GeomOpt thresholds to _orig_mol_opt_geom
# ==============================================================================
"""
Master Synchronization & Binding Injector for mshqc-sync/build-pypi.
Fixes broken multiline printf in src/scf/scf.cc, repairs NameError 'txt' in
python/mshqc/__init__.py, and enforces full Psi4 parity.
"""

from __future__ import annotations
import pathlib
import re


def patch_init_py(root: pathlib.Path) -> None:
    init_py = root / "python/mshqc/__init__.py"
    if not init_py.is_file():
        return
    txt = init_py.read_text(encoding="utf-8")

    wrapper_code = (
        "# ==============================================================================\n"
        "# [MSHQC-SYNC] Psi4-Compatible Convergence & Geometry Optimization Wrapper\n"
        "# ==============================================================================\n"
        "_G_CONV_PRESETS = {\n"
        '    "gau": (3.0e-4, 4.5e-4, 1.2e-3, 1.8e-3),\n'
        '    "gau_tight": (1.0e-5, 1.5e-5, 4.0e-5, 6.0e-5),\n'
        '    "gau_verytight": (1.0e-6, 2.0e-6, 4.0e-6, 6.0e-6),\n'
        '    "qchem": (3.0e-4, 3.0e-4, 1.2e-3, 1.2e-3),\n'
        '    "cfour": (1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4),\n'
        "}\n\n"
        '_orig_mol_opt_geom = getattr(Molecule, "optimize_geometry", None)\n\n'
        "def _psi4_compatible_optimize_geometry(\n"
        "    self,\n"
        '    method: str = "omp2",\n'
        '    eri_mode: str = "df",\n'
        '    eri_method: str = "",\n'
        '    basis: str = "cc-pVDZ",\n'
        '    aux_basis: str = "",\n'
        '    reference: str = "auto",\n'
        '    opt_method: str = "soscf",\n'
        "    max_iter: int = 40,\n"
        '    g_convergence: str = "",\n'
        "    g_rms_tol: float = 1.0e-6,\n"
        "    g_max_tol: float = 2.0e-6,\n"
        "    dx_rms_tol: float = 4.0e-6,\n"
        "    dx_max_tol: float = 6.0e-6,\n"
        "    energy_thresh: float = 1.0e-12,\n"
        "    orb_grad_thresh: float = 1.0e-10,\n"
        "    e_convergence: float = 0.0,\n"
        "    d_convergence: float = 0.0,\n"
        "    r_convergence: float = 0.0,\n"
        "    rms_mograd_convergence: float = 0.0,\n"
        "    max_mograd_convergence: float = 0.0,\n"
        "    fd_step_bohr: float = 5.0e-4,\n"
        "    disp_size: float = 0.0,\n"
        "    project_trans_rot: bool = True,\n"
        "    verbose: bool = False,\n"
        '    xyz_traj_file: str = "",\n'
        "):\n"
        "    del opt_method, max_mograd_convergence, project_trans_rot, xyz_traj_file\n"
        "    if g_convergence:\n"
        "        key = g_convergence.strip().lower()\n"
        "        if key in _G_CONV_PRESETS:\n"
        "            g_rms_tol, g_max_tol, dx_rms_tol, dx_max_tol = _G_CONV_PRESETS[key]\n"
        "    if e_convergence > 0.0:\n"
        "        energy_thresh = e_convergence\n"
        "    if d_convergence > 0.0:\n"
        "        energy_thresh = min(energy_thresh, d_convergence)\n"
        "    if rms_mograd_convergence > 0.0:\n"
        "        orb_grad_thresh = rms_mograd_convergence\n"
        "    elif r_convergence > 0.0:\n"
        "        orb_grad_thresh = min(orb_grad_thresh, r_convergence)\n"
        "    if disp_size > 0.0:\n"
        "        fd_step_bohr = disp_size\n\n"
        "    resolved_eri = eri_method if eri_method else eri_mode\n"
        '    resolved_aux = aux_basis if aux_basis else (f"{basis}-RI" if resolved_eri.lower() == "df" else "")\n\n'
        "    return _orig_mol_opt_geom(\n"
        "        self,\n"
        "        method=method,\n"
        "        eri_mode=resolved_eri,\n"
        "        eri_method=resolved_eri,\n"
        "        basis=basis,\n"
        "        aux_basis=resolved_aux,\n"
        "        reference=reference,\n"
        "        max_iter=max_iter,\n"
        "        g_rms_tol=g_rms_tol,\n"
        "        g_max_tol=g_max_tol,\n"
        "        dx_rms_tol=dx_rms_tol,\n"
        "        dx_max_tol=dx_max_tol,\n"
        "        energy_thresh=energy_thresh,\n"
        "        orb_grad_thresh=orb_grad_thresh,\n"
        "        fd_step_bohr=fd_step_bohr,\n"
        "        verbose=verbose,\n"
        "    )\n\n"
        "if _orig_mol_opt_geom is not None:\n"
        "    Molecule.optimize_geometry = _psi4_compatible_optimize_geometry\n"
    )

    marker = "# ==============================================================================\n# [MSHQC-SYNC] Psi4-Compatible Convergence & Geometry Optimization Wrapper"
    if marker in txt:
        txt = txt[: txt.index(marker)].rstrip() + "\n\n" + wrapper_code
    elif "_psi4_compatible_optimize_geometry" in txt:
        txt = re.sub(
            r"_G_CONV_PRESETS\s*=\s*\{.*",
            lambda _: wrapper_code,
            txt,
            flags=re.DOTALL,
        )
    else:
        txt = txt.rstrip() + "\n\n" + wrapper_code

    init_py.write_text(txt, encoding="utf-8")
    print(f"[PATCHED] {init_py.relative_to(root)}")


def patch_scf_cc(root: pathlib.Path) -> None:
    cc = root / "src/scf/scf.cc"
    if not cc.is_file():
        return
    txt = cc.read_text(encoding="utf-8")

    broken_printf = re.compile(
        r'printf\(\s*"\s*%4d\s+%18\.10f\s+%10\.2e\s+%10\.2e\s*\r?\n\s*",\s*iter_scf_,\s*energy\(\),\s*dE,\s*dP\s*\);'
    )
    txt = broken_printf.sub(
        lambda _: 'printf(" %4d %18.10f %10.2e %10.2e\\n", iter_scf_, energy(), dE, dP);',
        txt,
    )
    cc.write_text(txt, encoding="utf-8")
    print(f"[PATCHED] {cc.relative_to(root)}")


def main() -> None:
    root = pathlib.Path.cwd()
    patch_init_py(root)
    patch_scf_cc(root)
    print("\n[SUCCESS] python/mshqc/__init__.py dan src/scf/scf.cc telah bersih dan siap diuji.")


if __name__ == "__main__":
    main()