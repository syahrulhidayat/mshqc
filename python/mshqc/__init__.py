# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""mshqc v2.0 — Pure Thin-Core C++20 / Python Quantum Chemistry Engine."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


def _get_physical_cores() -> int:
    if sys.platform.startswith("linux"):
        try:
            cores = set()
            phys_id = "0"
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("physical id"):
                        phys_id = line.split(":", 1)[1].strip()
                    elif line.startswith("core id"):
                        core_id = line.split(":", 1)[1].strip()
                        cores.add(f"{phys_id}_{core_id}")
            if cores:
                return len(cores)
        except OSError:
            pass
    return max(1, (os.cpu_count() or 4) // 2)


def _configure_environment() -> None:
    try:
        os.sched_setaffinity(0, set(range(os.cpu_count() or 4)))
    except Exception:
        pass

    n_threads = os.environ.get("OMP_NUM_THREADS", str(_get_physical_cores()))
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    if sys.platform in ("darwin", "win32"):
        os.environ.setdefault("TBLIS_NUM_THREADS", "1")
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        os.environ.setdefault("KMP_BLOCKTIME", "0")
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    else:
        os.environ.setdefault("TBLIS_NUM_THREADS", str(n_threads))

    if sys.platform == "win32":
        os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")


def _preload_mkl() -> None:
    if sys.platform == "win32":
        pkg_dir = Path(__file__).resolve().parent
        win_dirs = [
            pkg_dir,
            pkg_dir.parent / "mshqc.libs",
            Path(sys.prefix) / "Library" / "bin",
            Path(sys.prefix) / "bin",
            Path(sys.prefix) / "Scripts",
            Path(sys.base_prefix) / "Library" / "bin",
            Path(sys.base_prefix) / "bin",
        ]
        valid_dirs = []
        for d in win_dirs:
            if d.is_dir():
                d_str = str(d)
                valid_dirs.append(d_str)
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(d_str)
                    except OSError:
                        pass
        if valid_dirs:
            os.environ["PATH"] = os.pathsep.join(valid_dirs) + os.pathsep + os.environ.get("PATH", "")
        for d_str in valid_dirs:
            for dll in (
                "mkl_core.3.dll", "mkl_core.2.dll", "mkl_core.dll",
                "mkl_sequential.3.dll", "mkl_sequential.2.dll", "mkl_sequential.dll",
                "mkl_rt.3.dll", "mkl_rt.2.dll", "mkl_rt.dll",
            ):
                p = Path(d_str) / dll
                if p.exists():
                    try:
                        ctypes.WinDLL(str(p))
                    except OSError:
                        pass
        return

    lib_dirs = [
        Path(sys.prefix) / "lib",
        Path(sys.prefix) / "lib64",
        Path(sys.base_prefix) / "lib",
        Path(sys.base_prefix) / "lib64",
    ]
    rtld_flags = getattr(ctypes, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_LAZY", 1)
    for d in lib_dirs:
        if not d.exists():
            continue
        for so in ("libmkl_rt.so.3", "libmkl_rt.so.2", "libmkl_rt.so", "libiomp5.so", "libtbb.so.12"):
            p = d / so
            if p.exists():
                try:
                    ctypes.CDLL(str(p), mode=rtld_flags)
                except OSError:
                    pass


def _resolve_mshqc_basis_dir() -> None:
    cur = os.environ.get("MSHQC_BASIS_PATH", "")
    if cur and Path(cur).is_dir() and any(Path(cur).glob("*.gbs")):
        return
    pkg_dir = Path(__file__).resolve().parent
    for cand in (
        pkg_dir / "basis",
        pkg_dir.parent.parent / "data" / "basis",
        pkg_dir.parent.parent.parent / "mshqc-basis" / "data" / "basis",
        Path.home() / "mshqc-basis" / "data" / "basis",
    ):
        if cand.is_dir() and any(cand.glob("*.gbs")):
            os.environ["MSHQC_BASIS_PATH"] = str(cand)
            return


_configure_environment()
_preload_mkl()
_resolve_mshqc_basis_dir()

from ._mshqc import (
    AngularMomentum,
    Atom,
    BasisSet,
    CholeskyDecompositionResult,
    CholeskyERI,
    ERITensor,
    GaussianPrimitive,
    IntegralEngine,
    Molecule,
    Shell,
    angstrom_to_bohr,
    bohr_to_angstrom,
    hartree_to_ev,
    hartree_to_kcal,
    kernels,
    standard_frozen_core,
)
from . import methods
from .calculators import MSHQCCalculator, build_molecule, optimize_geometry_universal, recommend_hpc_settings, run
from .molecules import atom, list_molecules, molecule
from .geom_opt import GeomOptResult, UniversalGeomOptimizer, project_translation_rotation
from .methods.geom_hybrid import (
    ThinCoreGeomResult,
    ThinCorePESPoint,
    ThinCoreVibResult,
    compute_frequencies_thin_core,
    optimize_geometry_thin_core,
    scan_bond_pes_thin_core,
)
from .methods.grad_hybrid import compute_df_gradient
from .session import MSHQCSession
from .utils import benchmark_basis_sets, compare_methods, quick_calculation

__version__ = "2.0.0"
THIN_CORE_ONLY = True

_G_CONV_PRESETS = {
    "gau": (3.0e-4, 4.5e-4, 1.2e-3, 1.8e-3),
    "gau_tight": (1.0e-5, 1.5e-5, 4.0e-5, 6.0e-5),
    "gau_verytight": (1.0e-6, 2.0e-6, 4.0e-6, 6.0e-6),
    "qchem": (3.0e-4, 3.0e-4, 1.2e-3, 1.2e-3),
    "cfour": (1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4),
}


def _mol_optimize_geometry(
    self,
    method: str = "omp2",
    eri_mode: str = "df",
    eri_method: str = "",
    basis: str = "cc-pVDZ",
    aux_basis: str = "",
    reference: str = "auto",
    max_iter: int = 40,
    g_convergence: str = "",
    g_rms_tol: float = 1.0e-6,
    g_max_tol: float = 2.0e-6,
    dx_rms_tol: float = 4.0e-6,
    dx_max_tol: float = 6.0e-6,
    energy_thresh: float = 1.0e-12,
    orb_grad_thresh: float = 1.0e-10,
    e_convergence: float = 0.0,
    d_convergence: float = 0.0,
    r_convergence: float = 0.0,
    rms_mograd_convergence: float = 0.0,
    fd_step_bohr: float = 5.0e-4,
    disp_size: float = 0.0,
    use_analytic_gradient: bool = True,
    verbose: bool = False,
    **_kw,
) -> ThinCoreGeomResult:
    del verbose
    if g_convergence:
        key = g_convergence.strip().lower()
        if key in _G_CONV_PRESETS:
            g_rms_tol, g_max_tol, dx_rms_tol, dx_max_tol = _G_CONV_PRESETS[key]
    if e_convergence > 0.0:
        energy_thresh = e_convergence
    if d_convergence > 0.0:
        energy_thresh = min(energy_thresh, d_convergence)
    if rms_mograd_convergence > 0.0:
        orb_grad_thresh = rms_mograd_convergence
    elif r_convergence > 0.0:
        orb_grad_thresh = min(orb_grad_thresh, r_convergence)
    if disp_size > 0.0:
        fd_step_bohr = disp_size

    return optimize_geometry_thin_core(
        self,
        method=method,
        eri_mode=(eri_method or eri_mode),
        basis=basis,
        aux_basis=(aux_basis or None),
        reference=reference,
        max_iter=max_iter,
        g_rms_tol=g_rms_tol,
        g_max_tol=g_max_tol,
        dx_rms_tol=dx_rms_tol,
        dx_max_tol=dx_max_tol,
        energy_thresh=energy_thresh,
        orb_grad_thresh=orb_grad_thresh,
        fd_step_bohr=fd_step_bohr,
        use_analytic_gradient=use_analytic_gradient,
        freeze_core=bool(_kw.get('freeze_core', False)),  # [INJECTED_V2_MOL_FORWARD_FC]
        n_frozen=int(_kw.get('n_frozen', 0)),
    )


def _mol_frequencies(
    self,
    method: str = "omp2",
    eri_mode: str = "df",
    eri_method: str = "",
    basis: str = "cc-pVDZ",
    reference: str = "auto",
    temperature_k: float = 298.15,
    verbose: bool = False,
    **_kw,
) -> ThinCoreVibResult:
    del verbose
    return compute_frequencies_thin_core(
        self,
        method=method,
        eri_mode=(eri_method or eri_mode),
        basis=basis,
        reference=reference,
        temperature_k=temperature_k,
        freeze_core=bool(_kw.get('freeze_core', False)),
        n_frozen=int(_kw.get('n_frozen', 0)),
    )


def _mol_gradient(
    self,
    method: str = "omp2",
    eri_mode: str = "df",
    eri_method: str = "",
    basis: str = "cc-pVDZ",
    aux_basis: str = "",
    reference: str = "auto",
    kappa: float = 1.45,
    **_kw,
):
    del eri_mode, eri_method
    return compute_df_gradient(
        self,
        method=method,
        basis_name=basis,
        aux_basis_name=(aux_basis or None),
        reference=reference,
        kappa=kappa,
    )


def _mol_scan_pes(
    self,
    atom_i: int,
    atom_j: int,
    distances_angstrom,
    method: str = "omp2",
    eri_mode: str = "df",
    basis: str = "cc-pVDZ",
    **_kw,
):
    return scan_bond_pes_thin_core(
        self,
        int(atom_i),
        int(atom_j),
        list(distances_angstrom),
        method=method,
        eri_mode=eri_mode,
        basis=basis,
        freeze_core=bool(_kw.get('freeze_core', False)),
        n_frozen=int(_kw.get('n_frozen', 0)),
    )


Molecule.optimize_geometry = _mol_optimize_geometry
Molecule.frequencies = _mol_frequencies
Molecule.gradient = _mol_gradient
Molecule.scan_pes = _mol_scan_pes

__all__ = [
    "__version__",
    "THIN_CORE_ONLY",
    "Atom", "Molecule", "GaussianPrimitive", "Shell", "AngularMomentum",
    "BasisSet", "IntegralEngine", "ERITensor",
    "CholeskyERI", "CholeskyDecompositionResult",
    "bohr_to_angstrom", "angstrom_to_bohr", "hartree_to_ev", "hartree_to_kcal",
    "standard_frozen_core",
    "kernels", "methods",
    "MSHQCCalculator", "MSHQCSession", "atom", "molecule", "list_molecules", "build_molecule", "run",
    "optimize_geometry_universal", "recommend_hpc_settings",
    "UniversalGeomOptimizer", "GeomOptResult", "project_translation_rotation",
    "ThinCoreGeomResult", "ThinCoreVibResult", "ThinCorePESPoint",
    "quick_calculation", "benchmark_basis_sets", "compare_methods",
]
