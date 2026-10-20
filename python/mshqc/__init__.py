# File: python/mshqc/__init__.py
import os
import sys
import glob
import ctypes
import multiprocessing

def _configure_environment():
    try:
        n_cores = multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        n_cores = 4

    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(n_cores)
    if "OPENBLAS_NUM_THREADS" not in os.environ:
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
    if "MKL_NUM_THREADS" not in os.environ:
        os.environ["MKL_NUM_THREADS"] = "1"
    if "VECLIB_MAXIMUM_THREADS" not in os.environ:
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    if "NUMEXPR_NUM_THREADS" not in os.environ:
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    if "MSHQC_BASIS_PATH" not in os.environ:
        pkg_basis = os.path.join(os.path.dirname(__file__), "basis")
        repo_basis = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "basis"))
        if os.path.isdir(pkg_basis):
            os.environ["MSHQC_BASIS_PATH"] = pkg_basis
        elif os.path.isdir(repo_basis):
            os.environ["MSHQC_BASIS_PATH"] = repo_basis

def _preload_mkl():
    """Preload MKL runtime libraries dari sys.prefix/lib agar _mshqc.so selalu menemukan simbol MKL."""
    search_dirs = [
        os.path.join(sys.prefix, "lib"),
        os.path.join(sys.base_prefix, "lib"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(sys.executable))), "lib"),
    ]
    lib_patterns = [
        "libtbb.so*",
        "libiomp5.so*",
        "libmkl_core.so*",
        "libmkl_gnu_thread.so*",
        "libmkl_intel_thread.so*",
        "libmkl_intel_lp64.so*",
        "libmkl_rt.so*",
    ]
    rtld_flags = getattr(ctypes, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_LAZY", 1)
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for pat in lib_patterns:
            for lib_path in sorted(glob.glob(os.path.join(d, pat))):
                try:
                    ctypes.CDLL(lib_path, mode=rtld_flags)
                except OSError:
                    pass

_configure_environment()
_preload_mkl()

from ._mshqc import *
from .calculators import MSHQCCalculator, build_molecule, run
from .session import MSHQCSession
from .utils import quick_calculation, benchmark_basis_sets, compare_methods

__version__ = "1.0.4.post3"

__all__ = [
    "Atom", "Molecule", "GaussianPrimitive", "Shell", "AngularMomentum",
    "BasisSet", "IntegralEngine", "ERITensor", "Screening", "ShellPair",
    "CholeskyERI", "CholeskyDecompositionResult",
    "PointGroup", "SymmetryOperation", "SymOpType", "PetiteList", "UniqueShellPair", "BasisSymmetrizer",
    "DIIS", "SCFConfig", "SCFResult", "RHF", "UHF", "ROHF",
    "MP2Config", "MP2Result", "RMP2", "UMP2", "OMP2",
    "MP3Result", "RMP3", "UMP3", "OMP3",
    "bohr_to_angstrom", "angstrom_to_bohr", "hartree_to_ev", "hartree_to_kcal",
    "MSHQCCalculator", "MSHQCSession", "build_molecule", "run",
    "quick_calculation", "benchmark_basis_sets", "compare_methods",
]