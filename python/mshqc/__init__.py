# Copyright 2026 Muhamad Syahrul Hidayat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import ctypes
from pathlib import Path

def _get_physical_cores() -> int:
    """Deteksi jumlah physical core secara dinamis tanpa batas hardcode."""
    # 1. Linux: baca pasangan unik (physical id, core id) dari /proc/cpuinfo
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

    # 2. Fallback universal: separuh dari logical CPU (asumsi SMT 2-way)
    logical = os.cpu_count() or 4
    return max(1, logical // 2)

def _configure_environment():
    # Buka CPU affinity ke seluruh core yang tersedia pada sistem
    try:
        os.sched_setaffinity(0, set(range(os.cpu_count() or 4)))
    except Exception:
        pass

    n_phys_cores = _get_physical_cores()
    n_threads = os.environ.get("OMP_NUM_THREADS", str(n_phys_cores))

    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("TBLIS_NUM_THREADS", str(n_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    if "MSHQC_BASIS_PATH" not in os.environ:
        pkg_basis = os.path.join(os.path.dirname(__file__), "basis")
        repo_basis = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "basis"))
        if os.path.isdir(pkg_basis):
            os.environ["MSHQC_BASIS_PATH"] = pkg_basis
        elif os.path.isdir(repo_basis):
            os.environ["MSHQC_BASIS_PATH"] = repo_basis
def _preload_mkl():
    """
    Preload MKL runtime libraries dari environment (sys.prefix/lib) 
    agar _mshqc.so selalu menemukan simbol MKL tanpa perlu set LD_LIBRARY_PATH.
    """
    _lib_dirs = [
        Path(sys.prefix) / "lib",
        Path(sys.prefix) / "lib64",
        Path(sys.base_prefix) / "lib",
        Path(sys.base_prefix) / "lib64",
    ]
    
    # Memuat library utama MKL dan OpenMP yang di-exclude oleh auditwheel
    _libs_to_load = [
        "libmkl_rt.so.3", "libmkl_rt.so.2", "libmkl_rt.so",
        "libiomp5.so", "libtbb.so.12"
    ]

    # RTLD_GLOBAL sangat penting agar _mshqc.so bisa ikut memakai library yang di-load ini
    rtld_flags = getattr(ctypes, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_LAZY", 1)
    
    for _d in _lib_dirs:
        if not _d.exists():
            continue
        for _lib_so in _libs_to_load:
            _lib_path = _d / _lib_so
            if _lib_path.exists():
                try:
                    ctypes.CDLL(str(_lib_path), mode=rtld_flags)
                except OSError:
                    pass

_configure_environment()
_preload_mkl()

from ._mshqc import *
from .calculators import MSHQCCalculator, build_molecule, run
from .session import MSHQCSession
from .utils import quick_calculation, benchmark_basis_sets, compare_methods

__version__ = "1.0.4.post16"

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

# --- Auto-resolve MSHQC_BASIS_PATH (mendukung repo terpisah mshqc-basis) ---
def _resolve_mshqc_basis_dir() -> None:
    import os as _os
    from pathlib import Path as _Path
    _pkg_dir = _Path(__file__).resolve().parent
    _candidates = [
        _pkg_dir / "basis",
        _pkg_dir.parent.parent / "data" / "basis",
        _pkg_dir.parent.parent.parent / "mshqc-basis" / "data" / "basis",
        _Path.home() / "mshqc-basis" / "data" / "basis",
    ]
    _cur = _os.environ.get("MSHQC_BASIS_PATH", "")
    if _cur and _Path(_cur).is_dir() and any(_Path(_cur).glob("*.gbs")):
        return
    for _cand in _candidates:
        if _cand.is_dir() and any(_cand.glob("*.gbs")):
            _os.environ["MSHQC_BASIS_PATH"] = str(_cand)
            break

_resolve_mshqc_basis_dir()

# === MSHQC NATIVE C++20 GEOMETRY OPTIMIZATION EXPORTS ===
for _sym in ("GeomOptConfig", "GeomOptResult", "VibThermoResult", "PESScanPoint"):
    if _sym in globals() and _sym not in __all__:
        __all__.append(_sym)
