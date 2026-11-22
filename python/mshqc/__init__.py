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
    # Di macOS & Windows, matikan dual-thread-pool TBLIS agar pthreads TCI tidak
    # spin-lock melawan thread pool OpenMP (penyebab macOS 5 menit vs Linux 4 detik)
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
    Preload MKL runtime libraries dari environment (Linux .so & Windows .dll)
    agar _mshqc selalu menemukan simbol MKL tanpa perlu set LD_LIBRARY_PATH / PATH manual.
    """
    if sys.platform == "win32":
        _pkg_dir = Path(__file__).resolve().parent
        _win_dirs = [
            _pkg_dir,
            _pkg_dir.parent / "mshqc.libs",
            Path(sys.prefix) / "Library" / "bin",
            Path(sys.prefix) / "bin",
            Path(sys.prefix) / "Scripts",
            Path(sys.base_prefix) / "Library" / "bin",
            Path(sys.base_prefix) / "bin",
        ]
        _valid_dirs = []
        for _d in _win_dirs:
            if _d.is_dir():
                _d_str = str(_d)
                _valid_dirs.append(_d_str)
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(_d_str)
                    except OSError:
                        pass

        # LoadLibraryA di dalam mkl_rt.3.dll wajib membaca dari PATH di Windows
        if _valid_dirs:
            os.environ["PATH"] = os.pathsep.join(_valid_dirs) + os.pathsep + os.environ.get("PATH", "")

        _win_dlls = [
            "mkl_core.3.dll", "mkl_core.2.dll", "mkl_core.dll",
            "mkl_sequential.3.dll", "mkl_sequential.2.dll", "mkl_sequential.dll",
            "mkl_rt.3.dll", "mkl_rt.2.dll", "mkl_rt.dll",
        ]
        for _d_str in _valid_dirs:
            for _dll in _win_dlls:
                _dll_path = Path(_d_str) / _dll
                if _dll_path.exists():
                    try:
                        ctypes.WinDLL(str(_dll_path))
                    except OSError:
                        pass
        return

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

__version__ = "1.0.4.post22"

__all__ = [
    "Atom", "Molecule", "GaussianPrimitive", "Shell", "AngularMomentum",
    "BasisSet", "IntegralEngine", "ERITensor", "Screening", "ShellPair",
    "CholeskyERI", "CholeskyDecompositionResult",
    "PointGroup", "SymmetryOperation", "SymOpType", "PetiteList", "UniqueShellPair", "BasisSymmetrizer",
    "DIIS", "SCFConfig", "SCFResult", "RHF", "UHF", "ROHF",
    "MP2Config", "MP2Result", "RMP2", "UMP2", "ROMP2", "OMP2",
    "MP3Result", "RMP3", "UMP3", "ROMP3", "OMP3",
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
for _sym in ("GeomOptConfig", "GeomOptResult", "VibThermoResult", "PESScanPoint", "HPCGeometryOptimizer"):
    if _sym in globals() and _sym not in __all__:
        __all__.append(_sym)
from .geom_opt import UniversalGeomOptimizer, GeomOptResult, project_translation_rotation
from .calculators import optimize_geometry_universal

# ==============================================================================
# [MSHQC-SYNC] Psi4-Compatible Convergence & Geometry Optimization Wrapper
# ==============================================================================
_G_CONV_PRESETS = {
    "gau": (3.0e-4, 4.5e-4, 1.2e-3, 1.8e-3),
    "gau_tight": (1.0e-5, 1.5e-5, 4.0e-5, 6.0e-5),
    "gau_verytight": (1.0e-6, 2.0e-6, 4.0e-6, 6.0e-6),
    "qchem": (3.0e-4, 3.0e-4, 1.2e-3, 1.2e-3),
    "cfour": (1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4),
}

_orig_mol_opt_geom = getattr(Molecule, "optimize_geometry", None)

def _psi4_compatible_optimize_geometry(
    self,
    method: str = "omp2",
    eri_mode: str = "df",
    eri_method: str = "",
    basis: str = "cc-pVDZ",
    aux_basis: str = "",
    reference: str = "auto",
    opt_method: str = "soscf",
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
    max_mograd_convergence: float = 0.0,
    fd_step_bohr: float = 5.0e-4,
    disp_size: float = 0.0,
    project_trans_rot: bool = True,
    use_analytic_gradient: bool = True,
    verbose: bool = False,
    xyz_traj_file: str = "",
):
    del opt_method, max_mograd_convergence, project_trans_rot, xyz_traj_file
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

    resolved_eri = eri_method if eri_method else eri_mode
    resolved_aux = aux_basis if aux_basis else (f"{basis}-RI" if resolved_eri.lower() == "df" else "")

    return _orig_mol_opt_geom(
        self,
        method=method,
        eri_mode=resolved_eri,
        eri_method=resolved_eri,
        basis=basis,
        aux_basis=resolved_aux,
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
        verbose=verbose,
    )

if _orig_mol_opt_geom is not None:
    Molecule.optimize_geometry = _psi4_compatible_optimize_geometry


from ._mshqc import kernels
if 'kernels' not in __all__:
    __all__.append('kernels')


from . import methods
if 'methods' not in __all__:
    __all__.append('methods')
