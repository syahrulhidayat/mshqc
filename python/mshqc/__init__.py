"""
MSHQC: Multi-State High-Quality Calculations
Python Package Initialization
"""

import os
import sys
import multiprocessing

# ==============================================================================
# AUTO-CONFIGURATION: THREADING STRATEGY
# ==============================================================================
# Mencegah "Thread Oversubscription" (Hang/Lag) akibat konflik OpenMP vs OpenBLAS.
# Strategi:
# 1. OpenMP (Loop C++ MSHQC) -> Gunakan SEMUA core yang tersedia.
# 2. OpenBLAS/MKL (Aljabar Linear) -> Gunakan 1 thread (Serial) per tugas.
# ==============================================================================

def _configure_threading_defaults():
    # 1. Deteksi jumlah core CPU (Physical/Logical)
    try:
        # Menggunakan cpu_count() standar
        n_cores = multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        n_cores = 4  # Fallback aman jika deteksi gagal

    # 2. Set OpenMP threads (Jika belum diset user di terminal)
    # Ini memastikan perhitungan integral/tensor menggunakan seluruh kekuatan CPU.
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(n_cores)

    # 3. Paksa Library Matriks (OpenBLAS/MKL) ke Mode Serial (Single-Thread)
    # Ini SANGAT KRUSIAL. Jika OpenBLAS mencoba parallel di dalam loop OpenMP,
    # aplikasi akan hang atau performa turun drastis (konflik resource).
    
    # OpenBLAS
    if "OPENBLAS_NUM_THREADS" not in os.environ:
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    # Intel MKL (jika ada di environment conda)
    if "MKL_NUM_THREADS" not in os.environ:
        os.environ["MKL_NUM_THREADS"] = "1"
        
    # Library lain
    if "VECLIB_MAXIMUM_THREADS" not in os.environ:
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    if "NUMEXPR_NUM_THREADS" not in os.environ:
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Jalankan konfigurasi sebelum modul C++ diload!
_configure_threading_defaults()


# ==============================================================================
# IMPORT LIBRARY (BINDINGS & UTILS)
# ==============================================================================

# Import C++ bindings
from ._mshqc import *

# Import Python utilities
# CICalculator dihapus karena binding CI belum aktif
from .calculators import MSHQCCalculator, MCSCFCalculator
from .session import MSHQCSession
from .utils import quick_calculation, benchmark_basis_sets, compare_methods

__version__ = "1.0.0"

__all__ = [
    # Core classes
    "Molecule", "BasisSet", "IntegralEngine", "Atom", "ERITensor",
    "CholeskyERI", "CholeskyDecompositionResult", "PointGroup",

    # SCF
    "SCFConfig", "SCFResult", "UHF", "RHF", "ROHF", 
    "CholeskyUHF", "CholeskyUHFConfig",
    "CholeskyROHF", "CholeskyROHFConfig",
    "CholeskyRHF", "CholeskyRHFConfig",

    # MP Methods
    "UMP2", "UMP2Result", "UMP3", "UMP3Result",
    "RMP2", "RMP2Result", "RMP3", "RMP3Result", 
    "OMP2", "OMP2Result", "OMP3", "OMP3Result",
    "CholeskyRMP2", "CholeskyRMP2Config", "CholeskyRMP2Result",
    "CholeskyOMP2", "CholeskyOMP2Config",
    "CholeskyUMP2", "CholeskyUMP2Config", "CholeskyUMP2Result",
    "CholeskyUMP3", "CholeskyUMP3Config", "CholeskyUMP3Result",
    "CholeskyOMP3", "CholeskyOMP3Config", "CholeskyOMP3Result",
    "CholeskyRMP3",
    # MCSCF / CAS
    "ActiveSpace", "CASResult", "CASSCF", 
    "CholeskyCASSCF", "UNOResult", "CholeskyUNO","CanonicalUNO", 
    
    # SA-CASSCF / PT2 / PT3
    "SACASConfig", "SACASResult", "CholeskySACASSCF","CanonicalSACASSCF",
    "CASPT2Result1", "CASPT2Config", "CASPT2", # Standard CASPT2
    "CASPT2Result", "CholeskySACASPT2", "CanonicalSACASPT2","CanonicalSACASPT3",
    "CASPT3Config", "CASPT3Result", "CholeskySACASPT3",

    # Python wrappers
    "MSHQCCalculator", "MCSCFCalculator", "MSHQCSession",
    "quick_calculation", "benchmark_basis_sets", "compare_methods",
]