"""
MSHQC: Multi-State High-Quality Calculations
Python Package Initialization
"""

import os
import sys
import multiprocessing


















def _configure_threading_defaults():
    

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



_configure_threading_defaults()











from ._mshqc import *





from .calculators import MSHQCCalculator, MCSCFCalculator
from .session import MSHQCSession
from .utils import quick_calculation, benchmark_basis_sets, compare_methods

__version__ = "1.0.0"

__all__ = [
    

    "Molecule", "BasisSet", "IntegralEngine", "Atom", "ERITensor",
    "CholeskyERI", "CholeskyDecompositionResult", "PointGroup",

    

    "SCFConfig", "SCFResult", "UHF", "RHF", "ROHF", 
    "CholeskyUHF", "CholeskyUHFConfig",
    "CholeskyROHF", "CholeskyROHFConfig",
    "CholeskyRHF", "CholeskyRHFConfig",

    

    "UMP2", "UMP2Result", "UMP3", "UMP3Result",
    "RMP2", "RMP2Result", "RMP3", "RMP3Result", 
    "OMP2", "OMP2Result", "OMP3", "OMP3Result",
    "CholeskyRMP2", "CholeskyRMP2Config", "CholeskyRMP2Result",
    "CholeskyOMP2", "CholeskyOMP2Config",
    "CholeskyUMP2", "CholeskyUMP2Config", "CholeskyUMP2Result",
    "CholeskyUMP3", "CholeskyUMP3Config", "CholeskyUMP3Result",
    "CholeskyOMP3", "CholeskyOMP3Config", "CholeskyOMP3Result",
    "CholeskyRMP3",
    

    "ActiveSpace", "CASResult", "CASSCF", 
    "CholeskyCASSCF", "UNOResult", "CholeskyUNO","CanonicalUNO", 
    
    

    "SACASConfig", "SACASResult", "CholeskySACASSCF","CanonicalSACASSCF",
    "CASPT2Result1", "CASPT2Config", "CASPT2", 

    "CASPT2Result", "CholeskySACASPT2", "CanonicalSACASPT2","CanonicalSACASPT3",
    "CASPT3Config", "CASPT3Result", "CholeskySACASPT3",

    

    "MSHQCCalculator", "MCSCFCalculator", "MSHQCSession",
    "quick_calculation", "benchmark_basis_sets", "compare_methods",
]