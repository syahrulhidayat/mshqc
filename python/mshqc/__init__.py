"""
MSHQC: Multi-State High-Quality Calculations
Python Package Initialization
"""

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
    "CholeskyERI", "CholeskyDecompositionResult",

    # SCF
    "SCFConfig", "SCFResult", "UHF", "RHF", "ROHF", 
    "CholeskyUHF", "CholeskyUHFConfig",

    # MP Methods
    "UMP2", "UMP2Result", "UMP3", "UMP3Result",
    "RMP2", "RMP2Result", "RMP3", "RMP3Result", # Added RMP
    "OMP2Result", # OMP2 class might be missing in pyi but Result is there
    "CholeskyUMP2", "CholeskyUMP2Config", "CholeskyUMP2Result",
    "CholeskyUMP3", "CholeskyUMP3Config", "CholeskyUMP3Result",

    # MCSCF / CAS
    "ActiveSpace", "CASResult", "CASSCF", 
    "CholeskyCASSCF", "UNOResult", "CholeskyUNO",
    
    # SA-CASSCF / PT2 / PT3
    "SACASConfig", "SACASResult", "CholeskySACASSCF",
    "CASPT2Result1", "CASPT2Config", "CASPT2", # Standard CASPT2
    "CASPT2Result", "CholeskySACASPT2", 
    "CASPT3Config", "CASPT3Result", "CholeskySACASPT3",

    # Python wrappers
    "MSHQCCalculator", "MCSCFCalculator", "MSHQCSession",
    "quick_calculation", "benchmark_basis_sets", "compare_methods",
]