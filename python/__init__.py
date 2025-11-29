"""
MSH-QC: Quantum Mechanics Library for Computational Chemistry

Comprehensive Python bindings for MSH-QC quantum chemistry library.
"""

import sys
import warnings

# Version
__version__ = "0.1.0"
__author__ = "Muhamad Sahrul Hidayat"

# Import core module with error handling
try:
    from ._core import *
except ImportError as e:
    error_msg = f"""
    Failed to import MSH-QC core module: {e}
    
    This usually means:
    1. The C++ extension was not built properly
    2. Missing dependencies (Eigen3, libint2, etc.)
    3. Incompatible Python/NumPy versions
    
    Try reinstalling with verbose mode:
        pip install --force-reinstall --no-cache-dir --verbose mshqc
    
    Or install with minimal dependencies:
        MSHQC_WITH_LIBINT2=OFF MSHQC_WITH_LIBCINT=OFF pip install .
    """
    raise ImportError(error_msg) from e

# Check NumPy compatibility
try:
    import numpy as np
    if np.__version__ < "1.22":
        warnings.warn(
            f"NumPy version {np.__version__} is older than recommended (1.22+). "
            "Some features may not work correctly.",
            UserWarning
        )
except ImportError:
    raise ImportError("NumPy is required but not installed. Install it with: pip install numpy>=1.22")

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_rhf(molecule, basis="sto-3g"):
    """Perform a quick RHF calculation
    
    Args:
        molecule: Molecule object
        basis: Basis set name (default: sto-3g)
    
    Returns:
        SCFResult object
    
    Example:
        >>> mol = create_h2_molecule()
        >>> result = quick_rhf(mol, "sto-3g")
        >>> print(f"Energy: {result.energy:.6f} Hartree")
    """
    bas = BasisSet()
    bas.load(basis, molecule)
    
    rhf = RHF(molecule, bas)
    result = rhf.solve()
    
    return result

def quick_uhf(molecule, basis="sto-3g"):
    """Perform a quick UHF calculation"""
    bas = BasisSet()
    bas.load(basis, molecule)
    
    uhf = UHF(molecule, bas)
    result = uhf.solve()
    
    return result

def quick_mp2(molecule, basis="sto-3g"):
    """Perform RHF followed by MP2 calculation"""
    bas = BasisSet()
    bas.load(basis, molecule)
    rhf = RHF(molecule, bas)
    scf_result = rhf.solve()
    
    integrals = IntegralEngine(molecule, bas)
    mp2 = RMP2(scf_result, integrals)
    mp2_result = mp2.solve()
    
    return scf_result, mp2_result

def quick_mp3(molecule, basis="sto-3g"):
    """Perform RHF followed by MP3 calculation"""
    bas = BasisSet()
    bas.load(basis, molecule)
    rhf = RHF(molecule, bas)
    scf_result = rhf.solve()
    
    integrals = IntegralEngine(molecule, bas)
    mp3 = RMP3(scf_result, integrals)
    mp3_result = mp3.solve()
    
    return scf_result, mp3_result

def quick_cisd(molecule, basis="sto-3g"):
    """Perform RHF followed by CISD calculation"""
    bas = BasisSet()
    bas.load(basis, molecule)
    rhf = RHF(molecule, bas)
    scf_result = rhf.solve()
    
    integrals = IntegralEngine(molecule, bas)
    cisd = CISD(scf_result, integrals)
    ci_result = cisd.solve()
    
    return scf_result, ci_result

def quick_fci(molecule, basis="sto-3g"):
    """Perform RHF followed by FCI calculation"""
    bas = BasisSet()
    bas.load(basis, molecule)
    rhf = RHF(molecule, bas)
    scf_result = rhf.solve()
    
    integrals = IntegralEngine(molecule, bas)
    fci = FCI(scf_result, integrals)
    ci_result = fci.solve()
    
    return scf_result, ci_result

def quick_casscf(molecule, n_electrons, n_orbitals, basis="sto-3g"):
    """Perform CASSCF calculation"""
    bas = BasisSet()
    bas.load(basis, molecule)
    
    casscf = CASSCF(molecule, bas)
    casscf.set_active_space(n_electrons, n_orbitals)
    result = casscf.solve()
    
    return result

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Core classes
    "Molecule", "BasisSet", "IntegralEngine",
    
    # SCF methods
    "SCFResult", "RHF", "ROHF", "UHF",
    
    # MP2/MP3 methods
    "MP2Result", "RMP2", "UMP2", "DFMP2",
    "UMP3Result", "UMP3", "RMP3",
    "UMP4", "UMP5",
    
    # CI methods
    "Determinant", "CIResult",
    "CIS", "CISD", "CISDT", "FCI", "MRCI", "CIPSI",
    
    # MCSCF methods
    "ActiveSpace", "CASSCFResult",
    "CASSCF", "SACASSCF", "CASPT2", "DFCASPT2", "MRMP2",
    
    # Gradient and optimization
    "NumericalGradient", "GeometryOptimizer",
    
    # Convenience functions
    "quick_rhf", "quick_uhf", "quick_mp2", "quick_mp3",
    "quick_cisd", "quick_fci", "quick_casscf",
    
    # Utility functions
    "create_h2_molecule", "create_water_molecule",
    "create_li_atom", "create_he_atom", "create_ne_atom",
]
