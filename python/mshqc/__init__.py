"""
MSH-QC: Quantum Mechanics Library for Computational Chemistry

This package provides Python bindings for the MSH-QC quantum chemistry library.
MSH-QC implements a range of electronic structure methods from Hartree-Fock
to multi-reference approaches.
"""

from ._core import *
import os

# Set default data path for basis sets
try:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    default_basis_path = os.path.join(os.path.dirname(this_dir), "data", "basis")
    if os.path.exists(default_basis_path):
        set_default_basis_path(default_basis_path)
except:
    pass  # If this fails, users can still specify path manually

# Version
__version__ = "0.1.0"
__author__ = "Muhamad Sahrul Hidayat"

# Convenience functions
def quick_rhf(molecule, basis="sto-3g"):
    """Perform a quick RHF calculation
    
    Args:
        molecule: Molecule object
        basis: Basis set name (default: sto-3g)
    
    Returns:
        SCFResult object
    """
    from . import BasisSet, RHF
    
    bas = BasisSet()
    bas.load_minimal(basis)
    
    rhf = RHF(molecule, bas)
    result = rhf.solve()
    
    return result

def quick_mp2(molecule, basis="sto-3g"):
    """Perform a quick RHF followed by MP2 calculation
    
    Args:
        molecule: Molecule object
        basis: Basis set name (default: sto-3g)
    
    Returns:
        Tuple of (RHF result, MP2 result)
    """
    from . import BasisSet, RHF, MP2
    
    # First RHF
    rhf_result = quick_rhf(molecule, basis)
    
    # Then MP2
    mp2 = MP2(rhf_result)
    mp2_result = mp2.solve()
    
    return rhf_result, mp2_result

def quick_cisd(molecule, basis="sto-3g"):
    """Perform a quick RHF followed by CISD calculation
    
    Args:
        molecule: Molecule object
        basis: Basis set name (default: sto-3g)
    
    Returns:
        Tuple of (RHF result, CISD result)
    """
    from . import BasisSet, RHF, CISD
    
    # First RHF
    rhf_result = quick_rhf(molecule, basis)
    
    # Then CISD
    cisd = CISD(rhf_result)
    cisd_result = cisd.solve()
    
    return rhf_result, cisd_result

# Export convenience functions
__all__ = [
    # Core classes
    "Molecule", "BasisSet", "SCFResult",
    
    # SCF methods
    "RHF", "UHF",
    
    # Correlation methods
    "MP2", "CISD", "CASSCF",
    
    # Convenience functions
    "quick_rhf", "quick_mp2", "quick_cisd",
    
    # Utility functions
    "create_water_molecule", "create_h2_molecule", "create_li_atom"
]