"""
MSH-QC: Quantum Mechanics Library for Computational Chemistry

This package provides Python bindings for the MSH-QC quantum chemistry library.
MSH-QC implements a range of electronic structure methods from Hartree-Fock
to multi-reference approaches.
"""

from ._core import *
import os

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

# Export convenience functions
__all__ = [
    # Core classes
    "Molecule", "BasisSet", "SCFResult",
    
    # SCF methods
    "RHF",
    
    # Convenience functions
    "quick_rhf",
    
    # Utility functions
    "create_water_molecule", "create_h2_molecule"
]