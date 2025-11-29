"""
MSH-QC: Quantum Mechanics Library for Computational Chemistry

Comprehensive Python bindings for MSH-QC quantum chemistry library.
MSH-QC implements a wide range of electronic structure methods from 
Hartree-Fock to advanced multi-reference approaches.
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
    bas.load(basis, molecule)
    
    rhf = RHF(molecule, bas)
    result = rhf.solve()
    
    return result

def quick_uhf(molecule, basis="sto-3g"):
    """Perform a quick UHF calculation
    
    Args:
        molecule: Molecule object
        basis: Basis set name (default: sto-3g)
    
    Returns:
        SCFResult object
    """
    from . import BasisSet, UHF
    
    bas = BasisSet()
    bas.load(basis, molecule)
    
    uhf = UHF(molecule, bas)
    result = uhf.solve()
    
    return result

def quick_mp2(molecule, basis="sto-3g"):
    """Perform RHF followed by MP2 calculation
    
    Args:
        molecule: Molecule object
        basis: Basis set name (default: sto-3g)
    
    Returns:
        Tuple of (SCFResult, MP2Result)
    """
    from . import BasisSet, RHF, RMP2, IntegralEngine
    
    # RHF
    bas = BasisSet()
    bas.load(basis, molecule)
    rhf = RHF(molecule, bas)
    scf_result = rhf.solve()
    
    # MP2
    integrals = IntegralEngine(molecule, bas)
    mp2 = RMP2(scf_result, integrals)
    mp2_result = mp2.solve()
    
    return scf_result, mp2_result

def quick_mp3(molecule, basis="sto-3g"):
    """Perform RHF followed by MP3 calculation
    
    Args:
        molecule: Molecule object
        basis: Basis set name (default: sto-3g)
    
    Returns:
        Tuple of (SCFResult, MP3Result)
    """
    from . import BasisSet, RHF, RMP3, IntegralEngine
    
    # RHF
    bas = BasisSet()
    bas.load(basis, molecule)
    rhf = RHF(molecule, bas)
    scf_result = rhf.solve()
    
    # MP3
    integrals = IntegralEngine(molecule, bas)
    mp3 = RMP3(scf_result, integrals)
    mp3_result = mp3.solve()
    
    return scf_result, mp3_result

def quick_cisd(molecule, basis="sto-3g"):
    """Perform RHF followed by CISD calculation
    
    Args:
        molecule: Molecule object
        basis: Basis set name (default: sto-3g)
    
    Returns:
        Tuple of (SCFResult, CIResult)
    """
    from . import BasisSet, RHF, CISD, IntegralEngine
    
    # RHF
    bas = BasisSet()
    bas.load(basis, molecule)
    rhf = RHF(molecule, bas)
    scf_result = rhf.solve()
    
    # CISD
    integrals = IntegralEngine(molecule, bas)
    cisd = CISD(scf_result, integrals)
    ci_result = cisd.solve()
    
    return scf_result, ci_result

def quick_fci(molecule, basis="sto-3g"):
    """Perform RHF followed by FCI calculation
    
    Args:
        molecule: Molecule object
        basis: Basis set name (default: sto-3g)
    
    Returns:
        Tuple of (SCFResult, CIResult)
    """
    from . import BasisSet, RHF, FCI, IntegralEngine
    
    # RHF
    bas = BasisSet()
    bas.load(basis, molecule)
    rhf = RHF(molecule, bas)
    scf_result = rhf.solve()
    
    # FCI
    integrals = IntegralEngine(molecule, bas)
    fci = FCI(scf_result, integrals)
    ci_result = fci.solve()
    
    return scf_result, ci_result

def quick_casscf(molecule, n_electrons, n_orbitals, basis="sto-3g"):
    """Perform CASSCF calculation
    
    Args:
        molecule: Molecule object
        n_electrons: Number of active electrons
        n_orbitals: Number of active orbitals
        basis: Basis set name (default: sto-3g)
    
    Returns:
        CASSCFResult object
    """
    from . import BasisSet, CASSCF
    
    bas = BasisSet()
    bas.load(basis, molecule)
    
    casscf = CASSCF(molecule, bas)
    casscf.set_active_space(n_electrons, n_orbitals)
    result = casscf.solve()
    
    return result

# Export all classes and functions
__all__ = [
    # Core classes
    "Molecule", "BasisSet", "IntegralEngine",
    
    # SCF methods and results
    "SCFResult", "RHF", "ROHF", "UHF",
    
    # MP2/MP3 methods and results
    "MP2Result", "RMP2", "UMP2", "DFMP2",
    "UMP3Result", "UMP3", "RMP3",
    "UMP4", "UMP5",
    
    # CI methods and results
    "Determinant", "CIResult",
    "CIS", "CISD", "CISDT", "FCI", "MRCI", "CIPSI",
    
    # MCSCF methods and results
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