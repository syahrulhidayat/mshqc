#!/usr/bin/env python3
"""
Example of using MSH-QC Python bindings
"""

import numpy as np
import mshqc

def h2_rhf_example():
    """Basic RHF calculation on H2 molecule"""
    print("=== H2 RHF Example ===")
    
    # Create H2 molecule with default bond distance
    mol = mshqc.create_h2_molecule()
    print(f"H2 molecule: {mol.natoms()} atoms, {mol.nelectrons()} electrons")
    
    # Perform RHF calculation
    result = mshqc.quick_rhf(mol, "sto-3g")
    print(f"RHF Energy: {result.energy:.6f} Hartree")
    
    # Get molecular orbital coefficients
    mo_coeffs = result.C
    print(f"MO coefficients shape: {mo_coeffs.shape}")
    
    # Get orbital energies
    orbital_energies = result.e
    print(f"Orbital energies: {orbital_energies}")
    
    return result

def water_mp2_example():
    """MP2 calculation on water molecule"""
    print("\n=== Water MP2 Example ===")
    
    # Create water molecule
    mol = mshqc.create_water_molecule()
    print(f"H2O molecule: {mol.natoms()} atoms, {mol.nelectrons()} electrons")
    
    # Perform RHF followed by MP2
    rhf_result, mp2 = mshqc.quick_mp2(mol, "sto-3g")
    print(f"RHF Energy: {rhf_result.energy:.6f} Hartree")
    print(f"MP2 Energy: {mp2.get_energy():.6f} Hartree")
    print(f"MP2 Correlation: {mp2.get_correlation_energy():.6f} Hartree")
    
    return mp2

def custom_molecule_example():
    """Example with custom molecule and basis set"""
    print("\n=== Custom Molecule Example ===")
    
    # Create a custom water molecule with slightly different geometry
    mol = mshqc.Molecule()
    mol.add_atom(8, 0.0, 0.0, 0.0)        # O at origin
    mol.add_atom(1, 0.0, 0.9, 0.4)        # H
    mol.add_atom(1, 0.0, -0.9, 0.4)       # H
    
    print(f"Custom H2O: charge={mol.charge()}, multiplicity={mol.multiplicity()}")
    
    # Load basis set
    basis = mshqc.BasisSet()
    basis.load_minimal("sto-3g")
    print(f"Basis set: {basis.nbasis()} basis functions")
    
    # Perform RHF
    rhf = mshqc.RHF(mol, basis)
    result = rhf.solve()
    
    print(f"RHF converged: {result.converged}")
    print(f"RHF Energy: {result.energy:.6f} Hartree")
    
    # Get density matrix
    density = result.D
    print(f"Density matrix shape: {density.shape}")
    
    return result

def li_casscf_example():
    """CASSCF calculation on Li atom"""
    print("\n=== Li CASSCF Example ===")
    
    # Create Li atom
    mol = mshqc.create_li_atom()
    print(f"Li atom: {mol.natoms()} atoms, {mol.nelectrons()} electrons")
    
    # Load basis set
    basis = mshqc.BasisSet()
    basis.load_minimal("sto-3g")
    
    # Set up CASSCF with (2,2) active space
    casscf = mshqc.CASSCF(mol, basis)
    casscf.set_active_space(2, 2)  # 2 electrons in 2 orbitals
    result = casscf.solve()
    
    print(f"CASSCF Energy: {result.energy:.6f} Hartree")
    
    casscf.save_orbitals("li_casscf.txt")
    
    return result

def analyze_wavefunction(result):
    """Analyze SCF result"""
    print("\n=== Wavefunction Analysis ===")
    
    # Get orbital energies
    energies = result.e
    n_occ = result.n_occ
    
    print(f"Number of occupied orbitals: {n_occ}")
    print(f"HOMO energy: {energies[n_occ-1]:.6f} Hartree")
    print(f"LUMO energy: {energies[n_occ]:.6f} Hartree")
    print(f"HOMO-LUMO gap: {(energies[n_occ]-energies[n_occ-1])*27.2114:.2f} eV")
    
    # Get density matrix
    density = result.D
    
    # Compute total number of electrons from density matrix
    n_electrons = np.trace(density)
    print(f"Electron count from density: {n_electrons:.2f}")
    
    # Compute Mulliken charges
    # First, we'd need access to overlap matrix which isn't in the result
    # So we'll skip this for now until we expose more functionality

if __name__ == "__main__":
    print("MSH-QC Python Bindings Example")
    print("==============================")
    
    # Run examples
    try:
        h2_rhf_example()
        water_mp2_example()
        custom_molecule_example()
        li_casscf_example()
        analyze_wavefunction(h2_rhf_example())
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()