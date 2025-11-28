/**
 * @file he_caspt2_test.cc
 * @brief Test CASPT2 on Helium atom
 * 
 * Test case: He atom with CAS(2,3) - partial active space
 * This should generate external space for PT2 correction
 * 
 * @author AI Agent 3
 * @date 2025-11-12
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/mcscf/casscf.h"
#include "mshqc/mcscf/caspt2.h"
#include <iostream>
#include <iomanip>
#include <memory>

using namespace mshqc;
using namespace mshqc::mcscf;

int main() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Helium CASPT2 Test\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // Setup He atom
    Molecule mol;
    mol.add_atom(2, 0.0, 0.0, 0.0);  // He (Z=2)
    
    std::cout << "Atom: He (Z=2)\n";
    std::cout << "Electrons: " << mol.n_electrons() << "\n\n";
    
    // Setup basis
    BasisSet basis("sto-3g", mol);
    int nbf = basis.n_basis_functions();
    
    std::cout << "Basis: STO-3G\n";
    std::cout << "Basis functions: " << nbf << "\n\n";
    
    // Compute integrals
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    
    // Run HF
    std::cout << "Running Hartree-Fock...\n";
    RHF rhf(mol, basis, integrals);
    auto hf_result = rhf.compute();
    
    std::cout << "  E(HF) = " << std::fixed << std::setprecision(10) 
              << hf_result.energy_total << " Ha\n\n";
    
    // Define active space: CAS(2,3)
    // He has 1 orbital (1s) in STO-3G, so we'll use all available
    std::cout << "Defining active space CAS(2," << nbf << ")...\n";
    
    int n_elec = mol.n_electrons();
    ActiveSpace active_space = ActiveSpace::CAS(
        n_elec,  // All electrons in active space
        nbf,     // All orbitals in active space
        nbf,
        n_elec
    );
    
    std::cout << "  " << active_space.to_string() << "\n";
    std::cout << "  Inactive: " << active_space.n_inactive() << "\n";
    std::cout << "  Active:   " << active_space.n_active() << "\n";
    std::cout << "  Virtual:  " << active_space.n_virtual() << "\n\n";
    
    // Run CASSCF
    std::cout << "Running CASSCF...\n\n";
    CASSCF casscf(mol, basis, integrals, active_space);
    casscf.set_max_iterations(20);
    
    auto casscf_result = casscf.compute(hf_result);
    
    if (!casscf_result.converged) {
        std::cerr << "ERROR: CASSCF did not converge!\n";
        return 1;
    }
    
    std::cout << "\nCASSCF converged!\n";
    std::cout << "  E(CASSCF) = " << std::fixed << std::setprecision(10) 
              << casscf_result.e_casscf << " Ha\n";
    std::cout << "  Determinants: " << casscf_result.n_determinants << "\n\n";
    
    // Run CASPT2
    std::cout << "Running CASPT2...\n\n";
    CASPT2 caspt2(mol, basis, integrals, casscf_result);
    
    auto caspt2_result = caspt2.compute();
    
    // Results
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Helium CASPT2 Results\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    std::cout << "Energy breakdown:\n";
    std::cout << "  E(HF)      = " << std::fixed << std::setprecision(10) 
              << hf_result.energy_total << " Ha\n";
    std::cout << "  E(CASSCF)  = " << casscf_result.e_casscf << " Ha\n";
    std::cout << "  E(PT2)     = " << caspt2_result.e_pt2 << " Ha\n";
    std::cout << "  E(CASPT2)  = " << caspt2_result.e_total << " Ha\n\n";
    
    double e_corr = casscf_result.e_casscf - hf_result.energy_total;
    std::cout << "Correlation:\n";
    std::cout << "  ΔE(CASSCF-HF) = " << std::scientific << std::setprecision(6)
              << e_corr << " Ha\n";
    std::cout << "  E(PT2)        = " << caspt2_result.e_pt2 << " Ha\n\n";
    
    // Validation
    std::cout << "Validation:\n";
    
    if (casscf_result.e_casscf <= hf_result.energy_total) {
        std::cout << "  ✓ E(CASSCF) <= E(HF)\n";
    } else {
        std::cout << "  ✗ E(CASSCF) > E(HF)\n";
    }
    
    if (caspt2_result.converged) {
        std::cout << "  ✓ CASPT2 converged\n";
    }
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Status: " << caspt2_result.status_message << "\n";
    std::cout << std::string(70, '=') << "\n";
    
    return 0;
}
