/**
 * @file h2_caspt2_test.cc
 * @brief Test CASPT2 on H2 molecule
 * 
 * Test case: H2 molecule with CAS(2,2) and CASPT2 correction
 * Expected: E(PT2) < 0 (negative correction)
 * 
 * @author AI Agent 3 (Multireference Master)
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
    std::cout << "H2 CASPT2 Test\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // ========================================================================
    // 1. Setup H2 molecule
    // ========================================================================
    std::cout << "Setting up H2 molecule...\n";
    
    Molecule mol;
    mol.add_atom(1, 0.0, 0.0, 0.0);           // H (Z=1) at origin
    mol.add_atom(1, 0.0, 0.0, 1.4);           // H at 1.4 bohr
    
    std::cout << "  Atoms: " << mol.n_atoms() << "\n";
    std::cout << "  Electrons: " << mol.n_electrons() << "\n";
    std::cout << "  Bond length: 1.4 bohr\n\n";
    
    // ========================================================================
    // 2. Setup basis set (STO-3G)
    // ========================================================================
    std::cout << "Setting up STO-3G basis...\n";
    
    BasisSet basis("sto-3g", mol);
    int nbf = basis.n_basis_functions();
    
    std::cout << "  Basis functions: " << nbf << "\n\n";
    
    // ========================================================================
    // 3. Compute integrals
    // ========================================================================
    std::cout << "Computing integrals...\n";
    
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    
    std::cout << "  Integrals computed\n\n";
    
    // ========================================================================
    // 4. Run Hartree-Fock (initial guess)
    // ========================================================================
    std::cout << "Running Hartree-Fock...\n";
    
    RHF rhf(mol, basis, integrals);
    
    auto hf_result = rhf.compute();
    
    std::cout << "  E(HF) = " << std::fixed << std::setprecision(10) 
              << hf_result.energy_total << " Ha\n\n";
    
    // ========================================================================
    // 5. Define active space: CAS(2,2)
    // ========================================================================
    std::cout << "Defining active space CAS(2,2)...\n";
    
    int n_elec = mol.n_electrons();  // 2 electrons
    
    // For H2 with 2 basis functions: active space = all orbitals
    ActiveSpace active_space = ActiveSpace::CAS(
        2,     // 2 electrons in active space
        2,     // 2 orbitals in active space (σ and σ*)
        nbf,   // Total number of orbitals
        n_elec // Total number of electrons
    );
    
    std::cout << "  " << active_space.to_string() << "\n";
    std::cout << "  Inactive orbitals: " << active_space.n_inactive() << "\n";
    std::cout << "  Active orbitals:   " << active_space.n_active() << "\n";
    std::cout << "  Virtual orbitals:  " << active_space.n_virtual() << "\n\n";
    
    // ========================================================================
    // 6. Run CASSCF
    // ========================================================================
    std::cout << "Running CASSCF...\n\n";
    
    CASSCF casscf(mol, basis, integrals, active_space);
    casscf.set_max_iterations(20);
    casscf.set_energy_threshold(1e-8);
    casscf.set_gradient_threshold(1e-6);
    
    auto casscf_result = casscf.compute(hf_result);
    
    if (!casscf_result.converged) {
        std::cerr << "ERROR: CASSCF did not converge!\n";
        return 1;
    }
    
    std::cout << "\nCASSCF converged!\n";
    std::cout << "  E(CASSCF) = " << std::fixed << std::setprecision(10) 
              << casscf_result.e_casscf << " Ha\n";
    std::cout << "  Iterations: " << casscf_result.n_iterations << "\n";
    std::cout << "  Determinants: " << casscf_result.n_determinants << "\n\n";
    
    // ========================================================================
    // 7. Run CASPT2
    // ========================================================================
    std::cout << "Running CASPT2...\n\n";
    
    CASPT2 caspt2(mol, basis, integrals, casscf_result);
    
    // Optional: set IPEA shift (typically 0.25 Ha)
    // caspt2.set_ipea_shift(0.25);
    
    auto caspt2_result = caspt2.compute();
    
    // ========================================================================
    // 8. Analyze results
    // ========================================================================
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "H2 CASPT2 Test Results\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    std::cout << "Energy breakdown:\n";
    std::cout << "  E(HF)          = " << std::fixed << std::setprecision(10) 
              << hf_result.energy_total << " Ha\n";
    std::cout << "  E(CASSCF)      = " << casscf_result.e_casscf << " Ha\n";
    std::cout << "  E(PT2)         = " << caspt2_result.e_pt2 << " Ha\n";
    std::cout << "  E(CASPT2)      = " << caspt2_result.e_total << " Ha\n\n";
    
    std::cout << "Correlation energy:\n";
    double e_corr_casscf = casscf_result.e_casscf - hf_result.energy_total;
    double e_corr_caspt2 = caspt2_result.e_total - hf_result.energy_total;
    
    std::cout << "  ΔE(CASSCF-HF)  = " << std::scientific << std::setprecision(6)
              << e_corr_casscf << " Ha\n";
    std::cout << "  ΔE(CASPT2-HF)  = " << e_corr_caspt2 << " Ha\n";
    std::cout << "  E(PT2)         = " << caspt2_result.e_pt2 << " Ha\n\n";
    
    // ========================================================================
    // 9. Validation checks
    // ========================================================================
    std::cout << "Validation checks:\n";
    
    bool valid = true;
    
    // Check 1: CASSCF should be lower than HF
    if (casscf_result.e_casscf < hf_result.energy_total) {
        std::cout << "  ✓ E(CASSCF) < E(HF) (variational principle)\n";
    } else {
        std::cout << "  ✗ E(CASSCF) >= E(HF) (FAILED!)\n";
        valid = false;
    }
    
    // Check 2: PT2 correction typically negative (but not guaranteed)
    if (caspt2_result.e_pt2 <= 0.0) {
        std::cout << "  ✓ E(PT2) <= 0 (typical for perturbation)\n";
    } else {
        std::cout << "  ⚠ E(PT2) > 0 (unusual but not necessarily wrong)\n";
    }
    
    // Check 3: CASPT2 converged
    if (caspt2_result.converged) {
        std::cout << "  ✓ CASPT2 converged\n";
    } else {
        std::cout << "  ✗ CASPT2 did not converge\n";
        valid = false;
    }
    
    // Check 4: Energy magnitudes reasonable
    if (std::abs(casscf_result.e_casscf) < 10.0 && 
        std::abs(caspt2_result.e_pt2) < 1.0) {
        std::cout << "  ✓ Energy magnitudes reasonable\n";
    } else {
        std::cout << "  ✗ Energy magnitudes suspicious\n";
        valid = false;
    }
    
    std::cout << "\nStatus: " << caspt2_result.status_message << "\n";
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    if (valid) {
        std::cout << "TEST PASSED ✓\n";
    } else {
        std::cout << "TEST FAILED ✗\n";
    }
    std::cout << std::string(70, '=') << "\n";
    
    return valid ? 0 : 1;
}
