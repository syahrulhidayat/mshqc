/**
 * @file casscf_h2_test.cc
 * @brief Test CASSCF on H2 molecule with CAS(2,2)
 * 
 * For H2 with minimal basis (STO-3G):
 * - 2 electrons, 2 basis functions
 * - CAS(2,2) = FCI (Full CI) = exact solution
 * - Should match FCI energy exactly
 * 
 * VALIDATION REFERENCE:
 * - Szabo & Ostlund, "Modern Quantum Chemistry", Example on H2
 * - H2 @ 0.74 Å (equilibrium): E_FCI ≈ -1.137 Ha (STO-3G)
 * 
 * @author Muhamad Sahrul Hidayat (AI Agent 3)
 * @date 2025-11-12
 * @license MIT
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/mcscf/active_space.h"
#include "mshqc/mcscf/casscf.h"
#include <iostream>
#include <iomanip>

using namespace mshqc;
using namespace mshqc::mcscf;

int main() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "CASSCF Test: H2 molecule with CAS(2,2)\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // ========================================================================
    // Setup H2 molecule at equilibrium (0.74 Angstrom)
    // ========================================================================
    
    Molecule mol;
    mol.add_atom(1, 0.0, 0.0, 0.0);           // H at origin
    mol.add_atom(1, 0.0, 0.0, 0.74 * 1.8897); // H at 0.74 Å (convert to bohr)
    
    std::cout << "Molecule: H2\n";
    std::cout << "Bond length: 0.74 Angstrom\n";
    std::cout << "Nuclear repulsion: " << std::fixed << std::setprecision(8)
              << mol.nuclear_repulsion_energy() << " Ha\n\n";
    
    // ========================================================================
    // Setup basis: STO-3G (minimal basis, 2 functions for H2)
    // ========================================================================
    
    BasisSet basis("sto-3g", mol);
    std::cout << "Basis set: STO-3G\n";
    std::cout << "Number of basis functions: " << basis.n_basis_functions() << "\n";
    std::cout << "Number of electrons: " << mol.n_electrons() << "\n\n";
    
    // ========================================================================
    // Setup integral engine
    // ========================================================================
    
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    std::cout << "Integral engine initialized\n\n";
    
    // ========================================================================
    // Step 1: Run RHF for initial orbitals
    // ========================================================================
    
    std::cout << std::string(70, '-') << "\n";
    std::cout << "Step 1: Hartree-Fock (initial guess)\n";
    std::cout << std::string(70, '-') << "\n\n";
    
    RHF rhf(mol, basis, integrals);
    
    auto hf_result = rhf.compute();
    
    std::cout << "\nRHF Results:\n";
    std::cout << "  E(HF)  = " << std::fixed << std::setprecision(10)
              << hf_result.energy_total << " Ha\n";
    std::cout << "  Converged in " << hf_result.iterations << " iterations\n";
    
    // ========================================================================
    // Step 2: Setup CASSCF active space - CAS(2,2)
    // ========================================================================
    
    std::cout << "\n" << std::string(70, '-') << "\n";
    std::cout << "Step 2: Setup CASSCF active space\n";
    std::cout << std::string(70, '-') << "\n\n";
    
    // For H2/STO-3G: 2 electrons in 2 orbitals
    // This is CAS(2,2) - all electrons in all orbitals = FCI
    int n_elec = mol.n_electrons();
    int n_orb = basis.n_basis_functions();
    
    auto active_space = ActiveSpace::CAS(n_elec, n_orb, n_orb, n_elec);
    
    std::cout << "Active space: " << active_space.to_string() << "\n";
    std::cout << "  n_inactive = " << active_space.n_inactive() << "\n";
    std::cout << "  n_active   = " << active_space.n_active() << "\n";
    std::cout << "  n_virtual  = " << active_space.n_virtual() << "\n";
    std::cout << "  n_elec_act = " << active_space.n_elec_active() << "\n";
    
    // Calculate expected number of determinants
    // For CAS(n_elec, n_orb) with n_alpha = n_beta = n_elec/2:
    // N_det = (n_orb choose n_alpha)^2
    int n_alpha = n_elec / 2;
    int n_det_expected = 1; // binomial(2, 1) * binomial(2, 1) = 2 * 2 = 4
    // Actually for CAS(2,2): det = C(2,1) * C(2,1) = 2 * 2 = 4
    // But there are also (2,0) and (0,2): total 6 determinants
    std::cout << "  Expected determinants: ~6 (full CI space)\n\n";
    
    // ========================================================================
    // Step 3: Run CASSCF
    // ========================================================================
    
    std::cout << std::string(70, '-') << "\n";
    std::cout << "Step 3: CASSCF Calculation\n";
    std::cout << std::string(70, '-') << "\n";
    
    CASSCF casscf(mol, basis, integrals, active_space);
    casscf.set_max_iterations(50);
    casscf.set_energy_threshold(1e-8);
    casscf.set_gradient_threshold(1e-6);
    
    auto result = casscf.compute(hf_result);
    
    // ========================================================================
    // Step 4: Analyze results
    // ========================================================================
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "CASSCF Results\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    if (result.converged) {
        std::cout << "✓ CASSCF CONVERGED!\n\n";
    } else {
        std::cout << "✗ CASSCF DID NOT CONVERGE\n\n";
    }
    
    std::cout << "Final energies:\n";
    std::cout << "  E(RHF)     = " << std::fixed << std::setprecision(10)
              << hf_result.energy_total << " Ha\n";
    std::cout << "  E(CASSCF)  = " << result.e_casscf << " Ha\n";
    std::cout << "  Correlation = " << result.e_casscf - hf_result.energy_total << " Ha\n";
    std::cout << "  E(nuclear) = " << result.e_nuclear << " Ha\n\n";
    
    std::cout << "Convergence:\n";
    std::cout << "  Iterations: " << result.n_iterations << "\n";
    std::cout << "  Status: " << (result.converged ? "CONVERGED" : "NOT CONVERGED") << "\n\n";
    
    // Print energy history
    if (!result.energy_history.empty()) {
        std::cout << "Energy history:\n";
        for (size_t i = 0; i < result.energy_history.size(); i++) {
            std::cout << "  Iter " << std::setw(2) << i + 1 << ": "
                      << std::fixed << std::setprecision(10)
                      << result.energy_history[i] << " Ha";
            if (i > 0) {
                double delta = result.energy_history[i] - result.energy_history[i-1];
                std::cout << "  (ΔE = " << std::scientific << std::setprecision(2)
                          << delta << ")";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    
    // ========================================================================
    // Validation
    // ========================================================================
    
    std::cout << std::string(70, '-') << "\n";
    std::cout << "Validation\n";
    std::cout << std::string(70, '-') << "\n\n";
    
    // For H2/STO-3G @ 0.74 Å:
    // Reference FCI energy ≈ -1.137 Ha (from literature)
    double e_ref_fci = -1.137;  // Approximate reference
    double error = std::abs(result.e_casscf - e_ref_fci);
    
    std::cout << "Reference comparison:\n";
    std::cout << "  E(reference) ≈ " << e_ref_fci << " Ha (FCI from literature)\n";
    std::cout << "  E(CASSCF)    = " << std::fixed << std::setprecision(10)
              << result.e_casscf << " Ha\n";
    std::cout << "  Error        = " << std::scientific << std::setprecision(4)
              << error << " Ha\n\n";
    
    // NOTE: For H2/STO-3G, CAS(2,2) = FCI (exact within basis)
    std::cout << "Note: CAS(2,2) for H2/STO-3G is equivalent to FCI\n";
    std::cout << "      (all electrons in all orbitals = full configuration space)\n\n";
    
    // Check if correlation energy is reasonable
    double e_corr = result.e_casscf - hf_result.energy_total;
    if (e_corr < 0) {
        std::cout << "✓ Correlation energy is negative (correct sign)\n";
    } else {
        std::cout << "✗ WARNING: Correlation energy should be negative!\n";
    }
    
    // Check if energy is lower than HF
    if (result.e_casscf < hf_result.energy_total) {
        std::cout << "✓ CASSCF energy lower than HF (variational principle)\n";
    } else {
        std::cout << "✗ WARNING: CASSCF energy should be lower than HF!\n";
    }
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Test completed\n";
    std::cout << std::string(70, '=') << "\n";
    
    return result.converged ? 0 : 1;
}
