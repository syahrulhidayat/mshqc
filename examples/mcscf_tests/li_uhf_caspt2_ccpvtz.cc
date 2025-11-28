/**
 * @file li_uhf_caspt2_ccpvtz.cc
 * @brief Lithium ground state with UHF → CASSCF → CASPT2 using cc-pVTZ basis
 * 
 * Complete test: UHF for open-shell Li, then CASSCF with CAS(3,5), finally CASPT2
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
    std::cout << "Li Ground State: UHF → CASSCF → CASPT2 (cc-pVTZ)\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // ========================================================================
    // 1. Setup Li atom
    // ========================================================================
    std::cout << "Setting up Lithium atom...\n";
    
    Molecule mol;
    mol.add_atom(3, 0.0, 0.0, 0.0);  // Li (Z=3) at origin
    
    int n_elec = mol.n_electrons();  // 3 electrons
    int n_alpha = 2;  // 1s² 2s¹ → 2 α, 1 β
    int n_beta = 1;
    
    std::cout << "  Atom: Li (Z=3)\n";
    std::cout << "  Electrons: " << n_elec << " (α=" << n_alpha 
              << ", β=" << n_beta << ")\n";
    std::cout << "  Ground state: ²S (doublet)\n\n";
    
    // ========================================================================
    // 2. Setup cc-pVTZ basis
    // ========================================================================
    std::cout << "Setting up cc-pVTZ basis...\n";
    
    BasisSet basis("cc-pvtz", mol);
    int nbf = basis.n_basis_functions();
    
    std::cout << "  Basis: cc-pVTZ\n";
    std::cout << "  Basis functions: " << nbf << "\n\n";
    
    // ========================================================================
    // 3. Compute integrals
    // ========================================================================
    std::cout << "Computing integrals...\n";
    
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    
    std::cout << "  Integrals computed\n\n";
    
    // ========================================================================
    // 4. Run UHF (for open-shell Li)
    // ========================================================================
    std::cout << "Running UHF (Unrestricted Hartree-Fock)...\n";
    
    SCFConfig config;
    config.max_iterations = 100;
    config.energy_threshold = 1e-8;
    config.density_threshold = 1e-6;
    
    UHF uhf(mol, basis, integrals, n_alpha, n_beta, config);
    
    auto uhf_result = uhf.compute();
    
    if (!uhf_result.converged) {
        std::cerr << "ERROR: UHF did not converge!\n";
        return 1;
    }
    
    std::cout << "\nUHF Results:\n";
    std::cout << "  E(UHF)     = " << std::fixed << std::setprecision(10) 
              << uhf_result.energy_total << " Ha\n";
    std::cout << "  Iterations = " << uhf_result.iterations << "\n";
    std::cout << "  Converged  = " << (uhf_result.converged ? "Yes" : "No") << "\n\n";
    
    // Compute spin contamination
    double s2 = uhf.compute_s_squared(uhf_result);
    double s2_exact = 0.75;  // S(S+1) for doublet: 0.5 * 1.5 = 0.75
    std::cout << "  ⟨S²⟩       = " << std::setprecision(6) << s2 << "\n";
    std::cout << "  ⟨S²⟩_exact = " << s2_exact << " (doublet)\n";
    std::cout << "  Spin contamination = " << std::setprecision(4) 
              << (s2 - s2_exact) << "\n\n";
    
    // ========================================================================
    // 5. Define active space: CAS(3,5)
    // ========================================================================
    std::cout << "Defining active space CAS(3,5)...\n";
    
    // Li: 1s² 2s¹ configuration
    // CAS(3,5): ALL 3 electrons in 5 orbitals (1s, 2s, 2p)
    // This captures full correlation including essential 1s-2s correlation
    
    ActiveSpace active_space = ActiveSpace::CAS(
        3,     // ALL 3 electrons in active space
        5,     // 5 orbitals in active space (1s, 2s, 2px, 2py, 2pz)
        nbf,   // Total number of orbitals
        n_elec // Total electrons (3)
    );
    
    std::cout << "  " << active_space.to_string() << "\n";
    std::cout << "  Inactive (core): " << active_space.n_inactive() 
              << " orbitals (none - full valence)\n";
    std::cout << "  Active:          " << active_space.n_active() 
              << " orbitals (1s, 2s, 2p)\n";
    std::cout << "  Virtual:         " << active_space.n_virtual() 
              << " orbitals\n\n";
    
    std::cout << "Expected determinants: ~50 (doublet with full correlation)\n\n";
    
    // ========================================================================
    // 6. Run CASSCF
    // ========================================================================
    std::cout << "Running CASSCF...\n\n";
    
    CASSCF casscf(mol, basis, integrals, active_space);
    casscf.set_max_iterations(50);
    casscf.set_energy_threshold(1e-8);
    casscf.set_gradient_threshold(1e-6);
    
    auto casscf_result = casscf.compute(uhf_result);
    
    if (!casscf_result.converged) {
        std::cerr << "ERROR: CASSCF did not converge!\n";
        return 1;
    }
    
    std::cout << "\nCASSCF Results:\n";
    std::cout << "  E(CASSCF)  = " << std::fixed << std::setprecision(10) 
              << casscf_result.e_casscf << " Ha\n";
    std::cout << "  Iterations = " << casscf_result.n_iterations << "\n";
    std::cout << "  Determinants = " << casscf_result.n_determinants << "\n";
    std::cout << "  Converged  = " << (casscf_result.converged ? "Yes" : "No") << "\n\n";
    
    double e_corr_casscf = casscf_result.e_casscf - uhf_result.energy_total;
    std::cout << "  Correlation (CASSCF-UHF) = " << std::scientific 
              << std::setprecision(6) << e_corr_casscf << " Ha\n\n";
    
    // ========================================================================
    // 7. Run CASPT2
    // ========================================================================
    std::cout << "Running CASPT2...\n\n";
    
    CASPT2 caspt2(mol, basis, integrals, casscf_result);
    
    // Optional: set IPEA shift
    // caspt2.set_ipea_shift(0.25);
    
    auto caspt2_result = caspt2.compute();
    
    // ========================================================================
    // 8. Final Results
    // ========================================================================
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Li Ground State - Final Results (cc-pVTZ)\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    std::cout << "Energy Breakdown:\n";
    std::cout << "  E(UHF)     = " << std::fixed << std::setprecision(10) 
              << uhf_result.energy_total << " Ha\n";
    std::cout << "  E(CASSCF)  = " << casscf_result.e_casscf << " Ha\n";
    std::cout << "  E(PT2)     = " << caspt2_result.e_pt2 << " Ha\n";
    std::cout << "  E(CASPT2)  = " << caspt2_result.e_total << " Ha\n\n";
    
    std::cout << "Correlation Energy:\n";
    std::cout << "  ΔE(CASSCF-UHF)  = " << std::scientific << std::setprecision(6)
              << e_corr_casscf << " Ha\n";
    std::cout << "  ΔE(CASPT2-UHF)  = " 
              << (caspt2_result.e_total - uhf_result.energy_total) << " Ha\n";
    std::cout << "  E(PT2)          = " << caspt2_result.e_pt2 << " Ha\n\n";
    
    // ========================================================================
    // 9. Validation
    // ========================================================================
    std::cout << "Validation:\n";
    
    bool valid = true;
    
    if (uhf_result.converged) {
        std::cout << "  ✓ UHF converged\n";
    } else {
        std::cout << "  ✗ UHF did not converge\n";
        valid = false;
    }
    
    if (casscf_result.converged) {
        std::cout << "  ✓ CASSCF converged\n";
    } else {
        std::cout << "  ✗ CASSCF did not converge\n";
        valid = false;
    }
    
    if (casscf_result.e_casscf <= uhf_result.energy_total) {
        std::cout << "  ✓ E(CASSCF) <= E(UHF) (variational)\n";
    } else {
        std::cout << "  ⚠ E(CASSCF) > E(UHF) (unusual)\n";
    }
    
    if (caspt2_result.converged) {
        std::cout << "  ✓ CASPT2 converged\n";
    } else {
        std::cout << "  ✗ CASPT2 did not converge\n";
        valid = false;
    }
    
    if (casscf_result.determinants.size() > 0) {
        std::cout << "  ✓ Determinants stored (" 
                  << casscf_result.determinants.size() << ")\n";
    }
    
    // Expected: Li ground state energy ~ -7.43 Ha (cc-pVTZ)
    double e_ref_approx = -7.43;
    std::cout << "\n  Reference E(Li) ~ " << e_ref_approx 
              << " Ha (literature, cc-pVTZ)\n";
    std::cout << "  Our E(CASPT2)   = " << std::fixed << std::setprecision(6)
              << caspt2_result.e_total << " Ha\n";
    
    std::cout << "\nStatus: " << caspt2_result.status_message << "\n";
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    if (valid) {
        std::cout << "TEST COMPLETED ✓\n";
    } else {
        std::cout << "TEST HAD ISSUES\n";
    }
    std::cout << std::string(70, '=') << "\n";
    
    return valid ? 0 : 1;
}
