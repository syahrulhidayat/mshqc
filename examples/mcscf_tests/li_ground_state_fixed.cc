/**
 * @file li_ground_state_fixed.cc
 * @brief Lithium ground state CASSCF/CASPT2 - FIXED VERSION
 * 
 * Li ground state: 1s² 2s¹ (²S)
 * Strategy: Use ROHF → CASSCF(3,5) → CASPT2
 * 
 * Active space: All 3 electrons in 5 orbitals (1s, 2s, 2px, 2py, 2pz)
 * This captures full correlation for Li
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
    std::cout << std::string(80, '=') << "\n";
    std::cout << "Li Ground State: ROHF → CASSCF(3,5) → CASPT2\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    // ========================================================================
    // 1. Setup Li atom with smaller basis (STO-3G for testing)
    // ========================================================================
    std::cout << "Setting up Lithium atom...\n";
    
    Molecule mol;
    mol.add_atom(3, 0.0, 0.0, 0.0);  // Li (Z=3) at origin
    
    int n_elec = mol.n_electrons();  // 3 electrons
    
    std::cout << "  Atom: Li (Z=3)\n";
    std::cout << "  Electrons: " << n_elec << "\n";
    std::cout << "  Ground state: ²S (doublet, 1s² 2s¹)\n\n";
    
    // ========================================================================
    // 2. Setup basis (start with STO-3G for validation)
    // ========================================================================
    std::cout << "Setting up basis...\n";
    
    BasisSet basis("sto-3g", mol);
    int nbf = basis.n_basis_functions();
    
    std::cout << "  Basis: STO-3G\n";
    std::cout << "  Basis functions: " << nbf << "\n\n";
    
    // ========================================================================
    // 3. Compute integrals
    // ========================================================================
    std::cout << "Computing integrals...\n";
    
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    
    std::cout << "  Integrals computed\n\n";
    
    // ========================================================================
    // 4. Run ROHF (Restricted Open-shell HF)
    // ========================================================================
    std::cout << "Running ROHF (Restricted Open-shell Hartree-Fock)...\n";
    
    int n_alpha = 2;  // 1s¹ 2s¹ (both α for high-spin reference)
    int n_beta = 1;   // 1s¹ (one β)
    
    SCFConfig config;
    config.max_iterations = 100;
    config.energy_threshold = 1e-8;
    config.density_threshold = 1e-6;
    
    // Use ROHF class
    ROHF rohf(mol, basis, integrals, config);
    
    auto rohf_result = rohf.compute();
    
    if (!rohf_result.converged) {
        std::cerr << "ERROR: ROHF did not converge!\n";
        std::cerr << "  This is expected - ROHF implementation may need debugging\n";
        std::cerr << "  Falling back to UHF...\n\n";
        
        // Fallback to UHF
        UHF uhf(mol, basis, integrals, n_alpha, n_beta, config);
        auto uhf_result = uhf.compute();
        
        if (!uhf_result.converged) {
            std::cerr << "ERROR: UHF also failed!\n";
            return 1;
        }
        
        std::cout << "\nUHF Results (fallback):\n";
        std::cout << "  E(UHF)     = " << std::fixed << std::setprecision(10) 
                  << uhf_result.energy_total << " Ha\n";
        std::cout << "  Iterations = " << uhf_result.iterations << "\n\n";
        
        // Use UHF result for CASSCF
        rohf_result = uhf_result;
    } else {
        std::cout << "\nROHF Results:\n";
        std::cout << "  E(ROHF)    = " << std::fixed << std::setprecision(10) 
                  << rohf_result.energy_total << " Ha\n";
        std::cout << "  Iterations = " << rohf_result.iterations << "\n\n";
    }
    
    // ========================================================================
    // 5. Define active space: CAS(3,5)
    // ========================================================================
    std::cout << "Defining active space CAS(3,5)...\n";
    
    // Li: 1s² 2s¹ configuration
    // CAS(3,5): ALL 3 electrons in 5 orbitals (1s, 2s, 2px, 2py, 2pz)
    // This is "full CI" for valence space
    
    ActiveSpace active_space = ActiveSpace::CAS(
        3,     // 3 electrons in active space (ALL electrons)
        5,     // 5 orbitals in active space (1s, 2s, 2p)
        nbf,   // Total number of orbitals
        n_elec // Total electrons (3)
    );
    
    std::cout << "  " << active_space.to_string() << "\n";
    std::cout << "  Inactive (core): " << active_space.n_inactive() 
              << " orbitals\n";
    std::cout << "  Active:          " << active_space.n_active() 
              << " orbitals (1s, 2s, 2p)\n";
    std::cout << "  Virtual:         " << active_space.n_virtual() 
              << " orbitals\n\n";
    
    // For 3 electrons in 5 orbitals, expect:
    // - High-spin: n_alpha=2, n_beta=1
    // - Number of dets: C(5,2) × C(5,1) = 10 × 5 = 50 determinants
    std::cout << "Expected determinants: ~50 (doublet state)\n\n";
    
    // ========================================================================
    // 6. Run CASSCF
    // ========================================================================
    std::cout << "Running CASSCF(3,5)...\n\n";
    
    CASSCF casscf(mol, basis, integrals, active_space);
    casscf.set_max_iterations(50);
    casscf.set_energy_threshold(1e-8);
    casscf.set_gradient_threshold(1e-6);
    
    auto casscf_result = casscf.compute(rohf_result);
    
    std::cout << "\nCASSCF Results:\n";
    std::cout << "  E(CASSCF)  = " << std::fixed << std::setprecision(10) 
              << casscf_result.e_casscf << " Ha\n";
    std::cout << "  Iterations = " << casscf_result.n_iterations << "\n";
    std::cout << "  Determinants = " << casscf_result.n_determinants << "\n";
    std::cout << "  Converged  = " << (casscf_result.converged ? "Yes" : "No") << "\n\n";
    
    if (!casscf_result.converged) {
        std::cerr << "WARNING: CASSCF did not converge!\n";
        std::cerr << "  This may indicate:\n";
        std::cerr << "    - Orbital optimization needs damping\n";
        std::cerr << "    - Step size too large\n";
        std::cerr << "    - Need better initial guess\n\n";
    }
    
    double e_corr_casscf = casscf_result.e_casscf - rohf_result.energy_total;
    std::cout << "  Correlation (CASSCF-HF) = " << std::scientific 
              << std::setprecision(6) << e_corr_casscf << " Ha\n\n";
    
    // ========================================================================
    // 7. Run CASPT2 (if CASSCF converged)
    // ========================================================================
    if (casscf_result.converged) {
        std::cout << "Running CASPT2...\n\n";
        
        CASPT2 caspt2(mol, basis, integrals, casscf_result);
        
        auto caspt2_result = caspt2.compute();
        
        // ====================================================================
        // 8. Final Results
        // ====================================================================
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "Li Ground State - Final Results (STO-3G)\n";
        std::cout << std::string(80, '=') << "\n\n";
        
        std::cout << "Energy Breakdown:\n";
        std::cout << "  E(HF)      = " << std::fixed << std::setprecision(10) 
                  << rohf_result.energy_total << " Ha\n";
        std::cout << "  E(CASSCF)  = " << casscf_result.e_casscf << " Ha\n";
        std::cout << "  E(PT2)     = " << caspt2_result.e_pt2 << " Ha\n";
        std::cout << "  E(CASPT2)  = " << caspt2_result.e_total << " Ha\n\n";
        
        std::cout << "Correlation Energy:\n";
        std::cout << "  ΔE(CASSCF-HF)   = " << std::scientific << std::setprecision(6)
                  << e_corr_casscf << " Ha\n";
        std::cout << "  ΔE(CASPT2-HF)   = " 
                  << (caspt2_result.e_total - rohf_result.energy_total) << " Ha\n";
        std::cout << "  E(PT2)          = " << caspt2_result.e_pt2 << " Ha\n\n";
        
        // ====================================================================
        // 9. Validation
        // ====================================================================
        std::cout << "Validation:\n";
        
        bool valid = true;
        
        if (rohf_result.converged) {
            std::cout << "  ✓ HF converged\n";
        } else {
            std::cout << "  ✗ HF did not converge\n";
            valid = false;
        }
        
        if (casscf_result.converged) {
            std::cout << "  ✓ CASSCF converged\n";
        } else {
            std::cout << "  ✗ CASSCF did not converge\n";
            valid = false;
        }
        
        if (casscf_result.e_casscf <= rohf_result.energy_total) {
            std::cout << "  ✓ E(CASSCF) <= E(HF) (variational principle)\n";
        } else {
            std::cout << "  ⚠ E(CASSCF) > E(HF) (violates variational principle!)\n";
        }
        
        if (caspt2_result.converged) {
            std::cout << "  ✓ CASPT2 completed\n";
        } else {
            std::cout << "  ✗ CASPT2 failed\n";
            valid = false;
        }
        
        if (casscf_result.determinants.size() > 0) {
            std::cout << "  ✓ Determinants stored (" 
                      << casscf_result.determinants.size() << ")\n";
        }
        
        // Expected: Li ground state energy ~ -7.43 Ha (exact)
        // With STO-3G: expect ~ -7.31 Ha
        double e_ref_sto3g = -7.31;
        std::cout << "\n  Reference E(Li, STO-3G) ~ " << e_ref_sto3g 
                  << " Ha (literature)\n";
        std::cout << "  Our E(CASPT2)           = " << std::fixed << std::setprecision(6)
                  << caspt2_result.e_total << " Ha\n";
        
        std::cout << "\nStatus: " << caspt2_result.status_message << "\n";
        
        std::cout << "\n" << std::string(80, '=') << "\n";
        if (valid) {
            std::cout << "TEST COMPLETED ✓\n";
        } else {
            std::cout << "TEST HAD ISSUES - See warnings above\n";
        }
        std::cout << std::string(80, '=') << "\n";
        
        return valid ? 0 : 1;
    } else {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "CASSCF FAILED TO CONVERGE - Cannot proceed to CASPT2\n";
        std::cout << std::string(80, '=') << "\n";
        return 1;
    }
}
