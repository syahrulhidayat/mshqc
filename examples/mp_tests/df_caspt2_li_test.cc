/**
 * @file df_caspt2_li_test.cc
 * @brief Test DF-CASPT2 on Li atom vs standard CASPT2
 * 
 * System: Li atom (3 electrons, ²S state)
 * Basis: cc-pVDZ (14 basis functions)
 * Active space: CAS(3e,5o)
 * Goal: Validate |E_DF - E_exact| < 10 µHa
 * 
 * REFERENCES:
 *   - M. Feyereisen et al., Chem. Phys. Lett. **208**, 359 (1993)
 *   - K. Andersson et al., J. Chem. Phys. **96**, 1218 (1992)
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-16
 * @license MIT License
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/mcscf/casscf.h"
#include "mshqc/mcscf/caspt2.h"
#include "mshqc/mcscf/df_caspt2.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace mshqc;

int main() {
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "DF-CASPT2 Test: Li Atom (cc-pVDZ)\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "System: Li (3e, ²S), CAS(3e,5o)\n";
    std::cout << "Basis: cc-pVDZ (14 functions)\n";
    std::cout << "Goal: |E_DF - E_exact| < 10 µHa\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    try {
        // ====================================================================
        // Step 1: Setup molecule and basis
        // ====================================================================
        
        Molecule mol;
        // Li atom (Z=3), doublet (2S+1 = 2)
        mol.add_atom(3, 0.0, 0.0, 0.0);
        mol.set_multiplicity(2);
        
        std::cout << "Loading basis sets...\n";
        BasisSet basis("cc-pvdz", mol, "data/basis");
        std::cout << "  Primary basis: cc-pVDZ (" << basis.n_basis_functions() << " functions)\n";
        
        // Load auxiliary basis for DF
        BasisSet aux_basis("cc-pvdz-ri", mol, "data/basis");
        std::cout << "  Auxiliary basis: cc-pVDZ-RI (" << aux_basis.n_basis_functions() << " functions)\n";
        std::cout << "  Ratio (aux/primary): " << std::fixed << std::setprecision(2)
                  << (double)aux_basis.n_basis_functions() / basis.n_basis_functions() << "\n\n";
        
        // Create integral engine
        auto integrals = std::make_shared<IntegralEngine>(mol, basis);
        
        // ====================================================================
        // Step 2: Run UHF (reference for CASSCF)
        // ====================================================================
        
        std::cout << "Running UHF...\n";
        // Li: N_alpha=2, N_beta=1
        SCFConfig scf_cfg;
        scf_cfg.max_iterations = 100;
        scf_cfg.energy_threshold = 1e-8;
        UHF uhf(mol, basis, integrals, /*n_alpha=*/2, /*n_beta=*/1, scf_cfg);
        
        auto uhf_result = uhf.compute();
        
        if (!uhf_result.converged) {
            std::cerr << "ERROR: UHF did not converge!\n";
            return 1;
        }
        
        std::cout << "  E(UHF) = " << std::fixed << std::setprecision(10) 
                  << uhf_result.energy_total << " Ha\n\n";
        
        // ====================================================================
        // Step 3: Run CASSCF
        // ====================================================================
        
        std::cout << "Running CASSCF(3e,5o)...\n";
        
        // Define CAS(3e,5o); determine total orbitals from UHF result
        int n_total_orb = uhf_result.C_alpha.cols();
        int n_total_elec = mol.n_electrons();
        mcscf::ActiveSpace active_space = mcscf::ActiveSpace::CAS(/*n_elec=*/3, /*n_orb=*/5,
                                                                  n_total_orb, n_total_elec);
        
        mcscf::CASSCF casscf(mol, basis, integrals, active_space);
        casscf.set_max_iterations(50);
        casscf.set_energy_threshold(1e-6);
        
        auto cas_result = casscf.compute(uhf_result);
        
        if (!cas_result.converged) {
            std::cerr << "ERROR: CASSCF did not converge!\n";
            return 1;
        }
        
        std::cout << "  E(CASSCF) = " << std::fixed << std::setprecision(10) 
                  << cas_result.e_casscf << " Ha\n\n";
        
        // ====================================================================
        // Step 4: Run standard CASPT2 (reference)
        // ====================================================================
        
        std::cout << "Running standard CASPT2 (exact)...\n";
        
        mcscf::CASPT2 caspt2_exact(mol, basis, integrals, cas_result);
        auto exact_result = caspt2_exact.compute();
        
        std::cout << "  E(CASPT2 exact) = " << std::fixed << std::setprecision(10) 
                  << exact_result.e_total << " Ha\n";
        std::cout << "  E(PT2 exact)    = " << std::scientific << std::setprecision(6)
                  << exact_result.e_pt2 << " Ha\n\n";
        
        // ====================================================================
        // Step 5: Run DF-CASPT2
        // ====================================================================
        
        std::cout << "Running DF-CASPT2...\n";
        
        auto cas_result_ptr = std::make_shared<mcscf::CASResult>(cas_result);
        mcscf::DFCASPT2 df_caspt2(mol, basis, aux_basis, integrals, cas_result_ptr);
        auto df_result = df_caspt2.compute();
        
        std::cout << "  E(DF-CASPT2) = " << std::fixed << std::setprecision(10) 
                  << df_result.e_total << " Ha\n";
        std::cout << "  E(PT2 DF)    = " << std::scientific << std::setprecision(6)
                  << df_result.e_pt2 << " Ha\n\n";
        
        // ====================================================================
        // Step 6: Compare results
        // ====================================================================
        
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "Comparison: DF-CASPT2 vs Exact CASPT2\n";
        std::cout << std::string(70, '=') << "\n\n";
        
        double error_pt2 = std::abs(df_result.e_pt2 - exact_result.e_pt2);
        double error_total = std::abs(df_result.e_total - exact_result.e_total);
        
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "E(CASPT2 exact):  " << exact_result.e_total << " Ha\n";
        std::cout << "E(DF-CASPT2):     " << df_result.e_total << " Ha\n";
        std::cout << "Difference:       " << std::scientific << std::setprecision(3) 
                  << error_total << " Ha\n";
        std::cout << "                  " << error_total * 1e6 << " µHa\n\n";
        
        std::cout << "E(PT2 exact):     " << std::scientific << std::setprecision(6)
                  << exact_result.e_pt2 << " Ha\n";
        std::cout << "E(PT2 DF):        " << df_result.e_pt2 << " Ha\n";
        std::cout << "Difference:       " << std::setprecision(3)
                  << error_pt2 << " Ha\n";
        std::cout << "                  " << error_pt2 * 1e6 << " µHa\n\n";
        
        // ====================================================================
        // Step 7: Validation
        // ====================================================================
        
        double threshold_uha = 10.0;  // 10 µHa target
        double error_uha = error_total * 1e6;
        
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Validation Results\n";
        std::cout << std::string(70, '=') << "\n\n";
        
        std::cout << "Target accuracy:  < " << threshold_uha << " µHa\n";
        std::cout << "Actual error:     " << std::fixed << std::setprecision(3) 
                  << error_uha << " µHa\n\n";
        
        if (error_uha < threshold_uha) {
            std::cout << "✅ PASS: DF-CASPT2 within target accuracy!\n";
            std::cout << "   Density fitting error: " << error_uha << " µHa < " 
                      << threshold_uha << " µHa\n";
        } else {
            std::cout << "❌ FAIL: DF-CASPT2 exceeds target accuracy\n";
            std::cout << "   Density fitting error: " << error_uha << " µHa > " 
                      << threshold_uha << " µHa\n";
        }
        
        std::cout << "\n" << std::string(70, '=') << "\n\n";
        
        // ====================================================================
        // Summary statistics
        // ====================================================================
        
        std::cout << "Summary Statistics:\n";
        std::cout << "  Auxiliary basis size: " << aux_basis.n_basis_functions() << " functions\n";
        std::cout << "  Auxiliary/Primary ratio: " << std::fixed << std::setprecision(2)
                  << (double)aux_basis.n_basis_functions() / basis.n_basis_functions() << "\n";
        std::cout << "  DF approximation error: " << std::scientific << std::setprecision(3)
                  << error_uha << " µHa\n";
        std::cout << "  Relative error: " << std::fixed << std::setprecision(4)
                  << (error_total / std::abs(exact_result.e_pt2)) * 100 << " %\n";
        
        std::cout << "\n";
        
        return (error_uha < threshold_uha) ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
