/**
 * @file dfmp2_test.cc
 * @brief Test DF-MP2 for Li atom (doublet)
 * 
 * Validates complete DF-MP2 implementation:
 * - ROHF convergen
ce
 * - 3-center integrals (\u03bc\u03bd|P)
 * - 2-center metric (P|Q) and inversion
 * - MO transformation
 * - MP2 energy computation
 * 
 * Expected result: E_corr \u2248 -0.011 Ha (Psi4 reference)
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-01-11
 * @license MIT License
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/scf.h"
#include "mshqc/integrals.h"
#include "mshqc/dfmp2.h"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  DF-MP2 Test: Li atom (doublet)\n";
    std::cout << "========================================\n\n";
    
    // Li atom doublet (Z=3, 3 electrons)
    mshqc::Molecule mol;
    mol.add_atom(3, 0.0, 0.0, 0.0);
    
    std::cout << "Molecule:\n";
    std::cout << "  Li atom (doublet, 2S+1=2)\n";
    std::cout << "  Electrons: 3 total (2 alpha, 1 beta)\n";
    std::cout << "  Occupation: \u03b1\u03b1\u03b2 (1 unpaired)\n\n";
    
    // Primary basis: cc-pV5Z (largest available)
    mshqc::BasisSet basis("cc-pV5Z", mol, "../data/basis");
    std::cout << "Primary basis: cc-pV5Z (" << basis.n_basis_functions() 
              << " functions)\n";
    
    // Auxiliary basis: cc-pV5Z-RI  
    mshqc::BasisSet aux_basis;
    try {
        aux_basis = mshqc::BasisSet("cc-pV5Z-RI", mol, "../data/basis");
        std::cout << "Auxiliary basis: cc-pV5Z-RI (" << aux_basis.n_basis_functions() 
                  << " functions)\n\n";
    } catch (const std::exception& e) {
        std::cout << "Error loading auxiliary basis: " << e.what() << "\n";
        std::cout << "Note: cc-pV5Z-RI must be in data/basis/\n";
        return 1;
    }
    
    // ROHF calculation
    std::cout << "========================================\n";
    std::cout << "  ROHF Calculation\n";
    std::cout << "========================================\n\n";
    
    // Electron config: 3 electrons (1s² 2s¹)
    // For ROHF: 2 doubly occupied + 1 singly occupied
    // So n_alpha = 2 (closed + open), n_beta = 1 (closed only)
    int n_alpha = 2;
    int n_beta = 1;
    
    mshqc::SCFConfig config;
    config.max_iterations = 100;
    config.energy_threshold = 1e-10;
    config.density_threshold = 1e-8;
    
    mshqc::ROHF rohf(mol, basis, n_alpha, n_beta, config);
    auto rohf_result = rohf.run();
    
    if (!rohf_result.converged) {
        std::cout << "ROHF not converged!\n";
        return 1;
    }
    
    std::cout << "\nROHF Energy: " << std::fixed << std::setprecision(10) 
              << rohf_result.energy_total << " Ha\n";
    
    // DF-MP2 calculation
    std::cout << "\n========================================\n";
    std::cout << "  DF-MP2 Calculation\n";
    std::cout << "========================================\n";
    
    // Create integrals engine for DF-MP2
    auto integrals = std::make_shared<mshqc::IntegralEngine>(mol, basis);
    mshqc::DFMP2 dfmp2(rohf_result, basis, aux_basis, integrals);
    auto mp2_result = dfmp2.compute();
    
    // Validation against expected
    std::cout << "\n========================================\n";
    std::cout << "  Validation\n";
    std::cout << "========================================\n\n";
    
    double expected_corr = -0.011;  // Approximate from Psi4
    double error = std::abs(mp2_result.e_corr - expected_corr);
    
    std::cout << "Expected correlation: ~" << expected_corr << " Ha\n";
    std::cout << "Computed correlation: " << std::fixed << std::setprecision(10)
              << mp2_result.e_corr << " Ha\n";
    std::cout << "Absolute error: " << std::scientific << std::setprecision(3)
              << error << " Ha\n\n";
    
    if (error < 0.002) {  // \u00b12 mHa tolerance
        std::cout << "\u2713 DF-MP2 correlation energy: PASS\n";
        std::cout << "\u2713 Within 2 mHa of expected value\n\n";
        return 0;
    } else if (error < 0.010) {
        std::cout << "\u26a0 DF-MP2 within 10 mHa (acceptable)\n\n";
        return 0;
    } else {
        std::cout << "\u2717 DF-MP2 error too large (>10 mHa)\n";
        std::cout << "  Check auxiliary basis or implementation\n\n";
        return 1;
    }
}
