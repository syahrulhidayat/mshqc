/**
 * @file rmp3_test.cc
 * @brief Test RMP3 (Restricted MP3) on H2O molecule
 * 
 * Demonstrates complete RHF → RMP2 → RMP3 workflow for closed-shell systems.
 * Tests third-order Møller-Plesset perturbation theory.
 * 
 * Expected results (H2O/cc-pVDZ):
 *   RHF:  ~ -76.027 Ha
 *   MP2:  ~ -0.204 Ha (correlation)
 *   MP3:  ~ -0.006 Ha (3rd-order correction)
 *   Total: ~ -76.237 Ha
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 * @license MIT
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/foundation/rmp2.h"
#include "mshqc/foundation/rmp3.h"
#include "mshqc/foundation/wavefunction.h"
#include <iostream>
#include <iomanip>
#include <memory>

using namespace mshqc;

int main() {
    std::cout << "====================================\n";
    std::cout << "  RMP3 Test: H2O/cc-pVDZ\n";
    std::cout << "====================================\n";
    
    // 1. Set up H2O molecule
    Molecule h2o;
    h2o.add_atom(8, 0.000000,  0.000000,  0.117176);  // O
    h2o.add_atom(1, 0.000000,  0.755453, -0.468706);  // H
    h2o.add_atom(1, 0.000000, -0.755453, -0.468706);  // H
    
    std::cout << "\nMolecule: H2O (10 electrons, closed-shell)\n";
    std::cout << "Geometry (Angstrom):\n";
    std::cout << "  O   0.000000   0.000000   0.117176\n";
    std::cout << "  H   0.000000   0.755453  -0.468706\n";
    std::cout << "  H   0.000000  -0.755453  -0.468706\n";
    
    // 2. Basis set
    BasisSet basis("cc-pVDZ", h2o);
    std::cout << "\nBasis: cc-pVDZ (" << basis.n_basis_functions() << " functions)\n";
    
    // 3. Integrals
    auto integrals = std::make_shared<IntegralEngine>(h2o, basis);
    
    // ========================================================================
    // Step 1: RHF
    // ========================================================================
    std::cout << "\n====================================\n";
    std::cout << "  Step 1: Restricted Hartree-Fock\n";
    std::cout << "====================================\n";
    
    SCFConfig config;
    config.max_iterations = 50;
    config.energy_threshold = 1e-8;
    config.density_threshold = 1e-6;
    config.print_level = 1;
    
    RHF rhf(h2o, basis, integrals, config);
    auto rhf_result = rhf.compute();
    
    std::cout << "\nRHF energy: " << std::fixed << std::setprecision(10) 
              << rhf_result.energy_total << " Ha\n";
    
    // ========================================================================
    // Step 2: RMP2
    // ========================================================================
    std::cout << "\n====================================\n";
    std::cout << "  Step 2: Restricted MP2\n";
    std::cout << "====================================\n";
    
    foundation::RMP2 rmp2(rhf_result, basis, integrals);
    auto rmp2_result = rmp2.compute();
    
    std::cout << "\nRMP2 Results:\n";
    std::cout << "  RHF energy:       " << std::setprecision(10) << rmp2_result.e_rhf << " Ha\n";
    std::cout << "  MP2 correlation:  " << rmp2_result.e_corr << " Ha\n";
    std::cout << "  RMP2 total:       " << rmp2_result.e_total << " Ha\n";
    
    // ========================================================================
    // Step 3: RMP3
    // ========================================================================
    std::cout << "\n====================================\n";
    std::cout << "  Step 3: Restricted MP3\n";
    std::cout << "====================================\n";
    
    foundation::RMP3 rmp3(rhf_result, rmp2_result, basis, integrals);
    auto rmp3_result = rmp3.compute();
    
    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "\n====================================\n";
    std::cout << "  FINAL SUMMARY\n";
    std::cout << "====================================\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "RHF energy:           " << rmp3_result.e_rhf << " Ha\n";
    std::cout << "MP2 correction:       " << rmp3_result.e_mp2 << " Ha\n";
    std::cout << "MP3 correction:       " << rmp3_result.e_mp3 << " Ha\n";
    std::cout << "Total correlation:    " << rmp3_result.e_corr_total << " Ha\n";
    std::cout << "RMP3 total energy:    " << rmp3_result.e_total << " Ha\n";
    
    std::cout << "\nPercentage contributions:\n";
    double mp2_percent = 100.0 * rmp3_result.e_mp2 / rmp3_result.e_corr_total;
    double mp3_percent = 100.0 * rmp3_result.e_mp3 / rmp3_result.e_corr_total;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  MP2: " << mp2_percent << "%\n";
    std::cout << "  MP3: " << mp3_percent << "%\n";
    
    // ========================================================================
    // Wavefunction analysis
    // ========================================================================
    std::cout << "\n====================================\n";
    std::cout << "  Wavefunction Analysis\n";
    std::cout << "====================================\n";
    
    // Create wavefunction container with both orders
    foundation::Wavefunction wfn(rmp3_result.n_occ, rmp3_result.n_occ,
                                  rmp3_result.n_virt, rmp3_result.n_virt);
    
    // Store T2^(1) and T2^(2) (for RMP, α = β so we store same in aa, bb, ab)
    wfn.set_t2_order_1(rmp3_result.t2_1, rmp3_result.t2_1, rmp3_result.t2_1);
    wfn.set_t2_order_2(rmp3_result.t2_2, rmp3_result.t2_2, rmp3_result.t2_2);
    
    std::cout << "\nStored amplitudes in wavefunction container:\n";
    std::cout << "  ✓ T2^(1) (first-order, from MP2)\n";
    std::cout << "  ✓ T2^(2) (second-order correction, from MP3)\n";
    
    // Print dominant excitations
    auto dominant_1st = wfn.dominant_amplitudes(0.05, 10);
    std::cout << "\nDominant first-order excitations (|t| > 0.05):\n";
    for (const auto& exc : dominant_1st) {
        std::cout << "  " << exc.to_string() << "\n";
    }
    
    std::cout << "\n====================================\n";
    std::cout << "  Test Complete!\n";
    std::cout << "====================================\n";
    
    return 0;
}
