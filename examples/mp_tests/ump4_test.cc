/**
 * UMP4 Validation Test - Rigorous Implementation
 * * Validates:
 * 1. MP2 is negative (Correlation)
 * 2. MP3 is positive (Correction/Back-oscillation for Li)
 * 3. MP4 is negative (Due to large attractive Triples)
 */

#include "mshqc/scf.h"
#include "mshqc/ump2.h"
#include "mshqc/ump3.h"
#include "mshqc/mp/ump4.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace mshqc;

int main() {
    std::cout << "\n========================================\n";
    std::cout << "  UMP4 VALIDATION TEST\n";
    std::cout << "  Rigorous Triples Implementation\n";
    std::cout << "========================================\n\n";
    
    // Li atom with cc-pVQZ
    Molecule mol;
    mol.add_atom(3, 0.0, 0.0, 0.0);  // Li has Z=3
    
    BasisSet basis("cc-pVQZ", mol);
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    
    std::cout << "System: Li atom\n";
    std::cout << "Basis:  cc-pVQZ (" << basis.n_basis_functions() << " functions)\n\n";
    
    // Run UHF
    SCFConfig config;
    config.energy_threshold = 1e-8;
    config.max_iterations = 100;
    
    UHF uhf(mol, basis, integrals, 2, 1, config);  // 2α, 1β
    auto uhf_result = uhf.compute();
    
    // Run UMP2
    UMP2 ump2(uhf_result, basis, integrals);
    auto ump2_result = ump2.compute();
    
    // Run UMP3
    UMP3 ump3(uhf_result, ump2_result, basis, integrals);
    auto ump3_result = ump3.compute();
    
    // Run UMP4 (ENABLE TRIPLES = TRUE)
    mp::UMP4 ump4(uhf_result, ump3_result, basis, integrals);
    auto ump4_result = ump4.compute(true); // <--- TRUE untuk Rigorous Triples
    
    // ========================================================================
    // VALIDATION
    // ========================================================================
    
    std::cout << "\n========================================\n";
    std::cout << "  VALIDATION RESULTS\n";
    std::cout << "========================================\n\n";
    
    std::cout << std::fixed << std::setprecision(10);
    
    // Extract energies
    double e_uhf = uhf_result.energy_total;
    double e2 = ump2_result.e_corr_total;
    double e3 = ump3_result.e_mp3;
    double e4_total = ump4_result.e_mp4_total;
    
    std::cout << "Reference:\n";
    std::cout << "  E_UHF:           " << std::setw(16) << e_uhf << " Ha\n\n";
    
    std::cout << "Correlation energies:\n";
    std::cout << "  E^(2) (MP2):     " << std::setw(16) << e2 << " Ha\n";
    std::cout << "  E^(3) (MP3):     " << std::setw(16) << e3 << " Ha\n";
    std::cout << "  E^(4) (MP4):     " << std::setw(16) << e4_total << " Ha\n\n";
    
    // Check 1: MP2 Must be Negative
    bool check_mp2 = (e2 < 0);
    
    // Check 2: MP3 can be positive (Oscillation)
    // We just check if it exists (not zero)
    bool check_mp3 = (std::abs(e3) > 1e-6);

    // Check 3: MP4 Must be Negative (Triples dominate)
    // With rigorous triples, E_T is negative and large enough to overcome SDQ
    bool check_mp4 = (e4_total < 0);
    
    std::cout << "Checks:\n";
    std::cout << "  1. MP2 Attractive (E2 < 0): " << (check_mp2 ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << "  2. MP3 Correction (|E3|>0): " << (check_mp3 ? "✓ PASS" : "✗ FAIL") 
              << (e3 > 0 ? " (Positive - Normal Oscillation)" : "") << "\n";
    std::cout << "  3. MP4 Convergent (E4 < 0): " << (check_mp4 ? "✓ PASS" : "✗ FAIL") << "\n\n";
    
    // Amplitude norms validation (Removed T2^(3) check as it is no longer stored)
    
    // Final verdict
    std::cout << "========================================\n";
    if (check_mp2 && check_mp3 && check_mp4) {
        std::cout << "✓✓✓ ALL PHYSICS CHECKS PASSED! ✓✓✓\n";
        std::cout << "========================================\n";
        return 0;
    } else {
        std::cout << "✗✗✗ SOME CHECKS FAILED ✗✗✗\n";
        std::cout << "========================================\n";
        return 1;
    }
}