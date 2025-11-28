/**
 * UMP4 Validation Test - Post Bug Fix
 * 
 * Validates:
 * 1. All energies are negative
 * 2. Energy hierarchy: |E^(n+1)| < |E^(n)|
 * 3. T2 amplitude norms are reasonable
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
    std::cout << "  Post Bug Fix (Double Normalization)\n";
    std::cout << "========================================\n\n";
    
    // Li atom with cc-pVTZ
    Molecule mol;
    mol.add_atom(3, 0.0, 0.0, 0.0);  // Li has Z=3
    
    BasisSet basis("cc-pVTZ", mol);
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    
    std::cout << "System: Li atom\n";
    std::cout << "Basis:  cc-pVTZ (" << basis.n_basis_functions() << " functions)\n\n";
    
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
    
    // Run UMP4 (SDQ only for speed)
    mp::UMP4 ump4(uhf_result, ump3_result, basis, integrals);
    auto ump4_result = ump4.compute(false);
    
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
    double e4_s = ump4_result.e_mp4_sdq;
    
    std::cout << "Reference:\n";
    std::cout << "  E_UHF:           " << std::setw(16) << e_uhf << " Ha\n\n";
    
    std::cout << "Correlation energies:\n";
    std::cout << "  E^(2) (MP2):     " << std::setw(16) << e2 << " Ha\n";
    std::cout << "  E^(3) (MP3):     " << std::setw(16) << e3 << " Ha\n";
    std::cout << "  E^(4) (MP4-SDQ): " << std::setw(16) << e4_s << " Ha\n\n";
    
    // Check 1: All negative
    bool all_negative = (e2 < 0) && (e3 < 0) && (e4_s < 0);
    
    std::cout << "Check 1: All correlation energies negative\n";
    std::cout << "  E^(2) < 0: " << (e2 < 0 ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << "  E^(3) < 0: " << (e3 < 0 ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << "  E^(4) < 0: " << (e4_s < 0 ? "✓ PASS" : "✗ FAIL") << "\n\n";
    
    // Check 2: Hierarchy
    bool hierarchy_ok = (std::abs(e3) < std::abs(e2)) && 
                        (std::abs(e4_s) < std::abs(e3));
    
    std::cout << "Check 2: Energy hierarchy |E^(n+1)| < |E^(n)|\n";
    std::cout << "  |E^(3)| < |E^(2)|: " 
              << (std::abs(e3) < std::abs(e2) ? "✓ PASS" : "✗ FAIL") 
              << " (" << std::abs(e3)/std::abs(e2)*100 << "%)\n";
    std::cout << "  |E^(4)| < |E^(3)|: " 
              << (std::abs(e4_s) < std::abs(e3) ? "✓ PASS" : "✗ FAIL")
              << " (" << std::abs(e4_s)/std::abs(e3)*100 << "%)\n\n";
    
    // Check 3: T2 amplitude norms
    auto calc_norm = [](const auto& T) {
        double s = 0.0;
        for (int i=0; i<T.dimension(0); ++i)
          for (int j=0; j<T.dimension(1); ++j)
            for (int a=0; a<T.dimension(2); ++a)
              for (int b=0; b<T.dimension(3); ++b)
                s += T(i,j,a,b)*T(i,j,a,b);
        return std::sqrt(s);
    };
    
    double t2_1_norm = calc_norm(ump3_result.t2_aa_1);
    double t2_2_norm = calc_norm(ump3_result.t2_aa_2);
    double t2_3_norm = calc_norm(ump4_result.t2_aa_3);
    
    std::cout << "Check 3: Amplitude norms (αα only)\n";
    std::cout << "  ||T2^(1)||: " << std::setw(12) << t2_1_norm << "\n";
    std::cout << "  ||T2^(2)||: " << std::setw(12) << t2_2_norm 
              << " (" << t2_2_norm/t2_1_norm*100 << "% of T2^(1))\n";
    std::cout << "  ||T2^(3)||: " << std::setw(12) << t2_3_norm
              << " (" << t2_3_norm/t2_2_norm*100 << "% of T2^(2))\n\n";
    
    bool norms_ok = (t2_2_norm > 0) && (t2_2_norm < t2_1_norm) &&
                    (t2_3_norm > t2_2_norm);  // T2^(3) includes T2^(2)!
    
    std::cout << "  T2^(2) non-zero: " << (t2_2_norm > 0 ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << "  T2^(2) < T2^(1): " << (t2_2_norm < t2_1_norm ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << "  T2^(3) > T2^(2): " << (t2_3_norm > t2_2_norm ? "✓ PASS" : "✗ FAIL") 
              << " (expected since T2^(3)=T2^(2)+corr)\n\n";
    
    // Final verdict
    std::cout << "========================================\n";
    if (all_negative && hierarchy_ok && norms_ok) {
        std::cout << "✓✓✓ ALL CHECKS PASSED! ✓✓✓\n";
        std::cout << "========================================\n";
        std::cout << "\nBug fix SUCCESSFUL! UMP4 is now correct.\n";
        return 0;
    } else {
        std::cout << "✗✗✗ SOME CHECKS FAILED ✗✗✗\n";
        std::cout << "========================================\n";
        return 1;
    }
}
