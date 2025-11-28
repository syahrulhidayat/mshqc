/**
 * Test COMPLETE UMP3 (with VVVV integrals) on Li atom / STO-3G
 * 
 * Expected behavior with complete MP3:
 * - E(2) ≈ -0.0034 Ha
 * - E(3) ≈ -0.0003 to -0.0008 Ha (10-25% of E(2)) - CONVERGENT!
 * - |E(3)| < |E(2)| (series converges)
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/ump2.h"
#include "mshqc/ump3.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace mshqc;

int main() {
    std::cout << "\n====================================\n";
    std::cout << "  COMPLETE UMP3 Test: Li / STO-3G\n";
    std::cout << "====================================\n";
    
    // Li atom
    Molecule li;
    li.add_atom(3, 0.0, 0.0, 0.0);
    
    std::cout << "\nMolecule: Li (2\u03b1, 1\u03b2)\n";
    
    // Basis - STO-3G (small, fast VVVV transform)
    BasisSet basis("STO-3G", li);
    std::cout << "Basis: STO-3G (" << basis.n_basis_functions() << " functions)\n";
    std::cout << "  [Small basis for fast VVVV transformation]\n\n";
    
    // Integrals
    auto integrals = std::make_shared<IntegralEngine>(li, basis);
    
    // UHF (quiet)
    SCFConfig config;
    config.max_iterations = 50;
    config.energy_threshold = 1e-8;
    config.density_threshold = 1e-6;
    config.print_level = 0;
    
    int n_alpha = 2;
    int n_beta = 1;
    
    UHF uhf(li, basis, integrals, n_alpha, n_beta, config);
    auto uhf_result = uhf.compute();
    
    std::cout << "UHF energy: " << std::fixed << std::setprecision(8) 
              << uhf_result.energy_total << " Ha\n";
    
    // UMP2
    UMP2 ump2(uhf_result, basis, integrals);
    auto ump2_result = ump2.compute();
    
    // UMP3 - NOW COMPLETE with VVVV!
    UMP3 ump3(uhf_result, ump2_result, basis, integrals);
    auto ump3_result = ump3.compute();
    
    // Validation
    std::cout << "\n====================================\n";
    std::cout << "VALIDATION:\n";
    std::cout << "====================================\n";
    std::cout << std::fixed << std::setprecision(8);
    
    std::cout << "UMP2 correlation: " << ump2_result.e_corr_total << " Ha\n";
    std::cout << "E(3) correction:  " << ump3_result.e_mp3_corr << " Ha\n";
    std::cout << "UMP3 correlation: " << ump3_result.e_corr_total << " Ha\n";
    std::cout << "\nUMP3 total:       " << ump3_result.e_total << " Ha\n";
    
    // Check convergence
    double ratio = std::abs(ump3_result.e_mp3_corr / ump2_result.e_corr_total);
    std::cout << "\nE(3)/E(2) ratio:  " << std::setprecision(2) << ratio*100 << "%\n\n";
    
    bool converged = ratio < 1.0;
    bool negative = ump3_result.e_mp3_corr < 0.0;
    bool monotonic = ump3_result.e_total < ump2_result.e_total;
    
    if (negative) {
        std::cout << "\u2713 E(3) is negative (correct sign)\n";
    } else {
        std::cout << "\u2717 E(3) is positive (ERROR!)\n";
    }
    
    if (converged) {
        std::cout << "\u2713 |E(3)| < |E(2)| (CONVERGENT series!)\n";
    } else {
        std::cout << "\u2717 |E(3)| > |E(2)| (divergent)\n";
    }
    
    if (monotonic) {
        std::cout << "\u2713 E(UMP3) < E(UMP2) (energy monotonicity)\n";
    } else {
        std::cout << "\u2717 E(UMP3) > E(UMP2) (non-monotonic)\n";
    }
    
    std::cout << "\n====================================\n";
    if (negative && converged && monotonic) {
        std::cout << "  COMPLETE UMP3 TEST: PASSED \u2713\n";
        std::cout << "  All three terms (HH + PP + PH) implemented!\n";
        std::cout << "====================================\n\n";
        return 0;
    } else {
        std::cout << "  COMPLETE UMP3 TEST: FAILED \u2717\n";
        std::cout << "====================================\n\n";
        return 1;
    }
}
