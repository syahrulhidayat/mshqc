/**
 * Test UMP3 on Li atom
 * 
 * Expected:
 * - UMP2: -0.0112 Ha ✓
 * - UMP3 correction: ~-0.0005 Ha (10% of MP2)
 * - Total: ~-0.0117 Ha
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
    std::cout << "====================================\n";
    std::cout << "  UMP3 Test: Lithium Atom\n";
    std::cout << "====================================\n";
    
    // Li atom
    Molecule li;
    li.add_atom(3, 0.0, 0.0, 0.0);
    
    std::cout << "\nMolecule: Li (2\u03b1, 1\u03b2)\n";
    
    // Basis
    BasisSet basis("cc-pVTZ", li);
    std::cout << "Basis: cc-pVTZ (" << basis.n_basis_functions() << " functions)\n";
    
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
    
    std::cout << "UHF energy: " << std::fixed << std::setprecision(6) 
              << uhf_result.energy_total << " Ha\n";
    
    // UMP2
    UMP2 ump2(uhf_result, basis, integrals);
    auto ump2_result = ump2.compute();
    
    // UMP3
    UMP3 ump3(uhf_result, ump2_result, basis, integrals);
    auto ump3_result = ump3.compute();
    
    // Validation
    std::cout << "\n====================================\n";
    std::cout << "VALIDATION:\n";
    std::cout << "====================================\n";
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "UMP2 correlation: " << ump2_result.e_corr_total << " Ha\n";
    std::cout << "E(3) correction:  " << ump3_result.e_mp3_corr << " Ha\n";
    std::cout << "UMP3 correlation: " << ump3_result.e_corr_total << " Ha\n";
    std::cout << "\nUMP3 total:       " << ump3_result.e_total << " Ha\n";
    
    // Check E(3) is reasonable (~10-20% of E(2))
    double ratio = std::abs(ump3_result.e_mp3_corr / ump2_result.e_corr_total);
    std::cout << "\nE(3)/E(2) ratio:  " << ratio*100 << "%\n";
    
    if(ratio > 0.05 && ratio < 0.30) {
        std::cout << "\u2713 E(3) correction reasonable (5-30% of E(2))\n";
    } else {
        std::cout << "\u2717 E(3) correction unexpected\n";
    }
    
    // Check total correlation improved
    if(std::abs(ump3_result.e_corr_total) > std::abs(ump2_result.e_corr_total)) {
        std::cout << "\u2713 UMP3 captures more correlation than UMP2\n";
    }
    
    std::cout << "\n====================================\n";
    std::cout << "  Test Complete!\n";
    std::cout << "====================================\n";
    
    return 0;
}
