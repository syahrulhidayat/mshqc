/**
 * @file test_ump3_oxygen.cc
 * @brief Test UMP3 on oxygen atom (triplet ground state)
 * 
 * System: O (4 alpha, 2 beta electrons)
 * Basis: cc-pVDZ (sufficient correlation, not too large)
 * 
 * Expected behavior:
 * - E(2) should be substantial (multiple electron pairs)
 * - E(3) should be negative and < |E(2)| (convergent)
 * - E(UMP3) < E(UMP2) < E(UHF) (energy monotonicity)
 * 
 * @author AI Agent
 * @date 2025-01-29
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/ump2.h"
#include "mshqc/ump3.h"
#include <iostream>
#include <iomanip>

using namespace mshqc;

int main() {
    std::cout << "\n====================================\n";
    std::cout << "  UMP3 Test: Oxygen Atom (Triplet)\n";
    std::cout << "====================================\n\n";
    
    // Oxygen atom: 8 electrons, triplet ground state (4α, 2β)
    Molecule mol;
    mol.add_atom(8, 0.0, 0.0, 0.0);  // O: Z=8
    
    std::cout << "Molecule: O (4α, 2β)\n";
    std::cout << "Basis: cc-pVDZ\n\n";
    
    // Load basis
    BasisSet basis("cc-pVDZ", mol);
    std::cout << "Basis functions: " << basis.n_basis_functions() << "\n\n";
    
    // Integrals
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    
    // Run UHF
    std::cout << "====================================\n";
    std::cout << "  Unrestricted Hartree-Fock (UHF)\n";
    std::cout << "====================================\n";
    
    SCFConfig config;
    config.max_iterations = 50;
    config.energy_threshold = 1e-8;
    config.density_threshold = 1e-6;
    config.print_level = 0;
    
    int n_alpha = 5;  // O: 1s² 2s² 2p⁴ → triplet: 5α, 3β
    int n_beta = 3;
    
    UHF uhf(mol, basis, integrals, n_alpha, n_beta, config);
    auto uhf_result = uhf.compute();
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "UHF energy: " << uhf_result.energy_total << " Ha\n\n";
    
    // Run UMP2
    UMP2 ump2(uhf_result, basis, integrals);
    auto ump2_result = ump2.compute();
    
    // Run UMP3
    UMP3 ump3(uhf_result, ump2_result, basis, integrals);
    auto ump3_result = ump3.compute();
    
    // Validation
    std::cout << "\n====================================\n";
    std::cout << "VALIDATION:\n";
    std::cout << "====================================\n";
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "UHF energy:       " << uhf_result.energy_total << " Ha\n";
    std::cout << "UMP2 correlation: " << ump2_result.e_corr_total << " Ha\n";
    std::cout << "E(3) correction:  " << ump3_result.e_mp3_corr << " Ha\n";
    std::cout << "UMP3 correlation: " << ump3_result.e_corr_total << " Ha\n";
    std::cout << "\n";
    std::cout << "UMP2 total:       " << ump2_result.e_total << " Ha\n";
    std::cout << "UMP3 total:       " << ump3_result.e_total << " Ha\n";
    std::cout << "\n";
    
    // Check convergence criterion
    double ratio = std::abs(ump3_result.e_mp3_corr / ump2_result.e_corr_total);
    std::cout << "E(3)/E(2) ratio:  " << std::setprecision(2) << (ratio * 100.0) << "%\n";
    
    bool converged = ratio < 1.0;
    bool monotonic = ump3_result.e_total < ump2_result.e_total;
    bool negative = ump3_result.e_mp3_corr < 0.0;
    
    std::cout << "\n";
    if (negative) {
        std::cout << "✓ E(3) is negative (correct sign)\n";
    } else {
        std::cout << "✗ E(3) is positive (WRONG - algorithm error)\n";
    }
    
    if (converged) {
        std::cout << "✓ |E(3)| < |E(2)| (convergent series)\n";
    } else {
        std::cout << "✗ |E(3)| > |E(2)| (divergent - bad test system or missing terms)\n";
    }
    
    if (monotonic) {
        std::cout << "✓ E(UMP3) < E(UMP2) (energy monotonicity)\n";
    } else {
        std::cout << "✗ E(UMP3) > E(UMP2) (non-monotonic)\n";
    }
    
    std::cout << "\n====================================\n";
    if (negative && converged && monotonic) {
        std::cout << "  Test PASSED ✓\n";
        std::cout << "====================================\n\n";
        return 0;
    } else {
        std::cout << "  Test FAILED or INCONCLUSIVE ✗\n";
        std::cout << "====================================\n\n";
        return 1;
    }
}
