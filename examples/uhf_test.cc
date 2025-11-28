/**
 * Test UHF on Li atom (doublet state)
 * 
 * Expected result (cc-pVTZ):
 * - Energy: ~-7.433 Ha  
 * - <S²>: ~0.75-0.76 (small spin contamination)
 * 
 * Compare with ROHF: -7.431 Ha (should be close)
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace mshqc;

int main() {
    std::cout << "====================================\n";
    std::cout << "  UHF Test: Lithium Atom\n";
    std::cout << "====================================\n";
    
    // Li atom at origin (3 electrons: 2α, 1β)
    Molecule li;
    li.add_atom(3, 0.0, 0.0, 0.0);  // Z=3
    
    std::cout << "\nMolecule:\n";
    std::cout << "  Li atom at origin\n";
    std::cout << "  Electrons: 3 (2α, 1β)\n";
    std::cout << "  Multiplicity: 2 (doublet)\n";
    
    // Basis: cc-pVTZ (30 functions)
    BasisSet basis("cc-pVTZ", li);
    std::cout << "\nBasis: cc-pVTZ\n";
    std::cout << "  Functions: " << basis.n_basis_functions() << "\n";
    
    // Integrals
    auto integrals = std::make_shared<IntegralEngine>(li, basis);
    
    // UHF calculation
    SCFConfig config;
    config.max_iterations = 50;
    config.energy_threshold = 1e-8;
    config.density_threshold = 1e-6;
    config.print_level = 1;
    
    int n_alpha = 2;  // Li: 3 electrons → 2α, 1β
    int n_beta = 1;
    
    UHF uhf(li, basis, integrals, n_alpha, n_beta, config);
    auto result = uhf.compute();
    
    // Validation
    std::cout << "\n====================================\n";
    std::cout << "VALIDATION:\n";
    std::cout << "====================================\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Total energy: " << result.energy_total << " Ha\n";
    
    // Check convergence
    if (!result.converged) {
        std::cout << "✗ SCF did not converge!\n";
        return 1;
    }
    std::cout << "✓ SCF converged in " << result.iterations << " iterations\n";
    
    // Check energy (compare with reference)
    double e_ref = -7.433;  // approximate cc-pVTZ reference
    double e_err = std::abs(result.energy_total - e_ref);
    if (e_err < 0.01) {
        std::cout << "✓ Energy within 10 mHa of reference\n";
    } else {
        std::cout << "✗ Energy differs from reference by " << e_err << " Ha\n";
    }
    
    // Check spin contamination
    double s2 = uhf.compute_s_squared(result);
    double s2_exact = 0.75;  // S=1/2 → S(S+1)=0.75
    double contam = s2 - s2_exact;
    
    std::cout << "\nSpin analysis:\n";
    std::cout << "  <S²> exact:    " << s2_exact << "\n";
    std::cout << "  <S²> UHF:      " << s2 << "\n";
    std::cout << "  Contamination: " << contam << "\n";
    
    if (contam < 0.02) {
        std::cout << "✓ Spin contamination acceptable (<2%)\n";
    } else {
        std::cout << "⚠ Spin contamination " << (contam/s2_exact*100) << "%\n";
    }
    
    std::cout << "\n====================================\n";
    std::cout << "  Test Complete!\n";
    std::cout << "====================================\n";
    
    return 0;
}
