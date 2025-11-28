/**
 * Test UMP2 on Li atom
 * 
 * Expected:
 * - UHF: -7.431 Ha ✓ (verified)
 * - UMP2 correlation: ~-0.011 Ha (matching Psi4 DF-MP2)
 * - Total: ~-7.442 Ha
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/ump2.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace mshqc;

int main() {
    std::cout << "====================================\n";
    std::cout << "  UMP2 Test: Lithium Atom\n";
    std::cout << "====================================\n";
    
    // Li atom
    Molecule li;
    li.add_atom(3, 0.0, 0.0, 0.0);
    
    std::cout << "\nMolecule: Li (2α, 1β)\n";
    
    // Basis
    BasisSet basis("cc-pVTZ", li);
    std::cout << "Basis: cc-pVTZ (" << basis.n_basis_functions() << " functions)\n";
    
    // Integrals
    auto integrals = std::make_shared<IntegralEngine>(li, basis);
    
    // UHF
    SCFConfig config;
    config.max_iterations = 50;
    config.energy_threshold = 1e-8;
    config.density_threshold = 1e-6;
    config.print_level = 0;  // Quiet UHF
    
    int n_alpha = 2;
    int n_beta = 1;
    
    UHF uhf(li, basis, integrals, n_alpha, n_beta, config);
    auto uhf_result = uhf.compute();
    
    // UMP2
    UMP2 ump2(uhf_result, basis, integrals);
    auto ump2_result = ump2.compute();
    
    // Validation
    std::cout << "\n====================================\n";
    std::cout << "VALIDATION:\n";
    std::cout << "====================================\n";
    std::cout << std::fixed << std::setprecision(6);
    
    // Check correlation energy
    double e_corr = ump2_result.e_corr_total;
    double e_ref = -0.011;  // Psi4 DF-MP2 reference
    double err = std::abs(e_corr - e_ref);
    
    std::cout << "Correlation:  " << e_corr << " Ha\n";
    std::cout << "Reference:    " << e_ref << " Ha\n";
    std::cout << "Error:        " << err << " Ha\n";
    
    if (err < 0.001) {
        std::cout << "✓ Correlation within 1 mHa of reference\n";
    } else {
        std::cout << "✗ Correlation differs by " << err*1000 << " mHa\n";
    }
    
    // Component breakdown
    std::cout << "\nComponent analysis:\n";
    std::cout << "  SS(αα): " << ump2_result.e_corr_ss_aa << " Ha\n";
    std::cout << "  SS(ββ): " << ump2_result.e_corr_ss_bb << " Ha\n";
    std::cout << "  OS(αβ): " << ump2_result.e_corr_os << " Ha\n";
    
    std::cout << "\n====================================\n";
    std::cout << "  Test Complete!\n";
    std::cout << "====================================\n";
    
    return 0;
}
