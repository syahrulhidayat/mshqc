/**
 * @file wavefunction_test.cc
 * @brief Test wavefunction container with UMP2 Li atom
 * 
 * This demonstrates the universal wavefunction container by:
 * 1. Running UMP2 on Li atom (doublet, open-shell)
 * 2. Storing T2 amplitudes in wavefunction object
 * 3. Analyzing dominant excitations
 * 4. Printing LaTeX-ready output
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 * @license MIT
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/ump2.h"
#include "mshqc/foundation/wavefunction.h"
#include <iostream>
#include <iomanip>
#include <memory>

using namespace mshqc;

int main() {
    std::cout << "=== Wavefunction Container Test: UMP2/cc-pVTZ for Li ===" << std::endl;
    
    // 1. Set up Li atom (doublet, 2α + 1β)
    Molecule li;
    li.add_atom(3, 0.0, 0.0, 0.0);  // Li at origin
    
    std::cout << "\nMolecule: Li atom (2α, 1β)" << std::endl;
    
    // 2. Basis set
    BasisSet basis("cc-pVTZ", li);
    std::cout << "Basis: cc-pVTZ (" << basis.n_basis_functions() << " functions)" << std::endl;
    
    // 3. Integrals
    auto integrals = std::make_shared<IntegralEngine>(li, basis);
    
    // 4. UHF
    std::cout << "\n--- Running UHF ---" << std::endl;
    SCFConfig config;
    config.max_iterations = 50;
    config.energy_threshold = 1e-8;
    config.density_threshold = 1e-6;
    config.print_level = 0;  // Quiet
    
    int n_alpha = 2;
    int n_beta = 1;
    
    UHF uhf(li, basis, integrals, n_alpha, n_beta, config);
    auto scf_result = uhf.compute();
    
    std::cout << "UHF energy: " << std::fixed << std::setprecision(10) 
              << scf_result.energy_total << " Ha" << std::endl;
    
    // 5. UMP2
    std::cout << "\n--- Running UMP2 ---" << std::endl;
    UMP2 ump2(scf_result, basis, integrals);
    auto mp2_result = ump2.compute();
    
    std::cout << "MP2 correlation energy: " << mp2_result.e_corr_total << " Ha" << std::endl;
    std::cout << "Total UMP2 energy: " << mp2_result.e_total << " Ha" << std::endl;
    std::cout << "\nSpin components:" << std::endl;
    std::cout << "  Same-spin (αα): " << std::setprecision(6) << mp2_result.e_corr_ss_aa << " Ha" << std::endl;
    std::cout << "  Same-spin (ββ): " << mp2_result.e_corr_ss_bb << " Ha" << std::endl;
    std::cout << "  Opposite-spin:  " << mp2_result.e_corr_os << " Ha" << std::endl;
    
    // 6. Create wavefunction container
    std::cout << "\n--- Building Wavefunction Container ---" << std::endl;
    
    int nocc_a = n_alpha;
    int nocc_b = n_beta;
    int nvirt_a = basis.n_basis_functions() - nocc_a;
    int nvirt_b = basis.n_basis_functions() - nocc_b;
    
    foundation::Wavefunction wfn(nocc_a, nocc_b, nvirt_a, nvirt_b);
    
    // Store T2 amplitudes from UMP2
    auto t2_tensors = ump2.get_t2_amplitudes();
    wfn.set_t2_order_1(t2_tensors.t2_aa, t2_tensors.t2_bb, t2_tensors.t2_ab);
    
    std::cout << "✓ Stored T2 amplitudes in wavefunction container" << std::endl;
    
    // 7. Analyze wavefunction
    wfn.print_summary(std::cout);
    
    // 8. LaTeX output
    std::cout << "\n--- LaTeX Format (top 5 excitations) ---" << std::endl;
    wfn.print_latex(std::cout, 5);
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    
    return 0;
}
