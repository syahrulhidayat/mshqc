/**
 * @file rmp3_he2_test.cc
 * @brief Test RMP3 on He2 dimer (minimal system)
 * 
 * Uses He2 with minimal basis to test RMP3 correctness quickly.
 * The O(N^6) algorithm is feasible for this tiny system.
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
#include <iostream>
#include <iomanip>
#include <memory>

using namespace mshqc;

int main() {
    std::cout << "====================================\n";
    std::cout << "  RMP3 Test: He2/STO-3G\n";
    std::cout << "====================================\n";
    
    // He2 dimer at 3.0 Bohr separation
    Molecule he2;
    he2.add_atom(2, 0.0, 0.0, 0.0);    // He
    he2.add_atom(2, 0.0, 0.0, 3.0);    // He
    
    std::cout << "\nMolecule: He2 (4 electrons, closed-shell)\n";
    std::cout << "Separation: 3.0 Bohr\n";
    
    // Minimal basis
    BasisSet basis("STO-3G", he2);
    std::cout << "Basis: STO-3G (" << basis.n_basis_functions() << " functions)\n";
    
    // Integrals
    auto integrals = std::make_shared<IntegralEngine>(he2, basis);
    
    // RHF
    std::cout << "\n--- Running RHF ---\n";
    SCFConfig config;
    config.max_iterations = 50;
    config.energy_threshold = 1e-8;
    config.density_threshold = 1e-6;
    config.print_level = 0;
    
    RHF rhf(he2, basis, integrals, config);
    auto rhf_result = rhf.compute();
    
    std::cout << "RHF energy: " << std::fixed << std::setprecision(10) 
              << rhf_result.energy_total << " Ha\n";
    
    // RMP2
    std::cout << "\n--- Running RMP2 ---\n";
    foundation::RMP2 rmp2(rhf_result, basis, integrals);
    auto rmp2_result = rmp2.compute();
    
    std::cout << "MP2 correlation: " << rmp2_result.e_corr << " Ha\n";
    std::cout << "RMP2 total:      " << rmp2_result.e_total << " Ha\n";
    
    // RMP3
    std::cout << "\n--- Running RMP3 ---\n";
    foundation::RMP3 rmp3(rhf_result, rmp2_result, basis, integrals);
    auto rmp3_result = rmp3.compute();
    
    // Summary
    std::cout << "\n====================================\n";
    std::cout << "  FINAL SUMMARY\n";
    std::cout << "====================================\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "RHF energy:           " << rmp3_result.e_rhf << " Ha\n";
    std::cout << "MP2 correction:       " << rmp3_result.e_mp2 << " Ha\n";
    std::cout << "MP3 correction:       " << rmp3_result.e_mp3 << " Ha\n";
    std::cout << "Total correlation:    " << rmp3_result.e_corr_total << " Ha\n";
    std::cout << "RMP3 total energy:    " << rmp3_result.e_total << " Ha\n";
    
    std::cout << "\n====================================\n";
    std::cout << "  Test Complete!\n";
    std::cout << "====================================\n";
    
    return 0;
}
