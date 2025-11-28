/**
 * @file cholesky_caspt2_test.cc
 * @brief Test Cholesky-CASPT2 vs exact CASPT2
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/mcscf/active_space.h"
#include "mshqc/mcscf/casscf.h"
#include "mshqc/mcscf/caspt2.h"
#include "mshqc/mcscf/cholesky_caspt2.h"
#include <iostream>
#include <iomanip>

using namespace mshqc;
using namespace mshqc::mcscf;

int main() {
    std::cout << "\n======================================================================\n";
    std::cout << "Cholesky-CASPT2 Test: Li Atom (cc-pVDZ)\n";
    std::cout << "======================================================================\n";
    std::cout << "System: Li (3e, ²S), CAS(3e,5o)\n";
    std::cout << "Basis: cc-pVDZ (14 functions)\n";
    std::cout << "Goal: Match exact CASPT2 within Cholesky threshold\n";
    std::cout << "======================================================================\n\n";

    // Li atom
    Molecule mol;
    mol.add_atom(3, 0.0, 0.0, 0.0);  // Li at origin
    
    // Load cc-pVDZ basis
    std::string basis_path = "cc-pVDZ";
    BasisSet basis(basis_path, mol);
    
    std::cout << "Basis: " << basis.name() << " (" 
              << basis.n_basis_functions() << " functions)\n\n";
    
    // Integrals
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    
    // UHF (Li has 1 unpaired electron)
    std::cout << "Running UHF...\n";
    SCFConfig scf_cfg;
    scf_cfg.max_iterations = 100;
    scf_cfg.energy_threshold = 1e-8;
    UHF uhf(mol, basis, integrals, /*n_alpha=*/2, /*n_beta=*/1, scf_cfg);
    auto uhf_result = uhf.compute();
    std::cout << "  E(UHF) = " << std::fixed << std::setprecision(10) 
              << uhf_result.energy_total << " Ha\n\n";
    
    // CASSCF(3e,5o)
    std::cout << "Running CASSCF(3e,5o)...\n";
    int nbf = basis.n_basis_functions();
    int nelec = 3;
    auto active_space = ActiveSpace::CAS(3, 5, nbf, nelec);
    
    CASSCF casscf(mol, basis, integrals, active_space);
    auto casscf_result = casscf.compute(uhf_result);
    std::cout << "  E(CASSCF) = " << std::fixed << std::setprecision(10) 
              << casscf_result.e_casscf << " Ha\n\n";
    
    // Exact CASPT2
    std::cout << "Running standard CASPT2 (exact)...\n";
    CASPT2 caspt2_exact(mol, basis, integrals, casscf_result);
    auto result_exact = caspt2_exact.compute();
    std::cout << "  E(CASPT2 exact) = " << result_exact.e_total << " Ha\n";
    std::cout << "  E(PT2 exact)    = " << std::scientific << std::setprecision(6) 
              << result_exact.e_pt2 << " Ha\n\n";
    
    // Cholesky-CASPT2
    std::cout << "Running Cholesky-CASPT2...\n";
    double chol_threshold = 1e-6;  // 1 µHa threshold
    CholeskyCASPT2 caspt2_chol(mol, basis, integrals, casscf_result, chol_threshold);
    auto result_chol = caspt2_chol.compute();
    
    // Compare
    std::cout << "\n======================================================================\n";
    std::cout << "Comparison: Cholesky vs Exact CASPT2\n";
    std::cout << "======================================================================\n\n";
    
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "E(CASPT2 exact):     " << result_exact.e_total << " Ha\n";
    std::cout << "E(Cholesky-CASPT2):  " << result_chol.e_total << " Ha\n";
    std::cout << "Difference:          " << std::scientific << std::setprecision(3)
              << (result_chol.e_total - result_exact.e_total) << " Ha\n";
    std::cout << "                     " << std::fixed << std::setprecision(3)
              << (result_chol.e_total - result_exact.e_total) * 1e6 << " µHa\n\n";
    
    std::cout << "E(PT2 exact):        " << std::scientific << std::setprecision(10) 
              << result_exact.e_pt2 << " Ha\n";
    std::cout << "E(PT2 Cholesky):     " << result_chol.e_pt2 << " Ha\n";
    std::cout << "Difference:          " << std::scientific << std::setprecision(3)
              << (result_chol.e_pt2 - result_exact.e_pt2) << " Ha\n";
    std::cout << "                     " << std::fixed << std::setprecision(3)
              << (result_chol.e_pt2 - result_exact.e_pt2) * 1e6 << " µHa\n\n";
    
    std::cout << "======================================================================\n";
    std::cout << "Validation Results\n";
    std::cout << "======================================================================\n\n";
    
    double error_uHa = std::abs(result_chol.e_total - result_exact.e_total) * 1e6;
    std::cout << "Cholesky threshold:  " << std::scientific << chol_threshold << " Ha\n";
    std::cout << "Cholesky max error:  " << result_chol.cholesky_error << " Ha\n";
    std::cout << "Cholesky vectors:    " << result_chol.n_cholesky_vectors << "\n";
    std::cout << "CASPT2 accuracy:     " << std::fixed << std::setprecision(3) 
              << error_uHa << " µHa\n\n";
    
    if (error_uHa < 10.0) {
        std::cout << "✓ PASS: Cholesky-CASPT2 matches exact CASPT2\n";
        std::cout << "   Error < 10 µHa (excellent agreement)\n";
    } else if (error_uHa < 100.0) {
        std::cout << "✓ ACCEPTABLE: Cholesky-CASPT2 reasonable accuracy\n";
        std::cout << "   Error < 100 µHa (may need tighter threshold)\n";
    } else {
        std::cout << "❌ FAIL: Cholesky-CASPT2 accuracy insufficient\n";
        std::cout << "   Error > 100 µHa (check implementation)\n";
    }
    
    std::cout << "\n======================================================================\n";
    
    return 0;
}
