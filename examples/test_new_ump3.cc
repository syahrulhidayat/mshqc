/**
 * @file test_new_ump3.cc
 * @brief Test NEW UMP3 implementation against Psi4 validation data
 * 
 * Expected results for Li atom (2α, 1β) / STO-3G:
 *   UHF:  -7.315526 Ha
 *   E(2): -0.000256 Ha
 *   E(3): -0.000042 Ha (16.4% of E(2), CONVERGENT)
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-01-29
 */

#include <mshqc/scf.h>
#include <mshqc/ump2.h>
#include <mshqc/ump3.h>
#include <mshqc/molecule.h>
#include <mshqc/basis.h>
#include <mshqc/integrals.h>
#include <iostream>
#include <iomanip>

using namespace mshqc;

int main() {
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "\n========================================\n";
    std::cout << "  NEW UMP3 Validation Test\n";
    std::cout << "  System: Li atom (2α, 1β)\n";
    std::cout << "  Basis: STO-3G\n";
    std::cout << "========================================\n";
    
    // Setup Li atom
    Molecule mol;
    mol.add_atom(3, 0.0, 0.0, 0.0);  // Li at origin
    mol.set_multiplicity(2);  // Doublet: 2α, 1β
    
    // Build basis
    BasisSet basis("sto-3g", mol);
    std::cout << "\nBasis functions: " << basis.n_basis_functions() << "\n";
    
    // Compute integrals
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    
    // Run UHF
    std::cout << "\n=== Running UHF ===\n";
    int n_alpha = 2;  // Li: 2α
    int n_beta = 1;   // Li: 1β
    UHF uhf_solver(mol, basis, integrals, n_alpha, n_beta);
    SCFResult uhf = uhf_solver.compute();
    
    std::cout << "\nUHF energy: " << uhf.energy_total << " Ha\n";
    std::cout << "Electrons: " << uhf.n_occ_alpha << "α + " << uhf.n_occ_beta << "β = " 
              << (uhf.n_occ_alpha + uhf.n_occ_beta) << "\n";
    
    // Run UMP2
    std::cout << "\n=== Running UMP2 ===\n";
    UMP2 ump2(uhf, basis, integrals);
    UMP2Result mp2_result = ump2.compute();
    
    std::cout << "\nE(2) total:  " << mp2_result.e_corr_total << " Ha\n";
    std::cout << "UMP2 energy: " << mp2_result.e_total << " Ha\n";
    
    // Run NEW UMP3
    std::cout << "\n=== Running NEW UMP3 (W-intermediate) ===\n";
    UMP3 ump3(uhf, mp2_result, basis, integrals);
    UMP3Result mp3_result = ump3.compute();
    
    // Validation against Psi4
    std::cout << "\n========================================\n";
    std::cout << "  Validation vs Psi4\n";
    std::cout << "========================================\n";
    
    double psi4_uhf = -7.3155259813;
    double psi4_e2  = -0.0002564896;
    double psi4_e3  = -0.0000420459;
    
    double err_uhf = std::abs(uhf.energy_total - psi4_uhf);
    double err_e2  = std::abs(mp2_result.e_corr_total - psi4_e2);
    double err_e3  = std::abs(mp3_result.e_mp3_corr - psi4_e3);
    
    std::cout << "\nQuantity       Our Value           Psi4 Value          Error\n";
    std::cout << "----------------------------------------------------------------\n";
    std::cout << "UHF         " << std::setw(16) << uhf.energy_total 
              << "  " << std::setw(16) << psi4_uhf 
              << "  " << std::scientific << std::setprecision(2) << err_uhf << "\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "E(2)        " << std::setw(16) << mp2_result.e_corr_total 
              << "  " << std::setw(16) << psi4_e2 
              << "  " << std::scientific << std::setprecision(2) << err_e2 << "\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "E(3)        " << std::setw(16) << mp3_result.e_mp3_corr 
              << "  " << std::setw(16) << psi4_e3 
              << "  " << std::scientific << std::setprecision(2) << err_e3 << "\n";
    
    // Check convergence
    std::cout << "\n========================================\n";
    std::cout << "  Convergence Check\n";
    std::cout << "========================================\n";
    
    double ratio = std::abs(mp3_result.e_mp3_corr / mp2_result.e_corr_total);
    std::cout << "\n|E(3)|/|E(2)| = " << std::fixed << std::setprecision(2) << ratio*100 << "%\n";
    
    if (ratio < 1.0) {
        std::cout << "✓ Series is CONVERGENT (|E(3)| < |E(2)|)\n";
    } else {
        std::cout << "✗ Series is DIVERGENT (|E(3)| > |E(2)|)\n";
    }
    
    // Final verdict
    std::cout << "\n========================================\n";
    std::cout << "  Final Verdict\n";
    std::cout << "========================================\n";
    
    bool uhf_ok = (err_uhf < 1e-4);  // 0.1 mHa tolerance
    bool e2_ok  = (err_e2 < 1e-6);   // 1 μHa tolerance
    bool e3_ok  = (err_e3 < 1e-5);   // 10 μHa tolerance (allow 10% error)
    bool conv_ok = (ratio < 1.0);
    
    std::cout << "\nUHF accuracy:  " << (uhf_ok ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << "E(2) accuracy: " << (e2_ok ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << "E(3) accuracy: " << (e3_ok ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << "Convergence:   " << (conv_ok ? "✓ PASS" : "✗ FAIL") << "\n";
    
    if (uhf_ok && e2_ok && e3_ok && conv_ok) {
        std::cout << "\n🎉 ALL TESTS PASSED! NEW UMP3 IS CORRECT!\n";
        return 0;
    } else {
        std::cout << "\n⚠ SOME TESTS FAILED. NEEDS DEBUGGING.\n";
        return 1;
    }
}
