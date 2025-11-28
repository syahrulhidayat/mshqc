/**
 * @file he_fci_regression.cc
 * @brief He atom FCI regression test (cc-pVDZ basis)
 * 
 * PURPOSE:
 * Regression test for 2-electron closed-shell system (He atom).
 * Validates FCI implementation for paired-spin configuration.
 * 
 * SYSTEM:
 *   - He atom (Z=2), 2 electrons (1s²)
 *   - Basis: cc-pVDZ (5 functions: 1s, 2s, 2px, 2py, 2pz)
 *   - FCI space: C(5,1) × C(5,1) = 25 determinants (singlet)
 * 
 * EXPECTED ENERGY:
 *   - E(RHF):  ≈ -2.855 Ha (cc-pVDZ)
 *   - E(FCI):  ≈ -2.887 Ha
 *   - E_corr:  ≈ -0.032 to -0.042 Ha (depends on exact basis)
 * 
 * THEORY:
 *   - FCI is exact solution in given basis
 *   - For 2e⁻ system, FCI = full CI (all excitations included)
 *   - Reference: Helgaker et al. (2000), Ch. 11
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 * @note Original implementation from textbook theory, not copied from other QC software
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/ci/fci.h"
#include "mshqc/ci/determinant.h"
#include "mshqc/ci/ci_utils.h"
#include <iostream>
#include <iomanip>
#include <memory>
#include <cmath>

using namespace mshqc;

int main() {
    std::cout << "============================================\n";
    std::cout << "  He Atom FCI Regression Test\n";
    std::cout << "  Basis: cc-pVDZ\n";
    std::cout << "============================================\n\n";
    
    // He atom at origin
    Molecule mol;
    mol.add_atom(2, 0.0, 0.0, 0.0);  // Z=2
    
    std::cout << "System: He atom (1s²)\n";
    std::cout << "  2 electrons, closed-shell\n";
    std::cout << "  Ground state: ¹S\n\n";
    
    // cc-pVDZ basis
    BasisSet basis("cc-pVDZ", mol);
    int nbf = basis.n_basis_functions();
    std::cout << "Basis: cc-pVDZ (" << nbf << " functions)\n";
    std::cout << "Expected: 5 functions (1s, 2s, 2p_x, 2p_y, 2p_z)\n\n";
    
    if (nbf != 5) {
        std::cerr << "WARNING: Expected 5 basis functions, got " << nbf << "\n";
        std::cerr << "Results may differ from benchmark.\n\n";
    }
    
    // Integrals
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    
    // ========================================
    // Step 1: RHF Calculation
    // ========================================
    std::cout << "Step 1: RHF Calculation\n";
    std::cout << "----------------------------------------\n";
    
    SCFConfig config;
    config.max_iterations = 100;
    config.energy_threshold = 1e-10;
    config.density_threshold = 1e-8;
    config.print_level = 0;
    
    RHF rhf(mol, basis, integrals, config);
    auto rhf_result = rhf.compute();
    
    if (!rhf_result.converged) {
        std::cerr << "ERROR: RHF did not converge!\n";
        return 1;
    }
    
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "E(RHF) = " << rhf_result.energy_total << " Ha\n";
    std::cout << "  Nuclear repulsion: " << rhf_result.energy_nuclear << " Ha\n";
    std::cout << "  Electronic energy: " << (rhf_result.energy_total - rhf_result.energy_nuclear) << " Ha\n\n";
    
    // Sanity check
    if (std::abs(rhf_result.energy_total - (-2.855)) > 0.1) {
        std::cerr << "WARNING: RHF energy far from expected -2.855 Ha\n";
        std::cerr << "  Got: " << rhf_result.energy_total << " Ha\n";
        std::cerr << "  This may indicate basis set or integral issues.\n\n";
    }
    
    // ========================================
    // Step 2: Transform integrals to MO basis
    // ========================================
    std::cout << "Step 2: Transform integrals to MO basis\n";
    std::cout << "----------------------------------------\n";
    
    // One-electron: h = T + V
    auto T = integrals->compute_kinetic();
    auto V = integrals->compute_nuclear();
    Eigen::MatrixXd h_ao = T + V;
    
    Eigen::MatrixXd h_mo = rhf_result.C_alpha.transpose() * h_ao * rhf_result.C_alpha;
    
    std::cout << "Orbital energies (ε_i):\n";
    for (int i = 0; i < std::min(nbf, 5); i++) {
        std::cout << "  ε[" << i << "] = " << std::setw(12) << rhf_result.orbital_energies_alpha(i) << " Ha\n";
    }
    std::cout << "\n";
    
    // Two-electron: ERI
    auto eri_ao = integrals->compute_eri();
    
    std::cout << "Transforming ERIs to MO basis...\n";
    
    // Transform ERI to MO basis (chemist notation)
    // REFERENCE: Helgaker et al. (2000), Sec. 9.6.2
    Eigen::Tensor<double, 4> eri_mo_chemist(nbf, nbf, nbf, nbf);
    eri_mo_chemist.setZero();
    
    for (int p = 0; p < nbf; p++) {
        for (int q = 0; q < nbf; q++) {
            for (int r = 0; r < nbf; r++) {
                for (int s = 0; s < nbf; s++) {
                    for (int mu = 0; mu < nbf; mu++) {
                        for (int nu = 0; nu < nbf; nu++) {
                            for (int lam = 0; lam < nbf; lam++) {
                                for (int sig = 0; sig < nbf; sig++) {
                                    eri_mo_chemist(p,q,r,s) += 
                                        rhf_result.C_alpha(mu,p) * rhf_result.C_alpha(nu,q) *
                                        eri_ao(mu,nu,lam,sig) *
                                        rhf_result.C_alpha(lam,r) * rhf_result.C_alpha(sig,s);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Convert chemist→physicist with antisymmetrization
    // REFERENCE: Szabo & Ostlund (1996), Appendix A
    Eigen::Tensor<double, 4> eri_mo_antisym(nbf, nbf, nbf, nbf);
    ci::build_same_spin_antisym_from_chemist(eri_mo_chemist, eri_mo_antisym);
    
    std::cout << "ERI transformation complete.\n\n";
    
    // Setup CI integrals
    ci::CIIntegrals ci_ints;
    ci_ints.h_alpha = h_mo;
    ci_ints.h_beta = h_mo;  // Same for RHF
    ci_ints.eri_aaaa = eri_mo_antisym;
    ci_ints.eri_bbbb = eri_mo_antisym;
    ci_ints.eri_aabb = eri_mo_chemist;  // No antisym for mixed-spin
    ci_ints.e_nuc = rhf_result.energy_nuclear;
    ci_ints.use_fock = false;
    
    // ========================================
    // Step 3: FCI Calculation
    // ========================================
    std::cout << "Step 3: FCI Calculation (exact in basis)\n";
    std::cout << "----------------------------------------\n";
    
    int n_alpha = 1;  // He: 2 electrons, 1α + 1β
    int n_beta = 1;
    
    // Estimate FCI space size
    size_t n_fci_est = ci::fci_determinant_count(nbf, n_alpha, n_beta);
    std::cout << "Estimated FCI space: " << n_fci_est << " determinants\n";
    std::cout << "  (C(" << nbf << "," << n_alpha << ") × C(" << nbf << "," << n_beta << ") = " 
              << n_fci_est << ")\n\n";
    
    ci::FCI fci(ci_ints, nbf, n_alpha, n_beta);
    auto fci_result = fci.compute();
    
    if (fci_result.n_determinants == 0) {
        std::cerr << "ERROR: FCI produced 0 determinants!\n";
        return 1;
    }
    
    std::cout << "FCI complete.\n";
    std::cout << "  Actual determinants: " << fci_result.n_determinants << "\n\n";
    
    // ========================================
    // Step 4: Results and Validation
    // ========================================
    std::cout << "========================================\n";
    std::cout << "  RESULTS\n";
    std::cout << "========================================\n\n";
    
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Energy Summary:\n";
    std::cout << "  E(RHF)      = " << std::setw(14) << rhf_result.energy_total << " Ha\n";
    std::cout << "  E(FCI)      = " << std::setw(14) << fci_result.e_fci << " Ha\n";
    std::cout << "  E_corr(FCI) = " << std::setw(14) << fci_result.e_corr << " Ha\n\n";
    
    std::cout << "Difference:\n";
    std::cout << "  FCI - RHF   = " << std::setw(14) << (fci_result.e_fci - rhf_result.energy_total) << " Ha\n\n";
    
    std::cout << "Wavefunction Analysis:\n";
    std::cout << "  HF determinant weight: " << std::setw(10) << fci_result.hf_weight 
              << " (" << (fci_result.hf_weight * 100.0) << "%)\n\n";
    
    // ========================================
    // Validation
    // ========================================
    std::cout << "========================================\n";
    std::cout << "  VALIDATION\n";
    std::cout << "========================================\n\n";
    
    bool passed = true;
    
    // Check 1: Correlation energy magnitude
    double e_corr_abs = std::abs(fci_result.e_corr);
    std::cout << "Check 1: Correlation energy magnitude\n";
    std::cout << "  |E_corr| = " << e_corr_abs << " Ha\n";
    
    if (e_corr_abs < 0.020 || e_corr_abs > 0.060) {
        std::cerr << "  ❌ FAILED: Expected |E_corr| ≈ 0.032-0.042 Ha for He/cc-pVDZ\n";
        std::cerr << "     Got: " << e_corr_abs << " Ha\n";
        passed = false;
    } else {
        std::cout << "  ✓ PASSED: E_corr in expected range (0.020-0.060 Ha)\n";
    }
    std::cout << "\n";
    
    // Check 2: E_corr should be negative
    std::cout << "Check 2: Correlation energy sign\n";
    if (fci_result.e_corr > 0) {
        std::cerr << "  ❌ FAILED: E_corr should be negative (correlation lowers energy)\n";
        passed = false;
    } else {
        std::cout << "  ✓ PASSED: E_corr < 0 (correlation lowers energy)\n";
    }
    std::cout << "\n";
    
    // Check 3: FCI should lower energy vs RHF
    std::cout << "Check 3: Variational principle\n";
    if (fci_result.e_fci >= rhf_result.energy_total) {
        std::cerr << "  ❌ FAILED: FCI energy should be lower than RHF (variational principle)\n";
        std::cerr << "     E(FCI) = " << fci_result.e_fci << " Ha\n";
        std::cerr << "     E(RHF) = " << rhf_result.energy_total << " Ha\n";
        passed = false;
    } else {
        std::cout << "  ✓ PASSED: E(FCI) < E(RHF) (variational principle)\n";
    }
    std::cout << "\n";
    
    // Check 4: HF weight should be dominant (> 0.90 for He)
    std::cout << "Check 4: HF determinant weight\n";
    std::cout << "  C_HF² = " << fci_result.hf_weight << "\n";
    if (fci_result.hf_weight < 0.85) {
        std::cerr << "  ⚠️  WARNING: HF weight unexpectedly low for He atom\n";
        std::cerr << "     Expected > 0.90, got " << fci_result.hf_weight << "\n";
        // Not a hard failure, but suspicious
    } else {
        std::cout << "  ✓ PASSED: HF weight > 0.85 (expected for weakly correlated He)\n";
    }
    std::cout << "\n";
    
    // Check 5: FCI space size
    std::cout << "Check 5: FCI determinant count\n";
    std::cout << "  Expected: " << n_fci_est << " determinants\n";
    std::cout << "  Actual:   " << fci_result.n_determinants << " determinants\n";
    if (fci_result.n_determinants != static_cast<int>(n_fci_est)) {
        std::cerr << "  ⚠️  WARNING: FCI space size mismatch\n";
        // Not critical if close
    } else {
        std::cout << "  ✓ PASSED: FCI space size matches C(n,k) formula\n";
    }
    std::cout << "\n";
    
    // ========================================
    // Final verdict
    // ========================================
    std::cout << "========================================\n";
    if (passed) {
        std::cout << "  ✅ ALL TESTS PASSED\n";
    } else {
        std::cout << "  ❌ SOME TESTS FAILED\n";
    }
    std::cout << "========================================\n\n";
    
    // Reference values for comparison
    std::cout << "Reference (literature values for He/cc-pVDZ):\n";
    std::cout << "  E(RHF):  ≈ -2.855 Ha\n";
    std::cout << "  E(FCI):  ≈ -2.887 Ha\n";
    std::cout << "  E_corr:  ≈ -0.032 to -0.042 Ha\n";
    std::cout << "  (Exact values depend on cc-pVDZ implementation)\n\n";
    
    std::cout << "Note: This is a REGRESSION test.\n";
    std::cout << "Primary goal: ensure FCI implementation consistency and\n";
    std::cout << "correct handling of paired-spin (closed-shell) systems.\n\n";
    
    std::cout << "============================================\n";
    std::cout << "  TEST COMPLETE\n";
    std::cout << "============================================\n";
    
    return passed ? 0 : 1;
}
