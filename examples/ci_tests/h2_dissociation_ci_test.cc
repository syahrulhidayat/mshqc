/**
 * @file h2_dissociation_ci_test.cc
 * @brief H2 dissociation curve regression test
 * 
 * PURPOSE:
 * Test CI methods (CISD, FCI, CISDT) on H2 at stretched geometry.
 * This is a critical test for multi-reference character.
 * 
 * SYSTEM:
 *   - H2 at R = 2.0 Å (stretched, near dissociation)
 *   - Basis: STO-3G (2 basis functions)
 *   - Expected: Strong multi-reference character
 * 
 * THEORY:
 *   At R → ∞, H2 dissociates to 2H atoms
 *   - RHF fails (gives ionic H⁺H⁻)
 *   - UHF better but has spin contamination
 *   - FCI exact (gives proper H· + H·)
 * 
 * EXPECTED:
 *   - HF weight should decrease (< 0.90) at large R
 *   - FCI should recover more correlation than CISD
 *   - E(FCI) should be significantly lower than E(CISD)
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 * @note Regression test for multi-reference systems
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/ci/cisd.h"
#include "mshqc/ci/fci.h"
#include "mshqc/ci/cisdt.h"
#include "mshqc/ci/wavefunction_analysis.h"
#include "mshqc/ci/ci_utils.h"
#include <iostream>
#include <iomanip>
#include <memory>
#include <cmath>

using namespace mshqc;

int main() {
    std::cout << "============================================\n";
    std::cout << "  H2 Dissociation CI Regression Test\n";
    std::cout << "  R = 2.0 Å (stretched geometry)\n";
    std::cout << "============================================\n\n";
    
    // H2 at stretched geometry (2.0 Å)
    double R = 2.0;  // Angstroms
    Molecule mol;
    mol.add_atom(1, 0.0, 0.0, 0.0);       // H1
    mol.add_atom(1, 0.0, 0.0, R);         // H2
    
    std::cout << "System: H2 molecule\n";
    std::cout << "  Bond length: R = " << R << " Å (stretched)\n";
    std::cout << "  Expected: Multi-reference character\n\n";
    
    // STO-3G basis
    BasisSet basis("STO-3G", mol);
    int nbf = basis.n_basis_functions();
    std::cout << "Basis: STO-3G (" << nbf << " functions)\n\n";
    
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    
    // ========================================
    // Step 1: UHF Calculation
    // ========================================
    std::cout << "Step 1: UHF Reference\n";
    std::cout << "========================================\n";
    
    SCFConfig config;
    config.max_iterations = 100;
    config.energy_threshold = 1e-10;
    config.density_threshold = 1e-8;
    config.print_level = 0;
    
    int n_alpha = 1;
    int n_beta = 1;
    
    UHF uhf(mol, basis, integrals, n_alpha, n_beta, config);
    auto uhf_result = uhf.compute();
    
    if (!uhf_result.converged) {
        std::cerr << "ERROR: UHF did not converge!\n";
        return 1;
    }
    
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "E(UHF) = " << uhf_result.energy_total << " Ha\n\n";
    
    // ========================================
    // Step 2: Transform integrals
    // ========================================
    std::cout << "Step 2: MO Integral Transformation\n";
    std::cout << "========================================\n";
    
    auto T = integrals->compute_kinetic();
    auto V = integrals->compute_nuclear();
    Eigen::MatrixXd h_ao = T + V;
    
    Eigen::MatrixXd h_mo_alpha = uhf_result.C_alpha.transpose() * h_ao * uhf_result.C_alpha;
    Eigen::MatrixXd h_mo_beta = uhf_result.C_beta.transpose() * h_ao * uhf_result.C_beta;
    
    auto eri_ao = integrals->compute_eri();
    
    // Transform ERI
    Eigen::Tensor<double, 4> eri_mo_chemist_aa(nbf, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> eri_mo_chemist_bb(nbf, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> eri_mo_chemist_ab(nbf, nbf, nbf, nbf);
    eri_mo_chemist_aa.setZero();
    eri_mo_chemist_bb.setZero();
    eri_mo_chemist_ab.setZero();
    
    for (int p = 0; p < nbf; p++) {
        for (int q = 0; q < nbf; q++) {
            for (int r = 0; r < nbf; r++) {
                for (int s = 0; s < nbf; s++) {
                    for (int mu = 0; mu < nbf; mu++) {
                        for (int nu = 0; nu < nbf; nu++) {
                            for (int lam = 0; lam < nbf; lam++) {
                                for (int sig = 0; sig < nbf; sig++) {
                                    double val = eri_ao(mu, nu, lam, sig);
                                    
                                    eri_mo_chemist_aa(p,q,r,s) += 
                                        uhf_result.C_alpha(mu,p) * uhf_result.C_alpha(nu,q) *
                                        uhf_result.C_alpha(lam,r) * uhf_result.C_alpha(sig,s) * val;
                                    
                                    eri_mo_chemist_bb(p,q,r,s) += 
                                        uhf_result.C_beta(mu,p) * uhf_result.C_beta(nu,q) *
                                        uhf_result.C_beta(lam,r) * uhf_result.C_beta(sig,s) * val;
                                    
                                    eri_mo_chemist_ab(p,q,r,s) += 
                                        uhf_result.C_alpha(mu,p) * uhf_result.C_alpha(nu,q) *
                                        uhf_result.C_beta(lam,r) * uhf_result.C_beta(sig,s) * val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    Eigen::Tensor<double, 4> eri_aaaa(nbf, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> eri_bbbb(nbf, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> eri_aabb(nbf, nbf, nbf, nbf);
    
    ci::build_same_spin_antisym_from_chemist(eri_mo_chemist_aa, eri_aaaa);
    ci::build_same_spin_antisym_from_chemist(eri_mo_chemist_bb, eri_bbbb);
    ci::build_alpha_beta_from_chemist(eri_mo_chemist_ab, eri_aabb);
    
    std::cout << "ERI transformation complete.\n\n";
    
    ci::CIIntegrals ci_ints;
    ci_ints.h_alpha = h_mo_alpha;
    ci_ints.h_beta = h_mo_beta;
    ci_ints.eri_aaaa = eri_aaaa;
    ci_ints.eri_bbbb = eri_bbbb;
    ci_ints.eri_aabb = eri_aabb;
    ci_ints.e_nuc = uhf_result.energy_nuclear;
    ci_ints.use_fock = false;
    
    ci::Determinant hf_det(std::vector<int>{0}, std::vector<int>{0});
    
    // ========================================
    // Step 3: CISD
    // ========================================
    std::cout << "Step 3: CISD Calculation\n";
    std::cout << "========================================\n";
    
    ci::CISD cisd(ci_ints, hf_det, 1, 1, nbf-1, nbf-1);
    auto cisd_result = cisd.compute();
    
    std::cout << "E(CISD)     = " << cisd_result.e_cisd << " Ha\n";
    std::cout << "E_corr(CISD)= " << cisd_result.e_corr << " Ha\n";
    std::cout << "N_det       = " << cisd_result.n_determinants << "\n\n";
    
    // ========================================
    // Step 4: FCI (exact)
    // ========================================
    std::cout << "Step 4: FCI Calculation (EXACT)\n";
    std::cout << "========================================\n";
    
    ci::FCI fci(ci_ints, nbf, n_alpha, n_beta);
    auto fci_result = fci.compute();
    
    std::cout << "E(FCI)      = " << fci_result.e_fci << " Ha\n";
    std::cout << "E_corr(FCI) = " << fci_result.e_corr << " Ha\n";
    std::cout << "N_det       = " << fci_result.n_determinants << "\n";
    std::cout << "HF weight   = " << fci_result.hf_weight << "\n\n";
    
    // ========================================
    // Step 5: Wavefunction Analysis
    // ========================================
    std::cout << "Step 5: Multi-reference Character Analysis\n";
    std::cout << "========================================\n";
    
    ci::WavefunctionAnalysis wf_analysis(fci_result.determinants, 
                                        fci_result.coefficients);
    auto diag = wf_analysis.compute_diagnostics(hf_det);
    
    std::cout << "HF weight: " << diag.hf_weight << "\n";
    std::cout << "Multi-reference character: " << diag.multireference_character << "\n";
    
    if (!diag.single_reference_ok) {
        std::cout << "✅ Correctly identified as multi-reference (HF weight < 0.90)\n";
    } else {
        std::cout << "⚠️  WARNING: Should be multi-reference at R = 2.0 Å\n";
    }
    std::cout << "\n";
    
    // ========================================
    // Step 6: Validation
    // ========================================
    std::cout << "========================================\n";
    std::cout << "  VALIDATION\n";
    std::cout << "========================================\n\n";
    
    bool all_pass = true;
    
    // Check 1: Energy hierarchy
    std::cout << "Check 1: Energy hierarchy\n";
    if (uhf_result.energy_total > cisd_result.e_cisd && 
        cisd_result.e_cisd > fci_result.e_fci) {
        std::cout << "  ✅ PASSED: E(UHF) > E(CISD) > E(FCI)\n";
    } else {
        std::cerr << "  ❌ FAILED: Energy hierarchy violated!\n";
        all_pass = false;
    }
    std::cout << "\n";
    
    // Check 2: FCI correlation
    std::cout << "Check 2: FCI correlation recovery\n";
    double fci_corr_abs = std::abs(fci_result.e_corr);
    if (fci_corr_abs > 0.01 && fci_corr_abs < 0.20) {
        std::cout << "  ✅ PASSED: |E_corr(FCI)| = " << fci_corr_abs << " Ha (reasonable)\n";
    } else {
        std::cerr << "  ⚠️  WARNING: |E_corr(FCI)| = " << fci_corr_abs << " Ha\n";
    }
    std::cout << "\n";
    
    // Check 3: CISD vs FCI gap
    std::cout << "Check 3: CISD-FCI gap\n";
    double gap = (cisd_result.e_cisd - fci_result.e_fci) * 1000.0;  // mHa
    std::cout << "  Gap: " << gap << " mHa\n";
    if (gap > 0.1) {
        std::cout << "  ✅ PASSED: FCI lowers energy beyond CISD\n";
    } else {
        std::cerr << "  ⚠️  WARNING: Very small CISD-FCI gap\n";
    }
    std::cout << "\n";
    
    // Check 4: Multi-reference character
    std::cout << "Check 4: Multi-reference character detection\n";
    if (diag.hf_weight < 0.90) {
        std::cout << "  ✅ PASSED: Multi-reference character detected\n";
    } else {
        std::cerr << "  ❌ FAILED: Should show multi-reference character at R=2.0 Å\n";
        all_pass = false;
    }
    std::cout << "\n";
    
    // ========================================
    // Summary
    // ========================================
    std::cout << "========================================\n";
    std::cout << "  SUMMARY\n";
    std::cout << "========================================\n\n";
    
    std::cout << "Method      Energy (Ha)      E_corr (Ha)\n";
    std::cout << "--------------------------------------------\n";
    std::cout << "UHF      " << std::setw(15) << uhf_result.energy_total 
              << std::setw(15) << 0.0 << "\n";
    std::cout << "CISD     " << std::setw(15) << cisd_result.e_cisd 
              << std::setw(15) << cisd_result.e_corr << "\n";
    std::cout << "FCI      " << std::setw(15) << fci_result.e_fci 
              << std::setw(15) << fci_result.e_corr << "\n";
    std::cout << "--------------------------------------------\n\n";
    
    if (all_pass) {
        std::cout << "✅ ALL CHECKS PASSED\n\n";
    } else {
        std::cout << "⚠️  SOME CHECKS FAILED\n\n";
    }
    
    std::cout << "============================================\n";
    std::cout << "  TEST COMPLETE\n";
    std::cout << "============================================\n";
    
    return all_pass ? 0 : 1;
}
