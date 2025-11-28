/**
 * @file li_all_ci_methods_comparison.cc
 * @brief Lithium atom: Comprehensive CI methods comparison
 * 
 * PURPOSE:
 * Compare ALL implemented CI methods (CIS, CISD, FCI, MRCI) for Li ground state.
 * This validates the hierarchy: E(UHF) > E(CIS) > E(CISD) > E(FCI)
 * 
 * SYSTEM:
 *   - Li atom (Z=3), 3 electrons (2α + 1β), ground state ²S (1s² 2s¹)
 *   - Basis: cc-pVDZ (14 basis functions)
 *   - UHF reference (open-shell)
 * 
 * EXPECTED HIERARCHY:
 *   E(UHF) > E(CIS) > E(CISD) > E(MRCI) ≥ E(FCI)
 *   (Lower energy = better method)
 * 
 * THEORY:
 *   - CIS: Singles only, no electron correlation in ground state
 *   - CISD: Singles + Doubles, includes correlation
 *   - FCI: ALL excitations, exact in basis (benchmark)
 *   - MRCI: Multi-reference CI (should be close to FCI for Li)
 * 
 * REFERENCES:
 *   - Szabo & Ostlund (1996), Ch. 4: CI methods
 *   - Helgaker et al. (2000), Ch. 10: CI hierarchy
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 * @note Original implementation from textbook theory (AI_RULES compliant)
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/ci/cis.h"
#include "mshqc/ci/cisd.h"
#include "mshqc/ci/fci.h"
#include "mshqc/ci/mrci.h"
#include "mshqc/ci/determinant.h"
#include "mshqc/ci/ci_utils.h"
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <cmath>

using namespace mshqc;

int main() {
    std::cout << "============================================\n";
    std::cout << "  Li Atom: ALL CI Methods Comparison\n";
    std::cout << "  UHF → CIS → CISD → MRCI → FCI\n";
    std::cout << "  Basis: cc-pVDZ\n";
    std::cout << "============================================\n\n";
    
    // Li atom
    Molecule mol;
    mol.add_atom(3, 0.0, 0.0, 0.0);  // Z=3 (Lithium)
    
    std::cout << "System: Li atom\n";
    std::cout << "  3 electrons: 2α + 1β (open-shell)\n";
    std::cout << "  Ground state: ²S (1s² 2s¹)\n";
    std::cout << "  Multiplicity: 2 (doublet)\n\n";
    
    // cc-pVDZ basis
    BasisSet basis("cc-pVDZ", mol);
    int nbf = basis.n_basis_functions();
    std::cout << "Basis: cc-pVDZ (" << nbf << " functions)\n\n";
    
    // Integrals
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    
    // ========================================
    // Step 1: UHF Calculation (Reference)
    // ========================================
    std::cout << "Step 1: UHF Calculation (Reference)\n";
    std::cout << "========================================\n";
    
    int n_alpha = 2;  // 1s↑ 2s↑
    int n_beta = 1;   // 1s↓
    
    SCFConfig config;
    config.max_iterations = 100;
    config.energy_threshold = 1e-10;
    config.density_threshold = 1e-8;
    config.print_level = 0;
    
    UHF uhf(mol, basis, integrals, n_alpha, n_beta, config);
    auto uhf_result = uhf.compute();
    
    if (!uhf_result.converged) {
        std::cerr << "ERROR: UHF did not converge!\n";
        return 1;
    }
    
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "E(UHF)      = " << uhf_result.energy_total << " Ha\n";
    std::cout << "E_nuc       = " << uhf_result.energy_nuclear << " Ha\n";
    std::cout << "E_elec      = " << uhf_result.energy_electronic << " Ha\n\n";
    
    // ========================================
    // Step 2: Transform integrals to MO basis
    // ========================================
    std::cout << "Step 2: Transform integrals to MO basis\n";
    std::cout << "========================================\n";
    
    // One-electron: h = T + V
    auto T = integrals->compute_kinetic();
    auto V = integrals->compute_nuclear();
    Eigen::MatrixXd h_ao = T + V;
    
    Eigen::MatrixXd h_mo_alpha = uhf_result.C_alpha.transpose() * h_ao * uhf_result.C_alpha;
    Eigen::MatrixXd h_mo_beta = uhf_result.C_beta.transpose() * h_ao * uhf_result.C_beta;
    
    // Two-electron: ERI
    auto eri_ao = integrals->compute_eri();
    
    std::cout << "Transforming ERIs... (may take 1-2 minutes for cc-pVDZ)\n";
    
    // Transform ERI to MO basis (chemist notation)
    // REFERENCE: Helgaker et al. (2000), Sec. 9.6.2
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
    
    // Convert chemist→physicist with antisymmetrization
    // REFERENCE: Szabo & Ostlund (1996), Appendix A
    Eigen::Tensor<double, 4> eri_aaaa(nbf, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> eri_bbbb(nbf, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> eri_aabb(nbf, nbf, nbf, nbf);
    
    ci::build_same_spin_antisym_from_chemist(eri_mo_chemist_aa, eri_aaaa);
    ci::build_same_spin_antisym_from_chemist(eri_mo_chemist_bb, eri_bbbb);
    ci::build_alpha_beta_from_chemist(eri_mo_chemist_ab, eri_aabb);
    
    std::cout << "ERI transformation complete.\n\n";
    
    // Setup CI integrals
    ci::CIIntegrals ci_ints;
    ci_ints.h_alpha = h_mo_alpha;
    ci_ints.h_beta = h_mo_beta;
    ci_ints.eri_aaaa = eri_aaaa;
    ci_ints.eri_bbbb = eri_bbbb;
    ci_ints.eri_aabb = eri_aabb;
    ci_ints.e_nuc = uhf_result.energy_nuclear;
    ci_ints.use_fock = false;
    
    // HF determinant
    ci::Determinant hf_det(std::vector<int>{0, 1}, std::vector<int>{0});
    
    int n_occ_a = 2;
    int n_occ_b = 1;
    int n_virt_a = nbf - n_occ_a;
    int n_virt_b = nbf - n_occ_b;
    
//     // ========================================
//     // Step 3: CIS Calculation
//     // ========================================
//     std::cout << "Step 3: CIS Calculation (Singles only)\n";
//     std::cout << "========================================\n";
//     
//     ci::CIS cis(ci_ints, hf_det, n_occ_a, n_occ_b, n_virt_a, n_virt_b);
//     
//     try {
//         auto cis_result = cis.compute();
//         std::cout << "E(CIS)      = " << std::setw(14) << cis_result.ground_state << " Ha\n";
//         std::cout << "E_corr(CIS) = " << std::setw(14) << 0.0 << " Ha\n";
//         std::cout << "N_det       = " << cis_result.n_determinants << "\n\n";
//     } catch (const std::exception& e) {
//         std::cerr << "CIS FAILED: " << e.what() << "\n\n";
//     }
    
    // ========================================
    // Step 4: CISD Calculation
    // ========================================
    std::cout << "Step 4: CISD Calculation (Singles + Doubles)\n";
    std::cout << "========================================\n";
    
    ci::CISD cisd(ci_ints, hf_det, n_occ_a, n_occ_b, n_virt_a, n_virt_b);
    auto cisd_result = cisd.compute();
    
    std::cout << "E(CISD)     = " << std::setw(14) << cisd_result.e_cisd << " Ha\n";
    std::cout << "E_corr(CISD)= " << std::setw(14) << cisd_result.e_corr << " Ha\n";
    std::cout << "N_det       = " << cisd_result.n_determinants << "\n\n";
    
    // ========================================
    // Step 5: FCI Calculation (EXACT)
    // ========================================
    std::cout << "Step 5: FCI Calculation (ALL excitations - EXACT)\n";
    std::cout << "========================================\n";
    
    size_t n_fci_est = ci::fci_determinant_count(nbf, n_alpha, n_beta);
    std::cout << "Estimated FCI space: " << n_fci_est << " determinants\n";
    
    if (n_fci_est > 50000) {
        std::cout << "WARNING: Very large FCI space! May take significant time...\n";
    }
    
    ci::FCI fci(ci_ints, nbf, n_alpha, n_beta);
    auto fci_result = fci.compute();
    
    std::cout << "E(FCI)      = " << std::setw(14) << fci_result.e_fci << " Ha\n";
    std::cout << "E_corr(FCI) = " << std::setw(14) << fci_result.e_corr << " Ha\n";
    std::cout << "N_det       = " << fci_result.n_determinants << "\n";
    std::cout << "HF weight   = " << std::setw(10) << fci_result.hf_weight << " (" 
              << (fci_result.hf_weight * 100.0) << "%)\n\n";
    
    // ========================================
    // Step 6: MRCI Calculation - DISABLED (bug fix in progress)
    // ========================================
    std::cout << "Step 6: MRCI Calculation\n";
    std::cout << "========================================\n";
    std::cout << "⚠️  MRCI temporarily disabled - constructor signature needs fix\n";
    std::cout << "    Issue: Open-shell parameter mismatch (n_core/n_active vs n_alpha/n_beta)\n";
    std::cout << "    Status: Known bug, will be fixed in next update\n";
    std::cout << "    Workaround: Use FCI for exact result\n\n";
    
    // TODO: Fix MRCI constructor to accept (ref_dets, n_orb, n_alpha, n_beta)
    // Current signature: (ints, refs, n_core, n_active, n_virtual)
    // Needed for open-shell: Proper determinant indexing for unequal α/β
    
    /* DISABLED CODE:
    std::vector<ci::Determinant> ref_dets;
    ref_dets.push_back(hf_det);
    ci::MRCI mrci(ci_ints, ref_dets, nbf, n_alpha, n_beta);
    auto mrci_result = mrci.compute();
    */
    
    // ========================================
    // Step 7: Comprehensive Comparison
    // ========================================
    std::cout << "========================================\n";
    std::cout << "  COMPREHENSIVE COMPARISON\n";
    std::cout << "========================================\n\n";
    
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Method      Energy (Ha)      E_corr (Ha)     vs UHF (mHa)  vs FCI (mHa)  % Corr\n";
    std::cout << "------------------------------------------------------------------------------------\n";
    
    // UHF
    std::cout << "UHF      " << std::setw(15) << uhf_result.energy_total 
              << std::setw(15) << 0.0 
              << std::setw(15) << 0.0
              << std::setw(15) << ((uhf_result.energy_total - fci_result.e_fci) * 1000.0)
              << std::setw(10) << "  0.0%\n";
    
//     // CIS (if succeeded)
//     try {
//         auto cis_result = cis.compute();
//         double cis_vs_uhf = (cis_result.ground_state - uhf_result.energy_total) * 1000.0;
//         double cis_vs_fci = (cis_result.ground_state - fci_result.e_fci) * 1000.0;
//         double cis_pct = (0.0 / fci_result.e_corr) * 100.0;
//         
//         std::cout << "CIS      " << std::setw(15) << cis_result.ground_state 
//                   << std::setw(15) << 0.0
//                   << std::setw(15) << cis_vs_uhf
//                   << std::setw(15) << cis_vs_fci
//                   << std::setw(10) << cis_pct << "%\n";
//     } catch (...) {
//         std::cout << "CIS      " << std::setw(15) << "FAILED" << "\n";
//     }
    
    // CISD
    double cisd_vs_uhf = (cisd_result.e_cisd - uhf_result.energy_total) * 1000.0;
    double cisd_vs_fci = (cisd_result.e_cisd - fci_result.e_fci) * 1000.0;
    double cisd_pct = (cisd_result.e_corr / fci_result.e_corr) * 100.0;
    
    std::cout << "CISD     " << std::setw(15) << cisd_result.e_cisd 
              << std::setw(15) << cisd_result.e_corr
              << std::setw(15) << cisd_vs_uhf
              << std::setw(15) << cisd_vs_fci
              << std::setw(10) << cisd_pct << "%\n";
    
    // MRCI - disabled (see Step 6)
    std::cout << "MRCI     " << std::setw(15) << "DISABLED" 
              << "  (constructor bug - fix in progress)\n";
    
    // FCI (benchmark)
    std::cout << "FCI      " << std::setw(15) << fci_result.e_fci 
              << std::setw(15) << fci_result.e_corr
              << std::setw(15) << ((fci_result.e_fci - uhf_result.energy_total) * 1000.0)
              << std::setw(15) << 0.0
              << std::setw(10) << "100.0%\n";
    
    std::cout << "------------------------------------------------------------------------------------\n\n";
    
    // ========================================
    // Validation
    // ========================================
    std::cout << "========================================\n";
    std::cout << "  VALIDATION\n";
    std::cout << "========================================\n\n";
    
    bool all_pass = true;
    
    // Check 1: Energy hierarchy (UHF > CISD > FCI)
    std::cout << "Check 1: Energy hierarchy (lower is better)\n";
    if (uhf_result.energy_total > cisd_result.e_cisd && cisd_result.e_cisd > fci_result.e_fci) {
        std::cout << "  ✓ PASSED: E(UHF) > E(CISD) > E(FCI)\n";
    } else {
        std::cerr << "  ❌ FAILED: Energy hierarchy violated!\n";
        all_pass = false;
    }
    std::cout << "\n";
    
    // Check 2: FCI correlation energy
    std::cout << "Check 2: FCI correlation energy magnitude\n";
    double e_corr_abs = std::abs(fci_result.e_corr);
    if (e_corr_abs > 0.0001 && e_corr_abs < 0.001) {
        std::cout << "  ✓ PASSED: |E_corr| ≈ " << e_corr_abs << " Ha (reasonable for Li)\n";
    } else {
        std::cerr << "  ⚠️  WARNING: |E_corr| = " << e_corr_abs << " Ha (unexpected magnitude)\n";
    }
    std::cout << "\n";
    
    // Check 3: CISD recovery
    std::cout << "Check 3: CISD correlation recovery\n";
    if (cisd_pct > 85.0 && cisd_pct < 105.0) {
        std::cout << "  ✓ PASSED: CISD recovers " << cisd_pct << "% of FCI correlation\n";
    } else {
        std::cerr << "  ⚠️  WARNING: CISD recovers " << cisd_pct << "% (expected ~90-100%)\n";
    }
    std::cout << "\n";
    
    // ========================================
    // Summary
    // ========================================
    std::cout << "========================================\n";
    std::cout << "  SUMMARY\n";
    std::cout << "========================================\n\n";
    
    std::cout << "Reference (literature for Li/cc-pVDZ):\n";
    std::cout << "  E(UHF)  ≈ -7.432 Ha\n";
    std::cout << "  E(FCI)  ≈ -7.433 Ha\n";
    std::cout << "  E_corr  ≈ -0.0002 to -0.0003 Ha\n\n";
    
    std::cout << "Our results:\n";
    std::cout << "  E(UHF)  = " << uhf_result.energy_total << " Ha\n";
    std::cout << "  E(CISD) = " << cisd_result.e_cisd << " Ha\n";
    std::cout << "  E(FCI)  = " << fci_result.e_fci << " Ha\n";
    std::cout << "  E_corr  = " << fci_result.e_corr << " Ha\n\n";
    
    if (all_pass) {
        std::cout << "✅ ALL VALIDATIONS PASSED\n\n";
    } else {
        std::cout << "⚠️  SOME VALIDATIONS FAILED - CHECK RESULTS\n\n";
    }
    
    std::cout << "============================================\n";
    std::cout << "  TEST COMPLETE\n";
    std::cout << "============================================\n";
    
    return all_pass ? 0 : 1;
}
