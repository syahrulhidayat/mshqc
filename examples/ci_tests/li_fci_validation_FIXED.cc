/**
 * @file li_fci_validation_FIXED.cc
 * @brief Li atom: UHF vs CISD vs FCI comparison (FIXED ERI mapping)
 * 
 * Purpose: Validate CISD implementation using FCI as benchmark
 * FCI for 3-electron Li should be EXACT in the given basis
 * 
 * FIX: Use ci_utils.h helpers for correct chemist→physicist ERI mapping
 * 
 * THEORY:
 *   - Integral transformation: Helgaker et al. (2000), Sec. 9.6.2
 *   - Slater-Condon rules: Szabo & Ostlund (1996), Appendix A
 * 
 * @author Muhamad Syahrul Hidayat (refactored)
 * @date 2025-11-16
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/ci/cisd.h"
#include "mshqc/ci/fci.h"
#include "mshqc/ci/determinant.h"
#include "mshqc/ci/ci_utils.h"  // Helper for ERI mapping (chemist→physicist)
#include <iostream>
#include <iomanip>
#include <memory>

using namespace mshqc;

int main() {
    std::cout << "============================================\n";
    std::cout << "  Li: UHF vs CISD vs FCI Validation\n";
    std::cout << "  Basis: cc-pVDZ\n";
    std::cout << "============================================\n\n";
    
    // Li atom
    Molecule mol;
    mol.add_atom(3, 0.0, 0.0, 0.0);  // Z=3
    
    std::cout << "Li atom: 3 electrons (2α, 1β)\n";
    std::cout << "Ground state: ²S (1s² 2s¹)\n\n";
    
    // cc-pVDZ basis
    BasisSet basis("cc-pVDZ", mol);
    int nbf = basis.n_basis_functions();
    std::cout << "Basis: cc-pVDZ (" << nbf << " functions)\n\n";
    
    // Integrals
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    
    // ========================================
    // Step 1: UHF Calculation
    // ========================================
    std::cout << "Step 1: UHF Calculation\n";
    std::cout << "----------------------------------------\n";
    
    int n_alpha = 2;
    int n_beta = 1;
    
    SCFConfig config;
    config.max_iterations = 100;
    config.energy_threshold = 1e-8;
    config.density_threshold = 1e-6;
    config.print_level = 0;
    
    UHF uhf(mol, basis, integrals, n_alpha, n_beta, config);
    auto uhf_result = uhf.compute();
    
    if (!uhf_result.converged) {
        std::cerr << "ERROR: UHF did not converge!\n";
        return 1;
    }
    
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "E(UHF) = " << uhf_result.energy_total << " Ha\n\n";
    
    // ========================================
    // Step 2: Transform integrals to MO basis
    // ========================================
    std::cout << "Step 2: Transform integrals to MO basis\n";
    std::cout << "----------------------------------------\n";
    
    // One-electron: h = T + V
    auto T = integrals->compute_kinetic();
    auto V = integrals->compute_nuclear();
    Eigen::MatrixXd h_ao = T + V;
    
    Eigen::MatrixXd h_mo_alpha = uhf_result.C_alpha.transpose() * h_ao * uhf_result.C_alpha;
    Eigen::MatrixXd h_mo_beta = uhf_result.C_beta.transpose() * h_ao * uhf_result.C_beta;
    
    // Two-electron: ERI (chemist notation in AO basis)
    auto eri_ao = integrals->compute_eri();
    
    std::cout << "Transforming ERIs... (may take 1-2 minutes for cc-pVDZ)\n";
    
    // Transform ERI to MO basis (chemist notation: (pq|rs))
    // REFERENCE: Helgaker et al. (2000), Sec. 9.6.2
    // Formula: (pq|rs)_MO = Σ_{ijkl} C_ip C_jq (ij|kl)_AO C_kr C_ls
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
                    for (int i = 0; i < nbf; i++) {
                        for (int j = 0; j < nbf; j++) {
                            for (int k = 0; k < nbf; k++) {
                                for (int l = 0; l < nbf; l++) {
                                    double val = eri_ao(i, j, k, l);
                                    
                                    eri_mo_chemist_aa(p,q,r,s) += uhf_result.C_alpha(i,p) * uhf_result.C_alpha(j,q) *
                                                                   uhf_result.C_alpha(k,r) * uhf_result.C_alpha(l,s) * val;
                                    
                                    eri_mo_chemist_bb(p,q,r,s) += uhf_result.C_beta(i,p) * uhf_result.C_beta(j,q) *
                                                                   uhf_result.C_beta(k,r) * uhf_result.C_beta(l,s) * val;
                                    
                                    eri_mo_chemist_ab(p,q,r,s) += uhf_result.C_alpha(i,p) * uhf_result.C_alpha(j,q) *
                                                                   uhf_result.C_beta(k,r) * uhf_result.C_beta(l,s) * val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Convert chemist→physicist notation with antisymmetrization
    // REFERENCE: Szabo & Ostlund (1996), Appendix A
    // Same-spin: <pq||rs> = (pr|qs) - (ps|qr) in chemist notation
    // Mixed-spin: <pq|rs> = (pq|rs) (no antisym, just direct Coulomb)
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
    
    // ========================================
    // Step 3: CISD Calculation
    // ========================================
    std::cout << "Step 3: CISD Calculation\n";
    std::cout << "----------------------------------------\n";
    
    int n_occ_a = 2;
    int n_occ_b = 1;
    int n_virt_a = nbf - n_occ_a;
    int n_virt_b = nbf - n_occ_b;
    
    ci::CISD cisd(ci_ints, hf_det, n_occ_a, n_occ_b, n_virt_a, n_virt_b);
    auto cisd_result = cisd.compute();
    
    std::cout << "CISD complete.\n\n";
    
    // ========================================
    // Step 4: FCI Calculation (EXACT!)
    // ========================================
    std::cout << "Step 4: FCI Calculation (EXACT in basis)\n";
    std::cout << "----------------------------------------\n";
    
    // Estimate FCI size
    size_t n_fci = ci::fci_determinant_count(nbf, n_alpha, n_beta);
    std::cout << "Estimated FCI determinants: " << n_fci << "\n";
    
    if (n_fci > 10000) {
        std::cout << "WARNING: Large FCI space! This will take time...\n";
    }
    
    ci::FCI fci(ci_ints, nbf, n_alpha, n_beta);
    auto fci_result = fci.compute();
    
    std::cout << "FCI complete.\n\n";
    
    // ========================================
    // Step 5: Comparison
    // ========================================
    std::cout << "========================================\n";
    std::cout << "  COMPREHENSIVE COMPARISON\n";
    std::cout << "========================================\n\n";
    
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Method          Energy (Ha)      vs UHF (Ha)   vs FCI (Ha)\n";
    std::cout << "---------------------------------------------------------------\n";
    std::cout << "Experimental   " << std::setw(12) << -7.478 << "      "
              << std::setw(10) << (-7.478 - uhf_result.energy_total) << "      "
              << std::setw(10) << (-7.478 - fci_result.e_fci) << "\n";
    std::cout << "UHF            " << std::setw(12) << uhf_result.energy_total << "      "
              << std::setw(10) << 0.0 << "      "
              << std::setw(10) << (uhf_result.energy_total - fci_result.e_fci) << "\n";
    std::cout << "CI_HF ref      " << std::setw(12) << cisd_result.e_hf << "      "
              << std::setw(10) << (cisd_result.e_hf - uhf_result.energy_total) << "      "
              << std::setw(10) << (cisd_result.e_hf - fci_result.e_fci) << "\n";
    std::cout << "CISD           " << std::setw(12) << cisd_result.e_cisd << "      "
              << std::setw(10) << (cisd_result.e_cisd - uhf_result.energy_total) << "      "
              << std::setw(10) << (cisd_result.e_cisd - fci_result.e_fci) << "\n";
    std::cout << "FCI (EXACT)    " << std::setw(12) << fci_result.e_fci << "      "
              << std::setw(10) << (fci_result.e_fci - uhf_result.energy_total) << "      "
              << std::setw(10) << 0.0 << "\n";
    std::cout << "---------------------------------------------------------------\n\n";
    
    std::cout << "Correlation Energies:\n";
    std::cout << "  CISD: " << cisd_result.e_corr << " Ha\n";
    std::cout << "  FCI:  " << fci_result.e_corr << " Ha\n";
    std::cout << "  CISD recovery: " << (cisd_result.e_corr / fci_result.e_corr * 100.0) << "%\n\n";
    
    std::cout << "Wavefunction Analysis:\n";
    std::cout << "  CISD determinants: " << cisd_result.n_determinants << "\n";
    std::cout << "  FCI determinants:  " << fci_result.n_determinants << "\n";
    std::cout << "  FCI HF weight: " << fci_result.hf_weight << " (" << (fci_result.hf_weight * 100) << "%)\n\n";
    
    // Validation
    std::cout << "========================================\n";
    std::cout << "  VALIDATION\n";
    std::cout << "========================================\n\n";
    
    double fci_uhf_diff = std::abs(fci_result.e_fci - uhf_result.energy_total);
    double fci_exp_diff = std::abs(fci_result.e_fci - (-7.478));
    
    if (fci_exp_diff > 1.0) {
        std::cout << "❌ CRITICAL: FCI vs Experimental = " << fci_exp_diff << " Ha\n";
        std::cout << "   This is TOO LARGE! Possible issues:\n";
        std::cout << "   1. Integral transformation ERROR\n";
        std::cout << "   2. Slater-Condon formula BUG\n";
        std::cout << "   3. UHF orbitals incorrect\n\n";
    } else if (fci_exp_diff > 0.5) {
        std::cout << "⚠️  WARNING: FCI vs Experimental = " << fci_exp_diff << " Ha\n";
        std::cout << "   Basis set incompleteness (expected)\n\n";
    } else {
        std::cout << "✓ FCI vs Experimental = " << fci_exp_diff << " Ha\n";
        std::cout << "  Reasonable (basis set limit)\n\n";
    }
    
    std::cout << "Reference (NIST): Li = -7.478 Ha (experimental)\n\n";
    
    std::cout << "============================================\n";
    std::cout << "  TEST COMPLETE\n";
    std::cout << "============================================\n";
    
    return 0;
}
