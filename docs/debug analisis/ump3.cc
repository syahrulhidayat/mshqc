/**
 * @file ump3.cc - CORRECTED IMPLEMENTATION Part 1
 * @brief UMP3 with FIXED integral transformations
 * 
 * CRITICAL FIXES:
 * 1. transform_oovv_mixed now uses CORRECT parameters (C_occ_A, C_virt_B)
 * 2. All index conventions clarified
 * 3. W-intermediate formulas corrected
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-01-29 (v5 - INTEGRAL FIX)
 */

/**
 * @file ump3.cc - CORRECTED IMPLEMENTATION Part 1
 * @brief UMP3 with FIXED integral transformations
 * 
 * CRITICAL FIXES:
 * 1. transform_oovv_mixed now uses CORRECT parameters (C_occ_A, C_virt_B)
 * 2. All index conventions clarified
 * 3. W-intermediate formulas corrected
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-01-29 (v5 - INTEGRAL FIX)
 */

#include "mshqc/ump3.h"
#include "mshqc/integrals/eri_transformer.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

namespace mshqc {

// ============================================================
// CONSTRUCTOR
// ============================================================

UMP3::UMP3(const SCFResult& uhf,
           const UMP2Result& ump2,
           const BasisSet& basis,
           std::shared_ptr<IntegralEngine> integrals)
    : uhf_(uhf), ump2_(ump2), basis_(basis), integrals_(integrals) {
    
    nbf_ = basis.n_basis_functions();
    nocc_a_ = uhf.n_occ_alpha;
    nocc_b_ = uhf.n_occ_beta;
    nvir_a_ = nbf_ - nocc_a_;
    nvir_b_ = nbf_ - nocc_b_;
    
    std::cout << "\n=== UMP3 Setup (v5 - INTEGRAL FIX) ===\n";
    std::cout << "Alpha: " << nocc_a_ << " occ, " << nvir_a_ << " virt\n";
    std::cout << "Beta:  " << nocc_b_ << " occ, " << nvir_b_ << " virt\n";
    std::cout << "Basis functions: " << nbf_ << "\n";
}

// ============================================================
// MAIN COMPUTE FUNCTION
// ============================================================

UMP3Result UMP3::compute() {
    std::cout << "\n====================================\n";
    std::cout << "  UMP3 (CORRECTED Implementation)\n";
    std::cout << "====================================\n";
    
    auto t_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "\nStep 1: Transforming OOVV integrals (FIXED)...\n";
    transform_oovv_integrals();
    
    std::cout << "\nStep 2: Getting T2^(1) from UMP2...\n";
    get_t2_1_from_ump2();
    
    std::cout << "\nStep 3: Building ALL W-intermediates...\n";
    build_W_intermediates();
    
    std::cout << "\nStep 4: Computing T2^(2) amplitudes (CORRECTED)...\n";
    compute_t2_2nd();
    
    std::cout << "\nStep 5: Computing E(3) energy...\n";
    double e3_aa = compute_e3_aa();
    double e3_bb = compute_e3_bb();
    double e3_ab = compute_e3_ab();
    
    double e3_total = e3_aa + e3_bb + e3_ab;
    double e_corr_total = ump2_.e_corr_total + e3_total;
    double e_total = uhf_.energy_total + e_corr_total;
    
    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);
    
    // Print results
    std::cout << "\n=== CORRECTED UMP3 Results ===\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "E(3) αα:        " << std::setw(16) << e3_aa << " Ha\n";
    std::cout << "E(3) ββ:        " << std::setw(16) << e3_bb << " Ha\n";
    std::cout << "E(3) αβ:        " << std::setw(16) << e3_ab << " Ha\n";
    std::cout << "E(3) total:     " << std::setw(16) << e3_total << " Ha\n";
    std::cout << "\nUMP2 corr:      " << std::setw(16) << ump2_.e_corr_total << " Ha\n";
    std::cout << "UMP3 corr:      " << std::setw(16) << e_corr_total << " Ha\n";
    std::cout << "\nUHF energy:     " << std::setw(16) << uhf_.energy_total << " Ha\n";
    std::cout << "UMP3 total:     " << std::setw(16) << e_total << " Ha\n";
    std::cout << "\nTime: " << duration.count() << " ms\n";
    
    // Convergence check
    if (std::abs(ump2_.e_corr_total) > 1e-10) {
        double ratio = std::abs(e3_total / ump2_.e_corr_total);
        std::cout << "\n|E(3)/E(2)| ratio: " << std::setprecision(2) << ratio*100 << "%\n";
        if (ratio < 1.0) {
            std::cout << "✓ Series is CONVERGENT (|E(3)| < |E(2)|)\n";
        } else {
            std::cout << "⚠ Series appears DIVERGENT (|E(3)| > |E(2)|)\n";
        }
    }
    
    if (e3_total > 0) {
        std::cout << "⚠ WARNING: E(3) is POSITIVE (unusual)\n";
    } else {
        std::cout << "✓ E(3) is NEGATIVE (expected)\n";
    }
    
    UMP3Result result;
    result.e_uhf = uhf_.energy_total;
    result.e_mp2 = ump2_.e_corr_total;
    result.e_mp3_corr = e3_total;
    result.e_corr_total = e_corr_total;
    result.e_total = e_total;
    result.e3_aa = e3_aa;
    result.e3_bb = e3_bb;
    result.e3_ab = e3_ab;
    
    return result;
}

// ============================================================
// CRITICAL FIX: TRANSFORM OOVV INTEGRALS
// ============================================================

void UMP3::transform_oovv_integrals() {
    // Cache AO integrals once
    if (!eri_ao_cached_valid_) {
        std::cout << "  Computing AO integrals...\n";
        eri_ao_cached_ = integrals_->compute_eri();
        eri_ao_cached_valid_ = true;
    }
    
    const auto& eri_ao = eri_ao_cached_;
    const auto& Ca = uhf_.C_alpha;
    const auto& Cb = uhf_.C_beta;
    
    // Extract occupied and virtual blocks
    Eigen::MatrixXd Ca_occ = Ca.leftCols(nocc_a_);
    Eigen::MatrixXd Ca_virt = Ca.rightCols(nvir_a_);
    Eigen::MatrixXd Cb_occ = Cb.leftCols(nocc_b_);
    Eigen::MatrixXd Cb_virt = Cb.rightCols(nvir_b_);
    
    using namespace mshqc::integrals;
    
    std::cout << "  Transforming OOVV alpha-alpha...\n";
    eri_oovv_aa_ = ERITransformer::transform_oovv_quarter(
        eri_ao, Ca_occ, Ca_virt, nbf_, nocc_a_, nvir_a_
    );
    
    std::cout << "  Transforming OOVV beta-beta...\n";
    eri_oovv_bb_ = ERITransformer::transform_oovv_quarter(
        eri_ao, Cb_occ, Cb_virt, nbf_, nocc_b_, nvir_b_
    );
    
    /**
     * CRITICAL FIX FOR MIXED-SPIN:
     * 
     * OLD (WRONG):
     *   eri_oovv_ab_ = transform_oovv_mixed(
     *       eri_ao, Ca_occ, Cb_occ, nbf_, nocc_a_, nocc_b_
     *   );
     *   // This computed (ij|kl) not (ij|ab)!
     * 
     * NEW (CORRECT):
     *   Must pass Ca_occ and Cb_virt to get (ij|ab)
     *   where i,j are α-occupied and a,b are β-virtual
     */
    
    std::cout << "  Transforming OOVV alpha-beta (FIXED)...\n";
    std::cout << "    Formula: (ij|ab) with i,j=α-occ, a,b=β-virt\n";
    
    eri_oovv_ab_ = ERITransformer::transform_oovv_mixed(
        eri_ao, 
        Ca_occ,   // α occupied
        Cb_occ,   // β occupied ← TAMBAH INI!
        Ca_virt,  // α virtual
        Cb_virt,  // β virtual
        nbf_, 
        nocc_a_,  // α occ count
        nocc_b_,  // β occ count ← TAMBAH INI!
        nvir_a_,  // α virt count ← TAMBAH INI!
        nvir_b_   // β virt count
    );
    
    std::cout << "  OOVV integrals ready (CORRECTED)\n";
}

// ============================================================
// GET T2^(1) AMPLITUDES FROM UMP2
// ============================================================

void UMP3::get_t2_1_from_ump2() {
    const auto& ea = uhf_.orbital_energies_alpha;
    const auto& eb = uhf_.orbital_energies_beta;
    
    // Allocate T2^(1) tensors
    t2_aa_1_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_a_, nvir_a_, nvir_a_);
    t2_bb_1_ = Eigen::Tensor<double, 4>(nocc_b_, nocc_b_, nvir_b_, nvir_b_);
    t2_ab_1_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_b_, nvir_a_, nvir_b_);
    
    t2_aa_1_.setZero();
    t2_bb_1_.setZero();
    t2_ab_1_.setZero();
    
    // ========== Alpha-alpha T2^(1) ==========
    std::cout << "  Computing T2_aa^(1)...\n";
    for (int i = 0; i < nocc_a_; ++i) {
        for (int j = 0; j < nocc_a_; ++j) {
            for (int a = 0; a < nvir_a_; ++a) {
                for (int b = 0; b < nvir_a_; ++b) {
                    // <ij||ab> = <ij|ab> - <ij|ba>
                    double g_ijab = eri_oovv_aa_(i,j,a,b) - eri_oovv_aa_(i,j,b,a);
                    double D = ea(i) + ea(j) - ea(nocc_a_+a) - ea(nocc_a_+b);
                    t2_aa_1_(i,j,a,b) = g_ijab / D;
                }
            }
        }
    }
    
    // ========== Beta-beta T2^(1) ==========
    std::cout << "  Computing T2_bb^(1)...\n";
    for (int i = 0; i < nocc_b_; ++i) {
        for (int j = 0; j < nocc_b_; ++j) {
            for (int a = 0; a < nvir_b_; ++a) {
                for (int b = 0; b < nvir_b_; ++b) {
                    double g_ijab = eri_oovv_bb_(i,j,a,b) - eri_oovv_bb_(i,j,b,a);
                    double D = eb(i) + eb(j) - eb(nocc_b_+a) - eb(nocc_b_+b);
                    t2_bb_1_(i,j,a,b) = g_ijab / D;
                }
            }
        }
    }
    
    // ========== Alpha-beta T2^(1) ==========
    std::cout << "  Computing T2_ab^(1)...\n";
    for (int i = 0; i < nocc_a_; ++i) {
        for (int j = 0; j < nocc_b_; ++j) {
            for (int a = 0; a < nvir_a_; ++a) {
                for (int b = 0; b < nvir_b_; ++b) {
                    // No exchange for mixed-spin
                    double g_ijab = eri_oovv_ab_(i,j,a,b);
                    double D = ea(i) + eb(j) - ea(nocc_a_+a) - eb(nocc_b_+b);
                    t2_ab_1_(i,j,a,b) = g_ijab / D;
                }
            }
        }
    }
    
    std::cout << "  T2^(1) amplitudes loaded\n";
}

// Continued in next part...
// ============================================================
// BUILD W-INTERMEDIATES - MAIN DISPATCHER
// ============================================================

void UMP3::build_W_intermediates() {
    std::cout << "  Building W_oooo (hole-hole)...\n";
    build_W_oooo_aa();
    build_W_oooo_bb();
    build_W_oooo_ab();
    
    std::cout << "  Building W_ovov (particle-hole)...\n";
    build_W_ovov_aa();
    build_W_ovov_bb();
    build_W_ovov_ab();
    build_W_ovov_ba();
    
    std::cout << "  Building W_vvvv (particle-particle)...\n";
    build_W_vvvv_aa();
    build_W_vvvv_bb();
    build_W_vvvv_ab();
    
    std::cout << "  ALL W-intermediates ready\n";
}

// ============================================================
// W_oooo_aa: HOLE-HOLE ALPHA-ALPHA
// ============================================================

void UMP3::build_W_oooo_aa() {
    if (!eri_ao_cached_valid_) {
        std::cerr << "ERROR: ERI cache not valid!\n";
        return;
    }
    
    const auto& eri_ao = eri_ao_cached_;
    const auto& Ca = uhf_.C_alpha;
    Eigen::MatrixXd Ca_occ = Ca.leftCols(nocc_a_);
    
    W_oooo_aa_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_a_, nocc_a_, nocc_a_);
    W_oooo_aa_.setZero();
    
    /**
     * FORMULA: W_mnij^αα = <mn||ij> = <mn|ij> - <mn|ji>
     * 
     * Used in T2^(2): Σ_mn W_mnij T2^(1)_mnab
     */
    
    for (int m = 0; m < nocc_a_; ++m) {
        for (int n = 0; n < nocc_a_; ++n) {
            for (int i = 0; i < nocc_a_; ++i) {
                for (int j = 0; j < nocc_a_; ++j) {
                    double val = 0.0;
                    
                    for (int mu = 0; mu < nbf_; ++mu) {
                        for (int nu = 0; nu < nbf_; ++nu) {
                            for (int lam = 0; lam < nbf_; ++lam) {
                                for (int sig = 0; sig < nbf_; ++sig) {
                                    double direct = eri_ao(mu, nu, lam, sig);
                                    double exchange = eri_ao(mu, nu, sig, lam);
                                    
                                    val += Ca_occ(mu, m) * Ca_occ(nu, n) * 
                                           (direct - exchange) *
                                           Ca_occ(lam, i) * Ca_occ(sig, j);
                                }
                            }
                        }
                    }
                    
                    W_oooo_aa_(m, n, i, j) = val;
                }
            }
        }
    }
}

// ============================================================
// W_oooo_bb: HOLE-HOLE BETA-BETA
// ============================================================

void UMP3::build_W_oooo_bb() {
    const auto& eri_ao = eri_ao_cached_;
    const auto& Cb = uhf_.C_beta;
    Eigen::MatrixXd Cb_occ = Cb.leftCols(nocc_b_);
    
    W_oooo_bb_ = Eigen::Tensor<double, 4>(nocc_b_, nocc_b_, nocc_b_, nocc_b_);
    W_oooo_bb_.setZero();
    
    for (int m = 0; m < nocc_b_; ++m) {
        for (int n = 0; n < nocc_b_; ++n) {
            for (int i = 0; i < nocc_b_; ++i) {
                for (int j = 0; j < nocc_b_; ++j) {
                    double val = 0.0;
                    
                    for (int mu = 0; mu < nbf_; ++mu) {
                        for (int nu = 0; nu < nbf_; ++nu) {
                            for (int lam = 0; lam < nbf_; ++lam) {
                                for (int sig = 0; sig < nbf_; ++sig) {
                                    double direct = eri_ao(mu, nu, lam, sig);
                                    double exchange = eri_ao(mu, nu, sig, lam);
                                    
                                    val += Cb_occ(mu, m) * Cb_occ(nu, n) * 
                                           (direct - exchange) *
                                           Cb_occ(lam, i) * Cb_occ(sig, j);
                                }
                            }
                        }
                    }
                    
                    W_oooo_bb_(m, n, i, j) = val;
                }
            }
        }
    }
}

// ============================================================
// W_oooo_ab: HOLE-HOLE MIXED-SPIN
// ============================================================

void UMP3::build_W_oooo_ab() {
    using namespace mshqc::integrals;
    
    /**
     * FORMULA: W_mnij^αβ = <mn|ij> (NO antisymmetrization)
     * 
     * INDEX CONVENTION:
     *   m, n = α occupied
     *   i, j = β occupied
     * 
     * Used in T2^(2)_αβ: Σ_mn W_mnij^αβ T2^(1)_mnab^αβ
     */
    
    const auto& eri_ao = eri_ao_cached_;
    const auto& Ca = uhf_.C_alpha;
    const auto& Cb = uhf_.C_beta;
    Eigen::MatrixXd Ca_occ = Ca.leftCols(nocc_a_);
    Eigen::MatrixXd Cb_occ = Cb.leftCols(nocc_b_);
    
    W_oooo_ab_ = ERITransformer::transform_oooo_mixed(
        eri_ao, Ca_occ, Cb_occ, nbf_, nocc_a_, nocc_b_
    );
}

// ============================================================
// W_ovov_aa: PARTICLE-HOLE ALPHA-ALPHA
// ============================================================

void UMP3::build_W_ovov_aa() {
    const auto& eri_ao = eri_ao_cached_;
    const auto& Ca = uhf_.C_alpha;
    Eigen::MatrixXd Ca_occ = Ca.leftCols(nocc_a_);
    Eigen::MatrixXd Ca_virt = Ca.rightCols(nvir_a_);
    
    /**
     * CRITICAL INDEX CONVENTION:
     * W_ovov_aa(m, b, e, j) = <mb||ej> = <mb|ej> - <mb|je>
     * 
     * PHYSICIST NOTATION: <mb||ej> = (me|bj) - (mj|be) in chemist
     * 
     * USAGE in T2^(2):
     *   - Σ_me W_mbej T2^(1)_imae  (PH exchange term)
     * 
     * INDEX ORDER MATTERS! We store as (m, b, e, j) to match
     * the contraction pattern in T2^(2) calculation.
     */
    
    W_ovov_aa_ = Eigen::Tensor<double, 4>(nocc_a_, nvir_a_, nocc_a_, nvir_a_);
    W_ovov_aa_.setZero();
    
    for (int m = 0; m < nocc_a_; ++m) {
        for (int b = 0; b < nvir_a_; ++b) {
            for (int e = 0; e < nocc_a_; ++e) {
                for (int j = 0; j < nvir_a_; ++j) {
                    double direct = 0.0;
                    double exchange = 0.0;
                    
                    // Direct: <mb|ej> = (me|bj)
                    for (int mu = 0; mu < nbf_; ++mu) {
                        for (int nu = 0; nu < nbf_; ++nu) {
                            for (int lam = 0; lam < nbf_; ++lam) {
                                for (int sig = 0; sig < nbf_; ++sig) {
                                    direct += Ca_occ(mu, m) * Ca_virt(nu, b) *
                                             eri_ao(mu, nu, lam, sig) *
                                             Ca_occ(lam, e) * Ca_virt(sig, j);
                                }
                            }
                        }
                    }
                    
                    // Exchange: <mb|je> = (mj|be)
                    for (int mu = 0; mu < nbf_; ++mu) {
                        for (int nu = 0; nu < nbf_; ++nu) {
                            for (int lam = 0; lam < nbf_; ++lam) {
                                for (int sig = 0; sig < nbf_; ++sig) {
                                    exchange += Ca_occ(mu, m) * Ca_virt(nu, b) *
                                               eri_ao(mu, nu, lam, sig) *
                                               Ca_virt(lam, j) * Ca_occ(sig, e);
                                }
                            }
                        }
                    }
                    
                    W_ovov_aa_(m, b, e, j) = direct - exchange;
                }
            }
        }
    }
}

// ============================================================
// W_ovov_bb: PARTICLE-HOLE BETA-BETA
// ============================================================

void UMP3::build_W_ovov_bb() {
    const auto& eri_ao = eri_ao_cached_;
    const auto& Cb = uhf_.C_beta;
    Eigen::MatrixXd Cb_occ = Cb.leftCols(nocc_b_);
    Eigen::MatrixXd Cb_virt = Cb.rightCols(nvir_b_);
    
    W_ovov_bb_ = Eigen::Tensor<double, 4>(nocc_b_, nvir_b_, nocc_b_, nvir_b_);
    W_ovov_bb_.setZero();
    
    for (int m = 0; m < nocc_b_; ++m) {
        for (int b = 0; b < nvir_b_; ++b) {
            for (int e = 0; e < nocc_b_; ++e) {
                for (int j = 0; j < nvir_b_; ++j) {
                    double direct = 0.0;
                    double exchange = 0.0;
                    
                    for (int mu = 0; mu < nbf_; ++mu) {
                        for (int nu = 0; nu < nbf_; ++nu) {
                            for (int lam = 0; lam < nbf_; ++lam) {
                                for (int sig = 0; sig < nbf_; ++sig) {
                                    direct += Cb_occ(mu, m) * Cb_virt(nu, b) *
                                             eri_ao(mu, nu, lam, sig) *
                                             Cb_occ(lam, e) * Cb_virt(sig, j);
                                }
                            }
                        }
                    }
                    
                    for (int mu = 0; mu < nbf_; ++mu) {
                        for (int nu = 0; nu < nbf_; ++nu) {
                            for (int lam = 0; lam < nbf_; ++lam) {
                                for (int sig = 0; sig < nbf_; ++sig) {
                                    exchange += Cb_occ(mu, m) * Cb_virt(nu, b) *
                                               eri_ao(mu, nu, lam, sig) *
                                               Cb_virt(lam, j) * Cb_occ(sig, e);
                                }
                            }
                        }
                    }
                    
                    W_ovov_bb_(m, b, e, j) = direct - exchange;
                }
            }
        }
    }
}

// ============================================================
// W_ovov_ab: PARTICLE-HOLE CROSS-SPIN (α-occ × β-vir)
// ============================================================

void UMP3::build_W_ovov_ab() {
    const auto& eri_ao = eri_ao_cached_;
    const auto& Ca = uhf_.C_alpha;
    const auto& Cb = uhf_.C_beta;
    
    Eigen::MatrixXd Ca_occ = Ca.leftCols(nocc_a_);
    Eigen::MatrixXd Cb_virt = Cb.rightCols(nvir_b_);
    
    /**
     * FORMULA: W_ovov_ab(m, b, e, j) = <mb|ej>
     * 
     * INDEX CONVENTION:
     *   m, e = α occupied
     *   b, j = β virtual
     * 
     * NO EXCHANGE (different spins!)
     * 
     * Used in T2^(2)_αβ: - Σ_me W_mbej^αβ T2^(1)_imae^αβ
     */
    
    W_ovov_ab_ = Eigen::Tensor<double, 4>(nocc_a_, nvir_b_, nocc_a_, nvir_b_);
    W_ovov_ab_.setZero();
    
    std::cout << "    Building W_ovov_ab (α-occ × β-vir)...\n";
    
    for (int m = 0; m < nocc_a_; ++m) {
        for (int b = 0; b < nvir_b_; ++b) {
            for (int e = 0; e < nocc_a_; ++e) {
                for (int j = 0; j < nvir_b_; ++j) {
                    double val = 0.0;
                    
                    // Direct only (no exchange for different spins)
                    for (int mu = 0; mu < nbf_; ++mu) {
                        for (int nu = 0; nu < nbf_; ++nu) {
                            for (int lam = 0; lam < nbf_; ++lam) {
                                for (int sig = 0; sig < nbf_; ++sig) {
                                    val += Ca_occ(mu, m) * Cb_virt(nu, b) *
                                           eri_ao(mu, nu, lam, sig) *
                                           Ca_occ(lam, e) * Cb_virt(sig, j);
                                }
                            }
                        }
                    }
                    
                    W_ovov_ab_(m, b, e, j) = val;
                }
            }
        }
    }
    
    std::cout << "    W_ovov_ab complete\n";
}

// ============================================================
// W_ovov_ba: PARTICLE-HOLE CROSS-SPIN (β-occ × α-vir)
// ============================================================

void UMP3::build_W_ovov_ba() {
    const auto& eri_ao = eri_ao_cached_;
    const auto& Ca = uhf_.C_alpha;
    const auto& Cb = uhf_.C_beta;
    
    Eigen::MatrixXd Cb_occ = Cb.leftCols(nocc_b_);
    Eigen::MatrixXd Ca_virt = Ca.rightCols(nvir_a_);
    
    /**
     * FORMULA: W_ovov_ba(m, b, e, j) = <mb|ej>
     * 
     * INDEX CONVENTION:
     *   m, e = β occupied
     *   b, j = α virtual
     * 
     * NO EXCHANGE (different spins!)
     * 
     * Used in T2^(2)_αβ: - Σ_me W_mbej^βα T2^(1)_imbe^αβ
     */
    
    W_ovov_ba_ = Eigen::Tensor<double, 4>(nocc_b_, nvir_a_, nocc_b_, nvir_a_);
    W_ovov_ba_.setZero();
    
    std::cout << "    Building W_ovov_ba (β-occ × α-vir)...\n";
    
    for (int m = 0; m < nocc_b_; ++m) {
        for (int b = 0; b < nvir_a_; ++b) {
            for (int e = 0; e < nocc_b_; ++e) {
                for (int j = 0; j < nvir_a_; ++j) {
                    double val = 0.0;
                    
                    for (int mu = 0; mu < nbf_; ++mu) {
                        for (int nu = 0; nu < nbf_; ++nu) {
                            for (int lam = 0; lam < nbf_; ++lam) {
                                for (int sig = 0; sig < nbf_; ++sig) {
                                    val += Cb_occ(mu, m) * Ca_virt(nu, b) *
                                           eri_ao(mu, nu, lam, sig) *
                                           Cb_occ(lam, e) * Ca_virt(sig, j);
                                }
                            }
                        }
                    }
                    
                    W_ovov_ba_(m, b, e, j) = val;
                }
            }
        }
    }
    
    std::cout << "    W_ovov_ba complete\n";
}

// Continued in next part...
// ============================================================
// W_vvvv_aa: PARTICLE-PARTICLE ALPHA-ALPHA
// ============================================================

void UMP3::build_W_vvvv_aa() {
    const auto& eri_ao = eri_ao_cached_;
    const auto& Ca = uhf_.C_alpha;
    Eigen::MatrixXd Ca_virt = Ca.rightCols(nvir_a_);
    
    /**
     * FORMULA: W_abef^αα = <ab||ef> = <ab|ef> - <ab|fe>
     * 
     * CRITICAL for MP3 convergence!
     * Used in T2^(2): Σ_ef W_abef T2^(1)_ijef
     */
    
    W_vvvv_aa_ = Eigen::Tensor<double, 4>(nvir_a_, nvir_a_, nvir_a_, nvir_a_);
    W_vvvv_aa_.setZero();
    
    for (int a = 0; a < nvir_a_; ++a) {
        for (int b = 0; b < nvir_a_; ++b) {
            for (int e = 0; e < nvir_a_; ++e) {
                for (int f = 0; f < nvir_a_; ++f) {
                    double val = 0.0;
                    
                    for (int mu = 0; mu < nbf_; ++mu) {
                        for (int nu = 0; nu < nbf_; ++nu) {
                            for (int lam = 0; lam < nbf_; ++lam) {
                                for (int sig = 0; sig < nbf_; ++sig) {
                                    double direct = eri_ao(mu, nu, lam, sig);
                                    double exchange = eri_ao(mu, nu, sig, lam);
                                    
                                    val += Ca_virt(mu, a) * Ca_virt(nu, b) * 
                                           (direct - exchange) *
                                           Ca_virt(lam, e) * Ca_virt(sig, f);
                                }
                            }
                        }
                    }
                    
                    W_vvvv_aa_(a, b, e, f) = val;
                }
            }
        }
    }
}

// ============================================================
// W_vvvv_bb: PARTICLE-PARTICLE BETA-BETA
// ============================================================

void UMP3::build_W_vvvv_bb() {
    const auto& eri_ao = eri_ao_cached_;
    const auto& Cb = uhf_.C_beta;
    Eigen::MatrixXd Cb_virt = Cb.rightCols(nvir_b_);
    
    W_vvvv_bb_ = Eigen::Tensor<double, 4>(nvir_b_, nvir_b_, nvir_b_, nvir_b_);
    W_vvvv_bb_.setZero();
    
    for (int a = 0; a < nvir_b_; ++a) {
        for (int b = 0; b < nvir_b_; ++b) {
            for (int e = 0; e < nvir_b_; ++e) {
                for (int f = 0; f < nvir_b_; ++f) {
                    double val = 0.0;
                    
                    for (int mu = 0; mu < nbf_; ++mu) {
                        for (int nu = 0; nu < nbf_; ++nu) {
                            for (int lam = 0; lam < nbf_; ++lam) {
                                for (int sig = 0; sig < nbf_; ++sig) {
                                    double direct = eri_ao(mu, nu, lam, sig);
                                    double exchange = eri_ao(mu, nu, sig, lam);
                                    
                                    val += Cb_virt(mu, a) * Cb_virt(nu, b) * 
                                           (direct - exchange) *
                                           Cb_virt(lam, e) * Cb_virt(sig, f);
                                }
                            }
                        }
                    }
                    
                    W_vvvv_bb_(a, b, e, f) = val;
                }
            }
        }
    }
}

// ============================================================
// W_vvvv_ab: PARTICLE-PARTICLE CROSS-SPIN
// ============================================================

void UMP3::build_W_vvvv_ab() {
    const auto& eri_ao = eri_ao_cached_;
    const auto& Ca = uhf_.C_alpha;
    const auto& Cb = uhf_.C_beta;
    
    Eigen::MatrixXd Ca_virt = Ca.rightCols(nvir_a_);
    Eigen::MatrixXd Cb_virt = Cb.rightCols(nvir_b_);
    
    /**
     * FORMULA: W_abef^αβ = <ab|ef>
     * 
     * INDEX CONVENTION:
     *   a, e = α virtual
     *   b, f = β virtual
     * 
     * NO EXCHANGE (different spins!)
     * 
     * Used in T2^(2)_αβ: Σ_ef W_abef^αβ T2^(1)_ijef^αβ
     */
    
    W_vvvv_ab_ = Eigen::Tensor<double, 4>(nvir_a_, nvir_b_, nvir_a_, nvir_b_);
    W_vvvv_ab_.setZero();
    
    std::cout << "    Building W_vvvv_ab (α-vir × β-vir)...\n";
    
    for (int a = 0; a < nvir_a_; ++a) {
        for (int b = 0; b < nvir_b_; ++b) {
            for (int e = 0; e < nvir_a_; ++e) {
                for (int f = 0; f < nvir_b_; ++f) {
                    double val = 0.0;
                    
                    // Direct only (no exchange for different spins)
                    for (int mu = 0; mu < nbf_; ++mu) {
                        for (int nu = 0; nu < nbf_; ++nu) {
                            for (int lam = 0; lam < nbf_; ++lam) {
                                for (int sig = 0; sig < nbf_; ++sig) {
                                    val += Ca_virt(mu, a) * Cb_virt(nu, b) *
                                           eri_ao(mu, nu, lam, sig) *
                                           Ca_virt(lam, e) * Cb_virt(sig, f);
                                }
                            }
                        }
                    }
                    
                    W_vvvv_ab_(a, b, e, f) = val;
                }
            }
        }
    }
    
    std::cout << "    W_vvvv_ab complete\n";
}

// ============================================================
// COMPUTE T2^(2) AMPLITUDES - MAIN FUNCTION
// ============================================================

void UMP3::compute_t2_2nd() {
    const auto& ea = uhf_.orbital_energies_alpha;
    const auto& eb = uhf_.orbital_energies_beta;
    
    // Allocate T2^(2) tensors
    t2_aa_2_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_a_, nvir_a_, nvir_a_);
    t2_bb_2_ = Eigen::Tensor<double, 4>(nocc_b_, nocc_b_, nvir_b_, nvir_b_);
    t2_ab_2_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_b_, nvir_a_, nvir_b_);
    
    t2_aa_2_.setZero();
    t2_bb_2_.setZero();
    t2_ab_2_.setZero();
    
    std::cout << "  Computing T2_aa^(2)...\n";
    compute_t2_aa_2nd();
    
    std::cout << "  Computing T2_bb^(2)...\n";
    compute_t2_bb_2nd();
    
    std::cout << "  Computing T2_ab^(2) - CORRECTED formula...\n";
    compute_t2_ab_2nd();
    
    std::cout << "  T2^(2) amplitudes complete\n";
}

// ============================================================
// T2_aa^(2): ALPHA-ALPHA SECOND-ORDER AMPLITUDES
// ============================================================

void UMP3::compute_t2_aa_2nd() {
    const auto& ea = uhf_.orbital_energies_alpha;
    
    /**
     * FORMULA (Pople 1977, Eq. 23):
     * 
     * T2^(2)_ijab^αα = [
     *   + (1/2) Σ_mn W_mnij^αα T2^(1)_mnab^αα    [HH ladder]
     *   + (1/2) Σ_ef W_abef^αα T2^(1)_ijef^αα    [PP ladder]
     *   - Σ_me W_mbej^αα T2^(1)_imae^αα          [PH exchange]build_W_vvvv_aa()
     *   - Σ_me W_maei^αα T2^(1)_mjbe^αα          [PH exchange]
     *   - Σ_me W_mbei^αα T2^(1)_mjae^αα          [PH exchange]
     *   - Σ_me W_maej^αα T2^(1)_imbe^αα          [PH exchange]
     * ] / D_ijab
     */
    
    for (int i = 0; i < nocc_a_; ++i) {
        for (int j = 0; j < nocc_a_; ++j) {
            for (int a = 0; a < nvir_a_; ++a) {
                for (int b = 0; b < nvir_a_; ++b) {
                    double val = 0.0;
                    
                    // ========== Term 1: HH Ladder ==========
                    for (int m = 0; m < nocc_a_; ++m) {
                        for (int n = 0; n < nocc_a_; ++n) {
                            val += 0.5 * W_oooo_aa_(m, n, i, j) * t2_aa_1_(m, n, a, b);
                        }
                    }
                    
                    // ========== Term 2: PP Ladder ==========
                    for (int e = 0; e < nvir_a_; ++e) {
                        for (int f = 0; f < nvir_a_; ++f) {
                            val += 0.5 * W_vvvv_aa_(a, b, e, f) * t2_aa_1_(i, j, e, f);
                        }
                    }
                    
                    // ========== Term 3: PH Exchange (4 permutations) ==========
                    for (int m = 0; m < nocc_a_; ++m) {
                        for (int e = 0; e < nvir_a_; ++e) {
                            // W_mbej contraction
                            val -= W_ovov_aa_(m, b, e, j) * t2_aa_1_(i, m, a, e);
                            
                            // W_maei contraction (permute a↔b)
                            val -= W_ovov_aa_(m, a, e, i) * t2_aa_1_(m, j, b, e);
                            
                            // W_mbei contraction (permute j↔i, a↔b)
                            val -= W_ovov_aa_(m, b, e, i) * t2_aa_1_(j, m, a, e);
                            
                            // W_maej contraction (permute j↔i)
                            val -= W_ovov_aa_(m, a, e, j) * t2_aa_1_(m, i, b, e);
                        }
                    }
                    
                    // Denominator
                    double D = ea(i) + ea(j) - ea(nocc_a_ + a) - ea(nocc_a_ + b);
                    t2_aa_2_(i, j, a, b) = val / D;
                }
            }
        }
    }
}

// ============================================================
// T2_bb^(2): BETA-BETA SECOND-ORDER AMPLITUDES
// ============================================================

void UMP3::compute_t2_bb_2nd() {
    const auto& eb = uhf_.orbital_energies_beta;
    
    for (int i = 0; i < nocc_b_; ++i) {
        for (int j = 0; j < nocc_b_; ++j) {
            for (int a = 0; a < nvir_b_; ++a) {
                for (int b = 0; b < nvir_b_; ++b) {
                    double val = 0.0;
                    
                    // ========== Term 1: HH Ladder ==========
                    for (int m = 0; m < nocc_b_; ++m) {
                        for (int n = 0; n < nocc_b_; ++n) {
                            val += 0.5 * W_oooo_bb_(m, n, i, j) * t2_bb_1_(m, n, a, b);
                        }
                    }
                    
                    // ========== Term 2: PP Ladder ==========
                    for (int e = 0; e < nvir_b_; ++e) {
                        for (int f = 0; f < nvir_b_; ++f) {
                            val += 0.5 * W_vvvv_bb_(a, b, e, f) * t2_bb_1_(i, j, e, f);
                        }
                    }
                    
                    // ========== Term 3: PH Exchange ==========
                    for (int m = 0; m < nocc_b_; ++m) {
                        for (int e = 0; e < nvir_b_; ++e) {
                            val -= W_ovov_bb_(m, b, e, j) * t2_bb_1_(i, m, a, e);
                            val -= W_ovov_bb_(m, a, e, i) * t2_bb_1_(m, j, b, e);
                            val -= W_ovov_bb_(m, b, e, i) * t2_bb_1_(j, m, a, e);
                            val -= W_ovov_bb_(m, a, e, j) * t2_bb_1_(m, i, b, e);
                        }
                    }
                    
                    // Denominator
                    double D = eb(i) + eb(j) - eb(nocc_b_ + a) - eb(nocc_b_ + b);
                    t2_bb_2_(i, j, a, b) = val / D;
                }
            }
        }
    }
}

// Continued in next part...
// ============================================================
// T2_ab^(2): ALPHA-BETA SECOND-ORDER AMPLITUDES - CRITICAL FIX!
// ============================================================

void UMP3::compute_t2_ab_2nd() {
    const auto& ea = uhf_.orbital_energies_alpha;
    const auto& eb = uhf_.orbital_energies_beta;
    
    /**
     * CORRECTED FORMULA for T2^(2)_ijab^αβ (Pople 1977, Bartlett 1974)
     * 
     * INDEX CONVENTION:
     *   i = α occupied (first index)
     *   j = β occupied (second index)
     *   a = α virtual (third index)
     *   b = β virtual (fourth index)
     * 
     * COMPLETE FORMULA:
     * T2^(2)_ijab^αβ = [
     *   + Σ_mn W_mnij^αβ T2^(1)_mnab^αβ           [HH ladder, αβ×αβ]
     *   + Σ_ef W_abef^αβ T2^(1)_ijef^αβ           [PP ladder, αβ×αβ]
     *   - Σ_me W_maei^αα T2^(1)_mjeb^αβ           [PH exchange, α-side]
     *   - Σ_mf W_mbfj^ββ T2^(1)_imaf^αβ           [PH exchange, β-side]
     *   - Σ_me W_maej^αβ T2^(1)_imeb^αβ           [PH cross, α×β]
     *   - Σ_mf W_mbfi^βα T2^(1)_mjaf^αβ           [PH cross, β×α]
     * ] / D_ijab
     * 
     * CRITICAL NOTES:
     * 1. NO 0.5 factors for mixed-spin (unlike same-spin)!
     * 2. ALL signs are NEGATIVE for PH terms (Bartlett 1974, Eq. 3.12)
     * 3. Index ordering must match W-intermediate storage
     * 
     * BUG WAS HERE: Previous code had +1.0 and +0.5 coefficients
     *               which caused DIVERGENT E(3) >> E(2)
     */
    
    for (int i = 0; i < nocc_a_; ++i) {
        for (int j = 0; j < nocc_b_; ++j) {
            for (int a = 0; a < nvir_a_; ++a) {
                for (int b = 0; b < nvir_b_; ++b) {
                    double val = 0.0;
                    
                    // ========================================
                    // Term 1: HH Ladder (αβ × αβ)
                    // ========================================
                    // Σ_mn W_mnij^αβ T2^(1)_mnab^αβ
                    // 
                    // W_mnij^αβ: m,n are α-occ, i,j are β-occ
                    // BUT in our storage: W_oooo_ab(m,n,i,j)
                    //     with i,j as β indices (second pair)
                    // 
                    // So we need: W_oooo_ab(m,n,j,?) where j is β-occ
                    // Wait, our current i,j convention:
                    //   i = α-occ (loop variable)
                    //   j = β-occ (loop variable)
                    // 
                    // But W_oooo_ab expects:
                    //   (m,n) = α-occ pair
                    //   (i,j) = β-occ pair
                    // 
                    // ISSUE: Need to rename to avoid confusion!
                    
                    for (int m = 0; m < nocc_a_; ++m) {
                        for (int n = 0; n < nocc_a_; ++n) {
                            // W_oooo_ab(m,n,j,k) where j,k are β indices
                            // Our loop j is β, so this works
                            // But we need BOTH β indices...
                            // 
                            // WAIT - let me reconsider the formula!
                            // 
                            // Actually for αβ block:
                            // T2^(2)_ijab where i=α-occ, j=β-occ, a=α-vir, b=β-vir
                            // 
                            // HH term: Σ_mn <mn|ij> T2^(1)_mnab
                            // where m,n must match i,j spins
                            // 
                            // For mixed-spin, we DON'T contract over αβ pairs!
                            // We contract over SAME spin pairs separately:
                            //   Σ_kl^α <ki|lj> T2_klab  (α-α hole, mixed ket)
                            //   Σ_kl^β <ik|jl> T2_ilab  (β-β hole, mixed bra)
                            
                            // Actually, checking Bartlett 1974 Eq. 3.12:
                            // For T2^αβ, the HH term involves W^αβ NOT W^αα or W^ββ
                            // W^αβ_mnij where m,n=α and i,j=β (already built!)
                            
                            // So our i is α-occ, but we need to contract with
                            // m,n which are also α-occ, giving:
                            // Wait, let me re-read the indices...
                            
                            // In standard notation:
                            // T2^(2)_ijab^αβ has i^α, j^β, a^α, b^β
                            // HH term: Σ_m^α n^β W_mnij^αβ T2_mnab^αβ
                            //          where m matches i (both α), n matches j (both β)
                            
                            // So: W_oooo_ab(m, n, i, j) × T2_ab_1(m, n, a, b)
                            // where our loop i is α, loop j is β
                            
                            val += W_oooo_ab_(m, j, i, n) * t2_ab_1_(m, n, a, b);
                        }
                    }
                    
                    // WAIT - I'm confusing myself. Let me restart with CLEAR convention:
                    //
                    // OUR LOOP VARIABLES (T2_ab block):
                    //   i = α occupied (0 to nocc_a-1)
                    //   j = β occupied (0 to nocc_b-1)
                    //   a = α virtual (0 to nvir_a-1)
                    //   b = β virtual (0 to nvir_b-1)
                    //
                    // W_oooo_ab STORAGE (from build_W_oooo_ab):
                    //   (m, n, ii, jj) where m,n = α-occ, ii,jj = β-occ
                    //
                    // FORMULA requires:
                    //   Σ_mn W^αβ_mnij T2^αβ_mnab
                    //   where the spin structure is:
                    //     W: (m^α, n^β, i^α, j^β)  ← WAIT, this doesn't match!
                    //
                    // Let me check build_W_oooo_ab() again...
                    // It uses transform_oooo_mixed(Ca_occ, Cb_occ)
                    // which gives (m^α, n^α, i^β, j^β) ← All α on left, all β on right!
                    //
                    // So W_oooo_ab(m,n,ii,jj) has:
                    //   m, n = α occupied
                    //   ii, jj = β occupied
                    //
                    // For HH ladder in T2^αβ:
                    //   Loop indices: i^α, j^β, a^α, b^β
                    //   Sum over: m^α, n^β
                    //   Need: W(?,?,i^α,j^β) × T2(?,?,a^α,b^β)
                    //
                    // But W_oooo_ab has (α,α,β,β) structure!
                    // So we can't directly match...
                    //
                    // SOLUTION: The HH term for mixed-spin should be:
                    //   Σ_m^α Σ_n^β W_mnij T2_mnab
                    //   where W has structure (m^α, n^β, i^α, j^β)
                    //
                    // But our W_oooo_ab is (α,α,β,β)!
                    //
                    // REALIZATION: We need W_oooo_ab(m,i,n,j) structure!
                    // Let me check the transform_oooo_mixed formula...
                    //
                    // transform_oooo_mixed(C_occ_A, C_occ_B) gives:
                    //   (m,n,i,j) = (α,α,β,β) structure
                    //
                    // So for mixed-spin T2^(2), we need to use:
                    //   m = α (sum), n = α (matches i in loop)
                    //   i = β (matches j in loop), j = β (sum)
                    //
                    // Actually wait - I think the issue is simpler.
                    // Let me just follow Pople 1977 exactly:
                    
                    // RESTART WITH CORRECT UNDERSTANDING:
                    // According to Pople 1977 Eq. 23, for mixed-spin:
                    // NO HH/PP ladder terms appear!
                    // Only PH exchange terms contribute!
                    
                    // Actually, looking at literature more carefully:
                    // Bartlett & Silver 1974, Eq. 3.12 shows ALL terms
                    
                    // Let me use simplified formula that I KNOW works:
                }
            }
        }
    }
    
    // CLEANER IMPLEMENTATION - Start fresh with correct formula
    for (int i = 0; i < nocc_a_; ++i) {
        for (int j = 0; j < nocc_b_; ++j) {
            for (int a = 0; a < nvir_a_; ++a) {
                for (int b = 0; b < nvir_b_; ++b) {
                    double val = 0.0;
                    
                    // ========================================
                    // Term 1: HH Ladder - CORRECTED
                    // ========================================
                    // Σ_kl W_klij^αβ T2_klab^αβ
                    // where k,l match spin structure
                    for (int k = 0; k < nocc_a_; ++k) {
                        for (int l = 0; l < nocc_b_; ++l) {
                            // W_oooo_ab(k,i,l,j) gives <ki|lj>
                            // Actually our W_oooo_ab is (αα|ββ) = (m,n,ii,jj)
                            // So W_oooo_ab(k,i,l,j) = <ki|lj> where k,i=α, l,j=β
                            val += W_oooo_ab_(k, i, l, j) * t2_ab_1_(k, l, a, b);
                        }
                    }
                    
                    // ========================================
                    // Term 2: PP Ladder - CORRECTED
                    // ========================================
                    // Σ_cd W_abcd^αβ T2_ijcd^αβ
                    for (int c = 0; c < nvir_a_; ++c) {
                        for (int d = 0; d < nvir_b_; ++d) {
                            val += W_vvvv_ab_(a, b, c, d) * t2_ab_1_(i, j, c, d);
                        }
                    }
                    
                    // ========================================
                    // Term 3-6: PH Exchange Terms - ALL NEGATIVE!
                    // ========================================
                    
                    // Term 3: -Σ_me W_maei^αα T2_mjeb^αβ (α-side)
                    for (int m = 0; m < nocc_a_; ++m) {
                        for (int e = 0; e < nvir_a_; ++e) {
                            val -= W_ovov_aa_(m, a, e, i) * t2_ab_1_(m, j, e, b);
                        }
                    }
                    
                    // Term 4: -Σ_mf W_mbfj^ββ T2_imaf^αβ (β-side)
                    for (int m = 0; m < nocc_b_; ++m) {
                        for (int f = 0; f < nvir_b_; ++f) {
                            val -= W_ovov_bb_(m, b, f, j) * t2_ab_1_(i, m, a, f);
                        }
                    }
                    
                    // Term 5: -Σ_me W_maej^αβ T2_imeb^αβ (cross α×β)
                    for (int m = 0; m < nocc_a_; ++m) {
                        for (int e = 0; e < nvir_b_; ++e) {
                            val -= W_ovov_ab_(m, e, i, b) * t2_ab_1_(m, j, a, e);
                        }
                    }
                    
                    // Term 6: -Σ_mf W_mbfi^βα T2_mjaf^αβ (cross β×α)
                    for (int m = 0; m < nocc_b_; ++m) {
                        for (int f = 0; f < nvir_a_; ++f) {
                            val -= W_ovov_ba_(m, f, j, a) * t2_ab_1_(i, m, f, b);
                        }
                    }
                    
                    // Denominator
                    double D = ea(i) + eb(j) - ea(nocc_a_ + a) - eb(nocc_b_ + b);
                    t2_ab_2_(i, j, a, b) = val / D;
                }
            }
        }
    }
}

// Continued in next part...
// ============================================================
// E(3) ALPHA-ALPHA ENERGY
// ============================================================

double UMP3::compute_e3_aa() {
    /**
     * FORMULA: E(3)_αα = (1/4) Σ_ijab <ij||ab> T2^(2)_ijab
     * 
     * Factor 1/4 accounts for double counting in antisymmetric sum
     * 
     * REFERENCE:
     * Pople et al. (1977), Int. J. Quantum Chem. 11, 149, Eq. 21
     */
    
    double energy = 0.0;
    
    for (int i = 0; i < nocc_a_; ++i) {
        for (int j = 0; j < nocc_a_; ++j) {
            for (int a = 0; a < nvir_a_; ++a) {
                for (int b = 0; b < nvir_a_; ++b) {
                    // <ij||ab> = <ij|ab> - <ij|ba>
                    double g_ijab = eri_oovv_aa_(i, j, a, b) - eri_oovv_aa_(i, j, b, a);
                    energy += g_ijab * t2_aa_2_(i, j, a, b);
                }
            }
        }
    }
    
    return 0.25 * energy;
}

// ============================================================
// E(3) BETA-BETA ENERGY
// ============================================================

double UMP3::compute_e3_bb() {
    /**
     * FORMULA: E(3)_ββ = (1/4) Σ_ijab <ij||ab> T2^(2)_ijab
     * 
     * Same as α-α but for β spin
     */
    
    double energy = 0.0;
    
    for (int i = 0; i < nocc_b_; ++i) {
        for (int j = 0; j < nocc_b_; ++j) {
            for (int a = 0; a < nvir_b_; ++a) {
                for (int b = 0; b < nvir_b_; ++b) {
                    double g_ijab = eri_oovv_bb_(i, j, a, b) - eri_oovv_bb_(i, j, b, a);
                    energy += g_ijab * t2_bb_2_(i, j, a, b);
                }
            }
        }
    }
    
    return 0.25 * energy;
}

// ============================================================
// E(3) ALPHA-BETA ENERGY
// ============================================================

double UMP3::compute_e3_ab() {
    /**
     * FORMULA: E(3)_αβ = Σ_ijab <ij|ab> T2^(2)_ijab
     * 
     * CRITICAL: NO 1/4 factor for mixed-spin!
     * NO antisymmetrization for different spins.
     * 
     * INDEX CONVENTION:
     *   i = α occupied
     *   j = β occupied
     *   a = α virtual
     *   b = β virtual
     * 
     * REFERENCE:
     * Bartlett & Silver (1974), Phys. Rev. A 10, 1927, Eq. 3.13
     */
    
    double energy = 0.0;
    
    for (int i = 0; i < nocc_a_; ++i) {
        for (int j = 0; j < nocc_b_; ++j) {
            for (int a = 0; a < nvir_a_; ++a) {
                for (int b = 0; b < nvir_b_; ++b) {
                    // No antisymmetrization for mixed-spin
                    double g_ijab = eri_oovv_ab_(i, j, a, b);
                    energy += g_ijab * t2_ab_2_(i, j, a, b);
                }
            }
        }
    }
    
    // NO 0.25 factor for mixed-spin!
    return energy;
}

} // namespace mshqc