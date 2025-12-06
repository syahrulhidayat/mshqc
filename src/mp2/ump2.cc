/**
 * @file ump2.cc
 * @brief Unrestricted Møller-Plesset 2nd-order perturbation theory (UMP2)
 * 
 * Implementation of UMP2 energy and T2^(1) amplitudes for open-shell systems.
 * Separate treatment for α-α, β-β, and α-β spin components.
 * 
 * Theory References:
 *   - C. Møller & M. S. Plesset, Phys. Rev. 46, 618 (1934)
 *     [Original MP2 perturbation theory]
 *   - J. A. Pople et al., J. Chem. Phys. 64, 2901 (1976)
 *     [Unrestricted MP2 formalism for open-shell systems]
 *   - J. A. Pople et al., Int. J. Quantum Chem. 10, 1 (1976)
 *     [UMP2 implementation details]
 *   - A. Szabo & N. S. Ostlund, "Modern Quantum Chemistry" (1996)
 *     [Textbook reference for integral transformations]
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-01-29
 * @license MIT License (see LICENSE file in project root)
 * 
 * @note This is an original implementation derived from published theory.
 *       No code was copied from existing quantum chemistry software.
 *       Algorithm based on equations from Pople et al. (1976).
 *       Physicist notation <ij|ab> used throughout (not chemist notation).
 */

/**
 /**
 * @file ump2.cc
 * @brief Unrestricted MP2 - FIXED TENSOR INDEXING
 * * FIX: ERITransformer returns (i,a,j,b). UMP2 needs (i,j,a,b).
 * Added .shuffle() to align dimensions.
 */

#include "mshqc/ump2.h"
#include "mshqc/integrals/eri_transformer.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace mshqc {

UMP2::UMP2(const SCFResult& uhf_result,
           const BasisSet& basis,
           std::shared_ptr<IntegralEngine> integrals)
    : uhf_(uhf_result), basis_(basis), integrals_(integrals) {
    
    nbf_ = basis.n_basis_functions();
    nocc_a_ = uhf_result.n_occ_alpha;
    nocc_b_ = uhf_result.n_occ_beta;
    nvir_a_ = nbf_ - nocc_a_;
    nvir_b_ = nbf_ - nocc_b_;
}

void UMP2::transform_integrals() {
    auto eri_ao = integrals_->compute_eri();
    
    Eigen::MatrixXd Ca_occ = uhf_.C_alpha.leftCols(nocc_a_);
    Eigen::MatrixXd Ca_vir = uhf_.C_alpha.rightCols(nvir_a_);
    Eigen::MatrixXd Cb_occ = uhf_.C_beta.leftCols(nocc_b_);
    Eigen::MatrixXd Cb_vir = uhf_.C_beta.rightCols(nvir_b_);
    
    std::cout << "\nTransforming integrals to MO basis (Fast + Reordering)...\n";
    
    using integrals::ERITransformer;
    
    // DEFINISI SHUFFLE: 
    // Input dari Transformer: (i, a, j, b) -> Indeks [0, 1, 2, 3]
    // Target UMP2 (Physicist): (i, j, a, b)
    // Maka kita butuh urutan: [0, 2, 1, 3] (tukar a dan j)
    Eigen::array<int, 4> shuffle_idxs = {0, 2, 1, 3};

    // 1. Alpha-Alpha
    // Output asli: (i, a, j, b) -> Shuffle jadi (i, j, a, b)
    eri_aaaa_ = ERITransformer::transform_oovv(
        eri_ao, Ca_occ, Ca_vir, nbf_, nocc_a_, nvir_a_
    ).shuffle(shuffle_idxs);
    
    // 2. Beta-Beta
    eri_bbbb_ = ERITransformer::transform_oovv(
        eri_ao, Cb_occ, Cb_vir, nbf_, nocc_b_, nvir_b_
    ).shuffle(shuffle_idxs);
    
    // 3. Alpha-Beta (Mixed)
    // Output asli: (i_alpha, a_alpha, j_beta, b_beta)
    // Target: (i, j, a, b)
    eri_aabb_ = ERITransformer::transform_oovv_mixed(
        eri_ao, Ca_occ, Cb_occ, Ca_vir, Cb_vir, 
        nbf_, nocc_a_, nocc_b_, nvir_a_, nvir_b_
    ).shuffle(shuffle_idxs);
    
    std::cout << "  Integral transformation & shuffling complete.\n";
}

double UMP2::compute_ss_alpha() {
    const auto& eps = uhf_.orbital_energies_alpha;
    double e = 0.0;
    
    for (int i = 0; i < nocc_a_; i++) {
        for (int j = 0; j < nocc_a_; j++) {
            for (int a = 0; a < nvir_a_; a++) {
                for (int b = 0; b < nvir_a_; b++) {
                    // SEKARANG eri_aaaa_ sudah benar (i, j, a, b)
                    double g_ijab = eri_aaaa_(i,j,a,b);
                    double g_ijba = eri_aaaa_(i,j,b,a);
                    
                    double d = eps(i) + eps(j) - eps(nocc_a_+a) - eps(nocc_a_+b);
                    if (std::abs(d) < 1e-12) continue;
                    
                    double t_ijab = g_ijab / d;
                    t2_aa_(i, j, a, b) = t_ijab; 
                    e += 0.5 * t_ijab * (g_ijab - g_ijba);
                }
            }
        }
    }
    return e;
}

double UMP2::compute_ss_beta() {
    const auto& eps = uhf_.orbital_energies_beta;
    double e = 0.0;
    
    for (int i = 0; i < nocc_b_; i++) {
        for (int j = 0; j < nocc_b_; j++) {
            for (int a = 0; a < nvir_b_; a++) {
                for (int b = 0; b < nvir_b_; b++) {
                    double g_ijab = eri_bbbb_(i,j,a,b);
                    double g_ijba = eri_bbbb_(i,j,b,a);
                    
                    double d = eps(i) + eps(j) - eps(nocc_b_+a) - eps(nocc_b_+b);
                    if (std::abs(d) < 1e-12) continue;

                    double t_ijab = g_ijab / d;
                    t2_bb_(i, j, a, b) = t_ijab;
                    e += 0.5 * t_ijab * (g_ijab - g_ijba);
                }
            }
        }
    }
    return e;
}

double UMP2::compute_os() {
    const auto& eps_a = uhf_.orbital_energies_alpha;
    const auto& eps_b = uhf_.orbital_energies_beta;
    double e = 0.0;
    
    for (int i = 0; i < nocc_a_; i++) {
        for (int j = 0; j < nocc_b_; j++) {
            for (int a = 0; a < nvir_a_; a++) {
                for (int b = 0; b < nvir_b_; b++) {
                    // SEKARANG eri_aabb_ sudah benar (i, j, a, b)
                    double g = eri_aabb_(i, j, a, b);
                    
                    double d = eps_a(i) + eps_b(j) - eps_a(nocc_a_+a) - eps_b(nocc_b_+b);
                    if (std::abs(d) < 1e-12) continue;

                    double t = g / d;
                    t2_ab_(i, j, a, b) = t; 
                    e += t * g;
                }
            }
        }
    }
    return e;
}

UMP2Result UMP2::compute() {
    std::cout << "\n====================================\n";
    std::cout << "  Unrestricted MP2 (UMP2)\n";
    std::cout << "====================================\n";
    
    transform_integrals();
    
    t2_aa_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_a_, nvir_a_, nvir_a_);
    t2_bb_ = Eigen::Tensor<double, 4>(nocc_b_, nocc_b_, nvir_b_, nvir_b_);
    t2_ab_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_b_, nvir_a_, nvir_b_);
    t2_aa_.setZero();
    t2_bb_.setZero();
    t2_ab_.setZero();
    
    std::cout << "  Computing MP2 energy...\n";
    
    double e_ss_aa = compute_ss_alpha();
    double e_ss_bb = compute_ss_beta();
    double e_os = compute_os();
    double e_corr = e_ss_aa + e_ss_bb + e_os;
    
    std::cout << "\n=== UMP2 Results ===\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "SS (αα):        " << std::setw(16) << e_ss_aa << " Ha\n";
    std::cout << "SS (ββ):        " << std::setw(16) << e_ss_bb << " Ha\n";
    std::cout << "OS (αβ):        " << std::setw(16) << e_os << " Ha\n";
    std::cout << "Correlation:    " << std::setw(16) << e_corr << " Ha\n";
    
    UMP2Result result;
    result.e_corr_ss_aa = e_ss_aa;
    result.e_corr_ss_bb = e_ss_bb;
    result.e_corr_os = e_os;
    result.e_corr_total = e_corr;
    result.e_total = uhf_.energy_total + e_corr;
    return result;
}

T2Amplitudes UMP2::get_t2_amplitudes() const {
    T2Amplitudes result;
    result.t2_aa = t2_aa_;
    result.t2_bb = t2_bb_;
    result.t2_ab = t2_ab_;
    return result;
}

} // namespace mshqc