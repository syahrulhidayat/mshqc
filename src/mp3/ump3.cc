/**
 * @file ump3.cc - PART 1: Setup & Integral Transformations
 * @author Muhamad Syahrul Hidayat
 * @date 2025-02-01 (CORRECTED v6)
 */

/**
 * @file ump3.cc
 * @brief UMP3 Implementation - FIXED INDEXING
 * * FIX: Added .shuffle({0, 2, 1, 3}) to OOVV, OOOO, and VVVV
 * * to convert Chemist (i,a,j,b) -> Physicist (i,j,a,b).
 */

/**
 * @file ump3.cc
 * @brief UMP3 Implementation - FIXED
 * * FIX: Explicitly evaluate shuffle expression to Eigen::Tensor
 * * to prevent "call of object" compilation errors.
 */

#include "mshqc/ump3.h"
#include "mshqc/integrals/eri_transformer.h"
#include <iostream>
#include <iomanip>
#include <chrono>

namespace mshqc {

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
}

UMP3Result UMP3::compute() {
    std::cout << "\n====================================\n";
    std::cout << "  UMP3 (Indexing Fixed + Eval)\n";
    std::cout << "====================================\n";
    
    auto t_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "\n[1/4] Transforming integrals (with shuffle)...\n";
    transform_all_integrals();
    
    std::cout << "\n[2/4] Getting T2^(1) amplitudes...\n";
    get_t2_from_ump2();
    
    std::cout << "\n[3/4] Computing T2^(2) amplitudes...\n";
    compute_t2_second_order();
    
    std::cout << "\n[4/4] Computing E(3) energy...\n";
    double e3_aa = compute_e3_aa();
    double e3_bb = compute_e3_bb();
    double e3_ab = compute_e3_ab();
    
    double e3_total = e3_aa + e3_bb + e3_ab;
    double e_corr = ump2_.e_corr_total + e3_total;
    double e_total = uhf_.energy_total + e_corr;
    
    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);
    
    std::cout << "\n=== UMP3 Results ===\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "E(3) αα:     " << e3_aa << " Ha\n";
    std::cout << "E(3) ββ:     " << e3_bb << " Ha\n";
    std::cout << "E(3) αβ:     " << e3_ab << " Ha\n";
    std::cout << "E(3) total:  " << e3_total << " Ha\n";
    std::cout << "\nUMP2 corr:   " << ump2_.e_corr_total << " Ha\n";
    std::cout << "UMP3 corr:   " << e_corr << " Ha\n";
    std::cout << "UMP3 total:  " << e_total << " Ha\n";
    std::cout << "\nTime: " << duration.count() << " ms\n";
    
    if (std::abs(ump2_.e_corr_total) > 1e-10) {
        double ratio = std::abs(e3_total / ump2_.e_corr_total);
        std::cout << "\n|E(3)/E(2)|: " << ratio*100 << "%\n";
    }

    UMP3Result result;
    result.e_uhf = uhf_.energy_total;
    result.e_mp2 = ump2_.e_corr_total;
    result.e_mp3 = e3_total;
    result.e_corr_total = e_corr;
    result.e_total = e_total;
    result.e3_aa = e3_aa;
    result.e3_bb = e3_bb;
    result.e3_ab = e3_ab;
    
    result.n_occ_alpha = nocc_a_;
    result.n_occ_beta = nocc_b_;
    result.n_virt_alpha = nvir_a_;
    result.n_virt_beta = nvir_b_;

    result.t2_aa_1 = t2_aa_1_;
    result.t2_bb_1 = t2_bb_1_;
    result.t2_ab_1 = t2_ab_1_;

    result.t2_aa_2 = t2_aa_2_;
    result.t2_bb_2 = t2_bb_2_;
    result.t2_ab_2 = t2_ab_2_;
    
    result.t3_2_computed = false; 

    return result;
}

void UMP3::transform_all_integrals() {
    std::cout << "  Computing OOVV..." << std::flush;
    transform_oovv();
    std::cout << " done\n";
    
    std::cout << "  Computing OOOO..." << std::flush;
    transform_oooo();
    std::cout << " done\n";
    
    std::cout << "  Computing VVVV..." << std::flush;
    transform_vvvv();
    std::cout << " done\n";
    
    std::cout << "  Computing OVOV..." << std::flush;
    transform_ovov();
    std::cout << " done\n";
}

void UMP3::transform_oovv() {
    auto eri_ao = integrals_->compute_eri();
    const auto& Ca = uhf_.C_alpha;
    const auto& Cb = uhf_.C_beta;
    
    Eigen::MatrixXd Ca_occ = Ca.leftCols(nocc_a_);
    Eigen::MatrixXd Ca_virt = Ca.rightCols(nvir_a_);
    Eigen::MatrixXd Cb_occ = Cb.leftCols(nocc_b_);
    Eigen::MatrixXd Cb_virt = Cb.rightCols(nvir_b_);
    
    using namespace mshqc::integrals;
    Eigen::array<int, 4> shuf = {0, 2, 1, 3};
    
    // FIX: Assign to explicit Tensor type to force evaluation
    Eigen::Tensor<double, 4> eri_aa = ERITransformer::transform_oovv_quarter(
        eri_ao, Ca_occ, Ca_virt, nbf_, nocc_a_, nvir_a_
    ).shuffle(shuf);
    
    g_oovv_aa_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_a_, nvir_a_, nvir_a_);
    for (int i=0; i<nocc_a_; ++i)
        for (int j=0; j<nocc_a_; ++j)
            for (int a=0; a<nvir_a_; ++a)
                for (int b=0; b<nvir_a_; ++b)
                    g_oovv_aa_(i,j,a,b) = eri_aa(i,j,a,b) - eri_aa(i,j,b,a);
    
    // Beta-beta
    Eigen::Tensor<double, 4> eri_bb = ERITransformer::transform_oovv_quarter(
        eri_ao, Cb_occ, Cb_virt, nbf_, nocc_b_, nvir_b_
    ).shuffle(shuf);
    
    g_oovv_bb_ = Eigen::Tensor<double, 4>(nocc_b_, nocc_b_, nvir_b_, nvir_b_);
    for (int i=0; i<nocc_b_; ++i)
        for (int j=0; j<nocc_b_; ++j)
            for (int a=0; a<nvir_b_; ++a)
                for (int b=0; b<nvir_b_; ++b)
                    g_oovv_bb_(i,j,a,b) = eri_bb(i,j,a,b) - eri_bb(i,j,b,a);
    
    // Alpha-beta
    g_oovv_ab_ = ERITransformer::transform_oovv_mixed(
        eri_ao, Ca_occ, Cb_occ, Ca_virt, Cb_virt,
        nbf_, nocc_a_, nocc_b_, nvir_a_, nvir_b_
    ).shuffle(shuf);
}

void UMP3::transform_oooo() {
    auto eri_ao = integrals_->compute_eri();
    const auto& Ca = uhf_.C_alpha;
    const auto& Cb = uhf_.C_beta;
    
    Eigen::MatrixXd Ca_occ = Ca.leftCols(nocc_a_);
    Eigen::MatrixXd Cb_occ = Cb.leftCols(nocc_b_);
    
    using namespace mshqc::integrals;
    Eigen::array<int, 4> shuf = {0, 2, 1, 3};
    
    // Alpha-alpha
    Eigen::Tensor<double, 4> eri_aa = ERITransformer::transform_oooo(
        eri_ao, Ca_occ, nbf_, nocc_a_
    ).shuffle(shuf);
    
    g_oooo_aa_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_a_, nocc_a_, nocc_a_);
    for (int i=0; i<nocc_a_; ++i)
        for (int j=0; j<nocc_a_; ++j)
            for (int k=0; k<nocc_a_; ++k)
                for (int l=0; l<nocc_a_; ++l)
                    g_oooo_aa_(i,j,k,l) = eri_aa(i,j,k,l) - eri_aa(i,j,l,k);
    
    // Beta-beta
    Eigen::Tensor<double, 4> eri_bb = ERITransformer::transform_oooo(
        eri_ao, Cb_occ, nbf_, nocc_b_
    ).shuffle(shuf);
    
    g_oooo_bb_ = Eigen::Tensor<double, 4>(nocc_b_, nocc_b_, nocc_b_, nocc_b_);
    for (int i=0; i<nocc_b_; ++i)
        for (int j=0; j<nocc_b_; ++j)
            for (int k=0; k<nocc_b_; ++k)
                for (int l=0; l<nocc_b_; ++l)
                    g_oooo_bb_(i,j,k,l) = eri_bb(i,j,k,l) - eri_bb(i,j,l,k);
    
    // Alpha-beta
    g_oooo_ab_ = ERITransformer::transform_oooo_mixed(
        eri_ao, Ca_occ, Cb_occ, nbf_, nocc_a_, nocc_b_
    ).shuffle(shuf);
}

void UMP3::transform_vvvv() {
    auto eri_ao = integrals_->compute_eri();
    const auto& Ca = uhf_.C_alpha;
    const auto& Cb = uhf_.C_beta;
    
    Eigen::MatrixXd Ca_virt = Ca.rightCols(nvir_a_);
    Eigen::MatrixXd Cb_virt = Cb.rightCols(nvir_b_);
    
    using namespace mshqc::integrals;
    Eigen::array<int, 4> shuf = {0, 2, 1, 3};
    
    // Alpha-alpha
    Eigen::Tensor<double, 4> eri_aa = ERITransformer::transform_vvvv(
        eri_ao, Ca_virt, nbf_, nvir_a_
    ).shuffle(shuf);
    
    g_vvvv_aa_ = Eigen::Tensor<double, 4>(nvir_a_, nvir_a_, nvir_a_, nvir_a_);
    for (int a=0; a<nvir_a_; ++a)
        for (int b=0; b<nvir_a_; ++b)
            for (int c=0; c<nvir_a_; ++c)
                for (int d=0; d<nvir_a_; ++d)
                    g_vvvv_aa_(a,b,c,d) = eri_aa(a,b,c,d) - eri_aa(a,b,d,c);
    
    // Beta-beta
    Eigen::Tensor<double, 4> eri_bb = ERITransformer::transform_vvvv(
        eri_ao, Cb_virt, nbf_, nvir_b_
    ).shuffle(shuf);
    
    g_vvvv_bb_ = Eigen::Tensor<double, 4>(nvir_b_, nvir_b_, nvir_b_, nvir_b_);
    for (int a=0; a<nvir_b_; ++a)
        for (int b=0; b<nvir_b_; ++b)
            for (int c=0; c<nvir_b_; ++c)
                for (int d=0; d<nvir_b_; ++d)
                    g_vvvv_bb_(a,b,c,d) = eri_bb(a,b,c,d) - eri_bb(a,b,d,c);
    
    // Alpha-beta
    g_vvvv_ab_ = ERITransformer::transform_vvvv_mixed(
        eri_ao, Ca_virt, Cb_virt, nbf_, nvir_a_, nvir_b_
    ).shuffle(shuf);
}

void UMP3::transform_ovov() {
    auto eri_ao = integrals_->compute_eri();
    const auto& Ca = uhf_.C_alpha;
    const auto& Cb = uhf_.C_beta;
    
    Eigen::MatrixXd Ca_occ = Ca.leftCols(nocc_a_);
    Eigen::MatrixXd Ca_virt = Ca.rightCols(nvir_a_);
    Eigen::MatrixXd Cb_occ = Cb.leftCols(nocc_b_);
    Eigen::MatrixXd Cb_virt = Cb.rightCols(nvir_b_);
    
    using namespace mshqc::integrals;
    
    // Alpha-alpha
    Eigen::Tensor<double, 4> eri_aa = ERITransformer::transform_ovov(
        eri_ao, Ca_occ, Ca_virt, nbf_, nocc_a_, nvir_a_
    );
    
    g_ovov_aa_ = Eigen::Tensor<double, 4>(nocc_a_, nvir_a_, nocc_a_, nvir_a_);
    for (int i=0; i<nocc_a_; ++i)
        for (int a=0; a<nvir_a_; ++a)
            for (int j=0; j<nocc_a_; ++j)
                for (int b=0; b<nvir_a_; ++b)
                    g_ovov_aa_(i,a,j,b) = eri_aa(i,a,j,b) - eri_aa(i,a,b,j);
    
    // Beta-beta
    Eigen::Tensor<double, 4> eri_bb = ERITransformer::transform_ovov(
        eri_ao, Cb_occ, Cb_virt, nbf_, nocc_b_, nvir_b_
    );
    
    g_ovov_bb_ = Eigen::Tensor<double, 4>(nocc_b_, nvir_b_, nocc_b_, nvir_b_);
    for (int i=0; i<nocc_b_; ++i)
        for (int a=0; a<nvir_b_; ++a)
            for (int j=0; j<nocc_b_; ++j)
                for (int b=0; b<nvir_b_; ++b)
                    g_ovov_bb_(i,a,j,b) = eri_bb(i,a,j,b) - eri_bb(i,a,b,j);
    
    g_ovov_ab_ = ERITransformer::transform_ovov_mixed(
        eri_ao, Ca_occ, Cb_virt, nbf_, nocc_a_, nvir_b_
    );
    
    g_ovov_ba_ = ERITransformer::transform_ovov_mixed(
        eri_ao, Cb_occ, Ca_virt, nbf_, nocc_b_, nvir_a_
    );
}

void UMP3::get_t2_from_ump2() {
    const auto& ea = uhf_.orbital_energies_alpha;
    const auto& eb = uhf_.orbital_energies_beta;
    
    t2_aa_1_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_a_, nvir_a_, nvir_a_);
    t2_bb_1_ = Eigen::Tensor<double, 4>(nocc_b_, nocc_b_, nvir_b_, nvir_b_);
    t2_ab_1_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_b_, nvir_a_, nvir_b_);
    
    t2_aa_1_.setZero();
    t2_bb_1_.setZero();
    t2_ab_1_.setZero();
    
    for (int i=0; i<nocc_a_; ++i) {
        for (int j=0; j<nocc_a_; ++j) {
            for (int a=0; a<nvir_a_; ++a) {
                for (int b=0; b<nvir_a_; ++b) {
                    double D = ea(i) + ea(j) - ea(nocc_a_+a) - ea(nocc_a_+b);
                    t2_aa_1_(i,j,a,b) = g_oovv_aa_(i,j,a,b) / D;
                }
            }
        }
    }
    
    for (int i=0; i<nocc_b_; ++i) {
        for (int j=0; j<nocc_b_; ++j) {
            for (int a=0; a<nvir_b_; ++a) {
                for (int b=0; b<nvir_b_; ++b) {
                    double D = eb(i) + eb(j) - eb(nocc_b_+a) - eb(nocc_b_+b);
                    t2_bb_1_(i,j,a,b) = g_oovv_bb_(i,j,a,b) / D;
                }
            }
        }
    }
    
    for (int i=0; i<nocc_a_; ++i) {
        for (int j=0; j<nocc_b_; ++j) {
            for (int a=0; a<nvir_a_; ++a) {
                for (int b=0; b<nvir_b_; ++b) {
                    double D = ea(i) + eb(j) - ea(nocc_a_+a) - eb(nocc_b_+b);
                    t2_ab_1_(i,j,a,b) = g_oovv_ab_(i,j,a,b) / D;
                }
            }
        }
    }
}

void UMP3::compute_t2_second_order() {
    t2_aa_2_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_a_, nvir_a_, nvir_a_);
    t2_bb_2_ = Eigen::Tensor<double, 4>(nocc_b_, nocc_b_, nvir_b_, nvir_b_);
    t2_ab_2_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_b_, nvir_a_, nvir_b_);
    
    t2_aa_2_.setZero();
    t2_bb_2_.setZero();
    t2_ab_2_.setZero();
    
    compute_t2_aa_second();
    compute_t2_bb_second();
    compute_t2_ab_second();
}

void UMP3::compute_t2_aa_second() {
    const auto& ea = uhf_.orbital_energies_alpha;
    
    for (int i=0; i<nocc_a_; ++i) {
        for (int j=0; j<nocc_a_; ++j) {
            for (int a=0; a<nvir_a_; ++a) {
                for (int b=0; b<nvir_a_; ++b) {
                    double val = 0.0;
                    for (int k=0; k<nocc_a_; ++k)
                        for (int l=0; l<nocc_a_; ++l)
                            val += 0.5 * g_oooo_aa_(k,l,i,j) * t2_aa_1_(k,l,a,b);
                    for (int c=0; c<nvir_a_; ++c)
                        for (int d=0; d<nvir_a_; ++d)
                            val += 0.5 * g_vvvv_aa_(a,b,c,d) * t2_aa_1_(i,j,c,d);
                    for (int k=0; k<nocc_a_; ++k) {
                        for (int c=0; c<nvir_a_; ++c) {
                            val -= g_ovov_aa_(k,b,j,c) * t2_aa_1_(i,k,a,c);
                            val -= g_ovov_aa_(k,a,i,c) * t2_aa_1_(k,j,b,c);
                        }
                    }
                    double D = ea(i) + ea(j) - ea(nocc_a_+a) - ea(nocc_a_+b);
                    t2_aa_2_(i,j,a,b) = val / D;
                }
            }
        }
    }
}

void UMP3::compute_t2_bb_second() {
    const auto& eb = uhf_.orbital_energies_beta;
    
    for (int i=0; i<nocc_b_; ++i) {
        for (int j=0; j<nocc_b_; ++j) {
            for (int a=0; a<nvir_b_; ++a) {
                for (int b=0; b<nvir_b_; ++b) {
                    double val = 0.0;
                    for (int k=0; k<nocc_b_; ++k)
                        for (int l=0; l<nocc_b_; ++l)
                            val += 0.5 * g_oooo_bb_(k,l,i,j) * t2_bb_1_(k,l,a,b);
                    for (int c=0; c<nvir_b_; ++c)
                        for (int d=0; d<nvir_b_; ++d)
                            val += 0.5 * g_vvvv_bb_(a,b,c,d) * t2_bb_1_(i,j,c,d);
                    for (int k=0; k<nocc_b_; ++k) {
                        for (int c=0; c<nvir_b_; ++c) {
                            val -= g_ovov_bb_(k,b,j,c) * t2_bb_1_(i,k,a,c);
                            val -= g_ovov_bb_(k,a,i,c) * t2_bb_1_(k,j,b,c);
                        }
                    }
                    double D = eb(i) + eb(j) - eb(nocc_b_+a) - eb(nocc_b_+b);
                    t2_bb_2_(i,j,a,b) = val / D;
                }
            }
        }
    }
}

void UMP3::compute_t2_ab_second() {
    const auto& ea = uhf_.orbital_energies_alpha;
    const auto& eb = uhf_.orbital_energies_beta;

    #pragma omp parallel for collapse(2)
    for (int i=0; i<nocc_a_; ++i) {
        for (int j=0; j<nocc_b_; ++j) {
            for (int a=0; a<nvir_a_; ++a) {
                for (int b=0; b<nvir_b_; ++b) {
                    double val = 0.0;
                    for (int k=0; k<nocc_a_; ++k)
                        for (int l=0; l<nocc_b_; ++l)
                            val += g_oooo_ab_(k,i,l,j) * t2_ab_1_(k,l,a,b);
                    for (int c=0; c<nvir_a_; ++c)
                        for (int d=0; d<nvir_b_; ++d)
                            val += g_vvvv_ab_(a,b,c,d) * t2_ab_1_(i,j,c,d);
                    for (int k=0; k<nocc_a_; ++k)
                        for (int c=0; c<nvir_a_; ++c)
                            val -= g_ovov_aa_(k,a,i,c) * t2_ab_1_(k,j,c,b);
                    for (int k=0; k<nocc_b_; ++k)
                        for (int c=0; c<nvir_b_; ++c)
                            val -= g_ovov_bb_(k,b,j,c) * t2_ab_1_(i,k,a,c);
                    for (int k=0; k<nocc_b_; ++k)
                        for (int c=0; c<nvir_a_; ++c)
                            val -= g_ovov_ba_(k,c,j,a) * t2_ab_1_(i,k,c,b);
                    for (int k=0; k<nocc_a_; ++k)
                        for (int c=0; c<nvir_b_; ++c)
                            val -= g_ovov_ab_(k,c,i,b) * t2_ab_1_(k,j,a,c);
                    
                    double D = ea(i) + eb(j) - ea(nocc_a_+a) - eb(nocc_b_+b);
                    t2_ab_2_(i,j,a,b) = val / D;
                }
            }
        }
    }
}

double UMP3::compute_e3_aa() {
    double energy = 0.0;
    for (int i=0; i<nocc_a_; ++i)
        for (int j=0; j<nocc_a_; ++j)
            for (int a=0; a<nvir_a_; ++a)
                for (int b=0; b<nvir_a_; ++b)
                    energy += g_oovv_aa_(i,j,a,b) * t2_aa_2_(i,j,a,b);
    return 0.25 * energy;
}

double UMP3::compute_e3_bb() {
    double energy = 0.0;
    for (int i=0; i<nocc_b_; ++i)
        for (int j=0; j<nocc_b_; ++j)
            for (int a=0; a<nvir_b_; ++a)
                for (int b=0; b<nvir_b_; ++b)
                    energy += g_oovv_bb_(i,j,a,b) * t2_bb_2_(i,j,a,b);
    return 0.25 * energy;
}

double UMP3::compute_e3_ab() {
    double energy = 0.0;
    #pragma omp parallel for collapse(2) reduction(+:energy)
    for (int i=0; i<nocc_a_; ++i)
        for (int j=0; j<nocc_b_; ++j)
            for (int a=0; a<nvir_a_; ++a)
                for (int b=0; b<nvir_b_; ++b)
                    energy += g_oovv_ab_(i,j,a,b) * t2_ab_2_(i,j,a,b);
    return energy;
}

} // namespace mshqc