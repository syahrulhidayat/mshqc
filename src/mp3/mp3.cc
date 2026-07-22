/**
 * @file src/mp3/mp3.cc
 * @brief Unified MP3 Implementation Powered by Native TBLIS
 */

#include "mshqc/mp3/mp3.h"
#include "mshqc/mp3/omp3.h"
#include "mshqc/integrals/eri_transformer.h"
#include "mshqc/gradient/optimizer.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <tblis/tblis.h>

namespace mshqc {

using integrals::ERITransformer;
using tblis::len_type;
using tblis::stride_type;
using tblis::varray_view;

BaseMP3::BaseMP3(const SCFResult& scf, const MP2Result& mp2, const MP2Config& config, std::shared_ptr<IntegralEngine> ints)
    : scf_(scf), mp2_(mp2), config_(config), ints_(ints) 
{
    nbf_ = scf_.C_alpha.rows();
    no_a_ = scf_.n_occ_alpha; no_b_ = scf_.n_occ_beta;
    nv_a_ = nbf_ - no_a_;     nv_b_ = nbf_ - no_b_;
    n_aux_ = scf_.L_mat.cols(); 
    
    if (mp2_.t2_aa.size() > 0) t2_aa_ = mp2_.t2_aa;
    if (mp2_.t2_bb.size() > 0) t2_bb_ = mp2_.t2_bb;
    if (mp2_.t2_ab.size() > 0) t2_ab_ = mp2_.t2_ab;
}

double BaseMP3::tensor_dot(const Eigen::Tensor<double, 4>& A, const Eigen::Tensor<double, 4>& B) const {
    Eigen::Map<const Eigen::VectorXd> vecA(A.data(), A.size());
    Eigen::Map<const Eigen::VectorXd> vecB(B.data(), B.size());
    return vecA.dot(vecB);
}

#define TBLIS_VIEW_4D(name, t, d1, d2, d3, d4) \
    varray_view<double> name({(len_type)d1, (len_type)d2, (len_type)d3, (len_type)d4}, t.data(), \
    {1, (stride_type)d1, (stride_type)(d1*d2), (stride_type)(d1*d2*d3)})

#define TBLIS_VIEW_3D(name, ptr, d1, d2, d3) \
    varray_view<double> name({(len_type)d1, (len_type)d2, (len_type)d3}, ptr, \
    {1, (stride_type)d1, (stride_type)(d1*d2)})

#define TBLIS_VIEW_2D(name, ptr, d1, d2) \
    varray_view<double> name({(len_type)d1, (len_type)d2}, ptr, \
    {1, (stride_type)d1})

MP3Result RMP3::compute() {
    auto t_start = std::chrono::high_resolution_clock::now();
    if(omp_get_thread_num() == 0) std::cout << "\n=== RMP3 (Unified Native TBLIS) ===\n";

    const auto& Co = scf_.C_alpha.leftCols(no_a_);
    const auto& Cv = scf_.C_alpha.rightCols(nv_a_);

    TBLIS_VIEW_4D(t_T, t2_aa_, no_a_, no_a_, nv_a_, nv_a_);
    Eigen::Tensor<double, 4> W(no_a_, no_a_, nv_a_, nv_a_); W.setZero();
    TBLIS_VIEW_4D(t_W, W, no_a_, no_a_, nv_a_, nv_a_);

    {
        auto V = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cv, Cv, Cv, Cv, ints_);
        TBLIS_VIEW_4D(t_V, V, nv_a_, nv_a_, nv_a_, nv_a_);
        tblis::mult<double>(1.0, t_T, "ijef", t_V, "eafb", 1.0, t_W, "ijab");
    }
    {
        auto V = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Co, Co, Co, Co, ints_);
        TBLIS_VIEW_4D(t_V, V, no_a_, no_a_, no_a_, no_a_);
        tblis::mult<double>(1.0, t_T, "mnab", t_V, "minj", 1.0, t_W, "ijab");
    }
    {
        auto V_ovov = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Co, Cv, Co, Cv, ints_);
        auto V_oovv = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Co, Co, Cv, Cv, ints_);
        TBLIS_VIEW_4D(t_Vovov, V_ovov, no_a_, nv_a_, no_a_, nv_a_);
        TBLIS_VIEW_4D(t_Voovv, V_oovv, no_a_, no_a_, nv_a_, nv_a_);

        tblis::mult<double>(2.0,  t_Vovov, "iakc", t_T, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikac", t_T, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(2.0,  t_T, "ikac", t_Vovov, "kcjb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T, "ikac", t_Voovv, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T, "ikca", t_Vovov, "kcjb", 1.0, t_W, "ijab"); 
        tblis::mult<double>(-1.0, t_Vovov, "iakc", t_T, "kjbc", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikbc", t_T, "kjac", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T, "ikcb", t_Voovv, "jkac", 1.0, t_W, "ijab"); 
    }

    double e_mp3 = 0.0;
    #pragma omp parallel for collapse(4) reduction(+:e_mp3)
    for (int i = 0; i < no_a_; ++i) {
        for (int j = 0; j < no_a_; ++j) {
            for (int a = 0; a < nv_a_; ++a) {
                for (int b = 0; b < nv_a_; ++b) {
                    e_mp3 += W(i, j, a, b) * (2.0 * t2_aa_(i, j, a, b) - t2_aa_(i, j, b, a));
                }
            }
        }
    }

    MP3Result res;
    res.e_hf = scf_.energy_total;
    res.e_mp2 = mp2_.energy_mp2_corr;
    res.e3_aa = 0.0; res.e3_ab = 0.0; res.e_mp3 = e_mp3;
    res.e_corr_total = res.e_mp2 + res.e_mp3;
    res.e_total = res.e_hf + res.e_corr_total;

    auto t_end = std::chrono::high_resolution_clock::now();
    if(omp_get_thread_num() == 0) {
        std::cout << "  E_MP3        : " << std::fixed << std::setprecision(8) << res.e_mp3 << " Ha\n";
        std::cout << "  Total Energy : " << res.e_total << " Ha\n";
        std::cout << "  Time         : " << std::chrono::duration<double>(t_end - t_start).count() << " s\n";
    }
    return res;
}

MP3Result UMP3::compute() {
    auto t_start = std::chrono::high_resolution_clock::now();
    if(omp_get_thread_num() == 0) std::cout << "\n=== UMP3 (Unified Native TBLIS) ===\n";

    const auto& Cao = scf_.C_alpha.leftCols(no_a_); const auto& Cav = scf_.C_alpha.rightCols(nv_a_);
    const auto& Cbo = scf_.C_beta.leftCols(no_b_);  const auto& Cbv = scf_.C_beta.rightCols(nv_b_);
    double e3_aa = 0.0, e3_bb = 0.0, e3_ab = 0.0;

    TBLIS_VIEW_4D(t_Taa, t2_aa_, no_a_, no_a_, nv_a_, nv_a_);
    TBLIS_VIEW_4D(t_Tbb, t2_bb_, no_b_, no_b_, nv_b_, nv_b_);
    TBLIS_VIEW_4D(t_Tab, t2_ab_, no_a_, no_b_, nv_a_, nv_b_);

    Eigen::Tensor<double, 4> Waa(no_a_, no_a_, nv_a_, nv_a_); TBLIS_VIEW_4D(t_Waa, Waa, no_a_, no_a_, nv_a_, nv_a_);
    Eigen::Tensor<double, 4> Wbb(no_b_, no_b_, nv_b_, nv_b_); TBLIS_VIEW_4D(t_Wbb, Wbb, no_b_, no_b_, nv_b_, nv_b_);
    Eigen::Tensor<double, 4> Wab(no_a_, no_b_, nv_a_, nv_b_); TBLIS_VIEW_4D(t_Wab, Wab, no_a_, no_b_, nv_a_, nv_b_);

    {
        auto Vaa = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cav, Cav, Cav, Cav, ints_);
        TBLIS_VIEW_4D(t_Vaa, Vaa, nv_a_, nv_a_, nv_a_, nv_a_); Waa.setZero();
        tblis::mult<double>(1.0, t_Taa, "ijef", t_Vaa, "eafb", 1.0, t_Waa, "ijab");
        tblis::mult<double>(-1.0, t_Taa, "ijef", t_Vaa, "ebfa", 1.0, t_Waa, "ijab");
        e3_aa += 0.125 * tensor_dot(t2_aa_, Waa);

        auto Vbb = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbv, Cbv, Cbv, Cbv, ints_);
        TBLIS_VIEW_4D(t_Vbb, Vbb, nv_b_, nv_b_, nv_b_, nv_b_); Wbb.setZero();
        tblis::mult<double>(1.0, t_Tbb, "ijef", t_Vbb, "eafb", 1.0, t_Wbb, "ijab");
        tblis::mult<double>(-1.0, t_Tbb, "ijef", t_Vbb, "ebfa", 1.0, t_Wbb, "ijab");
        e3_bb += 0.125 * tensor_dot(t2_bb_, Wbb);

        auto Vab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cav, Cav, Cbv, Cbv, ints_);
        TBLIS_VIEW_4D(t_Vab, Vab, nv_a_, nv_a_, nv_b_, nv_b_); Wab.setZero();
        tblis::mult<double>(1.0, t_Tab, "ijef", t_Vab, "eafb", 1.0, t_Wab, "ijab");
        e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
    }
    {
        auto Vaa = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cao, Cao, ints_);
        TBLIS_VIEW_4D(t_Vaa, Vaa, no_a_, no_a_, no_a_, no_a_); Waa.setZero();
        tblis::mult<double>(1.0, t_Taa, "mnab", t_Vaa, "minj", 1.0, t_Waa, "ijab");
        tblis::mult<double>(-1.0, t_Taa, "mnab", t_Vaa, "mjni", 1.0, t_Waa, "ijab");
        e3_aa += 0.125 * tensor_dot(t2_aa_, Waa);

        auto Vbb = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbo, Cbo, Cbo, ints_);
        TBLIS_VIEW_4D(t_Vbb, Vbb, no_b_, no_b_, no_b_, no_b_); Wbb.setZero();
        tblis::mult<double>(1.0, t_Tbb, "mnab", t_Vbb, "minj", 1.0, t_Wbb, "ijab");
        tblis::mult<double>(-1.0, t_Tbb, "mnab", t_Vbb, "mjni", 1.0, t_Wbb, "ijab");
        e3_bb += 0.125 * tensor_dot(t2_bb_, Wbb);

        auto Vab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cbo, Cbo, ints_);
        TBLIS_VIEW_4D(t_Vab, Vab, no_a_, no_a_, no_b_, no_b_); Wab.setZero();
        tblis::mult<double>(1.0, t_Tab, "mnab", t_Vab, "minj", 1.0, t_Wab, "ijab");
        e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
    }
    {
        auto ovov_aa = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cav, Cao, Cav, ints_);
        auto oovv_aa = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cav, Cav, ints_);
        TBLIS_VIEW_4D(t_ovov_aa, ovov_aa, no_a_, nv_a_, no_a_, nv_a_);
        TBLIS_VIEW_4D(t_oovv_aa, oovv_aa, no_a_, no_a_, nv_a_, nv_a_);

        auto ovov_bb = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbv, Cbo, Cbv, ints_);
        auto oovv_bb = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbo, Cbv, Cbv, ints_);
        TBLIS_VIEW_4D(t_ovov_bb, ovov_bb, no_b_, nv_b_, no_b_, nv_b_);
        TBLIS_VIEW_4D(t_oovv_bb, oovv_bb, no_b_, no_b_, nv_b_, nv_b_);

        auto ovov_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cav, Cbo, Cbv, ints_);
        auto oovv_ab_ex = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cbv, Cbv, ints_);
        auto oovv_ba_ex = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbo, Cav, Cav, ints_);
        TBLIS_VIEW_4D(t_ovov_ab, ovov_ab, no_a_, nv_a_, no_b_, nv_b_);
        TBLIS_VIEW_4D(t_oovv_ab_ex, oovv_ab_ex, no_a_, no_a_, nv_b_, nv_b_);
        TBLIS_VIEW_4D(t_oovv_ba_ex, oovv_ba_ex, no_b_, no_b_, nv_a_, nv_a_);

        Waa.setZero();
        tblis::mult<double>(1.0, t_ovov_aa, "iakc", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
        tblis::mult<double>(-1.0, t_oovv_aa, "ikac", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
        tblis::mult<double>(1.0, t_ovov_ab, "iakc", t_Tab, "jkbc", 1.0, t_Waa, "ijab");
        e3_aa += 1.0 * tensor_dot(t2_aa_, Waa);
        
        Wbb.setZero();
        tblis::mult<double>(1.0, t_ovov_bb, "iakc", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
        tblis::mult<double>(-1.0, t_oovv_bb, "ikac", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
        tblis::mult<double>(1.0, t_ovov_ab, "kcia", t_Tab, "kjcb", 1.0, t_Wbb, "ijab"); 
        e3_bb += 1.0 * tensor_dot(t2_bb_, Wbb);

        Wab.setZero();
        tblis::mult<double>(1.0,  t_ovov_aa, "iakc", t_Tab, "kjcb", 1.0, t_Wab, "ijab");
        tblis::mult<double>(-1.0, t_oovv_aa, "ikac", t_Tab, "kjcb", 1.0, t_Wab, "ijab");
        tblis::mult<double>(1.0,  t_Tab, "ikac", t_ovov_bb, "kcjb", 1.0, t_Wab, "ijab");
        tblis::mult<double>(-1.0, t_Tab, "ikac", t_oovv_bb, "kjcb", 1.0, t_Wab, "ijab");
        tblis::mult<double>(1.0,  t_Taa, "ikac", t_ovov_ab, "kcjb", 1.0, t_Wab, "ijab");
        tblis::mult<double>(1.0,  t_ovov_ab, "iakc", t_Tbb, "kjcb", 1.0, t_Wab, "ijab");
        tblis::mult<double>(-1.0, t_oovv_ab_ex, "ikbc", t_Tab, "kjac", 1.0, t_Wab, "ijab");
        tblis::mult<double>(-1.0, t_Tab, "ikcb", t_oovv_ba_ex, "jkac", 1.0, t_Wab, "ijab"); 
        e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
    }

    MP3Result res;
    res.e_hf = scf_.energy_total;
    res.e_mp2 = mp2_.energy_mp2_corr;
    res.e3_aa = e3_aa; res.e3_bb = e3_bb; res.e3_ab = e3_ab;
    res.e_mp3 = e3_aa + e3_bb + e3_ab;
    res.e_corr_total = res.e_mp2 + res.e_mp3;
    res.e_total = res.e_hf + res.e_corr_total;

    auto t_end = std::chrono::high_resolution_clock::now();
    if(omp_get_thread_num() == 0) {
        std::cout << "  E_MP3        : " << std::fixed << std::setprecision(8) << res.e_mp3 << " Ha\n";
        std::cout << "  Total Energy : " << res.e_total << " Ha\n";
        std::cout << "  Time         : " << std::chrono::duration<double>(t_end - t_start).count() << " s\n";
    }
    return res;
}







void OMP3::compute_mp3_correction() {
    if (na_ == 0 || va_ == 0) { e_mp3_tot_ = 0.0; return; }
    
    if (!config_.use_df) {
        throw std::runtime_error("[OMP3] Exact Integrals not yet supported. Please use DF.");
    }

    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    const Eigen::MatrixXd& Cao = scf_.C_alpha.leftCols(na_); 
    const Eigen::MatrixXd& Cav = scf_.C_alpha.rightCols(va_);
    const auto& ea = scf_.orbital_energies_alpha;
    
    auto* t2_aa_dense = t2_aa_.get_block(0,0,0,0);
    if(!t2_aa_dense) throw std::runtime_error("OMP3 missing T2 dense block.");
    Eigen::Tensor<double, 4> T2_aa_ijab(na_, na_, va_, va_);
    #pragma omp parallel for collapse(4) schedule(static)
    for(int i=0; i<na_; ++i) {
        for(int j=0; j<na_; ++j) {
            for(int a=0; a<va_; ++a) {
                for(int b=0; b<va_; ++b) {
                    T2_aa_ijab(i,j,a,b) = (*t2_aa_dense)(i,a,j,b);
                }
            }
        }
    }

    int n_aux = scf_.L_mat.cols();
    Eigen::Map<const Eigen::MatrixXd> L_flat(scf_.L_mat.data(), nbf_, nbf_ * n_aux);

    // =========================================================================
    //   ULTRA-FAST DF TENSOR BUILDER (O(N^4) to O(N^5) factorized GEMM)
    // =========================================================================
    auto build_B_mat = [&](const Eigen::MatrixXd& C_left, const Eigen::MatrixXd& C_right, int dim_L, int dim_R) {
        Eigen::MatrixXd B_mat = Eigen::MatrixXd::Zero(dim_L * dim_R, n_aux);
        Eigen::MatrixXd X_temp = C_left.transpose() * L_flat;
        #pragma omp parallel for schedule(static)
        for (int P = 0; P < n_aux; ++P) {
            Eigen::Map<Eigen::MatrixXd> X_P(X_temp.data() + P * dim_L * nbf_, dim_L, nbf_);
            Eigen::MatrixXd B_MO = X_P * C_right; 
            for (int i = 0; i < dim_L; ++i) {
                for (int j = 0; j < dim_R; ++j) {
                    B_mat(i * dim_R + j, P) = B_MO(i, j);
                }
            }
        }
        return B_mat;
    };

    auto map_4d = [](const Eigen::MatrixXd& M, int d1, int d2, int d3, int d4) {
        Eigen::Tensor<double, 4> T(d1, d2, d3, d4);
        #pragma omp parallel for collapse(4) schedule(static)
        for(int i=0; i<d1; ++i) {
            for(int j=0; j<d2; ++j) {
                for(int k=0; k<d3; ++k) {
                    for(int l=0; l<d4; ++l) {
                        T(i,j,k,l) = M(i*d2+j, k*d4+l);
                    }
                }
            }
        }
        return T;
    };

    Eigen::MatrixXd B_ij_a = build_B_mat(Cao, Cao, na_, na_);
    Eigen::MatrixXd B_ab_a = build_B_mat(Cav, Cav, va_, va_);

    // ================= RMP3 BLOCK ================= //
    if (is_restricted) {
        t2_3rd_aa_ = Eigen::Tensor<double, 4>(na_, na_, va_, va_);
        Eigen::Tensor<double, 4> W(na_, na_, va_, va_); W.setZero();
        
        TBLIS_VIEW_4D(t_T, T2_aa_ijab, na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_W, W, na_, na_, va_, va_);
      
        // Fast O(N^5) generation of MO integrals (NO ERITransformer)
        Eigen::Tensor<double, 4> V_oooo = map_4d(B_ij_a * B_ij_a.transpose(), na_, na_, na_, na_);
        Eigen::Tensor<double, 4> V_ovov = map_4d(B_ia_P_alpha_ * B_ia_P_alpha_.transpose(), na_, va_, na_, va_);
        Eigen::Tensor<double, 4> V_oovv = map_4d(B_ij_a * B_ab_a.transpose(), na_, na_, va_, va_);
        
        TBLIS_VIEW_4D(t_Voooo, V_oooo, na_, na_, na_, na_);
        TBLIS_VIEW_4D(t_Vovov, V_ovov, na_, va_, na_, va_);
        TBLIS_VIEW_4D(t_Voovv, V_oovv, na_, na_, va_, va_);

        tblis::mult<double>(1.0, t_T, "mnab", t_Voooo, "minj", 1.0, t_W, "ijab");
        tblis::mult<double>(2.0,  t_Vovov, "iakc", t_T, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikac", t_T, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(2.0,  t_T, "ikac", t_Vovov, "kcjb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T, "ikac", t_Voovv, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T, "ikca", t_Vovov, "kcjb", 1.0, t_W, "ijab"); 
        tblis::mult<double>(-1.0, t_Vovov, "iakc", t_T, "kjbc", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikbc", t_T, "kjac", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T, "ikcb", t_Voovv, "jkac", 1.0, t_W, "ijab"); 

        // Particle-Ladder DF O(N^5) Factored
        Eigen::Tensor<double, 4> X_temp(na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_Xtemp, X_temp, na_, na_, va_, va_);
        for (int P = 0; P < n_aux; ++P) {
            TBLIS_VIEW_2D(t_B_P, B_ab_a.col(P).data(), va_, va_);
            X_temp.setZero();
            tblis::mult<double>(1.0, t_T, "ijef", t_B_P, "fb", 0.0, t_Xtemp, "ijeb");
            tblis::mult<double>(1.0, t_Xtemp, "ijeb", t_B_P, "ea", 1.0, t_W, "ijab");
        }

        double e3_aa = 0.0;
        #pragma omp parallel for collapse(4) reduction(+:e3_aa) schedule(static)
        for (int i = 0; i < na_; ++i) {
            for (int j = 0; j < na_; ++j) {
                for (int a = 0; a < va_; ++a) {
                    for (int b = 0; b < va_; ++b) {
                        double D = ea(i) + ea(j) - ea(na_+a) - ea(na_+b);
                        t2_3rd_aa_(i,j,a,b) = W(i,j,a,b) / D; 
                        e3_aa += W(i, j, a, b) * (2.0 * T2_aa_ijab(i, j, a, b) - T2_aa_ijab(i, j, b, a));
                    }
                }
            }
        }
        e_mp3_tot_ = e3_aa;
        return; 
    }

    // ================= UMP3 BLOCK ================= //
    t2_3rd_aa_ = Eigen::Tensor<double, 4>(na_, na_, va_, va_); t2_3rd_aa_.setZero();
    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        t2_3rd_bb_ = Eigen::Tensor<double, 4>(nb_, nb_, vb_, vb_); t2_3rd_bb_.setZero();
        t2_3rd_ab_ = Eigen::Tensor<double, 4>(na_, nb_, va_, vb_); t2_3rd_ab_.setZero();
    }

    double e3_aa = 0.0, e3_bb = 0.0, e3_ab = 0.0;
    TBLIS_VIEW_4D(t_Taa, T2_aa_ijab, na_, na_, va_, va_);

    // --- ALPHA-ALPHA BLOCK ---
    {
        if (Waa_ladder_.size() == 0) Waa_ladder_.resize(na_, na_, va_, va_);
        if (Waa_ring_.size() == 0) Waa_ring_.resize(na_, na_, va_, va_);
        
        Waa_ladder_.setZero(); Waa_ring_.setZero();
        TBLIS_VIEW_4D(t_Waa_ladder, Waa_ladder_, na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_Waa_ring, Waa_ring_, na_, na_, va_, va_);

        Eigen::Tensor<double, 4> V_oooo_aa = map_4d(B_ij_a * B_ij_a.transpose(), na_, na_, na_, na_);
        Eigen::Tensor<double, 4> V_ovov_aa = map_4d(B_ia_P_alpha_ * B_ia_P_alpha_.transpose(), na_, va_, na_, va_);
        Eigen::Tensor<double, 4> V_oovv_aa = map_4d(B_ij_a * B_ab_a.transpose(), na_, na_, va_, va_);
        
        TBLIS_VIEW_4D(t_Voooo, V_oooo_aa, na_, na_, na_, na_);
        TBLIS_VIEW_4D(t_Vovov, V_ovov_aa, na_, va_, na_, va_);
        TBLIS_VIEW_4D(t_Voovv, V_oovv_aa, na_, na_, va_, va_);

        tblis::mult<double>(0.5, t_Taa, "mnab", t_Voooo, "minj", 1.0, t_Waa_ladder, "ijab");
        tblis::mult<double>(-0.5, t_Taa, "mnab", t_Voooo, "mjni", 1.0, t_Waa_ladder, "ijab");

        tblis::mult<double>(1.0,  t_Vovov, "iakc", t_Taa, "kjcb", 1.0, t_Waa_ring, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikac", t_Taa, "kjcb", 1.0, t_Waa_ring, "ijab");

        if (!is_restricted && nb_ > 0 && vb_ > 0) {
            Eigen::Tensor<double, 4> V_ovov_ab = map_4d(B_ia_P_alpha_ * B_ia_P_beta_.transpose(), na_, va_, nb_, vb_);
            TBLIS_VIEW_4D(t_Vovov_ab, V_ovov_ab, na_, va_, nb_, vb_);
            auto* t2_ab_dense = t2_ab_.get_block(0,0,0,0);
            TBLIS_VIEW_4D(t_Tab, (*t2_ab_dense), na_, nb_, va_, vb_);
            tblis::mult<double>(1.0, t_Vovov_ab, "iakc", t_Tab, "jkbc", 1.0, t_Waa_ring, "ijab");
        }

        Eigen::Tensor<double, 4> X_temp_aa(na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_Xtemp_aa, X_temp_aa, na_, na_, va_, va_);
        for (int P = 0; P < n_aux; ++P) {
            TBLIS_VIEW_2D(t_B_P_a, B_ab_a.col(P).data(), va_, va_);
            X_temp_aa.setZero();
            tblis::mult<double>(1.0, t_Taa, "ijef", t_B_P_a, "fb", 0.0, t_Xtemp_aa, "ijeb");
            tblis::mult<double>(0.5, t_Xtemp_aa, "ijeb", t_B_P_a, "ea", 1.0, t_Waa_ladder, "ijab");
            tblis::mult<double>(-0.5, t_Xtemp_aa, "ijea", t_B_P_a, "eb", 1.0, t_Waa_ladder, "ijab");
        }

        #pragma omp parallel for collapse(4) reduction(+:e3_aa) schedule(static)
        for(int i=0; i<na_; ++i) {
            for(int j=0; j<na_; ++j) {
                for(int a=0; a<va_; ++a) {
                    for(int b=0; b<va_; ++b) {
                        double r_asym = Waa_ring_(i,j,a,b) - Waa_ring_(j,i,a,b) - Waa_ring_(i,j,b,a) + Waa_ring_(j,i,b,a);
                        double w_tot = Waa_ladder_(i,j,a,b) + r_asym;
                        double D = ea(i) + ea(j) - ea(na_+a) - ea(na_+b);
                        t2_3rd_aa_(i,j,a,b) = w_tot / D;
                        e3_aa += 0.25 * T2_aa_ijab(i,j,a,b) * w_tot;
                    }
                }
            }
        }
    }

    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        const Eigen::MatrixXd& Cbo = scf_.C_beta.leftCols(nb_); 
        const Eigen::MatrixXd& Cbv = scf_.C_beta.rightCols(vb_);
        const auto& eb = scf_.orbital_energies_beta;
        
        Eigen::MatrixXd B_ij_b = build_B_mat(Cbo, Cbo, nb_, nb_);
        Eigen::MatrixXd B_ab_b = build_B_mat(Cbv, Cbv, vb_, vb_);

        auto* t2_bb_dense = t2_bb_.get_block(0,0,0,0);
        auto* t2_ab_dense = t2_ab_.get_block(0,0,0,0);

        TBLIS_VIEW_4D(t_Tbb, (*t2_bb_dense), nb_, nb_, vb_, vb_);
        TBLIS_VIEW_4D(t_Tab, (*t2_ab_dense), na_, nb_, va_, vb_);

        // ----- BETA-BETA BLOCK -----
        {
            Eigen::Tensor<double, 4> Wbb_ladder(nb_, nb_, vb_, vb_); Wbb_ladder.setZero();
            Eigen::Tensor<double, 4> Wbb_ring(nb_, nb_, vb_, vb_); Wbb_ring.setZero();
           
            TBLIS_VIEW_4D(t_Wbb_ladder, Wbb_ladder, nb_, nb_, vb_, vb_);
            TBLIS_VIEW_4D(t_Wbb_ring, Wbb_ring, nb_, nb_, vb_, vb_);

            Eigen::Tensor<double, 4> V_oooo_bb = map_4d(B_ij_b * B_ij_b.transpose(), nb_, nb_, nb_, nb_);
            Eigen::Tensor<double, 4> V_ovov_bb = map_4d(B_ia_P_beta_ * B_ia_P_beta_.transpose(), nb_, vb_, nb_, vb_);
            Eigen::Tensor<double, 4> V_oovv_bb = map_4d(B_ij_b * B_ab_b.transpose(), nb_, nb_, vb_, vb_);
            
            TBLIS_VIEW_4D(t_Voooo_bb, V_oooo_bb, nb_, nb_, nb_, nb_);
            TBLIS_VIEW_4D(t_Vovov_bb, V_ovov_bb, nb_, vb_, nb_, vb_);
            TBLIS_VIEW_4D(t_Voovv_bb, V_oovv_bb, nb_, nb_, vb_, vb_);

            tblis::mult<double>(0.5, t_Tbb, "mnab", t_Voooo_bb, "minj", 1.0, t_Wbb_ladder, "ijab");
            tblis::mult<double>(-0.5, t_Tbb, "mnab", t_Voooo_bb, "mjni", 1.0, t_Wbb_ladder, "ijab");
     
            tblis::mult<double>(1.0,  t_Vovov_bb, "iakc", t_Tbb, "kjcb", 1.0, t_Wbb_ring, "ijab");
            tblis::mult<double>(-1.0, t_Voovv_bb, "ikac", t_Tbb, "kjcb", 1.0, t_Wbb_ring, "ijab");

            Eigen::Tensor<double, 4> V_ovov_ab = map_4d(B_ia_P_alpha_ * B_ia_P_beta_.transpose(), na_, va_, nb_, vb_);
            TBLIS_VIEW_4D(t_Vovov_ab, V_ovov_ab, na_, va_, nb_, vb_);
            tblis::mult<double>(1.0, t_Vovov_ab, "kcia", t_Tab, "kjcb", 1.0, t_Wbb_ring, "ijab"); 

            Eigen::Tensor<double, 4> X_temp_bb(nb_, nb_, vb_, vb_);
            TBLIS_VIEW_4D(t_Xtemp_bb, X_temp_bb, nb_, nb_, vb_, vb_);
            for (int P = 0; P < n_aux; ++P) {
                TBLIS_VIEW_2D(t_B_P_b, B_ab_b.col(P).data(), vb_, vb_);
                X_temp_bb.setZero();
                tblis::mult<double>(1.0, t_Tbb, "ijef", t_B_P_b, "fb", 0.0, t_Xtemp_bb, "ijeb");
                tblis::mult<double>(0.5, t_Xtemp_bb, "ijeb", t_B_P_b, "ea", 1.0, t_Wbb_ladder, "ijab");
                tblis::mult<double>(-0.5, t_Xtemp_bb, "ijea", t_B_P_b, "eb", 1.0, t_Wbb_ladder, "ijab");
            }

            #pragma omp parallel for collapse(4) reduction(+:e3_bb) schedule(static)
            for(int i=0; i<nb_; ++i) {
                for(int j=0; j<nb_; ++j) {
                    for(int a=0; a<vb_; ++a) {
                        for(int b=0; b<vb_; ++b) {
                            double r_asym = Wbb_ring(i,j,a,b) - Wbb_ring(j,i,a,b) - Wbb_ring(i,j,b,a) + Wbb_ring(j,i,b,a);
                            double w_tot = Wbb_ladder(i,j,a,b) + r_asym;
                            double D = eb(i) + eb(j) - eb(nb_+a) - eb(nb_+b);
                            t2_3rd_bb_(i,j,a,b) = w_tot / D;
                            e3_bb += 0.25 * (*t2_bb_dense)(i,j,a,b) * w_tot;
                        }
                    }
                }
            }
        }

        // ----- ALPHA-BETA BLOCK -----
        {
            Eigen::Tensor<double, 4> Wab_ladder(na_, nb_, va_, vb_); Wab_ladder.setZero();
            Eigen::Tensor<double, 4> Wab_ring(na_, nb_, va_, vb_); Wab_ring.setZero();
            
            TBLIS_VIEW_4D(t_Wab_ladder, Wab_ladder, na_, nb_, va_, vb_);
            TBLIS_VIEW_4D(t_Wab_ring, Wab_ring, na_, nb_, va_, vb_);

            Eigen::Tensor<double, 4> V_oooo_ab = map_4d(B_ij_a * B_ij_b.transpose(), na_, na_, nb_, nb_);
            TBLIS_VIEW_4D(t_Voooo_ab, V_oooo_ab, na_, na_, nb_, nb_);
            tblis::mult<double>(1.0, t_Tab, "mnab", t_Voooo_ab, "minj", 1.0, t_Wab_ladder, "ijab");

            Eigen::Tensor<double, 4> V_ovov_aa = map_4d(B_ia_P_alpha_ * B_ia_P_alpha_.transpose(), na_, va_, na_, va_);
            Eigen::Tensor<double, 4> V_oovv_aa = map_4d(B_ij_a * B_ab_a.transpose(), na_, na_, va_, va_);
            TBLIS_VIEW_4D(t_Vovov_aa, V_ovov_aa, na_, va_, na_, va_);
            TBLIS_VIEW_4D(t_Voovv_aa, V_oovv_aa, na_, na_, va_, va_);

            tblis::mult<double>(1.0,  t_Vovov_aa, "iakc", t_Tab, "kjcb", 1.0, t_Wab_ring, "ijab");
            tblis::mult<double>(-1.0, t_Voovv_aa, "ikac", t_Tab, "kjcb", 1.0, t_Wab_ring, "ijab");

            Eigen::Tensor<double, 4> V_ovov_bb = map_4d(B_ia_P_beta_ * B_ia_P_beta_.transpose(), nb_, vb_, nb_, vb_);
            Eigen::Tensor<double, 4> V_oovv_bb = map_4d(B_ij_b * B_ab_b.transpose(), nb_, nb_, vb_, vb_);
            TBLIS_VIEW_4D(t_Vovov_bb, V_ovov_bb, nb_, vb_, nb_, vb_);
            TBLIS_VIEW_4D(t_Voovv_bb, V_oovv_bb, nb_, nb_, vb_, vb_);

            tblis::mult<double>(1.0,  t_Tab, "ikac", t_Vovov_bb, "kcjb", 1.0, t_Wab_ring, "ijab"); 
            tblis::mult<double>(-1.0, t_Tab, "ikac", t_Voovv_bb, "kjcb", 1.0, t_Wab_ring, "ijab");

            Eigen::Tensor<double, 4> V_ovov_ab = map_4d(B_ia_P_alpha_ * B_ia_P_beta_.transpose(), na_, va_, nb_, vb_);
            TBLIS_VIEW_4D(t_Vovov_ab, V_ovov_ab, na_, va_, nb_, vb_);

            tblis::mult<double>(1.0,  t_Taa, "ikac", t_Vovov_ab, "kcjb", 1.0, t_Wab_ring, "ijab");
            tblis::mult<double>(1.0,  t_Vovov_ab, "iakc", t_Tbb, "kjcb", 1.0, t_Wab_ring, "ijab");

            Eigen::Tensor<double, 4> V_oovv_ab_ex = map_4d(B_ij_a * B_ab_b.transpose(), na_, na_, vb_, vb_);
            Eigen::Tensor<double, 4> V_oovv_ba_ex = map_4d(B_ij_b * B_ab_a.transpose(), nb_, nb_, va_, va_);
            TBLIS_VIEW_4D(t_Voovv_ab_ex, V_oovv_ab_ex, na_, na_, vb_, vb_);
            TBLIS_VIEW_4D(t_Voovv_ba_ex, V_oovv_ba_ex, nb_, nb_, va_, va_);

            tblis::mult<double>(-1.0, t_Voovv_ab_ex, "ikbc", t_Tab, "kjac", 1.0, t_Wab_ring, "ijab");
            tblis::mult<double>(-1.0, t_Tab, "ikcb", t_Voovv_ba_ex, "jkac", 1.0, t_Wab_ring, "ijab");
          
            Eigen::Tensor<double, 4> X_temp_ab(na_, nb_, va_, vb_);
            TBLIS_VIEW_4D(t_Xtemp_ab, X_temp_ab, na_, nb_, va_, vb_);
            for (int P = 0; P < n_aux; ++P) {
                TBLIS_VIEW_2D(t_B_P_a, B_ab_a.col(P).data(), va_, va_);
                TBLIS_VIEW_2D(t_B_P_b, B_ab_b.col(P).data(), vb_, vb_);
                
                X_temp_ab.setZero();
                tblis::mult<double>(1.0, t_Tab, "ijef", t_B_P_b, "fb", 0.0, t_Xtemp_ab, "ijeb");
                tblis::mult<double>(1.0, t_Xtemp_ab, "ijeb", t_B_P_a, "ea", 1.0, t_Wab_ladder, "ijab");
            }

            #pragma omp parallel for collapse(4) reduction(+:e3_ab) schedule(static)
            for(int i=0; i<na_; ++i) {
                for(int j=0; j<nb_; ++j) {
                    for(int a=0; a<va_; ++a) {
                        for(int b=0; b<vb_; ++b) {
                            double w_tot = Wab_ladder(i,j,a,b) + Wab_ring(i,j,a,b); 
                            double D = ea(i) + eb(j) - ea(na_+a) - eb(nb_+b);
                            t2_3rd_ab_(i,j,a,b) = w_tot / D;
                            e3_ab += 1.0 * (*t2_ab_dense)(i,j,a,b) * w_tot;
                        }
                    }
                }
            }
        }
    }
    
    e_mp3_tot_ = e3_aa + e3_bb + e3_ab;
}



MP3Result OMP3::compute_omp3() {
    auto t_start = std::chrono::high_resolution_clock::now();
    if(omp_get_thread_num() == 0) {
        std::cout << "\n========================================================\n";
        std::cout << "      Orbital-Optimized MP3 (MP3 @ OMP2 Single-Shot)\n";
        std::cout << "========================================================\n";
    }
    
    // 1. Eksekusi siklus OMP2 murni sampai konvergen
    // Karena kita menghapus override execute_micro_iterations, ini akan memanggil OMP2 murni!
    MP2Result res2 = OMP2::compute();

    if(omp_get_thread_num() == 0) {
        std::cout << "\n  [OMP3] Orbital teroptimasi diperoleh. Memulai evaluasi energi MP3...\n";
    }

    // 2. Gunakan orbital OMP2 yang sudah konvergen untuk menghitung koreksi MP3
    compute_mp3_correction();

    // 3. Susun hasil akhir
    MP3Result res3;
    res3.converged = res2.converged;
    res3.iterations = res2.iterations;
    res3.e_hf = res2.energy_scf; // Energi referensi non-korelasi dari orbital OMP2
    res3.e_mp2 = e_ss_ + e_os_;  // Energi korelasi OMP2
    res3.e_mp3 = e_mp3_tot_;     // Energi korelasi MP3 (Evaluasi Single-Shot)
    res3.e_corr_total = res3.e_mp2 + res3.e_mp3;
    res3.e_total = res3.e_hf + res3.e_corr_total;

    auto t_end = std::chrono::high_resolution_clock::now();
    if(omp_get_thread_num() == 0) {
        std::cout << "--------------------------------------------------------\n";
        std::cout << "  E_MP2 (OMP2) : " << std::fixed << std::setprecision(8) << res3.e_mp2 << " Ha\n";
        std::cout << "  E_MP3 (Corr) : " << res3.e_mp3 << " Ha\n";
        std::cout << "  Total Energy : " << res3.e_total << " Ha\n";
        std::cout << "  Total Time   : " << std::chrono::duration<double>(t_end - t_start).count() << " s\n";
        std::cout << "========================================================\n";
    }

    return res3;
}
}