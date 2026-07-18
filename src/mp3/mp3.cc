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



double OMP3::get_correlation_energy() const {
    return e_ss_ + e_os_ + e_mp3_tot_;
}

double OMP3::execute_micro_iterations() {
    OMP2::execute_micro_iterations();
    
    compute_mp3_correction(); 
    
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1); 
    
    L2_aa_ = t2_3rd_aa_;
    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        L2_bb_ = t2_3rd_bb_;
        L2_ab_ = t2_3rd_ab_;
    }
    build_opdm_alpha();
    if (!is_restricted && nb_ > 0) {
        build_opdm_beta();
    } else if (is_restricted && nb_ > 0) {
        G_oo_beta_ = G_oo_alpha_;
    }

    return get_correlation_energy();
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
    #pragma omp parallel for collapse(4)
    for(int i=0; i<na_; ++i) {
        for(int j=0; j<na_; ++j) {
            for(int a=0; a<va_; ++a) {
                for(int b=0; b<va_; ++b) {
                    T2_aa_ijab(i,j,a,b) = (*t2_aa_dense)(i,a,j,b);
                }
            }
        }
    }

    if (is_restricted) {
        t2_3rd_aa_ = Eigen::Tensor<double, 4>(na_, na_, va_, va_);
        Eigen::Tensor<double, 4> W(na_, na_, va_, va_); W.setZero();
        
        TBLIS_VIEW_4D(t_T, T2_aa_ijab, na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_W, W, na_, na_, va_, va_);
      
        int n_aux = scf_.L_mat.cols();
        Eigen::MatrixXd B_vv_a = Eigen::MatrixXd::Zero(va_ * va_, n_aux);
        Eigen::Map<const Eigen::MatrixXd> L_flat(scf_.L_mat.data(), nbf_, nbf_ * n_aux);
        Eigen::MatrixXd X_a = Cav.transpose() * L_flat;
        #pragma omp parallel for schedule(static)
        for (int P = 0; P < n_aux; ++P) {
            Eigen::Map<Eigen::MatrixXd> X_P(X_a.data() + P * va_ * nbf_, va_, nbf_);
            Eigen::MatrixXd B_MO = X_P * Cav;
            std::copy(B_MO.data(), B_MO.data() + va_ * va_, B_vv_a.col(P).data());
        }
        
        Eigen::Map<const Eigen::MatrixXd> T2_flat(T2_aa_ijab.data(), na_*na_, va_*va_);
        Eigen::MatrixXd W_flat = (T2_flat * B_vv_a) * B_vv_a.transpose(); 
        #pragma omp parallel for collapse(4)
        for (int i = 0; i < na_; ++i) {
            for (int j = 0; j < na_; ++j) {
                for (int a = 0; a < va_; ++a) {
                    for (int b = 0; b < va_; ++b) {
                        W(i,j,a,b) = W_flat(i + j*na_, a + b*va_);
                    }
                }
            }
        }

        auto V_oooo = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cao, Cao, Cao, Cao, integrals_);
        auto V_ovov = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cao, Cav, Cao, Cav, integrals_);
        auto V_oovv = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cao, Cao, Cav, Cav, integrals_);
        
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

        double e3_aa = 0.0;
        #pragma omp parallel for collapse(4) reduction(+:e3_aa)
        for (int i = 0; i < na_; ++i) {
            for (int j = 0; j < na_; ++j) {
                for (int a = 0; a < va_; ++a) {
                    for (int b = 0; b < va_; ++b) {
                        double D = ea(i) + ea(j) - ea(na_+a) - ea(na_+b);
                        double reg_den = D / (D * D + 1e-20);
                        t2_3rd_aa_(i,j,a,b) = W(i,j,a,b) * reg_den; 
                        e3_aa += W(i, j, a, b) * (2.0 * T2_aa_ijab(i, j, a, b) - T2_aa_ijab(i, j, b, a));
                    }
                }
            }
        }
        e_mp3_tot_ = e3_aa;
        return; 
    }

    t2_3rd_aa_ = Eigen::Tensor<double, 4>(na_, na_, va_, va_); t2_3rd_aa_.setZero();
    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        t2_3rd_bb_ = Eigen::Tensor<double, 4>(nb_, nb_, vb_, vb_); t2_3rd_bb_.setZero();
        t2_3rd_ab_ = Eigen::Tensor<double, 4>(na_, nb_, va_, vb_); t2_3rd_ab_.setZero();
    }

    double e3_aa = 0.0, e3_bb = 0.0, e3_ab = 0.0;
    double e3_aa_ump3 = 0.0; 
    TBLIS_VIEW_4D(t_Taa, T2_aa_ijab, na_, na_, va_, va_);

    int n_aux = scf_.L_mat.cols();
    Eigen::MatrixXd B_vv_a, B_vv_b;
    B_vv_a.setZero(va_ * va_, n_aux);
    Eigen::Map<const Eigen::MatrixXd> L_flat(scf_.L_mat.data(), nbf_, nbf_ * n_aux);
    Eigen::MatrixXd X_a = Cav.transpose() * L_flat;
    
    #pragma omp parallel for schedule(static)
    for (int P = 0; P < n_aux; ++P) {
        Eigen::Map<Eigen::MatrixXd> X_P(X_a.data() + P * va_ * nbf_, va_, nbf_);
        Eigen::MatrixXd B_MO = X_P * Cav;
        std::copy(B_MO.data(), B_MO.data() + va_ * va_, B_vv_a.col(P).data());
    }

    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        B_vv_b.setZero(vb_ * vb_, n_aux);
        Eigen::MatrixXd X_b = scf_.C_beta.rightCols(vb_).transpose() * L_flat;
        #pragma omp parallel for schedule(static)
        for (int P = 0; P < n_aux; ++P) {
            Eigen::Map<Eigen::MatrixXd> X_P(X_b.data() + P * vb_ * nbf_, vb_, nbf_);
            Eigen::MatrixXd B_MO = X_P * scf_.C_beta.rightCols(vb_);
            std::copy(B_MO.data(), B_MO.data() + vb_ * vb_, B_vv_b.col(P).data());
        }
    }


    {
        Eigen::Tensor<double, 4> Waa_ladder(na_, na_, va_, va_); Waa_ladder.setZero();
        Eigen::Tensor<double, 4> Waa_ring(na_, na_, va_, va_); Waa_ring.setZero();
        TBLIS_VIEW_4D(t_Waa_ladder, Waa_ladder, na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_Waa_ring, Waa_ring, na_, na_, va_, va_);

      
        auto Vvvvv_aa = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cav, Cav, Cav, Cav, integrals_);
        TBLIS_VIEW_4D(t_Vvvvv_aa, Vvvvv_aa, va_, va_, va_, va_);

        tblis::mult<double>(0.5, t_Taa, "ijef", t_Vvvvv_aa, "eafb", 1.0, t_Waa_ladder, "ijab");
        tblis::mult<double>(-0.5, t_Taa, "ijef", t_Vvvvv_aa, "ebfa", 1.0, t_Waa_ladder, "ijab");

        auto V_oooo = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cao, Cao, Cao, Cao, integrals_);
        TBLIS_VIEW_4D(t_Voooo, V_oooo, na_, na_, na_, na_);
        tblis::mult<double>(0.5, t_Taa, "mnab", t_Voooo, "minj", 1.0, t_Waa_ladder, "ijab");
        tblis::mult<double>(-0.5, t_Taa, "mnab", t_Voooo, "mjni", 1.0, t_Waa_ladder, "ijab");

        auto V_ovov = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cao, Cav, Cao, Cav, integrals_);
        auto V_oovv = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cao, Cao, Cav, Cav, integrals_);
        TBLIS_VIEW_4D(t_Vovov, V_ovov, na_, va_, na_, va_);
        TBLIS_VIEW_4D(t_Voovv, V_oovv, na_, na_, va_, va_);

        tblis::mult<double>(1.0,  t_Vovov, "iakc", t_Taa, "kjcb", 1.0, t_Waa_ring, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikac", t_Taa, "kjcb", 1.0, t_Waa_ring, "ijab");

        if (!is_restricted && nb_ > 0 && vb_ > 0) {
            auto V_ovov_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cao, Cav, scf_.C_beta.leftCols(nb_), scf_.C_beta.rightCols(vb_), integrals_);
            TBLIS_VIEW_4D(t_Vovov_ab, V_ovov_ab, na_, va_, nb_, vb_);
            auto* t2_ab_dense = t2_ab_.get_block(0,0,0,0);
            TBLIS_VIEW_4D(t_Tab, (*t2_ab_dense), na_, nb_, va_, vb_);
            tblis::mult<double>(1.0, t_Vovov_ab, "iakc", t_Tab, "jkbc", 1.0, t_Waa_ring, "ijab");
        }

        #pragma omp parallel for collapse(4) reduction(+:e3_aa)
        for(int i=0; i<na_; ++i) {
            for(int j=0; j<na_; ++j) {
                for(int a=0; a<va_; ++a) {
                    for(int b=0; b<va_; ++b) {
                        double r_asym = Waa_ring(i,j,a,b) - Waa_ring(j,i,a,b) - Waa_ring(i,j,b,a) + Waa_ring(j,i,b,a);
                        double w_tot = Waa_ladder(i,j,a,b) + r_asym;
                        double D = ea(i) + ea(j) - ea(na_+a) - ea(na_+b);
                        double reg_den = D / (D * D + 1e-20);
                        t2_3rd_aa_(i,j,a,b) = w_tot * reg_den;
                        
                        if (is_restricted) {
                            e3_aa += w_tot * (2.0 * T2_aa_ijab(i,j,a,b) - T2_aa_ijab(i,j,b,a));
                        } else {
                            e3_aa += 0.25 * T2_aa_ijab(i,j,a,b) * w_tot;
                        }
                    }
                }
            }
        }
    }
    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        const Eigen::MatrixXd& Cbo = scf_.C_beta.leftCols(nb_); 
        const Eigen::MatrixXd& Cbv = scf_.C_beta.rightCols(vb_);
        const auto& eb = scf_.orbital_energies_beta;
        
        auto* t2_bb_dense = t2_bb_.get_block(0,0,0,0);
        auto* t2_ab_dense = t2_ab_.get_block(0,0,0,0);

        TBLIS_VIEW_4D(t_Tbb, (*t2_bb_dense), nb_, nb_, vb_, vb_);
        TBLIS_VIEW_4D(t_Tab, (*t2_ab_dense), na_, nb_, va_, vb_);

        // ----- BETA-BETA BLOCK -----
        {
            Eigen::Tensor<double, 4> Wbb_ladder(nb_, nb_, vb_, vb_); Wbb_ladder.setZero();
            Eigen::Tensor<double, 4> Wbb_ring(nb_, nb_, vb_, vb_); Wbb_ring.setZero();
           
            Eigen::Map<const Eigen::MatrixXd> T2bb_flat(t2_bb_dense->data(), nb_*nb_, vb_*vb_);
            Eigen::MatrixXd Wbb_flat = (T2bb_flat * B_vv_b) * B_vv_b.transpose();

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < nb_; ++i) {
                for (int j = 0; j < nb_; ++j) {
                    int ij = i + j * nb_;
                    for (int a = 0; a < vb_; ++a) {
                        for (int b = 0; b < vb_; ++b) {
                            int ab = a + b * vb_; 
                            int ba = b + a * vb_;
                            Wbb_ladder(i,j,a,b) = 0.5 * (Wbb_flat(ij, ab) - Wbb_flat(ij, ba));
                        }
                    }
                }
            }
            TBLIS_VIEW_4D(t_Wbb_ladder, Wbb_ladder, nb_, nb_, vb_, vb_);
            TBLIS_VIEW_4D(t_Wbb_ring, Wbb_ring, nb_, nb_, vb_, vb_);

            auto V_oooo = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cbo, Cbo, Cbo, Cbo, integrals_);
            TBLIS_VIEW_4D(t_Voooo, V_oooo, nb_, nb_, nb_, nb_);
            tblis::mult<double>(0.5, t_Tbb, "mnab", t_Voooo, "minj", 1.0, t_Wbb_ladder, "ijab");
            tblis::mult<double>(-0.5, t_Tbb, "mnab", t_Voooo, "mjni", 1.0, t_Wbb_ladder, "ijab");

            auto V_ovov = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cbo, Cbv, Cbo, Cbv, integrals_);
            auto V_oovv = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cbo, Cbo, Cbv, Cbv, integrals_);
            TBLIS_VIEW_4D(t_Vovov, V_ovov, nb_, vb_, nb_, vb_);
            TBLIS_VIEW_4D(t_Voovv, V_oovv, nb_, nb_, vb_, vb_);

            tblis::mult<double>(1.0,  t_Vovov, "iakc", t_Tbb, "kjcb", 1.0, t_Wbb_ring, "ijab");
            tblis::mult<double>(-1.0, t_Voovv, "ikac", t_Tbb, "kjcb", 1.0, t_Wbb_ring, "ijab");

            auto V_ovov_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cao, Cav, Cbo, Cbv, integrals_);
            TBLIS_VIEW_4D(t_Vovov_ab, V_ovov_ab, na_, va_, nb_, vb_);
            tblis::mult<double>(1.0, t_Vovov_ab, "kcia", t_Tab, "kjcb", 1.0, t_Wbb_ring, "ijab");

            #pragma omp parallel for collapse(4) reduction(+:e3_bb)
            for(int i=0; i<nb_; ++i) {
                for(int j=0; j<nb_; ++j) {
                    for(int a=0; a<vb_; ++a) {
                        for(int b=0; b<vb_; ++b) {
                            double r_asym = Wbb_ring(i,j,a,b) - Wbb_ring(j,i,a,b) - Wbb_ring(i,j,b,a) + Wbb_ring(j,i,b,a);
                            double w_tot = Wbb_ladder(i,j,a,b) + r_asym;
                            double D = eb(i) + eb(j) - eb(nb_+a) - eb(nb_+b);
                            double reg_den = D / (D * D + 1e-20);
                            t2_3rd_bb_(i,j,a,b) = w_tot * reg_den;
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
            
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < na_; ++i) {
                for (int j = 0; j < nb_; ++j) {
                    Eigen::MatrixXd T_ij(va_, vb_);
                    for(int a=0; a<va_; ++a) for(int b=0; b<vb_; ++b) T_ij(a,b) = (*t2_ab_dense)(i,j,a,b);
                    
                    Eigen::MatrixXd W_ij = Eigen::MatrixXd::Zero(va_, vb_);
                    for (int P = 0; P < n_aux; ++P) {
                        Eigen::Map<const Eigen::MatrixXd> B_Pa(B_vv_a.col(P).data(), va_, va_);
                        Eigen::Map<const Eigen::MatrixXd> B_Pb(B_vv_b.col(P).data(), vb_, vb_);
                        W_ij.noalias() += B_Pa * T_ij * B_Pb; 
                    }
                    for(int a=0; a<va_; ++a) for(int b=0; b<vb_; ++b) {
                        Wab_ladder(i,j,a,b) = W_ij(a,b); 
                    }
                }
            }

            TBLIS_VIEW_4D(t_Wab_ladder, Wab_ladder, na_, nb_, va_, vb_);
            TBLIS_VIEW_4D(t_Wab_ring, Wab_ring, na_, nb_, va_, vb_);

            auto V_oooo_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cao, Cao, Cbo, Cbo, integrals_);
            TBLIS_VIEW_4D(t_Voooo_ab, V_oooo_ab, na_, na_, nb_, nb_);
            tblis::mult<double>(1.0, t_Tab, "mnab", t_Voooo_ab, "minj", 1.0, t_Wab_ladder, "ijab");

            auto V_ovov_aa = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cao, Cav, Cao, Cav, integrals_);
            auto V_oovv_aa = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cao, Cao, Cav, Cav, integrals_);
            TBLIS_VIEW_4D(t_Vovov_aa, V_ovov_aa, na_, va_, na_, va_);
            TBLIS_VIEW_4D(t_Voovv_aa, V_oovv_aa, na_, na_, va_, va_);
            
            tblis::mult<double>(1.0,  t_Vovov_aa, "iakc", t_Tab, "kjcb", 1.0, t_Wab_ring, "ijab");
            tblis::mult<double>(-1.0, t_Voovv_aa, "ikac", t_Tab, "kjcb", 1.0, t_Wab_ring, "ijab");

            auto V_ovov_bb = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cbo, Cbv, Cbo, Cbv, integrals_);
            auto V_oovv_bb = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cbo, Cbo, Cbv, Cbv, integrals_);
            TBLIS_VIEW_4D(t_Vovov_bb, V_ovov_bb, nb_, vb_, nb_, vb_);
            TBLIS_VIEW_4D(t_Voovv_bb, V_oovv_bb, nb_, nb_, vb_, vb_);

            tblis::mult<double>(1.0,  t_Tab, "ikac", t_Vovov_bb, "kcjb", 1.0, t_Wab_ring, "ijab"); 
            tblis::mult<double>(-1.0, t_Tab, "ikac", t_Voovv_bb, "kjcb", 1.0, t_Wab_ring, "ijab"); 

            auto V_ovov_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cao, Cav, Cbo, Cbv, integrals_);
            TBLIS_VIEW_4D(t_Vovov_ab, V_ovov_ab, na_, va_, nb_, vb_);

            tblis::mult<double>(1.0,  t_Taa, "ikac", t_Vovov_ab, "kcjb", 1.0, t_Wab_ring, "ijab");
            tblis::mult<double>(1.0,  t_Vovov_ab, "iakc", t_Tbb, "kjcb", 1.0, t_Wab_ring, "ijab");

            auto V_oovv_ab_ex = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cao, Cao, Cbv, Cbv, integrals_);
            auto V_oovv_ba_ex = ERITransformer::get_mo_tensor(config_.use_df, n_aux, Cbo, Cbo, Cav, Cav, integrals_);
            TBLIS_VIEW_4D(t_Voovv_ab_ex, V_oovv_ab_ex, na_, na_, vb_, vb_);
            TBLIS_VIEW_4D(t_Voovv_ba_ex, V_oovv_ba_ex, nb_, nb_, va_, va_);

            tblis::mult<double>(-1.0, t_Voovv_ab_ex, "ikbc", t_Tab, "kjac", 1.0, t_Wab_ring, "ijab");
            tblis::mult<double>(-1.0, t_Tab, "ikcb", t_Voovv_ba_ex, "jkac", 1.0, t_Wab_ring, "ijab");
            
            #pragma omp parallel for collapse(4) reduction(+:e3_ab)
            for(int i=0; i<na_; ++i) {
                for(int j=0; j<nb_; ++j) {
                    for(int a=0; a<va_; ++a) {
                        for(int b=0; b<vb_; ++b) {
                            double w_tot = Wab_ladder(i,j,a,b) + Wab_ring(i,j,a,b); 
                            double D = ea(i) + eb(j) - ea(na_+a) - eb(nb_+b);
                            double reg_den = D / (D * D + 1e-20);
                            t2_3rd_ab_(i,j,a,b) = w_tot * reg_den;
                            e3_ab += 1.0 * (*t2_ab_dense)(i,j,a,b) * w_tot;
                        }
                    }
                }
            }
        }
    }
    
    e_mp3_tot_ = e3_aa + e3_bb + e3_ab;
}

void OMP3::build_opdm_alpha() {
    if (L2_aa_.size() == 0) {
        OMP2::build_opdm_alpha();
        return;
    }

    auto* t2_aa_dense = t2_aa_.get_block(0,0,0,0);
    if(!t2_aa_dense) return;

    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    G_oo_alpha_ = Eigen::MatrixXd::Zero(na_, na_);
    G_vv_alpha_ = Eigen::MatrixXd::Zero(va_, va_);
    
    Eigen::Tensor<double, 4> T2_aa_ijab(na_, na_, va_, va_);
    #pragma omp parallel for collapse(4)
    for(int i=0; i<na_; ++i) {
        for(int j=0; j<na_; ++j) {
            for(int a=0; a<va_; ++a) {
                for(int b=0; b<va_; ++b) {
                    T2_aa_ijab(i,j,a,b) = (*t2_aa_dense)(i,a,j,b);
                }
            }
        }
    }

    // ====================================================================
    // PERBAIKAN: Kontraksi Spasial Khusus Restricted
    // ====================================================================
    if (is_restricted) {
        #pragma omp parallel for
        for (int i = 0; i < na_; ++i) {
            for (int j = 0; j < na_; ++j) {
                double p_oo = 0.0;
                for (int k = 0; k < na_; ++k) {
                    for (int a = 0; a < va_; ++a) {
                        for (int b = 0; b < va_; ++b) {
                            double t2_ik = T2_aa_ijab(i, a, k, b);
                            double t2_jk = T2_aa_ijab(j, a, k, b);
                            double t2_jk_ex = T2_aa_ijab(j, b, k, a); 
                            
                            double t3_ik = L2_aa_(i, a, k, b);
                            double t3_jk = L2_aa_(j, a, k, b);
                            double t3_jk_ex = L2_aa_(j, b, k, a); 

                            p_oo -= 1.0 * t2_ik * (2.0 * t2_jk - t2_jk_ex);
                            p_oo -= 1.0 * t2_ik * (2.0 * t3_jk - t3_jk_ex); 
                            p_oo -= 1.0 * t3_ik * (2.0 * t2_jk - t2_jk_ex); 
                        }
                    }
                }
                G_oo_alpha_(i, j) = p_oo;
            }
        }
        #pragma omp parallel for
        for (int a = 0; a < va_; ++a) {
            for (int b = 0; b < va_; ++b) {
                double p_vv = 0.0;
                for (int i = 0; i < na_; ++i) {
                    for (int j = 0; j < na_; ++j) {
                        for (int c = 0; c < va_; ++c) {
                            double t2_ac = T2_aa_ijab(i, a, j, c);
                            double t2_bc = T2_aa_ijab(i, b, j, c);
                            double t2_cb = T2_aa_ijab(i, c, j, b);
                            
                            double t3_ac = L2_aa_(i, a, j, c);
                            double t3_bc = L2_aa_(i, b, j, c);
                            double t3_cb = L2_aa_(i, c, j, b);

                            p_vv += 1.0 * t2_ac * (2.0 * t2_bc - t2_cb);
                            p_vv += 1.0 * t2_ac * (2.0 * t3_bc - t3_cb); 
                            p_vv += 1.0 * t3_ac * (2.0 * t2_bc - t2_cb);
                        }
                    }
                }
                G_vv_alpha_(a, b) = p_vv;
            }
        }
        return;
    }

    TBLIS_VIEW_4D(t_T2aa, T2_aa_ijab, na_, na_, va_, va_);
    TBLIS_VIEW_4D(t_T3aa, L2_aa_, na_, na_, va_, va_);
    TBLIS_VIEW_2D(t_Goo_a, G_oo_alpha_.data(), na_, na_);
    TBLIS_VIEW_2D(t_Gvv_a, G_vv_alpha_.data(), va_, va_);

    tblis::mult<double>(-0.5, t_T2aa, "ikab", t_T2aa, "jkab", 1.0, t_Goo_a, "ij");
    tblis::mult<double>(-0.5, t_T2aa, "ikab", t_T3aa, "jkab", 1.0, t_Goo_a, "ij"); 
    tblis::mult<double>(-0.5, t_T3aa, "ikab", t_T2aa, "jkab", 1.0, t_Goo_a, "ij"); 
    tblis::mult<double>(0.5, t_T2aa, "ijac", t_T2aa, "ijbc", 1.0, t_Gvv_a, "ab");
    tblis::mult<double>(0.5, t_T2aa, "ijac", t_T3aa, "ijbc", 1.0, t_Gvv_a, "ab"); 
    tblis::mult<double>(0.5, t_T3aa, "ijac", t_T2aa, "ijbc", 1.0, t_Gvv_a, "ab");

    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        auto* t2_ab_dense = t2_ab_.get_block(0,0,0,0);
        TBLIS_VIEW_4D(t_T2ab, (*t2_ab_dense), na_, nb_, va_, vb_);
        TBLIS_VIEW_4D(t_T3ab, L2_ab_, na_, nb_, va_, vb_);

        tblis::mult<double>(-1.0, t_T2ab, "ikab", t_T2ab, "jkab", 1.0, t_Goo_a, "ij");
        tblis::mult<double>(1.0, t_T2ab, "ikab", t_T3ab, "jkab", 1.0, t_Goo_a, "ij"); 
        tblis::mult<double>(1.0, t_T3ab, "ikab", t_T2ab, "jkab", 1.0, t_Goo_a, "ij"); 
        tblis::mult<double>(1.0, t_T2ab, "ijac", t_T2ab, "ijbc", 1.0, t_Gvv_a, "ab");
        tblis::mult<double>(-1.0, t_T2ab, "ijac", t_T3ab, "ijbc", 1.0, t_Gvv_a, "ab"); 
        tblis::mult<double>(-1.0, t_T3ab, "ijac", t_T2ab, "ijbc", 1.0, t_Gvv_a, "ab"); 
    }
}
void OMP3::build_opdm_beta() {
    if (L2_bb_.size() == 0 && L2_ab_.size() == 0) {
        OMP2::build_opdm_beta(); 
        return;
    }
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    G_oo_beta_ = Eigen::MatrixXd::Zero(nb_, nb_);
    G_vv_beta_ = Eigen::MatrixXd::Zero(vb_, vb_);
    
    if (is_restricted || nb_ == 0 || vb_ == 0) return; 

    auto* t2_bb_dense = t2_bb_.get_block(0,0,0,0);
    auto* t2_ab_dense = t2_ab_.get_block(0,0,0,0);
    if(!t2_bb_dense || !t2_ab_dense) return;

    TBLIS_VIEW_4D(t_T2bb, (*t2_bb_dense), nb_, nb_, vb_, vb_);
    TBLIS_VIEW_4D(t_T3bb, L2_bb_, nb_, nb_, vb_, vb_);
    TBLIS_VIEW_4D(t_T2ab, (*t2_ab_dense), na_, nb_, va_, vb_);
    TBLIS_VIEW_4D(t_T3ab, L2_ab_, na_, nb_, va_, vb_);
    TBLIS_VIEW_2D(t_Goo_b, G_oo_beta_.data(), nb_, nb_);
    TBLIS_VIEW_2D(t_Gvv_b, G_vv_beta_.data(), vb_, vb_);

    tblis::mult<double>(-0.5, t_T2bb, "ikab", t_T2bb, "jkab", 1.0, t_Goo_b, "ij");
    tblis::mult<double>(0.5, t_T2bb, "ikab", t_T3bb, "jkab", 1.0, t_Goo_b, "ij"); 
    tblis::mult<double>(0.5, t_T3bb, "ikab", t_T2bb, "jkab", 1.0, t_Goo_b, "ij"); 

    tblis::mult<double>(0.5, t_T2bb, "ijac", t_T2bb, "ijbc", 1.0, t_Gvv_b, "ab");
    tblis::mult<double>(-0.5, t_T2bb, "ijac", t_T3bb, "ijbc", 1.0, t_Gvv_b, "ab"); 
    tblis::mult<double>(-0.5, t_T3bb, "ijac", t_T2bb, "ijbc", 1.0, t_Gvv_b, "ab"); 

    tblis::mult<double>(-1.0, t_T2ab, "kiab", t_T2ab, "kjab", 1.0, t_Goo_b, "ij");
    tblis::mult<double>(1.0, t_T2ab, "kiab", t_T3ab, "kjab", 1.0, t_Goo_b, "ij");
    tblis::mult<double>(1.0, t_T3ab, "kiab", t_T2ab, "kjab", 1.0, t_Goo_b, "ij"); 

    tblis::mult<double>(1.0, t_T2ab, "ijca", t_T2ab, "ijcb", 1.0, t_Gvv_b, "ab");
    tblis::mult<double>(-1.0, t_T2ab, "ijca", t_T3ab, "ijcb", 1.0, t_Gvv_b, "ab"); 
    tblis::mult<double>(-1.0, t_T3ab, "ijca", t_T2ab, "ijcb", 1.0, t_Gvv_b, "ab"); 
}

void OMP3::build_generalized_fock() {
   
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    
    Eigen::MatrixXd G_full_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
    G_full_a.block(0, 0, na_, na_) = G_oo_alpha_; 
    G_full_a.block(na_, na_, va_, va_) = G_vv_alpha_;
    Eigen::MatrixXd P_corr_a = scf_.C_alpha * G_full_a * scf_.C_alpha.transpose();
    
    Eigen::MatrixXd G_full_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    Eigen::MatrixXd P_corr_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    
    if (!is_restricted && nb_ > 0) {
        G_full_b.block(0, 0, nb_, nb_) = G_oo_beta_;
        G_full_b.block(nb_, nb_, vb_, vb_) = G_vv_beta_;
        P_corr_b = scf_.C_beta * G_full_b * scf_.C_beta.transpose();
    } else if (is_restricted) {
        P_corr_b = P_corr_a; 
    }

    Eigen::MatrixXd F_HF_ao_a, F_HF_ao_b;
    build_fock_fast(scf_.P_alpha, scf_.P_beta, F_HF_ao_a, F_HF_ao_b);
    
    Eigen::MatrixXd F_HF_mo_a = scf_.C_alpha.transpose() * F_HF_ao_a * scf_.C_alpha;
    Eigen::MatrixXd F_HF_mo_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    
    if (!is_restricted && nb_ > 0) F_HF_mo_b = scf_.C_beta.transpose() * F_HF_ao_b * scf_.C_beta;
    else if (is_restricted) F_HF_mo_b = F_HF_mo_a;

    Eigen::MatrixXd G_gamma_ao_a, G_gamma_ao_b;
    build_fock_fast(P_corr_a, P_corr_b, G_gamma_ao_a, G_gamma_ao_b);
    
    G_gamma_ao_a -= H_core_;
    if (!is_restricted && nb_ > 0) G_gamma_ao_b -= H_core_;
    else if (is_restricted) G_gamma_ao_b = G_gamma_ao_a;
    
    Eigen::MatrixXd G_gamma_mo_a = scf_.C_alpha.transpose() * G_gamma_ao_a * scf_.C_alpha;
    Eigen::MatrixXd G_gamma_mo_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    
    if (!is_restricted && nb_ > 0) G_gamma_mo_b = scf_.C_beta.transpose() * G_gamma_ao_b * scf_.C_beta;
    else if (is_restricted) G_gamma_mo_b = G_gamma_mo_a;

    F_gen_a_ = F_HF_mo_a + G_gamma_mo_a;
    if (na_ > 0 && va_ > 0) {
        Eigen::MatrixXd F_HF_vo_a = F_HF_mo_a.block(na_, 0, va_, na_);
        Eigen::MatrixXd L_sep_a = G_vv_alpha_ * F_HF_vo_a - F_HF_vo_a * G_oo_alpha_;
        F_gen_a_.block(na_, 0, va_, na_) += L_sep_a;
        F_gen_a_.block(0, na_, na_, va_) += L_sep_a.transpose();
    }

    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        F_gen_b_ = F_HF_mo_b + G_gamma_mo_b;
        Eigen::MatrixXd F_HF_vo_b = F_HF_mo_b.block(nb_, 0, vb_, nb_);
        Eigen::MatrixXd L_sep_b = G_vv_beta_ * F_HF_vo_b - F_HF_vo_b * G_oo_beta_;
        F_gen_b_.block(nb_, 0, vb_, nb_) += L_sep_b;
        F_gen_b_.block(0, nb_, nb_, vb_) += L_sep_b.transpose();
    } else if (is_restricted) {
        F_gen_b_ = F_gen_a_; 
    }

    int n_aux = scf_.L_mat.cols();
    Eigen::MatrixXd B_oo_a = Eigen::MatrixXd::Zero(na_*na_, n_aux);
    Eigen::MatrixXd B_vv_a = Eigen::MatrixXd::Zero(va_*va_, n_aux);
    Eigen::MatrixXd B_oo_b, B_vv_b;
    
    if (!is_restricted && nb_ > 0 && vb_ > 0) { 
        B_oo_b = Eigen::MatrixXd::Zero(nb_*nb_, n_aux); 
        B_vv_b = Eigen::MatrixXd::Zero(vb_*vb_, n_aux); 
    }
    
    const Eigen::MatrixXd& Ca_o = scf_.C_alpha.leftCols(na_);
    const Eigen::MatrixXd& Ca_v = scf_.C_alpha.rightCols(va_);
    const Eigen::MatrixXd& Cb_o = scf_.C_beta.leftCols(nb_);
    const Eigen::MatrixXd& Cb_v = scf_.C_beta.rightCols(vb_);

    #pragma omp parallel
    {
        Eigen::MatrixXd priv_oo_a = Eigen::MatrixXd::Zero(na_*na_, n_aux);
        Eigen::MatrixXd priv_vv_a = Eigen::MatrixXd::Zero(va_*va_, n_aux);
        Eigen::MatrixXd priv_oo_b, priv_vv_b;
        
        if (!is_restricted && nb_ > 0) {
            priv_oo_b = Eigen::MatrixXd::Zero(nb_*nb_, n_aux);
            priv_vv_b = Eigen::MatrixXd::Zero(vb_*vb_, n_aux);
        }
        
        #pragma omp for schedule(dynamic)
        for (int P = 0; P < n_aux; ++P) {
            Eigen::Map<const Eigen::MatrixXd> B_AO(scf_.L_mat.col(P).data(), nbf_, nbf_);
            Eigen::MatrixXd MO_oo_a = Ca_o.transpose() * (B_AO * Ca_o);
            Eigen::MatrixXd MO_vv_a = Ca_v.transpose() * (B_AO * Ca_v);
            for(int i=0; i<na_; ++i) for(int j=0; j<na_; ++j) priv_oo_a(i*na_+j, P) = MO_oo_a(i, j);
            for(int a=0; a<va_; ++a) for(int b=0; b<va_; ++b) priv_vv_a(a*va_+b, P) = MO_vv_a(a, b);
            
            if (!is_restricted && nb_ > 0 && vb_ > 0) {
                Eigen::MatrixXd MO_oo_b = Cb_o.transpose() * (B_AO * Cb_o);
                Eigen::MatrixXd MO_vv_b = Cb_v.transpose() * (B_AO * Cb_v);
                for(int i=0; i<nb_; ++i) for(int j=0; j<nb_; ++j) priv_oo_b(i*nb_+j, P) = MO_oo_b(i, j);
                for(int a=0; a<vb_; ++a) for(int b=0; b<vb_; ++b) priv_vv_b(a*vb_+b, P) = MO_vv_b(a, b);
            }
        }
        #pragma omp critical
        {
            B_oo_a += priv_oo_a; B_vv_a += priv_vv_a;
            if (!is_restricted && nb_ > 0) { B_oo_b += priv_oo_b; B_vv_b += priv_vv_b; }
        }
    }

    Eigen::Tensor<double, 4> Gamma_vvvv_aa(va_, va_, va_, va_); Gamma_vvvv_aa.setZero();
    Eigen::Tensor<double, 4> Gamma_oooo_aa(na_, na_, na_, na_); Gamma_oooo_aa.setZero();
    Eigen::Tensor<double, 4> Gamma_ovov_aa(na_, va_, na_, va_); Gamma_ovov_aa.setZero();

    Eigen::Tensor<double, 4> Gamma_vvvv_bb, Gamma_oooo_bb, Gamma_ovov_bb, Gamma_ovov_ab;
    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        Gamma_vvvv_bb = Eigen::Tensor<double, 4>(vb_, vb_, vb_, vb_); Gamma_vvvv_bb.setZero();
        Gamma_oooo_bb = Eigen::Tensor<double, 4>(nb_, nb_, nb_, nb_); Gamma_oooo_bb.setZero();
        Gamma_ovov_bb = Eigen::Tensor<double, 4>(nb_, vb_, nb_, vb_); Gamma_ovov_bb.setZero();
        Gamma_ovov_ab = Eigen::Tensor<double, 4>(na_, va_, nb_, vb_); Gamma_ovov_ab.setZero();
    }

    auto* t2_aa_dense = t2_aa_.get_block(0,0,0,0);

    Eigen::Tensor<double, 4> T2_aa_ijab(na_, na_, va_, va_);
    #pragma omp parallel for collapse(4)
    for(int i=0; i<na_; ++i) {
        for(int j=0; j<na_; ++j) {
            for(int a=0; a<va_; ++a) {
                for(int b=0; b<va_; ++b) {
                    T2_aa_ijab(i,j,a,b) = (*t2_aa_dense)(i,a,j,b);
                }
            }
        }
    }
    TBLIS_VIEW_4D(t_Taa, T2_aa_ijab, na_, na_, va_, va_);
    TBLIS_VIEW_4D(t_T3aa, L2_aa_, na_, na_, va_, va_);
    TBLIS_VIEW_4D(t_Gvvvv_aa, Gamma_vvvv_aa, va_, va_, va_, va_);
    TBLIS_VIEW_4D(t_Goooo_aa, Gamma_oooo_aa, na_, na_, na_, na_);
    TBLIS_VIEW_4D(t_Govov_aa, Gamma_ovov_aa, na_, va_, na_, va_);

    // 1. Matriks Densitas MP2 Murni (T1 * T1)
    tblis::mult<double>(0.5, t_Taa, "ijab", t_Taa, "ijcd", 1.0, t_Gvvvv_aa, "abcd");
    tblis::mult<double>(0.5, t_Taa, "ijab", t_Taa, "klab", 1.0, t_Goooo_aa, "ijkl");
    tblis::mult<double>(1.0, t_Taa, "ikac", t_Taa, "kjcb", 1.0, t_Govov_aa, "iajb"); 

    Eigen::MatrixXd Teff_aa = Eigen::MatrixXd::Zero(na_*va_, na_*va_);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < na_; ++i) {
        for (int a = 0; a < va_; ++a) {
            for (int j = 0; j < na_; ++j) {
                for (int b = 0; b < va_; ++b) {
                    double dir = T2_aa_ijab(i, j, a, b) + L2_aa_(i, j, a, b);
                    double ex  = T2_aa_ijab(i, j, b, a) + L2_aa_(i, j, b, a);
                    if (is_restricted) {
                        Teff_aa(i*va_+a, j*va_+b) = 2.0 * dir - 1.0 * ex;
                    } else {
                        Teff_aa(i*va_+a, j*va_+b) = dir - ex;
                    }
                }
            }
        }
    }

    Eigen::MatrixXd Teff_ab, Teff_bb;
    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        auto* t2_bb_dense = t2_bb_.get_block(0,0,0,0);
        auto* t2_ab_dense = t2_ab_.get_block(0,0,0,0);
        
        TBLIS_VIEW_4D(t_Tbb, (*t2_bb_dense), nb_, nb_, vb_, vb_);
        TBLIS_VIEW_4D(t_Gvvvv_bb, Gamma_vvvv_bb, vb_, vb_, vb_, vb_);
        TBLIS_VIEW_4D(t_Goooo_bb, Gamma_oooo_bb, nb_, nb_, nb_, nb_);
        TBLIS_VIEW_4D(t_Govov_bb, Gamma_ovov_bb, nb_, vb_, nb_, vb_);
        
        TBLIS_VIEW_4D(t_Tab, (*t2_ab_dense), na_, nb_, va_, vb_);
        TBLIS_VIEW_4D(t_Govov_ab, Gamma_ovov_ab, na_, va_, nb_, vb_);

        tblis::mult<double>(0.5, t_Tbb, "ijab", t_Tbb, "ijcd", 1.0, t_Gvvvv_bb, "abcd");
        tblis::mult<double>(0.5, t_Tbb, "ijab", t_Tbb, "klab", 1.0, t_Goooo_bb, "ijkl");
        tblis::mult<double>(1.0, t_Tbb, "ikac", t_Tbb, "kjcb", 1.0, t_Govov_bb, "iajb");
        
        tblis::mult<double>(1.0, t_Taa, "ikac", t_Tab, "kjcb", 1.0, t_Govov_ab, "iajb");
        tblis::mult<double>(1.0, t_Tab, "ikac", t_Tbb, "kjcb", 1.0, t_Govov_ab, "iajb");

        Teff_ab = Eigen::MatrixXd::Zero(na_*va_, nb_*vb_);
        Teff_bb = Eigen::MatrixXd::Zero(nb_*vb_, nb_*vb_);
        
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < na_; ++i) {
            for (int a = 0; a < va_; ++a) {
                for (int j = 0; j < nb_; ++j) {
                    for (int b = 0; b < vb_; ++b) {
                        Teff_ab(i*va_+a, j*vb_+b) = (*t2_ab_dense)(i, j, a, b) + L2_ab_(i, j, a, b);
                    }
                }
            }
        }
                
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < nb_; ++i) {
            for (int a = 0; a < vb_; ++a) {
                for (int j = 0; j < nb_; ++j) {
                    for (int b = 0; b < vb_; ++b) {
                        double dir = (*t2_bb_dense)(i, j, a, b) + L2_bb_(i, j, a, b);
                        double ex  = (*t2_bb_dense)(i, j, b, a) + L2_bb_(i, j, b, a);
                        Teff_bb(i*vb_+a, j*vb_+b) = dir - ex;
                    }
                }
            }
        }
    }

    Eigen::MatrixXd X_a = Teff_aa * B_ia_P_alpha_;
    if (!is_restricted && nb_ > 0 && vb_ > 0) X_a += Teff_ab * B_ia_P_beta_;

    Eigen::MatrixXd X_b;
    if (!is_restricted && nb_ > 0 && vb_ > 0) X_b = Teff_bb * B_ia_P_beta_ + Teff_ab.transpose() * B_ia_P_alpha_;

    Eigen::MatrixXd Z_mat_a = Eigen::MatrixXd::Zero(va_, na_);
    Eigen::MatrixXd Z_mat_b;
    if (!is_restricted && nb_ > 0 && vb_ > 0) Z_mat_b = Eigen::MatrixXd::Zero(vb_, nb_);

    TBLIS_VIEW_3D(t_Bvv_a, B_vv_a.data(), va_, va_, n_aux);
    TBLIS_VIEW_3D(t_Xa, X_a.data(), va_, na_, n_aux);
    TBLIS_VIEW_3D(t_Boo_a, B_oo_a.data(), na_, na_, n_aux);
    TBLIS_VIEW_2D(t_Za, Z_mat_a.data(), va_, na_);
    TBLIS_VIEW_3D(t_Bia_a, B_ia_P_alpha_.data(), va_, na_, n_aux); 

    tblis::mult<double>(1.0, t_Bvv_a, "baP", t_Xa, "biP", 1.0, t_Za, "ai");
    tblis::mult<double>(-1.0, t_Xa, "ajP", t_Boo_a, "jiP", 1.0, t_Za, "ai");

    Eigen::MatrixXd X_vv_a = Eigen::MatrixXd::Zero(va_ * va_, n_aux);
    TBLIS_VIEW_3D(t_Xvv_a, X_vv_a.data(), va_, va_, n_aux);
    tblis::mult<double>(1.0, t_Gvvvv_aa, "abcd", t_Bvv_a, "dcP", 0.0, t_Xvv_a, "baP"); 
    tblis::mult<double>(1.0, t_Xvv_a, "baP", t_Bia_a, "biP", 1.0, t_Za, "ai");        
    
    Eigen::MatrixXd X_oo_a = Eigen::MatrixXd::Zero(na_ * na_, n_aux);
    TBLIS_VIEW_3D(t_Xoo_a, X_oo_a.data(), na_, na_, n_aux);
    tblis::mult<double>(1.0, t_Goooo_aa, "ijkl", t_Boo_a, "lkP", 0.0, t_Xoo_a, "jiP"); 
    tblis::mult<double>(-1.0, t_Xoo_a, "jiP", t_Bia_a, "ajP", 1.0, t_Za, "ai");

    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        TBLIS_VIEW_3D(t_Bvv_b, B_vv_b.data(), vb_, vb_, n_aux);
        TBLIS_VIEW_3D(t_Xb, X_b.data(), vb_, nb_, n_aux);
        TBLIS_VIEW_3D(t_Boo_b, B_oo_b.data(), nb_, nb_, n_aux);
        TBLIS_VIEW_2D(t_Zb, Z_mat_b.data(), vb_, nb_);
        TBLIS_VIEW_3D(t_Bia_b, B_ia_P_beta_.data(), vb_, nb_, n_aux);
        TBLIS_VIEW_4D(t_Gvvvv_bb, Gamma_vvvv_bb, vb_, vb_, vb_, vb_);
        TBLIS_VIEW_4D(t_Goooo_bb, Gamma_oooo_bb, nb_, nb_, nb_, nb_);

        tblis::mult<double>(1.0, t_Bvv_b, "baP", t_Xb, "biP", 1.0, t_Zb, "ai");
        tblis::mult<double>(-1.0, t_Xb, "ajP", t_Boo_b, "jiP", 1.0, t_Zb, "ai");
        
        Eigen::MatrixXd X_vv_b = Eigen::MatrixXd::Zero(vb_ * vb_, n_aux);
        TBLIS_VIEW_3D(t_Xvv_b, X_vv_b.data(), vb_, vb_, n_aux);
        tblis::mult<double>(1.0, t_Gvvvv_bb, "abcd", t_Bvv_b, "dcP", 0.0, t_Xvv_b, "baP"); 
        tblis::mult<double>(1.0, t_Xvv_b, "baP", t_Bia_b, "biP", 1.0, t_Zb, "ai");        
        
        Eigen::MatrixXd X_oo_b = Eigen::MatrixXd::Zero(nb_ * nb_, n_aux);
        TBLIS_VIEW_3D(t_Xoo_b, X_oo_b.data(), nb_, nb_, n_aux);
        tblis::mult<double>(1.0, t_Goooo_bb, "ijkl", t_Boo_b, "lkP", 0.0, t_Xoo_b, "jiP"); 
        tblis::mult<double>(-1.0, t_Xoo_b, "jiP", t_Bia_b, "ajP", 1.0, t_Zb, "ai");
    }

    F_gen_a_.block(na_, 0, va_, na_) += Z_mat_a;
    F_gen_a_.block(0, na_, na_, va_) += Z_mat_a.transpose();
    
    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        F_gen_b_.block(nb_, 0, vb_, nb_) += Z_mat_b;
        F_gen_b_.block(0, nb_, nb_, vb_) += Z_mat_b.transpose();
    } else if (is_restricted) {
        F_gen_b_ = F_gen_a_; 
    }
}
void OMP3::build_hessian_diagonal(Eigen::VectorXd& diag_H, double grad_norm) {
    OMP2::build_hessian_diagonal(diag_H, grad_norm);
    int idx = 0;
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    double spin_factor = is_restricted ? 4.0 : 2.0;

    for (int i = 0; i < na_; ++i) {
        for (int a = 0; a < va_; ++a) {
            double delta_density = std::abs(G_vv_alpha_(a, a)) + std::abs(G_oo_alpha_(i, i));
            diag_H(idx) += spin_factor * 1.5 * delta_density; 
            idx++;
        }
    }

    if (!is_restricted && nb_ > 0) {
        for (int i = 0; i < nb_; ++i) {
            for (int a = 0; a < vb_; ++a) {
                double delta_density = std::abs(G_vv_beta_(a, a)) + std::abs(G_oo_beta_(i, i));
                diag_H(idx) += 2.0 * 1.5 * delta_density;
                idx++;
            }
        }
    }
}
MP3Result OMP3::compute_omp3() {
    if(omp_get_thread_num() == 0) {
        std::cout << "\n========================================================\n";
        std::cout << "      Orbital-Optimized MP3 (OMP3 - Professional)\n";
        std::cout << "========================================================\n";
    }    
    MP2Result res2 = OMP2::compute();

    MP3Result res3;
    res3.converged = res2.converged;
    res3.iterations = res2.iterations;
    res3.e_hf = res2.energy_scf;
    res3.e_mp2 = e_ss_ + e_os_; 
    res3.e_mp3 = e_mp3_tot_;
    res3.e_corr_total = res3.e_mp2 + res3.e_mp3;
    res3.e_total = res3.e_hf + res3.e_corr_total;

    return res3;
}

} 
