// Copyright 2026 Muhamad Syahrul Hidayat
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file src/mp3/mp3.cc
 * @brief Unified MP3 Implementation Powered by Native TBLISS
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
#include <unsupported/Eigen/CXX11/Tensor>

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
    Eigen::Tensor< double, 4 > W(no_a_, no_a_, nv_a_, nv_a_); W.setZero();
    TBLIS_VIEW_4D(t_W, W, no_a_, no_a_, nv_a_, nv_a_);

    // [PERBAIKAN] Bypass get_mo_tensor untuk mode eksak
    Eigen::Tensor< double, 4 > eri_ao;
    if (!config_.use_df) eri_ao = ints_->compute_eri();

    auto get_V = [&](const Eigen::MatrixXd& C1, const Eigen::MatrixXd& C2, const Eigen::MatrixXd& C3, const Eigen::MatrixXd& C4) {
        if (!config_.use_df) {
            return ERITransformer::transform_custom(eri_ao, C1, C2, C3, C4, nbf_, C1.cols(), C2.cols(), C3.cols(), C4.cols());
        } else {
            return ERITransformer::get_mo_tensor(true, n_aux_, C1, C2, C3, C4, ints_);
        }
    };

    {
        auto V = get_V(Cv, Cv, Cv, Cv);
        TBLIS_VIEW_4D(t_V, V, nv_a_, nv_a_, nv_a_, nv_a_);
        tblis::mult< double >(1.0, t_T, "ijef", t_V, "eafb", 1.0, t_W, "ijab");
    }
    {
        auto V = get_V(Co, Co, Co, Co);
        TBLIS_VIEW_4D(t_V, V, no_a_, no_a_, no_a_, no_a_);
        tblis::mult< double >(1.0, t_T, "mnab", t_V, "minj", 1.0, t_W, "ijab");
    }
    {
        auto V_ovov = get_V(Co, Cv, Co, Cv);
        auto V_oovv = get_V(Co, Co, Cv, Cv);
        TBLIS_VIEW_4D(t_Vovov, V_ovov, no_a_, nv_a_, no_a_, nv_a_);
        TBLIS_VIEW_4D(t_Voovv, V_oovv, no_a_, no_a_, nv_a_, nv_a_);

        tblis::mult< double >(2.0,  t_Vovov, "iakc", t_T, "kjcb", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_Voovv, "ikac", t_T, "kjcb", 1.0, t_W, "ijab");
        tblis::mult< double >(2.0,  t_T, "ikac", t_Vovov, "kcjb", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_T, "ikac", t_Voovv, "kjcb", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_T, "ikca", t_Vovov, "kcjb", 1.0, t_W, "ijab"); 
        tblis::mult< double >(-1.0, t_Vovov, "iakc", t_T, "kjbc", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_Voovv, "ikbc", t_T, "kjac", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_T, "ikcb", t_Voovv, "jkac", 1.0, t_W, "ijab"); 
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
        std::cout << "  Time         : " << std::chrono::duration< double >(t_end - t_start).count() << " s\n";
    }
    return res;
}

MP3Result UMP3::compute() {
    auto t_start = std::chrono::high_resolution_clock::now();
    if(omp_get_thread_num() == 0) std::cout << "\n=== UMP3 (Unified Native TBLIS) ===\n";

    const auto& Cao = scf_.C_alpha.leftCols(no_a_); const auto& Cav = scf_.C_alpha.rightCols(nv_a_);
    const auto& Cbo = scf_.C_beta.leftCols(no_b_);  const auto& Cbv = scf_.C_beta.rightCols(nv_b_);
    double e3_aa = 0.0, e3_bb = 0.0, e3_ab = 0.0;

    if (no_b_ > 0 && nv_b_ > 0) {
        if (t2_bb_.size() == 0) {
            t2_bb_ = Eigen::Tensor< double, 4 >(no_b_, no_b_, nv_b_, nv_b_);
            t2_bb_.setZero();
        }
        if (t2_ab_.size() == 0) {
            t2_ab_ = Eigen::Tensor< double, 4 >(no_a_, no_b_, nv_a_, nv_b_);
            t2_ab_.setZero();
        }
    }
    Eigen::Tensor< double, 4 > dummy_bb, dummy_ab;
    double* ptr_bb = nullptr;
    double* ptr_ab = nullptr;

    if (no_b_ > 0 && nv_b_ > 0) {
        if (t2_bb_.size() == 0) {
            dummy_bb = Eigen::Tensor< double, 4 >(no_b_, no_b_, nv_b_, nv_b_);
            dummy_bb.setZero();
            ptr_bb = dummy_bb.data();
        } else {
            ptr_bb = t2_bb_.data();
        }

        if (t2_ab_.size() == 0) {
            dummy_ab = Eigen::Tensor< double, 4 >(no_a_, no_b_, nv_a_, nv_b_);
            dummy_ab.setZero();
            ptr_ab = dummy_ab.data();
        } else {
            ptr_ab = t2_ab_.data();
        }
    }
    TBLIS_VIEW_4D(t_Taa, t2_aa_, no_a_, no_a_, nv_a_, nv_a_);

    varray_view< double > t_Tbb({(len_type)no_b_, (len_type)no_b_, (len_type)nv_b_, (len_type)nv_b_}, 
        ptr_bb, {1, (stride_type)no_b_, (stride_type)(no_b_*no_b_), (stride_type)(no_b_*no_b_*nv_b_)});
        
    varray_view< double > t_Tab({(len_type)no_a_, (len_type)no_b_, (len_type)nv_a_, (len_type)nv_b_}, 
        ptr_ab, {1, (stride_type)no_a_, (stride_type)(no_a_*no_b_), (stride_type)(no_a_*no_b_*nv_a_)});

    Eigen::Tensor< double, 4 > Waa(no_a_, no_a_, nv_a_, nv_a_); TBLIS_VIEW_4D(t_Waa, Waa, no_a_, no_a_, nv_a_, nv_a_);
    Eigen::Tensor< double, 4 > Wbb(no_b_, no_b_, nv_b_, nv_b_); TBLIS_VIEW_4D(t_Wbb, Wbb, no_b_, no_b_, nv_b_, nv_b_);
    Eigen::Tensor< double, 4 > Wab(no_a_, no_b_, nv_a_, nv_b_); TBLIS_VIEW_4D(t_Wab, Wab, no_a_, no_b_, nv_a_, nv_b_);

    // [PERBAIKAN] Bypass get_mo_tensor untuk mode eksak
    Eigen::Tensor< double, 4 > eri_ao;
    if (!config_.use_df) eri_ao = ints_->compute_eri();

    auto get_V = [&](const Eigen::MatrixXd& C1, const Eigen::MatrixXd& C2, const Eigen::MatrixXd& C3, const Eigen::MatrixXd& C4) {
        if (!config_.use_df) {
            return ERITransformer::transform_custom(eri_ao, C1, C2, C3, C4, nbf_, C1.cols(), C2.cols(), C3.cols(), C4.cols());
        } else {
            return ERITransformer::get_mo_tensor(true, n_aux_, C1, C2, C3, C4, ints_);
        }
    };

    {
        auto Vaa = get_V(Cav, Cav, Cav, Cav);
        TBLIS_VIEW_4D(t_Vaa, Vaa, nv_a_, nv_a_, nv_a_, nv_a_); Waa.setZero();
        tblis::mult< double >(1.0, t_Taa, "ijef", t_Vaa, "eafb", 1.0, t_Waa, "ijab");
        tblis::mult< double >(-1.0, t_Taa, "ijef", t_Vaa, "ebfa", 1.0, t_Waa, "ijab");
        e3_aa += 0.125 * tensor_dot(t2_aa_, Waa);

        auto Vbb = get_V(Cbv, Cbv, Cbv, Cbv);
        TBLIS_VIEW_4D(t_Vbb, Vbb, nv_b_, nv_b_, nv_b_, nv_b_); Wbb.setZero();
        tblis::mult< double >(1.0, t_Tbb, "ijef", t_Vbb, "eafb", 1.0, t_Wbb, "ijab");
        tblis::mult< double >(-1.0, t_Tbb, "ijef", t_Vbb, "ebfa", 1.0, t_Wbb, "ijab");
        e3_bb += 0.125 * tensor_dot(t2_bb_, Wbb);

        auto Vab = get_V(Cav, Cav, Cbv, Cbv);
        TBLIS_VIEW_4D(t_Vab, Vab, nv_a_, nv_a_, nv_b_, nv_b_); Wab.setZero();
        tblis::mult< double >(1.0, t_Tab, "ijef", t_Vab, "eafb", 1.0, t_Wab, "ijab");
        e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
    }
    {
        auto Vaa = get_V(Cao, Cao, Cao, Cao);
        TBLIS_VIEW_4D(t_Vaa, Vaa, no_a_, no_a_, no_a_, no_a_); Waa.setZero();
        tblis::mult< double >(1.0, t_Taa, "mnab", t_Vaa, "minj", 1.0, t_Waa, "ijab");
        tblis::mult< double >(-1.0, t_Taa, "mnab", t_Vaa, "mjni", 1.0, t_Waa, "ijab");
        e3_aa += 0.125 * tensor_dot(t2_aa_, Waa);

        auto Vbb = get_V(Cbo, Cbo, Cbo, Cbo);
        TBLIS_VIEW_4D(t_Vbb, Vbb, no_b_, no_b_, no_b_, no_b_); Wbb.setZero();
        tblis::mult< double >(1.0, t_Tbb, "mnab", t_Vbb, "minj", 1.0, t_Wbb, "ijab");
        tblis::mult< double >(-1.0, t_Tbb, "mnab", t_Vbb, "mjni", 1.0, t_Wbb, "ijab");
        e3_bb += 0.125 * tensor_dot(t2_bb_, Wbb);

        auto Vab = get_V(Cao, Cao, Cbo, Cbo);
        TBLIS_VIEW_4D(t_Vab, Vab, no_a_, no_a_, no_b_, no_b_); Wab.setZero();
        tblis::mult< double >(1.0, t_Tab, "mnab", t_Vab, "minj", 1.0, t_Wab, "ijab");
        e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
    }
    {
        auto ovov_aa = get_V(Cao, Cav, Cao, Cav);
        auto oovv_aa = get_V(Cao, Cao, Cav, Cav);
        TBLIS_VIEW_4D(t_ovov_aa, ovov_aa, no_a_, nv_a_, no_a_, nv_a_);
        TBLIS_VIEW_4D(t_oovv_aa, oovv_aa, no_a_, no_a_, nv_a_, nv_a_);

        auto ovov_bb = get_V(Cbo, Cbv, Cbo, Cbv);
        auto oovv_bb = get_V(Cbo, Cbo, Cbv, Cbv);
        TBLIS_VIEW_4D(t_ovov_bb, ovov_bb, no_b_, nv_b_, no_b_, nv_b_);
        TBLIS_VIEW_4D(t_oovv_bb, oovv_bb, no_b_, no_b_, nv_b_, nv_b_);

        auto ovov_ab = get_V(Cao, Cav, Cbo, Cbv);
        auto oovv_ab_ex = get_V(Cao, Cao, Cbv, Cbv);
        auto oovv_ba_ex = get_V(Cbo, Cbo, Cav, Cav);
        TBLIS_VIEW_4D(t_ovov_ab, ovov_ab, no_a_, nv_a_, no_b_, nv_b_);
        TBLIS_VIEW_4D(t_oovv_ab_ex, oovv_ab_ex, no_a_, no_a_, nv_b_, nv_b_);
        TBLIS_VIEW_4D(t_oovv_ba_ex, oovv_ba_ex, no_b_, no_b_, nv_a_, nv_a_);

        Waa.setZero();
        tblis::mult< double >(1.0, t_ovov_aa, "iakc", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
        tblis::mult< double >(-1.0, t_oovv_aa, "ikac", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
        tblis::mult< double >(1.0, t_ovov_ab, "iakc", t_Tab, "jkbc", 1.0, t_Waa, "ijab");
        e3_aa += 1.0 * tensor_dot(t2_aa_, Waa);
        
        Wbb.setZero();
        tblis::mult< double >(1.0, t_ovov_bb, "iakc", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
        tblis::mult< double >(-1.0, t_oovv_bb, "ikac", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
        tblis::mult< double >(1.0, t_ovov_ab, "kcia", t_Tab, "kjcb", 1.0, t_Wbb, "ijab"); 
        e3_bb += 1.0 * tensor_dot(t2_bb_, Wbb);

        Wab.setZero();
        tblis::mult< double >(1.0,  t_ovov_aa, "iakc", t_Tab, "kjcb", 1.0, t_Wab, "ijab");
        tblis::mult< double >(-1.0, t_oovv_aa, "ikac", t_Tab, "kjcb", 1.0, t_Wab, "ijab");
        tblis::mult< double >(1.0,  t_Tab, "ikac", t_ovov_bb, "kcjb", 1.0, t_Wab, "ijab");
        tblis::mult< double >(-1.0, t_Tab, "ikac", t_oovv_bb, "kjcb", 1.0, t_Wab, "ijab");
        tblis::mult< double >(1.0,  t_Taa, "ikac", t_ovov_ab, "kcjb", 1.0, t_Wab, "ijab");
        tblis::mult< double >(1.0,  t_ovov_ab, "iakc", t_Tbb, "kjcb", 1.0, t_Wab, "ijab");
        tblis::mult< double >(-1.0, t_oovv_ab_ex, "ikbc", t_Tab, "kjac", 1.0, t_Wab, "ijab");
        tblis::mult< double >(-1.0, t_Tab, "ikcb", t_oovv_ba_ex, "jkac", 1.0, t_Wab, "ijab"); 
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
        std::cout << "  Time         : " << std::chrono::duration< double >(t_end - t_start).count() << " s\n";
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

    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    const Eigen::MatrixXd& Cao = scf_.C_alpha.leftCols(na_); 
    const Eigen::MatrixXd& Cav = scf_.C_alpha.rightCols(va_);
    const auto& ea = scf_.orbital_energies_alpha;
    
    auto* t2_aa_dense = t2_aa_.get_block(0,0,0,0);
    if(!t2_aa_dense) throw std::runtime_error("OMP3 missing T2 dense block.");
    Eigen::Tensor< double, 4 > T2_aa_ijab(na_, na_, va_, va_);
    
    #pragma omp parallel for collapse(4) schedule(static)
    for(int i = 0; i < na_; ++i) {
        for(int j = 0; j < na_; ++j) {
            for(int a = 0; a < va_; ++a) {
                for(int b = 0; b < va_; ++b) {
                    T2_aa_ijab(i,j,a,b) = (*t2_aa_dense)(i,a,j,b);
                }
            }
        }
    }

    // =========================================================================
    // IMPLEMENTASI INTEGRAL EKSAK (CACHE AO INTEGRAL)
    // =========================================================================
    if (!config_.use_df) {
        if (is_restricted) {
            t2_3rd_aa_ = Eigen::Tensor< double, 4 >(na_, na_, va_, va_);
            Eigen::Tensor< double, 4 > W(na_, na_, va_, va_); W.setZero();
            
            TBLIS_VIEW_4D(t_T, T2_aa_ijab, na_, na_, va_, va_);
            TBLIS_VIEW_4D(t_W, W, na_, na_, va_, va_);

            // CACHE: Evaluasi AO ERI satu kali saja sepanjang siklus OMP3
            if (eri_ao_cached_.size() == 0) {
                eri_ao_cached_ = integrals_->compute_eri();
            }

            auto V_vvvv = integrals::ERITransformer::transform_custom(eri_ao_cached_, Cav, Cav, Cav, Cav, nbf_, va_, va_, va_, va_);
            TBLIS_VIEW_4D(t_Vvvvv, V_vvvv, va_, va_, va_, va_);
            tblis::mult< double >(1.0, t_T, "ijef", t_Vvvvv, "eafb", 1.0, t_W, "ijab");
            
            auto V_oooo = integrals::ERITransformer::transform_custom(eri_ao_cached_, Cao, Cao, Cao, Cao, nbf_, na_, na_, na_, na_);
            TBLIS_VIEW_4D(t_Voooo, V_oooo, na_, na_, na_, na_);
            tblis::mult< double >(1.0, t_T, "mnab", t_Voooo, "minj", 1.0, t_W, "ijab");
            
            auto V_ovov = integrals::ERITransformer::transform_custom(eri_ao_cached_, Cao, Cav, Cao, Cav, nbf_, na_, va_, na_, va_);
            auto V_oovv = integrals::ERITransformer::transform_custom(eri_ao_cached_, Cao, Cao, Cav, Cav, nbf_, na_, na_, va_, va_);
            TBLIS_VIEW_4D(t_Vovov, V_ovov, na_, va_, na_, va_);
            TBLIS_VIEW_4D(t_Voovv, V_oovv, na_, na_, va_, va_);

            tblis::mult< double >(2.0,  t_Vovov, "iakc", t_T, "kjcb", 1.0, t_W, "ijab");
            tblis::mult< double >(-1.0, t_Voovv, "ikac", t_T, "kjcb", 1.0, t_W, "ijab");
            tblis::mult< double >(2.0,  t_T, "ikac", t_Vovov, "kcjb", 1.0, t_W, "ijab");
            tblis::mult< double >(-1.0, t_T, "ikac", t_Voovv, "kjcb", 1.0, t_W, "ijab");
            tblis::mult< double >(-1.0, t_T, "ikca", t_Vovov, "kcjb", 1.0, t_W, "ijab"); 
            tblis::mult< double >(-1.0, t_Vovov, "iakc", t_T, "kjbc", 1.0, t_W, "ijab");
            tblis::mult< double >(-1.0, t_Voovv, "ikbc", t_T, "kjac", 1.0, t_W, "ijab");
            tblis::mult< double >(-1.0, t_T, "ikcb", t_Voovv, "jkac", 1.0, t_W, "ijab"); 

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
        } else {
            throw std::runtime_error("[OMP3] Unrestricted Exact Integrals belum didukung. Silakan gunakan DF.");
        }
    }

    // =========================================================================
    // IMPLEMENTASI DENSITY FITTING (MEMORI OPTIMAL - INPLACE ALLOCATION)
    // =========================================================================
    int n_aux = scf_.L_mat.cols();
    Eigen::Map< const Eigen::MatrixXd > L_flat(scf_.L_mat.data(), nbf_, nbf_ * n_aux);

    auto build_B_mat = [&](const Eigen::MatrixXd& C_left, const Eigen::MatrixXd& C_right, int dim_L, int dim_R) {
        Eigen::MatrixXd B_mat = Eigen::MatrixXd::Zero(dim_L * dim_R, n_aux);
        Eigen::MatrixXd X_temp = C_left.transpose() * L_flat;
        #pragma omp parallel for schedule(static)
        for (int P = 0; P < n_aux; ++P) {
            Eigen::Map< const Eigen::MatrixXd > X_P(X_temp.data() + P * dim_L * nbf_, dim_L, nbf_);
            Eigen::MatrixXd B_MO = X_P * C_right; 
            for (int i = 0; i < dim_L; ++i) {
                for (int j = 0; j < dim_R; ++j) {
                    B_mat(i * dim_R + j, P) = B_MO(i, j);
                }
            }
        }
        return B_mat;
    };

    auto map_4d_inplace = [](const Eigen::MatrixXd& M, Eigen::Tensor<double, 4>& T, int d1, int d2, int d3, int d4) {
        if (T.size() != d1*d2*d3*d4) T.resize(d1, d2, d3, d4);
        #pragma omp parallel for collapse(4) schedule(static)
        for(int i = 0; i < d1; ++i) {
            for(int j = 0; j < d2; ++j) {
                for(int k = 0; k < d3; ++k) {
                    for(int l = 0; l < d4; ++l) {
                        T(i,j,k,l) = M(i*d2+j, k*d4+l);
                    }
                }
            }
        }
    };

    Eigen::MatrixXd B_ij_a = build_B_mat(Cao, Cao, na_, na_);
    Eigen::MatrixXd B_ab_a = build_B_mat(Cav, Cav, va_, va_);

    if (is_restricted) {
        t2_3rd_aa_ = Eigen::Tensor< double, 4 >(na_, na_, va_, va_);
        Eigen::Tensor< double, 4 > W(na_, na_, va_, va_); W.setZero();
        
        TBLIS_VIEW_4D(t_T, T2_aa_ijab, na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_W, W, na_, na_, va_, va_);
      
        Eigen::Tensor< double, 4 > V_oooo, V_ovov, V_oovv;
        map_4d_inplace((B_ij_a * B_ij_a.transpose()).eval(), V_oooo, na_, na_, na_, na_);
        map_4d_inplace((B_ia_P_alpha_ * B_ia_P_alpha_.transpose()).eval(), V_ovov, na_, va_, na_, va_);
        map_4d_inplace((B_ij_a * B_ab_a.transpose()).eval(), V_oovv, na_, na_, va_, va_);
        
        TBLIS_VIEW_4D(t_Voooo, V_oooo, na_, na_, na_, na_);
        TBLIS_VIEW_4D(t_Vovov, V_ovov, na_, va_, na_, va_);
        TBLIS_VIEW_4D(t_Voovv, V_oovv, na_, na_, va_, va_);

        tblis::mult< double >(1.0, t_T, "mnab", t_Voooo, "minj", 1.0, t_W, "ijab");
        tblis::mult< double >(2.0,  t_Vovov, "iakc", t_T, "kjcb", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_Voovv, "ikac", t_T, "kjcb", 1.0, t_W, "ijab");
        tblis::mult< double >(2.0,  t_T, "ikac", t_Vovov, "kcjb", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_T, "ikac", t_Voovv, "kjcb", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_T, "ikca", t_Vovov, "kcjb", 1.0, t_W, "ijab"); 
        tblis::mult< double >(-1.0, t_Vovov, "iakc", t_T, "kjbc", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_Voovv, "ikbc", t_T, "kjac", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_T, "ikcb", t_Voovv, "jkac", 1.0, t_W, "ijab"); 

        Eigen::Tensor< double, 4 > X_temp(na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_Xtemp, X_temp, na_, na_, va_, va_);
        for (int P = 0; P < n_aux; ++P) {
            TBLIS_VIEW_2D(t_B_P, B_ab_a.col(P).data(), va_, va_);
            X_temp.setZero();
            tblis::mult< double >(1.0, t_T, "ijef", t_B_P, "fb", 0.0, t_Xtemp, "ijeb");
            tblis::mult< double >(1.0, t_Xtemp, "ijeb", t_B_P, "ea", 1.0, t_W, "ijab");
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

    t2_3rd_aa_ = Eigen::Tensor< double, 4 >(na_, na_, va_, va_); t2_3rd_aa_.setZero();
    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        t2_3rd_bb_ = Eigen::Tensor< double, 4 >(nb_, nb_, vb_, vb_); t2_3rd_bb_.setZero();
        t2_3rd_ab_ = Eigen::Tensor< double, 4 >(na_, nb_, va_, vb_); t2_3rd_ab_.setZero();
    }

    double e3_aa = 0.0, e3_bb = 0.0, e3_ab = 0.0;
    TBLIS_VIEW_4D(t_Taa, T2_aa_ijab, na_, na_, va_, va_);
    auto* t2_bb_dense_ptr = t2_bb_.get_block(0,0,0,0);
    Eigen::Tensor< double, 4 > dummy_bb_local;
    if (!t2_bb_dense_ptr && !is_restricted && nb_ > 0 && vb_ > 0) {
        dummy_bb_local = Eigen::Tensor< double, 4 >(nb_, nb_, vb_, vb_);
        dummy_bb_local.setZero();
        t2_bb_dense_ptr = &dummy_bb_local;
    }
    auto* t2_ab_dense = t2_ab_.get_block(0,0,0,0);
    Eigen::Tensor< double, 4 > dummy_ab;
    if (!t2_ab_dense && !is_restricted && nb_ > 0 && vb_ > 0) {
        dummy_ab = Eigen::Tensor< double, 4 >(na_, nb_, va_, vb_);
        dummy_ab.setZero();
        t2_ab_dense = &dummy_ab;
    }

    {
        if (Waa_ladder_.size() == 0) Waa_ladder_.resize(na_, na_, va_, va_);
        if (Waa_ring_.size() == 0) Waa_ring_.resize(na_, na_, va_, va_);
        
        Waa_ladder_.setZero(); Waa_ring_.setZero();
        TBLIS_VIEW_4D(t_Waa_ladder, Waa_ladder_, na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_Waa_ring, Waa_ring_, na_, na_, va_, va_);

        Eigen::Tensor< double, 4 > V_oooo_aa, V_ovov_aa, V_oovv_aa;
        map_4d_inplace((B_ij_a * B_ij_a.transpose()).eval(), V_oooo_aa, na_, na_, na_, na_);
        map_4d_inplace((B_ia_P_alpha_ * B_ia_P_alpha_.transpose()).eval(), V_ovov_aa, na_, va_, na_, va_);
        map_4d_inplace((B_ij_a * B_ab_a.transpose()).eval(), V_oovv_aa, na_, na_, va_, va_);
        
        TBLIS_VIEW_4D(t_Voooo, V_oooo_aa, na_, na_, na_, na_);
        TBLIS_VIEW_4D(t_Vovov, V_ovov_aa, na_, va_, na_, va_);
        TBLIS_VIEW_4D(t_Voovv, V_oovv_aa, na_, na_, va_, va_);

        tblis::mult< double >(0.5, t_Taa, "mnab", t_Voooo, "minj", 1.0, t_Waa_ladder, "ijab");
        tblis::mult< double >(-0.5, t_Taa, "mnab", t_Voooo, "mjni", 1.0, t_Waa_ladder, "ijab");

        tblis::mult< double >(1.0,  t_Vovov, "iakc", t_Taa, "kjcb", 1.0, t_Waa_ring, "ijab");
        tblis::mult< double >(-1.0, t_Voovv, "ikac", t_Taa, "kjcb", 1.0, t_Waa_ring, "ijab");

        if (!is_restricted && nb_ > 0 && vb_ > 0) {
            Eigen::Tensor< double, 4 > V_ovov_ab;
            map_4d_inplace((B_ia_P_alpha_ * B_ia_P_beta_.transpose()).eval(), V_ovov_ab, na_, va_, nb_, vb_);
            TBLIS_VIEW_4D(t_Vovov_ab, V_ovov_ab, na_, va_, nb_, vb_);
            TBLIS_VIEW_4D(t_Tab, (*t2_ab_dense), na_, nb_, va_, vb_);
            tblis::mult< double >(1.0, t_Vovov_ab, "iakc", t_Tab, "jkbc", 1.0, t_Waa_ring, "ijab");
        }

        Eigen::Tensor< double, 4 > X_temp_aa(na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_Xtemp_aa, X_temp_aa, na_, na_, va_, va_);
        for (int P = 0; P < n_aux; ++P) {
            TBLIS_VIEW_2D(t_B_P_a, B_ab_a.col(P).data(), va_, va_);
            X_temp_aa.setZero();
            tblis::mult< double >(1.0, t_Taa, "ijef", t_B_P_a, "fb", 0.0, t_Xtemp_aa, "ijeb");
            tblis::mult< double >(0.5, t_Xtemp_aa, "ijeb", t_B_P_a, "ea", 1.0, t_Waa_ladder, "ijab");
            tblis::mult< double >(-0.5, t_Xtemp_aa, "ijea", t_B_P_a, "eb", 1.0, t_Waa_ladder, "ijab");
        }

        #pragma omp parallel for collapse(4) reduction(+:e3_aa) schedule(static)
        for(int i = 0; i < na_; ++i) {
            for(int j = 0; j < na_; ++j) {
                for(int a = 0; a < va_; ++a) {
                    for(int b = 0; b < va_; ++b) {
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

        auto* t2_bb_dense_ptr = t2_bb_.get_block(0,0,0,0);
        Eigen::Tensor< double, 4 > dummy_bb_local;
        if (!t2_bb_dense_ptr && nb_ > 0 && vb_ > 0) {
            dummy_bb_local = Eigen::Tensor< double, 4 >(nb_, nb_, vb_, vb_);
            dummy_bb_local.setZero();
            t2_bb_dense_ptr = &dummy_bb_local;
        }
     
        TBLIS_VIEW_4D(t_Tbb, (*t2_bb_dense_ptr), nb_, nb_, vb_, vb_);
        TBLIS_VIEW_4D(t_Tab, (*t2_ab_dense), na_, nb_, va_, vb_);

        {
            Eigen::Tensor< double, 4 > Wbb_ladder(nb_, nb_, vb_, vb_); Wbb_ladder.setZero();
            Eigen::Tensor< double, 4 > Wbb_ring(nb_, nb_, vb_, vb_); Wbb_ring.setZero();
            
            TBLIS_VIEW_4D(t_Wbb_ladder, Wbb_ladder, nb_, nb_, vb_, vb_);
            TBLIS_VIEW_4D(t_Wbb_ring, Wbb_ring, nb_, nb_, vb_, vb_);

            Eigen::Tensor< double, 4 > V_oooo_bb, V_ovov_bb, V_oovv_bb;
            map_4d_inplace((B_ij_b * B_ij_b.transpose()).eval(), V_oooo_bb, nb_, nb_, nb_, nb_);
            map_4d_inplace((B_ia_P_beta_ * B_ia_P_beta_.transpose()).eval(), V_ovov_bb, nb_, vb_, nb_, vb_);
            map_4d_inplace((B_ij_b * B_ab_b.transpose()).eval(), V_oovv_bb, nb_, nb_, vb_, vb_);
            
            TBLIS_VIEW_4D(t_Voooo_bb, V_oooo_bb, nb_, nb_, nb_, nb_);
            TBLIS_VIEW_4D(t_Vovov_bb, V_ovov_bb, nb_, vb_, nb_, vb_);
            TBLIS_VIEW_4D(t_Voovv_bb, V_oovv_bb, nb_, nb_, vb_, vb_);

            tblis::mult< double >(0.5, t_Tbb, "mnab", t_Voooo_bb, "minj", 1.0, t_Wbb_ladder, "ijab");
            tblis::mult< double >(-0.5, t_Tbb, "mnab", t_Voooo_bb, "mjni", 1.0, t_Wbb_ladder, "ijab");
     
            tblis::mult< double >(1.0,  t_Vovov_bb, "iakc", t_Tbb, "kjcb", 1.0, t_Wbb_ring, "ijab");
            tblis::mult< double >(-1.0, t_Voovv_bb, "ikac", t_Tbb, "kjcb", 1.0, t_Wbb_ring, "ijab");

            Eigen::Tensor< double, 4 > V_ovov_ab;
            map_4d_inplace((B_ia_P_alpha_ * B_ia_P_beta_.transpose()).eval(), V_ovov_ab, na_, va_, nb_, vb_);
            TBLIS_VIEW_4D(t_Vovov_ab, V_ovov_ab, na_, va_, nb_, vb_);
            tblis::mult< double >(1.0, t_Vovov_ab, "kcia", t_Tab, "kjcb", 1.0, t_Wbb_ring, "ijab"); 

            Eigen::Tensor< double, 4 > X_temp_bb(nb_, nb_, vb_, vb_);
            TBLIS_VIEW_4D(t_Xtemp_bb, X_temp_bb, nb_, nb_, vb_, vb_);
            for (int P = 0; P < n_aux; ++P) {
                TBLIS_VIEW_2D(t_B_P_b, B_ab_b.col(P).data(), vb_, vb_);
                X_temp_bb.setZero();
                tblis::mult< double >(1.0, t_Tbb, "ijef", t_B_P_b, "fb", 0.0, t_Xtemp_bb, "ijeb");
                tblis::mult< double >(0.5, t_Xtemp_bb, "ijeb", t_B_P_b, "ea", 1.0, t_Wbb_ladder, "ijab");
                tblis::mult< double >(-0.5, t_Xtemp_bb, "ijea", t_B_P_b, "eb", 1.0, t_Wbb_ladder, "ijab");
            }

            #pragma omp parallel for collapse(4) reduction(+:e3_bb) schedule(static)
            for(int i = 0; i < nb_; ++i) {
                for(int j = 0; j < nb_; ++j) {
                    for(int a = 0; a < vb_; ++a) {
                        for(int b = 0; b < vb_; ++b) {
                            double r_asym = Wbb_ring(i,j,a,b) - Wbb_ring(j,i,a,b) - Wbb_ring(i,j,b,a) + Wbb_ring(j,i,b,a);
                            double w_tot = Wbb_ladder(i,j,a,b) + r_asym;
                            double D = eb(i) + eb(j) - eb(nb_+a) - eb(nb_+b);
                            t2_3rd_bb_(i,j,a,b) = w_tot / D;
                            e3_bb += 0.25 * (*t2_bb_dense_ptr)(i,j,a,b) * w_tot;
                        }
                    }
                }
            }
        }

        {
            Eigen::Tensor< double, 4 > Wab_ladder(na_, nb_, va_, vb_); Wab_ladder.setZero();
            Eigen::Tensor< double, 4 > Wab_ring(na_, nb_, va_, vb_); Wab_ring.setZero();
            
            TBLIS_VIEW_4D(t_Wab_ladder, Wab_ladder, na_, nb_, va_, vb_);
            TBLIS_VIEW_4D(t_Wab_ring, Wab_ring, na_, nb_, va_, vb_);

            Eigen::Tensor< double, 4 > V_oooo_ab;
            map_4d_inplace((B_ij_a * B_ij_b.transpose()).eval(), V_oooo_ab, na_, na_, nb_, nb_);
            TBLIS_VIEW_4D(t_Voooo_ab, V_oooo_ab, na_, na_, nb_, nb_);
            tblis::mult< double >(1.0, t_Tab, "mnab", t_Voooo_ab, "minj", 1.0, t_Wab_ladder, "ijab");

            Eigen::Tensor< double, 4 > V_ovov_aa, V_oovv_aa;
            map_4d_inplace((B_ia_P_alpha_ * B_ia_P_alpha_.transpose()).eval(), V_ovov_aa, na_, va_, na_, va_);
            map_4d_inplace((B_ij_a * B_ab_a.transpose()).eval(), V_oovv_aa, na_, na_, va_, va_);
            TBLIS_VIEW_4D(t_Vovov_aa, V_ovov_aa, na_, va_, na_, va_);
            TBLIS_VIEW_4D(t_Voovv_aa, V_oovv_aa, na_, na_, va_, va_);

            tblis::mult< double >(1.0,  t_Vovov_aa, "iakc", t_Tab, "kjcb", 1.0, t_Wab_ring, "ijab");
            tblis::mult< double >(-1.0, t_Voovv_aa, "ikac", t_Tab, "kjcb", 1.0, t_Wab_ring, "ijab");

            Eigen::Tensor< double, 4 > V_ovov_bb, V_oovv_bb;
            map_4d_inplace((B_ia_P_beta_ * B_ia_P_beta_.transpose()).eval(), V_ovov_bb, nb_, vb_, nb_, vb_);
            map_4d_inplace((B_ij_b * B_ab_b.transpose()).eval(), V_oovv_bb, nb_, nb_, vb_, vb_);
            TBLIS_VIEW_4D(t_Vovov_bb, V_ovov_bb, nb_, vb_, nb_, vb_);
            TBLIS_VIEW_4D(t_Voovv_bb, V_oovv_bb, nb_, nb_, vb_, vb_);

            tblis::mult< double >(1.0,  t_Tab, "ikac", t_Vovov_bb, "kcjb", 1.0, t_Wab_ring, "ijab"); 
            tblis::mult< double >(-1.0, t_Tab, "ikac", t_Voovv_bb, "kjcb", 1.0, t_Wab_ring, "ijab");

            Eigen::Tensor< double, 4 > V_ovov_ab;
            map_4d_inplace((B_ia_P_alpha_ * B_ia_P_beta_.transpose()).eval(), V_ovov_ab, na_, va_, nb_, vb_);
            TBLIS_VIEW_4D(t_Vovov_ab, V_ovov_ab, na_, va_, nb_, vb_);

            tblis::mult< double >(1.0,  t_Taa, "ikac", t_Vovov_ab, "kcjb", 1.0, t_Wab_ring, "ijab");
            tblis::mult< double >(1.0,  t_Vovov_ab, "iakc", t_Tbb, "kjcb", 1.0, t_Wab_ring, "ijab");

            Eigen::Tensor< double, 4 > V_oovv_ab_ex, V_oovv_ba_ex;
            map_4d_inplace((B_ij_a * B_ab_b.transpose()).eval(), V_oovv_ab_ex, na_, na_, vb_, vb_);
            map_4d_inplace((B_ij_b * B_ab_a.transpose()).eval(), V_oovv_ba_ex, nb_, nb_, va_, va_);
            TBLIS_VIEW_4D(t_Voovv_ab_ex, V_oovv_ab_ex, na_, na_, vb_, vb_);
            TBLIS_VIEW_4D(t_Voovv_ba_ex, V_oovv_ba_ex, nb_, nb_, va_, va_);

            tblis::mult< double >(-1.0, t_Voovv_ab_ex, "ikbc", t_Tab, "kjac", 1.0, t_Wab_ring, "ijab");
            tblis::mult< double >(-1.0, t_Tab, "ikcb", t_Voovv_ba_ex, "jkac", 1.0, t_Wab_ring, "ijab");
          
            Eigen::Tensor< double, 4 > X_temp_ab(na_, nb_, va_, vb_);
            TBLIS_VIEW_4D(t_Xtemp_ab, X_temp_ab, na_, nb_, va_, vb_);
            for (int P = 0; P < n_aux; ++P) {
                TBLIS_VIEW_2D(t_B_P_a, B_ab_a.col(P).data(), va_, va_);
                TBLIS_VIEW_2D(t_B_P_b, B_ab_b.col(P).data(), vb_, vb_);
                
                X_temp_ab.setZero();
                tblis::mult< double >(1.0, t_Tab, "ijef", t_B_P_b, "fb", 0.0, t_Xtemp_ab, "ijeb");
                tblis::mult< double >(1.0, t_Xtemp_ab, "ijeb", t_B_P_a, "ea", 1.0, t_Wab_ladder, "ijab");
            }

            #pragma omp parallel for collapse(4) reduction(+:e3_ab) schedule(static)
            for(int i = 0; i < na_; ++i) {
                for(int j = 0; j < nb_; ++j) {
                    for(int a = 0; a < va_; ++a) {
                        for(int b = 0; b < vb_; ++b) {
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

void OMP3::build_opdm_alpha() {
    if (L2_aa_.size() == 0) { OMP2::build_opdm_alpha(); return; }
    auto* t2_aa_dense = t2_aa_.get_block(0,0,0,0);
    if(!t2_aa_dense) return;

    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    G_oo_alpha_ = Eigen::MatrixXd::Zero(na_, na_);
    G_vv_alpha_ = Eigen::MatrixXd::Zero(va_, va_);
    
    Eigen::Tensor<double, 4> T2_aa_ijab(na_, na_, va_, va_);
    #pragma omp parallel for collapse(4) schedule(static)
    for(int i = 0; i < na_; ++i) {
        for(int j = 0; j < na_; ++j) {
            for(int a = 0; a < va_; ++a) {
                for(int b = 0; b < va_; ++b) {
                    T2_aa_ijab(i,j,a,b) = (*t2_aa_dense)(i,a,j,b);
                }
            }
        }
    }

    if (is_restricted) {
        Eigen::Tensor<double, 4> T2_tilde(na_, na_, va_, va_);
        Eigen::Tensor<double, 4> L2_tilde(na_, na_, va_, va_);
        
        #pragma omp parallel for collapse(4) schedule(static)
        for (int i = 0; i < na_; ++i) {
            for (int j = 0; j < na_; ++j) {
                for (int a = 0; a < va_; ++a) {
                    for (int b = 0; b < va_; ++b) {
                        T2_tilde(i,j,a,b) = 2.0 * T2_aa_ijab(i,j,a,b) - T2_aa_ijab(i,j,b,a);
                        L2_tilde(i,j,a,b) = 2.0 * L2_aa_(i,j,a,b) - L2_aa_(i,j,b,a);
                    }
                }
            }
        }

        TBLIS_VIEW_4D(t_T2t, T2_tilde, na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_L2t, L2_tilde, na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_T2, T2_aa_ijab, na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_L2, L2_aa_, na_, na_, va_, va_);

        TBLIS_VIEW_2D(t_Goo, G_oo_alpha_.data(), na_, na_);
        TBLIS_VIEW_2D(t_Gvv, G_vv_alpha_.data(), va_, va_);

        tblis::mult<double>(-1.0, t_T2, "ikab", t_T2t, "jkab", 0.0, t_Goo, "ij"); 
        tblis::mult<double>(-0.5, t_T2, "ikab", t_L2t, "jkab", 1.0, t_Goo, "ij"); 
        tblis::mult<double>(-0.5, t_L2, "ikab", t_T2t, "jkab", 1.0, t_Goo, "ij");  

        tblis::mult<double>(1.0, t_T2, "ijac", t_T2t, "ijbc", 0.0, t_Gvv, "ab"); 
        tblis::mult<double>(0.5, t_T2, "ijac", t_L2t, "ijbc", 1.0, t_Gvv, "ab"); 
        tblis::mult<double>(0.5, t_L2, "ijac", t_T2t, "ijbc", 1.0, t_Gvv, "ab"); 
        return;
    }

    TBLIS_VIEW_4D(t_T2aa, T2_aa_ijab, na_, na_, va_, va_);
    TBLIS_VIEW_4D(t_T3aa, L2_aa_, na_, na_, va_, va_);
    TBLIS_VIEW_2D(t_Goo_a, G_oo_alpha_.data(), na_, na_);
    TBLIS_VIEW_2D(t_Gvv_a, G_vv_alpha_.data(), va_, va_);
    
    tblis::mult<double>(-0.5,  t_T2aa, "ikab", t_T2aa, "jkab", 1.0, t_Goo_a, "ij"); 
    tblis::mult<double>(-0.25, t_T2aa, "ikab", t_T3aa, "jkab", 1.0, t_Goo_a, "ij"); 
    tblis::mult<double>(-0.25, t_T3aa, "ikab", t_T2aa, "jkab", 1.0, t_Goo_a, "ij"); 

    tblis::mult<double>(0.5,  t_T2aa, "ijac", t_T2aa, "ijbc", 1.0, t_Gvv_a, "ab"); 
    tblis::mult<double>(0.25, t_T2aa, "ijac", t_T3aa, "ijbc", 1.0, t_Gvv_a, "ab");  
    tblis::mult<double>(0.25, t_T3aa, "ijac", t_T2aa, "ijbc", 1.0, t_Gvv_a, "ab");

    auto* t2_ab_dense = t2_ab_.get_block(0,0,0,0);
    if (nb_ > 0 && vb_ > 0 && t2_ab_dense) {
        TBLIS_VIEW_4D(t_T2ab, (*t2_ab_dense), na_, nb_, va_, vb_);
        TBLIS_VIEW_4D(t_T3ab, L2_ab_, na_, nb_, va_, vb_);
        
        tblis::mult<double>(-1.0, t_T2ab, "ikab", t_T2ab, "jkab", 1.0, t_Goo_a, "ij"); 
        tblis::mult<double>(-0.5, t_T2ab, "ikab", t_T3ab, "jkab", 1.0, t_Goo_a, "ij"); 
        tblis::mult<double>(-0.5, t_T3ab, "ikab", t_T2ab, "jkab", 1.0, t_Goo_a, "ij"); 
        
        tblis::mult<double>(1.0, t_T2ab, "ijac", t_T2ab, "ijbc", 1.0, t_Gvv_a, "ab"); 
        tblis::mult<double>(0.5, t_T2ab, "ijac", t_T3ab, "ijbc", 1.0, t_Gvv_a, "ab");  
        tblis::mult<double>(0.5, t_T3ab, "ijac", t_T2ab, "ijbc", 1.0, t_Gvv_a, "ab"); 
    }
}

void OMP3::build_opdm_beta() {
    if (L2_bb_.size() == 0 && L2_ab_.size() == 0) { OMP2::build_opdm_beta(); return; }
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    G_oo_beta_ = Eigen::MatrixXd::Zero(nb_, nb_);
    G_vv_beta_ = Eigen::MatrixXd::Zero(vb_, vb_);
    
    if (is_restricted || nb_ == 0 || vb_ == 0) return; 
    auto* t2_bb_dense = t2_bb_.get_block(0,0,0,0);
    
    Eigen::Tensor<double, 4> dummy_bb;
    if (!t2_bb_dense) {
        dummy_bb = Eigen::Tensor<double, 4>(nb_, nb_, vb_, vb_); 
        dummy_bb.setZero();
        t2_bb_dense = &dummy_bb;
    }
    
    auto* t2_ab_dense = t2_ab_.get_block(0,0,0,0);
    Eigen::Tensor<double, 4> dummy_ab;
    if (!t2_ab_dense) {
        dummy_ab = Eigen::Tensor<double, 4>(na_, nb_, va_, vb_); 
        dummy_ab.setZero();
        t2_ab_dense = &dummy_ab;
    }

    TBLIS_VIEW_4D(t_T2bb, (*t2_bb_dense), nb_, nb_, vb_, vb_);
    TBLIS_VIEW_4D(t_T3bb, L2_bb_, nb_, nb_, vb_, vb_);
    TBLIS_VIEW_4D(t_T2ab, (*t2_ab_dense), na_, nb_, va_, vb_);
    TBLIS_VIEW_4D(t_T3ab, L2_ab_, na_, nb_, va_, vb_);
    TBLIS_VIEW_2D(t_Goo_b, G_oo_beta_.data(), nb_, nb_);
    TBLIS_VIEW_2D(t_Gvv_b, G_vv_beta_.data(), vb_, vb_);

    tblis::mult<double>(-0.5,  t_T2bb, "ikab", t_T2bb, "jkab", 1.0, t_Goo_b, "ij"); 
    tblis::mult<double>(-0.25, t_T2bb, "ikab", t_T3bb, "jkab", 1.0, t_Goo_b, "ij"); 
    tblis::mult<double>(-0.25, t_T3bb, "ikab", t_T2bb, "jkab", 1.0, t_Goo_b, "ij"); 

    tblis::mult<double>(0.5,  t_T2bb, "ijac", t_T2bb, "ijbc", 1.0, t_Gvv_b, "ab"); 
    tblis::mult<double>(0.25, t_T2bb, "ijac", t_T3bb, "ijbc", 1.0, t_Gvv_b, "ab");  
    tblis::mult<double>(0.25, t_T3bb, "ijac", t_T2bb, "ijbc", 1.0, t_Gvv_b, "ab"); 
    
    tblis::mult<double>(-1.0, t_T2ab, "kiab", t_T2ab, "kjab", 1.0, t_Goo_b, "ij"); 
    tblis::mult<double>(-0.5, t_T2ab, "kiab", t_T3ab, "kjab", 1.0, t_Goo_b, "ij");  
    tblis::mult<double>(-0.5, t_T3ab, "kiab", t_T2ab, "kjab", 1.0, t_Goo_b, "ij");  
        
    tblis::mult<double>(1.0, t_T2ab, "ijca", t_T2ab, "ijcb", 1.0, t_Gvv_b, "ab"); 
    tblis::mult<double>(0.5, t_T2ab, "ijca", t_T3ab, "ijcb", 1.0, t_Gvv_b, "ab");   
    tblis::mult<double>(0.5, t_T3ab, "ijca", t_T2ab, "ijcb", 1.0, t_Gvv_b, "ab");
}

void OMP3::build_generalized_fock() {
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    const auto& ea = scf_.orbital_energies_alpha;
    const auto& eb = scf_.orbital_energies_beta;
    double scale = is_restricted ? 0.25 : 0.5;

    const Eigen::MatrixXd& Cao = scf_.C_alpha.leftCols(na_);
    const Eigen::MatrixXd& Cav = scf_.C_alpha.rightCols(va_);
    const Eigen::MatrixXd& Cbo = scf_.C_beta.leftCols(nb_);
    const Eigen::MatrixXd& Cbv = scf_.C_beta.rightCols(vb_);

    Eigen::MatrixXd Z_mat_a = Eigen::MatrixXd::Zero(va_, na_);
    Eigen::MatrixXd dx_oo_a = Eigen::MatrixXd::Zero(na_, na_);
    Eigen::MatrixXd dx_vv_a = Eigen::MatrixXd::Zero(va_, va_);
    Eigen::MatrixXd Delta_G_ia_a = Eigen::MatrixXd::Zero(va_, na_);
    
    Eigen::MatrixXd Z_mat_b, dx_oo_b, dx_vv_b, Delta_G_ia_b;
    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        Z_mat_b = Eigen::MatrixXd::Zero(vb_, nb_);
        dx_oo_b = Eigen::MatrixXd::Zero(nb_, nb_);
        dx_vv_b = Eigen::MatrixXd::Zero(vb_, vb_);
        Delta_G_ia_b = Eigen::MatrixXd::Zero(vb_, nb_);
    }

    auto* t_aa_dense = t2_aa_.get_block(0,0,0,0);
    auto* t2_ab_dense = t2_ab_.get_block(0,0,0,0);
    auto* t2_bb_dense = t2_bb_.get_block(0,0,0,0);

    // =========================================================================
    // TAHAP 1 & 2: Z-VECTOR & CPHF SOLVER (EXACT CACHE vs DF ROUTING)
    // =========================================================================
    if (!config_.use_df) {
        if (!is_restricted) {
            throw std::runtime_error("[OMP3] Unrestricted Exact Z-vector belum didukung. Silakan gunakan DF.");
        }

        Eigen::Tensor< double, 4 > Teff(na_, na_, va_, va_);
        #pragma omp parallel for collapse(4) schedule(static)
        for(int i = 0; i < na_; ++i) {
            for(int j = 0; j < na_; ++j) {
                for(int a = 0; a < va_; ++a) {
                    for(int b = 0; b < va_; ++b) {
                        Teff(i,j,a,b) = (*t_aa_dense)(i,a,j,b) + L2_aa_(i,j,a,b);
                    }
                }
            }
        }
        TBLIS_VIEW_4D(t_Teff, Teff, na_, na_, va_, va_);

        // CACHE: Gunakan ERI yang sudah dihitung pada tahap compute_mp3_correction
        if (eri_ao_cached_.size() == 0) {
            eri_ao_cached_ = integrals_->compute_eri();
        }

        Eigen::MatrixXd Z_oo_mat = Eigen::MatrixXd::Zero(na_, na_);
        Eigen::MatrixXd Z_vv_mat = Eigen::MatrixXd::Zero(va_, va_);
        TBLIS_VIEW_2D(t_Zoo, Z_oo_mat.data(), na_, na_);
        TBLIS_VIEW_2D(t_Zvv, Z_vv_mat.data(), va_, va_);
        TBLIS_VIEW_2D(t_Zmat, Z_mat_a.data(), va_, na_);

        auto V_ovov = ERITransformer::transform_custom(eri_ao_cached_, Cao, Cav, Cao, Cav, nbf_, na_, va_, na_, va_);
        TBLIS_VIEW_4D(t_Vovov, V_ovov, na_, va_, na_, va_);
        tblis::mult< double >(1.0, t_Teff, "ikab", t_Vovov, "kbja", 0.0, t_Zoo, "ij");
        tblis::mult< double >(-1.0, t_Teff, "ikac", t_Vovov, "kcib", 0.0, t_Zvv, "ab");

        auto V_vvov = ERITransformer::transform_custom(eri_ao_cached_, Cav, Cav, Cao, Cav, nbf_, va_, va_, na_, va_);
        TBLIS_VIEW_4D(t_Vvvov, V_vvov, va_, va_, na_, va_);
        tblis::mult< double >(1.0, t_Vvvov, "abkc", t_Teff, "ikbc", 0.0, t_Zmat, "ai");

        auto V_ooov = ERITransformer::transform_custom(eri_ao_cached_, Cao, Cao, Cao, Cav, nbf_, na_, na_, na_, va_);
        TBLIS_VIEW_4D(t_Vooov, V_ooov, na_, na_, na_, va_);
        tblis::mult< double >(-1.0, t_Vooov, "jikc", t_Teff, "jkac", 1.0, t_Zmat, "ai");

        auto solve_mini_cphf_exact = [](const Eigen::MatrixXd& Z_in, const Eigen::VectorXd& eps,
                                        const Eigen::Tensor< double, 4 >& V_exact, int dim, int offset) -> Eigen::MatrixXd {
            if (dim == 0) return Eigen::MatrixXd::Zero(0, 0);
            int dim2 = dim * dim;

            Eigen::VectorXd Z_vec(dim2), eps_diff(dim2);
            for (int i = 0; i < dim; ++i) {
                for (int j = 0; j < dim; ++j) {
                    Z_vec(i * dim + j) = Z_in(i, j) - Z_in(j, i);
                    eps_diff(i * dim + j) = eps(offset + i) - eps(offset + j);
                }
            }

            Eigen::VectorXd x = Eigen::VectorXd::Zero(dim2);
            for (int k = 0; k < dim2; ++k) if (std::abs(eps_diff(k)) > 1e-5) x(k) = Z_vec(k) / eps_diff(k);

            auto apply_V = [&](const Eigen::VectorXd& vec) -> Eigen::VectorXd {
            
                Eigen::Map<Eigen::MatrixXd> M(const_cast<double*>(vec.data()), dim, dim);
                Eigen::VectorXd res(dim2);
                Eigen::Map<Eigen::MatrixXd> R(res.data(), dim, dim);
                R.setZero();
                Eigen::TensorMap<Eigen::Tensor<double, 4>> V_map(const_cast<double*>(V_exact.data()), dim, dim, dim, dim);

                TBLIS_VIEW_4D(t_V, V_map, dim, dim, dim, dim);
                TBLIS_VIEW_2D(t_M, M.data(), dim, dim);
                TBLIS_VIEW_2D(t_R, R.data(), dim, dim);
                
                tblis::mult< double >(1.0, t_V, "ijkl", t_M, "kl", 0.0, t_R, "ij");
                return res;
            };

            Eigen::VectorXd Ap_0 = eps_diff.cwiseProduct(x) + apply_V(x);
            Eigen::VectorXd r = Z_vec - Ap_0;
            Eigen::VectorXd z = Eigen::VectorXd::Zero(dim2);
            for (int k = 0; k < dim2; ++k) if (std::abs(eps_diff(k)) > 1e-5) z(k) = r(k) / eps_diff(k);
            
            Eigen::VectorXd p = z;
            double rz_old = r.dot(z);

            for (int iter = 0; iter < 20; ++iter) {
                if (r.norm() < 1e-8) break;
                Eigen::VectorXd Ap = eps_diff.cwiseProduct(p) + apply_V(p);
                double pAp = p.dot(Ap);
                if (std::abs(pAp) < 1e-14) break;
                
                double alpha = rz_old / pAp;
                x += alpha * p;
                r -= alpha * Ap;
                
                for (int k = 0; k < dim2; ++k) z(k) = (std::abs(eps_diff(k)) > 1e-5) ? r(k) / eps_diff(k) : 0.0;
                double rz_new = r.dot(z);
                p = z + (rz_new / rz_old) * p;
                rz_old = rz_new;
            }
            return Eigen::Map< Eigen::MatrixXd >(x.data(), dim, dim);
        };

        auto V_oooo = ERITransformer::transform_custom(eri_ao_cached_, Cao, Cao, Cao, Cao, nbf_, na_, na_, na_, na_);
        auto V_vvvv = ERITransformer::transform_custom(eri_ao_cached_, Cav, Cav, Cav, Cav, nbf_, va_, va_, va_, va_);
        
        dx_oo_a = solve_mini_cphf_exact(Z_oo_mat, ea, V_oooo, na_, 0);
        dx_vv_a = solve_mini_cphf_exact(Z_vv_mat, ea, V_vvvv, va_, na_);

        TBLIS_VIEW_2D(t_dG, Delta_G_ia_a.data(), va_, na_);
        auto V_ovoo = ERITransformer::transform_custom(eri_ao_cached_, Cao, Cav, Cao, Cao, nbf_, na_, va_, na_, na_);
        auto V_ovvv = ERITransformer::transform_custom(eri_ao_cached_, Cao, Cav, Cav, Cav, nbf_, na_, va_, va_, va_);
        
        TBLIS_VIEW_4D(t_Vovoo, V_ovoo, na_, va_, na_, na_);
        TBLIS_VIEW_4D(t_Vovvv, V_ovvv, na_, va_, va_, va_);
        TBLIS_VIEW_2D(t_xoo, dx_oo_a.data(), na_, na_);
        TBLIS_VIEW_2D(t_xvv, dx_vv_a.data(), va_, va_);

        tblis::mult< double >(scale, t_Vovoo, "iajk", t_xoo, "jk", 0.0, t_dG, "ai");
        tblis::mult< double >(scale, t_Vovvv, "iabc", t_xvv, "bc", 1.0, t_dG, "ai");

    } else {
        int n_aux = scf_.L_mat.cols();
        Eigen::MatrixXd B_oo_flat_a = Eigen::MatrixXd::Zero(na_ * na_, n_aux);
        Eigen::MatrixXd B_vv_flat_a = Eigen::MatrixXd::Zero(va_ * va_, n_aux);
        Eigen::MatrixXd B_oo_flat_b = Eigen::MatrixXd::Zero(nb_ * nb_, n_aux);
        Eigen::MatrixXd B_vv_flat_b = Eigen::MatrixXd::Zero(vb_ * vb_, n_aux);

        #pragma omp parallel
        {
            Eigen::MatrixXd priv_oo_a = Eigen::MatrixXd::Zero(na_ * na_, n_aux);
            Eigen::MatrixXd priv_vv_a = Eigen::MatrixXd::Zero(va_ * va_, n_aux);
            Eigen::MatrixXd priv_oo_b = Eigen::MatrixXd::Zero(nb_ * nb_, n_aux);
            Eigen::MatrixXd priv_vv_b = Eigen::MatrixXd::Zero(vb_ * vb_, n_aux);

            #pragma omp for schedule(dynamic)
            for (int P = 0; P < n_aux; ++P) {
                Eigen::Map< const Eigen::MatrixXd > B_AO(scf_.L_mat.col(P).data(), nbf_, nbf_);
                Eigen::MatrixXd MO_oo_a = Cao.transpose() * (B_AO * Cao);
                Eigen::MatrixXd MO_vv_a = Cav.transpose() * (B_AO * Cav);
                priv_oo_a.col(P) = Eigen::Map< Eigen::VectorXd >(MO_oo_a.data(), na_ * na_);
                priv_vv_a.col(P) = Eigen::Map< Eigen::VectorXd >(MO_vv_a.data(), va_ * va_);

                if (!is_restricted && nb_ > 0 && vb_ > 0) {
                    Eigen::MatrixXd MO_oo_b = Cbo.transpose() * (B_AO * Cbo);
                    Eigen::MatrixXd MO_vv_b = Cbv.transpose() * (B_AO * Cbv);
                    priv_oo_b.col(P) = Eigen::Map< Eigen::VectorXd >(MO_oo_b.data(), nb_ * nb_);
                    priv_vv_b.col(P) = Eigen::Map< Eigen::VectorXd >(MO_vv_b.data(), vb_ * vb_);
                }
            }
            #pragma omp critical
            {
                B_oo_flat_a += priv_oo_a; B_vv_flat_a += priv_vv_a;
                if (!is_restricted && nb_ > 0 && vb_ > 0) { B_oo_flat_b += priv_oo_b; B_vv_flat_b += priv_vv_b; }
            }
        }

        Eigen::MatrixXd Teff_aa = Eigen::MatrixXd::Zero(na_ * va_, na_ * va_);
        Eigen::MatrixXd Teff_ab = Eigen::MatrixXd::Zero(na_ * va_, nb_ * vb_);
        Eigen::MatrixXd Teff_bb = Eigen::MatrixXd::Zero(nb_ * vb_, nb_ * vb_);

        if (t_aa_dense) {
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < na_; ++i) {
                for (int a = 0; a < va_; ++a) {
                    for (int j = 0; j < na_; ++j) {
                        for (int b = 0; b < va_; ++b) {
                            Teff_aa(i * va_ + a, j * va_ + b) = (*t_aa_dense)(i, a, j, b) + L2_aa_(i, j, a, b);
                        }
                    }
                }
            }
        }
        
        if (t2_ab_dense && !is_restricted) {
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < na_; ++i) {
                for (int a = 0; a < va_; ++a) {
                    for (int j = 0; j < nb_; ++j) {
                        for (int b = 0; b < vb_; ++b) {
                            Teff_ab(i * va_ + a, j * vb_ + b) = (*t2_ab_dense)(i, j, a, b) + L2_ab_(i, j, a, b);
                        }
                    }
                }
            }
        }
        
        if (t2_bb_dense && !is_restricted) {
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < nb_; ++i) {
                for (int a = 0; a < vb_; ++a) {
                    for (int j = 0; j < nb_; ++j) {
                        for (int b = 0; b < vb_; ++b) {
                            Teff_bb(i * vb_ + a, j * vb_ + b) = (*t2_bb_dense)(i, j, a, b) + L2_bb_(i, j, a, b);
                        }
                    }
                }
            }
        }

        Eigen::MatrixXd X_a = Teff_aa * B_ia_P_alpha_;
        if (!is_restricted) X_a.noalias() += Teff_ab * B_ia_P_beta_;

        Eigen::MatrixXd Z_oo_a = Eigen::MatrixXd::Zero(na_, na_);
        Eigen::MatrixXd Z_vv_a = Eigen::MatrixXd::Zero(va_, va_);
        Eigen::MatrixXd X_b, Z_oo_b, Z_vv_b;
        if (!is_restricted && nb_ > 0 && vb_ > 0) {
            X_b = Teff_bb * B_ia_P_beta_ + Teff_ab.transpose() * B_ia_P_alpha_;
            Z_oo_b = Eigen::MatrixXd::Zero(nb_, nb_);
            Z_vv_b = Eigen::MatrixXd::Zero(vb_, vb_);
        }

        #pragma omp parallel
        {
            Eigen::MatrixXd Z_loc_a = Eigen::MatrixXd::Zero(va_, na_);
            Eigen::MatrixXd Z_oo_loc_a = Eigen::MatrixXd::Zero(na_, na_);
            Eigen::MatrixXd Z_vv_loc_a = Eigen::MatrixXd::Zero(va_, va_);
            Eigen::MatrixXd Z_loc_b, Z_oo_loc_b, Z_vv_loc_b;
            if (!is_restricted && nb_ > 0 && vb_ > 0) {
                Z_loc_b = Eigen::MatrixXd::Zero(vb_, nb_);
                Z_oo_loc_b = Eigen::MatrixXd::Zero(nb_, nb_);
                Z_vv_loc_b = Eigen::MatrixXd::Zero(vb_, vb_);
            }

            #pragma omp for schedule(dynamic)
            for (int P = 0; P < n_aux; ++P) {
                Eigen::Map< const Eigen::MatrixXd > X_ai(X_a.col(P).data(), va_, na_);
                Eigen::Map< const Eigen::MatrixXd > B_ai(B_ia_P_alpha_.col(P).data(), va_, na_);
                Eigen::Map< const Eigen::MatrixXd > V_a(B_vv_flat_a.col(P).data(), va_, va_);
                Eigen::Map< const Eigen::MatrixXd > O_a(B_oo_flat_a.col(P).data(), na_, na_);
                
                Z_loc_a.noalias() += V_a * X_ai - X_ai * O_a;
                Z_oo_loc_a.noalias() += X_ai.transpose() * B_ai;
                Z_vv_loc_a.noalias() -= X_ai * B_ai.transpose();

                if (!is_restricted && nb_ > 0 && vb_ > 0) {
                    Eigen::Map< const Eigen::MatrixXd > X_bi(X_b.col(P).data(), vb_, nb_);
                    Eigen::Map< const Eigen::MatrixXd > B_bi(B_ia_P_beta_.col(P).data(), vb_, nb_);
                    Eigen::Map< const Eigen::MatrixXd > V_b(B_vv_flat_b.col(P).data(), vb_, vb_);
                    Eigen::Map< const Eigen::MatrixXd > O_b(B_oo_flat_b.col(P).data(), nb_, nb_);
                    
                    Z_loc_b.noalias() += V_b * X_bi - X_bi * O_b;
                    Z_oo_loc_b.noalias() += X_bi.transpose() * B_bi;
                    Z_vv_loc_b.noalias() -= X_bi * B_bi.transpose();
                }
            }
            #pragma omp critical
            { 
                Z_mat_a += Z_loc_a; Z_oo_a += Z_oo_loc_a; Z_vv_a += Z_vv_loc_a; 
                if (!is_restricted && nb_ > 0 && vb_ > 0) { Z_mat_b += Z_loc_b; Z_oo_b += Z_oo_loc_b; Z_vv_b += Z_vv_loc_b; }
            }
        }

        auto solve_mini_cphf = [](const Eigen::MatrixXd& Z_in, const Eigen::VectorXd& eps,
                                  const Eigen::MatrixXd& B_flat, int dim, int offset) -> Eigen::MatrixXd {
            if (dim == 0) return Eigen::MatrixXd::Zero(0, 0);
            int dim2 = dim * dim;

            Eigen::VectorXd Z_vec(dim2), eps_diff(dim2);
            for (int i = 0; i < dim; ++i) {
                for (int j = 0; j < dim; ++j) {
                    Z_vec(i * dim + j) = Z_in(i, j) - Z_in(j, i);
                    eps_diff(i * dim + j) = eps(offset + i) - eps(offset + j);
                }
            }

            Eigen::VectorXd x = Eigen::VectorXd::Zero(dim2);
            for (int k = 0; k < dim2; ++k) if (std::abs(eps_diff(k)) > 1e-5) x(k) = Z_vec(k) / eps_diff(k);
            Eigen::VectorXd r = Z_vec - (eps_diff.cwiseProduct(x) + B_flat * (B_flat.transpose() * x));
            Eigen::VectorXd z = Eigen::VectorXd::Zero(dim2);
            for (int k = 0; k < dim2; ++k) if (std::abs(eps_diff(k)) > 1e-5) z(k) = r(k) / eps_diff(k);
            
            Eigen::VectorXd p = z;
            double rz_old = r.dot(z);

            for (int iter = 0; iter < 20; ++iter) {
                if (r.norm() < 1e-8) break;
                Eigen::VectorXd Ap = eps_diff.cwiseProduct(p) + B_flat * (B_flat.transpose() * p);
                double pAp = p.dot(Ap);
                if (std::abs(pAp) < 1e-14) break;
                
                double alpha = rz_old / pAp;
                x += alpha * p;
                r -= alpha * Ap;
                
                for (int k = 0; k < dim2; ++k) z(k) = (std::abs(eps_diff(k)) > 1e-5) ? r(k) / eps_diff(k) : 0.0;
                double rz_new = r.dot(z);
                p = z + (rz_new / rz_old) * p;
                rz_old = rz_new;
            }
            return Eigen::Map< Eigen::MatrixXd >(x.data(), dim, dim);
        };

        dx_oo_a = solve_mini_cphf(Z_oo_a, ea, B_oo_flat_a, na_, 0);
        dx_vv_a = solve_mini_cphf(Z_vv_a, ea, B_vv_flat_a, va_, na_);
        
        Eigen::Map< Eigen::VectorXd > vec_x_oo_a(dx_oo_a.data(), na_ * na_);
        Eigen::Map< Eigen::VectorXd > vec_x_vv_a(dx_vv_a.data(), va_ * va_);
        Eigen::VectorXd V_aux_a = B_oo_flat_a.transpose() * vec_x_oo_a + B_vv_flat_a.transpose() * vec_x_vv_a;
        Eigen::VectorXd delta_g_flat_a = scale * B_ia_P_alpha_ * V_aux_a; 
        Delta_G_ia_a = Eigen::Map< Eigen::MatrixXd >(delta_g_flat_a.data(), va_, na_);

        if (!is_restricted && nb_ > 0 && vb_ > 0) {
            dx_oo_b = solve_mini_cphf(Z_oo_b, eb, B_oo_flat_b, nb_, 0);
            dx_vv_b = solve_mini_cphf(Z_vv_b, eb, B_vv_flat_b, vb_, nb_);
            
            Eigen::Map< Eigen::VectorXd > vec_x_oo_b(dx_oo_b.data(), nb_ * nb_);
            Eigen::Map< Eigen::VectorXd > vec_x_vv_b(dx_vv_b.data(), vb_ * vb_);
            Eigen::VectorXd V_aux_b = B_oo_flat_b.transpose() * vec_x_oo_b + B_vv_flat_b.transpose() * vec_x_vv_b;
            Eigen::VectorXd delta_g_flat_b = scale * B_ia_P_beta_ * V_aux_b;
            Delta_G_ia_b = Eigen::Map< Eigen::MatrixXd >(delta_g_flat_b.data(), vb_, nb_);
        }
    }

    // =========================================================================
    // TAHAP 3: PERAKITAN 1-RDM MURNI & MATRIKS FOCK GENERALIZED (F_gen)
    // =========================================================================
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
        Eigen::MatrixXd L_sep_a = F_HF_vo_a * G_oo_alpha_ - G_vv_alpha_ * F_HF_vo_a;
        
        F_gen_a_.block(na_, 0, va_, na_) += L_sep_a + Z_mat_a + Delta_G_ia_a;
        F_gen_a_.block(0, na_, na_, va_) += (L_sep_a + Z_mat_a + Delta_G_ia_a).transpose();
    }

    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        F_gen_b_ = F_HF_mo_b + G_gamma_mo_b;
        Eigen::MatrixXd F_HF_vo_b = F_HF_mo_b.block(nb_, 0, vb_, nb_);
        Eigen::MatrixXd L_sep_b = F_HF_vo_b * G_oo_beta_ - G_vv_beta_ * F_HF_vo_b;

        F_gen_b_.block(nb_, 0, vb_, nb_) += L_sep_b + Z_mat_b + Delta_G_ia_b;
        F_gen_b_.block(0, nb_, nb_, vb_) += (L_sep_b + Z_mat_b + Delta_G_ia_b).transpose();
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

void OMP3::debug_gradient_fd(int i_target, int a_target) {
    std::cout << "\n--- [DEBUG] Membedah Komponen Gradien OMP3 ---\n";
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    
    // 1. Eksekusi state awal (tanpa rotasi)
    pseudocanonicalize(); 
    C_a_current_ = scf_.C_alpha;
    C_b_current_ = scf_.C_beta;
    transform_integrals();
    compute_t2_amplitudes();
    compute_mp2_energy();
    compute_mp3_correction();
    
    L2_aa_ = t2_3rd_aa_; 
    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        L2_bb_ = t2_3rd_bb_;
        L2_ab_ = t2_3rd_ab_;
    }
    
    build_opdm_alpha();
    if (!is_restricted && nb_ > 0) build_opdm_beta();
    else if (is_restricted) G_oo_beta_ = G_oo_alpha_;
    build_generalized_fock();

    // Otomatis cari target gradien terbesar
    double max_grad = -1.0;
    for(int i = 0; i < na_; ++i) {
        for(int a = 0; a < va_; ++a) {
            double val = std::abs(F_gen_a_(na_ + a, i));
            if (val > max_grad) {
                max_grad = val;
                i_target = i;
                a_target = a;
            }
        }
    }
    
    // Gradien Analitik
    double grad_ana_tot = is_restricted ? -4.0 * F_gen_a_(na_ + a_target, i_target) : -2.0 * F_gen_a_(na_ + a_target, i_target);
    
    Eigen::MatrixXd F_ao_a, F_ao_b;
    build_fock_fast(scf_.P_alpha, scf_.P_beta, F_ao_a, F_ao_b);
    Eigen::MatrixXd F_mo_a = scf_.C_alpha.transpose() * F_ao_a * scf_.C_alpha;
    double grad_ana_hf = is_restricted ? -4.0 * F_mo_a(na_ + a_target, i_target) : -2.0 * F_mo_a(na_ + a_target, i_target);
    double grad_ana_corr = grad_ana_tot - grad_ana_hf;
    
    double theta = 1e-5;
    Eigen::MatrixXd C_a_orig = scf_.C_alpha;
    Eigen::MatrixXd C_b_orig = scf_.C_beta;
    Eigen::MatrixXd P_a_orig = scf_.P_alpha;
    Eigen::MatrixXd P_b_orig = scf_.P_beta;
    
    // 2. Lambda Fungsi FD
    auto calc_energy_components = [&](double t, double& e_hf, double& e_mp2, double& e_mp3) {
        Eigen::MatrixXd U = Eigen::MatrixXd::Identity(nbf_, nbf_);
        U(i_target, na_ + a_target) = t;
        U(na_ + a_target, i_target) = -t;
        
        C_a_current_ = C_a_orig * U;
        if (!is_restricted) C_b_current_ = C_b_orig * U; 
        else C_b_current_ = C_a_current_;
        
        scf_.C_alpha = C_a_current_;
        scf_.C_beta = C_b_current_;
        scf_.P_alpha = scf_.C_alpha.leftCols(na_) * scf_.C_alpha.leftCols(na_).transpose();
        if (!is_restricted) scf_.P_beta = scf_.C_beta.leftCols(nb_) * scf_.C_beta.leftCols(nb_).transpose();
        else scf_.P_beta = scf_.P_alpha;
        
        pseudocanonicalize();
        C_a_current_ = scf_.C_alpha;
        C_b_current_ = scf_.C_beta;

        Eigen::MatrixXd F_ao_a_tmp, F_ao_b_tmp;
        build_fock_fast(scf_.P_alpha, scf_.P_beta, F_ao_a_tmp, F_ao_b_tmp);
        e_hf = 0.5 * (scf_.P_alpha.cwiseProduct(H_core_ + F_ao_a_tmp).sum() + 
                      scf_.P_beta.cwiseProduct(H_core_ + F_ao_b_tmp).sum()) 
               + mol_.nuclear_repulsion_energy();
                       
        transform_integrals(); 
        compute_t2_amplitudes();
        compute_mp2_energy();
        compute_mp3_correction();
        
        e_mp2 = e_ss_ + e_os_;
        e_mp3 = e_mp3_tot_;
    };

    // 3. Kalkulasi Numerik (+ dan -)
    double hf_plus, mp2_plus, mp3_plus;
    calc_energy_components(theta, hf_plus, mp2_plus, mp3_plus);
    
    double hf_minus, mp2_minus, mp3_minus;
    calc_energy_components(-theta, hf_minus, mp2_minus, mp3_minus);
    
    double g_num_hf  = (hf_plus - hf_minus) / (2.0 * theta);
    double g_num_mp2 = (mp2_plus - mp2_minus) / (2.0 * theta);
    double g_num_mp3 = (mp3_plus - mp3_minus) / (2.0 * theta);
    double g_num_corr = g_num_mp2 + g_num_mp3;
    double g_num_tot = g_num_hf + g_num_corr;

    // 4. Cetak Hasil Dekomposisi
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Target Rotasi        : (i=" << i_target << " [Occ], a=" << a_target << " [Vir])\n";
    std::cout << "--- Finite Difference (NUMERIK) ---\n";
    std::cout << "Gradien HF Numerik   : " << std::scientific << g_num_hf << "\n";
    std::cout << "Gradien MP2 Numerik  : " << std::scientific << g_num_mp2 << "\n";
    std::cout << "Gradien MP3 Numerik  : " << std::scientific << g_num_mp3 << "\n";
    std::cout << "Korelasi Numerik     : " << std::scientific << g_num_corr << "\n";
    std::cout << "TOTAL Numerik        : " << std::scientific << g_num_tot << "\n";
    std::cout << "--- Rumus Analitik (KODE) ---\n";
    std::cout << "Gradien HF Analitik  : " << std::scientific << grad_ana_hf << "\n";
    std::cout << "Korelasi Analitik    : " << std::scientific << grad_ana_corr << "\n";
    std::cout << "TOTAL Analitik       : " << std::scientific << grad_ana_tot << "\n";
    std::cout << "--- EVALUASI SELISIH ---\n";
    std::cout << "Selisih HF           : " << std::abs(g_num_hf - grad_ana_hf) << "\n";
    std::cout << "Selisih Korelasi     : " << std::abs(g_num_corr - grad_ana_corr) << "\n";
    std::cout << "Selisih TOTAL        : " << std::abs(g_num_tot - grad_ana_tot) << "\n";
    std::cout << "-------------------------------------------------------------------\n";
}

MP3Result OMP3::compute_omp3() {
    if(omp_get_thread_num() == 0) {
        std::cout << "\n========================================================\n";
        std::cout << "      Orbital-Optimized MP3 (OMP3 - Professional)\n";
        std::cout << "========================================================\n";
    }
    
    if(omp_get_thread_num() == 0) {
        if (na_ > 0 && va_ > 0) {
            debug_gradient_fd(na_ - 1, 0); 
        }
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

} // namespace mshqc