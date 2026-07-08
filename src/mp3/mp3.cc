/**
 * @file src/mp3/mp3.cc
 * @brief Unified MP3 Implementation Powered by Native TBLIS
 */

#include "mshqc/mp3.h"
#include "mshqc/integrals/eri_transformer.h"
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

// ============================================================================
// BASE MP3
// ============================================================================
BaseMP3::BaseMP3(const SCFResult& scf, const MP2Result& mp2, const MP2Config& config, std::shared_ptr<IntegralEngine> ints)
    : scf_(scf), mp2_(mp2), config_(config), ints_(ints) 
{
    nbf_ = scf_.C_alpha.rows();
    no_a_ = scf_.n_occ_alpha; no_b_ = scf_.n_occ_beta;
    nv_a_ = nbf_ - no_a_;     nv_b_ = nbf_ - no_b_;
    n_aux_ = scf_.L_mat.cols(); // Mendeteksi apakah kita pakai HDF5
    
    // Copy amplitudes dari MP2
    if (mp2_.t2_aa.size() > 0) t2_aa_ = mp2_.t2_aa;
    if (mp2_.t2_bb.size() > 0) t2_bb_ = mp2_.t2_bb;
    if (mp2_.t2_ab.size() > 0) t2_ab_ = mp2_.t2_ab;
}

double BaseMP3::tensor_dot(const Eigen::Tensor<double, 4>& A, const Eigen::Tensor<double, 4>& B) const {
    Eigen::Map<const Eigen::VectorXd> vecA(A.data(), A.size());
    Eigen::Map<const Eigen::VectorXd> vecB(B.data(), B.size());
    return vecA.dot(vecB);
}

// Macro Pembungkus TBLIS agar kode tidak kotor
#define TBLIS_VIEW_4D(name, t, d1, d2, d3, d4) \
    varray_view<double> name({(len_type)d1, (len_type)d2, (len_type)d3, (len_type)d4}, t.data(), \
    {1, (stride_type)d1, (stride_type)(d1*d2), (stride_type)(d1*d2*d3)})

// ============================================================================
// RESTRICTED MP3 (RMP3)
// ============================================================================
MP3Result RMP3::compute() {
    auto t_start = std::chrono::high_resolution_clock::now();
    if(omp_get_thread_num() == 0) std::cout << "\n=== RMP3 (Unified Native TBLIS) ===\n";

    const auto& Co = scf_.C_alpha.leftCols(no_a_);
    const auto& Cv = scf_.C_alpha.rightCols(nv_a_);
    double E_AA = 0.0, E_AB = 0.0;

    TBLIS_VIEW_4D(t_T2, t2_aa_, no_a_, no_a_, nv_a_, nv_a_);
    Eigen::Tensor<double, 4> W(no_a_, no_a_, nv_a_, nv_a_);
    TBLIS_VIEW_4D(t_W, W, no_a_, no_a_, nv_a_, nv_a_);

    // 1. Ladder VVVV
    {
        auto V = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cv, Cv, Cv, Cv, ints_);
        TBLIS_VIEW_4D(t_V, V, nv_a_, nv_a_, nv_a_, nv_a_);
        
        W.setZero(); // AA: T(i,j,e,f) * [V(e,a,f,b) - V(e,b,f,a)]
        tblis::mult<double>(1.0, t_T2, "ijef", t_V, "eafb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T2, "ijef", t_V, "ebfa", 1.0, t_W, "ijab");
        E_AA += 0.125 * tensor_dot(t2_aa_, W);

        W.setZero(); // AB: T(i,j,e,f) * V(e,a,f,b)
        tblis::mult<double>(1.0, t_T2, "ijef", t_V, "eafb", 0.0, t_W, "ijab");
        E_AB += 1.0 * tensor_dot(t2_aa_, W);
    }

    // 2. Ladder OOOO
    {
        auto V = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Co, Co, Co, Co, ints_);
        TBLIS_VIEW_4D(t_V, V, no_a_, no_a_, no_a_, no_a_);
        
        W.setZero(); // AA: T(m,n,a,b) * [V(m,i,n,j) - V(m,j,n,i)]
        tblis::mult<double>(1.0, t_T2, "mnab", t_V, "minj", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T2, "mnab", t_V, "mjni", 1.0, t_W, "ijab");
        E_AA += 0.125 * tensor_dot(t2_aa_, W);

        W.setZero(); // AB: T(m,n,a,b) * V(m,i,n,j)
        tblis::mult<double>(1.0, t_T2, "mnab", t_V, "minj", 0.0, t_W, "ijab");
        E_AB += 1.0 * tensor_dot(t2_aa_, W);
    }

    // 3. Ring Terms (OVOV dan OOVV)
    {
        auto V_ovov = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Co, Cv, Co, Cv, ints_);
        auto V_oovv = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Co, Co, Cv, Cv, ints_);
        TBLIS_VIEW_4D(t_Vovov, V_ovov, no_a_, nv_a_, no_a_, nv_a_);
        TBLIS_VIEW_4D(t_Voovv, V_oovv, no_a_, no_a_, nv_a_, nv_a_);

        W.setZero(); // Ring AA
        tblis::mult<double>(1.0,  t_Vovov, "iakc", t_T2, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Vovov, "iakc", t_T2, "kjbc", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikac", t_T2, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(1.0,  t_Voovv, "ikac", t_T2, "kjbc", 1.0, t_W, "ijab");
        E_AA += 1.0 * tensor_dot(t2_aa_, W);

        W.setZero(); // Ring AB (6 Term Klasik)
        tblis::mult<double>(1.0,  t_Vovov, "iakc", t_T2, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikac", t_T2, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(1.0,  t_T2, "ikac", t_Vovov, "kcjb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T2, "ikac", t_Voovv, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikbc", t_T2, "kjac", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T2, "kibc", t_Voovv, "kjac", 1.0, t_W, "ijab");
        E_AB += 1.0 * tensor_dot(t2_aa_, W);
    }

    MP3Result res;
    res.e_hf = scf_.energy_total;
    res.e_mp2 = mp2_.e_corr_total;
    res.e3_aa = E_AA; res.e3_ab = E_AB;
    res.e_mp3 = 2.0 * E_AA + E_AB;
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

// ============================================================================
// UNRESTRICTED MP3 (UMP3)
// ============================================================================
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

    // 1. Ladder Terms (VVVV)
    {
        auto Vaa = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cav, Cav, Cav, Cav, ints_);
        TBLIS_VIEW_4D(t_Vaa, Vaa, nv_a_, nv_a_, nv_a_, nv_a_);
        Waa.setZero();
        tblis::mult<double>(1.0, t_Taa, "ijef", t_Vaa, "eafb", 1.0, t_Waa, "ijab");
        tblis::mult<double>(-1.0, t_Taa, "ijef", t_Vaa, "ebfa", 1.0, t_Waa, "ijab");
        e3_aa += 0.125 * tensor_dot(t2_aa_, Waa);

        auto Vbb = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbv, Cbv, Cbv, Cbv, ints_);
        TBLIS_VIEW_4D(t_Vbb, Vbb, nv_b_, nv_b_, nv_b_, nv_b_);
        Wbb.setZero();
        tblis::mult<double>(1.0, t_Tbb, "ijef", t_Vbb, "eafb", 1.0, t_Wbb, "ijab");
        tblis::mult<double>(-1.0, t_Tbb, "ijef", t_Vbb, "ebfa", 1.0, t_Wbb, "ijab");
        e3_bb += 0.125 * tensor_dot(t2_bb_, Wbb);

        auto Vab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cav, Cav, Cbv, Cbv, ints_);
        TBLIS_VIEW_4D(t_Vab, Vab, nv_a_, nv_a_, nv_b_, nv_b_);
        Wab.setZero();
        tblis::mult<double>(1.0, t_Tab, "ijef", t_Vab, "eafb", 1.0, t_Wab, "ijab");
        e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
    }

    // 2. Ladder Terms (OOOO)
    {
        auto Vaa = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cao, Cao, ints_);
        TBLIS_VIEW_4D(t_Vaa, Vaa, no_a_, no_a_, no_a_, no_a_);
        Waa.setZero();
        tblis::mult<double>(1.0, t_Taa, "mnab", t_Vaa, "minj", 1.0, t_Waa, "ijab");
        tblis::mult<double>(-1.0, t_Taa, "mnab", t_Vaa, "mjni", 1.0, t_Waa, "ijab");
        e3_aa += 0.125 * tensor_dot(t2_aa_, Waa);

        auto Vbb = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbo, Cbo, Cbo, ints_);
        TBLIS_VIEW_4D(t_Vbb, Vbb, no_b_, no_b_, no_b_, no_b_);
        Wbb.setZero();
        tblis::mult<double>(1.0, t_Tbb, "mnab", t_Vbb, "minj", 1.0, t_Wbb, "ijab");
        tblis::mult<double>(-1.0, t_Tbb, "mnab", t_Vbb, "mjni", 1.0, t_Wbb, "ijab");
        e3_bb += 0.125 * tensor_dot(t2_bb_, Wbb);

        auto Vab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cbo, Cbo, ints_);
        TBLIS_VIEW_4D(t_Vab, Vab, no_a_, no_a_, no_b_, no_b_);
        Wab.setZero();
        tblis::mult<double>(1.0, t_Tab, "mnab", t_Vab, "minj", 1.0, t_Wab, "ijab");
        e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
    }

    // 3. Ring Terms
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
        auto oovv_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cbv, Cbv, ints_);
        auto oovv_ba = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbo, Cav, Cav, ints_);
        TBLIS_VIEW_4D(t_ovov_ab, ovov_ab, no_a_, nv_a_, no_b_, nv_b_);
        TBLIS_VIEW_4D(t_oovv_ab, oovv_ab, no_a_, no_a_, nv_b_, nv_b_);
        TBLIS_VIEW_4D(t_oovv_ba, oovv_ba, no_b_, no_b_, nv_a_, nv_a_);

        // Ring AA
        Waa.setZero();
        tblis::mult<double>(1.0,  t_ovov_aa, "iakc", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
        tblis::mult<double>(-1.0, t_ovov_aa, "iakc", t_Taa, "kjbc", 1.0, t_Waa, "ijab");
        tblis::mult<double>(-1.0, t_oovv_aa, "ikac", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
        tblis::mult<double>(1.0,  t_oovv_aa, "ikac", t_Taa, "kjbc", 1.0, t_Waa, "ijab");
        tblis::mult<double>(1.0,  t_ovov_ab, "iakc", t_Tab, "jkbc", 1.0, t_Waa, "ijab"); // Cross-spin
        e3_aa += 1.0 * tensor_dot(t2_aa_, Waa);

        // Ring BB
        Wbb.setZero();
        tblis::mult<double>(1.0,  t_ovov_bb, "iakc", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
        tblis::mult<double>(-1.0, t_ovov_bb, "iakc", t_Tbb, "kjbc", 1.0, t_Wbb, "ijab");
        tblis::mult<double>(-1.0, t_oovv_bb, "ikac", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
        tblis::mult<double>(1.0,  t_oovv_bb, "ikac", t_Tbb, "kjbc", 1.0, t_Wbb, "ijab");
        tblis::mult<double>(1.0,  t_ovov_ab, "kcia", t_Tab, "kjcb", 1.0, t_Wbb, "ijab"); // Cross-spin
        e3_bb += 1.0 * tensor_dot(t2_bb_, Wbb);

        // Ring AB
        Wab.setZero();
        tblis::mult<double>(1.0,  t_ovov_aa, "iakc", t_Tab, "kjcb", 1.0, t_Wab, "ijab");
        tblis::mult<double>(-1.0, t_oovv_aa, "ikac", t_Tab, "kjcb", 1.0, t_Wab, "ijab");
        tblis::mult<double>(1.0,  t_Tab, "ikac", t_ovov_bb, "kcjb", 1.0, t_Wab, "ijab");
        tblis::mult<double>(-1.0, t_Tab, "ikac", t_oovv_bb, "kjcb", 1.0, t_Wab, "ijab");
        tblis::mult<double>(1.0,  t_Taa, "ikac", t_ovov_ab, "kcjb", 1.0, t_Wab, "ijab");
        tblis::mult<double>(-1.0, t_Taa, "ikac", t_ovov_ab, "kjcb", 1.0, t_Wab, "ijab"); // Fix permutasi
        tblis::mult<double>(1.0,  t_ovov_ab, "iakc", t_Tbb, "kjcb", 1.0, t_Wab, "ijab");
        tblis::mult<double>(-1.0, t_oovv_ab, "ikbc", t_Tab, "kjac", 1.0, t_Wab, "ijab");
        tblis::mult<double>(-1.0, t_Tab, "kibc", t_oovv_ba, "kjac", 1.0, t_Wab, "ijab");
        e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
    }

    MP3Result res;
    res.e_hf = scf_.energy_total;
    res.e_mp2 = mp2_.e_corr_total;
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

// ============================================================================
// OMP3 (Stubs for next step)
// ============================================================================
MP3Result OMP3::compute() {
    std::cout << "OMP3 Iteration Logic goes here...\n";
    return MP3Result();
}
void OMP3::solve_zvector() {}

} // namespace mshqc
