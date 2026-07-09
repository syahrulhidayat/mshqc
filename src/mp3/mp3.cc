/**
 * @file src/mp3/mp3.cc
 * @brief Unified MP3 Implementation Powered by Native TBLIS
 */

#include "mshqc/mp3.h"
#include "mshqc/integrals/eri_transformer.h"
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




MP3Result RMP3::compute() {
    auto t_start = std::chrono::high_resolution_clock::now();
    if(omp_get_thread_num() == 0) std::cout << "\n=== RMP3 (Unified Native TBLIS) ===\n";

    const auto& Co = scf_.C_alpha.leftCols(no_a_);
    const auto& Cv = scf_.C_alpha.rightCols(nv_a_);
    double E_AA = 0.0, E_AB = 0.0;

    TBLIS_VIEW_4D(t_T2, t2_aa_, no_a_, no_a_, nv_a_, nv_a_);
    Eigen::Tensor<double, 4> W(no_a_, no_a_, nv_a_, nv_a_);
    TBLIS_VIEW_4D(t_W, W, no_a_, no_a_, nv_a_, nv_a_);

    
    {
        auto V = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cv, Cv, Cv, Cv, ints_);
        TBLIS_VIEW_4D(t_V, V, nv_a_, nv_a_, nv_a_, nv_a_);
        
        W.setZero(); 
        tblis::mult<double>(1.0, t_T2, "ijef", t_V, "eafb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T2, "ijef", t_V, "ebfa", 1.0, t_W, "ijab");
        E_AA += 0.125 * tensor_dot(t2_aa_, W);

        W.setZero(); 
        tblis::mult<double>(1.0, t_T2, "ijef", t_V, "eafb", 0.0, t_W, "ijab");
        E_AB += 1.0 * tensor_dot(t2_aa_, W);
    }

    
    {
        auto V = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Co, Co, Co, Co, ints_);
        TBLIS_VIEW_4D(t_V, V, no_a_, no_a_, no_a_, no_a_);
        
        W.setZero(); 
        tblis::mult<double>(1.0, t_T2, "mnab", t_V, "minj", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T2, "mnab", t_V, "mjni", 1.0, t_W, "ijab");
        E_AA += 0.125 * tensor_dot(t2_aa_, W);

        W.setZero(); 
        tblis::mult<double>(1.0, t_T2, "mnab", t_V, "minj", 0.0, t_W, "ijab");
        E_AB += 1.0 * tensor_dot(t2_aa_, W);
    }

    
    {
        auto V_ovov = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Co, Cv, Co, Cv, ints_);
        auto V_oovv = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Co, Co, Cv, Cv, ints_);
        TBLIS_VIEW_4D(t_Vovov, V_ovov, no_a_, nv_a_, no_a_, nv_a_);
        TBLIS_VIEW_4D(t_Voovv, V_oovv, no_a_, no_a_, nv_a_, nv_a_);

        W.setZero(); 
        tblis::mult<double>(1.0,  t_Vovov, "iakc", t_T2, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Vovov, "iakc", t_T2, "kjbc", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikac", t_T2, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(1.0,  t_Voovv, "ikac", t_T2, "kjbc", 1.0, t_W, "ijab");
        E_AA += 1.0 * tensor_dot(t2_aa_, W);

        W.setZero(); 
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
    res.e_mp2 = mp2_.energy_mp2_corr;
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

        
        Waa.setZero();
        tblis::mult<double>(1.0,  t_ovov_aa, "iakc", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
        tblis::mult<double>(-1.0, t_ovov_aa, "iakc", t_Taa, "kjbc", 1.0, t_Waa, "ijab");
        tblis::mult<double>(-1.0, t_oovv_aa, "ikac", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
        tblis::mult<double>(1.0,  t_oovv_aa, "ikac", t_Taa, "kjbc", 1.0, t_Waa, "ijab");
        tblis::mult<double>(1.0,  t_ovov_ab, "iakc", t_Tab, "jkbc", 1.0, t_Waa, "ijab"); 
        e3_aa += 1.0 * tensor_dot(t2_aa_, Waa);

        
        Wbb.setZero();
        tblis::mult<double>(1.0,  t_ovov_bb, "iakc", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
        tblis::mult<double>(-1.0, t_ovov_bb, "iakc", t_Tbb, "kjbc", 1.0, t_Wbb, "ijab");
        tblis::mult<double>(-1.0, t_oovv_bb, "ikac", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
        tblis::mult<double>(1.0,  t_oovv_bb, "ikac", t_Tbb, "kjbc", 1.0, t_Wbb, "ijab");
        tblis::mult<double>(1.0,  t_ovov_ab, "kcia", t_Tab, "kjcb", 1.0, t_Wbb, "ijab"); 
        e3_bb += 1.0 * tensor_dot(t2_bb_, Wbb);

        
        Wab.setZero();
        tblis::mult<double>(1.0,  t_ovov_aa, "iakc", t_Tab, "kjcb", 1.0, t_Wab, "ijab");
        tblis::mult<double>(-1.0, t_oovv_aa, "ikac", t_Tab, "kjcb", 1.0, t_Wab, "ijab");
        tblis::mult<double>(1.0,  t_Tab, "ikac", t_ovov_bb, "kcjb", 1.0, t_Wab, "ijab");
        tblis::mult<double>(-1.0, t_Tab, "ikac", t_oovv_bb, "kjcb", 1.0, t_Wab, "ijab");
        tblis::mult<double>(1.0,  t_Taa, "ikac", t_ovov_ab, "kcjb", 1.0, t_Wab, "ijab");
        tblis::mult<double>(-1.0, t_Taa, "ikac", t_ovov_ab, "kjcb", 1.0, t_Wab, "ijab"); 
        tblis::mult<double>(1.0,  t_ovov_ab, "iakc", t_Tbb, "kjcb", 1.0, t_Wab, "ijab");
        tblis::mult<double>(-1.0, t_oovv_ab, "ikbc", t_Tab, "kjac", 1.0, t_Wab, "ijab");
        tblis::mult<double>(-1.0, t_Tab, "kibc", t_oovv_ba, "kjac", 1.0, t_Wab, "ijab");
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






struct ZDIIS_Tensor {
    std::vector<Eigen::Tensor<double, 4>> tensor_hist; 
    std::vector<Eigen::Tensor<double, 4>> error_hist;  
    int max_vec = 6;
    void push(const Eigen::Tensor<double, 4>& val, const Eigen::Tensor<double, 4>& err) {
        tensor_hist.push_back(val); error_hist.push_back(err);
        if ((int)tensor_hist.size() > max_vec) { tensor_hist.erase(tensor_hist.begin()); error_hist.erase(error_hist.begin()); }
    }
    Eigen::Tensor<double, 4> extrapolate(int n1, int n2, int n3, int n4) {
        int n = tensor_hist.size();
        if (n < 2) return tensor_hist.back();
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n+1, n+1);
        for (int i = 0; i < n; ++i) {
            Eigen::Map<const Eigen::VectorXd> err_i(error_hist[i].data(), error_hist[i].size());
            for (int j = 0; j <= i; ++j) {
                Eigen::Map<const Eigen::VectorXd> err_j(error_hist[j].data(), error_hist[j].size());
                B(i,j) = err_i.dot(err_j); B(j,i) = B(i,j);
            }
            B(i,n) = -1.0; B(n,i) = -1.0;
        }
        B(n,n) = 0.0; Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n+1); rhs(n) = -1.0;
        Eigen::VectorXd c = B.colPivHouseholderQr().solve(rhs);
        Eigen::Tensor<double, 4> ext(n1, n2, n3, n4); ext.setZero();
        Eigen::Map<Eigen::VectorXd> ext_map(ext.data(), ext.size());
        for (int i = 0; i < n; ++i) {
            Eigen::Map<const Eigen::VectorXd> t_map(tensor_hist[i].data(), tensor_hist[i].size());
            ext_map += c(i) * t_map;
        }
        return ext;
    }
};

struct KappaDIIS {
    std::vector<Eigen::MatrixXd> kappa_hist; std::vector<Eigen::MatrixXd> grad_hist;
    int max_vec = 6;
    void push(const Eigen::MatrixXd& kappa, const Eigen::MatrixXd& grad) {
        kappa_hist.push_back(kappa); grad_hist.push_back(grad);
        if ((int)kappa_hist.size() > max_vec) { kappa_hist.erase(kappa_hist.begin()); grad_hist.erase(grad_hist.begin()); }
    }
    Eigen::MatrixXd extrapolate() {
        int n = kappa_hist.size(); if (n < 2) return kappa_hist.back();
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n+1, n+1);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) { B(i,j) = grad_hist[i].cwiseProduct(grad_hist[j]).sum(); B(j,i) = B(i,j); }
            B(i,n) = -1.0; B(n,i) = -1.0;
        }
        B(n,n) = 0.0; Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n+1); rhs(n) = -1.0;
        Eigen::VectorXd coeffs = B.colPivHouseholderQr().solve(rhs);
        Eigen::MatrixXd kappa_extrap = Eigen::MatrixXd::Zero(kappa_hist[0].rows(), kappa_hist[0].cols());
        for (int i = 0; i < n; ++i) kappa_extrap += coeffs(i) * kappa_hist[i];
        return kappa_extrap;
    }
};


static Eigen::MatrixXd pack_ladder_vvvv_as(const Eigen::Tensor<double, 4>& V, int nv) {
    if(nv == 0) return Eigen::MatrixXd();
    Eigen::MatrixXd M(nv*nv, nv*nv);
    #pragma omp parallel for collapse(2)
    for(int ab = 0; ab < nv*nv; ++ab) {
        for(int cd = 0; cd < nv*nv; ++cd) {
            int a = ab / nv; int b = ab % nv; int c = cd / nv; int d = cd % nv;
            M(ab, cd) = V(a,c,b,d) - V(a,d,b,c);
        }
    }
    return M;
}
static Eigen::MatrixXd pack_ladder_oooo_as(const Eigen::Tensor<double, 4>& V, int no) {
    if(no == 0) return Eigen::MatrixXd();
    Eigen::MatrixXd M(no*no, no*no);
    #pragma omp parallel for collapse(2)
    for(int kl = 0; kl < no*no; ++kl) {
        for(int ij = 0; ij < no*no; ++ij) {
            int k = kl / no; int l = kl % no; int i = ij / no; int j = ij % no;
            M(kl, ij) = V(k,i,l,j) - V(k,j,l,i);
        }
    }
    return M;
}
static Eigen::MatrixXd pack_t2_ij_ab(const Eigen::Tensor<double, 4>& T2, int no, int nv) {
    if(no == 0 || nv == 0) return Eigen::MatrixXd();
    Eigen::MatrixXd M(no*no, nv*nv);
    #pragma omp parallel for collapse(2)
    for(int ij = 0; ij < no*no; ++ij) {
        for(int ab = 0; ab < nv*nv; ++ab) {
            int i = ij / no; int j = ij % no; int a = ab / nv; int b = ab % nv;
            M(ij, ab) = T2(i,j,a,b);
        }
    }
    return M;
}




void OMP3::init_fast_integrals() {
    
    S_ = ints_->compute_overlap();
    H_core_ = ints_->compute_core_hamiltonian();
}
void OMP3::build_fock_fast(const Eigen::MatrixXd& P_a, const Eigen::MatrixXd& P_b, Eigen::MatrixXd& F_a, Eigen::MatrixXd& F_b) {
    
    int nbf = P_a.rows(); 
    if (nbf == 0) return; 

    Eigen::MatrixXd P_tot = P_a + P_b;
    Eigen::MatrixXd J_mat = Eigen::MatrixXd::Zero(nbf, nbf);
    Eigen::MatrixXd Ka_mat = Eigen::MatrixXd::Zero(nbf, nbf);
    Eigen::MatrixXd Kb_mat = Eigen::MatrixXd::Zero(nbf, nbf);

    if (scf_.L_mat.size() > 0) {
        int n_chol = scf_.L_mat.cols();
        
        #pragma omp parallel
        {
            Eigen::MatrixXd J_priv = Eigen::MatrixXd::Zero(nbf, nbf);
            Eigen::MatrixXd Ka_priv = Eigen::MatrixXd::Zero(nbf, nbf);
            Eigen::MatrixXd Kb_priv = Eigen::MatrixXd::Zero(nbf, nbf);
            Eigen::MatrixXd Ta_buf(nbf, nbf);
            Eigen::MatrixXd Tb_buf(nbf, nbf);

            #pragma omp for schedule(dynamic)
            for (int K = 0; K < n_chol; ++K) {
                Eigen::Map<const Eigen::MatrixXd> L_K(scf_.L_mat.col(K).data(), nbf, nbf);
                double val_J = (L_K.cwiseProduct(P_tot)).sum();
                J_priv += val_J * L_K;

                Ta_buf.noalias() = L_K * P_a;
                Ka_priv.noalias() += Ta_buf * L_K;
                if (no_b_ > 0) {
                    Tb_buf.noalias() = L_K * P_b;
                    Kb_priv.noalias() += Tb_buf * L_K;
                }
            }
            #pragma omp critical
            {
                J_mat += J_priv; Ka_mat += Ka_priv; if (no_b_ > 0) Kb_mat += Kb_priv;
            }
        }
    } else {
        auto eri_ao = ints_->compute_eri();
        for(int mu=0; mu<nbf; ++mu) {
            for(int nu=0; nu<nbf; ++nu) {
                for(int lam=0; lam<nbf; ++lam) {
                    for(int sig=0; sig<nbf; ++sig) {
                        J_mat(mu,nu) += eri_ao(mu,nu,lam,sig) * P_tot(lam,sig);
                        Ka_mat(mu,nu) += eri_ao(mu,lam,nu,sig) * P_a(lam,sig);
                        if (no_b_ > 0) Kb_mat(mu,nu) += eri_ao(mu,lam,nu,sig) * P_b(lam,sig);
                    }
                }
            }
        }
    }

    F_a = H_core_ + J_mat - Ka_mat;
    if (no_b_ > 0) F_b = H_core_ + J_mat - Kb_mat;
    else F_b = F_a;
}



void OMP3::pseudocanonicalize() {
    Eigen::MatrixXd F_alpha, F_beta;
    
    
    build_fock_fast(scf_.P_alpha, scf_.P_beta, F_alpha, F_beta);
    auto diag_block = [&](Eigen::MatrixXd& C, Eigen::VectorXd& eps, const Eigen::MatrixXd& F_ao, int n_occ, int n_virt) {
        Eigen::MatrixXd F_mo = C.transpose() * F_ao * C;
        eps.resize(n_occ + n_virt);
        if (n_occ > 0) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_o(F_mo.block(0, 0, n_occ, n_occ));
            eps.head(n_occ) = es_o.eigenvalues(); C.leftCols(n_occ) = C.leftCols(n_occ) * es_o.eigenvectors();
        }
        if (n_virt > 0) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_v(F_mo.block(n_occ, n_occ, n_virt, n_virt));
            eps.tail(n_virt) = es_v.eigenvalues(); C.rightCols(n_virt) = C.rightCols(n_virt) * es_v.eigenvectors();
        }
    };
    diag_block(scf_.C_alpha, scf_.orbital_energies_alpha, F_alpha, no_a_, nv_a_);
    diag_block(scf_.C_beta, scf_.orbital_energies_beta, F_beta, no_b_, nv_b_);
    scf_.P_alpha = scf_.C_alpha.leftCols(no_a_) * scf_.C_alpha.leftCols(no_a_).transpose();
    scf_.P_beta = scf_.C_beta.leftCols(no_b_) * scf_.C_beta.leftCols(no_b_).transpose();
}

double OMP3::compute_mp2_energy() {
    if (no_a_ == 0 || nv_a_ == 0) return 0.0;
    t2_aa_ = Eigen::Tensor<double, 4>(no_a_, no_a_, nv_a_, nv_a_);
    if (no_b_ > 0 && nv_b_ > 0) {
        t2_bb_ = Eigen::Tensor<double, 4>(no_b_, no_b_, nv_b_, nv_b_);
        t2_ab_ = Eigen::Tensor<double, 4>(no_a_, no_b_, nv_a_, nv_b_);
    }
    const Eigen::MatrixXd& Cao = scf_.C_alpha.leftCols(no_a_); const Eigen::MatrixXd& Cav = scf_.C_alpha.rightCols(nv_a_);
    const auto& ea = scf_.orbital_energies_alpha; double e_sum = 0.0;
    
    
    auto g_aa = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cav, Cao, Cav, ints_);
    double e_aa = 0.0;
    #pragma omp parallel for collapse(4) reduction(+:e_aa)
    for(int i=0; i<no_a_; ++i) for(int j=0; j<no_a_; ++j) 
        for(int a=0; a<nv_a_; ++a) for(int b=0; b<nv_a_; ++b) {
            double D = ea(i)+ea(j)-ea(no_a_+a)-ea(no_a_+b);
            if(std::abs(D)>1e-12) {
                double val = g_aa(i,a,j,b) - g_aa(i,b,j,a);
                t2_aa_(i,j,a,b) = val / D; e_aa += t2_aa_(i,j,a,b) * val;
            } else t2_aa_(i,j,a,b) = 0.0;
        }
    e_sum += 0.25 * e_aa;

    if (no_b_ > 0 && nv_b_ > 0) {
        const Eigen::MatrixXd& Cbo = scf_.C_beta.leftCols(no_b_); const Eigen::MatrixXd& Cbv = scf_.C_beta.rightCols(nv_b_);
        const auto& eb = scf_.orbital_energies_beta;
        
        auto g_bb = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbv, Cbo, Cbv, ints_);
        double e_bb = 0.0;
        #pragma omp parallel for collapse(4) reduction(+:e_bb)
        for(int i=0; i<no_b_; ++i) for(int j=0; j<no_b_; ++j)
            for(int a=0; a<nv_b_; ++a) for(int b=0; b<nv_b_; ++b) {
                double D = eb(i)+eb(j)-eb(no_b_+a)-eb(no_b_+b);
                if(std::abs(D)>1e-12) {
                    double val = g_bb(i,a,j,b) - g_bb(i,b,j,a);
                    t2_bb_(i,j,a,b) = val / D; e_bb += t2_bb_(i,j,a,b) * val;
                } else t2_bb_(i,j,a,b) = 0.0;
            }
        e_sum += 0.25 * e_bb;

        auto g_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cav, Cbo, Cbv, ints_);
        double e_ab = 0.0;
        #pragma omp parallel for collapse(4) reduction(+:e_ab)
        for(int i=0; i<no_a_; ++i) for(int j=0; j<no_b_; ++j)
            for(int a=0; a<nv_a_; ++a) for(int b=0; b<nv_b_; ++b) {
                double D = ea(i)+eb(j)-ea(no_a_+a)-eb(no_b_+b);
                if(std::abs(D)>1e-12) {
                    double val = g_ab(i,a,j,b);
                    t2_ab_(i,j,a,b) = val / D; e_ab += t2_ab_(i,j,a,b) * val;
                } else t2_ab_(i,j,a,b) = 0.0;
            }
        e_sum += e_ab;
    }
    return e_sum;
}

double OMP3::compute_mp3_correction() {
    if (no_a_ == 0 || nv_a_ == 0) return 0.0;
    
    t2_3rd_aa_ = Eigen::Tensor<double, 4>(no_a_, no_a_, nv_a_, nv_a_); t2_3rd_aa_.setZero();
    if (no_b_ > 0 && nv_b_ > 0) {
        t2_3rd_bb_ = Eigen::Tensor<double, 4>(no_b_, no_b_, nv_b_, nv_b_); t2_3rd_bb_.setZero();
        t2_3rd_ab_ = Eigen::Tensor<double, 4>(no_a_, no_b_, nv_a_, nv_b_); t2_3rd_ab_.setZero();
    }

    const Eigen::MatrixXd& Cao = scf_.C_alpha.leftCols(no_a_); const Eigen::MatrixXd& Cav = scf_.C_alpha.rightCols(nv_a_);
    const auto& ea = scf_.orbital_energies_alpha;
    double e3_aa = 0.0, e3_bb = 0.0, e3_ab = 0.0;

    TBLIS_VIEW_4D(t_Taa, t2_aa_, no_a_, no_a_, nv_a_, nv_a_);
    Eigen::Tensor<double, 4> Waa(no_a_, no_a_, nv_a_, nv_a_); TBLIS_VIEW_4D(t_Waa, Waa, no_a_, no_a_, nv_a_, nv_a_);

    
    
    
    {
        auto V_vvvv = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cav, Cav, Cav, Cav, ints_);
        TBLIS_VIEW_4D(t_Vvvvv, V_vvvv, nv_a_, nv_a_, nv_a_, nv_a_);
        Waa.setZero();
        tblis::mult<double>(1.0, t_Taa, "ijef", t_Vvvvv, "eafb", 1.0, t_Waa, "ijab");
        tblis::mult<double>(-1.0, t_Taa, "ijef", t_Vvvvv, "ebfa", 1.0, t_Waa, "ijab");

        auto V_oooo = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cao, Cao, ints_);
        TBLIS_VIEW_4D(t_Voooo, V_oooo, no_a_, no_a_, no_a_, no_a_);
        tblis::mult<double>(1.0, t_Taa, "mnab", t_Voooo, "minj", 1.0, t_Waa, "ijab");
        tblis::mult<double>(-1.0, t_Taa, "mnab", t_Voooo, "mjni", 1.0, t_Waa, "ijab");

        auto V_ovov = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cav, Cao, Cav, ints_);
        auto V_oovv = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cav, Cav, ints_);
        TBLIS_VIEW_4D(t_Vovov, V_ovov, no_a_, nv_a_, no_a_, nv_a_);
        TBLIS_VIEW_4D(t_Voovv, V_oovv, no_a_, no_a_, nv_a_, nv_a_);

        tblis::mult<double>(1.0,  t_Vovov, "iakc", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
        tblis::mult<double>(-1.0, t_Vovov, "iakc", t_Taa, "kjbc", 1.0, t_Waa, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikac", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
        tblis::mult<double>(1.0,  t_Voovv, "ikac", t_Taa, "kjbc", 1.0, t_Waa, "ijab");

        if (no_b_ > 0 && nv_b_ > 0) {
            auto V_ovov_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cav, scf_.C_beta.leftCols(no_b_), scf_.C_beta.rightCols(nv_b_), ints_);
            TBLIS_VIEW_4D(t_Vovov_ab, V_ovov_ab, no_a_, nv_a_, no_b_, nv_b_);
            TBLIS_VIEW_4D(t_Tab, t2_ab_, no_a_, no_b_, nv_a_, nv_b_);
            tblis::mult<double>(1.0, t_Vovov_ab, "iakc", t_Tab, "jkbc", 1.0, t_Waa, "ijab");
        }

        #pragma omp parallel for collapse(4) reduction(+:e3_aa)
        for(int i=0; i<no_a_; ++i) for(int j=0; j<no_a_; ++j) for(int a=0; a<nv_a_; ++a) for(int b=0; b<nv_a_; ++b) {
            double D = ea(i) + ea(j) - ea(no_a_+a) - ea(no_a_+b);
            if (std::abs(D) > 1e-12) {
                double val = Waa(i,j,a,b) / D;
                t2_3rd_aa_(i,j,a,b) = val;
                e3_aa += 0.125 * t2_aa_(i,j,a,b) * Waa(i,j,a,b);
            }
        }
    }

    if (no_b_ > 0 && nv_b_ > 0) {
        const Eigen::MatrixXd& Cbo = scf_.C_beta.leftCols(no_b_); const Eigen::MatrixXd& Cbv = scf_.C_beta.rightCols(nv_b_);
        const auto& eb = scf_.orbital_energies_beta;
        
        
        
        
        TBLIS_VIEW_4D(t_Tbb, t2_bb_, no_b_, no_b_, nv_b_, nv_b_);
        TBLIS_VIEW_4D(t_Tab, t2_ab_, no_a_, no_b_, nv_a_, nv_b_);

        Eigen::Tensor<double, 4> Wbb(no_b_, no_b_, nv_b_, nv_b_); TBLIS_VIEW_4D(t_Wbb, Wbb, no_b_, no_b_, nv_b_, nv_b_);
        
        {
            auto V_vvvv = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbv, Cbv, Cbv, Cbv, ints_);
            TBLIS_VIEW_4D(t_Vvvvv, V_vvvv, nv_b_, nv_b_, nv_b_, nv_b_);
            Wbb.setZero();
            tblis::mult<double>(1.0, t_Tbb, "ijef", t_Vvvvv, "eafb", 1.0, t_Wbb, "ijab");
            tblis::mult<double>(-1.0, t_Tbb, "ijef", t_Vvvvv, "ebfa", 1.0, t_Wbb, "ijab");

            auto V_oooo = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbo, Cbo, Cbo, ints_);
            TBLIS_VIEW_4D(t_Voooo, V_oooo, no_b_, no_b_, no_b_, no_b_);
            tblis::mult<double>(1.0, t_Tbb, "mnab", t_Voooo, "minj", 1.0, t_Wbb, "ijab");
            tblis::mult<double>(-1.0, t_Tbb, "mnab", t_Voooo, "mjni", 1.0, t_Wbb, "ijab");

            auto V_ovov = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbv, Cbo, Cbv, ints_);
            auto V_oovv = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbo, Cbv, Cbv, ints_);
            TBLIS_VIEW_4D(t_Vovov, V_ovov, no_b_, nv_b_, no_b_, nv_b_);
            TBLIS_VIEW_4D(t_Voovv, V_oovv, no_b_, no_b_, nv_b_, nv_b_);

            tblis::mult<double>(1.0,  t_Vovov, "iakc", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
            tblis::mult<double>(-1.0, t_Vovov, "iakc", t_Tbb, "kjbc", 1.0, t_Wbb, "ijab");
            tblis::mult<double>(-1.0, t_Voovv, "ikac", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
            tblis::mult<double>(1.0,  t_Voovv, "ikac", t_Tbb, "kjbc", 1.0, t_Wbb, "ijab");

            auto V_ovov_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cav, Cbo, Cbv, ints_);
            TBLIS_VIEW_4D(t_Vovov_ab, V_ovov_ab, no_a_, nv_a_, no_b_, nv_b_);
            tblis::mult<double>(1.0, t_Vovov_ab, "kcia", t_Tab, "kjcb", 1.0, t_Wbb, "ijab");

            #pragma omp parallel for collapse(4) reduction(+:e3_bb)
            for(int i=0; i<no_b_; ++i) for(int j=0; j<no_b_; ++j) for(int a=0; a<nv_b_; ++a) for(int b=0; b<nv_b_; ++b) {
                double D = eb(i) + eb(j) - eb(no_b_+a) - eb(no_b_+b);
                if (std::abs(D) > 1e-12) {
                    double val = Wbb(i,j,a,b) / D;
                    t2_3rd_bb_(i,j,a,b) = val;
                    e3_bb += 0.125 * t2_bb_(i,j,a,b) * Wbb(i,j,a,b);
                }
            }
        }

        
        
        
        {
            Eigen::Tensor<double, 4> Wab(no_a_, no_b_, nv_a_, nv_b_); TBLIS_VIEW_4D(t_Wab, Wab, no_a_, no_b_, nv_a_, nv_b_);
            
            auto V_vvvv_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cav, Cav, Cbv, Cbv, ints_);
            TBLIS_VIEW_4D(t_Vvvvv_ab, V_vvvv_ab, nv_a_, nv_a_, nv_b_, nv_b_);
            Wab.setZero();
            tblis::mult<double>(1.0, t_Tab, "ijef", t_Vvvvv_ab, "eafb", 1.0, t_Wab, "ijab");

            auto V_oooo_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cbo, Cbo, ints_);
            TBLIS_VIEW_4D(t_Voooo_ab, V_oooo_ab, no_a_, no_a_, no_b_, no_b_);
            tblis::mult<double>(1.0, t_Tab, "mnab", t_Voooo_ab, "minj", 1.0, t_Wab, "ijab");

            auto V_ovov_aa = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cav, Cao, Cav, ints_);
            auto V_oovv_aa = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cav, Cav, ints_);
            TBLIS_VIEW_4D(t_Vovov_aa, V_ovov_aa, no_a_, nv_a_, no_a_, nv_a_);
            TBLIS_VIEW_4D(t_Voovv_aa, V_oovv_aa, no_a_, no_a_, nv_a_, nv_a_);
            
            tblis::mult<double>(1.0,  t_Vovov_aa, "iakc", t_Tab, "kjcb", 1.0, t_Wab, "ijab");
            tblis::mult<double>(-1.0, t_Voovv_aa, "ikac", t_Tab, "kjcb", 1.0, t_Wab, "ijab");

            auto V_ovov_bb = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbv, Cbo, Cbv, ints_);
            auto V_oovv_bb = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbo, Cbv, Cbv, ints_);
            TBLIS_VIEW_4D(t_Vovov_bb, V_ovov_bb, no_b_, nv_b_, no_b_, nv_b_);
            TBLIS_VIEW_4D(t_Voovv_bb, V_oovv_bb, no_b_, no_b_, nv_b_, nv_b_);

            tblis::mult<double>(1.0,  t_Tab, "ikac", t_Vovov_bb, "kcjb", 1.0, t_Wab, "ijab"); 
            tblis::mult<double>(-1.0, t_Tab, "ikac", t_Voovv_bb, "kjcb", 1.0, t_Wab, "ijab"); 

            auto V_ovov_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cav, Cbo, Cbv, ints_);
            TBLIS_VIEW_4D(t_Vovov_ab, V_ovov_ab, no_a_, nv_a_, no_b_, nv_b_);

            tblis::mult<double>(1.0,  t_Taa, "ikac", t_Vovov_ab, "kcjb", 1.0, t_Wab, "ijab");
            tblis::mult<double>(-1.0, t_Taa, "ikac", t_Vovov_ab, "kjcb", 1.0, t_Wab, "ijab");
            tblis::mult<double>(1.0,  t_Vovov_ab, "iakc", t_Tbb, "kjcb", 1.0, t_Wab, "ijab");

            auto V_oovv_ab_ex = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cbv, Cbv, ints_);
            auto V_oovv_ba_ex = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbo, Cav, Cav, ints_);
            TBLIS_VIEW_4D(t_Voovv_ab_ex, V_oovv_ab_ex, no_a_, no_a_, nv_b_, nv_b_);
            TBLIS_VIEW_4D(t_Voovv_ba_ex, V_oovv_ba_ex, no_b_, no_b_, nv_a_, nv_a_);

            tblis::mult<double>(-1.0, t_Voovv_ab_ex, "ikbc", t_Tab, "kjac", 1.0, t_Wab, "ijab");
            tblis::mult<double>(-1.0, t_Tab, "kibc", t_Voovv_ba_ex, "kjac", 1.0, t_Wab, "ijab");

            #pragma omp parallel for collapse(4) reduction(+:e3_ab)
            for(int i=0; i<no_a_; ++i) for(int j=0; j<no_b_; ++j) for(int a=0; a<nv_a_; ++a) for(int b=0; b<nv_b_; ++b) {
                double D = ea(i) + eb(j) - ea(no_a_+a) - eb(no_b_+b);
                if (std::abs(D) > 1e-12) {
                    double val = Wab(i,j,a,b) / D;
                    t2_3rd_ab_(i,j,a,b) = val;
                    e3_ab += 1.0 * t2_ab_(i,j,a,b) * Wab(i,j,a,b);
                }
            }
        }
    }
    return e3_aa + e3_bb + e3_ab;
}




void OMP3::solve_zvector() {
    if (no_a_ == 0 || nv_a_ == 0) return;
    if (omp_get_thread_num() == 0) std::cout << "  [Z-Vector] Relaxing orbital parameters...\n";

    L2_aa_ = Eigen::Tensor<double, 4>(no_a_, no_a_, nv_a_, nv_a_);
    for(int i=0; i<t2_3rd_aa_.size(); ++i) L2_aa_.data()[i] = 1.0 * t2_3rd_aa_.data()[i];
    
    if (no_b_ > 0 && nv_b_ > 0) {
        L2_bb_ = Eigen::Tensor<double, 4>(no_b_, no_b_, nv_b_, nv_b_);
        for(int i=0; i<t2_3rd_bb_.size(); ++i) L2_bb_.data()[i] = 1.0 * t2_3rd_bb_.data()[i];
        L2_ab_ = Eigen::Tensor<double, 4>(no_a_, no_b_, nv_a_, nv_b_);
        for(int i=0; i<t2_3rd_ab_.size(); ++i) L2_ab_.data()[i] = 1.0 * t2_3rd_ab_.data()[i];
    }

    int max_z_iter = 50; double z_thresh = 1e-7; double rms_error = 1.0; int iter = 0;
    const Eigen::MatrixXd& Cao = scf_.C_alpha.leftCols(no_a_); const Eigen::MatrixXd& Cav = scf_.C_alpha.rightCols(nv_a_);
    const auto& ea = scf_.orbital_energies_alpha;

    
    auto V_VVVV_aa = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cav, Cav, Cav, Cav, ints_);
    Eigen::MatrixXd packed_vvvv_aa = pack_ladder_vvvv_as(V_VVVV_aa, nv_a_);
    auto V_OOOO_aa = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cao, Cao, ints_);
    Eigen::MatrixXd packed_oooo_aa = pack_ladder_oooo_as(V_OOOO_aa, no_a_);
    
    auto ovov_aa = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cav, Cao, Cav, ints_);
    auto oovv_aa = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cav, Cav, ints_);
    
    Eigen::MatrixXd V_AA(no_a_*nv_a_, no_a_*nv_a_);
    for(int ia=0; ia<no_a_*nv_a_; ++ia) for(int kc=0; kc<no_a_*nv_a_; ++kc) {
        int i=ia/nv_a_; int a=ia%nv_a_; int k=kc/nv_a_; int c=kc%nv_a_;
        V_AA(ia, kc) = ovov_aa(i,a,k,c) - oovv_aa(k,i,a,c);
    }

    Eigen::MatrixXd packed_vvvv_bb, packed_oooo_bb, V_BB, V_cr_ab_aa, V_cr_ab_bb, V_AB_ring, V_AB_T4_ring, V_C1, V_C2;
    Eigen::Tensor<double, 4> Vv_ab, Vo_ab, ovov_bb, oovv_bb, g_ovov_ab, g_oovv_ab_ex, g_oovv_ba_ex;
    
    if (no_b_ > 0 && nv_b_ > 0) {
        const Eigen::MatrixXd& Cbo = scf_.C_beta.leftCols(no_b_); const Eigen::MatrixXd& Cbv = scf_.C_beta.rightCols(nv_b_);
        auto V_VVVV_bb = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbv, Cbv, Cbv, Cbv, ints_);
        packed_vvvv_bb = pack_ladder_vvvv_as(V_VVVV_bb, nv_b_);
        auto V_OOOO_bb = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbo, Cbo, Cbo, ints_);
        packed_oooo_bb = pack_ladder_oooo_as(V_OOOO_bb, no_b_);

        ovov_bb = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbv, Cbo, Cbv, ints_);
        oovv_bb = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbo, Cbv, Cbv, ints_);
        V_BB = Eigen::MatrixXd::Zero(no_b_*nv_b_, no_b_*nv_b_);
        for(int ia=0; ia<no_b_*nv_b_; ++ia) for(int kc=0; kc<no_b_*nv_b_; ++kc) {
            int i=ia/nv_b_; int a=ia%nv_b_; int k=kc/nv_b_; int c=kc%nv_b_;
            V_BB(ia, kc) = ovov_bb(i,a,k,c) - oovv_bb(k,i,a,c);
        }

        g_ovov_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cav, Cbo, Cbv, ints_);
        V_cr_ab_aa = Eigen::MatrixXd::Zero(no_a_*nv_a_, no_b_*nv_b_);
        for(int i=0; i<no_a_; ++i) for(int a=0; a<nv_a_; ++a) for(int k=0; k<no_b_; ++k) for(int c=0; c<nv_b_; ++c)
            V_cr_ab_aa(i*nv_a_+a, k*nv_b_+c) = g_ovov_ab(i,a,k,c);
            
        V_cr_ab_bb = Eigen::MatrixXd::Zero(no_b_*nv_b_, no_a_*nv_a_);
        for(int i=0; i<no_b_; ++i) for(int a=0; a<nv_b_; ++a) for(int k=0; k<no_a_; ++k) for(int c=0; c<nv_a_; ++c)
            V_cr_ab_bb(i*nv_b_+a, k*nv_a_+c) = g_ovov_ab(k,c,i,a);

        Vv_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cav, Cav, Cbv, Cbv, ints_);
        Vo_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cbo, Cbo, ints_);

        V_AB_ring = Eigen::MatrixXd::Zero(no_a_*nv_a_, no_b_*nv_b_);
        for(int k=0; k<no_a_; ++k) for(int c=0; c<nv_a_; ++c) for(int j=0; j<no_b_; ++j) for(int b=0; b<nv_b_; ++b) 
            V_AB_ring(k*nv_a_+c, j*nv_b_+b) = g_ovov_ab(k,c,j,b);
            
        V_AB_T4_ring = Eigen::MatrixXd::Zero(no_a_*nv_a_, no_b_*nv_b_);
        for(int i=0; i<no_a_; ++i) for(int a=0; a<nv_a_; ++a) for(int k=0; k<no_b_; ++k) for(int c=0; c<nv_b_; ++c) 
            V_AB_T4_ring(i*nv_a_+a, k*nv_b_+c) = g_ovov_ab(i,a,k,c);

        g_oovv_ab_ex = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cbv, Cbv, ints_);
        g_oovv_ba_ex = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbo, Cav, Cav, ints_);
        
        V_C1 = Eigen::MatrixXd::Zero(no_a_*nv_b_, no_a_*nv_b_);
        for(int i=0; i<no_a_; ++i) for(int b=0; b<nv_b_; ++b) for(int k=0; k<no_a_; ++k) for(int c=0; c<nv_b_; ++c) 
            V_C1(i*nv_b_+b, k*nv_b_+c) = g_oovv_ab_ex(i,k,b,c);
            
        V_C2 = Eigen::MatrixXd::Zero(no_b_*nv_a_, no_b_*nv_a_);
        for(int j=0; j<no_b_; ++j) for(int a=0; a<nv_a_; ++a) for(int k=0; k<no_b_; ++k) for(int c=0; c<nv_a_; ++c) 
            V_C2(j*nv_a_+a, k*nv_a_+c) = g_oovv_ba_ex(j,k,a,c);
    }

    ZDIIS_Tensor diis_aa, diis_bb, diis_ab;
    auto unpack_ijab_to_iajb = [](const Eigen::MatrixXd& W_ijab, Eigen::MatrixXd& W_iajb, int no, int nv) {
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < no; ++i) {
            for(int j = 0; j < no; ++j) {
                for(int a = 0; a < nv; ++a) {
                    for(int b = 0; b < nv; ++b) W_iajb(i*nv + a, j*nv + b) += W_ijab(i*no + j, a*nv + b);
                }
            }
        }
    };

    while (rms_error > z_thresh && iter < max_z_iter) {
        rms_error = 0.0; int total_elements = no_a_*no_a_*nv_a_*nv_a_;
        Eigen::Tensor<double, 4> R_aa(no_a_, no_a_, nv_a_, nv_a_); R_aa.setZero();
        Eigen::Tensor<double, 4> R_bb, R_ab;
        Eigen::MatrixXd W_res_AA = Eigen::MatrixXd::Zero(no_a_*nv_a_, no_a_*nv_a_);
        Eigen::MatrixXd L2_mat_aa = pack_t2_ij_ab(L2_aa_, no_a_, nv_a_);
        Eigen::MatrixXd W_ijab_aa = 0.5 * (L2_mat_aa * packed_vvvv_aa) + 0.5 * (packed_oooo_aa.transpose() * L2_mat_aa);
        unpack_ijab_to_iajb(W_ijab_aa, W_res_AA, no_a_, nv_a_);

        Eigen::MatrixXd L2_iajb_AA(no_a_*nv_a_, no_a_*nv_a_);
        for(int ia=0; ia<no_a_*nv_a_; ++ia) for(int jb=0; jb<no_a_*nv_a_; ++jb) {
            int i=ia/nv_a_; int a=ia%nv_a_; int j=jb/nv_a_; int b=jb%nv_a_;
            L2_iajb_AA(ia, jb) = L2_aa_(i,j,a,b);
        }
        Eigen::MatrixXd Z_Ring_raw_AA = V_AA * L2_iajb_AA;
        
        if (no_b_ > 0 && nv_b_ > 0) {
            Eigen::MatrixXd L2_cr_ab(no_b_*nv_b_, no_a_*nv_a_);
            for (int k=0; k<no_b_; ++k) for (int c=0; c<nv_b_; ++c) for (int j=0; j<no_a_; ++j) for (int b=0; b<nv_a_; ++b)
                L2_cr_ab(k*nv_b_+c, j*nv_a_+b) = L2_ab_(j,k,b,c);
            Z_Ring_raw_AA += V_cr_ab_aa * L2_cr_ab;
        }

        #pragma omp parallel for collapse(2)
        for(int i=0; i<no_a_; ++i) {
            for(int j=0; j<no_a_; ++j) {
                for(int a=0; a<nv_a_; ++a) {
                    for(int b=0; b<nv_a_; ++b) {
                        W_res_AA(i*nv_a_+a, j*nv_a_+b) += Z_Ring_raw_AA(i*nv_a_+a, j*nv_a_+b) - Z_Ring_raw_AA(j*nv_a_+a, i*nv_a_+b) 
                                                        - Z_Ring_raw_AA(i*nv_a_+b, j*nv_a_+a) + Z_Ring_raw_AA(j*nv_a_+b, i*nv_a_+a);
                    }
                }
            }
        }

        Eigen::Tensor<double, 4> L2_next_aa(no_a_, no_a_, nv_a_, nv_a_);
        for(int ia=0; ia<no_a_*nv_a_; ++ia) for(int jb=0; jb<no_a_*nv_a_; ++jb) {
            int i=ia/nv_a_; int a=ia%nv_a_; int j=jb/nv_a_; int b=jb%nv_a_;
            double D = ea(i) + ea(j) - ea(no_a_+a) - ea(no_a_+b);
            double L2_new = 1.0 * t2_3rd_aa_(i,j,a,b);
            if (std::abs(D) > 1e-12) L2_new -= W_res_AA(ia, jb) / D;
            L2_next_aa(i,j,a,b) = L2_new;
            double err = L2_new - L2_aa_(i,j,a,b); R_aa(i,j,a,b) = err; rms_error += err * err;
        }

        Eigen::Tensor<double, 4> L2_next_bb, L2_next_ab;
        if (no_b_ > 0 && nv_b_ > 0) {
            const auto& eb = scf_.orbital_energies_beta;
            R_bb = Eigen::Tensor<double, 4>(no_b_, no_b_, nv_b_, nv_b_); R_bb.setZero();
            R_ab = Eigen::Tensor<double, 4>(no_a_, no_b_, nv_a_, nv_b_); R_ab.setZero();
            L2_next_bb = Eigen::Tensor<double, 4>(no_b_, no_b_, nv_b_, nv_b_);
            L2_next_ab = Eigen::Tensor<double, 4>(no_a_, no_b_, nv_a_, nv_b_);
            total_elements += (no_b_*no_b_*nv_b_*nv_b_) + (no_a_*no_b_*nv_a_*nv_b_);

            Eigen::MatrixXd W_res_BB = Eigen::MatrixXd::Zero(no_b_*nv_b_, no_b_*nv_b_);
            Eigen::MatrixXd L2_mat_bb = pack_t2_ij_ab(L2_bb_, no_b_, nv_b_);
            Eigen::MatrixXd W_ijab_bb = 0.5 * (L2_mat_bb * packed_vvvv_bb) + 0.5 * (packed_oooo_bb.transpose() * L2_mat_bb);
            unpack_ijab_to_iajb(W_ijab_bb, W_res_BB, no_b_, nv_b_);

            Eigen::MatrixXd L2_iajb_BB(no_b_*nv_b_, no_b_*nv_b_);
            for(int ia=0; ia<no_b_*nv_b_; ++ia) for(int jb=0; jb<no_b_*nv_b_; ++jb) {
                int i=ia/nv_b_; int a=ia%nv_b_; int j=jb/nv_b_; int b=jb%nv_b_;
                L2_iajb_BB(ia, jb) = L2_bb_(i,j,a,b);
            }
            Eigen::MatrixXd Z_Ring_raw_BB = V_BB * L2_iajb_BB;
            
            Eigen::MatrixXd L2_cr_ba(no_a_*nv_a_, no_b_*nv_b_);
            for (int k=0; k<no_a_; ++k) for (int c=0; c<nv_a_; ++c) for (int j=0; j<no_b_; ++j) for (int b=0; b<nv_b_; ++b)
                L2_cr_ba(k*nv_a_+c, j*nv_b_+b) = L2_ab_(k,j,c,b);
            Z_Ring_raw_BB += V_cr_ab_bb * L2_cr_ba;

            #pragma omp parallel for collapse(2)
            for(int i=0; i<no_b_; ++i) {
                for(int j=0; j<no_b_; ++j) {
                    for(int a=0; a<nv_b_; ++a) {
                        for(int b=0; b<nv_b_; ++b) {
                            W_res_BB(i*nv_b_+a, j*nv_b_+b) += Z_Ring_raw_BB(i*nv_b_+a, j*nv_b_+b) - Z_Ring_raw_BB(j*nv_b_+a, i*nv_b_+b) 
                                                            - Z_Ring_raw_BB(i*nv_b_+b, j*nv_b_+a) + Z_Ring_raw_BB(j*nv_b_+b, i*nv_b_+a);
                        }
                    }
                }
            }

            for(int ia=0; ia<no_b_*nv_b_; ++ia) for(int jb=0; jb<no_b_*nv_b_; ++jb) {
                int i=ia/nv_b_; int a=ia%nv_b_; int j=jb/nv_b_; int b=jb%nv_b_;
                double D = eb(i) + eb(j) - eb(no_b_+a) - eb(no_b_+b);
                double L2_new = 1.0 * t2_3rd_bb_(i,j,a,b);
                if (std::abs(D) > 1e-12) L2_new -= W_res_BB(ia, jb) / D;
                L2_next_bb(i,j,a,b) = L2_new;
                double err = L2_new - L2_bb_(i,j,a,b); R_bb(i,j,a,b) = err; rms_error += err * err;
            }

            
            Eigen::MatrixXd W_res_AB = Eigen::MatrixXd::Zero(no_a_*nv_a_, no_b_*nv_b_);
            #pragma omp parallel for collapse(2)
            for(int i=0; i<no_a_; ++i) for(int j=0; j<no_b_; ++j) {
                for(int a=0; a<nv_a_; ++a) for(int b=0; b<nv_b_; ++b) {
                    double sum = 0;
                    for(int c=0; c<nv_a_; ++c) for(int d=0; d<nv_b_; ++d) sum += Vv_ab(a,c,b,d) * L2_ab_(i,j,c,d);
                    for(int k=0; k<no_a_; ++k) for(int l=0; l<no_b_; ++l) sum += Vo_ab(k,i,l,j) * L2_ab_(k,l,a,b);
                    W_res_AB(i*nv_a_+a, j*nv_b_+b) += sum;
                }
            }

            Eigen::MatrixXd L2_iajb_AB(no_a_*nv_a_, no_b_*nv_b_);
            for(int i=0; i<no_a_; ++i) for(int a=0; a<nv_a_; ++a) for(int j=0; j<no_b_; ++j) for(int b=0; b<nv_b_; ++b) 
                L2_iajb_AB(i*nv_a_+a, j*nv_b_+b) = L2_ab_(i,j,a,b);
            W_res_AB += V_AA * L2_iajb_AB;
            
            Eigen::MatrixXd L2_AB_ik(no_a_*nv_a_, no_b_*nv_b_);
            for(int i=0; i<no_a_; ++i) for(int a=0; a<nv_a_; ++a) for(int k=0; k<no_b_; ++k) for(int c=0; c<nv_b_; ++c) 
                L2_AB_ik(i*nv_a_+a, k*nv_b_+c) = L2_ab_(i,k,a,c);
            W_res_AB += L2_AB_ik * V_BB;
            
            Eigen::MatrixXd L2_AA_mat(no_a_*nv_a_, no_a_*nv_a_);
            for(int i=0; i<no_a_; ++i) for(int a=0; a<nv_a_; ++a) for(int k=0; k<no_a_; ++k) for(int c=0; c<nv_a_; ++c) 
                L2_AA_mat(i*nv_a_+a, k*nv_a_+c) = L2_aa_(i,k,a,c);
            W_res_AB += L2_AA_mat * V_AB_ring;

            Eigen::MatrixXd L2_BB_mat(no_b_*nv_b_, no_b_*nv_b_);
            for(int k=0; k<no_b_; ++k) for(int c=0; c<nv_b_; ++c) for(int j=0; j<no_b_; ++j) for(int b=0; b<nv_b_; ++b) 
                L2_BB_mat(k*nv_b_+c, j*nv_b_+b) = L2_bb_(k,j,c,b);
            W_res_AB += V_AB_T4_ring * L2_BB_mat;
            
            Eigen::MatrixXd L2_C1(no_a_*nv_b_, no_b_*nv_a_);
            for(int k=0; k<no_a_; ++k) for(int c=0; c<nv_b_; ++c) for(int j=0; j<no_b_; ++j) for(int a=0; a<nv_a_; ++a) 
                L2_C1(k*nv_b_+c, j*nv_a_+a) = L2_ab_(k,j,a,c);
            Eigen::MatrixXd W_C1 = V_C1 * L2_C1;
            for(int i=0; i<no_a_; ++i) for(int a=0; a<nv_a_; ++a) for(int j=0; j<no_b_; ++j) for(int b=0; b<nv_b_; ++b) 
                W_res_AB(i*nv_a_+a, j*nv_b_+b) -= W_C1(i*nv_b_+b, j*nv_a_+a);

            Eigen::MatrixXd L2_C2(no_b_*nv_a_, no_a_*nv_b_);
            for(int k=0; k<no_b_; ++k) for(int c=0; c<nv_a_; ++c) for(int i=0; i<no_a_; ++i) for(int b=0; b<nv_b_; ++b) 
                L2_C2(k*nv_a_+c, i*nv_b_+b) = L2_ab_(i,k,c,b);
            Eigen::MatrixXd W_C2 = V_C2 * L2_C2;
            for(int i=0; i<no_a_; ++i) for(int a=0; a<nv_a_; ++a) for(int j=0; j<no_b_; ++j) for(int b=0; b<nv_b_; ++b) 
                W_res_AB(i*nv_a_+a, j*nv_b_+b) -= W_C2(j*nv_a_+a, i*nv_b_+b);

            for(int ia=0; ia<no_a_*nv_a_; ++ia) for(int jb=0; jb<no_b_*nv_b_; ++jb) {
                int i=ia/nv_a_; int a=ia%nv_a_; int j=jb/nv_b_; int b=jb%nv_b_;
                double D = ea(i) + eb(j) - ea(no_a_+a) - eb(no_b_+b);
                double L2_new = 1.0 * t2_3rd_ab_(i,j,a,b);
                if (std::abs(D) > 1e-12) L2_new -= W_res_AB(ia, jb) / D;
                L2_next_ab(i,j,a,b) = L2_new;
                double err = L2_new - L2_ab_(i,j,a,b); R_ab(i,j,a,b) = err; rms_error += err * err;
            }
        }

        rms_error = std::sqrt(rms_error / total_elements);
        if (rms_error < z_thresh) break;

        diis_aa.push(L2_next_aa, R_aa); L2_aa_ = diis_aa.extrapolate(no_a_, no_a_, nv_a_, nv_a_);
        if (no_b_ > 0 && nv_b_ > 0) {
            diis_bb.push(L2_next_bb, R_bb); L2_bb_ = diis_bb.extrapolate(no_b_, no_b_, nv_b_, nv_b_);
            diis_ab.push(L2_next_ab, R_ab); L2_ab_ = diis_ab.extrapolate(no_a_, no_b_, nv_a_, nv_b_);
        }
        iter++;
    }
}




void OMP3::build_opdm_alpha() {
    G_oo_alpha_ = Eigen::MatrixXd::Zero(no_a_, no_a_);
    G_vv_alpha_ = Eigen::MatrixXd::Zero(nv_a_, nv_a_);
    long va2 = nv_a_ * nv_a_;

    Eigen::MatrixXd T1_AA_o(no_a_, no_a_ * va2), T2_AA_o(no_a_, no_a_ * va2);
    for(int i=0; i<no_a_; ++i) for(int k=0; k<no_a_; ++k) for(int ab=0; ab<va2; ++ab) {
        int a = ab/nv_a_; int b = ab%nv_a_;
        T1_AA_o(i, k*va2+ab) = t2_aa_(i, k, a, b);
        T2_AA_o(i, k*va2+ab) = L2_aa_(i, k, a, b);
    }
    G_oo_alpha_ = -0.5 * (T1_AA_o * T1_AA_o.transpose());
    G_oo_alpha_ -= 0.5 * (T1_AA_o * T2_AA_o.transpose() + T2_AA_o * T1_AA_o.transpose());

    if (no_b_ > 0 && nv_b_ > 0) {
        Eigen::MatrixXd T1_AB_o(no_a_, no_b_ * nv_a_ * nv_b_), T2_AB_o(no_a_, no_b_ * nv_a_ * nv_b_);
        for(int i=0; i<no_a_; ++i) for(int k=0; k<no_b_; ++k) for(int a=0; a<nv_a_; ++a) for(int b=0; b<nv_b_; ++b) {
            int col = k*(nv_a_*nv_b_) + a*nv_b_ + b;
            T1_AB_o(i, col) = t2_ab_(i, k, a, b); T2_AB_o(i, col) = L2_ab_(i, k, a, b);
        }
        G_oo_alpha_ -= 1.0 * (T1_AB_o * T1_AB_o.transpose());
        G_oo_alpha_ -= 1.0 * (T1_AB_o * T2_AB_o.transpose() + T2_AB_o * T1_AB_o.transpose());
    }

    Eigen::MatrixXd T1_AA_v(nv_a_, no_a_ * no_a_ * nv_a_), T2_AA_v(nv_a_, no_a_ * no_a_ * nv_a_);
    for(int a=0; a<nv_a_; ++a) {
        int col = 0;
        for(int i=0; i<no_a_; ++i) for(int j=0; j<no_a_; ++j) for(int c=0; c<nv_a_; ++c) {
            T1_AA_v(a, col) = t2_aa_(i, j, a, c); T2_AA_v(a, col) = L2_aa_(i, j, a, c); col++;
        }
    }
    G_vv_alpha_ = 0.5 * (T1_AA_v * T1_AA_v.transpose());
    G_vv_alpha_ += 0.5 * (T1_AA_v * T2_AA_v.transpose() + T2_AA_v * T1_AA_v.transpose());

    if (no_b_ > 0 && nv_b_ > 0) {
        Eigen::MatrixXd T1_AB_v(nv_a_, no_a_ * no_b_ * nv_b_), T2_AB_v(nv_a_, no_a_ * no_b_ * nv_b_);
        for(int a=0; a<nv_a_; ++a) {
            int col = 0;
            for(int i=0; i<no_a_; ++i) for(int j=0; j<no_b_; ++j) for(int c=0; c<nv_b_; ++c) {
                T1_AB_v(a, col) = t2_ab_(i, j, a, c); T2_AB_v(a, col) = L2_ab_(i, j, a, c); col++;
            }
        }
        G_vv_alpha_ += 1.0 * (T1_AB_v * T1_AB_v.transpose());
        G_vv_alpha_ += 1.0 * (T1_AB_v * T2_AB_v.transpose() + T2_AB_v * T1_AB_v.transpose());
    }
}

void OMP3::build_opdm_beta() {
    G_oo_beta_ = Eigen::MatrixXd::Zero(no_b_, no_b_);
    G_vv_beta_ = Eigen::MatrixXd::Zero(nv_b_, nv_b_);
    if (no_b_ == 0 || nv_b_ == 0) return;
    long vb2 = nv_b_ * nv_b_;

    Eigen::MatrixXd T1_BB_o(no_b_, no_b_ * vb2), T2_BB_o(no_b_, no_b_ * vb2);
    for(int i=0; i<no_b_; ++i) for(int k=0; k<no_b_; ++k) for(int ab=0; ab<vb2; ++ab) {
        int a = ab/nv_b_; int b = ab%nv_b_;
        T1_BB_o(i, k*vb2+ab) = t2_bb_(i, k, a, b); T2_BB_o(i, k*vb2+ab) = L2_bb_(i, k, a, b);
    }
    G_oo_beta_ = -0.5 * (T1_BB_o * T1_BB_o.transpose());
    G_oo_beta_ -= 0.5 * (T1_BB_o * T2_BB_o.transpose() + T2_BB_o * T1_BB_o.transpose());

    Eigen::MatrixXd T1_BA_o(no_b_, no_a_ * nv_a_ * nv_b_), T2_BA_o(no_b_, no_a_ * nv_a_ * nv_b_);
    for(int i=0; i<no_b_; ++i) for(int k=0; k<no_a_; ++k) for(int a=0; a<nv_a_; ++a) for(int b=0; b<nv_b_; ++b) {
        int col = k*(nv_a_*nv_b_) + a*nv_b_ + b;
        T1_BA_o(i, col) = t2_ab_(k, i, a, b); T2_BA_o(i, col) = L2_ab_(k, i, a, b);
    }
    G_oo_beta_ -= 1.0 * (T1_BA_o * T1_BA_o.transpose());
    G_oo_beta_ -= 1.0 * (T1_BA_o * T2_BA_o.transpose() + T2_BA_o * T1_BA_o.transpose());

    Eigen::MatrixXd T1_BB_v(nv_b_, no_b_ * no_b_ * nv_b_), T2_BB_v(nv_b_, no_b_ * no_b_ * nv_b_);
    for(int a=0; a<nv_b_; ++a) {
        int col = 0;
        for(int i=0; i<no_b_; ++i) for(int j=0; j<no_b_; ++j) for(int c=0; c<nv_b_; ++c) {
            T1_BB_v(a, col) = t2_bb_(i, j, a, c); T2_BB_v(a, col) = L2_bb_(i, j, a, c); col++;
        }
    }
    G_vv_beta_ = 0.5 * (T1_BB_v * T1_BB_v.transpose());
    G_vv_beta_ += 0.5 * (T1_BB_v * T2_BB_v.transpose() + T2_BB_v * T1_BB_v.transpose());

    Eigen::MatrixXd T1_BA_v(nv_b_, no_a_ * no_b_ * nv_a_), T2_BA_v(nv_b_, no_a_ * no_b_ * nv_a_);
    for(int a=0; a<nv_b_; ++a) {
        int col = 0;
        for(int i=0; i<no_a_; ++i) for(int j=0; j<no_b_; ++j) for(int c=0; c<nv_a_; ++c) {
            T1_BA_v(a, col) = t2_ab_(i, j, c, a); T2_BA_v(a, col) = L2_ab_(i, j, c, a); col++;
        }
    }
    G_vv_beta_ += 1.0 * (T1_BA_v * T1_BA_v.transpose());
    G_vv_beta_ += 1.0 * (T1_BA_v * T2_BA_v.transpose() + T2_BA_v * T1_BA_v.transpose());
}




MP3Result OMP3::compute() {
    init_fast_integrals();
    std::string mode = (no_a_ == no_b_) ? "R" : "U";
    if(omp_get_thread_num() == 0) {
        std::cout << "\n========================================\n";
        std::cout << "  Iterative 2-RDM MP3 (" << mode << "-OMP3)   \n";
        std::cout << "========================================\n";
        std::cout << "Iter    E_Total (Ha)    E_Corr (Ha)     ||Grad||    Step\n";
        std::cout << "--------------------------------------------------------\n";
    }

    Eigen::MatrixXd C_a_last = scf_.C_alpha; Eigen::MatrixXd C_b_last = scf_.C_beta;
    Eigen::MatrixXd K_dir_a_last = Eigen::MatrixXd::Zero(nbf_, nbf_);
    Eigen::MatrixXd K_dir_b_last = Eigen::MatrixXd::Zero(nbf_, nbf_);
    double current_step = 0.5; int iter = 0; bool is_converged = false;
    double e_total_best = 1e99; double e_total_last = 1e99; double e_corr_last = 0.0; double grad_norm_last = 1.0;
    
    auto rotate_orbitals = [&](double scale, const Eigen::MatrixXd& Kdir_a, const Eigen::MatrixXd& Kdir_b) {
        Eigen::MatrixXd Ka = Kdir_a * scale; Eigen::MatrixXd Kb = Kdir_b * scale;
        for(int i=0; i<Ka.size(); ++i) {
            if (Ka(i) > 0.4) Ka(i) = 0.4; if (Ka(i) < -0.4) Ka(i) = -0.4;
            if (Kb(i) > 0.4) Kb(i) = 0.4; if (Kb(i) < -0.4) Kb(i) = -0.4;
        }
        scf_.C_alpha = C_a_last * (-Ka).exp();
        scf_.C_beta = (mode == "U") ? C_b_last * (-Kb).exp() : scf_.C_alpha;
        scf_.P_alpha = scf_.C_alpha.leftCols(no_a_) * scf_.C_alpha.leftCols(no_a_).transpose();
        scf_.P_beta  = scf_.C_beta.leftCols(no_b_)  * scf_.C_beta.leftCols(no_b_).transpose();
    };

    KappaDIIS diis_kappa_a, diis_kappa_b;

    while(iter < config_.max_iterations) {
        pseudocanonicalize();
        double e_mp2 = compute_mp2_energy();
        double e_mp3 = compute_mp3_correction();
        solve_zvector();
        
        Eigen::MatrixXd F_ao_a, F_ao_b;
        build_fock_fast(scf_.P_alpha, scf_.P_beta, F_ao_a, F_ao_b);
        
        double e_scf = 0.5 * (scf_.P_alpha.cwiseProduct(H_core_ + F_ao_a).sum() + scf_.P_beta.cwiseProduct(H_core_ + F_ao_b).sum()) + scf_.energy_total - 0.5*(scf_.P_alpha.cwiseProduct(H_core_+F_ao_a).sum()+scf_.P_beta.cwiseProduct(H_core_+F_ao_b).sum()); 
        
        e_scf = scf_.energy_total; 
        
        double e_corr = e_mp2 + e_mp3;
        double e_tot = e_scf + e_corr;
        double dE = (iter == 0) ? 1.0 : std::abs(e_total_last - e_tot); 
        
        if (e_tot < e_total_best) e_total_best = e_tot;
        e_total_last = e_tot; e_corr_last = e_corr;
        C_a_last = scf_.C_alpha; C_b_last = scf_.C_beta;
        
        build_opdm_alpha(); build_opdm_beta();

        Eigen::MatrixXd G_full_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
        G_full_a.block(0,0,no_a_,no_a_) = G_oo_alpha_; G_full_a.block(no_a_,no_a_,nv_a_,nv_a_) = G_vv_alpha_;
        Eigen::MatrixXd P_corr_a = scf_.C_alpha * G_full_a * scf_.C_alpha.transpose();
        
        Eigen::MatrixXd G_full_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
        if (no_b_ > 0) { G_full_b.block(0,0,no_b_,no_b_) = G_oo_beta_; G_full_b.block(no_b_,no_b_,nv_b_,nv_b_) = G_vv_beta_; }
        Eigen::MatrixXd P_corr_b = scf_.C_beta * G_full_b * scf_.C_beta.transpose();
        
        Eigen::MatrixXd F_gen_a, F_gen_b;
        build_fock_fast(scf_.P_alpha + P_corr_a, scf_.P_beta + P_corr_b, F_gen_a, F_gen_b);
        Eigen::MatrixXd F_mo_a = scf_.C_alpha.transpose() * F_gen_a * scf_.C_alpha;
        Eigen::MatrixXd F_mo_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
        if (no_b_ > 0) F_mo_b = scf_.C_beta.transpose() * F_gen_b * scf_.C_beta;

        
        if (no_a_ > 0 && nv_a_ > 0) {
            Eigen::MatrixXd L_nonsep_a = Eigen::MatrixXd::Zero(nv_a_, no_a_);
            const Eigen::MatrixXd& Cao = scf_.C_alpha.leftCols(no_a_); const Eigen::MatrixXd& Cav = scf_.C_alpha.rightCols(nv_a_);
            
            if (no_a_ >= 2 && nv_a_ >= 2) {
                
                auto V_vovv_aa = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cav, Cao, Cav, Cav, ints_);
                auto V_ooov_aa = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cao, Cav, ints_);
                
                Eigen::MatrixXd V_a_jbc(nv_a_, no_a_*nv_a_*nv_a_), T_i_jbc(no_a_, no_a_*nv_a_*nv_a_);
                Eigen::MatrixXd V_i_jkb(no_a_, no_a_*no_a_*nv_a_), T_a_jkb(nv_a_, no_a_*no_a_*nv_a_);
                
                #pragma omp parallel
                {
                    #pragma omp for collapse(3) nowait
                    for(int j=0; j<no_a_; ++j) for(int b=0; b<nv_a_; ++b) for(int c=0; c<nv_a_; ++c) {
                        int col = j*nv_a_*nv_a_ + b*nv_a_ + c;
                        for(int a=0; a<nv_a_; ++a) V_a_jbc(a, col) = V_vovv_aa(a,j,b,c);
                        for(int i=0; i<no_a_; ++i) T_i_jbc(i, col) = t2_aa_(i,j,b,c) + L2_aa_(i,j,b,c);
                    }
                    #pragma omp for collapse(3)
                    for(int j=0; j<no_a_; ++j) for(int k=0; k<no_a_; ++k) for(int b=0; b<nv_a_; ++b) {
                        int col = j*no_a_*nv_a_ + k*nv_a_ + b;
                        for(int i=0; i<no_a_; ++i) V_i_jkb(i, col) = V_ooov_aa(j,i,k,b);
                        for(int a=0; a<nv_a_; ++a) T_a_jkb(a, col) = t2_aa_(j,k,a,b) + L2_aa_(j,k,a,b);
                    }
                }
                L_nonsep_a += 0.5 * (V_a_jbc * T_i_jbc.transpose() - T_a_jkb * V_i_jkb.transpose());
            }
            
            if (no_b_ > 0 && nv_b_ > 0) {
                const Eigen::MatrixXd& Cbo = scf_.C_beta.leftCols(no_b_); const Eigen::MatrixXd& Cbv = scf_.C_beta.rightCols(nv_b_);
                auto V_vovv_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cav, Cbo, Cav, Cbv, ints_);
                auto V_ooov_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cbo, Cao, Cbv, ints_);
                
                Eigen::MatrixXd V_a_jbc_ab(nv_a_, no_b_*nv_a_*nv_b_), T_i_jbc_ab(no_a_, no_b_*nv_a_*nv_b_);
                Eigen::MatrixXd V_i_jkb_ab(no_a_, no_b_*no_a_*nv_b_), T_a_jkb_ab(nv_a_, no_b_*no_a_*nv_b_);
                
                #pragma omp parallel
                {
                    #pragma omp for collapse(3) nowait
                    for(int j=0; j<no_b_; ++j) for(int b=0; b<nv_a_; ++b) for(int c=0; c<nv_b_; ++c) {
                        int col = j*nv_a_*nv_b_ + b*nv_b_ + c;
                        for(int a=0; a<nv_a_; ++a) V_a_jbc_ab(a, col) = V_vovv_ab(a,j,b,c);
                        for(int i=0; i<no_a_; ++i) T_i_jbc_ab(i, col) = t2_ab_(i,j,b,c) + L2_ab_(i,j,b,c);
                    }
                    #pragma omp for collapse(3)
                    for(int j=0; j<no_b_; ++j) for(int k=0; k<no_a_; ++k) for(int b=0; b<nv_b_; ++b) {
                        int col = j*no_a_*nv_b_ + k*nv_b_ + b;
                        for(int i=0; i<no_a_; ++i) V_i_jkb_ab(i, col) = V_ooov_ab(k,i,j,b);
                        for(int a=0; a<nv_a_; ++a) T_a_jkb_ab(a, col) = t2_ab_(k,j,a,b) + L2_ab_(k,j,a,b);
                    }
                }
                L_nonsep_a += 1.0 * (V_a_jbc_ab * T_i_jbc_ab.transpose() - T_a_jkb_ab * V_i_jkb_ab.transpose());
            }
            Eigen::MatrixXd F_vo_a = F_mo_a.block(no_a_, 0, nv_a_, no_a_);
            Eigen::MatrixXd L_total_a = (G_vv_alpha_ * F_vo_a - F_vo_a * G_oo_alpha_) + L_nonsep_a;
            F_mo_a.block(no_a_, 0, nv_a_, no_a_) += L_total_a; F_mo_a.block(0, no_a_, no_a_, nv_a_) += L_total_a.transpose();
        }

        

        double grad_norm = 0.0; Eigen::MatrixXd K_dir_a = Eigen::MatrixXd::Zero(nbf_, nbf_); Eigen::MatrixXd K_dir_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
        double shift = 0.25;
        if (mode == "U") {
            Eigen::MatrixXd wa = 2.0 * F_mo_a.block(no_a_, 0, nv_a_, no_a_); Eigen::MatrixXd wb = 2.0 * F_mo_b.block(no_b_, 0, nv_b_, no_b_);
            grad_norm = std::sqrt(wa.squaredNorm() + wb.squaredNorm());
            for(int a=0; a<nv_a_; ++a) for(int i=0; i<no_a_; ++i) {
                double delta = scf_.orbital_energies_alpha(no_a_+a) - scf_.orbital_energies_alpha(i);
                double denom = (delta >= 0) ? (delta + shift) : (delta - shift); K_dir_a(no_a_+a, i) = wa(a,i)/denom; K_dir_a(i, no_a_+a) = -wa(a,i)/denom;
            }
        } else {
            Eigen::MatrixXd F_uni = Eigen::MatrixXd::Zero(nbf_, nbf_);
            F_uni.block(0, no_a_, no_b_, nv_a_) = 0.5 * (F_mo_a.block(0, no_a_, no_b_, nv_a_) + F_mo_b.block(0, no_a_, no_b_, nv_a_));
            for(int i=0; i<nbf_; ++i) for(int j=i+1; j<nbf_; ++j) {
                double val = 2.0 * F_uni(i, j);
                if (std::abs(val) > 1e-12) { grad_norm += val * val; double delta = scf_.orbital_energies_alpha(j) - scf_.orbital_energies_alpha(i); double denom = (delta >= 0) ? (delta + shift) : (delta - shift); K_dir_a(j, i) = val/denom; K_dir_a(i, j) = -val/denom; }
            }
            grad_norm = std::sqrt(grad_norm); K_dir_b = K_dir_a;
        }

        diis_kappa_a.push(K_dir_a, F_mo_a.block(no_a_, 0, nv_a_, no_a_));
        if (mode == "U") diis_kappa_b.push(K_dir_b, F_mo_b.block(no_b_, 0, nv_b_, no_b_));
        
        Eigen::MatrixXd K_dir_raw_a = K_dir_a;
        if (iter >= 2) { K_dir_a = diis_kappa_a.extrapolate(); if (mode == "U") K_dir_b = diis_kappa_b.extrapolate(); else K_dir_b = K_dir_a; }
        
        auto clamp_direction = [](Eigen::MatrixXd& K) { double max_val = K.cwiseAbs().maxCoeff(); if (max_val > 0.05) K *= (0.05 / max_val); };
        clamp_direction(K_dir_a); if (mode == "U") clamp_direction(K_dir_b); else K_dir_b = K_dir_a;
        K_dir_a_last = K_dir_a; K_dir_b_last = K_dir_b;

        if(omp_get_thread_num() == 0) std::cout << std::setw(4) << iter << "    " << std::fixed << std::setprecision(8) << e_tot << "    " << std::setprecision(8) << e_corr << "    " << std::scientific << std::setprecision(2) << grad_norm << "   " << std::fixed << std::setprecision(4) << current_step << "\n";
        
        if (grad_norm < config_.energy_threshold) { is_converged = true; break; }
        else if (iter > 1 && std::abs(dE) < 1e-11 && current_step < 0.05) { is_converged = true; break; }

        rotate_orbitals(current_step, K_dir_a_last, K_dir_b_last);
        iter++;
    }

    MP3Result result;
    result.e_total = e_total_last;
    result.e_corr_total = e_corr_last;
    result.converged = is_converged;
    result.iterations = iter;
    return result;
}
} 
