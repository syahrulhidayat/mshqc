/**
 * @file src/mp3/mp3.cc
 * @brief Unified MP3 Implementation Powered by Native TBLIS
 */

#include "mshqc/mp3/mp3.h"
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
    
    
    Eigen::Tensor<double, 4> W(no_a_, no_a_, nv_a_, nv_a_);
    W.setZero();
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
    res.e3_aa = 0.0;
    res.e3_ab = 0.0; 
    res.e_mp3 = e_mp3;
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

struct OrbitalLBFGS {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int m_max = 6;
    std::vector<Eigen::VectorXd> s_hist;
    std::vector<Eigen::VectorXd> y_hist;
    std::vector<double> rho_hist;
    Eigen::VectorXd g_prev;
    Eigen::VectorXd s_prev; 
    bool is_first = true;

    void reset() { s_hist.clear(); y_hist.clear(); rho_hist.clear(); is_first = true; }

    Eigen::VectorXd get_direction(const Eigen::VectorXd& g_curr, const Eigen::VectorXd& diag_H) {
        if (is_first) {
            g_prev = g_curr; 
            is_first = false;
            s_prev = -g_curr.cwiseQuotient(diag_H); 
            return s_prev; 
        }
        Eigen::VectorXd y = g_curr - g_prev;
        Eigen::VectorXd s = s_prev; 
        double ys = y.dot(s);
        Eigen::VectorXd Bs = s.cwiseProduct(diag_H); 
        double sBs = s.dot(Bs);
        
        double theta = 1.0;
        if (ys < 0.2 * sBs) theta = (0.8 * sBs) / (sBs - ys);
        
        Eigen::VectorXd y_mod = theta * y + (1.0 - theta) * Bs;
        double ys_mod = y_mod.dot(s);
        
        if (ys_mod > 1e-12) { 
            if ((int)s_hist.size() >= m_max) {
                s_hist.erase(s_hist.begin()); y_hist.erase(y_hist.begin()); rho_hist.erase(rho_hist.begin());
            }
            s_hist.push_back(s); y_hist.push_back(y_mod); rho_hist.push_back(1.0 / ys_mod);
        }
        
        g_prev = g_curr;
        Eigen::VectorXd q = g_curr;
        int k = s_hist.size();
        std::vector<double> alpha(k);
        for (int i = k - 1; i >= 0; --i) {
            alpha[i] = rho_hist[i] * s_hist[i].dot(q);
            q -= alpha[i] * y_hist[i];
        }
        
        Eigen::VectorXd r = q.cwiseQuotient(diag_H);
        for (int i = 0; i < k; ++i) {
            double beta = rho_hist[i] * y_hist[i].dot(r);
            r += s_hist[i] * (alpha[i] - beta);
        }
        
        s_prev = -r; 
        return s_prev; 
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
    const Eigen::MatrixXd& Cao = scf_.C_alpha.leftCols(no_a_); 
    const Eigen::MatrixXd& Cav = scf_.C_alpha.rightCols(nv_a_);
    const auto& ea = scf_.orbital_energies_alpha; 
    double e_sum = 0.0;
    
    auto g_aa = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cav, Cao, Cav, ints_);
    double e_aa = 0.0;
    #pragma omp parallel for collapse(4) reduction(+:e_aa)
    for(int i=0; i<no_a_; ++i) for(int j=0; j<no_a_; ++j) 
        for(int a=0; a<nv_a_; ++a) for(int b=0; b<nv_a_; ++b) {
            double D = ea(i) + ea(j) - ea(no_a_+a) - ea(no_a_+b);
            double val = g_aa(i,a,j,b) - g_aa(i,b,j,a);
            
            if (std::abs(D) > 1e-12) {
                t2_aa_(i,j,a,b) = val / D; 
                e_aa += t2_aa_(i,j,a,b) * val;
            } else { t2_aa_(i,j,a,b) = 0.0; }
        }
    e_sum += 0.25 * e_aa;

    if (no_b_ > 0 && nv_b_ > 0) {
        const Eigen::MatrixXd& Cbo = scf_.C_beta.leftCols(no_b_); 
        const Eigen::MatrixXd& Cbv = scf_.C_beta.rightCols(nv_b_);
        const auto& eb = scf_.orbital_energies_beta;
        
        auto g_bb = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbv, Cbo, Cbv, ints_);
        double e_bb = 0.0;
        #pragma omp parallel for collapse(4) reduction(+:e_bb)
        for(int i=0; i<no_b_; ++i) for(int j=0; j<no_b_; ++j)
            for(int a=0; a<nv_b_; ++a) for(int b=0; b<nv_b_; ++b) {
                double D = eb(i) + eb(j) - eb(no_b_+a) - eb(no_b_+b);
                double val = g_bb(i,a,j,b) - g_bb(i,b,j,a);
                if (std::abs(D) > 1e-12) {
                    t2_bb_(i,j,a,b) = val / D; 
                    e_bb += t2_bb_(i,j,a,b) * val;
                } else { t2_bb_(i,j,a,b) = 0.0; }
            }
        e_sum += 0.25 * e_bb;

        auto g_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cav, Cbo, Cbv, ints_);
        double e_ab = 0.0;
        #pragma omp parallel for collapse(4) reduction(+:e_ab)
        for(int i=0; i<no_a_; ++i) for(int j=0; j<no_b_; ++j)
            for(int a=0; a<nv_a_; ++a) for(int b=0; b<nv_b_; ++b) {
                double D = ea(i) + eb(j) - ea(no_a_+a) - eb(no_b_+b);
                double val = g_ab(i,a,j,b);
                if (std::abs(D) > 1e-12) {
                    t2_ab_(i,j,a,b) = val / D; 
                    e_ab += t2_ab_(i,j,a,b) * val;
                } else { t2_ab_(i,j,a,b) = 0.0; }
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

    const Eigen::MatrixXd& Cao = scf_.C_alpha.leftCols(no_a_); 
    const Eigen::MatrixXd& Cav = scf_.C_alpha.rightCols(nv_a_);
    int nbf_ = scf_.C_alpha.rows();
    const auto& ea = scf_.orbital_energies_alpha;
    double e3_aa = 0.0, e3_bb = 0.0, e3_ab = 0.0;

    TBLIS_VIEW_4D(t_Taa, t2_aa_, no_a_, no_a_, nv_a_, nv_a_);

    
    
    
    Eigen::MatrixXd B_vv_a, B_vv_b;
    if (config_.use_df && n_aux_ > 0) {
        B_vv_a.setZero(nv_a_ * nv_a_, n_aux_);
        Eigen::Map<const Eigen::MatrixXd> L_flat(scf_.L_mat.data(), nbf_, nbf_ * n_aux_);
        Eigen::MatrixXd X_a = Cav.transpose() * L_flat;
        #pragma omp parallel for
        for (int P = 0; P < n_aux_; ++P) {
            Eigen::Map<Eigen::MatrixXd> X_P(X_a.data() + P * nv_a_ * nbf_, nv_a_, nbf_);
            Eigen::MatrixXd B_MO = X_P * Cav;
            std::copy(B_MO.data(), B_MO.data() + nv_a_ * nv_a_, B_vv_a.col(P).data());
        }

        if (no_b_ > 0 && nv_b_ > 0) {
            B_vv_b.setZero(nv_b_ * nv_b_, n_aux_);
            Eigen::MatrixXd X_b = scf_.C_beta.rightCols(nv_b_).transpose() * L_flat;
            #pragma omp parallel for
            for (int P = 0; P < n_aux_; ++P) {
                Eigen::Map<Eigen::MatrixXd> X_P(X_b.data() + P * nv_b_ * nbf_, nv_b_, nbf_);
                Eigen::MatrixXd B_MO = X_P * scf_.C_beta.rightCols(nv_b_);
                std::copy(B_MO.data(), B_MO.data() + nv_b_ * nv_b_, B_vv_b.col(P).data());
            }
        }
    }

    
    
    
    {
        Eigen::Tensor<double, 4> Waa_ladder(no_a_, no_a_, nv_a_, nv_a_); Waa_ladder.setZero();
        Eigen::Tensor<double, 4> Waa_ring(no_a_, no_a_, nv_a_, nv_a_); Waa_ring.setZero();
        TBLIS_VIEW_4D(t_Waa_ladder, Waa_ladder, no_a_, no_a_, nv_a_, nv_a_);
        TBLIS_VIEW_4D(t_Waa_ring, Waa_ring, no_a_, no_a_, nv_a_, nv_a_);

        
        if (config_.use_df && n_aux_ > 0) {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < no_a_; ++i) {
                for (int j = 0; j < no_a_; ++j) {
                    Eigen::MatrixXd T_ij(nv_a_, nv_a_);
                    for(int a=0; a<nv_a_; ++a) for(int b=0; b<nv_a_; ++b) T_ij(a,b) = t2_aa_(i,j,a,b);
                    
                    Eigen::MatrixXd W_ij = Eigen::MatrixXd::Zero(nv_a_, nv_a_);
                    for (int P = 0; P < n_aux_; ++P) {
                        Eigen::Map<const Eigen::MatrixXd> B_P(B_vv_a.col(P).data(), nv_a_, nv_a_);
                        W_ij.noalias() += B_P * T_ij * B_P; 
                    }
                    for(int a=0; a<nv_a_; ++a) for(int b=0; b<nv_a_; ++b) {
                        Waa_ladder(i,j,a,b) += 0.5 * (W_ij(a,b) - W_ij(b,a)); 
                    }
                }
            }
        } else {
            auto V_vvvv = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cav, Cav, Cav, Cav, ints_);
            TBLIS_VIEW_4D(t_Vvvvv, V_vvvv, nv_a_, nv_a_, nv_a_, nv_a_);
            tblis::mult<double>(0.5, t_Taa, "ijef", t_Vvvvv, "eafb", 1.0, t_Waa_ladder, "ijab");
            tblis::mult<double>(-0.5, t_Taa, "ijef", t_Vvvvv, "ebfa", 1.0, t_Waa_ladder, "ijab");
        }

        
        auto V_oooo = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cao, Cao, ints_);
        TBLIS_VIEW_4D(t_Voooo, V_oooo, no_a_, no_a_, no_a_, no_a_);
        tblis::mult<double>(0.5, t_Taa, "mnab", t_Voooo, "minj", 1.0, t_Waa_ladder, "ijab");
        tblis::mult<double>(-0.5, t_Taa, "mnab", t_Voooo, "mjni", 1.0, t_Waa_ladder, "ijab");

        
        auto V_ovov = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cav, Cao, Cav, ints_);
        auto V_oovv = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cav, Cav, ints_);
        TBLIS_VIEW_4D(t_Vovov, V_ovov, no_a_, nv_a_, no_a_, nv_a_);
        TBLIS_VIEW_4D(t_Voovv, V_oovv, no_a_, no_a_, nv_a_, nv_a_);

        tblis::mult<double>(1.0,  t_Vovov, "iakc", t_Taa, "kjcb", 1.0, t_Waa_ring, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikac", t_Taa, "kjcb", 1.0, t_Waa_ring, "ijab");
        

        if (no_b_ > 0 && nv_b_ > 0) {
            auto V_ovov_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cav, scf_.C_beta.leftCols(no_b_), scf_.C_beta.rightCols(nv_b_), ints_);
            TBLIS_VIEW_4D(t_Vovov_ab, V_ovov_ab, no_a_, nv_a_, no_b_, nv_b_);
            TBLIS_VIEW_4D(t_Tab, t2_ab_, no_a_, no_b_, nv_a_, nv_b_);
            tblis::mult<double>(1.0, t_Vovov_ab, "iakc", t_Tab, "jkbc", 1.0, t_Waa_ring, "ijab");
        }

        
        #pragma omp parallel for collapse(4) reduction(+:e3_aa)
        for(int i=0; i<no_a_; ++i) {
            for(int j=0; j<no_a_; ++j) {
                for(int a=0; a<nv_a_; ++a) {
                    for(int b=0; b<nv_a_; ++b) {
                        double r_asym = Waa_ring(i,j,a,b) - Waa_ring(j,i,a,b) - Waa_ring(i,j,b,a) + Waa_ring(j,i,b,a);
                        double w_tot = Waa_ladder(i,j,a,b) + r_asym;
                        
                        double D = ea(i) + ea(j) - ea(no_a_+a) - ea(no_a_+b);
                        if (std::abs(D) > 1e-12) {
                            t2_3rd_aa_(i,j,a,b) = w_tot / D;
                            e3_aa += 0.25 * t2_aa_(i,j,a,b) * w_tot; 
                        } else { 
                            t2_3rd_aa_(i,j,a,b) = 0.0; 
                        }
                    }
                }
            }
        }
    }

    
    
    
    if (no_b_ > 0 && nv_b_ > 0) {
        const Eigen::MatrixXd& Cbo = scf_.C_beta.leftCols(no_b_); 
        const Eigen::MatrixXd& Cbv = scf_.C_beta.rightCols(nv_b_);
        const auto& eb = scf_.orbital_energies_beta;
        
        TBLIS_VIEW_4D(t_Tbb, t2_bb_, no_b_, no_b_, nv_b_, nv_b_);
        TBLIS_VIEW_4D(t_Tab, t2_ab_, no_a_, no_b_, nv_a_, nv_b_);

        
        {
            Eigen::Tensor<double, 4> Wbb_ladder(no_b_, no_b_, nv_b_, nv_b_); Wbb_ladder.setZero();
            Eigen::Tensor<double, 4> Wbb_ring(no_b_, no_b_, nv_b_, nv_b_); Wbb_ring.setZero();
            TBLIS_VIEW_4D(t_Wbb_ladder, Wbb_ladder, no_b_, no_b_, nv_b_, nv_b_);
            TBLIS_VIEW_4D(t_Wbb_ring, Wbb_ring, no_b_, no_b_, nv_b_, nv_b_);

            
            if (config_.use_df && n_aux_ > 0) {
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < no_b_; ++i) {
                    for (int j = 0; j < no_b_; ++j) {
                        Eigen::MatrixXd T_ij(nv_b_, nv_b_);
                        for(int a=0; a<nv_b_; ++a) for(int b=0; b<nv_b_; ++b) T_ij(a,b) = t2_bb_(i,j,a,b);
                        
                        Eigen::MatrixXd W_ij = Eigen::MatrixXd::Zero(nv_b_, nv_b_);
                        for (int P = 0; P < n_aux_; ++P) {
                            Eigen::Map<const Eigen::MatrixXd> B_P(B_vv_b.col(P).data(), nv_b_, nv_b_);
                            W_ij.noalias() += B_P * T_ij * B_P; 
                        }
                        for(int a=0; a<nv_b_; ++a) for(int b=0; b<nv_b_; ++b) {
                            Wbb_ladder(i,j,a,b) += 0.5 * (W_ij(a,b) - W_ij(b,a)); 
                        }
                    }
                }
            } else {
                auto V_vvvv = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbv, Cbv, Cbv, Cbv, ints_);
                TBLIS_VIEW_4D(t_Vvvvv, V_vvvv, nv_b_, nv_b_, nv_b_, nv_b_);
                tblis::mult<double>(0.5, t_Tbb, "ijef", t_Vvvvv, "eafb", 1.0, t_Wbb_ladder, "ijab");
                tblis::mult<double>(-0.5, t_Tbb, "ijef", t_Vvvvv, "ebfa", 1.0, t_Wbb_ladder, "ijab");
            }

            
            auto V_oooo = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbo, Cbo, Cbo, ints_);
            TBLIS_VIEW_4D(t_Voooo, V_oooo, no_b_, no_b_, no_b_, no_b_);
            tblis::mult<double>(0.5, t_Tbb, "mnab", t_Voooo, "minj", 1.0, t_Wbb_ladder, "ijab");
            tblis::mult<double>(-0.5, t_Tbb, "mnab", t_Voooo, "mjni", 1.0, t_Wbb_ladder, "ijab");

            
            auto V_ovov = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbv, Cbo, Cbv, ints_);
            auto V_oovv = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbo, Cbv, Cbv, ints_);
            TBLIS_VIEW_4D(t_Vovov, V_ovov, no_b_, nv_b_, no_b_, nv_b_);
            TBLIS_VIEW_4D(t_Voovv, V_oovv, no_b_, no_b_, nv_b_, nv_b_);

            tblis::mult<double>(1.0,  t_Vovov, "iakc", t_Tbb, "kjcb", 1.0, t_Wbb_ring, "ijab");
            tblis::mult<double>(-1.0, t_Voovv, "ikac", t_Tbb, "kjcb", 1.0, t_Wbb_ring, "ijab");

            auto V_ovov_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cav, Cbo, Cbv, ints_);
            TBLIS_VIEW_4D(t_Vovov_ab, V_ovov_ab, no_a_, nv_a_, no_b_, nv_b_);
            tblis::mult<double>(1.0, t_Vovov_ab, "kcia", t_Tab, "kjcb", 1.0, t_Wbb_ring, "ijab");

            
            #pragma omp parallel for collapse(4) reduction(+:e3_bb)
            for(int i=0; i<no_b_; ++i) {
                for(int j=0; j<no_b_; ++j) {
                    for(int a=0; a<nv_b_; ++a) {
                        for(int b=0; b<nv_b_; ++b) {
                            double r_asym = Wbb_ring(i,j,a,b) - Wbb_ring(j,i,a,b) - Wbb_ring(i,j,b,a) + Wbb_ring(j,i,b,a);
                            double w_tot = Wbb_ladder(i,j,a,b) + r_asym;
                            
                            double D = eb(i) + eb(j) - eb(no_b_+a) - eb(no_b_+b);
                            if (std::abs(D) > 1e-12) {
                                t2_3rd_bb_(i,j,a,b) = w_tot / D;
                                e3_bb += 0.25 * t2_bb_(i,j,a,b) * w_tot;
                            } else { t2_3rd_bb_(i,j,a,b) = 0.0; }
                        }
                    }
                }
            }
        }

        
        {
            Eigen::Tensor<double, 4> Wab_ladder(no_a_, no_b_, nv_a_, nv_b_); Wab_ladder.setZero();
            Eigen::Tensor<double, 4> Wab_ring(no_a_, no_b_, nv_a_, nv_b_); Wab_ring.setZero();
            TBLIS_VIEW_4D(t_Wab_ladder, Wab_ladder, no_a_, no_b_, nv_a_, nv_b_);
            TBLIS_VIEW_4D(t_Wab_ring, Wab_ring, no_a_, no_b_, nv_a_, nv_b_);

            
            if (config_.use_df && n_aux_ > 0) {
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < no_a_; ++i) {
                    for (int j = 0; j < no_b_; ++j) {
                        Eigen::MatrixXd T_ij(nv_a_, nv_b_);
                        for(int a=0; a<nv_a_; ++a) for(int b=0; b<nv_b_; ++b) T_ij(a,b) = t2_ab_(i,j,a,b);
                        
                        Eigen::MatrixXd W_ij = Eigen::MatrixXd::Zero(nv_a_, nv_b_);
                        for (int P = 0; P < n_aux_; ++P) {
                            Eigen::Map<const Eigen::MatrixXd> B_Pa(B_vv_a.col(P).data(), nv_a_, nv_a_);
                            Eigen::Map<const Eigen::MatrixXd> B_Pb(B_vv_b.col(P).data(), nv_b_, nv_b_);
                            W_ij.noalias() += B_Pa * T_ij * B_Pb; 
                        }
                        for(int a=0; a<nv_a_; ++a) for(int b=0; b<nv_b_; ++b) {
                            
                            Wab_ladder(i,j,a,b) += W_ij(a,b); 
                        }
                    }
                }
            } else {
                auto V_vvvv_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cav, Cav, Cbv, Cbv, ints_);
                TBLIS_VIEW_4D(t_Vvvvv_ab, V_vvvv_ab, nv_a_, nv_a_, nv_b_, nv_b_);
                tblis::mult<double>(1.0, t_Tab, "ijef", t_Vvvvv_ab, "eafb", 1.0, t_Wab_ladder, "ijab");
            }

            
            auto V_oooo_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cbo, Cbo, ints_);
            TBLIS_VIEW_4D(t_Voooo_ab, V_oooo_ab, no_a_, no_a_, no_b_, no_b_);
            tblis::mult<double>(1.0, t_Tab, "mnab", t_Voooo_ab, "minj", 1.0, t_Wab_ladder, "ijab");

            
            auto V_ovov_aa = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cav, Cao, Cav, ints_);
            auto V_oovv_aa = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cav, Cav, ints_);
            TBLIS_VIEW_4D(t_Vovov_aa, V_ovov_aa, no_a_, nv_a_, no_a_, nv_a_);
            TBLIS_VIEW_4D(t_Voovv_aa, V_oovv_aa, no_a_, no_a_, nv_a_, nv_a_);
            
            tblis::mult<double>(1.0,  t_Vovov_aa, "iakc", t_Tab, "kjcb", 1.0, t_Wab_ring, "ijab");
            tblis::mult<double>(-1.0, t_Voovv_aa, "ikac", t_Tab, "kjcb", 1.0, t_Wab_ring, "ijab");

            auto V_ovov_bb = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbv, Cbo, Cbv, ints_);
            auto V_oovv_bb = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbo, Cbv, Cbv, ints_);
            TBLIS_VIEW_4D(t_Vovov_bb, V_ovov_bb, no_b_, nv_b_, no_b_, nv_b_);
            TBLIS_VIEW_4D(t_Voovv_bb, V_oovv_bb, no_b_, no_b_, nv_b_, nv_b_);

            tblis::mult<double>(1.0,  t_Tab, "ikac", t_Vovov_bb, "kcjb", 1.0, t_Wab_ring, "ijab"); 
            tblis::mult<double>(-1.0, t_Tab, "ikac", t_Voovv_bb, "kjcb", 1.0, t_Wab_ring, "ijab"); 

            auto V_ovov_ab = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cav, Cbo, Cbv, ints_);
            TBLIS_VIEW_4D(t_Vovov_ab, V_ovov_ab, no_a_, nv_a_, no_b_, nv_b_);

            tblis::mult<double>(1.0,  t_Taa, "ikac", t_Vovov_ab, "kcjb", 1.0, t_Wab_ring, "ijab");
            tblis::mult<double>(1.0,  t_Vovov_ab, "iakc", t_Tbb, "kjcb", 1.0, t_Wab_ring, "ijab");

            auto V_oovv_ab_ex = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cao, Cao, Cbv, Cbv, ints_);
            auto V_oovv_ba_ex = ERITransformer::get_mo_tensor(config_.use_df, n_aux_, Cbo, Cbo, Cav, Cav, ints_);
            TBLIS_VIEW_4D(t_Voovv_ab_ex, V_oovv_ab_ex, no_a_, no_a_, nv_b_, nv_b_);
            TBLIS_VIEW_4D(t_Voovv_ba_ex, V_oovv_ba_ex, no_b_, no_b_, nv_a_, nv_a_);

            tblis::mult<double>(-1.0, t_Voovv_ab_ex, "ikbc", t_Tab, "kjac", 1.0, t_Wab_ring, "ijab");
            tblis::mult<double>(-1.0, t_Tab, "ikcb", t_Voovv_ba_ex, "jkac", 1.0, t_Wab_ring, "ijab");
            
            
            #pragma omp parallel for collapse(4) reduction(+:e3_ab)
            for(int i=0; i<no_a_; ++i) {
                for(int j=0; j<no_b_; ++j) {
                    for(int a=0; a<nv_a_; ++a) {
                        for(int b=0; b<nv_b_; ++b) {
                            double w_tot = Wab_ladder(i,j,a,b) + Wab_ring(i,j,a,b); 
                            double D = ea(i) + eb(j) - ea(no_a_+a) - eb(no_b_+b);
                            if (std::abs(D) > 1e-12) {
                                t2_3rd_ab_(i,j,a,b) = w_tot / D;
                                e3_ab += 1.0 * t2_ab_(i,j,a,b) * w_tot;
                            } else { 
                                t2_3rd_ab_(i,j,a,b) = 0.0; 
                            }
                        }
                    }
                }
            }
        }
    }
    
    return e3_aa + e3_bb + e3_ab;
}






void OMP3::build_opdm_alpha() {
    G_oo_alpha_ = Eigen::MatrixXd::Zero(no_a_, no_a_);
    G_vv_alpha_ = Eigen::MatrixXd::Zero(nv_a_, nv_a_);
    long va2 = nv_a_ * nv_a_;

    Eigen::MatrixXd T_tot_AA_o(no_a_, no_a_ * va2);
    for(int i=0; i<no_a_; ++i) for(int k=0; k<no_a_; ++k) for(int ab=0; ab<va2; ++ab) {
        int a = ab/nv_a_; int b = ab%nv_a_;
        T_tot_AA_o(i, k*va2+ab) = t2_aa_(i, k, a, b) + L2_aa_(i, k, a, b);
    }
    G_oo_alpha_ = -0.5 * (T_tot_AA_o * T_tot_AA_o.transpose());

    if (no_b_ > 0 && nv_b_ > 0) {
        Eigen::MatrixXd T_tot_AB_o(no_a_, no_b_ * nv_a_ * nv_b_);
        for(int i=0; i<no_a_; ++i) for(int k=0; k<no_b_; ++k) for(int a=0; a<nv_a_; ++a) for(int b=0; b<nv_b_; ++b) {
            int col = k*(nv_a_*nv_b_) + a*nv_b_ + b;
            T_tot_AB_o(i, col) = t2_ab_(i, k, a, b) + L2_ab_(i, k, a, b);
        }
        G_oo_alpha_ -= 1.0 * (T_tot_AB_o * T_tot_AB_o.transpose());
    }

    Eigen::MatrixXd T_tot_AA_v(nv_a_, no_a_ * no_a_ * nv_a_);
    for(int a=0; a<nv_a_; ++a) {
        int col = 0;
        for(int i=0; i<no_a_; ++i) for(int j=0; j<no_a_; ++j) for(int c=0; c<nv_a_; ++c) {
            T_tot_AA_v(a, col) = t2_aa_(i, j, a, c) + L2_aa_(i, j, a, c); col++;
        }
    }
    G_vv_alpha_ = 0.5 * (T_tot_AA_v * T_tot_AA_v.transpose());

    if (no_b_ > 0 && nv_b_ > 0) {
        Eigen::MatrixXd T_tot_AB_v(nv_a_, no_a_ * no_b_ * nv_b_);
        for(int a=0; a<nv_a_; ++a) {
            int col = 0;
            for(int i=0; i<no_a_; ++i) for(int j=0; j<no_b_; ++j) for(int c=0; c<nv_b_; ++c) {
                T_tot_AB_v(a, col) = t2_ab_(i, j, a, c) + L2_ab_(i, j, a, c); col++;
            }
        }
        G_vv_alpha_ += 1.0 * (T_tot_AB_v * T_tot_AB_v.transpose());
    }
}

void OMP3::build_opdm_beta() {
    G_oo_beta_ = Eigen::MatrixXd::Zero(no_b_, no_b_);
    G_vv_beta_ = Eigen::MatrixXd::Zero(nv_b_, nv_b_);
    if (no_b_ == 0 || nv_b_ == 0) return;
    long vb2 = nv_b_ * nv_b_;

    Eigen::MatrixXd T_tot_BB_o(no_b_, no_b_ * vb2);
    for(int i=0; i<no_b_; ++i) for(int k=0; k<no_b_; ++k) for(int ab=0; ab<vb2; ++ab) {
        int a = ab/nv_b_; int b = ab%nv_b_;
        T_tot_BB_o(i, k*vb2+ab) = t2_bb_(i, k, a, b) + L2_bb_(i, k, a, b);
    }
    G_oo_beta_ = -0.5 * (T_tot_BB_o * T_tot_BB_o.transpose());

    Eigen::MatrixXd T_tot_BA_o(no_b_, no_a_ * nv_a_ * nv_b_);
    for(int i=0; i<no_b_; ++i) for(int k=0; k<no_a_; ++k) for(int a=0; a<nv_a_; ++a) for(int b=0; b<nv_b_; ++b) {
        int col = k*(nv_a_*nv_b_) + a*nv_b_ + b;
        T_tot_BA_o(i, col) = t2_ab_(k, i, a, b) + L2_ab_(k, i, a, b);
    }
    G_oo_beta_ -= 1.0 * (T_tot_BA_o * T_tot_BA_o.transpose());

    Eigen::MatrixXd T_tot_BB_v(nv_b_, no_b_ * no_b_ * nv_b_);
    for(int a=0; a<nv_b_; ++a) {
        int col = 0;
        for(int i=0; i<no_b_; ++i) for(int j=0; j<no_b_; ++j) for(int c=0; c<nv_b_; ++c) {
            T_tot_BB_v(a, col) = t2_bb_(i, j, a, c) + L2_bb_(i, j, a, c); col++;
        }
    }
    G_vv_beta_ = 0.5 * (T_tot_BB_v * T_tot_BB_v.transpose());

    Eigen::MatrixXd T_tot_BA_v(nv_b_, no_a_ * no_b_ * nv_a_);
    for(int a=0; a<nv_b_; ++a) {
        int col = 0;
        for(int i=0; i<no_a_; ++i) for(int j=0; j<no_b_; ++j) for(int c=0; c<nv_a_; ++c) {
            T_tot_BA_v(a, col) = t2_ab_(i, j, c, a) + L2_ab_(i, j, c, a); col++;
        }
    }
    G_vv_beta_ += 1.0 * (T_tot_BA_v * T_tot_BA_v.transpose());
}

MP3Result OMP3::compute() {
    init_fast_integrals();
    std::string mode = (no_a_ == no_b_) ? "R" : "U";

    if (mode == "R" && scf_.C_beta.size() == 0) {
        scf_.C_beta = scf_.C_alpha;
        scf_.P_beta = scf_.P_alpha;
        scf_.orbital_energies_beta = scf_.orbital_energies_alpha;
    }

    if(omp_get_thread_num() == 0) {
        std::cout << "\n========================================================\n";
        std::cout << "      Orbital-Optimized MP3 (" << mode << "-OMP3)\n";
        std::cout << "      Trust-Region SOSCF (Exact Non-Iterative Z-Vector)\n";
        std::cout << "========================================================\n";
    }

    Eigen::MatrixXd F_ao_a_init, F_ao_b_init;
    build_fock_fast(scf_.P_alpha, scf_.P_beta, F_ao_a_init, F_ao_b_init);
    double e_elec_init = 0.5 * (scf_.P_alpha.cwiseProduct(H_core_ + F_ao_a_init).sum() + 
                                scf_.P_beta.cwiseProduct(H_core_ + F_ao_b_init).sum());
    double e_nuc = scf_.energy_total - e_elec_init;

    Eigen::MatrixXd C_a_current_ = scf_.C_alpha;
    Eigen::MatrixXd C_b_current_ = scf_.C_beta;
    Eigen::MatrixXd C_a_last = scf_.C_alpha; 
    Eigen::MatrixXd C_b_last = scf_.C_beta;
    
    double e_total_best = 1e99; double e_corr_best = 0.0; double e_total_last = 1e99;
    int macro_iter = 0; bool is_converged = false;
    double grad_norm = 1.0;
    
    double current_step = 1.0; 
    Eigen::VectorXd orbital_gradient_;

    
    int dim_a = nv_a_ * no_a_;
    int dim_b = (mode == "U" && no_b_ > 0) ? (nv_b_ * no_b_) : 0;
    int n_params = dim_a + dim_b;
    Eigen::VectorXd last_actual_step = Eigen::VectorXd::Zero(n_params);

    while (macro_iter < config_.max_iterations) {
        scf_.C_alpha = C_a_current_; scf_.C_beta = C_b_current_;
        
        pseudocanonicalize();
        
        
        C_a_current_ = scf_.C_alpha;
        C_b_current_ = scf_.C_beta;

        double e_mp2 = compute_mp2_energy();
        double e_mp3 = compute_mp3_correction();
        
        L2_aa_ = Eigen::Tensor<double, 4>(no_a_, no_a_, nv_a_, nv_a_);
        #pragma omp parallel for
        for (int i = 0; i < L2_aa_.size(); ++i) L2_aa_.data()[i] = t2_3rd_aa_.data()[i];

        if (no_b_ > 0 && nv_b_ > 0) { 
            L2_bb_ = Eigen::Tensor<double, 4>(no_b_, no_b_, nv_b_, nv_b_);
            L2_ab_ = Eigen::Tensor<double, 4>(no_a_, no_b_, nv_a_, nv_b_);
            #pragma omp parallel for
            for (int i = 0; i < L2_bb_.size(); ++i) L2_bb_.data()[i] = t2_3rd_bb_.data()[i];
            #pragma omp parallel for
            for (int i = 0; i < L2_ab_.size(); ++i) L2_ab_.data()[i] = t2_3rd_ab_.data()[i];
        }
        
        build_opdm_alpha(); 
        if (no_b_ > 0) build_opdm_beta();

        Eigen::MatrixXd F_ao_a, F_ao_b;
        build_fock_fast(scf_.P_alpha, scf_.P_beta, F_ao_a, F_ao_b);
        
        double e_scf = 0.5 * (scf_.P_alpha.cwiseProduct(H_core_ + F_ao_a).sum() + 
                              scf_.P_beta.cwiseProduct(H_core_ + F_ao_b).sum()) 
                              + e_nuc; 

        double e_mp3_corr = e_mp2 + e_mp3;
        double e_tot = e_scf + e_mp3_corr;

        if (macro_iter > 0 && e_tot > e_total_last + 1e-7) {
            current_step *= 0.5; 
            C_a_current_ = C_a_last; 
            C_b_current_ = C_b_last;
            
            Eigen::VectorXd scaled_step = last_actual_step * current_step;
            
            Eigen::MatrixXd Ka = Eigen::Map<const Eigen::MatrixXd>(scaled_step.data(), no_a_, nv_a_);
            Eigen::MatrixXd K_full = Eigen::MatrixXd::Zero(nbf_, nbf_);
            K_full.block(no_a_, 0, nv_a_, no_a_) = Ka.transpose(); K_full.block(0, no_a_, no_a_, nv_a_) = -Ka;
            C_a_current_ = C_a_current_ * K_full.exp();
            
            if (mode == "U" && no_b_ > 0) {
                Eigen::MatrixXd Kb = Eigen::Map<const Eigen::MatrixXd>(scaled_step.data() + dim_a, no_b_, nv_b_);
                Eigen::MatrixXd Kb_full = Eigen::MatrixXd::Zero(nbf_, nbf_);
                Kb_full.block(no_b_, 0, nv_b_, no_b_) = Kb.transpose(); Kb_full.block(0, no_b_, no_b_, nv_b_) = -Kb;
                C_b_current_ = C_b_current_ * Kb_full.exp();
            } else { C_b_current_ = C_a_current_; }
            
            scf_.P_alpha = C_a_current_.leftCols(no_a_) * C_a_current_.leftCols(no_a_).transpose();
            scf_.P_beta = (mode == "U") ? C_b_current_.leftCols(no_b_) * C_b_current_.leftCols(no_b_).transpose() : scf_.P_alpha;
            
            continue; 
        }

        current_step = std::min(1.0, current_step * 1.2);
        if (e_tot < e_total_best) { e_total_best = e_tot; e_corr_best = e_mp3_corr; }

        Eigen::MatrixXd G_full_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
        G_full_a.block(0,0,no_a_,no_a_) = G_oo_alpha_; G_full_a.block(no_a_,no_a_,nv_a_,nv_a_) = G_vv_alpha_;
        Eigen::MatrixXd P_corr_a = scf_.C_alpha * G_full_a * scf_.C_alpha.transpose();
        
        Eigen::MatrixXd P_corr_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
        if (no_b_ > 0) { 
            Eigen::MatrixXd G_full_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
            G_full_b.block(0,0,no_b_,no_b_) = G_oo_beta_; G_full_b.block(no_b_,no_b_,nv_b_,nv_b_) = G_vv_beta_; 
            P_corr_b = scf_.C_beta * G_full_b * scf_.C_beta.transpose();
        } else if (mode == "R") {
            P_corr_b = P_corr_a;
        }

        Eigen::MatrixXd F_HF_mo_a = scf_.C_alpha.transpose() * F_ao_a * scf_.C_alpha;
        Eigen::MatrixXd F_HF_mo_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
        if (no_b_ > 0) F_HF_mo_b = scf_.C_beta.transpose() * F_ao_b * scf_.C_beta;
        
        Eigen::MatrixXd G_gamma_a, G_gamma_b;
        build_fock_fast(P_corr_a, P_corr_b, G_gamma_a, G_gamma_b);
        G_gamma_a -= H_core_; if (no_b_ > 0) G_gamma_b -= H_core_;
        
        Eigen::MatrixXd F_gen_mo_a = F_HF_mo_a + scf_.C_alpha.transpose() * G_gamma_a * scf_.C_alpha;
        Eigen::MatrixXd F_gen_mo_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
        if (no_b_ > 0) F_gen_mo_b = F_HF_mo_b + scf_.C_beta.transpose() * G_gamma_b * scf_.C_beta;

        Eigen::MatrixXd F_vo_a = F_gen_mo_a.block(no_a_, 0, nv_a_, no_a_); 
        Eigen::MatrixXd L_sep_a = G_vv_alpha_ * F_vo_a - F_vo_a * G_oo_alpha_;
        F_gen_mo_a.block(no_a_, 0, nv_a_, no_a_) += L_sep_a;
        F_gen_mo_a.block(0, no_a_, no_a_, nv_a_) += L_sep_a.transpose();

        if (no_b_ > 0 && nv_b_ > 0) {
            Eigen::MatrixXd F_vo_b = F_gen_mo_b.block(no_b_, 0, nv_b_, no_b_); 
            Eigen::MatrixXd L_sep_b = G_vv_beta_ * F_vo_b - F_vo_b * G_oo_beta_;
            F_gen_mo_b.block(no_b_, 0, nv_b_, no_b_) += L_sep_b;
            F_gen_mo_b.block(0, no_b_, no_b_, nv_b_) += L_sep_b.transpose();
        }

        Eigen::MatrixXd B_ia_a = Eigen::MatrixXd::Zero(no_a_ * nv_a_, n_aux_);
        const Eigen::MatrixXd& Ca_o = C_a_current_.leftCols(no_a_);
        const Eigen::MatrixXd& Ca_v = C_a_current_.rightCols(nv_a_);
        Eigen::Map<const Eigen::MatrixXd> L_flat(scf_.L_mat.data(), nbf_, nbf_ * n_aux_);
        Eigen::MatrixXd X_a_tmp = Ca_v.transpose() * L_flat; 
        
        #pragma omp parallel for
        for (int P = 0; P < n_aux_; ++P) {
            Eigen::Map<Eigen::MatrixXd> X_P(X_a_tmp.data() + P * nv_a_ * nbf_, nv_a_, nbf_);
            Eigen::MatrixXd B_MO_a = X_P * Ca_o; 
            std::copy(B_MO_a.data(), B_MO_a.data() + no_a_ * nv_a_, B_ia_a.col(P).data());
        }

        Eigen::MatrixXd B_ia_b;
        if (no_b_ > 0 && nv_b_ > 0) {
            B_ia_b = Eigen::MatrixXd::Zero(no_b_ * nv_b_, n_aux_);
            const Eigen::MatrixXd& Cb_o = C_b_current_.leftCols(no_b_);
            const Eigen::MatrixXd& Cb_v = C_b_current_.rightCols(nv_b_);
            Eigen::MatrixXd X_b_tmp = Cb_v.transpose() * L_flat;
            #pragma omp parallel for
            for (int P = 0; P < n_aux_; ++P) {
                Eigen::Map<Eigen::MatrixXd> X_P(X_b_tmp.data() + P * nv_b_ * nbf_, nv_b_, nbf_);
                Eigen::MatrixXd B_MO_b = X_P * Cb_o; 
                std::copy(B_MO_b.data(), B_MO_b.data() + no_b_ * nv_b_, B_ia_b.col(P).data());
            }
        }

        Eigen::MatrixXd B_oo_a = Eigen::MatrixXd::Zero(no_a_*no_a_, n_aux_);
        Eigen::MatrixXd B_vv_a = Eigen::MatrixXd::Zero(nv_a_*nv_a_, n_aux_);
        Eigen::MatrixXd B_oo_b, B_vv_b;
        if (no_b_ > 0 && nv_b_ > 0) { B_oo_b = Eigen::MatrixXd::Zero(no_b_*no_b_, n_aux_); B_vv_b = Eigen::MatrixXd::Zero(nv_b_*nv_b_, n_aux_); }
        
        #pragma omp parallel for
        for (int P = 0; P < n_aux_; ++P) {
            Eigen::Map<const Eigen::MatrixXd> B_AO(scf_.L_mat.col(P).data(), nbf_, nbf_);
            Eigen::MatrixXd MO_oo_a = Ca_o.transpose() * (B_AO * Ca_o);
            Eigen::MatrixXd MO_vv_a = Ca_v.transpose() * (B_AO * Ca_v);
            for(int i=0; i<no_a_; ++i) for(int j=0; j<no_a_; ++j) B_oo_a(i*no_a_+j, P) = MO_oo_a(i, j);
            for(int a=0; a<nv_a_; ++a) for(int b=0; b<nv_a_; ++b) B_vv_a(a*nv_a_+b, P) = MO_vv_a(a, b);
            
            if (no_b_ > 0 && nv_b_ > 0) {
                Eigen::MatrixXd MO_oo_b = C_b_current_.leftCols(no_b_).transpose() * (B_AO * C_b_current_.leftCols(no_b_));
                Eigen::MatrixXd MO_vv_b = C_b_current_.rightCols(nv_b_).transpose() * (B_AO * C_b_current_.rightCols(nv_b_));
                for(int i=0; i<no_b_; ++i) for(int j=0; j<no_b_; ++j) B_oo_b(i*no_b_+j, P) = MO_oo_b(i, j);
                for(int a=0; a<nv_b_; ++a) for(int b=0; b<nv_b_; ++b) B_vv_b(a*nv_b_+b, P) = MO_vv_b(a, b);
            }
        }

        Eigen::MatrixXd Teff_aa = Eigen::MatrixXd::Zero(no_a_*nv_a_, no_a_*nv_a_);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < no_a_; ++i) for (int a = 0; a < nv_a_; ++a)
            for (int j = 0; j < no_a_; ++j) for (int b = 0; b < nv_a_; ++b)
                Teff_aa(i*nv_a_+a, j*nv_a_+b) = t2_aa_(i, j, a, b) + L2_aa_(i, j, a, b);

        Eigen::MatrixXd Teff_ab, Teff_bb;
        if (no_b_ > 0 && nv_b_ > 0) {
            Teff_ab = Eigen::MatrixXd::Zero(no_a_*nv_a_, no_b_*nv_b_);
            Teff_bb = Eigen::MatrixXd::Zero(no_b_*nv_b_, no_b_*nv_b_);
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < no_a_; ++i) for (int a = 0; a < nv_a_; ++a)
                for (int j = 0; j < no_b_; ++j) for (int b = 0; b < nv_b_; ++b)
                    Teff_ab(i*nv_a_+a, j*nv_b_+b) = t2_ab_(i, j, a, b) + L2_ab_(i, j, a, b);
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < no_b_; ++i) for (int a = 0; a < nv_b_; ++a)
                for (int j = 0; j < no_b_; ++j) for (int b = 0; b < nv_b_; ++b)
                    Teff_bb(i*nv_b_+a, j*nv_b_+b) = t2_bb_(i, j, a, b) + L2_bb_(i, j, a, b);
        }

        Eigen::MatrixXd X_a = Teff_aa * B_ia_a;
        if (no_b_ > 0 && nv_b_ > 0) X_a += Teff_ab * B_ia_b;

        Eigen::MatrixXd X_b;
        if (no_b_ > 0 && nv_b_ > 0) X_b = Teff_bb * B_ia_b + Teff_ab.transpose() * B_ia_a;

        Eigen::MatrixXd Z_mat_a = Eigen::MatrixXd::Zero(nv_a_, no_a_);
        Eigen::MatrixXd Z_mat_b;
        if (no_b_ > 0 && nv_b_ > 0) Z_mat_b = Eigen::MatrixXd::Zero(nv_b_, no_b_);

        TBLIS_VIEW_3D(t_Bvv_a, B_vv_a.data(), nv_a_, nv_a_, n_aux_);
        TBLIS_VIEW_3D(t_Xa, X_a.data(), nv_a_, no_a_, n_aux_);
        TBLIS_VIEW_3D(t_Boo_a, B_oo_a.data(), no_a_, no_a_, n_aux_);
        TBLIS_VIEW_2D(t_Za, Z_mat_a.data(), nv_a_, no_a_);

        tblis::mult<double>(1.0, t_Bvv_a, "baP", t_Xa, "biP", 1.0, t_Za, "ai");
        tblis::mult<double>(-1.0, t_Xa, "ajP", t_Boo_a, "jiP", 1.0, t_Za, "ai");

        if (no_b_ > 0 && nv_b_ > 0) {
            TBLIS_VIEW_3D(t_Bvv_b, B_vv_b.data(), nv_b_, nv_b_, n_aux_);
            TBLIS_VIEW_3D(t_Xb, X_b.data(), nv_b_, no_b_, n_aux_);
            TBLIS_VIEW_3D(t_Boo_b, B_oo_b.data(), no_b_, no_b_, n_aux_);
            TBLIS_VIEW_2D(t_Zb, Z_mat_b.data(), nv_b_, no_b_);

            tblis::mult<double>(1.0, t_Bvv_b, "baP", t_Xb, "biP", 1.0, t_Zb, "ai");
            tblis::mult<double>(-1.0, t_Xb, "ajP", t_Boo_b, "jiP", 1.0, t_Zb, "ai");
        }

        F_gen_mo_a.block(no_a_, 0, nv_a_, no_a_) += Z_mat_a;
        F_gen_mo_a.block(0, no_a_, no_a_, nv_a_) += Z_mat_a.transpose();
        if (no_b_ > 0 && nv_b_ > 0) {
            F_gen_mo_b.block(no_b_, 0, nv_b_, no_b_) += Z_mat_b;
            F_gen_mo_b.block(0, no_b_, no_b_, nv_b_) += Z_mat_b.transpose();
        }

        if (orbital_gradient_.size() != n_params) orbital_gradient_.resize(n_params);

        Eigen::MatrixXd wa = 2.0 * F_gen_mo_a.block(no_a_, 0, nv_a_, no_a_);
        Eigen::MatrixXd wb = 2.0 * F_gen_mo_b.block(no_b_, 0, nv_b_, no_b_);
        int idx = 0;
        
        if (mode == "U") {
            for (int a = 0; a < nv_a_; ++a) for (int i = 0; i < no_a_; ++i) orbital_gradient_(idx++) = wa(a, i);
            for (int b = 0; b < nv_b_; ++b) for (int i = 0; i < no_b_; ++i) orbital_gradient_(idx++) = wb(b, i);
        } else {
            for (int a = 0; a < nv_a_; ++a) for (int i = 0; i < no_a_; ++i) orbital_gradient_(idx++) = 0.5 * (wa(a, i) + wb(a, i));
        }
        
        grad_norm = orbital_gradient_.norm();

        if(omp_get_thread_num() == 0) {
            std::cout << std::setw(4) << macro_iter << "    " 
                      << std::fixed << std::setprecision(8) << e_tot << "    "
                      << std::setprecision(8) << e_mp3_corr << "    " 
                      << std::scientific << std::setprecision(2) << grad_norm << "\n";
        }

        if (macro_iter > 0 && grad_norm < config_.gradient_threshold && std::abs(e_tot - e_total_last) < config_.energy_threshold) {
            is_converged = true; break;
        }

        e_total_last = e_tot; C_a_last = C_a_current_; C_b_last = C_b_current_;

        Eigen::VectorXd diag_H(n_params);
        idx = 0; double level_shift = (grad_norm > 0.1) ? 0.05 : 0.005;
        
        for (int a = 0; a < nv_a_; ++a) {
            for (int i = 0; i < no_a_; ++i) {
                double eps_diff = scf_.orbital_energies_alpha(no_a_ + a) - scf_.orbital_energies_alpha(i);
                double J_ia = B_ia_a.row(i * nv_a_ + a).squaredNorm();
                diag_H(idx++) = 4.0 * std::abs(eps_diff) + 8.0 * J_ia + level_shift; 
            }
        }
        if (mode == "U" && no_b_ > 0) {
            for (int a = 0; a < nv_b_; ++a) {
                for (int i = 0; i < no_b_; ++i) {
                    double eps_diff = scf_.orbital_energies_beta(no_b_ + a) - scf_.orbital_energies_beta(i);
                    double J_ia = B_ia_b.row(i * nv_b_ + a).squaredNorm();
                    diag_H(idx++) = 4.0 * std::abs(eps_diff) + 8.0 * J_ia + level_shift;
                }
            }
        }

        mshqc::gradient::TrustRegionConfig tr_conf;
        tr_conf.micro_thresh = std::min(1e-4, grad_norm * 0.1); 
        mshqc::gradient::TrustRegionSOSCF soscf_engine(tr_conf);

        auto compute_hessian_vector = [&](const Eigen::VectorXd& p_vec) -> Eigen::VectorXd {
            Eigen::VectorXd Hp = diag_H.cwiseProduct(p_vec);
            
            Eigen::MatrixXd kappa_a = Eigen::Map<const Eigen::MatrixXd>(p_vec.data(), no_a_, nv_a_);
            Eigen::MatrixXd P1_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
            P1_a = C_a_current_.leftCols(no_a_) * kappa_a * C_a_current_.rightCols(nv_a_).transpose();
            P1_a += P1_a.transpose(); 
            
            Eigen::MatrixXd P1_b = P1_a;
            if (mode == "U" && dim_b > 0) {
                Eigen::MatrixXd kappa_b = Eigen::Map<const Eigen::MatrixXd>(p_vec.data() + dim_a, no_b_, nv_b_);
                P1_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
                P1_b = C_b_current_.leftCols(no_b_) * kappa_b * C_b_current_.rightCols(nv_b_).transpose();
                P1_b += P1_b.transpose();
            }
        
            Eigen::MatrixXd F1_a, F1_b;
            build_fock_fast(P1_a, P1_b, F1_a, F1_b);
            F1_a -= H_core_; 
            if (mode == "U" && no_b_ > 0) F1_b -= H_core_;
        
            double spin_factor = (mode == "R") ? 4.0 : 2.0;
            Eigen::MatrixXd H_kappa_a = C_a_current_.leftCols(no_a_).transpose() * F1_a * C_a_current_.rightCols(nv_a_);
            
            int idx_h = 0;
            for (int a = 0; a < nv_a_; ++a) for (int i = 0; i < no_a_; ++i) Hp(idx_h++) += spin_factor * H_kappa_a(i, a); 
            
            if (mode == "U" && dim_b > 0) {
                Eigen::MatrixXd H_kappa_b = C_b_current_.leftCols(no_b_).transpose() * F1_b * C_b_current_.rightCols(nv_b_);
                for (int a = 0; a < nv_b_; ++a) for (int i = 0; i < no_b_; ++i) Hp(idx_h++) += spin_factor * H_kappa_b(i, a);
            }
            return Hp;
        };

        mshqc::gradient::TrustRegionResult step_info = soscf_engine.solve(orbital_gradient_, diag_H, 0.50, compute_hessian_vector);
        Eigen::VectorXd actual_step = step_info.step;
        
        double max_rotation = 0.15; 
        for(int i = 0; i < actual_step.size(); ++i) {
            if (actual_step(i) > max_rotation) actual_step(i) = max_rotation;
            if (actual_step(i) < -max_rotation) actual_step(i) = -max_rotation;
        }

        Eigen::MatrixXd Ka = Eigen::Map<const Eigen::MatrixXd>(actual_step.data(), no_a_, nv_a_);
        Eigen::MatrixXd K_full = Eigen::MatrixXd::Zero(nbf_, nbf_);
        K_full.block(no_a_, 0, nv_a_, no_a_) = Ka.transpose(); K_full.block(0, no_a_, no_a_, nv_a_) = -Ka;
        C_a_current_ = C_a_current_ * K_full.exp();
        
        if (mode == "U" && no_b_ > 0) {
            Eigen::MatrixXd Kb = Eigen::Map<const Eigen::MatrixXd>(actual_step.data() + dim_a, no_b_, nv_b_);
            Eigen::MatrixXd Kb_full = Eigen::MatrixXd::Zero(nbf_, nbf_);
            Kb_full.block(no_b_, 0, nv_b_, no_b_) = Kb.transpose(); Kb_full.block(0, no_b_, no_b_, nv_b_) = -Kb;
            C_b_current_ = C_b_current_ * Kb_full.exp();
        } else { C_b_current_ = C_a_current_; }
        
        scf_.P_alpha = C_a_current_.leftCols(no_a_) * C_a_current_.leftCols(no_a_).transpose();
        scf_.P_beta = (mode == "U") ? C_b_current_.leftCols(no_b_) * C_b_current_.leftCols(no_b_).transpose() : scf_.P_alpha;
        
        last_actual_step = actual_step;
        macro_iter++;
    }

    MP3Result result;
    result.e_total = e_total_best;
    result.e_corr_total = e_corr_best;
    result.converged = is_converged;
    result.iterations = macro_iter;
    return result;
}
}