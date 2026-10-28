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
 * @brief Unified MP3 Implementation Powered by Native TBLIS & HPC Cholesky Factorization
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
    if(omp_get_thread_num() == 0) std::cout << "\n=== RMP3 (Unified Native TBLIS - Cholesky) ===\n";

    const auto& Co = scf_.C_alpha.leftCols(no_a_);
    const auto& Cv = scf_.C_alpha.rightCols(nv_a_);

    TBLIS_VIEW_4D(t_T, t2_aa_, no_a_, no_a_, nv_a_, nv_a_);
    Eigen::Tensor< double, 4 > W(no_a_, no_a_, nv_a_, nv_a_); W.setZero();
    TBLIS_VIEW_4D(t_W, W, no_a_, no_a_, nv_a_, nv_a_);

    const Eigen::Tensor<double, 4>* eri_ao_ptr = nullptr;
    if (config_.eri_method == "exact") eri_ao_ptr = &(ints_->compute_eri());

    auto get_V_exact = [&](const Eigen::MatrixXd& C1, const Eigen::MatrixXd& C2, const Eigen::MatrixXd& C3, const Eigen::MatrixXd& C4) {
        int d1 = C1.cols(); int d2 = C2.cols(); int d3 = C3.cols(); int d4 = C4.cols();
        return ERITransformer::transform_custom(*eri_ao_ptr, C1, C2, C3, C4, nbf_, d1, d2, d3, d4);
    };

    // Bypass TBLIS/MKL thread limits with explicit layout-respecting mapping
    auto make_V_from_B_inplace = [&](const Eigen::MatrixXd& B1, const Eigen::MatrixXd& B2, Eigen::Tensor<double, 4>& V_mo) {
        int d1 = B1.rows(); int d2 = B2.rows(); 
        Eigen::MatrixXd V_mat(d1, d2);
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < d1; ++i) {
            V_mat.row(i).noalias() = B1.row(i) * B2.transpose();
        }

        if (V_mat.cwiseAbs().maxCoeff() < 1e-12) {
            V_mo.setZero();
            return false;
        }
        
        int dim0 = V_mo.dimension(0);
        int dim1 = V_mo.dimension(1);
        int dim2 = V_mo.dimension(2);
        int dim3 = V_mo.dimension(3);

        #pragma omp parallel for collapse(4)
        for (int i = 0; i < dim0; ++i) {
            for (int j = 0; j < dim1; ++j) {
                for (int k = 0; k < dim2; ++k) {
                    for (int l = 0; l < dim3; ++l) {
                        int row = i * dim1 + j; 
                        int col = k * dim3 + l;
                        V_mo(i, j, k, l) = V_mat(row, col);
                    }
                }
            }
        }
        return true;
    };

    Eigen::MatrixXd B_oo, B_ov, B_vv;
    if (config_.eri_method != "exact") {
        B_ov = mp2_.B_ia_P_alpha; 
        
        auto build_B_fast = [&](const Eigen::MatrixXd& C1, const Eigen::MatrixXd& C2) {
            int d1 = C1.cols(); int d2 = C2.cols();
            Eigen::MatrixXd B = Eigen::MatrixXd::Zero(d1 * d2, n_aux_);
            Eigen::Map<const Eigen::MatrixXd> L_flat(scf_.L_mat.data(), nbf_, nbf_ * n_aux_);
            Eigen::MatrixXd temp1 = C1.transpose() * L_flat;
            #pragma omp parallel for schedule(static)
            for (int P = 0; P < n_aux_; ++P) {
                Eigen::Map<const Eigen::MatrixXd> tmp_P(temp1.data() + P * d1 * nbf_, d1, nbf_);
                Eigen::MatrixXd L_pq = tmp_P * C2;
                for (int x = 0; x < d1; ++x) {
                    for (int y = 0; y < d2; ++y) {
                        B(x * d2 + y, P) = L_pq(x, y);
                    }
                }
            }
            return B;
        };

        B_oo = build_B_fast(Co, Co);
        B_vv = build_B_fast(Cv, Cv);
    }

    // 1. Ladder term (vvvv)
    if (config_.eri_method == "exact") {
        Eigen::Tensor<double, 4> V_vvvv = get_V_exact(Cv, Cv, Cv, Cv);
        TBLIS_VIEW_4D(t_V, V_vvvv, nv_a_, nv_a_, nv_a_, nv_a_);
        tblis::mult< double >(1.0, t_T, "ijef", t_V, "eafb", 1.0, t_W, "ijab");
    } else {
        int n_pairs_sym = no_a_ * (no_a_ + 1) / 2;
        std::vector<Eigen::MatrixXd> T_pairs(n_pairs_sym, Eigen::MatrixXd(nv_a_, nv_a_));
        std::vector<Eigen::MatrixXd> W_pairs(n_pairs_sym, Eigen::MatrixXd::Zero(nv_a_, nv_a_));
        int p_idx = 0;
        for (int i = 0; i < no_a_; ++i) {
            for (int j = i; j < no_a_; ++j) {
                for (int e = 0; e < nv_a_; ++e)
                    for (int f = 0; f < nv_a_; ++f)
                        T_pairs[p_idx](e, f) = t2_aa_(i, j, e, f);
                p_idx++;
            }
        }
        #pragma omp parallel
        {
            std::vector<Eigen::MatrixXd> W_loc(n_pairs_sym, Eigen::MatrixXd::Zero(nv_a_, nv_a_));
            Eigen::MatrixXd X_tmp(nv_a_, nv_a_);
            #pragma omp for schedule(static) nowait
            for (int P = 0; P < n_aux_; ++P) {
                Eigen::Map<const Eigen::MatrixXd> B_P(B_vv.col(P).data(), nv_a_, nv_a_);
                for (int k = 0; k < n_pairs_sym; ++k) {
                    X_tmp.noalias() = T_pairs[k] * B_P;
                    W_loc[k].noalias() += B_P.transpose() * X_tmp;
                }
            }
            #pragma omp critical
            {
                for (int k = 0; k < n_pairs_sym; ++k) W_pairs[k] += W_loc[k];
            }
        }
        p_idx = 0;
        for (int i = 0; i < no_a_; ++i) {
            for (int j = i; j < no_a_; ++j) {
                for (int a = 0; a < nv_a_; ++a) {
                    for (int b = 0; b < nv_a_; ++b) {
                        W(i, j, a, b) += W_pairs[p_idx](a, b);
                        if (i != j) W(j, i, b, a) += W_pairs[p_idx](a, b);
                    }
                }
                p_idx++;
            }
        }
    }

    // 2. Ladder term (oooo)
    {
        Eigen::Tensor<double, 4> V(no_a_, no_a_, no_a_, no_a_);
        if (config_.eri_method == "exact") {
            V = get_V_exact(Co, Co, Co, Co);
            TBLIS_VIEW_4D(t_V, V, no_a_, no_a_, no_a_, no_a_);
            tblis::mult< double >(1.0, t_T, "mnab", t_V, "minj", 1.0, t_W, "ijab");
        } else {
            if (make_V_from_B_inplace(B_oo, B_oo, V)) {
                TBLIS_VIEW_4D(t_V, V, no_a_, no_a_, no_a_, no_a_);
                tblis::mult< double >(1.0, t_T, "mnab", t_V, "minj", 1.0, t_W, "ijab");
            }
        }
    }

    // 3. Ring terms (ovov & oovv)
    {
        Eigen::Tensor<double, 4> V_ovov(no_a_, nv_a_, no_a_, nv_a_);
        Eigen::Tensor<double, 4> V_oovv(no_a_, no_a_, nv_a_, nv_a_);
        bool calc_ovov = true, calc_oovv = true;

        if (config_.eri_method == "exact") {
            V_ovov = get_V_exact(Co, Cv, Co, Cv);
            V_oovv = get_V_exact(Co, Co, Cv, Cv);
        } else {
            calc_ovov = make_V_from_B_inplace(B_ov, B_ov, V_ovov);
            calc_oovv = make_V_from_B_inplace(B_oo, B_vv, V_oovv);
        }

        TBLIS_VIEW_4D(t_Vovov, V_ovov, no_a_, nv_a_, no_a_, nv_a_);
        TBLIS_VIEW_4D(t_Voovv, V_oovv, no_a_, no_a_, nv_a_, nv_a_);

        if (calc_ovov) {
            tblis::mult< double >(2.0,  t_Vovov, "iakc", t_T, "kjcb", 1.0, t_W, "ijab");
            tblis::mult< double >(2.0,  t_T, "ikac", t_Vovov, "kcjb", 1.0, t_W, "ijab");
            tblis::mult< double >(-1.0, t_T, "ikca", t_Vovov, "kcjb", 1.0, t_W, "ijab");
            tblis::mult< double >(-1.0, t_Vovov, "iakc", t_T, "kjbc", 1.0, t_W, "ijab");
        }
        if (calc_oovv) {
            tblis::mult< double >(-1.0, t_Voovv, "ikac", t_T, "kjcb", 1.0, t_W, "ijab");
            tblis::mult< double >(-1.0, t_T, "ikac", t_Voovv, "kjcb", 1.0, t_W, "ijab");
            tblis::mult< double >(-1.0, t_Voovv, "ikbc", t_T, "kjac", 1.0, t_W, "ijab");
            tblis::mult< double >(-1.0, t_T, "ikcb", t_Voovv, "jkac", 1.0, t_W, "ijab");
        }
    }

    double e_mp3 = 0.0;
    #pragma omp parallel for collapse(4) reduction(+:e_mp3)
    for (int i = 0; i < no_a_; ++i) {
        for (int j = 0; j < no_a_; ++j) {
            for (int a = 0; a < nv_a_; ++a) {
                for (int b = 0; b < nv_a_; ++b) {
                    // Menerapkan faktor adaptasi spin yang tepat untuk mengevaluasi korelasi spasial penuh
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
    if(omp_get_thread_num() == 0) std::cout << "\n=== UMP3 (Unified Native TBLIS - Cholesky) ===\n";

    const auto& Cao = scf_.C_alpha.leftCols(no_a_); const auto& Cav = scf_.C_alpha.rightCols(nv_a_);
    const auto& Cbo = scf_.C_beta.leftCols(no_b_);  const auto& Cbv = scf_.C_beta.rightCols(nv_b_);
    double e3_aa = 0.0, e3_bb = 0.0, e3_ab = 0.0;

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

    const Eigen::Tensor<double, 4>* eri_ao_ptr = nullptr;
    if (config_.eri_method == "exact") eri_ao_ptr = &(ints_->compute_eri());

    auto get_V_exact = [&](const Eigen::MatrixXd& C1, const Eigen::MatrixXd& C2, const Eigen::MatrixXd& C3, const Eigen::MatrixXd& C4) {
        int d1 = C1.cols(); int d2 = C2.cols(); int d3 = C3.cols(); int d4 = C4.cols();
        return ERITransformer::transform_custom(*eri_ao_ptr, C1, C2, C3, C4, nbf_, d1, d2, d3, d4);
    };

    auto make_V_from_B_inplace = [&](const Eigen::MatrixXd& B1, const Eigen::MatrixXd& B2, Eigen::Tensor<double, 4>& V_mo) {
        int d1 = B1.rows(); int d2 = B2.rows(); 
        Eigen::MatrixXd V_mat(d1, d2);
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < d1; ++i) {
            V_mat.row(i).noalias() = B1.row(i) * B2.transpose();
        }

        if (V_mat.cwiseAbs().maxCoeff() < 1e-12) {
            V_mo.setZero();
            return false;
        }

        int dim0 = V_mo.dimension(0);
        int dim1 = V_mo.dimension(1);
        int dim2 = V_mo.dimension(2);
        int dim3 = V_mo.dimension(3);

        #pragma omp parallel for collapse(4)
        for (int i = 0; i < dim0; ++i) {
            for (int j = 0; j < dim1; ++j) {
                for (int k = 0; k < dim2; ++k) {
                    for (int l = 0; l < dim3; ++l) {
                        int row = i * dim1 + j; 
                        int col = k * dim3 + l;
                        V_mo(i, j, k, l) = V_mat(row, col);
                    }
                }
            }
        }
        return true;
    };

    Eigen::MatrixXd B_oo_a, B_ov_a, B_vv_a;
    Eigen::MatrixXd B_oo_b, B_ov_b, B_vv_b;
    if (config_.eri_method != "exact") {
        B_ov_a = mp2_.B_ia_P_alpha; 
        if (no_b_ > 0 && nv_b_ > 0) B_ov_b = mp2_.B_ia_P_beta;

        auto build_B_fast = [&](const Eigen::MatrixXd& C1, const Eigen::MatrixXd& C2) {
            int d1 = C1.cols(); int d2 = C2.cols();
            Eigen::MatrixXd B = Eigen::MatrixXd::Zero(d1 * d2, n_aux_);
            Eigen::Map<const Eigen::MatrixXd> L_flat(scf_.L_mat.data(), nbf_, nbf_ * n_aux_);
            Eigen::MatrixXd temp1 = C1.transpose() * L_flat;
            #pragma omp parallel for schedule(static)
            for (int P = 0; P < n_aux_; ++P) {
                Eigen::Map<const Eigen::MatrixXd> tmp_P(temp1.data() + P * d1 * nbf_, d1, nbf_);
                Eigen::MatrixXd L_pq = tmp_P * C2;
                for (int x = 0; x < d1; ++x) {
                    for (int y = 0; y < d2; ++y) {
                        B(x * d2 + y, P) = L_pq(x, y);
                    }
                }
            }
            return B;
        };

        B_oo_a = build_B_fast(Cao, Cao);
        B_vv_a = build_B_fast(Cav, Cav);

        if (no_b_ > 0 && nv_b_ > 0) {
            B_oo_b = build_B_fast(Cbo, Cbo);
            B_vv_b = build_B_fast(Cbv, Cbv);
        }
    }

    // =========================================================================
    // 1. TAHAP LADDER (vvvv) 
    // =========================================================================
    if (config_.eri_method == "exact") {
        Waa.setZero();
        Eigen::Tensor<double, 4> Vvvvv_aa = get_V_exact(Cav, Cav, Cav, Cav);
        TBLIS_VIEW_4D(t_Vaa, Vvvvv_aa, nv_a_, nv_a_, nv_a_, nv_a_);
        tblis::mult< double >(1.0, t_Taa, "ijef", t_Vaa, "eafb", 1.0, t_Waa, "ijab");
        e3_aa += 0.25 * tensor_dot(t2_aa_, Waa);

        if (no_b_ > 0 && nv_b_ > 0) {
            Wbb.setZero();
            Eigen::Tensor<double, 4> Vvvvv_bb = get_V_exact(Cbv, Cbv, Cbv, Cbv);
            TBLIS_VIEW_4D(t_Vbb, Vvvvv_bb, nv_b_, nv_b_, nv_b_, nv_b_);
            tblis::mult< double >(1.0, t_Tbb, "ijef", t_Vbb, "eafb", 1.0, t_Wbb, "ijab");
            e3_bb += 0.25 * tensor_dot(t2_bb_, Wbb);

            Wab.setZero();
            Eigen::Tensor<double, 4> Vvvvv_ab = get_V_exact(Cav, Cav, Cbv, Cbv);
            TBLIS_VIEW_4D(t_Vab, Vvvvv_ab, nv_a_, nv_a_, nv_b_, nv_b_);
            tblis::mult< double >(1.0, t_Tab, "ijef", t_Vab, "eafb", 1.0, t_Wab, "ijab");
            e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
        }
    } else {
        int n_pairs_aa = no_a_ * (no_a_ - 1) / 2;
        if (n_pairs_aa > 0) {
            std::vector<Eigen::MatrixXd> T_aa_pairs(n_pairs_aa, Eigen::MatrixXd(nv_a_, nv_a_));
            std::vector<Eigen::MatrixXd> W_aa_pairs(n_pairs_aa, Eigen::MatrixXd::Zero(nv_a_, nv_a_));
            int p_idx = 0;
            for (int i = 0; i < no_a_; ++i) {
                for (int j = i + 1; j < no_a_; ++j) {
                    for (int e = 0; e < nv_a_; ++e)
                        for (int f = 0; f < nv_a_; ++f)
                            T_aa_pairs[p_idx](e, f) = t2_aa_(i, j, e, f);
                    p_idx++;
                }
            }
            #pragma omp parallel
            {
                std::vector<Eigen::MatrixXd> W_loc(n_pairs_aa, Eigen::MatrixXd::Zero(nv_a_, nv_a_));
                Eigen::MatrixXd X_tmp(nv_a_, nv_a_);
                #pragma omp for schedule(static) nowait
                for (int P = 0; P < n_aux_; ++P) {
                    Eigen::Map<const Eigen::MatrixXd> B_P(B_vv_a.col(P).data(), nv_a_, nv_a_);
                    for (int k = 0; k < n_pairs_aa; ++k) {
                        X_tmp.noalias() = T_aa_pairs[k] * B_P;
                        W_loc[k].noalias() += B_P.transpose() * X_tmp;
                    }
                }
                #pragma omp critical
                {
                    for (int k = 0; k < n_pairs_aa; ++k) W_aa_pairs[k] += W_loc[k];
                }
            }
            double sum_aa = 0.0;
            for (int k = 0; k < n_pairs_aa; ++k) {
                Eigen::MatrixXd W_asym = W_aa_pairs[k] - W_aa_pairs[k].transpose();
                sum_aa += 2.0 * (T_aa_pairs[k].cwiseProduct(W_asym)).sum();
            }
            e3_aa += 0.125 * sum_aa;
        }
        if (no_b_ > 0 && nv_b_ > 0) {
            int n_pairs_bb = no_b_ * (no_b_ - 1) / 2;
            if (n_pairs_bb > 0) {
                std::vector<Eigen::MatrixXd> T_bb_pairs(n_pairs_bb, Eigen::MatrixXd(nv_b_, nv_b_));
                std::vector<Eigen::MatrixXd> W_bb_pairs(n_pairs_bb, Eigen::MatrixXd::Zero(nv_b_, nv_b_));
                int p_idx = 0;
                for (int i = 0; i < no_b_; ++i) {
                    for (int j = i + 1; j < no_b_; ++j) {
                        for (int e = 0; e < nv_b_; ++e)
                            for (int f = 0; f < nv_b_; ++f)
                                T_bb_pairs[p_idx](e, f) = t2_bb_(i, j, e, f);
                        p_idx++;
                    }
                }
                #pragma omp parallel
                {
                    std::vector<Eigen::MatrixXd> W_loc(n_pairs_bb, Eigen::MatrixXd::Zero(nv_b_, nv_b_));
                    Eigen::MatrixXd X_tmp(nv_b_, nv_b_);
                    #pragma omp for schedule(static) nowait
                    for (int P = 0; P < n_aux_; ++P) {
                        Eigen::Map<const Eigen::MatrixXd> B_P(B_vv_b.col(P).data(), nv_b_, nv_b_);
                        for (int k = 0; k < n_pairs_bb; ++k) {
                            X_tmp.noalias() = T_bb_pairs[k] * B_P;
                            W_loc[k].noalias() += B_P.transpose() * X_tmp;
                        }
                    }
                    #pragma omp critical
                    {
                        for (int k = 0; k < n_pairs_bb; ++k) W_bb_pairs[k] += W_loc[k];
                    }
                }
                double sum_bb = 0.0;
                for (int k = 0; k < n_pairs_bb; ++k) {
                    Eigen::MatrixXd W_asym = W_bb_pairs[k] - W_bb_pairs[k].transpose();
                    sum_bb += 2.0 * (T_bb_pairs[k].cwiseProduct(W_asym)).sum();
                }
                e3_bb += 0.125 * sum_bb;
            }
            int n_pairs_ab = no_a_ * no_b_;
            std::vector<Eigen::MatrixXd> T_ab_pairs(n_pairs_ab, Eigen::MatrixXd(nv_a_, nv_b_));
            std::vector<Eigen::MatrixXd> W_ab_pairs(n_pairs_ab, Eigen::MatrixXd::Zero(nv_a_, nv_b_));
            for (int i = 0; i < no_a_; ++i) {
                for (int j = 0; j < no_b_; ++j) {
                    int k = i * no_b_ + j;
                    for (int e = 0; e < nv_a_; ++e)
                        for (int f = 0; f < nv_b_; ++f)
                            T_ab_pairs[k](e, f) = t2_ab_(i, j, e, f);
                }
            }
            #pragma omp parallel
            {
                std::vector<Eigen::MatrixXd> W_loc(n_pairs_ab, Eigen::MatrixXd::Zero(nv_a_, nv_b_));
                Eigen::MatrixXd X_tmp(nv_a_, nv_b_);
                #pragma omp for schedule(static) nowait
                for (int P = 0; P < n_aux_; ++P) {
                    Eigen::Map<const Eigen::MatrixXd> B_Pa(B_vv_a.col(P).data(), nv_a_, nv_a_);
                    Eigen::Map<const Eigen::MatrixXd> B_Pb(B_vv_b.col(P).data(), nv_b_, nv_b_);
                    for (int k = 0; k < n_pairs_ab; ++k) {
                        X_tmp.noalias() = T_ab_pairs[k] * B_Pb;
                        W_loc[k].noalias() += B_Pa.transpose() * X_tmp;
                    }
                }
                #pragma omp critical
                {
                    for (int k = 0; k < n_pairs_ab; ++k) W_ab_pairs[k] += W_loc[k];
                }
            }
            double sum_ab = 0.0;
            for (int k = 0; k < n_pairs_ab; ++k) {
                sum_ab += (T_ab_pairs[k].cwiseProduct(W_ab_pairs[k])).sum();
            }
            e3_ab += sum_ab;
        }
    }

    // =========================================================================
    // 2. TAHAP LADDER (oooo) 
    // =========================================================================
    {
        Eigen::Tensor<double, 4> Vaa(no_a_, no_a_, no_a_, no_a_);
        if (config_.eri_method == "exact") {
            Vaa = get_V_exact(Cao, Cao, Cao, Cao);
            TBLIS_VIEW_4D(t_Vaa, Vaa, no_a_, no_a_, no_a_, no_a_); Waa.setZero();
            tblis::mult< double >(1.0, t_Taa, "mnab", t_Vaa, "minj", 1.0, t_Waa, "ijab");
            tblis::mult< double >(-1.0, t_Taa, "mnab", t_Vaa, "mjni", 1.0, t_Waa, "ijab");
            e3_aa += 0.125 * tensor_dot(t2_aa_, Waa);
        } else {
            if (make_V_from_B_inplace(B_oo_a, B_oo_a, Vaa)) {
                TBLIS_VIEW_4D(t_Vaa, Vaa, no_a_, no_a_, no_a_, no_a_); Waa.setZero();
                tblis::mult< double >(1.0, t_Taa, "mnab", t_Vaa, "minj", 1.0, t_Waa, "ijab");
                tblis::mult< double >(-1.0, t_Taa, "mnab", t_Vaa, "mjni", 1.0, t_Waa, "ijab");
                e3_aa += 0.125 * tensor_dot(t2_aa_, Waa);
            }
        }

        if (no_b_ > 0 && nv_b_ > 0) {
            Eigen::Tensor<double, 4> Vbb(no_b_, no_b_, no_b_, no_b_);
            if (config_.eri_method == "exact") {
                Vbb = get_V_exact(Cbo, Cbo, Cbo, Cbo);
                TBLIS_VIEW_4D(t_Vbb, Vbb, no_b_, no_b_, no_b_, no_b_); Wbb.setZero();
                tblis::mult< double >(1.0, t_Tbb, "mnab", t_Vbb, "minj", 1.0, t_Wbb, "ijab");
                tblis::mult< double >(-1.0, t_Tbb, "mnab", t_Vbb, "mjni", 1.0, t_Wbb, "ijab");
                e3_bb += 0.125 * tensor_dot(t2_bb_, Wbb);
            } else {
                if (make_V_from_B_inplace(B_oo_b, B_oo_b, Vbb)) {
                    TBLIS_VIEW_4D(t_Vbb, Vbb, no_b_, no_b_, no_b_, no_b_); Wbb.setZero();
                    tblis::mult< double >(1.0, t_Tbb, "mnab", t_Vbb, "minj", 1.0, t_Wbb, "ijab");
                    tblis::mult< double >(-1.0, t_Tbb, "mnab", t_Vbb, "mjni", 1.0, t_Wbb, "ijab");
                    e3_bb += 0.125 * tensor_dot(t2_bb_, Wbb);
                }
            }

            Eigen::Tensor<double, 4> Vab(no_a_, no_a_, no_b_, no_b_);
            if (config_.eri_method == "exact") {
                Vab = get_V_exact(Cao, Cao, Cbo, Cbo);
                TBLIS_VIEW_4D(t_Vab, Vab, no_a_, no_a_, no_b_, no_b_); Wab.setZero();
                tblis::mult< double >(1.0, t_Tab, "mnab", t_Vab, "minj", 1.0, t_Wab, "ijab");
                e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
            } else {
                if (make_V_from_B_inplace(B_oo_a, B_oo_b, Vab)) {
                    TBLIS_VIEW_4D(t_Vab, Vab, no_a_, no_a_, no_b_, no_b_); Wab.setZero();
                    tblis::mult< double >(1.0, t_Tab, "mnab", t_Vab, "minj", 1.0, t_Wab, "ijab");
                    e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
                }
            }
        }
    }

    // =========================================================================
    // 3. TAHAP RING (ovov & oovv) 
    // =========================================================================
    {
        Eigen::Tensor<double, 4> ovov_aa(no_a_, nv_a_, no_a_, nv_a_); 
        Eigen::Tensor<double, 4> oovv_aa(no_a_, no_a_, nv_a_, nv_a_); 
        bool calc_aa = true;

        if (config_.eri_method == "exact") {
            ovov_aa = get_V_exact(Cao, Cav, Cao, Cav);
            oovv_aa = get_V_exact(Cao, Cao, Cav, Cav); 
        } else {
            calc_aa = make_V_from_B_inplace(B_ov_a, B_ov_a, ovov_aa);
            calc_aa &= make_V_from_B_inplace(B_oo_a, B_vv_a, oovv_aa);
        }

        TBLIS_VIEW_4D(t_ovov_aa, ovov_aa, no_a_, nv_a_, no_a_, nv_a_);
        TBLIS_VIEW_4D(t_oovv_aa, oovv_aa, no_a_, no_a_, nv_a_, nv_a_);

        Waa.setZero();
        if (calc_aa) {
            tblis::mult< double >(1.0, t_ovov_aa, "iakc", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
            tblis::mult< double >(-1.0, t_oovv_aa, "ikca", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
        }

        if (no_b_ > 0 && nv_b_ > 0) {
            Eigen::Tensor<double, 4> ovov_bb(no_b_, nv_b_, no_b_, nv_b_); 
            Eigen::Tensor<double, 4> oovv_bb(no_b_, no_b_, nv_b_, nv_b_); 
            Eigen::Tensor<double, 4> ovov_ab(no_a_, nv_a_, no_b_, nv_b_); 
            Eigen::Tensor<double, 4> oovv_ab_ex(no_a_, no_a_, nv_b_, nv_b_); 
            Eigen::Tensor<double, 4> oovv_ba_ex(no_b_, no_b_, nv_a_, nv_a_); 
            
            bool calc_bb = true, calc_ab = true;

            if (config_.eri_method == "exact") {
                ovov_bb    = get_V_exact(Cbo, Cbv, Cbo, Cbv);
                oovv_bb    = get_V_exact(Cbo, Cbo, Cbv, Cbv);
                ovov_ab    = get_V_exact(Cao, Cav, Cbo, Cbv);
                oovv_ab_ex = get_V_exact(Cao, Cao, Cbv, Cbv);
                oovv_ba_ex = get_V_exact(Cbo, Cbo, Cav, Cav);
            } else {
                calc_bb = make_V_from_B_inplace(B_ov_b, B_ov_b, ovov_bb);
                calc_bb &= make_V_from_B_inplace(B_oo_b, B_vv_b, oovv_bb);
                calc_ab = make_V_from_B_inplace(B_ov_a, B_ov_b, ovov_ab);
                calc_ab &= make_V_from_B_inplace(B_oo_a, B_vv_b, oovv_ab_ex);
                calc_ab &= make_V_from_B_inplace(B_oo_b, B_vv_a, oovv_ba_ex);
            }
            
            TBLIS_VIEW_4D(t_ovov_bb, ovov_bb, no_b_, nv_b_, no_b_, nv_b_);
            TBLIS_VIEW_4D(t_oovv_bb, oovv_bb, no_b_, no_b_, nv_b_, nv_b_);
            TBLIS_VIEW_4D(t_ovov_ab, ovov_ab, no_a_, nv_a_, no_b_, nv_b_);
            TBLIS_VIEW_4D(t_oovv_ab_ex, oovv_ab_ex, no_a_, no_a_, nv_b_, nv_b_);
            TBLIS_VIEW_4D(t_oovv_ba_ex, oovv_ba_ex, no_b_, no_b_, nv_a_, nv_a_);

            if (calc_ab) tblis::mult< double >(1.0, t_ovov_ab, "iakc", t_Tab, "jkbc", 1.0, t_Waa, "ijab");
            e3_aa += 1.0 * tensor_dot(t2_aa_, Waa);

            Wbb.setZero();
            if (calc_bb) {
                tblis::mult< double >(1.0, t_ovov_bb, "iakc", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
                tblis::mult< double >(-1.0, t_oovv_bb, "ikca", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
            }
            if (calc_ab) tblis::mult< double >(1.0, t_ovov_ab, "kcia", t_Tab, "kjcb", 1.0, t_Wbb, "ijab"); 
            e3_bb += 1.0 * tensor_dot(t2_bb_, Wbb);

            Wab.setZero();
            if (calc_aa) {
                tblis::mult< double >(1.0,  t_ovov_aa, "iakc", t_Tab, "kjcb", 1.0, t_Wab, "ijab");
                tblis::mult< double >(-1.0, t_oovv_aa, "ikca", t_Tab, "kjcb", 1.0, t_Wab, "ijab");
            }
            if (calc_bb) {
                tblis::mult< double >(1.0,  t_Tab, "ikac", t_ovov_bb, "kcjb", 1.0, t_Wab, "ijab");
                tblis::mult< double >(-1.0, t_Tab, "ikac", t_oovv_bb, "kjcb", 1.0, t_Wab, "ijab");
            }
            if (calc_ab) {
                tblis::mult< double >(1.0,  t_Taa, "ikac", t_ovov_ab, "kcjb", 1.0, t_Wab, "ijab");
                tblis::mult< double >(1.0,  t_ovov_ab, "iakc", t_Tbb, "kjcb", 1.0, t_Wab, "ijab");
                tblis::mult< double >(-1.0, t_oovv_ab_ex, "ikbc", t_Tab, "kjac", 1.0, t_Wab, "ijab");
                tblis::mult< double >(-1.0, t_Tab, "ikcb", t_oovv_ba_ex, "jkac", 1.0, t_Wab, "ijab"); 
            }
            e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
        } else {
            e3_aa += 1.0 * tensor_dot(t2_aa_, Waa);
        }
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
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    scf_.C_alpha = C_a_current_;
    scf_.C_beta  = C_b_current_;
    scf_.P_alpha.noalias() = scf_.C_alpha.leftCols(na_) * scf_.C_alpha.leftCols(na_).transpose();
    scf_.P_beta.noalias()  = scf_.C_beta.leftCols(nb_)  * scf_.C_beta.leftCols(nb_).transpose();

    pseudocanonicalize();

    C_a_current_ = scf_.C_alpha;
    C_b_current_ = scf_.C_beta;

    if (config_.eri_method == "cholesky" || config_.eri_method == "df") {
        scf_.irreps_alpha.assign(nbf_, 0);
        if (!is_restricted && nb_ > 0) {
            scf_.irreps_beta.assign(nbf_, 0);
        }
        transform_3center_mo_cholesky();
        compute_t2_and_energy_cholesky();
    } else {
        transform_integrals();
        compute_t2_amplitudes();
        compute_mp2_energy();
    }

    compute_mp3_correction(); 
    
    L2_aa_ = t2_3rd_aa_;
    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        L2_bb_ = t2_3rd_bb_;
        L2_ab_ = t2_3rd_ab_;
    }
    
    // Hanya dieksekusi 1x per iterasi makro setelah L2_aa_, L2_bb_, L2_ab_ diperbarui
    build_opdm_alpha();
    if (!is_restricted && nb_ > 0) {
        build_opdm_beta();
    } else if (is_restricted && nb_ > 0) {
        G_oo_beta_ = G_oo_alpha_;
        G_vv_beta_ = G_vv_alpha_;
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
    if (config_.eri_method == "exact") {
        if (is_restricted) {
            t2_3rd_aa_ = Eigen::Tensor< double, 4 >(na_, na_, va_, va_);
            Eigen::Tensor< double, 4 > W(na_, na_, va_, va_); W.setZero();
            
            TBLIS_VIEW_4D(t_T, T2_aa_ijab, na_, na_, va_, va_);
            TBLIS_VIEW_4D(t_W, W, na_, na_, va_, va_);

          
            const auto& eri_ao_cached_local = integrals_->compute_eri();

            Eigen::Tensor<double, 4> T_ao(na_, na_, nbf_, nbf_);
            Eigen::Tensor<double, 4> W_ao(na_, na_, nbf_, nbf_);
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < na_; ++i) {
                for (int j = 0; j < na_; ++j) {
                    Eigen::MatrixXd Tij_vv(va_, va_);
                    for (int e = 0; e < va_; ++e)
                        for (int f = 0; f < va_; ++f)
                            Tij_vv(e, f) = T2_aa_ijab(i, j, e, f);
                    Eigen::MatrixXd Tij_ao = Cav * Tij_vv * Cav.transpose();
                    for (int mu = 0; mu < nbf_; ++mu)
                        for (int lam = 0; lam < nbf_; ++lam)
                            T_ao(i, j, mu, lam) = Tij_ao(mu, lam);
                }
            }
            W_ao.setZero();
            TBLIS_VIEW_4D(t_Tao, T_ao, na_, na_, nbf_, nbf_);
            TBLIS_VIEW_4D(t_Wao, W_ao, na_, na_, nbf_, nbf_);
            tblis::varray_view<double> t_eri_ao(
                {(len_type)nbf_, (len_type)nbf_, (len_type)nbf_, (len_type)nbf_},
                const_cast<double*>(eri_ao_cached_local.data()),
                {1, (stride_type)nbf_, (stride_type)(nbf_*nbf_), (stride_type)(nbf_*nbf_*nbf_)}
            );
            tblis::mult<double>(1.0, t_Tao, "ijpr", t_eri_ao, "pqrs", 0.0, t_Wao, "ijqs");

            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < na_; ++i) {
                for (int j = 0; j < na_; ++j) {
                    Eigen::MatrixXd Wij_ao(nbf_, nbf_);
                    for (int nu = 0; nu < nbf_; ++nu)
                        for (int sig = 0; sig < nbf_; ++sig)
                            Wij_ao(nu, sig) = W_ao(i, j, nu, sig);
                    Eigen::MatrixXd Wij_vv = Cav.transpose() * Wij_ao * Cav;
                    for (int a = 0; a < va_; ++a)
                        for (int b = 0; b < va_; ++b)
                            W(i, j, a, b) += Wij_vv(a, b);
                }
            }
            
            if (Waa_ring_.size() != na_ * na_ * na_ * na_) Waa_ring_.resize(na_, na_, na_, na_);
            Waa_ring_ = integrals::ERITransformer::transform_custom(eri_ao_cached_local, Cao, Cao, Cao, Cao, nbf_, na_, na_, na_, na_);
            TBLIS_VIEW_4D(t_Voooo, Waa_ring_, na_, na_, na_, na_);
            tblis::mult< double >(1.0, t_T, "mnab", t_Voooo, "minj", 1.0, t_W, "ijab");

            auto* g_blk_mp3 = g_aa_.get_block(0,0,0,0);
            TBLIS_VIEW_4D(t_Vovov, (*g_blk_mp3), na_, va_, na_, va_);
            
            auto V_oovv = integrals::ERITransformer::transform_custom(eri_ao_cached_local, Cao, Cao, Cav, Cav, nbf_, na_, na_, va_, va_);
            TBLIS_VIEW_4D(t_Voovv, V_oovv, na_, na_, va_, va_);

            tblis::mult< double >(2.0,  t_Vovov, "iakc", t_T, "kjcb", 1.0, t_W, "ijab");
            tblis::mult< double >(-1.0, t_Voovv, "ikac", t_T, "kjcb", 1.0, t_W, "ijab"); //bisa ikca
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
            const auto& eri_ao_cached_local = integrals_->compute_eri();

            const Eigen::MatrixXd& Cbo = scf_.C_beta.leftCols(nb_);
            const Eigen::MatrixXd& Cbv = scf_.C_beta.rightCols(vb_);
            const auto& eb = scf_.orbital_energies_beta;

            auto get_V_exact = [&](const Eigen::MatrixXd& C1, const Eigen::MatrixXd& C2, const Eigen::MatrixXd& C3, const Eigen::MatrixXd& C4) {
                int d1 = C1.cols(); int d2 = C2.cols(); int d3 = C3.cols(); int d4 = C4.cols();
                return integrals::ERITransformer::transform_custom(eri_ao_cached_local, C1, C2, C3, C4, nbf_, d1, d2, d3, d4);
            };

            t2_3rd_aa_ = Eigen::Tensor< double, 4 >(na_, na_, va_, va_); t2_3rd_aa_.setZero();
            if (nb_ > 0 && vb_ > 0) {
                t2_3rd_bb_ = Eigen::Tensor< double, 4 >(nb_, nb_, vb_, vb_); t2_3rd_bb_.setZero();
                t2_3rd_ab_ = Eigen::Tensor< double, 4 >(na_, nb_, va_, vb_); t2_3rd_ab_.setZero();
            }

            Eigen::Tensor< double, 4 > Waa_ladder(na_, na_, va_, va_); Waa_ladder.setZero();
            Eigen::Tensor< double, 4 > Waa_ring(na_, na_, va_, va_); Waa_ring.setZero();
            Eigen::Tensor< double, 4 > Wbb_ladder(nb_, nb_, vb_, vb_); if (nb_ > 0) Wbb_ladder.setZero();
            Eigen::Tensor< double, 4 > Wbb_ring(nb_, nb_, vb_, vb_); if (nb_ > 0) Wbb_ring.setZero();
            Eigen::Tensor< double, 4 > Wab_ladder(na_, nb_, va_, vb_); if (nb_ > 0) Wab_ladder.setZero();
            Eigen::Tensor< double, 4 > Wab_ring(na_, nb_, va_, vb_); if (nb_ > 0) Wab_ring.setZero();

            TBLIS_VIEW_4D(t_Taa, T2_aa_ijab, na_, na_, va_, va_);
            TBLIS_VIEW_4D(t_Waa_ladder, Waa_ladder, na_, na_, va_, va_);
            TBLIS_VIEW_4D(t_Waa_ring, Waa_ring, na_, na_, va_, va_);

            auto* t2_bb_dense_ptr = t2_bb_.get_block(0,0,0,0);
            Eigen::Tensor< double, 4 > dummy_bb_local(nb_, nb_, vb_, vb_);
            if (!t2_bb_dense_ptr) {
                dummy_bb_local.setZero();
                t2_bb_dense_ptr = &dummy_bb_local;
            }
            
            auto* t2_ab_dense = t2_ab_.get_block(0,0,0,0);
            Eigen::Tensor< double, 4 > dummy_ab(na_, nb_, va_, vb_);
            if (!t2_ab_dense) {
                dummy_ab.setZero();
                t2_ab_dense = &dummy_ab;
            }

            TBLIS_VIEW_4D(t_Tbb, (*t2_bb_dense_ptr), nb_, nb_, vb_, vb_);
            TBLIS_VIEW_4D(t_Wbb_ladder, Wbb_ladder, nb_, nb_, vb_, vb_);
            TBLIS_VIEW_4D(t_Wbb_ring, Wbb_ring, nb_, nb_, vb_, vb_);
            
            TBLIS_VIEW_4D(t_Tab, (*t2_ab_dense), na_, nb_, va_, vb_);
            TBLIS_VIEW_4D(t_Wab_ladder, Wab_ladder, na_, nb_, va_, vb_);
            TBLIS_VIEW_4D(t_Wab_ring, Wab_ring, na_, nb_, va_, vb_);

            auto V_vvvv_aa = get_V_exact(Cav, Cav, Cav, Cav);
            TBLIS_VIEW_4D(t_Vvvvv_aa, V_vvvv_aa, va_, va_, va_, va_);
            tblis::mult< double >(1.0, t_Taa, "ijef", t_Vvvvv_aa, "eafb", 1.0, t_Waa_ladder, "ijab");

            if (nb_ > 0 && vb_ > 0) {
                auto V_vvvv_bb = get_V_exact(Cbv, Cbv, Cbv, Cbv);
                TBLIS_VIEW_4D(t_Vvvvv_bb, V_vvvv_bb, vb_, vb_, vb_, vb_);
                tblis::mult< double >(1.0, t_Tbb, "ijef", t_Vvvvv_bb, "eafb", 1.0, t_Wbb_ladder, "ijab");

                auto V_vvvv_ab = get_V_exact(Cav, Cav, Cbv, Cbv);
                TBLIS_VIEW_4D(t_Vvvvv_ab, V_vvvv_ab, va_, va_, vb_, vb_);
                tblis::mult< double >(1.0, t_Tab, "ijef", t_Vvvvv_ab, "eafb", 1.0, t_Wab_ladder, "ijab");
            }

            auto V_oooo_aa = get_V_exact(Cao, Cao, Cao, Cao);
            TBLIS_VIEW_4D(t_Voooo_aa, V_oooo_aa, na_, na_, na_, na_);
            tblis::mult< double >(1.0, t_Taa, "mnab", t_Voooo_aa, "minj", 1.0, t_Waa_ladder, "ijab");

            if (nb_ > 0 && vb_ > 0) {
                auto V_oooo_bb = get_V_exact(Cbo, Cbo, Cbo, Cbo);
                TBLIS_VIEW_4D(t_Voooo_bb, V_oooo_bb, nb_, nb_, nb_, nb_);
                tblis::mult< double >(1.0, t_Tbb, "mnab", t_Voooo_bb, "minj", 1.0, t_Wbb_ladder, "ijab");

                auto V_oooo_ab = get_V_exact(Cao, Cao, Cbo, Cbo);
                TBLIS_VIEW_4D(t_Voooo_ab, V_oooo_ab, na_, na_, nb_, nb_);
                tblis::mult< double >(1.0, t_Tab, "mnab", t_Voooo_ab, "minj", 1.0, t_Wab_ladder, "ijab");
            }

            auto V_ovov_aa = get_V_exact(Cao, Cav, Cao, Cav);
            auto V_oovv_aa = get_V_exact(Cao, Cao, Cav, Cav);
            TBLIS_VIEW_4D(t_Vovov_aa, V_ovov_aa, na_, va_, na_, va_);
            TBLIS_VIEW_4D(t_Voovv_aa, V_oovv_aa, na_, na_, va_, va_);

            tblis::mult< double >(1.0, t_Vovov_aa, "iakc", t_Taa, "kjcb", 1.0, t_Waa_ring, "ijab");
            tblis::mult< double >(-1.0, t_Voovv_aa, "ikac", t_Taa, "kjcb", 1.0, t_Waa_ring, "ijab");

            if (nb_ > 0 && vb_ > 0) {
                auto V_ovov_bb    = get_V_exact(Cbo, Cbv, Cbo, Cbv);
                auto V_oovv_bb    = get_V_exact(Cbo, Cbo, Cbv, Cbv);
                auto V_ovov_ab    = get_V_exact(Cao, Cav, Cbo, Cbv);
                auto V_oovv_ab_ex = get_V_exact(Cao, Cao, Cbv, Cbv);
                auto V_oovv_ba_ex = get_V_exact(Cbo, Cbo, Cav, Cav);

                TBLIS_VIEW_4D(t_Vovov_bb_vw, V_ovov_bb, nb_, vb_, nb_, vb_);
                TBLIS_VIEW_4D(t_Voovv_bb_vw, V_oovv_bb, nb_, nb_, vb_, vb_);
                TBLIS_VIEW_4D(t_Vovov_ab_vw, V_ovov_ab, na_, va_, nb_, vb_);
                TBLIS_VIEW_4D(t_Voovv_ab_ex_vw, V_oovv_ab_ex, na_, na_, vb_, vb_);
                TBLIS_VIEW_4D(t_Voovv_ba_ex_vw, V_oovv_ba_ex, nb_, nb_, va_, va_);

                tblis::mult< double >(1.0, t_Vovov_ab_vw, "iakc", t_Tab, "jkbc", 1.0, t_Waa_ring, "ijab");
                tblis::mult< double >(1.0, t_Vovov_bb_vw, "iakc", t_Tbb, "kjcb", 1.0, t_Wbb_ring, "ijab");
                tblis::mult< double >(-1.0, t_Voovv_bb_vw, "ikac", t_Tbb, "kjcb", 1.0, t_Wbb_ring, "ijab");
                tblis::mult< double >(1.0, t_Vovov_ab_vw, "kcia", t_Tab, "kjcb", 1.0, t_Wbb_ring, "ijab"); 
                tblis::mult< double >(1.0,  t_Vovov_aa, "iakc", t_Tab, "kjcb", 1.0, t_Wab_ring, "ijab");
                tblis::mult< double >(-1.0, t_Voovv_aa, "ikac", t_Tab, "kjcb", 1.0, t_Wab_ring, "ijab");
                tblis::mult< double >(1.0,  t_Tab, "ikac", t_Vovov_bb_vw, "kcjb", 1.0, t_Wab_ring, "ijab");
                tblis::mult< double >(-1.0, t_Tab, "ikac", t_Voovv_bb_vw, "kjcb", 1.0, t_Wab_ring, "ijab");
                tblis::mult< double >(1.0,  t_Taa, "ikac", t_Vovov_ab_vw, "kcjb", 1.0, t_Wab_ring, "ijab");
                tblis::mult< double >(1.0,  t_Vovov_ab_vw, "iakc", t_Tbb, "kjcb", 1.0, t_Wab_ring, "ijab");
                tblis::mult< double >(-1.0, t_Voovv_ab_ex_vw, "ikbc", t_Tab, "kjac", 1.0, t_Wab_ring, "ijab");
                tblis::mult< double >(-1.0, t_Tab, "ikcb", t_Voovv_ba_ex_vw, "jkac", 1.0, t_Wab_ring, "ijab"); 
            }

            double e3_aa = 0.0, e3_bb = 0.0, e3_ab = 0.0;
            
            #pragma omp parallel for collapse(4) reduction(+:e3_aa) schedule(static)
            for (int i = 0; i < na_; ++i) {
                for (int j = 0; j < na_; ++j) {
                    for (int a = 0; a < va_; ++a) {
                        for (int b = 0; b < va_; ++b) {
                            double l_asym = 0.5 * (Waa_ladder(i,j,a,b) - Waa_ladder(i,j,b,a));
                            double r_asym = Waa_ring(i,j,a,b) - Waa_ring(j,i,a,b) - Waa_ring(i,j,b,a) + Waa_ring(j,i,b,a);
                            double w_tot = l_asym + r_asym;
                            double D = ea(i) + ea(j) - ea(na_+a) - ea(na_+b);
                            t2_3rd_aa_(i,j,a,b) = w_tot / D;
                            e3_aa += 0.25 * T2_aa_ijab(i,j,a,b) * w_tot;
                        }
                    }
                }
            }

            if (nb_ > 0 && vb_ > 0) {
                #pragma omp parallel for collapse(4) reduction(+:e3_bb) schedule(static)
                for (int i = 0; i < nb_; ++i) {
                    for (int j = 0; j < nb_; ++j) {
                        for (int a = 0; a < vb_; ++a) {
                            for (int b = 0; b < vb_; ++b) {
                                double l_asym = 0.5 * (Wbb_ladder(i,j,a,b) - Wbb_ladder(i,j,b,a));
                                double r_asym = Wbb_ring(i,j,a,b) - Wbb_ring(j,i,a,b) - Wbb_ring(i,j,b,a) + Wbb_ring(j,i,b,a);
                                double w_tot = l_asym + r_asym;
                                double D = eb(i) + eb(j) - eb(nb_+a) - eb(nb_+b);
                                t2_3rd_bb_(i,j,a,b) = w_tot / D;
                                e3_bb += 0.25 * (*t2_bb_dense_ptr)(i,j,a,b) * w_tot;
                            }
                        }
                    }
                }

                #pragma omp parallel for collapse(4) reduction(+:e3_ab) schedule(static)
                for (int i = 0; i < na_; ++i) {
                    for (int j = 0; j < nb_; ++j) {
                        for (int a = 0; a < va_; ++a) {
                            for (int b = 0; b < vb_; ++b) {
                                double w_tot = Wab_ladder(i,j,a,b) + Wab_ring(i,j,a,b); 
                                double D = ea(i) + eb(j) - ea(na_+a) - eb(nb_+b);
                                t2_3rd_ab_(i,j,a,b) = w_tot / D;
                                e3_ab += 1.0 * (*t2_ab_dense)(i,j,a,b) * w_tot;
                            }
                        }
                    }
                }
            }
            
            e_mp3_tot_ = e3_aa + e3_bb + e3_ab;
            return;
        }
    }

    // =========================================================================
    // IMPLEMENTASI CHOLESKY & DENSITY FITTING (ZERO-COPY & PARALLEL GEMM)
    // =========================================================================
    int n_aux = scf_.L_mat.cols();

    // Helper: Parallel Tiled DGEMM (bebas dari throttling OPENBLAS/MKL_NUM_THREADS=1)
    auto parallel_gemm_abt = [](const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
        Eigen::MatrixXd C(A.rows(), B.rows());
        const int m = A.rows();
        constexpr int tile = 64;
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < m; i += tile) {
            int rows = std::min(tile, m - i);
            C.middleRows(i, rows).noalias() = A.middleRows(i, rows) * B.transpose();
        }
        return C;
    };

    // Helper: Zero-Copy 4D TBLIS View langsung di atas memori Eigen::MatrixXd (tanpa alokasi Tensor 4D)
    auto make_mat_view_4d = [](Eigen::MatrixXd& M, int d1, int d2, int d3, int d4) {
        return varray_view<double>(
            {(len_type)d1, (len_type)d2, (len_type)d3, (len_type)d4},
            M.data(),
            {(stride_type)d2, 1, (stride_type)(d1 * d2 * d4), (stride_type)(d1 * d2)}
        );
    };

    const Eigen::MatrixXd& B_ij_a = B_oo_P_alpha_;
    const Eigen::MatrixXd& B_ab_a = B_vv_P_alpha_;

    if (is_restricted) {
        t2_3rd_aa_ = Eigen::Tensor< double, 4 >(na_, na_, va_, va_);
        Eigen::Tensor< double, 4 > W(na_, na_, va_, va_); W.setZero();
        
        TBLIS_VIEW_4D(t_T, T2_aa_ijab, na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_W, W, na_, na_, va_, va_);
      
        Eigen::MatrixXd M_oooo = parallel_gemm_abt(B_ij_a, B_ij_a);
        Eigen::MatrixXd M_ovov = parallel_gemm_abt(B_ia_P_alpha_, B_ia_P_alpha_);
        Eigen::MatrixXd M_oovv = parallel_gemm_abt(B_ij_a, B_ab_a);
        
        auto t_Voooo = make_mat_view_4d(M_oooo, na_, na_, na_, na_);
        auto t_Vovov = make_mat_view_4d(M_ovov, na_, va_, na_, va_);
        auto t_Voovv = make_mat_view_4d(M_oovv, na_, na_, va_, va_);

        tblis::mult< double >(1.0,  t_T, "mnab", t_Voooo, "minj", 1.0, t_W, "ijab");
        tblis::mult< double >(2.0,  t_Vovov, "iakc", t_T, "kjcb", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_Voovv, "ikac", t_T, "kjcb", 1.0, t_W, "ijab");
        tblis::mult< double >(2.0,  t_T, "ikac", t_Vovov, "kcjb", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_T, "ikac", t_Voovv, "kjcb", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_T, "ikca", t_Vovov, "kcjb", 1.0, t_W, "ijab"); 
        tblis::mult< double >(-1.0, t_Vovov, "iakc", t_T, "kjbc", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_Voovv, "ikbc", t_T, "kjac", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_T, "ikcb", t_Voovv, "jkac", 1.0, t_W, "ijab"); 

        // Eksploitasi simetri W_ji(a,b) = W_ij(b,a): hanya hitung pasangan unik i <= j!
        int n_pairs_sym = na_ * (na_ + 1) / 2;
        std::vector<Eigen::MatrixXd> T_pairs(n_pairs_sym, Eigen::MatrixXd(va_, va_));
        std::vector<Eigen::MatrixXd> W_pairs(n_pairs_sym, Eigen::MatrixXd::Zero(va_, va_));

        int p_idx = 0;
        for (int i = 0; i < na_; ++i) {
            for (int j = i; j < na_; ++j) {
                for (int e = 0; e < va_; ++e)
                    for (int f = 0; f < va_; ++f)
                        T_pairs[p_idx](e, f) = T2_aa_ijab(i, j, e, f);
                p_idx++;
            }
        }

        #pragma omp parallel
        {
            std::vector<Eigen::MatrixXd> W_loc(n_pairs_sym, Eigen::MatrixXd::Zero(va_, va_));
            Eigen::MatrixXd X_tmp(va_, va_);

            #pragma omp for schedule(static) nowait
            for (int P = 0; P < n_aux; ++P) {
                Eigen::Map<const Eigen::MatrixXd> B_P(B_ab_a.col(P).data(), va_, va_);
                for (int k = 0; k < n_pairs_sym; ++k) {
                    X_tmp.noalias() = T_pairs[k] * B_P;
                    W_loc[k].noalias() += B_P.transpose() * X_tmp;
                }
            }

            #pragma omp critical
            {
                for (int k = 0; k < n_pairs_sym; ++k) W_pairs[k] += W_loc[k];
            }
        }

        p_idx = 0;
        for (int i = 0; i < na_; ++i) {
            for (int j = i; j < na_; ++j) {
                for (int a = 0; a < va_; ++a) {
                    for (int b = 0; b < va_; ++b) {
                        W(i, j, a, b) += W_pairs[p_idx](a, b);
                        if (i != j) W(j, i, b, a) += W_pairs[p_idx](a, b);
                    }
                }
                p_idx++;
            }
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

    Eigen::MatrixXd M_oooo_aa = parallel_gemm_abt(B_ij_a, B_ij_a);
    Eigen::MatrixXd M_ovov_aa = parallel_gemm_abt(B_ia_P_alpha_, B_ia_P_alpha_);
    Eigen::MatrixXd M_oovv_aa = parallel_gemm_abt(B_ij_a, B_ab_a);
    Eigen::MatrixXd M_ovov_ab;
    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        M_ovov_ab = parallel_gemm_abt(B_ia_P_alpha_, B_ia_P_beta_);
    }

    {
        if (Waa_ladder_.size() == 0) Waa_ladder_.resize(na_, na_, va_, va_);
        if (Waa_ring_.size() == 0) Waa_ring_.resize(na_, na_, va_, va_);
        
        Waa_ladder_.setZero(); Waa_ring_.setZero();
        TBLIS_VIEW_4D(t_Waa_ladder, Waa_ladder_, na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_Waa_ring, Waa_ring_, na_, na_, va_, va_);
        
        auto t_Voooo = make_mat_view_4d(M_oooo_aa, na_, na_, na_, na_);
        auto t_Vovov = make_mat_view_4d(M_ovov_aa, na_, va_, na_, va_);
        auto t_Voovv = make_mat_view_4d(M_oovv_aa, na_, na_, va_, va_);

        tblis::mult< double >(0.5,  t_Taa, "mnab", t_Voooo, "minj", 1.0, t_Waa_ladder, "ijab");
        tblis::mult< double >(-0.5, t_Taa, "mnab", t_Voooo, "mjni", 1.0, t_Waa_ladder, "ijab");

        tblis::mult< double >(1.0,  t_Vovov, "iakc", t_Taa, "kjcb", 1.0, t_Waa_ring, "ijab");
        tblis::mult< double >(-1.0, t_Voovv, "ikac", t_Taa, "kjcb", 1.0, t_Waa_ring, "ijab");

        if (!is_restricted && nb_ > 0 && vb_ > 0) {
            auto t_Vovov_ab = make_mat_view_4d(M_ovov_ab, na_, va_, nb_, vb_);
            TBLIS_VIEW_4D(t_Tab, (*t2_ab_dense), na_, nb_, va_, vb_);
            tblis::mult< double >(1.0, t_Vovov_ab, "iakc", t_Tab, "jkbc", 1.0, t_Waa_ring, "ijab");
        }

        // Antisimetri AA: hanya hitung pasangan i < j!
        int n_pairs_aa = na_ * (na_ - 1) / 2;
        if (n_pairs_aa > 0) {
            std::vector<Eigen::MatrixXd> T_aa_pairs(n_pairs_aa, Eigen::MatrixXd(va_, va_));
            std::vector<Eigen::MatrixXd> W_aa_pairs(n_pairs_aa, Eigen::MatrixXd::Zero(va_, va_));
            int p_idx = 0;
            for (int i = 0; i < na_; ++i) {
                for (int j = i + 1; j < na_; ++j) {
                    for (int e = 0; e < va_; ++e)
                        for (int f = 0; f < va_; ++f)
                            T_aa_pairs[p_idx](e, f) = T2_aa_ijab(i, j, e, f);
                    p_idx++;
                }
            }

            #pragma omp parallel for schedule(static)
            for (int k = 0; k < n_pairs_aa; ++k) {
                Eigen::MatrixXd X_tmp(va_, va_);
                for (int P = 0; P < n_aux; ++P) {
                    Eigen::Map<const Eigen::MatrixXd> B_P(B_ab_a.col(P).data(), va_, va_);
                    X_tmp.noalias() = T_aa_pairs[k] * B_P;
                    W_aa_pairs[k].noalias() += B_P.transpose() * X_tmp;
                }
            }

            p_idx = 0;
            for (int i = 0; i < na_; ++i) {
                for (int j = i + 1; j < na_; ++j) {
                    for (int a = 0; a < va_; ++a) {
                        for (int b = 0; b < va_; ++b) {
                            double val = 0.5 * (W_aa_pairs[p_idx](a, b) - W_aa_pairs[p_idx](b, a));
                            Waa_ladder_(i, j, a, b) += val;
                            Waa_ladder_(j, i, a, b) -= val;
                        }
                    }
                    p_idx++;
                }
            }
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
        Waa_ladder_.resize(0, 0, 0, 0);
        Waa_ring_.resize(0, 0, 0, 0);
        M_oooo_aa.resize(0, 0);
    }

    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        const auto& eb = scf_.orbital_energies_beta;
        const Eigen::MatrixXd& B_ij_b = B_oo_P_beta_;
        const Eigen::MatrixXd& B_ab_b = B_vv_P_beta_;
     
        TBLIS_VIEW_4D(t_Tbb, (*t2_bb_dense_ptr), nb_, nb_, vb_, vb_);
        TBLIS_VIEW_4D(t_Tab, (*t2_ab_dense), na_, nb_, va_, vb_);

        Eigen::MatrixXd M_oooo_bb = parallel_gemm_abt(B_ij_b, B_ij_b);
        Eigen::MatrixXd M_ovov_bb = parallel_gemm_abt(B_ia_P_beta_, B_ia_P_beta_);
        Eigen::MatrixXd M_oovv_bb = parallel_gemm_abt(B_ij_b, B_ab_b);

        {
            Eigen::Tensor< double, 4 > Wbb_ladder(nb_, nb_, vb_, vb_); Wbb_ladder.setZero();
            Eigen::Tensor< double, 4 > Wbb_ring(nb_, nb_, vb_, vb_); Wbb_ring.setZero();
            
            TBLIS_VIEW_4D(t_Wbb_ladder, Wbb_ladder, nb_, nb_, vb_, vb_);
            TBLIS_VIEW_4D(t_Wbb_ring, Wbb_ring, nb_, nb_, vb_, vb_);
            auto t_Voooo_bb = make_mat_view_4d(M_oooo_bb, nb_, nb_, nb_, nb_);
            auto t_Vovov_bb = make_mat_view_4d(M_ovov_bb, nb_, vb_, nb_, vb_);
            auto t_Voovv_bb = make_mat_view_4d(M_oovv_bb, nb_, nb_, vb_, vb_);
            auto t_Vovov_ab = make_mat_view_4d(M_ovov_ab, na_, va_, nb_, vb_);

            if (nb_ > 1) {
                tblis::mult< double >(0.5,  t_Tbb, "mnab", t_Voooo_bb, "minj", 1.0, t_Wbb_ladder, "ijab");
                tblis::mult< double >(-0.5, t_Tbb, "mnab", t_Voooo_bb, "mjni", 1.0, t_Wbb_ladder, "ijab");
         
                tblis::mult< double >(1.0,  t_Vovov_bb, "iakc", t_Tbb, "kjcb", 1.0, t_Wbb_ring, "ijab");
                tblis::mult< double >(-1.0, t_Voovv_bb, "ikac", t_Tbb, "kjcb", 1.0, t_Wbb_ring, "ijab");
                tblis::mult< double >(1.0,  t_Vovov_ab, "kcia", t_Tab, "kjcb", 1.0, t_Wbb_ring, "ijab"); 
            }

            // Antisimetri BB: hanya hitung jika nb_ > 1 untuk pasangan i < j!
            // Antisimetri BB: hanya hitung jika nb_ > 1 untuk pasangan i < j!
            int n_pairs_bb = nb_ * (nb_ - 1) / 2;
            if (n_pairs_bb > 0) {
                std::vector<Eigen::MatrixXd> T_bb_pairs(n_pairs_bb, Eigen::MatrixXd(vb_, vb_));
                std::vector<Eigen::MatrixXd> W_bb_pairs(n_pairs_bb, Eigen::MatrixXd::Zero(vb_, vb_));
                int p_idx = 0;
                for (int i = 0; i < nb_; ++i) {
                    for (int j = i + 1; j < nb_; ++j) {
                        for (int e = 0; e < vb_; ++e)
                            for (int f = 0; f < vb_; ++f)
                                T_bb_pairs[p_idx](e, f) = (*t2_bb_dense_ptr)(i, j, e, f);
                        p_idx++;
                    }
                }

                #pragma omp parallel for schedule(static)
                for (int k = 0; k < n_pairs_bb; ++k) {
                    Eigen::MatrixXd X_tmp(vb_, vb_);
                    for (int P = 0; P < n_aux; ++P) {
                        Eigen::Map<const Eigen::MatrixXd> B_P(B_ab_b.col(P).data(), vb_, vb_);
                        X_tmp.noalias() = T_bb_pairs[k] * B_P;
                        W_bb_pairs[k].noalias() += B_P.transpose() * X_tmp;
                    }
                }

                p_idx = 0;
                for (int i = 0; i < nb_; ++i) {
                    for (int j = i + 1; j < nb_; ++j) {
                        for (int a = 0; a < vb_; ++a) {
                            for (int b = 0; b < vb_; ++b) {
                                double val = 0.5 * (W_bb_pairs[p_idx](a, b) - W_bb_pairs[p_idx](b, a));
                                Wbb_ladder(i, j, a, b) += val;
                                Wbb_ladder(j, i, a, b) -= val;
                            }
                        }
                        p_idx++;
                    }
                }
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

            Eigen::MatrixXd M_oooo_ab = parallel_gemm_abt(B_ij_a, B_ij_b);
            auto t_Voooo_ab = make_mat_view_4d(M_oooo_ab, na_, na_, nb_, nb_);
            tblis::mult< double >(1.0, t_Tab, "mnab", t_Voooo_ab, "minj", 1.0, t_Wab_ladder, "ijab");

            auto t_Vovov_aa = make_mat_view_4d(M_ovov_aa, na_, va_, na_, va_);
            auto t_Voovv_aa = make_mat_view_4d(M_oovv_aa, na_, na_, va_, va_);
            tblis::mult< double >(1.0,  t_Vovov_aa, "iakc", t_Tab, "kjcb", 1.0, t_Wab_ring, "ijab");
            tblis::mult< double >(-1.0, t_Voovv_aa, "ikac", t_Tab, "kjcb", 1.0, t_Wab_ring, "ijab");

            auto t_Vovov_bb = make_mat_view_4d(M_ovov_bb, nb_, vb_, nb_, vb_);
            auto t_Voovv_bb = make_mat_view_4d(M_oovv_bb, nb_, nb_, vb_, vb_);
            tblis::mult< double >(1.0,  t_Tab, "ikac", t_Vovov_bb, "kcjb", 1.0, t_Wab_ring, "ijab"); 
            tblis::mult< double >(-1.0, t_Tab, "ikac", t_Voovv_bb, "kjcb", 1.0, t_Wab_ring, "ijab");

            auto t_Vovov_ab = make_mat_view_4d(M_ovov_ab, na_, va_, nb_, vb_);
            tblis::mult< double >(1.0,  t_Taa, "ikac", t_Vovov_ab, "kcjb", 1.0, t_Wab_ring, "ijab");
            if (nb_ > 1) {
                tblis::mult< double >(1.0,  t_Vovov_ab, "iakc", t_Tbb, "kjcb", 1.0, t_Wab_ring, "ijab");
            }

            Eigen::MatrixXd M_oovv_ab_ex = parallel_gemm_abt(B_ij_a, B_ab_b);
            Eigen::MatrixXd M_oovv_ba_ex = parallel_gemm_abt(B_ij_b, B_ab_a);
            auto t_Voovv_ab_ex = make_mat_view_4d(M_oovv_ab_ex, na_, na_, vb_, vb_);
            auto t_Voovv_ba_ex = make_mat_view_4d(M_oovv_ba_ex, nb_, nb_, va_, va_);

            tblis::mult< double >(-1.0, t_Voovv_ab_ex, "ikbc", t_Tab, "kjac", 1.0, t_Wab_ring, "ijab");
            tblis::mult< double >(-1.0, t_Tab, "ikcb", t_Voovv_ba_ex, "jkac", 1.0, t_Wab_ring, "ijab");
            M_oovv_ab_ex.resize(0, 0);
            M_oovv_ba_ex.resize(0, 0);

            int n_pairs_ab = na_ * nb_;
            std::vector<Eigen::MatrixXd> T_ab_pairs(n_pairs_ab, Eigen::MatrixXd(va_, vb_));
            std::vector<Eigen::MatrixXd> W_ab_pairs(n_pairs_ab, Eigen::MatrixXd::Zero(va_, vb_));
            for (int i = 0; i < na_; ++i) {
                for (int j = 0; j < nb_; ++j) {
                    int k = i * nb_ + j;
                    for (int e = 0; e < va_; ++e)
                        for (int f = 0; f < vb_; ++f)
                            T_ab_pairs[k](e, f) = (*t2_ab_dense)(i, j, e, f);
                }
            }

            #pragma omp parallel for schedule(static)
            for (int k = 0; k < n_pairs_ab; ++k) {
                Eigen::MatrixXd X_tmp(va_, vb_);
                for (int P = 0; P < n_aux; ++P) {
                    Eigen::Map<const Eigen::MatrixXd> B_Pa(B_ab_a.col(P).data(), va_, va_);
                    Eigen::Map<const Eigen::MatrixXd> B_Pb(B_ab_b.col(P).data(), vb_, vb_);
                    X_tmp.noalias() = T_ab_pairs[k] * B_Pb;
                    W_ab_pairs[k].noalias() += B_Pa.transpose() * X_tmp;
                }
            }

            for (int i = 0; i < na_; ++i) {
                for (int j = 0; j < nb_; ++j) {
                    int k = i * nb_ + j;
                    for (int a = 0; a < va_; ++a)
                        for (int b = 0; b < vb_; ++b)
                            Wab_ladder(i, j, a, b) += W_ab_pairs[k](a, b);
                }
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
        tblis::mult<double>(-1.0, t_T2, "ikab", t_L2t, "jkab", 1.0, t_Goo, "ij"); 
        tblis::mult<double>(-1.0, t_L2, "ikab", t_T2t, "jkab", 1.0, t_Goo, "ij");  

        tblis::mult<double>(1.0, t_T2, "ijac", t_T2t, "ijbc", 0.0, t_Gvv, "ab"); 
        tblis::mult<double>(1.0, t_T2, "ijac", t_L2t, "ijbc", 1.0, t_Gvv, "ab"); 
        tblis::mult<double>(1.0, t_L2, "ijac", t_T2t, "ijbc", 1.0, t_Gvv, "ab"); 
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

    auto* t2_ab_dense = t2_ab_.get_block(0,0,0,0);
    if (nb_ > 0 && vb_ > 0 && t2_ab_dense) {
        TBLIS_VIEW_4D(t_T2ab, (*t2_ab_dense), na_, nb_, va_, vb_);
        TBLIS_VIEW_4D(t_T3ab, L2_ab_, na_, nb_, va_, vb_);
        
        tblis::mult<double>(-1.0, t_T2ab, "ikab", t_T2ab, "jkab", 1.0, t_Goo_a, "ij"); 
        tblis::mult<double>(-1.0, t_T2ab, "ikab", t_T3ab, "jkab", 1.0, t_Goo_a, "ij"); 
        tblis::mult<double>(-1.0, t_T3ab, "ikab", t_T2ab, "jkab", 1.0, t_Goo_a, "ij"); 
        
        tblis::mult<double>(1.0, t_T2ab, "ijac", t_T2ab, "ijbc", 1.0, t_Gvv_a, "ab"); 
        tblis::mult<double>(1.0, t_T2ab, "ijac", t_T3ab, "ijbc", 1.0, t_Gvv_a, "ab");  
        tblis::mult<double>(1.0, t_T3ab, "ijac", t_T2ab, "ijbc", 1.0, t_Gvv_a, "ab"); 
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
    auto* t2_bb_dense = t2_bb_.get_block(0, 0, 0, 0);
    
    Eigen::Tensor<double, 4> dummy_bb;
    if (!t2_bb_dense) {
        dummy_bb = Eigen::Tensor<double, 4>(nb_, nb_, vb_, vb_); 
        dummy_bb.setZero();
        t2_bb_dense = &dummy_bb;
    }
    
    auto* t2_ab_dense = t2_ab_.get_block(0, 0, 0, 0);
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

    tblis::mult<double>(-0.5, t_T2bb, "ikab", t_T2bb, "jkab", 1.0, t_Goo_b, "ij"); 
    tblis::mult<double>(-0.5, t_T2bb, "ikab", t_T3bb, "jkab", 1.0, t_Goo_b, "ij"); 
    tblis::mult<double>(-0.5, t_T3bb, "ikab", t_T2bb, "jkab", 1.0, t_Goo_b, "ij"); 

    tblis::mult<double>(0.5, t_T2bb, "ijac", t_T2bb, "ijbc", 1.0, t_Gvv_b, "ab"); 
    tblis::mult<double>(0.5, t_T2bb, "ijac", t_T3bb, "ijbc", 1.0, t_Gvv_b, "ab");  
    tblis::mult<double>(0.5, t_T3bb, "ijac", t_T2bb, "ijbc", 1.0, t_Gvv_b, "ab"); 
    
    tblis::mult<double>(-1.0, t_T2ab, "kiab", t_T2ab, "kjab", 1.0, t_Goo_b, "ij"); 
    tblis::mult<double>(-1.0, t_T2ab, "kiab", t_T3ab, "kjab", 1.0, t_Goo_b, "ij");  
    tblis::mult<double>(-1.0, t_T3ab, "kiab", t_T2ab, "kjab", 1.0, t_Goo_b, "ij");  
        
    tblis::mult<double>(1.0, t_T2ab, "ijca", t_T2ab, "ijcb", 1.0, t_Gvv_b, "ab"); 
    tblis::mult<double>(1.0, t_T2ab, "ijca", t_T3ab, "ijcb", 1.0, t_Gvv_b, "ab");   
    tblis::mult<double>(1.0, t_T3ab, "ijca", t_T2ab, "ijcb", 1.0, t_Gvv_b, "ab");
}
void OMP3::build_generalized_fock() {
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);

    const Eigen::MatrixXd& Cao = scf_.C_alpha.leftCols(na_);
    const Eigen::MatrixXd& Cav = scf_.C_alpha.rightCols(va_);
    Eigen::MatrixXd Z_mat_a = Eigen::MatrixXd::Zero(va_, na_);
    
    const Eigen::MatrixXd& Cbo = scf_.C_beta.leftCols(nb_);
    const Eigen::MatrixXd& Cbv = scf_.C_beta.rightCols(vb_);
    Eigen::MatrixXd Z_mat_b = Eigen::MatrixXd::Zero(vb_, nb_);

    auto* t_aa_dense = t2_aa_.get_block(0,0,0,0);
    if (!t_aa_dense) return;

    // =========================================================================
    // 1. Z-VECTOR DASAR (DARI Teff OMP2 + OMP3)
    // =========================================================================
    Eigen::Tensor< double, 4 > Teff_aa(na_, na_, va_, va_);
    #pragma omp parallel for collapse(4) schedule(static)
    for(int i = 0; i < na_; ++i) {
        for(int j = 0; j < na_; ++j) {
            for(int a = 0; a < va_; ++a) {
                for(int b = 0; b < va_; ++b) {
                    if (is_restricted) {
                        double t_tilde = 2.0 * (*t_aa_dense)(i,a,j,b) - (*t_aa_dense)(i,b,j,a);
                        double l_tilde = 2.0 * L2_aa_(i,j,a,b) - L2_aa_(i,j,b,a);
                        Teff_aa(i,j,a,b) = t_tilde + l_tilde;
                    } else {
                        Teff_aa(i,j,a,b) = (*t_aa_dense)(i,a,j,b) + L2_aa_(i,j,a,b);
                    }
                }
            }
        }
    }
    TBLIS_VIEW_4D(t_Teff_aa, Teff_aa, na_, na_, va_, va_);
    TBLIS_VIEW_2D(t_Zmat_a, Z_mat_a.data(), va_, na_);

    auto parallel_gemm_abt = [](const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
        Eigen::MatrixXd C(A.rows(), B.rows());
        const int m = A.rows();
        constexpr int tile = 64;
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < m; i += tile) {
            int rows = std::min(tile, m - i);
            C.middleRows(i, rows).noalias() = A.middleRows(i, rows) * B.transpose();
        }
        return C;
    };

    auto make_mat_view_4d = [](Eigen::MatrixXd& M, int d1, int d2, int d3, int d4) {
        return varray_view<double>(
            {(len_type)d1, (len_type)d2, (len_type)d3, (len_type)d4},
            M.data(),
            {(stride_type)d2, 1, (stride_type)(d1 * d2 * d4), (stride_type)(d1 * d2)}
        );
    };

    Eigen::Tensor<double, 4> V_ovvv_aa, V_ooov_aa;
    Eigen::MatrixXd M_ovvv_aa, M_ooov_aa;
    const Eigen::MatrixXd& B_ij_a = B_oo_P_alpha_;
    const Eigen::MatrixXd& B_ab_a = B_vv_P_alpha_;

    if (config_.eri_method == "exact") {
        const auto& eri_ao_ref = integrals_->compute_eri();
        V_ovvv_aa = integrals::ERITransformer::transform_custom(eri_ao_ref, Cao, Cav, Cav, Cav, nbf_, na_, va_, va_, va_);
        V_ooov_aa = integrals::ERITransformer::transform_custom(eri_ao_ref, Cao, Cao, Cao, Cav, nbf_, na_, na_, na_, va_);
    } else {
        M_ovvv_aa = parallel_gemm_abt(B_ia_P_alpha_, B_ab_a);
        M_ooov_aa = parallel_gemm_abt(B_ij_a, B_ia_P_alpha_);
    }

    varray_view<double> t_Vovvv_aa = (config_.eri_method == "exact")
        ? varray_view<double>({(len_type)na_, (len_type)va_, (len_type)va_, (len_type)va_}, V_ovvv_aa.data(), {1, (stride_type)na_, (stride_type)(na_*va_), (stride_type)(na_*va_*va_)})
        : make_mat_view_4d(M_ovvv_aa, na_, va_, va_, va_);
    varray_view<double> t_Vooov_aa = (config_.eri_method == "exact")
        ? varray_view<double>({(len_type)na_, (len_type)na_, (len_type)na_, (len_type)va_}, V_ooov_aa.data(), {1, (stride_type)na_, (stride_type)(na_*na_), (stride_type)(na_*na_*na_)})
        : make_mat_view_4d(M_ooov_aa, na_, na_, na_, va_);

    tblis::mult< double >(1.0, t_Vovvv_aa, "kcab", t_Teff_aa, "ikbc", 0.0, t_Zmat_a, "ai"); 
    tblis::mult< double >(-1.0, t_Vooov_aa, "jikc", t_Teff_aa, "jkac", 1.0, t_Zmat_a, "ai");

    // =========================================================================
    // 2. KONTRIBUSI EKSPLISIT TPDM ORDE KE-3 (VVVV, OOOO, RING)
    // =========================================================================
    Eigen::Tensor<double, 4> T2_aa(na_, na_, va_, va_);
    Eigen::Tensor<double, 4> T2_tilde_aa(na_, na_, va_, va_);
    #pragma omp parallel for collapse(4) schedule(static)
    for(int i = 0; i < na_; ++i) {
        for(int j = 0; j < na_; ++j) {
            for(int a = 0; a < va_; ++a) {
                for(int b = 0; b < va_; ++b) {
                    T2_aa(i,j,a,b) = (*t_aa_dense)(i,a,j,b);
                    if (is_restricted) T2_tilde_aa(i,j,a,b) = 2.0 * (*t_aa_dense)(i,a,j,b) - (*t_aa_dense)(i,b,j,a);
                    else T2_tilde_aa(i,j,a,b) = (*t_aa_dense)(i,a,j,b); 
                }
            }
        }
    }
    TBLIS_VIEW_4D(t_T2, T2_aa, na_, na_, va_, va_);
    TBLIS_VIEW_4D(t_T2t, T2_tilde_aa, na_, na_, va_, va_);

    // --- A. KONTRIBUSI VVVV (Virtual Ladder) - Faktorisasi O(O^3 V^3) Tanpa Tensor G_vvvv! ---
    double f_ladd = is_restricted ? 0.25 : 0.125;
    Eigen::Tensor<double, 4> Y_ooov_aa(na_, na_, na_, va_); Y_ooov_aa.setZero();
    TBLIS_VIEW_4D(t_Yooov_aa, Y_ooov_aa, na_, na_, na_, va_);

    // Y(i,j,m,f) = sum_{a,b} T2_tilde(i,j,a,b) * V_ovvv(m,a,f,b)
    tblis::mult<double>(1.0, t_T2t, "ijab", t_Vovvv_aa, "mafb", 0.0, t_Yooov_aa, "ijmf");
    tblis::mult<double>(-4.0 * f_ladd, t_T2, "ijxf", t_Yooov_aa, "ijmf", 1.0, t_Zmat_a, "xm");

    // --- B. KONTRIBUSI OOOO (Occupied Ladder) 
    Eigen::Tensor<double, 4> G_oooo_aa(na_, na_, na_, na_); G_oooo_aa.setZero();
    TBLIS_VIEW_4D(t_Goooo_aa, G_oooo_aa, na_, na_, na_, na_);
    tblis::mult<double>(f_ladd, t_T2t, "ijab", t_T2, "mnab", 0.0, t_Goooo_aa, "minj");
    tblis::mult<double>(4.0, t_Goooo_aa, "minj", t_Vooov_aa, "njix", 1.0, t_Zmat_a, "xm");

    // --- C. KONTRIBUSI RING (OVOV & OOVV) ---
    Eigen::Tensor<double, 4> G_ovov_aa(na_, va_, na_, va_); G_ovov_aa.setZero();
    Eigen::Tensor<double, 4> G_oovv_aa(na_, na_, va_, va_); G_oovv_aa.setZero();
    TBLIS_VIEW_4D(t_Govov_aa, G_ovov_aa, na_, va_, na_, va_);
    TBLIS_VIEW_4D(t_Goovv_aa, G_oovv_aa, na_, na_, va_, va_);

    if (is_restricted) {
        tblis::mult<double>(0.50,  t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Govov_aa, "iakc");
        tblis::mult<double>(-0.25, t_T2, "kjbc", t_T2t, "ijab", 1.0, t_Govov_aa, "iakc");
        tblis::mult<double>(0.50,  t_T2, "miea", t_T2t, "mkec", 1.0, t_Govov_aa, "iakc");
        tblis::mult<double>(-0.25, t_T2, "miae", t_T2t, "mkec", 1.0, t_Govov_aa, "iakc");

        tblis::mult<double>(-0.25, t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Goovv_aa, "ikac");
        tblis::mult<double>(-0.25, t_T2, "miea", t_T2t, "mkec", 1.0, t_Goovv_aa, "ikac");
        tblis::mult<double>(-0.50, t_T2, "kmec", t_T2t, "imea", 1.0, t_Goovv_aa, "ikac");
    } else {
        tblis::mult<double>(0.5,  t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Govov_aa, "iakc");
        tblis::mult<double>(-0.5, t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Goovv_aa, "ikac");
        if (nb_ > 0 && vb_ > 0) {
            auto* t2_ab_dense_tmp = t2_ab_.get_block(0,0,0,0);
            if (t2_ab_dense_tmp) {
                TBLIS_VIEW_4D(t_T2_ab_tmp, (*t2_ab_dense_tmp), na_, nb_, va_, vb_);
                tblis::mult<double>(0.5,  t_T2_ab_tmp, "kjcb", t_T2_ab_tmp, "ijab", 1.0, t_Govov_aa, "iakc");
                tblis::mult<double>(-0.5, t_T2_ab_tmp, "kjcb", t_T2_ab_tmp, "ijab", 1.0, t_Goovv_aa, "ikac");
            }
        }
    }

    if (is_restricted) {
        // Karena G_ovov_aa simetris terhadap (i,a) <-> (k,c), 4 kontraksi cukup jadi 2:
        tblis::mult<double>(2.0,  t_Govov_aa, "makc", t_Vovvv_aa, "kcxa", 1.0, t_Zmat_a, "xm");
        tblis::mult<double>(-2.0, t_Govov_aa, "ixkc", t_Vooov_aa, "imkc", 1.0, t_Zmat_a, "xm");
    } else {
        tblis::mult<double>(1.0,  t_Govov_aa, "makc", t_Vovvv_aa, "kcxa", 1.0, t_Zmat_a, "xm");
        tblis::mult<double>(-1.0, t_Govov_aa, "ixkc", t_Vooov_aa, "imkc", 1.0, t_Zmat_a, "xm");
        tblis::mult<double>(1.0,  t_Govov_aa, "iamc", t_Vovvv_aa, "iaxc", 1.0, t_Zmat_a, "xm");
        tblis::mult<double>(-1.0, t_Govov_aa, "iakx", t_Vooov_aa, "kmia", 1.0, t_Zmat_a, "xm");
    }

    tblis::mult<double>(1.0,  t_Goovv_aa, "mkac", t_Vovvv_aa, "kxca", 1.0, t_Zmat_a, "xm");
    tblis::mult<double>(1.0,  t_Goovv_aa, "imac", t_Vovvv_aa, "ixac", 1.0, t_Zmat_a, "xm");
    tblis::mult<double>(-1.0, t_Goovv_aa, "ikxc", t_Vooov_aa, "ikmc", 1.0, t_Zmat_a, "xm");
    tblis::mult<double>(-1.0, t_Goovv_aa, "ikax", t_Vooov_aa, "ikma", 1.0, t_Zmat_a, "xm");
    M_ovvv_aa.resize(0, 0); M_ooov_aa.resize(0, 0);
    V_ovvv_aa.resize(0, 0, 0, 0); V_ooov_aa.resize(0, 0, 0, 0);
    Teff_aa.resize(0, 0, 0, 0); T2_tilde_aa.resize(0, 0, 0, 0);
    Y_ooov_aa.resize(0, 0, 0, 0); G_oooo_aa.resize(0, 0, 0, 0);
    G_ovov_aa.resize(0, 0, 0, 0); G_oovv_aa.resize(0, 0, 0, 0);

    // =========================================================================
    // 3. UNRESTRICTED EXTENSION (BB & AB COMPONENTS)
    // =========================================================================
    if (!is_restricted && nb_ > 0 && vb_ > 0) {
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

        TBLIS_VIEW_2D(t_Zmat_b, Z_mat_b.data(), vb_, nb_);

        Eigen::Tensor< double, 4 > Teff_bb(nb_, nb_, vb_, vb_);
        Eigen::Tensor< double, 4 > Teff_ab(na_, nb_, va_, vb_);

        #pragma omp parallel for collapse(4) schedule(static)
        for(int i=0; i<nb_; ++i) for(int j=0; j<nb_; ++j) for(int a=0; a<vb_; ++a) for(int b=0; b<vb_; ++b)
            Teff_bb(i,j,a,b) = (*t2_bb_dense)(i,j,a,b) + L2_bb_(i,j,a,b);
        #pragma omp parallel for collapse(4) schedule(static)
        for(int i=0; i<na_; ++i) for(int j=0; j<nb_; ++j) for(int a=0; a<va_; ++a) for(int b=0; b<vb_; ++b)
            Teff_ab(i,j,a,b) = (*t2_ab_dense)(i,j,a,b) + L2_ab_(i,j,a,b);

        TBLIS_VIEW_4D(t_Teff_bb, Teff_bb, nb_, nb_, vb_, vb_);
        TBLIS_VIEW_4D(t_Teff_ab, Teff_ab, na_, nb_, va_, vb_);

        Eigen::Tensor<double, 4> V_ovvv_bb, V_ooov_bb, V_ovvv_ab, V_ovvv_ba, V_ooov_aa_bb, V_ooov_bb_aa;
        Eigen::MatrixXd M_ovvv_bb, M_ooov_bb, M_ovvv_ab, M_ovvv_ba, M_ooov_aa_bb, M_ooov_bb_aa;

        if (config_.eri_method == "exact") {
            const auto& eri_ao_ref = integrals_->compute_eri();
            V_ovvv_bb = integrals::ERITransformer::transform_custom(eri_ao_ref, Cbo, Cbv, Cbv, Cbv, nbf_, nb_, vb_, vb_, vb_);
            V_ooov_bb = integrals::ERITransformer::transform_custom(eri_ao_ref, Cbo, Cbo, Cbo, Cbv, nbf_, nb_, nb_, nb_, vb_);
            V_ovvv_ab = integrals::ERITransformer::transform_custom(eri_ao_ref, Cao, Cav, Cbv, Cbv, nbf_, na_, va_, vb_, vb_);
            V_ovvv_ba = integrals::ERITransformer::transform_custom(eri_ao_ref, Cbo, Cbv, Cav, Cav, nbf_, nb_, vb_, va_, va_);
            V_ooov_aa_bb = integrals::ERITransformer::transform_custom(eri_ao_ref, Cao, Cao, Cbo, Cbv, nbf_, na_, na_, nb_, vb_);
            V_ooov_bb_aa = integrals::ERITransformer::transform_custom(eri_ao_ref, Cbo, Cbo, Cao, Cav, nbf_, nb_, nb_, na_, va_);
        } else {
            const Eigen::MatrixXd& B_ij_b = B_oo_P_beta_;
            const Eigen::MatrixXd& B_ab_b = B_vv_P_beta_;
            
            M_ovvv_bb    = parallel_gemm_abt(B_ia_P_beta_, B_ab_b);
            M_ooov_bb    = parallel_gemm_abt(B_ij_b, B_ia_P_beta_);
            M_ovvv_ab    = parallel_gemm_abt(B_ia_P_alpha_, B_ab_b);
            M_ovvv_ba    = parallel_gemm_abt(B_ia_P_beta_, B_ab_a);
            M_ooov_aa_bb = parallel_gemm_abt(B_ij_a, B_ia_P_beta_);
            M_ooov_bb_aa = parallel_gemm_abt(B_ij_b, B_ia_P_alpha_);
        }

        bool is_ex = (config_.eri_method == "exact");
        varray_view<double> t_Vovvv_bb = is_ex
            ? varray_view<double>({(len_type)nb_, (len_type)vb_, (len_type)vb_, (len_type)vb_}, V_ovvv_bb.data(), {1, (stride_type)nb_, (stride_type)(nb_*vb_), (stride_type)(nb_*vb_*vb_)})
            : make_mat_view_4d(M_ovvv_bb, nb_, vb_, vb_, vb_);
        varray_view<double> t_Vooov_bb = is_ex
            ? varray_view<double>({(len_type)nb_, (len_type)nb_, (len_type)nb_, (len_type)vb_}, V_ooov_bb.data(), {1, (stride_type)nb_, (stride_type)(nb_*nb_), (stride_type)(nb_*nb_*nb_)})
            : make_mat_view_4d(M_ooov_bb, nb_, nb_, nb_, vb_);
        varray_view<double> t_Vovvv_ab = is_ex
            ? varray_view<double>({(len_type)na_, (len_type)va_, (len_type)vb_, (len_type)vb_}, V_ovvv_ab.data(), {1, (stride_type)na_, (stride_type)(na_*va_), (stride_type)(na_*va_*vb_)})
            : make_mat_view_4d(M_ovvv_ab, na_, va_, vb_, vb_);
        varray_view<double> t_Vovvv_ba = is_ex
            ? varray_view<double>({(len_type)nb_, (len_type)vb_, (len_type)va_, (len_type)va_}, V_ovvv_ba.data(), {1, (stride_type)nb_, (stride_type)(nb_*vb_), (stride_type)(nb_*vb_*va_)})
            : make_mat_view_4d(M_ovvv_ba, nb_, vb_, va_, va_);
        varray_view<double> t_Vooov_aa_bb = is_ex
            ? varray_view<double>({(len_type)na_, (len_type)na_, (len_type)nb_, (len_type)vb_}, V_ooov_aa_bb.data(), {1, (stride_type)na_, (stride_type)(na_*na_), (stride_type)(na_*na_*nb_)})
            : make_mat_view_4d(M_ooov_aa_bb, na_, na_, nb_, vb_);
        varray_view<double> t_Vooov_bb_aa = is_ex
            ? varray_view<double>({(len_type)nb_, (len_type)nb_, (len_type)na_, (len_type)va_}, V_ooov_bb_aa.data(), {1, (stride_type)nb_, (stride_type)(nb_*nb_), (stride_type)(nb_*nb_*na_)})
            : make_mat_view_4d(M_ooov_bb_aa, nb_, nb_, na_, va_);

        // Kontribusi Teff_ab ke Z_mat_a
        tblis::mult< double >(1.0,  t_Vovvv_ba,    "kcab", t_Teff_ab, "ikbc", 1.0, t_Zmat_a, "ai");
        tblis::mult< double >(-1.0, t_Vooov_aa_bb, "jikc", t_Teff_ab, "jkac", 1.0, t_Zmat_a, "ai");

        // Kontribusi Teff_bb & Teff_ab ke Z_mat_b
        tblis::mult< double >(1.0,  t_Vovvv_bb,    "kcab", t_Teff_bb, "ikbc", 0.0, t_Zmat_b, "ai");
        tblis::mult< double >(-1.0, t_Vooov_bb,    "jikc", t_Teff_bb, "jkac", 1.0, t_Zmat_b, "ai");
        tblis::mult< double >(1.0,  t_Vovvv_ab,    "kcab", t_Teff_ab, "kicb", 1.0, t_Zmat_b, "ai"); 
        tblis::mult< double >(-1.0, t_Vooov_bb_aa, "jikc", t_Teff_ab, "kjca", 1.0, t_Zmat_b, "ai");

        // --- A. VVVV Unrestricted (Faktorisasi O(O^2 V^3) Tanpa Tensor G_vvvv) ---
        TBLIS_VIEW_4D(t_T2_bb, (*t2_bb_dense), nb_, nb_, vb_, vb_);
        TBLIS_VIEW_4D(t_T2_ab, (*t2_ab_dense), na_, nb_, va_, vb_);

        Eigen::Tensor<double, 4> Y_ooov_bb(nb_, nb_, nb_, vb_); Y_ooov_bb.setZero();
        TBLIS_VIEW_4D(t_Yooov_bb, Y_ooov_bb, nb_, nb_, nb_, vb_);
        tblis::mult<double>(1.0,  t_T2_bb, "ijab", t_Vovvv_bb, "mafb", 0.0, t_Yooov_bb, "ijmf");
        tblis::mult<double>(-0.5, t_T2_bb, "ijxf", t_Yooov_bb, "ijmf", 1.0, t_Zmat_b, "xm");

        Eigen::Tensor<double, 4> Y_ooov_ab_a(na_, nb_, na_, vb_); Y_ooov_ab_a.setZero();
        TBLIS_VIEW_4D(t_Yooov_ab_a, Y_ooov_ab_a, na_, nb_, na_, vb_);
        tblis::mult<double>(1.0,  t_T2_ab, "ijab", t_Vovvv_ab, "mafb", 0.0, t_Yooov_ab_a, "ijmf");
        tblis::mult<double>(-1.0, t_T2_ab, "ijxf", t_Yooov_ab_a, "ijmf", 1.0, t_Zmat_a, "xm");

        Eigen::Tensor<double, 4> Y_ooov_ab_b(na_, nb_, nb_, va_); Y_ooov_ab_b.setZero();
        TBLIS_VIEW_4D(t_Yooov_ab_b, Y_ooov_ab_b, na_, nb_, nb_, va_);
        tblis::mult<double>(1.0,  t_T2_ab, "ijab", t_Vovvv_ba, "mbea", 0.0, t_Yooov_ab_b, "ijme");
        tblis::mult<double>(-1.0, t_T2_ab, "ijex", t_Yooov_ab_b, "ijme", 1.0, t_Zmat_b, "xm");

        // --- B. OOOO Unrestricted ---
        Eigen::Tensor<double, 4> G_oooo_bb(nb_, nb_, nb_, nb_); G_oooo_bb.setZero();
        Eigen::Tensor<double, 4> G_oooo_ab(na_, na_, nb_, nb_); G_oooo_ab.setZero();
        TBLIS_VIEW_4D(t_Goooo_bb, G_oooo_bb, nb_, nb_, nb_, nb_);
        TBLIS_VIEW_4D(t_Goooo_ab, G_oooo_ab, na_, na_, nb_, nb_);

        tblis::mult<double>(0.125, t_T2_bb, "ijab", t_T2_bb, "mnab", 0.0, t_Goooo_bb, "minj");
        tblis::mult<double>(0.500, t_T2_ab, "ijab", t_T2_ab, "mnab", 0.0, t_Goooo_ab, "minj");

        tblis::mult<double>(1.0, t_Goooo_ab, "minj", t_Vooov_bb_aa, "njix", 1.0, t_Zmat_a, "xm");
        tblis::mult<double>(1.0, t_Goooo_ab, "imnj", t_Vooov_bb_aa, "njix", 1.0, t_Zmat_a, "xm");
        tblis::mult<double>(1.0, t_Goooo_ab, "inmj", t_Vooov_aa_bb, "injx", 1.0, t_Zmat_b, "xm");
        tblis::mult<double>(1.0, t_Goooo_ab, "injm", t_Vooov_aa_bb, "injx", 1.0, t_Zmat_b, "xm");

        tblis::mult<double>(1.0, t_Goooo_bb, "minj", t_Vooov_bb, "njix", 1.0, t_Zmat_b, "xm");
        tblis::mult<double>(1.0, t_Goooo_bb, "imnj", t_Vooov_bb, "njix", 1.0, t_Zmat_b, "xm"); 
        tblis::mult<double>(1.0, t_Goooo_bb, "inmj", t_Vooov_bb, "injx", 1.0, t_Zmat_b, "xm");
        tblis::mult<double>(1.0, t_Goooo_bb, "injm", t_Vooov_bb, "injx", 1.0, t_Zmat_b, "xm");

        // --- C. RING Unrestricted (BB & AB) ---
        Eigen::Tensor<double, 4> G_ovov_bb(nb_, vb_, nb_, vb_); G_ovov_bb.setZero();
        Eigen::Tensor<double, 4> G_oovv_bb(nb_, nb_, vb_, vb_); G_oovv_bb.setZero();
        TBLIS_VIEW_4D(t_Govov_bb, G_ovov_bb, nb_, vb_, nb_, vb_);
        TBLIS_VIEW_4D(t_Goovv_bb, G_oovv_bb, nb_, nb_, vb_, vb_);

        tblis::mult<double>(0.5,  t_T2_bb, "kjcb", t_T2_bb, "ijab", 0.0, t_Govov_bb, "iakc");
        tblis::mult<double>(0.5,  t_T2_ab, "miea", t_T2_ab, "mkec", 1.0, t_Govov_bb, "iakc");
        tblis::mult<double>(-0.5, t_T2_bb, "kjcb", t_T2_bb, "ijab", 0.0, t_Goovv_bb, "ikac");
        tblis::mult<double>(-0.5, t_T2_ab, "miea", t_T2_ab, "mkec", 1.0, t_Goovv_bb, "ikac");

        tblis::mult<double>(1.0,  t_Govov_bb, "makc", t_Vovvv_bb, "kcxa", 1.0, t_Zmat_b, "xm");
        tblis::mult<double>(-1.0, t_Govov_bb, "ixkc", t_Vooov_bb, "imkc", 1.0, t_Zmat_b, "xm");
        tblis::mult<double>(1.0,  t_Govov_bb, "iamc", t_Vovvv_bb, "iaxc", 1.0, t_Zmat_b, "xm");
        tblis::mult<double>(-1.0, t_Govov_bb, "iakx", t_Vooov_bb, "kmia", 1.0, t_Zmat_b, "xm");

        tblis::mult<double>(1.0,  t_Goovv_bb, "mkac", t_Vovvv_bb, "kxca", 1.0, t_Zmat_b, "xm");
        tblis::mult<double>(1.0,  t_Goovv_bb, "imac", t_Vovvv_bb, "ixac", 1.0, t_Zmat_b, "xm");
        tblis::mult<double>(-1.0, t_Goovv_bb, "ikxc", t_Vooov_bb, "ikmc", 1.0, t_Zmat_b, "xm");
        tblis::mult<double>(-1.0, t_Goovv_bb, "ikax", t_Vooov_bb, "ikma", 1.0, t_Zmat_b, "xm");

        Eigen::Tensor<double, 4> G_ovov_ab(na_, va_, nb_, vb_); G_ovov_ab.setZero();
        Eigen::Tensor<double, 4> G_oovv_ab_ex(na_, na_, vb_, vb_); G_oovv_ab_ex.setZero();
        Eigen::Tensor<double, 4> G_oovv_ba_ex(nb_, nb_, va_, va_); G_oovv_ba_ex.setZero();
        TBLIS_VIEW_4D(t_Govov_ab, G_ovov_ab, na_, va_, nb_, vb_);
        TBLIS_VIEW_4D(t_Goovv_ab_ex, G_oovv_ab_ex, na_, na_, vb_, vb_);
        TBLIS_VIEW_4D(t_Goovv_ba_ex, G_oovv_ba_ex, nb_, nb_, va_, va_);

        tblis::mult<double>(1.0,  t_T2,    "miea", t_T2_ab, "mkec", 0.0, t_Govov_ab, "iakc");
        tblis::mult<double>(1.0,  t_T2_bb, "kjcb", t_T2_ab, "ijab", 1.0, t_Govov_ab, "iakc");
        tblis::mult<double>(-0.5, t_T2_ab, "kmec", t_T2_ab, "imea", 0.0, t_Goovv_ab_ex, "ikac");
        tblis::mult<double>(-0.5, t_T2_ab, "mkce", t_T2_ab, "miae", 0.0, t_Goovv_ba_ex, "ikac");

        tblis::mult<double>(1.0,  t_Govov_ab,    "makc", t_Vovvv_ba,    "kcxa", 1.0, t_Zmat_a, "xm");
        tblis::mult<double>(-1.0, t_Govov_ab,    "ixkc", t_Vooov_aa_bb, "imkc", 1.0, t_Zmat_a, "xm");
        tblis::mult<double>(1.0,  t_Goovv_ab_ex, "mkac", t_Vovvv_ab,    "kxac", 1.0, t_Zmat_a, "xm");
        tblis::mult<double>(1.0,  t_Goovv_ab_ex, "imac", t_Vovvv_ab,    "ixac", 1.0, t_Zmat_a, "xm");
        tblis::mult<double>(-1.0, t_Goovv_ba_ex, "ikxc", t_Vooov_bb_aa, "ikmc", 1.0, t_Zmat_a, "xm");
        tblis::mult<double>(-1.0, t_Goovv_ba_ex, "ikax", t_Vooov_bb_aa, "ikma", 1.0, t_Zmat_a, "xm");

        tblis::mult<double>(1.0,  t_Govov_ab,    "iamc", t_Vovvv_ab,    "iaxc", 1.0, t_Zmat_b, "xm");
        tblis::mult<double>(-1.0, t_Govov_ab,    "iakx", t_Vooov_bb_aa, "kmia", 1.0, t_Zmat_b, "xm");
        tblis::mult<double>(1.0,  t_Goovv_ba_ex, "mkac", t_Vovvv_ba,    "kxac", 1.0, t_Zmat_b, "xm");
        tblis::mult<double>(1.0,  t_Goovv_ba_ex, "imac", t_Vovvv_ba,    "ixac", 1.0, t_Zmat_b, "xm");
        tblis::mult<double>(-1.0, t_Goovv_ab_ex, "ikxc", t_Vooov_aa_bb, "ikmc", 1.0, t_Zmat_b, "xm");
        tblis::mult<double>(-1.0, t_Goovv_ab_ex, "ikax", t_Vooov_aa_bb, "ikma", 1.0, t_Zmat_b, "xm");
    } else if (is_restricted) {
        Z_mat_b = Z_mat_a;
    }

    // =========================================================================
    // 4. PENYUSUNAN GENERALIZED FOCK AKHIR
    // =========================================================================
    Eigen::MatrixXd G_full_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
    G_full_a.block(0, 0, na_, na_) = G_oo_alpha_; 
    G_full_a.block(na_, na_, va_, va_) = G_vv_alpha_;
    Eigen::MatrixXd P_corr_a = scf_.C_alpha * G_full_a * scf_.C_alpha.transpose();

    Eigen::MatrixXd G_full_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    Eigen::MatrixXd P_corr_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    
    if (is_restricted) {
        G_full_b = G_full_a;
        P_corr_b = P_corr_a;
    } else if (nb_ > 0) {
        G_full_b.block(0, 0, nb_, nb_) = G_oo_beta_;
        G_full_b.block(nb_, nb_, vb_, vb_) = G_vv_beta_;
        P_corr_b = scf_.C_beta * G_full_b * scf_.C_beta.transpose();
    }

    const Eigen::MatrixXd& F_HF_ao_a = scf_.F_alpha;
    const Eigen::MatrixXd& F_HF_ao_b = scf_.F_beta;
    Eigen::MatrixXd F_HF_mo_a = scf_.C_alpha.transpose() * F_HF_ao_a * scf_.C_alpha;

    Eigen::MatrixXd F_HF_mo_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    if (nb_ > 0) F_HF_mo_b = scf_.C_beta.transpose() * F_HF_ao_b * scf_.C_beta;

    Eigen::MatrixXd G_gamma_ao_a, G_gamma_ao_b;
    build_fock_fast(P_corr_a, P_corr_b, G_gamma_ao_a, G_gamma_ao_b);
    G_gamma_ao_a -= H_core_;
    if (nb_ > 0) G_gamma_ao_b -= H_core_;

    Eigen::MatrixXd G_gamma_mo_a = scf_.C_alpha.transpose() * G_gamma_ao_a * scf_.C_alpha;
    Eigen::MatrixXd G_gamma_mo_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    if (nb_ > 0) G_gamma_mo_b = scf_.C_beta.transpose() * G_gamma_ao_b * scf_.C_beta;

    F_gen_a_ = F_HF_mo_a + G_gamma_mo_a;
    if (na_ > 0 && va_ > 0) {
        Eigen::MatrixXd F_HF_vo_a = F_HF_mo_a.block(na_, 0, va_, na_); 
        Eigen::MatrixXd L_sep_a = F_HF_vo_a * G_oo_alpha_ - G_vv_alpha_ * F_HF_vo_a;
        F_gen_a_.block(na_, 0, va_, na_) += L_sep_a + Z_mat_a;
        F_gen_a_.block(0, na_, na_, va_) += (L_sep_a + Z_mat_a).transpose();
    }

    F_gen_b_ = F_HF_mo_b + G_gamma_mo_b;
    if (nb_ > 0 && vb_ > 0) {
        Eigen::MatrixXd F_HF_vo_b = F_HF_mo_b.block(nb_, 0, vb_, nb_); 
        Eigen::MatrixXd L_sep_b = F_HF_vo_b * G_oo_beta_ - G_vv_beta_ * F_HF_vo_b;
        F_gen_b_.block(nb_, 0, vb_, nb_) += L_sep_b + Z_mat_b;
        F_gen_b_.block(0, nb_, nb_, vb_) += (L_sep_b + Z_mat_b).transpose();
    } else if (is_restricted) {
        F_gen_b_ = F_gen_a_;
    }
}

void OMP3::build_hessian_diagonal(Eigen::VectorXd& diag_H, double grad_norm) {
    // PSI4 BEHAVIOR: Gunakan Hessian OMP2 polos, abaikan koreksi densitas MP3
    OMP2::build_hessian_diagonal(diag_H, grad_norm);
    
    /* -- COMMENT OUT MSHQC HEURISTIC --
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
    ----------------------------------- */
}

void OMP3::debug_gradient_fd(int i_target, int a_target) {
    std::cout << "\n--- [DEBUG] Membedah Komponen Gradien OMP3 ---\n";
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    
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

    double hf_plus, mp2_plus, mp3_plus;
    calc_energy_components(theta, hf_plus, mp2_plus, mp3_plus);
    
    double hf_minus, mp2_minus, mp3_minus;
    calc_energy_components(-theta, hf_minus, mp2_minus, mp3_minus);
    
    double g_num_hf  = (hf_plus - hf_minus) / (2.0 * theta);
    double g_num_mp2 = (mp2_plus - mp2_minus) / (2.0 * theta);
    double g_num_mp3 = (mp3_plus - mp3_minus) / (2.0 * theta);
    double g_num_corr = g_num_mp2 + g_num_mp3;
    double g_num_tot = g_num_hf + g_num_corr;

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
    scf_.C_alpha = C_a_orig;
    scf_.C_beta  = C_b_orig;
    scf_.P_alpha = P_a_orig;
    scf_.P_beta  = P_b_orig;
    C_a_current_ = C_a_orig;
    C_b_current_ = C_b_orig;
}

MP3Result OMP3::compute_omp3() {
    MP2Result res2 = OMP2::compute();

    MP3Result res3;
    res3.converged = res2.converged;
    res3.iterations = res2.iterations;
    res3.e_hf = res2.energy_scf;
    res3.e_mp2 = e_ss_ + e_os_; 
    res3.e_mp3 = e_mp3_tot_;
    res3.e_corr_total = res3.e_mp2 + res3.e_mp3;
    res3.e_total = res3.e_hf + res3.e_corr_total;
    res3.C_alpha = res2.C_alpha;
    res3.C_beta = res2.C_beta;
    res3.n_occ_alpha = na_;
    res3.n_occ_beta = nb_;

    return res3;
}

} // namespace mshqc