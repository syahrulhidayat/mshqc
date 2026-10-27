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

#include "mshqc/mp2/mp2.h"
#include "mshqc/integrals/eri_transformer.h" 
#include <omp.h>
#include <tblis/tblis.h>
#include <iostream> 

namespace mshqc {

using integrals::ERITransformer;

#define TBLIS_VIEW_4D(name, t, d1, d2, d3, d4) \
    tblis::varray_view<double> name({(tblis::len_type)d1, (tblis::len_type)d2, (tblis::len_type)d3, (tblis::len_type)d4}, t.data(), \
    {1, (tblis::stride_type)d1, (tblis::stride_type)(d1*d2), (tblis::stride_type)(d1*d2*d3)})

#define TBLIS_VIEW_3D(name, ptr, d1, d2, d3) \
    tblis::varray_view<double> name({(tblis::len_type)d1, (tblis::len_type)d2, (tblis::len_type)d3}, ptr, \
    {1, (tblis::stride_type)d1, (tblis::stride_type)(d1*d2)})

#define TBLIS_VIEW_2D(name, ptr, d1, d2) \
    tblis::varray_view<double> name({(tblis::len_type)d1, (tblis::len_type)d2}, ptr, \
    {1, (tblis::stride_type)d1})

void OMP2::build_fock_fast(const Eigen::MatrixXd& P_a, const Eigen::MatrixXd& P_b,
                           Eigen::MatrixXd& F_a, Eigen::MatrixXd& F_b)
{
    if (config_.eri_method != "exact") {
        Eigen::MatrixXd P_tot = P_a + P_b;
        int n_chol = scf_.L_mat.cols();

        Eigen::Map<const Eigen::MatrixXd> L_flat_J(scf_.L_mat.data(), nbf_ * nbf_, n_chol);
        Eigen::Map<const Eigen::VectorXd> P_tot_flat(P_tot.data(), nbf_ * nbf_);
        Eigen::VectorXd X_J = L_flat_J.transpose() * P_tot_flat; 
        Eigen::VectorXd J_flat = L_flat_J * X_J;                 
        Eigen::Map<Eigen::MatrixXd> J_mat(J_flat.data(), nbf_, nbf_);

        bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
        bool is_idempotent_pa = (&P_a == &scf_.P_alpha);
        bool has_beta_unres = (!is_restricted && nb_ > 0);

        int max_threads = omp_get_max_threads();
        std::vector<Eigen::MatrixXd> Ka_tls(max_threads, Eigen::MatrixXd::Zero(nbf_, nbf_));
        std::vector<Eigen::MatrixXd> Kb_tls(is_restricted ? 0 : max_threads, Eigen::MatrixXd::Zero(nbf_, nbf_));

        // Gabungkan Ca_o dan Cb_o atau P_a dan P_b agar matriks L_K cukup dibaca 1x dari cache per iterasi K
        Eigen::MatrixXd Cab_o;
        Eigen::MatrixXd Pab_full;
        if (is_idempotent_pa) {
            int n_tot_o = na_ + (has_beta_unres ? nb_ : 0);
            Cab_o.resize(nbf_, n_tot_o);
            if (na_ > 0) Cab_o.leftCols(na_) = scf_.C_alpha.leftCols(na_);
            if (has_beta_unres) Cab_o.rightCols(nb_) = scf_.C_beta.leftCols(nb_);
        } else if (has_beta_unres) {
            Pab_full.resize(nbf_, 2 * nbf_);
            Pab_full.leftCols(nbf_)  = P_a;
            Pab_full.rightCols(nbf_) = P_b;
        }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            Eigen::MatrixXd& Ka_priv = Ka_tls[tid];
            Eigen::MatrixXd Tab_buf(nbf_, is_idempotent_pa ? Cab_o.cols() : (has_beta_unres ? 2 * nbf_ : nbf_));

            #pragma omp for schedule(static) nowait
            for (int K = 0; K < n_chol; ++K) {
                Eigen::Map<const Eigen::MatrixXd> L_K(scf_.L_mat.col(K).data(), nbf_, nbf_);

                if (is_idempotent_pa) {
                    Tab_buf.noalias() = L_K * Cab_o;
                    if (na_ > 0) {
                        auto Ta = Tab_buf.leftCols(na_);
                        Ka_priv.noalias() += Ta * Ta.transpose();
                    }
                    if (has_beta_unres) {
                        auto Tb = Tab_buf.rightCols(nb_);
                        Kb_tls[tid].noalias() += Tb * Tb.transpose();
                    }
                } else if (has_beta_unres) {
                    Tab_buf.noalias() = L_K * Pab_full;
                    Ka_priv.noalias() += Tab_buf.leftCols(nbf_) * L_K;
                    Kb_tls[tid].noalias() += Tab_buf.rightCols(nbf_) * L_K;
                } else {
                    Tab_buf.noalias() = L_K * P_a;
                    Ka_priv.noalias() += Tab_buf * L_K;
                }
            }
        }

        Eigen::MatrixXd Ka_mat = Eigen::MatrixXd::Zero(nbf_, nbf_);
        Eigen::MatrixXd Kb_mat = Eigen::MatrixXd::Zero(nbf_, nbf_);
        for (int t = 0; t < max_threads; ++t) {
            Ka_mat += Ka_tls[t];
            if (!is_restricted && nb_ > 0) Kb_mat += Kb_tls[t];
        }

        F_a = H_core_ + J_mat - Ka_mat;
        if (is_restricted) F_b = F_a;
        else if (nb_ > 0) F_b = H_core_ + J_mat - Kb_mat;
        else F_b = F_a;

        return;
    }

    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    Eigen::MatrixXd P_tot = P_a + P_b;
    const auto& ERI = integrals_->compute_eri();
    const double* __restrict__ eri_ptr = ERI.data();

    Eigen::Map<const Eigen::MatrixXd> ERI_mat(eri_ptr, static_cast<Eigen::Index>(nbf_) * nbf_, static_cast<Eigen::Index>(nbf_) * nbf_);
    Eigen::Map<const Eigen::VectorXd> P_tot_flat(P_tot.data(), static_cast<Eigen::Index>(nbf_) * nbf_);
    Eigen::VectorXd J_flat = ERI_mat * P_tot_flat;
    Eigen::Map<const Eigen::MatrixXd> J_mat(J_flat.data(), nbf_, nbf_);

    Eigen::MatrixXd K_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
    Eigen::MatrixXd K_b = Eigen::MatrixXd::Zero(nbf_, nbf_);

    #pragma omp parallel for schedule(dynamic, 4)
    for (int mu = 0; mu < nbf_; ++mu) {
        for (int nu = 0; nu <= mu; ++nu) {
            double ka = 0.0, kb = 0.0;
            for (int sig = 0; sig < nbf_; ++sig) {
                const double* __restrict__ e_col = eri_ptr + static_cast<size_t>(nbf_) * (mu + static_cast<size_t>(nbf_) * (nu + static_cast<size_t>(nbf_) * sig));
                const double* __restrict__ pa_col = P_a.data() + static_cast<size_t>(sig) * nbf_;
                if (is_restricted) {
                    #pragma omp simd reduction(+:ka)
                    for (int lam = 0; lam < nbf_; ++lam) {
                        ka += e_col[lam] * pa_col[lam];
                    }
                } else {
                    const double* __restrict__ pb_col = P_b.data() + static_cast<size_t>(sig) * nbf_;
                    #pragma omp simd reduction(+:ka, kb)
                    for (int lam = 0; lam < nbf_; ++lam) {
                        ka += e_col[lam] * pa_col[lam];
                        kb += e_col[lam] * pb_col[lam];
                    }
                }
            }
            K_a(mu, nu) = ka;
            K_a(nu, mu) = ka;
            if (!is_restricted) {
                K_b(mu, nu) = kb;
                K_b(nu, mu) = kb;
            }
        }
    }

    F_a = H_core_ + J_mat - K_a;
    F_b = is_restricted ? F_a : (H_core_ + J_mat - K_b);
}

void OMP2::build_opdm_alpha() {
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    G_oo_alpha_ = Eigen::MatrixXd::Zero(na_, na_);
    G_vv_alpha_ = Eigen::MatrixXd::Zero(va_, va_);
    
    auto* t_aa_blk = t2_aa_.get_block(0,0,0,0);
    if (!t_aa_blk) return;

    Eigen::Tensor<double, 4> T2_aa_ijab(na_, na_, va_, va_);
    #pragma omp parallel for collapse(4) schedule(static)
    for(int i = 0; i < na_; ++i) {
        for(int j = 0; j < na_; ++j) {
            for(int a = 0; a < va_; ++a) {
                for(int b = 0; b < va_; ++b) {
                    T2_aa_ijab(i,j,a,b) = (*t_aa_blk)(i,a,j,b);
                }
            }
        }
    }

    TBLIS_VIEW_2D(t_Goo_a, G_oo_alpha_.data(), na_, na_);
    TBLIS_VIEW_2D(t_Gvv_a, G_vv_alpha_.data(), va_, va_);

    if (is_restricted) {
        Eigen::Tensor<double, 4> T2_tilde(na_, na_, va_, va_);
        #pragma omp parallel for collapse(4) schedule(static)
        for (int i = 0; i < na_; ++i) {
            for (int j = 0; j < na_; ++j) {
                for (int a = 0; a < va_; ++a) {
                    for (int b = 0; b < va_; ++b) {
                        T2_tilde(i,j,a,b) = 2.0 * T2_aa_ijab(i,j,a,b) - T2_aa_ijab(i,j,b,a);
                    }
                }
            }
        }

        TBLIS_VIEW_4D(t_T2, T2_aa_ijab, na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_T2t, T2_tilde, na_, na_, va_, va_);

        tblis::mult<double>(-1.0, t_T2, "ikab", t_T2t, "jkab", 0.0, t_Goo_a, "ij");
        tblis::mult<double>(1.0, t_T2, "ijac", t_T2t, "ijbc", 0.0, t_Gvv_a, "ab");

        return;
    }

    TBLIS_VIEW_4D(t_T2aa, T2_aa_ijab, na_, na_, va_, va_);

    tblis::mult<double>(-0.5, t_T2aa, "ikab", t_T2aa, "jkab", 0.0, t_Goo_a, "ij");
    tblis::mult<double>(0.5, t_T2aa, "ijac", t_T2aa, "ijbc", 0.0, t_Gvv_a, "ab");

    auto* t_ab_blk = t2_ab_.get_block(0,0,0,0);
    if (nb_ > 0 && vb_ > 0 && t_ab_blk) {
        TBLIS_VIEW_4D(t_T2ab, (*t_ab_blk), na_, nb_, va_, vb_);
        
        tblis::mult<double>(-1.0, t_T2ab, "ikab", t_T2ab, "jkab", 1.0, t_Goo_a, "ij");
        tblis::mult<double>(1.0, t_T2ab, "ijac", t_T2ab, "ijbc", 1.0, t_Gvv_a, "ab");
    }
}

void OMP2::build_opdm_beta() {
    G_oo_beta_ = Eigen::MatrixXd::Zero(nb_, nb_);
    G_vv_beta_ = Eigen::MatrixXd::Zero(vb_, vb_);
    if (nb_ == 0 || vb_ == 0) return;

    auto* t_bb_blk = t2_bb_.get_block(0,0,0,0);
    auto* t_ab_blk = t2_ab_.get_block(0,0,0,0);

    TBLIS_VIEW_2D(t_Goo_b, G_oo_beta_.data(), nb_, nb_);
    TBLIS_VIEW_2D(t_Gvv_b, G_vv_beta_.data(), vb_, vb_);

    if (t_bb_blk) {
        TBLIS_VIEW_4D(t_T2bb, (*t_bb_blk), nb_, nb_, vb_, vb_);
        tblis::mult<double>(-0.5, t_T2bb, "ikab", t_T2bb, "jkab", 0.0, t_Goo_b, "ij");
        tblis::mult<double>(0.5, t_T2bb, "ijac", t_T2bb, "ijbc", 0.0, t_Gvv_b, "ab");
    }

    if (t_ab_blk) {
        TBLIS_VIEW_4D(t_T2ab, (*t_ab_blk), na_, nb_, va_, vb_);
        tblis::mult<double>(-1.0, t_T2ab, "kiab", t_T2ab, "kjab", 1.0, t_Goo_b, "ij");
        tblis::mult<double>(1.0, t_T2ab, "ijca", t_T2ab, "ijcb", 1.0, t_Gvv_b, "ab");
    }
}

void OMP2::build_generalized_fock() {
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    Eigen::MatrixXd Z_mat_a = Eigen::MatrixXd::Zero(va_, na_);
    Eigen::MatrixXd Z_mat_b = Eigen::MatrixXd::Zero(vb_, nb_);

    // =========================================================================
    // TAHAP 1: EVALUASI Z-VECTOR (EXACT / CHOLESKY / DF)
    // =========================================================================
    if (config_.eri_method == "exact") {
        Z_mat_a.setZero();
        Z_mat_b.setZero();

        const auto& eri_ao = integrals_->compute_eri();

        const Eigen::MatrixXd& Ca_o = scf_.C_alpha.leftCols(na_);
        const Eigen::MatrixXd& Ca_v = scf_.C_alpha.rightCols(va_);
        auto* t_aa_blk = t2_aa_.get_block(0, 0, 0, 0);

        if (t_aa_blk && na_ > 0 && va_ > 0) {
            Eigen::Tensor<double, 4> T2_eff_aa(na_, na_, va_, va_);
            #pragma omp parallel for collapse(4) schedule(static)
            for (int i = 0; i < na_; ++i) {
                for (int j = 0; j < na_; ++j) {
                    for (int a = 0; a < va_; ++a) {
                        for (int b = 0; b < va_; ++b) {
                            if (is_restricted) {
                                T2_eff_aa(i, j, a, b) = 2.0 * (*t_aa_blk)(i, a, j, b) - (*t_aa_blk)(i, b, j, a);
                            } else {
                                T2_eff_aa(i, j, a, b) = (*t_aa_blk)(i, a, j, b);
                            }
                        }
                    }
                }
            }

            auto V_ovvv_aa = integrals::ERITransformer::transform_custom(eri_ao, Ca_o, Ca_v, Ca_v, Ca_v, nbf_, na_, va_, va_, va_);
            auto V_ooov_aa = integrals::ERITransformer::transform_custom(eri_ao, Ca_o, Ca_o, Ca_o, Ca_v, nbf_, na_, na_, na_, va_);

            TBLIS_VIEW_4D(t_Teff_aa, T2_eff_aa, na_, na_, va_, va_);
            TBLIS_VIEW_4D(t_Vovvv_aa, V_ovvv_aa, na_, va_, va_, va_);
            TBLIS_VIEW_4D(t_Vooov_aa, V_ooov_aa, na_, na_, na_, va_);
            TBLIS_VIEW_2D(t_Zmat_a, Z_mat_a.data(), va_, na_);

            // Dipanggil di luar #pragma omp parallel agar TBLIS bebas memakai seluruh core CPU
            tblis::mult<double>(1.0,  t_Vovvv_aa, "kcab", t_Teff_aa, "ikbc", 0.0, t_Zmat_a, "ai");
            tblis::mult<double>(-1.0, t_Vooov_aa, "jikc", t_Teff_aa, "jkac", 1.0, t_Zmat_a, "ai");
        }

        if (is_restricted) {
            Z_mat_b = Z_mat_a;
        } else if (nb_ > 0 && vb_ > 0) {
            const Eigen::MatrixXd& Cb_o = scf_.C_beta.leftCols(nb_);
            const Eigen::MatrixXd& Cb_v = scf_.C_beta.rightCols(vb_);

            auto* t_bb_dense = t2_bb_.get_block(0, 0, 0, 0);
            auto* t_ab_dense = t2_ab_.get_block(0, 0, 0, 0);

            if (t_bb_dense && t_ab_dense) {
                auto V_ovvv_bb    = integrals::ERITransformer::transform_custom(eri_ao, Cb_o, Cb_v, Cb_v, Cb_v, nbf_, nb_, vb_, vb_, vb_);
                auto V_ooov_bb    = integrals::ERITransformer::transform_custom(eri_ao, Cb_o, Cb_o, Cb_o, Cb_v, nbf_, nb_, nb_, nb_, vb_);
                auto V_ovvv_ba_aa = integrals::ERITransformer::transform_custom(eri_ao, Cb_o, Cb_v, Ca_v, Ca_v, nbf_, nb_, vb_, va_, va_);
                auto V_ooov_aa_bb = integrals::ERITransformer::transform_custom(eri_ao, Ca_o, Ca_o, Cb_o, Cb_v, nbf_, na_, na_, nb_, vb_);
                auto V_ovvv_ab_bb = integrals::ERITransformer::transform_custom(eri_ao, Ca_o, Ca_v, Cb_v, Cb_v, nbf_, na_, va_, vb_, vb_);
                auto V_ooov_bb_aa = integrals::ERITransformer::transform_custom(eri_ao, Cb_o, Cb_o, Ca_o, Ca_v, nbf_, nb_, nb_, na_, va_);

                TBLIS_VIEW_4D(t_Tbb, (*t_bb_dense), nb_, nb_, vb_, vb_);
                TBLIS_VIEW_4D(t_Tab, (*t_ab_dense), na_, nb_, va_, vb_);
                TBLIS_VIEW_4D(t_Vovvv_bb, V_ovvv_bb, nb_, vb_, vb_, vb_);
                TBLIS_VIEW_4D(t_Vooov_bb, V_ooov_bb, nb_, nb_, nb_, vb_);
                TBLIS_VIEW_4D(t_Vovvv_ba, V_ovvv_ba_aa, nb_, vb_, va_, va_);
                TBLIS_VIEW_4D(t_Vooov_aa_bb, V_ooov_aa_bb, na_, na_, nb_, vb_);
                TBLIS_VIEW_4D(t_Vovvv_ab, V_ovvv_ab_bb, na_, va_, vb_, vb_);
                TBLIS_VIEW_4D(t_Vooov_bb_aa, V_ooov_bb_aa, nb_, nb_, na_, va_);

                TBLIS_VIEW_2D(t_Zmat_a, Z_mat_a.data(), va_, na_);
                TBLIS_VIEW_2D(t_Zmat_b, Z_mat_b.data(), vb_, nb_);

                // Kontribusi T2_ab ke Z_mat_a
                tblis::mult<double>(1.0,  t_Vovvv_ba,    "kcab", t_Tab, "ikbc", 1.0, t_Zmat_a, "ai");
                tblis::mult<double>(-1.0, t_Vooov_aa_bb, "jikc", t_Tab, "jkac", 1.0, t_Zmat_a, "ai");

                // Kontribusi T2_bb dan T2_ab ke Z_mat_b
                tblis::mult<double>(1.0,  t_Vovvv_bb,    "kcab", t_Tbb, "ikbc", 0.0, t_Zmat_b, "ai");
                tblis::mult<double>(-1.0, t_Vooov_bb,    "jikc", t_Tbb, "jkac", 1.0, t_Zmat_b, "ai");
                tblis::mult<double>(1.0,  t_Vovvv_ab,    "kcab", t_Tab, "kicb", 1.0, t_Zmat_b, "ai");
                tblis::mult<double>(-1.0, t_Vooov_bb_aa, "jikc", t_Tab, "kjca", 1.0, t_Zmat_b, "ai");
            }
        }
    } else {
        evaluate_z_vector_cholesky(Z_mat_a, Z_mat_b);
        if (is_restricted) Z_mat_b = Z_mat_a;
    }

    // =========================================================================
    // TAHAP 2: PERAKITAN 1-RDM MENGGUNAKAN DENSITAS YANG SUDAH RILEKS
    // =========================================================================
    Eigen::MatrixXd G_full_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
    G_full_a.block(0, 0, na_, na_) = G_oo_alpha_; 
    G_full_a.block(na_, na_, va_, va_) = G_vv_alpha_;
    Eigen::MatrixXd P_corr_a = scf_.C_alpha * G_full_a * scf_.C_alpha.transpose();

    Eigen::MatrixXd G_full_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    Eigen::MatrixXd P_corr_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    
    if (is_restricted) {
        G_oo_beta_ = G_oo_alpha_;
        G_vv_beta_ = G_vv_alpha_;
        G_full_b = G_full_a;
        P_corr_b = P_corr_a;
    } else if (nb_ > 0) {
        G_full_b.block(0, 0, nb_, nb_) = G_oo_beta_;
        G_full_b.block(nb_, nb_, vb_, vb_) = G_vv_beta_;
        P_corr_b = scf_.C_beta * G_full_b * scf_.C_beta.transpose();
    }

    // =========================================================================
    // TAHAP 3: KALKULASI FOCK DAN PENYUSUNAN FOCK UMUM (GENERALIZED FOCK)
    // =========================================================================
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
    
    if (omp_get_thread_num() == 0 && config_.print_level > 1) {
        std::cout << "\n  [DEBUG KOMPONEN FOCK OMP2]\n";
        std::cout << "  Trace G_oo_a + G_vv_a (Koreksi Partikel) : " 
                  << std::scientific << (G_oo_alpha_.trace() + G_vv_alpha_.trace()) << "\n";
        std::cout << "  Norm Z_mat_a (Z-Vector)                  : " 
                  << std::scientific << Z_mat_a.norm() << "\n";
        
        if (nb_ > 0 && vb_ > 0) {
            std::cout << "  Trace G_oo_b + G_vv_b (Koreksi Partikel) : " 
                      << std::scientific << (G_oo_beta_.trace() + G_vv_beta_.trace()) << "\n";
            std::cout << "  Norm Z_mat_b (Z-Vector)                  : " 
                      << std::scientific << Z_mat_b.norm() << "\n";
        }
    }
}

} // namespace mshqc