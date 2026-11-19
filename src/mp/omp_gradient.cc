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
#include "mshqc/integrals/packed_ao_utils.h"
#include "mshqc/utils/blas_thread_guard.h"
#include "mshqc/gradient/optimizer.h"
#include <omp.h>
#include <tblis/tblis.h>
#include <unsupported/Eigen/MatrixFunctions>

namespace mshqc {

void OMP2::evaluate_z_vector_cholesky(Eigen::MatrixXd& Z_mat_a, Eigen::MatrixXd& Z_mat_b) {
  const int n_chol = static_cast<int>(scf_.L_mat.cols());
  Z_mat_a.setZero(va_, na_);
  const bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
  const bool has_beta = (!is_restricted && nb_ > 0 && vb_ > 0);
  if (has_beta) Z_mat_b.setZero(vb_, nb_);

  auto* t_aa_blk = t2_aa_.get_block(0, 0, 0, 0);
  auto* t_ab_blk = has_beta ? t2_ab_.get_block(0, 0, 0, 0) : nullptr;
  auto* t_bb_blk = has_beta ? t2_bb_.get_block(0, 0, 0, 0) : nullptr;

  const int nf_a = std::clamp(n_frozen_, 0, na_);
  const int nf_b = std::clamp(n_frozen_, 0, nb_);

  // Zero-T2-copy contraction: X_a (na*va x n_chol) & X_b (nb*vb x n_chol)
  Eigen::MatrixXd X_a = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(na_) * va_, n_chol);
  Eigen::MatrixXd X_b;
  if (has_beta) X_b = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(nb_) * vb_, n_chol);

  if (t_aa_blk || (has_beta && t_ab_blk)) {
    #pragma omp parallel
    {
      Eigen::MatrixXd T_aa_ij(va_, va_);
      Eigen::MatrixXd T_ab_ij;
      if (has_beta && t_ab_blk) T_ab_ij.resize(va_, vb_);

      #pragma omp for schedule(dynamic, 1) nowait
      for (int i = nf_a; i < na_; ++i) {
        auto X_ai = X_a.middleRows(static_cast<Eigen::Index>(i) * va_, va_);
        if (t_aa_blk) {
          for (int j = nf_a; j < na_; ++j) {
            if (is_restricted) {
              for (int a = 0; a < va_; ++a)
                for (int b = 0; b < va_; ++b)
                  T_aa_ij(a, b) = 2.0 * (*t_aa_blk)(i, a, j, b) - (*t_aa_blk)(i, b, j, a);
            } else {
              for (int a = 0; a < va_; ++a)
                for (int b = 0; b < va_; ++b)
                  T_aa_ij(a, b) = (*t_aa_blk)(i, a, j, b);
            }
            X_ai.noalias() += T_aa_ij * B_ia_P_alpha_.middleRows(static_cast<Eigen::Index>(j) * va_, va_);
          }
        }
        if (has_beta && t_ab_blk) {
          for (int jb = nf_b; jb < nb_; ++jb) {
            for (int a = 0; a < va_; ++a)
              for (int b = 0; b < vb_; ++b)
                T_ab_ij(a, b) = (*t_ab_blk)(i, jb, a, b);
            X_ai.noalias() += T_ab_ij * B_ia_P_beta_.middleRows(static_cast<Eigen::Index>(jb) * vb_, vb_);
          }
        }
      }
    }
  }

  if (has_beta && (t_bb_blk || t_ab_blk)) {
    #pragma omp parallel
    {
      Eigen::MatrixXd T_bb_ij(vb_, vb_);
      Eigen::MatrixXd T_ba_ij(vb_, va_);

      #pragma omp for schedule(dynamic, 1) nowait
      for (int ib = nf_b; ib < nb_; ++ib) {
        auto X_bi = X_b.middleRows(static_cast<Eigen::Index>(ib) * vb_, vb_);
        if (t_bb_blk) {
          for (int jb = nf_b; jb < nb_; ++jb) {
            for (int a = 0; a < vb_; ++a)
              for (int b = 0; b < vb_; ++b)
                T_bb_ij(a, b) = (*t_bb_blk)(ib, jb, a, b);
            X_bi.noalias() += T_bb_ij * B_ia_P_beta_.middleRows(static_cast<Eigen::Index>(jb) * vb_, vb_);
          }
        }
        if (t_ab_blk) {
          for (int ia = nf_a; ia < na_; ++ia) {
            for (int a = 0; a < va_; ++a)
              for (int b = 0; b < vb_; ++b)
                T_ba_ij(b, a) = (*t_ab_blk)(ia, ib, a, b);
            X_bi.noalias() += T_ba_ij * B_ia_P_alpha_.middleRows(static_cast<Eigen::Index>(ia) * va_, va_);
          }
        }
      }
    }
  }

  #pragma omp parallel
  {
    Eigen::MatrixXd Z_loc_a = Eigen::MatrixXd::Zero(va_, na_);
    Eigen::MatrixXd Z_loc_b;
    if (has_beta) Z_loc_b = Eigen::MatrixXd::Zero(vb_, nb_);

    #pragma omp for schedule(static)
    for (int P = 0; P < n_chol; ++P) {
      Eigen::Map<const Eigen::MatrixXd> B_oo_a(B_oo_P_alpha_.col(P).data(), na_, na_);
      Eigen::Map<const Eigen::MatrixXd> B_vv_a(B_vv_P_alpha_.col(P).data(), va_, va_);
      Eigen::Map<const Eigen::MatrixXd> X_ai(X_a.col(P).data(), va_, na_);
      Z_loc_a.noalias() += B_vv_a * X_ai - X_ai * B_oo_a;

      if (has_beta) {
        Eigen::Map<const Eigen::MatrixXd> B_oo_b(B_oo_P_beta_.col(P).data(), nb_, nb_);
        Eigen::Map<const Eigen::MatrixXd> B_vv_b(B_vv_P_beta_.col(P).data(), vb_, vb_);
        Eigen::Map<const Eigen::MatrixXd> X_bi(X_b.col(P).data(), vb_, nb_);
        Z_loc_b.noalias() += B_vv_b * X_bi - X_bi * B_oo_b;
      }
    }

    #pragma omp critical
    {
      Z_mat_a += Z_loc_a;
      if (has_beta) Z_mat_b += Z_loc_b;
    }
  }
}

void OMP2::build_hessian_diagonal(Eigen::VectorXd& diag_H, double grad_norm) {
    int n_params = orbital_gradient_.size();
    if (diag_H.size() != n_params) diag_H.resize(n_params);
    
    int idx = 0;
    double level_shift = (grad_norm > 0.1) ? 0.05 : 0.005;
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
     
    
    double spin_factor = is_restricted ? 4.0 : 2.0;
    
    for (int i = 0; i < na_; ++i) {             
        for (int a = 0; a < va_; ++a) {        
            double eps_diff = scf_.orbital_energies_alpha(na_ + a) - scf_.orbital_energies_alpha(i);
            double safe_diff = std::max(std::abs(eps_diff), 1e-4);
            double J_ia = 0.0;
            if (config_.eri_method != "exact") {
                J_ia = B_ia_P_alpha_.row(i * va_ + a).squaredNorm(); 
            } else {
                auto* g_blk = g_aa_.get_block(0, 0, 0, 0);
                if (g_blk) J_ia = std::abs((*g_blk)(i, a, i, a));
            }
            
            // Singly occupied (alpha only) dievaluasi dengan factor setengah dari doubly
            double factor = spin_factor; 
            diag_H(idx++) = factor * safe_diff + 2.0 * factor * J_ia + level_shift;  
        }
    }
    
    if (!is_restricted && nb_ > 0) {
        for (int i = 0; i < nb_; ++i) {         
            for (int a = 0; a < vb_; ++a) {    
                double eps_diff = scf_.orbital_energies_beta(nb_ + a) - scf_.orbital_energies_beta(i);
                double safe_diff = std::max(std::abs(eps_diff), 1e-4);
                double J_ia = 0.0;
                if (config_.eri_method != "exact") {
                    J_ia = B_ia_P_beta_.row(i * vb_ + a).squaredNorm();
                } else {
                    auto* g_blk = g_bb_.get_block(0, 0, 0, 0);
                    if (g_blk) J_ia = std::abs((*g_blk)(i, a, i, a));
                }
                diag_H(idx++) = 2.0 * safe_diff + 4.0 * J_ia + level_shift;
            }
        }
    }
}

Eigen::VectorXd OMP2::compute_soscf_step(double trust_radius, double& expected_change) {
    int n_params = orbital_gradient_.size();
    if (n_params == 0) return Eigen::VectorXd::Zero(0);

    double grad_norm = orbital_gradient_.norm(); 

    build_hessian_diagonal(hessian_diag_, grad_norm);

    for(int i = 0; i < hessian_diag_.size(); ++i) {
        if(std::abs(hessian_diag_(i)) < 1e-12) hessian_diag_(i) = 1e-12; 
    }

    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    
    double spin_factor = is_restricted ? 4.0 : 2.0;

    mshqc::gradient::TrustRegionConfig tr_conf;
    tr_conf.micro_thresh = std::min(1e-4, grad_norm * 0.1); 
    mshqc::gradient::TrustRegionSOSCF soscf_engine(tr_conf);

    auto compute_hessian_vector = [&](const Eigen::VectorXd& p_vec) -> Eigen::VectorXd {
        Eigen::VectorXd Hp = Eigen::VectorXd::Zero(n_params);
        int dim_a = va_ * na_;
        int dim_b = is_restricted ? 0 : (vb_ * nb_); 
        
        int temp_idx = 0;
        for (int i = 0; i < na_; ++i) {
            for (int a = 0; a < va_; ++a) {
                double eps_diff = scf_.orbital_energies_alpha(na_ + a) - scf_.orbital_energies_alpha(i);
                double factor = spin_factor; 
                Hp(temp_idx) = factor * std::max(std::abs(eps_diff), 1e-4) * p_vec(temp_idx);
                temp_idx++;
            }
        }
        if (!is_restricted && dim_b > 0) {
            for (int i = 0; i < nb_; ++i) {
                for (int a = 0; a < vb_; ++a) {
                    double eps_diff = scf_.orbital_energies_beta(nb_ + a) - scf_.orbital_energies_beta(i);
                    Hp(temp_idx) = 2.0 * std::max(std::abs(eps_diff), 1e-4) * p_vec(temp_idx);
                    temp_idx++;
                }
            }
        }
        
        Eigen::MatrixXd kappa_a = Eigen::MatrixXd::Zero(na_, va_);
        if (dim_a > 0) {
            int k_idx = 0;
            for (int i = 0; i < na_; ++i) {
                for (int a = 0; a < va_; ++a) {
                    kappa_a(i, a) = p_vec(k_idx++);
                }
            }
        }
        
        Eigen::MatrixXd kappa_b = Eigen::MatrixXd::Zero(nb_, vb_);
        if (!is_restricted && dim_b > 0) {
            int k_idx = dim_a;
            for (int i = 0; i < nb_; ++i) {
                for (int a = 0; a < vb_; ++a) {
                    kappa_b(i, a) = p_vec(k_idx++);
                }
            }
        }
        
        // [INJECTED_FINAL_OPT_MO_SOSCF]
        const int n_aux = static_cast<int>(B_ia_P_alpha_.cols());
        const bool can_use_mo_df = (config_.eri_method != "exact" && n_aux > 0 &&
                                    B_oo_P_alpha_.cols() == n_aux && B_vv_P_alpha_.cols() == n_aux);
        if (can_use_mo_df) {
            Eigen::VectorXd w_aux = Eigen::VectorXd::Zero(n_aux);
            if (dim_a > 0) {
                Eigen::MatrixXd ka_T = kappa_a.transpose();
                Eigen::Map<const Eigen::VectorXd> ka_vec(ka_T.data(), dim_a);
                w_aux.noalias() += (is_restricted ? 4.0 : 2.0) * (B_ia_P_alpha_.transpose() * ka_vec);
            }
            if (!is_restricted && dim_b > 0 && B_ia_P_beta_.cols() == n_aux) {
                Eigen::MatrixXd kb_T = kappa_b.transpose();
                Eigen::Map<const Eigen::VectorXd> kb_vec(kb_T.data(), dim_b);
                w_aux.noalias() += 2.0 * (B_ia_P_beta_.transpose() * kb_vec);
            }

            Eigen::MatrixXd H_kappa_a = Eigen::MatrixXd::Zero(na_, va_);
            Eigen::MatrixXd H_kappa_b = (!is_restricted && dim_b > 0) ? Eigen::MatrixXd::Zero(nb_, vb_) : Eigen::MatrixXd();

            if (dim_a > 0) {
                Eigen::VectorXd j_vo_a = B_ia_P_alpha_ * w_aux;
                Eigen::Map<const Eigen::MatrixXd> J_vo_a_mat(j_vo_a.data(), va_, na_);
                H_kappa_a.noalias() = J_vo_a_mat.transpose();
            }
            if (!is_restricted && dim_b > 0 && B_ia_P_beta_.cols() == n_aux) {
                Eigen::VectorXd j_vo_b = B_ia_P_beta_ * w_aux;
                Eigen::Map<const Eigen::MatrixXd> J_vo_b_mat(j_vo_b.data(), vb_, nb_);
                H_kappa_b.noalias() = J_vo_b_mat.transpose();
            }

            const int max_t = omp_get_max_threads();
            std::vector<Eigen::MatrixXd> Ka_tls(max_t, Eigen::MatrixXd::Zero(na_, va_));
            std::vector<Eigen::MatrixXd> Kb_tls((!is_restricted && dim_b > 0) ? max_t : 0, Eigen::MatrixXd::Zero(nb_, vb_));

            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                // [INJECTED_AUDIT_FIX_SOSCF_O2V]
      Eigen::MatrixXd T_oo_a(na_, va_), T_oo2_a(na_, na_);
                Eigen::MatrixXd T_oo_b, T_oo2_b;
                if (!is_restricted && dim_b > 0) {
                  T_oo_b.resize(nb_, vb_);
                  T_oo2_b.resize(nb_, nb_);
                }

                #pragma omp for schedule(static) nowait
                for (int P = 0; P < n_aux; ++P) {
                    if (dim_a > 0) {
                        Eigen::Map<const Eigen::MatrixXd> Boo_a(B_oo_P_alpha_.col(P).data(), na_, na_);
                        Eigen::Map<const Eigen::MatrixXd> Bvo_a(B_ia_P_alpha_.col(P).data(), va_, na_);
                        Eigen::Map<const Eigen::MatrixXd> Bvv_a(B_vv_P_alpha_.col(P).data(), va_, va_);
                        T_oo_a.noalias() = Boo_a * kappa_a;
                        Ka_tls[tid].noalias() += T_oo_a * Bvv_a;
                        T_oo2_a.noalias() = kappa_a * Bvo_a;
          Ka_tls[tid].noalias() += T_oo2_a.transpose() * Bvo_a.transpose();
                    }
                    if (!is_restricted && dim_b > 0 && B_oo_P_beta_.cols() == n_aux) {
                        Eigen::Map<const Eigen::MatrixXd> Boo_b(B_oo_P_beta_.col(P).data(), nb_, nb_);
                        Eigen::Map<const Eigen::MatrixXd> Bvo_b(B_ia_P_beta_.col(P).data(), vb_, nb_);
                        Eigen::Map<const Eigen::MatrixXd> Bvv_b(B_vv_P_beta_.col(P).data(), vb_, vb_);
                        T_oo_b.noalias() = Boo_b * kappa_b;
                        Kb_tls[tid].noalias() += T_oo_b * Bvv_b;
                        T_oo2_b.noalias() = kappa_b * Bvo_b;
          Kb_tls[tid].noalias() += T_oo2_b.transpose() * Bvo_b.transpose();
                    }
                }
            }

            for (int t = 0; t < max_t; ++t) {
                if (dim_a > 0) H_kappa_a.noalias() -= Ka_tls[t];
                if (!is_restricted && dim_b > 0) H_kappa_b.noalias() -= Kb_tls[t];
            }

            if (dim_a > 0) {
                int idx_h = 0;
                for (int i = 0; i < na_; ++i) {
                    for (int a = 0; a < va_; ++a) {
                        Hp(idx_h++) += spin_factor * H_kappa_a(i, a);
                    }
                }
            }
            if (!is_restricted && dim_b > 0) {
                int idx_h = dim_a;
                for (int i = 0; i < nb_; ++i) {
                    for (int a = 0; a < vb_; ++a) {
                        Hp(idx_h++) += 2.0 * H_kappa_b(i, a);
                    }
                }
            }
        } else {
            Eigen::MatrixXd P1_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
            if (dim_a > 0) {
                P1_a = scf_.C_alpha.leftCols(na_) * kappa_a * scf_.C_alpha.rightCols(va_).transpose();
                P1_a = (P1_a + P1_a.transpose()).eval();
            }
            Eigen::MatrixXd P1_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
            if (!is_restricted && dim_b > 0) {
                P1_b = scf_.C_beta.leftCols(nb_) * kappa_b * scf_.C_beta.rightCols(vb_).transpose();
                P1_b = (P1_b + P1_b.transpose()).eval();
            } else if (is_restricted) {
                P1_b = P1_a;
            }
            Eigen::MatrixXd F1_a, F1_b;
            build_fock_fast(P1_a, P1_b, F1_a, F1_b);
            F1_a -= H_core_;
            if (!is_restricted || nb_ > 0) F1_b -= H_core_;
            if (dim_a > 0) {
                Eigen::MatrixXd H_kappa_a = scf_.C_alpha.leftCols(na_).transpose() * F1_a * scf_.C_alpha.rightCols(va_);
                int idx_h = 0;
                for (int i = 0; i < na_; ++i) {
                    for (int a = 0; a < va_; ++a) {
                        Hp(idx_h++) += spin_factor * H_kappa_a(i, a);
                    }
                }
            }
            if (!is_restricted && dim_b > 0) {
                Eigen::MatrixXd H_kappa_b = scf_.C_beta.leftCols(nb_).transpose() * F1_b * scf_.C_beta.rightCols(vb_);
                int idx_h = dim_a;
                for (int i = 0; i < nb_; ++i) {
                    for (int a = 0; a < vb_; ++a) {
                        Hp(idx_h++) += 2.0 * H_kappa_b(i, a);
                    }
                }
            }
        }

        bool use_sym = (!scf_.irreps_alpha.empty() && scf_.irreps_alpha[0] != -1);
        if (use_sym) {
            int idx_sym = 0;
            if (!is_restricted && dim_b > 0) {
                for (int i = 0; i < na_; ++i) {
                    for (int a = 0; a < va_; ++a) {
                        if ((scf_.irreps_alpha[i] ^ scf_.irreps_alpha[na_ + a]) != 0) {
                            Hp(idx_sym) = hessian_diag_(idx_sym) * p_vec(idx_sym);
                        }
                        idx_sym++;
                    }
                }
                for (int i = 0; i < nb_; ++i) {
                    for (int b = 0; b < vb_; ++b) {
                        if ((scf_.irreps_beta[i] ^ scf_.irreps_beta[nb_ + b]) != 0) {
                            Hp(idx_sym) = hessian_diag_(idx_sym) * p_vec(idx_sym);
                        }
                        idx_sym++;
                    }
                }
            } else {
                for (int i = 0; i < na_; ++i) {
                    for (int a = 0; a < va_; ++a) {
                        if ((scf_.irreps_alpha[i] ^ scf_.irreps_alpha[na_ + a]) != 0) {
                            Hp(idx_sym) = hessian_diag_(idx_sym) * p_vec(idx_sym);
                        }
                        idx_sym++;
                    }
                }
            }
        }
    if (n_frozen_ > 0) {
      const int nf_a = std::min(n_frozen_, na_);
      for (int i = 0; i < nf_a; ++i)
        for (int a = 0; a < va_; ++a)
          Hp(i * va_ + a) = hessian_diag_(i * va_ + a) * p_vec(i * va_ + a);
      if (!is_restricted && dim_b > 0) {
        const int nf_b = std::min(n_frozen_, nb_);
        for (int i = 0; i < nf_b; ++i)
          for (int b = 0; b < vb_; ++b)
            Hp(dim_a + i * vb_ + b) = hessian_diag_(dim_a + i * vb_ + b) * p_vec(dim_a + i * vb_ + b);
      }
    }
    return Hp;
    };

    mshqc::gradient::TrustRegionResult step_info = soscf_engine.solve(
        orbital_gradient_, hessian_diag_, trust_radius, compute_hessian_vector
    );
    
    expected_change = step_info.predicted_energy_change;
    return step_info.step;
}
void OMP2::apply_orbital_rotation(const Eigen::VectorXd& kappa) {
    if (kappa.norm() < 1e-12) return;
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    
    
    int idx = 0;
    int n_mo_a = na_ + va_;
    Eigen::MatrixXd K_a = Eigen::MatrixXd::Zero(n_mo_a, n_mo_a);

    for (int i = 0; i < na_; ++i) {
        for (int a = 0; a < va_; ++a) {
            double val = kappa(idx++);
            K_a(na_ + a, i) = val;
            K_a(i, na_ + a) = -val;
        }
    }
    {
    mshqc::utils::ScopedBlasThreads blas_exp_mt(n_mo_a, 96);
    C_a_current_ = C_a_current_ * K_a.exp();
  }

   

    // Lanjutkan rotasi UMP2 asli jika unrestricted
    int n_mo_b = nb_ + vb_;
    Eigen::MatrixXd K_b = Eigen::MatrixXd::Zero(n_mo_b, n_mo_b);

    if (!is_restricted && nb_ > 0) {
        for (int i = 0; i < nb_; ++i) {
            for (int a = 0; a < vb_; ++a) {
                double val = kappa(idx++);
                K_b(nb_ + a, i) = val;
                K_b(i, nb_ + a) = -val;
            }
        }
    } else if (is_restricted && nb_ > 0) {
        K_b = K_a; 
    }

    if (nb_ > 0) {
        {
      mshqc::utils::ScopedBlasThreads blas_exp_mt(n_mo_b, 96);
      C_b_current_ = C_b_current_ * K_b.exp();
    }
    }
}
Eigen::MatrixXd OMP2::solve_cphf_pcg_ov(const Eigen::MatrixXd& Z_ov, const Eigen::VectorXd& eps, int na, int va) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(va, na);
    Eigen::MatrixXd R = Z_ov;
    Eigen::MatrixXd P = R;
    
    double rsold = R.squaredNorm();
    const int max_iter = 50;
    const double tol = 1e-10;

    for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::MatrixXd AP = Eigen::MatrixXd::Zero(va, na);
        
        // 1. Evaluasi Komponen Diagonal HF (Selisih Eigenenergi)
        for (int i = 0; i < na; ++i) {
            for (int a = 0; a < va; ++a) {
                AP(a, i) = (eps(na + a) - eps(i)) * P(a, i);
            }
        }

        // 2. Evaluasi On-The-Fly Interaksi 2-Elektron (Hessian-Vector Product)
        // Membentuk matriks pseudo-densitas P1 dari tebakan P
        const int n_aux_cphf = static_cast<int>(B_ia_P_alpha_.cols());
        if (config_.eri_method != "exact" && n_aux_cphf > 0 && na == na_ && va == va_ &&
            B_oo_P_alpha_.cols() == n_aux_cphf && B_vv_P_alpha_.cols() == n_aux_cphf) {
            Eigen::Map<const Eigen::VectorXd> p_vec_flat(P.data(), va * na);
            Eigen::VectorXd w_cphf = 4.0 * (B_ia_P_alpha_.transpose() * p_vec_flat);
            Eigen::VectorXd j_vo = B_ia_P_alpha_ * w_cphf;
            Eigen::Map<const Eigen::MatrixXd> J_vo_mat(j_vo.data(), va, na);
            AP.noalias() += 2.0 * J_vo_mat;

            Eigen::MatrixXd kappa_ov = P.transpose();
            const int max_t = omp_get_max_threads();
            std::vector<Eigen::MatrixXd> K_tls(max_t, Eigen::MatrixXd::Zero(va, na));
            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                Eigen::MatrixXd T_oo(na, va), T_oo2(na, na);
                #pragma omp for schedule(static) nowait
                for (int K = 0; K < n_aux_cphf; ++K) {
                    Eigen::Map<const Eigen::MatrixXd> Boo(B_oo_P_alpha_.col(K).data(), na, na);
                    Eigen::Map<const Eigen::MatrixXd> Bvo(B_ia_P_alpha_.col(K).data(), va, na);
                    Eigen::Map<const Eigen::MatrixXd> Bvv(B_vv_P_alpha_.col(K).data(), va, va);
                    T_oo.noalias() = Boo * kappa_ov;
                    K_tls[tid].noalias() += (T_oo * Bvv).transpose();
                    T_oo2.noalias() = kappa_ov * Bvo;
          K_tls[tid].noalias() += Bvo * T_oo2;
                }
            }
            for (int t = 0; t < max_t; ++t) {
                AP.noalias() -= 2.0 * K_tls[t];
            }
        } else {
            Eigen::MatrixXd P1 = scf_.C_alpha.leftCols(na) * P.transpose() * scf_.C_alpha.rightCols(va).transpose();
            P1 = (P1 + P1.transpose()).eval();
            Eigen::MatrixXd F1_a, F1_b;
            build_fock_fast(P1, P1, F1_a, F1_b);
            F1_a -= H_core_;
            AP += 2.0 * scf_.C_alpha.rightCols(va).transpose() * F1_a * scf_.C_alpha.leftCols(na);
        }

        // 3. Update Preconditioned Conjugate Gradient
        double alpha = rsold / P.cwiseProduct(AP).sum();
        X += alpha * P;
        R -= alpha * AP;
        
        double rsnew = R.squaredNorm();
        if (std::sqrt(rsnew) < tol) {
            break; // CPHF Konvergen
        }
        
        P = R + (rsnew / rsold) * P;
        rsold = rsnew;
    }
    
    return X;
}

// [INJECTED_PRIORITAS_2E_ANALYTIC_GRAD]
OMP2GradientData OMP2::build_df_gradient_data() {
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    bool has_beta = (!is_restricted && nb_ > 0 && vb_ > 0);
    const int n_aux = static_cast<int>(scf_.L_mat.cols());

    scf_.C_alpha = C_a_current_;
    scf_.C_beta  = C_b_current_;
    execute_micro_iterations();

    const Eigen::MatrixXd Ca_o = scf_.C_alpha.leftCols(na_);
    const Eigen::MatrixXd Ca_v = scf_.C_alpha.rightCols(va_);
    const Eigen::MatrixXd Cb_o = has_beta ? scf_.C_beta.leftCols(nb_) : Eigen::MatrixXd();
    const Eigen::MatrixXd Cb_v = has_beta ? scf_.C_beta.rightCols(vb_) : Eigen::MatrixXd();

    auto* t_aa_blk = t2_aa_.get_block(0, 0, 0, 0);
    auto* t_ab_blk = has_beta ? t2_ab_.get_block(0, 0, 0, 0) : nullptr;
    auto* t_bb_blk = has_beta ? t2_bb_.get_block(0, 0, 0, 0) : nullptr;

    const int nf_a = std::clamp(n_frozen_, 0, na_);
  const int nf_b = std::clamp(n_frozen_, 0, nb_);
  Eigen::MatrixXd X_a = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(na_) * va_, n_aux);
  Eigen::MatrixXd X_b;
  if (has_beta) X_b = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(nb_) * vb_, n_aux);

  if (t_aa_blk || (has_beta && t_ab_blk)) {
    #pragma omp parallel
    {
      Eigen::MatrixXd T_aa_ij(va_, va_);
      Eigen::MatrixXd T_ab_ij;
      if (has_beta && t_ab_blk) T_ab_ij.resize(va_, vb_);
      #pragma omp for schedule(dynamic, 1) nowait
      for (int i = nf_a; i < na_; ++i) {
        auto X_ai = X_a.middleRows(static_cast<Eigen::Index>(i) * va_, va_);
        if (t_aa_blk) {
          for (int j = nf_a; j < na_; ++j) {
            for (int a = 0; a < va_; ++a)
              for (int b = 0; b < va_; ++b)
                T_aa_ij(a, b) = is_restricted
                    ? (2.0 * (*t_aa_blk)(i, a, j, b) - (*t_aa_blk)(i, b, j, a))
                    : (*t_aa_blk)(i, a, j, b);
            X_ai.noalias() += T_aa_ij * B_ia_P_alpha_.middleRows(static_cast<Eigen::Index>(j) * va_, va_);
          }
        }
        if (has_beta && t_ab_blk) {
          for (int jb = nf_b; jb < nb_; ++jb) {
            for (int a = 0; a < va_; ++a)
              for (int b = 0; b < vb_; ++b)
                T_ab_ij(a, b) = (*t_ab_blk)(i, jb, a, b);
            X_ai.noalias() += T_ab_ij * B_ia_P_beta_.middleRows(static_cast<Eigen::Index>(jb) * vb_, vb_);
          }
        }
      }
    }
  }
  if (has_beta && (t_bb_blk || t_ab_blk)) {
    #pragma omp parallel
    {
      Eigen::MatrixXd T_bb_ij(vb_, vb_), T_ba_ij(vb_, va_);
      #pragma omp for schedule(dynamic, 1) nowait
      for (int ib = nf_b; ib < nb_; ++ib) {
        auto X_bi = X_b.middleRows(static_cast<Eigen::Index>(ib) * vb_, vb_);
        if (t_bb_blk) {
          for (int jb = nf_b; jb < nb_; ++jb) {
            for (int a = 0; a < vb_; ++a)
              for (int b = 0; b < vb_; ++b)
                T_bb_ij(a, b) = (*t_bb_blk)(ib, jb, a, b);
            X_bi.noalias() += T_bb_ij * B_ia_P_beta_.middleRows(static_cast<Eigen::Index>(jb) * vb_, vb_);
          }
        }
        if (t_ab_blk) {
          for (int ia = nf_a; ia < na_; ++ia) {
            for (int a = 0; a < va_; ++a)
              for (int b = 0; b < vb_; ++b)
                T_ba_ij(b, a) = (*t_ab_blk)(ia, ib, a, b);
            X_bi.noalias() += T_ba_ij * B_ia_P_alpha_.middleRows(static_cast<Eigen::Index>(ia) * va_, va_);
          }
        }
      }
    }
  }

    Eigen::MatrixXd G_full_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
    G_full_a.block(0, 0, na_, na_)     = G_oo_alpha_;
    G_full_a.block(na_, na_, va_, va_) = G_vv_alpha_;
    Eigen::MatrixXd P_corr_a = scf_.C_alpha * G_full_a * scf_.C_alpha.transpose();

    Eigen::MatrixXd P_corr_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    if (is_restricted) {
        P_corr_b = P_corr_a;
    } else if (has_beta) {
        Eigen::MatrixXd G_full_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
        G_full_b.block(0, 0, nb_, nb_)     = G_oo_beta_;
        G_full_b.block(nb_, nb_, vb_, vb_) = G_vv_beta_;
        P_corr_b = scf_.C_beta * G_full_b * scf_.C_beta.transpose();
    }

    const Eigen::MatrixXd& P_ref_a = scf_.P_alpha;
    const Eigen::MatrixXd& P_ref_b = is_restricted ? scf_.P_alpha : scf_.P_beta;
    Eigen::MatrixXd P_ref_tot  = P_ref_a + P_ref_b;
    Eigen::MatrixXd P_corr_tot = P_corr_a + P_corr_b;

    Eigen::MatrixXd F_HF_mo_a = scf_.C_alpha.transpose() * scf_.F_alpha * scf_.C_alpha;
    Eigen::MatrixXd F_HF_mo_b = has_beta ? (scf_.C_beta.transpose() * scf_.F_beta * scf_.C_beta).eval() : Eigen::MatrixXd();

    Eigen::MatrixXd G_gamma_ao_a, G_gamma_ao_b;
    build_fock_fast(P_corr_a, P_corr_b, G_gamma_ao_a, G_gamma_ao_b);
    G_gamma_ao_a -= H_core_;
    if (has_beta) G_gamma_ao_b -= H_core_;

    Eigen::MatrixXd G_gamma_mo_a = scf_.C_alpha.transpose() * G_gamma_ao_a * scf_.C_alpha;
    Eigen::MatrixXd G_gamma_mo_b = has_beta ? (scf_.C_beta.transpose() * G_gamma_ao_b * scf_.C_beta).eval() : Eigen::MatrixXd();

    // Bagian 1-RDM + separabel dari Fock tergeneralisasi penuh 4-blok
    Eigen::MatrixXd F_1e_sep_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
    F_1e_sep_a.block(0, 0, na_, na_)     = F_HF_mo_a.block(0, 0, na_, na_)
                                         + F_HF_mo_a.block(0, 0, na_, na_) * G_oo_alpha_
                                         + G_gamma_mo_a.block(0, 0, na_, na_);
    F_1e_sep_a.block(na_, na_, va_, va_) = F_HF_mo_a.block(na_, na_, va_, va_) * G_vv_alpha_;
    F_1e_sep_a.block(na_, 0, va_, na_)   = F_HF_mo_a.block(na_, 0, va_, na_)
                                         + F_HF_mo_a.block(na_, 0, va_, na_) * G_oo_alpha_
                                         + G_gamma_mo_a.block(na_, 0, va_, na_);
    F_1e_sep_a.block(0, na_, na_, va_)   = F_HF_mo_a.block(0, na_, na_, va_) * G_vv_alpha_;

    Eigen::MatrixXd F_1e_sep_b;
    if (has_beta) {
        F_1e_sep_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
        F_1e_sep_b.block(0, 0, nb_, nb_)     = F_HF_mo_b.block(0, 0, nb_, nb_)
                                             + F_HF_mo_b.block(0, 0, nb_, nb_) * G_oo_beta_
                                             + G_gamma_mo_b.block(0, 0, nb_, nb_);
        F_1e_sep_b.block(nb_, nb_, vb_, vb_) = F_HF_mo_b.block(nb_, nb_, vb_, vb_) * G_vv_beta_;
        F_1e_sep_b.block(nb_, 0, vb_, nb_)   = F_HF_mo_b.block(nb_, 0, vb_, nb_)
                                             + F_HF_mo_b.block(nb_, 0, vb_, nb_) * G_oo_beta_
                                             + G_gamma_mo_b.block(nb_, 0, vb_, nb_);
        F_1e_sep_b.block(0, nb_, nb_, vb_)   = F_HF_mo_b.block(0, nb_, nb_, vb_) * G_vv_beta_;
    }

    Eigen::MatrixXd F_conn_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
    Eigen::MatrixXd F_conn_b = has_beta ? Eigen::MatrixXd::Zero(nbf_, nbf_) : Eigen::MatrixXd();

    // [INJECTED_PRIORITAS_2D_PACKED_AO]
    Eigen::VectorXd c_ref  = integrals::contract_ao_factor_with_sym(scf_.L_mat, P_ref_tot, nbf_);
    Eigen::VectorXd c_corr = integrals::contract_ao_factor_with_sym(scf_.L_mat, P_corr_tot, nbf_);

    OMP2GradientData out;
    out.P_eff_ao = P_ref_tot + P_corr_tot;
    out.Gamma_3c.resize(static_cast<Eigen::Index>(nbf_) * nbf_, n_aux);

    #pragma omp parallel
    {
        Eigen::MatrixXd Fa_loc = Eigen::MatrixXd::Zero(nbf_, nbf_);
        Eigen::MatrixXd Fb_loc = has_beta ? Eigen::MatrixXd::Zero(nbf_, nbf_) : Eigen::MatrixXd();
        Eigen::MatrixXd G_P(nbf_, nbf_), M_tmp(nbf_, nbf_), Conn_ao(nbf_, nbf_);

        #pragma omp for schedule(static) nowait
        for (int P = 0; P < n_aux; ++P) {
            Eigen::Map<const Eigen::MatrixXd> Boo_a(B_oo_P_alpha_.col(P).data(), na_, na_);
            Eigen::Map<const Eigen::MatrixXd> Bvv_a(B_vv_P_alpha_.col(P).data(), va_, va_);
            Eigen::Map<const Eigen::MatrixXd> Bvo_a(B_ia_P_alpha_.col(P).data(), va_, na_);
            Eigen::Map<const Eigen::MatrixXd> X_ai_a(X_a.col(P).data(), va_, na_);

            Fa_loc.block(0, 0, na_, na_).noalias()     += Bvo_a.transpose() * X_ai_a;
            Fa_loc.block(na_, na_, va_, va_).noalias() += Bvo_a * X_ai_a.transpose();
            Fa_loc.block(na_, 0, va_, na_).noalias()   += Bvv_a * X_ai_a;
            Fa_loc.block(0, na_, na_, va_).noalias()   += Boo_a * X_ai_a.transpose();

            // Coulomb separabel
            G_P.noalias() = out.P_eff_ao * c_ref(P) + P_ref_tot * c_corr(P);

            // Exchange separabel (alpha)
            const double k_scale = is_restricted ? 2.0 : 1.0;
            M_tmp.noalias() = Ca_o * Boo_a * Ca_o.transpose();
            Eigen::MatrixXd M_corr_a = Ca_o * (G_oo_alpha_ * Boo_a) * Ca_o.transpose()
                                     + Ca_v * (G_vv_alpha_ * Bvo_a) * Ca_o.transpose();
            G_P.noalias() -= k_scale * (M_tmp + M_corr_a + M_corr_a.transpose());

            // Connected 2-RDM (alpha): 0.5 untuk RHF (X_a = 1*E_corr), 0.25 untuk UHF (X_a+X_b = 2*E_corr)
            Conn_ao.noalias() = Ca_v * X_ai_a * Ca_o.transpose();
            // [FIXED_2E_SCALE_FACTORS]
            G_P.noalias() += (is_restricted ? 2.0 : 1.0) * (Conn_ao + Conn_ao.transpose());

            if (has_beta) {
                Eigen::Map<const Eigen::MatrixXd> Boo_b(B_oo_P_beta_.col(P).data(), nb_, nb_);
                Eigen::Map<const Eigen::MatrixXd> Bvv_b(B_vv_P_beta_.col(P).data(), vb_, vb_);
                Eigen::Map<const Eigen::MatrixXd> Bvo_b(B_ia_P_beta_.col(P).data(), vb_, nb_);
                Eigen::Map<const Eigen::MatrixXd> X_bi_b(X_b.col(P).data(), vb_, nb_);

                Fb_loc.block(0, 0, nb_, nb_).noalias()     += Bvo_b.transpose() * X_bi_b;
                Fb_loc.block(nb_, nb_, vb_, vb_).noalias() += Bvo_b * X_bi_b.transpose();
                Fb_loc.block(nb_, 0, vb_, nb_).noalias()   += Bvv_b * X_bi_b;
                Fb_loc.block(0, nb_, nb_, vb_).noalias()   += Boo_b * X_bi_b.transpose();

                M_tmp.noalias() = Cb_o * Boo_b * Cb_o.transpose();
                Eigen::MatrixXd M_corr_b = Cb_o * (G_oo_beta_ * Boo_b) * Cb_o.transpose()
                                         + Cb_v * (G_vv_beta_ * Bvo_b) * Cb_o.transpose();
                G_P.noalias() -= (M_tmp + M_corr_b + M_corr_b.transpose());

                Conn_ao.noalias() = Cb_v * X_bi_b * Cb_o.transpose();
                G_P.noalias() += 1.0 * (Conn_ao + Conn_ao.transpose());
            }

            out.Gamma_3c.col(P) = Eigen::Map<const Eigen::VectorXd>(G_P.data(), nbf_ * nbf_);
        }

        #pragma omp critical
        {
            F_conn_a.noalias() += Fa_loc;
            if (has_beta) F_conn_b.noalias() += Fb_loc;
        }
    }

    F_1e_sep_a = 0.5 * (F_1e_sep_a + F_1e_sep_a.transpose());
    F_conn_a   = 0.5 * (F_conn_a   + F_conn_a.transpose());

    if (is_restricted) {
        out.W_eff_ao = 2.0 * (scf_.C_alpha * (F_1e_sep_a + F_conn_a) * scf_.C_alpha.transpose());
    } else {
        out.W_eff_ao = scf_.C_alpha * (F_1e_sep_a + F_conn_a) * scf_.C_alpha.transpose();
        if (has_beta) {
            F_1e_sep_b = 0.5 * (F_1e_sep_b + F_1e_sep_b.transpose());
            F_conn_b   = 0.5 * (F_conn_b   + F_conn_b.transpose());
            out.W_eff_ao.noalias() += scf_.C_beta * (F_1e_sep_b + F_conn_b) * scf_.C_beta.transpose();
        }
    }

    out.gamma_2c = integrals::compute_gamma2c_from_full_and_L(out.Gamma_3c, scf_.L_mat, nbf_);

    return out;
}
} // namespace mshqc