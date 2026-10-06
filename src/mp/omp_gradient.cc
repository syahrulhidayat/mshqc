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
#include "mshqc/gradient/optimizer.h"
#include <omp.h>
#include <tblis/tblis.h>
#include <unsupported/Eigen/MatrixFunctions>

namespace mshqc {

void OMP2::evaluate_z_vector_cholesky(Eigen::MatrixXd& Z_mat_a, Eigen::MatrixXd& Z_mat_b) {
    int n_chol = scf_.L_mat.cols();
    Z_mat_a.setZero(va_, na_);
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    bool has_beta = (!is_restricted && nb_ > 0 && vb_ > 0);

    if (has_beta) Z_mat_b.setZero(vb_, nb_);

    auto* t_aa_blk = t2_aa_.get_block(0,0,0,0);
    auto* t_ab_blk = has_beta ? t2_ab_.get_block(0,0,0,0) : nullptr;
    auto* t_bb_blk = has_beta ? t2_bb_.get_block(0,0,0,0) : nullptr;
    
    Eigen::MatrixXd T2_aa = Eigen::MatrixXd::Zero(na_ * va_, na_ * va_);
    if (is_restricted && t_aa_blk) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < na_; ++i) {
            for (int a = 0; a < va_; ++a) {
                for (int j = 0; j < na_; ++j) {
                    for (int b = 0; b < va_; ++b) {
                        // PERBAIKAN AKURASI: Transposisi indeks virtual (2 * t_ij^ba - t_ij^ab)
                        T2_aa(i * va_ + a, j * va_ + b) = 
                            2.0 * (*t_aa_blk)(i, b, j, a) - 1.0 * (*t_aa_blk)(i, a, j, b);
                    }
                }
            }
        }
    } else if (!is_restricted && t_aa_blk) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < na_; ++i) {
            for (int a = 0; a < va_; ++a) {
                for (int j = 0; j < na_; ++j) {
                    for (int b = 0; b < va_; ++b) {
                        T2_aa(i * va_ + a, j * va_ + b) = (*t_aa_blk)(i, a, j, b);
                    }
                }
            }
        }
    }

    Eigen::MatrixXd T2_ab, T2_bb;
    if (has_beta) {
        T2_ab = Eigen::MatrixXd::Zero(na_ * va_, nb_ * vb_);
        if (t_ab_blk) {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < na_; ++i) {
                for (int a = 0; a < va_; ++a) {
                    for (int j = 0; j < nb_; ++j) {
                        for (int b = 0; b < vb_; ++b) {
                            T2_ab(i * va_ + a, j * vb_ + b) = (*t_ab_blk)(i, j, a, b);
                        }
                    }
                }
            }
        }
        T2_bb = Eigen::MatrixXd::Zero(nb_ * vb_, nb_ * vb_);
        if (t_bb_blk) {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < nb_; ++i) {
                for (int a = 0; a < vb_; ++a) {
                    for (int j = 0; j < nb_; ++j) {
                        for (int b = 0; b < vb_; ++b) {
                            T2_bb(i * vb_ + a, j * vb_ + b) = (*t_bb_blk)(i, j, a, b);
                        }
                    }
                }
            }
        }
    }

    // DGEMM Raksasa - O(N^3) Sangat efisien karena mengandalkan BLAS Level 3
    Eigen::MatrixXd X_a = T2_aa * B_ia_P_alpha_;
    Eigen::MatrixXd X_b;
    
    if (has_beta) {
        X_a += T2_ab * B_ia_P_beta_;
        X_b = T2_bb * B_ia_P_beta_ + T2_ab.transpose() * B_ia_P_alpha_;
    }

    #pragma omp parallel
    {
        Eigen::MatrixXd Z_loc_a = Eigen::MatrixXd::Zero(va_, na_);
        Eigen::MatrixXd Z_loc_b;
        if (has_beta) Z_loc_b = Eigen::MatrixXd::Zero(vb_, nb_);

        #pragma omp for schedule(dynamic)
        for (int P = 0; P < n_chol; ++P) {
            // Evaluasi in-place menggunakan Eigen::Map tanpa alokasi array baru
            Eigen::Map< const Eigen::MatrixXd > B_oo_a(B_oo_P_alpha_.col(P).data(), na_, na_);
            Eigen::Map< const Eigen::MatrixXd > B_vv_a(B_vv_P_alpha_.col(P).data(), va_, va_);
            Eigen::Map< const Eigen::MatrixXd > X_ai(X_a.col(P).data(), va_, na_);

            Z_loc_a.noalias() += B_vv_a * X_ai - X_ai * B_oo_a;

            if (has_beta) {
                Eigen::Map< const Eigen::MatrixXd > B_oo_b(B_oo_P_beta_.col(P).data(), nb_, nb_);
                Eigen::Map< const Eigen::MatrixXd > B_vv_b(B_vv_P_beta_.col(P).data(), vb_, vb_);
                Eigen::Map< const Eigen::MatrixXd > X_bi(X_b.col(P).data(), vb_, nb_);

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
    bool is_rohf = (n_params == na_ * va_ && !is_restricted); 
    
    double spin_factor = (is_restricted || is_rohf) ? 4.0 : 2.0;
    
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
            double factor = (is_rohf && i >= nb_) ? 2.0 : spin_factor; 
            diag_H(idx++) = factor * safe_diff + 2.0 * factor * J_ia + level_shift;  
        }
    }
    
    if (!is_restricted && nb_ > 0 && !is_rohf) {
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
    bool is_rohf = (n_params == na_ * va_ && !is_restricted);
    double spin_factor = (is_restricted || is_rohf) ? 4.0 : 2.0;

    mshqc::gradient::TrustRegionConfig tr_conf;
    tr_conf.micro_thresh = std::min(1e-4, grad_norm * 0.1); 
    mshqc::gradient::TrustRegionSOSCF soscf_engine(tr_conf);

    auto compute_hessian_vector = [&](const Eigen::VectorXd& p_vec) -> Eigen::VectorXd {
        Eigen::VectorXd Hp = Eigen::VectorXd::Zero(n_params);
        int dim_a = va_ * na_;
        int dim_b = (is_restricted || is_rohf) ? 0 : (vb_ * nb_); 
        
        int temp_idx = 0;
        for (int i = 0; i < na_; ++i) {
            for (int a = 0; a < va_; ++a) {
                double eps_diff = scf_.orbital_energies_alpha(na_ + a) - scf_.orbital_energies_alpha(i);
                double factor = (is_rohf && i >= nb_) ? 2.0 : spin_factor; 
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
        
        Eigen::MatrixXd P1_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
        if (dim_a > 0) {
            P1_a = scf_.C_alpha.leftCols(na_) * kappa_a * scf_.C_alpha.rightCols(va_).transpose();
            P1_a = (P1_a + P1_a.transpose()).eval(); 
        }
        
        Eigen::MatrixXd P1_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
        if (!is_restricted && dim_b > 0) {
            P1_b = scf_.C_beta.leftCols(nb_) * kappa_b * scf_.C_beta.rightCols(vb_).transpose();
            P1_b = (P1_b + P1_b.transpose()).eval();
        } else if (is_restricted || is_rohf) {
            P1_b = P1_a; 
        }
    
        Eigen::MatrixXd F1_a, F1_b;
        build_fock_fast(P1_a, P1_b, F1_a, F1_b);
        F1_a -= H_core_; 
        if (!is_restricted || nb_ > 0) F1_b -= H_core_;
    
        if (dim_a > 0) {
            Eigen::MatrixXd H_kappa_a = scf_.C_alpha.leftCols(na_).transpose() * F1_a * scf_.C_alpha.rightCols(va_);
            Eigen::MatrixXd H_kappa_b;
            if (is_rohf) {
                H_kappa_b = scf_.C_beta.leftCols(nb_).transpose() * F1_b * scf_.C_beta.rightCols(vb_);
            }
            
            int idx_h = 0;
            for (int i = 0; i < na_; ++i) {
                for (int a = 0; a < va_; ++a) {
                    double val = H_kappa_a(i, a);
                    if (is_rohf && i < nb_) {
                        int b_idx = (na_ + a) - nb_; 
                        val = 0.5 * (val + H_kappa_b(i, b_idx)); 
                    }
                    double factor = (is_rohf && i >= nb_) ? 2.0 : spin_factor; 
                    Hp(idx_h++) += factor * val; 
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
    bool is_rohf = (kappa.size() == na_ * va_ && !is_restricted);
    
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
    C_a_current_ = C_a_current_ * K_a.exp();

    // Sinkronisasi spasial absolut (Memutus cost overhead Matrix Exponentials)
    if (is_rohf) {
        C_b_current_ = C_a_current_;
        return; 
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
        C_b_current_ = C_b_current_ * K_b.exp();
    }
}
} // namespace mshqc