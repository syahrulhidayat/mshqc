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
    Eigen::MatrixXd T2_rmp2, T2_aa_mat, T2_ab_mat, T2_bb_mat;

    if (is_restricted && t_aa_blk) {
        T2_rmp2 = Eigen::MatrixXd::Zero(na_ * va_, na_ * va_);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < na_; ++i) {
            for (int a = 0; a < va_; ++a) {
                for (int j = 0; j < na_; ++j) {
                    for (int b = 0; b < va_; ++b) {
                        T2_rmp2(i * va_ + a, j * va_ + b) =
                            2.0 * (*t_aa_blk)(i, a, j, b) - 1.0 * (*t_aa_blk)(i, b, j, a);
                    }
                }
            }
        }
    } else if (!is_restricted) {
        if (t_aa_blk) {
            T2_aa_mat = Eigen::MatrixXd::Zero(na_ * va_, na_ * va_);
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < na_; ++i) {
                for (int a = 0; a < va_; ++a) {
                    for (int j = 0; j < na_; ++j) {
                        for (int b = 0; b < va_; ++b) {
                            T2_aa_mat(i * va_ + a, j * va_ + b) = (*t_aa_blk)(i, a, j, b);
                        }
                    }
                }
            }
        }

        if (t_ab_blk) {
            T2_ab_mat = Eigen::MatrixXd::Zero(na_ * va_, nb_ * vb_);
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < na_; ++i) {
                for (int a = 0; a < va_; ++a) {
                    for (int j = 0; j < nb_; ++j) {
                        for (int b = 0; b < vb_; ++b) {
                            T2_ab_mat(i * va_ + a, j * vb_ + b) = (*t_ab_blk)(i, j, a, b);
                        }
                    }
                }
            }
        }

        if (t_bb_blk) {
            T2_bb_mat = Eigen::MatrixXd::Zero(nb_ * vb_, nb_ * vb_);
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < nb_; ++i) {
                for (int a = 0; a < vb_; ++a) {
                    for (int j = 0; j < nb_; ++j) {
                        for (int b = 0; b < vb_; ++b) {
                            T2_bb_mat(i * vb_ + a, j * vb_ + b) = (*t_bb_blk)(i, j, a, b);
                        }
                    }
                }
            }
        }
    }
    const int CHUNK_SIZE = 128;

    #pragma omp parallel
    {
        Eigen::MatrixXd X_a_buf(na_ * va_, CHUNK_SIZE);
        Eigen::MatrixXd X_b_buf;
        if (has_beta) X_b_buf.resize(nb_ * vb_, CHUNK_SIZE);

        Eigen::MatrixXd Z_loc_a = Eigen::MatrixXd::Zero(va_, na_);
        Eigen::MatrixXd Z_loc_b = Eigen::MatrixXd::Zero(vb_, nb_);

        Eigen::MatrixXd B_oo_a(na_, na_), B_vv_a(va_, va_);
        Eigen::MatrixXd B_oo_b(nb_, nb_), B_vv_b(vb_, vb_);

        #pragma omp for schedule(dynamic)
        for (int P_start = 0; P_start < n_chol; P_start += CHUNK_SIZE) {
            int P_len = std::min(CHUNK_SIZE, n_chol - P_start);

            Eigen::Map<Eigen::MatrixXd> X_a_chunk(X_a_buf.data(), na_ * va_, P_len);
            X_a_chunk.setZero();

            Eigen::Map<Eigen::MatrixXd> X_b_chunk(X_b_buf.data(), has_beta ? nb_ * vb_ : 0, P_len);
            if (has_beta) X_b_chunk.setZero();

            Eigen::MatrixXd Bia_chunk = B_ia_P_alpha_.middleCols(P_start, P_len);

            if (t_aa_blk) {
                if (is_restricted) {
                    X_a_chunk.noalias() += T2_rmp2 * Bia_chunk;
                } else {
                    X_a_chunk.noalias() += 1.0 * T2_aa_mat * Bia_chunk;
                }
            }

            if (has_beta) {
                Eigen::MatrixXd Bib_chunk = B_ia_P_beta_.middleCols(P_start, P_len);

                if (t_ab_blk) {
                    X_a_chunk.noalias() += 1.0 * T2_ab_mat * Bib_chunk;
                    X_b_chunk.noalias() += 1.0 * T2_ab_mat.transpose() * Bia_chunk;
                }

                if (t_bb_blk) {
                    X_b_chunk.noalias() += 1.0 * T2_bb_mat * Bib_chunk;
                }
            }

            for (int p = 0; p < P_len; ++p) {
                int P_global = P_start + p;
                Eigen::Map<const Eigen::MatrixXd> B_AO(scf_.L_mat.col(P_global).data(), nbf_, nbf_);

                B_oo_a.noalias() = scf_.C_alpha.leftCols(na_).transpose() * (B_AO * scf_.C_alpha.leftCols(na_));
                B_vv_a.noalias() = scf_.C_alpha.rightCols(va_).transpose() * (B_AO * scf_.C_alpha.rightCols(va_));

                Eigen::Map<Eigen::MatrixXd> XT_a(X_a_chunk.col(p).data(), va_, na_);
                Z_loc_a.noalias() += B_vv_a * XT_a - XT_a * B_oo_a;

                if (has_beta) {
                    B_oo_b.noalias() = scf_.C_beta.leftCols(nb_).transpose() * (B_AO * scf_.C_beta.leftCols(nb_));
                    B_vv_b.noalias() = scf_.C_beta.rightCols(vb_).transpose() * (B_AO * scf_.C_beta.rightCols(vb_));

                    Eigen::Map<Eigen::MatrixXd> XT_b(X_b_chunk.col(p).data(), vb_, nb_);
                    Z_loc_b.noalias() += B_vv_b * XT_b - XT_b * B_oo_b;
                }
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
            diag_H(idx++) = spin_factor * safe_diff + 2.0 * spin_factor * J_ia + level_shift;
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
        int dim_b = (is_restricted) ? 0 : (vb_ * nb_);

        int temp_idx = 0;
        for (int i = 0; i < na_; ++i) {
            for (int a = 0; a < va_; ++a) {
                double eps_diff = scf_.orbital_energies_alpha(na_ + a) - scf_.orbital_energies_alpha(i);
                Hp(temp_idx) = spin_factor * std::max(std::abs(eps_diff), 1e-4) * p_vec(temp_idx);
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
}
