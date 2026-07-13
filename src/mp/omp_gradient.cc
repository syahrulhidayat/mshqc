#include "mshqc/mp2/mp2.h"
#include "mshqc/gradient/optimizer.h"
#include <omp.h>
#include <tblis/tblis.h>
#include <unsupported/Eigen/MatrixFunctions>

namespace mshqc {

// CUT dari mp2.cc dan PASTE ke sini:
void OMP2::evaluate_z_vector_cholesky(Eigen::MatrixXd& Z_mat_a, Eigen::MatrixXd& Z_mat_b) {
    int n_chol = scf_.L_mat.cols();
    Z_mat_a.setZero(va_, na_);
    if (nb_ > 0) Z_mat_b.setZero(vb_, nb_);

    auto* t_aa_blk = t2_aa_.get_block(0,0,0,0);
    auto* t_ab_blk = (nb_ > 0 && vb_ > 0) ? t2_ab_.get_block(0,0,0,0) : nullptr;
    auto* t_bb_blk = (nb_ > 0 && vb_ > 0) ? t2_bb_.get_block(0,0,0,0) : nullptr;

    const int CHUNK_SIZE = 128; 
    using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    #pragma omp parallel
    {
        // PERBAIKAN: Pre-alokasi buffer maksimal di LUAR kalang untuk mencegah Heap Lock!
        Eigen::MatrixXd X_a_buf(na_ * va_, CHUNK_SIZE);
        Eigen::MatrixXd X_b_buf;
        if (nb_ > 0 && vb_ > 0) X_b_buf.resize(nb_ * vb_, CHUNK_SIZE);

        Eigen::MatrixXd Z_loc_a = Eigen::MatrixXd::Zero(va_, na_);
        Eigen::MatrixXd Z_loc_b = Eigen::MatrixXd::Zero(vb_, nb_);

        Eigen::MatrixXd B_oo_a(na_, na_), B_vv_a(va_, va_);
        Eigen::MatrixXd B_oo_b(nb_, nb_), B_vv_b(vb_, vb_);

        #pragma omp for schedule(dynamic)
        for (int P_start = 0; P_start < n_chol; P_start += CHUNK_SIZE) {
            int P_len = std::min(CHUNK_SIZE, n_chol - P_start);

            // PERBAIKAN: Pemetaan dinamis Zero-Allocation menggunakan Eigen::Map
            Eigen::Map<Eigen::MatrixXd> X_a_chunk(X_a_buf.data(), na_ * va_, P_len);
            X_a_chunk.setZero();
            
            Eigen::Map<Eigen::MatrixXd> X_b_chunk(X_b_buf.data(), (nb_ > 0 && vb_ > 0) ? nb_ * vb_ : 0, P_len);
            if (nb_ > 0 && vb_ > 0) X_b_chunk.setZero();

            Eigen::MatrixXd Bia_chunk = B_ia_P_alpha_.middleCols(P_start, P_len);

            // Kontraksi In-the-fly dengan BLAS DGEMM
            if (t_aa_blk) {
                Eigen::Map<const MatrixXdRowMajor> T2_aa_map(t_aa_blk->data(), na_ * va_, na_ * va_);
                X_a_chunk.noalias() += T2_aa_map * Bia_chunk;
            }

            if (nb_ > 0 && vb_ > 0) {
                Eigen::MatrixXd Bib_chunk = B_ia_P_beta_.middleCols(P_start, P_len);
                
                if (t_ab_blk) {
                    Eigen::Map<const MatrixXdRowMajor> T2_ab_map(t_ab_blk->data(), na_ * va_, nb_ * vb_);
                    X_a_chunk.noalias() += T2_ab_map * Bib_chunk;
                    X_b_chunk.noalias() += T2_ab_map.transpose() * Bia_chunk;
                }

                if (t_bb_blk) {
                    Eigen::Map<const MatrixXdRowMajor> T2_bb_map(t_bb_blk->data(), nb_ * vb_, nb_ * vb_);
                    X_b_chunk.noalias() += T2_bb_map * Bib_chunk;
                }
            }

            // (Lanjutkan transformasi matriks seperti biasa...)
            for (int p = 0; p < P_len; ++p) {
                int P_global = P_start + p;
                Eigen::Map<const Eigen::MatrixXd> B_AO(scf_.L_mat.col(P_global).data(), nbf_, nbf_);

                B_oo_a.noalias() = scf_.C_alpha.leftCols(na_).transpose() * (B_AO * scf_.C_alpha.leftCols(na_));
                B_vv_a.noalias() = scf_.C_alpha.rightCols(va_).transpose() * (B_AO * scf_.C_alpha.rightCols(va_));

                Eigen::Map<Eigen::MatrixXd> XT_a(X_a_chunk.col(p).data(), va_, na_);
                Z_loc_a.noalias() += B_vv_a * XT_a - XT_a * B_oo_a;

                if (nb_ > 0 && vb_ > 0) {
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
            if (nb_ > 0) Z_mat_b += Z_loc_b;
        }
    }
}
Eigen::VectorXd OMP2::compute_soscf_step() {
    int n_params = orbital_gradient_.size();
    if (n_params == 0) return Eigen::VectorXd::Zero(0);

    if (hessian_diag_.size() != n_params) hessian_diag_.resize(n_params);
    int idx = 0;
    double grad_norm = orbital_gradient_.norm(); 
    
    double level_shift = (grad_norm > 0.1) ? 0.05 : 0.005; 
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    
    double spin_factor = is_restricted ? 4.0 : 2.0;
    for (int a = 0; a < va_; ++a) {
        for (int i = 0; i < na_; ++i) {
            double eps_diff = scf_.orbital_energies_alpha(na_ + a) - scf_.orbital_energies_alpha(i);
            double J_ia = 0.0;
            
            if (config_.eri_method != "exact") {
                J_ia = B_ia_P_alpha_.row(a * na_ + i).squaredNorm(); 
            } else {
                auto* g_blk = g_aa_.get_block(0, 0, 0, 0);
                if (g_blk) J_ia = std::abs((*g_blk)(i, a, i, a));
            }
            hessian_diag_(idx++) = spin_factor * std::abs(eps_diff) + spin_factor * J_ia + level_shift; 
        }
    }

    if (!is_restricted && nb_ > 0) {
        for (int a = 0; a < vb_; ++a) {
            for (int i = 0; i < nb_; ++i) {
                double eps_diff = scf_.orbital_energies_beta(nb_ + a) - scf_.orbital_energies_beta(i);
                double J_ia = 0.0;
                
                if (config_.eri_method != "exact") {
                    J_ia = B_ia_P_beta_.row(a * nb_ + i).squaredNorm();
                } else {
                    auto* g_blk = g_bb_.get_block(0, 0, 0, 0);
                    if (g_blk) J_ia = std::abs((*g_blk)(i, a, i, a));
                }
                hessian_diag_(idx++) = 2.0 * std::abs(eps_diff) + 2.0 * J_ia + level_shift;
            }
        }
    }

    for(int i = 0; i < hessian_diag_.size(); ++i) {
        if(std::abs(hessian_diag_(i)) < 1e-12) hessian_diag_(i) = 1e-12; 
    }

    Eigen::VectorXd kappa = -orbital_gradient_.cwiseQuotient(hessian_diag_);

    if (kappa.hasNaN() || !kappa.allFinite()) {
        std::cerr << "  [CRITICAL] Orbital gradient contains NaN/Inf! Fallback to zero rotation." << std::endl;
        kappa.setZero();
    }
    
    double max_step = 0.15;
    double step_norm = kappa.norm(); 
    
    if (step_norm > max_step) {
        kappa *= (max_step / step_norm); 
    }

    return kappa;
}
void OMP2::apply_orbital_rotation(const Eigen::VectorXd& kappa) {
    if (kappa.norm() < 1e-12) return;

    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);

    auto compute_exact_unitary = [](const Eigen::VectorXd& k_vec, int n_occ, int n_vir) -> Eigen::MatrixXd {
        int n_mo = n_occ + n_vir;
        Eigen::MatrixXd K_full = Eigen::MatrixXd::Zero(n_mo, n_mo);
        Eigen::Map<const Eigen::MatrixXd> kappa_mat(k_vec.data(), n_vir, n_occ);
        K_full.block(n_occ, 0, n_vir, n_occ) = kappa_mat;
        K_full.block(0, n_occ, n_occ, n_vir) = -kappa_mat.transpose();
        return K_full.exp(); 
    };
    int len_a = na_ * va_;
    Eigen::VectorXd kappa_a = kappa.head(len_a);
    Eigen::MatrixXd U_a = compute_exact_unitary(kappa_a, na_, va_);
    C_a_current_ = C_a_current_ * U_a;

    if (nb_ > 0 && vb_ > 0) {
        if (!is_restricted) {
            int len_b = nb_ * vb_;
            Eigen::VectorXd kappa_b = kappa.segment(len_a, len_b);
            Eigen::MatrixXd U_b = compute_exact_unitary(kappa_b, nb_, vb_);
            C_b_current_ = C_b_current_ * U_b;
        } else {
            C_b_current_ = C_b_current_ * U_a;
        }
    }
}
} // namespace mshqc