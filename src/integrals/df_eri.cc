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
 * @file src/integrals/df_eri.cc
 * @brief Implementasi Density Fitting Tensor Generator
 */

#include "mshqc/integrals/df_eri.h"
#include "mshqc/integrals/packed_ao_utils.h"
#include "mshqc/utils/hdf5_io.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include <omp.h>

namespace mshqc {
namespace integrals {

DensityFittingERI::DensityFittingERI(const BasisSet& primary_basis, 
                                     const BasisSet& aux_basis, 
                                     std::shared_ptr<IntegralEngine> integrals,
                                     double cutoff)
    : primary_basis_(&primary_basis), 
      aux_basis_(&aux_basis), 
      integrals_(integrals),
      is_computed_(false),
      cutoff_(cutoff) 
{
    n_primary_ = primary_basis_->n_basis_functions();
    n_aux_     = aux_basis_->n_basis_functions();
}

struct CholeskyResult {
    Eigen::MatrixXd L;
    std::vector<int> pivot_map;
    int rank;
};

CholeskyResult compute_pivoted_cholesky(Eigen::MatrixXd& J, double tol = 1e-10) {
    int n = J.rows();
    std::vector<int> p(n);
    std::iota(p.begin(), p.end(), 0); 
    
    Eigen::VectorXd diag = J.diagonal();
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(n, n);
    
    int rank = 0;
    for (int i = 0; i < n; ++i) {
        int max_idx = i;
        double max_val = diag(i);
        for (int j = i + 1; j < n; ++j) {
            if (diag(j) > max_val) {
                max_val = diag(j);
                max_idx = j;
            }
        }
        
        if (max_val < tol) break; 
        
        if (max_idx != i) {
            std::swap(p[i], p[max_idx]);
            std::swap(diag(i), diag(max_idx));
            J.row(i).swap(J.row(max_idx));
            J.col(i).swap(J.col(max_idx));
            L.row(i).swap(L.row(max_idx)); 
        }
        
        L(i, i) = std::sqrt(diag(i));
        double L_ii_inv = 1.0 / L(i, i);
        
        int rem = n - i - 1;
        if (rem > 0) {
            if (i > 0) {
                L.col(i).tail(rem).noalias() = J.col(i).tail(rem)
                    - L.bottomLeftCorner(rem, i) * L.row(i).head(i).transpose();
            } else {
                L.col(i).tail(rem) = J.col(i).tail(rem);
            }
            L.col(i).tail(rem) *= L_ii_inv;
            diag.tail(rem).array() -= L.col(i).tail(rem).array().square();
        }
        rank++;
    }

    L.conservativeResize(n, rank);
    return {L, p, rank};
}

// [INJECTED_PRIORITAS_2D_PACKED_AO]
void DensityFittingERI::compute(bool save_hdf5) {
    std::cout << "  Starting Density Fitting (RI) Decomposition (Packed Triangular AO)...\n";
    std::cout << "  Primary Basis: " << n_primary_ << " BF\n";
    std::cout << "  Auxiliary Basis: " << n_aux_ << " BF\n";

    const int n_shells_prim = primary_basis_->n_shells();
    const int n_shells_aux  = aux_basis_->n_shells();

    std::vector<int> aux_starts(n_shells_aux), aux_sizes(n_shells_aux);
    int offset_a = 0, max_dim_a = 1;
    for (int i = 0; i < n_shells_aux; ++i) {
        aux_starts[i] = offset_a;
        aux_sizes[i] = aux_basis_->shell(i).n_functions();
        max_dim_a = std::max(max_dim_a, aux_sizes[i]);
        offset_a += aux_sizes[i];
    }

    std::vector<int> prim_starts(n_shells_prim), prim_sizes(n_shells_prim);
    int offset_p = 0, max_dim_p = 1;
    for (int i = 0; i < n_shells_prim; ++i) {
        prim_starts[i] = offset_p;
        prim_sizes[i] = primary_basis_->shell(i).n_functions();
        max_dim_p = std::max(max_dim_p, prim_sizes[i]);
        offset_p += prim_sizes[i];
    }

    const size_t buf_2c_sz = std::max<size_t>(64, static_cast<size_t>(max_dim_a) * max_dim_a);
    const size_t buf_3c_sz = std::max<size_t>(64, static_cast<size_t>(max_dim_p) * max_dim_p * max_dim_a);
    const size_t buf_4c_sz = std::max<size_t>(64, static_cast<size_t>(max_dim_p) * max_dim_p * max_dim_p * max_dim_p);

    // 1. Matriks Metrik 2-Pusat J_metric
    Eigen::MatrixXd J_metric = Eigen::MatrixXd::Zero(n_aux_, n_aux_);
    #pragma omp parallel
    {
        std::vector<double> buffer(buf_2c_sz);
        #pragma omp for schedule(dynamic, 1) nowait
        for (int R = 0; R < n_shells_aux; ++R) {
            int abs_R = n_shells_prim + R;
            int dim_R = aux_sizes[R];
            int bf_R_start = aux_starts[R];

            for (int S = 0; S <= R; ++S) {
                int abs_S = n_shells_prim + S;
                int dim_S = aux_sizes[S];
                int bf_S_start = aux_starts[S];

                if (!integrals_->compute_2c2e_block_raw(abs_R, abs_S, buffer.data())) continue;

                for (int s = 0; s < dim_S; ++s) {
                    int idx_S = bf_S_start + s;
                    int buffer_offset = dim_R * s;
                    for (int r = 0; r < dim_R; ++r) {
                        int idx_R = bf_R_start + r;
                        double val = buffer[r + buffer_offset];
                        J_metric(idx_R, idx_S) = val;
                        if (idx_R != idx_S) J_metric(idx_S, idx_R) = val;
                    }
                }
            }
        }
    }

    std::vector<double> aux_schwarz(n_shells_aux, 0.0);
    for (int R = 0; R < n_shells_aux; ++R) {
        double max_diag = 0.0;
        for (int r = 0; r < aux_sizes[R]; ++r) {
            max_diag = std::max(max_diag, std::abs(J_metric(aux_starts[R] + r, aux_starts[R] + r)));
        }
        aux_schwarz[R] = std::sqrt(max_diag);
    }

    std::vector<double> prim_schwarz(static_cast<size_t>(n_shells_prim) * n_shells_prim, 0.0);
    #pragma omp parallel
    {
        std::vector<double> buffer(buf_4c_sz);
        #pragma omp for schedule(dynamic, 1) nowait
        for (int i = 0; i < n_shells_prim; ++i) {
            int dim_i = prim_sizes[i];
            for (int j = 0; j <= i; ++j) {
                int dim_j = prim_sizes[j];
                if (!integrals_->compute_shell_block_raw(i, j, i, j, buffer.data())) continue;
                double max_v = 0.0;
                for (int p_i = 0; p_i < dim_i; ++p_i) {
                    for (int p_j = 0; p_j < dim_j; ++p_j) {
                        size_t idx = p_i + dim_i * (p_j + dim_j * (p_i + dim_i * p_j));
                        max_v = std::max(max_v, std::abs(buffer[idx]));
                    }
                }
                double s_val = std::sqrt(max_v);
                prim_schwarz[i * n_shells_prim + j] = s_val;
                prim_schwarz[j * n_shells_prim + i] = s_val;
            }
        }
    }

    // 2. Dekomposisi Eigen J^(-1/2)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J_metric);
    const Eigen::VectorXd& evals = es.eigenvalues();
    const Eigen::MatrixXd& evecs = es.eigenvectors();

    double max_eval = (evals.size() > 0) ? std::max(1.0, evals.maxCoeff()) : 1.0;
    double rel_cond = (cutoff_ > 0.0 && cutoff_ <= 1e-10) ? cutoff_ : 1e-12;
    double eig_tol = std::max(1e-14, rel_cond * max_eval);

    int n_aux_rank = 0;
    for (int i = 0; i < n_aux_; ++i) {
        if (evals(i) >= eig_tol) n_aux_rank++;
    }

    Eigen::MatrixXd U_kept(n_aux_, n_aux_rank);
    Eigen::VectorXd inv_sqrt_evals(n_aux_rank);
    int col_idx = 0;
    for (int i = 0; i < n_aux_; ++i) {
        if (evals(i) >= eig_tol) {
            U_kept.col(col_idx) = evecs.col(i);
            inv_sqrt_evals(col_idx) = 1.0 / std::sqrt(evals(i));
            col_idx++;
        }
    }

    Eigen::MatrixXd U_scaled = U_kept * inv_sqrt_evals.asDiagonal();
    Eigen::MatrixXd J_inv_sqrt(n_aux_, n_aux_);
    constexpr int tile_j = 64;
    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < n_aux_; r += tile_j) {
        int rows = std::min(tile_j, n_aux_ - r);
        J_inv_sqrt.middleRows(r, rows).noalias() = U_scaled.middleRows(r, rows) * U_kept.transpose();
    }

    // 3. Evaluasi Integral 3-Pusat dalam Penyimpanan Segitiga Terkemas (n_tri x n_aux)
    const int n_tri = ao_tri_size(n_primary_);
    Eigen::MatrixXd V_raw = Eigen::MatrixXd::Zero(n_tri, n_aux_);
    const double screen_tol = 1e-14;

    #pragma omp parallel
    {
        std::vector<double> buffer(buf_3c_sz);
        #pragma omp for schedule(dynamic, 1) nowait
        for (int i = 0; i < n_shells_prim; ++i) {
            const int bf_i_start = prim_starts[i];
            const int dim_i = prim_sizes[i];

            for (int j = 0; j <= i; ++j) {
                const double q_ij = prim_schwarz[i * n_shells_prim + j];
                if (q_ij < screen_tol) continue;

                const int bf_j_start = prim_starts[j];
                const int dim_j = prim_sizes[j];

                for (int R = 0; R < n_shells_aux; ++R) {
                    if (q_ij * aux_schwarz[R] < screen_tol) continue;

                    const int abs_R = n_shells_prim + R;
                    const int dim_R = aux_sizes[R];
                    const int aux_start = aux_starts[R];

                    if (!integrals_->compute_3c2e_block_raw(i, j, abs_R, buffer.data())) continue;

                    for (int r = 0; r < dim_R; ++r) {
                        const int P = aux_start + r;
                        double* v_col = V_raw.col(P).data();
                        for (int p_j = 0; p_j < dim_j; ++p_j) {
                            const int gj = bf_j_start + p_j;
                            const int buffer_offset = dim_i * (p_j + dim_j * r);
                            for (int p_i = 0; p_i < dim_i; ++p_i) {
                                const int gi = bf_i_start + p_i;
                                if (gi < gj) continue;
                                v_col[gi * (gi + 1) / 2 + gj] = buffer[p_i + buffer_offset];
                            }
                        }
                    }
                }
            }
        }
    }

    // 4. Kontraksi OpenMP-Tiled BLAS-3 DGEMM pada n_tri baris (2x lebih hemat RAM & FLOPs)
    B_mat_.resize(n_tri, n_aux_);
    constexpr int tile_b = 256;
    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < n_tri; r += tile_b) {
        int rows = std::min(tile_b, n_tri - r);
        B_mat_.middleRows(r, rows).noalias() = V_raw.middleRows(r, rows) * J_inv_sqrt;
    }

    if (save_hdf5) {
        utils::HDF5TensorIO io("df_tensor.h5", utils::HDF5TensorIO::Mode::WRITE_TRUNCATE);
        std::array<long, 4> dims = {(long)n_aux_, (long)n_primary_, (long)n_primary_, 1};
        std::array<long, 4> chunks = {1, (long)n_primary_, (long)n_primary_, 1};
        io.create_dataset_4d("df_tensor", dims, chunks);

        Eigen::MatrixXd slice_sq(n_primary_, n_primary_);
        const bool packed = is_packed_ao_matrix(B_mat_, n_primary_);
        for (int P = 0; P < n_aux_; ++P) {
            unpack_ao_column(B_mat_.col(P).data(), n_primary_, packed, slice_sq);
            std::array<long, 4> offset = {P, 0, 0, 0};
            io.write_slice_4d("df_tensor", offset, chunks, slice_sq.data());
        }
    }

    is_computed_ = true;
}

} // namespace integrals
} // namespace mshqc