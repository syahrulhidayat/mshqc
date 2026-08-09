 // ==============================================================================
 // Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
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
 // ==============================================================================

#include "mshqc/integrals/df_eri.h"
#include "mshqc/utils/hdf5_io.h"
#include <iostream>
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

        for (int j = i + 1; j < n; ++j) {
            double sum = 0.0;

            for (int k = 0; k < i; ++k) {
                sum += L(j, k) * L(i, k);
            }
            L(j, i) = (J(j, i) - sum) * L_ii_inv;
            diag(j) -= L(j, i) * L(j, i);
        }
        rank++;
    }

    L.conservativeResize(n, rank);
    return {L, p, rank};
}
void DensityFittingERI::compute() {
    std::cout << "  Starting Density Fitting (RI) Decomposition (Ultra HPC)...\n";
    std::cout << "  Primary Basis: " << n_primary_ << " BF\n";
    std::cout << "  Auxiliary Basis: " << n_aux_ << " BF\n";

    int n_shells_prim = primary_basis_->n_shells();
    int n_shells_aux  = aux_basis_->n_shells();

    std::vector<int> aux_starts(n_shells_aux), aux_sizes(n_shells_aux);
    int offset_a = 0;
    for(int i = 0; i < n_shells_aux; ++i) {
        aux_starts[i] = offset_a;
        aux_sizes[i] = aux_basis_->shell(i).n_functions();
        offset_a += aux_sizes[i];
    }

    std::vector<int> prim_starts(n_shells_prim), prim_sizes(n_shells_prim);
    int offset_p = 0;
    for(int i = 0; i < n_shells_prim; ++i) {
        prim_starts[i] = offset_p;
        prim_sizes[i] = primary_basis_->shell(i).n_functions();
        offset_p += prim_sizes[i];
    }

    std::cout << "  Evaluating 2-Center Metric Matrix...\n";
    Eigen::MatrixXd J_metric = Eigen::MatrixXd::Zero(n_aux_, n_aux_);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int R = 0; R < n_shells_aux; ++R) {
        int abs_R = n_shells_prim + R;
        int dim_R = aux_sizes[R];
        int bf_R_start = aux_starts[R];

        for (int S = 0; S <= R; ++S) {
            int abs_S = n_shells_prim + S;
            int dim_S = aux_sizes[S];
            int bf_S_start = aux_starts[S];

            auto buffer = integrals_->compute_2c2e_block(abs_R, abs_S);
            if (buffer.empty()) continue;

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

    CholeskyResult chol = compute_pivoted_cholesky(J_metric, 1e-10);
    int n_aux_rank = chol.rank;
    std::cout << "  Pivoted Cholesky Rank: " << n_aux_rank << " / " << n_aux_ << " (Removed Linear Dependencies)\n";

    auto L_tri = chol.L.topLeftCorner(n_aux_rank, n_aux_rank).triangularView<Eigen::Lower>();

    std::vector<int> inv_pivot(n_aux_);
    for (int p = 0; p < n_aux_; ++p) {
        inv_pivot[chol.pivot_map[p]] = p;
    }

    int n_pairs = n_primary_ * n_primary_;
    B_mat_ = Eigen::MatrixXd::Zero(n_pairs, n_aux_);

    std::cout << "  Evaluating 3-Center Integrals and Pivoting...\n";

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < n_shells_prim; ++i) {
        int bf_i_start = prim_starts[i];
        int dim_i = prim_sizes[i];

        for (int j = 0; j <= i; ++j) {
            int bf_j_start = prim_starts[j];
            int dim_j = prim_sizes[j];

            for (int R = 0; R < n_shells_aux; ++R) {
                int abs_R = n_shells_prim + R;
                int dim_R = aux_sizes[R];
                int aux_start = aux_starts[R];

                auto buffer = integrals_->compute_3c2e_block(i, j, abs_R);
                if (buffer.empty()) continue;

                for (int r = 0; r < dim_R; ++r) {
                    int orig_P = aux_start + r;
                    int new_P  = inv_pivot[orig_P];

                    for (int p_j = 0; p_j < dim_j; ++p_j) {
                        int base_r1 = bf_j_start + p_j;
                        int base_r2 = (bf_j_start + p_j) * n_primary_;
                        int buffer_offset = dim_i * (p_j + dim_j * r);

                        for (int p_i = 0; p_i < dim_i; ++p_i) {
                            double val = buffer[p_i + buffer_offset];
                            int r1 = (bf_i_start + p_i) * n_primary_ + base_r1;
                            int r2 = base_r2 + (bf_i_start + p_i);

                            B_mat_(r1, new_P) = val;
                            if (r1 != r2) B_mat_(r2, new_P) = val;
                        }
                    }
                }
            }
        }
    }

    std::cout << "  In-Place Triangular Solve (cblas_dtrsm)...\n";
    auto V_rank = B_mat_.leftCols(n_aux_rank);
    L_tri.solveInPlace(V_rank.transpose());

    std::cout << "  Writing Density Fitting Tensor to HDF5 (Out-of-Core)...\n";
    utils::HDF5TensorIO io("df_tensor.h5", utils::HDF5TensorIO::Mode::WRITE_TRUNCATE);

    std::array<long, 4> dims = {(long)n_aux_rank, (long)n_primary_, (long)n_primary_, 1};
    std::array<long, 4> chunks = {1, (long)n_primary_, (long)n_primary_, 1};
    io.create_dataset_4d("df_tensor", dims, chunks);

    for (int P = 0; P < n_aux_rank; ++P) {
        std::array<long, 4> offset = {P, 0, 0, 0};
        io.write_slice_4d("df_tensor", offset, chunks, V_rank.col(P).data());
    }

    B_mat_ = V_rank;
    is_computed_ = true;
    std::cout << "  Density Fitting Decomposition Complete.\n";
}

}
}
