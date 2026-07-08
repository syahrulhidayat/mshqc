/**
 * @file src/integrals/df_eri.cc
 * @brief Implementasi Density Fitting Tensor Generator
 */

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

Eigen::MatrixXd DensityFittingERI::compute_J_inv_half() {
    Eigen::MatrixXd J_mat = Eigen::MatrixXd::Zero(n_aux_, n_aux_);
    int n_shells_aux = aux_basis_->n_shells();

    
    std::vector<int> aux_starts(n_shells_aux), aux_sizes(n_shells_aux);
    int offset = 0;
    for(int i = 0; i < n_shells_aux; ++i) {
        aux_starts[i] = offset;
        aux_sizes[i] = aux_basis_->shell(i).n_functions();
        offset += aux_sizes[i];
    }

    
    #pragma omp parallel for schedule(dynamic)
    for (int P = 0; P < n_shells_aux; ++P) {
        int abs_P = primary_basis_->n_shells() + P; 
        int bf_P_start = aux_starts[P];
        int dim_P = aux_sizes[P];

        for (int Q = 0; Q <= P; ++Q) {
            int abs_Q = primary_basis_->n_shells() + Q;
            int bf_Q_start = aux_starts[Q];
            int dim_Q = aux_sizes[Q];

            auto buffer = integrals_->compute_2c2e_block(abs_P, abs_Q);
            if (buffer.empty()) continue;

            for (int p = 0; p < dim_P; ++p) {
                for (int q = 0; q < dim_Q; ++q) {
                    size_t idx = p + dim_P * q;
                    double val = buffer[idx];
                    
                   
                    J_mat(bf_P_start + p, bf_Q_start + q) = val;
                    J_mat(bf_Q_start + q, bf_P_start + p) = val; 
                }
            }
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(J_mat);
    
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("FATAL: Eigen gagal mendiagonalisasi J_mat. Matriks berisi nilai ilegal.");
    }
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    Eigen::MatrixXd eigenvectors = solver.eigenvectors();

    double cutoff = cutoff_;
    Eigen::MatrixXd J_inv_half = Eigen::MatrixXd::Zero(n_aux_, n_aux_);
    
    for (int i = 0; i < n_aux_; ++i) {
        if (eigenvalues(i) > cutoff) {
            J_inv_half.col(i) = eigenvectors.col(i) * (1.0 / std::sqrt(eigenvalues(i)));
        }
    }

    return J_inv_half * eigenvectors.transpose();
}

void DensityFittingERI::compute() {
    std::cout << "  Starting Density Fitting (RI) Decomposition...\n";
    std::cout << "  Primary Basis: " << n_primary_ << " BF\n";
    std::cout << "  Auxiliary Basis: " << n_aux_ << " BF\n";

    Eigen::MatrixXd J_inv_half = compute_J_inv_half();

    int n_pairs = n_primary_ * n_primary_;
    B_mat_ = Eigen::MatrixXd::Zero(n_pairs, n_aux_);

    int n_shells_prim = primary_basis_->n_shells();
    int n_shells_aux  = aux_basis_->n_shells();

    
    std::vector<int> prim_starts(n_shells_prim), prim_sizes(n_shells_prim);
    int offset_p = 0;
    for(int i = 0; i < n_shells_prim; ++i) {
        prim_starts[i] = offset_p;
        prim_sizes[i] = primary_basis_->shell(i).n_functions();
        offset_p += prim_sizes[i];
    }

    
    std::vector<int> aux_starts(n_shells_aux), aux_sizes(n_shells_aux);
    int offset_a = 0;
    for(int i = 0; i < n_shells_aux; ++i) {
        aux_starts[i] = offset_a;
        aux_sizes[i] = aux_basis_->shell(i).n_functions();
        offset_a += aux_sizes[i];
    }

    
    
    
    int batch_size = 32; 
    for (int R_start = 0; R_start < n_shells_aux; R_start += batch_size) {
        int R_end = std::min(n_shells_aux, R_start + batch_size);
        int aux_bf_start = aux_starts[R_start];
        int aux_bf_end = (R_end < n_shells_aux) ? aux_starts[R_end] : n_aux_;
        int aux_bf_size = aux_bf_end - aux_bf_start;

        
        Eigen::MatrixXd V_batch = Eigen::MatrixXd::Zero(n_pairs, aux_bf_size);

        #pragma omp parallel for schedule(dynamic, 1)
        for (int R = R_start; R < R_end; ++R) {
            int abs_R = n_shells_prim + R; 
            int dim_R = aux_sizes[R];
            int col_offset = aux_starts[R] - aux_bf_start;

            for (int i = 0; i < n_shells_prim; ++i) {
                int bf_i_start = prim_starts[i];
                int dim_i = prim_sizes[i];

                for (int j = 0; j <= i; ++j) {
                    int bf_j_start = prim_starts[j];
                    int dim_j = prim_sizes[j];

                    auto buffer = integrals_->compute_3c2e_block(i, j, abs_R);
                    if (buffer.empty()) continue;

                    for (int r = 0; r < dim_R; ++r) {
                        int col_idx = col_offset + r; 
                        for (int p_j = 0; p_j < dim_j; ++p_j) {
                            int base_r1 = bf_j_start + p_j;
                            int base_r2 = (bf_j_start + p_j) * n_primary_;
                            int buffer_offset = dim_i * (p_j + dim_j * r);
                            
                            for (int p_i = 0; p_i < dim_i; ++p_i) {
                                double val = buffer[p_i + buffer_offset];
                                int r1 = (bf_i_start + p_i) * n_primary_ + base_r1;
                                int r2 = base_r2 + (bf_i_start + p_i);
                                
                                V_batch(r1, col_idx) = val;
                                if (r1 != r2) V_batch(r2, col_idx) = val;
                            }
                        }
                    }
                }
            }
        }
        std::cout << "  Computing V * J^{-1/2} for Shells " << R_start << " to " << R_end - 1 << " ...\n";
        B_mat_ += V_batch * J_inv_half.middleRows(aux_bf_start, aux_bf_size);
    }

    
    
    
    std::cout << "  Writing Density Fitting Tensor to HDF5 (Out-of-Core)...\n";
    utils::HDF5TensorIO io("df_tensor.h5", utils::HDF5TensorIO::Mode::WRITE_TRUNCATE);
    std::array<long, 4> dims = {(long)n_aux_, (long)n_primary_, (long)n_primary_, 1};
    std::array<long, 4> chunks = {1, (long)n_primary_, (long)n_primary_, 1};
    io.create_dataset_4d("df_tensor", dims, chunks);

    for (int P = 0; P < n_aux_; ++P) {
        std::array<long, 4> offset = {P, 0, 0, 0};
        io.write_slice_4d("df_tensor", offset, chunks, B_mat_.col(P).data());
    }

    
    B_mat_.resize(0, 0); 
    is_computed_ = true;
    std::cout << "  Density Fitting Decomposition Complete.\n";
}

} 
} 
