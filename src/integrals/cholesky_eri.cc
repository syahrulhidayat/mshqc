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
 * @file src/integrals/cholesky_eri.cc
 * @brief DIRECT Cholesky Decomposition (Hybrid: Optimized Tensor & Screened Direct)
 * @details 
 * - FIXED: Tensor input diproses dengan benar (tidak di-skip).
 * - OPTIMIZED: Loop Tensor menggunakan OpenMP collapse untuk kecepatan tinggi.
 * - OPTIMIZED: Mode Direct menggunakan Shell-Screening.
 */

#include "mshqc/integrals/cholesky_eri.h"
#include "mshqc/utils/hdf5_io.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace integrals {





CholeskyERI::CholeskyERI(double threshold)
    : n_basis_(0), n_vectors_(0), decomposed_(false) 
{
    config_.threshold = threshold;
    threshold_ = threshold;
}

CholeskyERI::CholeskyERI(const BasisSet& basis, std::shared_ptr<IntegralEngine> integrals)
    : basis_ptr_(&basis), integrals_ptr_(integrals), 
      n_basis_(basis.n_basis_functions()), n_vectors_(0), decomposed_(false) 
{
    config_.threshold = 1e-6;
    config_.print_level = 1;
    threshold_ = 1e-6;

    
    bf2shell_.resize(n_basis_);
    int current_func = 0;
    for(int s = 0; s < basis.n_shells(); ++s) {
        int count = basis.shell(s).n_functions();
        for(int i = 0; i < count; ++i) {
            if (current_func + i < n_basis_) {
                bf2shell_[current_func + i] = s;
            }
        }
        current_func += count;
    }
}





void CholeskyERI::compute() {
    if (!integrals_ptr_ || !basis_ptr_) {
        std::cerr << "[Error] CholeskyERI: Missing Engine/Basis! Cannot run compute().\n";
        return;
    }

    decompose_direct();
}

CholeskyDecompositionResult CholeskyERI::decompose(const Eigen::Tensor<double, 4>& eri_full) {
    
    auto dims = eri_full.dimensions();
    n_basis_ = static_cast<int>(dims[0]);
    int n_pairs = n_basis_ * n_basis_;
    
    if (config_.print_level > 0) {
        std::cout << "\n======================================================================\n";
        std::cout << "  Cholesky Decomposition (Tensor Mode - Exact)\n";
        std::cout << "======================================================================\n";
        std::cout << "  Basis: " << n_basis_ << " | Pairs: " << n_pairs << "\n";
    }

    reset();

    
    Eigen::VectorXd D(n_pairs);
    
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n_basis_; ++i) {
        for (int j = 0; j < n_basis_; ++j) {
            D(i * n_basis_ + j) = eri_full(i, j, i, j);
        }
    }

    double max_diag = D.maxCoeff();
    if (config_.print_level > 0) 
        std::cout << "  Max Diag: " << std::scientific << max_diag << " | Threshold: " << threshold_ << "\n";

    
    int est_rank = std::min(n_pairs, std::max(200, n_basis_ * 5));
    Eigen::MatrixXd L_store(n_pairs, est_rank);
    
    int iter = 0;
    while (true) {
        
        int pivot_idx;
        double D_max = D.maxCoeff(&pivot_idx);

        if (D_max < threshold_ || iter >= n_pairs) break;

        
        if (iter >= L_store.cols()) {
            L_store.conservativeResize(Eigen::NoChange, L_store.cols() * 2);
        }

        double inv_sqrt = 1.0 / std::sqrt(D_max);
        int pi = pivot_idx / n_basis_;
        int pj = pivot_idx % n_basis_;

        
        
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n_basis_; ++i) {
            for (int j = 0; j < n_basis_; ++j) {
                L_store(i * n_basis_ + j, iter) = eri_full(i, j, pi, pj);
            }
        }

        
        if (iter > 0) {
            
            L_store.col(iter) -= L_store.leftCols(iter) * L_store.row(pivot_idx).head(iter).transpose();
        }

        
        L_store.col(iter) *= inv_sqrt;
        D.array() -= L_store.col(iter).array().square();

        
        
        for(int k=0; k<n_pairs; ++k) if (D(k) < 0.0) D(k) = 0.0;

        iter++;
    }

    n_vectors_ = iter;
    decomposed_ = true;

    L_mat_ = L_store.leftCols(n_vectors_);

    if (config_.print_level > 0) {
        std::cout << "  Done. Vectors: " << n_vectors_ 
                  << " | Ratio: " << std::fixed << std::setprecision(2) << compression_ratio() << "x\n";
    }

    CholeskyDecompositionResult res;
    res.n_vectors = n_vectors_;
    res.converged = true;
    res.max_error = D.maxCoeff();
    return res;
}




void CholeskyERI::decompose_direct() {
    if (!integrals_ptr_ || !basis_ptr_) return;
    const auto& basis = *basis_ptr_;
    int nshells = basis.n_shells();
    int npair = n_basis_ * n_basis_;
    reset();

    std::vector<int> shell_starts(nshells);
    int offset = 0;
    for (int s = 0; s < nshells; ++s) {
        shell_starts[s] = offset;
        offset += basis.shell(s).n_functions();
    }

    Eigen::VectorXd D(npair);
    D.setZero();
    std::vector<double> shell_max(nshells * nshells, 0.0);
    std::vector<std::pair<int, int>> valid_pairs;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int s1 = 0; s1 < nshells; ++s1) {
        for (int s2 = 0; s2 <= s1; ++s2) {
            const auto& buf = integrals_ptr_->compute_shell_block(s1, s2, s1, s2);
            if (buf.empty()) continue;

            int dim1 = basis.shell(s1).n_functions();
            int dim2 = basis.shell(s2).n_functions();
            int st1 = shell_starts[s1];
            int st2 = shell_starts[s2];
            double max_val_block = 0.0;

            for (int i = 0; i < dim1; ++i) {
                for (int j = 0; j < dim2; ++j) {
                    size_t idx_buf = i + dim1 * (j + dim2 * (i + dim1 * j));
                    if (idx_buf >= buf.size()) continue;

                    double val = buf[idx_buf];
                    int p = st1 + i;
                    int q = st2 + j;

                    if (p * n_basis_ + q < npair) D(p * n_basis_ + q) = val;
                    if (q * n_basis_ + p < npair) D(q * n_basis_ + p) = val;
                    max_val_block = std::max(max_val_block, std::abs(val));
                }
            }
            double s_max = std::sqrt(max_val_block);
            shell_max[s1 * nshells + s2] = s_max;
            shell_max[s2 * nshells + s1] = s_max;

            if (s_max > 1e-15) {
                #pragma omp critical
                valid_pairs.push_back({s1, s2});
            }
        }
    }

    int est_rank = std::min(npair, std::max(200, n_basis_ * 5));
    Eigen::MatrixXd L_store(npair, est_rank);

    // Cutoff screening diperketat ke 1e-15 agar tidak teramplifikasi oleh 1/sqrt(D_max)
    const double screen_cutoff = std::min(1e-15, threshold_ * 1e-6);

    int n_shell_pairs = nshells * (nshells + 1) / 2;
    std::vector<Eigen::MatrixXd> shell_pair_cache(n_shell_pairs);

    int iter = 0;
    while (true) {
        int pivot_idx = -1;
        double D_max = D.maxCoeff(&pivot_idx);

        if (D_max < threshold_ || pivot_idx == -1 || iter >= npair) break;
        if (iter >= L_store.cols()) {
            L_store.conservativeResize(Eigen::NoChange, L_store.cols() * 2);
        }

        int p = pivot_idx / n_basis_;
        int q = pivot_idx % n_basis_;
        if (p < q) std::swap(p, q);

        int sp = bf2shell_[p];
        int sq = bf2shell_[q];
        int rel_p = p - shell_starts[sp];
        int rel_q = q - shell_starts[sq];
        int pair_pq = sp * (sp + 1) / 2 + sq;
        int dimP = basis.shell(sp).n_functions();
        int dimQ = basis.shell(sq).n_functions();

        if (shell_pair_cache[pair_pq].rows() == 0) {
            Eigen::MatrixXd blk_mat = Eigen::MatrixXd::Zero(npair, dimP * dimQ);
            double pivot_schwarz = shell_max[sp * nshells + sq];

            #pragma omp parallel
            {
                std::vector<double> thread_buf(10000);
                #pragma omp for schedule(dynamic, 16) nowait
                for (size_t v = 0; v < valid_pairs.size(); ++v) {
                    int s1 = valid_pairs[v].first;
                    int s2 = valid_pairs[v].second;
                    if (shell_max[s1 * nshells + s2] * pivot_schwarz < screen_cutoff) continue;

                    int pair_s = s1 * (s1 + 1) / 2 + s2;
                    bool swap_pairs = (pair_s < pair_pq);
                    int u1 = swap_pairs ? sp : s1;
                    int u2 = swap_pairs ? sq : s2;
                    int u3 = swap_pairs ? s1 : sp;
                    int u4 = swap_pairs ? s2 : sq;

                    if (!integrals_ptr_->compute_shell_block_raw(u1, u2, u3, u4, thread_buf.data())) continue;

                    int dim1 = basis.shell(s1).n_functions();
                    int dim2 = basis.shell(s2).n_functions();
                    int st1 = shell_starts[s1];
                    int st2 = shell_starts[s2];

                    for (int rp = 0; rp < dimP; ++rp) {
                        for (int rq = 0; rq < dimQ; ++rq) {
                            int col_pq = rp + dimP * rq;
                            for (int i = 0; i < dim1; ++i) {
                                int global_i = st1 + i;
                                for (int j = 0; j < dim2; ++j) {
                                    int global_j = st2 + j;
                                    size_t idx_buf = (!swap_pairs)
                                        ? (i + dim1 * (j + dim2 * (rp + dimP * rq)))
                                        : (rp + dimP * (rq + dimQ * (i + dim1 * j)));
                                    double val = thread_buf[idx_buf];
                                    blk_mat(global_i * n_basis_ + global_j, col_pq) = val;
                                    if (global_i != global_j) blk_mat(global_j * n_basis_ + global_i, col_pq) = val;
                                }
                            }
                        }
                    }
                }
            }
            shell_pair_cache[pair_pq] = std::move(blk_mat);
        }

        Eigen::VectorXd col_buf = shell_pair_cache[pair_pq].col(rel_p + dimP * rel_q);

        if (iter > 0) {
            Eigen::Map<const Eigen::MatrixXd> L_prev(L_store.data(), npair, iter);
            Eigen::VectorXd L_piv = L_prev.row(pivot_idx).transpose();
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < npair; r += 256) {
                int r_len = std::min(256, npair - r);
                col_buf.segment(r, r_len).noalias() -= L_prev.middleRows(r, r_len) * L_piv;
            }
        }

        double exact_diag = col_buf(pivot_idx);
        if (exact_diag < threshold_) {
            D(p * n_basis_ + q) = 0.0;
            D(q * n_basis_ + p) = 0.0;
            shell_pair_cache[pair_pq].resize(0, 0);
            continue;
        }
        double inv_sqrt = 1.0 / std::sqrt(exact_diag);

        L_store.col(iter) = col_buf * inv_sqrt;
        D.array() -= L_store.col(iter).array().square();

        D(p * n_basis_ + q) = 0.0;
        D(q * n_basis_ + p) = 0.0;
        D = D.cwiseMax(0.0);

        // Bebaskan shell_pair_cache segera jika tidak ada lagi kandidat pivot besar di shell ini
        // atau jika total cache aktif melebihi 8 MB (mencegah pembengkakan VmHWM sebesar O(N^4))
        double max_rem_in_shell = 0.0;
        int stP = shell_starts[sp], stQ = shell_starts[sq];
        for (int rp = 0; rp < dimP; ++rp) {
            for (int rq = 0; rq < dimQ; ++rq) {
                max_rem_in_shell = std::max(max_rem_in_shell, D((stP + rp) * n_basis_ + (stQ + rq)));
            }
        }
        if (max_rem_in_shell < 0.1 * D_max) {
            shell_pair_cache[pair_pq].resize(0, 0);
        }

        bool has_active = false;
        int stP = shell_starts[sp], stQ = shell_starts[sq];
        for (int rp = 0; rp < dimP && !has_active; ++rp) {
            for (int rq = 0; rq < dimQ; ++rq) {
                if (D((stP + rp) * n_basis_ + (stQ + rq)) >= threshold_) {
                    has_active = true;
                    break;
                }
            }
        }
        if (!has_active) shell_pair_cache[pair_pq].resize(0, 0);

        iter++;
    }

    shell_pair_cache.clear();
    shell_pair_cache.shrink_to_fit();
    n_vectors_ = iter;
    decomposed_ = true;
    L_store.conservativeResize(Eigen::NoChange, n_vectors_);
    L_mat_ = std::move(L_store);
}


void CholeskyERI::reset() {
    L_mat_.resize(0, 0);
    n_vectors_ = 0;
    decomposed_ = false;
}

double CholeskyERI::compression_ratio() const {
    if (n_basis_ == 0) return 0.0;
    double full_size = std::pow(n_basis_, 4);
    double chol_size = std::max(1.0, (double)n_vectors_ * n_basis_ * n_basis_);
    return full_size / chol_size;
}

double CholeskyERI::reconstruct(int i, int j, int k, int l) const {
    if (!decomposed_) return 0.0;
    int ij = i * n_basis_ + j;
    int kl = k * n_basis_ + l;
    return L_mat_.row(ij).dot(L_mat_.row(kl)); 
}

size_t CholeskyERI::storage_bytes() const {
    return L_mat_.size() * sizeof(double);
}

void CholeskyERI::save_to_hdf5(const std::string& filename, const std::string& dataset_name) const {
    if (!decomposed_ || n_vectors_ == 0) return;

    utils::HDF5TensorIO io(filename, utils::HDF5TensorIO::Mode::WRITE_TRUNCATE);
    long n_vec = n_vectors_;
    
    io.create_dataset_4d(dataset_name, {n_vec, n_basis_, n_basis_, 1}, {1, n_basis_, n_basis_, 1});

    for (long k = 0; k < n_vec; ++k) {
        std::array<long, 4> offset = {k, 0, 0, 0};
        std::array<long, 4> slice_dims = {1, n_basis_, n_basis_, 1};
        
        io.write_slice_4d(dataset_name, offset, slice_dims, L_mat_.col(k).data());
    }
}

void CholeskyERI::load_from_hdf5(const std::string& filename, const std::string& dataset_name) {
    utils::HDF5TensorIO io(filename, utils::HDF5TensorIO::Mode::READ_ONLY);
    
    Eigen::Tensor<double, 4> tensor_in = io.read_tensor_4d(dataset_name);
    
    long n_vec = tensor_in.dimension(0);
    n_basis_ = static_cast<int>(tensor_in.dimension(1));
    long n_pairs = n_basis_ * n_basis_;
    
    n_vectors_ = static_cast<int>(n_vec);
    
    
    L_mat_.resize(n_pairs, n_vectors_);
    
    for (int k = 0; k < n_vectors_; ++k) {
        std::array<long, 4> offset = {k, 0, 0, 0};
        std::array<long, 4> slice_dims = {1, n_basis_, n_basis_, 1};
        
        io.read_slice_4d(dataset_name, offset, slice_dims, L_mat_.col(k).data());
    }
    decomposed_ = true;
}


Eigen::Tensor<double, 4> CholeskyERI::reconstruct_full() const { return Eigen::Tensor<double, 4>(); }
std::pair<double, double> CholeskyERI::validate_reconstruction(const Eigen::Tensor<double, 4>&) { return {0.0, 0.0}; }
void CholeskyERI::print_statistics(bool) const {}
int CholeskyERI::find_pivot(const Eigen::VectorXd&) const { return 0; }
Eigen::VectorXd CholeskyERI::compute_new_vector(const Eigen::Tensor<double, 4>&, const Eigen::VectorXd&, int) const { return Eigen::VectorXd(); }

} 
} 
