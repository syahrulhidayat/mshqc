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
#include "mshqc/integrals/packed_ao_utils.h"
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
    const int nshells = basis.n_shells();
    const int n_tri = n_basis_ * (n_basis_ + 1) / 2;
    reset();

    std::vector<int> shell_starts(nshells);
    int max_shell_dim = 1;
    int offset = 0;
    for (int s = 0; s < nshells; ++s) {
        shell_starts[s] = offset;
        int dim_s = basis.shell(s).n_functions();
        max_shell_dim = std::max(max_shell_dim, dim_s);
        offset += dim_s;
    }
    const size_t max_buf_sz = std::max<size_t>(
        64, static_cast<size_t>(max_shell_dim) * max_shell_dim * max_shell_dim * max_shell_dim
    );

    auto tri_index = [](int a, int b) -> int {
        if (a < b) std::swap(a, b);
        return a * (a + 1) / 2 + b;
    };

    std::vector<int> tri_p(n_tri), tri_q(n_tri);
    for (int p = 0; p < n_basis_; ++p) {
        const int base = p * (p + 1) / 2;
        for (int q = 0; q <= p; ++q) {
            tri_p[base + q] = p;
            tri_q[base + q] = q;
        }
    }

    Eigen::VectorXd D = Eigen::VectorXd::Zero(n_tri);
    std::vector<double> shell_max(static_cast<size_t>(nshells) * nshells, 0.0);
    std::vector<std::pair<int, int>> valid_pairs;

    #pragma omp parallel
    {
        std::vector<double> buf(max_buf_sz);
        #pragma omp for schedule(dynamic, 1) nowait
        for (int s1 = 0; s1 < nshells; ++s1) {
            const int dim1 = basis.shell(s1).n_functions();
            const int st1  = shell_starts[s1];
            for (int s2 = 0; s2 <= s1; ++s2) {
                if (!integrals_ptr_->compute_shell_block_raw(s1, s2, s1, s2, buf.data())) continue;
                const int dim2 = basis.shell(s2).n_functions();
                const int st2  = shell_starts[s2];
                double max_val_block = 0.0;

                for (int i = 0; i < dim1; ++i) {
                    const int p = st1 + i;
                    for (int j = 0; j < dim2; ++j) {
                        const int q = st2 + j;
                        const size_t idx_buf = static_cast<size_t>(i) + dim1 * (j + dim2 * (i + dim1 * j));
                        const double val = buf[idx_buf];
                        if (p >= q) D(p * (p + 1) / 2 + q) = val;
                        max_val_block = std::max(max_val_block, std::abs(val));
                    }
                }
                const double s_max = std::sqrt(max_val_block);
                shell_max[s1 * nshells + s2] = s_max;
                shell_max[s2 * nshells + s1] = s_max;

                if (s_max > 1e-15) {
                    #pragma omp critical
                    valid_pairs.push_back({s1, s2});
                }
            }
        }
    }

    int est_rank = std::min(n_tri, std::max(200, n_basis_ * 5));
    Eigen::MatrixXd L_tri(n_tri, est_rank);

    const double screen_cutoff = std::min(1e-15, threshold_ * 1e-6);
    const int n_shell_pairs = nshells * (nshells + 1) / 2;
    std::vector<Eigen::MatrixXd> shell_pair_cache(n_shell_pairs);

    constexpr int kMaxBlock = 32;
    constexpr double kSpanFactor = 0.05;

    int iter = 0;
    std::vector<int> cand_idx;
    cand_idx.reserve(n_tri);

    while (iter < n_tri) {
        int max_piv = -1;
        double D_max = D.maxCoeff(&max_piv);
        if (D_max < threshold_ || max_piv < 0) break;

        // 1. Pilih batch hingga kMaxBlock poros teratas dalam span [max(threshold, 0.05*D_max), D_max]
        const double span_cut = std::max(threshold_, kSpanFactor * D_max);
        cand_idx.clear();
        for (int idx = 0; idx < n_tri; ++idx) {
            if (D(idx) >= span_cut) cand_idx.push_back(idx);
        }
        const int q_cnt = std::min(kMaxBlock, static_cast<int>(cand_idx.size()));
        if (q_cnt <= 0) break;

        std::partial_sort(cand_idx.begin(), cand_idx.begin() + q_cnt, cand_idx.end(),
                          [&D](int a, int b) { return D(a) > D(b); });
        cand_idx.resize(q_cnt);

        if (iter + q_cnt > L_tri.cols()) {
            L_tri.conservativeResize(Eigen::NoChange, std::max(iter + q_cnt, static_cast<int>(L_tri.cols()) * 2));
        }

        // 2. Evaluasi kolom integral mentah untuk seluruh poros di batch (hanya simpan n_tri baris)
        Eigen::MatrixXd V_panel(n_tri, q_cnt);
        std::vector<int> touched_shell_pairs;
        touched_shell_pairs.reserve(q_cnt);

        for (int c = 0; c < q_cnt; ++c) {
            const int piv = cand_idx[c];
            const int p = tri_p[piv];
            const int q = tri_q[piv];
            const int sp = bf2shell_[p];
            const int sq = bf2shell_[q];
            const int rel_p = p - shell_starts[sp];
            const int rel_q = q - shell_starts[sq];
            const int pair_pq = sp * (sp + 1) / 2 + sq;
            const int dimP = basis.shell(sp).n_functions();
            const int dimQ = basis.shell(sq).n_functions();

            touched_shell_pairs.push_back(pair_pq);

            if (shell_pair_cache[pair_pq].rows() == 0) {
                Eigen::MatrixXd blk_mat = Eigen::MatrixXd::Zero(n_tri, dimP * dimQ);
                const double pivot_schwarz = shell_max[sp * nshells + sq];

                #pragma omp parallel
                {
                    std::vector<double> thread_buf(max_buf_sz);
                    #pragma omp for schedule(dynamic, 16) nowait
                    for (size_t v = 0; v < valid_pairs.size(); ++v) {
                        const int s1 = valid_pairs[v].first;
                        const int s2 = valid_pairs[v].second;
                        if (shell_max[s1 * nshells + s2] * pivot_schwarz < screen_cutoff) continue;

                        const int pair_s = s1 * (s1 + 1) / 2 + s2;
                        const bool swap_pairs = (pair_s < pair_pq);
                        const int u1 = swap_pairs ? sp : s1;
                        const int u2 = swap_pairs ? sq : s2;
                        const int u3 = swap_pairs ? s1 : sp;
                        const int u4 = swap_pairs ? s2 : sq;

                        if (!integrals_ptr_->compute_shell_block_raw(u1, u2, u3, u4, thread_buf.data())) continue;

                        const int dim1 = basis.shell(s1).n_functions();
                        const int dim2 = basis.shell(s2).n_functions();
                        const int st1  = shell_starts[s1];
                        const int st2  = shell_starts[s2];

                        for (int rp = 0; rp < dimP; ++rp) {
                            for (int rq = 0; rq < dimQ; ++rq) {
                                const int col_pq = rp + dimP * rq;
                                double* dst_col = blk_mat.col(col_pq).data();
                                for (int i = 0; i < dim1; ++i) {
                                    const int gi = st1 + i;
                                    for (int j = 0; j < dim2; ++j) {
                                        const int gj = st2 + j;
                                        if (gi < gj) continue;
                                        const size_t idx_buf = (!swap_pairs)
                                            ? (static_cast<size_t>(i) + dim1 * (j + dim2 * (rp + dimP * rq)))
                                            : (static_cast<size_t>(rp) + dimP * (rq + dimQ * (i + dim1 * j)));
                                        dst_col[gi * (gi + 1) / 2 + gj] = thread_buf[idx_buf];
                                    }
                                }
                            }
                        }
                    }
                }
                shell_pair_cache[pair_pq] = std::move(blk_mat);
            }
            V_panel.col(c) = shell_pair_cache[pair_pq].col(rel_p + dimP * rel_q);
        }

        // 3. Proyeksi BLAS-3 DGEMM (1 pass atas L_tri untuk seluruh panel q_cnt kolom)
        if (iter > 0) {
            Eigen::MatrixXd L_piv_T(iter, q_cnt);
            for (int c = 0; c < q_cnt; ++c) {
                L_piv_T.col(c) = L_tri.row(cand_idx[c]).head(iter).transpose();
            }
            Eigen::Map<const Eigen::MatrixXd> L_prev(L_tri.data(), n_tri, iter);
            constexpr int row_tile = 256;
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < n_tri; r += row_tile) {
                const int r_len = std::min(row_tile, n_tri - r);
                V_panel.middleRows(r, r_len).noalias() -= L_prev.middleRows(r, r_len) * L_piv_T;
            }
        }

        // 4. Pivoted Cholesky di dalam panel (<= 32 kolom)
        std::vector<char> used(q_cnt, 0);
        int accepted_in_block = 0;

        for (int step = 0; step < q_cnt; ++step) {
            int best_c = -1;
            double best_d = -1.0;
            for (int c = 0; c < q_cnt; ++c) {
                if (used[c]) continue;
                const double d_val = V_panel(cand_idx[c], c);
                if (d_val > best_d) {
                    best_d = d_val;
                    best_c = c;
                }
            }
            if (best_c < 0) break;

            // Berhenti jika diagonal terbaik di panel turun di bawah span_cut (kecuali langkah pertama)
            if (best_d < threshold_) {
                if (accepted_in_block == 0) {
                    D(cand_idx[best_c]) = 0.0;
                }
                break;
            }
            if (accepted_in_block > 0 && best_d < span_cut) break;

            used[best_c] = 1;
            const int piv = cand_idx[best_c];
            const double inv_sqrt = 1.0 / std::sqrt(best_d);

            L_tri.col(iter).noalias() = V_panel.col(best_c) * inv_sqrt;
            D.array() -= L_tri.col(iter).array().square();
            D(piv) = 0.0;
            D = D.cwiseMax(0.0);

            // Perbarui kolom sisa di panel secara eksak: V(:, c) -= L_new * L_new(piv_c)
            #pragma omp parallel for schedule(static)
            for (int c = 0; c < q_cnt; ++c) {
                if (used[c]) continue;
                const double proj = L_tri(cand_idx[c], iter);
                V_panel.col(c).noalias() -= proj * L_tri.col(iter);
            }

            iter++;
            accepted_in_block++;
        }

        // 5. Eviksi cache blok shell yang sudah selesai
        std::sort(touched_shell_pairs.begin(), touched_shell_pairs.end());
        touched_shell_pairs.erase(std::unique(touched_shell_pairs.begin(), touched_shell_pairs.end()),
                                  touched_shell_pairs.end());
        const double evict_cut = std::max(threshold_, 0.1 * D_max);
        for (int pair_pq : touched_shell_pairs) {
            if (shell_pair_cache[pair_pq].rows() == 0) continue;
            // Cari sp, sq dari indeks segitiga pair_pq
            int sp = static_cast<int>((std::sqrt(8.0 * pair_pq + 1.0) - 1.0) * 0.5);
            while ((sp + 1) * (sp + 2) / 2 <= pair_pq) ++sp;
            while (sp * (sp + 1) / 2 > pair_pq) --sp;
            const int sq = pair_pq - sp * (sp + 1) / 2;

            double max_rem = 0.0;
            const int stP = shell_starts[sp], dimP = basis.shell(sp).n_functions();
            const int stQ = shell_starts[sq], dimQ = basis.shell(sq).n_functions();
            for (int rp = 0; rp < dimP; ++rp) {
                for (int rq = 0; rq < dimQ; ++rq) {
                    max_rem = std::max(max_rem, D(tri_index(stP + rp, stQ + rq)));
                }
            }
            if (max_rem < evict_cut) {
                shell_pair_cache[pair_pq].resize(0, 0);
            }
        }
    }

    shell_pair_cache.clear();
    shell_pair_cache.shrink_to_fit();
    n_vectors_ = iter;
    decomposed_ = true;

    // [INJECTED_PRIORITAS_2D_PACKED_AO]
    L_tri.conservativeResize(Eigen::NoChange, n_vectors_);
    L_mat_ = std::move(L_tri);
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
    const bool packed = is_packed_ao_matrix(L_mat_, n_basis_);
    int ij = packed ? ao_tri_index(i, j) : (i * n_basis_ + j);
    int kl = packed ? ao_tri_index(k, l) : (k * n_basis_ + l);
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

    const bool packed = is_packed_ao_matrix(L_mat_, n_basis_);
    Eigen::MatrixXd slice_sq(n_basis_, n_basis_);
    for (long k = 0; k < n_vec; ++k) {
        std::array<long, 4> offset = {k, 0, 0, 0};
        std::array<long, 4> slice_dims = {1, n_basis_, n_basis_, 1};
        unpack_ao_column(L_mat_.col(k).data(), n_basis_, packed, slice_sq);
        io.write_slice_4d(dataset_name, offset, slice_dims, slice_sq.data());
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
