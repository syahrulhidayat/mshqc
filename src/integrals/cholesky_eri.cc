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

// ============================================================================
// CONSTRUCTORS
// ============================================================================

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

    // Build bf2shell_ map
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

// ============================================================================
// MAIN METHODS
// ============================================================================

void CholeskyERI::compute() {
    if (!integrals_ptr_ || !basis_ptr_) {
        std::cerr << "[Error] CholeskyERI: Missing Engine/Basis! Cannot run compute().\n";
        return;
    }
    decompose_direct();
}

// [FIXED] Implementasi Benar: Memproses Tensor Input dengan Optimasi Code 1
CholeskyDecompositionResult CholeskyERI::decompose(const Eigen::Tensor<double, 4>& eri_full) {
    // 1. Setup dimensi
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

    // 2. Extract Diagonal (pq|pq) from Tensor
    Eigen::VectorXd D(n_pairs);
    
    // Optimasi: Collapse loop untuk efisiensi cache
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n_basis_; ++i) {
        for (int j = 0; j < n_basis_; ++j) {
            D(i * n_basis_ + j) = eri_full(i, j, i, j);
        }
    }

    double max_diag = D.maxCoeff();
    if (config_.print_level > 0) 
        std::cout << "  Max Diag: " << std::scientific << max_diag << " | Threshold: " << threshold_ << "\n";

    // 3. Main Loop (Menggunakan MatrixXd Storage)
    int est_rank = std::min(n_pairs, std::max(200, n_basis_ * 5));
    Eigen::MatrixXd L_store(n_pairs, est_rank);
    
    int iter = 0;
    while (true) {
        // A. Find Pivot
        int pivot_idx;
        double D_max = D.maxCoeff(&pivot_idx);

        if (D_max < threshold_ || iter >= n_pairs) break;

        // Resize storage if needed
        if (iter >= L_store.cols()) {
            L_store.conservativeResize(Eigen::NoChange, L_store.cols() * 2);
        }

        double inv_sqrt = 1.0 / std::sqrt(D_max);
        int pi = pivot_idx / n_basis_;
        int pj = pivot_idx % n_basis_;

        // B. Compute New Column (Ambil dari Tensor)
        // Optimasi: Hindari pembagian integer (/) dan modulus (%) di dalam loop
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n_basis_; ++i) {
            for (int j = 0; j < n_basis_; ++j) {
                L_store(i * n_basis_ + j, iter) = eri_full(i, j, pi, pj);
            }
        }

        // C. Update Column (DGEMV - Highly Optimized by Eigen/BLAS)
        if (iter > 0) {
            // L_curr -= L_prev * L_prev_row(pivot)^T
            L_store.col(iter) -= L_store.leftCols(iter) * L_store.row(pivot_idx).head(iter).transpose();
        }

        // D. Scale & Update Diagonal
        L_store.col(iter) *= inv_sqrt;
        D.array() -= L_store.col(iter).array().square();

        // Pruning numerical noise
        // #pragma omp parallel for (Opsional, array operation sudah kencang)
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

// ============================================================================
// DIRECT DECOMPOSITION IMPLEMENTATION (OPTIMIZED SHELL-BASED)
// ============================================================================

void CholeskyERI::decompose_direct() {
    if (!integrals_ptr_ || !basis_ptr_) return;
    const auto& basis = *basis_ptr_;
    int nshells = basis.n_shells();
    int npair = n_basis_ * n_basis_;
    reset();

    std::vector<int> shell_starts(nshells);
    int offset = 0;
    for(int s = 0; s < nshells; ++s) {
        shell_starts[s] = offset; offset += basis.shell(s).n_functions();
    }

    Eigen::VectorXd D(npair); D.setZero();
    std::vector<double> shell_max(nshells * nshells, 0.0);

    // HAPUS OPENMP DI SINI (Serial agar aman dari tabrakan memori libcint)
    for (int s1 = 0; s1 < nshells; ++s1) {
        for (int s2 = 0; s2 <= s1; ++s2) {
            const auto& buf = integrals_ptr_->compute_shell_block(s1, s2, s1, s2);
            if (buf.empty()) continue;
            int dim1 = basis.shell(s1).n_functions(); int dim2 = basis.shell(s2).n_functions();
            int st1 = shell_starts[s1]; int st2 = shell_starts[s2];
            double max_val_block = 0.0;
            for(int i=0; i<dim1; ++i) {
                for(int j=0; j<dim2; ++j) {
                    // [PERBAIKAN 1]: C-Order index untuk (s1, s2 | s1, s2)
                    // Setara dengan [i][j][i][j]
                    size_t idx_buf = i + dim1 * (j + dim2 * (i + dim1 * j));
                    
                    if (idx_buf >= buf.size()) continue;
                    double val = buf[idx_buf];
                    int p = st1 + i; int q = st2 + j;
                    if (p * n_basis_ + q < npair) D(p * n_basis_ + q) = val;
                    if (q * n_basis_ + p < npair) D(q * n_basis_ + p) = val;
                    max_val_block = std::max(max_val_block, std::abs(val));
                }
            }
            shell_max[s1 * nshells + s2] = std::sqrt(max_val_block);
            shell_max[s2 * nshells + s1] = std::sqrt(max_val_block);
        }
    }

    int est_rank = std::min(npair, std::max(200, n_basis_ * 5));
    Eigen::MatrixXd L_store(npair, est_rank);
    Eigen::VectorXd col_buf(npair); 

    int iter = 0;
    while (true) {
        int pivot_idx; double D_max = D.maxCoeff(&pivot_idx);
        if (D_max < threshold_ || iter >= npair) break;
        if (iter >= L_store.cols()) L_store.conservativeResize(Eigen::NoChange, L_store.cols() * 2);

        int p = pivot_idx / n_basis_; int q = pivot_idx % n_basis_;
        if (p < q) std::swap(p, q); 
        
        int sp = bf2shell_[p]; int sq = bf2shell_[q];
        int rel_p = p - shell_starts[sp]; int rel_q = q - shell_starts[sq];
        int pair_pq = sp * (sp + 1) / 2 + sq; double inv_sqrt = 1.0 / std::sqrt(D_max);

        col_buf.setZero();
        
        // HAPUS OPENMP DI SINI
        for (int s1 = 0; s1 < nshells; ++s1) {
            for (int s2 = 0; s2 <= s1; ++s2) {
                double bound = shell_max[s1 * nshells + s2] * shell_max[sp * nshells + sq];
                if (bound < 1e-12) continue; 

                int pair_s = s1 * (s1 + 1) / 2 + s2;
                bool swap_pairs = (pair_s < pair_pq);
                
                int u1 = swap_pairs ? sp : s1; int u2 = swap_pairs ? sq : s2;
                int u3 = swap_pairs ? s1 : sp; int u4 = swap_pairs ? s2 : sq;

                const auto& buf = integrals_ptr_->compute_shell_block(u1, u2, u3, u4);
                if (buf.empty()) continue;

                int dim1 = basis.shell(s1).n_functions(); int dim2 = basis.shell(s2).n_functions();
                int dimP = basis.shell(sp).n_functions(); int dimQ = basis.shell(sq).n_functions();
                int st1 = shell_starts[s1]; int st2 = shell_starts[s2];

                for (int i = 0; i < dim1; ++i) {
                    for (int j = 0; j < dim2; ++j) {
                        size_t idx_buf;
                        if (!swap_pairs) {
                            // Buffer adalah (s1, s2 | sp, sq) -> u1=s1, u2=s2, u3=sp, u4=sq
                            // Dims: dim1, dim2, dimP, dimQ
                            // --- UBAH BAGIAN INI MENJADI FORTRAN-ORDER (i bergerak paling cepat) ---
                            idx_buf = i + dim1 * (j + dim2 * (rel_p + dimP * rel_q));
                        } else {
                            // Buffer adalah (sp, sq | s1, s2) -> u1=sp, u2=sq, u3=s1, u4=s2
                            // Dims: dimP, dimQ, dim1, dim2
                            // --- UBAH BAGIAN INI MENJADI FORTRAN-ORDER (rel_p bergerak paling cepat) ---
                            idx_buf = rel_p + dimP * (rel_q + dimQ * (i + dim1 * j));
                        }

                        if (idx_buf >= buf.size()) continue;
                        double val = buf[idx_buf];

                        int global_i = st1 + i; int global_j = st2 + j;
                        col_buf(global_i * n_basis_ + global_j) = val;
                        if (global_i != global_j) col_buf(global_j * n_basis_ + global_i) = val;
                    }
                }
            }
        }
        if (iter > 0) col_buf -= L_store.leftCols(iter) * L_store.row(pivot_idx).head(iter).transpose();
        L_store.col(iter) = col_buf * inv_sqrt; D.array() -= L_store.col(iter).array().square();
        for(int k=0; k<npair; ++k) if (D(k) < 0.0) D(k) = 0.0;
        iter++;
    }

    n_vectors_ = iter; decomposed_ = true; L_mat_ = L_store.leftCols(n_vectors_);
}

// ============================================================================
// UTILITIES
// ============================================================================

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
    return L_mat_.row(ij).dot(L_mat_.row(kl)); // Cepat dengan SIMD dot-product
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
        // Menggunakan pointer memori L_mat_ kolom ke-k
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
    
    // Alokasi matriks utama sekaligus
    L_mat_.resize(n_pairs, n_vectors_);
    
    for (int k = 0; k < n_vectors_; ++k) {
        std::array<long, 4> offset = {k, 0, 0, 0};
        std::array<long, 4> slice_dims = {1, n_basis_, n_basis_, 1};
        // Tulis langsung ke memori matriks L_mat_
        io.read_slice_4d(dataset_name, offset, slice_dims, L_mat_.col(k).data());
    }
    decomposed_ = true;
}

// Legacy Stubs
Eigen::Tensor<double, 4> CholeskyERI::reconstruct_full() const { return Eigen::Tensor<double, 4>(); }
std::pair<double, double> CholeskyERI::validate_reconstruction(const Eigen::Tensor<double, 4>&) { return {0.0, 0.0}; }
void CholeskyERI::print_statistics(bool) const {}
int CholeskyERI::find_pivot(const Eigen::VectorXd&) const { return 0; }
Eigen::VectorXd CholeskyERI::compute_new_vector(const Eigen::Tensor<double, 4>&, const Eigen::VectorXd&, int) const { return Eigen::VectorXd(); }

} // namespace integrals
} // namespace mshqc