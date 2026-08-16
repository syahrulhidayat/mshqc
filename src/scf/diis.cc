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
 * @file src/scf/diis.cc
 * @brief High-Performance Incremental SVD-DIIS Solver (Fixed for std::deque)
 * @details 
 * - Uses SVD to handle linear dependencies automatically.
 * - Incremental B-matrix update: O(k * N^2).
 * - Removed density_history_ overhead.
 */

#include "mshqc/scf/diis.h"
#include <iostream>
#include <vector>
#include <Eigen/SVD>
#ifdef I
#undef I
#endif

namespace mshqc {

DIIS::DIIS(int max_vectors) : max_vectors_(max_vectors) {
    // Resize B_matrix agar pas untuk (max + 1)
    B_matrix_ = Eigen::MatrixXd::Zero(max_vectors + 1, max_vectors + 1);
    
    // [FIX] std::deque tidak memiliki .reserve(). Baris tersebut dihapus.
    // Deque mengelola memori secara otomatis dan efisien.
}

void DIIS::clear() {
    fock_history_.clear();
    error_history_.clear();
    // density_history_ dihapus untuk efisiensi RAM & CPU
    B_matrix_.setZero();
}

void DIIS::add_iteration(const Eigen::MatrixXd& F, 
                         const Eigen::MatrixXd& err, 
                         const Eigen::MatrixXd& /*P*/) // Parameter P diabaikan (Unused)
{
    // 1. Jika buffer penuh, buang elemen terlama (index 0)
    if (fock_history_.size() >= max_vectors_) {
        fock_history_.pop_front();  // [FIX] Gunakan pop_front() untuk std::deque
        error_history_.pop_front();
        
        // Geser B Matrix (Penting: Geser blok kiri-atas ke pojok)
        int n = max_vectors_; 
        // Logika: data index 1..n digeser ke 0..n-1
        // B_matrix ukuran (n+1)x(n+1), kita geser subblok n x n (sebelum push baru)
        // Sebenarnya kita menggeser (n-1)x(n-1) elemen yang tersisa.
        // Blok mulai dari (1,1) sebesar (n-1)x(n-1) digeser ke (0,0)
        Eigen::MatrixXd temp = B_matrix_.block(1, 1, n-1, n-1);
        B_matrix_.topLeftCorner(n-1, n-1) = temp;
    }

    fock_history_.push_back(F);
    error_history_.push_back(err);

    // 2. Update B Matrix (Hanya baris/kolom terakhir yang aktif)
    int n = fock_history_.size(); 
    int new_idx = n - 1; // Index vektor baru (0-based)
    
    // Flatten error vector baru untuk dot product cepat
    // Eigen::Map memungkinkan kita melihat Matrix sebagai Vector tanpa copy (Zero-Copy)
    Eigen::Map<const Eigen::VectorXd> vec_new(err.data(), err.size());

    for (int i = 0; i < n; ++i) {
        // Map error lama sebagai vector
        Eigen::Map<const Eigen::VectorXd> vec_old(error_history_[i].data(), error_history_[i].size());
        
        // Dot product: <e_i | e_new>
        double val = vec_old.dot(vec_new);
        
        B_matrix_(i, new_idx) = val;
        B_matrix_(new_idx, i) = val; 
    }
}

// -----------------------------------------------------------------------------
// SVD SOLVER (Stability King)
// -----------------------------------------------------------------------------
Eigen::VectorXd DIIS::solve_svd(const Eigen::MatrixXd& A) {
    // Lakukan JacobiSVD (Sangat robust untuk matriks singular)
    // ComputeThinU | ComputeThinV cukup karena matriks A kecil dan square
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
    // Filter nilai singular yang terlalu kecil (Noise removal)
    double threshold = 1e-12;
    
    int n = A.rows();
    Eigen::VectorXd singular_values = svd.singularValues();
    Eigen::MatrixXd S_inv = Eigen::MatrixXd::Zero(n, n);
    
    for (int i = 0; i < n; ++i) {
        if (singular_values(i) > threshold) {
            S_inv(i, i) = 1.0 / singular_values(i);
        } else {
            S_inv(i, i) = 0.0; // Truncate noise / Linear Dependency
        }
    }

    // RHS Vector: [0, 0, ..., -1] (Lagrange multiplier constraint)
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
    rhs(n - 1) = -1.0;

    // Solve: x = V * S^-1 * U^T * b
    return svd.matrixV() * S_inv * svd.matrixU().transpose() * rhs;
}

Eigen::MatrixXd DIIS::extrapolate() {
    int n = fock_history_.size();
    
    // Jika history terlalu sedikit, belum bisa ekstrapolasi
    if (n < 2) return fock_history_.back();

    // 1. Siapkan Sistem Linear Pulay
    // Ambil sub-matriks B yang relevan (n x n) dari cache
    Eigen::MatrixXd A(n + 1, n + 1);
    
    // Copy blok error overlap dari cache B_matrix
    A.topLeftCorner(n, n) = B_matrix_.topLeftCorner(n, n);
    
    // Set border Lagrange Multiplier (-1 di pinggir, 0 di pojok)
    A.col(n).head(n).setConstant(-1.0);
    A.row(n).head(n).setConstant(-1.0);
    A(n, n) = 0.0;

    // 2. Selesaikan dengan SVD
    Eigen::VectorXd coeffs_full = solve_svd(A);
    
    // Ambil n koefisien pertama (abaikan lagrange multiplier di akhir)
    Eigen::VectorXd coeffs = coeffs_full.head(n);

    // 3. Konstruksi Fock Matrix Baru: F_new = sum(c_i * F_i)
    Eigen::MatrixXd F_ext = Eigen::MatrixXd::Zero(fock_history_[0].rows(), fock_history_[0].cols());
    
    for (int i = 0; i < n; i++) {
        F_ext += coeffs(i) * fock_history_[i];
    }

    return F_ext;
}

// Stub function untuk kompatibilitas interface lama
double DIIS::compute_ediis_element(int, int) const { return 0.0; }
Eigen::VectorXd DIIS::solve_cdiis(const Eigen::MatrixXd&) { return Eigen::VectorXd(); }
Eigen::VectorXd DIIS::solve_ediis() { return Eigen::VectorXd(); }

} // namespace mshqc