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
#include <vector>
#include <cmath>
#include <Eigen/SVD>
#ifdef I
#undef I
#endif

namespace mshqc {

DIIS::DIIS(int max_vectors) : max_vectors_(max_vectors) {
    // Resize B_matrix agar pas untuk (max + 1)
    B_matrix_ = Eigen::MatrixXd::Zero(max_vectors + 1, max_vectors + 1);

}

void DIIS::clear() {
    fock_history_.clear();
    error_history_.clear();
    B_matrix_.setZero();
}

void DIIS::add_iteration(const Eigen::MatrixXd& F, const Eigen::MatrixXd& err)
{

    if (fock_history_.size() >= max_vectors_) {
        fock_history_.pop_front();
        error_history_.pop_front();
        int n = max_vectors_; 
        Eigen::MatrixXd temp = B_matrix_.block(1, 1, n-1, n-1);
        B_matrix_.topLeftCorner(n-1, n-1) = temp;
    }

    fock_history_.push_back(F);
    error_history_.push_back(err);
    int n = fock_history_.size(); 
    int new_idx = n - 1;
    Eigen::Map<const Eigen::VectorXd> vec_new(err.data(), err.size());

    for (int i = 0; i < n; ++i) {
        Eigen::Map<const Eigen::VectorXd> vec_old(error_history_[i].data(), error_history_[i].size());
        double val = vec_old.dot(vec_new); 
        B_matrix_(i, new_idx) = val;
        B_matrix_(new_idx, i) = val; 
    }
}

// -----------------------------------------------------------------------------
// SVD SOLVER (Relative Threshold for Equilibrated Pulay System)
// -----------------------------------------------------------------------------
Eigen::VectorXd DIIS::solve_svd(const Eigen::MatrixXd& A) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    int n = A.rows();
    Eigen::VectorXd singular_values = svd.singularValues();
    double max_sv = (singular_values.size() > 0) ? singular_values(0) : 1.0;
    double threshold = 1.0e-12 * std::max(1.0, max_sv);

    Eigen::MatrixXd S_inv = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        if (singular_values(i) > threshold) {
            S_inv(i, i) = 1.0 / singular_values(i);
        } else {
            S_inv(i, i) = 0.0;
        }
    }

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
    rhs(n - 1) = -1.0;

    return svd.matrixV() * S_inv * svd.matrixU().transpose() * rhs;
}

// [INJECTED_THIN_CORE_STAGE4_DIAG_DIIS]
Eigen::MatrixXd DIIS::extrapolate() {
    int n = static_cast<int>(fock_history_.size());
    if (n < 2) return fock_history_.back();

    Eigen::MatrixXd B_sub = B_matrix_.topLeftCorner(n, n);
    Eigen::VectorXd d(n);
    for (int i = 0; i < n; ++i) {
        const double bii = B_sub(i, i);
        if (!std::isfinite(bii) || bii <= 1.0e-30) {
            return fock_history_.back();
        }
        d(i) = std::sqrt(bii);
    }

    const double s_l = d.minCoeff();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n + 1, n + 1);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A(i, j) = B_sub(i, j) / (d(i) * d(j));
        }
        const double lag = -s_l / d(i);
        A(i, n) = lag;
        A(n, i) = lag;
    }
    A(n, n) = 0.0;

    Eigen::VectorXd y = solve_svd(A) * s_l;
    Eigen::VectorXd coeffs = y.head(n).cwiseQuotient(d);

    const double sum_c = coeffs.sum();
    if (!std::isfinite(sum_c) || std::abs(sum_c - 1.0) > 1.0e-4) {
        return fock_history_.back();
    }

    Eigen::MatrixXd F_new = Eigen::MatrixXd::Zero(fock_history_[0].rows(), fock_history_[0].cols());
    for (int i = 0; i < n; ++i) {
        F_new.noalias() += coeffs(i) * fock_history_[i];
    }
    return F_new;
}
} // namespace mshqc