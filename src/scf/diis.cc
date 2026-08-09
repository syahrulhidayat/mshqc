 // ==============================================================================
 // Copyright (c) 2026 Syahrul and mshqc contributors
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

#include "mshqc/scf/diis.h"
#include <iostream>
#include <vector>
#include <Eigen/SVD>
#ifdef I
#undef I
#endif

namespace mshqc {

DIIS::DIIS(int max_vectors) : max_vectors_(max_vectors) {

    B_matrix_ = Eigen::MatrixXd::Zero(max_vectors + 1, max_vectors + 1);

}

void DIIS::clear() {
    fock_history_.clear();
    error_history_.clear();

    B_matrix_.setZero();
}

void DIIS::add_iteration(const Eigen::MatrixXd& F,
                         const Eigen::MatrixXd& err,
                         const Eigen::MatrixXd& )
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

Eigen::VectorXd DIIS::solve_svd(const Eigen::MatrixXd& A) {

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    double threshold = 1e-12;

    int n = A.rows();
    Eigen::VectorXd singular_values = svd.singularValues();
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

Eigen::MatrixXd DIIS::extrapolate() {
    int n = fock_history_.size();

    if (n < 2) return fock_history_.back();

    Eigen::MatrixXd A(n + 1, n + 1);

    A.topLeftCorner(n, n) = B_matrix_.topLeftCorner(n, n);

    A.col(n).head(n).setConstant(-1.0);
    A.row(n).head(n).setConstant(-1.0);
    A(n, n) = 0.0;

    Eigen::VectorXd coeffs_full = solve_svd(A);

    Eigen::VectorXd coeffs = coeffs_full.head(n);

    Eigen::MatrixXd F_ext = Eigen::MatrixXd::Zero(fock_history_[0].rows(), fock_history_[0].cols());

    for (int i = 0; i < n; i++) {
        F_ext += coeffs(i) * fock_history_[i];
    }

    return F_ext;
}

double DIIS::compute_ediis_element(int, int) const { return 0.0; }
Eigen::VectorXd DIIS::solve_cdiis(const Eigen::MatrixXd&) { return Eigen::VectorXd(); }
Eigen::VectorXd DIIS::solve_ediis() { return Eigen::VectorXd(); }

}
