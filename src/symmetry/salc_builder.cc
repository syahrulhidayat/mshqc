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


#include "mshqc/symmetry/salc_builder.h"
#include <random>
#include <iostream>
#include <algorithm>

namespace mshqc {

SalcBuilder::SalcBuilder(BasisSymmetrizer* sym) : sym_(sym) {}

std::pair<Eigen::MatrixXd, std::vector<int>> SalcBuilder::build_salc(const Eigen::MatrixXd& S) {
    int nbf = S.rows();

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nbf, nbf);
    for (int i = 0; i < nbf; ++i) {
        for (int j = 0; j <= i; ++j) {
            double val = S(i, j) * (std::sin(i * 1.34 + j * 2.51) + 1.5);
            M(i, j) = val;
            M(j, i) = val;
        }
    }

    sym_->symmetrize(M);

    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(M, S);
    Eigen::MatrixXd C_salc = es.eigenvectors();

    std::vector<int> irreps = sym_->assign_mo_irreps(C_salc, 1e-5);

    std::vector<std::pair<int, int>> sort_data;
    for (int i = 0; i < nbf; ++i) {
        sort_data.push_back({irreps[i], i});
    }

    std::stable_sort(sort_data.begin(), sort_data.end(),
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.first < b.first;
        });

    Eigen::MatrixXd X_salc = Eigen::MatrixXd::Zero(nbf, nbf);
    std::vector<int> sorted_irreps(nbf);

    for (int i = 0; i < nbf; ++i) {
        X_salc.col(i) = C_salc.col(sort_data[i].second);
        sorted_irreps[i] = sort_data[i].first;
    }

    for (int i = 0; i < nbf; ++i) {
        double norm = std::sqrt(X_salc.col(i).transpose() * S * X_salc.col(i));
        X_salc.col(i) /= norm;
    }

    return {X_salc, sorted_irreps};
}

}
