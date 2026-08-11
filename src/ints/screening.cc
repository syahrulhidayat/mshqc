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

















#include "mshqc/integrals/screening.h"
#include "mshqc/ints/integrals.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#ifdef I
#undef I
#endif
#endif

namespace mshqc {
namespace integrals {

Screening::Screening(const BasisSet& basis)
    : basis_(basis), nshells_(basis.n_shells()), max_Q_(0.0)
{
    Q_matrix_ = Eigen::MatrixXd::Zero(nshells_, nshells_);
}

void Screening::reset() {
    max_Q_ = 0.0;
    Q_matrix_.setZero();
}

void Screening::compute(std::shared_ptr<mshqc::IntegralEngine> integrals) {
    if (!integrals) throw std::runtime_error("IntegralEngine null in Screening::compute");

    max_Q_ = 0.0;

    #pragma omp parallel for schedule(dynamic) reduction(max:max_Q_)
    for (int i = 0; i < nshells_; ++i) {
        for (int j = 0; j <= i; ++j) {

            auto buffer = integrals->compute_shell_block(i, j, i, j);

            double max_val = 0.0;
            for (double v : buffer) {
                max_val = std::max(max_val, std::abs(v));
            }

            double Q_val = std::sqrt(max_val);

            Q_matrix_(i, j) = Q_val;
            Q_matrix_(j, i) = Q_val;

            if (Q_val > max_Q_) max_Q_ = Q_val;
        }
    }

    if (max_Q_ < 1e-12) max_Q_ = 1.0;
}

std::vector<ShellPair> Screening::get_significant_pairs(double threshold) const {
    std::vector<ShellPair> pairs;
    pairs.reserve(nshells_ * nshells_ / 4);

    double safe_max_Q = (max_Q_ > 1e-12) ? max_Q_ : 1.0;
    double effective_cutoff = threshold / safe_max_Q;

    for (int i = 0; i < nshells_; ++i) {
        for (int j = 0; j <= i; ++j) {
            double val = Q_matrix_(i, j);
            if (val >= effective_cutoff) {
                pairs.push_back({i, j, val});
            }
        }
    }

    std::sort(pairs.begin(), pairs.end(),
        [](const ShellPair& a, const ShellPair& b) {
            return a.max_val > b.max_val;
        });

    return pairs;
}

bool Screening::is_significant(int sh_a, int sh_b, int sh_c, int sh_d, double threshold) const {
    double bound = Q_matrix_(sh_a, sh_b) * Q_matrix_(sh_c, sh_d);
    return bound >= threshold;
}

double Screening::get_schwarz_val(int sh_a, int sh_b) const {
    if (sh_a >= nshells_ || sh_b >= nshells_) return 0.0;
    return Q_matrix_(sh_a, sh_b);
}

void Screening::print_stats(double threshold) const {
    long long total_pairs = (long long)nshells_ * (nshells_ + 1) / 2;
    long long kept_pairs = 0;

    double safe_max_Q = (max_Q_ > 1e-12) ? max_Q_ : 1.0;
    double effective_cutoff = threshold / safe_max_Q;

    for (int i = 0; i < nshells_; ++i) {
        for (int j = 0; j <= i; ++j) {
            if (Q_matrix_(i, j) >= effective_cutoff) {
                kept_pairs++;
            }
        }
    }

    double percent = 100.0 * kept_pairs / total_pairs;

    std::cout << "\n  [Screening Stats]" << std::endl;
    std::cout << "  Threshold       : " << std::scientific << threshold << std::endl;
    std::cout << "  Max Schwarz (Q) : " << max_Q_ << std::endl;
    std::cout << "  Shell Pairs     : " << kept_pairs << " / " << total_pairs
              << " (" << std::fixed << std::setprecision(2) << percent << "% kept)" << std::endl;
    std::cout << "  Sparsity Gain   : " << (100.0 - percent) << "% computations skipped.\n" << std::endl;
}

void Screening::init_shell_centers() {}

void Screening::build_from_eri_tensor(const Eigen::Tensor<double, 4>& eri) {

}

}
}
