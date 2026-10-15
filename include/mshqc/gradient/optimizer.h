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

#ifndef MSHQC_OPTIMIZER_H
#define MSHQC_OPTIMIZER_H

#include <Eigen/Dense>
#include <functional>

#ifdef I
#undef I
#endif

namespace mshqc {
namespace gradient {

struct TrustRegionConfig {
    int max_micro_iter = 30;
    double micro_thresh = 1e-4;
    int print_level = 0;
    double precond_shift = 1e-4;
};

struct TrustRegionResult {
    Eigen::VectorXd step;
    double predicted_energy_change;
    bool hit_boundary;
};

class TrustRegionSOSCF {
public:
    explicit TrustRegionSOSCF(const TrustRegionConfig& config = TrustRegionConfig());

    TrustRegionResult solve(
        const Eigen::VectorXd& gradient,
        const Eigen::VectorXd& diag_hessian,
        double trust_radius,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> compute_hessian_vector
    );

private:
    TrustRegionConfig config_;

    double compute_boundary_intersection(
        const Eigen::VectorXd& z,
        const Eigen::VectorXd& p,
        double R
    );
};

} // namespace gradient
} // namespace mshqc

#endif // MSHQC_OPTIMIZER_H
