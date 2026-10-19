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

#include "mshqc/gradient/optimizer.h"
#include <cmath>
#include <algorithm>

#ifdef I
#undef I
#endif

namespace mshqc {
namespace gradient {

TrustRegionSOSCF::TrustRegionSOSCF(const TrustRegionConfig& config) : config_(config) {}

TrustRegionResult TrustRegionSOSCF::solve(
    const Eigen::VectorXd& gradient,
    const Eigen::VectorXd& diag_hessian,
    double trust_radius,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> compute_hessian_vector)
{
    int n = gradient.size();
    Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd Hz = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd r = gradient;

    Eigen::VectorXd M_inv = diag_hessian.cwiseMax(config_.precond_shift).cwiseInverse();
    Eigen::VectorXd p = -M_inv.cwiseProduct(r);

    double r_norm = r.norm();
    if (r_norm < config_.micro_thresh) return {z, 0.0, false};

    double r_M_r_old = r.dot(-p);

    for (int iter = 0; iter < config_.max_micro_iter; ++iter) {
        Eigen::VectorXd Hp = compute_hessian_vector(p);
        double kappa = p.dot(Hp);
        if (kappa <= 0.0) {
            double tau = compute_boundary_intersection(z, p, trust_radius);
            Eigen::VectorXd step = z + tau * p;
            Eigen::VectorXd H_step = Hz + tau * Hp;
            double m_energy = gradient.dot(step) + 0.5 * step.dot(H_step);
            return {step, m_energy, true};
        }

        double alpha = r_M_r_old / kappa;
        Eigen::VectorXd z_next = z + alpha * p;
        if (z_next.norm() >= trust_radius) {
            double tau = compute_boundary_intersection(z, p, trust_radius);
            Eigen::VectorXd step = z + tau * p;
            Eigen::VectorXd H_step = Hz + tau * Hp;
            double m_energy = gradient.dot(step) + 0.5 * step.dot(H_step);
            return {step, m_energy, true};
        }

        z = z_next;
        Hz += alpha * Hp;
        r += alpha * Hp;

        if (r.norm() < config_.micro_thresh) {
            double m_energy = gradient.dot(z) + 0.5 * z.dot(Hz);
            return {z, m_energy, false};
        }

        Eigen::VectorXd z_M_inv = M_inv.cwiseProduct(r);
        double r_M_r_new = r.dot(z_M_inv);

        double beta = r_M_r_new / r_M_r_old;
        p = -z_M_inv + beta * p;
        r_M_r_old = r_M_r_new;
    }

    double final_m_energy = gradient.dot(z) + 0.5 * z.dot(Hz);
    return {z, final_m_energy, false};
}

double TrustRegionSOSCF::compute_boundary_intersection(
    const Eigen::VectorXd& z,
    const Eigen::VectorXd& p,
    double R)
{
    double a = p.squaredNorm();
    double b = 2.0 * z.dot(p);
    double c = z.squaredNorm() - (R * R);

    double discriminant = (b * b) - (4.0 * a * c);
    if (discriminant < 0.0) return 0.0;
    return (-b + std::sqrt(discriminant)) / (2.0 * a);
}

} // namespace gradient
} // namespace mshqc
