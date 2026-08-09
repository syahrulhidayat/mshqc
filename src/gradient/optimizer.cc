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

#include "mshqc/gradient/optimizer.h"
#include "mshqc/basis.h"
#include "mshqc/ints/integrals.h"
#include "mshqc/scf/scf.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace gradient {

GeometryOptimizer::GeometryOptimizer(
    std::function<GradientResult(const Molecule&)> gradient_func,
    const OptConfig& config
) : gradient_func_(gradient_func), config_(config) {
}

OptResult GeometryOptimizer::optimize(const Molecule& initial_geom) {

    current_geom_ = initial_geom;
    current_coords_ = molecule_to_coords(current_geom_);

    energy_history_.clear();
    max_force_history_.clear();
    rms_force_history_.clear();

    auto grad_result = gradient_func_(current_geom_);
    current_energy_ = grad_result.energy;
    current_gradient_ = grad_result.gradient;

    if (config_.algorithm == OptAlgorithm::BFGS) {
        initialize_bfgs_hessian();
    }
    search_direction_ = -current_gradient_;

    if (config_.print_level > 0) {
        print_header();
    }

    OptResult result;
    result.algorithm = (config_.algorithm == OptAlgorithm::STEEPEST_DESCENT) ? "Steepest Descent" :
                       (config_.algorithm == OptAlgorithm::CONJUGATE_GRADIENT) ? "Conjugate Gradient" : "BFGS";

    bool converged = false;
    int iter = 0;

    for (iter = 0; iter < config_.max_iterations; ++iter) {

        energy_history_.push_back(current_energy_);

        double max_force, rms_force;
        compute_statistics(current_gradient_, rms_force, max_force);
        max_force_history_.push_back(max_force);
        rms_force_history_.push_back(rms_force);

        bool step_converged = step();

        if (step_converged) {
            converged = true;
            result.termination_reason = "Converged: All criteria satisfied";
            break;
        }
    }

    if (!converged) {
        result.termination_reason = "Maximum iterations reached";
    }

    result.converged = converged;
    result.n_iterations = iter + 1;
    result.final_geometry = current_geom_;
    result.final_energy = current_energy_;
    result.final_gradient = current_gradient_;
    result.energy_history = energy_history_;
    result.max_force_history = max_force_history_;
    result.rms_force_history = rms_force_history_;

    compute_statistics(current_gradient_, result.rms_force, result.max_force);

    if (iter > 0) {
        result.energy_change = std::abs(energy_history_.back() - energy_history_[energy_history_.size()-2]);
    }

    if (config_.print_level > 0) {
        print_results(result);
    }

    return result;
}

bool GeometryOptimizer::step() {

    Eigen::VectorXd prev_coords = current_coords_;
    prev_gradient_ = current_gradient_;
    double prev_energy = current_energy_;

    switch (config_.algorithm) {
        case OptAlgorithm::STEEPEST_DESCENT:
            steepest_descent_step();
            break;
        case OptAlgorithm::CONJUGATE_GRADIENT:
            conjugate_gradient_step();
            break;
        case OptAlgorithm::BFGS:
            bfgs_step();
            break;
    }

    double alpha = 1.0;
    if (config_.use_line_search) {
        alpha = line_search(search_direction_);
    }

    Eigen::VectorXd step_vector = alpha * search_direction_;

    if (config_.algorithm == OptAlgorithm::BFGS) {
        step_vector = apply_trust_region(step_vector);
    }

    current_coords_ += step_vector;
    current_geom_ = coords_to_molecule(current_coords_);

    auto grad_result = gradient_func_(current_geom_);
    current_energy_ = grad_result.energy;
    current_gradient_ = grad_result.gradient;

    if (config_.algorithm == OptAlgorithm::BFGS && energy_history_.size() > 0) {
        Eigen::VectorXd s = current_coords_ - prev_coords;
        Eigen::VectorXd y = current_gradient_ - prev_gradient_;
        update_bfgs_hessian(s, y);
    }

    if (config_.print_level > 0) {
        print_iteration(energy_history_.size(), step_vector);
    }

    return check_convergence(step_vector);
}

bool GeometryOptimizer::check_convergence(const Eigen::VectorXd& step_vector) {
    double max_force, rms_force;
    compute_statistics(current_gradient_, rms_force, max_force);

    double max_step, rms_step;
    compute_statistics(step_vector, rms_step, max_step);

    double energy_change = 0.0;
    if (energy_history_.size() > 0) {
        energy_change = std::abs(current_energy_ - energy_history_.back());
    }

    bool force_converged = (max_force < config_.max_force_thresh) &&
                           (rms_force < config_.rms_force_thresh);
    bool step_converged = (max_step < config_.max_step_thresh) &&
                          (rms_step < config_.rms_step_thresh);
    bool energy_converged = energy_change < config_.energy_thresh;

    return force_converged && (step_converged || energy_converged);
}

void GeometryOptimizer::steepest_descent_step() {
    search_direction_ = -current_gradient_;
}

void GeometryOptimizer::conjugate_gradient_step() {

    if (energy_history_.empty()) {

        search_direction_ = -current_gradient_;
    } else {

        Eigen::VectorXd grad_diff = current_gradient_ - prev_gradient_;
        double numerator = current_gradient_.dot(grad_diff);
        double denominator = prev_gradient_.squaredNorm();

        double beta = 0.0;
        if (denominator > 1e-10) {
            beta = std::max(0.0, numerator / denominator);
        }

        search_direction_ = -current_gradient_ + beta * search_direction_;

        if (search_direction_.dot(current_gradient_) >= 0) {
            search_direction_ = -current_gradient_;
        }
    }
}

void GeometryOptimizer::bfgs_step() {

    search_direction_ = -hessian_inverse_ * current_gradient_;

    if (search_direction_.dot(current_gradient_) >= 0) {

        search_direction_ = -current_gradient_;
    }
}

void GeometryOptimizer::initialize_bfgs_hessian() {

    int n = current_coords_.size();
    hessian_inverse_ = Eigen::MatrixXd::Identity(n, n);
}

void GeometryOptimizer::update_bfgs_hessian(const Eigen::VectorXd& s, const Eigen::VectorXd& y) {

    double ys = y.dot(s);
    if (ys < 1e-10) {
        return;
    }

    double rho = 1.0 / ys;
    int n = s.size();

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd V = I - rho * y * s.transpose();
    hessian_inverse_ = V.transpose() * hessian_inverse_ * V + rho * s * s.transpose();
}

double GeometryOptimizer::line_search(const Eigen::VectorXd& direction) {

    double alpha = config_.alpha_init;
    double grad_dot_dir = current_gradient_.dot(direction);
    double threshold = config_.c1 * grad_dot_dir;

    for (int i = 0; i < config_.max_line_search; ++i) {
        Eigen::VectorXd test_coords = current_coords_ + alpha * direction;
        Molecule test_geom = coords_to_molecule(test_coords);

        auto grad_result = gradient_func_(test_geom);
        double test_energy = grad_result.energy;
        if (test_energy <= current_energy_ + alpha * threshold) {
            return alpha;
        }
        alpha *= config_.rho;

        if (alpha < 1e-10) {
            return alpha;
        }
    }

    return alpha;
}

Eigen::VectorXd GeometryOptimizer::apply_trust_region(const Eigen::VectorXd& step) {

    double step_norm = step.norm();

    if (step_norm <= config_.trust_radius) {
        return step;
    }

    return step * (config_.trust_radius / step_norm);
}

Molecule GeometryOptimizer::coords_to_molecule(const Eigen::VectorXd& coords) const {
    Molecule mol;
    mol.set_charge(current_geom_.charge());
    mol.set_multiplicity(current_geom_.multiplicity());

    int natoms = current_geom_.n_atoms();
    for (int i = 0; i < natoms; ++i) {
        int Z = current_geom_.atom(i).atomic_number;
        double x = coords(3*i + 0);
        double y = coords(3*i + 1);
        double z = coords(3*i + 2);

        mol.add_atom(Z, x, y, z);
    }

    return mol;
}

Eigen::VectorXd GeometryOptimizer::molecule_to_coords(const Molecule& mol) const {
    int natoms = mol.n_atoms();
    Eigen::VectorXd coords(3 * natoms);

    for (int i = 0; i < natoms; ++i) {
        coords(3*i + 0) = mol.atom(i).x;
        coords(3*i + 1) = mol.atom(i).y;
        coords(3*i + 2) = mol.atom(i).z;
    }

    return coords;
}

void GeometryOptimizer::compute_statistics(const Eigen::VectorXd& vec, double& rms, double& max_val) {
    int n = vec.size();

    if (n == 0) {
        rms = 0.0;
        max_val = 0.0;
        return;
    }

    max_val = vec.cwiseAbs().maxCoeff();
    rms = std::sqrt(vec.squaredNorm() / n);
}

void GeometryOptimizer::print_header() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Geometry Optimization\n";
    std::cout << "========================================\n";
    std::cout << "Algorithm: ";
    if (config_.algorithm == OptAlgorithm::STEEPEST_DESCENT) {
        std::cout << "Steepest Descent\n";
    } else if (config_.algorithm == OptAlgorithm::CONJUGATE_GRADIENT) {
        std::cout << "Conjugate Gradient (Polak-Ribière)\n";
    } else {
        std::cout << "BFGS Quasi-Newton\n";
    }
    std::cout << "\n";
    std::cout << "Convergence criteria:\n";
    std::cout << "  Max force:  " << std::scientific << std::setprecision(2)
              << config_.max_force_thresh << " Ha/bohr\n";
    std::cout << "  RMS force:  " << config_.rms_force_thresh << " Ha/bohr\n";
    std::cout << "  Max step:   " << config_.max_step_thresh << " bohr\n";
    std::cout << "  RMS step:   " << config_.rms_step_thresh << " bohr\n";
    std::cout << "  ΔE:         " << config_.energy_thresh << " Ha\n";
    std::cout << "\n";
    std::cout << "Line search: " << (config_.use_line_search ? "Enabled (Armijo)" : "Disabled") << "\n";
    if (config_.algorithm == OptAlgorithm::BFGS) {
        std::cout << "Trust radius: " << std::fixed << std::setprecision(3)
                  << config_.trust_radius << " bohr\n";
    }
    std::cout << "\n";
    std::cout << "Iter    Energy (Ha)      ΔE (Ha)       Max Force    RMS Force    Max Step     RMS Step\n";
    std::cout << "--------------------------------------------------------------------------------------------\n";
}

void GeometryOptimizer::print_iteration(int iter, const Eigen::VectorXd& step_vector) {
    double max_force, rms_force;
    compute_statistics(current_gradient_, rms_force, max_force);

    double max_step, rms_step;
    compute_statistics(step_vector, rms_step, max_step);

    double energy_change = 0.0;
    if (energy_history_.size() > 0) {
        energy_change = current_energy_ - energy_history_.back();
    }

    std::cout << std::setw(4) << iter << "  ";
    std::cout << std::fixed << std::setprecision(8) << std::setw(15) << current_energy_ << "  ";
    std::cout << std::scientific << std::setprecision(2) << std::setw(12) << energy_change << "  ";
    std::cout << std::setw(12) << max_force << "  ";
    std::cout << std::setw(12) << rms_force << "  ";
    std::cout << std::setw(12) << max_step << "  ";
    std::cout << std::setw(12) << rms_step << "\n";

    if (config_.print_geometry && config_.print_level > 1) {
        std::cout << "\nCurrent geometry (Bohr):\n";
        int natoms = current_geom_.n_atoms();
        for (int i = 0; i < natoms; ++i) {
            const auto& atom = current_geom_.atom(i);
            std::cout << "  " << atom.atomic_number << "  "
                      << std::fixed << std::setprecision(6)
                      << std::setw(12) << atom.x << "  "
                      << std::setw(12) << atom.y << "  "
                      << std::setw(12) << atom.z << "\n";
        }
        std::cout << "\n";
    }
}

void GeometryOptimizer::print_results(const OptResult& result) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Optimization Complete\n";
    std::cout << "========================================\n";
    std::cout << "Status: " << (result.converged ? "CONVERGED ✓" : "NOT CONVERGED ✗") << "\n";
    std::cout << "Reason: " << result.termination_reason << "\n";
    std::cout << "Iterations: " << result.n_iterations << "\n";
    std::cout << "\n";
    std::cout << "Final energy: " << std::fixed << std::setprecision(10)
              << result.final_energy << " Ha\n";
    std::cout << "\n";
    std::cout << "Final gradient:\n";
    std::cout << "  Max force: " << std::scientific << std::setprecision(4)
              << result.max_force << " Ha/bohr";
    if (result.max_force < config_.max_force_thresh) std::cout << " ✓";
    std::cout << "\n";
    std::cout << "  RMS force: " << result.rms_force << " Ha/bohr";
    if (result.rms_force < config_.rms_force_thresh) std::cout << " ✓";
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "Final geometry (Bohr):\n";
    result.final_geometry.print();
    std::cout << "========================================\n";
}

OptResult optimize_rhf(
    const Molecule& initial_geom,
    const std::string& basis_name,
    const OptConfig& config
) {
    auto gradient_func = [basis_name](const Molecule& mol) -> GradientResult {
        BasisSet basis(basis_name, mol);
        auto integrals = std::make_shared<IntegralEngine>(mol, basis);

        SCFConfig scf_config;
        scf_config.print_level = 0;

        return compute_rhf_gradient_numerical(mol, basis, integrals, 0, scf_config);
    };

    GeometryOptimizer optimizer(gradient_func, config);
    return optimizer.optimize(initial_geom);
}

OptResult optimize_uhf(
    const Molecule& initial_geom,
    const std::string& basis_name,
    int charge,
    int multiplicity,
    const OptConfig& config
) {

    auto gradient_func = [basis_name, charge, multiplicity](const Molecule& mol) -> GradientResult {
        BasisSet basis(basis_name, mol);
        auto integrals = std::make_shared<IntegralEngine>(mol, basis);

        SCFConfig scf_config;
        scf_config.print_level = 0;

        return compute_uhf_gradient_numerical(mol, basis, integrals, charge, multiplicity, scf_config);
    };

    GeometryOptimizer optimizer(gradient_func, config);
    return optimizer.optimize(initial_geom);
}

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

double TrustRegionSOSCF::compute_boundary_intersection(const Eigen::VectorXd& z, const Eigen::VectorXd& p, double R) {
    double a = p.squaredNorm();
    double b = 2.0 * z.dot(p);
    double c = z.squaredNorm() - (R * R);

    double discriminant = (b * b) - (4.0 * a * c);
    if (discriminant < 0.0) return 0.0;
    return (-b + std::sqrt(discriminant)) / (2.0 * a);
}

}
}
