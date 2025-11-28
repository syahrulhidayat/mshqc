/**
 * @file optimizer.cc
 * @brief Implementation of geometry optimization algorithms
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 * 
 * @note Original implementation based on Nocedal & Wright (2006) and Schlegel (1987).
 *       No code copied from other quantum chemistry packages.
 */

#include "mshqc/gradient/optimizer.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

namespace mshqc {
namespace gradient {

// ============================================================================
// GeometryOptimizer Implementation
// ============================================================================

GeometryOptimizer::GeometryOptimizer(
    std::function<GradientResult(const Molecule&)> gradient_func,
    const OptConfig& config
) : gradient_func_(gradient_func), config_(config) {
}

OptResult GeometryOptimizer::optimize(const Molecule& initial_geom) {
    // Initialize
    current_geom_ = initial_geom;
    current_coords_ = molecule_to_coords(current_geom_);
    
    energy_history_.clear();
    max_force_history_.clear();
    rms_force_history_.clear();
    
    // Compute initial gradient
    auto grad_result = gradient_func_(current_geom_);
    current_energy_ = grad_result.energy;
    current_gradient_ = grad_result.gradient;
    
    // Initialize algorithm-specific state
    if (config_.algorithm == OptAlgorithm::BFGS) {
        initialize_bfgs_hessian();
    }
    search_direction_ = -current_gradient_;  // Initial direction: steepest descent
    
    // Print header
    if (config_.print_level > 0) {
        print_header();
    }
    
    // Optimization loop
    OptResult result;
    result.algorithm = (config_.algorithm == OptAlgorithm::STEEPEST_DESCENT) ? "Steepest Descent" :
                       (config_.algorithm == OptAlgorithm::CONJUGATE_GRADIENT) ? "Conjugate Gradient" : "BFGS";
    
    bool converged = false;
    int iter = 0;
    
    for (iter = 0; iter < config_.max_iterations; ++iter) {
        // Store current state
        energy_history_.push_back(current_energy_);
        
        double max_force, rms_force;
        compute_statistics(current_gradient_, rms_force, max_force);
        max_force_history_.push_back(max_force);
        rms_force_history_.push_back(rms_force);
        
        // Perform optimization step
        bool step_converged = step();
        
        // Check convergence
        if (step_converged) {
            converged = true;
            result.termination_reason = "Converged: All criteria satisfied";
            break;
        }
    }
    
    if (!converged) {
        result.termination_reason = "Maximum iterations reached";
    }
    
    // Fill result
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
    
    // Print results
    if (config_.print_level > 0) {
        print_results(result);
    }
    
    return result;
}

bool GeometryOptimizer::step() {
    // Save previous state
    Eigen::VectorXd prev_coords = current_coords_;
    prev_gradient_ = current_gradient_;
    double prev_energy = current_energy_;
    
    // Compute search direction based on algorithm
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
    
    // Apply line search if enabled
    double alpha = 1.0;
    if (config_.use_line_search) {
        alpha = line_search(search_direction_);
    }
    
    // Take step
    Eigen::VectorXd step_vector = alpha * search_direction_;
    
    // Apply trust region for BFGS
    if (config_.algorithm == OptAlgorithm::BFGS) {
        step_vector = apply_trust_region(step_vector);
    }
    
    current_coords_ += step_vector;
    current_geom_ = coords_to_molecule(current_coords_);
    
    // Compute new gradient
    auto grad_result = gradient_func_(current_geom_);
    current_energy_ = grad_result.energy;
    current_gradient_ = grad_result.gradient;
    
    // Update BFGS Hessian if using BFGS
    if (config_.algorithm == OptAlgorithm::BFGS && energy_history_.size() > 0) {
        Eigen::VectorXd s = current_coords_ - prev_coords;
        Eigen::VectorXd y = current_gradient_ - prev_gradient_;
        update_bfgs_hessian(s, y);
    }
    
    // Print iteration info
    if (config_.print_level > 0) {
        print_iteration(energy_history_.size(), step_vector);
    }
    
    // Check convergence
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
    
    // All criteria must be satisfied
    bool force_converged = (max_force < config_.max_force_thresh) && 
                           (rms_force < config_.rms_force_thresh);
    bool step_converged = (max_step < config_.max_step_thresh) && 
                          (rms_step < config_.rms_step_thresh);
    bool energy_converged = energy_change < config_.energy_thresh;
    
    return force_converged && (step_converged || energy_converged);
}

// ============================================================================
// Algorithm Implementations
// ============================================================================

void GeometryOptimizer::steepest_descent_step() {
    // Direction: d_k = -∇E_k
    // REFERENCE: Nocedal & Wright (2006), Equation (3.1)
    search_direction_ = -current_gradient_;
}

void GeometryOptimizer::conjugate_gradient_step() {
    // Polak-Ribière formula (more robust than Fletcher-Reeves)
    // β_k = max(0, g_k·(g_k - g_{k-1}) / ||g_{k-1}||²)
    // d_k = -g_k + β_k·d_{k-1}
    //
    // REFERENCE: Nocedal & Wright (2006), Equation (5.44)
    
    if (energy_history_.empty()) {
        // First iteration: steepest descent
        search_direction_ = -current_gradient_;
    } else {
        // Compute β using Polak-Ribière
        Eigen::VectorXd grad_diff = current_gradient_ - prev_gradient_;
        double numerator = current_gradient_.dot(grad_diff);
        double denominator = prev_gradient_.squaredNorm();
        
        double beta = 0.0;
        if (denominator > 1e-10) {
            beta = std::max(0.0, numerator / denominator);
        }
        
        // New search direction
        search_direction_ = -current_gradient_ + beta * search_direction_;
        
        // Restart if not descent direction
        if (search_direction_.dot(current_gradient_) >= 0) {
            search_direction_ = -current_gradient_;
        }
    }
}

void GeometryOptimizer::bfgs_step() {
    // Search direction: d_k = -H_k^{-1}·g_k
    // where H_k is approximate inverse Hessian
    //
    // REFERENCE: Nocedal & Wright (2006), Algorithm 6.1
    
    search_direction_ = -hessian_inverse_ * current_gradient_;
    
    // Ensure descent direction
    if (search_direction_.dot(current_gradient_) >= 0) {
        // Fall back to steepest descent if not descent direction
        search_direction_ = -current_gradient_;
    }
}

void GeometryOptimizer::initialize_bfgs_hessian() {
    // Initialize H_0^{-1} = I (identity matrix)
    // REFERENCE: Nocedal & Wright (2006), Section 6.1
    int n = current_coords_.size();
    hessian_inverse_ = Eigen::MatrixXd::Identity(n, n);
}

void GeometryOptimizer::update_bfgs_hessian(const Eigen::VectorXd& s, const Eigen::VectorXd& y) {
    // BFGS update formula:
    // H_{k+1} = (I - ρ s y^T) H_k (I - ρ y s^T) + ρ s s^T
    // where ρ = 1 / (y^T s)
    //
    // REFERENCE: Nocedal & Wright (2006), Equation (6.17)
    
    double ys = y.dot(s);
    
    // Check curvature condition
    if (ys < 1e-10) {
        // Skip update if curvature condition not satisfied
        return;
    }
    
    double rho = 1.0 / ys;
    int n = s.size();
    
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd V = I - rho * y * s.transpose();
    
    // H_{k+1} = V^T H_k V + ρ s s^T
    hessian_inverse_ = V.transpose() * hessian_inverse_ * V + rho * s * s.transpose();
}

// ============================================================================
// Line Search
// ============================================================================

double GeometryOptimizer::line_search(const Eigen::VectorXd& direction) {
    // Backtracking line search with Armijo condition
    // Find α such that: E(x + α·d) ≤ E(x) + c1·α·∇E^T·d
    //
    // REFERENCE: Nocedal & Wright (2006), Algorithm 3.1
    
    double alpha = config_.alpha_init;
    double grad_dot_dir = current_gradient_.dot(direction);
    
    // Armijo condition threshold
    double threshold = config_.c1 * grad_dot_dir;
    
    for (int i = 0; i < config_.max_line_search; ++i) {
        // Test geometry at x + α·d
        Eigen::VectorXd test_coords = current_coords_ + alpha * direction;
        Molecule test_geom = coords_to_molecule(test_coords);
        
        auto grad_result = gradient_func_(test_geom);
        double test_energy = grad_result.energy;
        
        // Check Armijo condition
        if (test_energy <= current_energy_ + alpha * threshold) {
            return alpha;
        }
        
        // Reduce step size
        alpha *= config_.rho;
        
        // Safety check
        if (alpha < 1e-10) {
            return alpha;
        }
    }
    
    return alpha;
}

Eigen::VectorXd GeometryOptimizer::apply_trust_region(const Eigen::VectorXd& step) {
    // Scale step to fit within trust radius
    // REFERENCE: Nocedal & Wright (2006), Section 4.1
    
    double step_norm = step.norm();
    
    if (step_norm <= config_.trust_radius) {
        return step;
    }
    
    // Scale down to trust radius
    return step * (config_.trust_radius / step_norm);
}

// ============================================================================
// Coordinate Conversion
// ============================================================================

Molecule GeometryOptimizer::coords_to_molecule(const Eigen::VectorXd& coords) const {
    // Rebuild molecule with new coordinates
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

// ============================================================================
// Utilities
// ============================================================================

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
    
    // Print geometry if requested
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

// ============================================================================
// Convenience Functions
// ============================================================================

OptResult optimize_rhf(
    const Molecule& initial_geom,
    const std::string& basis_name,
    const OptConfig& config
) {
    // Create gradient function for RHF
    auto gradient_func = [basis_name](const Molecule& mol) -> GradientResult {
        BasisSet basis(basis_name, mol);
        auto integrals = std::make_shared<IntegralEngine>(mol, basis);
        
        SCFConfig scf_config;
        scf_config.print_level = 0;  // Silence SCF output during optimization
        
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
    // Create gradient function for UHF
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

} // namespace gradient
} // namespace mshqc
