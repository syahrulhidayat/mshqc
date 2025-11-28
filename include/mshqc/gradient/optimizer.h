/**
 * @file optimizer.h
 * @brief Geometry optimization algorithms for molecular structure optimization
 * 
 * This module provides multiple optimization algorithms:
 * - Steepest Descent (SD): First-order, gradient-only
 * - Conjugate Gradient (CG): First-order with momentum
 * - BFGS Quasi-Newton: Builds approximate Hessian, fastest convergence
 * 
 * All algorithms use analytical or numerical gradients from gradient.h
 * and support arbitrary energy/gradient functions via std::function.
 * 
 * THEORY REFERENCES:
 * 1. J. Nocedal & S.J. Wright, "Numerical Optimization" (2006), Chapters 3, 5, 6
 *    - Comprehensive optimization theory and algorithms
 * 2. W.H. Press et al., "Numerical Recipes" (2007), Chapter 10
 *    - Practical optimization implementation details
 * 3. H.B. Schlegel, Adv. Chem. Phys. 67, 249 (1987)
 *    - Geometry optimization in quantum chemistry
 * 4. P. Pulay, Mol. Phys. 17, 197 (1969)
 *    - Force method for molecular geometry optimization
 * 5. C. Peng et al., J. Comput. Chem. 17, 49 (1996)
 *    - Combination of internal coordinates and delocalized coordinates
 * 6. J. Baker, J. Comput. Chem. 7, 385 (1986)
 *    - Convergence criteria for geometry optimization
 * 7. R. Fletcher & C.M. Reeves, Comput. J. 7, 149 (1964)
 *    - Conjugate gradient method for optimization
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 * 
 * @note This is original implementation based on theory papers.
 *       No code was copied from other quantum chemistry software.
 */

#ifndef MSHQC_OPTIMIZER_H
#define MSHQC_OPTIMIZER_H

#include "mshqc/gradient/gradient.h"
#include "mshqc/molecule.h"
#include <Eigen/Dense>
#include <functional>
#include <vector>
#include <string>
#include <memory>

namespace mshqc {
namespace gradient {

// ============================================================================
// Optimization Configuration
// ============================================================================

/**
 * @brief Optimization algorithm type
 */
enum class OptAlgorithm {
    STEEPEST_DESCENT,  ///< Steepest descent (SD) - simple, robust, slow
    CONJUGATE_GRADIENT, ///< Conjugate gradient (CG) - faster than SD
    BFGS               ///< BFGS quasi-Newton - fastest, builds Hessian approx
};

/**
 * @brief Configuration for geometry optimization
 * 
 * Default convergence criteria based on Baker (1986):
 * - max_force: 4.5×10⁻⁴ Ha/bohr
 * - rms_force: 3.0×10⁻⁴ Ha/bohr
 * - max_step:  1.8×10⁻³ bohr
 * - rms_step:  1.2×10⁻³ bohr
 */
struct OptConfig {
    OptAlgorithm algorithm = OptAlgorithm::BFGS;
    
    // Convergence thresholds
    double max_force_thresh = 4.5e-4;  ///< Max gradient component (Ha/bohr)
    double rms_force_thresh = 3.0e-4;  ///< RMS gradient (Ha/bohr)
    double max_step_thresh  = 1.8e-3;  ///< Max displacement (bohr)
    double rms_step_thresh  = 1.2e-3;  ///< RMS displacement (bohr)
    double energy_thresh    = 1.0e-6;  ///< Energy change (Ha)
    
    // Iteration control
    int max_iterations = 100;          ///< Maximum optimization steps
    
    // Line search parameters (Armijo-Goldstein)
    bool use_line_search = true;       ///< Enable backtracking line search
    double alpha_init = 1.0;           ///< Initial step size
    double alpha_max = 2.0;            ///< Maximum step size
    double rho = 0.5;                  ///< Step reduction factor (0 < rho < 1)
    double c1 = 1e-4;                  ///< Armijo condition parameter
    int max_line_search = 20;          ///< Max line search iterations
    
    // Trust region (for BFGS)
    double trust_radius = 0.3;         ///< Maximum step size (bohr)
    double trust_radius_max = 0.5;     ///< Maximum trust radius
    double trust_radius_min = 0.01;    ///< Minimum trust radius
    
    // Output control
    int print_level = 1;               ///< 0=silent, 1=normal, 2=verbose
    bool print_geometry = true;        ///< Print geometry each iteration
};

// ============================================================================
// Optimization Result
// ============================================================================

/**
 * @brief Result of geometry optimization
 */
struct OptResult {
    bool converged = false;            ///< Did optimization converge?
    int n_iterations = 0;              ///< Number of iterations performed
    
    Molecule final_geometry;           ///< Optimized molecular geometry
    double final_energy = 0.0;         ///< Final energy (Ha)
    Eigen::VectorXd final_gradient;    ///< Final gradient (Ha/bohr)
    
    double max_force = 0.0;            ///< Max gradient component (Ha/bohr)
    double rms_force = 0.0;            ///< RMS gradient (Ha/bohr)
    double max_step = 0.0;             ///< Max displacement in last step (bohr)
    double rms_step = 0.0;             ///< RMS displacement in last step (bohr)
    double energy_change = 0.0;        ///< Energy change in last step (Ha)
    
    std::vector<double> energy_history;      ///< Energy at each iteration
    std::vector<double> max_force_history;   ///< Max force at each iteration
    std::vector<double> rms_force_history;   ///< RMS force at each iteration
    
    std::string algorithm;             ///< Algorithm used
    std::string termination_reason;    ///< Why optimization stopped
};

// ============================================================================
// Geometry Optimizer Class
// ============================================================================

/**
 * @brief Geometry optimizer for molecular structure optimization
 * 
 * Supports three optimization algorithms:
 * 
 * 1. Steepest Descent (SD):
 *    x_new = x_old - α·∇E(x_old)
 *    Simple but slow convergence (linear rate)
 * 
 * 2. Conjugate Gradient (CG):
 *    Search direction: d_k = -∇E_k + β_k·d_{k-1}
 *    β_k computed via Fletcher-Reeves or Polak-Ribière formula
 *    Better than SD (superlinear convergence)
 * 
 * 3. BFGS Quasi-Newton:
 *    Builds approximate Hessian H_k via rank-2 updates
 *    Search direction: d_k = -H_k^{-1}·∇E_k
 *    Fastest convergence (superlinear to quadratic)
 * 
 * REFERENCES:
 * - Nocedal & Wright (2006), "Numerical Optimization"
 * - Schlegel (1987), Adv. Chem. Phys. 67, 249
 */
class GeometryOptimizer {
public:
    /**
     * @brief Construct geometry optimizer
     * @param gradient_func Function to compute gradient (takes Molecule, returns GradientResult)
     * @param config Optimization configuration
     */
    GeometryOptimizer(
        std::function<GradientResult(const Molecule&)> gradient_func,
        const OptConfig& config = OptConfig()
    );
    
    /**
     * @brief Optimize molecular geometry
     * @param initial_geom Starting molecular geometry
     * @return Optimization result with final geometry and convergence info
     */
    OptResult optimize(const Molecule& initial_geom);
    
    /**
     * @brief Get current configuration
     */
    const OptConfig& config() const { return config_; }
    
    /**
     * @brief Set configuration
     */
    void set_config(const OptConfig& config) { config_ = config; }
    
private:
    // Configuration and gradient function
    OptConfig config_;
    std::function<GradientResult(const Molecule&)> gradient_func_;
    
    // Optimization state
    Molecule current_geom_;
    Eigen::VectorXd current_coords_;     ///< Flattened coordinates (3N)
    Eigen::VectorXd current_gradient_;   ///< Current gradient (3N)
    double current_energy_;
    
    // Algorithm-specific state
    Eigen::VectorXd search_direction_;   ///< Current search direction
    Eigen::MatrixXd hessian_inverse_;    ///< Approximate H^{-1} (BFGS only)
    Eigen::VectorXd prev_gradient_;      ///< Previous gradient (CG, BFGS)
    Eigen::VectorXd prev_coords_;        ///< Previous coordinates (BFGS)
    
    // Iteration history
    std::vector<double> energy_history_;
    std::vector<double> max_force_history_;
    std::vector<double> rms_force_history_;
    
    // ========================================================================
    // Core optimization methods
    // ========================================================================
    
    /**
     * @brief Perform one optimization step
     * @return true if converged
     */
    bool step();
    
    /**
     * @brief Check convergence criteria
     * @param step_vector Displacement vector from last step
     * @return true if all criteria satisfied
     */
    bool check_convergence(const Eigen::VectorXd& step_vector);
    
    /**
     * @brief Convert Eigen vector to Molecule geometry
     */
    Molecule coords_to_molecule(const Eigen::VectorXd& coords) const;
    
    /**
     * @brief Extract coordinates from Molecule as Eigen vector
     */
    Eigen::VectorXd molecule_to_coords(const Molecule& mol) const;
    
    // ========================================================================
    // Algorithm implementations
    // ========================================================================
    
    /**
     * @brief Steepest descent step
     * 
     * REFERENCE: Nocedal & Wright (2006), Section 3.1
     * Direction: d_k = -∇E_k (steepest descent direction)
     * Step: x_{k+1} = x_k + α_k·d_k where α_k from line search
     */
    void steepest_descent_step();
    
    /**
     * @brief Conjugate gradient step
     * 
     * REFERENCE: Fletcher & Reeves (1964), Comput. J. 7, 149
     * Direction: d_k = -∇E_k + β_k·d_{k-1}
     * β_k = ||∇E_k||² / ||∇E_{k-1}||² (Fletcher-Reeves)
     * β_k = max(0, ∇E_k·(∇E_k - ∇E_{k-1}) / ||∇E_{k-1}||²) (Polak-Ribière)
     */
    void conjugate_gradient_step();
    
    /**
     * @brief BFGS quasi-Newton step
     * 
     * REFERENCE: Nocedal & Wright (2006), Section 6.1
     * Update inverse Hessian approximation H_k:
     * H_{k+1} = (I - ρ_k s_k y_k^T) H_k (I - ρ_k y_k s_k^T) + ρ_k s_k s_k^T
     * where s_k = x_{k+1} - x_k, y_k = ∇E_{k+1} - ∇E_k, ρ_k = 1/(y_k^T s_k)
     */
    void bfgs_step();
    
    /**
     * @brief Initialize BFGS Hessian inverse to identity
     */
    void initialize_bfgs_hessian();
    
    /**
     * @brief Update BFGS Hessian inverse approximation
     * @param s Step vector (x_new - x_old)
     * @param y Gradient change (g_new - g_old)
     */
    void update_bfgs_hessian(const Eigen::VectorXd& s, const Eigen::VectorXd& y);
    
    // ========================================================================
    // Line search
    // ========================================================================
    
    /**
     * @brief Backtracking line search (Armijo-Goldstein condition)
     * 
     * REFERENCE: Nocedal & Wright (2006), Section 3.1, Algorithm 3.1
     * Find α such that:
     * E(x + α·d) ≤ E(x) + c1·α·∇E^T·d (sufficient decrease)
     * 
     * @param direction Search direction
     * @return Step size α
     */
    double line_search(const Eigen::VectorXd& direction);
    
    /**
     * @brief Apply trust region constraint to step
     * @param step Proposed step vector
     * @return Scaled step within trust radius
     */
    Eigen::VectorXd apply_trust_region(const Eigen::VectorXd& step);
    
    // ========================================================================
    // Utilities
    // ========================================================================
    
    /**
     * @brief Print optimization header
     */
    void print_header();
    
    /**
     * @brief Print iteration information
     * @param iter Iteration number
     * @param step_vector Displacement in this step
     */
    void print_iteration(int iter, const Eigen::VectorXd& step_vector);
    
    /**
     * @brief Print final results
     * @param result Optimization result
     */
    void print_results(const OptResult& result);
    
    /**
     * @brief Compute RMS and max values of vector
     */
    void compute_statistics(const Eigen::VectorXd& vec, double& rms, double& max_val);
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Convenience function to optimize geometry with RHF gradients
 * @param initial_geom Starting geometry
 * @param basis_name Basis set name
 * @param config Optimization configuration
 * @return Optimization result
 */
OptResult optimize_rhf(
    const Molecule& initial_geom,
    const std::string& basis_name,
    const OptConfig& config = OptConfig()
);

/**
 * @brief Convenience function to optimize geometry with UHF gradients
 */
OptResult optimize_uhf(
    const Molecule& initial_geom,
    const std::string& basis_name,
    int charge,
    int multiplicity,
    const OptConfig& config = OptConfig()
);

} // namespace gradient
} // namespace mshqc

#endif // MSHQC_OPTIMIZER_H
