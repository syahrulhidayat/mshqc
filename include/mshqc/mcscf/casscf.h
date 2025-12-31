/**
 * @file casscf.h
 * @brief CASSCF (Complete Active Space SCF) implementation
 * 
 * Implements 2-step CASSCF algorithm:
 *   1. CI step: FCI in active space (uses Agent 2's CI solver)
 *   2. Orbital step: optimize orbitals with exp(κ) parametrization
 * 
 * THEORY REFERENCES:
 *   - B. O. Roos et al., Chem. Phys. 48, 157 (1980)
 *   - P. J. Knowles & H.-J. Werner, Chem. Phys. Lett. 145, 514 (1988)
 *   - T. Helgaker et al., "Molecular Electronic Structure Theory" (2000), Ch. 14
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 * @license MIT
 * 
 * @note Original implementation from published CASSCF theory.
 *       No code copied from existing quantum chemistry software.
 */

/**
 * @file casscf.h
 * @brief CASSCF (Complete Active Space SCF) implementation
 */

#ifndef MSHQC_MCSCF_CASSCF_H
#define MSHQC_MCSCF_CASSCF_H

#include "mshqc/mcscf/active_space.h"
#include "mshqc/ci/determinant.h"
#include "mshqc/ci/ci_utils.h" // Pastikan struct CIIntegrals ada di sini
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <memory>
#include <string>

namespace mshqc {

class Molecule;
class BasisSet;
class IntegralEngine;
struct SCFResult;

namespace mcscf {

struct CASResult {
    double e_casscf;
    double e_nuclear;
    double e_ci;
    int n_iterations;
    bool converged;
    
    Eigen::MatrixXd C_mo;
    Eigen::VectorXd orbital_energies;
    
    std::vector<double> ci_coeffs;
    std::vector<ci::Determinant> determinants;
    int n_determinants;
    
    ActiveSpace active_space;
    std::vector<double> energy_history;
};

class CASSCF {
public:
    CASSCF(const Molecule& mol,
           const BasisSet& basis,
           std::shared_ptr<IntegralEngine> integrals,
           const ActiveSpace& active_space);
    
    CASResult compute(const SCFResult& initial_guess);
    
    // Configuration
    void set_max_iterations(int max_iter) { max_iter_ = max_iter; }
    void set_energy_threshold(double thresh) { e_thresh_ = thresh; }
    void set_gradient_threshold(double thresh) { grad_thresh_ = thresh; }
    void set_convergence_mode(const std::string& mode) { conv_mode_ = mode; }
    void set_orbital_optimizer(const std::string& opt) { orbital_opt_ = opt; }
    void set_ci_solver(const std::string& solver) { ci_solver_ = solver; }
    void set_state_averaging(int nstates, const std::vector<double>& weights);

private:
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    ActiveSpace active_space_;
    
    // Parameters
    int max_iter_;
    double e_thresh_;
    double grad_thresh_;
    std::string conv_mode_;
    std::string orbital_opt_;
    std::string ci_solver_;
    
    // State averaging
    bool use_state_avg_;
    int n_states_;
    std::vector<double> state_weights_;
    
    // Cached dimensions
    int nbf_;
    int nmo_;
    int n_inactive_;
    int n_active_;
    int n_virtual_;
    int n_elec_active_;

    // Private Implementation Methods
    std::vector<ci::Determinant> generate_determinants() const;
    
    std::pair<double, std::vector<double>> 
    solve_ci_problem(const Eigen::MatrixXd& C_mo, int state_idx = 0);
    
    std::vector<std::pair<double, std::vector<double>>>
    solve_state_averaged_ci(const Eigen::MatrixXd& C_mo);
    
    ci::CIIntegrals transform_integrals_to_active_space(
        const Eigen::MatrixXd& C_mo) const;
        
    Eigen::MatrixXd compute_core_fock_contribution(
        const Eigen::MatrixXd& h_active,
        const Eigen::MatrixXd& C_mo) const;
        
    Eigen::Tensor<double, 4> transform_full_mo_eris(
        const Eigen::MatrixXd& C_mo) const;
        
    Eigen::MatrixXd compute_opdm(
        const std::vector<double>& ci_coeffs,
        const std::vector<ci::Determinant>& determinants) const;
        
    Eigen::Tensor<double, 4> compute_tpdm(
        const std::vector<double>& ci_coeffs,
        const std::vector<ci::Determinant>& determinants) const;
        
    std::pair<Eigen::MatrixXd, Eigen::Tensor<double, 4>>
    compute_state_averaged_density_matrices(
        const std::vector<Eigen::MatrixXd>& state_opdms,
        const std::vector<Eigen::Tensor<double, 4>>& state_tpdms) const;

    Eigen::MatrixXd compute_generalized_fock(
        const Eigen::MatrixXd& opdm,
        const Eigen::Tensor<double, 4>& tpdm,
        const Eigen::MatrixXd& C_mo) const;
    
    Eigen::VectorXd compute_orbital_gradient(
        const Eigen::MatrixXd& fock,
        const Eigen::MatrixXd& C_mo) const;
    
    Eigen::MatrixXd apply_orbital_rotation(
        const Eigen::MatrixXd& C_mo,
        const Eigen::VectorXd& kappa,
        double damping = 1.0) const;
    
    Eigen::VectorXd compute_orbital_step_newton(
        const Eigen::VectorXd& gradient,
        const Eigen::VectorXd& orbital_energies) const;
    
    double compute_total_energy(
        const Eigen::MatrixXd& C_mo,
        double e_ci) const;
    
    bool check_convergence(double delta_e, double grad_norm) const;
};

} // namespace mcscf
} // namespace mshqc

#endif