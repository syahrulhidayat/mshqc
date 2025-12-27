/**
 * @file cholesky_casscf.h
 * @brief Cholesky-based CASSCF Implementation
 * Uses Cholesky vectors for integral transformation (O(N^3) storage).
 * Author: Muhamad Syahrul Hidayat
 */


#ifndef MSHQC_MCSCF_CHOLESKY_CASSCF_H
#define MSHQC_MCSCF_CHOLESKY_CASSCF_H

#include "mshqc/mcscf/casscf.h"
#include "mshqc/integrals/cholesky_direct.h" 
#include "mshqc/cholesky_uhf.h" // Agar bisa ambil dari UHF
#include <vector>

namespace mshqc {
namespace mcscf {

class CholeskyCASSCF {
public:
    // Constructor 1: Standard (Decompose from scratch)
    CholeskyCASSCF(const Molecule& mol,
                   const BasisSet& basis,
                   std::shared_ptr<IntegralEngine> integrals,
                   const ActiveSpace& active_space);

    // Constructor 2: REUSE Vectors from UHF (Efficient!)
    CholeskyCASSCF(const Molecule& mol,
                   const BasisSet& basis,
                   std::shared_ptr<IntegralEngine> integrals,
                   const ActiveSpace& active_space,
                   const std::vector<Eigen::VectorXd>& L_vectors); // Reuse vectors

    CASResult compute(const SCFResult& initial_guess);

    // Configuration
    void set_max_iterations(int max_iter) { max_iter_ = max_iter; }
    void set_energy_threshold(double thresh) { e_thresh_ = thresh; }
    void set_gradient_threshold(double thresh) { grad_thresh_ = thresh; }
    void set_ci_solver(const std::string& solver) { ci_solver_ = solver; }
    void set_cholesky_threshold(double t) { cholesky_thresh_ = t; }

private:
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    ActiveSpace active_space_;
    
    // Cholesky Vectors (AO Basis)
    // Stored as flattened vectors (N*N) to match CholeskyERI format
    std::vector<Eigen::VectorXd> L_ao_vectors_; 
    bool vectors_provided_ = false;

    // Parameters
    int max_iter_ = 50;
    double e_thresh_ = 1e-8;
    double grad_thresh_ = 1e-6;
    double cholesky_thresh_ = 1e-6;
    std::string ci_solver_ = "auto";
    std::string orbital_opt_ = "newton";

    // --- Private Methods ---

    // Decompose if vectors not provided
    void ensure_cholesky_vectors();

    // Transform AO Vectors -> MO Vectors (L_pq)
    // Returns list of matrices L_pq^J
    std::vector<Eigen::MatrixXd> transform_cholesky_to_mo(const Eigen::MatrixXd& C_mo) const;

    // Reconstruct Active Integrals (h_act, eri_act) from L_mo
    ci::CIIntegrals construct_active_integrals(
        const std::vector<Eigen::MatrixXd>& L_mo, 
        const Eigen::MatrixXd& C_mo) const;

    // Build Generalized Fock Matrix using Cholesky (O(N^3))
    Eigen::MatrixXd compute_fock_cholesky(
        const Eigen::MatrixXd& opdm,
        const std::vector<Eigen::MatrixXd>& L_mo,
        const Eigen::MatrixXd& C_mo) const;
        
    // Calculate Total Energy
    double compute_total_energy(
        const Eigen::MatrixXd& C_mo, 
        double e_ci,
        const std::vector<Eigen::MatrixXd>& L_mo) const;

    // Standard CASSCF helpers (delegated/copied)
    std::vector<ci::Determinant> generate_determinants() const;
    std::pair<double, std::vector<double>> solve_ci_problem(
        const std::vector<ci::Determinant>& dets, 
        const ci::CIIntegrals& ints);
    Eigen::MatrixXd compute_opdm(
        const std::vector<double>& ci_coeffs,
        const std::vector<ci::Determinant>& determinants) const;
    Eigen::VectorXd compute_orbital_gradient(
        const Eigen::MatrixXd& fock,
        const Eigen::MatrixXd& C_mo) const;
    Eigen::VectorXd compute_orbital_step_newton(
        const Eigen::VectorXd& gradient,
        const Eigen::VectorXd& orbital_energies) const;
    Eigen::MatrixXd apply_orbital_rotation(
        const Eigen::MatrixXd& C_mo,
        const Eigen::VectorXd& kappa,
        double damping) const;
    bool check_convergence(double delta_e, double grad_norm) const;
};

} // namespace mcscf
} // namespace mshqc

#endif