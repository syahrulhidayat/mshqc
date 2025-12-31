/**
 * @file cholesky_sa_casscf.h
 * @brief State-Averaged CASSCF using Cholesky Decomposition (FULL SCF VERSION)
 * 
 * FEATURES:
 * 1. Supports initializing from UNO Orbitals.
 * 2. Reuses Cholesky Vectors from UHF (Speedup).
 * 3. Implements Orbital Rotation (Newton-Raphson/Gradient) to minimize Avg Energy.
 * 4. [NEW] Stores orbital energies (Fock diagonal) for PT2/PT3 denominators.
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-12-15
 */

#ifndef MSHQC_CHOLESKY_SA_CASSCF_H
#define MSHQC_CHOLESKY_SA_CASSCF_H

#include "mshqc/mcscf/active_space.h"
#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h" 
#include <vector>
#include <memory>
#include <Eigen/Dense>

namespace mshqc {
namespace mcscf {

struct SACASConfig {
    int n_states = 1;                     
    std::vector<double> weights;          
    int max_iter = 100;
    double e_thresh = 1e-8;
    double grad_thresh = 1e-5;
    double cholesky_thresh = 1e-6;
    int print_level = 1; 

    // Orbital Optimization Parameters
    double rotation_damping = 0.5;
    double shift = 0.0;

    void set_equal_weights(int n) {
        n_states = n;
        weights.assign(n, 1.0 / n);
    }
};

struct SACASResult {
    bool converged;
    int n_iterations;
    double e_avg;                         
    std::vector<double> state_energies;   
    
    Eigen::MatrixXd C_mo;
    
    // Properties for analysis
    std::vector<Eigen::VectorXd> ci_vectors; 
    std::vector<Eigen::MatrixXd> rdm1_states;
    std::vector<Eigen::MatrixXd> rdm2_states; 
    
    // [NEW] Ab initio orbital energies from Fock diagonal
    // REFERENCE: Szabo & Ostlund (1996), Eq. (3.154)
    // epsilon_p = F_pp = <p|h|p> + sum_q D_qq * [(pq|pq) - 0.5*(pp|qq)]
    std::vector<double> orbital_energies;
};

class CholeskySACASSCF {
public:
    // Constructor 1: Standard (Calculate Cholesky internally)
    CholeskySACASSCF(const Molecule& mol,
                     const BasisSet& basis,
                     std::shared_ptr<IntegralEngine> integrals,
                     const ActiveSpace& active_space,
                     const SACASConfig& config);

    // Constructor 2: Optimized (REUSE Cholesky Vectors form UHF)
    CholeskySACASSCF(const Molecule& mol,
                     const BasisSet& basis,
                     std::shared_ptr<IntegralEngine> integrals,
                     const ActiveSpace& active_space,
                     const SACASConfig& config,
                     const std::vector<Eigen::VectorXd>& L_vectors);

    /**
     * @brief Compute SA-CASSCF using specific orbital guess (e.g., from UNO)
     * @param initial_orbitals Matrix (N_basis x N_basis) from UNO result
     */
    SACASResult compute(const Eigen::MatrixXd& initial_orbitals);

    // Overload for convenience (extracts C from SCFResult)
    SACASResult compute(const SCFResult& initial_guess);

private:
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    ActiveSpace active_space_;
    SACASConfig config_;
    
    std::vector<Eigen::VectorXd> L_ao_vectors_;
    bool vectors_provided_;

    // --- Internal Logic ---
    void ensure_cholesky_vectors();
    
    std::vector<Eigen::MatrixXd> transform_cholesky_to_mo(const Eigen::MatrixXd& C_mo) const;
    
    Eigen::MatrixXd compute_generalized_fock(
        const std::vector<Eigen::MatrixXd>& rdm1_states,
        const std::vector<Eigen::MatrixXd>& L_mo,
        const Eigen::MatrixXd& C_mo
    ) const;

    Eigen::VectorXd compute_orbital_gradient(
        const Eigen::MatrixXd& F_gen, 
        const Eigen::MatrixXd& C_mo
    ) const;
    
    Eigen::MatrixXd apply_rotation(const Eigen::MatrixXd& C, const Eigen::VectorXd& kappa) const;
};

} // namespace mcscf
} // namespace mshqc

#endif