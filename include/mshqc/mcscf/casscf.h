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

#ifndef MSHQC_MCSCF_CASSCF_H
#define MSHQC_MCSCF_CASSCF_H

#include "active_space.h"
#include "mshqc/ci/determinant.h"  // Need full definition for std::vector<Determinant>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <memory>

namespace mshqc {

// Forward declarations (avoid circular dependencies)
class Molecule;
class BasisSet;
class IntegralEngine;
struct SCFResult;

namespace ci {
    struct CIIntegrals;
    // Forward declaration removed - include full header below to avoid incomplete type errors
}

namespace mcscf {

/**
 * @brief CASSCF calculation result
 */
struct CASResult {
    double e_casscf;              // Total CASSCF energy
    double e_nuclear;             // Nuclear repulsion
    int n_iterations;             // Macro-iterations
    bool converged;               // Convergence status
    
    Eigen::MatrixXd C_mo;         // Optimized MO coefficients
    Eigen::VectorXd orbital_energies;
    
    // CI wavefunction in active space
    std::vector<double> ci_coeffs;           // CI coefficients
    std::vector<ci::Determinant> determinants; // CAS determinants (for CASPT2)
    int n_determinants;                       // Number of determinants
    
    // Active space definition (for CASPT2)
    ActiveSpace active_space;
    
    // Convergence history
    std::vector<double> energy_history;
};

/**
 * @brief CASSCF solver using 2-step algorithm
 * 
 * Example usage:
 *   ActiveSpace cas = ActiveSpace::CAS(2, 2, nbf, nelec);  // CAS(2,2)
 *   CASSCF casscf(mol, basis, integrals, cas);
 *   auto result = casscf.compute(hf_result);  // Initial guess from HF
 */
class CASSCF {
public:
    // Constructor
    CASSCF(const Molecule& mol,
           const BasisSet& basis,
           std::shared_ptr<IntegralEngine> integrals,
           const ActiveSpace& active_space);
    
    /**
     * @brief Run CASSCF calculation
     * @param initial_guess Initial orbitals (usually from HF)
     * @return CASSCF result
     */
    CASResult compute(const SCFResult& initial_guess);
    
    /**
     * @brief State-averaged CASSCF (SA-CASSCF)
     * @param initial_guess Initial orbitals
     * @param nstates Number of states to average
     * @param weights State weights (should sum to 1.0)
     * @return SA-CASSCF result
     */
    CASResult compute_sa(const SCFResult& initial_guess,
                         int nstates,
                         const std::vector<double>& weights);
    
    /**
     * @brief Public wrappers to compute density matrices from CI wavefunction
     */
    Eigen::MatrixXd opdm_from_ci(const std::vector<double>& ci_coeffs,
                                 const std::vector<ci::Determinant>& ci_dets) const {
        return compute_opdm_from_ci(ci_coeffs, ci_dets);
    }
    Eigen::Tensor<double, 4> tpdm_from_ci(const std::vector<double>& ci_coeffs,
                                          const std::vector<ci::Determinant>& ci_dets) const {
        return compute_tpdm_from_ci(ci_coeffs, ci_dets);
    }
    
    // Convergence parameters
    void set_max_iterations(int max_iter) { max_iter_ = max_iter; }
    void set_energy_threshold(double thresh) { e_thresh_ = thresh; }
    void set_gradient_threshold(double thresh) { grad_thresh_ = thresh; }
    
private:
    // Molecule and basis
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    ActiveSpace active_space_;
    
    // Convergence parameters
    int max_iter_;
    double e_thresh_;
    double grad_thresh_;
    
    // Adaptive damping for orbital rotation
    // REFERENCE: Werner & Knowles (1988), Section 4
    double damping_factor_;          // Current damping (α ∈ [0,1])
    double damping_min_;              // Minimum damping (conservative)
    double damping_max_;              // Maximum damping (aggressive)
    int n_energy_increase_;           // Counter for energy increases
    
    // CI state from last iteration (for gradient calculation)
    // REFERENCE: Werner & Knowles (1988), Section 3
    std::vector<ci::Determinant> last_ci_dets_;
    std::vector<double> last_ci_coeffs_;
    Eigen::MatrixXd last_C_mo_;
    
    // =================================================================
    // TODO: Waiting for Agent 2 (CI Specialist)
    // =================================================================
    
    /**
     * @brief CI step: solve FCI in active space
     * @param C_mo Current MO coefficients
     * @return CI energy and coefficients
     * 
     * TODO: This needs Agent 2's CI solver:
     *   - Determinant generation for active space
     *   - Davidson diagonalization or dense FCI
     *   - Slater-Condon matrix elements
     * 
     * BLOCKED: Cannot implement until Agent 2 delivers:
     *   - include/mshqc/ci/determinant.h
     *   - include/mshqc/ci/ci_solver.h
     */
    std::pair<double, std::vector<double>>
        solve_ci_step(const Eigen::MatrixXd& C_mo);
    
    /**
     * @brief Generate all determinants in active space
     * 
     * TODO: Use Agent 2's determinant infrastructure
     */
    // std::vector<Determinant> generate_active_determinants();
    
    // =================================================================
    // Density Matrices from CI Wavefunction
    // =================================================================
    
    /**
     * @brief Compute one-particle density matrix (OPDM) from CI wavefunction
     * 
    * Calculates γ_pq = ⟨Ψ|E_pq|Ψ⟩ = Σ_IJ c_I c_J ⟨Φ_I|E_pq|Φ_J⟩
     * THEORY: Helgaker et al. (2000), Ch. 11, Eq. 11.6.10
     */
    Eigen::MatrixXd compute_opdm_from_ci(
        const std::vector<double>& ci_coeffs,
        const std::vector<ci::Determinant>& ci_dets) const;
    
    /**
     * @brief Compute two-particle density matrix (TPDM) from CI wavefunction
     * 
     * Calculates Γ_pqrs = ⟨Ψ|E_pq E_rs|Ψ⟩ (Helgaker et al., Eq. 11.7.5)
     */
    Eigen::Tensor<double, 4> compute_tpdm_from_ci(
        const std::vector<double>& ci_coeffs,
        const std::vector<ci::Determinant>& ci_dets) const;
    
    // =================================================================
    // Can implement now (no dependencies)
    // =================================================================
    
    /**
     * @brief Orbital step: optimize orbitals
     * @param C_mo Current MO coefficients
     * @param ci_coeffs CI coefficients from CI step
     * @return Updated MO coefficients
     * 
     * REFERENCE: Knowles & Werner (1988), Section 3
     * Update: C_new = C_old * exp(κ)
     */
    Eigen::MatrixXd optimize_orbitals(const Eigen::MatrixXd& C_mo,
                                       const std::vector<double>& ci_coeffs);
    
    /**
     * @brief Compute generalized Fock matrix from OPDM/TPDM
     * @param opdm One-particle density matrix
     * @param tpdm Two-particle density matrix
     * @param C_mo MO coefficients
     * @return Generalized Fock matrix F
     * 
     * REFERENCE: Werner & Knowles (1988), Eq. (8-10)
     */
    Eigen::MatrixXd compute_generalized_fock(
        const Eigen::MatrixXd& opdm,
        const Eigen::Tensor<double, 4>& tpdm,
        const Eigen::MatrixXd& C_mo) const;
    
    /**
     * @brief Compute orbital gradient
     * @param ci_coeffs CI coefficients
     * @return Gradient vector
     * 
     * REFERENCE: Helgaker et al. (2000), Eq. (14.3.10)
     * g_pq = 2⟨Ψ|[E_pq, H]|Ψ⟩
     */
    Eigen::VectorXd compute_orbital_gradient(const std::vector<double>& ci_coeffs);
    
    /**
     * @brief Compute orbital rotation parameters κ
     * @param gradient Orbital gradient
     * @return Rotation parameters
     * 
     * Simple version: κ = -gradient / (ε_p - ε_q)
     */
    Eigen::VectorXd compute_rotation_parameters(const Eigen::VectorXd& gradient);
    
    /**
     * @brief Apply exp(κ) rotation to orbitals
     * @param C_mo Current MO coefficients
     * @param kappa Rotation parameters
     * @return Rotated MO coefficients
     * 
     * REFERENCE: Helgaker et al. (2000), Section 14.4
     */
    Eigen::MatrixXd apply_orbital_rotation(const Eigen::MatrixXd& C_mo,
                                           const Eigen::VectorXd& kappa);
    
    /**
     * @brief Check convergence
     * @param delta_e Energy change
     * @param gradient Orbital gradient
     * @return true if converged
     */
    bool is_converged(double delta_e, const Eigen::VectorXd& gradient) const;
    
    /**
     * @brief Extract active orbital block from MO matrix
     * @param C_mo Full MO coefficients
     * @return Active orbital coefficients
     */
    Eigen::MatrixXd extract_active_orbitals(const Eigen::MatrixXd& C_mo) const;
    
    /**
     * @brief Transform integrals to active space MO basis
     * @param C_mo Current MO coefficients
     * @return CIIntegrals in active space
     * 
     * Transforms AO integrals to MO basis for active orbitals only.
     * Includes antisymmetrization for same-spin ERIs.
     */
    ci::CIIntegrals transform_integrals_to_active(const Eigen::MatrixXd& C_mo) const;
    
    /**
     * @brief Transform ALL ERIs from AO to MO basis
     * @param C_mo Current MO coefficients
     * @return Full MO-basis ERI tensor (pq|rs)
     * 
     * Transforms complete ERI tensor to MO basis for generalized Fock construction.
     * Uses efficient 4-step algorithm: O(N^5) instead of O(N^8).
     * 
     * REFERENCE: Helgaker et al. (2000), Section 9.7
     */
    Eigen::Tensor<double, 4> transform_all_mo_eris(const Eigen::MatrixXd& C_mo) const;
    
    /**
     * @brief Compute core Hartree-Fock potential (vhf_c)
     * @param mo_eris MO-basis ERIs
     * @return Core contribution to Fock matrix
     * 
     * Computes: vhf_c_pq = Σ_i [2(pq|ii) - (pi|qi)]
     * where i runs over inactive (core) orbitals.
     */
    Eigen::MatrixXd compute_core_fock(const Eigen::Tensor<double, 4>& mo_eris) const;
    
    /**
     * @brief Compute active space Hartree-Fock potential (vhf_a)
     * @param mo_eris MO-basis ERIs
     * @param opdm One-particle density matrix
     * @return Active space contribution to Fock matrix
     * 
     * Computes: vhf_a_pq = Σ_tu γ_tu [2(pq|tu) - (pt|uq)]
     * where t,u run over active orbitals.
     */
    Eigen::MatrixXd compute_active_fock(const Eigen::Tensor<double, 4>& mo_eris,
                                        const Eigen::MatrixXd& opdm) const;
};

} // namespace mcscf
} // namespace mshqc

#endif // MSHQC_MCSCF_CASSCF_H
