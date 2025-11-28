/**
 * @file multi_root_ci.h
 * @brief Multi-root CI solver wrapper for SA-CASSCF
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 * 
 * THEORY:
 * Solves CI eigenvalue problem for multiple electronic states:
 *   H_CI |ψ_I⟩ = E_I |ψ_I⟩  for I = 0, 1, ..., N-1
 * 
 * Uses Agent 2's FCI module with Davidson diagonalization.
 * Extracts active space from full MO space and computes RDMs.
 * 
 * REFERENCES:
 * - Davidson (1975), J. Comput. Phys. 17, 87
 *   "The iterative calculation of a few of the lowest eigenvalues"
 * - Knowles & Handy (1984), Chem. Phys. Lett. 111, 315
 *   "A new determinant-based full configuration interaction method"
 * - Helgaker et al. (2000), Ch. 11
 *   "Configuration Interaction theory and RDMs"
 * - Werner & Meyer (1981), J. Chem. Phys. 74, 5794
 *   "A quadratically convergent MCSCF method" (SA-CASSCF)
 * 
 * ORIGINALITY:
 * This wrapper integrates Agent 2's FCI implementation with SA-CASSCF.
 * RDM computation algorithms derived from published theory.
 */

#ifndef MSHQC_MCSCF_MULTI_ROOT_CI_H
#define MSHQC_MCSCF_MULTI_ROOT_CI_H

#include "mshqc/ci/fci.h"
#include "mshqc/ci/davidson.h"
#include "mshqc/ci/determinant.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <memory>

namespace mshqc {
namespace mcscf {

/**
 * @brief Result for single CI state
 */
struct CIState {
    double energy;                          ///< State energy (Ha)
    Eigen::VectorXd ci_vector;              ///< CI coefficients
    std::vector<ci::Determinant> determinants; ///< Determinant basis
    Eigen::MatrixXd rdm1;                   ///< 1-RDM in active space
    Eigen::MatrixXd rdm2;                   ///< 2-RDM in active space (flattened)
    bool converged;                         ///< Convergence flag
    int iterations;                         ///< Davidson iterations
};

/**
 * @brief Multi-root CI solver for SA-CASSCF
 * 
 * Wraps Agent 2's FCI solver for multiple electronic states.
 * Handles active space extraction and RDM computation.
 * 
 * USAGE:
 * ```cpp
 * MultiRootCI ci_solver(n_states, n_active_orb, n_active_elec);
 * auto states = ci_solver.solve(h_mo, eri_mo, C_active);
 * 
 * for (auto& state : states) {
 *     std::cout << "E = " << state.energy << "\n";
 *     // Use state.rdm1, state.rdm2 for SA-CASSCF
 * }
 * ```
 */
class MultiRootCI {
public:
    /**
     * @brief Constructor
     * 
     * @param n_states Number of electronic states to compute
     * @param n_active_orb Number of active space orbitals
     * @param n_active_elec Number of active space electrons
     */
    MultiRootCI(int n_states, int n_active_orb, int n_active_elec);
    
    /**
     * @brief Solve for multiple states
     * 
     * Algorithm:
     * 1. Extract active space integrals from full MO space
     * 2. Determine alpha/beta electrons from multiplicity
     * 3. Call FCI with n_roots = n_states
     * 4. Compute RDMs for each state
     * 
     * @param h_mo Core Hamiltonian in full MO basis
     * @param eri_mo ERIs in full MO basis (nbasis⁴)
     * @param C_mo MO coefficients for orbital indexing
     * @param n_inactive Number of inactive (frozen core) orbitals
     * @return Vector of CIState (one per state)
     */
    std::vector<CIState> solve(
        const Eigen::MatrixXd& h_mo,
        const Eigen::Tensor<double, 4>& eri_mo,
        const Eigen::MatrixXd& C_mo,
        int n_inactive
    );
    
    /**
     * @brief Set spin multiplicity (2S+1)
     * 
     * Used to determine n_alpha and n_beta from n_electrons.
     * Default: multiplicity = 1 (singlet, closed-shell)
     * 
     * @param multiplicity Spin multiplicity (1=singlet, 2=doublet, 3=triplet, ...)
     */
    void set_multiplicity(int multiplicity);
    
    /**
     * @brief Get estimated number of determinants
     * 
     * Uses binomial coefficient: N_det = C(n_orb, n_alpha) × C(n_orb, n_beta)
     * 
     * @return Estimated determinant count
     */
    size_t estimate_n_determinants() const;
    
private:
    int n_states_;        ///< Number of states to compute
    int n_active_orb_;    ///< Active space orbitals
    int n_active_elec_;   ///< Active space electrons
    int multiplicity_;    ///< Spin multiplicity (2S+1)
    int n_alpha_;         ///< Alpha electrons in active space
    int n_beta_;          ///< Beta electrons in active space
    
    /**
     * @brief Extract active space block from MO integrals
     * 
     * @param h_mo Full MO core Hamiltonian
     * @param start First active orbital index
     * @param size Number of active orbitals
     * @return h_active (n_active × n_active)
     */
    Eigen::MatrixXd extract_active_h(
        const Eigen::MatrixXd& h_mo,
        int start,
        int size
    );
    
    /**
     * @brief Extract active space ERIs
     * 
     * @param eri_mo Full MO ERIs (n × n × n × n)
     * @param start First active orbital index
     * @param size Number of active orbitals
     * @return eri_active (size⁴)
     */
    Eigen::Tensor<double, 4> extract_active_eri(
        const Eigen::Tensor<double, 4>& eri_mo,
        int start,
        int size
    );
    
    /**
     * @brief Compute 1-RDM from CI vector
     * 
     * THEORY:
     * 1-RDM element: γ_pq = ⟨ψ|a_p† a_q|ψ⟩
     * 
     * Algorithm:
     * For all determinant pairs I, J:
     *   if J = a_p† a_q |I⟩:  (single excitation p→q)
     *     γ_pq += c_I * c_J * phase
     * 
     * REFERENCE: Helgaker et al. (2000), Eq. (11.7.1)
     * 
     * @param ci_vector CI coefficients
     * @param dets Determinant basis
     * @param n_orb Number of orbitals
     * @return 1-RDM (n_orb × n_orb)
     */
    Eigen::MatrixXd compute_rdm1(
        const Eigen::VectorXd& ci_vector,
        const std::vector<ci::Determinant>& dets,
        int n_orb
    );
    
    /**
     * @brief Compute 2-RDM from CI vector
     * 
     * THEORY:
     * 2-RDM element: Γ_pqrs = ⟨ψ|a_p† a_q† a_s a_r|ψ⟩
     * 
     * Algorithm:
     * For all determinant pairs I, J:
     *   if J = a_p† a_q† a_s a_r |I⟩:  (double excitation rs→pq)
     *     Γ_pqrs += c_I * c_J * phase
     * 
     * REFERENCE: Helgaker et al. (2000), Eq. (11.7.2)
     * 
     * @param ci_vector CI coefficients
     * @param dets Determinant basis
     * @param n_orb Number of orbitals
     * @return 2-RDM flattened (n_orb² × n_orb²)
     */
    Eigen::MatrixXd compute_rdm2(
        const Eigen::VectorXd& ci_vector,
        const std::vector<ci::Determinant>& dets,
        int n_orb
    );
    
    /**
     * @brief Determine n_alpha, n_beta from total electrons and multiplicity
     * 
     * THEORY:
     * Multiplicity = 2S + 1, where S = (n_alpha - n_beta) / 2
     * n_alpha + n_beta = n_elec
     * 
     * Solving: n_alpha = (n_elec + 2S) / 2
     *          n_beta  = (n_elec - 2S) / 2
     */
    void determine_spin_config();
    
    /**
     * @brief Compute excitation level between two determinants
     * 
     * @param det_i First determinant
     * @param det_j Second determinant
     * @return Excitation level (0, 1, 2, ...)
     */
    int excitation_level(
        const ci::Determinant& det_i,
        const ci::Determinant& det_j
    );
    
    /**
     * @brief Get orbital indices for single excitation I→J
     * 
     * @param det_i Initial determinant
     * @param det_j Final determinant
     * @param p_alpha Destroyed alpha orbital (output)
     * @param q_alpha Created alpha orbital (output)
     * @param p_beta Destroyed beta orbital (output)
     * @param q_beta Created beta orbital (output)
     * @return Phase factor (±1)
     */
    int get_single_excitation_indices(
        const ci::Determinant& det_i,
        const ci::Determinant& det_j,
        int& p_alpha, int& q_alpha,
        int& p_beta, int& q_beta
    );
};

} // namespace mcscf
} // namespace mshqc

#endif // MSHQC_MCSCF_MULTI_ROOT_CI_H
