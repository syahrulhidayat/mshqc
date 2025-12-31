/**
 * @file sa_casscf.h
 * @brief State-Averaged Complete Active Space SCF (SA-CASSCF)
 * 
 * SA-CASSCF optimizes orbitals for multiple electronic states simultaneously
 * by averaging Fock matrices and densities with specified weights.
 * 
 * Essential for:
 * - Excited state calculations
 * - Conical intersections
 * - Non-adiabatic dynamics
 * - Spectroscopy (transition properties)
 * 
 * THEORY:
 * Energy: E_avg = Σ_I w_I E_I  (w_I = state weights, Σw_I = 1)
 * Fock:   F_avg = Σ_I w_I F_I
 * Density: D_avg = Σ_I w_I D_I
 * 
 * Orbital rotation optimizes E_avg while keeping all states coupled.
 * 
 * THEORY REFERENCES:
 * 1. B.O. Roos et al., Chem. Phys. 48, 157 (1980)
 *    - Original SA-CASSCF method
 * 2. P.E.M. Siegbahn et al., Phys. Scr. 21, 323 (1980)
 *    - Multi-state CASSCF theory
 * 3. H.-J. Werner & W. Meyer, J. Chem. Phys. 74, 5794 (1981)
 *    - Efficient SA-CASSCF implementation
 * 4. P. Celani & H.-J. Werner, J. Chem. Phys. 112, 5546 (2000)
 *    - Analytical gradients for SA-CASSCF
 * 5. J. Finley et al., Chem. Phys. Lett. 288, 299 (1998)
 *    - State-specific vs state-averaged comparison
 * 6. M. Schreiber et al., J. Chem. Phys. 128, 134110 (2008)
 *    - Benchmarking excited states with SA-CASSCF
 * 7. D.R. Yarkony, Rev. Mod. Phys. 68, 985 (1996)
 *    - Non-adiabatic processes and conical intersections
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 * 
 * @note Original implementation based on theory papers.
 *       No code copied from other quantum chemistry software.
 */

#ifndef MSHQC_SA_CASSCF_H
#define MSHQC_SA_CASSCF_H

#include "mshqc/mcscf/casscf.h"
#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace mshqc {
namespace mcscf {

// ============================================================================
// SA-CASSCF Configuration
// ============================================================================

/**
 * @brief Configuration for state-averaged CASSCF
 */
struct SACASConfig1 {
    // State averaging
    int n_states = 2;                          ///< Number of states to average
    std::vector<double> state_weights;         ///< Weights for each state (Σw_I=1)
    
    // Active space (same as regular CASSCF)
    int n_active_electrons = 0;                ///< Electrons in active space
    int n_active_orbitals = 0;                 ///< Orbitals in active space
    
    // Convergence
    double energy_thresh = 1e-8;               ///< Energy convergence (Ha)
    double gradient_thresh = 1e-6;             ///< Orbital gradient threshold
    int max_iterations = 100;                  ///< Maximum macro iterations
    int max_ci_iterations = 50;                ///< Max CI iterations per state
    
    // Output
    int print_level = 1;                       ///< 0=silent, 1=normal, 2=verbose
    bool print_ci_vectors = false;             ///< Print CI coefficients
    
    /**
     * @brief Set equal weights for all states
     */
    void set_equal_weights() {
        state_weights.clear();
        double w = 1.0 / n_states;
        for (int i = 0; i < n_states; ++i) {
            state_weights.push_back(w);
        }
    }
    
    /**
     * @brief Set custom weights (must sum to 1.0)
     */
    void set_custom_weights(const std::vector<double>& weights) {
        if (weights.size() != static_cast<size_t>(n_states)) {
            throw std::runtime_error("Number of weights must equal n_states");
        }
        double sum = 0.0;
        for (double w : weights) sum += w;
        if (std::abs(sum - 1.0) > 1e-10) {
            throw std::runtime_error("State weights must sum to 1.0");
        }
        state_weights = weights;
    }
};

// ============================================================================
// SA-CASSCF Result
// ============================================================================

/**
 * @brief Result from SA-CASSCF calculation
 */
struct SACASResult1 {
    bool converged = false;                    ///< Did calculation converge?
    int n_iterations = 0;                      ///< Number of iterations
    
    // Energies
    double energy_averaged = 0.0;              ///< State-averaged energy
    std::vector<double> state_energies;        ///< Individual state energies
    
    // Orbitals
    Eigen::MatrixXd mo_coefficients;           ///< MO coefficients (optimized for avg)
    Eigen::VectorXd orbital_energies;          ///< Orbital energies
    
    // CI for each state
    std::vector<Eigen::VectorXd> ci_vectors;   ///< CI vectors for each state
    std::vector<Eigen::MatrixXd> rdm1_states;  ///< 1-RDM for each state
    std::vector<Eigen::MatrixXd> rdm2_states;  ///< 2-RDM for each state
    
    // State-averaged quantities
    Eigen::MatrixXd rdm1_averaged;             ///< State-averaged 1-RDM
    Eigen::MatrixXd rdm2_averaged;             ///< State-averaged 2-RDM
    
    // Configuration
    int n_states;
    std::vector<double> state_weights;
    int n_active_electrons;
    int n_active_orbitals;
};

// ============================================================================
// Transition Properties
// ============================================================================

/**
 * @brief Transition properties between two states
 * 
 * For spectroscopy and excited state analysis.
 */
struct TransitionProperties {
    int state_i;                               ///< Initial state index
    int state_j;                               ///< Final state index
    
    double energy_diff;                        ///< E_j - E_i (Ha)
    double wavelength;                         ///< Transition wavelength (nm)
    double frequency;                          ///< Transition frequency (cm⁻¹)
    
    Eigen::Vector3d transition_dipole;         ///< μ_ij = ⟨i|μ|j⟩ (au)
    double dipole_strength;                    ///< |μ_ij|² (au²)
    double oscillator_strength;                ///< f_ij (dimensionless)
    
    double einstein_A;                         ///< A_ji (spontaneous emission, s⁻¹)
    double einstein_B_absorption;              ///< B_ij (absorption)
    double einstein_B_emission;                ///< B_ji (stimulated emission)
    
    /**
     * @brief Compute derived quantities from energy and dipole
     */
    void compute_derived_quantities();
};

/**
 * @brief State-specific properties (for one state)
 */
struct StateProperties {
    int state_index;                           ///< Which state
    double energy;                             ///< State energy (Ha)
    
    Eigen::Vector3d dipole_moment;             ///< μ = ⟨ψ|μ|ψ⟩ (Debye)
    Eigen::Matrix3d quadrupole_moment;         ///< Q (au)
    
    Eigen::MatrixXd natural_orbitals;          ///< Natural orbitals from 1-RDM
    Eigen::VectorXd occupation_numbers;        ///< Occupation numbers
    
    double s_squared;                          ///< ⟨S²⟩ spin contamination
};

// ============================================================================
// SA-CASSCF Solver
// ============================================================================

/**
 * @brief State-Averaged Complete Active Space SCF
 * 
 * Simultaneously optimizes orbitals for multiple electronic states
 * by minimizing the state-averaged energy:
 * 
 * E_avg = Σ_I w_I E_I
 * 
 * Algorithm:
 * 1. Initialize orbitals (from RHF or guess)
 * 2. For each state I:
 *    a. Diagonalize CI Hamiltonian → E_I, |ψ_I⟩
 *    b. Compute state-specific 1-RDM and 2-RDM
 * 3. Form state-averaged densities: D_avg = Σ_I w_I D_I
 * 4. Build state-averaged Fock matrix: F_avg = Σ_I w_I F_I
 * 5. Optimize orbitals via orbital rotation
 * 6. Repeat until convergence
 * 
 * REFERENCE: Werner & Meyer (1981), J. Chem. Phys. 74, 5794
 */
class SACASSCF {
public:
    /**
     * @brief Construct SA-CASSCF calculator
     * @param mol Molecule
     * @param basis Basis set
     * @param integrals Integral engine
     * @param config SA-CASSCF configuration
     */
    SACASSCF(
        const Molecule& mol,
        const BasisSet& basis,
        std::shared_ptr<IntegralEngine> integrals,
        const SACASConfig1& config
    );
    
    /**
     * @brief Run SA-CASSCF calculation
     * @return SA-CASSCF result with all states
     */
    SACASResult1 compute();
    
    /**
     * @brief Compute transition properties between two states
     * @param result SA-CASSCF result
     * @param state_i Initial state index
     * @param state_j Final state index
     * @return Transition properties
     */
    TransitionProperties compute_transition_properties(
        const SACASResult1& result,
        int state_i,
        int state_j
    );
    
    /**
     * @brief Compute properties for specific state
     * @param result SA-CASSCF result
     * @param state_idx State index
     * @return State properties
     */
    StateProperties compute_state_properties(
        const SACASResult1& result,
        int state_idx
    );
    
    /**
     * @brief Compute all transition properties (full matrix)
     * @return Matrix of transition properties
     */
    std::vector<std::vector<TransitionProperties>> compute_all_transitions(
        const SACASResult1& result
    );
    
private:
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    SACASConfig1 config_;
    
    size_t nbasis_;
    int n_elec_;
    
    // Current orbitals and integrals
    Eigen::MatrixXd C_;                        ///< MO coefficients
    Eigen::MatrixXd H_core_;                   ///< Core Hamiltonian
    
    // Per-state quantities
    std::vector<double> state_energies_;
    std::vector<Eigen::VectorXd> ci_vectors_;
    std::vector<Eigen::MatrixXd> rdm1_states_;
    std::vector<Eigen::MatrixXd> rdm2_states_;
    
    // State-averaged quantities
    Eigen::MatrixXd rdm1_avg_;
    Eigen::MatrixXd rdm2_avg_;
    Eigen::MatrixXd fock_avg_;
    
    /**
     * @brief Initialize orbitals (from RHF)
     */
    void initialize_orbitals();
    
    /**
     * @brief Solve CI problem for all states
     */
    void solve_ci_all_states();
    
    /**
     * @brief Form state-averaged density matrices
     */
    void form_averaged_densities();
    
    /**
     * @brief Build state-averaged Fock matrix
     */
    void build_averaged_fock();
    
    /**
     * @brief Optimize orbitals via orbital rotation
     */
    void optimize_orbitals();
    
    /**
     * @brief Check convergence
     */
    bool check_convergence(double energy_change, double gradient_norm);
    
    /**
     * @brief Print iteration info
     */
    void print_iteration(int iter, double energy_avg);
    
    /**
     * @brief Print final results
     */
    void print_results(const SACASResult1& result);
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Compute transition dipole moment between two states
 * 
 * μ_IJ = ⟨ψ_I|μ|ψ_J⟩ = Σ_pq γ_pq^IJ μ_pq
 * 
 * where γ^IJ is transition 1-RDM
 * 
 * REFERENCE: Roos et al. (1980), Chem. Phys. 48, 157
 */
Eigen::Vector3d compute_transition_dipole(
    const Eigen::VectorXd& ci_i,
    const Eigen::VectorXd& ci_j,
    const Eigen::MatrixXd& mo_coefficients,
    const std::vector<Eigen::MatrixXd>& dipole_integrals
);

/**
 * @brief Compute oscillator strength from transition dipole
 * 
 * f_ij = (2/3) ΔE_ij |μ_ij|²
 * 
 * REFERENCE: Helgaker et al. (2000), Eq. (14.5.4)
 */
double compute_oscillator_strength(
    double energy_diff,
    const Eigen::Vector3d& transition_dipole
);

/**
 * @brief Compute Einstein A coefficient (spontaneous emission)
 * 
 * A_ji = (4 α³ ω³)/(3) |μ_ij|²
 * 
 * where α = fine structure constant, ω = transition frequency
 * 
 * REFERENCE: Cohen-Tannoudji "Quantum Mechanics" Vol. 2
 */
double compute_einstein_A(
    double energy_diff,
    double dipole_strength
);

/**
 * @brief Compute natural orbitals from 1-RDM
 * 
 * Diagonalize 1-RDM: D = U n U^†
 * Natural orbitals are columns of U
 * Occupation numbers are diagonal elements of n
 */
std::pair<Eigen::MatrixXd, Eigen::VectorXd> compute_natural_orbitals(
    const Eigen::MatrixXd& rdm1
);

/**
 * @brief Print spectrum (all transitions)
 */
void print_spectrum(
    const std::vector<std::vector<TransitionProperties>>& transitions,
    double intensity_threshold = 1e-4
);

} // namespace mcscf
} // namespace mshqc

#endif // MSHQC_SA_CASSCF_H
