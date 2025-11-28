/**
 * @file cipsi.h
 * @brief CIPSI - Configuration Interaction with Perturbative Selection Iteratively
 * 
 * Theory References (Ab Initio - No Code Copying):
 * - Original CIPSI: B. Huron et al., J. Chem. Phys. **58**, 5745 (1973)
 * - Epstein-Nesbet PT: R. K. Nesbet, Phys. Rev. **109**, 1632 (1958)
 * - Modern CIPSI: E. Giner et al., J. Chem. Phys. **143**, 124305 (2015)
 * - Quantum Package: A. Scemama et al., J. Comp. Chem. **37**, 1866 (2016)
 * - Stochastic CIPSI: Y. Garniron et al., J. Chem. Phys. **149**, 064103 (2018)
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 * 
 * @note Original implementation from theory papers (Huron 1973, Giner 2015).
 *       CIPSI iteratively selects important determinants based on perturbative
 *       energy contribution, dramatically reducing CI space (10³-10⁶× reduction).
 *       This enables large active spaces (N > 20 orbitals) that are intractable
 *       with full CI.
 * 
 * Key Concepts:
 * - Variational space: Selected determinants (internal space)
 * - External space: All other determinants
 * - Selection criterion: ΔE_PT2 = |⟨Ψ_var|H|Φ_ext⟩|² / (E_var - E_ext)
 * - Iterative: Add most important external dets → converge to FCI limit
 * 
 * Algorithm:
 * 1. Start with small variational space (HF + singles)
 * 2. Compute variational energy E_var
 * 3. Screen all external determinants
 * 4. Select dets with largest |ΔE_PT2|
 * 5. Add to variational space
 * 6. Repeat until ΔE_PT2_total < threshold
 * 
 * Advantages over full CI:
 * - 10³-10⁶× fewer determinants
 * - Converges to FCI limit systematically
 * - Size-consistent (unlike truncated CI)
 * - Black-box (no active space choice needed)
 * 
 * @copyright MIT License
 */

#ifndef MSHQC_CI_CIPSI_H
#define MSHQC_CI_CIPSI_H

#include "mshqc/ci/determinant.h"
#include "mshqc/ci/slater_condon.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <unordered_set>
#include <string>

namespace mshqc {
namespace ci {

/**
 * @brief Result structure for CIPSI calculation
 */
struct CIPSIResult {
    // Energies
    double e_var;           ///< Variational energy (selected space)
    double e_pt2;           ///< 2nd-order perturbative correction
    double e_total;         ///< E_var + E_PT2 (estimate of FCI)
    
    // Wavefunction
    std::vector<Determinant> variational_space;  ///< Selected determinants
    Eigen::VectorXd coefficients;                ///< CI coefficients
    
    // Convergence info
    int n_iterations;       ///< Number of selection iterations
    int n_selected;         ///< Total determinants selected
    int n_external;         ///< External determinants screened (last iter)
    double pt2_norm;        ///< ||E_PT2|| (sum of all ΔE_PT2)
    
    // Timing
    double time_diag;       ///< Diagonalization time
    double time_pt2;        ///< PT2 screening time
    double time_total;      ///< Total computation time
    
    // Diagnostics
    bool converged;         ///< True if E_PT2 < threshold
    std::string conv_reason;///< Convergence/divergence reason
};

/**
 * @brief Configuration for CIPSI algorithm
 */
struct CIPSIConfig {
    // Selection parameters
    double e_pt2_threshold = 1.0e-4;    ///< Stop when E_PT2 < this (Ha)
    int max_determinants = 100000;       ///< Maximum variational space size
    int max_iterations = 50;             ///< Maximum selection iterations
    int n_select_per_iter = 1000;        ///< Determinants added per iteration
    
    // PT2 screening
    double pt2_selection_threshold = 1.0e-6;  ///< Min |ΔE_PT2| to add det
    bool use_epstein_nesbet = true;      ///< EN vs Møller-Plesset denominator
    
    // Initial space
    bool start_from_hf = true;           ///< Start with HF determinant
    bool include_singles = true;         ///< Add all singles initially
    bool include_doubles = false;        ///< Add all doubles initially (expensive!)
    
    // Excitation generation
    int max_excitation_level = 2;        ///< Max excitation rank (singles, doubles, ...)
    bool use_symmetry = false;           ///< Use spatial symmetry (future)
    
    // Performance
    int n_threads = 1;                   ///< OpenMP threads (future)
    bool verbose = true;                 ///< Print iteration info
};

/**
 * @brief CIPSI - Configuration Interaction with Perturbative Selection
 * 
 * Theory: Iterative algorithm that builds variational space by selecting
 *         determinants with largest perturbative energy contributions.
 * 
 * Algorithm (Huron et al. 1973):
 * 1. Initialize variational space Ψ_var = {HF, singles}
 * 2. Diagonalize H in variational space → E_var, |Ψ_var⟩
 * 3. Screen external determinants |Φ_ext⟩:
 *    ΔE_PT2(Φ) = |⟨Ψ_var|H|Φ_ext⟩|² / (E_var - ⟨Φ_ext|H|Φ_ext⟩)
 * 4. Select determinants with largest |ΔE_PT2|
 * 5. Add to variational space
 * 6. If E_PT2_total < threshold: converged, else goto 2
 * 
 * Key Features:
 * - Variational: E_var ≥ E_FCI (upper bound)
 * - PT2 correction: E_var + E_PT2 ≈ E_FCI (accurate estimate)
 * - Size-consistent: Energy scales linearly with system size
 * - Systematic: Converges to FCI as more dets added
 * 
 * Reference: Huron et al., J. Chem. Phys. 58, 5745 (1973)
 */
class CIPSI {
public:
    /**
     * @brief Constructor
     * 
     * @param ints CI integrals (includes alpha/beta separation for UHF)
     * @param n_orb Number of spatial orbitals
     * @param n_alpha Number of alpha electrons
     * @param n_beta Number of beta electrons
     * @param config CIPSI configuration parameters
     */
    CIPSI(const CIIntegrals& ints,
          int n_orb, int n_alpha, int n_beta,
          const CIPSIConfig& config = CIPSIConfig());
    
    /**
     * @brief Run CIPSI calculation
     * 
     * Algorithm:
     * 1. Initialize variational space (HF + singles)
     * 2. Iterate:
     *    a. Diagonalize H_var → E_var, Ψ_var
     *    b. Screen all external determinants
     *    c. Compute ΔE_PT2 for each external det
     *    d. Select n_select largest ΔE_PT2
     *    e. Add to variational space
     *    f. Check convergence: E_PT2_total < threshold
     * 3. Return result
     * 
     * @return CIPSI result with energies and wavefunction
     * 
     * Theory: Huron et al., J. Chem. Phys. 58, 5745 (1973)
     */
    CIPSIResult compute();
    
    /**
     * @brief Get current variational space
     */
    const std::vector<Determinant>& get_variational_space() const {
        return variational_space_;
    }
    
    /**
     * @brief Get current coefficients
     */
    const Eigen::VectorXd& get_coefficients() const {
        return coefficients_;
    }

private:
    // System parameters
    const CIIntegrals& ints_;
    int n_orb_;
    int n_alpha_;
    int n_beta_;
    CIPSIConfig config_;
    
    // Variational space
    std::vector<Determinant> variational_space_;
    Eigen::VectorXd coefficients_;
    double e_var_;
    
    // Helper: Initialize variational space
    void initialize_variational_space();
    
    // Helper: Build Hamiltonian matrix in variational space
    Eigen::MatrixXd build_hamiltonian_variational();
    
    // Helper: Diagonalize Hamiltonian → lowest eigenvalue + eigenvector
    void diagonalize_variational();
    
    // Helper: Generate external determinants (on-the-fly)
    std::vector<Determinant> generate_external_determinants(int max_count = 10000);
    
    // Helper: Compute PT2 contribution for one external determinant
    // Theory: Epstein-Nesbet or Møller-Plesset denominator
    // ΔE_PT2 = |⟨Ψ_var|H|Φ_ext⟩|² / (E_var - ⟨Φ_ext|H|Φ_ext⟩)
    double compute_pt2_contribution(const Determinant& det_ext);
    
    // Helper: Select determinants with largest PT2 contributions
    std::vector<Determinant> select_important_determinants(
        const std::vector<Determinant>& external_dets,
        const std::vector<double>& pt2_contribs,
        int n_select
    );
    
    // Helper: Check if determinant already in variational space
    bool is_in_variational_space(const Determinant& det) const;
    
    // Helper: Compute matrix element between wavefunction and determinant
    // ⟨Ψ_var|H|Φ_ext⟩ = Σ_i c_i ⟨Φ_i|H|Φ_ext⟩
    double compute_hamiltonian_element_wfn_det(const Determinant& det_ext);
    
    // Helper: Print iteration info
    void print_iteration(int iter, int n_var, int n_ext, 
                        double e_var, double e_pt2, double pt2_norm);
};

/**
 * @brief Helper: Generate all single excitations from a determinant
 * 
 * Theory: Single excitation |Φ_i^a⟩ = a_a† a_i |Φ_0⟩
 * 
 * @param det Reference determinant
 * @param n_orb Number of orbitals
 * @return Vector of single excitation determinants
 */
std::vector<Determinant> generate_singles(const Determinant& det, int n_orb);

/**
 * @brief Helper: Generate all double excitations from a determinant
 * 
 * Theory: Double excitation |Φ_ij^ab⟩ = a_a† a_b† a_j a_i |Φ_0⟩
 * 
 * @param det Reference determinant
 * @param n_orb Number of orbitals
 * @return Vector of double excitation determinants
 */
std::vector<Determinant> generate_doubles(const Determinant& det, int n_orb);

/**
 * @brief Helper: Generate connected excitations from variational space
 * 
 * Connected excitations: Dets that differ from any variational det by 1-2 excitations
 * This is more efficient than generating all possible determinants.
 * 
 * @param variational_space Current selected determinants
 * @param n_orb Number of orbitals
 * @param max_excitation_level Max excitation rank (1=singles, 2=doubles, ...)
 * @return Vector of connected determinants (not in variational space)
 */
std::vector<Determinant> generate_connected_excitations(
    const std::vector<Determinant>& variational_space,
    int n_orb,
    int max_excitation_level = 2
);

} // namespace ci
} // namespace mshqc

#endif // MSHQC_CI_CIPSI_H
