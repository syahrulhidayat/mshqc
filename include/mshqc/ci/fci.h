/**
 * @file fci.h
 * @brief Full Configuration Interaction (exact solution)
 * 
 * THEORY REFERENCES:
 *   - Knowles & Handy (1984), Chem. Phys. Lett. 111, 315
 *   - Olsen et al. (1988), J. Chem. Phys. 89, 2185
 *   - Helgaker et al. (2000), Ch. 11
 * 
 * FCI includes ALL possible determinants in the space.
 * It provides the EXACT solution within the given basis set.
 * 
 * Scaling: N_dets = C(n_orb, n_elec) ~ exponential
 * Only feasible for small systems (< 20 orbitals, < 10 electrons)
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 */

#ifndef MSHQC_CI_FCI_H
#define MSHQC_CI_FCI_H

#include "mshqc/ci/determinant.h"
#include "mshqc/ci/slater_condon.h"
#include "mshqc/ci/davidson.h"
#include <Eigen/Dense>
#include <vector>

namespace mshqc {
namespace ci {

/**
 * FCI result structure
 */
struct FCIResult {
    double e_fci;              // FCI ground state energy (EXACT in basis)
    double e_hf;               // HF reference energy
    double e_corr;             // Correlation energy (total)
    int n_determinants;        // Total number of determinants
    bool converged;            // Solver convergence
    int iterations;            // Davidson iterations (if used)
    
    // Wavefunction data
    std::vector<Determinant> determinants;
    Eigen::VectorXd coefficients;
    
    // Analysis
    double hf_weight;          // |c_0|^2 (HF contribution)
    double singles_weight;     // Singles contribution
    double doubles_weight;     // Doubles contribution
    double higher_weight;      // Triples, quadruples, etc.
    
    // Excited states (if requested)
    std::vector<double> excited_energies;
    std::vector<Eigen::VectorXd> excited_states;
};

/**
 * Full Configuration Interaction solver
 * 
 * Generates ALL possible determinants and solves exactly.
 * This is the gold standard for benchmarking other methods.
 * 
 * USAGE:
 *   FCI fci(ints, n_orb, n_alpha, n_beta);
 *   auto result = fci.compute();
 * 
 * LIMITATIONS:
 *   - Exponential scaling: only for small systems
 *   - H2 (2e, 2orb): 6 determinants
 *   - He (2e, 5orb): 15 determinants
 *   - H2O (10e, 24orb): ~10^12 determinants (IMPOSSIBLE!)
 * 
 * VALIDATION:
 *   FCI is exact, so use to validate:
 *   - CISD correlation recovery (~95%)
 *   - CCSD should match FCI very closely
 *   - CASSCF with full active space = FCI
 */
class FCI {
public:
    /**
     * Constructor
     * 
     * @param ints MO integrals (must be complete basis, not just active space)
     * @param n_orbitals Total number of spatial orbitals
     * @param n_alpha Number of alpha electrons
     * @param n_beta Number of beta electrons
     * @param n_roots Number of states to compute (1 = ground state only)
     */
    FCI(const CIIntegrals& ints,
        int n_orbitals,
        int n_alpha,
        int n_beta,
        int n_roots = 1);
    
    /**
     * Compute FCI ground state (and excited states if requested)
     * 
     * Algorithm:
     * 1. Generate ALL determinants
     * 2. Check size: use dense or Davidson
     * 3. Solve eigenvalue problem
     * 4. Analyze wavefunction
     * 
     * @return FCIResult with exact energy and wavefunction
     */
    FCIResult compute();
    
    /**
     * Compute with comparison to approximate method
     * 
     * @param approx_energy Energy from CISD, CCSD, etc.
     * @param method_name Name of the approximate method
     * @return FCIResult with comparison data
     */
    FCIResult compute_with_comparison(double approx_energy, 
                                      const std::string& method_name);
    
    /**
     * Get all determinants (without solving)
     * Useful for analysis or external use
     */
    std::vector<Determinant> get_determinants() const;
    
    /**
     * Estimate number of determinants before generation
     * Use binomial coefficient: C(n, k)
     * 
     * @return Estimated number of determinants
     */
    size_t estimate_n_determinants() const;
    
private:
    const CIIntegrals& ints_;
    int n_orb_;
    int n_alpha_;
    int n_beta_;
    int n_roots_;
    
    /**
     * Generate all possible determinants
     * 
     * Algorithm (Combinatorial):
     * 1. Generate all alpha strings: C(n_orb, n_alpha)
     * 2. Generate all beta strings: C(n_orb, n_beta)
     * 3. Cartesian product: all (alpha, beta) pairs
     * 
     * REFERENCE: Knowles & Handy (1984)
     */
    std::vector<Determinant> generate_all_determinants();
    
    /**
     * Generate all combinations of k items from n
     * 
     * Uses recursive generation algorithm.
     * Returns occupation lists, not bit strings.
     * 
     * @param n Total items
     * @param k Items to choose
     * @return All C(n,k) combinations
     */
    std::vector<std::vector<int>> generate_combinations(int n, int k);
    
    /**
     * Analyze FCI wavefunction
     * 
     * Breaks down contribution by excitation level:
     * - HF (reference)
     * - Singles
     * - Doubles
     * - Triples
     * - Quadruples, etc.
     */
    void analyze_wavefunction(const Eigen::VectorXd& c,
                             const std::vector<Determinant>& dets,
                             FCIResult& result);
    
};

/**
 * Compute binomial coefficient C(n, k)
 * Used for estimating determinant count
 */
size_t fci_binomial(int n, int k);

/**
 * Helper: Estimate FCI determinant count
 * 
 * @param n_orbitals Number of spatial orbitals
 * @param n_alpha Number of alpha electrons
 * @param n_beta Number of beta electrons
 * @return Estimated determinant count
 */
size_t fci_determinant_count(int n_orbitals, int n_alpha, int n_beta);

/**
 * Helper: Check if FCI is feasible
 * 
 * @param n_orbitals Number of orbitals
 * @param n_alpha Alpha electrons
 * @param n_beta Beta electrons
 * @param max_dets Maximum allowed determinants (default 1M)
 * @return true if FCI is computationally feasible
 */
bool is_fci_feasible(int n_orbitals, int n_alpha, int n_beta, 
                     size_t max_dets = 1000000);

} // namespace ci
} // namespace mshqc

#endif // MSHQC_CI_FCI_H
