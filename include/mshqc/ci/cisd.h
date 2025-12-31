/**
 * @file cisd.h
 * @brief CISD (Configuration Interaction Singles + Doubles)
 * 
 * Production CI method including singles and doubles excitations.
 * More accurate than CIS, comparable to MP2 but variational.
 * 
 * THEORY REFERENCES:
 *   - Shavitt (1998), Mol. Phys. 94, 3
 *   - Szabo & Ostlund (1996), Ch. 4.3
 *   - Helgaker et al. (2000), Ch. 10
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 * @license MIT
 * 
 * @note Original implementation from published CI theory.
 *       No code copied from existing quantum chemistry software.
 */

#ifndef MSHQC_CI_CISD_H
#define MSHQC_CI_CISD_H

#include "mshqc/ci/determinant.h"
#include "mshqc/ci/slater_condon.h"
#include "mshqc/ci/davidson.h"
#include <vector>

namespace mshqc {
namespace ci {

/**
 * CISD computation options
 */
struct CISDOptions {
    bool use_sparse = false;        // Use sparse matrix representation
    double sparse_eps = 1e-12;      // Screening threshold for sparse H
    int sparse_threshold = 3000;    // Auto-switch to sparse if n_dets > threshold
    bool auto_sparse = true;        // Automatically decide dense vs sparse
    
    // On-the-fly sigma-vector mode (for large systems)
    bool use_onthefly = false;      // Enable on-the-fly sigma-vector
    bool auto_onthefly = true;      // Auto-enable for N > 200
    int onthefly_threshold = 200;   // Auto-switch threshold
    
    bool verbose = true;            // Print progress information
};

/**
 * CISD result structure
 */
struct CISDResult {
    double e_hf;           // HF energy (CI reference, bare H)
    double e_cisd;         // CISD total energy (bare H)
    double e_corr;         // Correlation energy
    double e_uhf_ref;      // UHF reference energy (for comparison)
    double e_cisd_corrected; // CISD corrected to UHF reference
    int n_determinants;    // Total # of determinants
    bool converged;        // Davidson convergence
    int iterations;        // # of Davidson iterations
    
    // CI wavefunction
    Eigen::VectorXd coefficients;
    std::vector<Determinant> determinants;
    
    // Comparison with MP2 (if available)
    double mp2_correlation = 0.0;
    double cisd_vs_mp2_diff = 0.0;
};

/**
 * CISD (Configuration Interaction Singles + Doubles)
 * 
 * THEORY: Include HF + singles + doubles
 * |Ψ⟩ = c_0 |HF⟩ + Σ_ia c_i^a |Φ_i^a⟩ + Σ_ijab c_ij^ab |Φ_ij^ab⟩
 * 
 * Properties:
 * - Variational (upper bound to exact energy)
 * - Size-consistent (unlike truncated CI with triples)
 * - Comparable accuracy to MP2
 * - More expensive: O(N^6) vs MP2 O(N^5)
 * 
 * Scaling:
 * # determinants ~ n_occ^2 * n_virt^2 ~ O(N^4)
 * Example: H2O/cc-pVDZ: ~10^5 determinants
 */
class CISD {
public:
    /**
     * Constructor
     * @param ints MO integrals in physicist notation
     * @param hf_det HF reference determinant
     * @param n_occ_alpha # occupied α orbitals
     * @param n_occ_beta # occupied β orbitals
     * @param n_virt_alpha # virtual α orbitals
     * @param n_virt_beta # virtual β orbitals
     */
    CISD(const CIIntegrals& ints,
         const Determinant& hf_det,
         int n_occ_alpha, int n_occ_beta,
         int n_virt_alpha, int n_virt_beta);
    
    /**
     * Compute CISD energy
     * 
     * REFERENCE: Shavitt (1998), Mol. Phys. 94, 3
     * 
     * @param opts Computation options (sparse/dense, thresholds)
     * @return CISD result with energy and wavefunction
     */
    CISDResult compute(const CISDOptions& opts = CISDOptions());
    
    /**
     * Compute and compare with MP2
     * @param mp2_correlation MP2 correlation energy for comparison
     */
    CISDResult compute_with_mp2_comparison(double mp2_correlation);
    
    /**
     * Get all CISD determinants (HF + singles + doubles)
     */
    std::vector<Determinant> get_determinants() const;
    
private:
    const CIIntegrals& ints_;
    Determinant hf_det_;
    int nocc_a_, nocc_b_;
    int nvirt_a_, nvirt_b_;
    
    /**
     * Generate singles excitations
     * REFERENCE: Szabo & Ostlund (1996), Section 4.2
     */
    std::vector<Determinant> generate_singles();
    
    /**
     * Generate doubles excitations
     * REFERENCE: Szabo & Ostlund (1996), Section 4.3
     * 
     * Three spin cases:
     * - αα: i_α j_α → a_α b_α
     * - ββ: i_β j_β → a_β b_β
     * - αβ: i_α j_β → a_α b_β
     */
    std::vector<Determinant> generate_doubles();
    
    /**
     * Analyze CI wavefunction
     * Extract dominant configurations
     */
    void analyze_wavefunction(const Eigen::VectorXd& c,
                              const std::vector<Determinant>& dets);
};

} // namespace ci
} // namespace mshqc

#endif // MSHQC_CI_CISD_H
