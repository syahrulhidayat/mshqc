/**
 * @file cisdt.h
 * @brief Configuration Interaction Singles Doubles Triples (CISDT)
 * 
 * THEORY REFERENCES:
 *   - Pople et al. (1977), Int. J. Quantum Chem. 11, 149
 *   - Raghavachari et al. (1989), J. Chem. Phys. 91, 1062
 *   - Helgaker et al. (2000), Ch. 10.6
 * 
 * CISDT extends CISD by including triple excitations |ijk⟩ → |abc⟩.
 * This captures higher-order correlation effects beyond doubles.
 * 
 * USAGE:
 *   CISDT cisdt(ci_ints, hf_det, n_occ_a, n_occ_b, n_virt_a, n_virt_b);
 *   auto result = cisdt.compute();
 * 
 * SCALING:
 *   - Determinants: O(N_occ³ × N_virt³) for triples
 *   - Hamiltonian: O(N_det²) or use Davidson
 *   - Much more expensive than CISD!
 * 
 * TYPICAL USE:
 *   - Small molecules (< 10 electrons)
 *   - Benchmarking higher excitations
 *   - Systems with strong triple character
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 * @note Original implementation from textbook theory (AI_RULES compliant)
 */

#ifndef MSHQC_CI_CISDT_H
#define MSHQC_CI_CISDT_H

#include "mshqc/ci/determinant.h"
#include "mshqc/ci/slater_condon.h"
#include "mshqc/ci/davidson.h"
#include <Eigen/Dense>
#include <vector>

namespace mshqc {
namespace ci {

/**
 * CISDT result structure
 */
struct CISDTResult {
    double e_cisdt;         // CISDT energy (total)
    double e_hf;            // HF reference energy
    double e_corr;          // Correlation energy (CISDT - HF)
    int n_determinants;     // Total determinants (1 + S + D + T)
    int n_singles;          // Number of singles
    int n_doubles;          // Number of doubles
    int n_triples;          // Number of triples
    bool converged;         // Davidson convergence
    
    // Wavefunction data
    std::vector<Determinant> determinants;
    Eigen::VectorXd coefficients;
    
    // Analysis
    double hf_weight;       // |c_HF|²
    double singles_weight;  // Σ|c_singles|²
    double doubles_weight;  // Σ|c_doubles|²
    double triples_weight;  // Σ|c_triples|²
};

/**
 * CISDT Configuration Options
 */
struct CISDTOptions {
    bool use_davidson = true;       // Use Davidson (default) or dense
    int max_davidson_iter = 100;    // Davidson iterations
    double davidson_tol = 1e-6;     // Davidson convergence
    bool verbose = true;            // Print progress
    
    // Size thresholds
    int davidson_threshold = 1000;  // Use Davidson if N_det > threshold
};

/**
 * Configuration Interaction Singles Doubles Triples
 * 
 * CISDT Wavefunction:
 *   |Ψ⟩ = c₀|HF⟩ + Σᵢₐ cᵢₐ|i→a⟩ + Σᵢⱼₐb cᵢⱼₐb|ij→ab⟩ + Σᵢⱼₖₐbc cᵢⱼₖₐbc|ijk→abc⟩
 * 
 * THEORY:
 *   Triple excitations capture connected triple substitutions:
 *   - T₃ amplitude: t_ijk^abc = ⟨ijk||abc⟩ / (εᵢ + εⱼ + εₖ - εₐ - εb - εc)
 *   - Includes O(N⁶) terms
 * 
 * IMPORTANT:
 *   - Very expensive! Only practical for small systems
 *   - Consider CISDTQ for even higher accuracy
 *   - For production, use CCSD(T) instead (cheaper)
 */
class CISDT {
public:
    /**
     * Constructor
     * 
     * @param ints MO integrals (h, ERI)
     * @param hf_det HF reference determinant
     * @param n_occ_alpha Number of occupied alpha orbitals
     * @param n_occ_beta Number of occupied beta orbitals
     * @param n_virt_alpha Number of virtual alpha orbitals
     * @param n_virt_beta Number of virtual beta orbitals
     */
    CISDT(const CIIntegrals& ints,
          const Determinant& hf_det,
          int n_occ_alpha, int n_occ_beta,
          int n_virt_alpha, int n_virt_beta);
    
    /**
     * Compute CISDT energy
     * 
     * Algorithm:
     * 1. Generate singles excitations
     * 2. Generate doubles excitations
     * 3. Generate triples excitations (NEW!)
     * 4. Build full determinant list
     * 5. Solve with Davidson (or dense if small)
     * 6. Extract energy and wavefunction
     * 
     * @param opts CISDT options
     * @return CISDTResult with energy and diagnostics
     */
    CISDTResult compute(const CISDTOptions& opts = CISDTOptions());
    
    /**
     * Generate single excitations
     * Same as CISD
     * 
     * @return Vector of singly-excited determinants
     */
    std::vector<Determinant> generate_singles();
    
    /**
     * Generate double excitations
     * Same as CISD
     * 
     * @return Vector of doubly-excited determinants
     */
    std::vector<Determinant> generate_doubles();
    
    /**
     * Generate triple excitations
     * 
     * Creates all |ijk⟩ → |abc⟩ excitations with i<j<k, a<b<c
     * 
     * SPIN CASES:
     * - αααα: i_α j_α k_α → a_α b_α c_α
     * - ββββ: i_β j_β k_β → a_β b_β c_β  
     * - ααβ:  i_α j_α k_β → a_α b_α c_β (2 variants)
     * - αββ:  i_α j_β k_β → a_α b_β c_β (2 variants)
     * 
     * REFERENCE: Raghavachari et al. (1989)
     * 
     * @return Vector of triply-excited determinants
     */
    std::vector<Determinant> generate_triples();
    
    /**
     * Estimate number of determinants
     * 
     * @return Estimated N_det = 1 + N_S + N_D + N_T
     */
    size_t estimate_n_determinants() const;
    
    /**
     * Get all determinants (without solving)
     * Useful for analysis
     */
    std::vector<Determinant> get_determinants() const;
    
private:
    const CIIntegrals& ints_;
    Determinant hf_det_;
    int nocc_a_, nocc_b_;
    int nvirt_a_, nvirt_b_;
};

/**
 * Estimate CISDT determinant count
 * 
 * Formula:
 *   N_det = 1 + N_S + N_D + N_T
 *   N_S = n_occ × n_virt
 *   N_D = C(n_occ,2) × C(n_virt,2)
 *   N_T = C(n_occ,3) × C(n_virt,3)
 * 
 * @param n_orb_alpha Alpha orbitals
 * @param n_orb_beta Beta orbitals
 * @param n_occ_alpha Occupied alpha
 * @param n_occ_beta Occupied beta
 * @return Estimated determinant count
 */
size_t cisdt_determinant_count(int n_orb_alpha, int n_orb_beta,
                               int n_occ_alpha, int n_occ_beta);

} // namespace ci
} // namespace mshqc

#endif // MSHQC_CI_CISDT_H
