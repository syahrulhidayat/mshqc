/**
 * @file natural_orbitals.h
 * @brief Natural orbital analysis for CI wavefunctions
 * 
 * THEORY REFERENCES:
 *   - Löwdin (1955), Phys. Rev. 97, 1474
 *   - McWeeny (1989), Methods of Molecular Quantum Mechanics, Ch. 6
 *   - Helgaker et al. (2000), Ch. 14.8
 * 
 * Natural orbitals (NOs) are eigenvectors of the 1-electron reduced density matrix (1-RDM).
 * They provide the most compact representation of a many-electron wavefunction.
 * 
 * USAGE:
 *   // After CI computation
 *   NaturalOrbitalAnalysis no_analysis(ci_result.determinants, ci_result.coefficients);
 *   auto no_result = no_analysis.compute(n_orbitals);
 *   
 *   // Print occupations
 *   no_result.print_occupations();
 * 
 * KEY CONCEPTS:
 *   - Natural occupations: eigenvalues of 1-RDM (0 ≤ n_i ≤ 2)
 *   - Natural orbitals: eigenvectors of 1-RDM
 *   - Sum rule: Σ n_i = N_electrons
 *   - Correlation measure: deviation from integer occupations
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 * @note Original implementation from textbook theory (AI_RULES compliant)
 */

#ifndef MSHQC_CI_NATURAL_ORBITALS_H
#define MSHQC_CI_NATURAL_ORBITALS_H

#include "mshqc/ci/determinant.h"
#include <Eigen/Dense>
#include <vector>

namespace mshqc {
namespace ci {

/**
 * Natural orbital result structure
 */
struct NaturalOrbitalResult {
    // Eigenvalues of 1-RDM (natural occupations, 0 ≤ n_i ≤ 2)
    Eigen::VectorXd occupations_alpha;
    Eigen::VectorXd occupations_beta;
    
    // Eigenvectors of 1-RDM (natural orbitals in MO basis)
    Eigen::MatrixXd orbitals_alpha;  // (n_orb × n_orb)
    Eigen::MatrixXd orbitals_beta;
    
    // Diagnostics
    double total_occupation_alpha;   // Should = N_alpha
    double total_occupation_beta;    // Should = N_beta
    double correlation_measure;      // Deviation from HF (integer occupations)
    int n_strongly_occupied;         // Occupations > 1.95
    int n_weakly_occupied;           // Occupations < 0.05
    int n_fractional;                // 0.05 < occ < 1.95 (correlation)
    
    // Print summary
    void print_summary() const;
    void print_occupations(int n_print = 10) const;
};

/**
 * Natural Orbital Analysis
 * 
 * Computes 1-electron reduced density matrix (1-RDM) from CI wavefunction
 * and diagonalizes to obtain natural orbitals and occupation numbers.
 * 
 * THEORY:
 * 1-RDM: γ_pq = ⟨Ψ| a†_p a_q |Ψ⟩ = Σ_I Σ_J c_I c_J ⟨I| a†_p a_q |J⟩
 * 
 * Natural orbitals: eigenvectors of γ
 * Natural occupations: eigenvalues of γ (sum to N_electrons)
 */
class NaturalOrbitalAnalysis {
public:
    /**
     * Constructor
     * 
     * @param dets List of CI determinants
     * @param coeffs CI coefficients (normalized)
     */
    NaturalOrbitalAnalysis(const std::vector<Determinant>& dets,
                          const Eigen::VectorXd& coeffs);
    
    /**
     * Compute natural orbitals
     * 
     * Algorithm:
     * 1. Build 1-RDM from CI wavefunction
     * 2. Diagonalize 1-RDM: γ = U n U†
     * 3. Extract natural occupations (n) and orbitals (U)
     * 4. Compute diagnostics
     * 
     * @param n_orb Number of molecular orbitals
     * @return NaturalOrbitalResult with occupations and orbitals
     */
    NaturalOrbitalResult compute(int n_orb);
    
    /**
     * Build 1-electron reduced density matrix (1-RDM)
     * 
     * γ_pq^α = ⟨Ψ| a†_p^α a_q^α |Ψ⟩
     * γ_pq^β = ⟨Ψ| a†_p^β a_q^β |Ψ⟩
     * 
     * REFERENCE: Helgaker et al. (2000), Eq. (14.8.3)
     * 
     * @param n_orb Number of orbitals
     * @param rdm_alpha Output: alpha 1-RDM
     * @param rdm_beta Output: beta 1-RDM
     */
    void build_1rdm(int n_orb,
                   Eigen::MatrixXd& rdm_alpha,
                   Eigen::MatrixXd& rdm_beta);
    
private:
    const std::vector<Determinant>& dets_;
    const Eigen::VectorXd& coeffs_;
    
    /**
     * Compute 1-RDM matrix element ⟨I| a†_p a_q |J⟩
     * 
     * Uses Slater-Condon rules:
     * - I = J: ⟨I|I⟩ = 1 if q occupied in I, else 0
     * - I ≠ J by 1 excitation i→a: ⟨I| a†_p a_q |J⟩ = δ_pa δ_qi × phase
     * - Otherwise: 0
     * 
     * @param det_i Bra determinant
     * @param det_j Ket determinant
     * @param p Creation orbital
     * @param q Annihilation orbital
     * @param alpha True for alpha spin, false for beta
     * @return Matrix element value
     */
    double rdm_element(const Determinant& det_i,
                      const Determinant& det_j,
                      int p, int q, bool alpha);
    
    /**
     * Compute correlation measure from natural occupations
     * Measures deviation from Hartree-Fock (integer occupations)
     * 
     * REFERENCE: Head-Gordon (2003), Chem. Phys. Lett. 372, 508
     * 
     * @param occupations Natural occupation numbers
     * @return Correlation measure (0 = HF, larger = more correlation)
     */
    double compute_correlation_measure(const Eigen::VectorXd& occupations);
};

} // namespace ci
} // namespace mshqc

#endif // MSHQC_CI_NATURAL_ORBITALS_H
