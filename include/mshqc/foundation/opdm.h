/**
 * @file opdm.h
 * @brief One-Particle Density Matrix (OPDM) for multireference wavefunctions
 * 
 * THEORY REFERENCES:
 *   - T. Helgaker, P. Jørgensen, & J. Olsen, "Molecular Electronic-Structure 
 *     Theory" (2000), Chapter 11 [Reduced density matrices]
 *   - E. R. Davidson, Chem. Phys. Lett. 21, 565 (1976)
 *     [Reduced density matrices in quantum chemistry]
 *   - R. McWeeny, Rev. Mod. Phys. 32, 335 (1960)
 *     [Density matrix theory]
 *   - A. Szabo & N. S. Ostlund, "Modern Quantum Chemistry" (1996), Sec. 2.4
 *     [Density matrix formalism]
 * 
 * FORMULA (Helgaker Eq. 11.2.1):
 *   γ_pq = ⟨Ψ| a†_p a_q |Ψ⟩
 * 
 * where:
 *   |Ψ⟩ = Σ_I c_I |Φ_I⟩  (CI wavefunction)
 *   a†_p creates electron in orbital p
 *   a_q annihilates electron in orbital q
 * 
 * Used for:
 *   - Natural orbitals (eigenvectors of γ)
 *   - Natural occupation numbers (eigenvalues of γ)
 *   - CASSCF orbital optimization gradients
 *   - Expectation values of one-electron operators
 *   - Population analysis
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 * @license MIT License
 * 
 * Copyright (c) 2025 Muhamad Sahrul Hidayat
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 * @note This is an original implementation derived from published theory.
 *       No code was copied from existing quantum chemistry software.
 */

#ifndef MSHQC_FOUNDATION_OPDM_H
#define MSHQC_FOUNDATION_OPDM_H

#include "mshqc/ci/determinant.h"
#include <Eigen/Dense>
#include <vector>
#include <utility>

namespace mshqc {
namespace foundation {

/**
 * @brief One-Particle Density Matrix (OPDM)
 * 
 * THEORY: Helgaker et al. (2000), Chapter 11
 * 
 * Computes reduced one-particle density matrix from CI wavefunction:
 *   γ_pq = ⟨Ψ| a†_p a_q |Ψ⟩ = Σ_IJ c_I c_J ⟨Φ_I| a†_p a_q |Φ_J⟩
 * 
 * Properties:
 *   - Hermitian: γ_pq = γ*_qp
 *   - Trace: Tr(γ) = N_electrons
 *   - Eigenvalues: 0 ≤ n_i ≤ 1 (natural occupations)
 * 
 * Applications:
 *   1. Natural orbitals: Diagonalization γ = U Λ U†
 *   2. Correlation measure: Entropy S = -Σ n_i ln(n_i)
 *   3. CASSCF gradients: ∂E/∂C = function(γ, h)
 *   4. Expectation values: ⟨O⟩ = Σ γ_pq O_qp
 */
class OPDM {
public:
    /**
     * @brief Construct OPDM from CI wavefunction
     * 
     * ALGORITHM (Helgaker Ch. 11):
     * 1. Loop over determinant pairs (I, J)
     * 2. Compute ⟨Φ_I| a†_p a_q |Φ_J⟩ using Slater-Condon rules
     * 3. Accumulate γ_pq += c_I c_J ⟨I|a†_p a_q|J⟩
     * 
     * Slater-Condon rules:
     *   - Same determinant: γ_pp += c_I² (diagonal)
     *   - Single excitation: γ_ij += phase × c_I c_J
     *   - Higher excitations: zero contribution
     * 
     * @param ci_coeffs CI coefficients c_I (normalized: Σ c_I² = 1)
     * @param determinants Slater determinants |Φ_I⟩
     * @param n_orbitals Number of spatial orbitals
     * 
     * @throws std::invalid_argument if inputs inconsistent
     */
    OPDM(const std::vector<double>& ci_coeffs,
         const std::vector<ci::Determinant>& determinants,
         int n_orbitals);
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    /**
     * @brief Get α-spin density matrix
     * @return γ^α_pq (n_orb × n_orb matrix)
     */
    const Eigen::MatrixXd& alpha() const { return opdm_alpha_; }
    
    /**
     * @brief Get β-spin density matrix
     * @return γ^β_pq (n_orb × n_orb matrix)
     */
    const Eigen::MatrixXd& beta() const { return opdm_beta_; }
    
    /**
     * @brief Get total (spin-summed) density matrix
     * @return γ^total_pq = γ^α_pq + γ^β_pq
     */
    Eigen::MatrixXd total() const;
    
    /**
     * @brief Get specific matrix element
     * @param p Orbital index (bra)
     * @param q Orbital index (ket)
     * @param alpha true for α-spin, false for β-spin
     * @return γ_pq value
     */
    double operator()(int p, int q, bool alpha) const;
    
    /**
     * @brief Number of spatial orbitals
     */
    int n_orbitals() const { return n_orbitals_; }
    
    /**
     * @brief Number of determinants in CI expansion
     */
    int n_determinants() const { return n_determinants_; }
    
    // ========================================================================
    // Properties
    // ========================================================================
    
    /**
     * @brief Trace of density matrix
     * 
     * PROPERTY (Helgaker Eq. 11.2.3):
     *   Tr(γ^α) = N_α (number of α electrons)
     *   Tr(γ^β) = N_β (number of β electrons)
     * 
     * @param alpha true for α-spin, false for β-spin
     * @return Trace Σ_p γ_pp
     */
    double trace(bool alpha) const;
    
    /**
     * @brief Check N-representability conditions
     * 
     * CONDITIONS (Davidson 1976):
     * 1. Hermiticity: γ_pq = γ*_qp
     * 2. Positive semidefinite: all eigenvalues ≥ 0
     * 3. Pauli exclusion: all eigenvalues ≤ 1
     * 4. Particle conservation: Tr(γ) = N_electrons
     * 
     * @param tolerance Numerical tolerance for checks
     * @return true if all conditions satisfied
     */
    bool is_n_representable(double tolerance = 1e-6) const;
    
    // ========================================================================
    // Natural Orbitals
    // ========================================================================
    
    /**
     * @brief Compute natural orbitals and occupation numbers
     * 
     * THEORY: Löwdin (1955), Phys. Rev. 97, 1474
     * 
     * Natural orbitals diagonalize the density matrix:
     *   γ |φ_i⟩ = n_i |φ_i⟩
     * 
     * where:
     *   - |φ_i⟩ are natural orbitals
     *   - n_i are natural occupation numbers (0 ≤ n_i ≤ 1)
     * 
     * Physical interpretation:
     *   - n_i ≈ 2: Strongly occupied (doubly occupied orbital)
     *   - n_i ≈ 1: Singly occupied (open-shell orbital)
     *   - n_i ≈ 0: Virtual (unoccupied orbital)
     *   - 0 < n_i < 2: Fractional occupation (correlation)
     * 
     * ALGORITHM:
     * 1. Diagonalize: γ = U Λ U†  (Eigen::SelfAdjointEigenSolver)
     * 2. Sort eigenvalues descending: n_0 ≥ n_1 ≥ ... ≥ n_N
     * 3. Return (occupations n_i, orbital coefficients U)
     * 
     * @param alpha true for α-spin, false for β-spin
     * @return Pair of (natural occupations, natural orbital coefficients)
     *         - occupations: Vector of n_i (length n_orb, sorted descending)
     *         - orbitals: Matrix C_μi (n_orb × n_orb)
     */
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> 
    natural_orbitals(bool alpha) const;
    
    /**
     * @brief Get natural occupation numbers only
     * 
     * Convenience function that just returns eigenvalues
     * 
     * @param alpha true for α-spin, false for β-spin
     * @return Natural occupations n_i (sorted descending)
     */
    Eigen::VectorXd natural_occupations(bool alpha) const;
    
    /**
     * @brief Entropy of natural occupation distribution
     * 
     * DEFINITION (Von Neumann entropy):
     *   S = -Σ_i [n_i ln(n_i) + (1-n_i) ln(1-n_i)]
     * 
     * Physical meaning:
     *   - S = 0: Single determinant (Hartree-Fock)
     *   - S > 0: Multireference character present
     *   - Larger S: Stronger static correlation
     * 
     * Used to quantify:
     *   - Multireference character
     *   - Correlation strength
     *   - Active space quality
     * 
     * @param alpha true for α-spin, false for β-spin
     * @return Entropy S (bits)
     */
    double entropy(bool alpha) const;
    
    // ========================================================================
    // Expectation Values
    // ========================================================================
    
    /**
     * @brief Expectation value of one-electron operator
     * 
     * FORMULA (Helgaker Eq. 11.2.5):
     *   ⟨O⟩ = Σ_pq γ_pq O_qp
     * 
     * where O_pq is one-electron operator in MO basis
     * 
     * Common operators:
     *   - O = h (core Hamiltonian) → kinetic + nuclear attraction
     *   - O = r (position) → dipole moment
     *   - O = S (overlap in AO basis) → Mulliken populations
     *   - O = δ(r-R) → electron density at point R
     * 
     * @param operator_matrix One-electron operator O_pq (n_orb × n_orb)
     * @param alpha true for α-spin, false for β-spin
     * @return Expectation value ⟨O⟩
     */
    double expectation_value(const Eigen::MatrixXd& operator_matrix,
                             bool alpha) const;
    
    /**
     * @brief One-electron energy contribution
     * 
     * FORMULA (Helgaker Eq. 14.8.1):
     *   E_1e = Σ_pq (γ^α_pq + γ^β_pq) h_pq
     * 
     * where h_pq is core Hamiltonian (kinetic + nuclear attraction)
     * 
     * Note: This is only ONE-ELECTRON part of total energy!
     *       Full energy needs two-electron part from TPDM.
     * 
     * @param h_core Core Hamiltonian h_pq in MO basis
     * @return One-electron energy E_1e
     */
    double one_electron_energy(const Eigen::MatrixXd& h_core) const;
    
    // ========================================================================
    // Validation & Debugging
    // ========================================================================
    
    /**
     * @brief Print OPDM statistics
     * 
     * Output includes:
     *   - Matrix dimensions
     *   - Trace (should equal N_electrons)
     *   - Eigenvalue range (min, max)
     *   - Largest off-diagonal elements
     *   - Hermiticity check
     *   - N-representability status
     */
    void print_statistics() const;
    
    /**
     * @brief Print natural orbitals and occupations
     * 
     * Formatted table:
     *   Orbital    n_α      n_β      Total
     *   ----------------------------------------
     *   0          1.982    1.982    3.964
     *   1          1.956    1.956    3.912
     *   2          0.044    0.044    0.088
     *   ...
     */
    void print_natural_orbitals() const;

private:
    // Density matrices
    Eigen::MatrixXd opdm_alpha_;  ///< γ^α_pq (n_orb × n_orb)
    Eigen::MatrixXd opdm_beta_;   ///< γ^β_pq (n_orb × n_orb)
    
    // Metadata
    int n_orbitals_;              ///< Number of spatial orbitals
    int n_determinants_;          ///< Number of CI determinants
    
    // Reference data (stored for potential recomputation)
    std::vector<double> ci_coeffs_;           ///< CI coefficients c_I
    std::vector<ci::Determinant> determinants_; ///< Slater determinants
    
    // ========================================================================
    // Internal computation methods
    // ========================================================================
    
    /**
     * @brief Main OPDM computation routine
     * 
     * ALGORITHM:
     * 1. Initialize γ^α = 0, γ^β = 0
     * 2. Loop over determinant pairs (I, J):
     *    a. Compute excitation level |Φ_I - Φ_J|
     *    b. If level ≤ 1: compute matrix element
     *    c. Add contribution: γ_pq += c_I c_J ⟨I|a†_p a_q|J⟩
     * 3. Validate result
     * 
     * Called by constructor
     */
    void compute_opdm();
    
    /**
     * @brief Process contribution from determinant pair
     * 
     * Dispatches to diagonal or off-diagonal handler based on I == J
     * 
     * @param I First determinant index
     * @param J Second determinant index
     * @param alpha Process α-spin (true) or β-spin (false)
     */
    void add_pair_contribution(int I, int J, bool alpha);
    
    /**
     * @brief Handle diagonal contribution (same determinant)
     * 
     * Slater-Condon rule for I = J:
     *   ⟨Φ| a†_p a_q |Φ⟩ = δ_pq if orbital p occupied, 0 otherwise
     * 
     * Implementation:
     *   For each occupied orbital p:
     *     γ_pp += c_I²
     * 
     * @param I Determinant index
     * @param alpha Process α-spin or β-spin
     */
    void add_diagonal_contribution(int I, bool alpha);
    
    /**
     * @brief Handle off-diagonal (different determinants)
     * 
     * For single excitation j → i:
     *   ⟨Φ_I| a†_i a_j |Φ_J⟩ = phase × δ_pi δ_qj
     * 
     * Phase: (-1)^{number of orbitals between j and i}
     * 
     * Implementation:
     * 1. Identify excitation: which orbital changed
     * 2. Compute phase from orbital ordering
     * 3. Add: γ_ij += phase × c_I c_J
     * 4. Add: γ_ji += phase × c_J c_I  (Hermitian)
     * 
     * @param I First determinant index
     * @param J Second determinant index
     * @param alpha Process α-spin or β-spin
     */
    void add_off_diagonal_contribution(int I, int J, bool alpha);
    
    /**
     * @brief Validate OPDM after construction
     * 
     * Checks:
     * 1. Hermiticity: ||γ - γ†|| < tol
     * 2. Trace: |Tr(γ) - N_electrons| < tol
     * 3. Eigenvalues: 0 - tol ≤ λ_i ≤ 1 + tol
     * 
     * @throws std::runtime_error if validation fails
     */
    void validate() const;
};

} // namespace foundation
} // namespace mshqc

#endif // MSHQC_FOUNDATION_OPDM_H
