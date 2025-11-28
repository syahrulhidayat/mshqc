#ifndef MSHQC_INTEGRALS_H
#define MSHQC_INTEGRALS_H

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <memory>

/**
 * @file integrals.h
 * @brief Gaussian integral evaluation using Libint2
 * 
 * REFERENCES:
 * Szabo & Ostlund (1996), Chapter 3, pp. 153-217
 * Helgaker et al. (2000), Chapters 9-10, pp. 315-426
 * Valeev, E. F. Libint2 (http://libint.valeyev.net/)
 * 
 * One-electron: S, T, V | Two-electron: (μν|λσ) = ⟨μν|r₁₂⁻¹|λσ⟩
 */

namespace mshqc {

/**
 * @brief Four-dimensional tensor for electron repulsion integrals
 * 
 * Chemist's notation: ERI(μ,ν,λ,σ) = ⟨μν|λσ⟩
 * Symmetries: (μν|λσ) = (νμ|λσ) = (μν|σλ) = (λσ|μν)
 */
using ERITensor = Eigen::Tensor<double, 4>;

/**
 * @brief Integral engine for computing molecular integrals
 * 
 * Libint2 wrapper for one-electron (S, T, V) and two-electron (ERI) integrals
 */
class IntegralEngine {
public:
    /**
     * @brief Construct integral engine for molecule and basis
     * @param mol Molecule object
     * @param basis Basis set
     */
    IntegralEngine(const Molecule& mol, const BasisSet& basis);
    
    /**
     * @brief Destructor (cleanup Libint2)
     */
    ~IntegralEngine();
    
    // Disable copy (Libint2 engine not copyable)
    IntegralEngine(const IntegralEngine&) = delete;
    IntegralEngine& operator=(const IntegralEngine&) = delete;
    
    // Allow move
    IntegralEngine(IntegralEngine&&) = default;
    IntegralEngine& operator=(IntegralEngine&&) = default;
    
    /**
     * @brief Compute overlap matrix S
     * 
     * REFERENCE: Szabo & Ostlund (1996), Eq. (3.1), p. 155
     * S_μν = ⟨μ|ν⟩ (symmetric, positive definite)
     */
    Eigen::MatrixXd compute_overlap();
    
    /**
     * @brief Compute kinetic energy matrix T
     * 
     * REFERENCE: Szabo & Ostlund (1996), Eq. (3.153), p. 180
     * T_μν = ⟨μ|-½∇²|ν⟩ (symmetric, positive definite)
     */
    Eigen::MatrixXd compute_kinetic();
    
    /**
     * @brief Compute nuclear attraction matrix V
     * 
     * REFERENCE: Szabo & Ostlund (1996), Eq. (3.154), p. 180
     * V_μν = ⟨μ|V_nuc|ν⟩ = -∫ φ_μ [Σ_A Z_A/r_A] φ_ν dr (negative definite)
     */
    Eigen::MatrixXd compute_nuclear();
    
    /**
     * @brief Compute core Hamiltonian H = T + V
     * 
     * REFERENCE: Szabo & Ostlund (1996), Eq. (3.152), p. 179
     * H_μν = T_μν + V_μν = ⟨μ|-½∇² + V_nuc|ν⟩
     */
    Eigen::MatrixXd compute_core_hamiltonian();
    
    /**
     * @brief Compute electron repulsion integrals (ERI)
     * 
     * Two-electron integral in chemist's notation:
     *   (μν|λσ) = ⟨μν|1/r₁₂|λσ⟩
     *           = ∫∫ φ_μ(r₁)φ_ν(r₁) (1/|r₁-r₂|) φ_λ(r₂)φ_σ(r₂) dr₁dr₂
     * 
     * Symmetries (8-fold):
     *   (μν|λσ) = (νμ|λσ) = (μν|σλ) = (νμ|σλ)
     *           = (λσ|μν) = (σλ|μν) = (λσ|νμ) = (σλ|νμ)
     * 
     * Storage: Full tensor (nbasis⁴ elements)
     * For STO-3G/Li: 5⁴ = 625 elements (~5 KB)
     * For larger basis: use symmetry or Schwarz screening
     * 
     * REFERENCE:
     * Szabo & Ostlund (1996), Eq. (3.155), p. 181
     * "The two-electron repulsion integral represents the Coulomb
     *  repulsion between two electron distributions"
     * 
     * Helgaker et al. (2000), Section 10.3, pp. 386-400
     * "Efficient evaluation of ERIs using recursion relations"
     * 
     * @return ERI tensor (nbasis × nbasis × nbasis × nbasis)
     */
    Eigen::Tensor<double, 4> compute_eri();
    
    /**
     * @brief Compute ERI with Schwarz screening
     * 
     * Uses Schwarz inequality to skip negligible integrals:
     *   |(μν|λσ)| ≤ √[(μν|μν)(λσ|λσ)]
     * 
     * If √[(μν|μν)(λσ|λσ)] < threshold, skip (μν|λσ)
     * 
     * This can significantly reduce computation time for large basis sets.
     * 
     * REFERENCE:
     * Häser & Ahlrichs, J. Comput. Chem. 10, 104 (1989)
     * "Improvements on the direct SCF method"
     * 
     * @param threshold Schwarz screening threshold (default: 1e-12)
     * @return ERI tensor (with screened integrals set to zero)
     */
    Eigen::Tensor<double, 4> compute_eri_screened(double threshold = 1e-12);
    
    /**
     * @brief Get number of basis functions
     */
    size_t nbasis() const { return nbasis_; }
    
    /**
     * @brief Print integral statistics
     * 
     * Shows timing, number of integrals computed, screening statistics, etc.
     */
    void print_statistics() const;
    
    /**
     * @brief Compute 3-center electron repulsion integrals for DF-MP2
     * 
     * Computes (μν|P) integrals where μ,ν are primary basis functions
     * and P runs over auxiliary basis functions.
     * 
     * Formula: (μν|P) = ∫∫ φ_μ(r1)φ_ν(r1) r12^-1 χ_P(r2) dr1 dr2
     * 
     * REFERENCE:
     * Weigend & Häser (1997), Theor. Chem. Acc. 97, 331, Eq. (2)
     * "3-center integrals for density fitting approximation"
     * 
     * @param aux_basis Auxiliary basis set (e.g. cc-pVTZ-RI)
     * @return 3-center tensor [nbasis × nbasis × naux]
     */
    Eigen::Tensor<double, 3> compute_3center_eri(const BasisSet& aux_basis);
    
    /**
     * @brief Compute 2-center auxiliary metric (P|Q)
     * 
     * Computes Coulomb integrals between auxiliary basis functions.
     * 
     * Formula: (P|Q) = ∫∫ χ_P(r1) r12^-1 χ_Q(r2) dr1 dr2
     * 
     * This metric is inverted to form J^-1 in DF-MP2.
     * 
     * REFERENCE:
     * Feyereisen et al. (1993), Chem. Phys. Lett. 208, 359, Eq. (8)
     * "Auxiliary basis metric for resolution-of-identity"
     * 
     * @param aux_basis Auxiliary basis set
     * @return Metric matrix [naux × naux]
     */
    Eigen::MatrixXd compute_2center_eri(const BasisSet& aux_basis);
    
private:
    const Molecule& mol_;         ///< Molecule reference
    const BasisSet& basis_;       ///< Basis set reference
    size_t nbasis_;              ///< Number of basis functions
    
    // Forward declaration of Libint2 shell type
    // (actual implementation will use libint2::Shell)
    struct LibintShellData;
    std::unique_ptr<LibintShellData> libint_shells_;  ///< Libint2 shells
    
    /**
     * @brief Initialize Libint2 library
     * 
     * Must be called before any integral computation.
     * Automatically called in constructor.
     */
    void initialize_libint();
    
    /**
     * @brief Finalize Libint2 library
     * 
     * Cleanup Libint2 resources.
     * Automatically called in destructor.
     */
    void finalize_libint();
    
    /**
     * @brief Convert QuantChem basis to Libint2 format
     * 
     * Translates our Shell objects to libint2::Shell format.
     */
    void convert_basis_to_libint();
};

/**
 * @brief Compute Fock matrix from density matrix
 * 
 * The Fock matrix for closed-shell systems:
 *   F_μν = H_μν + Σ_λσ P_λσ [2(μν|λσ) - (μλ|νσ)]
 *        = H_μν + G_μν
 * 
 * where:
 * - H = core Hamiltonian (T + V)
 * - G = two-electron contribution
 * - P = density matrix
 * - (μν|λσ) = electron repulsion integrals
 * 
 * For open-shell (ROHF), separate α and β Fock matrices are needed.
 * 
 * REFERENCE:
 * Szabo & Ostlund (1996), Eq. (3.154), p. 139
 * "The Fock operator in the Hartree-Fock equations"
 * 
 * @param H Core Hamiltonian matrix
 * @param P Density matrix
 * @param ERI Electron repulsion integrals
 * @return Fock matrix F = H + G(P)
 */
Eigen::MatrixXd compute_fock_matrix(
    const Eigen::MatrixXd& H,
    const Eigen::MatrixXd& P,
    const Eigen::Tensor<double, 4>& ERI);

/**
 * @brief Transform integrals from AO to MO basis
 * 
 * Four-index transformation:
 *   (pq|rs)_MO = Σ_μνλσ C_μp C_νq C_λr C_σs (μν|λσ)_AO
 * 
 * This is the most expensive step in post-HF methods (O(N⁵) scaling).
 * 
 * Algorithm: Quarter transformations (4 steps)
 * 1. (μν|λσ) → (pν|λσ)
 * 2. (pν|λσ) → (pq|λσ)
 * 3. (pq|λσ) → (pq|rσ)
 * 4. (pq|rσ) → (pq|rs)
 * 
 * REFERENCE:
 * Helgaker et al. (2000), Section 10.7, pp. 409-413
 * "Transformation of two-electron integrals"
 * 
 * @param ERI_AO ERIs in AO basis
 * @param C MO coefficient matrix
 * @return ERIs in MO basis
 */
Eigen::Tensor<double, 4> transform_eri_to_mo(
    const Eigen::Tensor<double, 4>& ERI_AO,
    const Eigen::MatrixXd& C);

} // namespace mshqc

#endif // MSHQC_INTEGRALS_H
