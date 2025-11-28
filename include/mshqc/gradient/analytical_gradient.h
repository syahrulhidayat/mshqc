/**
 * @file analytical_gradient.h
 * @brief Analytical energy gradients for SCF methods (RHF, UHF)
 * 
 * Implements analytical first derivatives of SCF energy with respect to
 * nuclear coordinates. Much faster than numerical gradients (10-50× speedup).
 * 
 * THEORY:
 * Energy gradient: ∂E/∂R_A = ∂E_nuc/∂R_A + ∂E_elec/∂R_A
 * 
 * Electronic contribution:
 * ∂E_elec/∂R_A = Tr[W·∂S/∂R_A] + Tr[P·∂H/∂R_A] + Σ_μνλσ G_μνλσ·∂(μν|λσ)/∂R_A
 * 
 * where W is energy-weighted density matrix (computed via Z-vector method)
 * 
 * THEORY REFERENCES:
 * 1. P. Pulay, Mol. Phys. 17, 197 (1969)
 *    - Original force method for analytical gradients
 * 2. J.A. Pople et al., Int. J. Quantum Chem. Symp. 13, 225 (1979)
 *    - Derivative methods in quantum chemistry
 * 3. T. Helgaker et al., "Molecular Electronic-Structure Theory" (2000), Chapter 13
 *    - Comprehensive gradient theory
 * 4. H.B. Schlegel, J. Comput. Chem. 3, 214 (1982)
 *    - Analytical gradients for closed-shell systems
 * 5. P.M.W. Gill et al., Adv. Quantum Chem. 25, 141 (1994)
 *    - Integral derivative algorithms
 * 6. J. Gauss et al., J. Chem. Phys. 95, 2623 (1991)
 *    - Analytical gradients for open-shell methods
 * 7. Y. Yamaguchi et al., "A New Dimension to Quantum Chemistry" (1994)
 *    - Practical implementation of analytical derivatives
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 * 
 * @note Original implementation based on theory papers.
 *       No code copied from other quantum chemistry software.
 */

#ifndef MSHQC_ANALYTICAL_GRADIENT_H
#define MSHQC_ANALYTICAL_GRADIENT_H

#include "mshqc/gradient/gradient.h"
#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include <Eigen/Dense>
#include <memory>

namespace mshqc {
namespace gradient {

// ============================================================================
// Integral Derivatives
// ============================================================================

/**
 * @brief Compute derivative of overlap matrix: ∂S/∂R_A
 * 
 * For Gaussian basis functions:
 * S_μν = ⟨μ|ν⟩
 * ∂S_μν/∂R_A = ⟨∂μ/∂R_A|ν⟩ + ⟨μ|∂ν/∂R_A⟩
 * 
 * Uses translational invariance and Hermiticity.
 * 
 * REFERENCE: Helgaker et al. (2000), Eq. (13.2.15)
 * 
 * @param basis Basis set
 * @param mol Molecule
 * @param atom_idx Atom index A (which atom to differentiate w.r.t.)
 * @param coord Direction (0=x, 1=y, 2=z)
 * @return ∂S/∂R_A matrix (nbasis × nbasis)
 */
Eigen::MatrixXd compute_overlap_derivative(
    const BasisSet& basis,
    const Molecule& mol,
    int atom_idx,
    int coord
);

/**
 * @brief Compute derivative of core Hamiltonian: ∂H/∂R_A
 * 
 * H = T + V_ne
 * ∂H/∂R_A = ∂T/∂R_A + ∂V_ne/∂R_A
 * 
 * Kinetic: ∂T/∂R_A from Gaussian derivative rules
 * Nuclear: ∂V_ne/∂R_A involves both nuclear and basis function movement
 * 
 * REFERENCE: Helgaker et al. (2000), Eq. (13.2.16)-(13.2.17)
 * 
 * @return ∂H/∂R_A matrix (nbasis × nbasis)
 */
Eigen::MatrixXd compute_core_hamiltonian_derivative(
    const BasisSet& basis,
    const Molecule& mol,
    int atom_idx,
    int coord
);

/**
 * @brief Compute derivative of two-electron integral: ∂(μν|λσ)/∂R_A
 * 
 * Most expensive part of analytical gradient.
 * Uses translational invariance to reduce number of derivatives.
 * 
 * For efficiency, returns as 4-index tensor contribution to gradient:
 * Σ_μνλσ G_μνλσ·∂(μν|λσ)/∂R_A
 * 
 * where G is two-electron density matrix.
 * 
 * REFERENCE: Gill et al. (1994), Adv. Quantum Chem. 25, 141
 * 
 * @param density_matrix Density matrix P
 * @return Contribution to gradient from ERI derivatives
 */
double compute_eri_derivative_contribution(
    const BasisSet& basis,
    const Molecule& mol,
    const Eigen::MatrixXd& density_matrix,
    int atom_idx,
    int coord
);

// ============================================================================
// Z-Vector Method (Coupled-Perturbed HF)
// ============================================================================

/**
 * @brief Compute energy-weighted density matrix W via Z-vector method
 * 
 * Avoids computing full orbital response ∂C/∂R by solving:
 * W·S = -P·F^[1] + (energy-weighted terms)
 * 
 * where F^[1] is derivative of Fock matrix at fixed orbitals.
 * 
 * REFERENCE: Handy & Schaefer, J. Chem. Phys. 81, 5031 (1984)
 * 
 * For RHF:
 * W_μν = Σ_i^occ Σ_a^virt (ε_i - ε_a)^{-1} U_ia C_μi C_νa
 * 
 * where U_ia from CPHF equations:
 * (ε_a - ε_i)U_ia = B_ia^x (x = nuclear displacement)
 * 
 * @param C MO coefficients
 * @param eps Orbital energies
 * @param n_occ Number of occupied orbitals
 * @param F_deriv Derivative of Fock matrix at fixed orbitals
 * @param S Overlap matrix
 * @return Energy-weighted density W
 */
Eigen::MatrixXd compute_z_vector_rhf(
    const Eigen::MatrixXd& C,
    const Eigen::VectorXd& eps,
    int n_occ,
    const Eigen::MatrixXd& F_deriv,
    const Eigen::MatrixXd& S
);

/**
 * @brief Z-vector for UHF (separate alpha and beta)
 */
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> compute_z_vector_uhf(
    const Eigen::MatrixXd& C_alpha,
    const Eigen::MatrixXd& C_beta,
    const Eigen::VectorXd& eps_alpha,
    const Eigen::VectorXd& eps_beta,
    int n_alpha,
    int n_beta,
    const Eigen::MatrixXd& F_alpha_deriv,
    const Eigen::MatrixXd& F_beta_deriv,
    const Eigen::MatrixXd& S
);

// ============================================================================
// Analytical Gradient Classes
// ============================================================================

/**
 * @brief RHF analytical gradient calculator
 * 
 * Computes ∂E/∂R_A analytically for closed-shell RHF.
 * 
 * Algorithm:
 * 1. Compute density matrix P and MO coefficients C from converged SCF
 * 2. For each atom A and direction α:
 *    a. Compute ∂S/∂R_A and ∂H/∂R_A (integral derivatives)
 *    b. Compute W via Z-vector method (orbital relaxation)
 *    c. Compute ∂(μν|λσ)/∂R_A contribution
 *    d. Assemble: ∂E/∂R_A = Tr[W·∂S] + Tr[P·∂H] + ∂(ERI) + ∂V_nuc
 * 
 * Cost: O(N^4) same as SCF, but ~10× faster than numerical (6N+1 SCFs)
 */
class RHFAnalyticalGradient {
public:
    /**
     * @brief Construct RHF analytical gradient calculator
     * @param mol Molecule
     * @param basis Basis set
     * @param integrals Integral engine
     * @param scf_result Converged RHF result (need P, C, eps)
     */
    RHFAnalyticalGradient(
        const Molecule& mol,
        const BasisSet& basis,
        std::shared_ptr<IntegralEngine> integrals,
        const SCFResult& scf_result
    );
    
    /**
     * @brief Compute analytical gradient
     * @return Gradient result (3N vector)
     */
    GradientResult compute();
    
private:
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    SCFResult scf_result_;
    
    Eigen::MatrixXd P_;    ///< Density matrix
    Eigen::MatrixXd W_;    ///< Energy-weighted density (from Z-vector)
    
    /**
     * @brief Compute gradient for one atom
     */
    Eigen::Vector3d compute_atom_gradient(int atom_idx);
    
    /**
     * @brief Compute nuclear repulsion gradient
     */
    Eigen::Vector3d compute_nuclear_gradient(int atom_idx);
    
    /**
     * @brief Compute electronic gradient contribution
     */
    Eigen::Vector3d compute_electronic_gradient(int atom_idx);
};

/**
 * @brief UHF analytical gradient calculator
 * 
 * Similar to RHF but handles separate alpha/beta densities and orbitals.
 */
class UHFAnalyticalGradient {
public:
    UHFAnalyticalGradient(
        const Molecule& mol,
        const BasisSet& basis,
        std::shared_ptr<IntegralEngine> integrals,
        const SCFResult& scf_result
    );
    
    GradientResult compute();
    
private:
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    SCFResult scf_result_;
    
    Eigen::MatrixXd P_alpha_;
    Eigen::MatrixXd P_beta_;
    Eigen::MatrixXd W_alpha_;
    Eigen::MatrixXd W_beta_;
    
    Eigen::Vector3d compute_atom_gradient(int atom_idx);
    Eigen::Vector3d compute_nuclear_gradient(int atom_idx);
    Eigen::Vector3d compute_electronic_gradient(int atom_idx);
};

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * @brief Compute RHF analytical gradient (convenience wrapper)
 * 
 * Runs RHF SCF to convergence, then computes analytical gradient.
 * 
 * @param mol Molecule
 * @param basis Basis set
 * @param integrals Integral engine
 * @param config SCF configuration
 * @return Gradient result
 */
GradientResult compute_rhf_gradient_analytical(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    const SCFConfig& config = SCFConfig()
);

/**
 * @brief Compute UHF analytical gradient
 */
GradientResult compute_uhf_gradient_analytical(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    int n_alpha,
    int n_beta,
    const SCFConfig& config = SCFConfig()
);

/**
 * @brief Validate analytical gradient against numerical
 * 
 * Computes both analytical and numerical gradients and compares.
 * Useful for debugging and validation.
 * 
 * @return Maximum absolute difference between analytical and numerical
 */
double validate_analytical_gradient(
    const Molecule& mol,
    const BasisSet& basis,
    const std::string& method,  // "RHF" or "UHF"
    double numerical_step = 1e-5
);

} // namespace gradient
} // namespace mshqc

#endif // MSHQC_ANALYTICAL_GRADIENT_H
