/**
 * @file gradient.h
 * @brief Molecular gradient calculations for geometry optimization
 * 
 * Implementation of numerical gradients via finite differences for all
 * electronic structure methods. Analytical gradients interface defined
 * for future method-specific implementations.
 * 
 * THEORY BACKGROUND:
 * Molecular gradients describe how energy changes with nuclear positions:
 * ∇E = ∂E/∂R_A for each atom A
 * 
 * Two approaches:
 * 1. Numerical gradients: Finite difference ∂E/∂x ≈ [E(x+δ) - E(x-δ)]/(2δ)
 * 2. Analytical gradients: Explicit derivative formulas (method-specific)
 * 
 * THEORY REFERENCES:
 * [1] P. Pulay, Mol. Phys. **17**, 197 (1969)
 *     - Analytic energy derivatives (original force method)
 * 
 * [2] J. A. Pople et al., Int. J. Quantum Chem. Symp. **13**, 225 (1979)
 *     - Derivative methods in quantum chemistry
 * 
 * [3] T. Helgaker & P. Jørgensen, Methods Comput. Chem. **3**, 1 (1988)
 *     - Comprehensive review of gradient theory
 * 
 * [4] P. J. Knowles & H.-J. Werner, Chem. Phys. Lett. **145**, 514 (1988)
 *     - CASSCF analytical gradients
 * 
 * TEXTBOOK:
 * [5] T. Helgaker et al., "Molecular Electronic-Structure Theory" (2000)
 *     Chapter 13: Molecular Properties
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 * @license MIT (recommended)
 * 
 * @note Original implementation derived from published theory.
 *       No code copied from Psi4, PySCF, or other software.
 *       Finite difference formulas from standard numerical analysis.
 */

#ifndef MSHQC_GRADIENT_H
#define MSHQC_GRADIENT_H

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <string>

namespace mshqc {
namespace gradient {

// ============================================================================
// Gradient Results
// ============================================================================

/**
 * @brief Results from gradient calculation
 */
struct GradientResult {
    /// Gradient vector (3N components: x,y,z for each atom)
    /// Layout: [∂E/∂x₁, ∂E/∂y₁, ∂E/∂z₁, ∂E/∂x₂, ...]
    Eigen::VectorXd gradient;
    
    /// Energy at current geometry (Ha)
    double energy;
    
    /// RMS gradient norm ||∇E||/√(3N)
    double rms_gradient;
    
    /// Maximum absolute gradient component
    double max_gradient;
    
    /// Method used (e.g., "RHF", "MP2", "CASSCF")
    std::string method;
    
    /// Whether gradient was computed analytically or numerically
    bool is_analytical;
    
    /// Gradient by atom (N × 3 matrix for convenience)
    /// Row i: [∂E/∂x_i, ∂E/∂y_i, ∂E/∂z_i]
    Eigen::MatrixXd gradient_by_atom;
    
    /**
     * @brief Convert flat gradient to per-atom format
     * @param natoms Number of atoms
     */
    void populate_gradient_by_atom(int natoms);
};

// ============================================================================
// Energy Function Type
// ============================================================================

/**
 * @brief Function signature for energy calculation
 * 
 * Takes a molecule and returns its energy (Ha)
 * Used for numerical gradients with finite differences
 * 
 * Example:
 * @code
 * auto energy_func = [&](const Molecule& mol) -> double {
 *     RHF rhf(mol, basis, integrals, 0, config);
 *     auto result = rhf.compute();
 *     return result.energy_total;
 * };
 * @endcode
 */
using EnergyFunction = std::function<double(const Molecule&)>;

// ============================================================================
// Gradient Calculator
// ============================================================================

/**
 * @brief Numerical gradient calculator using finite differences
 * 
 * THEORY:
 * For each nuclear coordinate x, the gradient is approximated as:
 * 
 * Central difference (default, O(δ²) error):
 *   ∂E/∂x ≈ [E(x+δ) - E(x-δ)] / (2δ)
 * 
 * Forward difference (O(δ) error, less accurate):
 *   ∂E/∂x ≈ [E(x+δ) - E(x)] / δ
 * 
 * REFERENCE: Press et al., "Numerical Recipes" (2007), Section 5.7
 * 
 * OPTIMAL STEP SIZE:
 * For central differences with double precision (~10⁻¹⁶ machine epsilon):
 * δ_opt ≈ ε^(1/3) ≈ 10⁻⁵ au (≈ 0.0003 Å)
 * 
 * REFERENCE: Pople et al., J. Comput. Chem. 3, 468 (1982), Eq. (5)
 * 
 * PERFORMANCE:
 * - Requires 2×3N energy evaluations (central diff)
 * - Can be parallelized trivially (each displacement independent)
 * - Memory: O(N) for storing molecule copies
 */
class NumericalGradient {
public:
    /**
     * @brief Construct numerical gradient calculator
     * @param energy_func Function that computes energy for given molecule
     * @param delta Step size for finite difference (au, default: 1e-5)
     * @param use_central Use central difference (true) or forward (false)
     */
    explicit NumericalGradient(
        EnergyFunction energy_func,
        double delta = 1e-5,
        bool use_central = true
    );
    
    /**
     * @brief Compute numerical gradient at current geometry
     * @param mol Molecule at geometry to compute gradient
     * @return GradientResult with gradient vector and statistics
     * 
     * ALGORITHM:
     * For each atom A and Cartesian component x,y,z:
     *   1. Displace R_A by +δ along component
     *   2. Compute E₊ = E(R_A + δ)
     *   3. Displace R_A by -δ along component  
     *   4. Compute E₋ = E(R_A - δ)
     *   5. Gradient: ∂E/∂R_A = (E₊ - E₋)/(2δ)
     * 
     * Total evaluations: 1 (E₀) + 6N (for 3N coordinates × 2 directions)
     */
    GradientResult compute(const Molecule& mol);
    
    /**
     * @brief Compute gradient for single coordinate (debugging)
     * @param mol Molecule
     * @param atom_index Atom to displace (0-based)
     * @param coord_index Coordinate: 0=x, 1=y, 2=z
     * @return Gradient component ∂E/∂R_A
     */
    double compute_component(const Molecule& mol, int atom_index, int coord_index);
    
    /**
     * @brief Set step size for finite difference
     * @param delta Step size (au)
     */
    void set_delta(double delta) { delta_ = delta; }
    
    /**
     * @brief Get current step size
     */
    double get_delta() const { return delta_; }
    
    /**
     * @brief Set whether to use central vs forward difference
     * @param use_central True for central difference, false for forward
     */
    void set_use_central(bool use_central) { use_central_ = use_central; }
    
private:
    EnergyFunction energy_func_;  ///< Function to compute energy
    double delta_;                ///< Finite difference step size (au)
    bool use_central_;            ///< Use central (true) or forward (false) diff
    
    /**
     * @brief Displace molecule along one coordinate
     * @param mol Original molecule
     * @param atom_idx Atom to displace
     * @param coord_idx Coordinate (0=x, 1=y, 2=z)
     * @param displacement Displacement amount (au)
     * @return New molecule with displaced atom
     */
    Molecule displace_coordinate(
        const Molecule& mol,
        int atom_idx,
        int coord_idx,
        double displacement
    ) const;
};

// ============================================================================
// Analytical Gradient Interface (Future)
// ============================================================================

/**
 * @brief Base class for analytical gradient implementations
 * 
 * Each method (RHF, MP2, CASSCF) will derive from this
 * and implement compute_analytical_gradient()
 * 
 * STATUS: Interface defined, implementations to follow
 */
class AnalyticalGradient {
public:
    virtual ~AnalyticalGradient() = default;
    
    /**
     * @brief Compute analytical gradient (method-specific)
     * @param mol Molecule
     * @return Gradient result
     * 
     * To be implemented by derived classes:
     * - RHFGradient: HF gradient via coupled-perturbed equations
     * - MP2Gradient: MP2 gradient (density matrix approach)
     * - CASSCFGradient: MCSCF gradient (CP-MCSCF)
     */
    virtual GradientResult compute(const Molecule& mol) = 0;
    
protected:
    /**
     * @brief Compute nuclear repulsion gradient
     * ∇V_nn = Σ_{A<B} Z_A Z_B (R_A - R_B)/|R_A - R_B|³
     * 
     * REFERENCE: Helgaker et al. (2000), Eq. (13.2.3)
     */
    static Eigen::VectorXd compute_nuclear_gradient(const Molecule& mol);
};

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * @brief Compute numerical gradient for RHF
 * @param mol Molecule
 * @param basis Basis set
 * @param integrals Integral engine
 * @param charge Molecular charge
 * @param config SCF configuration
 * @param delta Finite difference step size (au)
 * @return Gradient result
 */
GradientResult compute_rhf_gradient_numerical(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    int charge,
    const SCFConfig& config = SCFConfig(),
    double delta = 1e-5
);

/**
 * @brief Compute numerical gradient for UHF
 */
GradientResult compute_uhf_gradient_numerical(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    int charge,
    int multiplicity,
    const SCFConfig& config = SCFConfig(),
    double delta = 1e-5
);

/**
 * @brief Print gradient result in readable format
 * @param result Gradient result
 * @param mol Molecule (for atom labels)
 */
void print_gradient(const GradientResult& result, const Molecule& mol);

/**
 * @brief Check if gradient is converged for optimization
 * @param gradient Gradient vector
 * @param rms_threshold RMS gradient threshold (Ha/au, default: 3e-4)
 * @param max_threshold Maximum gradient threshold (Ha/au, default: 4.5e-4)
 * @return True if converged
 * 
 * Standard thresholds (Baker, J. Comp. Chem. 7, 385 (1986)):
 * - RMS gradient: 3×10⁻⁴ Ha/bohr
 * - Max gradient: 4.5×10⁻⁴ Ha/bohr
 */
bool is_gradient_converged(
    const Eigen::VectorXd& gradient,
    double rms_threshold = 3e-4,
    double max_threshold = 4.5e-4
);

} // namespace gradient
} // namespace mshqc

#endif // MSHQC_GRADIENT_H
