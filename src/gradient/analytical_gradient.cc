/**
 * @file analytical_gradient.cc
 * @brief Implementation of analytical SCF gradients
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 * 
 * @note Skeleton implementation with TODOs for detailed integral derivatives.
 *       Core structure complete, integral derivative implementations require
 *       Obara-Saika or McMurchie-Davidson recursion (future work).
 */

#include "mshqc/gradient/analytical_gradient.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace mshqc {
namespace gradient {

// ============================================================================
// Integral Derivatives
// ============================================================================

Eigen::MatrixXd compute_overlap_derivative(
    const BasisSet& basis,
    const Molecule& mol,
    int atom_idx,
    int coord
) {
    // Compute ∂S/∂R_A for overlap matrix
    //
    // THEORY: S_μν = ⟨μ|ν⟩
    // ∂S_μν/∂R_A = ⟨∂μ/∂R_A|ν⟩ + ⟨μ|∂ν/∂R_A⟩
    //
    // Uses translational invariance:
    // ∂S_μν/∂R_A = 0 if neither μ nor ν on atom A
    //
    // REFERENCE: Helgaker et al. (2000), Eq. (13.2.15)
    
    size_t nbasis = basis.n_basis_functions();
    Eigen::MatrixXd dS = Eigen::MatrixXd::Zero(nbasis, nbasis);
    
    // TODO: Implement overlap derivative using Obara-Saika recursion
    // For now, return zero matrix (analytical gradient will not work correctly)
    //
    // Implementation requires:
    // 1. Identify shells on atom A
    // 2. For each pair of shells (μ,ν) involving atom A:
    //    a. Apply Gaussian derivative rules: ∂/∂R_A exp(-α|r-R_A|²)
    //    b. Use recursion relations for derivative integrals
    // 3. Use translational invariance to reduce computation
    
    std::cerr << "WARNING: compute_overlap_derivative not fully implemented\n";
    std::cerr << "         Using placeholder (returns zero matrix)\n";
    
    return dS;
}

Eigen::MatrixXd compute_core_hamiltonian_derivative(
    const BasisSet& basis,
    const Molecule& mol,
    int atom_idx,
    int coord
) {
    // Compute ∂H/∂R_A where H = T + V_ne
    //
    // ∂H/∂R_A = ∂T/∂R_A + ∂V_ne/∂R_A
    //
    // Kinetic: ∂T_μν/∂R_A = ⟨∂μ/∂R_A|∇²|ν⟩ + ⟨μ|∇²|∂ν/∂R_A⟩
    // Nuclear: ∂V_μν/∂R_A involves both nucleus movement and basis function movement
    //
    // REFERENCE: Helgaker et al. (2000), Eq. (13.2.16)-(13.2.17)
    
    size_t nbasis = basis.n_basis_functions();
    Eigen::MatrixXd dH = Eigen::MatrixXd::Zero(nbasis, nbasis);
    
    // TODO: Implement kinetic + nuclear attraction derivatives
    //
    // Kinetic derivative:
    // Use relation: ∂²/∂R_A = -∂²/∂r (translational invariance)
    // Can reuse overlap derivative machinery with additional ∇² operator
    //
    // Nuclear attraction derivative has two contributions:
    // 1. Derivative of basis functions: ∂⟨μ|-Z/|r-R_B|ν⟩/∂R_A
    // 2. Derivative of nucleus position (only if B=A): ∂(-Z/|r-R_A|)/∂R_A
    
    std::cerr << "WARNING: compute_core_hamiltonian_derivative not fully implemented\n";
    std::cerr << "         Using placeholder (returns zero matrix)\n";
    
    return dH;
}

double compute_eri_derivative_contribution(
    const BasisSet& basis,
    const Molecule& mol,
    const Eigen::MatrixXd& density_matrix,
    int atom_idx,
    int coord
) {
    // Compute ERI derivative contribution to gradient
    //
    // Returns: Σ_μνλσ G_μνλσ ∂(μν|λσ)/∂R_A
    //
    // where G is two-electron density matrix:
    // G_μνλσ = P_μλ P_νσ - 0.5 P_μσ P_νλ (for RHF)
    //
    // REFERENCE: Gill et al. (1994), Adv. Quantum Chem. 25, 141
    
    double contrib = 0.0;
    
    // TODO: Implement ERI derivative
    //
    // Most expensive part of analytical gradient (O(N⁴) loop)
    // Requires:
    // 1. Identify quartets (μνλσ) involving atom A
    // 2. Compute ∂(μν|λσ)/∂R_A using:
    //    - McMurchie-Davidson recursion, or
    //    - Obara-Saika recursion, or
    //    - Rys quadrature with derivatives
    // 3. Contract with two-electron density matrix
    // 4. Use permutational symmetry (8-fold for real basis)
    // 5. Use translational invariance to reduce derivatives
    //
    // Optimization: Can use Cauchy-Schwarz screening
    
    std::cerr << "WARNING: compute_eri_derivative_contribution not fully implemented\n";
    std::cerr << "         Using placeholder (returns 0.0)\n";
    
    return contrib;
}

// ============================================================================
// Z-Vector Method
// ============================================================================

Eigen::MatrixXd compute_z_vector_rhf(
    const Eigen::MatrixXd& C,
    const Eigen::VectorXd& eps,
    int n_occ,
    const Eigen::MatrixXd& F_deriv,
    const Eigen::MatrixXd& S
) {
    // Compute energy-weighted density W via Z-vector method
    //
    // Solves CPHF equations implicitly to avoid computing ∂C/∂R
    //
    // THEORY:
    // W_μν = Σ_i^occ Σ_a^virt U_ia/(ε_a - ε_i) C_μi C_νa
    //
    // where U_ia satisfies:
    // (ε_a - ε_i)U_ia + Σ_jb [(ia|jb) - (ij|ab)] U_jb = -F_ia^[1]
    //
    // F^[1] = derivative of Fock matrix at fixed orbitals
    //
    // REFERENCE: Handy & Schaefer, J. Chem. Phys. 81, 5031 (1984)
    
    int nbasis = C.rows();
    int nvirt = nbasis - n_occ;
    
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(nbasis, nbasis);
    
    if (nvirt <= 0) {
        // No virtual orbitals (shouldn't happen for reasonable basis)
        return W;
    }
    
    // TODO: Implement Z-vector solver
    //
    // Algorithm:
    // 1. Transform F_deriv to MO basis: F_ia^[1] = C_i^T F_deriv C_a
    // 2. Build CPHF matrix A_iajb = (ε_a - ε_i)δ_ij δ_ab + 2(ia|jb) - (ij|ab)
    // 3. Solve linear system: A·U = -F^[1]
    //    - Can use iterative solver (conjugate gradient)
    //    - Exploit structure: (ε_a - ε_i) diagonal dominance
    // 4. Construct W from U: W = Σ_ia U_ia C_i C_a^T
    //
    // Simplified approach (if iterative solver not available):
    // W ≈ -P·F_deriv·P (assumes diagonal approximation)
    // This is approximate but better than nothing
    
    // Use simplified diagonal approximation as fallback
    Eigen::MatrixXd P = 2.0 * C.leftCols(n_occ) * C.leftCols(n_occ).transpose();
    W = -P * F_deriv * P;
    
    std::cerr << "WARNING: compute_z_vector_rhf using diagonal approximation\n";
    std::cerr << "         Full CPHF solver not yet implemented\n";
    
    return W;
}

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
) {
    // UHF Z-vector: separate alpha and beta
    //
    // Similar to RHF but independent equations for each spin
    
    int nbasis = C_alpha.rows();
    
    // Simplified diagonal approximation
    Eigen::MatrixXd P_alpha = C_alpha.leftCols(n_alpha) * C_alpha.leftCols(n_alpha).transpose();
    Eigen::MatrixXd P_beta = C_beta.leftCols(n_beta) * C_beta.leftCols(n_beta).transpose();
    
    Eigen::MatrixXd W_alpha = -P_alpha * F_alpha_deriv * P_alpha;
    Eigen::MatrixXd W_beta = -P_beta * F_beta_deriv * P_beta;
    
    std::cerr << "WARNING: compute_z_vector_uhf using diagonal approximation\n";
    
    return {W_alpha, W_beta};
}

// ============================================================================
// RHF Analytical Gradient
// ============================================================================

RHFAnalyticalGradient::RHFAnalyticalGradient(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    const SCFResult& scf_result
) : mol_(mol), basis_(basis), integrals_(integrals), scf_result_(scf_result) {
    
    // Extract density matrix from SCF result
    int n_occ = scf_result.C_alpha.cols() / 2;  // Assuming closed-shell
    P_ = 2.0 * scf_result.C_alpha.leftCols(n_occ) * 
         scf_result.C_alpha.leftCols(n_occ).transpose();
    
    // Will compute W_ on-demand for each nuclear displacement
    W_ = Eigen::MatrixXd::Zero(P_.rows(), P_.cols());
}

GradientResult RHFAnalyticalGradient::compute() {
    int natoms = mol_.n_atoms();
    Eigen::VectorXd gradient(3 * natoms);
    
    std::cout << "\n========================================\n";
    std::cout << "  Analytical Gradient Calculation (RHF)\n";
    std::cout << "========================================\n";
    std::cout << "Method: RHF analytical\n";
    std::cout << "Atoms:  " << natoms << "\n";
    std::cout << "========================================\n\n";
    
    // Compute gradient for each atom
    for (int atom = 0; atom < natoms; ++atom) {
        Eigen::Vector3d grad_atom = compute_atom_gradient(atom);
        
        gradient(3*atom + 0) = grad_atom(0);
        gradient(3*atom + 1) = grad_atom(1);
        gradient(3*atom + 2) = grad_atom(2);
        
        std::cout << "Atom " << (atom+1) << "  "
                  << std::scientific << std::setprecision(6)
                  << grad_atom(0) << "  "
                  << grad_atom(1) << "  "
                  << grad_atom(2) << "\n";
    }
    
    // Build result
    GradientResult result;
    result.gradient = gradient;
    result.energy = scf_result_.energy_total;
    result.method = "RHF (analytical)";
    result.is_analytical = true;
    result.populate_gradient_by_atom(natoms);
    
    double rms = std::sqrt(gradient.squaredNorm() / (3 * natoms));
    double max_component = gradient.cwiseAbs().maxCoeff();
    result.rms_gradient = rms;
    result.max_gradient = max_component;
    
    std::cout << "\n";
    std::cout << "RMS gradient: " << std::scientific << std::setprecision(4) 
              << rms << " Ha/bohr\n";
    std::cout << "Max gradient: " << max_component << " Ha/bohr\n";
    std::cout << "========================================\n\n";
    
    return result;
}

Eigen::Vector3d RHFAnalyticalGradient::compute_atom_gradient(int atom_idx) {
    Eigen::Vector3d grad = Eigen::Vector3d::Zero();
    
    // Nuclear repulsion gradient (always analytical)
    Eigen::Vector3d grad_nuc = compute_nuclear_gradient(atom_idx);
    
    // Electronic gradient (requires integral derivatives)
    Eigen::Vector3d grad_elec = compute_electronic_gradient(atom_idx);
    
    grad = grad_nuc + grad_elec;
    
    return grad;
}

Eigen::Vector3d RHFAnalyticalGradient::compute_nuclear_gradient(int atom_idx) {
    // Nuclear repulsion gradient: ∂V_nn/∂R_A
    //
    // V_nn = Σ_{A<B} Z_A Z_B / R_AB
    // ∂V_nn/∂R_A = Σ_{B≠A} Z_A Z_B (R_A - R_B) / R_AB³
    //
    // REFERENCE: Helgaker et al. (2000), Eq. (13.2.3)
    
    Eigen::Vector3d grad_nuc = Eigen::Vector3d::Zero();
    
    const auto& atom_A = mol_.atom(atom_idx);
    double Z_A = atom_A.atomic_number;
    Eigen::Vector3d R_A(atom_A.x, atom_A.y, atom_A.z);
    
    int natoms = mol_.n_atoms();
    for (int B = 0; B < natoms; ++B) {
        if (B == atom_idx) continue;
        
        const auto& atom_B = mol_.atom(B);
        double Z_B = atom_B.atomic_number;
        Eigen::Vector3d R_B(atom_B.x, atom_B.y, atom_B.z);
        
        Eigen::Vector3d R_AB = R_A - R_B;
        double dist = R_AB.norm();
        double dist3 = dist * dist * dist;
        
        grad_nuc += Z_A * Z_B * R_AB / dist3;
    }
    
    return grad_nuc;
}

Eigen::Vector3d RHFAnalyticalGradient::compute_electronic_gradient(int atom_idx) {
    // Electronic gradient: ∂E_elec/∂R_A
    //
    // ∂E_elec/∂R_A = Tr[W·∂S/∂R_A] + Tr[P·∂H/∂R_A] + ∂(ERI)/∂R_A
    //
    // REFERENCE: Helgaker et al. (2000), Eq. (13.1.1)
    
    Eigen::Vector3d grad_elec = Eigen::Vector3d::Zero();
    
    for (int coord = 0; coord < 3; ++coord) {
        // Overlap derivative contribution
        Eigen::MatrixXd dS = compute_overlap_derivative(basis_, mol_, atom_idx, coord);
        double overlap_contrib = (W_.array() * dS.array()).sum();
        
        // Core Hamiltonian derivative contribution
        Eigen::MatrixXd dH = compute_core_hamiltonian_derivative(basis_, mol_, atom_idx, coord);
        double core_contrib = (P_.array() * dH.array()).sum();
        
        // ERI derivative contribution
        double eri_contrib = compute_eri_derivative_contribution(basis_, mol_, P_, atom_idx, coord);
        
        grad_elec(coord) = overlap_contrib + core_contrib + eri_contrib;
    }
    
    return grad_elec;
}

// ============================================================================
// UHF Analytical Gradient
// ============================================================================

UHFAnalyticalGradient::UHFAnalyticalGradient(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    const SCFResult& scf_result
) : mol_(mol), basis_(basis), integrals_(integrals), scf_result_(scf_result) {
    
    // TODO: Extract alpha/beta densities from UHF SCFResult
    // For now, use placeholder
    
    std::cerr << "WARNING: UHFAnalyticalGradient not fully implemented\n";
}

GradientResult UHFAnalyticalGradient::compute() {
    GradientResult result;
    result.method = "UHF (analytical) - NOT IMPLEMENTED";
    result.is_analytical = true;
    
    std::cerr << "ERROR: UHF analytical gradient not yet implemented\n";
    
    return result;
}

Eigen::Vector3d UHFAnalyticalGradient::compute_atom_gradient(int atom_idx) {
    return Eigen::Vector3d::Zero();
}

Eigen::Vector3d UHFAnalyticalGradient::compute_nuclear_gradient(int atom_idx) {
    return Eigen::Vector3d::Zero();
}

Eigen::Vector3d UHFAnalyticalGradient::compute_electronic_gradient(int atom_idx) {
    return Eigen::Vector3d::Zero();
}

// ============================================================================
// Convenience Functions
// ============================================================================

GradientResult compute_rhf_gradient_analytical(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    const SCFConfig& config
) {
    // Run RHF to convergence
    RHF rhf(mol, basis, integrals, config);
    auto scf_result = rhf.compute();
    
    if (!scf_result.converged) {
        std::cerr << "WARNING: SCF did not converge\n";
    }
    
    // Compute analytical gradient
    RHFAnalyticalGradient grad_calc(mol, basis, integrals, scf_result);
    return grad_calc.compute();
}

GradientResult compute_uhf_gradient_analytical(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    int n_alpha,
    int n_beta,
    const SCFConfig& config
) {
    // Run UHF to convergence
    UHF uhf(mol, basis, integrals, n_alpha, n_beta, config);
    auto scf_result = uhf.compute();
    
    if (!scf_result.converged) {
        std::cerr << "WARNING: SCF did not converge\n";
    }
    
    // Compute analytical gradient
    UHFAnalyticalGradient grad_calc(mol, basis, integrals, scf_result);
    return grad_calc.compute();
}

double validate_analytical_gradient(
    const Molecule& mol,
    const BasisSet& basis,
    const std::string& method,
    double numerical_step
) {
    std::cout << "\n========================================\n";
    std::cout << "  Gradient Validation\n";
    std::cout << "========================================\n";
    std::cout << "Method: " << method << "\n";
    std::cout << "Comparing analytical vs numerical\n";
    std::cout << "Numerical step: " << numerical_step << " au\n";
    std::cout << "========================================\n\n";
    
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    SCFConfig config;
    config.print_level = 0;
    
    GradientResult grad_analytical, grad_numerical;
    
    if (method == "RHF") {
        grad_analytical = compute_rhf_gradient_analytical(mol, basis, integrals, config);
        grad_numerical = compute_rhf_gradient_numerical(mol, basis, integrals, 0, config, numerical_step);
    } else {
        std::cerr << "ERROR: Only RHF validation currently supported\n";
        return -1.0;
    }
    
    // Compare gradients
    Eigen::VectorXd diff = grad_analytical.gradient - grad_numerical.gradient;
    double max_diff = diff.cwiseAbs().maxCoeff();
    double rms_diff = std::sqrt(diff.squaredNorm() / diff.size());
    
    std::cout << "\nComparison:\n";
    std::cout << "  Max difference: " << std::scientific << std::setprecision(4) 
              << max_diff << " Ha/bohr\n";
    std::cout << "  RMS difference: " << rms_diff << " Ha/bohr\n";
    std::cout << "\n";
    
    if (max_diff < 1e-5) {
        std::cout << "✓ Analytical gradient VALIDATED (excellent agreement)\n";
    } else if (max_diff < 1e-4) {
        std::cout << "⚠ Analytical gradient acceptable (good agreement)\n";
    } else {
        std::cout << "✗ Analytical gradient FAILED (poor agreement)\n";
        std::cout << "  Note: Integral derivatives not fully implemented\n";
    }
    
    std::cout << "========================================\n\n";
    
    return max_diff;
}

} // namespace gradient
} // namespace mshqc
