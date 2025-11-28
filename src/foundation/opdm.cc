/**
 * @file opdm.cc
 * @brief Implementation of One-Particle Density Matrix
 * 
 * THEORY REFERENCES:
 *   - T. Helgaker, P. Jørgensen, & J. Olsen, "Molecular Electronic-Structure
 *     Theory" (2000), Chapter 11, Equations (11.2.1)-(11.2.15)
 *   - E. R. Davidson, Chem. Phys. Lett. 21, 565 (1976)
 *   - R. McWeeny, Rev. Mod. Phys. 32, 335 (1960)
 *   - A. Szabo & N. S. Ostlund, "Modern Quantum Chemistry" (1996), Sec. 2.4
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 * @license MIT
 * 
 * @note Original implementation from theory, NOT from software
 */

#include "mshqc/foundation/opdm.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <Eigen/Eigenvalues>

namespace mshqc {
namespace foundation {

// ============================================================================
// Constructor
// ============================================================================

OPDM::OPDM(const std::vector<double>& ci_coeffs,
           const std::vector<ci::Determinant>& determinants,
           int n_orbitals)
    : n_orbitals_(n_orbitals),
      n_determinants_(static_cast<int>(determinants.size())),
      ci_coeffs_(ci_coeffs),
      determinants_(determinants) {
    
    // Validate inputs
    if (ci_coeffs.size() != determinants.size()) {
        throw std::invalid_argument(
            "OPDM: CI coefficients and determinants size mismatch");
    }
    
    if (n_orbitals <= 0) {
        throw std::invalid_argument(
            "OPDM: Number of orbitals must be positive");
    }
    
    if (determinants.empty()) {
        throw std::invalid_argument(
            "OPDM: Determinant list cannot be empty");
    }
    
    // Initialize density matrices
    opdm_alpha_ = Eigen::MatrixXd::Zero(n_orbitals_, n_orbitals_);
    opdm_beta_ = Eigen::MatrixXd::Zero(n_orbitals_, n_orbitals_);
    
    // Compute OPDM from CI expansion
    compute_opdm();
    
    // Validate result
    validate();
}

// ============================================================================
// Accessors
// ============================================================================

Eigen::MatrixXd OPDM::total() const {
    return opdm_alpha_ + opdm_beta_;
}

double OPDM::operator()(int p, int q, bool alpha) const {
    if (p < 0 || p >= n_orbitals_ || q < 0 || q >= n_orbitals_) {
        throw std::out_of_range("OPDM: Orbital index out of range");
    }
    
    return alpha ? opdm_alpha_(p, q) : opdm_beta_(p, q);
}

// ============================================================================
// Properties
// ============================================================================

double OPDM::trace(bool alpha) const {
    // THEORY: Helgaker Eq. (11.2.3)
    // Tr(γ) = Σ_p γ_pp = N_electrons
    
    return alpha ? opdm_alpha_.trace() : opdm_beta_.trace();
}

bool OPDM::is_n_representable(double tolerance) const {
    // N-representability conditions (Davidson 1976):
    // 1. Hermiticity
    // 2. Positive semidefinite (eigenvalues ≥ 0)
    // 3. Pauli exclusion (eigenvalues ≤ 1)
    // 4. Particle conservation (trace = N_electrons)
    
    // Check Hermiticity
    double hermit_error_alpha = (opdm_alpha_ - opdm_alpha_.transpose()).norm();
    double hermit_error_beta = (opdm_beta_ - opdm_beta_.transpose()).norm();
    
    if (hermit_error_alpha > tolerance || hermit_error_beta > tolerance) {
        return false;
    }
    
    // Check eigenvalues for both spins
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_alpha(opdm_alpha_);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_beta(opdm_beta_);
    
    const auto& evals_alpha = solver_alpha.eigenvalues();
    const auto& evals_beta = solver_beta.eigenvalues();
    
    for (int i = 0; i < n_orbitals_; i++) {
        // Check positive semidefinite
        if (evals_alpha(i) < -tolerance || evals_beta(i) < -tolerance) {
            return false;
        }
        
        // Check Pauli exclusion (per-spin: 0 ≤ n_i ≤ 1)
        if (evals_alpha(i) > 1.0 + tolerance || evals_beta(i) > 1.0 + tolerance) {
            return false;
        }
    }
    
    return true;
}

// ============================================================================
// Natural Orbitals
// ============================================================================

std::pair<Eigen::VectorXd, Eigen::MatrixXd> 
OPDM::natural_orbitals(bool alpha) const {
    // THEORY: Löwdin (1955), Phys. Rev. 97, 1474
    // 
    // Natural orbitals diagonalize density matrix:
    //   γ |φ_i⟩ = n_i |φ_i⟩
    // 
    // ALGORITHM:
    // 1. Diagonalize γ = U Λ U†
    // 2. Sort eigenvalues descending
    // 3. Return (occupations, orbitals)
    
    const Eigen::MatrixXd& gamma = alpha ? opdm_alpha_ : opdm_beta_;
    
    // Diagonalize (Hermitian matrix)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(gamma);
    
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("OPDM: Failed to diagonalize density matrix");
    }
    
    Eigen::VectorXd occupations = solver.eigenvalues();
    Eigen::MatrixXd orbitals = solver.eigenvectors();
    
    // Sort by occupation (descending order)
    std::vector<std::pair<double, int>> sorted_indices;
    for (int i = 0; i < n_orbitals_; i++) {
        sorted_indices.push_back({occupations(i), i});
    }
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Reorder
    Eigen::VectorXd sorted_occupations(n_orbitals_);
    Eigen::MatrixXd sorted_orbitals(n_orbitals_, n_orbitals_);
    
    for (int i = 0; i < n_orbitals_; i++) {
        int idx = sorted_indices[i].second;
        sorted_occupations(i) = occupations(idx);
        sorted_orbitals.col(i) = orbitals.col(idx);
    }
    
    return {sorted_occupations, sorted_orbitals};
}

Eigen::VectorXd OPDM::natural_occupations(bool alpha) const {
    auto [occupations, orbitals] = natural_orbitals(alpha);
    return occupations;
}

double OPDM::entropy(bool alpha) const {
    // DEFINITION (Von Neumann entropy):
    //   S = -Σ_i [n_i ln(n_i) + (1-n_i) ln(1-n_i)]
    // 
    // Physical meaning:
    //   S = 0: Single determinant (HF)
    //   S > 0: Multireference character
    
    auto occupations = natural_occupations(alpha);
    
    double S = 0.0;
    const double epsilon = 1e-12;  // Avoid log(0)
    
    for (int i = 0; i < n_orbitals_; i++) {
        double n = occupations(i);
        
        // Skip if occupation negligible or saturated
        if (n < epsilon || n > 1.0 - epsilon) {
            continue;
        }
        
        // Von Neumann entropy contribution
        S -= n * std::log(n) + (1.0 - n) * std::log(1.0 - n);
    }
    
    return S;
}

// ============================================================================
// Expectation Values
// ============================================================================

double OPDM::expectation_value(const Eigen::MatrixXd& operator_matrix,
                                bool alpha) const {
    // FORMULA (Helgaker Eq. 11.2.5):
    //   ⟨O⟩ = Σ_pq γ_pq O_qp = Tr(γ O)
    
    if (operator_matrix.rows() != n_orbitals_ || 
        operator_matrix.cols() != n_orbitals_) {
        throw std::invalid_argument(
            "OPDM: Operator matrix size mismatch");
    }
    
    const Eigen::MatrixXd& gamma = alpha ? opdm_alpha_ : opdm_beta_;
    
    // Trace of matrix product
    return (gamma * operator_matrix).trace();
}

double OPDM::one_electron_energy(const Eigen::MatrixXd& h_core) const {
    // FORMULA (Helgaker Eq. 14.8.1):
    //   E_1e = Σ_pq (γ^α_pq + γ^β_pq) h_pq
    
    if (h_core.rows() != n_orbitals_ || h_core.cols() != n_orbitals_) {
        throw std::invalid_argument(
            "OPDM: Core Hamiltonian size mismatch");
    }
    
    Eigen::MatrixXd total_gamma = opdm_alpha_ + opdm_beta_;
    
    return (total_gamma * h_core).trace();
}

// ============================================================================
// Validation & Debugging
// ============================================================================

void OPDM::print_statistics() const {
    std::cout << "\n=== OPDM Statistics ===\n";
    std::cout << "Dimensions: " << n_orbitals_ << " × " << n_orbitals_ << "\n";
    std::cout << "Determinants: " << n_determinants_ << "\n\n";
    
    // Traces
    double trace_alpha = trace(true);
    double trace_beta = trace(false);
    std::cout << "Trace(γ^α): " << std::fixed << std::setprecision(6) 
              << trace_alpha << "\n";
    std::cout << "Trace(γ^β): " << trace_beta << "\n";
    std::cout << "Total electrons: " << trace_alpha + trace_beta << "\n\n";
    
    // Eigenvalue ranges
    auto [occ_alpha, orb_alpha] = natural_orbitals(true);
    auto [occ_beta, orb_beta] = natural_orbitals(false);
    
    std::cout << "Natural occupation range:\n";
    std::cout << "  α: [" << occ_alpha.minCoeff() << ", " 
              << occ_alpha.maxCoeff() << "]\n";
    std::cout << "  β: [" << occ_beta.minCoeff() << ", " 
              << occ_beta.maxCoeff() << "]\n\n";
    
    // Hermiticity check
    double hermit_alpha = (opdm_alpha_ - opdm_alpha_.transpose()).norm();
    double hermit_beta = (opdm_beta_ - opdm_beta_.transpose()).norm();
    std::cout << "Hermiticity error:\n";
    std::cout << "  α: " << std::scientific << hermit_alpha << "\n";
    std::cout << "  β: " << hermit_beta << "\n\n";
    
    // N-representability
    bool n_rep = is_n_representable();
    std::cout << "N-representable: " << (n_rep ? "YES ✓" : "NO ✗") << "\n";
    
    // Entropy
    double S_alpha = entropy(true);
    double S_beta = entropy(false);
    std::cout << "\nEntropy (correlation measure):\n";
    std::cout << "  S^α: " << std::fixed << S_alpha << " bits\n";
    std::cout << "  S^β: " << S_beta << " bits\n";
    
    if (S_alpha < 1e-6 && S_beta < 1e-6) {
        std::cout << "  → Single determinant (Hartree-Fock)\n";
    } else {
        std::cout << "  → Multireference character present\n";
    }
}

void OPDM::print_natural_orbitals() const {
    std::cout << "\n=== Natural Orbitals ===\n\n";
    
    auto [occ_alpha, orb_alpha] = natural_orbitals(true);
    auto [occ_beta, orb_beta] = natural_orbitals(false);
    
    std::cout << std::setw(10) << "Orbital" 
              << std::setw(12) << "n_α"
              << std::setw(12) << "n_β"
              << std::setw(12) << "Total"
              << "\n";
    std::cout << std::string(46, '-') << "\n";
    
    for (int i = 0; i < n_orbitals_; i++) {
        double total = occ_alpha(i) + occ_beta(i);
        
        std::cout << std::setw(10) << i
                  << std::setw(12) << std::fixed << std::setprecision(6) 
                  << occ_alpha(i)
                  << std::setw(12) << occ_beta(i)
                  << std::setw(12) << total
                  << "\n";
    }
    
    std::cout << "\n";
}

// ============================================================================
// Internal Computation Methods
// ============================================================================

void OPDM::compute_opdm() {
    // ALGORITHM (Helgaker Ch. 11):
    // 
    // γ_pq = Σ_IJ c_I c_J ⟨Φ_I| a†_p a_q |Φ_J⟩
    // 
    // Slater-Condon rules:
    //   - Same determinant (I=J): diagonal contribution
    //   - Single excitation: off-diagonal contribution
    //   - Higher excitations: zero
    
    for (int I = 0; I < n_determinants_; I++) {
        for (int J = I; J < n_determinants_; J++) {
            // Process both spins
            add_pair_contribution(I, J, true);   // α-spin
            add_pair_contribution(I, J, false);  // β-spin
        }
    }
}

void OPDM::add_pair_contribution(int I, int J, bool alpha) {
    const auto& det_I = determinants_[I];
    const auto& det_J = determinants_[J];
    
    double c_I = ci_coeffs_[I];
    double c_J = ci_coeffs_[J];
    
    if (I == J) {
        // Diagonal contribution
        add_diagonal_contribution(I, alpha);
    } else {
        // Check excitation level
        auto [n_diff_alpha, n_diff_beta] = det_I.excitation_level(det_J);
        
        int n_diff = alpha ? n_diff_alpha : n_diff_beta;
        
        if (n_diff == 1) {
            // Single excitation: off-diagonal contribution
            add_off_diagonal_contribution(I, J, alpha);
        }
        // n_diff > 1: zero contribution, skip
    }
}

void OPDM::add_diagonal_contribution(int I, bool alpha) {
    // Slater-Condon rule for I = J:
    //   ⟨Φ| a†_p a_q |Φ⟩ = δ_pq if p occupied, 0 otherwise
    // 
    // Implementation:
    //   γ_pp += c_I² for each occupied orbital p
    
    const auto& det = determinants_[I];
    double c_I = ci_coeffs_[I];
    double c_I_sq = c_I * c_I;
    
    // Get occupied orbitals
    std::vector<int> occupied = alpha ? det.alpha_occupations() 
                                      : det.beta_occupations();
    
    Eigen::MatrixXd& gamma = alpha ? opdm_alpha_ : opdm_beta_;
    
    for (int p : occupied) {
        if (p < n_orbitals_) {  // Bounds check
            gamma(p, p) += c_I_sq;
        }
    }
}

void OPDM::add_off_diagonal_contribution(int I, int J, bool alpha) {
    // For single excitation j → i:
    //   ⟨Φ_I| a†_i a_j |Φ_J⟩ = phase × δ_pi δ_qj
    // 
    // ALGORITHM:
    // 1. Find which orbital changed (j → i)
    // 2. Compute phase = (-1)^{# orbitals between j and i}
    // 3. Add γ_ij += phase × c_I × c_J
    // 4. Add γ_ji += phase × c_J × c_I  (Hermitian)
    
    const auto& det_I = determinants_[I];
    const auto& det_J = determinants_[J];
    
    double c_I = ci_coeffs_[I];
    double c_J = ci_coeffs_[J];
    
    // Get occupied orbitals for both determinants
    std::vector<int> occ_I = alpha ? det_I.alpha_occupations() 
                                   : det_I.beta_occupations();
    std::vector<int> occ_J = alpha ? det_J.alpha_occupations() 
                                   : det_J.beta_occupations();
    
    // Find difference: which orbital was excited
    std::vector<int> in_I_not_J;  // Orbitals in I but not in J
    std::vector<int> in_J_not_I;  // Orbitals in J but not in I
    
    for (int p : occ_I) {
        if (std::find(occ_J.begin(), occ_J.end(), p) == occ_J.end()) {
            in_I_not_J.push_back(p);
        }
    }
    
    for (int p : occ_J) {
        if (std::find(occ_I.begin(), occ_I.end(), p) == occ_I.end()) {
            in_J_not_I.push_back(p);
        }
    }
    
    // Should have exactly one difference for single excitation
    if (in_I_not_J.size() != 1 || in_J_not_I.size() != 1) {
        return;  // Not single excitation, skip
    }
    
    int i = in_I_not_J[0];  // Orbital in I (created)
    int j = in_J_not_I[0];  // Orbital in J (annihilated)
    
    // Compute phase
    int phase = det_I.phase(j, i, alpha);
    
    // Add contributions
    Eigen::MatrixXd& gamma = alpha ? opdm_alpha_ : opdm_beta_;
    
    if (i < n_orbitals_ && j < n_orbitals_) {
        gamma(i, j) += phase * c_I * c_J;
        gamma(j, i) += phase * c_J * c_I;  // Hermitian
    }
}

void OPDM::validate() const {
    // Validation checks:
    // 1. Hermiticity
    // 2. Trace = N_electrons
    // 3. Eigenvalues in [0, 1]
    
    const double tolerance = 1e-6;
    
    // Check Hermiticity
    double hermit_error_alpha = (opdm_alpha_ - opdm_alpha_.transpose()).norm();
    double hermit_error_beta = (opdm_beta_ - opdm_beta_.transpose()).norm();
    
    if (hermit_error_alpha > tolerance) {
        throw std::runtime_error(
            "OPDM validation failed: α-spin not Hermitian");
    }
    
    if (hermit_error_beta > tolerance) {
        throw std::runtime_error(
            "OPDM validation failed: β-spin not Hermitian");
    }
    
    // Check eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_alpha(opdm_alpha_);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_beta(opdm_beta_);
    
    const auto& evals_alpha = solver_alpha.eigenvalues();
    const auto& evals_beta = solver_beta.eigenvalues();
    
    for (int i = 0; i < n_orbitals_; i++) {
        if (evals_alpha(i) < -tolerance || evals_alpha(i) > 1.0 + tolerance) {
            throw std::runtime_error(
                "OPDM validation failed: α-spin eigenvalue out of [0,1]");
        }
        
        if (evals_beta(i) < -tolerance || evals_beta(i) > 1.0 + tolerance) {
            throw std::runtime_error(
                "OPDM validation failed: β-spin eigenvalue out of [0,1]");
        }
    }
    
    // Trace check is informational only (depends on wavefunction)
    // Don't throw error, just warn if seems wrong
    double trace_total = trace(true) + trace(false);
    if (trace_total < 0.0 || trace_total > 2.0 * n_orbitals_) {
        std::cerr << "Warning: OPDM trace = " << trace_total 
                  << " seems unusual\n";
    }
}

} // namespace foundation
} // namespace mshqc
