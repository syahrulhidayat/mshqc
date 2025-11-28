/**
 * @file davidson.cc
 * @brief Implementation of Davidson iterative diagonalization
 * 
 * REFERENCES:
 * - Davidson (1975), J. Comput. Phys. 17, 87-94
 * - Liu (1978), Math. Program. 45, 503-528  
 * - Sleijpen & van der Vorst (1996), SIAM J. Matrix Anal. Appl. 17, 401
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

#include "mshqc/ci/davidson.h"
#include "mshqc/ci/hamiltonian_sparse.h"
#include "mshqc/ci/excitation_generator.h"
#include "mshqc/ci/ci_utils.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mshqc {
namespace ci {

DavidsonSolver::DavidsonSolver(const DavidsonOptions& opts)
    : opts_(opts) {}

DavidsonSolver::~DavidsonSolver() {
    // Clean up owned det_map if we created it
    if (owns_det_map_ && det_map_ != nullptr) {
        delete det_map_;
        det_map_ = nullptr;
    }
}

// Main Davidson algorithm
// REFERENCE: Davidson (1975), J. Comput. Phys. 17, 87
DavidsonResult DavidsonSolver::solve(
    const std::vector<Determinant>& dets,
    const CIIntegrals& ints,
    const Eigen::VectorXd& guess) {
    
    int n = dets.size();
    
    if (opts_.verbose) {
        std::cout << "\n=== Davidson Diagonalization ===\n";
        std::cout << "Matrix size: " << n << " x " << n << "\n";
        std::cout << "Max iterations: " << opts_.max_iter << "\n";
        std::cout << "Convergence: " << opts_.conv_tol << "\n\n";
    }
    
    // Compute diagonal for preconditioner
    Eigen::VectorXd H_diag = hamiltonian_diagonal(dets, ints);
    
    // Initialize subspace with guess vector (normalized)
    std::vector<Eigen::VectorXd> subspace;
    Eigen::VectorXd b0 = guess.normalized();
    subspace.push_back(b0);
    
        // Compute sigma vector: σ0 = H * b0
    std::vector<Eigen::VectorXd> sigma_vectors;
    sigma_vectors.push_back(compute_sigma(dets, b0, ints));
    
    double energy_old = 0.0;
    DavidsonResult result;
    result.converged = false;
    
    // Main Davidson iteration
    for (int iter = 0; iter < opts_.max_iter; iter++) {
        // Build subspace Hamiltonian: H_sub = B^T * H * B
        Eigen::MatrixXd H_sub = build_subspace_hamiltonian(subspace, sigma_vectors);
        
        // Diagonalize subspace Hamiltonian
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_sub);
        double energy = solver.eigenvalues()(0);  // Lowest eigenvalue
        Eigen::VectorXd alpha = solver.eigenvectors().col(0);  // Eigenvector in subspace
        
        // Expand to full space: b = Σ_i α_i b_i
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
        for (size_t i = 0; i < subspace.size(); i++) {
            b += alpha(i) * subspace[i];
        }
        
        // Compute sigma: σ = Σ_i α_i σ_i
        Eigen::VectorXd sigma = Eigen::VectorXd::Zero(n);
        for (size_t i = 0; i < sigma_vectors.size(); i++) {
            sigma += alpha(i) * sigma_vectors[i];
        }
        
        // Residual: r = σ - E*b = (H - E)*b
        Eigen::VectorXd residual = sigma - energy * b;
        double res_norm = residual.norm();
        
        if (opts_.verbose) {
            std::cout << "Iter " << std::setw(3) << iter 
                      << ": E = " << std::fixed << std::setprecision(10) << energy
                      << ", dE = " << std::scientific << std::setprecision(2) 
                      << (energy - energy_old)
                      << ", |r| = " << res_norm
                      << ", subspace = " << subspace.size() << "\n";
        }
        
        // Check convergence
        if (is_converged(std::abs(energy - energy_old), res_norm)) {
            if (opts_.verbose) {
                std::cout << "\n✓ Davidson converged!\n";
                std::cout << "Final energy: " << std::setprecision(12) << energy << "\n";
            }
            
            result.energy = energy;
            result.eigenvector = b;
            result.iterations = iter + 1;
            result.converged = true;
            result.residual_norm = res_norm;
            return result;
        }
        
        // Compute correction vector
        // REFERENCE: Davidson (1975), Eq. (13)
        // δb = r / (E - H_diag)
        Eigen::VectorXd delta_b = compute_correction(residual, energy, H_diag);
        
        // Orthogonalize against existing subspace
        delta_b = orthogonalize(delta_b, subspace);
        
        // Check if correction is too small (numerical issue)
        if (delta_b.norm() < 1e-10) {
            if (opts_.verbose) {
                std::cout << "\n⚠ Correction vector too small. Stopping.\n";
            }
            result.energy = energy;
            result.eigenvector = b;
            result.iterations = iter + 1;
            result.converged = false;
            result.residual_norm = res_norm;
            return result;
        }
        
        // Normalize and add to subspace
        delta_b.normalize();
        expand_subspace(subspace, delta_b);
        
        // Compute new sigma vector
        sigma_vectors.push_back(compute_sigma(dets, delta_b, ints));
        
        // Collapse subspace if too large
        if (static_cast<int>(subspace.size()) > opts_.max_subspace) {
            if (opts_.verbose) {
                std::cout << "  Collapsing subspace: " << subspace.size() 
                          << " → " << (opts_.max_subspace/2) << "\n";
            }
            collapse_subspace(subspace, sigma_vectors, opts_.max_subspace / 2);
        }
        
        energy_old = energy;
    }
    
    // Max iterations reached without convergence
    if (opts_.verbose) {
        std::cout << "\n✗ Davidson did not converge in " << opts_.max_iter << " iterations\n";
    }
    
    // Return best result so far
    Eigen::MatrixXd H_sub = build_subspace_hamiltonian(subspace, sigma_vectors);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_sub);
    double energy = solver.eigenvalues()(0);
    Eigen::VectorXd alpha = solver.eigenvectors().col(0);
    
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
    for (size_t i = 0; i < subspace.size(); i++) {
        b += alpha(i) * subspace[i];
    }
    
    result.energy = energy;
    result.eigenvector = b;
    result.iterations = opts_.max_iter;
    result.converged = false;
    result.residual_norm = -1.0;
    
    return result;
}

// Sparse Davidson algorithm using precomputed CSR matrix
// REFERENCE: Davidson (1975), same algorithm but with sparse matvec
DavidsonResult DavidsonSolver::solve_sparse(
    const SparseCSR& Hcsr,
    const Eigen::VectorXd& diag,
    const Eigen::VectorXd& guess) {
    
    int n = Hcsr.n_rows();
    
    if (opts_.verbose) {
        std::cout << "\n=== Davidson Diagonalization (Sparse) ===\n";
        std::cout << "Matrix size: " << n << " x " << n << "\n";
        std::cout << "Max iterations: " << opts_.max_iter << "\n";
        std::cout << "Convergence: " << opts_.conv_tol << "\n\n";
    }
    
    // Initialize subspace with guess vector (normalized)
    std::vector<Eigen::VectorXd> subspace;
    Eigen::VectorXd b0 = guess.normalized();
    subspace.push_back(b0);
    
    // Compute sigma vector: σ0 = H * b0 (using sparse matvec)
    std::vector<Eigen::VectorXd> sigma_vectors;
    sigma_vectors.push_back(sigma_vector_sparse(Hcsr, b0));
    
    double energy_old = 0.0;
    DavidsonResult result;
    result.converged = false;
    
    // Main Davidson iteration
    for (int iter = 0; iter < opts_.max_iter; iter++) {
        // Build subspace Hamiltonian: H_sub = B^T * H * B
        Eigen::MatrixXd H_sub = build_subspace_hamiltonian(subspace, sigma_vectors);
        
        // Diagonalize subspace Hamiltonian
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_sub);
        double energy = solver.eigenvalues()(0);  // Lowest eigenvalue
        Eigen::VectorXd alpha = solver.eigenvectors().col(0);  // Eigenvector in subspace
        
        // Expand to full space: b = Σ_i α_i b_i
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
        for (size_t i = 0; i < subspace.size(); i++) {
            b += alpha(i) * subspace[i];
        }
        
        // Compute sigma: σ = Σ_i α_i σ_i
        Eigen::VectorXd sigma = Eigen::VectorXd::Zero(n);
        for (size_t i = 0; i < sigma_vectors.size(); i++) {
            sigma += alpha(i) * sigma_vectors[i];
        }
        
        // Residual: r = σ - E*b = (H - E)*b
        Eigen::VectorXd residual = sigma - energy * b;
        double res_norm = residual.norm();
        
        if (opts_.verbose) {
            std::cout << "Iter " << std::setw(3) << iter 
                      << ": E = " << std::fixed << std::setprecision(10) << energy
                      << ", dE = " << std::scientific << std::setprecision(2) 
                      << (energy - energy_old)
                      << ", |r| = " << res_norm
                      << ", subspace = " << subspace.size() << "\n";
        }
        
        // Check convergence
        if (is_converged(std::abs(energy - energy_old), res_norm)) {
            if (opts_.verbose) {
                std::cout << "\n✓ Davidson converged!\n";
                std::cout << "Final energy: " << std::setprecision(12) << energy << "\n";
            }
            
            result.energy = energy;
            result.eigenvector = b;
            result.iterations = iter + 1;
            result.converged = true;
            result.residual_norm = res_norm;
            return result;
        }
        
        // Compute correction vector: δb = r / (E - H_diag)
        Eigen::VectorXd delta_b = compute_correction(residual, energy, diag);
        
        // Orthogonalize against existing subspace
        delta_b = orthogonalize(delta_b, subspace);
        
        // Check if correction is too small
        if (delta_b.norm() < 1e-10) {
            if (opts_.verbose) {
                std::cout << "\n⚠ Correction vector too small. Stopping.\n";
            }
            result.energy = energy;
            result.eigenvector = b;
            result.iterations = iter + 1;
            result.converged = false;
            result.residual_norm = res_norm;
            return result;
        }
        
        // Normalize and add to subspace
        delta_b.normalize();
        expand_subspace(subspace, delta_b);
        
        // Compute new sigma vector (using sparse matvec)
        sigma_vectors.push_back(sigma_vector_sparse(Hcsr, delta_b));
        
        // Collapse subspace if too large
        if (static_cast<int>(subspace.size()) > opts_.max_subspace) {
            if (opts_.verbose) {
                std::cout << "  Collapsing subspace: " << subspace.size() 
                          << " → " << (opts_.max_subspace/2) << "\n";
            }
            collapse_subspace(subspace, sigma_vectors, opts_.max_subspace / 2);
        }
        
        energy_old = energy;
    }
    
    // Max iterations reached without convergence
    if (opts_.verbose) {
        std::cout << "\n✗ Davidson did not converge in " << opts_.max_iter << " iterations\n";
    }
    
    // Return best result so far
    Eigen::MatrixXd H_sub = build_subspace_hamiltonian(subspace, sigma_vectors);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_sub);
    double energy = solver.eigenvalues()(0);
    Eigen::VectorXd alpha = solver.eigenvectors().col(0);
    
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
    for (size_t i = 0; i < subspace.size(); i++) {
        b += alpha(i) * subspace[i];
    }
    
    result.energy = energy;
    result.eigenvector = b;
    result.iterations = opts_.max_iter;
    result.converged = false;
    result.residual_norm = -1.0;
    
    return result;
}

// Build subspace Hamiltonian
Eigen::MatrixXd DavidsonSolver::build_subspace_hamiltonian(
    const std::vector<Eigen::VectorXd>& subspace,
    const std::vector<Eigen::VectorXd>& sigma_vectors) {
    
    int m = subspace.size();
    Eigen::MatrixXd H_sub(m, m);
    
    // H_sub(i,j) = b_i^T * σ_j = b_i^T * H * b_j
    for (int i = 0; i < m; i++) {
        for (int j = i; j < m; j++) {
            H_sub(i,j) = subspace[i].dot(sigma_vectors[j]);
            if (i != j) {
                H_sub(j,i) = H_sub(i,j);  // Hermitian
            }
        }
    }
    
    return H_sub;
}

// Orthogonalize vector against subspace
// REFERENCE: Sleijpen & van der Vorst (1996) - Modified Gram-Schmidt
Eigen::VectorXd DavidsonSolver::orthogonalize(
    const Eigen::VectorXd& vec,
    const std::vector<Eigen::VectorXd>& subspace) {
    
    Eigen::VectorXd result = vec;
    
    // Modified Gram-Schmidt (more stable than classical)
    for (const auto& b : subspace) {
        double proj = b.dot(result);
        result -= proj * b;
    }
    
    return result;
}

// Compute Davidson correction
// REFERENCE: Davidson (1975), Eq. (13)
Eigen::VectorXd DavidsonSolver::compute_correction(
    const Eigen::VectorXd& residual,
    double eigenvalue,
    const Eigen::VectorXd& diag) {
    
    int n = residual.size();
    Eigen::VectorXd delta(n);
    
    // δb_i = r_i / (E - H_ii)
    // Preconditioner using diagonal elements
    for (int i = 0; i < n; i++) {
        double denom = eigenvalue - diag(i);
        
        // Avoid division by zero
        if (std::abs(denom) < 1e-12) {
            denom = (denom > 0) ? 1e-12 : -1e-12;
        }
        
        delta(i) = residual(i) / denom;
    }
    
    return delta;
}

void DavidsonSolver::expand_subspace(
    std::vector<Eigen::VectorXd>& subspace,
    const Eigen::VectorXd& new_vec) {
    
    subspace.push_back(new_vec);
}

bool DavidsonSolver::is_converged(double delta_e, double residual_norm) const {
    return (std::abs(delta_e) < opts_.conv_tol) && 
           (residual_norm < opts_.residual_tol);
}

void DavidsonSolver::collapse_subspace(
    std::vector<Eigen::VectorXd>& subspace,
    std::vector<Eigen::VectorXd>& sigma_vectors,
    int keep_size) {
    
    // Keep only first keep_size vectors (most important)
    subspace.resize(keep_size);
    sigma_vectors.resize(keep_size);
}

// Generate initial guess
Eigen::VectorXd generate_davidson_guess(
    const std::vector<Determinant>& dets,
    const CIIntegrals& ints) {
    
    int n = dets.size();
    
    // Find determinant with lowest diagonal element
    Eigen::VectorXd diag = hamiltonian_diagonal(dets, ints);
    
    int min_idx = 0;
    double min_val = diag(0);
    for (int i = 1; i < n; i++) {
        if (diag(i) < min_val) {
            min_val = diag(i);
            min_idx = i;
        }
    }
    
    // Create guess: unit vector at min_idx
    Eigen::VectorXd guess = Eigen::VectorXd::Zero(n);
    guess(min_idx) = 1.0;
    
    return guess;
}

// Multiple root solver (for excited states)
std::vector<DavidsonResult> DavidsonSolver::solve_multiple(
    const std::vector<Determinant>& dets,
    const CIIntegrals& ints,
    int nroots) {
    
    std::vector<DavidsonResult> results;
    
    // For now, solve sequentially (can optimize later)
    for (int root = 0; root < nroots; root++) {
        if (opts_.verbose) {
            std::cout << "\n=== Root " << root << " ===\n";
        }
        
        // Generate guess for this root
        auto guesses = generate_multiple_guesses(dets, ints, root + 1);
        auto result = solve(dets, ints, guesses[root]);
        
        results.push_back(result);
    }
    
    return results;
}

std::vector<Eigen::VectorXd> generate_multiple_guesses(
    const std::vector<Determinant>& dets,
    const CIIntegrals& ints,
    int nroots) {
    
    int n = dets.size();
    Eigen::VectorXd diag = hamiltonian_diagonal(dets, ints);
    
    // Sort diagonal elements
    std::vector<std::pair<double, int>> sorted_diag;
    for (int i = 0; i < n; i++) {
        sorted_diag.push_back({diag(i), i});
    }
    std::sort(sorted_diag.begin(), sorted_diag.end());
    
    // Generate guesses from lowest nroots determinants
    std::vector<Eigen::VectorXd> guesses;
    for (int r = 0; r < std::min(nroots, n); r++) {
        Eigen::VectorXd guess = Eigen::VectorXd::Zero(n);
        guess(sorted_diag[r].second) = 1.0;
        guesses.push_back(guess);
    }
    
    return guesses;
}

// ============================================================================
// ON-THE-FLY SIGMA-VECTOR SUPPORT
// ============================================================================

void DavidsonSolver::set_onthefly_mode(
    bool use_onthefly,
    int n_orb,
    const std::unordered_map<Determinant, int>* det_map) {
    
    if (n_orb <= 0) {
        throw std::invalid_argument("n_orb must be positive");
    }
    
    use_onthefly_ = use_onthefly;
    n_orb_ = n_orb;
    det_map_ = det_map;
    owns_det_map_ = false;  // Caller owns the map
}

Eigen::VectorXd DavidsonSolver::compute_sigma(
    const std::vector<Determinant>& dets,
    const Eigen::VectorXd& c,
    const CIIntegrals& ints) {
    
    if (use_onthefly_) {
        // On-the-fly mode: O(N) memory, O(N×N_conn) time
        // REFERENCE: Knowles & Handy (1984), Olsen et al. (1988)
        
        const int n = static_cast<int>(dets.size());
        Eigen::VectorXd sigma = Eigen::VectorXd::Zero(n);
        
        // PARALLEL OPTIMIZATION: OpenMP parallelization of sigma-vector
        // Each thread computes independent σ_i values
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 16)
        #endif
        for (int i = 0; i < n; ++i) {
            const auto& det_i = dets[i];
            
            // Diagonal: ⟨Φ_i|H|Φ_i⟩
            double H_ii = diagonal_element(det_i, ints);
            sigma(i) += H_ii * c(i);
            
            // Off-diagonal: generate connected determinants
            generate_connected_excitations(det_i, n_orb_, 
                [&](const GeneratedExcitation& exc) {
                    // O(1) hash lookup
                    auto it = det_map_->find(exc.det);
                    if (it != det_map_->end()) {
                        int j = it->second;
                        
                        // Skip if same determinant (already handled in diagonal)
                        if (i == j) return;
                        
                        // Compute ⟨Φ_i|H|Φ_j⟩ on-the-fly
                        double H_ij = hamiltonian_element(det_i, exc.det, ints);
                        
                        // Accumulate: σ_i += H_ij * c_j
                        sigma(i) += H_ij * c(j);
                    }
                });
        }
        
        return sigma;
    } else {
        // Dense mode: O(N²) memory, O(N²) time
        // Standard sigma-vector computation
        return sigma_vector(dets, c, ints);
    }
}

} // namespace ci
} // namespace mshqc
