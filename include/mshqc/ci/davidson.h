#ifndef MSHQC_CI_DAVIDSON_H
#define MSHQC_CI_DAVIDSON_H

#include "mshqc/ci/determinant.h"
#include "mshqc/ci/slater_condon.h"
#include "mshqc/ci/sparse_csr.h"
#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <unordered_map>

/**
 * @file davidson.h
 * @brief Davidson iterative diagonalization for large sparse matrices
 * 
 * REFERENCES:
 * - Davidson (1975), J. Comput. Phys. 17, 87-94
 * - Liu (1978), Math. Program. 45, 503-528
 * - Sleijpen & van der Vorst (1996), SIAM J. Matrix Anal. Appl. 17, 401
 * 
 * ALGORITHM:
 * For eigenvalue problem HC = EC:
 * 1. Start with guess vector b0
 * 2. Iteratively expand subspace with σ = H*b
 * 3. Diagonalize small subspace Hamiltonian
 * 4. Compute correction: δb = r / (E - H_diag)
 * 5. Orthogonalize and add to subspace
 * 6. Repeat until converged
 * 
 * KEY ADVANTAGE: Never store full H matrix!
 * Only needs σ = H*c (matrix-vector product)
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

namespace mshqc {
namespace ci {

/**
 * Davidson convergence criteria
 */
struct DavidsonOptions {
    int max_iter = 50;           // Max iterations
    int max_subspace = 20;       // Max subspace size
    double conv_tol = 1e-8;      // Energy convergence
    double residual_tol = 1e-6;  // Residual norm threshold
    bool verbose = true;         // Print progress
};

/**
 * Davidson solver result
 */
struct DavidsonResult {
    double energy;               // Converged eigenvalue
    Eigen::VectorXd eigenvector; // Converged eigenvector
    int iterations;              // # iterations
    bool converged;              // Convergence flag
    double residual_norm;        // Final residual norm
};

/**
 * Davidson iterative diagonalization
 * 
 * USAGE:
 * ```cpp
 * DavidsonSolver solver;
 * auto result = solver.solve(dets, ints, guess);
 * std::cout << "Ground state: " << result.energy << "\n";
 * ```
 */
class DavidsonSolver {
public:
    /**
     * Constructor with options
     */
    DavidsonSolver(const DavidsonOptions& opts = DavidsonOptions());
    
    /**
     * Destructor (cleanup owned resources)
     */
    ~DavidsonSolver();
    
    /**
     * Solve for lowest eigenvalue/eigenvector
     * 
     * REFERENCE: Davidson (1975), J. Comput. Phys. 17, 87
     * 
     * @param dets List of determinants
     * @param ints MO integrals
     * @param guess Initial guess vector
     * @return Lowest eigenvalue and eigenvector
     */
    DavidsonResult solve(const std::vector<Determinant>& dets,
                         const CIIntegrals& ints,
                         const Eigen::VectorXd& guess);
    
    /**
     * Solve for multiple roots (excited states)
     * 
     * @param dets List of determinants
     * @param ints MO integrals
     * @param nroots Number of roots to find
     * @return Vector of eigenvalues/eigenvectors
     */
    std::vector<DavidsonResult> solve_multiple(
        const std::vector<Determinant>& dets,
        const CIIntegrals& ints,
        int nroots);
    
    /**
     * Solve using sparse Hamiltonian representation (CSR format)
     * 
     * This method uses precomputed sparse Hamiltonian matrix instead of
     * computing sigma vectors on-the-fly. Suitable for large systems where
     * the Hamiltonian is sparse and CSR matvec is faster than dense operations.
     * 
     * REFERENCE: Davidson (1975), J. Comput. Phys. 17, 87
     * 
     * @param Hcsr Sparse Hamiltonian in CSR format
     * @param diag Diagonal elements for preconditioner
     * @param guess Initial guess vector
     * @return Lowest eigenvalue and eigenvector
     */
    DavidsonResult solve_sparse(
        const SparseCSR& Hcsr,
        const Eigen::VectorXd& diag,
        const Eigen::VectorXd& guess);
    
    /**
     * Enable on-the-fly sigma-vector mode for large systems
     * 
     * For large CI calculations (N > 200), avoids storing H matrix.
     * Uses hash-based determinant lookup for O(1) indexing.
     * 
     * REFERENCE: Direct CI methods
     *   - Knowles & Handy (1984), Chem. Phys. Lett. 111, 315
     *   - Olsen et al. (1988), J. Chem. Phys. 89, 2185
     * 
     * @param use_onthefly Enable on-the-fly mode
     * @param n_orb Number of orbitals (for excitation generation)
     * @param det_map Determinant->index hash map
     */
    void set_onthefly_mode(bool use_onthefly,
                           int n_orb,
                           const std::unordered_map<Determinant, int>* det_map);
    
private:
    DavidsonOptions opts_;
    
    // On-the-fly mode configuration
    bool use_onthefly_ = false;
    int n_orb_ = 0;
    const std::unordered_map<Determinant, int>* det_map_ = nullptr;
    bool owns_det_map_ = false;  // Track if we created det_map
    
    /**
     * Expand subspace with new vector
     * REFERENCE: Davidson (1975), Eq. (12)
     */
    void expand_subspace(std::vector<Eigen::VectorXd>& subspace,
                         const Eigen::VectorXd& new_vec);
    
    /**
     * Orthogonalize vector against subspace (Modified Gram-Schmidt)
     * REFERENCE: Sleijpen & van der Vorst (1996)
     */
    Eigen::VectorXd orthogonalize(const Eigen::VectorXd& vec,
                                   const std::vector<Eigen::VectorXd>& subspace);
    
    /**
     * Build subspace Hamiltonian: H_sub = B^T * H * B
     * where B = [b1, b2, ..., bn] is subspace basis
     */
    Eigen::MatrixXd build_subspace_hamiltonian(
        const std::vector<Eigen::VectorXd>& subspace,
        const std::vector<Eigen::VectorXd>& sigma_vectors);
    
    /**
     * Compute Davidson correction vector
     * REFERENCE: Davidson (1975), Eq. (13)
     * δb = r / (E - H_diag)
     * where r = (H - E)b is residual
     */
    Eigen::VectorXd compute_correction(
        const Eigen::VectorXd& residual,
        double eigenvalue,
        const Eigen::VectorXd& diag);
    
    /**
     * Check convergence
     */
    bool is_converged(double delta_e, double residual_norm) const;
    
    /**
     * Collapse subspace if too large
     * Keep only important vectors
     */
    void collapse_subspace(std::vector<Eigen::VectorXd>& subspace,
                            std::vector<Eigen::VectorXd>& sigma_vectors,
                            int keep_size);
    
    /**
     * Compute sigma-vector with automatic method selection
     * Chooses dense or on-the-fly based on configuration
     * 
     * @param dets List of determinants
     * @param c CI coefficient vector
     * @param ints MO integrals
     * @return σ = H·c
     */
    Eigen::VectorXd compute_sigma(
        const std::vector<Determinant>& dets,
        const Eigen::VectorXd& c,
        const CIIntegrals& ints);
};

/**
 * Generate initial guess for Davidson
 * 
 * Strategy: Use determinant with lowest diagonal element
 * (usually HF determinant for ground state)
 */
Eigen::VectorXd generate_davidson_guess(
    const std::vector<Determinant>& dets,
    const CIIntegrals& ints);

/**
 * Generate guess for excited states
 * Use determinants with low diagonal elements
 */
std::vector<Eigen::VectorXd> generate_multiple_guesses(
    const std::vector<Determinant>& dets,
    const CIIntegrals& ints,
    int nroots);

} // namespace ci
} // namespace mshqc

#endif // MSHQC_CI_DAVIDSON_H
