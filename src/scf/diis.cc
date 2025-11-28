/**
 * @file diis.cc
 * @brief Direct Inversion in the Iterative Subspace (DIIS) convergence accelerator
 * 
 * Implementation of Pulay's DIIS method for accelerating SCF convergence.
 * Extrapolates Fock matrix from error vectors to minimize residual.
 * 
 * Theory References:
 *   - P. Pulay, Chem. Phys. Lett. 73, 393 (1980)
 *     [Original DIIS formulation, Eq. (11)]
 *   - P. Pulay, J. Comput. Chem. 3, 556 (1982)
 *     [Improved DIIS convergence, Eq. (8)]
 *   - C. D. Sherrill, "An Introduction to Hartree-Fock Molecular Orbital Theory"
 *     [Pedagogical DIIS explanation]
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-01-29
 * @license MIT License (see LICENSE file in project root)
 * 
 * @note This is an original implementation derived from published theory.
 *       No code was copied from existing quantum chemistry software.
 *       DIIS equation: minimize ||\u03a3 c_i e_i||\u00b2 subject to \u03a3 c_i = 1.
 */

#include "mshqc/scf.h"
#include <stdexcept>

namespace mshqc {

DIIS::DIIS(int max_vectors) : max_vectors_(max_vectors) {
    if (max_vectors < 2) {
        throw std::invalid_argument("DIIS requires at least 2 vectors");
    }
}

void DIIS::add_iteration(const Eigen::MatrixXd& F, const Eigen::MatrixXd& error) {
    fock_matrices_.push_back(F);
    error_vectors_.push_back(error);
    
    // Keep only recent history
    if (static_cast<int>(fock_matrices_.size()) > max_vectors_) {
        fock_matrices_.erase(fock_matrices_.begin());
        error_vectors_.erase(error_vectors_.begin());
    }
}

Eigen::MatrixXd DIIS::extrapolate() {
    // REFERENCE: Pulay (1980), Chem. Phys. Lett. 73, 393, Eq. (11)
    // Solve DIIS equation: minimize ||Σ c_i e_i||² subject to Σ c_i = 1
    
    if (!can_extrapolate()) {
        throw std::runtime_error("DIIS: not enough vectors");
    }
    
    size_t n = fock_matrices_.size();
    
    // Build B matrix: B_ij = Tr(e_i^† e_j)
    Eigen::MatrixXd B = build_B_matrix();
    
    // Augmented system with Lagrange multiplier
    // [ B  -1 ] [ c ] = [ 0 ]
    // [-1   0 ] [ λ ]   [-1 ]
    Eigen::MatrixXd A(n + 1, n + 1);
    A.topLeftCorner(n, n) = B;
    A.block(0, n, n, 1).setConstant(-1.0);
    A.block(n, 0, 1, n).setConstant(-1.0);
    A(n, n) = 0.0;
    
    Eigen::VectorXd b(n + 1);
    b.setZero();
    b(n) = -1.0;
    
    Eigen::VectorXd x = A.fullPivLu().solve(b);
    Eigen::VectorXd c = x.head(n);  // extract coefficients
    
    // Extrapolated Fock: F = Σ c_i F_i
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(
        fock_matrices_[0].rows(),
        fock_matrices_[0].cols()
    );
    
    for (size_t i = 0; i < n; i++) {
        F += c(i) * fock_matrices_[i];
    }
    
    return F;
}

void DIIS::clear() {
    fock_matrices_.clear();
    error_vectors_.clear();
}

Eigen::MatrixXd DIIS::build_B_matrix() const {
    // REFERENCE: Pulay (1980), Chem. Phys. Lett. 73, 393, Eq. (11)
    // B_ij = Tr(e_i^† e_j) = Σ_pq e_i[pq] × e_j[pq]
    
    size_t n = error_vectors_.size();
    Eigen::MatrixXd B(n, n);
    
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            B(i, j) = (error_vectors_[i].array() * error_vectors_[j].array()).sum();
        }
    }
    
    return B;
}

} // namespace mshqc
