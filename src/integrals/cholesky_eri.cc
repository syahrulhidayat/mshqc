/**
 * @file cholesky_eri.cc
 * @brief Cholesky Decomposition for Electron Repulsion Integrals - Implementation
 * 
 * THEORY REFERENCES:
 * - N. H. F. Beebe & J. Linderberg, Int. J. Quantum Chem. **12**, 683 (1977)
 * - H. Koch et al., J. Chem. Phys. **118**, 9481 (2003)
 * - F. Aquilante et al., J. Chem. Phys. **129**, 024113 (2008)
 * 
 * @author Muhamad Syahrul Hidayat (Agent 2)
 * @date 2025-11-16
 * @license MIT License
 */

#include "mshqc/integrals/cholesky_eri.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace mshqc {
namespace integrals {

// ============================================================================
// CONSTRUCTOR
// ============================================================================

CholeskyERI::CholeskyERI(double threshold)
    : threshold_(threshold) {
    if (threshold_ <= 0.0) {
        throw std::invalid_argument("Cholesky threshold must be positive");
    }
}

// ============================================================================
// MAIN DECOMPOSITION ALGORITHM (Koch et al. 2003)
// ============================================================================

CholeskyDecompositionResult CholeskyERI::decompose(
    const Eigen::Tensor<double, 4>& eri_full
) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Cholesky Decomposition of ERIs\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Theory: Koch et al. (2003), Beebe & Linderberg (1977)\n";
    std::cout << "Threshold: " << std::scientific << threshold_ << " Ha\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    reset();
    
    auto dims = eri_full.dimensions();
    n_basis_ = static_cast<int>(dims[0]);
    int n_pairs = n_basis_ * n_basis_;
    
    std::cout << "Basis functions: " << n_basis_ << "\n";
    std::cout << "Orbital pairs: " << n_pairs << "\n";
    std::cout << "Full ERI storage: " << (n_pairs * n_pairs * 8.0 / 1024.0 / 1024.0) 
              << " MB\n\n";
    
    // Initialize diagonal
    Eigen::VectorXd D(n_pairs);
    D.setZero();
    
    std::cout << "Initializing diagonal elements...\n";
    for (int i = 0; i < n_basis_; ++i) {
        for (int j = 0; j < n_basis_; ++j) {
            int ij = composite_index(i, j);
            D(ij) = eri_full(i, j, i, j);
        }
    }
    
    double D_max_initial = D.maxCoeff();
    std::cout << "Initial max diagonal: " << std::scientific << D_max_initial << " Ha\n";
    std::cout << "Target threshold: " << threshold_ << " Ha\n\n";
    
    if (D_max_initial < threshold_) {
        std::cout << "WARNING: All diagonal elements below threshold!\n";
        decomposed_ = true;
        
        CholeskyDecompositionResult result;
        result.n_vectors = 0;
        result.n_basis = n_basis_;
        result.threshold = threshold_;
        result.converged = true;
        return result;
    }
    
    // Modified Cholesky algorithm
    std::cout << "Starting Cholesky decomposition...\n";
    int max_vectors = n_pairs;
    int iter = 0;
    
    while (iter < max_vectors) {
        int pivot_ij = find_pivot(D);
        double D_pivot = D(pivot_ij);
        
        if (D_pivot < threshold_) {
            std::cout << "\nConverged at vector " << iter << "\n";
            std::cout << "Max remaining diagonal: " << std::scientific 
                      << D_pivot << " Ha\n";
            break;
        }
        
        Eigen::VectorXd L_new = compute_new_vector(eri_full, D, pivot_ij);
        
        for (int ij = 0; ij < n_pairs; ++ij) {
            D(ij) -= L_new(ij) * L_new(ij);
            if (D(ij) < 0.0) D(ij) = 0.0;
        }
        
        L_vectors_.push_back(L_new);
        n_vectors_++;
        iter++;
        
        /*if (iter % 10 == 0 || iter < 10) {
            std::cout << "Vector " << std::setw(4) << iter 
                      << ": D_max = " << std::scientific << std::setprecision(3)
                      << D_pivot << " Ha, "
                      << "||L|| = " << std::fixed << std::setprecision(6)
                      << L_new.norm() << "\n";
        }*/
    }
    
    decomposed_ = true;
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Decomposition Complete\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Cholesky vectors: " << n_vectors_ << "\n";
    std::cout << "Basis functions: " << n_basis_ << "\n";
    std::cout << "Ratio M/N: " << std::fixed << std::setprecision(2) 
              << (double)n_vectors_ / n_basis_ << "\n";
    
    double comp_ratio = compression_ratio();
    std::cout << "Compression ratio: " << std::fixed << std::setprecision(1)
              << comp_ratio << "×\n";
    std::cout << "Storage: " << (storage_bytes() / 1024.0 / 1024.0) << " MB\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    CholeskyDecompositionResult result;
    result.n_vectors = n_vectors_;
    result.n_basis = n_basis_;
    result.threshold = threshold_;
    result.compression_ratio = comp_ratio;
    result.converged = (iter < max_vectors);
    
    return result;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

int CholeskyERI::find_pivot(const Eigen::VectorXd& D) const {
    int pivot = 0;
    double max_val = D(0);
    
    for (int ij = 1; ij < D.size(); ++ij) {
        if (D(ij) > max_val) {
            max_val = D(ij);
            pivot = ij;
        }
    }
    
    return pivot;
}

Eigen::VectorXd CholeskyERI::compute_new_vector(
    const Eigen::Tensor<double, 4>& eri_full,
    const Eigen::VectorXd& D,
    int pivot_ij
) const {
    int n_pairs = n_basis_ * n_basis_;
    Eigen::VectorXd L_new(n_pairs);
    L_new.setZero();
    
    int i_pivot = pivot_ij / n_basis_;
    int j_pivot = pivot_ij % n_basis_;
    
    for (int i = 0; i < n_basis_; ++i) {
        for (int j = 0; j < n_basis_; ++j) {
            int ij = composite_index(i, j);
            
            double val = eri_full(i, j, i_pivot, j_pivot);
            
            for (int k = 0; k < n_vectors_; ++k) {
                val -= L_vectors_[k](ij) * L_vectors_[k](pivot_ij);
            }
            
            L_new(ij) = val;
        }
    }
    
    double norm_factor = std::sqrt(D(pivot_ij));
    if (norm_factor > 1e-12) {
        L_new /= norm_factor;
    } else {
        std::cerr << "WARNING: Near-zero pivot in Cholesky decomposition\n";
    }
    
    return L_new;
}

// ============================================================================
// RECONSTRUCTION
// ============================================================================

double CholeskyERI::reconstruct(int i, int j, int a, int b) const {
    if (!decomposed_) {
        throw std::runtime_error("Cannot reconstruct: decomposition not performed");
    }
    
    int ij = composite_index(i, j);
    int ab = composite_index(a, b);
    
    double value = 0.0;
    for (int k = 0; k < n_vectors_; ++k) {
        value += L_vectors_[k](ij) * L_vectors_[k](ab);
    }
    
    return value;
}

Eigen::Tensor<double, 4> CholeskyERI::reconstruct_full() const {
    if (!decomposed_) {
        throw std::runtime_error("Cannot reconstruct: decomposition not performed");
    }
    
    std::cout << "Reconstructing full ERI tensor (validation mode)...\n";
    
    Eigen::Tensor<double, 4> eri_recon(n_basis_, n_basis_, n_basis_, n_basis_);
    eri_recon.setZero();
    
    for (int i = 0; i < n_basis_; ++i) {
        for (int j = 0; j < n_basis_; ++j) {
            for (int a = 0; a < n_basis_; ++a) {
                for (int b = 0; b < n_basis_; ++b) {
                    eri_recon(i, j, a, b) = reconstruct(i, j, a, b);
                }
            }
        }
    }
    
    return eri_recon;
}

// ============================================================================
// VALIDATION
// ============================================================================

std::pair<double, double> CholeskyERI::validate_reconstruction(
    const Eigen::Tensor<double, 4>& eri_exact
) {
    std::cout << "Validating Cholesky reconstruction...\n";
    
    auto dims = eri_exact.dimensions();
    if (static_cast<int>(dims[0]) != n_basis_) {
        throw std::invalid_argument("ERI dimensions mismatch");
    }
    
    double max_error = 0.0;
    double sum_sq_error = 0.0;
    long long n_elements = 0;
    
    for (int i = 0; i < n_basis_; ++i) {
        for (int j = 0; j < n_basis_; ++j) {
            for (int a = 0; a < n_basis_; ++a) {
                for (int b = 0; b < n_basis_; ++b) {
                    double exact = eri_exact(i, j, a, b);
                    double approx = reconstruct(i, j, a, b);
                    double error = std::abs(exact - approx);
                    
                    if (error > max_error) {
                        max_error = error;
                    }
                    sum_sq_error += error * error;
                    n_elements++;
                }
            }
        }
    }
    
    double rms_error = std::sqrt(sum_sq_error / n_elements);
    
    max_error_ = max_error;
    rms_error_ = rms_error;
    
    std::cout << "Validation complete:\n";
    std::cout << "  Max error: " << std::scientific << max_error << " Ha\n";
    std::cout << "  RMS error: " << rms_error << " Ha\n";
    std::cout << "  Elements checked: " << n_elements << "\n\n";
    
    return {max_error, rms_error};
}

// ============================================================================
// UTILITIES
// ============================================================================

double CholeskyERI::compression_ratio() const {
    if (n_basis_ == 0 || n_vectors_ == 0) return 0.0;
    
    double full_size = std::pow(n_basis_, 4);
    double cholesky_size = n_vectors_ * n_basis_ * n_basis_;
    
    return full_size / cholesky_size;
}

size_t CholeskyERI::storage_bytes() const {
    size_t bytes = 0;
    for (const auto& vec : L_vectors_) {
        bytes += vec.size() * sizeof(double);
    }
    return bytes;
}

void CholeskyERI::print_statistics(bool verbose) const {
    std::cout << "\n=== Cholesky ERI Statistics ===\n";
    std::cout << "Decomposed: " << (decomposed_ ? "Yes" : "No") << "\n";
    
    if (!decomposed_) {
        std::cout << "No decomposition performed yet.\n";
        return;
    }
    
    std::cout << "Basis functions (N): " << n_basis_ << "\n";
    std::cout << "Cholesky vectors (M): " << n_vectors_ << "\n";
    std::cout << "Ratio M/N: " << std::fixed << std::setprecision(2)
              << (double)n_vectors_ / n_basis_ << "\n";
    std::cout << "Compression ratio: " << std::fixed << std::setprecision(1)
              << compression_ratio() << "×\n";
    std::cout << "Storage: " << (storage_bytes() / 1024.0 / 1024.0) 
              << " MB\n";
    std::cout << "Threshold used: " << std::scientific << threshold_ << " Ha\n";
    
    if (max_error_ > 0.0) {
        std::cout << "\nValidation errors:\n";
        std::cout << "  Max error: " << std::scientific << max_error_ << " Ha\n";
        std::cout << "  RMS error: " << rms_error_ << " Ha\n";
    }
    
    if (verbose && n_vectors_ > 0) {
        std::cout << "\nFirst 5 vector norms:\n";
        for (int k = 0; k < std::min(5, n_vectors_); ++k) {
            std::cout << "  L[" << k << "]: " << std::fixed << std::setprecision(6)
                      << L_vectors_[k].norm() << "\n";
        }
    }
    
    std::cout << "==============================\n\n";
}

void CholeskyERI::reset() {
    L_vectors_.clear();
    n_vectors_ = 0;
    n_basis_ = 0;
    decomposed_ = false;
    max_error_ = 0.0;
    rms_error_ = 0.0;
}

} // namespace integrals
} // namespace mshqc