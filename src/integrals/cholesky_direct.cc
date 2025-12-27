// FILE: cholesky_direct.cc
// ============================================================================
#include "mshqc/integrals/cholesky_direct.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

namespace mshqc {
namespace integrals {

using namespace libint2;

DirectCholesky::DirectCholesky(const std::vector<Shell>& shells, double threshold)
    : shells_(shells), threshold_(threshold), n_basis_(0) {
    
    // Compute total basis functions
    for (const auto& s : shells_) {
        n_basis_ += s.size();
    }
    
    // Build shell -> basis function mapping
    shell2bf_.resize(shells_.size());
    size_t bf = 0;
    for (size_t i = 0; i < shells_.size(); i++) {
        shell2bf_[i].resize(1);
        shell2bf_[i][0] = bf;
        bf += shells_[i].size();
    }
}

void DirectCholesky::compute() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Direct Cholesky Decomposition (On-the-fly ERIs)\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Basis functions: " << n_basis_ << "\n";
    std::cout << "Threshold: " << std::scientific << threshold_ << " Ha\n";

    // 1. Auto-detect max_l and max_nprim from shells
    int max_l = 0;
    int max_nprim = 0;
    for (const auto& s : shells_) {
        for (const auto& c : s.contr) {
            max_l = std::max(max_l, c.l);
            max_nprim = std::max(max_nprim, (int)c.coeff.size());
        }
    }
    
    std::cout << "Detected: l_max = " << max_l 
              << ", max_nprim = " << max_nprim << "\n";
    std::cout << std::string(70, '-') << "\n\n";

    // 2. Initialize diagonal (on-the-fly)
    diag_.resize(n_basis_ * n_basis_);
    diag_.setZero();
    
    std::cout << "Computing diagonal ERIs (mu nu | mu nu)...\n";
    
    #pragma omp parallel
    {
        Engine engine_thread(Operator::coulomb, max_nprim, max_l, 0);
        
        #pragma omp for schedule(dynamic)
        for (size_t s1 = 0; s1 < shells_.size(); s1++) {
            size_t bf1_start = shell2bf_[s1][0];
            size_t n1 = shells_[s1].size();
            
            for (size_t s2 = 0; s2 <= s1; s2++) {
                size_t bf2_start = shell2bf_[s2][0];
                size_t n2 = shells_[s2].size();
                
                // Compute (s1 s2 | s1 s2)
                engine_thread.compute(shells_[s1], shells_[s2], 
                                     shells_[s1], shells_[s2]);
                
                const auto& buf_vec = engine_thread.results();
                if (buf_vec[0] == nullptr) continue;
                
                // Extract diagonal elements (f1 f2 | f1 f2)
                for (size_t f1 = 0; f1 < n1; f1++) {
                    for (size_t f2 = 0; f2 < n2; f2++) {
                        // Index in integral buffer
                        size_t idx = f1*n2*n2*n2 + f2*n2*n2 + f1*n2 + f2;
                        
                        size_t bf1 = bf1_start + f1;
                        size_t bf2 = bf2_start + f2;
                        int diag_idx = bf1 * n_basis_ + bf2;
                        
                        double val = buf_vec[0][idx];
                        
                        #pragma omp critical
                        {
                            diag_(diag_idx) = val;
                            if (s1 != s2) {
                                int diag_idx_sym = bf2 * n_basis_ + bf1;
                                diag_(diag_idx_sym) = val;
                            }
                        }
                    }
                }
            }
        }
    }

    double max_diag = diag_.maxCoeff();
    std::cout << "Initial max diagonal: " << std::scientific << max_diag << " Ha\n";
    std::cout << "Target threshold: " << threshold_ << " Ha\n\n";

    // 3. Main Cholesky loop
    L_vectors_.clear();
    int n_vec = 0;

    std::cout << "Starting Cholesky decomposition...\n";
    
    while (max_diag > threshold_) {
        // Find pivot
        int pivot_idx = 0;
        max_diag = diag_.maxCoeff(&pivot_idx);

        if (max_diag < threshold_) {
            std::cout << "\nConverged at vector " << n_vec << "\n";
            break;
        }
        
        if (max_diag < 0.0) {
            if (max_diag > -1e-10) {
                std::cout << "\nSmall negative diagonal (" << max_diag 
                          << ") - treating as zero\n";
                break;
            }
            std::cout << "\nWarning: Large negative diagonal (" << max_diag 
                      << ") - stopping\n";
            break;
        }

        // Compute new Cholesky vector
        Eigen::VectorXd L_new = compute_column(pivot_idx, max_l, max_nprim);

        // Update diagonal: D(ij) -= L_new(ij)^2
        #pragma omp parallel for
        for (int k = 0; k < diag_.size(); k++) {
            diag_(k) -= L_new(k) * L_new(k);
            // Clean up tiny negative values from numerical errors
            if (diag_(k) < 0.0 && diag_(k) > -1e-10) {
                diag_(k) = 0.0;
            }
        }

        L_vectors_.push_back(L_new);
        n_vec++;

        // Print progress
        if (n_vec <= 10 || n_vec % 10 == 0) {
            std::cout << "Vector " << std::setw(4) << n_vec 
                      << ": D_max = " << std::scientific << std::setprecision(3)
                      << max_diag << " Ha, ||L|| = " << std::fixed 
                      << std::setprecision(6) << L_new.norm() << "\n";
        }
        
        // Safety check
        if (n_vec > n_basis_ * n_basis_ / 2) {
            std::cout << "\nWarning: Too many vectors (" << n_vec 
                      << ") - stopping\n";
            break;
        }
    }
    
    // Final statistics
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Decomposition Complete\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Cholesky vectors: " << n_vec << "\n";
    std::cout << "Basis functions: " << n_basis_ << "\n";
    std::cout << "Ratio M/N: " << std::fixed << std::setprecision(2)
              << ((double)n_vec / n_basis_) << "\n";
    
    long long full_size = (long long)n_basis_ * n_basis_ * n_basis_ * n_basis_;
    long long chol_size = (long long)n_vec * n_basis_ * n_basis_;
    double compression = (double)full_size / chol_size;
    
    std::cout << "Compression ratio: " << std::setprecision(1)
              << compression << "×\n";
    
    double chol_mb = (chol_size * sizeof(double)) / (1024.0 * 1024.0);
    std::cout << "Storage: " << std::setprecision(1) << chol_mb << " MB\n";
    std::cout << std::string(70, '=') << "\n\n";
}

double DirectCholesky::compute_eri(int mu, int nu, int lam, int sig, 
                                   Engine& engine) {
    // Find shells containing these basis functions
    size_t s_mu = 0, s_nu = 0, s_lam = 0, s_sig = 0;
    size_t f_mu = 0, f_nu = 0, f_lam = 0, f_sig = 0;
    
    for (size_t s = 0; s < shells_.size(); s++) {
        size_t bf_start = shell2bf_[s][0];
        size_t bf_end = bf_start + shells_[s].size();
        
        if (bf_start <= (size_t)mu && (size_t)mu < bf_end) {
            s_mu = s;
            f_mu = mu - bf_start;
        }
        if (bf_start <= (size_t)nu && (size_t)nu < bf_end) {
            s_nu = s;
            f_nu = nu - bf_start;
        }
        if (bf_start <= (size_t)lam && (size_t)lam < bf_end) {
            s_lam = s;
            f_lam = lam - bf_start;
        }
        if (bf_start <= (size_t)sig && (size_t)sig < bf_end) {
            s_sig = s;
            f_sig = sig - bf_start;
        }
    }
    
    // Compute shell quartet (s_mu s_nu | s_lam s_sig)
    engine.compute(shells_[s_mu], shells_[s_nu], 
                   shells_[s_lam], shells_[s_sig]);
    
    const auto& buf_vec = engine.results();
    if (buf_vec[0] == nullptr) return 0.0;
    
    // Extract specific integral (mu nu | lam sig)
    size_t n_nu = shells_[s_nu].size();
    size_t n_lam = shells_[s_lam].size();
    size_t n_sig = shells_[s_sig].size();
    
    size_t idx = f_mu*n_nu*n_lam*n_sig + f_nu*n_lam*n_sig + f_lam*n_sig + f_sig;
    
    return buf_vec[0][idx];
}

Eigen::VectorXd DirectCholesky::compute_column(int pivot_idx, int max_l, 
                                               int max_nprim) {
    // Decompose pivot index into basis function indices
    int pi = pivot_idx / n_basis_;
    int pj = pivot_idx % n_basis_;
    
    Eigen::VectorXd col(n_basis_ * n_basis_);
    col.setZero();
    
    // Check for valid pivot
    if (diag_(pivot_idx) <= 0.0) {
        std::cerr << "Warning: Non-positive pivot diagonal " 
                  << diag_(pivot_idx) << "\n";
        return col;
    }
    
    double L_pivot = std::sqrt(diag_(pivot_idx));

    // 1. Compute raw integrals (i j | pi pj) for all i,j
    #pragma omp parallel
    {
        Engine engine_thread(Operator::coulomb, max_nprim, max_l, 0);
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n_basis_; i++) {
            for (int j = 0; j < n_basis_; j++) {
                int idx = i * n_basis_ + j;
                col(idx) = compute_eri(i, j, pi, pj, engine_thread);
            }
        }
    }

    // 2. Subtract contributions from previous vectors
    // L_new(ij) = [(ij|pivot) - sum_k L^k(ij) * L^k(pivot)] / sqrt(D(pivot))
    for (const auto& L_prev : L_vectors_) {
        double L_pivot_k = L_prev(pivot_idx);
        col -= L_prev * L_pivot_k;
    }

    // 3. Normalize by pivot
    col /= L_pivot;
    
    return col;
}

} // namespace integrals
} // namespace mshqc