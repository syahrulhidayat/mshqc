/**
 * @file test_cholesky_mo_transform.cc
 * @brief Validate Cholesky vector MO transformation for CD-CASPT2
 * 
 * Tests:
 * 1. Transform Cholesky vectors from AO to MO basis
 * 2. Reconstruct MO ERIs from transformed vectors
 * 3. Compare with direct AO→MO ERI transformation
 * 4. Verify accuracy < 1 µHa
 * 
 * @author Agent 2 - CD-CASPT2 Integration
 * @date 2025-11-17
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include "mshqc/integrals/cholesky_eri.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace mshqc::integrals;

// Generate mock molecular orbital coefficients (random for testing)
Eigen::MatrixXd generate_mo_coefficients(int n_bf, int n_mo) {
    Eigen::MatrixXd C = Eigen::MatrixXd::Random(n_bf, n_mo);
    
    // Orthonormalize columns (Gram-Schmidt)
    for (int i = 0; i < n_mo; ++i) {
        // Orthogonalize against previous vectors
        for (int j = 0; j < i; ++j) {
            double overlap = C.col(i).dot(C.col(j));
            C.col(i) -= overlap * C.col(j);
        }
        // Normalize
        C.col(i).normalize();
    }
    
    return C;
}

// Generate mock positive semidefinite ERI
Eigen::Tensor<double, 4> generate_eri(int n_bf) {
    Eigen::Tensor<double, 4> eri(n_bf, n_bf, n_bf, n_bf);
    eri.setZero();
    
    std::mt19937 gen(42);
    std::normal_distribution<> dis(0.0, 0.05);
    
    int rank = std::max(n_bf / 2, 3);
    int n_pairs = n_bf * n_bf;
    
    Eigen::MatrixXd L = Eigen::MatrixXd::Random(n_pairs, rank) * 0.15;
    Eigen::MatrixXd V = L * L.transpose();
    
    for (int ij = 0; ij < n_pairs; ++ij) {
        V(ij, ij) += 0.05 + std::abs(dis(gen));
    }
    
    for (int i = 0; i < n_bf; ++i) {
        for (int j = 0; j < n_bf; ++j) {
            int ij = i * n_bf + j;
            for (int k = 0; k < n_bf; ++k) {
                for (int l = 0; l < n_bf; ++l) {
                    eri(i, j, k, l) = V(ij, k * n_bf + l);
                }
            }
        }
    }
    
    return eri;
}

// Transform Cholesky vectors from AO to MO basis
std::vector<Eigen::MatrixXd> transform_cholesky_to_mo(
    const std::vector<Eigen::VectorXd>& L_ao_vec,
    const Eigen::MatrixXd& C_mo,
    int n_bf
) {
    int M = L_ao_vec.size();
    int n_mo = C_mo.cols();
    
    std::vector<Eigen::MatrixXd> L_mo(M);
    
    for (int k = 0; k < M; ++k) {
        // Reshape L^k from vector (n_bf^2) to matrix (n_bf × n_bf)
        Eigen::MatrixXd L_k_ao(n_bf, n_bf);
        for (int mu = 0; mu < n_bf; ++mu) {
            for (int nu = 0; nu < n_bf; ++nu) {
                L_k_ao(mu, nu) = L_ao_vec[k](mu * n_bf + nu);
            }
        }
        
        // Transform: L^k_MO = C^T · L^k_AO · C
        L_mo[k] = C_mo.transpose() * L_k_ao * C_mo;
    }
    
    return L_mo;
}

// Reconstruct MO ERI from Cholesky vectors (chemist notation)
double reconstruct_mo_eri(
    int p, int q, int r, int s,
    const std::vector<Eigen::MatrixXd>& L_mo
) {
    double val = 0.0;
    for (size_t k = 0; k < L_mo.size(); ++k) {
        val += L_mo[k](p, q) * L_mo[k](r, s);
    }
    return val;
}

// Direct AO→MO transformation (traditional O(N^5) method)
Eigen::Tensor<double, 4> direct_ao_to_mo_transform(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_mo,
    int n_bf,
    int n_mo
) {
    Eigen::Tensor<double, 4> eri_mo(n_mo, n_mo, n_mo, n_mo);
    eri_mo.setZero();
    
    // Simplified direct transformation (chemist notation)
    for (int p = 0; p < n_mo; ++p) {
        for (int q = 0; q < n_mo; ++q) {
            for (int r = 0; r < n_mo; ++r) {
                for (int s = 0; s < n_mo; ++s) {
                    double val = 0.0;
                    
                    // (pq|rs) = Σ_μνλσ C_μp C_νq C_λr C_σs (μν|λσ)
                    for (int mu = 0; mu < n_bf; ++mu) {
                        for (int nu = 0; nu < n_bf; ++nu) {
                            for (int lam = 0; lam < n_bf; ++lam) {
                                for (int sig = 0; sig < n_bf; ++sig) {
                                    val += C_mo(mu, p) * C_mo(nu, q) * 
                                           C_mo(lam, r) * C_mo(sig, s) *
                                           eri_ao(mu, nu, lam, sig);
                                }
                            }
                        }
                    }
                    
                    eri_mo(p, q, r, s) = val;
                }
            }
        }
    }
    
    return eri_mo;
}

int main() {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout <<   "║  Cholesky MO Transformation Test - CD-CASPT2 Prototype          ║\n";
    std::cout <<   "╚═══════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Agent 2 - Week 4 Day 1\n";
    std::cout << "Validates: Cholesky vector AO→MO transformation\n";
    std::cout << "Theory: Aquilante et al., J. Chem. Phys. 129, 024113 (2008)\n\n";
    
    // Test parameters
    const int n_bf = 7;   // Small system (like H2O/STO-3G)
    const int n_mo = 7;   // All orbitals
    const double threshold = 1e-6;
    
    std::cout << "System: N_bf = " << n_bf << ", N_mo = " << n_mo << "\n";
    std::cout << "Cholesky threshold: τ = " << std::scientific << threshold << "\n\n";
    
    // ========================================================================
    // Step 1: Generate mock data
    // ========================================================================
    
    std::cout << "Step 1: Generating mock data...\n";
    
    auto eri_ao = generate_eri(n_bf);
    auto C_mo = generate_mo_coefficients(n_bf, n_mo);
    
    std::cout << "  AO ERIs generated (" << n_bf << "^4 = " 
              << std::pow(n_bf, 4) << " elements)\n";
    std::cout << "  MO coefficients generated (" << n_bf << " × " 
              << n_mo << " matrix)\n\n";
    
    // ========================================================================
    // Step 2: Cholesky decomposition
    // ========================================================================
    
    std::cout << "Step 2: Cholesky decomposition of AO ERIs...\n";
    
    CholeskyERI cholesky(threshold);
    
    auto t_chol_start = std::chrono::high_resolution_clock::now();
    auto chol_result = cholesky.decompose(eri_ao);
    auto t_chol_end = std::chrono::high_resolution_clock::now();
    
    double t_chol_ms = std::chrono::duration<double, std::milli>(
        t_chol_end - t_chol_start).count();
    
    const auto& L_ao = cholesky.get_L_vectors();
    int M = L_ao.size();
    
    std::cout << "  Cholesky vectors (M): " << M << "\n";
    std::cout << "  M/N ratio: " << std::fixed << std::setprecision(2) 
              << (double)M / n_bf << "\n";
    std::cout << "  Compression: " << std::setprecision(1) 
              << cholesky.compression_ratio() << "×\n";
    std::cout << "  Time: " << t_chol_ms << " ms\n\n";
    
    // ========================================================================
    // Step 3: Transform Cholesky vectors to MO basis
    // ========================================================================
    
    std::cout << "Step 3: Transforming Cholesky vectors to MO basis...\n";
    std::cout << "  Method: L^k_MO = C^T · L^k_AO · C\n";
    std::cout << "  Cost: O(M × N^3) vs O(N^5) traditional\n";
    
    auto t_trans_start = std::chrono::high_resolution_clock::now();
    auto L_mo = transform_cholesky_to_mo(L_ao, C_mo, n_bf);
    auto t_trans_end = std::chrono::high_resolution_clock::now();
    
    double t_trans_ms = std::chrono::duration<double, std::milli>(
        t_trans_end - t_trans_start).count();
    
    std::cout << "  MO Cholesky vectors: " << L_mo.size() << "\n";
    std::cout << "  Each vector size: " << n_mo << " × " << n_mo << "\n";
    std::cout << "  Transform time: " << t_trans_ms << " ms\n\n";
    
    // ========================================================================
    // Step 4: Direct AO→MO transformation (reference)
    // ========================================================================
    
    std::cout << "Step 4: Direct AO→MO transformation (reference)...\n";
    std::cout << "  Method: Traditional O(N^5) algorithm\n";
    
    auto t_direct_start = std::chrono::high_resolution_clock::now();
    auto eri_mo_direct = direct_ao_to_mo_transform(eri_ao, C_mo, n_bf, n_mo);
    auto t_direct_end = std::chrono::high_resolution_clock::now();
    
    double t_direct_ms = std::chrono::duration<double, std::milli>(
        t_direct_end - t_direct_start).count();
    
    std::cout << "  MO ERIs computed (" << n_mo << "^4 = " 
              << std::pow(n_mo, 4) << " elements)\n";
    std::cout << "  Transform time: " << t_direct_ms << " ms\n\n";
    
    // ========================================================================
    // Step 5: Validate reconstruction
    // ========================================================================
    
    std::cout << "Step 5: Validating Cholesky reconstruction...\n";
    std::cout << "  Comparing CD-reconstructed vs direct MO ERIs\n";
    
    double max_error = 0.0;
    double sum_sq_error = 0.0;
    int n_elements = 0;
    
    int max_err_p = 0, max_err_q = 0, max_err_r = 0, max_err_s = 0;
    
    for (int p = 0; p < n_mo; ++p) {
        for (int q = 0; q < n_mo; ++q) {
            for (int r = 0; r < n_mo; ++r) {
                for (int s = 0; s < n_mo; ++s) {
                    double direct = eri_mo_direct(p, q, r, s);
                    double cholesky = reconstruct_mo_eri(p, q, r, s, L_mo);
                    
                    double error = std::abs(direct - cholesky);
                    
                    if (error > max_error) {
                        max_error = error;
                        max_err_p = p; max_err_q = q;
                        max_err_r = r; max_err_s = s;
                    }
                    
                    sum_sq_error += error * error;
                    n_elements++;
                }
            }
        }
    }
    
    double rms_error = std::sqrt(sum_sq_error / n_elements);
    
    std::cout << "  Elements checked: " << n_elements << "\n";
    std::cout << "  Max error: " << std::scientific << std::setprecision(6) 
              << max_error << " Ha (" << std::fixed << std::setprecision(2)
              << max_error * 1e6 << " µHa)\n";
    std::cout << "    at (pqrs) = (" << max_err_p << "," << max_err_q << ","
              << max_err_r << "," << max_err_s << ")\n";
    std::cout << "  RMS error: " << std::scientific << rms_error << " Ha\n\n";
    
    // ========================================================================
    // Performance comparison
    // ========================================================================
    
    std::cout << "╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Performance Summary                                              ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n\n";
    
    double total_cholesky = t_chol_ms + t_trans_ms;
    double speedup = t_direct_ms / total_cholesky;
    
    std::cout << "┌────────────────────────────┬──────────────┬──────────────┐\n";
    std::cout << "│ Method                     │ Time (ms)    │ Speedup      │\n";
    std::cout << "├────────────────────────────┼──────────────┼──────────────┤\n";
    std::cout << "│ Traditional O(N^5)         │ " << std::setw(12) << std::fixed 
              << std::setprecision(3) << t_direct_ms << " │ 1.0×         │\n";
    std::cout << "│ Cholesky decomp + transform│ " << std::setw(12) 
              << total_cholesky << " │ " << std::setw(12) << std::setprecision(1) 
              << speedup << "× │\n";
    std::cout << "│   - Decomposition          │ " << std::setw(12) << std::setprecision(3)
              << t_chol_ms << " │              │\n";
    std::cout << "│   - MO transform           │ " << std::setw(12) 
              << t_trans_ms << " │              │\n";
    std::cout << "└────────────────────────────┴──────────────┴──────────────┘\n\n";
    
    std::cout << "Storage reduction: " << std::fixed << std::setprecision(1)
              << cholesky.compression_ratio() << "×\n";
    std::cout << "  Traditional: " << std::pow(n_mo, 4) << " doubles = "
              << (std::pow(n_mo, 4) * 8 / 1024.0) << " KB\n";
    std::cout << "  Cholesky:    " << (M * n_mo * n_mo) << " doubles = "
              << (M * n_mo * n_mo * 8 / 1024.0) << " KB\n\n";
    
    // ========================================================================
    // Test result
    // ========================================================================
    
    bool pass = (max_error * 1e6 < 1.0) && (speedup > 0.5);
    
    std::cout << "╔═══════════════════════════════════════════════════════════════════╗\n";
    
    if (pass) {
        std::cout << "║  ✓ TEST PASSED - CD-MO transformation validated!                 ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n\n";
        std::cout << "Validation successful:\n";
        std::cout << "  ✓ Accuracy: " << max_error * 1e6 << " µHa < 1 µHa target\n";
        std::cout << "  ✓ Method validated for CD-CASPT2 integration\n";
        std::cout << "  ✓ Ready for production implementation\n\n";
    } else {
        std::cout << "║  ✗ TEST FAILED - Check implementation                            ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n\n";
        std::cout << "Issues:\n";
        if (max_error * 1e6 >= 1.0) {
            std::cout << "  ✗ Error " << max_error * 1e6 << " µHa exceeds 1 µHa\n";
        }
        if (speedup <= 0.5) {
            std::cout << "  ✗ Performance worse than traditional\n";
        }
        std::cout << "\n";
    }
    
    std::cout << "Next steps:\n";
    std::cout << "1. Integrate into CASPT2::compute() method\n";
    std::cout << "2. Test with real molecular systems (H2O/STO-3G)\n";
    std::cout << "3. Benchmark speedup on larger systems\n";
    std::cout << "4. Extend to CASPT3/4/5\n\n";
    
    exit(pass ? 0 : 1);
}
