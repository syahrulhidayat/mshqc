/**
 * @file test_cholesky_n20.cc
 * @brief Test Cholesky ERI for N=20 system
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <random>
#include "mshqc/integrals/cholesky_eri.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace mshqc::integrals;

Eigen::Tensor<double, 4> generate_eri(int n) {
    Eigen::Tensor<double, 4> eri(n, n, n, n);
    eri.setZero();
    
    std::mt19937 gen(42 + n);
    std::normal_distribution<> dis(0.0, 0.05);
    
    int rank = std::max(n / 2, 3);
    int n_pairs = n * n;
    
    Eigen::MatrixXd L = Eigen::MatrixXd::Random(n_pairs, rank) * 0.15;
    Eigen::MatrixXd V = L * L.transpose();
    
    for (int ij = 0; ij < n_pairs; ++ij) {
        V(ij, ij) += 0.05 + std::abs(dis(gen));
    }
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int ij = i * n + j;
            for (int k = 0; k < n; ++k) {
                for (int l = 0; l < n; ++l) {
                    eri(i, j, k, l) = V(ij, k * n + l);
                }
            }
        }
    }
    
    return eri;
}

int main() {
    const int N = 20;
    const double threshold = 1e-6;
    
    std::cout << "\n=== Cholesky ERI Test: N=" << N << " ===\n\n";
    
    auto eri = generate_eri(N);
    CholeskyERI cholesky(threshold);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = cholesky.decompose(eri);
    auto end = std::chrono::high_resolution_clock::now();
    
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Validate
    int sample = 5;
    double max_err = 0.0;
    double sum_sq = 0.0;
    int n_samples = 0;
    
    for (int i = 0; i < sample; ++i) {
        for (int j = 0; j < sample; ++j) {
            for (int k = 0; k < sample; ++k) {
                for (int l = 0; l < sample; ++l) {
                    double exact = eri(i, j, k, l);
                    double recon = cholesky.reconstruct(i, j, k, l);
                    double err = std::abs(exact - recon);
                    max_err = std::max(max_err, err);
                    sum_sq += err * err;
                    n_samples++;
                }
            }
        }
    }
    
    double rms = std::sqrt(sum_sq / n_samples);
    size_t exact_bytes = std::pow(N, 4) * sizeof(double);
    size_t cd_bytes = cholesky.storage_bytes();
    double compression = cholesky.compression_ratio();
    
    std::cout << "Cholesky vectors: " << result.n_vectors << "\n";
    std::cout << "M/N ratio:        " << std::fixed << std::setprecision(2) 
              << (double)result.n_vectors / N << "\n";
    std::cout << "Compression:      " << std::setprecision(1) << compression << "×\n";
    std::cout << "Time:             " << time_ms << " ms\n";
    std::cout << "Max error:        " << std::scientific << max_err << " Ha ("
              << std::fixed << max_err * 1e6 << " µHa)\n";
    std::cout << "RMS error:        " << std::scientific << rms << " Ha\n";
    std::cout << "Storage:          " << std::fixed << (cd_bytes / 1024.0) 
              << " KB (exact: " << (exact_bytes / 1024.0 / 1024.0) << " MB)\n";
    std::cout << "Memory savings:   " << std::setprecision(1) 
              << 100.0 * (1.0 - (double)cd_bytes / exact_bytes) << "%\n";
    
    bool pass = (max_err * 1e6 < 10.0) && (result.n_vectors > 0) && (compression >= 1.0);
    std::cout << "\nStatus: " << (pass ? "✓ PASS" : "✗ FAIL") << "\n\n";
    
    exit(pass ? 0 : 1);
}
