/**
 * @file cisd_onthefly_test.cc
 * @brief Integration test for on-the-fly Davidson sigma-vector
 * 
 * Tests:
 * 1. Compare dense vs on-the-fly CISD energies
 * 2. Validate numerical accuracy (< 1e-12 Ha)
 * 3. Measure performance and memory usage
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-14
 */

#include "mshqc/ci/cisd.h"
#include "mshqc/ci/determinant.h"
#include "mshqc/ci/slater_condon.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

using namespace mshqc::ci;
using namespace std::chrono;

// ============================================================================
// MOCK INTEGRALS FOR TESTING
// ============================================================================

CIIntegrals generate_mock_integrals(int n_orb) {
    CIIntegrals ints;
    
    ints.h_alpha = Eigen::MatrixXd::Zero(n_orb, n_orb);
    ints.h_beta = Eigen::MatrixXd::Zero(n_orb, n_orb);
    
    // Diagonal one-electron energies
    for (int i = 0; i < n_orb; i++) {
        double e = -2.0 + 0.3 * i;
        ints.h_alpha(i,i) = e;
        ints.h_beta(i,i) = e;
    }
    
    // Off-diagonal one-electron coupling
    for (int i = 0; i < n_orb; i++) {
        for (int j = i+1; j < n_orb; j++) {
            double v = 0.05 * std::exp(-0.2 * std::abs(i-j));
            ints.h_alpha(i,j) = v;
            ints.h_alpha(j,i) = v;
            ints.h_beta(i,j) = v;
            ints.h_beta(j,i) = v;
        }
    }
    
    // Two-electron integrals (simplified model)
    ints.eri_aaaa = Eigen::Tensor<double, 4>(n_orb, n_orb, n_orb, n_orb);
    ints.eri_bbbb = Eigen::Tensor<double, 4>(n_orb, n_orb, n_orb, n_orb);
    ints.eri_aabb = Eigen::Tensor<double, 4>(n_orb, n_orb, n_orb, n_orb);
    ints.eri_aaaa.setZero();
    ints.eri_bbbb.setZero();
    ints.eri_aabb.setZero();
    
    for (int i = 0; i < n_orb; i++) {
        for (int j = 0; j < n_orb; j++) {
            for (int k = 0; k < n_orb; k++) {
                for (int l = 0; l < n_orb; l++) {
                    double r_ij = std::abs(i - j) + 0.5;
                    double r_kl = std::abs(k - l) + 0.5;
                    double coulomb = 0.4 / (r_ij * r_kl + 1.0);
                    
                    double r_ik = std::abs(i - k) + 0.5;
                    double r_jl = std::abs(j - l) + 0.5;
                    double exchange = 0.2 / (r_ik * r_jl + 1.0);
                    
                    ints.eri_aaaa(i,j,k,l) = coulomb - exchange;
                    ints.eri_bbbb(i,j,k,l) = coulomb - exchange;
                    ints.eri_aabb(i,j,k,l) = coulomb;
                }
            }
        }
    }
    
    ints.e_nuc = 0.0;
    return ints;
}

// ============================================================================
// TEST CASES
// ============================================================================

void test_accuracy(int nocc_a, int nocc_b, int nvirt_a, int nvirt_b) {
    int n_orb = nocc_a + nvirt_a;
    
    std::cout << "\n============================================================\n";
    std::cout << "ACCURACY TEST: " << nocc_a + nocc_b << " electrons in " 
              << n_orb << " orbitals\n";
    std::cout << "  Occupied: α=" << nocc_a << ", β=" << nocc_b << "\n";
    std::cout << "  Virtual:  α=" << nvirt_a << ", β=" << nvirt_b << "\n";
    std::cout << "============================================================\n";
    
    // Setup
    auto ints = generate_mock_integrals(n_orb);
    
    std::vector<int> occ_a, occ_b;
    for (int i = 0; i < nocc_a; i++) occ_a.push_back(i);
    for (int i = 0; i < nocc_b; i++) occ_b.push_back(i);
    Determinant hf_det(occ_a, occ_b);
    
    CISD cisd(ints, hf_det, nocc_a, nocc_b, nvirt_a, nvirt_b);
    
    // Test 1: Dense mode
    std::cout << "\n--- TEST 1: Dense Mode ---\n";
    CISDOptions opts_dense;
    opts_dense.use_onthefly = false;
    opts_dense.auto_onthefly = false;
    opts_dense.auto_sparse = false;
    opts_dense.verbose = true;
    
    auto start_dense = high_resolution_clock::now();
    auto result_dense = cisd.compute(opts_dense);
    auto end_dense = high_resolution_clock::now();
    double time_dense = duration<double, std::milli>(end_dense - start_dense).count();
    
    // Test 2: On-the-fly mode
    std::cout << "\n--- TEST 2: On-the-Fly Mode ---\n";
    CISDOptions opts_onthefly;
    opts_onthefly.use_onthefly = true;
    opts_onthefly.auto_sparse = false;
    opts_onthefly.verbose = true;
    
    auto start_onthefly = high_resolution_clock::now();
    auto result_onthefly = cisd.compute(opts_onthefly);
    auto end_onthefly = high_resolution_clock::now();
    double time_onthefly = duration<double, std::milli>(end_onthefly - start_onthefly).count();
    
    // Compare results
    std::cout << "\n=== COMPARISON ===\n";
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Dense energy:      " << result_dense.e_cisd << " Ha\n";
    std::cout << "On-the-fly energy: " << result_onthefly.e_cisd << " Ha\n";
    
    double energy_diff = std::abs(result_dense.e_cisd - result_onthefly.e_cisd);
    std::cout << "\nEnergy difference: " << std::scientific << std::setprecision(3) 
              << energy_diff << " Ha\n";
    
    // Check accuracy
    double threshold = 1e-12;
    if (energy_diff < threshold) {
        std::cout << "✅ PASSED: Accuracy within " << threshold << " Ha\n";
    } else {
        std::cout << "❌ FAILED: Accuracy exceeds " << threshold << " Ha\n";
    }
    
    // Performance comparison
    std::cout << "\n=== PERFORMANCE ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Dense time:        " << time_dense << " ms\n";
    std::cout << "On-the-fly time:   " << time_onthefly << " ms\n";
    std::cout << "Speedup:           " << (time_dense / time_onthefly) << "×\n";
    
    // Memory estimation
    int N = result_dense.n_determinants;
    size_t mem_dense = static_cast<size_t>(N) * N * sizeof(double);
    size_t mem_onthefly = N * 80;  // ~80 bytes per determinant (hash map)
    
    std::cout << "\n=== MEMORY ===\n";
    std::cout << "Dense:             " << (mem_dense / 1024.0 / 1024.0) << " MB\n";
    std::cout << "On-the-fly:        " << (mem_onthefly / 1024.0 / 1024.0) << " MB\n";
    std::cout << "Reduction:         " << std::setprecision(1) 
              << (100.0 * (1.0 - static_cast<double>(mem_onthefly) / mem_dense)) << "%\n";
}

void test_auto_switching() {
    std::cout << "\n============================================================\n";
    std::cout << "AUTO-SWITCHING TEST\n";
    std::cout << "============================================================\n";
    
    // Small system: N < 200 (should use dense)
    std::cout << "\n--- Small System (N ~ 100) ---\n";
    {
        auto ints = generate_mock_integrals(6);
        std::vector<int> occ_a = {0, 1};
        std::vector<int> occ_b = {0, 1};
        Determinant hf_det(occ_a, occ_b);
        
        CISD cisd(ints, hf_det, 2, 2, 4, 4);
        
        CISDOptions opts;
        opts.auto_onthefly = true;
        opts.onthefly_threshold = 200;
        opts.auto_sparse = false;
        opts.verbose = true;
        
        auto result = cisd.compute(opts);
        std::cout << "Expected: Dense mode (N < 200)\n";
    }
    
    // Large system: N > 200 (should use on-the-fly)
    std::cout << "\n--- Large System (N ~ 300) ---\n";
    {
        auto ints = generate_mock_integrals(8);
        std::vector<int> occ_a = {0, 1, 2};
        std::vector<int> occ_b = {0, 1, 2};
        Determinant hf_det(occ_a, occ_b);
        
        CISD cisd(ints, hf_det, 3, 3, 5, 5);
        
        CISDOptions opts;
        opts.auto_onthefly = true;
        opts.onthefly_threshold = 200;
        opts.auto_sparse = false;
        opts.verbose = true;
        
        auto result = cisd.compute(opts);
        std::cout << "Expected: On-the-fly mode (N > 200)\n";
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║  CISD ON-THE-FLY INTEGRATION TEST                        ║\n";
    std::cout << "║  Testing Davidson on-the-fly sigma-vector implementation ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n";
    
    try {
        // Test 1: Small system (N ~ 93)
        test_accuracy(2, 2, 3, 3);
        
        // Test 2: Medium system (N ~ 216)
        test_accuracy(3, 2, 3, 3);
        
        // Test 3: Auto-switching logic
        test_auto_switching();
        
        std::cout << "\n╔═══════════════════════════════════════════════════════════╗\n";
        std::cout << "║  ALL TESTS COMPLETED                                      ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ ERROR: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
