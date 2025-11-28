/**
 * @file excitation_generator_test.cc
 * @brief Test excitation generator for on-the-fly methods
 * 
 * Verifies that excitation generation works correctly and counts are accurate.
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-14
 */

#include <iostream>
#include <iomanip>
#include "mshqc/ci/determinant.h"
#include "mshqc/ci/excitation_generator.h"

using namespace mshqc::ci;

int main() {
    std::cout << "=============================================================\n";
    std::cout << "Excitation Generator Test\n";
    std::cout << "=============================================================\n\n";
    
    // Test system: 3 electrons in 6 orbitals
    // HF configuration: |α:0,1⟩ |β:0⟩
    std::vector<int> occ_alpha = {0, 1};
    std::vector<int> occ_beta = {0};
    Determinant hf_det(occ_alpha, occ_beta);
    
    int n_orb = 6;
    int nocc_a = 2;
    int nocc_b = 1;
    int nvirt_a = n_orb - nocc_a;  // 4
    int nvirt_b = n_orb - nocc_b;   // 5
    
    std::cout << "System info:\n";
    std::cout << "  n_orb = " << n_orb << "\n";
    std::cout << "  Occupied alpha: {0, 1}\n";
    std::cout << "  Occupied beta:  {0}\n";
    std::cout << "  Virtual alpha:  {2, 3, 4, 5}\n";
    std::cout << "  Virtual beta:   {1, 2, 3, 4, 5}\n\n";
    
    // Expected counts
    int expected_singles_a = nocc_a * nvirt_a;  // 2 * 4 = 8
    int expected_singles_b = nocc_b * nvirt_b;  // 1 * 5 = 5
    int expected_singles = expected_singles_a + expected_singles_b;  // 13
    
    int expected_doubles_aa = (nocc_a * (nocc_a - 1) / 2) * (nvirt_a * (nvirt_a - 1) / 2);  // 1 * 6 = 6
    int expected_doubles_bb = (nocc_b * (nocc_b - 1) / 2) * (nvirt_b * (nvirt_b - 1) / 2);  // 0 * 10 = 0
    int expected_doubles_ab = nocc_a * nocc_b * nvirt_a * nvirt_b;  // 2 * 1 * 4 * 5 = 40
    int expected_doubles = expected_doubles_aa + expected_doubles_bb + expected_doubles_ab;  // 46
    
    int expected_total = expected_singles + expected_doubles;  // 59
    
    std::cout << "Expected excitation counts:\n";
    std::cout << "  Singles (alpha): " << expected_singles_a << "\n";
    std::cout << "  Singles (beta):  " << expected_singles_b << "\n";
    std::cout << "  Total singles:   " << expected_singles << "\n";
    std::cout << "  Doubles (aa):    " << expected_doubles_aa << "\n";
    std::cout << "  Doubles (bb):    " << expected_doubles_bb << "\n";
    std::cout << "  Doubles (ab):    " << expected_doubles_ab << "\n";
    std::cout << "  Total doubles:   " << expected_doubles << "\n";
    std::cout << "  Total connected: " << expected_total << "\n\n";
    
    // Test 1: Count singles
    std::cout << "=== Test 1: Generate Singles ===\n";
    int count_singles = 0;
    generate_singles(hf_det, n_orb, [&](const GeneratedExcitation& exc) {
        count_singles++;
    });
    
    std::cout << "Generated " << count_singles << " singles\n";
    std::cout << "Expected:  " << expected_singles << " singles\n";
    if (count_singles == expected_singles) {
        std::cout << "✓ PASS\n\n";
    } else {
        std::cout << "✗ FAIL\n\n";
    }
    
    // Test 2: Count doubles
    std::cout << "=== Test 2: Generate Doubles ===\n";
    int count_doubles = 0;
    generate_doubles(hf_det, n_orb, [&](const GeneratedExcitation& exc) {
        count_doubles++;
    });
    
    std::cout << "Generated " << count_doubles << " doubles\n";
    std::cout << "Expected:  " << expected_doubles << " doubles\n";
    if (count_doubles == expected_doubles) {
        std::cout << "✓ PASS\n\n";
    } else {
        std::cout << "✗ FAIL\n\n";
    }
    
    // Test 3: Total connected excitations
    std::cout << "=== Test 3: All Connected Excitations ===\n";
    int count_total = 0;
    generate_connected_excitations(hf_det, n_orb, [&](const GeneratedExcitation& exc) {
        count_total++;
    });
    
    std::cout << "Generated " << count_total << " total\n";
    std::cout << "Expected:  " << expected_total << " total\n";
    if (count_total == expected_total) {
        std::cout << "✓ PASS\n\n";
    } else {
        std::cout << "✗ FAIL\n\n";
    }
    
    // Test 4: Count function
    std::cout << "=== Test 4: Count Function ===\n";
    int count_func = count_connected_excitations(hf_det, n_orb);
    std::cout << "count_connected_excitations() = " << count_func << "\n";
    std::cout << "Expected:                       " << expected_total << "\n";
    if (count_func == expected_total) {
        std::cout << "✓ PASS\n\n";
    } else {
        std::cout << "✗ FAIL\n\n";
    }
    
    // Test 5: Sample a few excitations
    std::cout << "=== Test 5: Sample Excitations ===\n";
    std::cout << "First 5 singles:\n";
    int sample_count = 0;
    generate_singles(hf_det, n_orb, [&](const GeneratedExcitation& exc) {
        if (sample_count < 5) {
            std::cout << "  " << sample_count + 1 << ". ";
            std::cout << "i=" << exc.i << ", a=" << exc.a << ", ";
            std::cout << "spin=" << (exc.spin_i ? "alpha" : "beta") << "\n";
            sample_count++;
        }
    });
    std::cout << "\n";
    
    std::cout << "First 5 doubles:\n";
    sample_count = 0;
    generate_doubles(hf_det, n_orb, [&](const GeneratedExcitation& exc) {
        if (sample_count < 5) {
            std::cout << "  " << sample_count + 1 << ". ";
            std::cout << "i=" << exc.i << ", j=" << exc.j << ", ";
            std::cout << "a=" << exc.a << ", b=" << exc.b << ", ";
            std::cout << "spins=" << (exc.spin_i ? "a" : "b") << (exc.spin_j ? "a" : "b") << "\n";
            sample_count++;
        }
    });
    std::cout << "\n";
    
    // Summary
    std::cout << "=============================================================\n";
    std::cout << "Summary:\n";
    bool all_pass = (count_singles == expected_singles) && 
                    (count_doubles == expected_doubles) && 
                    (count_total == expected_total) && 
                    (count_func == expected_total);
    
    if (all_pass) {
        std::cout << "✓ All tests PASSED!\n";
        std::cout << "Excitation generator is working correctly.\n";
    } else {
        std::cout << "✗ Some tests FAILED!\n";
    }
    std::cout << "=============================================================\n";
    
    return all_pass ? 0 : 1;
}
