/**
 * @file determinant_test.cc
 * @brief Test Determinant class basics
 * 
 * Tests:
 * 1. Construction from occupation lists
 * 2. Single and double excitations
 * 3. Phase calculations
 * 4. Excitation level counting
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
 */

#include "mshqc/ci/determinant.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace mshqc::ci;

void test_construction() {
    std::cout << "=== Test 1: Determinant Construction ===\n";
    
    // H2 molecule: 2 electrons in 2 orbitals
    // Ground state: |11⟩ = both electrons in orbital 0
    Determinant hf_det(std::vector<int>{0}, std::vector<int>{0});
    
    std::cout << "HF determinant: " << hf_det.to_string() << "\n";
    std::cout << "  n_alpha = " << hf_det.n_alpha() << "\n";
    std::cout << "  n_beta  = " << hf_det.n_beta() << "\n";
    
    // Check occupation
    if (hf_det.is_occupied(0, true) && hf_det.is_occupied(0, false)) {
        std::cout << "✓ Orbital 0 occupied (α and β)\n";
    } else {
        std::cout << "✗ FAILED: Orbital 0 should be occupied\n";
    }
    
    if (!hf_det.is_occupied(1, true)) {
        std::cout << "✓ Orbital 1 virtual\n";
    } else {
        std::cout << "✗ FAILED: Orbital 1 should be virtual\n";
    }
    
    std::cout << "\n";
}

void test_single_excitation() {
    std::cout << "=== Test 2: Single Excitation ===\n";
    
    // Start from |↑↓, 0, 0, ...⟩
    Determinant hf(std::vector<int>{0}, std::vector<int>{0});
    
    // Single excitation: α electron 0 → 1
    // Result: |0, ↑↓, 0, ...⟩
    Determinant excited = hf.single_excite(0, 1, true);
    
    std::cout << "HF:      " << hf.to_string() << "\n";
    std::cout << "Excited: " << excited.to_string() << "\n";
    
    // Check excitation level
    int diff = hf.count_differences(excited);
    if (diff == 1) {
        std::cout << "✓ Single excitation detected (diff = " << diff << ")\n";
    } else {
        std::cout << "✗ FAILED: Expected diff = 1, got " << diff << "\n";
    }
    
    // Check phase
    int phase = hf.phase(0, 1, true);
    std::cout << "  Phase factor: " << phase << "\n";
    
    std::cout << "\n";
}

void test_double_excitation() {
    std::cout << "=== Test 3: Double Excitation ===\n";
    
    // H2O: 5 occupied orbitals (simplified)
    // α: 0, 1, 2, 3, 4
    // β: 0, 1, 2, 3, 4
    Determinant hf({0, 1, 2, 3, 4}, {0, 1, 2, 3, 4});
    
    // Double excitation: αα (3, 4) → (5, 6)
    Determinant excited = hf.double_excite(3, 4, 5, 6, true, true);
    
    std::cout << "HF:      5 α, 5 β electrons\n";
    std::cout << "Excited: " << excited.to_string() << "\n";
    
    int diff = hf.count_differences(excited);
    if (diff == 2) {
        std::cout << "✓ Double excitation detected (diff = " << diff << ")\n";
    } else {
        std::cout << "✗ FAILED: Expected diff = 2, got " << diff << "\n";
    }
    
    auto [n_alpha, n_beta] = hf.excitation_level(excited);
    std::cout << "  Excitation: " << n_alpha << " α, " << n_beta << " β\n";
    
    if (n_alpha == 2 && n_beta == 0) {
        std::cout << "✓ Correct spin breakdown\n";
    }
    
    std::cout << "\n";
}

void test_excitation_finder() {
    std::cout << "=== Test 4: Excitation Finder ===\n";
    
    Determinant hf({0, 1}, {0, 1});
    Determinant exc = hf.single_excite(1, 3, true);
    
    auto excitation = find_excitation(hf, exc);
    
    std::cout << "Excitation level: " << excitation.level << "\n";
    std::cout << "Occupied α: ";
    for (int i : excitation.occ_alpha) std::cout << i << " ";
    std::cout << "\n";
    std::cout << "Virtual α:  ";
    for (int a : excitation.virt_alpha) std::cout << a << " ";
    std::cout << "\n";
    
    if (excitation.level == 1 && 
        excitation.occ_alpha.size() == 1 &&
        excitation.occ_alpha[0] == 1 &&
        excitation.virt_alpha[0] == 3) {
        std::cout << "✓ Correct excitation detected: 1 → 3 (α)\n";
    } else {
        std::cout << "✗ FAILED\n";
    }
    
    std::cout << "\n";
}

void test_comparison() {
    std::cout << "=== Test 5: Comparison Operators ===\n";
    
    Determinant det1({0, 1}, {0});
    Determinant det2({0, 1}, {0});
    Determinant det3({0, 2}, {0});
    
    if (det1 == det2) {
        std::cout << "✓ Equality operator works\n";
    } else {
        std::cout << "✗ FAILED: det1 should equal det2\n";
    }
    
    if (det1 != det3) {
        std::cout << "✓ Inequality operator works\n";
    } else {
        std::cout << "✗ FAILED: det1 should not equal det3\n";
    }
    
    // Test sorting (important for FCI)
    std::vector<Determinant> dets = {det3, det1, det2};
    std::sort(dets.begin(), dets.end());
    
    std::cout << "✓ Determinants can be sorted\n";
    std::cout << "\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "   Determinant Class Test Suite\n";
    std::cout << "   AI Agent 2 (CI Specialist)\n";
    std::cout << "========================================\n\n";
    
    try {
        test_construction();
        test_single_excitation();
        test_double_excitation();
        test_excitation_finder();
        test_comparison();
        
        std::cout << "========================================\n";
        std::cout << "   All Tests Completed!\n";
        std::cout << "========================================\n";
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
