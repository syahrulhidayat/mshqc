/**
 * @file mp_ci_adapter.cc
 * @brief Implementation of MP-CI integration adapter
 * 
 * INTEGRATION TASK 1.1: Determinant infrastructure reuse
 * 
 * This module provides clean interface between  (MP) and  (CI)
 * without modifying their original code. Following separation of concerns principle.
 * 
 * Theory References:
 *   - Raghavachari et al. (1989), Chem. Phys. Lett. 157, 479
 *   - Pople et al. (1977), Int. J. Quantum Chem. 11, 149
 *   - Szabo & Ostlund (1996), Modern Quantum Chemistry, Appendix A
 * 
 * @author Muhamad Syahrul Hidayat ()
 * @date 2025-11-16
 * @license MIT License
 * 
 * Copyright (c) 2025 Muhamad Syahrul Hidayat
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
 * @note ORIGINAL IMPLEMENTATION - Integration between CI and MP modules
 * @note NO CODE COPIED from existing quantum chemistry software
 */

#include "mshqc/integration/mp_ci_adapter.h"
#include <algorithm>

namespace mshqc {
namespace integration {

// Helper: Build HF determinant from occupation numbers
ci::Determinant build_hf_determinant(int n_occ_alpha, int n_occ_beta) {
    // Assume orbitals filled from 0 to n_occ-1 (Aufbau principle)
    std::vector<int> alpha_occ;
    std::vector<int> beta_occ;
    
    for(int i = 0; i < n_occ_alpha; i++) {
        alpha_occ.push_back(i);
    }
    
    for(int i = 0; i < n_occ_beta; i++) {
        beta_occ.push_back(i);
    }
    
    return ci::Determinant(alpha_occ, beta_occ);
}

// Generate alpha-alpha-alpha triple excitations
// 
// Algorithm:
//   For each combination of (i,j,k) from occupied α orbitals:
//     For each combination of (a,b,c) from virtual α orbitals:
//       Generate |Φ_ijk^abc⟩ using CI determinant infrastructure
//       Call callback with excitation indices
//
// This replaces manual 6-deep loops:
//   for i in occ:
//     for j in occ:
//       for k in occ:
//         for a in virt:
//           for b in virt:
//             for c in virt:
//               // compute T3 amplitude
//
// Theory: Pople et al. (1977), Eq. 13
void generate_triples_alpha(const ci::Determinant& hf_det,
                             int n_occ_alpha,
                             int n_virt_alpha,
                             std::function<void(const TripleExcitation&)> callback) {
    
    // Need at least 3 occupied orbitals for triples!
    if(n_occ_alpha < 3) {
        // No triples possible (e.g. Li with 2α electrons)
        return;
    }
    
    if(n_virt_alpha < 3) {
        // No triples possible (need 3 virtual orbitals)
        return;
    }
    
    // Get α occupied orbitals from HF determinant
    auto occ_alpha = hf_det.alpha_occupations();
    
    // Generate all occupied indices (i < j < k)
    // Use ordered loop to avoid redundant excitations
    for(int ii = 0; ii < n_occ_alpha; ii++) {
        for(int jj = ii + 1; jj < n_occ_alpha; jj++) {
            for(int kk = jj + 1; kk < n_occ_alpha; kk++) {
                int i = occ_alpha[ii];
                int j = occ_alpha[jj];
                int k = occ_alpha[kk];
                
                // Generate all virtual indices (a < b < c)
                // Virtual orbitals: n_occ_alpha, n_occ_alpha+1, ..., n_basis-1
                // Indexed as: 0, 1, ..., n_virt_alpha-1 (offset by n_occ_alpha)
                for(int aa = 0; aa < n_virt_alpha; aa++) {
                    for(int bb = aa + 1; bb < n_virt_alpha; bb++) {
                        for(int cc = bb + 1; cc < n_virt_alpha; cc++) {
                            // Virtual orbital indices (offset by n_occ)
                            int a = aa;
                            int b = bb;
                            int c = cc;
                            
                            // Create triple excitation struct
                            TripleExcitation exc(i, j, k, a, b, c, true);
                            
                            // Call user callback
                            callback(exc);
                        }
                    }
                }
            }
        }
    }
}

// Generate beta-beta-beta triple excitations
// Same algorithm as alpha but for β spin
void generate_triples_beta(const ci::Determinant& hf_det,
                            int n_occ_beta,
                            int n_virt_beta,
                            std::function<void(const TripleExcitation&)> callback) {
    
    if(n_occ_beta < 3 || n_virt_beta < 3) {
        return;
    }
    
    auto occ_beta = hf_det.beta_occupations();
    
    for(int ii = 0; ii < n_occ_beta; ii++) {
        for(int jj = ii + 1; jj < n_occ_beta; jj++) {
            for(int kk = jj + 1; kk < n_occ_beta; kk++) {
                int i = occ_beta[ii];
                int j = occ_beta[jj];
                int k = occ_beta[kk];
                
                for(int aa = 0; aa < n_virt_beta; aa++) {
                    for(int bb = aa + 1; bb < n_virt_beta; bb++) {
                        for(int cc = bb + 1; cc < n_virt_beta; cc++) {
                            int a = aa;
                            int b = bb;
                            int c = cc;
                            
                            TripleExcitation exc(i, j, k, a, b, c, true);
                            callback(exc);
                        }
                    }
                }
            }
        }
    }
}

// Count triple excitations (memory estimation)
// Formula: C(n_occ, 3) * C(n_virt, 3)
size_t count_triple_excitations(int n_occ, int n_virt) {
    if(n_occ < 3 || n_virt < 3) {
        return 0;
    }
    
    // Binomial coefficient: C(n, k) = n! / (k! * (n-k)!)
    // C(n, 3) = n * (n-1) * (n-2) / 6
    size_t n_occ_comb = (n_occ * (n_occ - 1) * (n_occ - 2)) / 6;
    size_t n_virt_comb = (n_virt * (n_virt - 1) * (n_virt - 2)) / 6;
    
    return n_occ_comb * n_virt_comb;
}

} // namespace integration
} // namespace mshqc
