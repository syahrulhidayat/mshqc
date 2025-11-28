/**
 * @file slater_condon.cc
 * @brief Implementation of Slater-Condon rules
 * 
 * REFERENCES:
 * - Slater (1929), Phys. Rev. 34, 1293
 * - Condon (1930), Phys. Rev. 36, 1121
 * - Szabo & Ostlund (1996), Appendix A
 * - Shavitt & Bartlett (2009), Ch. 3, Sec. 3.2
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

#include "mshqc/ci/slater_condon.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace mshqc {
namespace ci {

// Main dispatcher: compute ⟨Φ|H|Φ'⟩ based on excitation level
double hamiltonian_element(const Determinant& bra, 
                           const Determinant& ket,
                           const CIIntegrals& ints) {
    
    // Find excitation between bra and ket
    auto exc = find_excitation(bra, ket);
    
    if (exc.level == 0) {
        // Same determinant: diagonal element
        return diagonal_element(bra, ints);
    }
    else if (exc.level == 1) {
        // Single excitation
        bool is_alpha = !exc.occ_alpha.empty();
        int i = is_alpha ? exc.occ_alpha[0] : exc.occ_beta[0];
        int a = is_alpha ? exc.virt_alpha[0] : exc.virt_beta[0];
        
        return single_excitation_element(bra, i, a, is_alpha, ints);
    }
    else if (exc.level == 2) {
        // Double excitation
        // Need to determine spin cases: αα, ββ, or αβ
        
        if (exc.occ_alpha.size() == 2) {
            // Double α excitation: i_α j_α → a_α b_α
            return double_excitation_element(bra, 
                exc.occ_alpha[0], exc.occ_alpha[1],
                exc.virt_alpha[0], exc.virt_alpha[1],
                true, true, ints);
        }
        else if (exc.occ_beta.size() == 2) {
            // Double β excitation
            return double_excitation_element(bra,
                exc.occ_beta[0], exc.occ_beta[1],
                exc.virt_beta[0], exc.virt_beta[1],
                false, false, ints);
        }
        else {
            // Mixed αβ excitation
            return double_excitation_element(bra,
                exc.occ_alpha[0], exc.occ_beta[0],
                exc.virt_alpha[0], exc.virt_beta[0],
                true, false, ints);
        }
    }
    else {
        // Differ by more than 2 electrons: matrix element = 0
        // This is Slater-Condon rule #4
        return 0.0;
    }
}

// Diagonal element ⟨Φ|H|Φ⟩
// REFERENCE: Szabo & Ostlund (1996), Eq. (A.5)
double diagonal_element(const Determinant& det, const CIIntegrals& ints) {
    double energy = 0.0;
    
    auto occ_a = det.alpha_occupations();
    auto occ_b = det.beta_occupations();
    
    // DEBUG: First call only
    static bool first_call = true;
    bool do_debug = first_call && occ_a.size() == 2 && occ_b.size() == 1;
    if (do_debug) {
        first_call = false;
        std::cout << "\n=== DEBUG: diagonal_element() TRACE ===\n";
        std::cout << "Alpha occ: ";
        for (int i : occ_a) std::cout << i << " ";
        std::cout << "\nBeta occ: ";
        for (int i : occ_b) std::cout << i << " ";
        std::cout << "\n";
    }
    
    if (ints.use_fock) {
        // Fock-based formulation: just sum orbital energies (diagonal of Fock matrix)
        // The Fock matrix already includes mean-field two-electron effects
        // E = Σ_i F_ii (no additional ERI terms needed)
        for (int i : occ_a) {
            energy += ints.h_alpha(i, i);
        }
        for (int i : occ_b) {
            energy += ints.h_beta(i, i);
        }
    }
    else {
        // Bare Hamiltonian formulation (original)
        // E = Σ_i h_ii + (1/2) Σ_ij <ij||ij>
        
        // One-electron contribution: Σ_i h_ii
        if (do_debug) {
            std::cout << "\n--- One-electron contributions ---\n";
        }
        for (int i : occ_a) {
            if (do_debug) {
                std::cout << "h_alpha(" << i << "," << i << ") = " << ints.h_alpha(i,i) << " Ha\n";
            }
            energy += ints.h_alpha(i, i);
        }
        for (int i : occ_b) {
            if (do_debug) {
                std::cout << "h_beta(" << i << "," << i << ") = " << ints.h_beta(i,i) << " Ha\n";
            }
            energy += ints.h_beta(i, i);
        }
        
        // Two-electron contribution (αα): Σ_i>j <ij||ij>^α
        // NOTE: eri_aaaa already antisymmetrized, so use unique pairs only!
        if (do_debug) {
            std::cout << "\n--- Two-electron (αα) contributions ---\n";
        }
        for (size_t idx_i = 0; idx_i < occ_a.size(); idx_i++) {
            for (size_t idx_j = idx_i + 1; idx_j < occ_a.size(); idx_j++) {
                int i = occ_a[idx_i];
                int j = occ_a[idx_j];
                // <ij||ij> already antisymmetrized
                double contrib = ints.eri_aaaa(i,j,i,j);
                if (do_debug) {
                    std::cout << "eri_aaaa(" << i << "," << j << "," << i << "," << j << ") = " << contrib << " Ha\n";
                }
                energy += contrib;
            }
        }
        
        // Two-electron contribution (ββ): Σ_i>j <ij||ij>^β
        for (size_t idx_i = 0; idx_i < occ_b.size(); idx_i++) {
            for (size_t idx_j = idx_i + 1; idx_j < occ_b.size(); idx_j++) {
                int i = occ_b[idx_i];
                int j = occ_b[idx_j];
                // <ij||ij> already antisymmetrized
                energy += ints.eri_bbbb(i,j,i,j);  // No 1/2 for unique pairs
            }
        }
        
        // Two-electron contribution (αβ): Σ_i∈α Σ_J∈β <iJ|iJ>
        // Mapping to chemist ERI (pq|rs): <iJ|iJ>_phys = (ii|JJ)_chem
        if (do_debug) {
            std::cout << "\n--- Two-electron (αβ) contributions ---\n";
        }
        for (int i : occ_a) {
            for (int j : occ_b) {
                double contrib = ints.eri_aabb(i, i, j, j);
                if (do_debug) {
                    std::cout << "eri_aabb(" << i << "," << i << "," << j << "," << j << ") = " << contrib << " Ha\n";
                }
                energy += contrib;
            }
        }
    }
    
    if (do_debug) {
        std::cout << "\n--- TOTAL diagonal element ---\n";
        std::cout << "E(det) = " << energy << " Ha\n";
        std::cout << "Expected HF: -7.3155 Ha\n";
        std::cout << "Difference: " << (energy + 7.3155)*1000 << " mHa\n";
    }
    
    return energy;
}

// Single excitation element ⟨Φ|H|Φ_i^a⟩
// REFERENCE: Szabo & Ostlund (1996), Eq. (A.6)
// H = h_ia + Σ_j <ij||aj> (bare H) OR F_ia (Fock-based)
double single_excitation_element(const Determinant& bra,
                                  int i, int a, bool spin_alpha,
                                  const CIIntegrals& ints) {
    
    // Phase factor from orbital permutations
    int phase_sign = bra.phase(i, a, spin_alpha);
    
    double elem = 0.0;
    
    if (ints.use_fock) {
        // Fock-based: just use off-diagonal Fock matrix element
        // F_ia already includes mean-field coupling with occupied orbitals
        if (spin_alpha) {
            elem = ints.h_alpha(i, a);  // F_ia
        } else {
            elem = ints.h_beta(i, a);   // F_ia
        }
    }
    else {
        // Bare Hamiltonian formulation (original)
        if (spin_alpha) {
            // α excitation: i_α → a_α
            
            // One-electron part: h_ia
            elem = ints.h_alpha(i, a);
            
            // Two-electron part: Σ_j <ij||aj>^α
            auto occ_a = bra.alpha_occupations();
            for (int j : occ_a) {
                if (j != i) {  // Skip i (it's being excited)
                    elem += ints.eri_aaaa(i,j,a,j);
                }
            }
            
            // αβ coupling: Σ_J <iJ|aJ>
            // Mapping to chemist ERI: <iJ|aJ>_phys = (ia|JJ)_chem
            auto occ_b = bra.beta_occupations();
            for (int j : occ_b) {
                elem += ints.eri_aabb(i, a, j, j);
            }
        }
        else {
            // β excitation: i_β → a_β
            
            elem = ints.h_beta(i, a);
            
            // ββ coupling
            auto occ_b = bra.beta_occupations();
            for (int j : occ_b) {
                if (j != i) {
                    elem += ints.eri_bbbb(i,j,a,j);
                }
            }
            
            // βα coupling: Σ_I <Ii|Ia>
            // Mapping to chemist ERI: <Ii|Ia>_phys = (II|ia)_chem
            auto occ_a = bra.alpha_occupations();
            for (int j : occ_a) {
                elem += ints.eri_aabb(j, j, i, a);
            }
        }
    }
    
    return phase_sign * elem;
}

// Double excitation element ⟨Φ|H|Φ_ij^ab⟩
// REFERENCE: Szabo & Ostlund (1996), Eq. (A.7)
// H = <ij||ab>
double double_excitation_element(const Determinant& bra,
                                  int i, int j, int a, int b,
                                  bool spin1, bool spin2,
                                  const CIIntegrals& ints) {
    
    double elem = 0.0;
    
    // Compute phase (product of two single excitation phases)
    int phase_i = bra.phase(i, a, spin1);
    
    // For second excitation, compute phase on intermediate determinant
    Determinant interm = bra.single_excite(i, a, spin1);
    int phase_j = interm.phase(j, b, spin2);
    
    int total_phase = phase_i * phase_j;
    
    if (spin1 && spin2) {
        // αα double excitation
        // <ij||ab>^α = <ij|ab> - <ij|ba>
        elem = ints.eri_aaaa(i,j,a,b);
    }
    else if (!spin1 && !spin2) {
        // ββ double excitation
        elem = ints.eri_bbbb(i,j,a,b);
    }
    else {
        // αβ mixed excitation (no antisymmetrization)
        // <iJ|aB> with mapping to chemist ERI: (ia|JB)
        elem = ints.eri_aabb(i, a, j, b);
    }
    
    return total_phase * elem;
}

// Build full Hamiltonian matrix
Eigen::MatrixXd build_hamiltonian(const std::vector<Determinant>& dets,
                                  const CIIntegrals& ints) {
    int n = dets.size();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
    
    // DEBUG: Compute HF determinant energy for comparison
    auto hf_occ_a = dets[0].alpha_occupations();
    auto hf_occ_b = dets[0].beta_occupations();
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "\n=== DEBUG: First determinant ===\n";
    std::cout << "Alpha occupations: ";
    for (int i : hf_occ_a) std::cout << i << " ";
    std::cout << "\nBeta occupations: ";
    for (int i : hf_occ_b) std::cout << i << " ";
    std::cout << "\n";
    
    // Build H_ij = ⟨Φ_i|H|Φ_j⟩
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {  // Only upper triangle (Hermitian)
            H(i,j) = hamiltonian_element(dets[i], dets[j], ints);
            if (i != j) {
                H(j,i) = H(i,j);  // Hermitian symmetry
            }
        }
        
        // Progress for large matrices
        if (n > 1000 && i % 100 == 0) {
            std::cout << "Building H: " << i << "/" << n << "\n";
        }
    }
    
    // DEBUG: Print H(0,0)
    std::cout << "\n=== DEBUG: CI Hamiltonian Matrix ===\n";
    std::cout << "H(0,0) = " << H(0,0) << " Ha\n";
    std::cout << "This should match HF determinant energy from integrals\n";
    
    return H;
}

// Compute only diagonal of H (for Davidson preconditioner)
Eigen::VectorXd hamiltonian_diagonal(const std::vector<Determinant>& dets,
                                     const CIIntegrals& ints) {
    int n = dets.size();
    Eigen::VectorXd diag(n);
    
    for (int i = 0; i < n; i++) {
        diag(i) = diagonal_element(dets[i], ints);
    }
    
    return diag;
}

// Sigma-vector product: σ = H * c
// CRITICAL for Davidson - avoid storing full H!
Eigen::VectorXd sigma_vector(const std::vector<Determinant>& dets,
                              const Eigen::VectorXd& c,
                              const CIIntegrals& ints) {
    int n = dets.size();
    Eigen::VectorXd sigma = Eigen::VectorXd::Zero(n);
    
    // σ_i = Σ_j H_ij c_j
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sigma(i) += hamiltonian_element(dets[i], dets[j], ints) * c(j);
        }
    }
    
    return sigma;
}

} // namespace ci
} // namespace mshqc
