#ifndef MSHQC_CI_DETERMINANT_H
#define MSHQC_CI_DETERMINANT_H

#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <functional>

/**
 * @file determinant.h
 * @brief Slater determinant representation with bit strings
 * 
 * REFERENCES:
 * - Slater (1929), Phys. Rev. 34, 1293 [original determinant theory]
 * - Shavitt & Bartlett (2009), Many-Body Methods, Ch. 3 [modern treatment]
 * - Knowles & Handy (1984), Chem. Phys. Lett. 111, 315 [FCI algorithms]
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

namespace mshqc {
namespace ci {

/**
 * Slater determinant using bit representation
 * 
 * CONCEPT: Each orbital occupation stored as bit
 * Example: |α: 110010⟩ |β: 100010⟩
 *   → α electrons in orbitals 1,2,5
 *   → β electrons in orbitals 1,5
 * 
 * Bit operations are FAST: O(1) for comparisons!
 */
class Determinant {
public:
    /**
     * Default constructor (vacuum state)
     */
    Determinant();
    
    /**
     * Construct from occupation lists
     * @param alpha_occ List of α-occupied orbital indices
     * @param beta_occ List of β-occupied orbital indices
     * 
     * Example: alpha_occ = {0, 1, 2}, beta_occ = {0, 1}
     *   → |↑↑↑↓↓⟩ in first 3 orbitals
     */
    Determinant(const std::vector<int>& alpha_occ,
                const std::vector<int>& beta_occ);
    
    /**
     * Construct from bit strings directly (advanced)
     * @param alpha_bits Bit string for α electrons
     * @param beta_bits Bit string for β electrons
     */
    explicit Determinant(uint64_t alpha_bits, uint64_t beta_bits);
    
    // Accessors
    uint64_t alpha_bits() const { return alpha_; }
    uint64_t beta_bits() const { return beta_; }
    int n_alpha() const { return n_alpha_; }
    int n_beta() const { return n_beta_; }
    
    /**
     * Check if orbital is occupied
     * @param orb Orbital index
     * @param alpha true for α, false for β
     */
    bool is_occupied(int orb, bool alpha) const;
    
    /**
     * Get list of occupied orbitals
     */
    std::vector<int> alpha_occupations() const;
    std::vector<int> beta_occupations() const;
    
    /**
     * Create single excitation: i → a
     * 
     * REFERENCE: Shavitt & Bartlett (2009), Eq. (3.12)
     * |Φ_i^a⟩ = a^† i |Φ⟩
     * 
     * @param i Occupied orbital (to remove electron)
     * @param a Virtual orbital (to add electron)
     * @param alpha true for α-spin, false for β-spin
     * @return New determinant with excitation applied
     */
    Determinant single_excite(int i, int a, bool alpha) const;
    
    /**
     * Create double excitation: ij → ab
     * 
     * REFERENCE: Shavitt & Bartlett (2009), Eq. (3.13)
     * |Φ_ij^ab⟩ = a^† b^† j i |Φ⟩
     * 
     * @param i First occupied orbital
     * @param j Second occupied orbital
     * @param a First virtual orbital
     * @param b Second virtual orbital
     * @param spin1 Spin of first excitation
     * @param spin2 Spin of second excitation
     * @return New determinant
     */
    Determinant double_excite(int i, int j, int a, int b,
                               bool spin1, bool spin2) const;
    
    /**
     * Create triple excitation: ijk → abc
     * 
     * THEORY REFERENCES:
     * - Raghavachari, K., Trucks, G. W., Pople, J. A., & Head-Gordon, M. (1989).
     *   "A fifth-order perturbation comparison of electron correlation theories"
     *   Chemical Physics Letters, 157(6), 479-483.
     *   DOI: 10.1016/S0009-2614(89)87395-6
     *   [Triple excitation operators in CI theory]
     * 
     * - Helgaker, T., Jørgensen, P., & Olsen, J. (2000).
     *   "Molecular Electronic-Structure Theory", Ch. 10.6
     *   Wiley. ISBN: 978-0-471-96755-2
     *   [Phase factors for multiple excitations]
     * 
     * - Szabo, A., & Ostlund, N. S. (1996).
     *   "Modern Quantum Chemistry", Appendix A
     *   Dover. ISBN: 978-0486691862
     *   [Creation/annihilation operator algebra]
     * 
     * FORMULA:
     *   |Φ_ijk^abc⟩ = a^† b^† c^† k j i |Φ⟩
     * 
     * Phase factor from fermion anticommutation:
     *   Phase = (-1)^{n_perm}
     *   where n_perm = number of electron permutations
     * 
     * ALGORITHM:
     *   1. Check all orbitals i,j,k are occupied in correct spin
     *   2. Check all orbitals a,b,c are virtual (not occupied)
     *   3. Remove electrons from i,j,k (annihilation: k,j,i order)
     *   4. Add electrons to a,b,c (creation: a,b,c order)
     *   5. Compute phase from orbital ordering and intermediate states
     * 
     * @param i First occupied orbital
     * @param j Second occupied orbital  
     * @param k Third occupied orbital
     * @param a First virtual orbital
     * @param b Second virtual orbital
     * @param c Third virtual orbital
     * @param spin_i Spin of electron i (true=α, false=β)
     * @param spin_j Spin of electron j (true=α, false=β)
     * @param spin_k Spin of electron k (true=α, false=β)
     * @return New determinant with triple excitation
     * @throws std::runtime_error if orbitals invalid
     * 
     * @note ORIGINAL IMPLEMENTATION - Author: Muhamad Syahrul Hidayat
     * @note NO CODE COPIED from PySCF, Psi4, or other packages
     * @date 2025-11-16
     */
    Determinant triple_excite(int i, int j, int k,
                              int a, int b, int c,
                              bool spin_i, bool spin_j, bool spin_k) const;
    
    /**
     * Count number of different orbitals between determinants
     * 
     * Used for Slater-Condon rules:
     * - diff = 0: same determinant
     * - diff = 1: single excitation
     * - diff = 2: double excitation
     * - diff > 2: matrix element = 0
     */
    int count_differences(const Determinant& other) const;
    
    /**
     * Get excitation level between determinants
     * @return {n_diff_alpha, n_diff_beta}
     */
    std::pair<int, int> excitation_level(const Determinant& other) const;
    
    /**
     * Compute phase factor for excitation
     * 
     * REFERENCE: Szabo & Ostlund (1996), Appendix A
     * Phase = (-1)^n where n = # of orbital permutations
     * 
     * Example: |...↑...⟩ → |...↑...⟩
     * Count electrons between i and a positions
     */
    int phase(int i, int a, bool alpha) const;
    
    /**
     * String representation for debugging
     * Example: "|αααβ:1100,ββ:0011⟩"
     */
    std::string to_string() const;
    
    /**
     * Comparison operators (for sorting)
     */
    bool operator==(const Determinant& other) const;
    bool operator!=(const Determinant& other) const;
    bool operator<(const Determinant& other) const;
    
    /**
     * Count set bits (popcount)
     * Fast bit manipulation: counts 1's in bit string
     */
    static int popcount(uint64_t bits);
    
    /**
     * Find position of differences between bit strings
     * Returns orbital indices where bits differ
     */
    static std::vector<int> find_differences(uint64_t bits1, uint64_t bits2);
    
private:
    uint64_t alpha_;  // α-spin bit string
    uint64_t beta_;   // β-spin bit string
    int n_alpha_;     // # of α electrons
    int n_beta_;      // # of β electrons
};

/**
 * Excitation descriptor
 * Stores information about excitation between determinants
 */
struct Excitation {
    std::vector<int> occ_alpha;   // Occupied α orbitals
    std::vector<int> occ_beta;    // Occupied β orbitals
    std::vector<int> virt_alpha;  // Virtual α orbitals
    std::vector<int> virt_beta;   // Virtual β orbitals
    int level;                    // Excitation level (0,1,2,...)
};

/**
 * Find excitation between two determinants
 * 
 * USAGE:
 * auto exc = find_excitation(det1, det2);
 * if (exc.level == 1) {
 *     // Single excitation: i → a
 *     int i = exc.occ_alpha[0];
 *     int a = exc.virt_alpha[0];
 * }
 */
Excitation find_excitation(const Determinant& bra, const Determinant& ket);

} // namespace ci
} // namespace mshqc

// ============================================================================
// Hash Function for Determinant (for unordered_map/unordered_set)
// ============================================================================

namespace std {
    /**
     * @brief Hash functor for Determinant to enable O(1) lookup
     * 
     * Uses bit-string representation for fast hashing.
     * Combines alpha and beta bits using XOR and bit rotation.
     * 
     * PERFORMANCE: O(1) hash computation
     * COLLISION: Very low (64-bit hash space)
     * 
     * Usage:
     * ```cpp
     * std::unordered_map<Determinant, int, std::hash<Determinant>> det_to_idx;
     * ```
     * 
     * @author Muhamad Syahrul Hidayat
     * @date 2025-11-14
     */
    template<>
    struct hash<mshqc::ci::Determinant> {
        std::size_t operator()(const mshqc::ci::Determinant& det) const noexcept {
            // Combine alpha and beta bit strings
            // Use standard hash_combine pattern
            std::size_t h1 = std::hash<uint64_t>{}(det.alpha_bits());
            std::size_t h2 = std::hash<uint64_t>{}(det.beta_bits());
            
            // Boost-style hash_combine:
            // h = h1 XOR (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2))
            return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        }
    };
}

#endif // MSHQC_CI_DETERMINANT_H
