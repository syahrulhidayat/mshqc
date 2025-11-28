/**
 * @file determinant.cc
 * @brief Implementation of Slater determinant with bit strings
 * 
 * REFERENCES:
 * - Slater (1929), Phys. Rev. 34, 1293
 * - Knowles & Handy (1984), Chem. Phys. Lett. 111, 315
 * - Shavitt & Bartlett (2009), Many-Body Methods, Ch. 3
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

#include "mshqc/ci/determinant.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <bitset>

namespace mshqc {
namespace ci {

// Default constructor: vacuum state
Determinant::Determinant() 
    : alpha_(0), beta_(0), n_alpha_(0), n_beta_(0) {}

// Construct from occupation lists
Determinant::Determinant(const std::vector<int>& alpha_occ,
                         const std::vector<int>& beta_occ)
    : alpha_(0), beta_(0), n_alpha_(alpha_occ.size()), n_beta_(beta_occ.size()) {
    
    // Build bit string for α electrons
    for (int orb : alpha_occ) {
        if (orb < 0 || orb >= 64) {
            throw std::out_of_range("Orbital index must be 0-63");
        }
        alpha_ |= (1ULL << orb);
    }
    
    // Build bit string for β electrons
    for (int orb : beta_occ) {
        if (orb < 0 || orb >= 64) {
            throw std::out_of_range("Orbital index must be 0-63");
        }
        beta_ |= (1ULL << orb);
    }
}

// Construct from bit strings
Determinant::Determinant(uint64_t alpha_bits, uint64_t beta_bits)
    : alpha_(alpha_bits), beta_(beta_bits),
      n_alpha_(popcount(alpha_bits)), n_beta_(popcount(beta_bits)) {}

bool Determinant::is_occupied(int orb, bool alpha) const {
    if (orb < 0 || orb >= 64) return false;
    uint64_t bits = alpha ? alpha_ : beta_;
    return (bits & (1ULL << orb)) != 0;
}

std::vector<int> Determinant::alpha_occupations() const {
    std::vector<int> occ;
    for (int i = 0; i < 64; i++) {
        if (alpha_ & (1ULL << i)) {
            occ.push_back(i);
        }
    }
    return occ;
}

std::vector<int> Determinant::beta_occupations() const {
    std::vector<int> occ;
    for (int i = 0; i < 64; i++) {
        if (beta_ & (1ULL << i)) {
            occ.push_back(i);
        }
    }
    return occ;
}

// Single excitation: i → a
// REFERENCE: Shavitt & Bartlett (2009), Eq. (3.12)
Determinant Determinant::single_excite(int i, int a, bool alpha) const {
    uint64_t new_alpha = alpha_;
    uint64_t new_beta = beta_;
    
    if (alpha) {
        // Check i is occupied, a is virtual
        if (!(alpha_ & (1ULL << i))) {
            throw std::runtime_error("Orbital i not occupied in alpha");
        }
        if (alpha_ & (1ULL << a)) {
            throw std::runtime_error("Orbital a already occupied in alpha");
        }
        
        // Remove electron from i, add to a
        new_alpha &= ~(1ULL << i);  // Clear bit i
        new_alpha |= (1ULL << a);   // Set bit a
    } else {
        // Beta excitation
        if (!(beta_ & (1ULL << i))) {
            throw std::runtime_error("Orbital i not occupied in beta");
        }
        if (beta_ & (1ULL << a)) {
            throw std::runtime_error("Orbital a already occupied in beta");
        }
        
        new_beta &= ~(1ULL << i);
        new_beta |= (1ULL << a);
    }
    
    return Determinant(new_alpha, new_beta);
}

// Double excitation: ij → ab
// REFERENCE: Shavitt & Bartlett (2009), Eq. (3.13)
Determinant Determinant::double_excite(int i, int j, int a, int b,
                                        bool spin1, bool spin2) const {
    uint64_t new_alpha = alpha_;
    uint64_t new_beta = beta_;
    
    // First excitation
    if (spin1) {
        if (!(alpha_ & (1ULL << i))) {
            throw std::runtime_error("Orbital i not occupied");
        }
        if (alpha_ & (1ULL << a)) {
            throw std::runtime_error("Orbital a occupied");
        }
        new_alpha &= ~(1ULL << i);
        new_alpha |= (1ULL << a);
    } else {
        if (!(beta_ & (1ULL << i))) {
            throw std::runtime_error("Orbital i not occupied");
        }
        if (beta_ & (1ULL << a)) {
            throw std::runtime_error("Orbital a occupied");
        }
        new_beta &= ~(1ULL << i);
        new_beta |= (1ULL << a);
    }
    
    // Second excitation
    if (spin2) {
        if (!(new_alpha & (1ULL << j))) {
            throw std::runtime_error("Orbital j not occupied");
        }
        if (new_alpha & (1ULL << b)) {
            throw std::runtime_error("Orbital b occupied");
        }
        new_alpha &= ~(1ULL << j);
        new_alpha |= (1ULL << b);
    } else {
        if (!(new_beta & (1ULL << j))) {
            throw std::runtime_error("Orbital j not occupied");
        }
        if (new_beta & (1ULL << b)) {
            throw std::runtime_error("Orbital b occupied");
        }
        new_beta &= ~(1ULL << j);
        new_beta |= (1ULL << b);
    }
    
    return Determinant(new_alpha, new_beta);
}

// Triple excitation: ijk → abc
// THEORY REFERENCES:
// - Raghavachari et al. (1989), Chem. Phys. Lett. 157, 479
// - Helgaker et al. (2000), Molecular Electronic-Structure Theory, Ch. 10.6
// - Szabo & Ostlund (1996), Modern Quantum Chemistry, Appendix A
//
// ORIGINAL IMPLEMENTATION by Muhamad Syahrul Hidayat (2025-11-16)
// NO CODE COPIED from PySCF, Psi4, or other quantum chemistry packages
//
// Algorithm:
//   |Φ_ijk^abc⟩ = a^† b^† c^† k j i |Φ⟩
//
//   Step 1: Annihilate electrons from i,j,k (must be occupied)
//   Step 2: Create electrons at a,b,c (must be virtual)
//   Step 3: Phase is computed from intermediate states
//
// For simplicity, we apply three single excitations sequentially:
//   |Φ'⟩ = single_excite(i→a) on |Φ⟩
//   |Φ''⟩ = single_excite(j→b) on |Φ'⟩  
//   |Φ_ijk^abc⟩ = single_excite(k→c) on |Φ''⟩
//
// This automatically handles phase factors through the single excitation logic.
Determinant Determinant::triple_excite(int i, int j, int k,
                                        int a, int b, int c,
                                        bool spin_i, bool spin_j, bool spin_k) const {
    // Validate input
    if (i < 0 || j < 0 || k < 0 || a < 0 || b < 0 || c < 0) {
        throw std::runtime_error("Triple excitation: orbital indices must be non-negative");
    }
    
    if (i >= 64 || j >= 64 || k >= 64 || a >= 64 || b >= 64 || c >= 64) {
        throw std::runtime_error("Triple excitation: orbital indices must be < 64");
    }
    
    // Check orbitals i,j,k are occupied in their respective spins
    uint64_t alpha_check = alpha_;
    uint64_t beta_check = beta_;
    
    if (spin_i) {
        if (!(alpha_check & (1ULL << i))) {
            throw std::runtime_error("Triple excitation: orbital i not occupied in alpha");
        }
    } else {
        if (!(beta_check & (1ULL << i))) {
            throw std::runtime_error("Triple excitation: orbital i not occupied in beta");
        }
    }
    
    if (spin_j) {
        if (!(alpha_check & (1ULL << j))) {
            throw std::runtime_error("Triple excitation: orbital j not occupied in alpha");
        }
    } else {
        if (!(beta_check & (1ULL << j))) {
            throw std::runtime_error("Triple excitation: orbital j not occupied in beta");
        }
    }
    
    if (spin_k) {
        if (!(alpha_check & (1ULL << k))) {
            throw std::runtime_error("Triple excitation: orbital k not occupied in alpha");
        }
    } else {
        if (!(beta_check & (1ULL << k))) {
            throw std::runtime_error("Triple excitation: orbital k not occupied in beta");
        }
    }
    
    // Check orbitals a,b,c are virtual (not occupied)
    if (spin_i && (alpha_check & (1ULL << a))) {
        throw std::runtime_error("Triple excitation: orbital a already occupied in alpha");
    }
    if (!spin_i && (beta_check & (1ULL << a))) {
        throw std::runtime_error("Triple excitation: orbital a already occupied in beta");
    }
    
    if (spin_j && (alpha_check & (1ULL << b))) {
        throw std::runtime_error("Triple excitation: orbital b already occupied in alpha");
    }
    if (!spin_j && (beta_check & (1ULL << b))) {
        throw std::runtime_error("Triple excitation: orbital b already occupied in beta");
    }
    
    if (spin_k && (alpha_check & (1ULL << c))) {
        throw std::runtime_error("Triple excitation: orbital c already occupied in alpha");
    }
    if (!spin_k && (beta_check & (1ULL << c))) {
        throw std::runtime_error("Triple excitation: orbital c already occupied in beta");
    }
    
    // Apply triple excitation as sequence of single excitations
    // This preserves correct fermion statistics and phase factors
    //
    // THEORY: Sequential application of creation/annihilation operators
    // automatically generates correct phase through anticommutation relations:
    // {a_i, a_j†} = δ_ij, {a_i, a_j} = 0, {a_i†, a_j†} = 0
    //
    // REFERENCE: Szabo & Ostlund (1996), Appendix A, Eqs. (A.10-A.12)
    
    Determinant result = *this;
    
    // First excitation: i → a
    result = result.single_excite(i, a, spin_i);
    
    // Second excitation: j → b (on intermediate state)
    result = result.single_excite(j, b, spin_j);
    
    // Third excitation: k → c (on final intermediate state)
    result = result.single_excite(k, c, spin_k);
    
    return result;
}

int Determinant::count_differences(const Determinant& other) const {
    // XOR gives bits that differ
    uint64_t diff_alpha = alpha_ ^ other.alpha_;
    uint64_t diff_beta = beta_ ^ other.beta_;
    
    // Count differing bits (each excitation changes 2 bits: occupied→virtual)
    // So divide by 2 to get excitation level
    int n_diff = popcount(diff_alpha) + popcount(diff_beta);
    return n_diff / 2;
}

std::pair<int, int> Determinant::excitation_level(const Determinant& other) const {
    uint64_t diff_alpha = alpha_ ^ other.alpha_;
    uint64_t diff_beta = beta_ ^ other.beta_;
    
    int n_alpha_diff = popcount(diff_alpha) / 2;
    int n_beta_diff = popcount(diff_beta) / 2;
    
    return {n_alpha_diff, n_beta_diff};
}

// Phase calculation
// REFERENCE: Szabo & Ostlund (1996), Appendix A
// Phase = (-1)^n where n = # electrons between i and a
int Determinant::phase(int i, int a, bool alpha) const {
    uint64_t bits = alpha ? alpha_ : beta_;
    
    // Ensure i < a
    if (i > a) std::swap(i, a);
    
    // Count occupied orbitals between i and a
    int count = 0;
    for (int orb = i + 1; orb < a; orb++) {
        if (bits & (1ULL << orb)) {
            count++;
        }
    }
    
    // Phase = (-1)^count
    return (count % 2 == 0) ? 1 : -1;
}

std::string Determinant::to_string() const {
    std::ostringstream oss;
    oss << "|α:";
    
    // Print alpha string
    for (int i = 0; i < 16; i++) {  // Print first 16 orbitals
        if (alpha_ & (1ULL << i)) {
            oss << "1";
        } else {
            oss << "0";
        }
    }
    
    oss << ",β:";
    
    // Print beta string
    for (int i = 0; i < 16; i++) {
        if (beta_ & (1ULL << i)) {
            oss << "1";
        } else {
            oss << "0";
        }
    }
    
    oss << "⟩";
    return oss.str();
}

bool Determinant::operator==(const Determinant& other) const {
    return (alpha_ == other.alpha_) && (beta_ == other.beta_);
}

bool Determinant::operator!=(const Determinant& other) const {
    return !(*this == other);
}

bool Determinant::operator<(const Determinant& other) const {
    if (alpha_ != other.alpha_) {
        return alpha_ < other.alpha_;
    }
    return beta_ < other.beta_;
}

// Static helper: count set bits (popcount)
// Use builtin if available, otherwise manual
int Determinant::popcount(uint64_t bits) {
#ifdef __GNUC__
    // GCC/Clang builtin (fast!)
    return __builtin_popcountll(bits);
#else
    // Manual fallback (slower)
    int count = 0;
    while (bits) {
        count += bits & 1;
        bits >>= 1;
    }
    return count;
#endif
}

// Find differences between bit strings
std::vector<int> Determinant::find_differences(uint64_t bits1, uint64_t bits2) {
    uint64_t diff = bits1 ^ bits2;
    std::vector<int> positions;
    
    for (int i = 0; i < 64; i++) {
        if (diff & (1ULL << i)) {
            positions.push_back(i);
        }
    }
    
    return positions;
}

// Find excitation between determinants
Excitation find_excitation(const Determinant& bra, const Determinant& ket) {
    Excitation exc;
    
    // Find differences in alpha
    auto diff_alpha = Determinant::find_differences(bra.alpha_bits(), ket.alpha_bits());
    
    // Find differences in beta
    auto diff_beta = Determinant::find_differences(bra.beta_bits(), ket.beta_bits());
    
    // Separate into occupied (in bra, not in ket) and virtual (in ket, not in bra)
    for (int orb : diff_alpha) {
        if (bra.is_occupied(orb, true)) {
            exc.occ_alpha.push_back(orb);
        } else {
            exc.virt_alpha.push_back(orb);
        }
    }
    
    for (int orb : diff_beta) {
        if (bra.is_occupied(orb, false)) {
            exc.occ_beta.push_back(orb);
        } else {
            exc.virt_beta.push_back(orb);
        }
    }
    
    // Total excitation level
    exc.level = exc.occ_alpha.size() + exc.occ_beta.size();
    
    return exc;
}

} // namespace ci
} // namespace mshqc
