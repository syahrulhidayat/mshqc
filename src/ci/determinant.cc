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
/**
 * @file determinant.cc
 * @brief Implementation of Slater determinant with Arbitrary Size Bitset
 * * FIX: Ensures alpha_occupations returns SORTED lists for set_difference.
 * * FIX: Implements robust find_excitation to prevent Segfaults downstream.
 */

#include "mshqc/ci/determinant.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iterator> // Penting untuk std::back_inserter

namespace mshqc {
namespace ci {

// ==========================================================
// Helper Private Methods (Bit Manipulation)
// ==========================================================

void Determinant::set_bit(std::vector<uint64_t>& bits, int orb) {
    size_t required_idx = orb / 64;
    if (bits.size() <= required_idx) {
        // Resize dan isi dengan 0
        bits.resize(required_idx + 1, 0);
    }
    bits[required_idx] |= (1ULL << (orb % 64));
}

void Determinant::clear_bit(std::vector<uint64_t>& bits, int orb) {
    size_t idx = orb / 64;
    if (idx < bits.size()) {
        bits[idx] &= ~(1ULL << (orb % 64));
    }
}

bool Determinant::check_bit(const std::vector<uint64_t>& bits, int orb) const {
    size_t idx = orb / 64;
    if (idx >= bits.size()) return false;
    return (bits[idx] & (1ULL << (orb % 64))) != 0;
}

// ==========================================================
// Constructors
// ==========================================================

Determinant::Determinant(const std::vector<int>& alpha_occ,
                         const std::vector<int>& beta_occ)
    : n_alpha_(alpha_occ.size()), n_beta_(beta_occ.size()) 
{
    alpha_bits_.push_back(0);
    beta_bits_.push_back(0);
    
    for (int orb : alpha_occ) set_bit(alpha_bits_, orb);
    for (int orb : beta_occ) set_bit(beta_bits_, orb);
}

Determinant::Determinant(const std::vector<uint64_t>& alpha, 
                         const std::vector<uint64_t>& beta)
    : alpha_bits_(alpha), beta_bits_(beta) 
{
    n_alpha_ = 0;
    for (auto w : alpha) n_alpha_ += popcount(w);
    n_beta_ = 0;
    for (auto w : beta) n_beta_ += popcount(w);
}

// ==========================================================
// Orbital Queries
// ==========================================================

bool Determinant::is_occupied(int orb, bool alpha) const {
    return check_bit(alpha ? alpha_bits_ : beta_bits_, orb);
}

// [CRITICAL] Fungsi ini WAJIB mengembalikan list yang TERURUT (Sorted)
// Implementasi loop ini menjamin urutan dari kecil ke besar.
std::vector<int> Determinant::alpha_occupations() const {
    std::vector<int> occ;
    occ.reserve(n_alpha_); 
    for (size_t i = 0; i < alpha_bits_.size(); ++i) {
        uint64_t word = alpha_bits_[i];
        if (word == 0) continue;
        for (int b = 0; b < 64; ++b) {
            if (word & (1ULL << b)) {
                occ.push_back(static_cast<int>(i * 64 + b));
            }
        }
    }
    return occ;
}

std::vector<int> Determinant::beta_occupations() const {
    std::vector<int> occ;
    occ.reserve(n_beta_);
    for (size_t i = 0; i < beta_bits_.size(); ++i) {
        uint64_t word = beta_bits_[i];
        if (word == 0) continue;
        for (int b = 0; b < 64; ++b) {
            if (word & (1ULL << b)) {
                occ.push_back(static_cast<int>(i * 64 + b));
            }
        }
    }
    return occ;
}

// ==========================================================
// Excitations & Phase
// ==========================================================

Determinant Determinant::single_excite(int i, int a, bool alpha) const {
    Determinant det = *this;
    if (alpha) {
        det.clear_bit(det.alpha_bits_, i);
        det.set_bit(det.alpha_bits_, a);
    } else {
        det.clear_bit(det.beta_bits_, i);
        det.set_bit(det.beta_bits_, a);
    }
    return det;
}

Determinant Determinant::double_excite(int i, int j, int a, int b,
                                        bool spin1, bool spin2) const {
    Determinant det = *this;
    // First excitation
    if (spin1) { det.clear_bit(det.alpha_bits_, i); det.set_bit(det.alpha_bits_, a); }
    else       { det.clear_bit(det.beta_bits_, i);  det.set_bit(det.beta_bits_, a); }
    
    // Second excitation
    if (spin2) { det.clear_bit(det.alpha_bits_, j); det.set_bit(det.alpha_bits_, b); }
    else       { det.clear_bit(det.beta_bits_, j);  det.set_bit(det.beta_bits_, b); }
    
    return det;
}

int Determinant::phase(int i, int a, bool alpha) const {
    if (i > a) std::swap(i, a);
    const auto& bits = alpha ? alpha_bits_ : beta_bits_;
    
    int count = 0;
    // Simple loop check (aman & pasti benar)
    for (int k = i + 1; k < a; ++k) {
        if (check_bit(bits, k)) count++;
    }
    return (count % 2 == 0) ? 1 : -1;
}

// ==========================================================
// Utilities & Operators
// ==========================================================

int Determinant::count_differences(const Determinant& other) const {
    int diff = 0;
    size_t max_sz = std::max(alpha_bits_.size(), other.alpha_bits_.size());
    for (size_t i = 0; i < max_sz; ++i) {
        uint64_t w1 = (i < alpha_bits_.size()) ? alpha_bits_[i] : 0;
        uint64_t w2 = (i < other.alpha_bits_.size()) ? other.alpha_bits_[i] : 0;
        diff += popcount(w1 ^ w2);
    }
    max_sz = std::max(beta_bits_.size(), other.beta_bits_.size());
    for (size_t i = 0; i < max_sz; ++i) {
        uint64_t w1 = (i < beta_bits_.size()) ? beta_bits_[i] : 0;
        uint64_t w2 = (i < other.beta_bits_.size()) ? other.beta_bits_[i] : 0;
        diff += popcount(w1 ^ w2);
    }
    return diff / 2;
}

std::pair<int, int> Determinant::excitation_level(const Determinant& other) const {
    int da = 0;
    size_t max_sz = std::max(alpha_bits_.size(), other.alpha_bits_.size());
    for (size_t i = 0; i < max_sz; ++i) {
        uint64_t w1 = (i < alpha_bits_.size()) ? alpha_bits_[i] : 0;
        uint64_t w2 = (i < other.alpha_bits_.size()) ? other.alpha_bits_[i] : 0;
        da += popcount(w1 ^ w2);
    }
    int db = 0;
    max_sz = std::max(beta_bits_.size(), other.beta_bits_.size());
    for (size_t i = 0; i < max_sz; ++i) {
        uint64_t w1 = (i < beta_bits_.size()) ? beta_bits_[i] : 0;
        uint64_t w2 = (i < other.beta_bits_.size()) ? other.beta_bits_[i] : 0;
        db += popcount(w1 ^ w2);
    }
    return {da / 2, db / 2};
}

bool Determinant::operator==(const Determinant& other) const {
    return alpha_bits_ == other.alpha_bits_ && beta_bits_ == other.beta_bits_;
}
bool Determinant::operator!=(const Determinant& other) const { return !(*this == other); }
bool Determinant::operator<(const Determinant& other) const {
    if (alpha_bits_ != other.alpha_bits_) return alpha_bits_ < other.alpha_bits_;
    return beta_bits_ < other.beta_bits_;
}

int Determinant::popcount(uint64_t bits) {
#ifdef __GNUC__
    return __builtin_popcountll(bits);
#else
    int c = 0;
    for (; bits; c++) bits &= bits - 1;
    return c;
#endif
}

std::string Determinant::to_string() const {
    std::ostringstream oss;
    oss << "|A:";
    for(int i : alpha_occupations()) oss << i << " ";
    oss << " B:";
    for(int i : beta_occupations()) oss << i << " ";
    oss << ">";
    return oss.str();
}

// ==========================================================
// GLOBAL HELPER: find_excitation (Sumber Crash Utama)
// ==========================================================

Excitation find_excitation(const Determinant& bra, const Determinant& ket) {
    Excitation exc;
    
    // 1. Ambil list orbital (Pasti Sorted karena implementasi di atas)
    auto bra_a = bra.alpha_occupations();
    auto ket_a = ket.alpha_occupations();
    auto bra_b = bra.beta_occupations();
    auto ket_b = ket.beta_occupations();
    
    // 2. Gunakan std::set_difference untuk mencari perbedaan
    // set_difference butuh input terurut.
    
    // Cari orbital di BRA tapi tidak di KET (Holes / Occupied yang hilang)
    std::set_difference(bra_a.begin(), bra_a.end(), 
                        ket_a.begin(), ket_a.end(), 
                        std::back_inserter(exc.occ_alpha));
                        
    // Cari orbital di KET tapi tidak di BRA (Particles / Virtual yang terisi)
    std::set_difference(ket_a.begin(), ket_a.end(), 
                        bra_a.begin(), bra_a.end(), 
                        std::back_inserter(exc.virt_alpha));
    
    // Ulangi untuk Beta
    std::set_difference(bra_b.begin(), bra_b.end(), 
                        ket_b.begin(), ket_b.end(), 
                        std::back_inserter(exc.occ_beta));
                        
    std::set_difference(ket_b.begin(), ket_b.end(), 
                        bra_b.begin(), bra_b.end(), 
                        std::back_inserter(exc.virt_beta));
                        
    exc.level = exc.occ_alpha.size() + exc.occ_beta.size();
    return exc;
}

} // namespace ci
} // namespace mshqc