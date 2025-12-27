#ifndef MSHQC_CI_DETERMINANT_H
#define MSHQC_CI_DETERMINANT_H

#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <functional>
#include <algorithm> // for std::max

/**
 * @file determinant.h
 * @brief Slater determinant representation with Arbitrary Size Bitset
 * * UPGRADE: Now supports > 64 orbitals using multi-word bitset (std::vector<uint64_t>).
 * Essential for large basis sets like cc-pV5Z (91 orbitals).
 * * @author Muhamad Sahrul Hidayat
 * @date 2025-12-15
 */

namespace mshqc {
namespace ci {

/**
 * Slater determinant using multi-word bit representation
 */
class Determinant {
public:
    // Default constructor
    Determinant() : n_alpha_(0), n_beta_(0) {
        // Default small size, will expand if needed
        alpha_bits_.push_back(0);
        beta_bits_.push_back(0);
    }
    
    // Construct from occupation lists
    Determinant(const std::vector<int>& alpha_occ,
                const std::vector<int>& beta_occ);
    
    // Advanced: Construct from raw bit vectors
    Determinant(const std::vector<uint64_t>& alpha, 
                const std::vector<uint64_t>& beta);

    // --- Accessors ---
    int n_alpha() const { return n_alpha_; }
    int n_beta() const { return n_beta_; }
    
    // --- Orbital Queries ---
    
    // Check if orbital is occupied
    bool is_occupied(int orb, bool alpha) const;
    
    // Get full occupation lists
    std::vector<int> alpha_occupations() const;
    std::vector<int> beta_occupations() const;
    
    // --- Excitations ---
    
    // Single excitation: i -> a
    Determinant single_excite(int i, int a, bool alpha) const;
    
    // Double excitation: ij -> ab
    Determinant double_excite(int i, int j, int a, int b,
                               bool spin1, bool spin2) const;
    
    // --- Analysis ---
    
    // Count differences (for Slater-Condon)
    int count_differences(const Determinant& other) const;
    
    // Get excitation level {n_diff_alpha, n_diff_beta}
    std::pair<int, int> excitation_level(const Determinant& other) const;
    
    // Compute phase factor (-1)^N_perm
    int phase(int i, int a, bool alpha) const;
    
    std::string to_string() const;
    
    // --- Operators ---
    bool operator==(const Determinant& other) const;
    bool operator!=(const Determinant& other) const;
    bool operator<(const Determinant& other) const;
    
    // --- Static Helpers ---
    static int popcount(uint64_t bits);
    static std::vector<int> find_differences(const std::vector<uint64_t>& bits1, 
                                             const std::vector<uint64_t>& bits2);

private:
    // Multi-word storage: each uint64_t holds 64 orbitals
    // Index 0: Orb 0-63, Index 1: Orb 64-127, etc.
    std::vector<uint64_t> alpha_bits_;
    std::vector<uint64_t> beta_bits_;
    
    int n_alpha_;
    int n_beta_;
    
    // Helper to set/clear bits safely
    void set_bit(std::vector<uint64_t>& bits, int orb);
    void clear_bit(std::vector<uint64_t>& bits, int orb);
    bool check_bit(const std::vector<uint64_t>& bits, int orb) const;
};

// --- Excitation Descriptor ---
struct Excitation {
    std::vector<int> occ_alpha;   
    std::vector<int> occ_beta;    
    std::vector<int> virt_alpha;  
    std::vector<int> virt_beta;   
    int level;                    
};

Excitation find_excitation(const Determinant& bra, const Determinant& ket);

} // namespace ci
} // namespace mshqc

// Hash Function Update
namespace std {
    template<>
    struct hash<mshqc::ci::Determinant> {
        std::size_t operator()(const mshqc::ci::Determinant& det) const noexcept {
            std::size_t seed = 0;
            auto combine = [&](uint64_t v) {
                seed ^= std::hash<uint64_t>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            };
            
            // Hash alpha vector content (need accessor or friend, but for now assuming public access or helper)
            // Since we don't have direct access to vector here, we rely on a hypothetical `hash_value()` method
            // or simply hash the occupations which are public.
            // For efficiency, Determinant should expose a hash method.
            
            // Temporary simple hash based on occupations (Safe but slower)
            auto occ_a = det.alpha_occupations();
            for(int i : occ_a) combine(i);
            
            auto occ_b = det.beta_occupations();
            for(int i : occ_b) combine(i + 10000); // Offset for beta
            
            return seed;
        }
    };
}

#endif // MSHQC_CI_DETERMINANT_H