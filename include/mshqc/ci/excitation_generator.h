/**
 * @file excitation_generator.h
 * @brief Generate excited determinants for on-the-fly CI methods
 * 
 * For large CI calculations, we avoid storing full Hamiltonian matrix.
 * Instead, we generate connected determinants on-the-fly during sigma-vector.
 * 
 * Theory References:
 *   - Knowles & Handy (1984), Chem. Phys. Lett. 111, 315
 *   - Olsen et al. (1988), J. Chem. Phys. 89, 2185
 *   - Evangelisti et al. (1983), Chem. Phys. 75, 91 [Direct CI]
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-14
 * 
 * @note Original implementation from published theory.
 *       No code copied from existing quantum chemistry software.
 */

#ifndef MSHQC_CI_EXCITATION_GENERATOR_H
#define MSHQC_CI_EXCITATION_GENERATOR_H

#include <vector>
#include <functional>
#include "mshqc/ci/determinant.h"

namespace mshqc {
namespace ci {

/**
 * Excitation descriptor for generated determinants
 */
struct GeneratedExcitation {
    Determinant det;      // Excited determinant
    int exc_level;        // 1 = single, 2 = double
    int i, j;             // Occupied orbitals (j=-1 for singles)
    int a, b;             // Virtual orbitals (b=-1 for singles)
    bool spin_i, spin_j;  // Spins (false = alpha, true = beta)
};

/**
 * Generate all single excitations from a determinant
 * 
 * For determinant |Φ⟩, generates all |Φ_i^a⟩ where:
 *   - i is occupied orbital
 *   - a is virtual orbital
 *   - Both alpha and beta excitations
 * 
 * @param det Reference determinant
 * @param n_orb Total number of orbitals
 * @param callback Function called for each generated excitation
 * 
 * Example usage:
 *   generate_singles(det, n_orb, [](const GeneratedExcitation& exc) {
 *       // Process excitation
 *   });
 */
void generate_singles(const Determinant& det, 
                      int n_orb,
                      std::function<void(const GeneratedExcitation&)> callback);

/**
 * Generate all double excitations from a determinant
 * 
 * For determinant |Φ⟩, generates all |Φ_ij^ab⟩ where:
 *   - i, j are occupied orbitals
 *   - a, b are virtual orbitals
 *   - α-α, β-β, and α-β excitations
 * 
 * @param det Reference determinant
 * @param n_orb Total number of orbitals
 * @param callback Function called for each generated excitation
 */
void generate_doubles(const Determinant& det,
                      int n_orb,
                      std::function<void(const GeneratedExcitation&)> callback);

/**
 * Generate all connected excitations (singles + doubles)
 * 
 * Convenience function that calls both generate_singles and generate_doubles.
 * 
 * @param det Reference determinant
 * @param n_orb Total number of orbitals
 * @param callback Function called for each generated excitation
 */
void generate_connected_excitations(const Determinant& det,
                                    int n_orb,
                                    std::function<void(const GeneratedExcitation&)> callback);

/**
 * Count number of connected excitations (for memory estimation)
 * 
 * Returns: n_singles + n_doubles
 * 
 * For closed-shell system with nocc occupied, nvirt virtual:
 *   Singles = 2 * nocc * nvirt
 *   Doubles = nocc*(nocc-1)/2 * nvirt*(nvirt-1)/2 * 3  (aa, bb, ab)
 * 
 * @param det Reference determinant
 * @param n_orb Total number of orbitals
 * @return Number of connected determinants
 */
int count_connected_excitations(const Determinant& det, int n_orb);

} // namespace ci
} // namespace mshqc

#endif // MSHQC_CI_EXCITATION_GENERATOR_H
