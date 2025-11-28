/**
 * @file mp_ci_adapter.h
 * @brief Integration adapter: MP module <-> CI determinant infrastructure
 * 
 * PURPOSE:
 *   Provide clean interface for MP perturbation theory (Agent 1) to reuse
 *   CI determinant infrastructure (Agent 2) without modifying original code.
 * 
 * INTEGRATION TASK 1.1 (Week 1, Agent 3):
 *   - Replace manual 6-deep loops in MP T3 computation
 *   - Reuse CI triple_excite() for automatic fermion statistics
 *   - Maintain compatibility with existing MP code
 * 
 * Theory References:
 *   - Raghavachari et al. (1989), Chem. Phys. Lett. 157, 479
 *     [Triple excitations in MP perturbation theory]
 *   - Pople et al. (1977), Int. J. Quantum Chem. 11, 149
 *     [Third-order MP3 theory with T3^(2) amplitudes]
 *   - Szabo & Ostlund (1996), Modern Quantum Chemistry, Appendix A
 *     [Fermion statistics and phase factors]
 * 
 * @author Muhamad Syahrul Hidayat (Agent 3)
 * @date 2025-11-16
 * @license MIT License
 * 
 * @note ORIGINAL IMPLEMENTATION - Integration layer for CI-MP synergy
 * @note NO CODE COPIED from existing quantum chemistry software
 */

#ifndef MSHQC_INTEGRATION_MP_CI_ADAPTER_H
#define MSHQC_INTEGRATION_MP_CI_ADAPTER_H

#include "mshqc/ci/determinant.h"
#include <vector>
#include <functional>

namespace mshqc {
namespace integration {

/**
 * Triple excitation indices for MP perturbation theory
 * 
 * Stores orbital indices for T3 amplitude: t_ijk^abc
 * Formula: t_ijk^abc(2) = <ijk||abc> / D_ijk^abc
 */
struct TripleExcitation {
    int i, j, k;  // Occupied orbital indices
    int a, b, c;  // Virtual orbital indices
    bool same_spin;  // true for aaa or bbb, false for mixed
    
    // Constructor for convenience
    TripleExcitation(int i_, int j_, int k_, int a_, int b_, int c_,
                     bool same_spin_ = true)
        : i(i_), j(j_), k(k_), a(a_), b(b_), c(c_), same_spin(same_spin_) {}
};

/**
 * Generate all alpha-alpha-alpha triple excitations
 * 
 * For HF determinant |Φ₀⟩ with n_occ_alpha occupied α orbitals and
 * n_virt_alpha virtual α orbitals, generates ALL triple excitations:
 *   |Φ_ijk^abc⟩ where i,j,k ∈ {occupied} and a,b,c ∈ {virtual}
 * 
 * Theory:
 *   T3^(2) amplitude formula (Pople 1977, Eq. 13):
 *   t_ijk^abc(2) = <ijk||abc> / (ε_i + ε_j + ε_k - ε_a - ε_b - ε_c)
 * 
 * Benefit over manual loops:
 *   - Automatic fermion statistics via CI Determinant class
 *   - Eliminates 6-deep nested loops (i,j,k,a,b,c)
 *   - Consistent with CI excitation generation
 *   - Easy to debug and maintain
 * 
 * @param hf_det Reference Hartree-Fock determinant |Φ₀⟩
 * @param n_occ_alpha Number of α occupied orbitals
 * @param n_virt_alpha Number of α virtual orbitals
 * @param callback Function called for each triple excitation
 *                  Signature: void callback(const TripleExcitation& exc)
 * 
 * Usage example (in UMP3::compute_t3_2nd_order):
 *   ```cpp
 *   // Build HF determinant
 *   std::vector<int> alpha_occ = {0, 1, 2};  // Occupied orbitals
 *   ci::Determinant hf_det(alpha_occ, beta_occ);
 *   
 *   // Generate all triples
 *   generate_triples_alpha(hf_det, nocc_a_, nvir_a_,
 *       [&](const TripleExcitation& exc) {
 *           // Compute T3 amplitude
 *           double D = ea(exc.i) + ea(exc.j) + ea(exc.k)
 *                    - ea(nocc_a_+exc.a) - ea(nocc_a_+exc.b) - ea(nocc_a_+exc.c);
 *           double numerator = compute_3body_integral(exc.i, exc.j, exc.k,
 *                                                      exc.a, exc.b, exc.c);
 *           t3_aaa_2_(exc.i, exc.j, exc.k, exc.a, exc.b, exc.c) = numerator / D;
 *       });
 *   ```
 * 
 * @note This is equivalent to manual 6-deep loops but cleaner and safer
 * @note Phase factors handled automatically by CI Determinant::triple_excite()
 */
void generate_triples_alpha(const ci::Determinant& hf_det,
                             int n_occ_alpha,
                             int n_virt_alpha,
                             std::function<void(const TripleExcitation&)> callback);

/**
 * Generate all beta-beta-beta triple excitations
 * 
 * Same as generate_triples_alpha but for β-spin electrons.
 * 
 * @param hf_det Reference HF determinant
 * @param n_occ_beta Number of β occupied orbitals
 * @param n_virt_beta Number of β virtual orbitals
 * @param callback Function called for each triple excitation
 */
void generate_triples_beta(const ci::Determinant& hf_det,
                            int n_occ_beta,
                            int n_virt_beta,
                            std::function<void(const TripleExcitation&)> callback);

/**
 * Helper: Build HF determinant from occupation numbers
 * 
 * Convenience function to create CI Determinant from MP/SCF orbital info.
 * 
 * @param n_occ_alpha Number of α occupied orbitals (0, 1, ..., n_occ_alpha-1)
 * @param n_occ_beta Number of β occupied orbitals (0, 1, ..., n_occ_beta-1)
 * @return CI Determinant object representing |Φ_HF⟩
 * 
 * Example:
 *   For Li (3 electrons, 2α+1β):
 *     n_occ_alpha = 2 → α occupied: {0, 1}
 *     n_occ_beta = 1  → β occupied: {0}
 *     Returns: Determinant with α={0,1}, β={0}
 */
ci::Determinant build_hf_determinant(int n_occ_alpha, int n_occ_beta);

/**
 * Count total number of triple excitations (memory estimation)
 * 
 * Formula:
 *   N_triples(α³) = C(n_occ_α, 3) × C(n_virt_α, 3)
 *                 = [n_occ × (n_occ-1) × (n_occ-2) / 6] ×
 *                   [n_virt × (n_virt-1) × (n_virt-2) / 6]
 * 
 * @param n_occ Number of occupied orbitals
 * @param n_virt Number of virtual orbitals
 * @return Total number of triple excitations
 * 
 * Example:
 *   Li/cc-pVDZ: n_occ=2, n_virt=12
 *   N_triples = 0 (need ≥3 electrons!)
 *   
 *   Be/cc-pVDZ: n_occ=2, n_virt=12  
 *   N_triples = C(2,3) × C(12,3) = 0 × 220 = 0 (still need ≥3)
 */
size_t count_triple_excitations(int n_occ, int n_virt);

} // namespace integration
} // namespace mshqc

#endif // MSHQC_INTEGRATION_MP_CI_ADAPTER_H
