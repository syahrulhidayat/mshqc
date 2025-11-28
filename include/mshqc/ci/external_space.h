/*
 * MSH-QC: External Space Generator for CASPT
 * 
 * Purpose: Generate determinants OUTSIDE the CAS active space
 *          for CASPT2-CASPT5 perturbation theory calculations
 * 
 * Theory: Andersson et al. (1990) - CASPT2
 *         J. Phys. Chem. 94, 5483-5488
 * 
 * Author: Muhamad Syahrul Hidayat
 * For: AI Agent 3 (CASPT Implementation)
 * Project: MSH-QC by Muhamad Sahrul Hidayat
 * Date: 2025-11-12
 * 
 * License: MIT
 */

#ifndef MSHQC_CI_EXTERNAL_SPACE_H
#define MSHQC_CI_EXTERNAL_SPACE_H

#include <vector>
#include <set>
#include <mshqc/ci/determinant.h>

namespace mshqc {
namespace ci {

/**
 * ExternalSpace
 * 
 * Generates excitations OUTSIDE the CAS active space for CASPT calculations.
 * 
 * Orbital Partitioning:
 * -------------------------------------------------------------------
 * | Core (inactive) | Active (CAS) | Virtual (external) |
 * -------------------------------------------------------------------
 *   0 ... n_core-1    n_core ... n_active-1    n_active ... n_orb-1
 * 
 * External Excitation Classes:
 * 
 * 1. SD (Singles/Doubles):
 *    - Core → Active (hole in core)
 *    - Active → Virtual (particle in virtual)
 *    - Core → Virtual (core-to-virtual)
 * 
 * 2. TQ (Triples/Quadruples):
 *    - Combinations from coupling multiple references
 * 
 * Mathematical Framework:
 * 
 * CASPT wavefunction:
 * |Ψ⟩ = |Ψ₀⟩ + Σ_K t_K |Φ_K⟩
 * 
 * where:
 * - |Ψ₀⟩ is CASSCF reference (multi-configurational)
 * - |Φ_K⟩ are EXTERNAL configurations
 * - t_K are first-order amplitudes
 * 
 * Second-order energy:
 * E⁽²⁾ = Σ_K |⟨Φ_K|V|Ψ₀⟩|² / (E₀ - E_K)
 */
class ExternalSpace {
public:
    /**
     * Generate all external configurations for CASPT
     * 
     * @param cas_dets      CAS space determinants (from CASSCF)
     * @param n_core        Number of core (inactive) orbitals
     * @param n_active      Number of active orbitals
     * @param n_virtual     Number of virtual orbitals
     * @param max_excit     Maximum excitation level (2=SD, 4=SDTQ)
     * 
     * @return Vector of external determinants
     * 
     * Theory: Andersson et al. (1990), Section II
     * 
     * Complexity: O(N_ref × N_core × N_virt) for SD
     *             O(N_ref × N_core² × N_virt²) for TQ
     */
    static std::vector<Determinant> generate_external(
        const std::vector<Determinant>& cas_dets,
        int n_core,
        int n_active,
        int n_virtual,
        int max_excit = 2  // 2=SD (CASPT2), 4=SDTQ (CASPT3+)
    );
    
    /**
     * Generate SD (singles/doubles) external space
     * 
     * Three types of SD excitations:
     * 
     * 1. Semi-internal (Active → Virtual):
     *    |Φᵃᵛ⟩ = a†_v a_a |Ψ₀⟩
     *    Creates particle in virtual space
     * 
     * 2. Semi-external (Core → Active):
     *    |Φᶜᵃ⟩ = a†_a a_c |Ψ₀⟩
     *    Creates hole in core space
     * 
     * 3. Doubly external (Core → Virtual):
     *    |Φᶜᵛ⟩ = a†_v a_c |Ψ₀⟩
     *    Core hole + virtual particle
     * 
     * @param cas_ref    Single CAS reference determinant
     * @param n_core     Core orbitals
     * @param n_active   Active orbitals
     * @param n_virtual  Virtual orbitals
     * 
     * @return SD external configurations from this reference
     * 
     * Theory: Andersson et al. (1990), Eq. 10-12
     */
    static std::vector<Determinant> generate_sd_external(
        const Determinant& cas_ref,
        int n_core,
        int n_active,
        int n_virtual
    );
    
    /**
     * Generate TQ (triples/quadruples) external space
     * 
     * Higher-order excitations from coupling:
     * 
     * 1. Triples:
     *    - Core → Active (2 holes) + Active → Virtual (1 particle)
     *    - Core → Active (1 hole) + Active → Virtual (2 particles)
     * 
     * 2. Quadruples:
     *    - Core → Active (2 holes) + Active → Virtual (2 particles)
     * 
     * Required for CASPT3 and higher.
     * 
     * @param cas_ref    Single CAS reference determinant
     * @param n_core     Core orbitals
     * @param n_active   Active orbitals
     * @param n_virtual  Virtual orbitals
     * 
     * @return TQ external configurations
     * 
     * Theory: Andersson & Roos (1993) - CASPT3
     *         Int. J. Quantum Chem. 45, 591-607
     */
    static std::vector<Determinant> generate_tq_external(
        const Determinant& cas_ref,
        int n_core,
        int n_active,
        int n_virtual
    );
    
    /**
     * Remove duplicate determinants
     * 
     * Multiple CAS references can generate the same external configurations.
     * This function removes duplicates efficiently.
     * 
     * @param dets  Vector of determinants (modified in-place)
     * 
     * Algorithm: Sort + unique (O(N log N))
     * 
     * Note: Determinant class must implement operator< for sorting
     */
    static void remove_duplicates(std::vector<Determinant>& dets);
    
    /**
     * Count external configurations (without generating)
     * 
     * Useful for memory estimation before generation.
     * 
     * @param n_cas_ref  Number of CAS reference determinants
     * @param n_core     Core orbitals
     * @param n_active   Active orbitals
     * @param n_virtual  Virtual orbitals
     * @param max_excit  Maximum excitation level
     * 
     * @return Estimated number of external determinants
     * 
     * Formula (SD only):
     * N_ext ≈ N_ref × (N_act × N_virt + N_core × N_act + N_core × N_virt)
     */
    static size_t count_external(
        size_t n_cas_ref,
        int n_core,
        int n_active,
        int n_virtual,
        int max_excit = 2
    );
    
    /**
     * Classify excitation type
     * 
     * @param det        External determinant
     * @param cas_ref    CAS reference determinant
     * @param n_core     Core orbitals
     * @param n_active   Active orbitals
     * 
     * @return String: "semi-internal", "semi-external", "doubly-external", etc.
     * 
     * Useful for CASPT energy decomposition.
     */
    static std::string classify_excitation(
        const Determinant& det,
        const Determinant& cas_ref,
        int n_core,
        int n_active
    );

private:
    // Helper: Check if orbital is in core region
    static bool is_core(int orb, int n_core);
    
    // Helper: Check if orbital is in active region
    static bool is_active(int orb, int n_core, int n_active);
    
    // Helper: Check if orbital is in virtual region
    static bool is_virtual(int orb, int n_core, int n_active);
    
    // Helper: Generate single excitation (i → a) from determinant
    static Determinant single_excitation(
        const Determinant& det,
        int from_orb,
        int to_orb,
        bool alpha
    );
    
    // Helper: Generate double excitation (i,j → a,b) from determinant
    static Determinant double_excitation(
        const Determinant& det,
        int from_i, int from_j,
        int to_a, int to_b,
        bool alpha_i, bool alpha_j
    );
};

/**
 * Usage Example (for Agent 3):
 * 
 * ```cpp
 * // After CASSCF convergence
 * auto cas_dets = casscf_result.ci_determinants;
 * 
 * // Define orbital spaces
 * int n_core = 5;      // First 5 orbitals (frozen)
 * int n_active = 4;    // CAS(8,4) - 4 active orbitals
 * int n_virtual = 10;  // Virtual orbitals
 * 
 * // Generate external space for CASPT2
 * auto external_dets = ExternalSpace::generate_external(
 *     cas_dets, n_core, n_active, n_virtual, 2  // max_excit=2 for SD
 * );
 * 
 * std::cout << "CAS refs: " << cas_dets.size() << std::endl;
 * std::cout << "External: " << external_dets.size() << std::endl;
 * 
 * // Compute CASPT2 energy
 * double e2 = 0.0;
 * for (const auto& ext_det : external_dets) {
 *     double matrix_elem = caspt_matrix_element(cas_dets, ext_det, integrals);
 *     double denominator = E_cas - external_energy(ext_det);
 *     e2 += (matrix_elem * matrix_elem) / denominator;
 * }
 * 
 * std::cout << "CASPT2 correlation: " << e2 << " Ha" << std::endl;
 * ```
 */

} // namespace ci
} // namespace mshqc

#endif // MSHQC_CI_EXTERNAL_SPACE_H
