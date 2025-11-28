// Author: Muhamad Syahrul Hidayat
// Date: 2025-11-16
//
// Size-consistency corrections for truncated CI methods (Davidson +Q, etc.)
//
// Theory References:
// - Davidson, E. R. (1974). "Size consistency in the dilute helium gas electronic structure"
//   J. Chem. Phys. 62, 400.
//   DOI: 10.1063/1.430484
//
// - Langhoff, S. R. & Davidson, E. R. (1974). "Configuration interaction calculations 
//   on the nitrogen molecule"
//   Int. J. Quantum Chem. 8, 61-72.
//   DOI: 10.1002/qua.560080106
//
// - Pople, J. A., Head-Gordon, M., & Raghavachari, K. (1987). 
//   "Quadratic configuration interaction - A general technique for determining 
//   electron correlation energies"
//   J. Chem. Phys. 87, 5968-5975.
//   DOI: 10.1063/1.453520
//
// Davidson +Q Correction:
//   E(CISD+Q) = E(CISD) + ΔE_Q
//   
//   ΔE_Q = (1 - c₀²) * ΔE_corr
//   
//   where:
//     c₀ = coefficient of HF determinant in CISD wavefunction
//     ΔE_corr = E(CISD) - E(HF)
//
// This correction approximately accounts for higher-order excitations (triples, quadruples)
// and improves size-consistency of CISD for weakly interacting systems.
//
// Pople's Quadratic CI (QCI):
//   More sophisticated approach using second-order perturbation theory with CISD as reference.
//   E(QCI) = E(CISD) + <0|H|T>/ΔE
//   where |T> are triple excitations, ΔE are energy denominators
//
// ============================================================================
// ORIGINAL IMPLEMENTATION - NO CODE COPIED FROM PYSCF/PSI4
// ============================================================================

#ifndef MSHQC_CI_SIZE_CONSISTENCY_H
#define MSHQC_CI_SIZE_CONSISTENCY_H

#include <vector>
#include <string>

namespace mshqc {
namespace ci {

/**
 * Size-consistency correction for truncated CI methods
 * 
 * Truncated CI methods (CISD, CISDT, etc.) are not size-consistent:
 * For non-interacting fragments A...B, E(A...B) ≠ E(A) + E(B)
 * 
 * Davidson +Q correction empirically restores size-consistency for CISD
 * by accounting for higher excitations via the HF reference weight.
 */
class SizeConsistencyCorrection {
public:
    /**
     * Davidson +Q correction for CISD
     * 
     * Formula: E(CISD+Q) = E(CISD) + (1 - c₀²) * ΔE_corr
     * 
     * Physical interpretation:
     *   - If c₀ ≈ 1 (dominated by HF), correction is small (single-reference)
     *   - If c₀ << 1 (multi-reference), correction is large
     *   - Accounts for quadruples via perturbation theory
     * 
     * @param e_cisd CISD energy (Hartree)
     * @param e_hf Hartree-Fock reference energy (Hartree)
     * @param c0 Coefficient of HF determinant in CISD wavefunction
     * @return Davidson +Q correction energy ΔE_Q (Hartree)
     */
    static double davidson_q_correction(double e_cisd, double e_hf, double c0);

    /**
     * Compute Davidson +Q corrected energy
     * 
     * @param e_cisd CISD energy
     * @param e_hf HF energy
     * @param c0 HF coefficient
     * @return E(CISD+Q) = E(CISD) + ΔE_Q
     */
    static double cisd_plus_q(double e_cisd, double e_hf, double c0);

    /**
     * Size-consistency error metric for CISD
     * 
     * Measures deviation from size-extensivity:
     *   Error(n) = E(n*A) - n*E(A)
     * 
     * For size-consistent methods: Error → 0 as n → ∞
     * For CISD: Error scales as O(n)
     * 
     * @param e_single Energy of single system A
     * @param e_multiple Energy of n non-interacting copies
     * @param n Number of copies
     * @return Size-consistency error (should be zero for exact methods)
     */
    static double size_consistency_error(double e_single, double e_multiple, int n);

    /**
     * Renormalized Davidson correction (for multi-reference cases)
     * 
     * Modified formula accounting for CAS reference instead of HF:
     *   E(MRCISD+Q) = E(MRCISD) + (1 - Σᵢc²ᵢ) * ΔE_corr
     * 
     * where sum is over reference determinants in CAS space
     * 
     * @param e_mrcisd MRCISD energy
     * @param e_cas CAS reference energy
     * @param ref_weights Vector of reference determinant coefficients
     * @return Renormalized +Q correction
     */
    static double renormalized_q_correction(
        double e_mrcisd, 
        double e_cas,
        const std::vector<double>& ref_weights
    );

    /**
     * Pople's Quadratic CI (QCISD) approximation
     * 
     * More accurate than Davidson +Q, accounts for connected triples:
     *   E(QCISD) ≈ E(CISD) + k * (1 - c₀²)² * ΔE_corr
     * 
     * where k is empirical constant (typically ~0.5-1.0)
     * 
     * Reference: Pople et al., J. Chem. Phys. 87, 5968 (1987)
     * 
     * @param e_cisd CISD energy
     * @param e_hf HF energy
     * @param c0 HF coefficient
     * @param k Empirical scaling factor (default 0.75)
     * @return QCISD approximation energy
     */
    static double qcisd_approximation(
        double e_cisd, 
        double e_hf, 
        double c0,
        double k = 0.75
    );

    /**
     * Diagnostic: Check if Davidson +Q is reliable
     * 
     * Davidson +Q works well when:
     *   - 0.90 < c₀² < 0.99 (weakly correlated, single-reference)
     *   - System is near equilibrium geometry
     * 
     * Warning if:
     *   - c₀² < 0.90 (multi-reference, use MRCI instead)
     *   - c₀² > 0.99 (almost HF, correction negligible)
     * 
     * @param c0 HF coefficient
     * @return True if Davidson +Q is appropriate
     */
    static bool is_davidson_q_reliable(double c0);

    /**
     * Get diagnostic message about correction applicability
     * 
     * @param c0 HF coefficient
     * @return Human-readable diagnostic string
     */
    static std::string get_diagnostic_message(double c0);
};

} // namespace ci
} // namespace mshqc

#endif // MSHQC_CI_SIZE_CONSISTENCY_H
