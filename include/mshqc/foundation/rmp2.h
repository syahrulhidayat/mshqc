/**
 * @file rmp2.h
 * @brief Restricted Møller-Plesset 2nd-order perturbation theory (closed-shell)
 * 
 * Implementation of RMP2 energy and T2 amplitudes for closed-shell systems.
 * Simpler than UMP2 because all electrons are paired (α=β orbitals).
 * 
 * THEORY REFERENCES:
 *   - C. Møller & M. S. Plesset, Phys. Rev. 46, 618 (1934) [original MP theory]
 *   - A. Szabo & N. S. Ostlund, "Modern Quantum Chemistry" (1996), Eq. (6.74), p. 354
 *   - J. S. Binkley & J. A. Pople, Int. J. Quantum Chem. 9, 229 (1975) [RMP2 details]
 * 
 * FORMULA (spin-adapted):
 *   E^(2) = Σ_ijab (2<ij|ab> - <ij|ba>) * t_ij^ab
 *   t_ij^ab = <ij|ab> / (ε_i + ε_j - ε_a - ε_b)
 * 
 * where i,j are occupied orbital indices, a,b are virtual indices.
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 * @license MIT
 * 
 * @note This is an original implementation derived from published theory.
 *       No code was copied from existing quantum chemistry software.
 */

#ifndef MSHQC_FOUNDATION_RMP2_H
#define MSHQC_FOUNDATION_RMP2_H

#include "mshqc/scf.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

namespace mshqc {
namespace foundation {

/**
 * @brief RMP2 calculation result
 */
struct RMP2Result {
    double e_rhf;         // RHF reference energy
    double e_corr;        // MP2 correlation energy
    double e_total;       // Total RHF + MP2 energy
    int n_occ;            // Number of occupied orbitals
    int n_virt;           // Number of virtual orbitals
    
    // T2 amplitudes (for wavefunction analysis)
    Eigen::Tensor<double, 4> t2;  // t_ijab amplitudes (occ×occ×virt×virt)
};

/**
 * @brief Restricted Møller-Plesset 2nd order (closed-shell MP2)
 * 
 * REFERENCE: Szabo & Ostlund (1996), Eq. (6.74), p. 354
 * 
 * For closed-shell systems (N electrons, all paired):
 *   - Reference: RHF wavefunction (single determinant)
 *   - Perturbation: electron correlation (instantaneous repulsion)
 *   - 2nd-order energy: E^(2) = Σ_ijab (2<ij|ab> - <ij|ba>) t_ij^ab
 * 
 * Computational scaling: O(N^5) due to 4-index integral transformation
 * 
 * NOTE: This is simpler than UMP2 because:
 *   - Only one spin case (α = β orbitals)
 *   - Spin-adapted formula (factor of 2 from spin)
 *   - No separate same-spin/opposite-spin contributions
 */
class RMP2 {
public:
    /**
     * @brief Construct RMP2 solver
     * @param rhf_result RHF SCF result (must be converged closed-shell)
     * @param basis Basis set
     * @param integrals Integral engine for ERIs
     */
    RMP2(const SCFResult& rhf_result,
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals);
    
    /**
     * @brief Compute RMP2 correlation energy and amplitudes
     * @return RMP2Result containing energy and T2 amplitudes
     * 
     * Algorithm:
     *   1. Transform ERIs from AO → MO basis (4-index transformation)
     *   2. Compute T2 amplitudes: t_ij^ab = <ij|ab> / D_ij^ab
     *   3. Calculate E^(2) = Σ (2<ij|ab> - <ij|ba>) t_ij^ab
     */
    RMP2Result compute();
    
    /**
     * @brief Get T2 amplitudes (for wavefunction container)
     * @return 4D tensor t_ijab (occupied × occupied × virtual × virtual)
     * 
     * NOTE: Only available after compute() has been called
     */
    const Eigen::Tensor<double, 4>& get_t2_amplitudes() const;
    
private:
    const SCFResult& rhf_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    
    // Dimensions
    int nbf_;      // # basis functions
    int nocc_;     // # occupied orbitals (N/2 for closed-shell)
    int nvirt_;    // # virtual orbitals
    
    // MO integrals and amplitudes
    Eigen::Tensor<double, 4> eri_mo_;  // <ij|ab> in MO basis (physicist notation)
    Eigen::Tensor<double, 4> t2_;      // T2 amplitudes
    
    /**
     * @brief Transform AO integrals to MO basis
     * 
     * REFERENCE: Szabo & Ostlund (1996), Eq. (2.282)
     * 
     * Four-index transformation:
     *   <pq|rs>_MO = Σ_μνλσ C_μp C_νq (μν|λσ)_AO C_λr C_σs
     * 
     * This is the bottleneck: O(N^5) scaling
     * Uses physicist notation: <12|12> = ∫ φ1*(1) φ2*(2) (1/r12) φ1(1) φ2(2)
     * 
     * NOTE: Could be optimized with quarter-transformation, but we use
     *       naive algorithm first for correctness.
     */
    void transform_integrals_ao_to_mo();
    
    /**
     * @brief Compute T2 amplitudes
     * 
     * REFERENCE: Szabo & Ostlund (1996), Eq. (6.63)
     * 
     * Amplitude formula:
     *   t_ij^ab = <ij|ab> / (ε_i + ε_j - ε_a - ε_b)
     * 
     * where ε_i, ε_j are occupied orbital energies (negative)
     *       ε_a, ε_b are virtual orbital energies (positive for bound states)
     * 
     * Denominator is always negative (occ energies < virt energies)
     */
    void compute_t2_amplitudes();
    
    /**
     * @brief Compute MP2 correlation energy from T2 amplitudes
     * 
     * REFERENCE: Szabo & Ostlund (1996), Eq. (6.74), p. 354
     * 
     * Energy formula (spin-adapted):
     *   E^(2) = Σ_ijab (2<ij|ab> - <ij|ba>) * t_ij^ab
     *         = Σ_ijab <ij|ab> * (2*t_ij^ab - t_ij^ba)
     * 
     * Factor of 2 from spin (αα and ββ contribute equally)
     * Exchange term <ij|ba> differs by permutation b ↔ a
     * 
     * @return Correlation energy E^(2) (negative for bound states)
     */
    double compute_correlation_energy();
};

} // namespace foundation
} // namespace mshqc

#endif // MSHQC_FOUNDATION_RMP2_H
