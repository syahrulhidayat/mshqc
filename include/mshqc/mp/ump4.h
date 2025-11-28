/**
 * @file ump4.h
 * @brief Unrestricted Møller-Plesset 4th-order perturbation theory
 * 
 * THEORY REFERENCES:
 *   - K. Raghavachari, G. W. Trucks, J. A. Pople, & M. Head-Gordon,
 *     Chem. Phys. Lett. 157, 479 (1989)
 *     [MP4 formulation and implementation strategies]
 *   - J. A. Pople, R. Krishnan, H. B. Schlegel, & J. S. Binkley,
 *     Int. J. Quantum Chem. 14, 545 (1978)
 *     [Fourth-order MBPT for molecules]
 *   - R. J. Bartlett & D. M. Silver, Phys. Rev. A 10, 1927 (1974)
 *     [Many-body perturbation theory diagrams]
 *   - T. Helgaker, P. Jørgensen, & J. Olsen,
 *     "Molecular Electronic-Structure Theory" (2000), Section 14.4
 *     [Møller-Plesset perturbation theory, Eq. (14.66)-(14.70)]
 * 
 * FORMULA (fourth-order energy):
 *   E^(4) = E_S^(4) + E_D^(4) + E_Q^(4) + E_T^(4)
 *   
 *   where:
 *     E_S: Singles contribution (from T1^(3) amplitudes)
 *     E_D: Doubles contribution (from T2^(3) amplitudes)
 *     E_Q: Quadruples contribution (O(N^8)!)
 *     E_T: Triples contribution (O(N^7))
 * 
 * MP4(SDQ): Singles + Doubles + Quadruples (no triples)
 * MP4(SDTQ): Full MP4 (with triples)
 * 
 * Computational scaling:
 *   - T1^(3): O(N^4)
 *   - T2^(3): O(N^6)
 *   - T3^(3): O(N^7)
 *   - T4^(3): O(N^8) - bottleneck!
 * 
 * @author Syahrul
 * @date 2025-11-12
 * @license MIT
 * 
 * @note Original implementation from published theory.
 *       No code copied from existing quantum chemistry software.
 */

#ifndef MSHQC_MP_UMP4_H
#define MSHQC_MP_UMP4_H

#include "mshqc/ump3.h"  // Requires UMP3 for T2^(1) and T2^(2)
#include "mshqc/ump2.h"  // Also needs UMP2 result
#include "mshqc/scf.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

namespace mshqc {
namespace mp {

/**
 * @brief UMP4 calculation result
 * 
 * Contains all energy components and amplitudes from fourth-order MP.
 */
struct UMP4Result {
    double e_uhf;          ///< UHF reference energy
    double e_mp2;          ///< MP2 correlation (2nd order)
    double e_mp3;          ///< MP3 correction (3rd order)
    double e_mp4_sdq;      ///< MP4(SDQ): Singles + Doubles + Quadruples
    double e_mp4_t;        ///< MP4(T): Triples only
    double e_mp4_total;    ///< Total MP4 = SDQ + T
    double e_corr_total;   ///< Total correlation (MP2 + MP3 + MP4)
    double e_total;        ///< UHF + correlation
    
    int n_occ_alpha;       ///< # occupied α orbitals
    int n_occ_beta;        ///< # occupied β orbitals
    int n_virt_alpha;      ///< # virtual α orbitals
    int n_virt_beta;       ///< # virtual β orbitals
    
    // Amplitudes (for wavefunction analysis)
    Eigen::Tensor<double, 2> t1_alpha_3;   ///< T1^(3) α (occ_α × virt_α)
    Eigen::Tensor<double, 2> t1_beta_3;    ///< T1^(3) β
    Eigen::Tensor<double, 4> t2_aa_3;      ///< T2^(3) αα (occ_α^2 × virt_α^2)
    Eigen::Tensor<double, 4> t2_bb_3;      ///< T2^(3) ββ
    Eigen::Tensor<double, 4> t2_ab_3;      ///< T2^(3) αβ
    
    // Second-order amplitudes (for UMP5)
    Eigen::Tensor<double, 4> t2_aa_2;      ///< T2^(2) αα (from UMP3)
    Eigen::Tensor<double, 4> t2_bb_2;      ///< T2^(2) ββ
    Eigen::Tensor<double, 4> t2_ab_2;      ///< T2^(2) αβ
    
    // Second-order triples (for 4th-order wavefunction Ψ^(4))
    Eigen::Tensor<double, 6> t3_aaa_2;     ///< T3^(2) ααα (from UMP3)
    Eigen::Tensor<double, 6> t3_bbb_2;     ///< T3^(2) βββ
    Eigen::Tensor<double, 6> t3_aab_2;     ///< T3^(2) ααβ
    Eigen::Tensor<double, 6> t3_abb_2;     ///< T3^(2) αββ
    
    bool t3_2_available = false;           ///< Flag: whether T3^(2) is stored
};

/**
 * @brief Unrestricted Møller-Plesset 4th order perturbation theory
 * 
 * REFERENCE: Raghavachari et al., Chem. Phys. Lett. 157, 479 (1989)
 * 
 * Fourth-order Møller-Plesset perturbation theory for unrestricted (open-shell)
 * wavefunctions. Requires UMP2 and UMP3 as prerequisites.
 * 
 * The fourth-order energy E^(4) includes contributions from:
 *   1. Singles (S): T1^(3) amplitudes → E_S^(4)
 *      - New at MP4 (no T1 at MP2/MP3 for canonical orbitals)
 *      - Couples Fock matrix with T2 amplitudes
 *      - Scaling: O(N^4)
 * 
 *   2. Doubles (D): T2^(3) amplitudes → E_D^(4)
 *      - Extends MP3 T2^(2) with T1 couplings
 *      - pp, hh, ph diagrams + T1-T2 contractions
 *      - Scaling: O(N^6)
 * 
 *   3. Quadruples (Q): T4^(3) amplitudes → E_Q^(4)
 *      - Four-electron excitations (ijkl → abcd)
 *      - Direct formula: t_ijkl^abcd = <ijkl||abcd> / D
 *      - Scaling: O(N^8) - **BOTTLENECK!**
 * 
 *   4. Triples (T): T3^(3) amplitudes → E_T^(4)
 *      - Three-electron excitations (ijk → abc)
 *      - Complex T2-ERI contractions
 *      - Scaling: O(N^7)
 * 
 * Common approximations:
 *   - MP4(SDQ): Compute S + D + Q only (skip expensive triples)
 *   - MP4(SDTQ): Full MP4 (all terms)
 * 
 * Algorithm outline:
 *   1. Start from UMP3 result (provides T2^(1) and T2^(2))
 *   2. Transform integrals to MO basis (if not cached)
 *   3. Compute T1^(3): Fock-T2 contractions [O(N^4)]
 *   4. Compute T2^(3): pp/hh/ph + T1-T2 couplings [O(N^6)]
 *   5. Compute E_S^(4): <i||a> T1_i^a [O(N^2)]
 *   6. Compute E_D^(4): <ij||ab> T2_ij^ab [O(N^4)]
 *   7. Compute E_Q^(4): <ijkl||abcd>^2 / D (on-the-fly) [O(N^8)]
 *   8. (Optional) Compute T3^(3) and E_T^(4) [O(N^7)]
 * 
 * Memory considerations:
 *   - T1: 2 × N_occ × N_virt (small)
 *   - T2: 3 × N_occ^2 × N_virt^2 (manageable)
 *   - T3: 2 × N_occ^3 × N_virt^3 (large! defer storage)
 *   - T4: NEVER STORE (would be N_occ^4 × N_virt^4 → terabytes!)
 * 
 * Implementation strategy:
 *   - Phase 1: Implement MP4(SDQ) first (skip triples)
 *   - Phase 2: Add MP4(T) triples as separate module
 *   - Phase 3: Optimize with integral screening, symmetry
 */
class UMP4 {
public:
    /**
     * @brief Construct UMP4 solver
     * @param uhf_result UHF SCF result (must be converged)
     * @param ump3_result UMP3 result (provides T2^(1), T2^(2), MO integrals)
     * @param basis Basis set
     * @param integrals Integral engine for ERIs
     * 
     * NOTE: UMP3 must be run first to provide:
     *       - T2^(1) amplitudes (from UMP2)
     *       - T2^(2) amplitudes (from UMP3)
     *       - Orbital energies (from UHF)
     */
    UMP4(const SCFResult& uhf_result,
         const UMP3Result& ump3_result,
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals);
    
    /**
     * @brief Compute UMP4 correlation energy
     * @param include_triples If true, compute full MP4(SDTQ); if false, MP4(SDQ) only
     * @return UMP4Result containing all energy components and amplitudes
     * 
     * Algorithm:
     *   1. Transform ERIs to MO basis (occ-occ-virt-virt blocks)
     *   2. Compute T1^(3) amplitudes: Fock-T2 contractions
     *   3. Compute T2^(3) amplitudes: pp/hh/ph + T1-T2 couplings
     *   4. Calculate E_S^(4), E_D^(4), E_Q^(4)
     *   5. (Optional) Calculate T3^(3) and E_T^(4) if include_triples=true
     * 
     * Computational cost:
     *   - MP4(SDQ): O(N^8) from quadruples
     *   - MP4(T): Additional O(N^7) from triples
     * 
     * Typical usage:
     *   // First run UHF, UMP2, UMP3
     *   UMP4 ump4(uhf_result, ump3_result, basis, integrals);
     *   auto result = ump4.compute(true);  // Full MP4(SDTQ)
     *   // or
     *   auto result_sdq = ump4.compute(false);  // MP4(SDQ) only
     */
    UMP4Result compute(bool include_triples = true);
    
    /**
     * @brief Get T1 amplitudes (3rd order)
     * @return Pair of (T1_alpha, T1_beta) tensors
     * 
     * NOTE: Only available after compute() has been called
     */
    std::pair<const Eigen::Tensor<double, 2>&, 
              const Eigen::Tensor<double, 2>&> get_t1_amplitudes() const;
    
    /**
     * @brief Get T2 amplitudes (3rd order)
     * @return Tuple of (T2_aa, T2_bb, T2_ab) tensors
     * 
     * NOTE: Only available after compute() has been called
     */
    std::tuple<const Eigen::Tensor<double, 4>&,
               const Eigen::Tensor<double, 4>&,
               const Eigen::Tensor<double, 4>&> get_t2_amplitudes() const;
    
private:
    const SCFResult& uhf_;
    const UMP3Result& ump3_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    
    // Dimensions
    int nbf_;              ///< # basis functions
    int nocc_a_, nocc_b_;  ///< # occupied α, β orbitals
    int nvirt_a_, nvirt_b_; ///< # virtual α, β orbitals
    
    // MO integrals (only occ-occ-virt-virt blocks needed)
    Eigen::Tensor<double, 4> eri_ooov_aa_;  ///< <ij|ab> αα (occ-occ-virt-virt)
    Eigen::Tensor<double, 4> eri_ooov_bb_;  ///< <ij|ab> ββ
    Eigen::Tensor<double, 4> eri_ooov_ab_;  ///< <ij|ab> αβ
    
    // Fock matrix in MO basis
    Eigen::MatrixXd fock_mo_a_;  ///< F_α in MO basis
    Eigen::MatrixXd fock_mo_b_;  ///< F_β in MO basis
    
    // Amplitudes
    Eigen::Tensor<double, 2> t1_a_3_, t1_b_3_;     ///< T1^(3) α, β
    Eigen::Tensor<double, 4> t2_aa_3_, t2_bb_3_, t2_ab_3_;  ///< T2^(3) αα, ββ, αβ
    
    // ========================================================================
    // Implementation helpers
    // ========================================================================
    
    /**
     * @brief Transform AO integrals to MO basis
     * 
     * REFERENCE: Szabo & Ostlund (1996), Eq. (2.282)
     * 
     * Four-index transformation: <pq|rs>_MO = Σ_μνλσ C_μp C_νq (μν|λσ)_AO C_λr C_σs
     * 
     * For efficiency, only transform occ-occ × virt-virt blocks:
     *   - <ij|ab> where i,j are occupied, a,b are virtual
     * 
     * Scaling: O(N^5) per spin case
     * Storage: 3 tensors × N_occ^2 × N_virt^2 (manageable)
     */
    void transform_integrals_to_mo();
    
    /**
     * @brief Build Fock matrix in MO basis
     * 
     * F_MO = C^T F_AO C
     * 
     * For canonical orbitals, F_MO is diagonal with eigenvalues ε_p.
     * But we compute full matrix for generality (non-canonical orbitals).
     */
    void build_fock_mo();
    
    /**
     * @brief Compute T1 amplitudes at 3rd order
     * 
     * REFERENCE: Raghavachari et al. (1989), Eq. (7)-(9)
     * 
     * T1 appears first at MP4 (not present in MP2/MP3 for canonical orbitals).
     * 
     * Amplitude equation:
     *   t_i^a(3) = (1/D_i^a) × [ F_ia + Σ_{jbc} <ij||bc> t_jb^(1) t_ic^(1) + ... ]
     * 
     * where D_i^a = ε_i - ε_a (orbital energy denominator)
     * 
     * Main contributions:
     *   1. Fock-T2 contractions: Σ_{jb} F_jb t_ij^ab
     *   2. T2-T2 contractions: Σ_{jbc} <ij||bc> t_jb t_ic
     * 
     * Scaling: O(N^4) from T2-T2 contractions
     */
    void compute_t1_third_order();
    
    /**
     * @brief Compute T2 amplitudes at 3rd order
     * 
     * REFERENCE: Raghavachari et al. (1989), Eq. (10)-(15)
     * 
     * Extends MP3 T2^(2) with additional T1 coupling terms.
     * 
     * Amplitude equation:
     *   t_ij^ab(3) = (1/D_ij^ab) × [ residual_MP3 + T1_couplings ]
     * 
     * where D_ij^ab = ε_i + ε_j - ε_a - ε_b
     * 
     * New T1 coupling terms (not in MP3):
     *   - T1-Fock: Σ_c [ F_ac t_i^c × t_j^b + permutations ]
     *   - T1-T1-ERI: Σ_cd <ab||cd> t_i^c t_j^d
     *   - T1-T2-ERI: Σ_{kc} <ki||ca> t_k^c t_j^b
     * 
     * Scaling: O(N^6) from ERI contractions
     */
    void compute_t2_third_order();
    
    /**
     * @brief Compute singles contribution to E^(4)
     * 
     * REFERENCE: Raghavachari et al. (1989), Eq. (5)
     * 
     * Energy formula:
     *   E_S^(4) = Σ_{ia} F_ia t_i^a(3)
     * 
     * where F_ia is off-diagonal Fock matrix element (should be ~0 for canonical).
     * 
     * For canonical RHF/UHF: F_ia ≈ 0 → E_S ≈ 0
     * For non-canonical: F_ia ≠ 0 → E_S contributes
     * 
     * Scaling: O(N^2)
     * 
     * @return Singles energy contribution E_S^(4)
     */
    double compute_singles_energy();
    
    /**
     * @brief Compute doubles contribution to E^(4)
     * 
     * REFERENCE: Raghavachari et al. (1989), Eq. (6)
     * 
     * Energy formula:
     *   E_D^(4) = Σ_{ijab} <ij||ab> t_ij^ab(3)
     * 
     * Spin-summed:
     *   E_D = Σ <ij_αα|ab_αα> t_ij^ab_αα + Σ <ij_ββ|ab_ββ> t_ij^ab_ββ + 
     *         Σ <ij_αβ|ab_αβ> t_ij^ab_αβ
     * 
     * Scaling: O(N^4)
     * 
     * @return Doubles energy contribution E_D^(4)
     */
    double compute_doubles_energy();
    
    /**
     * @brief Compute quadruples contribution to E^(4)
     * 
     * REFERENCE: Raghavachari et al. (1989), Eq. (16)-(18)
     * 
     * Energy formula:
     *   E_Q^(4) = Σ_{ijkl,abcd} <ijkl||abcd>^2 / D_ijkl^abcd
     * 
     * where D_ijkl^abcd = ε_i + ε_j + ε_k + ε_l - ε_a - ε_b - ε_c - ε_d
     * 
     * This is the O(N^8) bottleneck!
     * 
     * Implementation strategy:
     *   - Do NOT store T4 amplitudes (would be ~TB!)
     *   - Compute <ijkl||abcd> on-the-fly
     *   - Use integral screening: skip if |<ijkl||abcd>| < threshold
     *   - Exploit permutational symmetry
     * 
     * Scaling: O(N^8)
     * 
     * @return Quadruples energy contribution E_Q^(4)
     */
    double compute_quadruples_energy();
    
    /**
     * @brief Compute triples contribution to E^(4)
     * 
     * REFERENCE: Raghavachari et al. (1989), Eq. (19)-(25)
     * 
     * Energy formula:
     *   E_T^(4) = Σ_{ijk,abc} <ijk||abc> t_ijk^abc(3)
     * 
     * where t_ijk^abc(3) requires complex T2-ERI contractions:
     *   t_ijk^abc = (1/D) × [ <ijk||abc> + Σ <ab||de> t_ijk^dec + ... ]
     * 
     * This is expensive O(N^7) and has ~15 diagram terms.
     * 
     * Implementation:
     *   - Start with dominant diagrams only
     *   - Defer full implementation to optimization phase
     * 
     * Scaling: O(N^7)
     * 
     * @return Triples energy contribution E_T^(4)
     * 
     * NOTE: This is the most complex term in MP4!
     */
    double compute_triples_energy();
};

} // namespace mp
} // namespace mshqc

#endif // MSHQC_MP_UMP4_H
