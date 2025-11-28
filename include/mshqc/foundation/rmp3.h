/**
 * @file rmp3.h
 * @brief Restricted Møller-Plesset 3rd-order perturbation theory (closed-shell)
 * 
 * Implementation of RMP3 energy correction for closed-shell systems.
 * Computes third-order correction E^(3) using first-order amplitudes T2^(1) from RMP2
 * and second-order amplitudes T2^(2).
 * 
 * THEORY REFERENCES:
 *   - J. A. Pople, R. Seeger, & R. Krishnan, Int. J. Quantum Chem. Symp. 11, 149 (1977)
 *     [RMP3 formulation for closed-shell systems]
 *   - R. J. Bartlett & D. M. Silver, Phys. Rev. A 10, 1927 (1974)
 *     [Third-order MBPT diagrams and expressions]
 *   - K. Raghavachari, J. A. Pople, et al., Chem. Phys. Lett. 157, 479 (1989)
 *     [Efficient implementation strategies]
 *   - A. Szabo & N. S. Ostlund, "Modern Quantum Chemistry" (1996), Ch. 6
 *     [Textbook reference for MP perturbation theory]
 * 
 * FORMULA (third-order energy):
 *   E^(3) = Σ_ijab <ij|ab> * t_ij^ab(2)
 * 
 * where t_ij^ab(2) is the second-order correction to amplitudes computed from:
 *   - Fock matrix couplings (occupied-occupied, virtual-virtual)
 *   - Four-index ERI contractions with T2^(1)
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 * @license MIT
 * 
 * @note This is an original implementation derived from published theory.
 *       No code was copied from existing quantum chemistry software.
 *       Algorithm based on Pople et al. (1977) equations.
 */

#ifndef MSHQC_FOUNDATION_RMP3_H
#define MSHQC_FOUNDATION_RMP3_H

#include "mshqc/foundation/rmp2.h"
#include "mshqc/scf.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

namespace mshqc {
namespace foundation {

/**
 * @brief RMP3 calculation result
 */
struct RMP3Result {
    double e_rhf;         // RHF reference energy
    double e_mp2;         // MP2 correlation energy (2nd order)
    double e_mp3;         // MP3 correction energy (3rd order)
    double e_corr_total;  // Total correlation (MP2 + MP3)
    double e_total;       // Total energy (RHF + MP2 + MP3)
    
    int n_occ;            // Number of occupied orbitals
    int n_virt;           // Number of virtual orbitals
    
    // T2 amplitudes at different orders
    Eigen::Tensor<double, 4> t2_1;  // First-order (from RMP2)
    Eigen::Tensor<double, 4> t2_2;  // Second-order correction
};

/**
 * @brief Restricted Møller-Plesset 3rd order (closed-shell MP3)
 * 
 * REFERENCE: Pople et al., Int. J. Quantum Chem. Symp. 11, 149 (1977)
 * 
 * Third-order Møller-Plesset perturbation theory for closed-shell systems.
 * Requires RMP2 results as input (T2^(1) amplitudes).
 * 
 * The third-order energy E^(3) includes contributions from:
 *   1. Particle-particle (pp) ladder diagram
 *   2. Hole-hole (hh) ladder diagram  
 *   3. Particle-hole (ph) ring diagram
 *   4. Three-body terms (typically small)
 * 
 * Computational scaling: O(N^6) due to T2^(2) computation
 * 
 * Algorithm outline:
 *   1. Start with RHF and RMP2 results
 *   2. Compute T2^(2) amplitudes from various contractions
 *   3. Calculate E^(3) = <Φ_HF| H^(0) |Ψ^(2)>
 * 
 * NOTE: MP3 often provides ~90% of correlation energy recovered by CCSD(T)
 *       at much lower cost, making it useful for large systems.
 */
class RMP3 {
public:
    /**
     * @brief Construct RMP3 solver
     * @param rhf_result RHF SCF result (must be converged closed-shell)
     * @param rmp2_result RMP2 result containing T2^(1) amplitudes
     * @param basis Basis set
     * @param integrals Integral engine for ERIs
     * 
     * NOTE: RMP2 must be run first to provide T2^(1) amplitudes
     */
    RMP3(const SCFResult& rhf_result,
         const RMP2Result& rmp2_result,
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals);
    
    /**
     * @brief Compute RMP3 third-order energy correction
     * @return RMP3Result containing all energy components and amplitudes
     * 
     * Algorithm:
     *   1. Build Fock matrix in MO basis
     *   2. Transform ERIs to MO basis (if not already done)
     *   3. Compute T2^(2) amplitudes from Fock and ERI contractions
     *   4. Calculate E^(3) = Σ <ij|ab> t_ij^ab(2)
     */
    RMP3Result compute();
    
    /**
     * @brief Get T2^(2) amplitudes (for wavefunction container)
     * @return 4D tensor t_ijab (occupied × occupied × virtual × virtual)
     * 
     * NOTE: Only available after compute() has been called
     */
    const Eigen::Tensor<double, 4>& get_t2_second_order() const;
    
private:
    const SCFResult& rhf_;
    const RMP2Result& rmp2_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    
    // Dimensions
    int nbf_;      // # basis functions
    int nocc_;     // # occupied orbitals
    int nvirt_;    // # virtual orbitals
    
    // MO quantities
    Eigen::MatrixXd fock_mo_;              // Fock matrix in MO basis
    Eigen::Tensor<double, 4> eri_mo_;      // <ij|ab> in MO basis
    Eigen::Tensor<double, 4> t2_2_;        // T2^(2) amplitudes
    
    /**
     * @brief Build Fock matrix in MO basis
     * 
     * F_pq = C_p^T F_AO C_q
     * 
     * where F_AO is the Fock matrix from RHF in AO basis.
     * In canonical RHF orbitals, F_MO is diagonal with eigenvalues ε_i.
     */
    void build_fock_mo();
    
    /**
     * @brief Transform AO integrals to MO basis
     * 
     * Same as RMP2 transformation: <pq|rs>_MO = Σ_μνλσ C_μp C_νq (μν|λσ)_AO C_λr C_σs
     * 
     * NOTE: If RMP2 already transformed integrals, could reuse them.
     *       For now, we transform independently.
     */
    void transform_integrals_ao_to_mo();
    
    /**
     * @brief Compute T2^(2) amplitudes (second-order correction)
     * 
     * REFERENCE: Pople et al. (1977), Eq. (15)-(17)
     * 
     * T2^(2) has contributions from:
     * 
     * 1. Virtual-virtual Fock coupling:
     *    t_ij^ab(2) += Σ_c F_ac t_ij^cb(1) + F_bc t_ij^ac(1)
     * 
     * 2. Occupied-occupied Fock coupling:
     *    t_ij^ab(2) -= Σ_k F_ki t_kj^ab(1) + F_kj t_ik^ab(1)
     * 
     * 3. Four-index ERI contractions:
     *    t_ij^ab(2) += Σ_kc <kc|ab> t_ij^kc(1)    [particle-particle]
     *    t_ij^ab(2) += Σ_kl <kl|ij> t_kl^ab(1)    [hole-hole]
     *    t_ij^ab(2) += Σ_kc <ki|ca> t_kj^cb(1)    [particle-hole, ring]
     *    (and permutations)
     * 
     * All divided by energy denominator: D_ij^ab = ε_i + ε_j - ε_a - ε_b
     * 
     * Computational cost: O(N^6) from Σ_kc contractions
     */
    void compute_t2_second_order();
    
    /**
     * @brief Compute MP3 third-order energy
     * 
     * REFERENCE: Pople et al. (1977), Eq. (12)
     * 
     * E^(3) = Σ_ijab <ij|ab> * t_ij^ab(2)
     * 
     * where <ij|ab> are the same MO integrals used in MP2,
     * and t_ij^ab(2) are the second-order amplitude corrections.
     * 
     * The spin-adapted formula includes factor of (2<ij|ab> - <ij|ba>):
     * E^(3) = Σ_ijab (2<ij|ab> - <ij|ba>) * t_ij^ab(2)
     * 
     * @return Third-order energy correction E^(3)
     */
    double compute_third_order_energy();
};

} // namespace foundation
} // namespace mshqc

#endif // MSHQC_FOUNDATION_RMP3_H
