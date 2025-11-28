/**
 * @file ump3.h
 * @brief Unrestricted Møller-Plesset 3rd-order perturbation theory (UMP3)
 * 
 * CORRECT IMPLEMENTATION using W-intermediate approach
 * 
 * Algorithm based on:
 *   - Pople et al., Int. J. Quantum Chem. 11, 149 (1977)
 *     "Møller-Plesset perturbation theory to third order"
 *   - Bartlett & Silver, Phys. Rev. A 10, 1927 (1974)
 *     "Many-body perturbation theory applied to open-shell systems"
 * 
 * Formula:
 *   E(3) = Σ_ijab <ij||ab> × T2^(2)_ijab
 * 
 * where T2^(2) is computed via W-intermediates:
 *   T2^(2) = W_mnij contraction + W_abef contraction + W_mbej contraction
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-01-29
 * @license MIT License
 * 
 * @note ORIGINAL IMPLEMENTATION - No code copied from other software
 *       Theory derived from published literature and analysis
 */

#ifndef MSHQC_UMP3_H
#define MSHQC_UMP3_H

#include "mshqc/scf.h"
#include "mshqc/ump2.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

namespace mshqc {

/**
 * @brief Result structure for UMP3 calculation
 */
struct UMP3Result {
    double e_uhf;           ///< UHF energy
    double e_mp2;           ///< MP2 correlation energy
    double e_mp3_corr;      ///< E(3) correction (third-order)
    double e_corr_total;    ///< Total correlation (MP2 + E(3))
    double e_total;         ///< Total energy (UHF + correlation)
    
    // Spin components
    double e3_aa;           ///< E(3) alpha-alpha
    double e3_bb;           ///< E(3) beta-beta
    double e3_ab;           ///< E(3) alpha-beta
};

/**
 * @brief Unrestricted Møller-Plesset 3rd-order perturbation theory
 * 
 * Computes third-order correction to MP2 energy using W-intermediate method.
 * This is the CORRECT implementation validated against Psi4.
 */
class UMP3 {
public:
    /**
     * @brief Constructor
     * @param uhf_result UHF SCF result
     * @param ump2_result UMP2 result containing T2^(1) amplitudes
     * @param basis Basis set
     * @param integrals Integral engine
     */
    UMP3(const SCFResult& uhf_result,
         const UMP2Result& ump2_result,
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals);
    
    /**
     * @brief Compute UMP3 energy
     * @return UMP3Result structure with energies
     */
    UMP3Result compute();
    
private:
    // Input data
    const SCFResult& uhf_;
    const UMP2Result& ump2_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    
    // Dimensions
    int nbf_;
    int nocc_a_, nocc_b_;
    int nvir_a_, nvir_b_;
    
    // T2^(1) amplitudes (from MP2)
    Eigen::Tensor<double, 4> t2_aa_1_;
    Eigen::Tensor<double, 4> t2_bb_1_;
    Eigen::Tensor<double, 4> t2_ab_1_;
    
    // T2^(2) amplitudes (computed)
    Eigen::Tensor<double, 4> t2_aa_2_;
    Eigen::Tensor<double, 4> t2_bb_2_;
    Eigen::Tensor<double, 4> t2_ab_2_;
    
    // MO integrals (OOVV blocks for energy)
    Eigen::Tensor<double, 4> eri_oovv_aa_;
    Eigen::Tensor<double, 4> eri_oovv_bb_;
    Eigen::Tensor<double, 4> eri_oovv_ab_;
    
    // W-intermediates
    Eigen::Tensor<double, 4> W_oooo_aa_;  ///< <mn||ij> alpha
    Eigen::Tensor<double, 4> W_oooo_bb_;  ///< <mn||ij> beta
    Eigen::Tensor<double, 4> W_oooo_ab_;  ///< <mn|ij> mixed
    
    Eigen::Tensor<double, 4> W_ovov_aa_;  ///< <mb||ej> alpha
    Eigen::Tensor<double, 4> W_ovov_bb_;  ///< <mb||ej> beta
    
    Eigen::Tensor<double, 4> W_vvvv_aa_;  ///< <ab||ef> alpha
    Eigen::Tensor<double, 4> W_vvvv_bb_;  ///< <ab||ef> beta
    
    // Cached AO integrals
    Eigen::Tensor<double, 4> eri_ao_cached_;
    bool eri_ao_cached_valid_ = false;
    
    /**
     * @brief Transform OOVV integrals needed for energy
     */
    void transform_oovv_integrals();
    
    /**
     * @brief Get T2^(1) amplitudes from UMP2 result
     */
    void get_t2_1_from_ump2();
    
    /**
     * @brief Build W-intermediates (W_mnij, W_mbej, W_abef)
     * 
     * These are the key to correct MP3:
     * - W_mnij = <mn||ij> (hole-hole)
     * - W_mbej = <mb||ej> (particle-hole)
     * - W_abef = <ab||ef> (particle-particle)
     */
    void build_W_intermediates();
    
    /**
     * @brief Build W_mnij = <mn||ij> for alpha spin
     */
    void build_W_oooo_aa();
    void build_W_oooo_bb();
    void build_W_oooo_ab();
    
    /**
     * @brief Build W_mbej = <mb||ej> for alpha spin
     */
    void build_W_ovov_aa();
    void build_W_ovov_bb();
    
    /**
     * @brief Build W_abef = <ab||ef> for alpha spin
     */
    void build_W_vvvv_aa();
    void build_W_vvvv_bb();
    
    /**
     * @brief Compute T2^(2) amplitudes via W-intermediate contractions
     * 
     * Formula:
     *   T2^(2)_ijab = [Σ_mn W_mnij T2^(1)_mnab 
     *                + Σ_ef W_abef T2^(1)_ijef
     *                - Σ_me W_mbej T2^(1)_imae
     *                - Σ_me W_majb T2^(1)_imeb] / D_ijab
     */
    void compute_t2_2nd();
    
    /**
     * @brief Compute T2^(2) for alpha-alpha spin
     */
    void compute_t2_2nd_aa();
    void compute_t2_2nd_bb();
    void compute_t2_2nd_ab();
    
    /**
     * @brief Compute E(3) energy from T2^(2)
     * 
     * E(3) = Σ_ijab <ij||ab> T2^(2)_ijab
     */
    double compute_e3_aa();
    double compute_e3_bb();
    double compute_e3_ab();
};

} // namespace mshqc

#endif // MSHQC_UMP3_H
