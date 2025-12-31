/**
 * @file cholesky_ump3.h
 * @brief UMP3 with Cholesky-decomposed ERIs
 * 
 * THEORY:
 * Third-order Møller-Plesset perturbation theory using Cholesky vectors
 * to reduce memory from O(N⁴) to O(N²M) where M ≈ N to 2N typically.
 * 
 * UMP3 energy has two contributions:
 * 1. Second-order amplitudes T2^(1) (from UMP2)
 * 2. Third-order correction involving T2^(2) and W intermediates
 * 
 * REFERENCES:
 * - Pople, Binkley, & Seeger (1976), Int. J. Quantum Chem. 10, 1
 *   [Original MP3 formulation]
 * - Knowles et al. (1985), Chem. Phys. Lett. 113, 8
 *   [UMP3 for open-shell systems]
 * - Bartlett & Silver (1975), J. Chem. Phys. 62, 3258
 *   [Third-order perturbation theory]
 * - Aquilante et al. (2008), J. Chem. Phys. 129, 024113
 *   [Cholesky decomposition for correlation methods]
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-12-11
 * @license MIT
 */

#ifndef MSHQC_CHOLESKY_UMP3_H
#define MSHQC_CHOLESKY_UMP3_H

#include "mshqc/scf.h"
#include "mshqc/cholesky_ump2.h"
#include "mshqc/integrals/cholesky_eri.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

namespace mshqc {

// Forward declarations
struct CholeskyUMP3Result;

/**
 * @brief Configuration for Cholesky-UMP3
 */
struct CholeskyUMP3Config {
    double cholesky_threshold = 1e-6;
    bool use_ump2_result = true;      // Reuse UMP2 T2 if available
    bool store_intermediates = false; // Store W matrices (memory intensive)
    bool validate_energy = false;
    int print_level = 1;
};

/**
 * @brief Results from Cholesky-UMP3 calculation
 */
struct CholeskyUMP3Result {
    // UMP2 components (for reference)
    double e_mp2_ss_aa = 0.0;
    double e_mp2_ss_bb = 0.0;
    double e_mp2_os = 0.0;
    double e_mp2_total = 0.0;
    
    // UMP3 correction components
    double e_mp3_ss_aa = 0.0;
    double e_mp3_ss_bb = 0.0;
    double e_mp3_os = 0.0;
    double e_mp3_total = 0.0;
    
    // Total energies
    double e_corr_total = 0.0;  // MP2 + MP3
    double e_total = 0.0;       // SCF + correlation
    
    // Diagnostics
    int n_cholesky_vectors = 0;
    double compression_ratio = 0.0;
    double memory_mb = 0.0;
    
    // Timing
    double time_mp2_s = 0.0;
    double time_mp3_s = 0.0;
    double time_total_s = 0.0;
};

/**
 * @brief W intermediate matrices for UMP3
 * 
 * REFERENCE: Bartlett & Silver (1975), J. Chem. Phys. 62, 3258
 * W matrices encode higher-order correlation effects
 */
struct WIntermediates {
    // W_mnij: Occupied-occupied block
    Eigen::Tensor<double, 4> W_mnij_aa;  // (m,n,i,j) all alpha
    Eigen::Tensor<double, 4> W_mnij_bb;  // (m,n,i,j) all beta
    Eigen::Tensor<double, 4> W_mnij_ab;  // (m,n,i,j) mixed
    
    // W_abef: Virtual-virtual block
    Eigen::Tensor<double, 4> W_abef_aa;
    Eigen::Tensor<double, 4> W_abef_bb;
    Eigen::Tensor<double, 4> W_abef_ab;
    
    // W_mbej: Mixed occupied-virtual
    Eigen::Tensor<double, 4> W_mbej_aa;
    Eigen::Tensor<double, 4> W_mbej_bb;
    Eigen::Tensor<double, 4> W_mbej_ab;
};

/**
 * @brief UMP3 using Cholesky-decomposed ERIs
 * 
 * Computes third-order correlation energy efficiently using Cholesky vectors.
 * Builds upon UMP2 by computing T2^(2) amplitudes and W intermediates.
 * 
 * REFERENCE: Knowles et al. (1985), Chem. Phys. Lett. 113, 8
 * UMP3 energy expressions for unrestricted reference
 */
class CholeskyUMP3 {
public:
    /**
     * @brief Construct Cholesky-UMP3 solver
     * @param uhf_result UHF SCF results
     * @param basis Basis set
     * @param integrals Integral engine
     * @param config Configuration parameters
     */
    CholeskyUMP3(const SCFResult& uhf_result,
                 const BasisSet& basis,
                 std::shared_ptr<IntegralEngine> integrals,
                 const CholeskyUMP3Config& config = CholeskyUMP3Config());
    
    /**
     * @brief Construct from existing UMP2 calculation (efficient)
     */
    CholeskyUMP3(const CholeskyUMP2& ump2_solver,
                 const CholeskyUMP3Config& config = CholeskyUMP3Config());
    
    ~CholeskyUMP3() = default;
    
    /**
     * @brief Compute UMP3 energy
     * @return Complete MP2 and MP3 results
     */
    CholeskyUMP3Result compute();
    
    /**
     * @brief Get W intermediates (if stored)
     */
    const WIntermediates& get_intermediates() const { return W_; }
    void initialize_cholesky();
    void transform_cholesky_vectors();
private:
    // System
    const SCFResult& uhf_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    CholeskyUMP3Config config_;
    
    // Dimensions
    int nbf_, nocc_a_, nocc_b_, nvir_a_, nvir_b_;
    
    // Cholesky decomposition
    std::unique_ptr<integrals::CholeskyERI> cholesky_;
    int n_cholesky_ = 0;
    
    // Transformed Cholesky vectors (from UMP2 or computed here)
    std::vector<Eigen::MatrixXd> L_ia_alpha_;
    std::vector<Eigen::MatrixXd> L_ia_beta_;
    std::vector<Eigen::MatrixXd> L_ij_alpha_;  // occ-occ
    std::vector<Eigen::MatrixXd> L_ij_beta_;
    std::vector<Eigen::MatrixXd> L_ab_alpha_;  // virt-virt
    std::vector<Eigen::MatrixXd> L_ab_beta_;
    
    // T2 amplitudes from MP2
    Eigen::Tensor<double, 4> t2_aa_;
    Eigen::Tensor<double, 4> t2_bb_;
    Eigen::Tensor<double, 4> t2_ab_;
    
    // W intermediates
    WIntermediates W_;
    
    // UMP2 result (if reusing)
    std::unique_ptr<CholeskyUMP2Result> mp2_result_;
    
    /**
     * @brief Setup: decompose and transform Cholesky vectors
     */
    
    
    /**
     * @brief Transform Cholesky vectors to all needed MO blocks
     * 
     * REFERENCE: Aquilante et al. (2008), Eq. (12)
     * Need: L_ia (occ-virt), L_ij (occ-occ), L_ab (virt-virt)
     */
    
    
    /**
     * @brief Compute T2^(1) amplitudes (MP2 level)
     * 
     * REFERENCE: Szabo & Ostlund (1996), Eq. (6.63)
     * T2^(1)_ijab = <ij||ab> / Δ_ijab
     */
    void compute_t2_first_order();
    

    /**
     * @brief Helper to reconstruct integral block (d1*d2) x (d3*d4) using DGEMM
     */
    Eigen::MatrixXd reconstruct_block_mat(
        const std::vector<Eigen::MatrixXd>& L1, 
        const std::vector<Eigen::MatrixXd>& L2, 
        int d1, int d2, int d3, int d4
    );
    /**
     * @brief Build W intermediates for MP3
     * 
     * REFERENCE: Bartlett & Silver (1975), Eq. (15)-(17)
     * W_mnij, W_abef, W_mbej encode correlation structure
     */
    void build_W_intermediates();
    
    /**
     * @brief Compute MP3 same-spin αα correction
     * 
     * REFERENCE: Knowles et al. (1985), Eq. (10)
     * E3_ss^αα = Σ T2^(1) * W * T2^(1) / Δ
     */
    double compute_mp3_ss_alpha();
    
    /**
     * @brief Compute MP3 same-spin ββ correction
     */
    double compute_mp3_ss_beta();
    
    /**
     * @brief Compute MP3 opposite-spin correction
     * 
     * REFERENCE: Knowles et al. (1985), Eq. (11)
     */
    double compute_mp3_os();
    
    /**
     * @brief Reconstruct integral using Cholesky vectors
     * Generic version for any orbital pair types
     */
    double reconstruct_integral(
        int p, int q, int r, int s,
        const std::vector<Eigen::MatrixXd>& L_pq,
        const std::vector<Eigen::MatrixXd>& L_rs
    );
    
    /**
     * @brief Print detailed statistics
     */
    void print_statistics(const CholeskyUMP3Result& result) const;
};

} // namespace mshqc

#endif // MSHQC_CHOLESKY_UMP3_H