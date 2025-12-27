/**
 * @file cholesky_ump2.h
 * @brief UMP2 with Cholesky-decomposed ERIs
 * 
 * THEORY:
 * Standard UMP2 but uses Cholesky vectors L^K to avoid storing full (ia|jb):
 * (ia|jb) ≈ Σ_K L^K_ia L^K_jb
 * 
 * Memory: O(N²M) vs O(N⁴) for full integrals
 * Speed: Comparable or faster due to better cache usage
 * 
 * REFERENCES:
 * - Møller & Plesset (1934), Phys. Rev. 46, 618 [MP2 theory]
 * - Pople et al. (1976), Int. J. Quantum Chem. 10, 1 [UMP2]
 * - Beebe & Linderberg (1977), Int. J. Quantum Chem. 12, 683 [Cholesky ERI]
 * - Koch et al. (2003), J. Chem. Phys. 118, 9481 [Modern Cholesky]
 * - Aquilante et al. (2008), J. Chem. Phys. 129, 024113 [Cholesky MP2]
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-12-11
 * @license MIT
 */

#ifndef MSHQC_CHOLESKY_UMP2_H
#define MSHQC_CHOLESKY_UMP2_H

#include "mshqc/scf.h"
#include "mshqc/integrals/cholesky_eri.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

namespace mshqc {

// Forward declarations
struct CholeskyUMP2Result;

/**
 * @brief Configuration for Cholesky-UMP2
 */
struct CholeskyUMP2Config {
    double cholesky_threshold = 1e-6;  // ERI decomposition accuracy
    bool use_on_the_fly = false;       // Compute L vectors on-the-fly
    bool validate_energy = false;      // Compare with full ERI result
    int print_level = 1;               // 0=minimal, 1=normal, 2=verbose
};

/**
 * @brief Results from Cholesky-UMP2 calculation
 */
struct CholeskyUMP2Result {
    // Energy components
    double e_corr_ss_aa = 0.0;   // Same-spin αα
    double e_corr_ss_bb = 0.0;   // Same-spin ββ
    double e_corr_os = 0.0;      // Opposite-spin αβ
    double e_corr_total = 0.0;   // Total correlation
    double e_total = 0.0;        // Total energy (SCF + corr)
    
    // Diagnostics
    int n_cholesky_vectors = 0;
    double compression_ratio = 0.0;
    double memory_mb = 0.0;
    
    // Timing (optional)
    double time_cholesky_s = 0.0;
    double time_transform_s = 0.0;
    double time_energy_s = 0.0;
};

/**
 * @brief UMP2 using Cholesky-decomposed ERIs
 * 
 * Main advantage: Memory O(N²M) instead of O(N⁴)
 * 
 * REFERENCE: Aquilante et al. (2008), J. Chem. Phys. 129, 024113
 * Efficient MP2 energy via Cholesky decomposition
 */
class CholeskyUMP2 {
    // Allow CholeskyUMP3 to access private members for efficiency
    friend class CholeskyUMP3;

    
public:
    /**
     * @brief Construct Cholesky-UMP2 solver
     * @param uhf_result UHF SCF results (orbitals, energies, etc)
     * @param basis Basis set
     * @param integrals Integral engine
     * @param config Configuration parameters
     */
    CholeskyUMP2(const SCFResult& uhf_result,
                 const BasisSet& basis,
                 std::shared_ptr<IntegralEngine> integrals,
                 const CholeskyUMP2Config& config,
                 const integrals::CholeskyERI& cholesky_from_uhf 
                );
                 
    
    ~CholeskyUMP2() = default;
    
    /**
     * @brief Compute UMP2 energy using Cholesky vectors
     * @return MP2 results with energy breakdown
     */
    CholeskyUMP2Result compute();
    
    /**
     * @brief Get T2 amplitudes (if stored)
     * NOTE: Optional - can be expensive for large systems
     */
    void compute_t2_amplitudes();
    
    /**
     * @brief Get Cholesky decomposition object
     */
    const integrals::CholeskyERI& get_cholesky() const {
        return *cholesky_;
    }
    
private:
    // System info
    const SCFResult& uhf_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    CholeskyUMP2Config config_;
    
    // Dimensions
    int nbf_;       // basis functions
    int nocc_a_;    // occupied α
    int nocc_b_;    // occupied β
    int nvir_a_;    // virtual α
    int nvir_b_;    // virtual β
    
    // Cholesky decomposition
    std::unique_ptr<integrals::CholeskyERI> cholesky_;
    int n_cholesky_ = 0;
    
    // Transformed Cholesky vectors in MO basis
    // L^K_ia where i=occ, a=virt
    std::vector<Eigen::MatrixXd> L_ia_alpha_;  // [K](i,a)
    std::vector<Eigen::MatrixXd> L_ia_beta_;   // [K](i,a)
    
    // Optional: T2 amplitudes (memory-intensive)
    Eigen::Tensor<double, 4> t2_aa_;  // (i,j,a,b) α-α
    Eigen::Tensor<double, 4> t2_bb_;  // (i,j,a,b) β-β
    Eigen::Tensor<double, 4> t2_ab_;  // (i,j,a,b) α-β
    
    /**
     * @brief Decompose AO-basis ERIs into Cholesky vectors
     * 
     * REFERENCE: Koch et al. (2003), J. Chem. Phys. 118, 9481
     * Modified Cholesky decomposition of ERI matrix
     */
    void decompose_eri();
    
    /**
     * @brief Transform Cholesky vectors from AO to MO basis
     * 
     * REFERENCE: Aquilante et al. (2008), Eq. (12)
     * L^K_ia = Σ_μν L^K_μν C_μi C_νa
     * 
     * This is O(N³M) operation (much better than O(N⁵) full transform)
     */
    void transform_cholesky_vectors();
    
    /**
     * @brief Compute same-spin αα contribution
     * 
     * REFERENCE: Szabo & Ostlund (1996), Eq. (6.74)
     * E_ss^αα = (1/4) Σ_ijab |<ij||ab>|² / Δ_ijab
     * 
     * Using Cholesky:
     * <ij||ab> = <ij|ab> - <ij|ba>
     * <ij|ab> = Σ_K L^K_ia L^K_jb
     */
    double compute_ss_alpha();
    
    /**
     * @brief Compute same-spin ββ contribution
     */
    double compute_ss_beta();
    
    /**
     * @brief Compute opposite-spin αβ contribution
     * 
     * REFERENCE: Szabo & Ostlund (1996), Eq. (6.73)
     * E_os = Σ_ijab <ij|ab>² / Δ_ijab
     * 
     * Using Cholesky:
     * <ij|ab> = Σ_K L^K_ia L^K_jb (no antisymmetrization)
     */
    double compute_os();
    
    /**
     * @brief Reconstruct integral <ij|ab> from Cholesky vectors
     * For validation/debugging
     */
    double reconstruct_integral(int i, int j, int a, int b, bool alpha_i, bool alpha_j);
    
    /**
     * @brief Print timing and memory statistics
     */
    void print_statistics(const CholeskyUMP2Result& result) const;
};

} // namespace mshqc

#endif // MSHQC_CHOLESKY_UMP2_H