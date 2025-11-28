#ifndef MSHQC_MPN_HIERARCHY_H
#define MSHQC_MPN_HIERARCHY_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <string>

/**
 * @file mpn_hierarchy.h
 * @brief Complete Møller-Plesset perturbation hierarchy (orders 0-5)
 * 
 * Provides unified structure for:
 * - Energies: E^(0) through E^(5)
 * - Wavefunctions: Ψ^(0) through Ψ^(4)
 * 
 * All formulas are EXACT from Rayleigh-Schrödinger perturbation theory.
 * No approximations except basis set truncation.
 * 
 * THEORY REFERENCES:
 * - Møller & Plesset (1934): Original MP theory
 * - Szabo & Ostlund (1996): Textbook derivation (Chapter 6)
 * - Helgaker et al. (2000): Complete theory (Chapter 14)
 * 
 * @author MSH-QC Project (Original Implementation)
 * @date 2025-01-16
 * @license MIT
 */

namespace mshqc {

/**
 * @brief Complete perturbation hierarchy result
 * 
 * Contains all orders of energy and wavefunction from
 * exact Møller-Plesset perturbation theory.
 */
struct MPnHierarchyResult {
    // ========================================================================
    // ENERGY HIERARCHY (Exact from Perturbation Theory)
    // ========================================================================
    
    /// E^(0): Hartree-Fock reference energy
    double e0_hf;
    
    /// E^(1): First-order energy = 0 (Brillouin's theorem)
    double e1;  // Always 0 for canonical HF
    
    /// E^(2): Second-order MP2 correlation energy (EXACT)
    double e2_mp2;
    double e2_aa;     ///< α-α spin contribution
    double e2_bb;     ///< β-β spin contribution  
    double e2_ab;     ///< α-β spin contribution
    
    /// E^(3): Third-order MP3 correlation energy (EXACT)
    double e3_mp3;
    double e3_aa;     ///< α-α contribution
    double e3_bb;     ///< β-β contribution
    double e3_ab;     ///< α-β contribution
    
    /// E^(4): Fourth-order MP4 correlation energy (EXACT)
    double e4_mp4;
    double e4_s;      ///< Singles contribution
    double e4_d;      ///< Doubles contribution
    double e4_t;      ///< Triples contribution
    double e4_q;      ///< Quadruples contribution
    
    /// E^(5): Fifth-order MP5 correlation energy (EXACT)
    double e5_mp5;
    double e5_t;      ///< Triples from t^(2)
    double e5_q;      ///< Quadruples mixed
    double e5_p;      ///< Pentuples
    
    // ========================================================================
    // CUMULATIVE ENERGIES
    // ========================================================================
    
    /// Total energy at each order
    double e_total_mp0;  ///< = E^(0)
    double e_total_mp1;  ///< = E^(0) + E^(1) = E^(0)
    double e_total_mp2;  ///< = E^(0) + E^(2)
    double e_total_mp3;  ///< = E^(0) + E^(2) + E^(3)
    double e_total_mp4;  ///< = E^(0) + E^(2) + E^(3) + E^(4)
    double e_total_mp5;  ///< = E^(0) + E^(2) + ... + E^(5)
    
    // ========================================================================
    // WAVEFUNCTION HIERARCHY (Exact Amplitudes)
    // ========================================================================
    
    /// Ψ^(0): HF determinant (reference)
    /// Represented implicitly by MO coefficients
    
    /// Ψ^(1): First-order wavefunction
    /// |Ψ^(1)⟩ = Σ t_ij^ab(1) |Ψ_ij^ab⟩
    Eigen::Tensor<double, 4> t2_aa_1;  ///< t^(1) α-α doubles
    Eigen::Tensor<double, 4> t2_bb_1;  ///< t^(1) β-β doubles
    Eigen::Tensor<double, 4> t2_ab_1;  ///< t^(1) α-β doubles
    
    /// Ψ^(2): Second-order wavefunction (NOT NEEDED FOR ENERGY)
    /// Intermediate, not stored
    
    /// Ψ^(3): Third-order wavefunction
    /// |Ψ^(3)⟩ = Σ t_i^a(2) |Ψ_i^a⟩ + Σ t_ij^ab(2) |Ψ_ij^ab⟩
    Eigen::Tensor<double, 2> t1_a_2;   ///< t^(2) α singles
    Eigen::Tensor<double, 2> t1_b_2;   ///< t^(2) β singles
    Eigen::Tensor<double, 4> t2_aa_2;  ///< t^(2) α-α doubles
    Eigen::Tensor<double, 4> t2_bb_2;  ///< t^(2) β-β doubles
    Eigen::Tensor<double, 4> t2_ab_2;  ///< t^(2) α-β doubles
    
    /// Ψ^(4): Fourth-order wavefunction
    /// |Ψ^(4)⟩ = Σ t_i^a(3) |Ψ_i^a⟩ + Σ t_ij^ab(3) |Ψ_ij^ab⟩ + Σ t_ijk^abc(2) |Ψ_ijk^abc⟩
    Eigen::Tensor<double, 2> t1_a_3;   ///< t^(3) α singles
    Eigen::Tensor<double, 2> t1_b_3;   ///< t^(3) β singles
    Eigen::Tensor<double, 4> t2_aa_3;  ///< t^(3) α-α doubles
    Eigen::Tensor<double, 4> t2_bb_3;  ///< t^(3) β-β doubles
    Eigen::Tensor<double, 4> t2_ab_3;  ///< t^(3) α-β doubles
    Eigen::Tensor<double, 6> t3_aaa_2; ///< t^(2) α-α-α triples
    Eigen::Tensor<double, 6> t3_bbb_2; ///< t^(2) β-β-β triples
    Eigen::Tensor<double, 6> t3_aab_2; ///< t^(2) α-α-β triples
    Eigen::Tensor<double, 6> t3_abb_2; ///< t^(2) α-β-β triples
    
    // ========================================================================
    // ORBITAL INFORMATION
    // ========================================================================
    
    int n_occ_alpha;   ///< Number of occupied α orbitals
    int n_occ_beta;    ///< Number of occupied β orbitals
    int n_virt_alpha;  ///< Number of virtual α orbitals
    int n_virt_beta;   ///< Number of virtual β orbitals
    int n_basis;       ///< Total basis functions
    
    // ========================================================================
    // WAVEFUNCTION NORMS (Diagnostics)
    // ========================================================================
    
    double norm_t2_1;   ///< ||T2^(1)|| (first-order doubles)
    double norm_t1_2;   ///< ||T1^(2)|| (second-order singles)
    double norm_t2_2;   ///< ||T2^(2)|| (second-order doubles)
    double norm_t3_2;   ///< ||T3^(2)|| (second-order triples)
    double norm_t1_3;   ///< ||T1^(3)|| (third-order singles)
    double norm_t2_3;   ///< ||T2^(3)|| (third-order doubles)
    
    // ========================================================================
    // METADATA
    // ========================================================================
    
    bool mp2_computed;  ///< E^(2) available?
    bool mp3_computed;  ///< E^(3) available?
    bool mp4_computed;  ///< E^(4) available?
    bool mp5_computed;  ///< E^(5) available?
    
    bool psi1_computed; ///< Ψ^(1) available?
    bool psi3_computed; ///< Ψ^(3) available?
    bool psi4_computed; ///< Ψ^(4) available?
    
    std::string basis_name;  ///< Basis set used
    std::string molecule;    ///< Molecular formula
    
    /**
     * @brief Print complete energy hierarchy table
     * 
     * Shows convergence of perturbation series:
     * Order | Energy        | Contribution | Cumulative
     * ------|---------------|--------------|------------
     * E^(0) | ...           | ...          | ...
     * E^(1) | 0.0           | 0.0          | ...
     * E^(2) | ...           | ...          | ...
     * etc.
     */
    void print_energy_table() const;
    
    /**
     * @brief Print wavefunction components summary
     * 
     * Shows which amplitudes are available:
     * Ψ^(n) | Singles | Doubles | Triples | Available?
     * ------|---------|---------|---------|------------
     * etc.
     */
    void print_wavefunction_summary() const;
    
    /**
     * @brief Print complete hierarchy summary
     * 
     * Combined energy table + wavefunction summary + convergence analysis
     */
    void print() const;
};

/**
 * @brief Build complete MPn hierarchy from individual results
 * 
 * Combines UHF, UMP2, UMP3, UMP4, UMP5 results into unified structure.
 * 
 * @param uhf_result HF reference
 * @param ump2_result MP2 result (required)
 * @param ump3_result MP3 result (optional, nullptr if not computed)
 * @param ump4_result MP4 result (optional, nullptr if not computed)
 * @param ump5_result MP5 result (optional, nullptr if not computed)
 * @return Complete hierarchy
 */
MPnHierarchyResult build_mpn_hierarchy(
    const struct SCFResult& uhf_result,
    const struct UMP2Result& ump2_result,
    const struct UMP3Result* ump3_result = nullptr,
    const void* ump4_result = nullptr,  // UMP4Result when implemented
    const void* ump5_result = nullptr   // UMP5Result when implemented
);

} // namespace mshqc

#endif // MSHQC_MPN_HIERARCHY_H
