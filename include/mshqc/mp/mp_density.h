/**
 * @file mp_density.h
 * @brief Density matrix analysis for Møller-Plesset wavefunctions
 * 
 * Theory References:
 * - 1-RDM from PT: R. J. Bartlett, Ann. Rev. Phys. Chem. **32**, 359 (1981)
 * - Natural Orbitals: P.-O. Löwdin, Phys. Rev. **97**, 1474 (1955)
 * - MP Wavefunction: J. A. Pople et al., Int. J. Quantum Chem. **14**, 545 (1978)
 * - OPDM Theory: T. Helgaker et al., "Molecular Electronic Structure Theory" (2000), Ch. 10
 * - Response Theory: J. Olsen & P. Jørgensen, J. Chem. Phys. **82**, 3235 (1985)
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 * 
 * @note Original implementation from theory papers. The one-particle density matrix
 *       (1-RDM) for MP wavefunctions is computed from cluster amplitudes T1, T2, T3.
 *       Natural orbitals are eigenfunctions of 1-RDM and provide insight into
 *       electron correlation and multi-reference character.
 * 
 * @copyright MIT License
 */

#ifndef MSHQC_MP_DENSITY_H
#define MSHQC_MP_DENSITY_H

#include "mshqc/ump3.h"
#include "mshqc/mp/ump4.h"
#include "mshqc/mp/ump5.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

namespace mshqc {
namespace mp {

/**
 * @brief Natural orbital analysis result for MP wavefunctions
 * 
 * Theory: Natural orbitals are eigenfunctions of 1-RDM
 *         γ = Σ_i n_i |φ_i⟩⟨φ_i|
 * 
 * Reference: P.-O. Löwdin, Phys. Rev. 97, 1474 (1955)
 */
struct MPNaturalOrbitalResult {
    // Natural orbital occupations
    Eigen::VectorXd occupations;        ///< n_i (0 ≤ n_i ≤ 2)
    
    // Natural orbital coefficients
    Eigen::MatrixXd orbitals;           ///< Natural orbitals (MO basis)
    
    // Correlation measures
    double total_correlation;           ///< Σ(1-n_occ) + Σ(n_virt)
    double singles_correlation;         ///< From T1 amplitudes
    double doubles_correlation;         ///< From T2 amplitudes
    double triples_correlation;         ///< From T3 amplitudes (if available)
    
    // Diagnostic numbers
    double t1_diagnostic;               ///< ||T1|| / √(n_occ)
    double t2_diagnostic;               ///< ||T2|| / √(n_occ²)
    double largest_t1_amplitude;        ///< max|t_i^a|
    double largest_t2_amplitude;        ///< max|t_ij^ab|
    
    // Multi-reference character
    bool is_multi_reference;            ///< True if strong correlation detected
    double multi_ref_score;             ///< 0=single-ref, 1=multi-ref
    std::string multi_ref_reason;       ///< Why flagged as multi-ref
    
    // System info
    int n_electrons;
    int n_orbitals;
    std::string level;                  ///< "MP2", "MP3", "MP4", "MP5"
};

/**
 * @brief One-Particle Density Matrix (1-RDM) for MP wavefunctions
 * 
 * Theory: The 1-RDM for MP wavefunctions includes contributions from:
 *         γ_pq = ⟨Ψ_MP|p†q|Ψ_MP⟩
 *              = γ_pq^(0) + γ_pq^(1) + γ_pq^(2) + ...
 * 
 * where:
 *   γ^(0) = HF density
 *   γ^(1) = T1 contribution
 *   γ^(2) = T2 contribution + T1×T1
 *   γ^(3) = T3 contribution + T1×T2 + ...
 * 
 * Reference: Bartlett, Ann. Rev. Phys. Chem. 32, 359 (1981)
 */
class MPDensityMatrix {
public:
    /**
     * @brief Compute 1-RDM from UMP2 wavefunction
     * 
     * Theory: γ = γ^(HF) + γ^(T1) + γ^(T2)
     * 
     * For UMP2 (no T1):
     *   γ_pq = δ_pq × n_p^(HF) + Σ_ijab t_ij^ab × ⟨0|p†q|ijab⟩
     * 
     * @param ump2_result UMP2 computation result
     * @param uhf_result UHF reference
     * @return 1-RDM matrix (n_orb × n_orb)
     */
    static Eigen::MatrixXd compute_opdm_mp2(
        const UMP2Result& ump2_result,
        const SCFResult& uhf_result
    );
    
    /**
     * @brief Compute 1-RDM from UMP3 wavefunction
     * 
     * Theory: γ = γ^(HF) + γ^(T1) + γ^(T2^(1)) + γ^(T2^(2)) + γ^(T3^(2))
     * 
     * Includes:
     *   - HF reference density
     *   - T1^(2) singles (2nd order)
     *   - T2^(1) doubles (1st order, from MP2)
     *   - T2^(2) doubles (2nd order)
     *   - T3^(2) triples (2nd order, if computed)
     * 
     * @param ump3_result UMP3 computation result
     * @param uhf_result UHF reference
     * @return 1-RDM matrix
     */
    static Eigen::MatrixXd compute_opdm_mp3(
        const UMP3Result& ump3_result,
        const SCFResult& uhf_result
    );
    
    /**
     * @brief Compute natural orbitals from 1-RDM
     * 
     * Theory: Diagonalize 1-RDM
     *         γ|φ_i⟩ = n_i|φ_i⟩
     * 
     * Properties:
     *   - 0 ≤ n_i ≤ 2 (occupation per spatial orbital)
     *   - Σ n_i = N_electrons (sum rule)
     *   - Occupied: n_i ≈ 2.0
     *   - Virtual: n_i ≈ 0.0
     *   - Partially occupied: 0 < n_i < 2 (correlation!)
     * 
     * @param opdm 1-RDM matrix
     * @param level "MP2", "MP3", etc.
     * @return Natural orbital analysis result
     * 
     * Reference: Löwdin, Phys. Rev. 97, 1474 (1955)
     */
    static MPNaturalOrbitalResult compute_natural_orbitals(
        const Eigen::MatrixXd& opdm,
        const std::string& level,
        int n_electrons,
        int n_orbitals
    );
    
    /**
     * @brief Analyze MP wavefunction quality
     * 
     * Computes diagnostic measures:
     *   - T1 diagnostic: ||T1|| / √(n_occ)
     *   - T2 diagnostic: ||T2|| / √(n_occ²)
     *   - Multi-reference character from natural orbital occupations
     * 
     * Thresholds (empirical):
     *   - T1 < 0.02: single-reference
     *   - 0.02 < T1 < 0.05: weakly multi-reference
     *   - T1 > 0.05: strongly multi-reference
     * 
     * @param ump3_result UMP3 result with amplitudes
     * @param no_result Natural orbital analysis
     * @return Updated NO result with diagnostics
     * 
     * Reference: Lee & Taylor, Int. J. Quantum Chem. Symp. 23, 199 (1989)
     */
    static MPNaturalOrbitalResult analyze_wavefunction(
        const UMP3Result& ump3_result,
        MPNaturalOrbitalResult no_result
    );
    
    /**
     * @brief Print natural orbital analysis report
     * 
     * Output includes:
     *   - Natural orbital occupations (sorted)
     *   - Correlation measures
     *   - Diagnostic numbers (T1, T2)
     *   - Multi-reference assessment
     * 
     * @param result Natural orbital analysis result
     * @param verbose Print all orbitals if true
     */
    static void print_report(
        const MPNaturalOrbitalResult& result,
        bool verbose = false
    );
    
    /**
     * @brief Compare MP natural orbitals with CI natural orbitals
     * 
     * Useful for validation:
     *   - MP and CI should give similar NOs
     *   - Differences indicate MP approximation error
     * 
     * @param mp_no MP natural orbital result
     * @param ci_no CI natural orbital result (from FCI/CISD)
     * @return RMS difference in occupations
     */
    static double compare_with_ci(
        const MPNaturalOrbitalResult& mp_no,
        const MPNaturalOrbitalResult& ci_no
    );

private:
    // Helper: Compute HF contribution to 1-RDM
    static Eigen::MatrixXd compute_hf_density(
        const SCFResult& uhf_result,
        int n_orb,
        int n_alpha,
        int n_beta
    );
    
    // Helper: Compute T1 contribution to 1-RDM
    // γ_pq += Σ_i t_i^p t_i^q (particle-particle block)
    static void add_t1_contribution(
        Eigen::MatrixXd& opdm,
        const Eigen::Tensor<double, 2>& t1_alpha,
        const Eigen::Tensor<double, 2>& t1_beta,
        int n_occ_alpha,
        int n_occ_beta
    );
    
    // Helper: Compute T2 contribution to 1-RDM
    // γ_pq += Σ_ijab t_ij^ap t_ij^bq (complex contractions)
    static void add_t2_contribution(
        Eigen::MatrixXd& opdm,
        const Eigen::Tensor<double, 4>& t2_aa,
        const Eigen::Tensor<double, 4>& t2_bb,
        const Eigen::Tensor<double, 4>& t2_ab,
        int n_occ_alpha,
        int n_occ_beta,
        int n_virt_alpha,
        int n_virt_beta
    );
    
    // Helper: Compute T3 contribution (if available)
    static void add_t3_contribution(
        Eigen::MatrixXd& opdm,
        const Eigen::Tensor<double, 6>& t3_aaa,
        const Eigen::Tensor<double, 6>& t3_bbb,
        int n_occ_alpha,
        int n_occ_beta
    );
    
    // Helper: Check multi-reference character
    static bool check_multi_reference(
        const Eigen::VectorXd& occupations,
        int n_electrons,
        double& score,
        std::string& reason
    );
    
    // Helper: Compute correlation energy from natural orbital occupations
    static double compute_correlation_from_nos(
        const Eigen::VectorXd& occupations,
        int n_electrons
    );
};

} // namespace mp
} // namespace mshqc

#endif // MSHQC_MP_DENSITY_H
