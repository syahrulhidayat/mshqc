/**
 * @file mp_vs_fci.h
 * @brief Validation of Møller-Plesset perturbation theory against Full CI
 * 
 * Theory References:
 * - Møller-Plesset Theory: C. Møller & M. S. Plesset, Phys. Rev. **46**, 618 (1934)
 * - UMP Theory: J. A. Pople et al., J. Chem. Phys. **64**, 2901 (1976)
 * - FCI Benchmark: P. J. Knowles & N. C. Handy, Chem. Phys. Lett. **111**, 315 (1984)
 * - Convergence Analysis: J. A. Pople et al., Int. J. Quantum Chem. **14**, 545 (1978)
 * - Perturbation Series: R. J. Bartlett & M. Musiał, Rev. Mod. Phys. **79**, 291 (2007)
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 * 
 * @note Original implementation from theory papers. Full Configuration Interaction
 *       (FCI) provides exact solution within a given basis set, making it the
 *       gold standard for validating approximate methods like Møller-Plesset
 *       perturbation theory.
 * 
 * @copyright MIT License
 */

#ifndef MSHQC_VALIDATION_MP_VS_FCI_H
#define MSHQC_VALIDATION_MP_VS_FCI_H

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

namespace mshqc {
namespace validation {

/**
 * @brief Report structure for MP series convergence analysis
 * 
 * Theory: Perturbation theory convergence requires |E^(n+1)| < |E^(n)|
 * Reference: J. A. Pople et al., Int. J. Quantum Chem. **14**, 545 (1978)
 */
struct MPConvergenceReport {
    // Reference energies
    double e_hf;            ///< Hartree-Fock energy (zeroth-order)
    double e_fci;           ///< Full CI energy (exact in basis)
    
    // MP series energies
    double e_mp2;           ///< MP2 total energy
    double e_mp3;           ///< MP3 total energy
    double e_mp4;           ///< MP4 total energy
    double e_mp5;           ///< MP5 total energy
    
    // Correlation energies (E^(n))
    double ec_mp2;          ///< E^(2) correlation
    double ec_mp3;          ///< E^(2) + E^(3)
    double ec_mp4;          ///< E^(2) + E^(3) + E^(4)
    double ec_mp5;          ///< E^(2) + ... + E^(5)
    double ec_fci;          ///< Exact correlation (FCI)
    
    // Errors vs FCI (in Hartree)
    double error_mp2;       ///< |E_MP2 - E_FCI|
    double error_mp3;       ///< |E_MP3 - E_FCI|
    double error_mp4;       ///< |E_MP4 - E_FCI|
    double error_mp5;       ///< |E_MP5 - E_FCI|
    
    // Convergence ratios
    double ratio_32;        ///< |E^(3)|/|E^(2)|
    double ratio_43;        ///< |E^(4)|/|E^(3)|
    double ratio_54;        ///< |E^(5)|/|E^(4)|
    
    // Convergence diagnostics
    bool is_converging;     ///< True if ratios decrease
    bool mp5_converged;     ///< True if error_mp5 < threshold
    double convergence_rate;///< Estimated convergence rate
    
    // Correlation recovery (percentage)
    double corr_mp2_pct;    ///< % correlation recovered by MP2
    double corr_mp3_pct;    ///< % correlation recovered by MP3
    double corr_mp4_pct;    ///< % correlation recovered by MP4
    double corr_mp5_pct;    ///< % correlation recovered by MP5
    
    // System info
    std::string system_name;
    int n_electrons;
    int n_basis;
    std::string basis_name;
};

/**
 * @brief Validator class for MP hierarchy vs FCI benchmark
 * 
 * Theory: FCI is exact within basis set, so |E_MP - E_FCI| measures
 *         approximation error in perturbation theory.
 * 
 * Reference: P. J. Knowles & N. C. Handy, Chem. Phys. Lett. **111**, 315 (1984)
 */
class MPFCIValidator {
public:
    /**
     * @brief Validate MP series convergence against FCI
     * 
     * Theory: For well-behaved single-reference systems, MP series
     *         should converge monotonically to FCI energy.
     * 
     * @param e_hf Hartree-Fock reference energy
     * @param e_mp2 MP2 total energy
     * @param e_mp3 MP3 total energy  
     * @param e_mp4 MP4 total energy
     * @param e_mp5 MP5 total energy
     * @param e_fci FCI exact energy (gold standard)
     * @param system_name Molecule/atom name (e.g., "Li", "Be")
     * @param n_electrons Number of electrons
     * @param n_basis Number of basis functions
     * @param basis_name Basis set name (e.g., "cc-pVDZ")
     * @param threshold Convergence threshold in Hartree (default: 10 µHa)
     * 
     * @return MPConvergenceReport with detailed analysis
     * 
     * @note For open-shell systems, MP3 can overshoot (Pople 1977)
     */
    static MPConvergenceReport validate(
        double e_hf,
        double e_mp2,
        double e_mp3,
        double e_mp4,
        double e_mp5,
        double e_fci,
        const std::string& system_name = "Unknown",
        int n_electrons = 0,
        int n_basis = 0,
        const std::string& basis_name = "Unknown",
        double threshold = 10.0e-6  // 10 microHartree
    );
    
    /**
     * @brief Print detailed convergence report
     * 
     * Output includes:
     * - Energy table (HF, MP2-5, FCI)
     * - Error analysis vs FCI
     * - Convergence ratios
     * - Correlation recovery percentages
     * - Diagnostic warnings
     * 
     * @param report Validation report to print
     * @param verbose If true, print extended diagnostics
     */
    static void print_report(
        const MPConvergenceReport& report,
        bool verbose = true
    );
    
    /**
     * @brief Check if MP series is converging properly
     * 
     * Theory: Convergent series requires |E^(n+1)| < |E^(n)|
     * 
     * @param report Validation report
     * @return True if series converges
     */
    static bool is_series_converging(const MPConvergenceReport& report);
    
    /**
     * @brief Estimate convergence rate from ratios
     * 
     * Theory: λ = |E^(n+1)|/|E^(n)| estimates convergence
     *         If λ < 1: converging
     *         If λ > 1: diverging
     * 
     * @param report Validation report
     * @return Estimated convergence rate λ
     */
    static double estimate_convergence_rate(const MPConvergenceReport& report);
    
    /**
     * @brief Predict MP6 error from observed convergence
     * 
     * Theory: If λ is constant, E^(6) ≈ E^(5) × λ
     * 
     * @param report Validation report
     * @return Predicted |E_MP6 - E_FCI| error
     */
    static double predict_mp6_error(const MPConvergenceReport& report);
    
    /**
     * @brief Generate LaTeX table for publication
     * 
     * @param report Validation report
     * @return LaTeX tabular code
     */
    static std::string to_latex_table(const MPConvergenceReport& report);
    
    /**
     * @brief Export report to JSON format
     * 
     * @param report Validation report
     * @return JSON string
     */
    static std::string to_json(const MPConvergenceReport& report);
    
private:
    // Helper functions
    static double compute_correlation_energy(double e_total, double e_hf);
    static double compute_percentage(double partial, double total);
    static double safe_ratio(double numerator, double denominator);
};

} // namespace validation
} // namespace mshqc

#endif // MSHQC_VALIDATION_MP_VS_FCI_H
