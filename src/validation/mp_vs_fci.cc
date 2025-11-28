/**
 * @file mp_vs_fci.cc
 * @brief Implementation of MP vs FCI validation pipeline
 * 
 * Theory References:
 * - Møller-Plesset Theory: C. Møller & M. S. Plesset, Phys. Rev. **46**, 618 (1934)
 * - Convergence Analysis: J. A. Pople et al., Int. J. Quantum Chem. **14**, 545 (1978)
 * - FCI Theory: P. J. Knowles & N. C. Handy, Chem. Phys. Lett. **111**, 315 (1984)
 * - Perturbation Theory Review: R. J. Bartlett & M. Musiał, Rev. Mod. Phys. **79**, 291 (2007)
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 * 
 * @note Original implementation. This validation framework uses FCI (exact within
 *       basis set) as gold standard to assess MP series convergence. For single-
 *       reference systems, MP2-5 should converge monotonically to FCI energy.
 * 
 * @copyright MIT License
 */

#include "mshqc/validation/mp_vs_fci.h"
#include <cmath>
#include <sstream>
#include <algorithm>

namespace mshqc {
namespace validation {

// Helper: compute correlation energy
double MPFCIValidator::compute_correlation_energy(double e_total, double e_hf) {
    return e_total - e_hf;
}

// Helper: compute percentage safely
double MPFCIValidator::compute_percentage(double partial, double total) {
    if (std::abs(total) < 1e-12) return 0.0;
    return 100.0 * partial / total;
}

// Helper: safe division for ratios
double MPFCIValidator::safe_ratio(double numerator, double denominator) {
    if (std::abs(denominator) < 1e-12) return 0.0;
    return std::abs(numerator) / std::abs(denominator);
}

// Main validation function
MPConvergenceReport MPFCIValidator::validate(
    double e_hf,
    double e_mp2,
    double e_mp3,
    double e_mp4,
    double e_mp5,
    double e_fci,
    const std::string& system_name,
    int n_electrons,
    int n_basis,
    const std::string& basis_name,
    double threshold
) {
    MPConvergenceReport report;
    
    // System info
    report.system_name = system_name;
    report.n_electrons = n_electrons;
    report.n_basis = n_basis;
    report.basis_name = basis_name;
    
    // Store energies
    report.e_hf = e_hf;
    report.e_mp2 = e_mp2;
    report.e_mp3 = e_mp3;
    report.e_mp4 = e_mp4;
    report.e_mp5 = e_mp5;
    report.e_fci = e_fci;
    
    // Compute correlation energies
    // Theory: E_corr = E_total - E_HF (Møller & Plesset 1934)
    report.ec_mp2 = compute_correlation_energy(e_mp2, e_hf);
    report.ec_mp3 = compute_correlation_energy(e_mp3, e_hf);
    report.ec_mp4 = compute_correlation_energy(e_mp4, e_hf);
    report.ec_mp5 = compute_correlation_energy(e_mp5, e_hf);
    report.ec_fci = compute_correlation_energy(e_fci, e_hf);
    
    // Compute absolute errors vs FCI (gold standard)
    // Theory: FCI is exact within basis, so |E_MP - E_FCI| = approximation error
    // Reference: Knowles & Handy, Chem. Phys. Lett. 111, 315 (1984)
    report.error_mp2 = std::abs(e_mp2 - e_fci);
    report.error_mp3 = std::abs(e_mp3 - e_fci);
    report.error_mp4 = std::abs(e_mp4 - e_fci);
    report.error_mp5 = std::abs(e_mp5 - e_fci);
    
    // Compute convergence ratios
    // Theory: For convergent series, |E^(n+1)| < |E^(n)|
    // Reference: Pople et al., Int. J. Quantum Chem. 14, 545 (1978)
    double e3 = report.ec_mp3 - report.ec_mp2;  // E^(3) correction
    double e4 = report.ec_mp4 - report.ec_mp3;  // E^(4) correction
    double e5 = report.ec_mp5 - report.ec_mp4;  // E^(5) correction
    
    report.ratio_32 = safe_ratio(e3, report.ec_mp2);  // |E^(3)|/|E^(2)|
    report.ratio_43 = safe_ratio(e4, e3);              // |E^(4)|/|E^(3)|
    report.ratio_54 = safe_ratio(e5, e4);              // |E^(5)|/|E^(4)|
    
    // Check convergence
    // Theory: Series converges if ratios decrease (λ_n < λ_{n-1})
    report.is_converging = is_series_converging(report);
    report.convergence_rate = estimate_convergence_rate(report);
    
    // Check if MP5 converged within threshold
    report.mp5_converged = (report.error_mp5 < threshold);
    
    // Compute correlation recovery percentages
    // Shows how much of FCI correlation each MP level captures
    report.corr_mp2_pct = compute_percentage(report.ec_mp2, report.ec_fci);
    report.corr_mp3_pct = compute_percentage(report.ec_mp3, report.ec_fci);
    report.corr_mp4_pct = compute_percentage(report.ec_mp4, report.ec_fci);
    report.corr_mp5_pct = compute_percentage(report.ec_mp5, report.ec_fci);
    
    return report;
}

// Check if MP series is converging
bool MPFCIValidator::is_series_converging(const MPConvergenceReport& report) {
    // Theory: Convergent series has decreasing correction ratios
    // For well-behaved systems: λ_n < λ_{n-1} < 1.0
    
    // Need at least 2 ratios to determine trend
    if (report.ratio_32 < 1e-12 && report.ratio_43 < 1e-12) {
        return true;  // Near-zero corrections (already converged)
    }
    
    // Check if ratios are decreasing
    bool ratio_43_smaller = (report.ratio_43 < report.ratio_32);
    bool ratio_54_smaller = (report.ratio_54 < report.ratio_43);
    
    // Series converges if ratios decrease OR stay below 1.0
    return (ratio_43_smaller && ratio_54_smaller) || 
           (report.ratio_43 < 1.0 && report.ratio_54 < 1.0);
}

// Estimate convergence rate
double MPFCIValidator::estimate_convergence_rate(const MPConvergenceReport& report) {
    // Theory: Average convergence rate λ ≈ geometric mean of ratios
    // If λ < 1: converging, if λ > 1: diverging
    
    // Use geometric mean of available ratios
    double product = 1.0;
    int count = 0;
    
    if (report.ratio_32 > 1e-12) { product *= report.ratio_32; count++; }
    if (report.ratio_43 > 1e-12) { product *= report.ratio_43; count++; }
    if (report.ratio_54 > 1e-12) { product *= report.ratio_54; count++; }
    
    if (count == 0) return 0.0;
    
    return std::pow(product, 1.0 / count);
}

// Predict MP6 error
double MPFCIValidator::predict_mp6_error(const MPConvergenceReport& report) {
    // Theory: If convergence rate is constant (λ), then
    //         E^(6) ≈ E^(5) × λ
    //         error_MP6 ≈ error_MP5 × λ
    
    double lambda = report.convergence_rate;
    return report.error_mp5 * lambda;
}

// Print detailed report
void MPFCIValidator::print_report(const MPConvergenceReport& report, bool verbose) {
    using namespace std;
    
    cout << "\n";
    cout << "================================================================================\n";
    cout << "  MP SERIES VALIDATION vs FCI (Gold Standard)\n";
    cout << "================================================================================\n";
    cout << "System: " << report.system_name << " / " << report.basis_name << "\n";
    cout << "Electrons: " << report.n_electrons << ", Basis functions: " << report.n_basis << "\n";
    cout << "\n";
    
    // Energy table
    cout << "ENERGIES (Hartree):\n";
    cout << "--------------------------------------------------------------------------------\n";
    cout << fixed << setprecision(10);
    cout << "E(HF)  = " << setw(16) << report.e_hf << " (reference)\n";
    cout << "E(MP2) = " << setw(16) << report.e_mp2 << "\n";
    cout << "E(MP3) = " << setw(16) << report.e_mp3 << "\n";
    cout << "E(MP4) = " << setw(16) << report.e_mp4 << "\n";
    cout << "E(MP5) = " << setw(16) << report.e_mp5 << "\n";
    cout << "E(FCI) = " << setw(16) << report.e_fci << " (exact)\n";
    cout << "\n";
    
    // Correlation energies
    cout << "CORRELATION ENERGIES:\n";
    cout << "--------------------------------------------------------------------------------\n";
    cout << "E_c(MP2) = " << setw(14) << report.ec_mp2 
         << "  (" << setw(5) << fixed << setprecision(2) << report.corr_mp2_pct << "% of FCI)\n";
    cout << "E_c(MP3) = " << setw(14) << report.ec_mp3 
         << "  (" << setw(5) << report.corr_mp3_pct << "% of FCI)\n";
    cout << "E_c(MP4) = " << setw(14) << report.ec_mp4 
         << "  (" << setw(5) << report.corr_mp4_pct << "% of FCI)\n";
    cout << "E_c(MP5) = " << setw(14) << report.ec_mp5 
         << "  (" << setw(5) << report.corr_mp5_pct << "% of FCI)\n";
    cout << "E_c(FCI) = " << setw(14) << report.ec_fci << "  (100.00%)\n";
    cout << "\n";
    
    // Error analysis
    cout << "ERROR vs FCI:\n";
    cout << "--------------------------------------------------------------------------------\n";
    cout << scientific << setprecision(3);
    cout << "MP2: " << setw(12) << report.error_mp2 << " Ha  = " 
         << setw(10) << report.error_mp2 * 1e6 << " µHa\n";
    cout << "MP3: " << setw(12) << report.error_mp3 << " Ha  = " 
         << setw(10) << report.error_mp3 * 1e6 << " µHa\n";
    cout << "MP4: " << setw(12) << report.error_mp4 << " Ha  = " 
         << setw(10) << report.error_mp4 * 1e6 << " µHa\n";
    cout << "MP5: " << setw(12) << report.error_mp5 << " Ha  = " 
         << setw(10) << report.error_mp5 * 1e6 << " µHa ";
    
    if (report.mp5_converged) {
        cout << " ✓ CONVERGED\n";
    } else {
        cout << " ⚠ NOT CONVERGED\n";
    }
    cout << "\n";
    
    // Convergence analysis
    if (verbose) {
        cout << "CONVERGENCE ANALYSIS:\n";
        cout << "--------------------------------------------------------------------------------\n";
        cout << fixed << setprecision(4);
        cout << "Ratio |E^(3)|/|E^(2)| = " << report.ratio_32 << "\n";
        cout << "Ratio |E^(4)|/|E^(3)| = " << report.ratio_43 << "\n";
        cout << "Ratio |E^(5)|/|E^(4)| = " << report.ratio_54 << "\n";
        cout << "Convergence rate λ    = " << report.convergence_rate << "\n";
        cout << "\n";
        
        cout << "STATUS: ";
        if (report.is_converging) {
            cout << "✓ Series is CONVERGING (λ < 1)\n";
        } else {
            cout << "⚠ Series may be DIVERGING or oscillating\n";
        }
        
        // Predict MP6 if converging
        if (report.is_converging && report.convergence_rate < 1.0) {
            double mp6_pred = predict_mp6_error(report);
            cout << "Predicted MP6 error: " << scientific << setprecision(2) 
                 << mp6_pred << " Ha = " << mp6_pred * 1e6 << " µHa\n";
        }
        cout << "\n";
    }
    
    // Diagnostic warnings
    cout << "DIAGNOSTICS:\n";
    cout << "--------------------------------------------------------------------------------\n";
    
    if (report.error_mp5 < 1e-6) {
        cout << "✓ Excellent agreement (< 1 µHa)\n";
    } else if (report.error_mp5 < 10e-6) {
        cout << "✓ Good agreement (< 10 µHa)\n";
    } else if (report.error_mp5 < 100e-6) {
        cout << "⚠ Fair agreement (< 100 µHa) - consider MP6 or larger basis\n";
    } else {
        cout << "⚠ Poor agreement (≥ 100 µHa) - check for multi-reference character\n";
    }
    
    if (report.ratio_32 > 1.5) {
        cout << "⚠ Large E^(3)/E^(2) ratio - possible open-shell overshoot (Pople 1977)\n";
    }
    
    if (!report.is_converging) {
        cout << "⚠ Series not converging - system may be multi-reference\n";
    }
    
    cout << "================================================================================\n";
    cout << "\n";
}

// Generate LaTeX table
std::string MPFCIValidator::to_latex_table(const MPConvergenceReport& report) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(8);
    
    oss << "\\begin{table}[h]\n";
    oss << "\\centering\n";
    oss << "\\caption{MP Series Convergence for " << report.system_name 
        << " / " << report.basis_name << "}\n";
    oss << "\\begin{tabular}{lcc}\n";
    oss << "\\hline\n";
    oss << "Method & Energy (Ha) & Error vs FCI ($\\mu$Ha) \\\\\n";
    oss << "\\hline\n";
    oss << "HF   & " << report.e_hf << " & - \\\\\n";
    oss << "MP2  & " << report.e_mp2 << " & " << report.error_mp2 * 1e6 << " \\\\\n";
    oss << "MP3  & " << report.e_mp3 << " & " << report.error_mp3 * 1e6 << " \\\\\n";
    oss << "MP4  & " << report.e_mp4 << " & " << report.error_mp4 * 1e6 << " \\\\\n";
    oss << "MP5  & " << report.e_mp5 << " & " << report.error_mp5 * 1e6 << " \\\\\n";
    oss << "FCI  & " << report.e_fci << " & 0.000 \\\\\n";
    oss << "\\hline\n";
    oss << "\\end{tabular}\n";
    oss << "\\end{table}\n";
    
    return oss.str();
}

// Export to JSON
std::string MPFCIValidator::to_json(const MPConvergenceReport& report) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(10);
    
    oss << "{\n";
    oss << "  \"system\": \"" << report.system_name << "\",\n";
    oss << "  \"basis\": \"" << report.basis_name << "\",\n";
    oss << "  \"n_electrons\": " << report.n_electrons << ",\n";
    oss << "  \"n_basis\": " << report.n_basis << ",\n";
    oss << "  \"energies\": {\n";
    oss << "    \"hf\": " << report.e_hf << ",\n";
    oss << "    \"mp2\": " << report.e_mp2 << ",\n";
    oss << "    \"mp3\": " << report.e_mp3 << ",\n";
    oss << "    \"mp4\": " << report.e_mp4 << ",\n";
    oss << "    \"mp5\": " << report.e_mp5 << ",\n";
    oss << "    \"fci\": " << report.e_fci << "\n";
    oss << "  },\n";
    oss << "  \"errors_microHa\": {\n";
    oss << "    \"mp2\": " << report.error_mp2 * 1e6 << ",\n";
    oss << "    \"mp3\": " << report.error_mp3 * 1e6 << ",\n";
    oss << "    \"mp4\": " << report.error_mp4 * 1e6 << ",\n";
    oss << "    \"mp5\": " << report.error_mp5 * 1e6 << "\n";
    oss << "  },\n";
    oss << "  \"convergence\": {\n";
    oss << "    \"is_converging\": " << (report.is_converging ? "true" : "false") << ",\n";
    oss << "    \"mp5_converged\": " << (report.mp5_converged ? "true" : "false") << ",\n";
    oss << "    \"rate\": " << report.convergence_rate << "\n";
    oss << "  }\n";
    oss << "}\n";
    
    return oss.str();
}

} // namespace validation
} // namespace mshqc
