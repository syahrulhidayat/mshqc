/**
 * @file wavefunction_analysis.h
 * @brief Tools for analyzing CI wavefunctions
 * 
 * THEORY REFERENCES:
 *   - Szabo & Ostlund (1996), Ch. 4.4: CI coefficient interpretation
 *   - Helgaker et al. (2000), Ch. 11.7: Wavefunction analysis
 *   - Janssen & Nielsen (1998), Chem. Phys. Lett. 290, 423
 * 
 * USAGE:
 *   WavefunctionAnalysis analysis(ci_result.determinants, ci_result.coefficients);
 *   analysis.print_dominant_determinants(10);  


 *   analysis.print_excitation_composition();   


 *   auto diag = analysis.compute_diagnostics(hf_det);
 * 
 * KEY FEATURES:
 *   - Dominant determinant identification
 *   - Excitation level composition (singles/doubles/triples contribution)
 *   - CI correlation diagnostics (T1, D1, %HF)
 *   - Determinant character analysis
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 * @note Original implementation from textbook theory (AI_RULES compliant)
 */

#ifndef MSHQC_CI_WAVEFUNCTION_ANALYSIS_H
#define MSHQC_CI_WAVEFUNCTION_ANALYSIS_H

#include "mshqc/ci/determinant.h"
#include <Eigen/Dense>
#include <vector>
#include <string>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace ci {

/**
 * Determinant contribution structure
 */
struct DeterminantContribution {
    int index;                    


    Determinant det;              


    double coefficient;           


    double weight;                


    std::string excitation_type;  


    int excitation_level;         


};

/**
 * Excitation composition (singles/doubles/triples breakdown)
 */
struct ExcitationComposition {
    int n_hf;           


    int n_singles;      


    int n_doubles;      


    int n_triples;      


    int n_higher;       


    
    double weight_hf;       


    double weight_singles;  


    double weight_doubles;  


    double weight_triples;  


    double weight_higher;   


    
    


    double percent_hf;
    double percent_singles;
    double percent_doubles;
    double percent_triples;
    double percent_higher;
};

/**
 * CI diagnostics structure
 * Measures deviation from single-reference character
 */
struct CIDiagnostics {
    double hf_weight;           


    double t1_diagnostic;       


    double d1_diagnostic;       


    double leading_det_weight;  


    
    


    bool single_reference_ok;   


    std::string multireference_character;  


};

/**
 * CI Wavefunction Analysis
 * 
 * Provides tools to interpret CI wavefunction:
 * - Dominant determinants
 * - Excitation composition
 * - Correlation diagnostics
 */
class WavefunctionAnalysis {
public:
    /**
     * Constructor
     * 
     * @param dets CI determinants
     * @param coeffs CI coefficients (normalized)
     */
    WavefunctionAnalysis(const std::vector<Determinant>& dets,
                        const Eigen::VectorXd& coeffs);
    
    /**
     * Print dominant determinants
     * Shows determinants with largest |c_i|²
     * 
     * @param n_print Number to print (default 10)
     */
    void print_dominant_determinants(int n_print = 10) const;
    
    /**
     * Print excitation composition
     * Shows breakdown by singles/doubles/triples
     * 
     * @param hf_det Reference determinant (usually HF)
     */
    void print_excitation_composition(const Determinant& hf_det) const;
    
    /**
     * Compute CI diagnostics
     * Measures multireference character
     * 
     * REFERENCE: Janssen & Nielsen (1998)
     * 
     * @param hf_det HF reference determinant
     * @return CIDiagnostics structure
     */
    CIDiagnostics compute_diagnostics(const Determinant& hf_det) const;
    
    /**
     * Get dominant determinants sorted by weight
     * 
     * @param n_get Number to return (0 = all)
     * @return Vector of DeterminantContribution
     */
    std::vector<DeterminantContribution> get_dominant_determinants(int n_get = 0) const;
    
    /**
     * Analyze excitation composition
     * 
     * @param hf_det Reference determinant
     * @return ExcitationComposition structure
     */
    ExcitationComposition analyze_excitation_composition(const Determinant& hf_det) const;
    
    /**
     * Print full wavefunction analysis
     * Comprehensive report with all diagnostics
     * 
     * @param hf_det Reference determinant
     */
    void print_full_analysis(const Determinant& hf_det) const;
    
private:
    const std::vector<Determinant>& dets_;
    const Eigen::VectorXd& coeffs_;
    
    /**
     * Determine excitation level between two determinants
     * 
     * @param det Target determinant
     * @param ref Reference determinant
     * @return Excitation level (0=same, 1=single, 2=double, etc.)
     */
    int excitation_level(const Determinant& det, const Determinant& ref) const;
    
    /**
     * Get excitation type string
     * 
     * @param level Excitation level
     * @return String like "HF", "S", "D", "T", etc.
     */
    std::string excitation_type_string(int level) const;
};

} 


} 



#endif 


