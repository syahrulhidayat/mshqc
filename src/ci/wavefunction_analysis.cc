/**
 * @file wavefunction_analysis.cc
 * @brief CI wavefunction analysis implementation
 * 
 * THEORY REFERENCES:
 *   - Szabo & Ostlund (1996), Ch. 4.4
 *   - Helgaker et al. (2000), Ch. 11.7
 *   - Janssen & Nielsen (1998), Chem. Phys. Lett. 290, 423
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 */

#include "mshqc/ci/wavefunction_analysis.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace mshqc {
namespace ci {

// Constructor
WavefunctionAnalysis::WavefunctionAnalysis(
    const std::vector<Determinant>& dets,
    const Eigen::VectorXd& coeffs)
    : dets_(dets), coeffs_(coeffs) {}

// Determine excitation level
int WavefunctionAnalysis::excitation_level(
    const Determinant& det,
    const Determinant& ref) const {
    
    auto occ_det_a = det.alpha_occupations();
    auto occ_det_b = det.beta_occupations();
    auto occ_ref_a = ref.alpha_occupations();
    auto occ_ref_b = ref.beta_occupations();
    
    // Count differences
    int diff_a = 0, diff_b = 0;
    
    for (int orb : occ_det_a) {
        if (std::find(occ_ref_a.begin(), occ_ref_a.end(), orb) == occ_ref_a.end()) {
            diff_a++;
        }
    }
    
    for (int orb : occ_det_b) {
        if (std::find(occ_ref_b.begin(), occ_ref_b.end(), orb) == occ_ref_b.end()) {
            diff_b++;
        }
    }
    
    return diff_a + diff_b;
}

// Get excitation type string
std::string WavefunctionAnalysis::excitation_type_string(int level) const {
    switch (level) {
        case 0: return "HF";
        case 1: return "S";
        case 2: return "D";
        case 3: return "T";
        case 4: return "Q";
        default: return "H";  // Higher
    }
}

// Get dominant determinants
std::vector<DeterminantContribution> 
WavefunctionAnalysis::get_dominant_determinants(int n_get) const {
    
    std::vector<DeterminantContribution> contribs;
    
    for (size_t i = 0; i < dets_.size(); i++) {
        DeterminantContribution contrib;
        contrib.index = i;
        contrib.det = dets_[i];
        contrib.coefficient = coeffs_(i);
        contrib.weight = coeffs_(i) * coeffs_(i);
        contribs.push_back(contrib);
    }
    
    // Sort by weight (descending)
    std::sort(contribs.begin(), contribs.end(),
             [](const DeterminantContribution& a, const DeterminantContribution& b) {
                 return a.weight > b.weight;
             });
    
    // Return requested number
    if (n_get > 0 && n_get < static_cast<int>(contribs.size())) {
        contribs.resize(n_get);
    }
    
    return contribs;
}

// Print dominant determinants
void WavefunctionAnalysis::print_dominant_determinants(int n_print) const {
    std::cout << "\n=== Dominant Determinants ===\n\n";
    
    auto contribs = get_dominant_determinants(n_print);
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << " Rank   Index    Coefficient      Weight      Occupation (α|β)\n";
    std::cout << "----------------------------------------------------------------\n";
    
    for (size_t rank = 0; rank < contribs.size(); rank++) {
        auto& c = contribs[rank];
        
        std::cout << std::setw(4) << (rank + 1) << "  "
                  << std::setw(6) << c.index << "  "
                  << std::setw(14) << c.coefficient << "  "
                  << std::setw(12) << c.weight << "  ";
        
        // Print occupation
        auto occ_a = c.det.alpha_occupations();
        auto occ_b = c.det.beta_occupations();
        
        for (int orb : occ_a) std::cout << orb << " ";
        std::cout << "| ";
        for (int orb : occ_b) std::cout << orb << " ";
        std::cout << "\n";
    }
    
    std::cout << "\n";
}

// Analyze excitation composition
ExcitationComposition 
WavefunctionAnalysis::analyze_excitation_composition(const Determinant& hf_det) const {
    
    ExcitationComposition comp;
    comp.n_hf = 0;
    comp.n_singles = 0;
    comp.n_doubles = 0;
    comp.n_triples = 0;
    comp.n_higher = 0;
    
    comp.weight_hf = 0.0;
    comp.weight_singles = 0.0;
    comp.weight_doubles = 0.0;
    comp.weight_triples = 0.0;
    comp.weight_higher = 0.0;
    
    for (size_t i = 0; i < dets_.size(); i++) {
        int level = excitation_level(dets_[i], hf_det);
        double weight = coeffs_(i) * coeffs_(i);
        
        switch (level) {
            case 0:
                comp.n_hf++;
                comp.weight_hf += weight;
                break;
            case 1:
                comp.n_singles++;
                comp.weight_singles += weight;
                break;
            case 2:
                comp.n_doubles++;
                comp.weight_doubles += weight;
                break;
            case 3:
                comp.n_triples++;
                comp.weight_triples += weight;
                break;
            default:
                comp.n_higher++;
                comp.weight_higher += weight;
                break;
        }
    }
    
    // Compute percentages
    double total_weight = comp.weight_hf + comp.weight_singles + comp.weight_doubles +
                         comp.weight_triples + comp.weight_higher;
    
    comp.percent_hf = (comp.weight_hf / total_weight) * 100.0;
    comp.percent_singles = (comp.weight_singles / total_weight) * 100.0;
    comp.percent_doubles = (comp.weight_doubles / total_weight) * 100.0;
    comp.percent_triples = (comp.weight_triples / total_weight) * 100.0;
    comp.percent_higher = (comp.weight_higher / total_weight) * 100.0;
    
    return comp;
}

// Print excitation composition
void WavefunctionAnalysis::print_excitation_composition(const Determinant& hf_det) const {
    std::cout << "\n=== Excitation Composition ===\n\n";
    
    auto comp = analyze_excitation_composition(hf_det);
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << " Type        Count      Weight      Percentage\n";
    std::cout << "------------------------------------------------\n";
    
    if (comp.n_hf > 0) {
        std::cout << " HF       " << std::setw(8) << comp.n_hf
                  << "  " << std::setw(10) << comp.weight_hf
                  << "  " << std::setw(10) << comp.percent_hf << "%\n";
    }
    
    if (comp.n_singles > 0) {
        std::cout << " Singles  " << std::setw(8) << comp.n_singles
                  << "  " << std::setw(10) << comp.weight_singles
                  << "  " << std::setw(10) << comp.percent_singles << "%\n";
    }
    
    if (comp.n_doubles > 0) {
        std::cout << " Doubles  " << std::setw(8) << comp.n_doubles
                  << "  " << std::setw(10) << comp.weight_doubles
                  << "  " << std::setw(10) << comp.percent_doubles << "%\n";
    }
    
    if (comp.n_triples > 0) {
        std::cout << " Triples  " << std::setw(8) << comp.n_triples
                  << "  " << std::setw(10) << comp.weight_triples
                  << "  " << std::setw(10) << comp.percent_triples << "%\n";
    }
    
    if (comp.n_higher > 0) {
        std::cout << " Higher   " << std::setw(8) << comp.n_higher
                  << "  " << std::setw(10) << comp.weight_higher
                  << "  " << std::setw(10) << comp.percent_higher << "%\n";
    }
    
    std::cout << "------------------------------------------------\n";
    std::cout << " Total    " << std::setw(8) << dets_.size()
              << "  " << std::setw(10) << 1.0
              << "  " << std::setw(10) << 100.0 << "%\n\n";
}

// Compute diagnostics
CIDiagnostics WavefunctionAnalysis::compute_diagnostics(const Determinant& hf_det) const {
    CIDiagnostics diag;
    
    // Find HF determinant
    diag.hf_weight = 0.0;
    for (size_t i = 0; i < dets_.size(); i++) {
        if (dets_[i] == hf_det) {
            diag.hf_weight = coeffs_(i) * coeffs_(i);
            break;
        }
    }
    
    // Find leading determinant weight
    diag.leading_det_weight = 0.0;
    for (int i = 0; i < coeffs_.size(); i++) {
        double weight = coeffs_(i) * coeffs_(i);
        if (weight > diag.leading_det_weight) {
            diag.leading_det_weight = weight;
        }
    }
    
    // T1 diagnostic: sqrt(Σ|c_singles|²)
    // D1 diagnostic: sqrt(Σ|c_doubles|²)
    double singles_weight = 0.0;
    double doubles_weight = 0.0;
    
    for (size_t i = 0; i < dets_.size(); i++) {
        int level = excitation_level(dets_[i], hf_det);
        double weight = coeffs_(i) * coeffs_(i);
        
        if (level == 1) singles_weight += weight;
        if (level == 2) doubles_weight += weight;
    }
    
    diag.t1_diagnostic = std::sqrt(singles_weight);
    diag.d1_diagnostic = std::sqrt(doubles_weight);
    
    // Interpretation
    diag.single_reference_ok = (diag.hf_weight > 0.90);
    
    if (diag.hf_weight > 0.95) {
        diag.multireference_character = "single";
    } else if (diag.hf_weight > 0.85) {
        diag.multireference_character = "moderate";
    } else {
        diag.multireference_character = "strong";
    }
    
    return diag;
}

// Print full analysis
void WavefunctionAnalysis::print_full_analysis(const Determinant& hf_det) const {
    std::cout << "\n========================================\n";
    std::cout << "  CI Wavefunction Analysis\n";
    std::cout << "========================================\n";
    
    std::cout << "\nTotal determinants: " << dets_.size() << "\n";
    
    // Diagnostics
    auto diag = compute_diagnostics(hf_det);
    
    std::cout << "\n--- CI Diagnostics ---\n\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "HF weight:         " << diag.hf_weight << " (" 
              << (diag.hf_weight * 100.0) << "%)\n";
    std::cout << "Leading det weight: " << diag.leading_det_weight << "\n";
    std::cout << "T1 diagnostic:     " << diag.t1_diagnostic << "\n";
    std::cout << "D1 diagnostic:     " << diag.d1_diagnostic << "\n\n";
    
    std::cout << "Multireference character: " << diag.multireference_character << "\n";
    
    if (diag.single_reference_ok) {
        std::cout << "✅ Single-reference appropriate (HF weight > 0.90)\n";
    } else {
        std::cout << "⚠️  Multireference needed (HF weight < 0.90)\n";
    }
    
    // Excitation composition
    print_excitation_composition(hf_det);
    
    // Dominant determinants
    print_dominant_determinants(10);
    
    std::cout << "========================================\n\n";
}

} // namespace ci
} // namespace mshqc
