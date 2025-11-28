/**
 * @file natural_orbitals.cc
 * @brief Natural orbital analysis implementation
 * 
 * THEORY REFERENCES:
 *   - Löwdin (1955), Phys. Rev. 97, 1474
 *   - McWeeny (1989), Methods of Molecular Quantum Mechanics, Ch. 6
 *   - Helgaker et al. (2000), Ch. 14.8
 *   - Head-Gordon (2003), Chem. Phys. Lett. 372, 508
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 */

#include "mshqc/ci/natural_orbitals.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

namespace mshqc {
namespace ci {

// Constructor
NaturalOrbitalAnalysis::NaturalOrbitalAnalysis(
    const std::vector<Determinant>& dets,
    const Eigen::VectorXd& coeffs)
    : dets_(dets), coeffs_(coeffs) {}

// Compute 1-RDM matrix element using Slater-Condon rules
// REFERENCE: McWeeny (1989), Section 6.3
double NaturalOrbitalAnalysis::rdm_element(
    const Determinant& det_i,
    const Determinant& det_j,
    int p, int q, bool alpha) {
    
    // Get occupation lists
    auto occ_i = alpha ? det_i.alpha_occupations() : det_i.beta_occupations();
    auto occ_j = alpha ? det_j.alpha_occupations() : det_j.beta_occupations();
    
    // Check if orbitals p and q are occupied
    bool p_in_i = std::find(occ_i.begin(), occ_i.end(), p) != occ_i.end();
    bool q_in_i = std::find(occ_i.begin(), occ_i.end(), q) != occ_i.end();
    bool p_in_j = std::find(occ_j.begin(), occ_j.end(), p) != occ_j.end();
    bool q_in_j = std::find(occ_j.begin(), occ_j.end(), q) != occ_j.end();
    
    // Case 1: Same determinant
    // ⟨I| a†_p a_q |I⟩ = δ_pq if p occupied in I, else 0
    if (det_i == det_j) {
        if (p == q && p_in_i) {
            return 1.0;
        }
        return 0.0;
    }
    
    // Case 2: Differ by single excitation
    // ⟨I| a†_p a_q |J⟩ for J = a†_a a_i |I⟩
    // Result: δ_pa δ_qi × phase
    
    // Find differences in occupations
    std::vector<int> only_in_i, only_in_j;
    for (int orb : occ_i) {
        if (std::find(occ_j.begin(), occ_j.end(), orb) == occ_j.end()) {
            only_in_i.push_back(orb);
        }
    }
    for (int orb : occ_j) {
        if (std::find(occ_i.begin(), occ_i.end(), orb) == occ_i.end()) {
            only_in_j.push_back(orb);
        }
    }
    
    // Single excitation: one orbital different
    if (only_in_i.size() == 1 && only_in_j.size() == 1) {
        int i_orb = only_in_i[0];  // Hole in I
        int a_orb = only_in_j[0];  // Particle in J
        
        // Check if this matches a†_p a_q: need p=a, q=i
        if (p == a_orb && q == i_orb) {
            // Compute phase factor
            // Phase = (-1)^{number of electrons between i and a}
            int phase_count = 0;
            for (int orb : occ_i) {
                if (orb > std::min(i_orb, a_orb) && orb < std::max(i_orb, a_orb)) {
                    phase_count++;
                }
            }
            double phase = (phase_count % 2 == 0) ? 1.0 : -1.0;
            return phase;
        }
    }
    
    // All other cases: zero
    return 0.0;
}

// Build 1-RDM
// REFERENCE: Helgaker et al. (2000), Eq. (14.8.3)
void NaturalOrbitalAnalysis::build_1rdm(
    int n_orb,
    Eigen::MatrixXd& rdm_alpha,
    Eigen::MatrixXd& rdm_beta) {
    
    rdm_alpha.resize(n_orb, n_orb);
    rdm_beta.resize(n_orb, n_orb);
    rdm_alpha.setZero();
    rdm_beta.setZero();
    
    int n_dets = dets_.size();
    
    // γ_pq = Σ_I Σ_J c_I c_J ⟨I| a†_p a_q |J⟩
    for (int I = 0; I < n_dets; I++) {
        for (int J = 0; J < n_dets; J++) {
            double ci_cj = coeffs_(I) * coeffs_(J);
            
            // Skip if coefficient product negligible
            if (std::abs(ci_cj) < 1e-12) continue;
            
            // Compute RDM elements for all p, q pairs
            for (int p = 0; p < n_orb; p++) {
                for (int q = 0; q < n_orb; q++) {
                    double elem_alpha = rdm_element(dets_[I], dets_[J], p, q, true);
                    double elem_beta = rdm_element(dets_[I], dets_[J], p, q, false);
                    
                    rdm_alpha(p, q) += ci_cj * elem_alpha;
                    rdm_beta(p, q) += ci_cj * elem_beta;
                }
            }
        }
    }
}

// Compute correlation measure
// REFERENCE: Head-Gordon (2003), Chem. Phys. Lett. 372, 508
double NaturalOrbitalAnalysis::compute_correlation_measure(
    const Eigen::VectorXd& occupations) {
    
    double measure = 0.0;
    
    for (int i = 0; i < occupations.size(); i++) {
        double n_i = occupations(i);
        
        // Deviation from nearest integer (0, 1, or 2)
        double nearest = std::round(n_i);
        double deviation = std::abs(n_i - nearest);
        
        measure += deviation;
    }
    
    return measure;
}

// Main compute function
NaturalOrbitalResult NaturalOrbitalAnalysis::compute(int n_orb) {
    
    std::cout << "\n=== Natural Orbital Analysis ===\n";
    std::cout << "Orbitals: " << n_orb << "\n";
    std::cout << "Determinants: " << dets_.size() << "\n\n";
    
    // Build 1-RDM
    std::cout << "Building 1-electron reduced density matrix...\n";
    Eigen::MatrixXd rdm_alpha, rdm_beta;
    build_1rdm(n_orb, rdm_alpha, rdm_beta);
    
    // Diagonalize to get natural orbitals
    std::cout << "Diagonalizing 1-RDM...\n";
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_alpha(rdm_alpha);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_beta(rdm_beta);
    
    NaturalOrbitalResult result;
    
    // Extract eigenvalues (natural occupations) - already sorted ascending
    result.occupations_alpha = solver_alpha.eigenvalues();
    result.occupations_beta = solver_beta.eigenvalues();
    
    // Extract eigenvectors (natural orbitals)
    result.orbitals_alpha = solver_alpha.eigenvectors();
    result.orbitals_beta = solver_beta.eigenvectors();
    
    // Compute diagnostics
    result.total_occupation_alpha = result.occupations_alpha.sum();
    result.total_occupation_beta = result.occupations_beta.sum();
    
    // Count occupation types
    result.n_strongly_occupied = 0;
    result.n_weakly_occupied = 0;
    result.n_fractional = 0;
    
    for (int i = 0; i < n_orb; i++) {
        double occ_a = result.occupations_alpha(i);
        double occ_b = result.occupations_beta(i);
        
        for (double occ : {occ_a, occ_b}) {
            if (occ > 1.95) result.n_strongly_occupied++;
            else if (occ < 0.05) result.n_weakly_occupied++;
            else result.n_fractional++;
        }
    }
    
    // Correlation measure (both spins)
    Eigen::VectorXd all_occs(2 * n_orb);
    all_occs << result.occupations_alpha, result.occupations_beta;
    result.correlation_measure = compute_correlation_measure(all_occs);
    
    std::cout << "Natural orbital analysis complete.\n\n";
    
    return result;
}

// Print summary
void NaturalOrbitalResult::print_summary() const {
    std::cout << "=== Natural Orbital Summary ===\n\n";
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Total occupation (α): " << total_occupation_alpha << "\n";
    std::cout << "Total occupation (β): " << total_occupation_beta << "\n";
    std::cout << "Total electrons:      " << (total_occupation_alpha + total_occupation_beta) << "\n\n";
    
    std::cout << "Occupation character:\n";
    std::cout << "  Strongly occupied (n > 1.95): " << n_strongly_occupied << "\n";
    std::cout << "  Fractional (0.05 < n < 1.95):  " << n_fractional << "\n";
    std::cout << "  Weakly occupied (n < 0.05):    " << n_weakly_occupied << "\n\n";
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Correlation measure: " << correlation_measure << "\n";
    std::cout << "  (deviation from integer occupations)\n";
    std::cout << "  0.000 = Hartree-Fock\n";
    std::cout << "  Larger = more electron correlation\n\n";
}

// Print occupations
void NaturalOrbitalResult::print_occupations(int n_print) const {
    std::cout << "=== Natural Orbital Occupations ===\n\n";
    
    int n_orb = occupations_alpha.size();
    int n_to_print = std::min(n_print, n_orb);
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << " Orbital     α Occupation     β Occupation     Total\n";
    std::cout << "--------------------------------------------------------\n";
    
    // Print in descending order (highest occupations first)
    for (int i = n_orb - 1; i >= n_orb - n_to_print; i--) {
        double occ_a = occupations_alpha(i);
        double occ_b = occupations_beta(i);
        double total = occ_a + occ_b;
        
        std::cout << std::setw(4) << (n_orb - i) << "  "
                  << std::setw(14) << occ_a << "  "
                  << std::setw(14) << occ_b << "  "
                  << std::setw(14) << total << "\n";
    }
    
    if (n_to_print < n_orb) {
        std::cout << " ... (showing top " << n_to_print << " of " << n_orb << " orbitals)\n";
    }
    std::cout << "\n";
}

} // namespace ci
} // namespace mshqc
