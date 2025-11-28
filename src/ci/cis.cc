/**
 * @file cis.cc
 * @brief CIS implementation for excited state calculations
 * 
 * THEORY REFERENCES:
 *   - Foresman et al. (1992), J. Phys. Chem. 96, 135
 *   - Szabo & Ostlund (1996), Ch. 4.2
 *   - Helgaker et al. (2000), Ch. 10.7
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 */

#include "mshqc/ci/cis.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

namespace mshqc {
namespace ci {

CIS::CIS(const CIIntegrals& ints,
         const Determinant& hf_det,
         int n_occ_alpha, int n_occ_beta,
         int n_virt_alpha, int n_virt_beta)
    : ints_(ints), hf_det_(hf_det),
      nocc_a_(n_occ_alpha), nocc_b_(n_occ_beta),
      nvirt_a_(n_virt_alpha), nvirt_b_(n_virt_beta) {}

// Generate all single excitations from HF determinant
// REFERENCE: Szabo & Ostlund (1996), Section 4.2
std::vector<Determinant> CIS::generate_singles() {
    std::vector<Determinant> singles;
    
    // Add HF determinant first (reference)
    singles.push_back(hf_det_);
    
    auto occ_a = hf_det_.alpha_occupations();
    auto occ_b = hf_det_.beta_occupations();
    
    // Generate α single excitations: i_α → a_α
    for (int i : occ_a) {
        for (int a = nocc_a_; a < nocc_a_ + nvirt_a_; a++) {
            try {
                Determinant excited = hf_det_.single_excite(i, a, true);
                singles.push_back(excited);
            } catch (...) {
                // Skip invalid excitations
            }
        }
    }
    
    // Generate β single excitations: i_β → a_β
    for (int i : occ_b) {
        for (int a = nocc_b_; a < nocc_b_ + nvirt_b_; a++) {
            try {
                Determinant excited = hf_det_.single_excite(i, a, false);
                singles.push_back(excited);
            } catch (...) {
                // Skip invalid excitations
            }
        }
    }
    
    return singles;
}

std::vector<Determinant> CIS::get_determinants() const {
    return const_cast<CIS*>(this)->generate_singles();
}

// Main CIS computation
// REFERENCE: Foresman et al. (1992), J. Phys. Chem. 96, 135
CISResult CIS::compute(int n_states) {
    
    std::cout << "\n=== CIS (Configuration Interaction Singles) ===\n";
    std::cout << "Occupied: α=" << nocc_a_ << ", β=" << nocc_b_ << "\n";
    std::cout << "Virtual:  α=" << nvirt_a_ << ", β=" << nvirt_b_ << "\n";
    
    // Generate all single excitations
    auto dets = generate_singles();
    int n_dets = dets.size();
    
    std::cout << "Total determinants: " << n_dets << "\n";
    std::cout << "  HF reference: 1\n";
    std::cout << "  Singles:      " << (n_dets - 1) << "\n";
    
    // For small CIS, can use dense diagonalization
    if (n_dets <= 5000) {
        std::cout << "\nUsing dense diagonalization (small matrix)\n";
        
        // Build full Hamiltonian
        Eigen::MatrixXd H = build_hamiltonian(dets, ints_);
        
        // Diagonalize
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
        auto eigenvalues = solver.eigenvalues();
        auto eigenvectors = solver.eigenvectors();
        
        // Extract results
        CISResult result;
        result.ground_state = eigenvalues(0);
        result.n_states = std::min(n_states, n_dets - 1);
        result.converged = true;
        
        std::cout << "\n--- CIS Results ---\n";
        std::cout << "Ground state: " << std::fixed << std::setprecision(8) 
                  << result.ground_state << " Ha\n\n";
        
        std::cout << "State   Excitation (Ha)   Excitation (eV)   Wavelength (nm)\n";
        std::cout << "-------------------------------------------------------------\n";
        
        // Process excited states (skip ground state at index 0)
        for (int i = 1; i <= result.n_states; i++) {
            CISExcitation exc;
            exc.energy = eigenvalues(i) - result.ground_state;  // Excitation energy
            exc.wavelength = 45.56 / exc.energy;  // λ (nm) = 1240 eV·nm / ΔE
            exc.amplitudes = eigenvectors.col(i);
            
            // Compute oscillator strength (simplified, needs dipole integrals)
            exc.oscillator_strength = 0.0;  // TODO: implement with dipole integrals
            
            // Get dominant transitions
            exc.dominant = get_dominant_transitions(dets, exc.amplitudes, 0.1);
            
            result.excitations.push_back(exc);
            
            double exc_ev = exc.energy * 27.211;  // Ha to eV
            std::cout << std::setw(3) << i 
                      << std::fixed << std::setprecision(6)
                      << std::setw(18) << exc.energy
                      << std::setw(18) << exc_ev
                      << std::setw(18) << exc.wavelength << "\n";
            
            // Print dominant transitions
            if (!exc.dominant.empty()) {
                std::cout << "      Dominant: ";
                for (size_t j = 0; j < std::min(size_t(3), exc.dominant.size()); j++) {
                    auto& t = exc.dominant[j];
                    std::cout << t.i << "→" << t.a;
                    std::cout << (t.alpha ? "α" : "β");
                    std::cout << " (" << std::setprecision(3) << t.coeff << ") ";
                }
                std::cout << "\n";
            }
        }
        
        return result;
        
    } else {
        // Use Davidson for large CIS
        std::cout << "\nUsing Davidson solver (large matrix)\n";
        
        DavidsonOptions opts;
        opts.max_iter = 50;
        opts.conv_tol = 1e-6;
        opts.verbose = false;
        
        DavidsonSolver solver(opts);
        
        // Solve for multiple roots
        auto results = solver.solve_multiple(dets, ints_, n_states + 1);
        
        CISResult cis_result;
        cis_result.ground_state = results[0].energy;
        cis_result.n_states = results.size() - 1;
        cis_result.converged = true;
        
        std::cout << "\n--- CIS Results ---\n";
        std::cout << "Ground state: " << std::fixed << std::setprecision(8) 
                  << cis_result.ground_state << " Ha\n\n";
        
        std::cout << "State   Excitation (Ha)   Excitation (eV)   Wavelength (nm)\n";
        std::cout << "-------------------------------------------------------------\n";
        
        for (size_t i = 1; i < results.size(); i++) {
            CISExcitation exc;
            exc.energy = results[i].energy - cis_result.ground_state;
            exc.wavelength = 45.56 / exc.energy;
            exc.amplitudes = results[i].eigenvector;
            exc.oscillator_strength = 0.0;
            exc.dominant = get_dominant_transitions(dets, exc.amplitudes, 0.1);
            
            cis_result.excitations.push_back(exc);
            
            double exc_ev = exc.energy * 27.211;
            std::cout << std::setw(3) << i 
                      << std::fixed << std::setprecision(6)
                      << std::setw(18) << exc.energy
                      << std::setw(18) << exc_ev
                      << std::setw(18) << exc.wavelength << "\n";
            
            if (!exc.dominant.empty()) {
                std::cout << "      Dominant: ";
                for (size_t j = 0; j < std::min(size_t(3), exc.dominant.size()); j++) {
                    auto& t = exc.dominant[j];
                    std::cout << t.i << "→" << t.a;
                    std::cout << (t.alpha ? "α" : "β");
                    std::cout << " (" << std::setprecision(3) << t.coeff << ") ";
                }
                std::cout << "\n";
            }
        }
        
        return cis_result;
    }
}

// Extract dominant transitions
std::vector<CISExcitation::Transition> CIS::get_dominant_transitions(
    const std::vector<Determinant>& dets,
    const Eigen::VectorXd& c,
    double threshold) {
    
    std::vector<CISExcitation::Transition> trans;
    
    // Skip HF determinant (index 0)
    for (size_t idx = 1; idx < dets.size(); idx++) {
        double coeff = c(idx);
        if (std::abs(coeff) > threshold) {
            // Find excitation from HF to this determinant
            auto exc = find_excitation(dets[0], dets[idx]);
            
            if (exc.level == 1) {
                CISExcitation::Transition t;
                
                if (!exc.occ_alpha.empty()) {
                    t.i = exc.occ_alpha[0];
                    t.a = exc.virt_alpha[0];
                    t.alpha = true;
                } else {
                    t.i = exc.occ_beta[0];
                    t.a = exc.virt_beta[0];
                    t.alpha = false;
                }
                
                t.coeff = coeff;
                trans.push_back(t);
            }
        }
    }
    
    // Sort by coefficient magnitude
    std::sort(trans.begin(), trans.end(),
              [](const CISExcitation::Transition& a, 
                 const CISExcitation::Transition& b) {
                  return std::abs(a.coeff) > std::abs(b.coeff);
              });
    
    return trans;
}

// Compute oscillator strength
// REFERENCE: Helgaker et al. (2000), Eq. (10.148)
// f = (2/3) * ΔE * |⟨Ψ_0|μ|Ψ_i⟩|²
double CIS::compute_oscillator_strength(
    const Eigen::VectorXd& ground,
    const Eigen::VectorXd& excited,
    double delta_e) {
    
    // TODO: Need dipole moment integrals
    // For now, return 0 (placeholder)
    // Full implementation requires:
    // 1. Dipole integrals in AO basis
    // 2. Transform to MO basis
    // 3. Compute transition dipole ⟨Ψ_0|μ|Ψ_i⟩
    
    return 0.0;
}

} // namespace ci
} // namespace mshqc
