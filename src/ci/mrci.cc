/**
 * @file mrci.cc
 * @brief Multi-Reference CI implementation (simplified for Agent 3)
 * 
 * THEORY REFERENCES:
 *   - Werner & Knowles (1988), J. Chem. Phys. 89, 5803
 *   - Szalay et al. (2012), Chem. Rev. 112, 108
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 */

#include "mshqc/ci/mrci.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <set>
#include <stdexcept>

namespace mshqc {
namespace ci {

// Constructor
MRCI::MRCI(const CIIntegrals& ints,
           const std::vector<Determinant>& references,
           int n_core,
           int n_active,
           int n_virtual)
    : ints_(ints), references_(references),
      n_core_(n_core), n_active_(n_active), n_virtual_(n_virtual) {
    
    if (references.empty()) {
        throw std::invalid_argument("MRCI requires at least one reference determinant");
    }
    
    std::cout << "MRCI initialized with " << references.size() << " references\n";
    std::cout << "Space: " << n_core_ << " core, " << n_active_ << " active, " 
              << n_virtual_ << " virtual\n";
}

// Generate singles from a single determinant
std::vector<Determinant> MRCI::generate_singles_from_det(const Determinant& det) {
    std::vector<Determinant> singles;
    
    auto occ_a = det.alpha_occupations();
    auto occ_b = det.beta_occupations();
    
    // Alpha singles: i → a
    for (int i : occ_a) {
        // Only excite from active+core to virtual
        int n_total_occ = n_core_ + n_active_;
        for (int a = n_total_occ; a < n_total_occ + n_virtual_; a++) {
            try {
                singles.push_back(det.single_excite(i, a, true));
            } catch (...) {}
        }
    }
    
    // Beta singles: i → a
    for (int i : occ_b) {
        int n_total_occ = n_core_ + n_active_;
        for (int a = n_total_occ; a < n_total_occ + n_virtual_; a++) {
            try {
                singles.push_back(det.single_excite(i, a, false));
            } catch (...) {}
        }
    }
    
    return singles;
}

// Generate doubles from a single determinant  
std::vector<Determinant> MRCI::generate_doubles_from_det(const Determinant& det) {
    std::vector<Determinant> doubles;
    
    auto occ_a = det.alpha_occupations();
    auto occ_b = det.beta_occupations();
    
    int n_total_occ = n_core_ + n_active_;
    
    // αα doubles (same-spin)
    for (size_t idx_i = 0; idx_i < occ_a.size(); idx_i++) {
        for (size_t idx_j = idx_i + 1; idx_j < occ_a.size(); idx_j++) {
            int i = occ_a[idx_i];
            int j = occ_a[idx_j];
            
            for (int a = n_total_occ; a < n_total_occ + n_virtual_; a++) {
                for (int b = a + 1; b < n_total_occ + n_virtual_; b++) {
                    try {
                        doubles.push_back(det.double_excite(i, j, a, b, true, true));
                    } catch (...) {}
                }
            }
        }
    }
    
    // ββ doubles (same-spin)
    for (size_t idx_i = 0; idx_i < occ_b.size(); idx_i++) {
        for (size_t idx_j = idx_i + 1; idx_j < occ_b.size(); idx_j++) {
            int i = occ_b[idx_i];
            int j = occ_b[idx_j];
            
            for (int a = n_total_occ; a < n_total_occ + n_virtual_; a++) {
                for (int b = a + 1; b < n_total_occ + n_virtual_; b++) {
                    try {
                        doubles.push_back(det.double_excite(i, j, a, b, false, false));
                    } catch (...) {}
                }
            }
        }
    }
    
    // αβ doubles (opposite-spin)
    for (int i : occ_a) {
        for (int j : occ_b) {
            for (int a = n_total_occ; a < n_total_occ + n_virtual_; a++) {
                for (int b = n_total_occ; b < n_total_occ + n_virtual_; b++) {
                    try {
                        doubles.push_back(det.double_excite(i, j, a, b, true, false));
                    } catch (...) {}
                }
            }
        }
    }
    
    return doubles;
}

// Remove duplicates
std::vector<Determinant> MRCI::remove_duplicates(const std::vector<Determinant>& dets) {
    std::set<Determinant> unique_set(dets.begin(), dets.end());
    return std::vector<Determinant>(unique_set.begin(), unique_set.end());
}

// Check if determinant is a reference
bool MRCI::is_reference(const Determinant& det) const {
    return std::find(references_.begin(), references_.end(), det) != references_.end();
}

// Generate singles from all references
std::vector<Determinant> MRCI::generate_singles_from_references() {
    std::vector<Determinant> all_singles;
    
    for (const auto& ref : references_) {
        auto singles = generate_singles_from_det(ref);
        all_singles.insert(all_singles.end(), singles.begin(), singles.end());
    }
    
    // Remove duplicates
    return remove_duplicates(all_singles);
}

// Generate doubles from all references
std::vector<Determinant> MRCI::generate_doubles_from_references() {
    std::vector<Determinant> all_doubles;
    
    for (const auto& ref : references_) {
        auto doubles = generate_doubles_from_det(ref);
        all_doubles.insert(all_doubles.end(), doubles.begin(), doubles.end());
    }
    
    // Remove duplicates
    return remove_duplicates(all_doubles);
}

std::vector<Determinant> MRCI::get_determinants() const {
    std::vector<Determinant> all_dets = references_;
    
    auto singles = const_cast<MRCI*>(this)->generate_singles_from_references();
    all_dets.insert(all_dets.end(), singles.begin(), singles.end());
    
    auto doubles = const_cast<MRCI*>(this)->generate_doubles_from_references();
    all_dets.insert(all_dets.end(), doubles.begin(), doubles.end());
    
    return all_dets;
}

size_t MRCI::estimate_n_determinants() const {
    int n_occ = n_core_ + n_active_;
    int n_refs = references_.size();
    
    // Rough estimate: n_refs + singles + doubles
    size_t n_singles = n_refs * n_occ * n_virtual_;
    size_t n_doubles = n_refs * (n_occ * (n_occ-1) / 2) * (n_virtual_ * (n_virtual_-1) / 2);
    
    return n_refs + n_singles + n_doubles;
}

// Main MRCI computation
MRCIResult MRCI::compute() {
    std::cout << "\n=== MRCI (Multi-Reference Configuration Interaction) ===\n";
    std::cout << "References: " << references_.size() << "\n";
    
    // Generate all determinants
    std::cout << "Generating singles from references...\n";
    auto singles = generate_singles_from_references();
    
    std::cout << "Generating doubles from references...\n";
    auto doubles = generate_doubles_from_references();
    
    std::vector<Determinant> all_dets = references_;
    all_dets.insert(all_dets.end(), singles.begin(), singles.end());
    all_dets.insert(all_dets.end(), doubles.begin(), doubles.end());
    
    int n_dets = all_dets.size();
    
    std::cout << "\nDeterminant count:\n";
    std::cout << "  References: " << references_.size() << "\n";
    std::cout << "  Singles:    " << singles.size() << "\n";
    std::cout << "  Doubles:    " << doubles.size() << "\n";
    std::cout << "  Total:      " << n_dets << "\n\n";
    
    MRCIResult result;
    result.n_determinants = n_dets;
    result.n_references = references_.size();
    result.n_singles = singles.size();
    result.n_doubles = doubles.size();
    result.determinants = all_dets;
    
    // Compute reference energy (from first reference diagonal)
    result.e_ref = diagonal_element(references_[0], ints_);
    
    // Solve eigenvalue problem
    const size_t MAX_DENSE = 10000;
    
    if (n_dets <= static_cast<int>(MAX_DENSE)) {
        std::cout << "Using dense diagonalization\n";
        
        auto H = build_hamiltonian(all_dets, ints_);
        // Apply diagonal shift using reference energy
        double e0 = result.e_ref;
        H.diagonal().array() -= e0;
        
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
        
        // Add shift back for reporting
        result.e_mrci = solver.eigenvalues()(0) + e0;
        result.coefficients = solver.eigenvectors().col(0);
        result.converged = true;
        result.iterations = 0;
        
    } else {
        std::cout << "Using Davidson solver\n";
        
        DavidsonOptions opts;
        opts.max_iter = 100;
        opts.conv_tol = 1e-8;
        opts.verbose = true;
        
        DavidsonSolver solver(opts);
        auto guess = generate_davidson_guess(all_dets, ints_);
        auto davidson_result = solver.solve(all_dets, ints_, guess);
        
        double e0 = result.e_ref;
        result.e_mrci = davidson_result.energy + e0;
        result.coefficients = davidson_result.eigenvector;
        result.converged = davidson_result.converged;
        result.iterations = davidson_result.iterations;
    }
    
    result.e_corr = result.e_mrci - result.e_ref;
    
    // Analyze wavefunction
    analyze_wavefunction(result.coefficients, all_dets, result);
    
    // Print results
    std::cout << "\n=== MRCI Results ===\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Reference energy:   " << result.e_ref << " Ha\n";
    std::cout << "MRCI energy:        " << result.e_mrci << " Ha\n";
    std::cout << "Correlation energy: " << result.e_corr << " Ha\n";
    
    return result;
}

MRCIResult MRCI::compute_with_comparison(double casscf_energy) {
    auto result = compute();
    
    double diff = result.e_mrci - casscf_energy;
    
    std::cout << "\n=== Comparison with CASSCF ===\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "CASSCF energy: " << casscf_energy << " Ha\n";
    std::cout << "MRCI energy:   " << result.e_mrci << " Ha\n";
    std::cout << "Difference:    " << diff << " Ha\n";
    std::cout << "MRCI adds:     " << std::setprecision(6) 
              << (diff * 627.509) << " kcal/mol dynamic correlation\n";
    
    return result;
}

// Analyze wavefunction
void MRCI::analyze_wavefunction(const Eigen::VectorXd& c,
                               const std::vector<Determinant>& dets,
                               MRCIResult& result) {
    
    std::cout << "\n=== Wavefunction Analysis ===\n";
    
    result.ref_weight = 0.0;
    result.singles_weight = 0.0;
    result.doubles_weight = 0.0;
    
    // Compute weights
    for (size_t i = 0; i < dets.size(); i++) {
        double weight = c(i) * c(i);
        
        if (i < references_.size()) {
            result.ref_weight += weight;
        } else if (i < references_.size() + result.n_singles) {
            result.singles_weight += weight;
        } else {
            result.doubles_weight += weight;
        }
    }
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Reference weight: " << result.ref_weight 
              << " (" << (result.ref_weight * 100) << "%)\n";
    std::cout << "Singles weight:   " << result.singles_weight
              << " (" << (result.singles_weight * 100) << "%)\n";
    std::cout << "Doubles weight:   " << result.doubles_weight
              << " (" << (result.doubles_weight * 100) << "%)\n";
    
    // Find dominant configurations
    std::vector<std::pair<int, double>> configs;
    for (int i = 0; i < c.size(); i++) {
        double weight = c(i) * c(i);
        if (weight > 0.01) {  // 1% threshold
            configs.push_back({i, weight});
        }
    }
    
    std::sort(configs.begin(), configs.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });
    
    std::cout << "\nDominant configurations (> 1%):\n";
    std::cout << "Index   Weight     Type\n";
    std::cout << "────────────────────────────\n";
    
    int count = 0;
    for (const auto& [idx, weight] : configs) {
        if (count++ >= 10) break;
        
        std::cout << std::setw(5) << idx
                  << std::setw(10) << std::setprecision(4) << weight;
        
        if (static_cast<size_t>(idx) < references_.size()) {
            std::cout << "  Reference";
        } else if (static_cast<size_t>(idx) < references_.size() + result.n_singles) {
            std::cout << "  Single";
        } else {
            std::cout << "  Double";
        }
        std::cout << "\n";
    }
}

// Helper functions
std::vector<Determinant> mrci_references_from_casscf(
    const std::vector<Determinant>& casscf_dets,
    const Eigen::VectorXd& casscf_coeffs,
    double threshold) {
    
    std::vector<Determinant> refs;
    
    for (int i = 0; i < casscf_coeffs.size(); i++) {
        double weight = casscf_coeffs(i) * casscf_coeffs(i);
        if (weight > threshold) {
            refs.push_back(casscf_dets[i]);
        }
    }
    
    std::cout << "Selected " << refs.size() << " references from CASSCF (threshold=" 
              << threshold << ")\n";
    
    return refs;
}

size_t estimate_mrci_size(int n_refs, int n_occ_core, 
                          int n_occ_active, int n_virt) {
    int n_occ = n_occ_core + n_occ_active;
    
    size_t n_singles = n_refs * n_occ * n_virt;
    size_t n_doubles = n_refs * (n_occ * (n_occ-1) / 2) * (n_virt * (n_virt-1) / 2);
    
    return n_refs + n_singles + n_doubles;
}

} // namespace ci
} // namespace mshqc
