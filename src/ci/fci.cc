/**
 * @file fci.cc
 * @brief Full Configuration Interaction implementation
 * 
 * THEORY REFERENCES:
 *   - Knowles & Handy (1984), Chem. Phys. Lett. 111, 315
 *   - Olsen et al. (1988), J. Chem. Phys. 89, 2185
 *   - Helgaker et al. (2000), Ch. 11
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 */

#include "mshqc/ci/fci.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mshqc {
namespace ci {

// Constructor
FCI::FCI(const CIIntegrals& ints,
         int n_orbitals,
         int n_alpha,
         int n_beta,
         int n_roots)
    : ints_(ints), n_orb_(n_orbitals), 
      n_alpha_(n_alpha), n_beta_(n_beta), n_roots_(n_roots) {
    
    if (n_alpha < 0 || n_beta < 0) {
        throw std::invalid_argument("Number of electrons must be non-negative");
    }
    if (n_alpha > n_orb_ || n_beta > n_orb_) {
        throw std::invalid_argument("Number of electrons exceeds number of orbitals");
    }
}

// Binomial coefficient C(n, k) = n! / (k! * (n-k)!)
size_t fci_binomial(int n, int k) {
    if (k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k;  // Symmetry C(n,k) = C(n,n-k)
    
    size_t result = 1;
    for (int i = 0; i < k; i++) {
        result *= (n - i);
        result /= (i + 1);
    }
    return result;
}

// Estimate determinant count
size_t FCI::estimate_n_determinants() const {
    size_t n_alpha_strings = fci_binomial(n_orb_, n_alpha_);
    size_t n_beta_strings = fci_binomial(n_orb_, n_beta_);
    return n_alpha_strings * n_beta_strings;
}

// Generate all combinations C(n, k)
// REFERENCE: Knuth, TAOCP Vol. 4A, Algorithm 7.2.1.3T
std::vector<std::vector<int>> FCI::generate_combinations(int n, int k) {
    std::vector<std::vector<int>> combinations;
    
    if (k == 0) {
        combinations.push_back({});
        return combinations;
    }
    
    if (k > n) {
        return combinations;
    }
    
    // Initial combination: [0, 1, 2, ..., k-1]
    std::vector<int> current(k);
    for (int i = 0; i < k; i++) {
        current[i] = i;
    }
    
    while (true) {
        combinations.push_back(current);
        
        // Find rightmost element that can be incremented
        int i = k - 1;
        while (i >= 0 && current[i] == n - k + i) {
            i--;
        }
        
        // All combinations generated
        if (i < 0) break;
        
        // Increment and reset subsequent elements
        current[i]++;
        for (int j = i + 1; j < k; j++) {
            current[j] = current[j-1] + 1;
        }
    }
    
    return combinations;
}

// Generate all determinants
// REFERENCE: Knowles & Handy (1984), Chem. Phys. Lett. 111, 315
std::vector<Determinant> FCI::generate_all_determinants() {
    std::vector<Determinant> dets;
    
    // Generate all alpha occupation strings
    auto alpha_strings = generate_combinations(n_orb_, n_alpha_);
    
    // Generate all beta occupation strings
    auto beta_strings = generate_combinations(n_orb_, n_beta_);
    
    // Cartesian product: all (alpha, beta) pairs
    dets.reserve(alpha_strings.size() * beta_strings.size());
    
    for (const auto& alpha : alpha_strings) {
        for (const auto& beta : beta_strings) {
            dets.emplace_back(alpha, beta);
        }
    }
    
    return dets;
}

std::vector<Determinant> FCI::get_determinants() const {
    return const_cast<FCI*>(this)->generate_all_determinants();
}

// Main FCI computation
// REFERENCE: Olsen et al. (1988), J. Chem. Phys. 89, 2185
FCIResult FCI::compute() {
    std::cout << "\n=== FCI (Full Configuration Interaction) ===\n";
    std::cout << "Electrons: α=" << n_alpha_ << ", β=" << n_beta_ 
              << " (total=" << (n_alpha_ + n_beta_) << ")\n";
    std::cout << "Orbitals: " << n_orb_ << "\n";
    
    // Estimate size
    size_t estimated = estimate_n_determinants();
    std::cout << "Estimated determinants: " << estimated << "\n";
    
    // Check feasibility
    const size_t MAX_DENSE = 50000;  // 50k determinants for dense
    const size_t MAX_DAVIDSON = 5000000;  // 5M for Davidson
    
    if (estimated > MAX_DAVIDSON) {
        throw std::runtime_error(
            "FCI too large: " + std::to_string(estimated) + " determinants. "
            "Maximum " + std::to_string(MAX_DAVIDSON) + " allowed."
        );
    }
    
    std::cout << "\nGenerating all determinants...\n";
    auto dets = generate_all_determinants();
    int n_dets = dets.size();
    
    std::cout << "Actual determinants: " << n_dets << "\n";
    
    if (n_dets != static_cast<int>(estimated)) {
        std::cout << "Warning: Generated " << n_dets 
                  << " but estimated " << estimated << "\n";
    }
    
    FCIResult result;
    result.n_determinants = n_dets;
    result.determinants = dets;
    
    // Find HF determinant (lowest orbitals occupied)
    std::vector<int> hf_alpha(n_alpha_), hf_beta(n_beta_);
    for (int i = 0; i < n_alpha_; i++) hf_alpha[i] = i;
    for (int i = 0; i < n_beta_; i++) hf_beta[i] = i;
    Determinant hf_det(hf_alpha, hf_beta);
    
    result.e_hf = diagonal_element(hf_det, ints_);
    
    // Solve eigenvalue problem
    std::cout << "\n";
    
    if (n_dets <= static_cast<int>(MAX_DENSE)) {
        // Dense diagonalization
        std::cout << "Using dense diagonalization\n";
        std::cout << "Building Hamiltonian matrix (" << n_dets << "×" << n_dets << ")...\n";
        
        auto H = build_hamiltonian(dets, ints_);
        
        // DEBUG: Print H(0,0) before shift
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "\n=== DEBUG: CI Hamiltonian Diagonal ===\n";
        std::cout << "H(0,0) = " << H(0,0) << " Ha\n";
        std::cout << "E(HF determinant) from integrals = " << result.e_hf << " Ha\n";
        std::cout << "Difference = " << (H(0,0) - result.e_hf)*1000 << " mHa\n";
        
        // Apply diagonal shift so HF reference is zero (standard CI convention)
        double e0 = result.e_hf;
        H.diagonal().array() -= e0;
        
        std::cout << "Diagonalizing...\n";
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
        
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Dense diagonalization failed");
        }
        
        // Add shift back for reporting
        result.e_fci = solver.eigenvalues()(0) + e0;
        result.coefficients = solver.eigenvectors().col(0);
        result.converged = true;
        result.iterations = 0;
        
        // Store excited states if requested
        for (int i = 1; i < std::min(n_roots_, n_dets); i++) {
            result.excited_energies.push_back(solver.eigenvalues()(i));
            result.excited_states.push_back(solver.eigenvectors().col(i));
        }
        
    } else {
        // Davidson solver
        std::cout << "Using Davidson iterative solver\n";
        
        DavidsonOptions opts;
        opts.max_iter = 150;
        opts.conv_tol = 1e-10;  // FCI should be very accurate
        opts.residual_tol = 1e-8;
        opts.verbose = true;
        
        DavidsonSolver solver(opts);
        
        if (n_roots_ > 1) {
            // Multiple roots requested
            auto results = solver.solve_multiple(dets, ints_, n_roots_);
            
            if (!results[0].converged) {
                std::cout << "Warning: Davidson did not fully converge\n";
            }
            
            double e0 = result.e_hf;
            result.e_fci = results[0].energy + e0;
            result.coefficients = results[0].eigenvector;
            result.converged = results[0].converged;
            result.iterations = results[0].iterations;
            
            // Store excited states
            for (int i = 1; i < n_roots_ && i < static_cast<int>(results.size()); i++) {
                result.excited_energies.push_back(results[i].energy);
                result.excited_states.push_back(results[i].eigenvector);
            }
        } else {
            // Single root (ground state only)
            auto guess = generate_davidson_guess(dets, ints_);
            auto davidson_result = solver.solve(dets, ints_, guess);
            
            if (!davidson_result.converged) {
                std::cout << "Warning: Davidson did not fully converge\n";
            }
            
            double e0 = result.e_hf;
            result.e_fci = davidson_result.energy + e0;
            result.coefficients = davidson_result.eigenvector;
            result.converged = davidson_result.converged;
            result.iterations = davidson_result.iterations;
        }
    }
    
    // Correlation energy
    result.e_corr = result.e_fci - result.e_hf;
    
    // Analyze wavefunction
    analyze_wavefunction(result.coefficients, dets, result);
    
    // Print results
    std::cout << "\n=== FCI Results ===\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "HF energy:          " << result.e_hf << " Ha\n";
    std::cout << "FCI energy:         " << result.e_fci << " Ha\n";
    std::cout << "Correlation energy: " << result.e_corr << " Ha\n";
    std::cout << "% Correlation:      " << std::setprecision(2) 
              << (result.e_corr / result.e_fci * 100) << "%\n";
    
    if (!result.excited_energies.empty()) {
        std::cout << "\nExcited states:\n";
        for (size_t i = 0; i < result.excited_energies.size(); i++) {
            double exc_energy = result.excited_energies[i] - result.e_fci;
            std::cout << "  State " << (i+1) << ": " 
                      << std::setprecision(6) << exc_energy << " Ha ("
                      << (exc_energy * 27.211) << " eV)\n";
        }
    }
    
    return result;
}

// Compute with comparison
FCIResult FCI::compute_with_comparison(double approx_energy,
                                       const std::string& method_name) {
    auto result = compute();
    
    double error = approx_energy - result.e_fci;
    double corr_recovery = (approx_energy - result.e_hf) / result.e_corr * 100.0;
    
    std::cout << "\n=== Comparison with " << method_name << " ===\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << method_name << " energy: " << approx_energy << " Ha\n";
    std::cout << "FCI energy:         " << result.e_fci << " Ha\n";
    std::cout << "Error:              " << error << " Ha\n";
    std::cout << "Error (μHa):        " << (error * 1e6) << " μHa\n";
    std::cout << "Correlation recovery: " << std::setprecision(2) 
              << corr_recovery << "%\n";
    
    if (std::abs(error) < 1e-6) {
        std::cout << "✓ Excellent agreement (< 1 μHa)\n";
    } else if (std::abs(error) < 1e-5) {
        std::cout << "✓ Good agreement (< 10 μHa)\n";
    } else if (std::abs(error) < 1e-4) {
        std::cout << "○ Acceptable agreement (< 100 μHa)\n";
    } else {
        std::cout << "✗ Poor agreement (> 100 μHa)\n";
    }
    
    return result;
}

// Analyze wavefunction by excitation level
void FCI::analyze_wavefunction(const Eigen::VectorXd& c,
                               const std::vector<Determinant>& dets,
                               FCIResult& result) {
    
    std::cout << "\n=== Wavefunction Analysis ===\n";
    
    // Find HF determinant (should be first or near first)
    std::vector<int> hf_alpha(n_alpha_), hf_beta(n_beta_);
    for (int i = 0; i < n_alpha_; i++) hf_alpha[i] = i;
    for (int i = 0; i < n_beta_; i++) hf_beta[i] = i;
    Determinant hf_det(hf_alpha, hf_beta);
    
    // Find HF in determinant list
    int hf_idx = -1;
    for (size_t i = 0; i < dets.size(); i++) {
        if (dets[i] == hf_det) {
            hf_idx = i;
            break;
        }
    }
    
    result.hf_weight = 0.0;
    result.singles_weight = 0.0;
    result.doubles_weight = 0.0;
    result.higher_weight = 0.0;
    
    if (hf_idx >= 0) {
        result.hf_weight = c(hf_idx) * c(hf_idx);
        
        // Analyze by excitation level
        for (size_t i = 0; i < dets.size(); i++) {
            if (static_cast<int>(i) == hf_idx) continue;
            
            auto exc = find_excitation(hf_det, dets[i]);
            double weight = c(i) * c(i);
            
            if (exc.level == 1) {
                result.singles_weight += weight;
            } else if (exc.level == 2) {
                result.doubles_weight += weight;
            } else if (exc.level > 2) {
                result.higher_weight += weight;
            }
        }
    } else {
        std::cout << "Warning: HF determinant not found in list\n";
        // Still compute by excitation from first determinant
        for (size_t i = 1; i < dets.size(); i++) {
            auto exc = find_excitation(dets[0], dets[i]);
            double weight = c(i) * c(i);
            
            if (exc.level == 1) result.singles_weight += weight;
            else if (exc.level == 2) result.doubles_weight += weight;
            else if (exc.level > 2) result.higher_weight += weight;
        }
        result.hf_weight = c(0) * c(0);
    }
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "HF weight:      " << result.hf_weight 
              << " (" << (result.hf_weight * 100) << "%)\n";
    std::cout << "Singles weight: " << result.singles_weight
              << " (" << (result.singles_weight * 100) << "%)\n";
    std::cout << "Doubles weight: " << result.doubles_weight
              << " (" << (result.doubles_weight * 100) << "%)\n";
    std::cout << "Higher weight:  " << result.higher_weight
              << " (" << (result.higher_weight * 100) << "%)\n";
    
    // Find dominant configurations
    std::vector<std::pair<int, double>> configs;
    for (int i = 0; i < c.size(); i++) {
        double weight = c(i) * c(i);
        if (weight > 0.01) {  // Threshold 1%
            configs.push_back({i, weight});
        }
    }
    
    // Sort by weight
    std::sort(configs.begin(), configs.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });
    
    std::cout << "\nDominant configurations (> 1%):\n";
    std::cout << "Index   Weight     Type\n";
    std::cout << "──────────────────────────\n";
    
    int count = 0;
    for (const auto& [idx, weight] : configs) {
        if (count++ >= 15) break;  // Top 15
        
        std::cout << std::setw(5) << idx
                  << std::setw(10) << std::setprecision(4) << weight;
        
        if (hf_idx >= 0 && idx == hf_idx) {
            std::cout << "  HF";
        } else if (hf_idx >= 0) {
            auto exc = find_excitation(hf_det, dets[idx]);
            if (exc.level == 1) std::cout << "  Single";
            else if (exc.level == 2) std::cout << "  Double";
            else if (exc.level == 3) std::cout << "  Triple";
            else if (exc.level == 4) std::cout << "  Quadruple";
            else std::cout << "  Higher";
        }
        std::cout << "\n";
    }
}

// Helper functions
size_t fci_determinant_count(int n_orbitals, int n_alpha, int n_beta) {
    size_t n_alpha_strings = fci_binomial(n_orbitals, n_alpha);
    size_t n_beta_strings = fci_binomial(n_orbitals, n_beta);
    return n_alpha_strings * n_beta_strings;
}

bool is_fci_feasible(int n_orbitals, int n_alpha, int n_beta, size_t max_dets) {
    size_t n_dets = fci_determinant_count(n_orbitals, n_alpha, n_beta);
    return n_dets <= max_dets;
}

} // namespace ci
} // namespace mshqc