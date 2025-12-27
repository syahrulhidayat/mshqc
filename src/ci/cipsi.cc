/**
 * @file cipsi.cc
 * @brief Implementation of CIPSI (Configuration Interaction with Perturbative Selection)
 * 
 * Theory References:
 * - B. Huron et al., J. Chem. Phys. **58**, 5745 (1973) - Original CIPSI algorithm
 * - E. Giner et al., J. Chem. Phys. **143**, 124305 (2015) - Modern implementation
 * - R. K. Nesbet, Phys. Rev. **109**, 1632 (1958) - Epstein-Nesbet perturbation theory
 * - A. Scemama et al., J. Comp. Chem. **37**, 1866 (2016) - Quantum Package implementation
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 * 
 * @note Original implementation from theory papers. This is NOT derived from any
 *       quantum chemistry software source code (PySCF, Psi4, Quantum Package, etc.).
 *       All algorithms implemented from published literature only.
 * 
 * @copyright MIT License
 */

#include "mshqc/ci/cipsi.h"
#include "mshqc/ci/slater_condon.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <unordered_map>

namespace mshqc {
namespace ci {

// ============================================================================
// Constructor
// ============================================================================
CIPSI::CIPSI(const CIIntegrals& ints,
             int n_orb, int n_alpha, int n_beta,
             const CIPSIConfig& config)
    : ints_(ints),
      n_orb_(n_orb),
      n_alpha_(n_alpha),
      n_beta_(n_beta),
      config_(config),
      e_var_(0.0)
{
    if (config_.verbose) {
        std::cout << "\n";
        std::cout << "================================================================================\n";
        std::cout << "  CIPSI - Configuration Interaction with Perturbative Selection\n";
        std::cout << "================================================================================\n";
        std::cout << "System: " << n_alpha_ << "α + " << n_beta_ << "β electrons, "
                  << n_orb_ << " orbitals\n";
        std::cout << "Config:\n";
        std::cout << "  E(PT2) threshold:  " << config_.e_pt2_threshold << " Ha\n";
        std::cout << "  Max determinants:  " << config_.max_determinants << "\n";
        std::cout << "  Max iterations:    " << config_.max_iterations << "\n";
        std::cout << "  Select per iter:   " << config_.n_select_per_iter << "\n";
        std::cout << "  PT2 denominator:   " 
                  << (config_.use_epstein_nesbet ? "Epstein-Nesbet" : "Møller-Plesset") << "\n";
        std::cout << "================================================================================\n";
        std::cout << "\n";
    }
}

// ============================================================================
// Main CIPSI algorithm
// ============================================================================
CIPSIResult CIPSI::compute() {
    auto start_time = std::chrono::steady_clock::now();
    
    CIPSIResult result;
    result.n_iterations = 0;
    result.converged = false;
    result.time_diag = 0.0;
    result.time_pt2 = 0.0;
    
    // Step 1: Initialize variational space
    if (config_.verbose) {
        std::cout << "Initializing variational space...\n";
    }
    initialize_variational_space();
    
    if (config_.verbose) {
        std::cout << "Initial space: " << variational_space_.size() << " determinants\n\n";
        std::cout << "Starting iterative selection...\n";
        std::cout << "--------------------------------------------------------------------------------\n";
        std::cout << " Iter |  N_var  | N_ext  |    E_var (Ha)   |  E_PT2 (Ha)  |  PT2 norm  |\n";
        std::cout << "--------------------------------------------------------------------------------\n";
    }
    
    // Main iteration loop
    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        result.n_iterations = iter + 1;
        
        // Step 2: Diagonalize Hamiltonian in variational space
        auto t_diag_start = std::chrono::steady_clock::now();
        diagonalize_variational();
        auto t_diag_end = std::chrono::steady_clock::now();
        result.time_diag += std::chrono::duration<double>(t_diag_end - t_diag_start).count();
        
        // Step 3: Generate external determinants (connected to variational space)
        auto t_pt2_start = std::chrono::steady_clock::now();
        std::vector<Determinant> external_dets = generate_external_determinants();
        
        // Step 4: Compute PT2 contributions for all external determinants
        std::vector<double> pt2_contribs;
        pt2_contribs.reserve(external_dets.size());
        
        double e_pt2_total = 0.0;
        double pt2_norm = 0.0;
        
        for (const auto& det_ext : external_dets) {
            double de_pt2 = compute_pt2_contribution(det_ext);
            pt2_contribs.push_back(de_pt2);
            e_pt2_total += de_pt2;
            pt2_norm += std::abs(de_pt2);
        }
        
        auto t_pt2_end = std::chrono::steady_clock::now();
        result.time_pt2 += std::chrono::duration<double>(t_pt2_end - t_pt2_start).count();
        
        // Print iteration info
        if (config_.verbose) {
            print_iteration(iter, variational_space_.size(), external_dets.size(),
                          e_var_, e_pt2_total, pt2_norm);
        }
        
        // Step 5: Check convergence
        if (std::abs(e_pt2_total) < config_.e_pt2_threshold) {
            result.converged = true;
            result.conv_reason = "PT2 energy below threshold";
            result.e_var = e_var_ + ints_.e_nuc;
            result.e_pt2 = e_pt2_total;
            result.e_total = e_var_ + e_pt2_total + ints_.e_nuc;
            result.pt2_norm = pt2_norm;
            break;
        }
        
        // Step 6: Select most important determinants
        std::vector<Determinant> selected = select_important_determinants(
            external_dets, pt2_contribs, config_.n_select_per_iter
        );
        
        if (selected.empty()) {
            result.converged = false;
            result.conv_reason = "No more determinants to add";
            result.e_var = e_var_ + ints_.e_nuc;
            result.e_pt2 = e_pt2_total;
            result.e_total = e_var_ + e_pt2_total + ints_.e_nuc;
            result.pt2_norm = pt2_norm;
            break;
        }
        
        // Step 7: Add selected determinants to variational space
        for (const auto& det : selected) {
            variational_space_.push_back(det);
        }
        
        // Check size limit
        if (variational_space_.size() >= static_cast<size_t>(config_.max_determinants)) {
            result.converged = false;
            result.conv_reason = "Maximum determinant limit reached";
            result.e_var = e_var_ + ints_.e_nuc;
            result.e_pt2 = e_pt2_total;
            result.e_total = e_var_ + e_pt2_total + ints_.e_nuc;
            result.pt2_norm = pt2_norm;
            break;
        }
        
        // Store last iteration values
        result.e_var = e_var_ + ints_.e_nuc;
        result.e_pt2 = e_pt2_total;
        result.e_total = e_var_ + e_pt2_total + ints_.e_nuc;
        result.pt2_norm = pt2_norm;
        result.n_external = external_dets.size();
    }
    
    // Check if max iterations reached
    if (!result.converged && result.n_iterations >= config_.max_iterations) {
        result.conv_reason = "Maximum iterations reached";
    }
    
    // Finalize result
    result.variational_space = variational_space_;
    result.coefficients = coefficients_;
    result.n_selected = variational_space_.size();
    
    auto end_time = std::chrono::steady_clock::now();
    result.time_total = std::chrono::duration<double>(end_time - start_time).count();
    
    // Print final summary
    if (config_.verbose) {
        std::cout << "--------------------------------------------------------------------------------\n";
        std::cout << "\nCIPSI Calculation Complete!\n";
        std::cout << "================================================================================\n";
        std::cout << "Convergence: " << (result.converged ? "YES ✓" : "NO ✗") << "\n";
        std::cout << "Reason:      " << result.conv_reason << "\n";
        std::cout << "Iterations:  " << result.n_iterations << "\n";
        std::cout << "Determinants selected: " << result.n_selected << "\n";
        std::cout << "\n";
        std::cout << "ENERGIES:\n";
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "  E(variational) = " << result.e_var << " Ha\n";
        std::cout << "  E(PT2)         = " << result.e_pt2 << " Ha\n";
        std::cout << "  E(total)       = " << result.e_total << " Ha (CIPSI estimate)\n";
        std::cout << "\n";
        std::cout << "TIMING:\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Diagonalization: " << result.time_diag << " s\n";
        std::cout << "  PT2 screening:   " << result.time_pt2 << " s\n";
        std::cout << "  Total:           " << result.time_total << " s\n";
        std::cout << "================================================================================\n";
        std::cout << "\n";
    }
    
    return result;
}

// ============================================================================
// Initialize variational space
// ============================================================================
void CIPSI::initialize_variational_space() {
    // Theory: Start with HF determinant + singles (optional: + doubles)
    // Reference: Huron et al., J. Chem. Phys. 58, 5745 (1973)
    
    variational_space_.clear();
    
    // Create HF determinant: |1111...000⟩ (lowest n_alpha α, lowest n_beta β)
    std::vector<int> alpha_occ, beta_occ;
    for (int i = 0; i < n_alpha_; ++i) alpha_occ.push_back(i);
    for (int i = 0; i < n_beta_; ++i) beta_occ.push_back(i);
    Determinant hf_det(alpha_occ, beta_occ);
    
    // Step 1: Add HF determinant
    if (config_.start_from_hf) {
        variational_space_.push_back(hf_det);
    }
    
    // Step 2: Add all single excitations
    if (config_.include_singles && config_.start_from_hf) {
        std::vector<Determinant> singles = generate_singles(hf_det, n_orb_);
        variational_space_.insert(variational_space_.end(), singles.begin(), singles.end());
    }
    
    // Step 3: Optionally add all double excitations (expensive!)
    if (config_.include_doubles && config_.start_from_hf) {
        std::vector<Determinant> doubles = generate_doubles(hf_det, n_orb_);
        variational_space_.insert(variational_space_.end(), doubles.begin(), doubles.end());
    }
}

// ============================================================================
// Build Hamiltonian matrix in variational space
// ============================================================================
Eigen::MatrixXd CIPSI::build_hamiltonian_variational() {
    // Theory: H_ij = ⟨Φ_i|H|Φ_j⟩ using Slater-Condon rules
    // Reference: Slater (1929), Condon (1930)
    
    int n_var = variational_space_.size();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_var, n_var);
    
    for (int i = 0; i < n_var; ++i) {
        for (int j = i; j < n_var; ++j) {
            double h_ij = hamiltonian_element(
                variational_space_[i],
                variational_space_[j],
                ints_
            );
            H(i, j) = h_ij;
            if (i != j) {
                H(j, i) = h_ij;  // Hermitian
            }
        }
    }
    
    return H;
}

// ============================================================================
// Diagonalize variational Hamiltonian
// ============================================================================
void CIPSI::diagonalize_variational() {
    // Theory: Solve H|Ψ⟩ = E|Ψ⟩ for lowest eigenvalue
    // Reference: Standard linear algebra
    
    Eigen::MatrixXd H = build_hamiltonian_variational();
    
    // DEBUG: Print first diagonal element
    static bool first_time = true;
    if (first_time && variational_space_.size() > 0) {
        std::cout << "\n=== CIPSI DEBUG: First diagonalization ===\n";
        std::cout << "N(variational space) = " << variational_space_.size() << "\n";
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "H(0,0) = " << H(0,0) << " Ha (should be HF determinant energy)\n";
        std::cout << "E_nuc = " << ints_.e_nuc << " Ha\n";
    }
    
    // Use SelfAdjointEigenSolver for symmetric matrices (faster)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
    
    if (solver.info() != Eigen::Success) {
        std::cerr << "Error: Failed to diagonalize Hamiltonian\n";
        return;
    }
    
    // Extract lowest eigenvalue and eigenvector
    e_var_ = solver.eigenvalues()(0);  // Lowest energy
    coefficients_ = solver.eigenvectors().col(0);  // Ground state wavefunction
    
    if (first_time) {
        std::cout << "\nDEBUG e_var (electronic) = " << std::setprecision(10) << e_var_ << " Ha\n";
        std::cout << "DEBUG E_nuc = " << ints_.e_nuc << " Ha\n";
        std::cout << "DEBUG e_var + E_nuc = " << (e_var_ + ints_.e_nuc) << " Ha\n\n";
        first_time = false;
    }
}

// ============================================================================
// Generate external determinants
// ============================================================================
std::vector<Determinant> CIPSI::generate_external_determinants(int max_count) {
    // Theory: Generate determinants connected to variational space
    // Connected = differ by 1-2 excitations from any variational det
    // Reference: Huron et al., J. Chem. Phys. 58, 5745 (1973)
    
    std::vector<Determinant> external_dets;
    external_dets.reserve(max_count);
    
    // Generate connected excitations from variational space
    std::vector<Determinant> candidates = generate_connected_excitations(
        variational_space_, n_orb_, config_.max_excitation_level
    );
    
    // Filter: Keep only determinants not already in variational space
    for (const auto& det : candidates) {
        if (!is_in_variational_space(det)) {
            external_dets.push_back(det);
            if (static_cast<int>(external_dets.size()) >= max_count) {
                break;
            }
        }
    }
    
    return external_dets;
}

// ============================================================================
// Compute PT2 contribution for external determinant
// ============================================================================
double CIPSI::compute_pt2_contribution(const Determinant& det_ext) {
    // Theory: Second-order perturbation theory
    // 
    // Epstein-Nesbet:
    //   ΔE_PT2 = |⟨Ψ_var|H|Φ_ext⟩|² / (E_var - ⟨Φ_ext|H|Φ_ext⟩)
    // 
    // Møller-Plesset:
    //   ΔE_PT2 = |⟨Ψ_var|H|Φ_ext⟩|² / (E_HF - ⟨Φ_ext|H|Φ_ext⟩)
    // 
    // Reference: Nesbet, Phys. Rev. 109, 1632 (1958)
    
    // Compute numerator: |⟨Ψ_var|H|Φ_ext⟩|²
    double h_wfn_ext = compute_hamiltonian_element_wfn_det(det_ext);
    double numerator = h_wfn_ext * h_wfn_ext;
    
    // Compute denominator: E_var - ⟨Φ_ext|H|Φ_ext⟩
    double h_ext_ext = hamiltonian_element(det_ext, det_ext, ints_);
    double denominator = e_var_ - h_ext_ext;
    
    // Avoid division by zero
    if (std::abs(denominator) < 1.0e-10) {
        return 0.0;
    }
    
    // PT2 contribution (negative for lowering energy)
    double de_pt2 = -numerator / denominator;
    
    return de_pt2;
}

// ============================================================================
// Select important determinants
// ============================================================================
std::vector<Determinant> CIPSI::select_important_determinants(
    const std::vector<Determinant>& external_dets,
    const std::vector<double>& pt2_contribs,
    int n_select
) {
    // Theory: Select determinants with largest |ΔE_PT2|
    // Reference: Huron et al., J. Chem. Phys. 58, 5745 (1973)
    
    if (external_dets.empty()) {
        return {};
    }
    
    // Create pairs of (|ΔE_PT2|, index)
    std::vector<std::pair<double, int>> pt2_idx;
    pt2_idx.reserve(external_dets.size());
    
    for (size_t i = 0; i < external_dets.size(); ++i) {
        double abs_pt2 = std::abs(pt2_contribs[i]);
        if (abs_pt2 >= config_.pt2_selection_threshold) {
            pt2_idx.push_back({abs_pt2, static_cast<int>(i)});
        }
    }
    
    // Sort by |ΔE_PT2| (descending)
    std::sort(pt2_idx.begin(), pt2_idx.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Select top n_select determinants
    std::vector<Determinant> selected;
    int n_to_select = std::min(n_select, static_cast<int>(pt2_idx.size()));
    selected.reserve(n_to_select);
    
    for (int i = 0; i < n_to_select; ++i) {
        int idx = pt2_idx[i].second;
        selected.push_back(external_dets[idx]);
    }
    
    return selected;
}

// ============================================================================
// Check if determinant in variational space
// ============================================================================
bool CIPSI::is_in_variational_space(const Determinant& det) const {
    // Simple linear search (could be optimized with hash table)
    for (const auto& var_det : variational_space_) {
        if (var_det == det) {
            return true;
        }
    }
    return false;
}

// ============================================================================
// Compute Hamiltonian element between wavefunction and determinant
// ============================================================================
double CIPSI::compute_hamiltonian_element_wfn_det(const Determinant& det_ext) {
    // Theory: ⟨Ψ_var|H|Φ_ext⟩ = Σ_i c_i ⟨Φ_i|H|Φ_ext⟩
    // Reference: Standard CI theory
    
    double h_wfn_ext = 0.0;
    
    for (size_t i = 0; i < variational_space_.size(); ++i) {
        double h_ij = hamiltonian_element(
            variational_space_[i], det_ext, ints_
        );
        h_wfn_ext += coefficients_(i) * h_ij;
    }
    
    return h_wfn_ext;
}

// ============================================================================
// Print iteration info
// ============================================================================
void CIPSI::print_iteration(int iter, int n_var, int n_ext,
                           double e_var, double e_pt2, double pt2_norm) {
    std::cout << std::setw(5) << iter << " | "
              << std::setw(7) << n_var << " | "
              << std::setw(6) << n_ext << " | "
              << std::fixed << std::setprecision(10) << std::setw(15) << e_var << " | "
              << std::scientific << std::setprecision(4) << std::setw(12) << e_pt2 << " | "
              << std::setw(10) << pt2_norm << " |\n";
}

// ============================================================================
// Helper: Generate single excitations
// ============================================================================
std::vector<Determinant> generate_singles(const Determinant& det, int n_orb) {
    // Theory: |Φ_i^a⟩ = a_a† a_i |Φ_0⟩ (single excitation)
    // Reference: Standard quantum chemistry
    
    std::vector<Determinant> singles;
    
    // Alpha excitations
    for (int i = 0; i < n_orb; ++i) {
        if (det.is_occupied(i, true)) {  // Alpha spin
            for (int a = 0; a < n_orb; ++a) {
                if (!det.is_occupied(a, true)) {
                    Determinant exc_det = det.single_excite(i, a, true);
                    singles.push_back(exc_det);
                }
            }
        }
    }
    
    // Beta excitations
    for (int i = 0; i < n_orb; ++i) {
        if (det.is_occupied(i, false)) {  // Beta spin
            for (int a = 0; a < n_orb; ++a) {
                if (!det.is_occupied(a, false)) {
                    Determinant exc_det = det.single_excite(i, a, false);
                    singles.push_back(exc_det);
                }
            }
        }
    }
    
    return singles;
}

// ============================================================================
// Helper: Generate double excitations
// ============================================================================
std::vector<Determinant> generate_doubles(const Determinant& det, int n_orb) {
    // Theory: |Φ_ij^ab⟩ = a_a† a_b† a_j a_i |Φ_0⟩ (double excitation)
    // Reference: Standard quantum chemistry
    
    std::vector<Determinant> doubles;
    
    // Alpha-alpha excitations
    for (int i = 0; i < n_orb; ++i) {
        if (!det.is_occupied(i, true)) continue;
        for (int j = i+1; j < n_orb; ++j) {
            if (!det.is_occupied(j, true)) continue;
            for (int a = 0; a < n_orb; ++a) {
                if (det.is_occupied(a, true)) continue;
                for (int b = a+1; b < n_orb; ++b) {
                    if (det.is_occupied(b, true)) continue;
                    Determinant exc_det = det.double_excite(i, j, a, b, true, true);
                    doubles.push_back(exc_det);
                }
            }
        }
    }
    
    // Beta-beta excitations
    for (int i = 0; i < n_orb; ++i) {
        if (!det.is_occupied(i, false)) continue;
        for (int j = i+1; j < n_orb; ++j) {
            if (!det.is_occupied(j, false)) continue;
            for (int a = 0; a < n_orb; ++a) {
                if (det.is_occupied(a, false)) continue;
                for (int b = a+1; b < n_orb; ++b) {
                    if (det.is_occupied(b, false)) continue;
                    Determinant exc_det = det.double_excite(i, j, a, b, false, false);
                    doubles.push_back(exc_det);
                }
            }
        }
    }
    
    // Alpha-beta excitations
    for (int i = 0; i < n_orb; ++i) {
        if (!det.is_occupied(i, true)) continue;
        for (int j = 0; j < n_orb; ++j) {
            if (!det.is_occupied(j, false)) continue;
            for (int a = 0; a < n_orb; ++a) {
                if (det.is_occupied(a, true)) continue;
                for (int b = 0; b < n_orb; ++b) {
                    if (det.is_occupied(b, false)) continue;
                    Determinant exc_det = det.double_excite(i, j, a, b, true, false);
                    doubles.push_back(exc_det);
                }
            }
        }
    }
    
    return doubles;
}

// ============================================================================
// Helper: Generate connected excitations from variational space
// ============================================================================
std::vector<Determinant> generate_connected_excitations(
    const std::vector<Determinant>& variational_space,
    int n_orb,
    int max_excitation_level
) {
    // Theory: Generate all determinants connected to variational space
    // Connected = differ by 1-2 excitations from ANY variational determinant
    // Reference: Giner et al., J. Chem. Phys. 143, 124305 (2015)
    
    std::vector<Determinant> connected;
    
    // Use hash set to avoid duplicates
    std::unordered_set<std::string> seen;
    
    for (const auto& det : variational_space) {
        // Generate singles
        if (max_excitation_level >= 1) {
            std::vector<Determinant> singles = generate_singles(det, n_orb);
            for (const auto& single_det : singles) {
                std::string key = single_det.to_string();
                if (seen.find(key) == seen.end()) {
                    connected.push_back(single_det);
                    seen.insert(key);
                }
            }
        }
        
        // Generate doubles
        if (max_excitation_level >= 2) {
            std::vector<Determinant> doubles = generate_doubles(det, n_orb);
            for (const auto& double_det : doubles) {
                std::string key = double_det.to_string();
                if (seen.find(key) == seen.end()) {
                    connected.push_back(double_det);
                    seen.insert(key);
                }
            }
        }
    }
    
    return connected;
}

} // namespace ci
} // namespace mshqc
