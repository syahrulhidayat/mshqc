/**
 * @file cisd.cc
 * @brief CISD implementation
 * 
 * THEORY REFERENCES:
 *   - Shavitt (1998), Mol. Phys. 94, 3
 *   - Szabo & Ostlund (1996), Ch. 4.3
 *   - Helgaker et al. (2000), Ch. 10.5
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 */

#include "mshqc/ci/cisd.h"
#include "mshqc/ci/hamiltonian_sparse.h"
#include "mshqc/ci/ci_utils.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

namespace mshqc {
namespace ci {

CISD::CISD(const CIIntegrals& ints,
           const Determinant& hf_det,
           int n_occ_alpha, int n_occ_beta,
           int n_virt_alpha, int n_virt_beta)
    : ints_(ints), hf_det_(hf_det),
      nocc_a_(n_occ_alpha), nocc_b_(n_occ_beta),
      nvirt_a_(n_virt_alpha), nvirt_b_(n_virt_beta) {}

// Generate singles excitations
// REFERENCE: Szabo & Ostlund (1996), Section 4.2
std::vector<Determinant> CISD::generate_singles() {
    std::vector<Determinant> singles;
    
    auto occ_a = hf_det_.alpha_occupations();
    auto occ_b = hf_det_.beta_occupations();
    
    // α singles: i_α → a_α
    for (int i : occ_a) {
        for (int a = nocc_a_; a < nocc_a_ + nvirt_a_; a++) {
            try {
                singles.push_back(hf_det_.single_excite(i, a, true));
            } catch (...) {}
        }
    }
    
    // β singles: i_β → a_β
    for (int i : occ_b) {
        for (int a = nocc_b_; a < nocc_b_ + nvirt_b_; a++) {
            try {
                singles.push_back(hf_det_.single_excite(i, a, false));
            } catch (...) {}
        }
    }
    
    return singles;
}

// Generate doubles excitations
// REFERENCE: Szabo & Ostlund (1996), Section 4.3
std::vector<Determinant> CISD::generate_doubles() {
    std::vector<Determinant> doubles;
    
    auto occ_a = hf_det_.alpha_occupations();
    auto occ_b = hf_det_.beta_occupations();
    
    // αα doubles: i_α j_α → a_α b_α
    for (size_t idx_i = 0; idx_i < occ_a.size(); idx_i++) {
        for (size_t idx_j = idx_i + 1; idx_j < occ_a.size(); idx_j++) {
            int i = occ_a[idx_i];
            int j = occ_a[idx_j];
            
            for (int a = nocc_a_; a < nocc_a_ + nvirt_a_; a++) {
                for (int b = a + 1; b < nocc_a_ + nvirt_a_; b++) {
                    try {
                        auto excited = hf_det_.double_excite(i, j, a, b, true, true);
                        doubles.push_back(excited);
                    } catch (...) {}
                }
            }
        }
    }
    
    // ββ doubles: i_β j_β → a_β b_β
    for (size_t idx_i = 0; idx_i < occ_b.size(); idx_i++) {
        for (size_t idx_j = idx_i + 1; idx_j < occ_b.size(); idx_j++) {
            int i = occ_b[idx_i];
            int j = occ_b[idx_j];
            
            for (int a = nocc_b_; a < nocc_b_ + nvirt_b_; a++) {
                for (int b = a + 1; b < nocc_b_ + nvirt_b_; b++) {
                    try {
                        auto excited = hf_det_.double_excite(i, j, a, b, false, false);
                        doubles.push_back(excited);
                    } catch (...) {}
                }
            }
        }
    }
    
    // αβ doubles: i_α j_β → a_α b_β
    for (int i : occ_a) {
        for (int j : occ_b) {
            for (int a = nocc_a_; a < nocc_a_ + nvirt_a_; a++) {
                for (int b = nocc_b_; b < nocc_b_ + nvirt_b_; b++) {
                    try {
                        auto excited = hf_det_.double_excite(i, j, a, b, true, false);
                        doubles.push_back(excited);
                    } catch (...) {}
                }
            }
        }
    }
    
    return doubles;
}

std::vector<Determinant> CISD::get_determinants() const {
    std::vector<Determinant> dets;
    
    // Add HF reference
    dets.push_back(hf_det_);
    
    // Add singles
    auto singles = const_cast<CISD*>(this)->generate_singles();
    dets.insert(dets.end(), singles.begin(), singles.end());
    
    // Add doubles
    auto doubles = const_cast<CISD*>(this)->generate_doubles();
    dets.insert(dets.end(), doubles.begin(), doubles.end());
    
    return dets;
}

// Main CISD computation
// REFERENCE: Shavitt (1998), Mol. Phys. 94, 3
CISDResult CISD::compute(const CISDOptions& opts) {
    
    if (opts.verbose) {
        std::cout << "\n=== CISD (Configuration Interaction Singles + Doubles) ===\n";
        std::cout << "Occupied: α=" << nocc_a_ << ", β=" << nocc_b_ << "\n";
        std::cout << "Virtual:  α=" << nvirt_a_ << ", β=" << nvirt_b_ << "\n\n";
    }
    
    // Generate determinants
    if (opts.verbose) std::cout << "Generating determinants...\n";
    auto singles = generate_singles();
    auto doubles = generate_doubles();
    
    std::vector<Determinant> all_dets;
    all_dets.push_back(hf_det_);
    all_dets.insert(all_dets.end(), singles.begin(), singles.end());
    all_dets.insert(all_dets.end(), doubles.begin(), doubles.end());
    
    int n_dets = all_dets.size();
    
    if (opts.verbose) {
        std::cout << "Determinant count:\n";
        std::cout << "  HF reference:  1\n";
        std::cout << "  Singles:       " << singles.size() << "\n";
        std::cout << "  Doubles:       " << doubles.size() << "\n";
        std::cout << "  Total:         " << n_dets << "\n\n";
    }
    
    // Decide whether to use sparse representation
    bool use_sparse = opts.use_sparse;
    if (opts.auto_sparse && !opts.use_sparse) {
        use_sparse = (n_dets > opts.sparse_threshold);
    }
    
    // Decide whether to use on-the-fly mode
    bool use_onthefly = opts.use_onthefly;
    if (opts.auto_onthefly && !opts.use_onthefly) {
        use_onthefly = (n_dets > opts.onthefly_threshold);
    }
    
    // On-the-fly takes precedence over sparse (more efficient for large systems)
    if (use_onthefly && use_sparse) {
        use_sparse = false;  // Disable sparse if on-the-fly is enabled
    }
    
    CISDResult result;
    result.n_determinants = n_dets;
    
    // Small systems: use dense diagonalization
    if (n_dets <= 1000 && !opts.use_sparse) {
        if (opts.verbose) {
            std::cout << "Using dense diagonalization (small system)\n";
            std::cout << "Building Hamiltonian matrix...\n";
        }
        
        // Apply diagonal shift so that HF reference has zero energy.
        // This matches the standard CI convention and improves numerical stability.
        double e0 = diagonal_element(hf_det_, ints_);
        auto H = build_hamiltonian(all_dets, ints_);
        H.diagonal().array() -= e0;
        
        if (opts.verbose) {
            std::cout << "Applying diagonal shift: E0 = " << std::fixed << std::setprecision(10)
                      << e0 << " Ha (HF reference -> 0)\n";
            std::cout << "Diagonalizing...\n";
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
        
        // Add the shift back to eigenvalue for reporting total electronic CI energy
        result.e_cisd = solver.eigenvalues()(0) + e0;
        result.coefficients = solver.eigenvectors().col(0);
        result.converged = true;
        result.iterations = 0;
        
    } else if (use_sparse) {
        // Large sparse systems: use sparse Davidson
        if (opts.verbose) {
            std::cout << "Using sparse representation (n_dets = " << n_dets << ")\n";
            std::cout << "Building sparse Hamiltonian (eps = " << opts.sparse_eps << ")...\n";
        }
        
        // Build sparse Hamiltonian in CSR format
        SparseCSR Hcsr;
        build_hamiltonian_csr(all_dets, ints_, Hcsr, opts.sparse_eps);
        
        if (opts.verbose) {
            size_t nnz = Hcsr.values().size();
            size_t dense_size = static_cast<size_t>(n_dets) * n_dets;
            double sparsity = 100.0 * (1.0 - static_cast<double>(nnz) / dense_size);
            std::cout << "  Nonzeros: " << nnz << " / " << dense_size 
                      << " (sparsity: " << std::fixed << std::setprecision(1) 
                      << sparsity << "%)\n";
        }
        
        // Compute diagonal for preconditioner
        Eigen::VectorXd H_diag = hamiltonian_diagonal(all_dets, ints_);
        
        // Davidson with sparse matvec
        DavidsonOptions dav_opts;
        dav_opts.max_iter = 100;
        dav_opts.conv_tol = 1e-8;
        dav_opts.residual_tol = 1e-6;
        dav_opts.verbose = opts.verbose;
        
        DavidsonSolver solver(dav_opts);
        
        // Generate initial guess
        auto guess = generate_davidson_guess(all_dets, ints_);
        
        auto davidson_result = solver.solve_sparse(Hcsr, H_diag, guess);
        
        // Apply diagonal shift after solve (constant shift does not change eigenvectors)
        double e0 = diagonal_element(hf_det_, ints_);
        result.e_cisd = davidson_result.energy + e0;
        result.coefficients = davidson_result.eigenvector;
        result.converged = davidson_result.converged;
        result.iterations = davidson_result.iterations;
        
    } else {
        // Medium systems: use dense Davidson
        if (use_onthefly) {
            if (opts.verbose) {
                std::cout << "Using Davidson solver with ON-THE-FLY sigma-vector\n";
                std::cout << "  (N = " << n_dets << " > threshold = " << opts.onthefly_threshold << ")\n";
            }
        } else {
            if (opts.verbose) std::cout << "Using Davidson solver (dense)\n";
        }
        
        DavidsonOptions dav_opts;
        dav_opts.max_iter = 100;
        dav_opts.conv_tol = 1e-8;
        dav_opts.residual_tol = 1e-6;
        dav_opts.verbose = opts.verbose;
        
        DavidsonSolver solver(dav_opts);
        
        // Configure on-the-fly mode if enabled
        if (use_onthefly) {
            // Build hash map for O(1) determinant lookup
            auto det_map = build_determinant_index_map(all_dets);
            int n_orb = nocc_a_ + nvirt_a_;  // Total number of orbitals
            
            // Enable on-the-fly mode
            solver.set_onthefly_mode(true, n_orb, &det_map);
        }
        
        // Generate initial guess
        auto guess = generate_davidson_guess(all_dets, ints_);
        
        auto davidson_result = solver.solve(all_dets, ints_, guess);
        
        // Apply diagonal shift after solve (constant shift does not change eigenvectors)
        double e0 = diagonal_element(hf_det_, ints_);
        result.e_cisd = davidson_result.energy + e0;
        result.coefficients = davidson_result.eigenvector;
        result.converged = davidson_result.converged;
        result.iterations = davidson_result.iterations;
    }
    
    // Compute HF energy and correlation
    result.e_hf = diagonal_element(hf_det_, ints_);
    result.e_corr = result.e_cisd - result.e_hf;
    result.determinants = all_dets;
    
    std::cout << "\n--- CISD Results ---\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "HF energy:          " << result.e_hf << " Ha\n";
    std::cout << "CISD energy:        " << result.e_cisd << " Ha\n";
    std::cout << "Correlation energy: " << result.e_corr << " Ha\n";
    
    // Analyze wavefunction
    analyze_wavefunction(result.coefficients, all_dets);
    
    return result;
}

CISDResult CISD::compute_with_mp2_comparison(double mp2_correlation) {
    auto result = compute();
    
    result.mp2_correlation = mp2_correlation;
    result.cisd_vs_mp2_diff = result.e_corr - mp2_correlation;
    
    std::cout << "\n--- Comparison with MP2 ---\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "MP2 correlation:    " << mp2_correlation << " Ha\n";
    std::cout << "CISD correlation:   " << result.e_corr << " Ha\n";
    std::cout << "Difference:         " << result.cisd_vs_mp2_diff << " Ha\n";
    std::cout << "CISD vs MP2:        " << (result.cisd_vs_mp2_diff > 0 ? "less" : "more") 
              << " correlated\n";
    
    return result;
}

// Analyze CI wavefunction
void CISD::analyze_wavefunction(const Eigen::VectorXd& c,
                                const std::vector<Determinant>& dets) {
    
    std::cout << "\n--- Wavefunction Analysis ---\n";
    
    // HF weight
    double c0_squared = c(0) * c(0);
    std::cout << "HF weight: " << std::setprecision(4) << c0_squared 
              << " (" << (c0_squared * 100) << "%)\n";
    
    // Find dominant configurations
    struct Config {
        int idx;
        double coeff;
        double weight;
    };
    
    std::vector<Config> configs;
    for (int i = 0; i < c.size(); i++) {
        if (std::abs(c(i)) > 0.05) {  // Threshold 5%
            configs.push_back({i, c(i), c(i)*c(i)});
        }
    }
    
    // Sort by weight
    std::sort(configs.begin(), configs.end(),
              [](const Config& a, const Config& b) {
                  return a.weight > b.weight;
              });
    
    std::cout << "\nDominant configurations (threshold > 5%):\n";
    std::cout << "Index   Coefficient    Weight\n";
    std::cout << "-----------------------------------\n";
    
    int count = 0;
    for (const auto& cfg : configs) {
        if (count++ >= 10) break;  // Show top 10
        
        std::cout << std::setw(5) << cfg.idx
                  << std::fixed << std::setprecision(6)
                  << std::setw(14) << cfg.coeff
                  << std::setw(12) << cfg.weight;
        
        if (cfg.idx == 0) {
            std::cout << "  (HF)";
        } else {
            // Determine if singles or doubles
            auto exc = find_excitation(dets[0], dets[cfg.idx]);
            if (exc.level == 1) {
                std::cout << "  (single)";
            } else if (exc.level == 2) {
                std::cout << "  (double)";
            }
        }
        std::cout << "\n";
    }
    
    // Singles vs doubles contribution
    double singles_weight = 0.0;
    double doubles_weight = 0.0;
    
    for (size_t i = 1; i < dets.size(); i++) {
        auto exc = find_excitation(dets[0], dets[i]);
        double w = c(i) * c(i);
        
        if (exc.level == 1) {
            singles_weight += w;
        } else if (exc.level == 2) {
            doubles_weight += w;
        }
    }
    
    std::cout << "\nExcitation contributions:\n";
    std::cout << "  HF:      " << std::setprecision(4) << c0_squared << "\n";
    std::cout << "  Singles: " << singles_weight << "\n";
    std::cout << "  Doubles: " << doubles_weight << "\n";
}

} // namespace ci
} // namespace mshqc
