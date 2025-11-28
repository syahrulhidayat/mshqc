/**
 * @file cisdt.cc
 * @brief CISDT implementation with triple excitations
 * 
 * THEORY REFERENCES:
 *   - Pople et al. (1977), Int. J. Quantum Chem. 11, 149
 *   - Raghavachari et al. (1989), J. Chem. Phys. 91, 1062
 *   - Helgaker et al. (2000), Ch. 10.6
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 */

#include "mshqc/ci/cisdt.h"
#include "mshqc/ci/hamiltonian_sparse.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace mshqc {
namespace ci {

// Binomial coefficient
static size_t binomial(int n, int k) {
    if (k > n || k < 0) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k;
    
    size_t result = 1;
    for (int i = 0; i < k; i++) {
        result *= (n - i);
        result /= (i + 1);
    }
    return result;
}

// Constructor
CISDT::CISDT(const CIIntegrals& ints,
             const Determinant& hf_det,
             int n_occ_alpha, int n_occ_beta,
             int n_virt_alpha, int n_virt_beta)
    : ints_(ints), hf_det_(hf_det),
      nocc_a_(n_occ_alpha), nocc_b_(n_occ_beta),
      nvirt_a_(n_virt_alpha), nvirt_b_(n_virt_beta) {}

// Generate singles (same as CISD)
std::vector<Determinant> CISDT::generate_singles() {
    std::vector<Determinant> singles;
    
    auto occ_a = hf_det_.alpha_occupations();
    auto occ_b = hf_det_.beta_occupations();
    
    // α singles
    for (int i : occ_a) {
        for (int a = nocc_a_; a < nocc_a_ + nvirt_a_; a++) {
            try {
                singles.push_back(hf_det_.single_excite(i, a, true));
            } catch (...) {}
        }
    }
    
    // β singles
    for (int i : occ_b) {
        for (int a = nocc_b_; a < nocc_b_ + nvirt_b_; a++) {
            try {
                singles.push_back(hf_det_.single_excite(i, a, false));
            } catch (...) {}
        }
    }
    
    return singles;
}

// Generate doubles (same as CISD)
std::vector<Determinant> CISDT::generate_doubles() {
    std::vector<Determinant> doubles;
    
    auto occ_a = hf_det_.alpha_occupations();
    auto occ_b = hf_det_.beta_occupations();
    
    // αα doubles
    for (size_t idx_i = 0; idx_i < occ_a.size(); idx_i++) {
        for (size_t idx_j = idx_i + 1; idx_j < occ_a.size(); idx_j++) {
            int i = occ_a[idx_i];
            int j = occ_a[idx_j];
            
            for (int a = nocc_a_; a < nocc_a_ + nvirt_a_; a++) {
                for (int b = a + 1; b < nocc_a_ + nvirt_a_; b++) {
                    try {
                        doubles.push_back(hf_det_.double_excite(i, j, a, b, true, true));
                    } catch (...) {}
                }
            }
        }
    }
    
    // ββ doubles
    for (size_t idx_i = 0; idx_i < occ_b.size(); idx_i++) {
        for (size_t idx_j = idx_i + 1; idx_j < occ_b.size(); idx_j++) {
            int i = occ_b[idx_i];
            int j = occ_b[idx_j];
            
            for (int a = nocc_b_; a < nocc_b_ + nvirt_b_; a++) {
                for (int b = a + 1; b < nocc_b_ + nvirt_b_; b++) {
                    try {
                        doubles.push_back(hf_det_.double_excite(i, j, a, b, false, false));
                    } catch (...) {}
                }
            }
        }
    }
    
    // αβ doubles
    for (int i : occ_a) {
        for (int j : occ_b) {
            for (int a = nocc_a_; a < nocc_a_ + nvirt_a_; a++) {
                for (int b = nocc_b_; b < nocc_b_ + nvirt_b_; b++) {
                    try {
                        doubles.push_back(hf_det_.double_excite(i, j, a, b, true, false));
                    } catch (...) {}
                }
            }
        }
    }
    
    return doubles;
}

// Generate triple excitations
// REFERENCE: Raghavachari et al. (1989), J. Chem. Phys. 91, 1062
std::vector<Determinant> CISDT::generate_triples() {
    std::vector<Determinant> triples;
    
    auto occ_a = hf_det_.alpha_occupations();
    auto occ_b = hf_det_.beta_occupations();
    
    // αααα triples: i_α j_α k_α → a_α b_α c_α
    for (size_t idx_i = 0; idx_i < occ_a.size(); idx_i++) {
        for (size_t idx_j = idx_i + 1; idx_j < occ_a.size(); idx_j++) {
            for (size_t idx_k = idx_j + 1; idx_k < occ_a.size(); idx_k++) {
                int i = occ_a[idx_i];
                int j = occ_a[idx_j];
                int k = occ_a[idx_k];
                
                for (int a = nocc_a_; a < nocc_a_ + nvirt_a_; a++) {
                    for (int b = a + 1; b < nocc_a_ + nvirt_a_; b++) {
                        for (int c = b + 1; c < nocc_a_ + nvirt_a_; c++) {
                            try {
                                triples.push_back(hf_det_.triple_excite(
                                    i, j, k, a, b, c, true, true, true));
                            } catch (...) {}
                        }
                    }
                }
            }
        }
    }
    
    // ββββ triples: i_β j_β k_β → a_β b_β c_β
    for (size_t idx_i = 0; idx_i < occ_b.size(); idx_i++) {
        for (size_t idx_j = idx_i + 1; idx_j < occ_b.size(); idx_j++) {
            for (size_t idx_k = idx_j + 1; idx_k < occ_b.size(); idx_k++) {
                int i = occ_b[idx_i];
                int j = occ_b[idx_j];
                int k = occ_b[idx_k];
                
                for (int a = nocc_b_; a < nocc_b_ + nvirt_b_; a++) {
                    for (int b = a + 1; b < nocc_b_ + nvirt_b_; b++) {
                        for (int c = b + 1; c < nocc_b_ + nvirt_b_; c++) {
                            try {
                                triples.push_back(hf_det_.triple_excite(
                                    i, j, k, a, b, c, false, false, false));
                            } catch (...) {}
                        }
                    }
                }
            }
        }
    }
    
    // ααβ triples: i_α j_α k_β → a_α b_α c_β
    for (size_t idx_i = 0; idx_i < occ_a.size(); idx_i++) {
        for (size_t idx_j = idx_i + 1; idx_j < occ_a.size(); idx_j++) {
            int i = occ_a[idx_i];
            int j = occ_a[idx_j];
            
            for (int k : occ_b) {
                for (int a = nocc_a_; a < nocc_a_ + nvirt_a_; a++) {
                    for (int b = a + 1; b < nocc_a_ + nvirt_a_; b++) {
                        for (int c = nocc_b_; c < nocc_b_ + nvirt_b_; c++) {
                            try {
                                triples.push_back(hf_det_.triple_excite(
                                    i, j, k, a, b, c, true, true, false));
                            } catch (...) {}
                        }
                    }
                }
            }
        }
    }
    
    // αββ triples: i_α j_β k_β → a_α b_β c_β
    for (int i : occ_a) {
        for (size_t idx_j = 0; idx_j < occ_b.size(); idx_j++) {
            for (size_t idx_k = idx_j + 1; idx_k < occ_b.size(); idx_k++) {
                int j = occ_b[idx_j];
                int k = occ_b[idx_k];
                
                for (int a = nocc_a_; a < nocc_a_ + nvirt_a_; a++) {
                    for (int b = nocc_b_; b < nocc_b_ + nvirt_b_; b++) {
                        for (int c = b + 1; c < nocc_b_ + nvirt_b_; c++) {
                            try {
                                triples.push_back(hf_det_.triple_excite(
                                    i, j, k, a, b, c, true, false, false));
                            } catch (...) {}
                        }
                    }
                }
            }
        }
    }
    
    return triples;
}

// Estimate determinant count
size_t CISDT::estimate_n_determinants() const {
    size_t n_singles_a = nocc_a_ * nvirt_a_;
    size_t n_singles_b = nocc_b_ * nvirt_b_;
    
    size_t n_doubles_aa = binomial(nocc_a_, 2) * binomial(nvirt_a_, 2);
    size_t n_doubles_bb = binomial(nocc_b_, 2) * binomial(nvirt_b_, 2);
    size_t n_doubles_ab = nocc_a_ * nocc_b_ * nvirt_a_ * nvirt_b_;
    
    size_t n_triples_aaa = binomial(nocc_a_, 3) * binomial(nvirt_a_, 3);
    size_t n_triples_bbb = binomial(nocc_b_, 3) * binomial(nvirt_b_, 3);
    size_t n_triples_aab = binomial(nocc_a_, 2) * nocc_b_ * binomial(nvirt_a_, 2) * nvirt_b_;
    size_t n_triples_abb = nocc_a_ * binomial(nocc_b_, 2) * nvirt_a_ * binomial(nvirt_b_, 2);
    
    return 1 + n_singles_a + n_singles_b + 
           n_doubles_aa + n_doubles_bb + n_doubles_ab +
           n_triples_aaa + n_triples_bbb + n_triples_aab + n_triples_abb;
}

// Get all determinants
std::vector<Determinant> CISDT::get_determinants() const {
    std::vector<Determinant> dets;
    dets.push_back(hf_det_);
    
    auto singles = const_cast<CISDT*>(this)->generate_singles();
    dets.insert(dets.end(), singles.begin(), singles.end());
    
    auto doubles = const_cast<CISDT*>(this)->generate_doubles();
    dets.insert(dets.end(), doubles.begin(), doubles.end());
    
    auto triples = const_cast<CISDT*>(this)->generate_triples();
    dets.insert(dets.end(), triples.begin(), triples.end());
    
    return dets;
}

// Main compute function
CISDTResult CISDT::compute(const CISDTOptions& opts) {
    
    if (opts.verbose) {
        std::cout << "\n=== CISDT (Singles + Doubles + Triples) ===\n";
        std::cout << "Occupied: α=" << nocc_a_ << ", β=" << nocc_b_ << "\n";
        std::cout << "Virtual:  α=" << nvirt_a_ << ", β=" << nvirt_b_ << "\n\n";
    }
    
    // Estimate size
    size_t estimated = estimate_n_determinants();
    if (opts.verbose) {
        std::cout << "Estimated determinants: " << estimated << "\n";
        if (estimated > 10000) {
            std::cout << "⚠️  WARNING: Large CISDT space! This may take time...\n";
        }
        std::cout << "\n";
    }
    
    // Generate determinants
    if (opts.verbose) std::cout << "Generating singles...\n";
    auto singles = generate_singles();
    
    if (opts.verbose) std::cout << "Generating doubles...\n";
    auto doubles = generate_doubles();
    
    if (opts.verbose) std::cout << "Generating triples (may take time)...\n";
    auto triples = generate_triples();
    
    // Build full list
    std::vector<Determinant> all_dets;
    all_dets.push_back(hf_det_);
    all_dets.insert(all_dets.end(), singles.begin(), singles.end());
    all_dets.insert(all_dets.end(), doubles.begin(), doubles.end());
    all_dets.insert(all_dets.end(), triples.begin(), triples.end());
    
    int n_dets = all_dets.size();
    
    if (opts.verbose) {
        std::cout << "\nDeterminant count:\n";
        std::cout << "  HF reference:  1\n";
        std::cout << "  Singles:       " << singles.size() << "\n";
        std::cout << "  Doubles:       " << doubles.size() << "\n";
        std::cout << "  Triples:       " << triples.size() << "\n";
        std::cout << "  Total:         " << n_dets << "\n\n";
    }
    
    CISDTResult result;
    result.n_determinants = n_dets;
    result.n_singles = singles.size();
    result.n_doubles = doubles.size();
    result.n_triples = triples.size();
    result.determinants = all_dets;
    
    // Compute HF energy
    double e0 = diagonal_element(hf_det_, ints_);
    result.e_hf = e0;
    
    // Solve eigenvalue problem
    if (n_dets <= opts.davidson_threshold && !opts.use_davidson) {
        // Dense diagonalization
        if (opts.verbose) {
            std::cout << "Using dense diagonalization\n";
            std::cout << "Building Hamiltonian matrix...\n";
        }
        
        auto H = build_hamiltonian(all_dets, ints_);
        H.diagonal().array() -= e0;
        
        if (opts.verbose) std::cout << "Diagonalizing...\n";
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
        
        result.e_cisdt = solver.eigenvalues()(0) + e0;
        result.coefficients = solver.eigenvectors().col(0);
        result.converged = true;
        
    } else {
        // Davidson solver
        if (opts.verbose) {
            std::cout << "Using Davidson iterative solver\n";
        }
        
        DavidsonOptions dav_opts;
        dav_opts.max_iter = opts.max_davidson_iter;
        dav_opts.conv_tol = opts.davidson_tol;
        dav_opts.verbose = false;
        
        // Initial guess: HF determinant with weight 1
        Eigen::VectorXd guess = Eigen::VectorXd::Zero(n_dets);
        guess(0) = 1.0;
        
        DavidsonSolver solver(dav_opts);
        auto dav_result = solver.solve(all_dets, ints_, guess);
        
        result.e_cisdt = dav_result.energy + e0;
        result.coefficients = dav_result.eigenvector;
        result.converged = dav_result.converged;
    }
    
    result.e_corr = result.e_cisdt - result.e_hf;
    
    // Analyze wavefunction
    result.hf_weight = result.coefficients(0) * result.coefficients(0);
    
    result.singles_weight = 0.0;
    for (size_t i = 1; i <= singles.size(); i++) {
        result.singles_weight += result.coefficients(i) * result.coefficients(i);
    }
    
    result.doubles_weight = 0.0;
    for (size_t i = 1 + singles.size(); i <= singles.size() + doubles.size(); i++) {
        result.doubles_weight += result.coefficients(i) * result.coefficients(i);
    }
    
    result.triples_weight = 0.0;
    for (size_t i = 1 + singles.size() + doubles.size(); i < all_dets.size(); i++) {
        result.triples_weight += result.coefficients(i) * result.coefficients(i);
    }
    
    if (opts.verbose) {
        std::cout << "\n=== CISDT Results ===\n";
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "E(CISDT)     = " << result.e_cisdt << " Ha\n";
        std::cout << "E_corr(CISDT)= " << result.e_corr << " Ha\n";
        std::cout << "\nWavefunction analysis:\n";
        std::cout << "  HF weight:      " << std::setw(10) << result.hf_weight << "\n";
        std::cout << "  Singles weight: " << std::setw(10) << result.singles_weight << "\n";
        std::cout << "  Doubles weight: " << std::setw(10) << result.doubles_weight << "\n";
        std::cout << "  Triples weight: " << std::setw(10) << result.triples_weight << "\n\n";
    }
    
    return result;
}

// Estimate determinant count function
size_t cisdt_determinant_count(int n_orb_alpha, int n_orb_beta,
                               int n_occ_alpha, int n_occ_beta) {
    int nvirt_a = n_orb_alpha - n_occ_alpha;
    int nvirt_b = n_orb_beta - n_occ_beta;
    
    size_t n_s = n_occ_alpha * nvirt_a + n_occ_beta * nvirt_b;
    size_t n_d = binomial(n_occ_alpha, 2) * binomial(nvirt_a, 2) +
                 binomial(n_occ_beta, 2) * binomial(nvirt_b, 2) +
                 n_occ_alpha * n_occ_beta * nvirt_a * nvirt_b;
    size_t n_t = binomial(n_occ_alpha, 3) * binomial(nvirt_a, 3) +
                 binomial(n_occ_beta, 3) * binomial(nvirt_b, 3) +
                 binomial(n_occ_alpha, 2) * n_occ_beta * binomial(nvirt_a, 2) * nvirt_b +
                 n_occ_alpha * binomial(n_occ_beta, 2) * nvirt_a * binomial(nvirt_b, 2);
    
    return 1 + n_s + n_d + n_t;
}

} // namespace ci
} // namespace mshqc
