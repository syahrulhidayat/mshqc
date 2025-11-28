/**
 * @file cholesky_caspt2.cc
 * @brief Cholesky-Decomposed CASPT2 implementation
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 * @license MIT License
 */

#include "mshqc/mcscf/cholesky_caspt2.h"
#include "mshqc/mcscf/external_space.h"
#include "mshqc/ci/slater_condon.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

namespace mshqc {
namespace mcscf {

// ============================================================================
// Constructor
// ============================================================================

CholeskyCASPT2::CholeskyCASPT2(const Molecule& mol,
                               const BasisSet& basis,
                               std::shared_ptr<IntegralEngine> integrals,
                               const CASResult& casscf_result,
                               double cholesky_threshold)
    : mol_(mol), basis_(basis), integrals_(std::move(integrals)),
      casscf_(casscf_result), cholesky_threshold_(cholesky_threshold) {
    
    nbf_ = static_cast<int>(basis.n_basis_functions());
    n_mo_ = casscf_.C_mo.cols();
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Cholesky-CASPT2 Setup\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Basis functions:    " << nbf_ << "\n";
    std::cout << "MO orbitals:        " << n_mo_ << "\n";
    std::cout << "Cholesky threshold: " << std::scientific << cholesky_threshold_ << " Ha\n";
    std::cout << std::string(70, '=') << "\n\n";
}

// ============================================================================
// Main compute function
// ============================================================================

CholeskyCASPT2Result CholeskyCASPT2::compute() {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Cholesky-CASPT2 Calculation\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Reference: Andersson et al. (1992) + Koch et al. (2003)\n";
    std::cout << "E(CASSCF) = " << std::fixed << std::setprecision(10) 
              << casscf_.e_casscf << " Ha\n";
    std::cout << "IPEA shift = " << ipea_shift_ << " Ha\n";
    std::cout << "Imaginary shift = " << imaginary_shift_ << " Ha\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    CholeskyCASPT2Result res;
    res.e_casscf = casscf_.e_casscf;
    res.ipea_shift_used = ipea_shift_;
    res.imaginary_shift_used = imaginary_shift_;
    res.cholesky_threshold = cholesky_threshold_;
    
    // Verify CASResult has required data
    if (casscf_.determinants.empty()) {
        std::cerr << "ERROR: CASResult missing determinants\n";
        res.e_pt2 = 0.0;
        res.e_total = res.e_casscf;
        res.converged = false;
        res.status_message = "Failed: No determinants in CASResult";
        return res;
    }
    
    // ========================================================================
    // Phase 1: Cholesky decomposition of AO ERIs
    // ========================================================================
    std::cout << "Phase 1: Cholesky decomposition of AO ERIs...\n";
    compute_cholesky_decomposition();
    std::cout << "  Cholesky vectors: " << cholesky_result_.n_vectors << "\n";
    std::cout << "  Compression ratio: " << std::fixed << std::setprecision(1)
              << cholesky_result_.compression_ratio << "×\n";
    std::cout << "  Max error: " << std::scientific << cholesky_result_.max_error << " Ha\n\n";
    
    res.n_cholesky_vectors = cholesky_result_.n_vectors;
    res.cholesky_error = cholesky_result_.max_error;
    
    // ========================================================================
    // Phase 2: Transform Cholesky vectors to MO basis
    // ========================================================================
    std::cout << "Phase 2: Transforming Cholesky vectors to MO basis...\n";
    transform_cholesky_to_mo();
    std::cout << "  Cholesky MO vectors ready\n\n";
    
    // ========================================================================
    // Phase 3: Compute PT2 energy using Cholesky-reconstructed ERIs
    // ========================================================================
    std::cout << "Phase 3: Computing Cholesky-CASPT2 energy...\n";
    
    double E_PT2 = compute_pt2_energy_cholesky();
    
    std::cout << "  PT2 correction computed\n";
    std::cout << "  E(PT2) = " << std::scientific << std::setprecision(6) 
              << E_PT2 << " Ha\n\n";
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Results
    res.e_pt2 = E_PT2;
    res.e_total = res.e_casscf + res.e_pt2;
    res.converged = true;
    res.status_message = "Cholesky-CASPT2 completed successfully";
    
    // Print results
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Cholesky-CASPT2 Results\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "E(CASSCF)         = " << res.e_casscf << " Ha\n";
    std::cout << "E(PT2)            = " << res.e_pt2 << " Ha\n";
    std::cout << "E(Cholesky-CASPT2) = " << res.e_total << " Ha\n";
    std::cout << "\nCholesky vectors: " << res.n_cholesky_vectors << "\n";
    std::cout << "Cholesky threshold: " << std::scientific << res.cholesky_threshold << " Ha\n";
    std::cout << "Cholesky max error: " << res.cholesky_error << " Ha\n";
    std::cout << "Computation time: " << std::fixed << std::setprecision(2) 
              << elapsed.count() << " seconds\n";
    std::cout << "Status: " << res.status_message << "\n";
    std::cout << std::string(70, '=') << "\n";
    
    return res;
}

// ============================================================================
// Compute Cholesky decomposition
// ============================================================================

void CholeskyCASPT2::compute_cholesky_decomposition() {
    // Compute full AO ERIs
    std::cout << "  Computing full AO ERIs...\n";
    auto eri_ao = integrals_->compute_eri();
    
    // Perform Cholesky decomposition
    cholesky_eri_ = std::make_unique<integrals::CholeskyERI>(cholesky_threshold_);
    cholesky_result_ = cholesky_eri_->decompose(eri_ao);
}

// ============================================================================
// Transform Cholesky vectors to MO basis
// ============================================================================

void CholeskyCASPT2::transform_cholesky_to_mo() {
    const auto& L_ao = cholesky_eri_->get_L_vectors();
    const auto& C = casscf_.C_mo;
    
    int n_chol = cholesky_result_.n_vectors;
    L_mo_.resize(n_chol);
    
    std::cout << "  Transforming " << n_chol << " Cholesky vectors...\n";
    
    // Transform each Cholesky vector: L^K_μν → L^K_pq
    // L^K_pq = Σ_μν C_μp L^K_μν C_νq
    for (int K = 0; K < n_chol; K++) {
        // Reshape L^K from vector to matrix
        Eigen::MatrixXd L_K_ao(nbf_, nbf_);
        for (int mu = 0; mu < nbf_; mu++) {
            for (int nu = 0; nu < nbf_; nu++) {
                int idx = mu * nbf_ + nu;
                L_K_ao(mu, nu) = L_ao[K](idx);
            }
        }
        
        // Transform: C^T * L^K * C
        L_mo_[K] = C.transpose() * L_K_ao * C;
        
        if ((K+1) % 10 == 0 || K < 10) {
            std::cout << "    Transformed vector " << (K+1) << "/" << n_chol << "\n";
        }
    }
}

// ============================================================================
// Compute PT2 energy using Cholesky vectors
// ============================================================================

double CholeskyCASPT2::compute_pt2_energy_cholesky() {
    // Generate external space (same as standard CASPT2)
    ExternalSpaceGenerator gen(casscf_.active_space, n_mo_);
    auto external_dets = gen.generate(casscf_.determinants);
    
    std::cout << "  External space: " << external_dets.size() << " determinants\n";
    
    if (external_dets.empty()) {
        std::cout << "  No external space - no PT2 correction\n";
        return 0.0;
    }
    
    // Reconstruct MO ERIs from Cholesky vectors: (pq|rs) = Σ_K L^K_pq L^K_rs
    std::cout << "  Reconstructing MO ERIs from Cholesky vectors...\n";
    
    // Build integrals structure for hamiltonian_element
    ci::CIIntegrals integrals_mo;
    
    // Compute h_core in MO basis: h_pq = Σ_μν C_μp h_μν C_νq
    auto h_core_ao = integrals_->compute_core_hamiltonian();
    integrals_mo.h_alpha = casscf_.C_mo.transpose() * h_core_ao * casscf_.C_mo;
    integrals_mo.h_beta = integrals_mo.h_alpha;  // Same for RHF-based CASSCF
    
    // Allocate ERI tensors
    integrals_mo.eri_aaaa = Eigen::Tensor<double, 4>(n_mo_, n_mo_, n_mo_, n_mo_);
    integrals_mo.eri_bbbb = Eigen::Tensor<double, 4>(n_mo_, n_mo_, n_mo_, n_mo_);
    integrals_mo.eri_aabb = Eigen::Tensor<double, 4>(n_mo_, n_mo_, n_mo_, n_mo_);
    integrals_mo.eri_aaaa.setZero();
    integrals_mo.eri_bbbb.setZero();
    integrals_mo.eri_aabb.setZero();
    
    // Reconstruct ERIs in chemist notation: (pq|rs) = Σ_K L^K_pq L^K_rs
    int n_chol = static_cast<int>(L_mo_.size());
    
    // First build chemist notation ERIs
    Eigen::Tensor<double, 4> eri_chem(n_mo_, n_mo_, n_mo_, n_mo_);
    eri_chem.setZero();
    
    for (int p = 0; p < n_mo_; p++) {
        for (int q = 0; q < n_mo_; q++) {
            for (int r = 0; r < n_mo_; r++) {
                for (int s = 0; s < n_mo_; s++) {
                    for (int K = 0; K < n_chol; K++) {
                        eri_chem(p, q, r, s) += L_mo_[K](p, q) * L_mo_[K](r, s);
                    }
                }
            }
        }
    }
    
    // Convert chemist to physicist notation (same as CASSCF line 353-368)
    // REFERENCE: CASSCF::transform_integrals_to_active()
    for (int p = 0; p < n_mo_; p++) {
        for (int q = 0; q < n_mo_; q++) {
            for (int r = 0; r < n_mo_; r++) {
                for (int s = 0; s < n_mo_; s++) {
                    // Same-spin: antisymmetrized physicist notation
                    // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)_chem
                    integrals_mo.eri_aaaa(p, q, r, s) = eri_chem(p, r, q, s) - eri_chem(p, s, q, r);
                    integrals_mo.eri_bbbb(p, q, r, s) = eri_chem(p, r, q, s) - eri_chem(p, s, q, r);
                    
                    // Mixed-spin: chemist notation (NO antisymmetrization)
                    // CI expects eri_aabb(i,i,j,j) = (ii|jj)_chem
                    integrals_mo.eri_aabb(p, q, r, s) = eri_chem(p, q, r, s);
                }
            }
        }
    }
    
    std::cout << "  MO ERIs reconstructed from Cholesky\n";
    std::cout << "  Computing PT2 energy...\n";
    
    // Compute PT2 energy correction
    // E_PT2 = Σ_K |V_K0|² / (E_0 - E_K)
    double E_PT2 = 0.0;
    int n_cas = static_cast<int>(casscf_.determinants.size());
    
    // Loop over external determinants
    for (size_t K = 0; K < external_dets.size(); K++) {
        const auto& ext_det = external_dets[K];
        
        // Compute matrix element: V_K0 = Σ_I c_I * H_KI
        double V_K0 = 0.0;
        
        for (int I = 0; I < n_cas; I++) {
            double H_KI = ci::hamiltonian_element(
                ext_det,
                casscf_.determinants[I],
                integrals_mo
            );
            V_K0 += casscf_.ci_coeffs[I] * H_KI;
        }
        
        // Compute denominator: E₀ - E_K
        double E_K = 0.0;
        for (int p = 0; p < n_mo_; p++) {
            if (ext_det.is_occupied(p, true)) {
                E_K += casscf_.orbital_energies(p);
            }
            if (ext_det.is_occupied(p, false)) {
                E_K += casscf_.orbital_energies(p);
            }
        }
        
        double denom = casscf_.e_casscf - E_K;
        
        // Add shifts
        if (ipea_shift_ > 0.0) {
            denom -= ipea_shift_;
        }
        if (imaginary_shift_ > 0.0) {
            denom -= imaginary_shift_;
        }
        
        // Accumulate PT2 contribution
        if (std::abs(denom) > 1e-10) {
            E_PT2 += (V_K0 * V_K0) / denom;
        }
    }
    
    return E_PT2;
}

} // namespace mcscf
} // namespace mshqc
