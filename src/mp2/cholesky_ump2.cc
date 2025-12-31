/**
 * @file cholesky_ump2.cc
 * @brief Cholesky-based UMP2 Implementation
 * 
 * THEORY:
 * Uses Cholesky decomposition (ia|jb) ≈ Σ_K L^K_ia L^K_jb to reduce memory
 * 
 * REFERENCES:
 * - Møller & Plesset (1934), Phys. Rev. 46, 618
 * - Aquilante et al. (2008), J. Chem. Phys. 129, 024113
 * - Koch et al. (2003), J. Chem. Phys. 118, 9481
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-12-11
 * @license MIT
 */

#include "mshqc/cholesky_ump2.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

namespace mshqc {

// ============================================================================
// CONSTRUCTOR - REUSE Cholesky from UHF (NO RE-DECOMPOSITION)
// ============================================================================

CholeskyUMP2::CholeskyUMP2(
    const SCFResult& uhf_result,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    const CholeskyUMP2Config& config,
    const integrals::CholeskyERI& cholesky_from_uhf
) : uhf_(uhf_result), 
    basis_(basis), 
    integrals_(integrals),
    config_(config)
{
    nbf_ = basis.n_basis_functions();
    nocc_a_ = uhf_result.n_occ_alpha;
    nocc_b_ = uhf_result.n_occ_beta;
    nvir_a_ = nbf_ - nocc_a_;
    nvir_b_ = nbf_ - nocc_b_;
    
    if (config_.print_level > 0) {
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "Cholesky-based UMP2 (REUSING UHF Cholesky)\n";
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Basis functions: " << nbf_ << "\n";
        std::cout << "Occupied (α/β): " << nocc_a_ << " / " << nocc_b_ << "\n";
        std::cout << "Virtual (α/β):  " << nvir_a_ << " / " << nvir_b_ << "\n";
        std::cout << "Reusing " << cholesky_from_uhf.n_vectors() 
                  << " Cholesky vectors from UHF\n";
        std::cout << std::string(70, '=') << "\n\n";
    }
    
    // CRITICAL: Copy Cholesky object (shallow copy is OK, vectors are const)
    cholesky_ = std::make_unique<integrals::CholeskyERI>(cholesky_from_uhf);
    n_cholesky_ = cholesky_->n_vectors();
}

// ============================================================================
// MAIN COMPUTE ROUTINE
// ============================================================================

CholeskyUMP2Result CholeskyUMP2::compute() {
    CholeskyUMP2Result result;
    
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // Check if Cholesky already exists (from UHF reuse)
    bool need_decompose = (cholesky_->n_vectors() == 0);
    
    if (need_decompose) {
        // Step 1: Decompose ERIs in AO basis
        if (config_.print_level > 0) {
            std::cout << "Step 1: Cholesky decomposition of ERIs...\n";
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        decompose_eri();
        auto t2 = std::chrono::high_resolution_clock::now();
        result.time_cholesky_s = std::chrono::duration<double>(t2 - t1).count();
        
        n_cholesky_ = cholesky_->n_vectors();
        result.n_cholesky_vectors = n_cholesky_;
        result.compression_ratio = cholesky_->compression_ratio();
        
        if (config_.print_level > 0) {
            std::cout << "  Cholesky vectors: " << n_cholesky_ << "\n";
            std::cout << "  Compression: " << std::fixed << std::setprecision(1)
                      << result.compression_ratio << "×\n";
            std::cout << "  Time: " << std::fixed << std::setprecision(2)
                      << result.time_cholesky_s << " s\n\n";
        }
    } else {
        // Reusing existing Cholesky from UHF
        if (config_.print_level > 0) {
            std::cout << "Step 1: Cholesky decomposition...\n";
            std::cout << "  ✅ SKIPPED - Reusing " << cholesky_->n_vectors() 
                      << " vectors from UHF\n";
            std::cout << "  Compression: " << std::fixed << std::setprecision(1)
                      << cholesky_->compression_ratio() << "×\n\n";
        }
        result.time_cholesky_s = 0.0; // No decomposition overhead!
        n_cholesky_ = cholesky_->n_vectors();
        result.n_cholesky_vectors = n_cholesky_;
        result.compression_ratio = cholesky_->compression_ratio();
    }
    
    // Step 2: Transform to MO basis
    if (config_.print_level > 0) {
        std::cout << "Step 2: Transform Cholesky vectors to MO basis...\n";
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    transform_cholesky_vectors();
    auto t2 = std::chrono::high_resolution_clock::now();
    result.time_transform_s = std::chrono::duration<double>(t2 - t1).count();
    
    // ... rest of compute() remains unchanged ...
    if (config_.print_level > 0) {
        std::cout << "  Alpha L_ia stored: " << L_ia_alpha_.size() << " vectors\n";
        std::cout << "  Beta L_ia stored:  " << L_ia_beta_.size() << " vectors\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(2)
                  << result.time_transform_s << " s\n\n";
    }
    
    // Step 3: Compute MP2 energy
    if (config_.print_level > 0) {
        std::cout << "Step 3: Computing MP2 correlation energy...\n";
    }
    t1 = std::chrono::high_resolution_clock::now();
    
    result.e_corr_ss_aa = compute_ss_alpha();
    result.e_corr_ss_bb = compute_ss_beta();
    result.e_corr_os = compute_os();
    result.e_corr_total = result.e_corr_ss_aa + result.e_corr_ss_bb + result.e_corr_os;
    result.e_total = uhf_.energy_total + result.e_corr_total;
    
    t2 = std::chrono::high_resolution_clock::now();
    result.time_energy_s = std::chrono::duration<double>(t2 - t1).count();
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();
    
    // Memory estimate
    size_t mem_bytes = 0;
    for (const auto& L : L_ia_alpha_) {
        mem_bytes += L.size() * sizeof(double);
    }
    for (const auto& L : L_ia_beta_) {
        mem_bytes += L.size() * sizeof(double);
    }
    result.memory_mb = mem_bytes / (1024.0 * 1024.0);
    
    // Print results
    if (config_.print_level > 0) {
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "Cholesky-UMP2 Results\n";
        std::cout << std::string(70, '=') << "\n";
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "SS (αα):        " << std::setw(18) << result.e_corr_ss_aa << " Ha\n";
        std::cout << "SS (ββ):        " << std::setw(18) << result.e_corr_ss_bb << " Ha\n";
        std::cout << "OS (αβ):        " << std::setw(18) << result.e_corr_os << " Ha\n";
        std::cout << "Correlation:    " << std::setw(18) << result.e_corr_total << " Ha\n";
        std::cout << "Total energy:   " << std::setw(18) << result.e_total << " Ha\n";
        std::cout << std::string(70, '=') << "\n\n";
        
        print_statistics(result);
    }
    
    return result;
}

// ============================================================================
// CHOLESKY DECOMPOSITION
// ============================================================================

void CholeskyUMP2::decompose_eri() {
    if (config_.print_level > 0) std::cout << "Step 1: On-the-Fly Cholesky Decomposition...\n";

    // 1. Reset Cholesky object
    cholesky_->clear(); 
    
    // 2. Compute Diagonal (O(N^2))
    // NOTE: You must implement this in IntegralEngine.cc!
    Eigen::VectorXd diag = integrals_->compute_eri_diagonal(); 
    
    double max_diag = diag.maxCoeff();
    double threshold = config_.cholesky_threshold;
    int nbf = basis_.n_basis_functions();
    int n_pairs = nbf * (nbf + 1) / 2;
    int n_vec = 0;

    // 3. Main Cholesky Loop
    while (max_diag > threshold) {
        // A. Find Pivot
        int pivot_idx = 0;
        max_diag = diag.maxCoeff(&pivot_idx);
        
        if (max_diag < threshold) break;

        // B. Compute one column of integrals (O(N^2))
        // NOTE: You must implement this in IntegralEngine.cc!
        Eigen::VectorXd col = integrals_->compute_eri_column(pivot_idx);
        
        // C. Subtract contributions from previous vectors (Screening)
        // L_new = Col - sum(L_prev * L_prev_pivot)
        // We do this manually since we don't have the full matrix
        const auto& L_vecs = cholesky_->get_L_vectors();
        for (const auto& L_prev : L_vecs) {
            double L_pivot_val = L_prev(pivot_idx);
            col -= L_prev * L_pivot_val;
        }

        // D. Normalize to get new vector
        double pivot_val = std::sqrt(max_diag);
        Eigen::VectorXd new_vector = col / pivot_val; // This defines 'new_vector'

        // E. Store vector
        cholesky_->add_vector(new_vector);
        n_vec++;

        // F. Update Diagonal (D_i = D_i - L_i^2)
        // This is why Cholesky is fast: we update diagonal to find next best pivot
        for (int i = 0; i < n_pairs; i++) {
            diag(i) -= new_vector(i) * new_vector(i);
        }
        
        // Safety break
        if (n_vec > n_pairs) break;
    }
    
    if (config_.print_level > 0) {
        std::cout << "  Decomposition done. Vectors: " << n_vec << "\n";
    }
}

// ============================================================================
// TRANSFORM CHOLESKY VECTORS TO MO BASIS
// ============================================================================

void CholeskyUMP2::transform_cholesky_vectors() {
    // REFERENCE: Aquilante et al. (2008), J. Chem. Phys. 129, 024113, Eq. (12)
    // Transform L^K_μν → L^K_ia where i=occ, a=virt
    // L^K_ia = Σ_μν L^K_μν C_μi C_νa
    
    const auto& L_vectors = cholesky_->get_L_vectors();
    n_cholesky_ = L_vectors.size();
    
    // Extract occupied and virtual blocks from MO coefficients
    Eigen::MatrixXd Ca_occ = uhf_.C_alpha.leftCols(nocc_a_);
    Eigen::MatrixXd Ca_vir = uhf_.C_alpha.rightCols(nvir_a_);
    Eigen::MatrixXd Cb_occ = uhf_.C_beta.leftCols(nocc_b_);
    Eigen::MatrixXd Cb_vir = uhf_.C_beta.rightCols(nvir_b_);
    
    L_ia_alpha_.resize(n_cholesky_);
    L_ia_beta_.resize(n_cholesky_);
    
    if (config_.print_level > 1) {
        std::cout << "  Transforming " << n_cholesky_ << " Cholesky vectors...\n";
    }
    
    for (int K = 0; K < n_cholesky_; K++) {
        // Reshape L^K from vector to matrix
        // L^K is stored as vector of length nbf*nbf
        Eigen::MatrixXd L_K(nbf_, nbf_);
        for (int mu = 0; mu < nbf_; mu++) {
            for (int nu = 0; nu < nbf_; nu++) {
                int idx = mu * nbf_ + nu;
                L_K(mu, nu) = L_vectors[K](idx);
            }
        }
        
        // Transform alpha: L^K_ia = Σ_μν L^K_μν C_μi C_νa
        // This is equivalent to: L_ia = C_occ^T * L_K * C_vir
        L_ia_alpha_[K] = Ca_occ.transpose() * L_K * Ca_vir;
        
        // Transform beta
        L_ia_beta_[K] = Cb_occ.transpose() * L_K * Cb_vir;
        
        if (config_.print_level > 1 && (K % 50 == 0 || K < 5)) {
            std::cout << "    Vector " << std::setw(4) << K 
                      << ": ||L_ia^α|| = " << std::fixed << std::setprecision(6)
                      << L_ia_alpha_[K].norm() << "\n";
        }
    }
    
    if (config_.print_level > 1) {
        std::cout << "  Transformation complete.\n";
    }
}

// ============================================================================
// ENERGY COMPUTATION - SAME-SPIN ALPHA
// ============================================================================

double CholeskyUMP2::compute_ss_alpha() {
    // REFERENCE: Szabo & Ostlund (1996), Eq. (6.74), p. 354
    // Same-spin αα correlation:
    // E_ss^αα = (1/4) Σ_ijab |<ij||ab>|² / Δ_ijab
    // where <ij||ab> = <ij|ab> - <ij|ba> (antisymmetrized)
    
    // Using Cholesky vectors:
    // <ij|ab> = Σ_K L^K_ia L^K_jb
    
    const auto& eps = uhf_.orbital_energies_alpha;
    double e_ss = 0.0;
    
    if (config_.print_level > 1) {
        std::cout << "  Computing same-spin (αα) contribution...\n";
    }
    
    for (int i = 0; i < nocc_a_; i++) {
        for (int j = 0; j < nocc_a_; j++) {
            for (int a = 0; a < nvir_a_; a++) {
                for (int b = 0; b < nvir_a_; b++) {
                    
                    // Compute <ij|ab> using Cholesky vectors
                    // <ij|ab> = Σ_K L^K_ia L^K_jb
                    double g_ijab = 0.0;
                    for (int K = 0; K < n_cholesky_; K++) {
                        g_ijab += L_ia_alpha_[K](i, a) * L_ia_alpha_[K](j, b);
                    }
                    
                    // Compute <ij|ba> for antisymmetrization
                    double g_ijba = 0.0;
                    for (int K = 0; K < n_cholesky_; K++) {
                        g_ijba += L_ia_alpha_[K](i, b) * L_ia_alpha_[K](j, a);
                    }
                    
                    // Antisymmetrized integral
                    double g_antisym = g_ijab - g_ijba;
                    
                    // Orbital energy denominator
                    // REFERENCE: Szabo & Ostlund (1996), Eq. (6.65)
                    // Δ = εi + εj - εa - εb (always negative for stability)
                    double denom = eps(i) + eps(j) 
                                 - eps(nocc_a_ + a) - eps(nocc_a_ + b);
                    
                    if (std::abs(denom) < 1e-12) continue;
                    
                    // MP2 energy contribution
                    // Factor 0.25 = 1/4 from spin integration
                    e_ss += 0.25 * g_antisym * g_antisym / denom;
                }
            }
        }
    }
    
    if (config_.print_level > 1) {
        std::cout << "    E_ss(αα) = " << std::fixed << std::setprecision(10)
                  << e_ss << " Ha\n";
    }
    
    return e_ss;
}

// ============================================================================
// ENERGY COMPUTATION - SAME-SPIN BETA
// ============================================================================

double CholeskyUMP2::compute_ss_beta() {
    // REFERENCE: Szabo & Ostlund (1996), Eq. (6.74)
    // Same-spin ββ correlation (identical formula, different orbitals)
    
    const auto& eps = uhf_.orbital_energies_beta;
    double e_ss = 0.0;
    
    if (config_.print_level > 1) {
        std::cout << "  Computing same-spin (ββ) contribution...\n";
    }
    
    for (int i = 0; i < nocc_b_; i++) {
        for (int j = 0; j < nocc_b_; j++) {
            for (int a = 0; a < nvir_b_; a++) {
                for (int b = 0; b < nvir_b_; b++) {
                    
                    // <ij|ab> = Σ_K L^K_ia L^K_jb
                    double g_ijab = 0.0;
                    for (int K = 0; K < n_cholesky_; K++) {
                        g_ijab += L_ia_beta_[K](i, a) * L_ia_beta_[K](j, b);
                    }
                    
                    // <ij|ba> for antisymmetrization
                    double g_ijba = 0.0;
                    for (int K = 0; K < n_cholesky_; K++) {
                        g_ijba += L_ia_beta_[K](i, b) * L_ia_beta_[K](j, a);
                    }
                    
                    double g_antisym = g_ijab - g_ijba;
                    
                    double denom = eps(i) + eps(j) 
                                 - eps(nocc_b_ + a) - eps(nocc_b_ + b);
                    
                    if (std::abs(denom) < 1e-12) continue;
                    
                    e_ss += 0.25 * g_antisym * g_antisym / denom;
                }
            }
        }
    }
    
    if (config_.print_level > 1) {
        std::cout << "    E_ss(ββ) = " << std::fixed << std::setprecision(10)
                  << e_ss << " Ha\n";
    }
    
    return e_ss;
}

// ============================================================================
// ENERGY COMPUTATION - OPPOSITE-SPIN
// ============================================================================

double CholeskyUMP2::compute_os() {
    // REFERENCE: Szabo & Ostlund (1996), Eq. (6.73), p. 353
    // Opposite-spin αβ correlation:
    // E_os = Σ_ijab <ij|ab>² / Δ_ijab
    // NO antisymmetrization (different spins)
    
    // Using Cholesky:
    // <i_α j_β|a_α b_β> = Σ_K L^K_ia(α) L^K_jb(β)
    
    const auto& eps_a = uhf_.orbital_energies_alpha;
    const auto& eps_b = uhf_.orbital_energies_beta;
    double e_os = 0.0;
    
    if (config_.print_level > 1) {
        std::cout << "  Computing opposite-spin (αβ) contribution...\n";
    }
    
    for (int i = 0; i < nocc_a_; i++) {
        for (int j = 0; j < nocc_b_; j++) {
            for (int a = 0; a < nvir_a_; a++) {
                for (int b = 0; b < nvir_b_; b++) {
                    
                    // Compute mixed-spin integral
                    // <i_α j_β|a_α b_β> = Σ_K L^K_ia(α) L^K_jb(β)
                    double g_ijab = 0.0;
                    for (int K = 0; K < n_cholesky_; K++) {
                        g_ijab += L_ia_alpha_[K](i, a) * L_ia_beta_[K](j, b);
                    }
                    
                    // Mixed denominator (α and β orbital energies)
                    double denom = eps_a(i) + eps_b(j) 
                                 - eps_a(nocc_a_ + a) - eps_b(nocc_b_ + b);
                    
                    if (std::abs(denom) < 1e-12) continue;
                    
                    // No factor 1/4 (full weight for opposite-spin)
                    e_os += g_ijab * g_ijab / denom;
                }
            }
        }
    }
    
    if (config_.print_level > 1) {
        std::cout << "    E_os(αβ) = " << std::fixed << std::setprecision(10)
                  << e_os << " Ha\n";
    }
    
    return e_os;
}

// ============================================================================
// T2 AMPLITUDES (OPTIONAL)
// ============================================================================

void CholeskyUMP2::compute_t2_amplitudes() {
    // NOTE: This stores full T2 tensor - memory intensive!
    // Only use for small systems or when needed
    
    if (config_.print_level > 0) {
        std::cout << "\nComputing T2 amplitudes (memory-intensive)...\n";
    }
    
    // Allocate tensors
    t2_aa_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_a_, nvir_a_, nvir_a_);
    t2_bb_ = Eigen::Tensor<double, 4>(nocc_b_, nocc_b_, nvir_b_, nvir_b_);
    t2_ab_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_b_, nvir_a_, nvir_b_);
    t2_aa_.setZero();
    t2_bb_.setZero();
    t2_ab_.setZero();
    
    const auto& eps_a = uhf_.orbital_energies_alpha;
    const auto& eps_b = uhf_.orbital_energies_beta;
    
    // Alpha-alpha amplitudes
    for (int i = 0; i < nocc_a_; i++) {
        for (int j = 0; j < nocc_a_; j++) {
            for (int a = 0; a < nvir_a_; a++) {
                for (int b = 0; b < nvir_a_; b++) {
                    double g_ijab = 0.0;
                    for (int K = 0; K < n_cholesky_; K++) {
                        g_ijab += L_ia_alpha_[K](i,a) * L_ia_alpha_[K](j,b);
                    }
                    
                    double denom = eps_a(i) + eps_a(j) 
                                 - eps_a(nocc_a_+a) - eps_a(nocc_a_+b);
                    
                    if (std::abs(denom) > 1e-12) {
                        t2_aa_(i, j, a, b) = g_ijab / denom;
                    }
                }
            }
        }
    }
    
    // Beta-beta amplitudes
    for (int i = 0; i < nocc_b_; i++) {
        for (int j = 0; j < nocc_b_; j++) {
            for (int a = 0; a < nvir_b_; a++) {
                for (int b = 0; b < nvir_b_; b++) {
                    double g_ijab = 0.0;
                    for (int K = 0; K < n_cholesky_; K++) {
                        g_ijab += L_ia_beta_[K](i,a) * L_ia_beta_[K](j,b);
                    }
                    
                    double denom = eps_b(i) + eps_b(j) 
                                 - eps_b(nocc_b_+a) - eps_b(nocc_b_+b);
                    
                    if (std::abs(denom) > 1e-12) {
                        t2_bb_(i, j, a, b) = g_ijab / denom;
                    }
                }
            }
        }
    }
    
    // Alpha-beta amplitudes
    for (int i = 0; i < nocc_a_; i++) {
        for (int j = 0; j < nocc_b_; j++) {
            for (int a = 0; a < nvir_a_; a++) {
                for (int b = 0; b < nvir_b_; b++) {
                    double g_ijab = 0.0;
                    for (int K = 0; K < n_cholesky_; K++) {
                        g_ijab += L_ia_alpha_[K](i,a) * L_ia_beta_[K](j,b);
                    }
                    
                    double denom = eps_a(i) + eps_b(j) 
                                 - eps_a(nocc_a_+a) - eps_b(nocc_b_+b);
                    
                    if (std::abs(denom) > 1e-12) {
                        t2_ab_(i, j, a, b) = g_ijab / denom;
                    }
                }
            }
        }
    }
    
    if (config_.print_level > 0) {
        std::cout << "T2 amplitudes computed and stored.\n";
    }
}
// ============================================================================
// INTEGRAL RECONSTRUCTION (VALIDATION)
// ============================================================================

double CholeskyUMP2::reconstruct_integral(
    int i, int j, int a, int b, 
    bool alpha_i, bool alpha_j
) {
    // Reconstruct <ij|ab> from Cholesky vectors for validation
    // REFERENCE: Koch et al. (2003), J. Chem. Phys. 118, 9481
    // (ia|jb) ≈ Σ_K L^K_ia L^K_jb
    
    double integral = 0.0;
    
    if (alpha_i && alpha_j) {
        // Both alpha
        for (int K = 0; K < n_cholesky_; K++) {
            integral += L_ia_alpha_[K](i, a) * L_ia_alpha_[K](j, b);
        }
    } else if (!alpha_i && !alpha_j) {
        // Both beta
        for (int K = 0; K < n_cholesky_; K++) {
            integral += L_ia_beta_[K](i, a) * L_ia_beta_[K](j, b);
        }
    } else {
        // Mixed (alpha-beta or beta-alpha)
        if (alpha_i) {
            for (int K = 0; K < n_cholesky_; K++) {
                integral += L_ia_alpha_[K](i, a) * L_ia_beta_[K](j, b);
            }
        } else {
            for (int K = 0; K < n_cholesky_; K++) {
                integral += L_ia_beta_[K](i, a) * L_ia_alpha_[K](j, b);
            }
        }
    }
    
    return integral;
}

// ============================================================================
// STATISTICS & TIMING
// ============================================================================

void CholeskyUMP2::print_statistics(const CholeskyUMP2Result& result) const {
    std::cout << "Performance Statistics:\n";
    std::cout << std::string(70, '-') << "\n";
    
    // Timing breakdown
    double total_time = result.time_cholesky_s 
                      + result.time_transform_s 
                      + result.time_energy_s;
    
    std::cout << "Timing breakdown:\n";
    std::cout << "  Cholesky decomp:  " << std::fixed << std::setprecision(3)
              << std::setw(8) << result.time_cholesky_s << " s ("
              << std::setw(5) << std::setprecision(1)
              << (result.time_cholesky_s / total_time * 100) << "%)\n";
    
    std::cout << "  MO transform:     " << std::fixed << std::setprecision(3)
              << std::setw(8) << result.time_transform_s << " s ("
              << std::setw(5) << std::setprecision(1)
              << (result.time_transform_s / total_time * 100) << "%)\n";
    
    std::cout << "  Energy compute:   " << std::fixed << std::setprecision(3)
              << std::setw(8) << result.time_energy_s << " s ("
              << std::setw(5) << std::setprecision(1)
              << (result.time_energy_s / total_time * 100) << "%)\n";
    
    std::cout << "  Total:            " << std::fixed << std::setprecision(3)
              << std::setw(8) << total_time << " s\n";
    
    std::cout << "\n";
    
    // Memory usage
    std::cout << "Memory usage:\n";
    std::cout << "  Cholesky vectors: " << result.n_cholesky_vectors << "\n";
    std::cout << "  L_ia storage:     " << std::fixed << std::setprecision(2)
              << result.memory_mb << " MB\n";
    std::cout << "  Compression:      " << std::fixed << std::setprecision(1)
              << result.compression_ratio << "× vs full ERI\n";
    
    // Theoretical memory comparison
    long long full_eri_size = (long long)nbf_ * nbf_ * nbf_ * nbf_;
    long long cholesky_size = (long long)n_cholesky_ * nbf_ * nbf_;
    double full_mb = full_eri_size * 8.0 / (1024.0 * 1024.0);
    double cholesky_mb = cholesky_size * 8.0 / (1024.0 * 1024.0);
    
    std::cout << "  Full ERI (N⁴):    " << std::fixed << std::setprecision(1)
              << full_mb << " MB (theoretical)\n";
    std::cout << "  Cholesky (N²M):   " << std::fixed << std::setprecision(1)
              << cholesky_mb << " MB (actual AO)\n";
    
    std::cout << "\n";
    
    // Orbital information
    std::cout << "Orbital dimensions:\n";
    std::cout << "  Occupied (α):     " << nocc_a_ << "\n";
    std::cout << "  Virtual (α):      " << nvir_a_ << "\n";
    std::cout << "  Occupied (β):     " << nocc_b_ << "\n";
    std::cout << "  Virtual (β):      " << nvir_b_ << "\n";
    std::cout << "  Basis functions:  " << nbf_ << "\n";
    
    std::cout << "\n";
    
    // Integral counts
    long long n_ss_aa = (long long)nocc_a_ * nocc_a_ * nvir_a_ * nvir_a_;
    long long n_ss_bb = (long long)nocc_b_ * nocc_b_ * nvir_b_ * nvir_b_;
    long long n_os = (long long)nocc_a_ * nocc_b_ * nvir_a_ * nvir_b_;
    long long n_total = n_ss_aa + n_ss_bb + n_os;
    
    std::cout << "Integral evaluations:\n";
    std::cout << "  Same-spin (αα):   " << n_ss_aa << "\n";
    std::cout << "  Same-spin (ββ):   " << n_ss_bb << "\n";
    std::cout << "  Opposite-spin:    " << n_os << "\n";
    std::cout << "  Total:            " << n_total << "\n";
    
    std::cout << std::string(70, '=') << "\n\n";
}

} // namespace mshqc