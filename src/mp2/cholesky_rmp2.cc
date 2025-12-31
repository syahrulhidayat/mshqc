/**
 * @file cholesky_rmp2.cc
 * @brief Implementation of Cholesky-Decomposed RMP2
 * @author Muhamad Syahrul Hidayat
 * @date 2025-01-11
 */

#include "mshqc/cholesky_rmp2.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <vector>

namespace mshqc {

// ============================================================================
// CONSTRUCTOR 1: DECOMPOSE INTERNALLY
// ============================================================================
CholeskyRMP2::CholeskyRMP2(const Molecule& mol,
                           const BasisSet& basis,
                           std::shared_ptr<IntegralEngine> integrals,
                           const SCFResult& rhf_result,
                           const CholeskyRMP2Config& config)
    : mol_(mol), basis_(basis), integrals_(integrals),
      rhf_(rhf_result), config_(config), cholesky_ptr_(nullptr) {
    
    nbf_ = basis.n_basis_functions();
    nocc_ = rhf_.n_occ_alpha;
    nvirt_ = nbf_ - nocc_;

    // Perform decomposition if not provided
    if (config_.print_level > 0) {
        std::cout << "\n[CholeskyRMP2] Decomposing integrals internally (Threshold=" 
                  << config_.cholesky_threshold << ")...\n";
    }
    internal_cholesky_ = std::make_unique<integrals::CholeskyERI>(config_.cholesky_threshold);
    auto eri = integrals_->compute_eri();
    internal_cholesky_->decompose(eri);
    cholesky_ptr_ = internal_cholesky_.get();
}

// ============================================================================
// CONSTRUCTOR 2: REUSE VECTORS
// ============================================================================
CholeskyRMP2::CholeskyRMP2(const Molecule& mol,
                           const BasisSet& basis,
                           std::shared_ptr<IntegralEngine> integrals,
                           const SCFResult& rhf_result,
                           const CholeskyRMP2Config& config,
                           const integrals::CholeskyERI& existing_cholesky)
    : mol_(mol), basis_(basis), integrals_(integrals),
      rhf_(rhf_result), config_(config), cholesky_ptr_(&existing_cholesky) {
    
    nbf_ = basis.n_basis_functions();
    nocc_ = rhf_.n_occ_alpha;
    nvirt_ = nbf_ - nocc_;
    
    if (config_.print_level > 0) {
        std::cout << "\n[CholeskyRMP2] Initialized with reused vectors (" 
                  << cholesky_ptr_->n_vectors() << " vectors)\n";
    }
}

// ============================================================================
// HELPER: TRANSFORM VECTORS (AO -> MO)
// ============================================================================
std::vector<Eigen::MatrixXd> CholeskyRMP2::transform_vectors() {
    const auto& L_ao = cholesky_ptr_->get_L_vectors();
    int n_chol = cholesky_ptr_->n_vectors();
    
    // Hasil: Vector of matrices (nocc x nvirt)
    // Q_ia^K = sum_uv C_ui * C_va * L_uv^K
    std::vector<Eigen::MatrixXd> L_mo(n_chol);
    
    // Pre-slice coefficients
    Eigen::MatrixXd C_occ = rhf_.C_alpha.leftCols(nocc_);
    Eigen::MatrixXd C_virt = rhf_.C_alpha.rightCols(nvirt_);

    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < n_chol; k++) {
        // Reconstruct square AO matrix from flattened vector L_ao[k]
        Eigen::Map<const Eigen::MatrixXd> L_k_ao(L_ao[k].data(), nbf_, nbf_);
        
        // Transform: Q = C_occ^T * L_ao * C_virt
        // Dimensi: (nocc x nbf) * (nbf x nbf) * (nbf x nvirt) -> (nocc x nvirt)
        Eigen::MatrixXd Temp = L_k_ao * C_virt; // (N x V)
        L_mo[k] = C_occ.transpose() * Temp;     // (O x V)
    }
    
    return L_mo;
}

// ============================================================================
// MAIN COMPUTE
// ============================================================================
foundation::CholeskyRMP2Result CholeskyRMP2::compute() {
    if (config_.print_level > 0) {
        std::cout << "  Transforming Cholesky vectors to MO basis...\n";
    }

    // 1. Transform Vectors AO -> MO (Q_ia)
    auto Q_ia = transform_vectors(); 
    
    // 2. Compute Energy & Amplitudes
    double e_corr = 0.0;
    const auto& eps = rhf_.orbital_energies_alpha;
    
    // Initialize T2 tensor (for MP3 reuse)
    t2_ = Eigen::Tensor<double, 4>(nocc_, nocc_, nvirt_, nvirt_);
    t2_.setZero();
    
    int n_chol = cholesky_ptr_->n_vectors();

    if (config_.print_level > 0) std::cout << "  Computing energy and amplitudes...\n";

    // Loop Paralel
    #pragma omp parallel for reduction(+:e_corr) collapse(2)
    for (int i = 0; i < nocc_; i++) {
        for (int j = 0; j < nocc_; j++) {
            
            // Rekonstruksi blok integral (ia|jb) untuk i, j tetap
            // Matrix (V x V)
            Eigen::MatrixXd Int_dir = Eigen::MatrixXd::Zero(nvirt_, nvirt_);
            
            for (int k = 0; k < n_chol; k++) {
                // Outer product: Q(i, :) * Q(j, :)
                // Q_ia[k] is (Occ x Virt). Row(i) is vector length Virt.
                Int_dir += Q_ia[k].row(i).transpose() * Q_ia[k].row(j);
            }
            
            // Int_ex (ib|ja) adalah Transpose dari Int_dir (ia|jb)
            // Karena (ib|ja) = <ib|ja> = <ja|ib> (real) = Int_dir(b, a)
            
            for (int a = 0; a < nvirt_; a++) {
                for (int b = 0; b < nvirt_; b++) {
                    
                    double denom = eps(i) + eps(j) - eps(nocc_ + a) - eps(nocc_ + b);
                    if (std::abs(denom) < 1e-12) continue;

                    double iajb = Int_dir(a, b);
                    double ibja = Int_dir(b, a); // Transpose access
                    
                    // Amplitude standard t_ij^ab = (ia|jb) / D_ijab
                    double t_val = iajb / denom;
                    
                    // Simpan amplitudo
                    t2_(i, j, a, b) = t_val;
                    
                    // Update Energi: (2 <ia|jb> - <ib|ja>) * t_ij^ab
                    e_corr += (2.0 * iajb - ibja) * t_val;
                }
            }
        }
    }
    
    // 3. Populate Result (Gunakan CholeskyRMP2Result)
    foundation::CholeskyRMP2Result result;
    result.e_rhf = rhf_.energy_total;
    result.e_corr = e_corr;
    result.e_total = result.e_rhf + e_corr;
    result.t2 = t2_;
    result.n_occ = nocc_;
    result.n_virt = nvirt_;
    
    // [CRITICAL FOR MP3] Pass Cholesky vectors for reuse
    result.chol_vectors = cholesky_ptr_->get_L_vectors();
    result.n_chol_vectors = n_chol;
    
    if (config_.print_level > 0) {
        std::cout << std::setprecision(8);
        std::cout << "  RHF Energy:      " << std::setw(14) << result.e_rhf << " Ha\n";
        std::cout << "  Cholesky RMP2:   " << std::setw(14) << result.e_corr << " Ha\n";
        std::cout << "  Total Energy:    " << std::setw(14) << result.e_total << " Ha\n";
        std::cout << "==============================================\n";
    }
    
    return result;
}

} // namespace mshqc