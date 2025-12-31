/**
 * @file dfmp2.cc
 * @brief Density-Fitted MP2 (DF-MP2) for ROHF
 * 
 * Implements Resolution-of-Identity (RI) approximation for MP2:
 * Approximate 4-center ERIs (μν|λσ) using 3-center integrals (μν|P)
 * where P runs over auxiliary basis functions.
 * 
 * Approximation: (μν|λσ) ≈ Σ_PQ (μν|P) [J^-1]_PQ (Q|λσ)
 * where J_PQ = (P|Q) is auxiliary basis metric.
 * 
 * This reduces MP2 scaling from O(N^5) to O(N^4) with negligible error
 * (~μHa accuracy with proper auxiliary basis like cc-pVTZ-RI).
 * 
 * Theory References:
 *   - M. Feyereisen et al., Chem. Phys. Lett. 208, 359 (1993)
 *     [Original RI-MP2 formulation, Eq. (7): fitted integral formula]
 *   - F. Weigend & M. Häser, Theor. Chem. Acc. 97, 331 (1997)
 *     [Auxiliary basis design principles for RI-MP2]
 *   - F. Weigend et al., Chem. Phys. Lett. 294, 143 (1998)
 *     [Optimized cc-pVTZ-RI auxiliary basis, Eq. (3): metric inversion]
 *   - A. Szabo & N. S. Ostlund, "Modern Quantum Chemistry" (1996)
 *     [MP2 energy formulas, Section 6.4]
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-01-11
 * @license MIT License (see LICENSE file in project root)
 * 
 * @note Original implementation from published equations.
 *       No code copied from Psi4, PySCF, or other software.
 *       Algorithm derived from Feyereisen et al. (1993) paper.
 */

#include "mshqc/dfmp2.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <Eigen/Eigenvalues>

namespace mshqc {

DFMP2::DFMP2(const SCFResult& rohf_result,
             const BasisSet& basis,
             const BasisSet& aux_basis,
             std::shared_ptr<IntegralEngine> integrals)
    : rohf_(rohf_result), basis_(basis), aux_basis_(aux_basis),
      integrals_(integrals) {
    
    nbf_ = static_cast<int>(basis.n_basis_functions());
    naux_ = static_cast<int>(aux_basis.n_basis_functions());
    nocc_ = rohf_.n_occ_beta;  // ROHF: use beta (closed-shell count)
    nvir_ = nbf_ - rohf_.n_occ_alpha;  // virtual starts after all alpha
    
    std::cout << "\nDF-MP2 Setup:\n";
    std::cout << "  Primary basis:   " << nbf_ << " functions\n";
    std::cout << "  Auxiliary basis: " << naux_ << " functions\n";
    std::cout << "  Occupied:        " << nocc_ << "\n";
    std::cout << "  Virtual:         " << nvir_ << "\n";
}

DFMP2Result DFMP2::compute() {
    std::cout << "\n========================================\n";
    std::cout << "  DF-MP2 Calculation\n";
    std::cout << "========================================\n";
    
    // Step 1: Build auxiliary metric J = (P|Q) and invert
    // REFERENCE: Feyereisen et al. (1993), Eq. (8)
    // J_PQ = ∫∫ χ_P(r1) r12^-1 χ_Q(r2) dr1 dr2
    std::cout << "\nBuilding auxiliary metric J...\n";
    compute_metric();
    std::cout << "  Metric computed and inverted.\n";
    
    // Step 2: Transform 3-center to MO basis
    // REFERENCE: Weigend et al. (1998), Eq. (2)
    // B^P_ia = Σ_μν C_μi (μν|P) C_νa
    std::cout << "\nTransforming 3-center integrals to MO...\n";
    transform_3center();
    std::cout << "  Transformation complete.\n";
    
    // Step 3: Compute MP2 energy
    // REFERENCE: Feyereisen et al. (1993), Eq. (11)
    std::cout << "\nComputing DF-MP2 energy...\n";
    
    double e_ss = compute_ss_energy();
    double e_os = compute_os_energy();
    double e_corr = e_ss + e_os;
    
    DFMP2Result result;
    result.e_ss = e_ss;
    result.e_os = e_os;
    result.e_corr = e_corr;
    result.e_total = rohf_.energy_total + e_corr;
    
    // Print
    std::cout << "\n========================================\n";
    std::cout << "  DF-MP2 Results\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "\nSCF energy:      " << rohf_.energy_total << " Ha\n";
    std::cout << "\nCorrelation:\n";
    std::cout << "  Same-spin:     " << e_ss << " Ha\n";
    std::cout << "  Opposite-spin: " << e_os << " Ha\n";
    std::cout << "  Total:         " << e_corr << " Ha\n";
    std::cout << "\nTotal DF-MP2:    " << result.e_total << " Ha\n";
    std::cout << "========================================\n\n";
    
    return result;
}

void DFMP2::compute_metric() {
    // REFERENCE: Feyereisen et al. (1993), Eq. (8)
    // Auxiliary metric: J_PQ = (P|Q) = ∫∫ χ_P(r1) r12^-1 χ_Q(r2) dr1 dr2
    //
    // This is 2-center ERI over auxiliary basis
    // Then invert via Cholesky: J must be positive definite
    
    // Compute (P|Q) using IntegralEngine (Phase 1)
    J_ = integrals_->compute_2center_eri(aux_basis_);
    
    // Check positive definiteness via eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(J_);
    double min_eval = eig.eigenvalues().minCoeff();
    double max_eval = eig.eigenvalues().maxCoeff();
    double cond = max_eval / min_eval;
    
    std::cout << "  Metric condition number: " << std::scientific 
              << std::setprecision(2) << cond << std::fixed << "\n";
    
    if (min_eval <= 0) {
        throw std::runtime_error("DF-MP2: Metric not positive definite!");
    }
    
    // Compute J^{-1/2} for DF fitting
    // REFERENCE: Weigend et al. (1998), Eq. (6)
    // B̃^P = Σ_Q B^Q (J^{-1/2})_QP
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_solver(J_);
    const auto& eigenvec = eig_solver.eigenvectors();
    const auto& eigenval = eig_solver.eigenvalues();
    
    J_inv_ = Eigen::MatrixXd::Zero(naux_, naux_);
    for (int P = 0; P < naux_; P++) {
        for (int Q = 0; Q < naux_; Q++) {
            for (int k = 0; k < naux_; k++) {
                J_inv_(P, Q) += eigenvec(P, k) * (1.0 / std::sqrt(eigenval(k))) * eigenvec(Q, k);
            }
        }
    }
    
    // Validate: J^{-1/2} * J * J^{-1/2} should = I
    Eigen::MatrixXd test = J_inv_ * J_ * J_inv_;
    double max_offdiag = 0.0;
    for (int i = 0; i < naux_; i++) {
        for (int j = 0; j < naux_; j++) {
            if (i != j) {
                max_offdiag = std::max(max_offdiag, std::abs(test(i,j)));
            }
        }
    }
    std::cout << "  J^{-1/2} validated: max off-diagonal = " << std::scientific 
              << max_offdiag << std::fixed << "\n";
}

void DFMP2::transform_3center() {
    // REFERENCE: Weigend et al. (1998), Section 2
    // Transform (μν|P) to (ia|P) using MO coefficients
    // B^P_ia = Σ_μν C_μi (μν|P) C_νa
    //
    // Index convention:
    // i = occupied (0..nocc-1)
    // a = virtual (nocc..nbf-1) 
    // P = auxiliary (0..naux-1)
    
    // Get 3-center integrals (dummy s-function approximation)
    auto B_ao = integrals_->compute_3center_eri(aux_basis_);
    
    std::cout << "  Using raw 3-center integrals (no additional fitting)\n";
    
    // DEBUG: Check B_ao values
    std::cout << "  DEBUG: B_ao(0,0,0) = " << std::scientific << B_ao(0,0,0) 
              << ", B_ao(0,1,0) = " << B_ao(0,1,0) << std::fixed << "\n";
    
    // Allocate MO 3-center: [nocc*nvir, naux]
    // Flatten (i,a) to single index: idx = i*nvir + a
    B_ia_ = Eigen::MatrixXd::Zero(nocc_ * nvir_, naux_);
    
    const auto& C = rohf_.C_alpha;  // use alpha MOs
    
    // DEBUG: Check C matrix
    std::cout << "  DEBUG: C shape = [" << C.rows() << ", " << C.cols() << "]\n";
    std::cout << "  DEBUG: C(0,0) = " << std::scientific << C(0,0) 
              << ", C(1,0) = " << C(1,0) << std::fixed << "\n";
    std::cout << "  DEBUG: C(0,2) = " << std::scientific << C(0,2) 
              << ", C(1,2) = " << C(1,2) << std::fixed << "\n";
    
    // Check if virtual MOs are populated
    double c_norm_virt = 0.0;
    for (int mu = 0; mu < nbf_; mu++) {
        c_norm_virt += C(mu, 2) * C(mu, 2);
    }
    std::cout << "  DEBUG: ||C(:,2)|| = " << std::scientific << std::sqrt(c_norm_virt) << std::fixed << "\n";
    
    // Transform via quarter-transformation: (μν|P) → (iν|P) → (ia|P)
    // Step 1: First index μ → i
    std::vector<double> B_i_data(nocc_ * nbf_ * naux_, 0.0);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> B_i(
        B_i_data.data(), nocc_, nbf_, naux_
    );
    
    for (int i = 0; i < nocc_; i++) {
        for (int nu = 0; nu < nbf_; nu++) {
            for (int P = 0; P < naux_; P++) {
                double sum = 0.0;
                for (int mu = 0; mu < nbf_; mu++) {
                    sum += C(mu, i) * B_ao(mu, nu, P);
                }
                B_i(i, nu, P) = sum;
            }
        }
    }
    
    // DEBUG: Check B_i after first transform
    std::cout << "  DEBUG: B_i(0,0,0) = " << std::scientific << B_i(0,0,0) 
              << ", B_i(0,1,0) = " << B_i(0,1,0) << std::fixed << "\n";
    std::cout << "  DEBUG: B_i(0,3,0) = " << std::scientific << B_i(0,3,0) 
              << ", B_i(0,5,0) = " << B_i(0,5,0) << std::fixed << "\n";
    
    // Manual check of dot product for first virtual
    double manual_sum = 0.0;
    for (int nu = 0; nu < nbf_; nu++) {
        manual_sum += B_i(0, nu, 0) * C(nu, 2);
    }
    std::cout << "  DEBUG: manual B_ia(0,0) = " << std::scientific << manual_sum << std::fixed << "\n";
    
    // Step 2: Second index ν → a (virtual)
    for (int i = 0; i < nocc_; i++) {
        for (int a = 0; a < nvir_; a++) {
            int a_mo = rohf_.n_occ_alpha + a;  // virtual MO index
            for (int P = 0; P < naux_; P++) {
                double sum = 0.0;
                for (int nu = 0; nu < nbf_; nu++) {
                    sum += B_i(i, nu, P) * C(nu, a_mo);
                }
                int idx = i * nvir_ + a;
                B_ia_(idx, P) = sum;
            }
        }
    }
    
    std::cout << "  B_ia shape: [" << nocc_ * nvir_ << ", " << naux_ << "]\n";
    
    // DEBUG: Check B_ia values
    double max_B = 0.0;
    double norm_B = 0.0;
    int nonzero_count = 0;
    for (int i = 0; i < nocc_ * nvir_; i++) {
        for (int P = 0; P < naux_; P++) {
            double val = std::abs(B_ia_(i, P));
            max_B = std::max(max_B, val);
            norm_B += val * val;
            if (val > 1e-15) nonzero_count++;
        }
    }
    norm_B = std::sqrt(norm_B);
    std::cout << "  DEBUG: max|B_ia| = " << std::scientific << max_B 
              << ", norm = " << norm_B 
              << ", nonzeros = " << nonzero_count << std::fixed << "\n";
    std::cout << "  DEBUG: B_ia(0,0) = " << std::scientific << B_ia_(0,0) 
              << ", B_ia(0,1) = " << B_ia_(0,1) << std::fixed << "\n";
    
    // Find which (i*nvir+a) indices have nonzeros (first 5 only)
    std::cout << "  DEBUG: First 5 nonzero ia combinations:\n";
    int count = 0;
    for (int ia = 0; ia < nocc_ * nvir_ && count < 5; ia++) {
        double max_for_ia = 0.0;
        for (int P = 0; P < naux_; P++) {
            max_for_ia = std::max(max_for_ia, std::abs(B_ia_(ia, P)));
        }
        if (max_for_ia > 1e-10) {
            int i = ia / nvir_;
            int a = ia % nvir_;
            int a_mo = rohf_.n_occ_alpha + a;
            std::cout << "    ia=" << ia << " (i=" << i << ", a=" << a << ", a_mo=" << a_mo
                      << "): max|B| = " << std::scientific << max_for_ia << std::fixed << "\n";
            count++;
        }
    }
}

double DFMP2::compute_ss_energy() {
    // REFERENCE: Feyereisen et al. (1993), Eq. (11) + Szabo & Ostlund Eq. (6.74)
    // Same-spin MP2 with DF approximation
    //
    // Formula: E_SS = (1/4) Σ_ijab [<ij||ab>]^2 / D_ijab
    // where <ij||ab> = <ij|ab> - <ij|ba> (antisymmetrized)
    // and D_ijab = ε_i + ε_j - ε_a - ε_b
    //
    // DF integrals: <ij|ab> ≈ Σ_P B_tilde_ia * B_jb
    // where B_tilde = B * J^-1
    
    double e = 0.0;
    const auto& eps = rohf_.orbital_energies_alpha;
    
    // Apply J^{-1/2} transformation
    Eigen::MatrixXd B_tilde = B_ia_ * J_inv_;
    
    // Loop over occupied and virtual pairs
    for (int i = 0; i < nocc_; i++) {
        for (int j = 0; j < nocc_; j++) {
            for (int a = 0; a < nvir_; a++) {
                int a_mo = rohf_.n_occ_alpha + a;
                for (int b = 0; b < nvir_; b++) {
                    int b_mo = rohf_.n_occ_alpha + b;
                    
                    // Compute DF integrals
                    int idx_ia = i * nvir_ + a;
                    int idx_ib = i * nvir_ + b;
                    int idx_ja = j * nvir_ + a;
                    int idx_jb = j * nvir_ + b;
                    
                    // Direct: <ij|ab> with J^{-1/2}
                    double g_direct = 0.0;
                    for (int P = 0; P < naux_; P++) {
                        g_direct += B_tilde(idx_ia, P) * B_tilde(idx_jb, P);  // Both tilde!
                    }
                    
                    // Exchange: <ij|ba> with J^{-1/2}
                    double g_exch = 0.0;
                    for (int P = 0; P < naux_; P++) {
                        g_exch += B_tilde(idx_ib, P) * B_tilde(idx_ja, P);  // Both tilde!
                    }
                    
                    // Antisymmetrized: <ij||ab> = <ij|ab> - <ij|ba>
                    double g_antisym = g_direct - g_exch;
                    
                    // Energy denominator
                    double denom = eps(i) + eps(j) - eps(a_mo) - eps(b_mo);
                    
                    // Same-spin contribution with 1/4 factor
                    e += 0.25 * g_antisym * g_antisym / denom;
                }
            }
        }
    }
    
    return e;
}

double DFMP2::compute_os_energy() {
    // REFERENCE: Feyereisen et al. (1993), Eq. (11)
    // Opposite-spin MP2 with DF for ROHF
    //
    // Formula: E_OS = Σ_ijab <ij|ab>_DF^2 / D_ijab
    // where i,j = doubly occupied, a,b = virtual
    //
    // DF integral: <ij|ab> ≈ Σ_PQ B^P_ia [J^-1]_PQ B^Q_jb
    // 
    // For ROHF: use closed-shell (doubly occ) orbitals only
    
    double e = 0.0;
    const auto& eps = rohf_.orbital_energies_alpha;
    
    std::cout << "  DEBUG OS: nocc=" << nocc_ << ", nvir=" << nvir_ << "\n";
    
    // Apply J^{-1/2} transformation
    // With J^{-1/2}, integral is: <ij|ab> = Σ_P B̃_ia * B̃_jb
    Eigen::MatrixXd B_tilde = B_ia_ * J_inv_;
    
    int n_computed = 0;
    
    // Loop over doubly-occupied pairs (i,j) and virtual pairs (a,b)
    for (int i = 0; i < nocc_; i++) {
        for (int j = 0; j < nocc_; j++) {
            for (int a = 0; a < nvir_; a++) {
                int a_mo = rohf_.n_occ_alpha + a;
                for (int b = 0; b < nvir_; b++) {
                    int b_mo = rohf_.n_occ_alpha + b;
                    
                    // Compute DF integral <ij|ab> = Σ_P B̃_ia * B̃_jb (both tilde!)
                    int idx_ia = i * nvir_ + a;
                    int idx_jb = j * nvir_ + b;
                    
                    double g = 0.0;
                    for (int P = 0; P < naux_; P++) {
                        g += B_tilde(idx_ia, P) * B_tilde(idx_jb, P);  // Both use B_tilde!
                    }
                    
                    // Energy denominator: D = ε_i + ε_j - ε_a - ε_b
                    double denom = eps(i) + eps(j) - eps(a_mo) - eps(b_mo);
                    
                    // Opposite-spin contribution (no exchange)
                    e += g * g / denom;
                    n_computed++;
                    
                    if (n_computed == 1) {
                        std::cout << "  DEBUG: first term: i=" << i << " j=" << j 
                                  << " a=" << a << " b=" << b << "\n";
                        std::cout << "    g=" << std::scientific << g 
                                  << " denom=" << denom << std::fixed << "\n";
                    }
                }
            }
        }
    }
    
    std::cout << "  DEBUG: computed " << n_computed << " terms\n";
    
    return e;
}

} // namespace mshqc
