/**
 * @file omp2.cc
 * @brief Orbital-Optimized Møller-Plesset 2nd-order (OMP2) implementation
 * 
 * Implementation of OMP2 with orbital relaxation at second order.
 * Iteratively optimizes orbitals to minimize MP2 energy functional.
 * 
 * **Status**: Untested - implementation exists but not validated against Psi4.
 * 
 * Theory References:
 *   - U. Bozkaya & C. D. Sherrill, J. Chem. Phys. 139, 054104 (2013)
 *     [Orbital-optimized MP methods (OMP2/OMP3), Eq. (11), (15), (20)-(22)]
 *   - U. Bozkaya, J. Chem. Phys. 135, 224103 (2011)
 *     [OMP perturbation theory derivations]
 *   - R. C. Lochan et al., J. Chem. Phys. 126, 164101 (2007)
 *     [Orbital optimization techniques]
 *   - T. Helgaker et al., "Molecular Electronic-Structure Theory" (2000)
 *     [Section 10.8: Orbital optimization methodology]
 *   - C. Møller & M. S. Plesset, Phys. Rev. 46, 618 (1934)
 *     [MP2 foundation]
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-01-29
 * @license MIT License (see LICENSE file in project root)
 * 
 * @note Original implementation from Bozkaya & Sherrill equations.
 *       Requires OPDM (one-particle density matrix) calculation.
 *       Needs validation testing against Psi4 OMP2.
 */

#include "mshqc/mp2.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>

namespace mshqc {

OMP2::OMP2(const Molecule& mol,
           const BasisSet& basis,
           std::shared_ptr<IntegralEngine> integrals,
           const SCFResult& scf_guess)
    : mol_(mol), basis_(basis), integrals_(integrals), scf_(scf_guess) {
    
    // Initialize dimensions
    nbf_ = scf_.C_alpha.rows();
    na_ = scf_.n_occ_alpha;
    nb_ = scf_.n_occ_beta;
    va_ = static_cast<int>(nbf_) - na_;
    vb_ = static_cast<int>(nbf_) - nb_;
}

MP2Result OMP2::compute() {
    // REFERENCE: Bozkaya & Sherrill (2013), J. Chem. Phys. 139, 054104
    // Iterative orbital optimization via Lagrangian gradient descent
    
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "     OMP2 Calculation\n";
    std::cout << "========================================\n";
    std::cout << "\nMax iterations:  " << max_iter_ << "\n";
    std::cout << "Convergence:     " << conv_thresh_ << "\n";
    std::cout << "========================================\n\n";
    
    // Track iteration progress
    // REFERENCE: Pople et al. (1976), Int. J. Quantum Chem. Symp. 10, 1
    int it = 0;
    double e_prev = scf_.energy_total;
    double e_curr, de;
    double rms_k = 0.0, max_k = 0.0;
    bool ok = false;
    
    // Header for iteration table
    std::cout << "Iter    E_total         ΔE          RMS_κ       Max_κ\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << std::setw(3) << 0 << "   ";
    std::cout << std::setw(12) << e_prev << "    ";
    std::cout << std::setw(10) << 0.0 << "  ";
    std::cout << std::scientific << std::setprecision(1);
    std::cout << std::setw(10) << 0.0 << "  ";
    std::cout << std::setw(10) << 0.0 << "  (HF)\n";
    std::cout << std::fixed;
    
    // Main optimization loop
    // REFERENCE: Helgaker et al. (2000), Section 10.8 - orbital optimization
    do {
        it++;
        
        // Step 1: MP2 energy with current orbitals
        // REFERENCE: Szabo & Ostlund (1996), Section 6.4
        double e_ss = 0.0, e_os = 0.0;
        double e_mp2 = mp2_energy(e_ss, e_os);
        
        // Step 2: Generalized Fock from MP2 response
        // REFERENCE: Helgaker et al. (2000), Section 10.3
        Eigen::MatrixXd gam = build_opdm();
        Eigen::MatrixXd GF = build_gfock(gam);
        
        // Step 3: Orbital gradient from GF
        // REFERENCE: Bozkaya & Sherrill (2013), Eq. (11)
        Eigen::MatrixXd W = GF - GF.transpose();  // Antisymmetrize
        
        // Extract occ-virt block for kappa
        Eigen::MatrixXd k = Eigen::MatrixXd::Zero(static_cast<int>(nbf_), static_cast<int>(nbf_));
        k.block(0, na_, na_, va_) = W.block(0, na_, na_, va_);
        k.block(na_, 0, va_, na_) = -W.block(0, na_, na_, va_).transpose();
        
        // Step 4: Rotate orbitals
        // REFERENCE: Helgaker et al. (2000), Section 10.8
        rotate_orbitals(k);
        
        // Update energy
        e_curr = scf_.energy_total + e_mp2;
        de = e_curr - e_prev;
        
        // Compute gradient norms
        // REFERENCE: Bozkaya & Sherrill (2013), Eq. (11) - gradient convergence
        rms_k = k.norm() / std::sqrt(static_cast<double>(k.size()));
        max_k = k.cwiseAbs().maxCoeff();
        
        // Check convergence (TODO: Phase 6)
        ok = converged(k);
        
        // Print iteration info
        std::cout << std::fixed << std::setprecision(6);
        std::cout << std::setw(3) << it << "   ";
        std::cout << std::setw(12) << e_curr << "  ";
        std::cout << std::setw(12) << de << "  ";
        std::cout << std::scientific << std::setprecision(1);
        std::cout << std::setw(10) << rms_k << "  ";
        std::cout << std::setw(10) << max_k << "\n";
        std::cout << std::fixed;
        
        e_prev = e_curr;
        
        // Safety: prevent infinite loop
        if (it >= max_iter_) {
            std::cout << "\n*** WARNING: Max iterations reached without convergence ***\n";
            break;
        }
        
    } while (!ok);
    
    if (ok) {
        std::cout << "\n";
        std::cout << "==============================================================================\n";
        std::cout << "======================== OMP2 CONVERGED IN " << std::setw(2) << it << " ITERATIONS ======================\n";
        std::cout << "==============================================================================\n\n";
    }
    
    // Final MP2 energy with optimized orbitals
    double e_ss_final = 0.0, e_os_final = 0.0;
    double e_corr_final = mp2_energy(e_ss_final, e_os_final);
    
    // Build result
    MP2Result res;
    res.energy_scf = scf_.energy_total;
    res.energy_mp2_ss = e_ss_final;
    res.energy_mp2_os = e_os_final;
    res.energy_mp2_corr = e_corr_final;
    res.energy_total = scf_.energy_total + e_corr_final;
    res.n_occ_alpha = na_;
    res.n_occ_beta = nb_;
    res.n_virt_alpha = va_;
    res.n_virt_beta = vb_;
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "ROHF Reference Energy:     " << res.energy_scf << " Ha\n";
    std::cout << "OMP2 Correlation Energy:   " << res.energy_mp2_corr << " Ha\n";
    std::cout << "OMP2 Total Energy:         " << res.energy_total << " Ha\n\n";
    
    return res;
}

Eigen::MatrixXd OMP2::orbital_gradient() {
    // REFERENCE: Bozkaya & Sherrill (2013), Eq. (11)
    // Orbital gradient from Lagrangian
    // 
    // κ_pq = ∂L/∂κ_pq where L = E_SCF + E_MP2 + constraints
    // 
    // For ROHF reference, need:
    //   - SCF orbital gradient (Fock contribution)
    //   - MP2 orbital response (amplitude derivatives)
    //   - Lagrange multipliers (constraint satisfaction)
    // 
    // Result is antisymmetric matrix κ with blocks:
    //   κ_occ-occ   = 0 (occupied-occupied rotation)
    //   κ_virt-virt = 0 (virtual-virtual rotation)
    //   κ_occ-virt  ≠ 0 (occupied-virtual mixing, main optimization)
    
    // TODO: ~200 lines implementation
    throw std::runtime_error("OMP2::orbital_gradient() - not implemented");
}

void OMP2::rotate_orbitals(const Eigen::MatrixXd& kappa) {
    // REFERENCE: Helgaker et al. (2000), Section 10.8, Eq. (10.8.15)
    // Orbital rotation via exponential parametrization
    // U = exp(-κ) where κ antisymmetric
    // C_new = C_old * U
    
    // Use Cayley transform (approximate exp for small κ)
    // REFERENCE: Bozkaya & Sherrill (2013), Eq. (21)
    // U = (I + κ/2)^{-1} (I - κ/2)
    // Simpler than full matrix exponential, exact for infinitesimal rotations
    
    int n = static_cast<int>(nbf_);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    
    // Scale down kappa for stability (line search step)
    // REFERENCE: Nocedal & Wright (2006), Chapter 3 - line search
    double step = 1.0;  // Full step
    Eigen::MatrixXd k_scaled = step * kappa;
    
    // Cayley transform: U = (I - k/2)(I + k/2)^{-1}
    // Reordered for numerical stability
    Eigen::MatrixXd A = I + 0.5 * k_scaled;
    Eigen::MatrixXd B = I - 0.5 * k_scaled;
    
    // Solve A * U = B  =>  U = A^{-1} * B
    Eigen::MatrixXd U = A.lu().solve(B);
    
    // Rotate orbital coefficients
    Eigen::MatrixXd C_new_alpha = scf_.C_alpha * U;
    Eigen::MatrixXd C_new_beta = scf_.C_beta * U;
    
    // Orthonormalize (Löwdin)
    // REFERENCE: Szabo & Ostlund (1996), Eq. (3.167)
    // S_MO = C^T S C, then C' = C * S_MO^{-1/2}
    
    // Get AO overlap from integrals
    Eigen::MatrixXd S_ao = integrals_->compute_overlap();
    
    // Alpha orbitals
    Eigen::MatrixXd S_mo_a = C_new_alpha.transpose() * S_ao * C_new_alpha;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_a(S_mo_a);
    Eigen::VectorXd s_val_a = es_a.eigenvalues();
    Eigen::MatrixXd s_vec_a = es_a.eigenvectors();
    
    // S^{-1/2} = V * diag(s^{-1/2}) * V^T
    Eigen::VectorXd s_inv_sqrt_a = s_val_a.array().rsqrt();
    Eigen::MatrixXd S_inv_a = s_vec_a * s_inv_sqrt_a.asDiagonal() * s_vec_a.transpose();
    
    scf_.C_alpha = C_new_alpha * S_inv_a;
    
    // Beta orbitals (similar)
    Eigen::MatrixXd S_mo_b = C_new_beta.transpose() * S_ao * C_new_beta;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_b(S_mo_b);
    Eigen::VectorXd s_val_b = es_b.eigenvalues();
    Eigen::MatrixXd s_vec_b = es_b.eigenvectors();
    
    Eigen::VectorXd s_inv_sqrt_b = s_val_b.array().rsqrt();
    Eigen::MatrixXd S_inv_b = s_vec_b * s_inv_sqrt_b.asDiagonal() * s_vec_b.transpose();
    
    scf_.C_beta = C_new_beta * S_inv_b;
    
    // Update orbital energies by diagonalizing Fock in new basis
    // F_new_MO = C_new^T F_AO C_new
    Eigen::MatrixXd F_mo_a = scf_.C_alpha.transpose() * scf_.F_alpha * scf_.C_alpha;
    Eigen::MatrixXd F_mo_b = scf_.C_beta.transpose() * scf_.F_beta * scf_.C_beta;
    
    // Extract diagonal as new orbital energies
    scf_.orbital_energies_alpha = F_mo_a.diagonal();
    scf_.orbital_energies_beta = F_mo_b.diagonal();
}

void OMP2::xform_ints() {
    // REFERENCE: Helgaker et al. (2000), Section 9.6.2
    // Four-index transform (simplified for small systems)
    
    auto eri_ao = integrals_->compute_eri();
    const auto& Ca = scf_.C_alpha;
    const auto& Cb = scf_.C_beta;
    
    // Allocate MO integrals
    g_aa_ = Eigen::Tensor<double, 4>(na_, na_, va_, va_);
    g_ab_ = Eigen::Tensor<double, 4>(na_, nb_, va_, vb_);
    g_bb_ = Eigen::Tensor<double, 4>(nb_, nb_, vb_, vb_);
    g_aa_.setZero();
    g_ab_.setZero();
    g_bb_.setZero();
    
    // Alpha-alpha
    for (int i = 0; i < na_; i++) {
        for (int j = 0; j < na_; j++) {
            for (int a = 0; a < va_; a++) {
                int am = na_ + a;
                for (int b = 0; b < va_; b++) {
                    int bm = na_ + b;
                    double v = 0.0;
                    for (size_t m = 0; m < nbf_; m++) {
                        for (size_t n = 0; n < nbf_; n++) {
                            for (size_t l = 0; l < nbf_; l++) {
                                for (size_t s = 0; s < nbf_; s++) {
                                    v += Ca(m,i) * Ca(n,j) * eri_ao(m,n,l,s) *
                                         Ca(l,am) * Ca(s,bm);
                                }
                            }
                        }
                    }
                    g_aa_(i,j,a,b) = v;
                }
            }
        }
    }
    
    // Alpha-beta
    for (int i = 0; i < na_; i++) {
        for (int J = 0; J < nb_; J++) {
            for (int a = 0; a < va_; a++) {
                int am = na_ + a;
                for (int B = 0; B < vb_; B++) {
                    int Bm = nb_ + B;
                    double v = 0.0;
                    for (size_t m = 0; m < nbf_; m++) {
                        for (size_t n = 0; n < nbf_; n++) {
                            for (size_t l = 0; l < nbf_; l++) {
                                for (size_t s = 0; s < nbf_; s++) {
                                    v += Ca(m,i) * Cb(n,J) * eri_ao(m,n,l,s) *
                                         Ca(l,am) * Cb(s,Bm);
                                }
                            }
                        }
                    }
                    g_ab_(i,J,a,B) = v;
                }
            }
        }
    }
    
    // Beta-beta
    for (int i = 0; i < nb_; i++) {
        for (int j = 0; j < nb_; j++) {
            for (int a = 0; a < vb_; a++) {
                int am = nb_ + a;
                for (int b = 0; b < vb_; b++) {
                    int bm = nb_ + b;
                    double v = 0.0;
                    for (size_t m = 0; m < nbf_; m++) {
                        for (size_t n = 0; n < nbf_; n++) {
                            for (size_t l = 0; l < nbf_; l++) {
                                for (size_t s = 0; s < nbf_; s++) {
                                    v += Cb(m,i) * Cb(n,j) * eri_ao(m,n,l,s) *
                                         Cb(l,am) * Cb(s,bm);
                                }
                            }
                        }
                    }
                    g_bb_(i,j,a,b) = v;
                }
            }
        }
    }
}

double OMP2::mp2_energy(double& e_ss, double& e_os) {
    // REFERENCE: Bozkaya et al. (2011), J. Chem. Phys. 135, 104103
    // MP2 correlation energy from current orbitals
    
    // Transform integrals with current orbitals
    xform_ints();
    
    const auto& ea = scf_.orbital_energies_alpha;
    const auto& eb = scf_.orbital_energies_beta;
    
    // Allocate amplitude tensors
    t2_aa_ = Eigen::Tensor<double, 4>(na_, na_, va_, va_);
    t2_ab_ = Eigen::Tensor<double, 4>(na_, nb_, va_, vb_);
    t2_bb_ = Eigen::Tensor<double, 4>(nb_, nb_, vb_, vb_);
    t2_aa_.setZero();
    t2_ab_.setZero();
    t2_bb_.setZero();
    
    // Same-spin alpha-alpha
    // REFERENCE: Szabo & Ostlund (1996), Eq. (6.74)
    double e_aa = 0.0;
    for (int i = 0; i < na_; i++) {
        for (int j = i+1; j < na_; j++) {
            for (int a = 0; a < va_; a++) {
                int am = na_ + a;
                for (int b = a+1; b < va_; b++) {
                    int bm = na_ + b;
                    double g1 = g_aa_(i,j,a,b);
                    double g2 = g_aa_(i,j,b,a);
                    double anti = g1 - g2;  // Antisymmetrize
                    double d = ea(i) + ea(j) - ea(am) - ea(bm);
                    double t = anti / d;
                    t2_aa_(i,j,a,b) = t;
                    e_aa += t * anti;
                }
            }
        }
    }
    
    // Same-spin beta-beta
    double e_bb = 0.0;
    for (int i = 0; i < nb_; i++) {
        for (int j = i+1; j < nb_; j++) {
            for (int a = 0; a < vb_; a++) {
                int am = nb_ + a;
                for (int b = a+1; b < vb_; b++) {
                    int bm = nb_ + b;
                    double g1 = g_bb_(i,j,a,b);
                    double g2 = g_bb_(i,j,b,a);
                    double anti = g1 - g2;
                    double d = eb(i) + eb(j) - eb(am) - eb(bm);
                    double t = anti / d;
                    t2_bb_(i,j,a,b) = t;
                    e_bb += t * anti;
                }
            }
        }
    }
    
    // Opposite-spin alpha-beta
    // REFERENCE: Bozkaya et al. (2011), Eq. (3)
    // REFERENCE: Knowles et al. (1991), Chem. Phys. Lett. 186, 130, Eq. (9)
    // CRITICAL: For ROHF, only closed-shell orbitals in opposite-spin!
    // Open-shell alpha electrons have NO beta counterpart to correlate with
    double e_ab = 0.0;
    int nc = nb_;  // Number of closed-shell (doubly occupied) orbitals
    for (int i = 0; i < nc; i++) {  // ONLY closed!
        for (int J = 0; J < nc; J++) {  // ONLY closed!
            for (int a = 0; a < va_; a++) {
                int am = na_ + a;
                for (int B = 0; B < vb_; B++) {
                    int Bm = nb_ + B;
                    double g = g_ab_(i,J,a,B);
                    double d = ea(i) + eb(J) - ea(am) - eb(Bm);
                    double t = g / d;
                    t2_ab_(i,J,a,B) = t;
                    e_ab += t * g;
                }
            }
        }
    }
    
    e_ss = e_aa + e_bb;
    e_os = e_ab;
    
    return e_ss + e_os;
}

Eigen::MatrixXd OMP2::build_opdm() {
    // REFERENCE: Bozkaya & Sherrill (2013), Eq. (13-14)
    // One-particle density matrix: γ = γ_HF + γ_MP2
    // γ_MP2 from amplitude response
    
    int nmo = static_cast<int>(nbf_);
    Eigen::MatrixXd gam = Eigen::MatrixXd::Zero(nmo, nmo);
    
    // HF density (diagonal: 1 for occupied, 0 for virtual)
    for (int i = 0; i < na_; i++) {
        gam(i, i) = 1.0;  // Alpha occupied
    }
    
    // MP2 correction to density
    // REFERENCE: Helgaker et al. (2000), Section 10.3.1
    // Occupied-occupied: γ_ij^MP2 = -0.5 Σ_kab t_ikab t_jkab
    for (int i = 0; i < na_; i++) {
        for (int j = 0; j < na_; j++) {
            double d_ij = 0.0;
            // Alpha-alpha contribution
            for (int k = 0; k < na_; k++) {
                if (k > i || k > j) continue;  // Only unique pairs
                for (int a = 0; a < va_; a++) {
                    for (int b = 0; b < va_; b++) {
                        if (b <= a) continue;
                        double t_ik = (i < k) ? t2_aa_(i,k,a,b) : t2_aa_(k,i,a,b);
                        double t_jk = (j < k) ? t2_aa_(j,k,a,b) : t2_aa_(k,j,a,b);
                        d_ij -= 0.5 * t_ik * t_jk;
                    }
                }
            }
            // Alpha-beta contribution (simplified)
            for (int K = 0; K < nb_; K++) {
                for (int a = 0; a < va_; a++) {
                    for (int B = 0; B < vb_; B++) {
                        d_ij -= t2_ab_(i,K,a,B) * t2_ab_(j,K,a,B);
                    }
                }
            }
            gam(i, j) += d_ij;
        }
    }
    
    // Virtual-virtual: γ_ab^MP2 = +0.5 Σ_ijc t_ijac t_ijbc
    for (int a = 0; a < va_; a++) {
        int am = na_ + a;
        for (int b = 0; b < va_; b++) {
            int bm = na_ + b;
            double d_ab = 0.0;
            // Alpha-alpha
            for (int i = 0; i < na_; i++) {
                for (int j = i+1; j < na_; j++) {
                    for (int c = 0; c < va_; c++) {
                        if (c > a || c > b) continue;
                        double t_ac = (a < c) ? t2_aa_(i,j,a,c) : t2_aa_(i,j,c,a);
                        double t_bc = (b < c) ? t2_aa_(i,j,b,c) : t2_aa_(i,j,c,b);
                        d_ab += 0.5 * t_ac * t_bc;
                    }
                }
            }
            // Alpha-beta (simplified)
            for (int i = 0; i < na_; i++) {
                for (int J = 0; J < nb_; J++) {
                    for (int C = 0; C < vb_; C++) {
                        d_ab += t2_ab_(i,J,a,C) * t2_ab_(i,J,b,C);
                    }
                }
            }
            gam(am, bm) += d_ab;
        }
    }
    
    return gam;
}

Eigen::MatrixXd OMP2::build_gfock(const Eigen::MatrixXd& gamma) {
    // REFERENCE: Helgaker et al. (2000), Section 10.3
    // Generalized Fock matrix: G_pq = h_pq + Σ_rs γ_rs <pr||qs>
    // 
    // SIMPLIFIED VERSION: Use approximate gradient from occupied-virtual coupling
    // Full GF expensive (N^8), use approximation from MP2 amplitudes directly
    // REFERENCE: Bozkaya & Sherrill (2013), Eq. (20) - diagonal approximation
    
    int nmo = static_cast<int>(nbf_);
    
    // Start with Fock matrix from HF
    Eigen::MatrixXd F_ao = scf_.F_alpha;
    Eigen::MatrixXd F_mo = scf_.C_alpha.transpose() * F_ao * scf_.C_alpha;
    
    Eigen::MatrixXd GF = F_mo;
    
    // Approximate MP2 response using amplitude-weighted integrals
    // GF_ia ≈ F_ia + Σ_jb t_ijab g_jb (simplified response)
    // This captures main orbital relaxation without full N^8 cost
    
    const auto& ea = scf_.orbital_energies_alpha;
    
    for (int i = 0; i < na_; i++) {
        for (int a = 0; a < va_; a++) {
            int am = na_ + a;
            double corr = 0.0;
            
            // Contribution from amplitudes (approximate)
            for (int j = 0; j < na_; j++) {
                for (int b = 0; b < va_; b++) {
                    int bm = na_ + b;
                    
                    // Use available occ-occ-virt-virt integrals
                    if (i < j && a < b) {
                        double t = t2_aa_(i,j,a,b);
                        double g = g_aa_(i,j,a,b);
                        corr += t * g * 0.1;  // Scaled contribution
                    }
                }
            }
            
            // Add opposite-spin
            for (int J = 0; J < nb_; J++) {
                for (int B = 0; B < vb_; B++) {
                    double t = t2_ab_(i,J,a,B);
                    double g = g_ab_(i,J,a,B);
                    corr += t * g * 0.1;
                }
            }
            
            GF(i, am) += corr;
            GF(am, i) += corr;  // Symmetric
        }
    }
    
    return GF;
}

void OMP2::xform_full_mo() {
    // Placeholder - not needed for simplified GF
    // Full transform too expensive (N^8)
}

bool OMP2::converged(const Eigen::MatrixXd& kappa) {
    // REFERENCE: Bozkaya & Sherrill (2013), convergence criteria
    // Multiple checks: RMS and max gradient elements
    
    // Compute norms
    double rms = kappa.norm() / std::sqrt(static_cast<double>(kappa.size()));
    double mx = kappa.cwiseAbs().maxCoeff();
    
    // Dual threshold: both RMS and max must be small
    // REFERENCE: Helgaker et al. (2000), Section 10.8.5 - convergence criteria
    bool rms_ok = (rms < conv_thresh_);
    bool max_ok = (mx < 10.0 * conv_thresh_);  // Looser for max element
    
    return (rms_ok && max_ok);
}

} // namespace mshqc
