/**
 * @file rmp2.cc
 * @brief Implementation of Restricted MP2 for closed-shell systems
 * 
 * THEORY REFERENCES:
 *   - C. Møller & M. S. Plesset, Phys. Rev. 46, 618 (1934)
 *   - A. Szabo & N. S. Ostlund, "Modern Quantum Chemistry" (1996), Ch. 6
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 * @license MIT
 */

#include "mshqc/foundation/rmp2.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace mshqc {
namespace foundation {

RMP2::RMP2(const SCFResult& rhf_result,
           const BasisSet& basis,
           std::shared_ptr<IntegralEngine> integrals)
    : rhf_(rhf_result), basis_(basis), integrals_(integrals) {
    
    nbf_ = basis_.n_basis_functions();
    nocc_ = rhf_.n_occ_alpha;  // For RHF, n_occ_alpha = n_occ_beta
    nvirt_ = nbf_ - nocc_;
    
    // Check that this is actually closed-shell (sanity check)
    if (rhf_.n_occ_alpha != rhf_.n_occ_beta) {
        throw std::runtime_error("RMP2 requires closed-shell reference (n_alpha == n_beta)");
    }
    
    std::cout << "RMP2: " << nocc_ << " occupied, " << nvirt_ << " virtual orbitals\n";
}

RMP2Result RMP2::compute() {
    std::cout << "\n=== RMP2 Calculation ===\n";
    std::cout << "Basis functions: " << nbf_ << "\n";
    std::cout << "Occupied orbitals: " << nocc_ << "\n";
    std::cout << "Virtual orbitals: " << nvirt_ << "\n";
    
    // Step 1: Transform integrals AO → MO
    std::cout << "Transforming integrals to MO basis...\n";
    transform_integrals_ao_to_mo();
    
    // Step 2: Compute T2 amplitudes
    std::cout << "Computing T2 amplitudes...\n";
    compute_t2_amplitudes();
    
    // Step 3: Calculate correlation energy
    std::cout << "Computing MP2 energy...\n";
    double e_corr = compute_correlation_energy();
    
    // Build result
    RMP2Result result;
    result.e_rhf = rhf_.energy_total;
    result.e_corr = e_corr;
    result.e_total = rhf_.energy_total + e_corr;
    result.n_occ = nocc_;
    result.n_virt = nvirt_;
    result.t2 = t2_;  // Copy T2 amplitudes
    
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "\nRHF energy:          " << result.e_rhf << " Ha\n";
    std::cout << "MP2 correlation:     " << result.e_corr << " Ha\n";
    std::cout << "Total RMP2 energy:   " << result.e_total << " Ha\n";
    
    return result;
}

void RMP2::transform_integrals_ao_to_mo() {
    // REFERENCE: Szabo & Ostlund (1996), Eq. (2.282)
    // Four-index transformation: <pq|rs>_MO = Σ_μνλσ C_μp C_νq (μν|λσ)_AO C_λr C_σs
    
    // Get MO coefficients (use alpha coefficients, same as beta for RHF)
    const Eigen::MatrixXd& C = rhf_.C_alpha;
    
    // Compute all AO integrals first
    std::cout << "  Computing AO integrals...\n";
    Eigen::Tensor<double, 4> eri_ao = integrals_->compute_eri();
    
    // Allocate MO integral tensor
    // Only need occupied-occupied-virtual-virtual block <ij|ab>
    eri_mo_ = Eigen::Tensor<double, 4>(nocc_, nocc_, nvirt_, nvirt_);
    eri_mo_.setZero();
    
    std::cout << "  Transforming integrals AO -> MO...\n";
    
    // Naive O(N^8) transformation (can optimize later)
    // For now, correctness > speed
    for (int i = 0; i < nocc_; i++) {
        for (int j = 0; j < nocc_; j++) {
            for (int a = 0; a < nvirt_; a++) {
                int a_mo = nocc_ + a;  // Virtual MO index
                for (int b = 0; b < nvirt_; b++) {
                    int b_mo = nocc_ + b;
                    
                    double eri_ijab = 0.0;
                    
                    // Sum over AO basis
                    for (int mu = 0; mu < nbf_; mu++) {
                        for (int nu = 0; nu < nbf_; nu++) {
                            for (int lam = 0; lam < nbf_; lam++) {
                                for (int sig = 0; sig < nbf_; sig++) {
                                    // Transform to MO (physicist notation)
                                    // <ij|ab> = (ia|jb) in chemist notation
                                    eri_ijab += C(mu, i) * C(lam, a_mo) * eri_ao(mu, nu, lam, sig) * 
                                               C(nu, j) * C(sig, b_mo);
                                }
                            }
                        }
                    }
                    
                    eri_mo_(i, j, a, b) = eri_ijab;
                }
            }
        }
    }
    
    std::cout << "Integral transformation complete.\n";
}

void RMP2::compute_t2_amplitudes() {
    // REFERENCE: Szabo & Ostlund (1996), Eq. (6.63)
    // Amplitude: t_ij^ab = <ij|ab> / (ε_i + ε_j - ε_a - ε_b)
    
    const Eigen::VectorXd& eps = rhf_.orbital_energies_alpha;
    
    // Allocate T2 tensor
    t2_ = Eigen::Tensor<double, 4>(nocc_, nocc_, nvirt_, nvirt_);
    
    for (int i = 0; i < nocc_; i++) {
        for (int j = 0; j < nocc_; j++) {
            for (int a = 0; a < nvirt_; a++) {
                int a_mo = nocc_ + a;
                for (int b = 0; b < nvirt_; b++) {
                    int b_mo = nocc_ + b;
                    
                    // Orbital energy denominator
                    // NOTE: Always negative for occ→virt excitation
                    double denom = eps(i) + eps(j) - eps(a_mo) - eps(b_mo);
                    
                    if (std::abs(denom) < 1e-10) {
                        throw std::runtime_error("MP2: near-zero denominator (degenerate orbitals?)");
                    }
                    
                    // T2 amplitude from direct integral
                    t2_(i, j, a, b) = eri_mo_(i, j, a, b) / denom;
                }
            }
        }
    }
    
    std::cout << "T2 amplitudes computed.\n";
}

double RMP2::compute_correlation_energy() {
    // REFERENCE: Szabo & Ostlund (1996), Eq. (6.74), p. 354
    // E^(2) = Σ_ijab (2<ij|ab> - <ij|ba>) * t_ij^ab
    //
    // This is the spin-adapted formula for closed-shell systems.
    // Factor of 2 accounts for spin (α and β contribute equally).
    
    double e_mp2 = 0.0;
    
    for (int i = 0; i < nocc_; i++) {
        for (int j = 0; j < nocc_; j++) {
            for (int a = 0; a < nvirt_; a++) {
                for (int b = 0; b < nvirt_; b++) {
                    double direct = eri_mo_(i, j, a, b);   // <ij|ab>
                    double exchange = eri_mo_(i, j, b, a); // <ij|ba> (b and a swapped)
                    double t_ijab = t2_(i, j, a, b);
                    
                    // Spin-adapted formula
                    e_mp2 += (2.0 * direct - exchange) * t_ijab;
                }
            }
        }
    }
    
    return e_mp2;
}

const Eigen::Tensor<double, 4>& RMP2::get_t2_amplitudes() const {
    if (t2_.size() == 0) {
        throw std::runtime_error("T2 amplitudes not available. Call compute() first.");
    }
    return t2_;
}

} // namespace foundation
} // namespace mshqc
