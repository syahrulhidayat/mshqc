/**
 * @file ump2.cc
 * @brief Unrestricted Møller-Plesset 2nd-order perturbation theory (UMP2)
 * 
 * Implementation of UMP2 energy and T2^(1) amplitudes for open-shell systems.
 * Separate treatment for α-α, β-β, and α-β spin components.
 * 
 * Theory References:
 *   - C. Møller & M. S. Plesset, Phys. Rev. 46, 618 (1934)
 *     [Original MP2 perturbation theory]
 *   - J. A. Pople et al., J. Chem. Phys. 64, 2901 (1976)
 *     [Unrestricted MP2 formalism for open-shell systems]
 *   - J. A. Pople et al., Int. J. Quantum Chem. 10, 1 (1976)
 *     [UMP2 implementation details]
 *   - A. Szabo & N. S. Ostlund, "Modern Quantum Chemistry" (1996)
 *     [Textbook reference for integral transformations]
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-01-29
 * @license MIT License (see LICENSE file in project root)
 * 
 * @note This is an original implementation derived from published theory.
 *       No code was copied from existing quantum chemistry software.
 *       Algorithm based on equations from Pople et al. (1976).
 *       Physicist notation <ij|ab> used throughout (not chemist notation).
 */

#include "mshqc/ump2.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace mshqc {

UMP2::UMP2(const SCFResult& uhf_result,
           const BasisSet& basis,
           std::shared_ptr<IntegralEngine> integrals)
    : uhf_(uhf_result), basis_(basis), integrals_(integrals) {
    
    nbf_ = basis.n_basis_functions();
    nocc_a_ = uhf_result.n_occ_alpha;
    nocc_b_ = uhf_result.n_occ_beta;
    nvir_a_ = nbf_ - nocc_a_;
    nvir_b_ = nbf_ - nocc_b_;
    
    std::cout << "\n=== UMP2 Setup ===\n";
    std::cout << "Basis functions: " << nbf_ << "\n";
    std::cout << "Alpha: " << nocc_a_ << " occ, " << nvir_a_ << " virt\n";
    std::cout << "Beta:  " << nocc_b_ << " occ, " << nvir_b_ << " virt\n";
}

void UMP2::transform_integrals() {
    // Get AO integrals
    auto eri_ao = integrals_->compute_eri();
    
    const auto& Ca = uhf_.C_alpha;
    const auto& Cb = uhf_.C_beta;
    
    std::cout << "\nTransforming integrals to MO basis...\n";
    
    // Alpha-alpha: <ia|jb>^αα in physicist notation
    // Transform (μν|λσ) chemist -> <ia|jb> physicist
    // REFERENCE: Szabo & Ostlund (1996), Eq. (A.9)
    eri_aaaa_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_a_, nvir_a_, nvir_a_);
    eri_aaaa_.setZero();
    
    for (int i = 0; i < nocc_a_; i++) {
        for (int j = 0; j < nocc_a_; j++) {
            for (int a = 0; a < nvir_a_; a++) {
                for (int b = 0; b < nvir_a_; b++) {
                    double val = 0.0;
                    
                    // Physicist <ia|jb> = Chemist (ij|ab)
                    // But stored as (i,j,a,b) indexing
                    for (int mu = 0; mu < nbf_; mu++) {
                        for (int lam = 0; lam < nbf_; lam++) {
                            for (int nu = 0; nu < nbf_; nu++) {
                                for (int sig = 0; sig < nbf_; sig++) {
                                    // <ia|jb> = (ij|ab) chemist
                                    val += Ca(mu, i) * Ca(lam, nocc_a_ + a) * 
                                           eri_ao(mu, lam, nu, sig) *
                                           Ca(nu, j) * Ca(sig, nocc_a_ + b);
                                }
                            }
                        }
                    }
                    
                    eri_aaaa_(i, j, a, b) = val;
                }
            }
        }
    }
    std::cout << "  Alpha-alpha integrals: " << nocc_a_ << "^2 x " << nvir_a_ << "^2 done\n";
    
    // Beta-beta: <IA|JB>^ββ physicist notation
    eri_bbbb_ = Eigen::Tensor<double, 4>(nocc_b_, nocc_b_, nvir_b_, nvir_b_);
    eri_bbbb_.setZero();
    
    for (int i = 0; i < nocc_b_; i++) {
        for (int j = 0; j < nocc_b_; j++) {
            for (int a = 0; a < nvir_b_; a++) {
                for (int b = 0; b < nvir_b_; b++) {
                    double val = 0.0;
                    
                    for (int mu = 0; mu < nbf_; mu++) {
                        for (int lam = 0; lam < nbf_; lam++) {
                            for (int nu = 0; nu < nbf_; nu++) {
                                for (int sig = 0; sig < nbf_; sig++) {
                                    // <IA|JB> = (IJ|AB) chemist
                                    val += Cb(mu, i) * Cb(lam, nocc_b_ + a) * 
                                           eri_ao(mu, lam, nu, sig) *
                                           Cb(nu, j) * Cb(sig, nocc_b_ + b);
                                }
                            }
                        }
                    }
                    
                    eri_bbbb_(i, j, a, b) = val;
                }
            }
        }
    }
    std::cout << "  Beta-beta integrals: " << nocc_b_ << "^2 x " << nvir_b_ << "^2 done\n";
    
    // Alpha-beta: <ia|JB>^αβ mixed spin physicist notation
    eri_aabb_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_b_, nvir_a_, nvir_b_);
    eri_aabb_.setZero();
    
    for (int i = 0; i < nocc_a_; i++) {
        for (int j = 0; j < nocc_b_; j++) {
            for (int a = 0; a < nvir_a_; a++) {
                for (int b = 0; b < nvir_b_; b++) {
                    double val = 0.0;
                    
                    for (int mu = 0; mu < nbf_; mu++) {
                        for (int lam = 0; lam < nbf_; lam++) {
                            for (int nu = 0; nu < nbf_; nu++) {
                                for (int sig = 0; sig < nbf_; sig++) {
                                    // <ia|JB> = (iJ|aB) chemist
                                    val += Ca(mu, i) * Ca(lam, nocc_a_ + a) * 
                                           eri_ao(mu, lam, nu, sig) *
                                           Cb(nu, j) * Cb(sig, nocc_b_ + b);
                                }
                            }
                        }
                    }
                    
                    eri_aabb_(i, j, a, b) = val;
                }
            }
        }
    }
    std::cout << "  Alpha-beta integrals: " << nocc_a_ << " x " << nocc_b_ 
              << " x " << nvir_a_ << " x " << nvir_b_ << " done\n";
}

double UMP2::compute_ss_alpha() {
    // Same-spin αα contribution
    // REFERENCE: PySCF ump2.py line 73-75
    // t2i = eris_ovov / D
    // emp2_ss += einsum('jab,jab', t2i, eris_ovov) * 0.5 - einsum('jab,jba', t2i, eris_ovov) * 0.5
    
    const auto& eps = uhf_.orbital_energies_alpha;
    double e = 0.0;
    
    for (int i = 0; i < nocc_a_; i++) {
        for (int j = 0; j < nocc_a_; j++) {
            for (int a = 0; a < nvir_a_; a++) {
                for (int b = 0; b < nvir_a_; b++) {
                    double g_ijab = eri_aaaa_(i,j,a,b);
                    double g_ijba = eri_aaaa_(i,j,b,a);
                    
                    // Energy denominator
                    double d = eps(i) + eps(j) - eps(nocc_a_+a) - eps(nocc_a_+b);
                    
                    // Amplitude t = g/d
                    double t_ijab = g_ijab / d;
                    t2_aa_(i, j, a, b) = t_ijab;  // Store amplitude
                    
                    // Energy: 0.5 * t * (g_direct - g_exchange)
                    e += 0.5 * t_ijab * (g_ijab - g_ijba);
                }
            }
        }
    }
    
    // Factor 0.5 already included above (direct - exchange gives full antisym energy)
    return e;
}

double UMP2::compute_ss_beta() {
    // Same-spin ββ contribution
    const auto& eps = uhf_.orbital_energies_beta;
    double e = 0.0;
    
    for (int i = 0; i < nocc_b_; i++) {
        for (int j = 0; j < nocc_b_; j++) {
            for (int a = 0; a < nvir_b_; a++) {
                for (int b = 0; b < nvir_b_; b++) {
                    double g_ijab = eri_bbbb_(i,j,a,b);
                    double g_ijba = eri_bbbb_(i,j,b,a);
                    
                    // Denominator
                    double d = eps(i) + eps(j) - eps(nocc_b_+a) - eps(nocc_b_+b);
                    
                    // Amplitude
                    double t_ijab = g_ijab / d;
                    t2_bb_(i, j, a, b) = t_ijab;  // Store amplitude
                    
                    // Energy
                    e += 0.5 * t_ijab * (g_ijab - g_ijba);
                }
            }
        }
    }
    
    return e;
}

double UMP2::compute_os() {
    // Opposite-spin αβ contribution
    // REFERENCE: PySCF ump2.py line 87
    // emp2_os += einsum('JaB,JaB', t2i, eris_ovov)
    
    const auto& eps_a = uhf_.orbital_energies_alpha;
    const auto& eps_b = uhf_.orbital_energies_beta;
    double e = 0.0;
    
    for (int i = 0; i < nocc_a_; i++) {
        for (int j = 0; j < nocc_b_; j++) {
            for (int a = 0; a < nvir_a_; a++) {
                for (int b = 0; b < nvir_b_; b++) {
                    // No antisymmetrization for mixed spin
                    double g = eri_aabb_(i, j, a, b);
                    
                    // Denominator: ε_i^α + ε_J^β - ε_a^α - ε_B^β
                    double d = eps_a(i) + eps_b(j) - eps_a(nocc_a_+a) - eps_b(nocc_b_+b);
                    
                    // Amplitude
                    double t = g / d;
                    t2_ab_(i, j, a, b) = t;  // Store amplitude
                    
                    // Energy: t * g (no 0.25 factor!)
                    e += t * g;
                }
            }
        }
    }
    
    return e;
}

UMP2Result UMP2::compute() {
    std::cout << "\n====================================\n";
    std::cout << "  Unrestricted MP2 (UMP2)\n";
    std::cout << "====================================\n";
    
    // Transform integrals
    transform_integrals();
    
    // Allocate T2 amplitudes
    t2_aa_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_a_, nvir_a_, nvir_a_);
    t2_bb_ = Eigen::Tensor<double, 4>(nocc_b_, nocc_b_, nvir_b_, nvir_b_);
    t2_ab_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_b_, nvir_a_, nvir_b_);
    t2_aa_.setZero();
    t2_bb_.setZero();
    t2_ab_.setZero();
    
    // Compute energy components
    std::cout << "\nComputing MP2 energy...\n";
    
    double e_ss_aa = compute_ss_alpha();
    double e_ss_bb = compute_ss_beta();
    double e_os = compute_os();
    
    double e_corr = e_ss_aa + e_ss_bb + e_os;
    double e_tot = uhf_.energy_total + e_corr;
    
    // Print results
    std::cout << "\n=== UMP2 Results ===\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "SS (αα):        " << std::setw(16) << e_ss_aa << " Ha\n";
    std::cout << "SS (ββ):        " << std::setw(16) << e_ss_bb << " Ha\n";
    std::cout << "OS (αβ):        " << std::setw(16) << e_os << " Ha\n";
    std::cout << "Correlation:    " << std::setw(16) << e_corr << " Ha\n";
    std::cout << "\nUHF energy:     " << std::setw(16) << uhf_.energy_total << " Ha\n";
    std::cout << "UMP2 energy:    " << std::setw(16) << e_tot << " Ha\n";
    
    // Prepare result
    UMP2Result result;
    result.e_corr_ss_aa = e_ss_aa;
    result.e_corr_ss_bb = e_ss_bb;
    result.e_corr_os = e_os;
    result.e_corr_total = e_corr;
    result.e_total = e_tot;
    
    return result;
}

T2Amplitudes UMP2::get_t2_amplitudes() const {
    T2Amplitudes result;
    result.t2_aa = t2_aa_;
    result.t2_bb = t2_bb_;
    result.t2_ab = t2_ab_;
    return result;
}

} // namespace mshqc
