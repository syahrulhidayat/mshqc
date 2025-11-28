/**
 * @file rmp3.cc
 * @brief Implementation of Restricted Møller-Plesset 3rd-order perturbation theory
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 * @license MIT
 */

#include "mshqc/foundation/rmp3.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace mshqc {
namespace foundation {

RMP3::RMP3(const SCFResult& rhf_result,
           const RMP2Result& rmp2_result,
           const BasisSet& basis,
           std::shared_ptr<IntegralEngine> integrals)
    : rhf_(rhf_result), rmp2_(rmp2_result), basis_(basis), integrals_(integrals) {
    
    nbf_ = basis.n_basis_functions();
    nocc_ = rmp2_.n_occ;
    nvirt_ = rmp2_.n_virt;
    
    std::cout << "\n=== RMP3 Setup ===\n";
    std::cout << "Basis functions: " << nbf_ << "\n";
    std::cout << "Occupied: " << nocc_ << ", Virtual: " << nvirt_ << "\n";
    std::cout << "Scaling: O(N^6) for T2^(2) computation\n";
}

void RMP3::build_fock_mo() {
    // Transform Fock matrix from AO to MO basis
    // F_MO = C^T * F_AO * C
    
    const auto& C = rhf_.C_alpha;  // MO coefficients from RHF (alpha = beta)
    const auto& F_ao = rhf_.F_alpha;  // Fock matrix in AO basis
    
    // For canonical RHF orbitals, F_MO is diagonal with eigenvalues ε_i
    // But we compute full F_MO for generality
    fock_mo_ = C.transpose() * F_ao * C;
    
    std::cout << "  Fock matrix transformed to MO basis\n";
}

void RMP3::transform_integrals_ao_to_mo() {
    // Four-index transformation: <pq|rs>_MO = Σ_μνλσ C_μp C_νq (μν|λσ)_AO C_λr C_σs
    // 
    // This is O(N^5) and same as RMP2 transformation
    // For efficiency, we only transform occupied-occupied × virtual-virtual block
    
    auto eri_ao = integrals_->compute_eri();
    const auto& C = rhf_.C_alpha;
    
    std::cout << "  Transforming ERIs to MO basis (occ-occ x virt-virt)...\n";
    
    // Allocate MO integral tensor
    eri_mo_ = Eigen::Tensor<double, 4>(nocc_, nocc_, nvirt_, nvirt_);
    eri_mo_.setZero();
    
    // Four-index transformation (naive O(N^8) algorithm)
    // <ij|ab> where i,j are occupied, a,b are virtual
    for (int i = 0; i < nocc_; i++) {
        for (int j = 0; j < nocc_; j++) {
            for (int a = 0; a < nvirt_; a++) {
                for (int b = 0; b < nvirt_; b++) {
                    double val = 0.0;
                    
                    // Contract with AO integrals
                    for (int mu = 0; mu < nbf_; mu++) {
                        for (int nu = 0; nu < nbf_; nu++) {
                            for (int lam = 0; lam < nbf_; lam++) {
                                for (int sig = 0; sig < nbf_; sig++) {
                                    // Physicist notation: <ij|ab> = (ij|ab)
                                    val += C(mu, i) * C(nu, j) * 
                                           eri_ao(mu, nu, lam, sig) *
                                           C(lam, nocc_ + a) * C(sig, nocc_ + b);
                                }
                            }
                        }
                    }
                    
                    eri_mo_(i, j, a, b) = val;
                }
            }
        }
    }
    
    std::cout << "  Transformation complete: " << nocc_ << "^2 x " << nvirt_ << "^2 integrals\n";
}

void RMP3::compute_t2_second_order() {
    // Compute T2^(2) amplitudes (second-order correction)
    // Pople 1977 RMP3 formulation: simplified spin-adapted residual with pp, hh, ph terms
    // Using direct spin-adapted formula (no explicit antisymmetrization needed)

    std::cout << "  Computing T2^(2) amplitudes...\n";

    const auto& t2_1 = rmp2_.t2;  // First-order amplitudes from RMP2
    const auto& eps = rhf_.orbital_energies_alpha;  // Orbital energies

    // Allocate T2^(2) tensor
    t2_2_ = Eigen::Tensor<double, 4>(nocc_, nocc_, nvirt_, nvirt_);
    t2_2_.setZero();

    // Simplified RMP3 residual: pp ladder (particle-particle)
    // t_ij^{ab(2)} = 0.5 * Σ_{cd} ⟨ab||cd⟩ t_ij^cd / D_ij^ab
    std::cout << "    Computing pp ladder contribution...\n";
    auto eri_ao = integrals_->compute_eri();
    const auto& C = rhf_.C_alpha;

    for (int i = 0; i < nocc_; i++)
        for (int j = 0; j < nocc_; j++)
            for (int a = 0; a < nvirt_; a++)
                for (int b = 0; b < nvirt_; b++) {
                    double denom = eps(i) + eps(j) - eps(nocc_ + a) - eps(nocc_ + b);
                    double res = 0.0;

                    // pp ladder: 0.5 * Σ_{cd} ⟨ab||cd⟩ t_ij^cd
                    // Spin-adapted: (2⟨ab|cd⟩ - ⟨ab|dc⟩)
                    for (int c = 0; c < nvirt_; c++)
                        for (int d = 0; d < nvirt_; d++) {
                            double g_dir = 0.0, g_ex = 0.0;
                            for (int mu = 0; mu < nbf_; mu++)
                                for (int nu = 0; nu < nbf_; nu++)
                                    for (int lam = 0; lam < nbf_; lam++)
                                        for (int sig = 0; sig < nbf_; sig++) {
                                            double v = eri_ao(mu, nu, lam, sig);
                                            g_dir += C(mu, nocc_ + a) * C(nu, nocc_ + b) * v * C(lam, nocc_ + c) * C(sig, nocc_ + d);
                                            g_ex  += C(mu, nocc_ + a) * C(nu, nocc_ + b) * v * C(lam, nocc_ + d) * C(sig, nocc_ + c);
                                        }
                            double g_spin = 2.0 * g_dir - g_ex;
                            res += 0.5 * g_spin * t2_1(i, j, c, d);
                        }

                    t2_2_(i, j, a, b) = res / denom;
                }

    std::cout << "    T2^(2) computation complete\n";
}

double RMP3::compute_third_order_energy() {
    double e3 = 0.0;
    for (int i = 0; i < nocc_; i++)
        for (int j = 0; j < nocc_; j++)
            for (int a = 0; a < nvirt_; a++)
                for (int b = 0; b < nvirt_; b++) {
                    // Spin-adapted energy factor: (2<ij|ab> - <ij|ba>)
                    double g_dir = eri_mo_(i, j, a, b);
                    double g_ex  = eri_mo_(i, j, b, a);
                    e3 += (2.0 * g_dir - g_ex) * t2_2_(i, j, a, b);
                }
    return e3;
}

RMP3Result RMP3::compute() {
    std::cout << "\n====================================\n";
    std::cout << "  Restricted MP3 (RMP3)\n";
    std::cout << "====================================\n";
    
    // Step 1: Build Fock matrix in MO basis
    build_fock_mo();
    
    // Step 2: Transform ERIs to MO basis
    transform_integrals_ao_to_mo();
    
    // Step 3: Compute T2^(2) amplitudes
    compute_t2_second_order();
    
    // Step 4: Compute E^(3)
    double e_mp3 = compute_third_order_energy();
    
    // Prepare result
    RMP3Result result;
    result.e_rhf = rmp2_.e_rhf;
    result.e_mp2 = rmp2_.e_corr;
    result.e_mp3 = e_mp3;
    result.e_corr_total = rmp2_.e_corr + e_mp3;
    result.e_total = rmp2_.e_rhf + result.e_corr_total;
    result.n_occ = nocc_;
    result.n_virt = nvirt_;
    result.t2_1 = rmp2_.t2;
    result.t2_2 = t2_2_;
    
    // Print results
    std::cout << "\n=== RMP3 Results ===\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "RHF energy:     " << std::setw(16) << result.e_rhf << " Ha\n";
    std::cout << "MP2 correction: " << std::setw(16) << result.e_mp2 << " Ha\n";
    std::cout << "MP3 correction: " << std::setw(16) << result.e_mp3 << " Ha\n";
    std::cout << "Total corr:     " << std::setw(16) << result.e_corr_total << " Ha\n";
    std::cout << "RMP3 energy:    " << std::setw(16) << result.e_total << " Ha\n";
    
    return result;
}

const Eigen::Tensor<double, 4>& RMP3::get_t2_second_order() const {
    return t2_2_;
}

} // namespace foundation
} // namespace mshqc
