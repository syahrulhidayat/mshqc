/**
 * @file caspt2.cc
 * @brief CASPT2 implementation - Second-order perturbation on CASSCF
 * 
 * THEORY REFERENCES:
 * - K. Andersson et al., J. Phys. Chem. **94**, 5483 (1990)
 *   "Second-order perturbation theory with a CASSCF reference function"
 * - B. O. Roos, Adv. Chem. Phys. **69**, 399 (1987)
 * - A. Ghigo et al., Chem. Phys. Lett. **396**, 142 (2004) [IPEA shift]
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-16
 * @license MIT License
 * 
 * Copyright (c) 2025 Muhamad Sahrul Hidayat
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 * @note This is an original implementation derived from published theory.
 *       No code was copied from existing quantum chemistry software.
 */

#include "mshqc/mcscf/caspt2.h"
#include "mshqc/mcscf/external_space.h"
#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/ci/slater_condon.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace mshqc {
namespace mcscf {

// ============================================================================
// Constructor
// ============================================================================

CASPT2::CASPT2(const Molecule& mol,
               const BasisSet& basis,
               std::shared_ptr<IntegralEngine> integrals,
               const CASResult& casscf_result)
    : mol_(mol), basis_(basis), integrals_(std::move(integrals)),
      casscf_(casscf_result) {}

// ============================================================================
// CASPT2 Energy Calculation
// ============================================================================
// THEORY: Andersson et al. (1990), Eq. 15-18
// E_PT2 = Σ_K |⟨Φ_K|Ĥ|Ψ₀⟩|² / (E₀ - E_K)
// ============================================================================

CASPT2Result CASPT2::compute() {
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "CASPT2 Calculation\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Reference: Andersson et al. (1990)\n";
    std::cout << "E(CASSCF) = " << std::fixed << std::setprecision(10) 
              << casscf_.e_casscf << " Ha\n";
    std::cout << "IPEA shift = " << ipea_shift_ << " Ha\n";
    std::cout << "Imaginary shift = " << imaginary_shift_ << " Ha\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    CASPT2Result res;
    res.e_casscf = casscf_.e_casscf;
    res.ipea_shift_used = ipea_shift_;
    res.imaginary_shift_used = imaginary_shift_;
    
    // Verify CASResult has required data
    if (casscf_.determinants.empty()) {
        std::cerr << "ERROR: CASResult missing determinants (CASSCF bug)\n";
        res.e_pt2 = 0.0;
        res.e_total = res.e_casscf;
        res.converged = false;
        res.status_message = "Failed: No determinants in CASResult";
        return res;
    }
    
    // ========================================================================
    // PHASE 1: Generate External Space
    // ========================================================================
    // THEORY: Andersson et al. (1990), Eq. 10-12
    // Three types: Semi-Internal (SI), Semi-External (SE), Doubly-External (DE)
    
    std::cout << "Phase 1: Generating external space...\n";
    
    ExternalSpaceGenerator gen(
        casscf_.active_space,
        casscf_.C_mo.cols()  // Total number of orbitals
    );
    
    auto external_dets = gen.generate(casscf_.determinants);
    auto stats = gen.get_statistics(external_dets);
    
    std::cout << "  External space generated:\n";
    std::cout << "    SI (active→virtual):  " << stats.n_semi_internal << " determinants\n";
    std::cout << "    SE (core→active):     " << stats.n_semi_external << " determinants\n";
    std::cout << "    DE (core→virtual):    " << stats.n_doubly_external << " determinants\n";
    std::cout << "    Total external dets:  " << stats.n_total << "\n\n";
    
    if (external_dets.empty()) {
        std::cout << "WARNING: No external determinants generated\n";
        res.e_pt2 = 0.0;
        res.e_total = res.e_casscf;
        res.converged = true;
        res.status_message = "No external space (already at exact limit)";
        return res;
    }
    
    // ========================================================================
    // PHASE 2: Transform Integrals to MO Basis
    // ========================================================================
    // Need full MO basis integrals for matrix elements
    
    std::cout << "Phase 2: Transforming integrals to MO basis...\n";
    
    // One-electron integrals
    Eigen::MatrixXd h_ao = integrals_->compute_kinetic() + 
                           integrals_->compute_nuclear();
    Eigen::MatrixXd h_mo = casscf_.C_mo.transpose() * h_ao * casscf_.C_mo;
    
    // Two-electron integrals (full MO transform, brute-force for small systems)
    int nbf = basis_.n_basis_functions();
    int n_mo = casscf_.C_mo.cols();

    ci::CIIntegrals integrals_mo;
    integrals_mo.h_alpha = h_mo;
    integrals_mo.h_beta = h_mo;
    integrals_mo.e_nuc = mol_.nuclear_repulsion_energy();

    // Allocate MO ERI tensors
    integrals_mo.eri_aaaa = Eigen::Tensor<double, 4>(n_mo, n_mo, n_mo, n_mo);
    integrals_mo.eri_bbbb = Eigen::Tensor<double, 4>(n_mo, n_mo, n_mo, n_mo);
    integrals_mo.eri_aabb = Eigen::Tensor<double, 4>(n_mo, n_mo, n_mo, n_mo);
    integrals_mo.eri_aaaa.setZero();
    integrals_mo.eri_bbbb.setZero();
    integrals_mo.eri_aabb.setZero();

    std::cout << "  Transforming AO ERIs → MO (O(N^5) algorithm)...\n";

    // Get AO ERIs
    auto eri_ao = integrals_->compute_eri();

    // 4-index transformation: (μν|λσ) → (pq|rs)
    // REFERENCE: Helgaker, Jorgensen, Olsen "Molecular Electronic Structure Theory" (2000)
    //            Section 9.7.3, Eq. 9.7.30-33
    // Algorithm: Sequential transformation of each index (O(N^5) scaling)
    //   (pν|λσ) = Σ_μ C_μp (μν|λσ)     [N_mo × N_bf^4]
    //   (pq|λσ) = Σ_ν C_νq (pν|λσ)     [N_mo^2 × N_bf^3]
    //   (pq|rσ) = Σ_λ C_λr (pq|λσ)     [N_mo^3 × N_bf^2]
    //   (pq|rs) = Σ_σ C_σs (pq|rσ)     [N_mo^4 × N_bf]
    // Total: O(N_mo^4 × N_bf) = O(N^5) for N_mo ~ N_bf
    
    // Allocate intermediate tensors
    Eigen::Tensor<double, 4> I1(n_mo, nbf, nbf, nbf);    // (p ν λ σ)
    Eigen::Tensor<double, 4> I2(n_mo, n_mo, nbf, nbf);   // (p q λ σ)
    Eigen::Tensor<double, 4> I3(n_mo, n_mo, n_mo, nbf);  // (p q r σ)
    Eigen::Tensor<double, 4> eri_chem(n_mo, n_mo, n_mo, n_mo); // (p q r s) chemist
    
    I1.setZero();
    I2.setZero();
    I3.setZero();
    eri_chem.setZero();
    
    // Transform first index: μ → p
    for (int p = 0; p < n_mo; ++p) {
        for (int nu = 0; nu < nbf; ++nu) {
            for (int lam = 0; lam < nbf; ++lam) {
                for (int sig = 0; sig < nbf; ++sig) {
                    double val = 0.0;
                    for (int mu = 0; mu < nbf; ++mu) {
                        val += casscf_.C_mo(mu, p) * eri_ao(mu, nu, lam, sig);
                    }
                    I1(p, nu, lam, sig) = val;
                }
            }
        }
    }
    
    // Transform second index: ν → q
    for (int p = 0; p < n_mo; ++p) {
        for (int q = 0; q < n_mo; ++q) {
            for (int lam = 0; lam < nbf; ++lam) {
                for (int sig = 0; sig < nbf; ++sig) {
                    double val = 0.0;
                    for (int nu = 0; nu < nbf; ++nu) {
                        val += casscf_.C_mo(nu, q) * I1(p, nu, lam, sig);
                    }
                    I2(p, q, lam, sig) = val;
                }
            }
        }
    }
    
    // Transform third index: λ → r
    for (int p = 0; p < n_mo; ++p) {
        for (int q = 0; q < n_mo; ++q) {
            for (int r = 0; r < n_mo; ++r) {
                for (int sig = 0; sig < nbf; ++sig) {
                    double val = 0.0;
                    for (int lam = 0; lam < nbf; ++lam) {
                        val += casscf_.C_mo(lam, r) * I2(p, q, lam, sig);
                    }
                    I3(p, q, r, sig) = val;
                }
            }
        }
    }
    
    // Transform fourth index: σ → s
    for (int p = 0; p < n_mo; ++p) {
        for (int q = 0; q < n_mo; ++q) {
            for (int r = 0; r < n_mo; ++r) {
                for (int s = 0; s < n_mo; ++s) {
                    double val = 0.0;
                    for (int sig = 0; sig < nbf; ++sig) {
                        val += casscf_.C_mo(sig, s) * I3(p, q, r, sig);
                    }
                    eri_chem(p, q, r, s) = val;
                }
            }
        }
    }
    
    // Convert from chemist notation (pq|rs) to physicist notation <pq||rs>
    // REFERENCE: Szabo & Ostlund "Modern Quantum Chemistry" (1996), Appendix A
    // Physicist: <pq|rs> = (pr|qs) in chemist notation
    // Antisymmetrized: <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
    for (int p = 0; p < n_mo; ++p) {
        for (int q = 0; q < n_mo; ++q) {
            for (int r = 0; r < n_mo; ++r) {
                for (int s = 0; s < n_mo; ++s) {
                    // For same-spin (α-α or β-β): antisymmetrized
                    double antisym = eri_chem(p, r, q, s) - eri_chem(p, s, q, r);
                    integrals_mo.eri_aaaa(p, q, r, s) = antisym;
                    integrals_mo.eri_bbbb(p, q, r, s) = antisym;
                    
                    // For mixed-spin (α-β): direct Coulomb only
                    integrals_mo.eri_aabb(p, q, r, s) = eri_chem(p, q, r, s);
                }
            }
        }
    }

    std::cout << "  MO integral transform complete\n\n";
    
    // ========================================================================
    // PHASE 3: Compute PT2 Energy
    // ========================================================================
    // THEORY: E_PT2 = Σ_K |V_K0|² / (E₀ - E_K)
    // where V_K0 = ⟨Φ_K|Ĥ|Ψ₀⟩ = Σ_I c_I ⟨Φ_K|Ĥ|Φ_I⟩
    
    std::cout << "Phase 3: Computing PT2 energy correction...\n";
    
    double E_PT2 = 0.0;
    int n_cas = static_cast<int>(casscf_.determinants.size());
    
    // Loop over external determinants
    for (size_t K = 0; K < external_dets.size(); K++) {
        
        const auto& ext_det = external_dets[K];
        
        // Compute matrix element: V_K0 = Σ_I c_I * H_KI
        // THEORY: Andersson et al. (1990), Eq. 16, using Slater-Condon rules
        double V_K0 = 0.0;
        
        for (int I = 0; I < n_cas; I++) {
            double H_KI = ci::hamiltonian_element(
                ext_det,                    // ⟨Φ_K|
                casscf_.determinants[I],    // |Φ_I⟩
                integrals_mo                // MO integrals
            );
            V_K0 += casscf_.ci_coeffs[I] * H_KI;
        }
        
        // Compute denominator: E₀ - E_K
        // E_K = sum of orbital energies for occupied orbitals in Φ_K
        double E_K = 0.0;
        
        // For each occupied orbital in external determinant
        for (int p = 0; p < n_mo; p++) {
            if (ext_det.is_occupied(p, true)) {
                E_K += casscf_.orbital_energies(p);
            }
            if (ext_det.is_occupied(p, false)) {
                E_K += casscf_.orbital_energies(p);
            }
        }
        
        double denom = casscf_.e_casscf - E_K;
        
        // Add IPEA shift (prevents denominators from being too small)
        // THEORY: Ghigo et al. (2004), typically ε_IPEA ≈ 0.25 Ha
        if (ipea_shift_ > 0.0) {
            denom -= ipea_shift_;
        }
        
        // Add imaginary shift (for intruder states)
        if (imaginary_shift_ > 0.0) {
            denom -= imaginary_shift_;
        }
        
        // Accumulate PT2 contribution
        if (std::abs(denom) > 1e-10) {
            E_PT2 += (V_K0 * V_K0) / denom;
        }
    }
    
    std::cout << "  PT2 correction computed\n";
    std::cout << "  E(PT2) = " << std::scientific << std::setprecision(6) 
              << E_PT2 << " Ha\n\n";
    
    // ========================================================================
    // Results
    // ========================================================================
    
    res.e_pt2 = E_PT2;
    res.e_total = res.e_casscf + res.e_pt2;
    res.converged = true;
    res.status_message = "CASPT2 completed successfully";
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "CASPT2 Results\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "E(CASSCF)     = " << std::fixed << std::setprecision(10) 
              << res.e_casscf << " Ha\n";
    std::cout << "E(PT2)        = " << std::setprecision(10) 
              << res.e_pt2 << " Ha\n";
    std::cout << "E(CASPT2)     = " << std::setprecision(10) 
              << res.e_total << " Ha\n";
    std::cout << "\nExternal space: " << external_dets.size() << " determinants\n";
    std::cout << "Status: " << res.status_message << "\n";
    std::cout << std::string(70, '=') << "\n";
    
    return res;
}

} // namespace mcscf
} // namespace mshqc
