/**
 * @file mrmp2.cc
 * @brief MRMP2 implementation - Multi-reference Møller-Plesset order 2
 * 
 * THEORY REFERENCES:
 * - K. Hirao, Chem. Phys. Lett. **190**, 374 (1992)
 *   "Multireference Møller-Plesset method"
 * - K. Hirao, Chem. Phys. Lett. **196**, 397 (1992)
 *   "State-specific multireference Møller-Plesset perturbation treatment"
 * 
 * KEY DIFFERENCE FROM CASPT2:
 * - Denominator: D = Σ_occ ε_i - Σ_virt ε_a (Møller-Plesset style)
 * - CASPT2 uses: D = E_CASSCF - E_K
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

#include "mshqc/mcscf/mrmp2.h"
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

MRMP2::MRMP2(const Molecule& mol,
             const BasisSet& basis,
             std::shared_ptr<IntegralEngine> integrals,
             const CASResult& casscf_result)
    : mol_(mol), basis_(basis), integrals_(std::move(integrals)),
      casscf_(casscf_result) {}

// ============================================================================
// MRMP2 Energy Calculation
// ============================================================================
// THEORY: Hirao (1992), Eq. 10-15
// E_MRMP2 = Σ_K |V_K0|² / D_K
// where D_K = Σ_occ ε_i - Σ_virt ε_a (Møller-Plesset denominator)
// ============================================================================

MRMP2Result MRMP2::compute() {
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "MRMP2 Calculation\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Reference: Hirao (1992)\n";
    std::cout << "E(CASSCF) = " << std::fixed << std::setprecision(10) 
              << casscf_.e_casscf << " Ha\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    MRMP2Result res;
    res.e_casscf = casscf_.e_casscf;
    
    // Verify CASResult has required data
    if (casscf_.determinants.empty()) {
        std::cerr << "ERROR: CASResult missing determinants\n";
        res.e_mrmp2_correction = 0.0;
        res.e_total = res.e_casscf;
        res.converged = false;
        res.status_message = "Failed: No determinants in CASResult";
        return res;
    }
    
    // ========================================================================
    // PHASE 1: Generate External Space
    // ========================================================================
    
    std::cout << "Phase 1: Generating external space...\n";
    
    ExternalSpaceGenerator gen(
        casscf_.active_space,
        casscf_.C_mo.cols()
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
        res.e_mrmp2_correction = 0.0;
        res.e_total = res.e_casscf;
        res.converged = true;
        res.status_message = "No external space";
        return res;
    }
    
    // ========================================================================
    // PHASE 2: Transform Integrals to MO Basis
    // ========================================================================
    
    std::cout << "Phase 2: Transforming integrals to MO basis...\n";
    
    // One-electron integrals
    Eigen::MatrixXd h_ao = integrals_->compute_kinetic() + 
                           integrals_->compute_nuclear();
    Eigen::MatrixXd h_mo = casscf_.C_mo.transpose() * h_ao * casscf_.C_mo;
    
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

    // 4-index transformation
    // REFERENCE: Helgaker, Jorgensen, Olsen (2000), Section 9.7.3
    Eigen::Tensor<double, 4> I1(n_mo, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> I2(n_mo, n_mo, nbf, nbf);
    Eigen::Tensor<double, 4> I3(n_mo, n_mo, n_mo, nbf);
    Eigen::Tensor<double, 4> eri_chem(n_mo, n_mo, n_mo, n_mo);
    
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
    
    // Convert chemist → physicist antisymmetrized
    for (int p = 0; p < n_mo; ++p) {
        for (int q = 0; q < n_mo; ++q) {
            for (int r = 0; r < n_mo; ++r) {
                for (int s = 0; s < n_mo; ++s) {
                    double antisym = eri_chem(p, r, q, s) - eri_chem(p, s, q, r);
                    integrals_mo.eri_aaaa(p, q, r, s) = antisym;
                    integrals_mo.eri_bbbb(p, q, r, s) = antisym;
                    integrals_mo.eri_aabb(p, q, r, s) = eri_chem(p, q, r, s);
                }
            }
        }
    }

    std::cout << "  MO integral transform complete\n\n";
    
    // ========================================================================
    // PHASE 3: Compute MRMP2 Energy
    // ========================================================================
    // KEY DIFFERENCE: Use Møller-Plesset denominator instead of CASPT2
    
    std::cout << "Phase 3: Computing MRMP2 energy correction...\n";
    
    double E_MRMP2 = 0.0;
    int n_cas = static_cast<int>(casscf_.determinants.size());
    
    // Determine reference determinant occupations
    const auto& ref_det = casscf_.determinants[0];
    
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
        
        // ====================================================================
        // MØLLER-PLESSET DENOMINATOR (KEY DIFFERENCE FROM CASPT2!)
        // ====================================================================
        // THEORY: Hirao (1992), Eq. 12
        // D_K = Σ_i ε_i - Σ_a ε_a
        // where i runs over occupied in ref, a over virtual in excitation
        
        double denom = 0.0;
        
        // Add occupied orbital energies from reference
        for (int p = 0; p < n_mo; p++) {
            if (ref_det.is_occupied(p, true)) {  // alpha in reference
                denom += casscf_.orbital_energies(p);
            }
            if (ref_det.is_occupied(p, false)) {  // beta in reference
                denom += casscf_.orbital_energies(p);
            }
        }
        
        // Subtract occupied orbital energies from external determinant
        for (int p = 0; p < n_mo; p++) {
            if (ext_det.is_occupied(p, true)) {
                denom -= casscf_.orbital_energies(p);
            }
            if (ext_det.is_occupied(p, false)) {
                denom -= casscf_.orbital_energies(p);
            }
        }
        
        // Accumulate MRMP2 contribution
        if (std::abs(denom) > 1e-10) {
            E_MRMP2 += (V_K0 * V_K0) / denom;
        }
    }
    
    std::cout << "  MRMP2 correction computed\n";
    std::cout << "  E(MRMP2) = " << std::scientific << std::setprecision(6) 
              << E_MRMP2 << " Ha\n\n";
    
    // ========================================================================
    // Results
    // ========================================================================
    
    res.e_mrmp2_correction = E_MRMP2;
    res.e_total = res.e_casscf + res.e_mrmp2_correction;
    res.converged = true;
    res.status_message = "MRMP2 completed successfully";
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "MRMP2 Results\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "E(CASSCF)     = " << std::fixed << std::setprecision(10) 
              << res.e_casscf << " Ha\n";
    std::cout << "E(MRMP2)      = " << std::setprecision(10) 
              << res.e_mrmp2_correction << " Ha\n";
    std::cout << "E(total)      = " << std::setprecision(10) 
              << res.e_total << " Ha\n";
    std::cout << "\nExternal space: " << external_dets.size() << " determinants\n";
    std::cout << "Status: " << res.status_message << "\n";
    std::cout << std::string(70, '=') << "\n";
    
    return res;
}

} // namespace mcscf
} // namespace mshqc
