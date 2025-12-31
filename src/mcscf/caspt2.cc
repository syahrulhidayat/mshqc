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

/**
 * @file caspt2.cc
 * @brief Optimized CASPT2 Implementation using Fast Integral Transforms
 * * FEATURES:
 * - Uses O(N^5) integral transformation via ERITransformer (not O(N^8) brute force)
 * - Implements IPEA and Imaginary Shift
 * - Correct handling of external space determinants
 * * THEORY REFERENCES:
 * - K. Andersson et al., J. Phys. Chem. 94, 5483 (1990)
 * - A. Ghigo et al., Chem. Phys. Lett. 396, 142 (2004)
 */

/**
 * @file caspt2.cc
 * @brief Optimized CASPT2 Implementation - DEBUG MODE (Single Threaded)
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-16
 */

/**
 * @file caspt2.cc
 * @brief CASPT2 Implementation - DEEP DEBUG MODE
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-16
 */

#include "mshqc/mcscf/caspt2.h"
#include "mshqc/mcscf/external_space.h"
#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/ci/slater_condon.h"
#include "mshqc/integrals/eri_transformer.h" 
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

namespace mshqc {
namespace mcscf {

CASPT2::CASPT2(const Molecule& mol,
               const BasisSet& basis,
               std::shared_ptr<IntegralEngine> integrals,
               const CASResult& casscf_result)
    : mol_(mol), basis_(basis), integrals_(std::move(integrals)),
      casscf_(casscf_result) {}

CASPT2Result1 CASPT2::compute() {
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "CASPT2 Calculation (DEEP DEBUG MODE)\n";
    std::cout << std::string(70, '=') << "\n";
    
    CASPT2Result1 res;
    res.e_casscf = casscf_.e_casscf;
    
    int nbf = basis_.n_basis_functions();
    int n_mo = casscf_.C_mo.cols();

    std::cout << "DEBUG INFO:\n";
    std::cout << "  N Basis (AO): " << nbf << "\n";
    std::cout << "  N MO    (MO): " << n_mo << " (Valid Indices: 0 to " << n_mo - 1 << ")\n";

    if (casscf_.determinants.empty()) {
        std::cerr << "ERROR: CASResult missing determinants.\n";
        return res;
    }
    
    // ========================================================================
    // PHASE 1: Generate External Space
    // ========================================================================
    std::cout << "Phase 1: Generating external space...\n";
    
    ExternalSpaceGenerator gen(
        casscf_.active_space,
        n_mo  // Passing total MOs to generator
    );
    
    auto external_dets = gen.generate(casscf_.determinants);
    std::cout << "  Total external dets: " << external_dets.size() << "\n";

    if (external_dets.empty()) return res;

    // ========================================================================
    // PHASE 1.5: VALIDATE DETERMINANTS (CRITICAL CHECK)
    // ========================================================================
    std::cout << "Phase 1.5: Validating Determinant Indices...\n";
    bool det_error = false;
    for (size_t i = 0; i < external_dets.size(); ++i) {
        auto occ_a = external_dets[i].alpha_occupations();
        auto occ_b = external_dets[i].beta_occupations();
        
        for (int idx : occ_a) {
            if (idx < 0 || idx >= n_mo) {
                std::cerr << "CRITICAL ERROR: Det " << i << " has invalid Alpha orbital index: " << idx << "\n";
                det_error = true;
            }
        }
        for (int idx : occ_b) {
            if (idx < 0 || idx >= n_mo) {
                std::cerr << "CRITICAL ERROR: Det " << i << " has invalid Beta orbital index: " << idx << "\n";
                det_error = true;
            }
        }
        if (det_error) break; 
    }

    if (det_error) {
        std::cerr << "ABORTING: External determinants contain invalid indices (Out of Bounds).\n";
        std::cerr << "This indicates a bug in ExternalSpaceGenerator.\n";
        exit(1); 
    }
    std::cout << "  All determinant indices are valid (within 0.." << n_mo-1 << ").\n";

    // ========================================================================
    // PHASE 2: Transform Integrals
    // ========================================================================
    std::cout << "Phase 2: Transforming integrals...\n";
    
    Eigen::MatrixXd h_ao = integrals_->compute_kinetic() + integrals_->compute_nuclear();
    Eigen::MatrixXd h_mo = casscf_.C_mo.transpose() * h_ao * casscf_.C_mo;
    
    ci::CIIntegrals integrals_mo;
    integrals_mo.h_alpha = h_mo;
    integrals_mo.h_beta = h_mo;
    integrals_mo.e_nuc = mol_.nuclear_repulsion_energy();

    // Allocate tensors
    std::cout << "  Allocating Tensors (" << n_mo << "^4)...\n";
    integrals_mo.eri_aaaa = Eigen::Tensor<double, 4>(n_mo, n_mo, n_mo, n_mo);
    integrals_mo.eri_bbbb = Eigen::Tensor<double, 4>(n_mo, n_mo, n_mo, n_mo);
    integrals_mo.eri_aabb = Eigen::Tensor<double, 4>(n_mo, n_mo, n_mo, n_mo);

    auto eri_ao = integrals_->compute_eri();
    auto eri_chem = integrals::ERITransformer::transform_vvvv(eri_ao, casscf_.C_mo, nbf, n_mo);
    
    // Validate Dimensions
    if (eri_chem.dimension(0) != n_mo) {
         std::cerr << "CRITICAL ERROR: eri_chem dim 0 mismatch! Expected " << n_mo << ", got " << eri_chem.dimension(0) << "\n";
         exit(1);
    }
    
    // Convert to Physicist notation
    std::cout << "  Converting notation...\n";
    for (int p = 0; p < n_mo; ++p) {
        for (int q = 0; q < n_mo; ++q) {
            for (int r = 0; r < n_mo; ++r) {
                for (int s = 0; s < n_mo; ++s) {
                    double val_dir = eri_chem(p, r, q, s);
                    double val_exc = eri_chem(p, s, q, r);
                    integrals_mo.eri_aaaa(p, q, r, s) = val_dir - val_exc;
                    integrals_mo.eri_bbbb(p, q, r, s) = val_dir - val_exc;
                    integrals_mo.eri_aabb(p, q, r, s) = eri_chem(p, q, r, s);
                }
            }
        }
    }
    std::cout << "  Transformation complete.\n";

    // ========================================================================
    // PHASE 2.5: Fock Matrix Repair
    // ========================================================================
    if (casscf_.orbital_energies.size() != n_mo) {
        std::cout << "  [AUTO-REPAIR] Recomputing Fock Diagonal...\n";
        casscf_.orbital_energies.resize(n_mo);
        casscf_.orbital_energies.setZero();

        int n_core = casscf_.active_space.n_inactive(); 
        int n_act  = casscf_.active_space.n_active();

        Eigen::VectorXd occ_active = Eigen::VectorXd::Zero(n_act);
        
        for (size_t I = 0; I < casscf_.determinants.size(); ++I) {
            double weight = casscf_.ci_coeffs[I] * casscf_.ci_coeffs[I];
            if (weight < 1e-12) continue;
            auto occ_a = casscf_.determinants[I].alpha_occupations();
            auto occ_b = casscf_.determinants[I].beta_occupations();
            for(int p : occ_a) if(p >= n_core && p < n_core + n_act) occ_active(p - n_core) += weight;
            for(int p : occ_b) if(p >= n_core && p < n_core + n_act) occ_active(p - n_core) += weight;
        }

        for (int p = 0; p < n_mo; ++p) {
            double f_pp = h_mo(p, p); 
            for (int i = 0; i < n_core; ++i) {
                double J = integrals_mo.eri_aabb(i, i, p, p);
                double K = integrals_mo.eri_aabb(i, p, p, i);
                f_pp += 2.0 * J - K;
            }
            for (int t = 0; t < n_act; ++t) {
                int mo_t = n_core + t;
                double occ_t = occ_active(t);
                if (std::abs(occ_t) > 1e-9) {
                     double J_act = integrals_mo.eri_aabb(mo_t, mo_t, p, p);
                     double K_act = integrals_mo.eri_aabb(mo_t, p, p, mo_t);
                     f_pp += occ_t * (J_act - 0.5 * K_act);
                }
            }
            casscf_.orbital_energies(p) = f_pp;
        }
    }

    // ========================================================================
    // PHASE 3: Compute PT2 Energy
    // ========================================================================
    
    std::cout << "Phase 3: Computing PT2 energy correction...\n";
    
    double E_PT2 = 0.0;
    int n_cas = static_cast<int>(casscf_.determinants.size());
    long size_orb = casscf_.orbital_energies.size();
    
    std::cout << "  Params: n_cas=" << n_cas << ", n_ext=" << external_dets.size() << "\n";

    for (size_t K = 0; K < external_dets.size(); K++) {
        
        // EXTREME VERBOSITY FOR K=0
        if (K == 0) std::cout << "  [DEBUG] Loop K=0 started.\n" << std::flush;

        const auto& ext_det = external_dets[K];
        double V_K0 = 0.0;
        
        for (int I = 0; I < n_cas; I++) {
            
            // Filter coefficients
            if (std::abs(casscf_.ci_coeffs[I]) < 1e-12) continue;

            // Excitation level check
            auto diff = ext_det.excitation_level(casscf_.determinants[I]);
            if (diff.first + diff.second > 2) continue;

            // TRACE: Call Hamiltonian
            if (K == 0 && I == 0) {
                 std::cout << "  [DEBUG] Calling hamiltonian_element(Ext, Ref[0])...\n" << std::flush;
                 std::cout << "          Integrals dimensions check: " << integrals_mo.eri_aaaa.dimension(0) << "\n" << std::flush;
            }

            double H_KI = ci::hamiltonian_element(
                ext_det,                    
                casscf_.determinants[I],    
                integrals_mo                
            );
            
            if (K == 0 && I == 0) std::cout << "  [DEBUG] hamiltonian_element success. Val=" << H_KI << "\n" << std::flush;

            V_K0 += casscf_.ci_coeffs[I] * H_KI;
        }
        
        if (std::abs(V_K0) < 1e-14) continue;

        double E_K = 0.0;
        auto occ_a = ext_det.alpha_occupations();
        auto occ_b = ext_det.beta_occupations();
        
        for (int p : occ_a) E_K += casscf_.orbital_energies(p);
        for (int p : occ_b) E_K += casscf_.orbital_energies(p);
        
        double denom = casscf_.e_casscf - E_K;
        if (ipea_shift_ > 0.0) denom -= ipea_shift_;
        if (imaginary_shift_ > 0.0) denom = (denom * denom) / (denom + (imaginary_shift_ * imaginary_shift_));
        
        if (std::abs(denom) > 1e-10) {
            E_PT2 += (V_K0 * V_K0) / denom;
        }
    }
    
    std::cout << "  PT2 correction computed successfully.\n";
    std::cout << "  E(PT2) = " << std::scientific << std::setprecision(6) << E_PT2 << " Ha\n\n";
    
    res.e_pt2 = E_PT2;
    res.e_total = res.e_casscf + res.e_pt2;
    res.converged = true;
    res.status_message = "CASPT2 completed successfully";
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "CASPT2 Results\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "E(CASSCF)     = " << std::fixed << std::setprecision(10) << res.e_casscf << " Ha\n";
    std::cout << "E(PT2)        = " << std::setprecision(10) << res.e_pt2 << " Ha\n";
    std::cout << "E(CASPT2)     = " << std::setprecision(10) << res.e_total << " Ha\n";
    std::cout << std::string(70, '=') << "\n";
    
    return res;
}

} // namespace mcscf
} // namespace mshqc