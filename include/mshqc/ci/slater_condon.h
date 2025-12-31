#ifndef MSHQC_CI_SLATER_CONDON_H
#define MSHQC_CI_SLATER_CONDON_H

#include "mshqc/ci/determinant.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

/**
 * @file slater_condon.h
 * @brief Slater-Condon rules for CI matrix elements
 * 
 * REFERENCES:
 * - Slater (1929), Phys. Rev. 34, 1293 [original rules]
 * - Condon (1930), Phys. Rev. 36, 1121 [extension]
 * - Szabo & Ostlund (1996), Appendix A [modern formulation]
 * - Shavitt & Bartlett (2009), Ch. 3 [comprehensive treatment]
 * 
 * SLATER-CONDON RULES:
 * Given two determinants |Φ⟩ and |Φ'⟩, compute ⟨Φ|H|Φ'⟩
 * 
 * Case 1: Same determinant (Φ = Φ')
 *   ⟨Φ|H|Φ⟩ = Σ_i h_ii + (1/2) Σ_ij [<ij||ij>]
 * 
 * Case 2: Single excitation (Φ' = Φ_i^a)
 *   ⟨Φ|H|Φ_i^a⟩ = h_ia + Σ_j <ij||aj>
 * 
 * Case 3: Double excitation (Φ' = Φ_ij^ab)
 *   ⟨Φ|H|Φ_ij^ab⟩ = <ij||ab>
 * 
 * Case 4: Higher excitation (differ by > 2)
 *   ⟨Φ|H|Φ'⟩ = 0
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
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

namespace mshqc {
namespace ci {

/**
 * Integral container for Slater-Condon rules
 * 
 * Stores one- and two-electron integrals in MO basis
 */
/*struct CIIntegrals {
    Eigen::MatrixXd h_alpha;  // One-electron α (i,a) - can be bare H or Fock
    Eigen::MatrixXd h_beta;   // One-electron β - can be bare H or Fock
    
    // Two-electron integrals (antisymmetrized physicist notation)
    // <ij||ab> = <ij|ab> - <ij|ba>
    Eigen::Tensor<double, 4> eri_aaaa;  // α-α
    Eigen::Tensor<double, 4> eri_bbbb;  // β-β
    Eigen::Tensor<double, 4> eri_aabb;  // α-β (no antisym)
    
    double e_nuc;  // Nuclear repulsion
    
    // Flag: if true, h_alpha/h_beta are Fock matrices (include mean-field effects)
    // If false (default), h_alpha/h_beta are bare Hamiltonian (T + V only)
    bool use_fock = false;
};*/

/**
 * Integral container for Slater-Condon rules
 * [FIX] UNCOMMENTED THIS STRUCT - IT IS REQUIRED
 */
struct CIIntegrals {
    Eigen::MatrixXd h_alpha;  // One-electron α
    Eigen::MatrixXd h_beta;   // One-electron β
    
    // Two-electron integrals <ij||ab>
    Eigen::Tensor<double, 4> eri_aaaa;  // α-α
    Eigen::Tensor<double, 4> eri_bbbb;  // β-β
    Eigen::Tensor<double, 4> eri_aabb;  // α-β
    
    double e_nuc = 0.0;  // Nuclear repulsion
    bool use_fock = false;
};

// ... (Fungsi-fungsi lama Anda tetap ada) ...

double hamiltonian_element(const Determinant& bra, 
                           const Determinant& ket,
                           const CIIntegrals& ints);

double diagonal_element(const Determinant& det, const CIIntegrals& ints);

double single_excitation_element(const Determinant& bra,
                                  int i, int a, bool spin_alpha,
                                  const CIIntegrals& ints);

double double_excitation_element(const Determinant& bra,
                                  int i, int j, int a, int b,
                                  bool spin1, bool spin2,
                                  const CIIntegrals& ints);

// [FIX] Ini fungsi yang dicari oleh ci_utils.cc
double evaluate_matrix_element(const Determinant& det_i, 
                               const Determinant& det_j, 
                               const CIIntegrals& integrals);

Eigen::MatrixXd build_hamiltonian(const std::vector<Determinant>& dets,
                                  const CIIntegrals& ints);

Eigen::VectorXd hamiltonian_diagonal(const std::vector<Determinant>& dets,
                                     const CIIntegrals& ints);

Eigen::VectorXd sigma_vector(const std::vector<Determinant>& dets,
                              const Eigen::VectorXd& c,
                              const CIIntegrals& ints);

} // namespace ci
} // namespace mshqc
#endif // MSHQC_CI_SLATER_CONDON_H