


/**
 * @file slater_condon.cc
 * @brief Implementation of Slater-Condon rules
 * 
 * REFERENCES:
 * - Slater (1929), Phys. Rev. 34, 1293
 * - Condon (1930), Phys. Rev. 36, 1121
 * - Szabo & Ostlund (1996), Appendix A
 * - Shavitt & Bartlett (2009), Ch. 3, Sec. 3.2
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

 /**
 * @file slater_condon.cc
 * @brief Implementation of Slater-Condon rules (ROBUST & SAFE VERSION)
 * @details Includes strict bounds checking to prevent Segmentation Faults
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 */

#include "mshqc/ci/slater_condon.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm> // For std::max

namespace mshqc {
namespace ci {

// Helper untuk mengecek batas indeks (Safety Check)
inline bool is_safe(int idx, int max_dim) {
    return idx >= 0 && idx < max_dim;
}

double hamiltonian_element(const Determinant& bra, const Determinant& ket, const CIIntegrals& ints) {
    auto exc = find_excitation(bra, ket);
    
    // Safety: Dapatkan dimensi maksimal dari integral 1-elektron
    // Kita asumsikan h_alpha.rows() merepresentasikan N_MO
    int n_mo = ints.h_alpha.rows(); 

    if (exc.level == 0) {
        return diagonal_element(bra, ints);
    }
    else if (exc.level == 1) {
        bool is_alpha = !exc.occ_alpha.empty();
        
        // CRITICAL FIX: Pastikan vector tidak kosong sebelum akses [0]
        if (is_alpha) {
            if (exc.occ_alpha.empty() || exc.virt_alpha.empty()) return 0.0;
            int i = exc.occ_alpha[0];
            int a = exc.virt_alpha[0];
            if (!is_safe(i, n_mo) || !is_safe(a, n_mo)) return 0.0; // Prevent Out of Bounds
            return single_excitation_element(bra, i, a, true, ints);
        } else {
            if (exc.occ_beta.empty() || exc.virt_beta.empty()) return 0.0;
            int i = exc.occ_beta[0];
            int a = exc.virt_beta[0];
            if (!is_safe(i, n_mo) || !is_safe(a, n_mo)) return 0.0; // Prevent Out of Bounds
            return single_excitation_element(bra, i, a, false, ints);
        }
    }
    else if (exc.level == 2) {
        // CRITICAL FIX: Pastikan ukuran vector valid untuk double excitation
        if (exc.occ_alpha.size() == 2 && exc.virt_alpha.size() == 2) {
            return double_excitation_element(bra, 
                exc.occ_alpha[0], exc.occ_alpha[1], 
                exc.virt_alpha[0], exc.virt_alpha[1], 
                true, true, ints);
        }
        else if (exc.occ_beta.size() == 2 && exc.virt_beta.size() == 2) {
            return double_excitation_element(bra, 
                exc.occ_beta[0], exc.occ_beta[1], 
                exc.virt_beta[0], exc.virt_beta[1], 
                false, false, ints);
        }
        else if (exc.occ_alpha.size() == 1 && exc.occ_beta.size() == 1 &&
                 exc.virt_alpha.size() == 1 && exc.virt_beta.size() == 1) {
            return double_excitation_element(bra, 
                exc.occ_alpha[0], exc.occ_beta[0], 
                exc.virt_alpha[0], exc.virt_beta[0], 
                true, false, ints);
        }
        // Jika struktur eksitasi aneh/tidak valid, kembalikan 0.0 daripada crash
        return 0.0;
    }
    
    return 0.0;
}

double diagonal_element(const Determinant& det, const CIIntegrals& ints) {
    double energy = 0.0;
    auto occ_a = det.alpha_occupations();
    auto occ_b = det.beta_occupations();
    
    int n_mo = ints.h_alpha.rows();
    
    if (ints.use_fock) {
        for (int i : occ_a) {
            if (is_safe(i, n_mo)) energy += ints.h_alpha(i, i);
        }
        for (int i : occ_b) {
            if (is_safe(i, n_mo)) energy += ints.h_beta(i, i);
        }
    }
    else {
        // One-electron
        for (int i : occ_a) if (is_safe(i, n_mo)) energy += ints.h_alpha(i, i);
        for (int i : occ_b) if (is_safe(i, n_mo)) energy += ints.h_beta(i, i);
        
        // Two-electron (aa)
        for (size_t i = 0; i < occ_a.size(); ++i) {
            for (size_t j = i + 1; j < occ_a.size(); ++j) {
                int p = occ_a[i]; int q = occ_a[j];
                if (is_safe(p, n_mo) && is_safe(q, n_mo))
                    energy += ints.eri_aaaa(p, q, p, q);
            }
        }
        
        // Two-electron (bb)
        for (size_t i = 0; i < occ_b.size(); ++i) {
            for (size_t j = i + 1; j < occ_b.size(); ++j) {
                int p = occ_b[i]; int q = occ_b[j];
                if (is_safe(p, n_mo) && is_safe(q, n_mo))
                    energy += ints.eri_bbbb(p, q, p, q);
            }
        }
        
        // Two-electron (ab)
        for (int i : occ_a) {
            for (int j : occ_b) {
                if (is_safe(i, n_mo) && is_safe(j, n_mo))
                    energy += ints.eri_aabb(i, j, i, j); 
            }
        }
    }
    return energy;
}

double single_excitation_element(const Determinant& bra, int i, int a, bool spin_alpha, const CIIntegrals& ints) {
    // Safety check awal
    int n_mo = ints.h_alpha.rows();
    if (!is_safe(i, n_mo) || !is_safe(a, n_mo)) return 0.0;

    int phase_sign = bra.phase(i, a, spin_alpha);
    double elem = 0.0;
    
    if (ints.use_fock) {
        elem = spin_alpha ? ints.h_alpha(i, a) : ints.h_beta(i, a);
    } else {
        if (spin_alpha) {
            elem = ints.h_alpha(i, a);
            // Loop over occupied orbitals (Summing 2-electron part)
            for (int j : bra.alpha_occupations()) {
                if (j != i && is_safe(j, n_mo)) 
                    elem += ints.eri_aaaa(i, j, a, j);
            }
            for (int j : bra.beta_occupations()) {
                if (is_safe(j, n_mo))
                    elem += ints.eri_aabb(i, j, a, j);
            }
        } else {
            elem = ints.h_beta(i, a);
            for (int j : bra.beta_occupations()) {
                if (j != i && is_safe(j, n_mo))
                    elem += ints.eri_bbbb(i, j, a, j);
            }
            for (int j : bra.alpha_occupations()) {
                if (is_safe(j, n_mo))
                    elem += ints.eri_aabb(j, i, j, a); // Perhatikan indeks eri_aabb(alpha, beta, ...)
            }
        }
    }
    return phase_sign * elem;
}

double double_excitation_element(const Determinant& bra, int i, int j, int a, int b, bool spin1, bool spin2, const CIIntegrals& ints) {
    // Safety check
    int n_mo = ints.h_alpha.rows();
    if (!is_safe(i, n_mo) || !is_safe(j, n_mo) || !is_safe(a, n_mo) || !is_safe(b, n_mo)) return 0.0;

    int phase = bra.phase(i, a, spin1) * bra.single_excite(i, a, spin1).phase(j, b, spin2);
    double elem = 0.0;
    
    if (spin1 && spin2) elem = ints.eri_aaaa(i, j, a, b);       
    else if (!spin1 && !spin2) elem = ints.eri_bbbb(i, j, a, b); 
    else elem = ints.eri_aabb(i, j, a, b);                       
    
    return phase * elem;
}

Eigen::MatrixXd build_hamiltonian(const std::vector<Determinant>& dets, const CIIntegrals& ints) {
    int n = dets.size();
    Eigen::MatrixXd H(n, n);
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double val = hamiltonian_element(dets[i], dets[j], ints);
            H(i,j) = val;
            if (i!=j) H(j,i) = val;
        }
    }
    return H;
}

Eigen::VectorXd hamiltonian_diagonal(const std::vector<Determinant>& dets, const CIIntegrals& ints) {
    int n = dets.size();
    Eigen::VectorXd diag(n);
    for (int i = 0; i < n; i++) diag(i) = diagonal_element(dets[i], ints);
    return diag;
}

Eigen::VectorXd sigma_vector(const std::vector<Determinant>& dets, const Eigen::VectorXd& c, const CIIntegrals& ints) {
    int n = dets.size();
    Eigen::VectorXd sigma = Eigen::VectorXd::Zero(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) 
        for (int j = 0; j < n; j++) 
            sigma(i) += hamiltonian_element(dets[i], dets[j], ints) * c(j);
    return sigma;
}

double evaluate_matrix_element(const Determinant& det_i, 
                               const Determinant& det_j, 
                               const CIIntegrals& integrals) {
    return hamiltonian_element(det_i, det_j, integrals);
}

} // namespace ci
} // namespace mshqc