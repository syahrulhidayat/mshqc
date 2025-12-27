/**
 * @file caspt2.h
 * @brief CASPT2 - Complete Active Space Second-Order Perturbation Theory
 * 
 * THEORY REFERENCES:
 * - K. Andersson et al., J. Phys. Chem. **94**, 5483 (1990)
 *   "Second-order perturbation theory with a CASSCF reference function"
 * - B. O. Roos, Adv. Chem. Phys. **69**, 399 (1987)
 *   "The Complete Active Space SCF method"
 * - A. Ghigo et al., Chem. Phys. Lett. **396**, 142 (2004)
 *   "A modified definition of the zeroth-order Hamiltonian (IPEA)"
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
#ifndef MSHQC_MCSCF_CASPT2_H
#define MSHQC_MCSCF_CASPT2_H

#include "casscf.h"
#include <memory>
#include <string>

namespace mshqc {
class Molecule;
class BasisSet;
class IntegralEngine;

namespace mcscf {

struct CASPT2Result1 {
    double e_casscf = 0.0;
    double e_pt2 = 0.0;
    double e_total = 0.0;
    double ipea_shift_used = 0.25;
    double imaginary_shift_used = 0.0;
    bool converged = false;
    std::string status_message;
};

class CASPT2 {
public:
    CASPT2(const Molecule& mol,
           const BasisSet& basis,
           std::shared_ptr<IntegralEngine> integrals,
           const CASResult& casscf_result);

    void set_ipea_shift(double s) { ipea_shift_ = s; }
    void set_imaginary_shift(double s) { imaginary_shift_ = s; }
    double get_ipea_shift() const { return ipea_shift_; }
    double get_imaginary_shift() const { return imaginary_shift_; }

    CASPT2Result1 compute();
    CASPT2Result1 compute_with_decomposition() { return compute(); }

private:
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    CASResult casscf_;
    double ipea_shift_ = 0.0;
    double imaginary_shift_ = 0.0;
};

} // namespace mcscf
} // namespace mshqc

#endif // MSHQC_MCSCF_CASPT2_H
