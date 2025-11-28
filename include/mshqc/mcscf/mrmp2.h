/**
 * @file mrmp2.h
 * @brief MRMP2 - Multi-Reference Møller-Plesset Perturbation Theory Order 2
 * 
 * THEORY REFERENCES:
 * - K. Hirao, Chem. Phys. Lett. **190**, 374 (1992)
 *   "Multireference Møller-Plesset method"
 * - K. Hirao, Chem. Phys. Lett. **196**, 397 (1992)
 *   "State-specific multireference Møller-Plesset perturbation treatment"
 * - H. Nakano, J. Chem. Phys. **99**, 7983 (1993)
 *   "Quasidegenerate perturbation theory with multiconfigurational SCF"
 * 
 * COMPARISON WITH CASPT2:
 * - Same external space (SI/SE/DE excitations)
 * - Different zeroth-order Hamiltonian (Møller-Plesset vs generalized Fock)
 * - Different denominator formula (orbital energy differences)
 * - No IPEA shift in original formulation
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
#ifndef MSHQC_MCSCF_MRMP2_H
#define MSHQC_MCSCF_MRMP2_H

#include "casscf.h"
#include <memory>
#include <string>

namespace mshqc {
class Molecule;
class BasisSet;
class IntegralEngine;

namespace mcscf {

struct MRMP2Result {
    double e_casscf = 0.0;
    double e_mrmp2_correction = 0.0;
    double e_total = 0.0;
    bool converged = false;
    std::string status_message;
};

class MRMP2 {
public:
    MRMP2(const Molecule& mol,
          const BasisSet& basis,
          std::shared_ptr<IntegralEngine> integrals,
          const CASResult& casscf_result);

    MRMP2Result compute();

private:
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    CASResult casscf_;
};

} // namespace mcscf
} // namespace mshqc

#endif // MSHQC_MCSCF_MRMP2_H
