/**
 * @file libcint_wrapper.h
 * @brief Libcint wrapper for 3-center electron repulsion integrals
 * 
 * Provides native 3-center ERI computation (μν|P) using Libcint library.
 * This enables accurate density-fitting approximations for CASPT2.
 * 
 * THEORY REFERENCES:
 *   - Q. Sun, J. Comp. Chem. 36, 1664 (2015)
 *     "Libcint: An efficient general integral library for Gaussian basis functions"
 *   - F. Weigend & M. Häser, Theor. Chem. Acc. 97, 331 (1997)
 *     "RI-MP2: first derivatives and global consistency"
 *     [Equation (2): Definition of 3-center integrals]
 * 
 * ALGORITHM:
 *   Uses Libcint's int3c2e_sph() function for native (μν|P) computation.
 *   No approximation needed - these are exact 3-center integrals.
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
 * @note This is an original wrapper implementation.
 *       Libcint library itself is BSD-licensed (compatible with MIT).
 *       No Libcint source code is copied - we only call public API functions.
 */

#ifndef MSHQC_LIBCINT_WRAPPER_H
#define MSHQC_LIBCINT_WRAPPER_H

#include "mshqc/basis.h"
#include "mshqc/molecule.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <memory>

namespace mshqc {

/**
 * @brief Libcint wrapper for 3-center integrals
 * 
 * Computes genuine 3-center electron repulsion integrals (μν|P)
 * using Libcint's native int3c2e operator.
 * 
 * FORMULA:
 *   (μν|P) = ∫∫ φ_μ(r₁) φ_ν(r₁) r₁₂⁻¹ χ_P(r₂) dr₁ dr₂
 * 
 * where:
 *   - μ, ν: primary basis functions
 *   - P: auxiliary basis function
 *   - r₁₂ = |r₁ - r₂|
 * 
 * USAGE:
 *   LibcintWrapper wrapper(mol, basis);
 *   auto B = wrapper.compute_3center_eri(aux_basis);
 *   // B(μ, ν, P) contains exact 3-center integrals
 */
class LibcintWrapper {
public:
    /**
     * @brief Constructor
     * @param mol Molecule object
     * @param basis Primary basis set
     */
    LibcintWrapper(const Molecule& mol, const BasisSet& basis);
    
    /**
     * @brief Destructor
     */
    ~LibcintWrapper();
    
    // Disable copy (Libcint uses raw pointers)
    LibcintWrapper(const LibcintWrapper&) = delete;
    LibcintWrapper& operator=(const LibcintWrapper&) = delete;
    
    /**
     * @brief Compute true 3-center electron repulsion integrals
     * 
     * Uses Libcint's int3c2e_sph() for native (μν|P) computation.
     * This is the EXACT 3-center integral, no approximation.
     * 
     * REFERENCE:
     * Sun (2015), Section 2.2: "3-center 2-electron integrals"
     * 
     * @param aux_basis Auxiliary basis set (e.g., cc-pVDZ-RI)
     * @return 3-center tensor [nbasis × nbasis × naux]
     */
    Eigen::Tensor<double, 3> compute_3center_eri(const BasisSet& aux_basis);
    
    /**
     * @brief Compute 2-center auxiliary metric (P|Q)
     * 
     * Uses Libcint's int2c2e_sph() for 2-center Coulomb metric.
     * 
     * FORMULA:
     *   J_PQ = (P|Q) = ∫∫ χ_P(r₁) r₁₂⁻¹ χ_Q(r₂) dr₁ dr₂
     * 
     * @param aux_basis Auxiliary basis set
     * @return Metric matrix [naux × naux]
     */
    Eigen::MatrixXd compute_2center_eri(const BasisSet& aux_basis);
    
private:
    const Molecule& mol_;
    const BasisSet& basis_;
    size_t nbasis_;
    
    // Libcint data structures (opaque pointers)
    struct LibcintData;
    std::unique_ptr<LibcintData> libcint_data_;
    
    /**
     * @brief Convert our basis to Libcint format
     * @param basis Basis set to convert
     * @param natm Number of atoms
     * @param atm Libcint atom array (output)
     * @param nbas Number of basis shells
     * @param bas Libcint basis array (output)
     * @param env Libcint environment array (output)
     */
    void convert_basis_to_libcint(
        const BasisSet& basis,
        int& natm, int*& atm,
        int& nbas, int*& bas,
        double*& env
    );
    
    /**
     * @brief Free Libcint arrays
     */
    void free_libcint_arrays(int* atm, int* bas, double* env);
};

} // namespace mshqc

#endif // MSHQC_LIBCINT_WRAPPER_H
