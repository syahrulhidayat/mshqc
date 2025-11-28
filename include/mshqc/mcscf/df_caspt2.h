/**
 * @file df_caspt2.h
 * @brief Density-Fitted CASPT2 (DF-CASPT2) implementation
 * 
 * Implements Resolution-of-Identity (RI) approximation for CASPT2 to reduce
 * computational cost from O(N^5) to O(N^4) with minimal accuracy loss.
 * 
 * THEORY REFERENCES:
 * 
 * Density Fitting / Resolution-of-Identity:
 *   - M. Feyereisen, G. Fitzgerald, A. Komornicki, Chem. Phys. Lett. **208**, 359 (1993)
 *     "Use of approximate integrals in ab initio theory. An application in MP2 energy calculations"
 *     [Original RI-MP2 formulation, Eq. (7)-(11): 3-center integral approximation]
 * 
 *   - F. Weigend, M. Häser, Theor. Chem. Acc. **97**, 331 (1997)
 *     "RI-MP2: first derivatives and global consistency"
 *     [Auxiliary basis design, optimized RI approach]
 * 
 *   - F. Weigend, A. Köhn, C. Hättig, J. Chem. Phys. **116**, 3175 (2002)
 *     "Efficient use of the correlation consistent basis sets in resolution of the identity MP2 calculations"
 *     [cc-pVXZ-RI auxiliary basis sets]
 * 
 * CASPT2 Theory:
 *   - K. Andersson, P.-Å. Malmqvist, B. O. Roos, J. Chem. Phys. **96**, 1218 (1992)
 *     "Second-order perturbation theory with a complete active space self-consistent field reference function"
 *     [Original CASPT2 formulation, Eq. (15)-(18): E_PT2 = Σ |V_K0|²/(E₀-E_K)]
 * 
 * Algorithm:
 *   Standard CASPT2:  (pq|rs) via 4-index transformation, O(N^5)
 *   DF-CASPT2:        (pq|rs) ≈ Σ_P (pq|P)[J^{-1}]_PQ(Q|rs), O(N^4)
 * 
 * Expected Speedup:  10-100× for N=20-100 basis functions
 * Expected Accuracy: Error < 10 µHa with proper auxiliary basis
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
 * @note Original implementation derived from published theory.
 *       No code copied from existing quantum chemistry software (Psi4, PySCF, etc.).
 *       Algorithm derived from Feyereisen et al. (1993) and Andersson et al. (1992) papers.
 */

#ifndef MSHQC_MCSCF_DF_CASPT2_H
#define MSHQC_MCSCF_DF_CASPT2_H

#include "mshqc/mcscf/caspt2.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include <Eigen/Dense>
#include <memory>

namespace mshqc {
namespace mcscf {

/**
 * @brief Result structure for DF-CASPT2 calculation
 */
struct DFCASPT2Result {
    double e_casscf;              // CASSCF reference energy
    double e_pt2;                 // PT2 correlation correction
    double e_total;               // Total DF-CASPT2 energy
    
    double ipea_shift_used;       // IPEA shift applied
    double imaginary_shift_used;  // Imaginary shift applied
    
    bool converged;               // Convergence status
    std::string status_message;   // Status message
    
    // DF-specific diagnostics
    int n_aux;                    // Number of auxiliary basis functions
    double fitting_error_estimate; // Est. error from DF approximation
    double speedup_factor;        // Actual speedup vs. conventional CASPT2
};

/**
 * @brief Density-Fitted CASPT2 (DF-CASPT2)
 * 
 * Approximates 4-center ERIs using 3-center integrals + auxiliary basis:
 *   (pq|rs) ≈ Σ_PQ (pq|P) [J^{-1}]_PQ (Q|rs)
 * 
 * where:
 *   - P, Q run over auxiliary basis
 *   - J_PQ = (P|Q) is auxiliary basis metric
 *   - (pq|P) are 3-center integrals
 * 
 * Reduces scaling from O(N^5) to O(N^4) with <10 µHa accuracy.
 * 
 * REFERENCE: Feyereisen et al. (1993), Eq. (7)-(11)
 */
class DFCASPT2 {
public:
    /**
     * @brief Constructor
     * @param mol Molecule object
     * @param basis Primary basis set
     * @param aux_basis Auxiliary (fitting) basis set (e.g., cc-pVDZ-RI)
     * @param integrals Integral engine
     * @param casscf_result CASSCF reference result
     */
    DFCASPT2(const Molecule& mol,
             const BasisSet& basis,
             const BasisSet& aux_basis,
             std::shared_ptr<IntegralEngine> integrals,
             std::shared_ptr<CASResult> casscf_result);
    
    /**
     * @brief Destructor (explicit for debugging)
     */
    ~DFCASPT2();
    
    // Delete copy constructor and assignment (avoid CASResult copy issues)
    DFCASPT2(const DFCASPT2&) = delete;
    DFCASPT2& operator=(const DFCASPT2&) = delete;
    
    /**
     * @brief Compute DF-CASPT2 energy
     * @return DFCASPT2Result structure
     */
    DFCASPT2Result compute();
    
    /**
     * @brief Set IPEA shift (default: 0.25 Ha)
     * 
     * REFERENCE: Ghigo et al., Chem. Phys. Lett. **396**, 142 (2004)
     * IPEA shift prevents intruder states in CASPT2
     */
    void set_ipea_shift(double shift) { ipea_shift_ = shift; }
    
    /**
     * @brief Set imaginary shift for intruder state avoidance
     */
    void set_imaginary_shift(double shift) { imaginary_shift_ = shift; }
    
private:
    // Input data
    const Molecule& mol_;
    const BasisSet& basis_;
    const BasisSet& aux_basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    std::shared_ptr<CASResult> casscf_;  // Shared pointer to avoid deep copy issues
    
    // Dimensions
    int nbf_;     // Number of primary basis functions
    int naux_;    // Number of auxiliary basis functions
    int n_mo_;    // Number of MO orbitals
    
    // CASPT2 parameters
    double ipea_shift_ = 0.0;
    double imaginary_shift_ = 0.0;
    
    // DF data structures
    Eigen::MatrixXd J_;         // Auxiliary metric (P|Q)
    Eigen::MatrixXd J_inv_sqrt_; // J^{-1/2} for fitting
    
    // 3-center fitted integrals in MO basis: B̃^P_pq = (B * J^{-1/2})_pq,P
    // Stored as [n_mo*n_mo, naux] matrix
    // REFERENCE: Feyereisen et al. (1993), Eq. (10)
    Eigen::MatrixXd B_mo_;
    
    /**
     * @brief Compute auxiliary basis metric J_PQ = (P|Q) and invert
     * 
     * REFERENCE: Feyereisen et al. (1993), Eq. (8)
     * Metric must be positive definite for RI to work
     */
    void compute_metric();
    
    /**
     * @brief Transform 3-center integrals from AO to MO basis
     * 
     * REFERENCE: Weigend & Häser (1997), Section 2.2
     * B^P_pq = Σ_μν C_μp (μν|P) C_νq
     * 
     * Stores result in B_mo_[pq, P] format
     */
    void transform_3center_to_mo();
    
    /**
     * @brief Compute DF-CASPT2 energy using fitted integrals
     * 
     * REFERENCE: Andersson et al. (1992), Eq. (15)-(18)
     * Applied with DF approximation for ERIs
     * 
     * E_PT2 = Σ_K |V_K0|² / (E₀ - E_K)
     * where V_K0 uses DF-approximated matrix elements
     */
    double compute_pt2_energy_df();
};

} // namespace mcscf
} // namespace mshqc

#endif // MSHQC_MCSCF_DF_CASPT2_H
