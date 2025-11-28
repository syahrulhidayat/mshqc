/**
 * @file cholesky_caspt2.h
 * @brief Cholesky-Decomposed CASPT2 Implementation
 * 
 * Uses Cholesky decomposition of ERIs for efficient CASPT2.
 * Mathematically equivalent to DF-CASPT2 but uses adaptive Cholesky vectors
 * instead of auxiliary basis RI approximation.
 * 
 * THEORY REFERENCES:
 *   - K. Andersson et al., J. Chem. Phys. **96**, 1218 (1992)
 *     [CASPT2 theory, E_PT2 = Σ |V_K0|²/(E₀-E_K)]
 *   - H. Koch et al., J. Chem. Phys. **118**, 9481 (2003)
 *     [Cholesky decomposition for quantum chemistry]
 *   - F. Aquilante et al., J. Chem. Phys. **129**, 024113 (2008)
 *     [Cholesky-based CASPT2 implementation]
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 * @license MIT License
 */

#ifndef MSHQC_MCSCF_CHOLESKY_CASPT2_H
#define MSHQC_MCSCF_CHOLESKY_CASPT2_H

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/mcscf/casscf.h"
#include "mshqc/integrals/cholesky_eri.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <vector>

namespace mshqc {
namespace mcscf {

/**
 * @brief Result structure for Cholesky-CASPT2 calculation
 */
struct CholeskyCASPT2Result {
    double e_casscf = 0.0;              ///< CASSCF reference energy
    double e_pt2 = 0.0;                 ///< PT2 correlation energy
    double e_total = 0.0;               ///< Total energy (CASSCF + PT2)
    
    double ipea_shift_used = 0.0;       ///< IPEA shift applied
    double imaginary_shift_used = 0.0;  ///< Imaginary shift applied
    
    int n_cholesky_vectors = 0;         ///< Number of Cholesky vectors used
    double cholesky_threshold = 0.0;    ///< Cholesky decomposition threshold
    double cholesky_error = 0.0;        ///< Max Cholesky reconstruction error
    
    bool converged = false;             ///< Whether calculation converged
    std::string status_message;         ///< Human-readable status
};

/**
 * @brief Cholesky-Decomposed CASPT2
 * 
 * Computes second-order perturbation correction to CASSCF using
 * Cholesky-decomposed electron repulsion integrals.
 * 
 * Algorithm:
 *   1. Decompose AO ERIs: (μν|λσ) = Σ_K L^K_μν L^K_λσ
 *   2. Transform to MO: L^K_μν → L^K_pq
 *   3. Reconstruct MO ERIs from Cholesky vectors
 *   4. Compute PT2 energy correction
 * 
 * Advantages over DF-CASPT2:
 *   - No auxiliary basis needed
 *   - Adaptive: only keeps significant vectors
 *   - Error-controlled by threshold
 *   - Exact reconstruction (within threshold)
 */
class CholeskyCASPT2 {
public:
    /**
     * @brief Constructor
     * @param mol Molecule
     * @param basis Basis set
     * @param integrals Integral engine
     * @param casscf_result CASSCF result (reference wavefunction)
     * @param cholesky_threshold Threshold for Cholesky decomposition (default 1e-6)
     */
    CholeskyCASPT2(const Molecule& mol,
                   const BasisSet& basis,
                   std::shared_ptr<IntegralEngine> integrals,
                   const CASResult& casscf_result,
                   double cholesky_threshold = 1e-6);
    
    /**
     * @brief Compute Cholesky-CASPT2 energy
     * @return CholeskyCASPT2Result with energies and statistics
     */
    CholeskyCASPT2Result compute();
    
    /**
     * @brief Set IPEA shift (ionization potential - electron affinity)
     * 
     * IPEA shift modifies zeroth-order Hamiltonian to improve agreement
     * with experiment. Standard value: 0.25 Ha.
     * 
     * REFERENCE: G. Ghigo et al., Chem. Phys. Lett. 396, 142 (2004)
     */
    void set_ipea_shift(double shift) { ipea_shift_ = shift; }
    
    /**
     * @brief Set imaginary level shift for numerical stability
     * 
     * Small imaginary shift added to denominators to avoid singularities.
     * Standard value: 0.1-0.3 Ha.
     */
    void set_imaginary_shift(double shift) { imaginary_shift_ = shift; }
    
    /**
     * @brief Get Cholesky decomposition statistics
     */
    const integrals::CholeskyDecompositionResult& cholesky_stats() const {
        return cholesky_result_;
    }

private:
    // Member data
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    const CASResult& casscf_;
    
    int nbf_;                    ///< Number of basis functions
    int n_mo_;                   ///< Number of MOs
    double ipea_shift_ = 0.0;    ///< IPEA shift
    double imaginary_shift_ = 0.0; ///< Imaginary shift
    double cholesky_threshold_;  ///< Cholesky threshold
    
    // Cholesky data
    std::unique_ptr<integrals::CholeskyERI> cholesky_eri_;
    integrals::CholeskyDecompositionResult cholesky_result_;
    std::vector<Eigen::MatrixXd> L_mo_;  ///< Cholesky vectors in MO basis [K](p,q)
    
    /**
     * @brief Compute Cholesky decomposition of AO ERIs
     */
    void compute_cholesky_decomposition();
    
    /**
     * @brief Transform Cholesky vectors from AO to MO basis
     * 
     * L^K_μν → L^K_pq using C_μp MO coefficients
     */
    void transform_cholesky_to_mo();
    
    /**
     * @brief Compute PT2 energy using Cholesky-reconstructed ERIs
     * 
     * Reconstructs (pq|rs) = Σ_K L^K_pq L^K_rs on-the-fly
     * to avoid storing full 4-index MO ERI tensor.
     */
    double compute_pt2_energy_cholesky();
};

} // namespace mcscf
} // namespace mshqc

#endif // MSHQC_MCSCF_CHOLESKY_CASPT2_H
