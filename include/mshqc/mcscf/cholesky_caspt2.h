/**
 * @file cholesky_caspt2.h
 * @brief Complete Ab Initio Cholesky-CASPT2 Implementation
 * 
 * THEORY: Pure Rayleigh-Schrödinger perturbation theory (2nd order)
 * E_PT2 = Σ_K |⟨K|H|0⟩|² / (E₀ - E_K)
 * 
 * REFERENCES:
 * - Andersson et al., J. Phys. Chem. 94, 5483 (1990)
 * - Andersson et al., J. Chem. Phys. 96, 1218 (1992)
 * - Roos et al., Chem. Phys. Lett. 245, 215 (1995)
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-12-14
 */

#ifndef MSHQC_MCSCF_CHOLESKY_CASPT2_H
#define MSHQC_MCSCF_CHOLESKY_CASPT2_H

#include "mshqc/mcscf/casscf.h"
#include "mshqc/integrals/cholesky_eri.h"
#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/ci/determinant.h"
#include "mshqc/ci/slater_condon.h"  // Contains CIIntegrals
#include <vector>
#include <memory>
#include <string>
#include <Eigen/Dense>

namespace mshqc {
namespace mcscf {

// ============================================================================
// HELPER STRUCTURES - DEFINED FIRST
// ============================================================================

/**
 * @brief Analysis of PT2 contributions by excitation type
 */
struct PT2Analysis {
    double E_SD = 0.0;   ///< Singles: Inactive → Active
    double E_ST = 0.0;   ///< Singles: Active → Virtual
    double E_D = 0.0;    ///< Doubles
    double E_TQ = 0.0;   ///< Triples/Quadruples
    
    int n_SD = 0;        ///< Number of SD excitations
    int n_ST = 0;        ///< Number of ST excitations
    int n_D = 0;         ///< Number of double excitations
    int n_TQ = 0;        ///< Number of higher excitations
    
    /**
     * @brief Print analysis to console
     */
    void print() const;
};

// Forward declarations
// (None needed now since PT2Analysis is defined above)

/**
 * @brief Result structure for CASPT2 calculation
 */
struct CholeskyCASPT2Result {
    double e_casscf = 0.0;        ///< CASSCF reference energy
    double e_pt2 = 0.0;           ///< PT2 correlation energy
    double e_total = 0.0;         ///< Total CASPT2 energy
    int n_cholesky_vectors = 0;   ///< Number of Cholesky vectors used
    double time_total_s = 0.0;    ///< Total computation time
    bool converged = false;       ///< Convergence status
    std::string status_message;   ///< Status message
};

/**
 * @brief Complete Ab Initio Cholesky-CASPT2 Implementation
 * 
 * Features:
 * - Pure Rayleigh-Schrödinger perturbation theory (no semi-empirics)
 * - Cholesky decomposition for efficient integral handling
 * - IPEA shift correction (optional)
 * - Imaginary shift regularization (optional)
 * - Detailed excitation analysis
 */
class CholeskyCASPT2 {
public:
    // ========================================================================
    // CONSTRUCTORS
    // ========================================================================
    
    /**
     * @brief Standard constructor (will decompose ERIs)
     */
    CholeskyCASPT2(
        const Molecule& mol, 
        const BasisSet& basis,
        std::shared_ptr<IntegralEngine> integrals,
        const CASResult& casscf_result,
        double cholesky_threshold = 1e-6
    );
    
    /**
     * @brief Constructor with existing Cholesky vectors (reuse)
     */
    CholeskyCASPT2(
        const Molecule& mol,
        const BasisSet& basis,
        std::shared_ptr<IntegralEngine> integrals,
        const CASResult& casscf_result,
        const integrals::CholeskyERI& existing_cholesky
    );
    
    // ========================================================================
    // CONFIGURATION
    // ========================================================================
    
    /**
     * @brief Set IPEA shift (default: 0.25 Ha)
     * Roos et al., Chem. Phys. Lett. 245, 215 (1995)
     */
    void set_ipea_shift(double shift) { ipea_shift_ = shift; }
    
    /**
     * @brief Set imaginary shift for regularization (default: 0.0 Ha)
     */
    void set_imaginary_shift(double shift) { imaginary_shift_ = shift; }
    
    /**
     * @brief Get current IPEA shift
     */
    double get_ipea_shift() const { return ipea_shift_; }
    
    /**
     * @brief Get current imaginary shift
     */
    double get_imaginary_shift() const { return imaginary_shift_; }
    
    // ========================================================================
    // MAIN COMPUTATION
    // ========================================================================
    
    /**
     * @brief Compute CASPT2 energy
     * @return Complete CASPT2 result
     */
    CholeskyCASPT2Result compute();
    
    /**
     * @brief Compute with detailed diagnostics
     * @return CASPT2 result with excitation analysis
     */
    CholeskyCASPT2Result compute_with_diagnostics();
    
private:
    // ========================================================================
    // MEMBER VARIABLES
    // ========================================================================
    
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    CASResult casscf_;
    
    // Cholesky decomposition
    std::unique_ptr<integrals::CholeskyERI> cholesky_eri_;
    std::vector<Eigen::MatrixXd> L_mo_;  ///< Cholesky vectors in MO basis
    bool vectors_provided_ = false;
    
    // Configuration
    double cholesky_threshold_ = 1e-6;
    double ipea_shift_ = 0.25;
    double imaginary_shift_ = 0.0;
    
    // Dimensions
    int nbf_ = 0;   ///< Number of AO basis functions
    int n_mo_ = 0;  ///< Number of MO basis functions
    
    // ========================================================================
    // PRIVATE METHODS
    // ========================================================================
    
    /**
     * @brief Ensure Cholesky vectors are available
     */
    void ensure_cholesky_vectors();
    
    /**
     * @brief Transform Cholesky vectors from AO to MO basis
     */
    void transform_cholesky_to_mo();
    
    /**
     * @brief Build complete MO integrals from Cholesky vectors
     * @return CI integrals structure with h_pq and (pq|rs)
     */
    ci::CIIntegrals build_mo_integrals();
    
    /**
     * @brief Compute PT2 energy using complete ab initio formula
     * @param external_dets External excitation determinants
     * @param integrals_mo MO integrals
     * @return PT2 correlation energy
     */
    double compute_pt2_energy_complete(
        const std::vector<ci::Determinant>& external_dets,
        const ci::CIIntegrals& integrals_mo
    );
    
    /**
     * @brief Compute PT2 with detailed excitation analysis
     * @param external_dets External excitation determinants
     * @param integrals_mo MO integrals
     * @param analysis Output structure for analysis
     * @return PT2 correlation energy
     */
    double compute_pt2_with_analysis(
        const std::vector<ci::Determinant>& external_dets,
        const ci::CIIntegrals& integrals_mo,
        PT2Analysis& analysis
    ) const;
    
    /**
     * @brief Get orbital energy for given orbital index
     * @param p Orbital index
     * @return Orbital energy in Ha
     */
    double get_orbital_energy(int p) const;
    
    /**
     * @brief Compute Fock matrix in MO basis
     * @param ints CI integrals
     * @return Fock matrix
     */
    Eigen::MatrixXd compute_fock_mo(const ci::CIIntegrals& ints) const;
};

} // namespace mcscf
} // namespace mshqc

#endif // MSHQC_MCSCF_CHOLESKY_CASPT2_H