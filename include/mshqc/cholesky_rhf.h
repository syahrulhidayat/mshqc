/**
 * @file mshqc/cholesky_rhf.h
 * @brief Header for Cholesky-Decomposed Restricted Hartree-Fock (RHF)
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 */

#ifndef MSHQC_CHOLESKY_RHF_H
#define MSHQC_CHOLESKY_RHF_H

#include <memory>
#include <vector>
#include <Eigen/Dense>

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h" // Menggunakan SCFConfig, SCFResult, dan DIIS dari sini
#include "mshqc/integrals/cholesky_eri.h"

namespace mshqc {

/**
 * @brief Configuration for Cholesky-RHF
 * Inherits standard SCF parameters and adds Cholesky-specific options.
 */
struct CholeskyRHFConfig : public SCFConfig {
    /// Threshold for Cholesky Decomposition (if internal decomposition is used)
    double cholesky_threshold = 1e-6;

    /// Enable screening of exchange matrix construction (skip small L vectors)
    bool screen_exchange = true;

    CholeskyRHFConfig() {
        // Default overrides for Cholesky RHF if needed
        print_level = 1;
    }
};

/**
 * @class CholeskyRHF
 * @brief Efficient RHF implementation using Cholesky Decomposition
 * * Uses vectors L_uv^P instead of 4-index ERI tensor (uv|rs).
 * - J matrix constructed via O(N^3) trace operations.
 * - K matrix constructed via O(N^3) matrix multiplications.
 */
class CholeskyRHF {
public:
    /**
     * @brief Constructor 1: Standard (Decompose internally)
     * Performs Cholesky decomposition of the ERI tensor during initialization.
     */
    CholeskyRHF(const Molecule& mol,
                const BasisSet& basis,
                std::shared_ptr<IntegralEngine> integrals,
                const CholeskyRHFConfig& config);

    /**
     * @brief Constructor 2: Reuse Vectors (High Efficiency)
     * Uses existing Cholesky vectors (e.g., from a previous calculation or pipeline).
     * Does NOT re-compute integrals or decomposition.
     */
    CholeskyRHF(const Molecule& mol,
                const BasisSet& basis,
                std::shared_ptr<IntegralEngine> integrals,
                const CholeskyRHFConfig& config,
                const integrals::CholeskyERI& existing_cholesky);

    /**
     * @brief Run the SCF procedure
     * @return SCFResult containing energy, coefficients, and convergence status
     */
    SCFResult compute();

    /**
     * @brief Get current total energy
     */
    double energy() const { return energy_; }

private:
    // References & Config
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    CholeskyRHFConfig config_;
    
    // Convergence Accelerator
    DIIS diis_;

    // Cholesky Vectors Management
    // cholesky_ptr_ always points to valid vectors (either internal or external)
    const integrals::CholeskyERI* cholesky_ptr_;
    std::unique_ptr<integrals::CholeskyERI> internal_cholesky_;

    // Matrices
    Eigen::MatrixXd S_;    ///< Overlap Matrix
    Eigen::MatrixXd H_;    ///< Core Hamiltonian
    Eigen::MatrixXd X_;    ///< Orthogonalization Matrix (S^-1/2)
    Eigen::MatrixXd C_;    ///< MO Coefficients
    Eigen::MatrixXd P_;    ///< Density Matrix
    Eigen::MatrixXd F_;    ///< Fock Matrix
    Eigen::VectorXd eps_;  ///< Orbital Energies

    // Scalars
    double energy_;
    int nbasis_;
    int n_occ_;

    // --- Private Helper Methods ---

    /// Initialize 1-electron integrals (S, T, V) and orthogonalizer X
    void init_integrals();

    /// Generate initial guess (Core Hamiltonian)
    void initial_guess();

    /// Construct Density Matrix P = 2 * C_occ * C_occ^T
    Eigen::MatrixXd build_density();

    /// Construct Fock Matrix using Cholesky Vectors
    /// F = H + J[L] - 0.5 * K[L]
    void build_fock();

    /// Compute Total Energy
    double compute_energy();

    /// Solve F' C' = C' e (Standard Eigenvalue Problem)
    void solve_fock();
};

} // namespace mshqc

#endif // MSHQC_CHOLESKY_RHF_H