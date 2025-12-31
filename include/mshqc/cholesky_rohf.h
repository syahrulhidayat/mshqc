/**
 * @file cholesky_rohf.h
 * @brief Cholesky-based Restricted Open-Shell Hartree-Fock (ROHF)
 * * ALGORITHM:
 * Uses Cholesky vectors L^K_μν to construct Fock matrices efficiently.
 * - Coulomb (J): Computed in AO basis via scalar contraction.
 * - Exchange (K): Computed using half-transformed vectors (L^K_μi) to reduce scaling.
 * * SCALING: O(N^3) instead of O(N^4)
 * * @author Muhamad Syahrul Hidayat
 * @date 2025-02-01
 */

#ifndef MSHQC_CHOLESKY_ROHF_H
#define MSHQC_CHOLESKY_ROHF_H

#include "mshqc/scf.h"
#include "mshqc/integrals/cholesky_eri.h"
#include <memory>
#include <vector>

namespace mshqc {

/**
 * @brief Configuration for Cholesky ROHF
 */
struct CholeskyROHFConfig : public SCFConfig {
    double cholesky_threshold = 1e-6;
    bool screen_exchange = true;
};

class CholeskyROHF {
public:
    // ------------------------------------------------------------------------
    // CONSTRUCTORS
    // ------------------------------------------------------------------------
    
    /**
     * @brief Constructor Standard (Melakukan Dekomposisi Sendiri)
     */
    CholeskyROHF(const Molecule& mol,
                 const BasisSet& basis,
                 std::shared_ptr<IntegralEngine> integrals,
                 int n_alpha,
                 int n_beta,
                 const CholeskyROHFConfig& config = CholeskyROHFConfig());

    /**
     * @brief Constructor REUSE (Menggunakan Vektor yang Sudah Ada)
     * Sangat efisien jika dijalankan setelah Cholesky-UHF atau perhitungan lain.
     */
    CholeskyROHF(const Molecule& mol,
                 const BasisSet& basis,
                 std::shared_ptr<IntegralEngine> integrals,
                 int n_alpha,
                 int n_beta,
                 const CholeskyROHFConfig& config,
                 const integrals::CholeskyERI& existing_cholesky);

    // ------------------------------------------------------------------------
    // MAIN COMPUTE
    // ------------------------------------------------------------------------
    SCFResult compute();

private:
    // System
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    CholeskyROHFConfig config_;
    
    size_t nbasis_;
    int n_alpha_;
    int n_beta_;

    // Cholesky Data
    std::unique_ptr<integrals::CholeskyERI> cholesky_;
    bool own_cholesky_; // True jika kita yang manage memori cholesky_

    // Matrices
    Eigen::MatrixXd S_, H_, X_;
    Eigen::MatrixXd C_alpha_, C_beta_;
    Eigen::MatrixXd P_alpha_, P_beta_;
    Eigen::MatrixXd F_alpha_, F_beta_;
    Eigen::VectorXd eps_alpha_, eps_beta_;
    
    double energy_ = 0.0;
    double energy_old_ = 0.0;

    // Helpers
    void init_integrals();
    void initial_guess();
    
    /**
     * @brief Build Fock Matrix using Cholesky Vectors (The Magic Happens Here)
     * F = H + J_tot - K_spin
     */
    void build_fock_cholesky();
    
    Eigen::MatrixXd build_density(const Eigen::MatrixXd& C, int n_occ);
    double compute_energy();
    void solve_fock(const Eigen::MatrixXd& F, Eigen::MatrixXd& C, Eigen::VectorXd& eps);
    bool check_convergence();
};

} // namespace mshqc

#endif // MSHQC_CHOLESKY_ROHF_H