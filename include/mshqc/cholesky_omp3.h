/**
 * @file mshqc/cholesky_omp3.h
 * @brief Header for Cholesky-Decomposed Orbital-Optimized MP3 (Cholesky-OMP3)
 * @author Muhamad Syahrul Hidayat
 * @date 2025-01-11
 */

#ifndef MSHQC_CHOLESKY_OMP3_H
#define MSHQC_CHOLESKY_OMP3_H

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/mp2.h" // Untuk OMP2Result
#include "mshqc/integrals/cholesky_eri.h"

namespace mshqc {

/**
 * @class CholeskyOMP3Config
 * @brief Configuration parameters for Cholesky-OMP3
 */
class CholeskyOMP3Config {
public:
    int max_iterations;
    double energy_threshold;
    double gradient_threshold;
    double cholesky_threshold;
    int print_level;

    CholeskyOMP3Config()
        : max_iterations(50), 
          energy_threshold(1e-6), 
          gradient_threshold(1e-5),
          cholesky_threshold(1e-4),
          print_level(1) {}
};

/**
 * @class CholeskyOMP3Result
 * @brief Results container for Cholesky-OMP3 calculation
 */
class CholeskyOMP3Result {
public:
    double energy_scf;
    double energy_mp2_corr;
    double energy_mp3_corr;
    double energy_total;
    
    bool converged;
    int iterations;

    // Final Optimized Orbitals
    Eigen::MatrixXd C_alpha;
    Eigen::MatrixXd C_beta;
    Eigen::VectorXd orbital_energies_alpha;
    Eigen::VectorXd orbital_energies_beta;

    CholeskyOMP3Result() 
        : energy_scf(0.0), energy_mp2_corr(0.0), energy_mp3_corr(0.0),
          energy_total(0.0), converged(false), iterations(0) {}
};

/**
 * @class CholeskyOMP3
 * @brief Efficient Orbital-Optimized MP3 using Cholesky Decomposition
 * * Uses an OMP2.5 scheme: Optimizes orbitals based on MP2-like gradients
 * but evaluates MP3 energy at each step. Reuses Cholesky vectors
 * to avoid N^4 storage and reconstruction.
 */
class CholeskyOMP3 {
public:
    /**
     * @brief Constructor that reuses existing Cholesky vectors
     * @param mol The molecule object
     * @param basis The basis set
     * @param integrals Integral engine
     * @param omp2_guess Result from previous OMP2 calculation (for orbital guess)
     * @param config Configuration options
     * @param cholesky_vectors Pre-computed Cholesky vectors (Reuse!)
     */
    CholeskyOMP3(const Molecule& mol,
                 const BasisSet& basis,
                 std::shared_ptr<IntegralEngine> integrals,
                 const OMP2Result& omp2_guess,
                 const CholeskyOMP3Config& config,
                 const integrals::CholeskyERI& cholesky_vectors);

    /**
     * @brief Run the Cholesky-OMP3 optimization
     */
    CholeskyOMP3Result compute();

private:
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    CholeskyOMP3Config config_;
    
    // Pointer to existing Cholesky vectors (No ownership, pure reuse)
    const integrals::CholeskyERI* cholesky_ptr_;

    // Internal State
    SCFResult scf_; // Holds current C, P, and Orbital Energies
    
    // Dimensions
    int nbf_;
    int na_, nb_;
    int va_, vb_;

    // Amplitudes Storage (T2)
    Eigen::Tensor<double, 4> t2_aa_;
    Eigen::Tensor<double, 4> t2_bb_;
    Eigen::Tensor<double, 4> t2_ab_;

    // Intermediate Density Matrices (One-Particle)
    Eigen::MatrixXd G_oo_alpha_, G_vv_alpha_;
    Eigen::MatrixXd G_oo_beta_, G_vv_beta_;

    // --- Private Helper Methods ---

    // 1. Vector Transformation
    std::vector<Eigen::MatrixXd> transform_vectors(
        const Eigen::MatrixXd& C_left, 
        const Eigen::MatrixXd& C_right);

    // 2. Amplitudes & Energy
    void compute_t2_amplitudes();
    double compute_mp2_energy();
    double compute_mp3_energy(); // The "lightweight" version

    // 3. Density & Gradient Construction
    void build_opdm_alpha();
    // void build_opdm_beta(); // Implemented inside compute/inlined or symmetric

    Eigen::MatrixXd build_gfock(const Eigen::MatrixXd& P_tot, const Eigen::MatrixXd& P_spin);

    // 4. Orbital Rotation
    void rotate_orbitals(Eigen::MatrixXd& C, const Eigen::MatrixXd& w, 
                         const Eigen::MatrixXd& F_mo, int n_occ, int n_virt);
};

} // namespace mshqc

#endif // MSHQC_CHOLESKY_OMP3_H