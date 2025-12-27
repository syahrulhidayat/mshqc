/**
 * @file include/mshqc/omp3.h
 * @brief Orbital-Optimized MP3 (OMP3) Header - FIXED
 * @details Declares OMP3 class with pseudocanonicalize() and correct members.
 * * @author Muhamad Syahrul Hidayat
 * @date 2025-01-11
 * @license MIT License
 */

#ifndef MSHQC_OMP3_H
#define MSHQC_OMP3_H

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/scf.h"
#include "mshqc/integrals.h"
#include "mshqc/mp2.h" // Include OMP2Result definition
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

namespace mshqc {

/**
 * @struct OMP3Result
 * @brief Results from OMP3 calculation
 */
struct OMP3Result {
    double energy_total;        ///< Total OMP3 energy (HF + MP2 + MP3)
    double energy_mp2_corr;     ///< MP2 correlation energy
    double energy_mp3_corr;     ///< MP3 correlation energy
    double energy_omp2;         ///< Converged OMP2 total energy
    double energy_omp3;         ///< Converged OMP3 total energy
    bool converged;             ///< Convergence status
    int iterations;             ///< Number of iterations
    
    // Orbital energies at convergence
    Eigen::VectorXd orbital_energies_alpha;
    Eigen::VectorXd orbital_energies_beta;
    
    // Final orbitals
    Eigen::MatrixXd C_alpha;
    Eigen::MatrixXd C_beta;
};

/**
 * @class OMP3
 * @brief Orbital-Optimized Third-Order Møller-Plesset Perturbation Theory
 */
class OMP3 {
public:
    /**
     * @brief Constructor from OMP2 result
     */
    OMP3(const Molecule& mol,
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals,
         const OMP2Result& omp2_result);
    
    /**
     * @brief Run OMP3 calculation
     */
    OMP3Result compute();
    
    // Setters
    void set_max_iterations(int max_iter) { max_iter_ = max_iter; }
    void set_convergence_threshold(double thresh) { conv_thresh_ = thresh; }
    void set_gradient_threshold(double thresh) { grad_thresh_ = thresh; }

private:
    // ========================================================================
    // MEMBER VARIABLES
    // ========================================================================
    
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    
    int nbf_;       ///< Number of basis functions
    int na_;        ///< Number of alpha occupied orbitals
    int nb_;        ///< Number of beta occupied orbitals
    int va_;        ///< Number of alpha virtual orbitals
    int vb_;        ///< Number of beta virtual orbitals
    
    SCFResult scf_; // Stores current orbitals and energies
    
    // T2 Amplitudes
    Eigen::Tensor<double, 4> t2_aa_;
    Eigen::Tensor<double, 4> t2_bb_;
    Eigen::Tensor<double, 4> t2_ab_;
    
    // Density Matrices
    Eigen::MatrixXd G_oo_alpha_, G_vv_alpha_;
    Eigen::MatrixXd G_oo_beta_, G_vv_beta_;
    Eigen::MatrixXd Gamma_oo_alpha_, Gamma_vv_alpha_;
    Eigen::MatrixXd Gamma_oo_beta_, Gamma_vv_beta_;
    
    // Parameters
    int max_iter_;
    double conv_thresh_;
    double grad_thresh_;
    
    // ========================================================================
    // HELPER METHODS
    // ========================================================================

    /**
     * @brief Update orbital energies via diagonalization.
     * @details Defined in private to ensure internal consistency.
     */
    void pseudocanonicalize();

    void compute_t2_amplitudes();
    double compute_mp2_energy_from_t2();
    
    double compute_mp3_energy();
    double compute_mp3_particle_hole();
    double compute_mp3_particle_particle();
    double compute_mp3_hole_hole();
    
    void build_opdm_alpha();
    void build_opdm_beta();
    
    void build_mp3_density_contributions_alpha();
    void build_mp3_density_contributions_beta();
    
    Eigen::MatrixXd build_gfock_alpha(const Eigen::MatrixXd& G_mo_alpha, const Eigen::MatrixXd& Gamma_alpha);
    Eigen::MatrixXd build_gfock_beta(const Eigen::MatrixXd& G_mo_beta, const Eigen::MatrixXd& Gamma_beta);
    
    Eigen::MatrixXd compute_orbital_gradient_alpha(const Eigen::MatrixXd& F_mo);
    Eigen::MatrixXd compute_orbital_gradient_beta(const Eigen::MatrixXd& F_mo);
    
    void rotate_orbitals_alpha(const Eigen::MatrixXd& w_ai, const Eigen::MatrixXd& F_mo);
    void rotate_orbitals_beta(const Eigen::MatrixXd& w_ai, const Eigen::MatrixXd& F_mo);
    
    bool converged(const Eigen::MatrixXd& w_alpha, const Eigen::MatrixXd& w_beta, double e_new, double e_old);
};

} // namespace mshqc

#endif // MSHQC_OMP3_H