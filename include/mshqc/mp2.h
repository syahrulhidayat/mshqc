/**
 * @file mp2.h
 * @brief Restricted Open-shell Møller-Plesset 2nd order perturbation theory
 * 
 * THEORY BACKGROUND:
 * MP2 treats electron correlation through 2nd-order perturbation theory,
 * calculating energy correction from double excitations.
 * 
 * For open-shell systems (ROHF reference), we have three spin cases:
 * - Same-spin alpha-alpha (αα)
 * - Mixed-spin alpha-beta (αβ)  
 * - Same-spin beta-beta (ββ)
 * 
 * REFERENCES:
 * [1] Møller, C. & Plesset, M. S., Phys. Rev. 46, 618 (1934)
 *     - Original MP perturbation theory
 * 
 * [2] Szabo, A. & Ostlund, N. S., Modern Quantum Chemistry (1996)
 *     Dover Publications, ISBN: 0-486-69186-1
 *     - Chapter 6: Many-Body Perturbation Theory
 *     - Section 6.4: Møller-Plesset Perturbation Theory
 * 
 * [3] Bozkaya, U., Turney, J. M., Yamaguchi, Y., Schaefer III, H. F., 
 *     & Sherrill, C. D., J. Chem. Phys. 135, 104103 (2011)
 *     DOI: 10.1063/1.3631129
 *     - Section II.A: ROHF-based MP2
 *     - Equations for spin-adapted amplitudes
 * 
 * [4] Pople, J. A., Binkley, J. S., & Seeger, R., 
 *     Int. J. Quantum Chem. Symp. 10, 1 (1976)
 *     - Spin-restricted open-shell MP2
 */

/**
 * @file mp2.h
 * @brief Møller-Plesset 2nd order perturbation theory (MP2/OMP2)
 */

#ifndef MSHQC_MP2_H
#define MSHQC_MP2_H

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/scf.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/MatrixFunctions>
#include <memory>

namespace mshqc {

struct OMP2Result {
    double energy_scf;
    double energy_mp2_ss;
    double energy_mp2_os;
    double energy_mp2_corr;
    double energy_total;
    
    int n_occ_alpha;
    int n_occ_beta;
    int n_virt_alpha;
    int n_virt_beta;
    
    Eigen::VectorXd orbital_energies_alpha;
    Eigen::VectorXd orbital_energies_beta;
    Eigen::MatrixXd C_alpha;
    Eigen::MatrixXd C_beta;

    
    bool converged = false;
    int iterations = 0;
};

// ============================================================================
// ROMP2: Standard MP2 (Non-optimized)
// ============================================================================
class ROMP2 {
public:
    ROMP2(const SCFResult& scf_result, 
          std::shared_ptr<IntegralEngine> integrals);
    
    OMP2Result compute();
    
private:
    SCFResult scf_result_;
    std::shared_ptr<IntegralEngine> integrals_;
    
    size_t nbasis_;
    int n_occ_alpha_;
    int n_occ_beta_;
    int n_virt_alpha_;
    int n_virt_beta_;
    
    Eigen::Tensor<double, 4> eri_mo_aaaa_;
    Eigen::Tensor<double, 4> eri_mo_aabb_;
    Eigen::Tensor<double, 4> eri_mo_bbbb_;
    
    void transform_integrals();
    double compute_mp2_ss_alpha();
    double compute_mp2_ss_beta();
    double compute_mp2_os();
    double get_antisym_integral(const Eigen::Tensor<double, 4>& eri,
                                int i, int j, int a, int b);
};

// ============================================================================
// ROHF_MBPT2: Semi-canonical MP2
// ============================================================================
class ROHF_MBPT2 {
public:
    ROHF_MBPT2(const SCFResult& scf_result,
               std::shared_ptr<IntegralEngine> integrals);
    
    OMP2Result compute();
    
private:
    SCFResult scf_result_;
    std::shared_ptr<IntegralEngine> integrals_;
    
    size_t nbasis_;
    int n_c_;
    int n_o_;
    int n_v_;
    
    Eigen::MatrixXd C_semi_;
    Eigen::VectorXd eps_semi_;
    Eigen::MatrixXd t1_;
    
    void semi_canonical_transform();
    void compute_t1_amplitudes();
    double t1_energy();
    double compute_mp2_correlation();
};

// ============================================================================
// OMP2: Orbital-Optimized MP2 (COMPLETE IMPLEMENTATION)
// ============================================================================
class OMP2 {
public:
    friend class OMP3;
    OMP2(const Molecule& mol,
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals,
         const SCFResult& scf_guess);
    
    OMP2Result compute();
    
    // Configuration
    void set_max_iterations(int max_iter) { max_iter_ = max_iter; }
    void set_convergence_threshold(double thresh) { conv_thresh_ = thresh; }
    void set_gradient_threshold(double thresh) { grad_thresh_ = thresh; }
    
private:
    // System data
    Molecule mol_;
    BasisSet basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    SCFResult scf_;
    
    // Convergence parameters
    int max_iter_;
    double conv_thresh_;
    double grad_thresh_;
    
    // Dimensions
    int nbf_;
    int na_, nb_;
    int va_, vb_;
    
    // ========================================================================
    // AMPLITUDES & DENSITY MATRICES
    // ========================================================================
    
    // T2 amplitudes: t_ijab
    Eigen::Tensor<double, 4> t2_aa_;  // Alpha-alpha
    Eigen::Tensor<double, 4> t2_bb_;  // Beta-beta
    Eigen::Tensor<double, 4> t2_ab_;  // Alpha-beta
    
    // OPDM (One-Particle Density Matrix) components
    Eigen::MatrixXd G_oo_alpha_;  // Occupied-occupied (alpha)
    Eigen::MatrixXd G_vv_alpha_;  // Virtual-virtual (alpha)
    Eigen::MatrixXd G_oo_beta_;   // Occupied-occupied (beta)
    Eigen::MatrixXd G_vv_beta_;   // Virtual-virtual (beta)
    
    // ========================================================================
    // PART 1: OPDM BUILDER (from Part 1)
    // ========================================================================
    // REF: Psi4 omp2_opdm.cc
    // G_IJ = \sum_{K,A,B} t_IK^AB t_JK^AB
    // G_AB = -\sum_{I,J,C} t_IJ^AC t_IJ^BC
    
    void build_opdm_alpha();
    void build_opdm_beta();
    Eigen::MatrixXd build_opdm();
    
    // ========================================================================
    // PART 2: ORBITAL ROTATION ENGINE (from Part 2)
    // ========================================================================
    // REF: Bozkaya & Sherrill (2013), J. Chem. Phys. 139, 054104
    
    // Generalized Fock matrix: F_pq = h_pq + \sum_r G_pr <pr||qr>
    Eigen::MatrixXd build_gfock_alpha(const Eigen::MatrixXd& G_opdm);
    Eigen::MatrixXd build_gfock_beta(const Eigen::MatrixXd& G_opdm);
    
    // Orbital gradient: w_ai = 2 * F_ai
    Eigen::MatrixXd compute_orbital_gradient_alpha(const Eigen::MatrixXd& F_mo);
    Eigen::MatrixXd compute_orbital_gradient_beta(const Eigen::MatrixXd& F_mo);
    
    // Orbital rotation: C_new = C_old * exp(-kappa)
    void rotate_orbitals_alpha(const Eigen::MatrixXd& w_ai, 
                               const Eigen::MatrixXd& F_mo);
    void rotate_orbitals_beta(const Eigen::MatrixXd& w_ai,
                              const Eigen::MatrixXd& F_mo);
    
    // Convergence check
    bool converged(const Eigen::MatrixXd& w_alpha, 
                   const Eigen::MatrixXd& w_beta,
                   double e_new, double e_old);
    
    // ========================================================================
    // PART 3: AMPLITUDE & ENERGY COMPUTATION (from Part 3)
    // ========================================================================
    
    // Compute T2 amplitudes with current orbitals
    void compute_t2_amplitudes();
    
    // Compute MP2 energy from T2 amplitudes
    double compute_mp2_energy_from_t2();
    
    // ========================================================================
    // LEGACY METHODS (for backward compatibility)
    // ========================================================================
    
    // Non-iterative MP2 (single-shot)
    double compute_rmp2_energy();
    double compute_ump2_energy(double& e_ss, double& e_os);
    
    // Stub methods (not used in OMP2)
    void xform_ints() {}
    void xform_full_mo() {}
    double mp2_energy(double& e_ss, double& e_os) { return 0.0; }
    void rotate_orbitals(const Eigen::MatrixXd& kappa) {}
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Semi-canonical transformation for ROHF
SCFResult semicanonicalize(const SCFResult& rohf_result);

} // namespace mshqc

#endif // MSHQC_MP2_H