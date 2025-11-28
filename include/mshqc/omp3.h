#ifndef MSHQC_OMP3_H
#define MSHQC_OMP3_H

#include "mshqc/scf.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include <Eigen/Dense>

/**
 * @file omp3.h
 * @brief Orbital-Optimized MP3 (OMP3) for open-shell systems
 * 
 * OMP3 extends MP3 by optimizing the orbitals simultaneously with
 * the correlation energy. This makes it more accurate and variational,
 * especially for open-shell systems like radicals.
 * 
 * THEORY REFERENCES:
 * - Bozkaya & Sherrill, J. Chem. Phys. 139, 054104 (2013)
 *   "Orbital-optimized third-order Møller-Plesset perturbation theory"
 *   Equations (1)-(15) - OMP3 energy and gradient
 * 
 * - Bozkaya et al., J. Chem. Phys. 135, 104103 (2011)
 *   "Orbital-optimized MP2 and MP3"
 *   Theory and implementation details
 * 
 * - Lochan & Head-Gordon, J. Chem. Phys. 126, 164101 (2007)
 *   "Orbital-optimized opposite-spin scaled MP2"
 *   Foundational OMP2 theory
 * 
 * Algorithm:
 * 1. Start from ROHF orbitals
 * 2. Compute MP3 energy and amplitudes
 * 3. Compute orbital gradient
 * 4. Update orbitals via orbital rotation
 * 5. Iterate until convergence
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-01-11
 * @license MIT License
 */

namespace mshqc {

/**
 * OMP3 result structure
 */
struct OMP3Result {
    double e_ref;        // Reference (ROHF) energy
    double e_corr_mp2;   // MP2 correlation
    double e_corr_mp3;   // MP3 correlation  
    double e_corr;       // Total correlation (MP2 + MP3)
    double e_total;      // Total OMP3 energy
    int iterations;      // Number of orbital optimization iterations
    bool converged;      // Convergence status
};

/**
 * Orbital-Optimized MP3 for ROHF
 * 
 * Implements OMP3 theory with orbital optimization for improved
 * accuracy over standard MP3, especially for open-shell systems.
 * 
 * REFERENCE: Bozkaya & Sherrill, J. Chem. Phys. 139, 054104 (2013)
 */
class OMP3 {
public:
    /**
     * Constructor
     * @param rohf_result Initial ROHF result
     * @param basis Basis set
     * @param integrals Integral engine
     */
    OMP3(const SCFResult& rohf_result,
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals);
    
    /**
     * Compute OMP3 energy
     * @param max_iter Maximum orbital optimization iterations
     * @param e_conv Energy convergence threshold
     * @param grad_conv Gradient convergence threshold
     * @return OMP3 result
     */
    OMP3Result compute(int max_iter = 50, 
                      double e_conv = 1e-8,
                      double grad_conv = 1e-6);
    
private:
    SCFResult rohf_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    
    // Dimensions
    int nbf_;    // basis functions
    int nocc_;   // occupied (doubly)
    int nvir_;   // virtual
    
    // MO integrals
    Eigen::Tensor<double, 4> ERI_MO_;
    
    // Current orbitals and energies
    Eigen::MatrixXd C_;
    Eigen::VectorXd orbital_energies_;
    
    /**
     * Transform ERIs to MO basis
     */
    void transform_integrals();
    
    /**
     * Compute MP2 energy and amplitudes
     * REFERENCE: Szabo & Ostlund, Section 6.4
     */
    double compute_mp2_energy(Eigen::Tensor<double, 4>& t2);
    
    /**
     * Compute MP3 energy correction
     * REFERENCE: Paldus & Bartlett, Adv. Quantum Chem. 20, 291 (1989)
     */
    double compute_mp3_energy(const Eigen::Tensor<double, 4>& t2);
    
    /**
     * Compute orbital gradient for optimization
     * REFERENCE: Bozkaya & Sherrill (2013), Eq. (10)
     */
    Eigen::MatrixXd compute_orbital_gradient(const Eigen::Tensor<double, 4>& t2);
    
    /**
     * Perform orbital rotation update
     * REFERENCE: Helgaker et al., Section 10.8
     */
    void rotate_orbitals(const Eigen::MatrixXd& gradient, double step_size);
    
    /**
     * Update Fock matrix with correlation contributions
     */
    void update_fock_matrix();
};

} // namespace mshqc

#endif // MSHQC_OMP3_H
