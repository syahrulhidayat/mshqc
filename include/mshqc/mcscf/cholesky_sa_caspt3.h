/**
 * @file cholesky_sa_caspt3.h
 * @brief Header for Cholesky State-Specific CASPT3 (Full Loop Implementation)
 * @details 
 * Defines the CholeskySACASPT3 class which implements the third-order 
 * perturbation theory correction using Cholesky decomposed integrals.
 * This version uses explicit loops to ensure full index matching accuracy.
 */

#ifndef MSHQC_MCSCF_CHOLESKY_SA_CASPT3_H
#define MSHQC_MCSCF_CHOLESKY_SA_CASPT3_H

// Include CASSCF result definitions
#include "mshqc/mcscf/cholesky_sa_casscf.h"

// [WAJIB] Include ini agar struct 'PT2Amplitudes' dikenali
// PT2Amplitudes diperlukan untuk mentransfer amplitudo T2 dari PT2 ke PT3
#include "mshqc/mcscf/cholesky_sa_caspt2.h"

#include <vector>
#include <functional> // Diperlukan untuk std::function
#include <Eigen/Dense>

namespace mshqc {
namespace mcscf {

/**
 * @brief Configuration struct for CASPT3 calculation
 */
struct CASPT3Config {
    double shift = 0.0;         ///< Level shift parameter to avoid intruder states
    double zero_thresh = 1e-12; ///< Threshold for zero values
    int print_level = 1;        ///< Output verbosity (0=Silent, 1=Summary, 2=Detail)
};

/**
 * @brief Container for CASPT3 results
 */
struct CASPT3Result {
    std::vector<double> e_cas;   ///< Reference CASSCF energies
    std::vector<double> e_pt2;   ///< Second-order corrections
    std::vector<double> e_pt3;   ///< Third-order corrections
    std::vector<double> e_total; ///< Total energies (CAS + PT2 + PT3)
};

/**
 * @class CholeskySACASPT3
 * @brief Main driver class for CASPT3 calculations using Cholesky vectors.
 */
class CholeskySACASPT3 {
public:
    /**
     * @brief Constructor
     * @param result Results from the preceding CASSCF calculation
     * @param vecs Cholesky vectors in AO basis
     * @param n_basis Number of basis functions
     * @param active_space Definition of active, inactive, and virtual spaces
     * @param config Configuration parameters
     */
    CholeskySACASPT3(const SACASResult& result,
                     const std::vector<Eigen::VectorXd>& vecs,
                     int n_basis,
                     const ActiveSpace& active_space,
                     const CASPT3Config& config);

    /**
     * @brief Executes the CASPT3 calculation for all states.
     * @return CASPT3Result struct containing energies for all states.
     */
    CASPT3Result compute();

private:
    // Internal data storage
    SACASResult cas_res_;
    std::vector<Eigen::VectorXd> L_ao_; 
    std::vector<Eigen::MatrixXd> L_mo_; 
    
    int nbasis_;
    ActiveSpace active_space_;
    CASPT3Config config_;
    
    int n_inact_, n_act_, n_virt_;

    // Helper method (Placeholder/Legacy)
    void transform_cholesky_to_mo();
    
    // Legacy / Verification Kernels
    double compute_pt2_verify(int state_idx, const Eigen::VectorXd& eps);
    double compute_state_pt3_fast(int state_idx, const Eigen::VectorXd& eps);

    /**
     * @brief Core PT3 Calculation Kernel (Full Loop Implementation)
     * @param state_idx Index of the electronic state being calculated
     * @param eps Canonical orbital energies
     * @param get_int Lambda function to retrieve integrals on-the-fly
     * @param amps Reference to PT2 amplitudes (Source of T2)
     * @return Third-order energy correction (E3)
     */
    double compute_state_pt3_optimized(
        int state_idx, 
        const Eigen::VectorXd& eps,
        const std::function<double(int,int,int,int)>& get_int,
        const PT2Amplitudes& amps 
    );
};

} // namespace mcscf
} // namespace mshqc

#endif // MSHQC_MCSCF_CHOLESKY_SA_CASPT3_H