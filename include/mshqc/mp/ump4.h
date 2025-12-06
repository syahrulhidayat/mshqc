/**
 * @file ump4.h
 * @brief Unrestricted Møller-Plesset 4th-order perturbation theory
 * 
 * THEORY REFERENCES:
 *   - K. Raghavachari, G. W. Trucks, J. A. Pople, & M. Head-Gordon,
 *     Chem. Phys. Lett. 157, 479 (1989)
 *     [MP4 formulation and implementation strategies]
 *   - J. A. Pople, R. Krishnan, H. B. Schlegel, & J. S. Binkley,
 *     Int. J. Quantum Chem. 14, 545 (1978)
 *     [Fourth-order MBPT for molecules]
 *   - R. J. Bartlett & D. M. Silver, Phys. Rev. A 10, 1927 (1974)
 *     [Many-body perturbation theory diagrams]
 *   - T. Helgaker, P. Jørgensen, & J. Olsen,
 *     "Molecular Electronic-Structure Theory" (2000), Section 14.4
 *     [Møller-Plesset perturbation theory, Eq. (14.66)-(14.70)]
 * 
 * FORMULA (fourth-order energy):
 *   E^(4) = E_S^(4) + E_D^(4) + E_Q^(4) + E_T^(4)
 *   
 *   where:
 *     E_S: Singles contribution (from T1^(3) amplitudes)
 *     E_D: Doubles contribution (from T2^(3) amplitudes)
 *     E_Q: Quadruples contribution (O(N^8)!)
 *     E_T: Triples contribution (O(N^7))
 * 
 * MP4(SDQ): Singles + Doubles + Quadruples (no triples)
 * MP4(SDTQ): Full MP4 (with triples)
 * 
 * Computational scaling:
 *   - T1^(3): O(N^4)
 *   - T2^(3): O(N^6)
 *   - T3^(3): O(N^7)
 *   - T4^(3): O(N^8) - bottleneck!
 * 
 * @author Syahrul
 * @date 2025-11-12
 * @license MIT
 * 
 * @note Original implementation from published theory.
 *       No code copied from existing quantum chemistry software.
 */

/**
 * @file ump4.h
 * @brief Unrestricted MP4 Header - Added Triples Support
 */

#ifndef MSHQC_MP_UMP4_H
#define MSHQC_MP_UMP4_H

#include "mshqc/ump3.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <tuple>

namespace mshqc {
namespace mp {

struct UMP4Result {
    double e_uhf;
    double e_mp2;
    double e_mp3;
    double e_mp4_sdq;
    double e_mp4_t;      // Triples Energy
    double e_mp4_total;
    double e_corr_total;
    double e_total;
    
    int n_occ_alpha, n_occ_beta;
    int n_virt_alpha, n_virt_beta;
    
    Eigen::Tensor<double, 2> t1_alpha_3;
    Eigen::Tensor<double, 2> t1_beta_3;
    Eigen::Tensor<double, 4> t2_aa_3;
    Eigen::Tensor<double, 4> t2_bb_3;
    Eigen::Tensor<double, 4> t2_ab_3;
    
    // For validation
    Eigen::Tensor<double, 4> t2_aa_2; 
    Eigen::Tensor<double, 4> t2_bb_2;
    Eigen::Tensor<double, 4> t2_ab_2;
};

class UMP4 {
public:
    UMP4(const SCFResult& uhf_result,
         const UMP3Result& ump3_result,
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals);
    
    UMP4Result compute(bool include_triples = true);
    
    auto get_t1_amplitudes() const -> std::pair<const Eigen::Tensor<double, 2>&, const Eigen::Tensor<double, 2>&>;
    auto get_t2_amplitudes() const -> std::tuple<const Eigen::Tensor<double, 4>&, const Eigen::Tensor<double, 4>&, const Eigen::Tensor<double, 4>&>;

private:
    const SCFResult& uhf_;
    const UMP3Result& ump3_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    
    int nbf_;
    int nocc_a_, nocc_b_;
    int nvirt_a_, nvirt_b_;
    
    // Integrals for SDQ (Occ-Occ-Virt-Virt)
    Eigen::Tensor<double, 4> eri_ooov_aa_;
    Eigen::Tensor<double, 4> eri_ooov_bb_;
    Eigen::Tensor<double, 4> eri_ooov_ab_;
    
    // Integrals for Triples (Virt-Virt-Virt-Occ) [NEW]
    Eigen::Tensor<double, 4> eri_vvvo_aa_; // <ab|ci>
    Eigen::Tensor<double, 4> eri_vvvo_bb_; // <AB|CI>
    Eigen::Tensor<double, 4> eri_vvvo_ab_; // <aB|cI> mixed
    Eigen::Tensor<double, 4> eri_vvvo_ba_; // <Ab|Ci> mixed
    
    Eigen::MatrixXd fock_mo_a_;
    Eigen::MatrixXd fock_mo_b_;
    
    Eigen::Tensor<double, 2> t1_a_3_, t1_b_3_;
    Eigen::Tensor<double, 4> t2_aa_3_, t2_bb_3_, t2_ab_3_;
    
    void build_fock_mo();
    void transform_integrals_to_mo();
    void transform_triples_integrals(); // [NEW]
    
    void compute_t1_third_order();
    void compute_t2_third_order();
    
    double compute_singles_energy();
    double compute_doubles_energy();
    double compute_quadruples_energy();
    double compute_triples_energy(); // [UPDATED]
};

} // namespace mp
} // namespace mshqc

<<<<<<< HEAD
#endif
=======
#endif
>>>>>>> 9767215 (update saya)
