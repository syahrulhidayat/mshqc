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
/**
/**
 * @file ump4.h
 * @brief Unrestricted MP4 Header (Corrected)
 * @details Ensures UMP4Result struct has all members used in implementation.
 */

/**
 * @file ump4.h
 * @brief Unrestricted MP4 (Full SDTQ) Header
 * @details Supports explicit Singles, Doubles, Triples, and Quadruples calculation.
 */

#ifndef MSHQC_MP_UMP4_H
#define MSHQC_MP_UMP4_H

#include "mshqc/ump3.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

namespace mshqc {
namespace mp {

struct UMP4Result {
    double e_uhf, e_mp2, e_mp3;
    double e_s, e_d, e_q, e_t;
    double e_mp4_sdq, e_mp4_total;
    double e_corr_total, e_total;
};

class UMP4 {
public:
    UMP4(const SCFResult& uhf_result,
         const UMP3Result& ump3_result,
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals);

    UMP4Result compute(bool include_triples = true);

private:
    const SCFResult& uhf_;
    const UMP3Result& ump3_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;

    int nbf_;
    int nocc_a_, nocc_b_;
    int nvirt_a_, nvirt_b_;

    // Integrals
    Eigen::Tensor<double, 4> eri_oovv_aa_;
    Eigen::Tensor<double, 4> eri_oovv_bb_; 
    Eigen::Tensor<double, 4> eri_oovv_ab_;
    
    // Integrals KHUSUS untuk Triples (Baru)
    Eigen::Tensor<double, 4> eri_vvvo_aab_; // <eb|ck>
    Eigen::Tensor<double, 4> eri_vooo_aab_; // <mc|jk> (INI YANG ANDA CARI)

    // Orbital Energies
    Eigen::VectorXd eps_a_, eps_b_;

    // Helper functions
    void setup_integrals();
    
    // DEKLARASI FUNGSI BARU (Wajib ada!)
    void compute_vooo_integrals(const Eigen::Tensor<double, 4>& eri_ao); 

    double energy_singles();    
    double energy_doubles();    
    double energy_quadruples(); 
    double energy_triples();    
};

} // namespace mp
} // namespace mshqc

#endif // MSHQC_MP_UMP4_H