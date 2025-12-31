/**
 * @file ump4.h
 * @brief Unrestricted Møller-Plesset 4th-order (UMP4) - RIGOROUS IMPLEMENTATION
 * 
 * ============================================================================
 * THEORY REFERENCES (PRIMARY SOURCES):
 * ============================================================================
 * [1] K. Raghavachari, G. W. Trucks, J. A. Pople, & M. Head-Gordon,
 *     Chem. Phys. Lett. 157, 479 (1989)
 *     "A fifth-order perturbation comparison of electron correlation theories"
 *     [THE definitive MP4(SDTQ) reference with explicit formulas]
 * 
 * [2] J. A. Pople, R. Krishnan, H. B. Schlegel, & J. S. Binkley,
 *     Int. J. Quantum Chem. 14, 545 (1978)
 *     "Derivative studies in Hartree-Fock and Møller-Plesset theories"
 *     [Original MP4 formulation]
 * 
 * [3] R. J. Bartlett & D. M. Silver, Phys. Rev. A 10, 1927 (1974)
 *     "Many-body perturbation theory applied to electron pair correlation energies"
 *     [Diagrammatic derivation of MP terms]
 * 
 * [4] T. Helgaker, P. Jørgensen, & J. Olsen,
 *     "Molecular Electronic-Structure Theory" (Wiley, 2000)
 *     Section 14.4: Møller-Plesset Perturbation Theory
 *     Equations (14.66)-(14.74): Explicit MP4 energy expressions
 */
 

/**
 * @file ump4.h
 * @brief Unrestricted Møller-Plesset 4th-order (UMP4) - RIGOROUS IMPLEMENTATION
 * * THEORY: Raghavachari et al., Chem. Phys. Lett. 157, 479 (1989)
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
    UMP4(const SCFResult& uhf, const UMP3Result& ump3,
         const BasisSet& basis, std::shared_ptr<IntegralEngine> integrals);
    
    UMP4Result compute(bool include_triples = true);

private:
    const SCFResult& uhf_;
    const UMP3Result& ump3_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    
    int nbf_, nocc_a_, nocc_b_, nvirt_a_, nvirt_b_;
    Eigen::VectorXd eps_a_, eps_b_;
    
    // Integrals storage
    Eigen::Tensor<double, 4> g_oovv_aa_, g_oovv_bb_, g_oovv_ab_;
    Eigen::Tensor<double, 4> g_vvvo_aa_, g_vvvo_bb_, g_vvvo_ab_, g_vvvo_ba_;
    Eigen::Tensor<double, 4> g_vooo_aa_, g_vooo_bb_, g_vooo_ab_, g_vooo_ba_;
    
    void transform_integrals();
    
    // Core computation modules
    double compute_singles(); // FIXED: Added missing declaration
    double compute_doubles();
    double compute_quadruples();
    double compute_triples();
    
    void print_results(const UMP4Result& r);
};

} // namespace mp
} // namespace mshqc

#endif