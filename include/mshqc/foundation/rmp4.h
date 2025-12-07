/**
 * @file rmp4.h
 * @brief Restricted Møller-Plesset 4th-order perturbation theory (MP4(SDQ))
 * * Implementation of MP4 energy correction focusing on Singles, Doubles, 
 * and Quadruples (SDQ) contributions.
 * * THEORY REFERENCES:
 * - R. Krishnan & J. A. Pople, Int. J. Quantum Chem. 14, 91 (1978)
 * [Approximate Fourth-Order Perturbation Theory]
 * - A. Szabo & N. S. Ostlund, "Modern Quantum Chemistry" (1996)
 * * FORMULA:
 * E(MP4) = E(MP3) + E_S(4) + E_D(4) + E_Q(4)
 * * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 * @license MIT
 */

#/**
 * @file rmp4.h
 * @brief Restricted MP4 (Full SDTQ) Header - CORRECTED
 * @details Supports explicit Singles, Doubles, Triples, and Quadruples calculation.
 */

#ifndef MSHQC_FOUNDATION_RMP4_H
#define MSHQC_FOUNDATION_RMP4_H

#include "mshqc/foundation/rmp3.h"
#include "mshqc/scf.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

namespace mshqc {
namespace foundation {

/**
 * @brief Container for RMP4 results
 */
struct RMP4Result {
    double e_rhf;
    double e_mp2;
    double e_mp3;
    
    // Components (Previously missing members added here)
    double e_s;       // Singles
    double e_d;       // Doubles
    double e_q;       // Quadruples
    double e_t;       // Triples (New!)

    // Totals
    double e_mp4_sdq;    // S + D + Q
    double e_mp4_total;  // S + D + Q + T (New!)
    double e_corr_total; // MP2 + MP3 + MP4 (New!)
    double e_total;      // Total Electronic Energy
};

/**
 * @brief Restricted MP4 Solver (Full SDTQ)
 */
class RMP4 {
public:
    RMP4(const SCFResult& rhf,
         const RMP2Result& mp2,
         const RMP3Result& mp3,
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals);

    /**
     * @brief Compute RMP4 energy
     * @param include_triples If true, compute (T) term (Expensive O(N^7))
     */
    RMP4Result compute(bool include_triples = true);

private:
    const SCFResult& rhf_;
    const RMP2Result& mp2_;
    const RMP3Result& mp3_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;

    int nocc_;
    int nvirt_;
    int nbf_;

    // Full MO Integrals (Chemist Notation)
    Eigen::Tensor<double, 4> eri_mo_; 
    
    // Specialized Integral for Triples <ab|ck>
    Eigen::Tensor<double, 4> eri_vvvo_;

    // Helper functions
    void ensure_mo_integrals();
    
    // Ini yang sebelumnya hilang dan menyebabkan error:
    void transform_triples_integrals(); 

    double compute_singles_energy();    
    double compute_doubles_energy();    
    double compute_quadruples_energy(); 
    
    // Ini juga hilang sebelumnya:
    double compute_triples_energy();    
};

} // namespace foundation
} // namespace mshqc

#endif // MSHQC_FOUNDATION_RMP4_H