#ifndef MSHQC_UMP2_H
#define MSHQC_UMP2_H

#include "mshqc/scf.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <memory> // [Wajib] Untuk std::shared_ptr
#ifdef I
#undef I
#endif

/**
 * @file ump2.h
 * @brief Unrestricted MP2 for open-shell systems
 */

namespace mshqc {

// [Wajib] Forward declaration agar UMP2 mengenali tipe PointGroup
// tanpa harus include full header di sini (menghemat waktu kompilasi)
class PointGroup;

/**
 * UMP2 result structure
 */
struct UMP2Result {
    double e_corr_ss_aa;  // Same-spin αα
    double e_corr_ss_bb;  // Same-spin ββ
    double e_corr_os;     // Opposite-spin αβ
    double e_corr_total;  // Total correlation
    double e_total;       // UHF + correlation
};

/**
 * T2 amplitude tensors for wavefunction analysis
 */
struct T2Amplitudes {
    Eigen::Tensor<double, 4> t2_aa;  // αα amplitudes
    Eigen::Tensor<double, 4> t2_bb;  // ββ amplitudes
    Eigen::Tensor<double, 4> t2_ab;  // αβ amplitudes
};

/**
 * Unrestricted Møller-Plesset 2nd order
 */
class UMP2 {
public:
    /**
     * Constructor
     * @param uhf_result UHF SCF result (must contain C_alpha, C_beta, eps_alpha, eps_beta)
     * @param basis Basis set
     * @param integrals Integral engine
     * @param pg Point Group symmetry object (Optional/Shared Pointer)
     */
    UMP2(const SCFResult& uhf_result,
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals,
         std::shared_ptr<PointGroup> pg = nullptr); // [Update] Tambah parameter PG
    
    /**
     * Compute UMP2 energy
     */
    UMP2Result compute();
    
    /**
     * Get T2 amplitudes for wavefunction analysis
     * Must be called after compute()
     */
    T2Amplitudes get_t2_amplitudes() const;
    
private:
    const SCFResult& uhf_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    std::shared_ptr<PointGroup> pg_; // [Update] Tambah member PG
    
    // Dimensions
    int nbf_;       // # basis functions
    int nocc_a_;    // # α occupied
    int nocc_b_;    // # β occupied  
    int nvir_a_;    // # α virtual
    int nvir_b_;    // # β virtual
    
    // MO integrals (ijab notation: i,j=occ, a,b=virt)
    Eigen::Tensor<double, 4> eri_aaaa_;  // <ij|ab>^αα (antisym)
    Eigen::Tensor<double, 4> eri_bbbb_;  // <IJ|AB>^ββ (antisym)
    Eigen::Tensor<double, 4> eri_aabb_;  // <iJ|aB>^αβ (no antisym)
    
    // T2 amplitudes (stored after compute())
    Eigen::Tensor<double, 4> t2_aa_;  // t_ij^ab (αα)
    Eigen::Tensor<double, 4> t2_bb_;  // t_IJ^AB (ββ)
    Eigen::Tensor<double, 4> t2_ab_;  // t_iJ^aB (αβ)

    // Symmetry Irreps (Added for optimization)
    std::vector<int> irreps_occ_a_;
    std::vector<int> irreps_vir_a_;
    std::vector<int> irreps_occ_b_;
    std::vector<int> irreps_vir_b_;
    
    void transform_integrals();
    double compute_ss_alpha();
    double compute_ss_beta();
    double compute_os();
};

} // namespace mshqc

#endif // MSHQC_UMP2_H