/**
 * @file ump3.h
 * @brief Unrestricted Møller-Plesset 3rd-order perturbation theory (UMP3)
 * 
 * CORRECTED IMPLEMENTATION - Opsi A (Full Fix)
 * 
 * Theory References:
 *   - Pople et al., Int. J. Quantum Chem. 11, 149 (1977)
 *   - Bartlett & Silver, Phys. Rev. A 10, 1927 (1974)
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-02-01 (CORRECTED v6)
 * @license MIT License
 */



#ifndef MSHQC_UMP3_H
#define MSHQC_UMP3_H

#include "mshqc/scf.h"
#include "mshqc/ump2.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

namespace mshqc {

struct UMP3Result {
    double e_uhf;
    double e_mp2;
    double e_mp3;      // E(3)
    double e_corr_total;    // E(2) + E(3)
    double e_total;
    
    double e3_aa;
    double e3_bb;
    double e3_ab;
    
    // === TAMBAHKAN DIMENSI (Dibutuhkan UMP4) ===
    int n_occ_alpha;
    int n_occ_beta;
    int n_virt_alpha;
    int n_virt_beta;
    // ===========================================

    // === TAMBAHAN PENTING UNTUK mp_density.cc ===
    // Density matrix analyzer membutuhkan T1 (Singles). 
    // Meskipun di Canonical MP3 nilainya nol, struct harus menyediakannya.
    Eigen::MatrixXd t1_a_2;
    Eigen::MatrixXd t1_b_2;
    // ============================================

    // T2 Amplitudo Orde-1 (dari UMP2, tapi diteruskan lewat sini untuk kemudahan)
    Eigen::Tensor<double, 4> t2_aa_1;
    Eigen::Tensor<double, 4> t2_bb_1;
    Eigen::Tensor<double, 4> t2_ab_1;

    // T2 Amplitudo Orde-2 (Dihitung oleh UMP3)
    Eigen::Tensor<double, 4> t2_aa_2;
    Eigen::Tensor<double, 4> t2_bb_2;
    Eigen::Tensor<double, 4> t2_ab_2;

    // Flag untuk T3 (karena UMP3 Anda belum menghitung T3, set false dulu)
    bool t3_2_computed = false;
    // Placeholder jika nanti implementasi T3 ditambahkan
    Eigen::Tensor<double, 6> t3_aaa_2; 
    Eigen::Tensor<double, 6> t3_bbb_2; 
    Eigen::Tensor<double, 6> t3_aab_2; 
    Eigen::Tensor<double, 6> t3_abb_2;
};

class UMP3 {
public:
    UMP3(const SCFResult& uhf_result,
         const UMP2Result& ump2_result,
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals);
    
    UMP3Result compute();

private:
    // Input
    const SCFResult& uhf_;
    const UMP2Result& ump2_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    
    // Dimensions
    int nbf_;
    int nocc_a_, nocc_b_;
    int nvir_a_, nvir_b_;
    
    // T2 amplitudes
    Eigen::Tensor<double, 4> t2_aa_1_;  // from MP2
    Eigen::Tensor<double, 4> t2_bb_1_;
    Eigen::Tensor<double, 4> t2_ab_1_;
    
    Eigen::Tensor<double, 4> t2_aa_2_;  // computed
    Eigen::Tensor<double, 4> t2_bb_2_;
    Eigen::Tensor<double, 4> t2_ab_2_;
    
    // MO integrals - ALL blocks needed
    Eigen::Tensor<double, 4> g_oovv_aa_;  // <ij||ab>
    Eigen::Tensor<double, 4> g_oovv_bb_;
    Eigen::Tensor<double, 4> g_oovv_ab_;  // <ij|ab> no antisym
    
    Eigen::Tensor<double, 4> g_oooo_aa_;  // <ij||kl>
    Eigen::Tensor<double, 4> g_oooo_bb_;
    Eigen::Tensor<double, 4> g_oooo_ab_;  // <ij|kl>
    
    Eigen::Tensor<double, 4> g_vvvv_aa_;  // <ab||cd>
    Eigen::Tensor<double, 4> g_vvvv_bb_;
    Eigen::Tensor<double, 4> g_vvvv_ab_;  // <ab|cd>
    
    Eigen::Tensor<double, 4> g_ovov_aa_;  // <ia||jb>
    Eigen::Tensor<double, 4> g_ovov_bb_;
    Eigen::Tensor<double, 4> g_ovov_ab_;  // <ia|jb>
    Eigen::Tensor<double, 4> g_ovov_ba_;  // <ia|jb> (beta-alpha)
    
    // Core methods
    void transform_all_integrals();
    void get_t2_from_ump2();
    void compute_t2_second_order();
    double compute_e3_energy();
    
    // Integral transformations
    void transform_oovv();
    void transform_oooo();
    void transform_vvvv();
    void transform_ovov();
    
    // T2^(2) by spin case
    void compute_t2_aa_second();
    void compute_t2_bb_second();
    void compute_t2_ab_second();
    
    // Energy by spin case
    double compute_e3_aa();
    double compute_e3_bb();
    double compute_e3_ab();
};

} // namespace mshqc

#endif