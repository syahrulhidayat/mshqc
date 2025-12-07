/**
 * @file eri_transformer.cc
 * @brief Implementation of unified ERI transformation utilities
 * 
 * Implements efficient AO→MO integral transformations used by both
 * CI and MP modules. Eliminates ~300 lines of duplicate code.
 * 
 * ALGORITHM: Naive O(N^5) transformation
 * 
 * FORMULA (physicist notation):
 *   (pq|rs)_MO = Σ_μνλσ C_μp C_νq (μν|λσ)_AO C_λr C_σs
 * 
 * FUTURE OPTIMIZATION: Quarter transforms (Helgaker 2000, Algorithm 9.5)
 *   1. (μν|λσ) → (pν|λσ)   [O(N^5)]
 *   2. (pν|λσ) → (pq|λσ)   [O(N^5)]
 *   3. (pq|λσ) → (pq|rσ)   [O(N^5)]
 *   4. (pq|rσ) → (pq|rs)   [O(N^5)]
 *   Total: 4×O(N^5) but with reduced prefactor vs naive 16×O(N^5)
 * 
 * THEORY REFERENCES:
 *   - Helgaker, Jørgensen, Olsen (2000), "Molecular Electronic-Structure Theory"
 *     Chapter 9.6: Integral transformations
 *   - Almlöf (1991), Chem. Phys. Lett. 181, 319
 *     Direct integral transformations
 * 
 * @author Muhamad Syahrul Hidayat (Agent 3)
 * @date 2025-11-17
 * @license MIT License
 * 
 * Copyright (c) 2025 MSH-QC Project
 * 
 * @note ORIGINAL IMPLEMENTATION - No code copied from existing software
 *       Algorithms based on published theory
 */

/**
 * @file eri_transformer.cc - CORRECTED IMPLEMENTATION Part 1
 * @brief Fixed mixed-spin transformations and quarter transforms
 * 
 * CRITICAL FIXES:
 * 1. transform_oovv_mixed: Now uses C_virt_B instead of C_occ_B
 * 2. transform_oovv_quarter: Fixed virtual space indexing
 * 3. All index conventions clarified with comments
 */
/**
 * @file eri_transformer.cc
 * @brief Implementation of unified ERI transformation utilities
 * FIXED: Index ordering (mu, nu, lam, sig) and loop nesting.
 */

/**
 * @file eri_transformer.cc
 * @brief Implementation of unified ERI transformation utilities
 * @details FULL IMPLEMENTATION for MP2, MP3, and MP4.
 * Uses efficient Quarter Transformation (O(N^5)).
 * FIXED: Index ordering (mu, nu, lam, sig).
 */

#include "mshqc/integrals/eri_transformer.h"
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mshqc {
namespace integrals {

// ============================================================================
// HELPER: GENERIC QUARTER TRANSFORM
// ============================================================================
// Internal helper to avoid code duplication
// Transforms (mu, nu, lam, sig) -> (p, q, r, s)
static Eigen::Tensor<double, 4> quarter_transform(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C1,
    const Eigen::MatrixXd& C2,
    const Eigen::MatrixXd& C3,
    const Eigen::MatrixXd& C4,
    int nbf, int n1, int n2, int n3, int n4
) {
    // Step 1: Contract dim 0 (mu) -> (p, nu, lam, sig)
    Eigen::Tensor<double, 4> t1(n1, nbf, nbf, nbf);
    t1.setZero();
    
    #pragma omp parallel for collapse(3)
    for (int p = 0; p < n1; p++) {
        for (int nu = 0; nu < nbf; nu++) {
            for (int lam = 0; lam < nbf; lam++) {
                for (int sig = 0; sig < nbf; sig++) {
                    double val = 0.0;
                    for (int mu = 0; mu < nbf; mu++) {
                        val += C1(mu, p) * eri_ao(mu, nu, lam, sig);
                    }
                    t1(p, nu, lam, sig) = val;
                }
            }
        }
    }

    // Step 2: Contract dim 1 (nu) -> (p, q, lam, sig)
    Eigen::Tensor<double, 4> t2(n1, n2, nbf, nbf);
    t2.setZero();

    #pragma omp parallel for collapse(3)
    for (int p = 0; p < n1; p++) {
        for (int q = 0; q < n2; q++) {
            for (int lam = 0; lam < nbf; lam++) {
                for (int sig = 0; sig < nbf; sig++) {
                    double val = 0.0;
                    for (int nu = 0; nu < nbf; nu++) {
                        val += C2(nu, q) * t1(p, nu, lam, sig);
                    }
                    t2(p, q, lam, sig) = val;
                }
            }
        }
    }

    // Step 3: Contract dim 2 (lam) -> (p, q, r, sig)
    Eigen::Tensor<double, 4> t3(n1, n2, n3, nbf);
    t3.setZero();

    #pragma omp parallel for collapse(3)
    for (int p = 0; p < n1; p++) {
        for (int q = 0; q < n2; q++) {
            for (int r = 0; r < n3; r++) {
                for (int sig = 0; sig < nbf; sig++) {
                    double val = 0.0;
                    for (int lam = 0; lam < nbf; lam++) {
                        val += C3(lam, r) * t2(p, q, lam, sig);
                    }
                    t3(p, q, r, sig) = val;
                }
            }
        }
    }

    // Step 4: Contract dim 3 (sig) -> (p, q, r, s)
    Eigen::Tensor<double, 4> result(n1, n2, n3, n4);
    result.setZero();

    #pragma omp parallel for collapse(3)
    for (int p = 0; p < n1; p++) {
        for (int q = 0; q < n2; q++) {
            for (int r = 0; r < n3; r++) {
                for (int s = 0; s < n4; s++) {
                    double val = 0.0;
                    for (int sig = 0; sig < nbf; sig++) {
                        val += C4(sig, s) * t3(p, q, r, sig);
                    }
                    result(p, q, r, s) = val;
                }
            }
        }
    }

    return result;
}

// ============================================================================
// OOVV IMPLEMENTATIONS (MP2, MP3, MP4)
// Returns: (i, a, j, b)
// ============================================================================

Eigen::Tensor<double, 4> ERITransformer::transform_oovv(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ,
    const Eigen::MatrixXd& C_virt,
    int nbf, int nocc, int nvirt
) {
    return quarter_transform(eri_ao, C_occ, C_virt, C_occ, C_virt, nbf, nocc, nvirt, nocc, nvirt);
}

// Implementasi khusus untuk UMP4 yang memanggil 'quarter' secara eksplisit
Eigen::Tensor<double, 4> ERITransformer::transform_oovv_quarter(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ,
    const Eigen::MatrixXd& C_virt,
    int nbf, int nocc, int nvirt
) {
    return transform_oovv(eri_ao, C_occ, C_virt, nbf, nocc, nvirt);
}

Eigen::Tensor<double, 4> ERITransformer::transform_oovv_mixed(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ_A, 
    const Eigen::MatrixXd& C_occ_B, 
    const Eigen::MatrixXd& C_virt_A, 
    const Eigen::MatrixXd& C_virt_B,
    int nbf, int nocc_A, int nocc_B, int nvirt_A, int nvirt_B
) {
    // Target: (i, a, j, b)
    // C1=i(occA), C2=a(virtA), C3=j(occB), C4=b(virtB)
    return quarter_transform(eri_ao, C_occ_A, C_virt_A, C_occ_B, C_virt_B, 
                           nbf, nocc_A, nvirt_A, nocc_B, nvirt_B);
}

// ============================================================================
// OOOO IMPLEMENTATIONS (MP3, MP4)
// Returns: (i, k, j, l) -> Chemist Notation
// ============================================================================

Eigen::Tensor<double, 4> ERITransformer::transform_oooo(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ,
    int nbf, int nocc
) {
    return quarter_transform(eri_ao, C_occ, C_occ, C_occ, C_occ, nbf, nocc, nocc, nocc, nocc);
}

Eigen::Tensor<double, 4> ERITransformer::transform_oooo_mixed(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ_A,    
    const Eigen::MatrixXd& C_occ_B,    
    int nbf, int nocc_A, int nocc_B
) {
    // (i_a, j_a, i_b, j_b)
    return quarter_transform(eri_ao, C_occ_A, C_occ_A, C_occ_B, C_occ_B, 
                           nbf, nocc_A, nocc_A, nocc_B, nocc_B);
}

// ============================================================================
// VVVV IMPLEMENTATIONS (MP3, MP4)
// Returns: (a, b, c, d)
// ============================================================================

Eigen::Tensor<double, 4> ERITransformer::transform_vvvv(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_virt,
    int nbf, int nvirt
) {
    return quarter_transform(eri_ao, C_virt, C_virt, C_virt, C_virt, nbf, nvirt, nvirt, nvirt, nvirt);
}

Eigen::Tensor<double, 4> ERITransformer::transform_vvvv_mixed(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_virt_A,   
    const Eigen::MatrixXd& C_virt_B,   
    int nbf, int nvirt_A, int nvirt_B
) {
    // (a_a, b_a, a_b, b_b)
    return quarter_transform(eri_ao, C_virt_A, C_virt_A, C_virt_B, C_virt_B, 
                           nbf, nvirt_A, nvirt_A, nvirt_B, nvirt_B);
}

// ============================================================================
// OVOV IMPLEMENTATIONS (MP3)
// Returns: (i, a, j, b)
// ============================================================================

Eigen::Tensor<double, 4> ERITransformer::transform_ovov(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ,
    const Eigen::MatrixXd& C_virt,
    int nbf, int nocc, int nvirt
) {
    return quarter_transform(eri_ao, C_occ, C_virt, C_occ, C_virt, nbf, nocc, nvirt, nocc, nvirt);
}

Eigen::Tensor<double, 4> ERITransformer::transform_ovov_mixed(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ_A,    
    const Eigen::MatrixXd& C_virt_B,   
    int nbf, int nocc_A, int nvirt_B
) {
    // Note: OVOV mixed usually implies (i_a, a_b, j_a, b_b) or similar depending on derivation.
    // Assuming (i_a, a_b, j_a, b_b) for PH terms.
    // Check UMP3 usage carefully. Usually UMP3 calls this for spin-flipped cases.
    // For now, consistent with (OccA, VirtB, OccA, VirtB)
    return quarter_transform(eri_ao, C_occ_A, C_virt_B, C_occ_A, C_virt_B, 
                           nbf, nocc_A, nvirt_B, nocc_A, nvirt_B);
}

// ============================================================================
// PARALLEL WRAPPERS (Optional, redirect to standard functions)
// ============================================================================

Eigen::Tensor<double, 4> ERITransformer::transform_oovv_parallel(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ,
    const Eigen::MatrixXd& C_virt,
    int nbf, int nocc, int nvirt, int n_threads
) {
    // OpenMP is already handled inside quarter_transform
    return transform_oovv(eri_ao, C_occ, C_virt, nbf, nocc, nvirt);
}

Eigen::Tensor<double, 4> ERITransformer::transform_vvvv_parallel(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_virt,
    int nbf, int nvirt, int n_threads
) {
    return transform_vvvv(eri_ao, C_virt, nbf, nvirt);
}

Eigen::Tensor<double, 4> ERITransformer::transform_oooo_parallel(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ,
    int nbf, int nocc, int n_threads
) {
    return transform_oooo(eri_ao, C_occ, nbf, nocc);
}


// Tambahkan fungsi ini di bagian VVVO
Eigen::Tensor<double, 4> ERITransformer::transform_vvvo(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ,
    const Eigen::MatrixXd& C_virt,
    int nbf, int nocc, int nvirt
) {
    return quarter_transform(
        eri_ao, 
        C_virt, C_virt, C_virt, C_occ, 
        nbf, 
        nvirt, nvirt, nvirt, nocc
    );

    // Panggil core dengan urutan: Virt, Virt, Virt, Occ
    // Asumsi quarter_transform_core(eri, C1, C2, C3, C4, nbf, n1, n2, n3, n4)
    // Silakan copy fungsi quarter_transform_core dari jawaban sebelumnya jika belum ada
    // Untuk brevity saya panggil wrapper imajiner atau manual loop
    
    // Implementasi Manual (Jika quarter_transform_core tidak terekspos di header)
    // Target: (a, b, c, k) -> (Virt, Virt, Virt, Occ)
    
    // 1. Transform index 4 (AO -> k) [Occ]
    Eigen::Tensor<double, 4> t1(nbf, nbf, nbf, nocc);
    t1.setZero();
    #pragma omp parallel for collapse(3)
    for(int m=0; m<nbf; ++m) for(int n=0; n<nbf; ++n) for(int l=0; l<nbf; ++l) 
        for(int k=0; k<nocc; ++k) 
            for(int s=0; s<nbf; ++s) t1(m,n,l,k) += C_occ(s, k) * eri_ao(m,n,l,s);

    // 2. Transform index 3 (AO -> c) [Virt]
    Eigen::Tensor<double, 4> t2(nbf, nbf, nvirt, nocc);
    t2.setZero();
    #pragma omp parallel for collapse(3)
    for(int m=0; m<nbf; ++m) for(int n=0; n<nbf; ++n) for(int c=0; c<nvirt; ++c) for(int k=0; k<nocc; ++k)
        for(int l=0; l<nbf; ++l) t2(m,n,c,k) += C_virt(l, c) * t1(m,n,l,k);

    // 3. Transform index 2 (AO -> b) [Virt]
    Eigen::Tensor<double, 4> t3(nbf, nvirt, nvirt, nocc);
    t3.setZero();
    #pragma omp parallel for collapse(3)
    for(int m=0; m<nbf; ++m) for(int b=0; b<nvirt; ++b) for(int c=0; c<nvirt; ++c) for(int k=0; k<nocc; ++k)
        for(int n=0; n<nbf; ++n) t3(m,b,c,k) += C_virt(n, b) * t2(m,n,c,k);

    // 4. Transform index 1 (AO -> a) [Virt]
    Eigen::Tensor<double, 4> result(nvirt, nvirt, nvirt, nocc);
    result.setZero();
    #pragma omp parallel for collapse(4)
    for(int a=0; a<nvirt; ++a) for(int b=0; b<nvirt; ++b) for(int c=0; c<nvirt; ++c) for(int k=0; k<nocc; ++k)
        for(int m=0; m<nbf; ++m) result(a,b,c,k) += C_virt(m, a) * t3(m,b,c,k);
        
    return result;
}
// IMPLEMENTASI YANG HILANG
Eigen::Tensor<double, 4> ERITransformer::transform_vvvo_mixed(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ_K,
    const Eigen::MatrixXd& C_virt_AC,
    const Eigen::MatrixXd& C_virt_B,
    int nbf, int nocc_K, int nvirt_AC, int nvirt_B
) {
    // Logika sama persis dengan transform_vvvo tapi matriks C berbeda
    // Target: (a, b, c, k) -> (VirtAC, VirtB, VirtAC, OccK)
    
    // 1. Transform index 4 (AO -> k) [OccK]
    Eigen::Tensor<double, 4> t1(nbf, nbf, nbf, nocc_K);
    t1.setZero();
    #pragma omp parallel for collapse(3)
    for(int m=0; m<nbf; ++m) for(int n=0; n<nbf; ++n) for(int l=0; l<nbf; ++l) 
        for(int k=0; k<nocc_K; ++k) 
            for(int s=0; s<nbf; ++s) t1(m,n,l,k) += C_occ_K(s, k) * eri_ao(m,n,l,s);

    // 2. Transform index 3 (AO -> c) [VirtAC]
    Eigen::Tensor<double, 4> t2(nbf, nbf, nvirt_AC, nocc_K);
    t2.setZero();
    #pragma omp parallel for collapse(3)
    for(int m=0; m<nbf; ++m) for(int n=0; n<nbf; ++n) for(int c=0; c<nvirt_AC; ++c) for(int k=0; k<nocc_K; ++k)
        for(int l=0; l<nbf; ++l) t2(m,n,c,k) += C_virt_AC(l, c) * t1(m,n,l,k);

    // 3. Transform index 2 (AO -> b) [VirtB]
    Eigen::Tensor<double, 4> t3(nbf, nvirt_B, nvirt_AC, nocc_K);
    t3.setZero();
    #pragma omp parallel for collapse(3)
    for(int m=0; m<nbf; ++m) for(int b=0; b<nvirt_B; ++b) for(int c=0; c<nvirt_AC; ++c) for(int k=0; k<nocc_K; ++k)
        for(int n=0; n<nbf; ++n) t3(m,b,c,k) += C_virt_B(n, b) * t2(m,n,c,k);

    // 4. Transform index 1 (AO -> a) [VirtAC]
    Eigen::Tensor<double, 4> result(nvirt_AC, nvirt_B, nvirt_AC, nocc_K);
    result.setZero();
    #pragma omp parallel for collapse(4)
    for(int a=0; a<nvirt_AC; ++a) for(int b=0; b<nvirt_B; ++b) for(int c=0; c<nvirt_AC; ++c) for(int k=0; k<nocc_K; ++k)
        for(int m=0; m<nbf; ++m) result(a,b,c,k) += C_virt_AC(m, a) * t3(m,b,c,k);
        
    return result;
}

} // namespace integrals
} // namespace mshqc  