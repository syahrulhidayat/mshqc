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

#include "mshqc/integrals/eri_transformer.h"
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mshqc {
namespace integrals {

// ============================================================================
// SAME-SPIN OOVV TRANSFORMATION (Naive O(N^8) algorithm)
// ============================================================================

Eigen::Tensor<double, 4> ERITransformer::transform_oovv(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ,
    const Eigen::MatrixXd& C_virt,
    int nbf,
    int nocc,
    int nvirt
) {
    Eigen::Tensor<double, 4> eri_mo(nocc, nocc, nvirt, nvirt);
    eri_mo.setZero();
    
    // FORMULA: (ij|ab) = Σ_μνλσ C_μi C_νa (μν|λσ) C_λj C_σb
    // Physicist notation: <ij|ab> = (ia|jb) in chemist notation
    
    for (int i = 0; i < nocc; i++) {
        for (int j = 0; j < nocc; j++) {
            for (int a = 0; a < nvirt; a++) {
                for (int b = 0; b < nvirt; b++) {
                    double val = 0.0;
                    
                    for (int mu = 0; mu < nbf; mu++) {
                        for (int nu = 0; nu < nbf; nu++) {
                            for (int lam = 0; lam < nbf; lam++) {
                                for (int sig = 0; sig < nbf; sig++) {
                                    // Transform: μ→i, ν→a, λ→j, σ→b
                                    val += C_occ(mu, i) * C_virt(nu, a) *
                                           eri_ao(mu, nu, lam, sig) *
                                           C_occ(lam, j) * C_virt(sig, b);
                                }
                            }
                        }
                    }
                    
                    eri_mo(i, j, a, b) = val;
                }
            }
        }
    }
    
    return eri_mo;
}

// ============================================================================
// CRITICAL FIX: MIXED-SPIN OOVV TRANSFORMATION
// ============================================================================
    Eigen::Tensor<double, 4> ERITransformer::transform_oovv_mixed(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& C_occ_A,     // α occupied
        const Eigen::MatrixXd& C_occ_B,     // β occupied
        const Eigen::MatrixXd& C_virt_A,    // α virtual
        const Eigen::MatrixXd& C_virt_B,    // β virtual
        int nbf,
        int nocc_A,
        int nocc_B,
        int nvirt_A,
        int nvirt_B
    ) {
        // Output: (i,j,a,b) where i=α-occ, j=β-occ, a=α-virt, b=β-virt
        Eigen::Tensor<double, 4> eri_mo(nocc_A, nocc_B, nvirt_A, nvirt_B);
        eri_mo.setZero();
        
        for (int i = 0; i < nocc_A; i++) {      // α occ
            for (int j = 0; j < nocc_B; j++) {  // β occ ← FIX!
                for (int a = 0; a < nvirt_A; a++) {  // α virt
                    for (int b = 0; b < nvirt_B; b++) {  // β virt
                        double val = 0.0;
                        
                        for (int mu = 0; mu < nbf; mu++) {
                            for (int nu = 0; nu < nbf; nu++) {
                                for (int lam = 0; lam < nbf; lam++) {
                                    for (int sig = 0; sig < nbf; sig++) {
                                        // Transform: μ→i(α), ν→j(β), λ→a(α), σ→b(β)
                                        val += C_occ_A(mu, i) * C_occ_B(nu, j) *
                                            eri_ao(mu, nu, lam, sig) *
                                            C_virt_A(lam, a) * C_virt_B(sig, b);
                                    }
                                }
                            }
                        }
                        
                        eri_mo(i, j, a, b) = val;
                    }
                }
            }
        }
        
        return eri_mo;
    }
// ============================================================================
// QUARTER TRANSFORM ALGORITHM (Helgaker Algorithm 9.5)
// ============================================================================

Eigen::Tensor<double, 4> ERITransformer::transform_oovv_quarter(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ,
    const Eigen::MatrixXd& C_virt,
    int nbf,
    int nocc,
    int nvirt
) {
    /**
     * HELGAKER ALGORITHM 9.5 - 4-Step Transformation
     * 
     * Reduces cost from O(N^8) → O(N^5) by factorizing contractions
     * 
     * STEPS:
     *   1. (μν|λσ) → (iν|λσ)   [Contract μ with C_occ → occupied i]
     *   2. (iν|λσ) → (ia|λσ)   [Contract ν with C_virt → virtual a]
     *   3. (ia|λσ) → (ia|jσ)   [Contract λ with C_occ → occupied j]
     *   4. (ia|jσ) → (ia|jb)   [Contract σ with C_virt → virtual b]
     *   5. Rearrange (ia|jb) → (ij|ab)
     */
    
    // ========================================================================
    // STEP 1: (μν|λσ) → (iν|λσ)
    // Contract first index (μ) with C_occ to get occupied index i
    // ========================================================================
    
    Eigen::Tensor<double, 4> temp1(nocc, nbf, nbf, nbf);
    temp1.setZero();
    
    for (int i = 0; i < nocc; i++) {
        for (int nu = 0; nu < nbf; nu++) {
            for (int lam = 0; lam < nbf; lam++) {
                for (int sig = 0; sig < nbf; sig++) {
                    double val = 0.0;
                    for (int mu = 0; mu < nbf; mu++) {
                        val += C_occ(mu, i) * eri_ao(mu, nu, lam, sig);
                    }
                    temp1(i, nu, lam, sig) = val;
                }
            }
        }
    }
    
    // ========================================================================
    // STEP 2: (iν|λσ) → (ia|λσ)
    // Contract second index (ν) with C_virt to get virtual index a
    // ========================================================================
    
    Eigen::Tensor<double, 4> temp2(nocc, nvirt, nbf, nbf);
    temp2.setZero();
    
    for (int i = 0; i < nocc; i++) {
        for (int a = 0; a < nvirt; a++) {
            for (int lam = 0; lam < nbf; lam++) {
                for (int sig = 0; sig < nbf; sig++) {
                    double val = 0.0;
                    for (int nu = 0; nu < nbf; nu++) {
                        val += C_virt(nu, a) * temp1(i, nu, lam, sig);
                    }
                    temp2(i, a, lam, sig) = val;
                }
            }
        }
    }
    
    // ========================================================================
    // STEP 3: (ia|λσ) → (ia|jσ)
    // Contract third index (λ) with C_occ to get occupied index j
    // ========================================================================
    
    Eigen::Tensor<double, 4> temp3(nocc, nvirt, nocc, nbf);
    temp3.setZero();
    
    for (int i = 0; i < nocc; i++) {
        for (int a = 0; a < nvirt; a++) {
            for (int j = 0; j < nocc; j++) {
                for (int sig = 0; sig < nbf; sig++) {
                    double val = 0.0;
                    for (int lam = 0; lam < nbf; lam++) {
                        val += C_occ(lam, j) * temp2(i, a, lam, sig);
                    }
                    temp3(i, a, j, sig) = val;
                }
            }
        }
    }
    
    // ========================================================================
    // STEP 4: (ia|jσ) → (ia|jb)
    // Contract fourth index (σ) with C_virt to get virtual index b
    // ========================================================================
    
    Eigen::Tensor<double, 4> temp4(nocc, nvirt, nocc, nvirt);
    temp4.setZero();
    
    for (int i = 0; i < nocc; i++) {
        for (int a = 0; a < nvirt; a++) {
            for (int j = 0; j < nocc; j++) {
                for (int b = 0; b < nvirt; b++) {
                    double val = 0.0;
                    for (int sig = 0; sig < nbf; sig++) {
                        val += C_virt(sig, b) * temp3(i, a, j, sig);
                    }
                    temp4(i, a, j, b) = val;
                }
            }
        }
    }
    
    // ========================================================================
    // STEP 5: Rearrange (ia|jb) → (ij|ab)
    // Reorder tensor to match physicist notation
    // ========================================================================
    
    Eigen::Tensor<double, 4> eri_mo(nocc, nocc, nvirt, nvirt);
    
    for (int i = 0; i < nocc; i++) {
        for (int j = 0; j < nocc; j++) {
            for (int a = 0; a < nvirt; a++) {
                for (int b = 0; b < nvirt; b++) {
                    eri_mo(i, j, a, b) = temp4(i, a, j, b);
                }
            }
        }
    }
    
    return eri_mo;
}

// ============================================================================
// VVOV TRANSFORMATION (Particle-Particle-Hole)
// ============================================================================

Eigen::Tensor<double, 4> ERITransformer::transform_vvov(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ,
    const Eigen::MatrixXd& C_virt,
    int nbf,
    int nocc,
    int nvirt
) {
    Eigen::Tensor<double, 4> eri_mo(nvirt, nvirt, nocc, nvirt);
    eri_mo.setZero();
    
    // FORMULA: (ab|kc) = Σ_μνλσ C_μa C_νb (μν|λσ) C_λk C_σc
    
    for (int a = 0; a < nvirt; a++) {
        for (int b = 0; b < nvirt; b++) {
            for (int k = 0; k < nocc; k++) {
                for (int c = 0; c < nvirt; c++) {
                    double val = 0.0;
                    
                    for (int mu = 0; mu < nbf; mu++) {
                        for (int nu = 0; nu < nbf; nu++) {
                            for (int lam = 0; lam < nbf; lam++) {
                                for (int sig = 0; sig < nbf; sig++) {
                                    val += C_virt(mu, a) * C_virt(nu, b) *
                                           eri_ao(mu, nu, lam, sig) *
                                           C_occ(lam, k) * C_virt(sig, c);
                                }
                            }
                        }
                    }
                    
                    eri_mo(a, b, k, c) = val;
                }
            }
        }
    }
    
    return eri_mo;
}

// Continued in next part...
// ============================================================================
// OOOO TRANSFORMATION (Hole-Hole)
// ============================================================================

Eigen::Tensor<double, 4> ERITransformer::transform_oooo(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ,
    int nbf,
    int nocc
) {
    Eigen::Tensor<double, 4> eri_mo(nocc, nocc, nocc, nocc);
    eri_mo.setZero();
    
    // FORMULA: (ik|jl) = Σ_μνλσ C_μi C_νk (μν|λσ) C_λj C_σl
    // Used in: W_oooo intermediate for MP3 HH-ladder
    
    for (int i = 0; i < nocc; i++) {
        for (int k = 0; k < nocc; k++) {
            for (int j = 0; j < nocc; j++) {
                for (int l = 0; l < nocc; l++) {
                    double val = 0.0;
                    
                    for (int mu = 0; mu < nbf; mu++) {
                        for (int nu = 0; nu < nbf; nu++) {
                            for (int lam = 0; lam < nbf; lam++) {
                                for (int sig = 0; sig < nbf; sig++) {
                                    val += C_occ(mu, i) * C_occ(nu, k) *
                                           eri_ao(mu, nu, lam, sig) *
                                           C_occ(lam, j) * C_occ(sig, l);
                                }
                            }
                        }
                    }
                    
                    eri_mo(i, k, j, l) = val;
                }
            }
        }
    }
    
    return eri_mo;
}

// ============================================================================
// MIXED-SPIN OOOO TRANSFORMATION (α-occ × β-occ)
// ============================================================================

Eigen::Tensor<double, 4> ERITransformer::transform_oooo_mixed(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ_A,    // α occupied
    const Eigen::MatrixXd& C_occ_B,    // β occupied
    int nbf,
    int nocc_A,
    int nocc_B
) {
    /**
     * FORMULA FOR W_oooo_ab:
     *   (mn|ij)_MO = Σ_μνλσ C^α_μm C^α_νn (μν|λσ)_AO C^β_λi C^β_σj
     * 
     * INDEX CONVENTION:
     *   m, n = α occupied (bra side)
     *   i, j = β occupied (ket side)
     * 
     * QUARTER TRANSFORM for efficiency:
     *   Step 1: (μν|λσ) → (mν|λσ)   [Contract μ with α]
     *   Step 2: (mν|λσ) → (mn|λσ)   [Contract ν with α]
     *   Step 3: (mn|λσ) → (mn|iσ)   [Contract λ with β]
     *   Step 4: (mn|iσ) → (mn|ij)   [Contract σ with β]
     */
    
    // Step 1: (μν|λσ) → (mν|λσ)
    Eigen::Tensor<double, 4> temp1(nocc_A, nbf, nbf, nbf);
    temp1.setZero();
    
    for (int m = 0; m < nocc_A; m++) {
        for (int nu = 0; nu < nbf; nu++) {
            for (int lam = 0; lam < nbf; lam++) {
                for (int sig = 0; sig < nbf; sig++) {
                    double val = 0.0;
                    for (int mu = 0; mu < nbf; mu++) {
                        val += C_occ_A(mu, m) * eri_ao(mu, nu, lam, sig);
                    }
                    temp1(m, nu, lam, sig) = val;
                }
            }
        }
    }
    
    // Step 2: (mν|λσ) → (mn|λσ)
    Eigen::Tensor<double, 4> temp2(nocc_A, nocc_A, nbf, nbf);
    temp2.setZero();
    
    for (int m = 0; m < nocc_A; m++) {
        for (int n = 0; n < nocc_A; n++) {
            for (int lam = 0; lam < nbf; lam++) {
                for (int sig = 0; sig < nbf; sig++) {
                    double val = 0.0;
                    for (int nu = 0; nu < nbf; nu++) {
                        val += C_occ_A(nu, n) * temp1(m, nu, lam, sig);
                    }
                    temp2(m, n, lam, sig) = val;
                }
            }
        }
    }
    
    // Step 3: (mn|λσ) → (mn|iσ)
    Eigen::Tensor<double, 4> temp3(nocc_A, nocc_A, nocc_B, nbf);
    temp3.setZero();
    
    for (int m = 0; m < nocc_A; m++) {
        for (int n = 0; n < nocc_A; n++) {
            for (int i = 0; i < nocc_B; i++) {
                for (int sig = 0; sig < nbf; sig++) {
                    double val = 0.0;
                    for (int lam = 0; lam < nbf; lam++) {
                        val += C_occ_B(lam, i) * temp2(m, n, lam, sig);
                    }
                    temp3(m, n, i, sig) = val;
                }
            }
        }
    }
    
    // Step 4: (mn|iσ) → (mn|ij)
    Eigen::Tensor<double, 4> eri_mo(nocc_A, nocc_A, nocc_B, nocc_B);
    eri_mo.setZero();
    
    for (int m = 0; m < nocc_A; m++) {
        for (int n = 0; n < nocc_A; n++) {
            for (int i = 0; i < nocc_B; i++) {
                for (int j = 0; j < nocc_B; j++) {
                    double val = 0.0;
                    for (int sig = 0; sig < nbf; sig++) {
                        val += C_occ_B(sig, j) * temp3(m, n, i, sig);
                    }
                    eri_mo(m, n, i, j) = val;
                }
            }
        }
    }
    
    return eri_mo;
}

// ============================================================================
// VVVV TRANSFORMATION (Particle-Particle) - CRITICAL FOR MP3!
// ============================================================================

Eigen::Tensor<double, 4> ERITransformer::transform_vvvv(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_virt,
    int nbf,
    int nvirt
) {
    /**
     * FORMULA: (ab|cd) = Σ_μνλσ C_μa C_νb (μν|λσ) C_λc C_σd
     * 
     * CRITICAL for MP3 convergence!
     * Without this term, E(3) diverges (too negative).
     * 
     * QUARTER TRANSFORM for efficiency:
     *   Step 1: (μν|λσ) → (aν|λσ)
     *   Step 2: (aν|λσ) → (ab|λσ)
     *   Step 3: (ab|λσ) → (ab|cσ)
     *   Step 4: (ab|cσ) → (ab|cd)
     * 
     * COST: O(nbf^4 × nvirt^4) - very expensive!
     * STORAGE: O(nvirt^4)
     *   cc-pVDZ (nvirt=9):  6,561 elements ≈ 52 KB
     *   cc-pVTZ (nvirt=28): 614,656 elements ≈ 4.9 MB
     */
    
    std::cout << "  Transforming <ab|cd> (vvvv) - quarter algorithm..." << std::flush;
    
    // Step 1: (μν|λσ) → (aν|λσ)
    Eigen::Tensor<double, 4> temp1(nvirt, nbf, nbf, nbf);
    temp1.setZero();
    
    for (int a = 0; a < nvirt; a++) {
        for (int nu = 0; nu < nbf; nu++) {
            for (int lam = 0; lam < nbf; lam++) {
                for (int sig = 0; sig < nbf; sig++) {
                    double val = 0.0;
                    for (int mu = 0; mu < nbf; mu++) {
                        val += C_virt(mu, a) * eri_ao(mu, nu, lam, sig);
                    }
                    temp1(a, nu, lam, sig) = val;
                }
            }
        }
    }
    
    // Step 2: (aν|λσ) → (ab|λσ)
    Eigen::Tensor<double, 4> temp2(nvirt, nvirt, nbf, nbf);
    temp2.setZero();
    
    for (int a = 0; a < nvirt; a++) {
        for (int b = 0; b < nvirt; b++) {
            for (int lam = 0; lam < nbf; lam++) {
                for (int sig = 0; sig < nbf; sig++) {
                    double val = 0.0;
                    for (int nu = 0; nu < nbf; nu++) {
                        val += C_virt(nu, b) * temp1(a, nu, lam, sig);
                    }
                    temp2(a, b, lam, sig) = val;
                }
            }
        }
    }
    
    // Step 3: (ab|λσ) → (ab|cσ)
    Eigen::Tensor<double, 4> temp3(nvirt, nvirt, nvirt, nbf);
    temp3.setZero();
    
    for (int a = 0; a < nvirt; a++) {
        for (int b = 0; b < nvirt; b++) {
            for (int c = 0; c < nvirt; c++) {
                for (int sig = 0; sig < nbf; sig++) {
                    double val = 0.0;
                    for (int lam = 0; lam < nbf; lam++) {
                        val += C_virt(lam, c) * temp2(a, b, lam, sig);
                    }
                    temp3(a, b, c, sig) = val;
                }
            }
        }
    }
    
    // Step 4: (ab|cσ) → (ab|cd)
    Eigen::Tensor<double, 4> eri_mo(nvirt, nvirt, nvirt, nvirt);
    eri_mo.setZero();
    
    for (int a = 0; a < nvirt; a++) {
        for (int b = 0; b < nvirt; b++) {
            for (int c = 0; c < nvirt; c++) {
                for (int d = 0; d < nvirt; d++) {
                    double val = 0.0;
                    for (int sig = 0; sig < nbf; sig++) {
                        val += C_virt(sig, d) * temp3(a, b, c, sig);
                    }
                    eri_mo(a, b, c, d) = val;
                }
            }
        }
    }
    
    std::cout << " done\n";
    
    // Report storage
    size_t mem_bytes = nvirt * nvirt * nvirt * nvirt * 8;
    double mem_mb = mem_bytes / (1024.0 * 1024.0);
    std::cout << "  VVVV storage: " << mem_mb << " MB (" 
              << nvirt << "^4 elements)\n";
    
    return eri_mo;
}

// ============================================================================
// OPENMP PARALLELIZED OOVV TRANSFORMATION
// ============================================================================

Eigen::Tensor<double, 4> ERITransformer::transform_oovv_parallel(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ,
    const Eigen::MatrixXd& C_virt,
    int nbf,
    int nocc,
    int nvirt,
    int n_threads
) {
#ifdef _OPENMP
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }
    
    Eigen::Tensor<double, 4> eri_mo(nocc, nocc, nvirt, nvirt);
    eri_mo.setZero();
    
    // Parallelize over outer loops (i, j)
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < nocc; i++) {
        for (int j = 0; j < nocc; j++) {
            for (int a = 0; a < nvirt; a++) {
                for (int b = 0; b < nvirt; b++) {
                    double val = 0.0;
                    
                    for (int mu = 0; mu < nbf; mu++) {
                        for (int nu = 0; nu < nbf; nu++) {
                            for (int lam = 0; lam < nbf; lam++) {
                                for (int sig = 0; sig < nbf; sig++) {
                                    val += C_occ(mu, i) * C_virt(nu, a) *
                                           eri_ao(mu, nu, lam, sig) *
                                           C_occ(lam, j) * C_virt(sig, b);
                                }
                            }
                        }
                    }
                    
                    eri_mo(i, j, a, b) = val;
                }
            }
        }
    }
    
    return eri_mo;
#else
    // OpenMP not available, fall back to serial
    std::cerr << "INFO: OpenMP not available, using serial version\n";
    (void)n_threads;
    return transform_oovv(eri_ao, C_occ, C_virt, nbf, nocc, nvirt);
#endif
}

}} // namespace mshqc::integrals