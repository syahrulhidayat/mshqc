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
 * @author Muhamad Syahrul Hidayat ()
 * @date 2025-11-17
 * @license MIT License
 * 
 * Copyright (c) 2025 MSH-QC Project
 * 
 * @note ORIGINAL IMPLEMENTATION - No code copied from existing software
 *       Algorithms based on published theory
 */

#include "mshqc/integrals/eri_transformer.h"
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mshqc {
namespace integrals {

Eigen::Tensor<double, 4> ERITransformer::transform_oovv(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ,
    const Eigen::MatrixXd& C_virt,
    int nbf,
    int nocc,
    int nvirt
) {
    // Allocate output tensor (ij|ab) = (nocc × nocc × nvirt × nvirt)
    Eigen::Tensor<double, 4> eri_mo(nocc, nocc, nvirt, nvirt);
    eri_mo.setZero();
    
    // NAIVE ALGORITHM: Direct contraction
    // (ij|ab)_MO = Σ_μνλσ C_μi C_νj (μν|λσ)_AO C_λa C_σb
    // 
    // Loop ordering: i-j-a-b (MO) outer, μ-ν-λ-σ (AO) inner
    // This is not optimal but simple and correct
    
    for (int i = 0; i < nocc; i++) {
        for (int j = 0; j < nocc; j++) {
            for (int a = 0; a < nvirt; a++) {
                for (int b = 0; b < nvirt; b++) {
                    double val = 0.0;
                    
                    // Contract over all AO indices
                    for (int mu = 0; mu < nbf; mu++) {
                        for (int nu = 0; nu < nbf; nu++) {
                            for (int lam = 0; lam < nbf; lam++) {
                                for (int sig = 0; sig < nbf; sig++) {
                                    // Physicist notation: <ij|ab> = (ia|jb)
                                    // ERI storage: (μν|λσ)
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

Eigen::Tensor<double, 4> ERITransformer::transform_vvov(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ,
    const Eigen::MatrixXd& C_virt,
    int nbf,
    int nocc,
    int nvirt
) {
    // Allocate output tensor (ab|kc) = (nvirt × nvirt × nocc × nvirt)
    Eigen::Tensor<double, 4> eri_mo(nvirt, nvirt, nocc, nvirt);
    eri_mo.setZero();
    
    // FORMULA: (ab|kc)_MO = Σ_μνλσ C_μa C_νb (μν|λσ)_AO C_λk C_σc
    // Used in MP3 particle-particle ladder: Σ_kc <ab||kc> t_ij^kc
    
    for (int a = 0; a < nvirt; a++) {
        for (int b = 0; b < nvirt; b++) {
            for (int k = 0; k < nocc; k++) {
                for (int c = 0; c < nvirt; c++) {
                    double val = 0.0;
                    
                    for (int mu = 0; mu < nbf; mu++) {
                        for (int nu = 0; nu < nbf; nu++) {
                            for (int lam = 0; lam < nbf; lam++) {
                                for (int sig = 0; sig < nbf; sig++) {
                                    // Transform: μ→a, ν→b, λ→k, σ→c
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

Eigen::Tensor<double, 4> ERITransformer::transform_oooo(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ,
    int nbf,
    int nocc
) {
    // Allocate output tensor (ik|jl) = (nocc × nocc × nocc × nocc)
    Eigen::Tensor<double, 4> eri_mo(nocc, nocc, nocc, nocc);
    eri_mo.setZero();
    
    // FORMULA: (ik|jl)_MO = Σ_μνλσ C_μi C_νk (μν|λσ)_AO C_λj C_σl
    // Used in MP3 hole-hole ladder: Σ_kl <ik||jl> t_kl^ab
    
    for (int i = 0; i < nocc; i++) {
        for (int k = 0; k < nocc; k++) {
            for (int j = 0; j < nocc; j++) {
                for (int l = 0; l < nocc; l++) {
                    double val = 0.0;
                    
                    for (int mu = 0; mu < nbf; mu++) {
                        for (int nu = 0; nu < nbf; nu++) {
                            for (int lam = 0; lam < nbf; lam++) {
                                for (int sig = 0; sig < nbf; sig++) {
                                    // Transform: μ→i, ν→k, λ→j, σ→l
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

Eigen::Tensor<double, 4> ERITransformer::transform_oovv_mixed(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ_A,
    const Eigen::MatrixXd& C_virt_B,
    int nbf,
    int nocc_A,
    int nvirt_B
) {
    // Mixed-spin transformation for UMP αβ blocks
    // (ij|ab)_MO where i,j are spin A and a,b are spin B
    
    Eigen::Tensor<double, 4> eri_mo(nocc_A, nocc_A, nvirt_B, nvirt_B);
    eri_mo.setZero();
    
    // FORMULA: (ij|ab)_MO = Σ_μνλσ C^A_μi C^A_νj (μν|λσ)_AO C^B_λa C^B_σb
    
    for (int i = 0; i < nocc_A; i++) {
        for (int j = 0; j < nocc_A; j++) {
            for (int a = 0; a < nvirt_B; a++) {
                for (int b = 0; b < nvirt_B; b++) {
                    double val = 0.0;
                    
                    for (int mu = 0; mu < nbf; mu++) {
                        for (int nu = 0; nu < nbf; nu++) {
                            for (int lam = 0; lam < nbf; lam++) {
                                for (int sig = 0; sig < nbf; sig++) {
                                    val += C_occ_A(mu, i) * C_occ_A(nu, j) *
                                           eri_ao(mu, nu, lam, sig) *
                                           C_virt_B(lam, a) * C_virt_B(sig, b);
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

Eigen::Tensor<double, 4> ERITransformer::transform_oovv_parallel(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ,
    const Eigen::MatrixXd& C_virt,
    int nbf,
    int nocc,
    int nvirt,
    int n_threads
) {
    // OpenMP parallelized version (Month 2, Week 3, Task 3.1)
    // Expected speedup: 2-3× on 4-core CPU
    
#ifdef _OPENMP
    // Set number of threads if specified
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }
    
    // Allocate output tensor
    Eigen::Tensor<double, 4> eri_mo(nocc, nocc, nvirt, nvirt);
    eri_mo.setZero();
    
    // Parallelize over outer loops (i, j)
    // Each thread computes independent (i,j,a,b) elements
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < nocc; i++) {
        for (int j = 0; j < nocc; j++) {
            for (int a = 0; a < nvirt; a++) {
                for (int b = 0; b < nvirt; b++) {
                    double val = 0.0;
                    
                    // Contract over all AO indices (thread-private accumulation)
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
    // OpenMP not available, fall back to serial version
    std::cerr << "INFO: OpenMP not available, using serial version\n";
    (void)n_threads;
    return transform_oovv(eri_ao, C_occ, C_virt, nbf, nocc, nvirt);
#endif
}

// ============================================================================
// QUARTER TRANSFORM ALGORITHM (Helgaker Algorithm 9.5)
// Month 2, Week 3, Task 3.2
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
     * HELGAKER ALGORITHM 9.5 - Quarter Transform
     * 
     * THEORY:
     *   Naive:   (ij|ab) = Σ_μνλσ C_μi C_νa (μν|λσ) C_λj C_σb
     *            Cost: O(nocc^2 × nvirt^2 × nbf^4) ≈ O(N^8)
     * 
     *   Quarter: Four-step contraction reduces prefactor 16 → 4
     *            Cost: O(nbf^5 + nbf^4×nocc + ...) ≈ O(N^5)
     * 
     * STEPS:
     *   1. (μν|λσ) → (iν|λσ)   [Contract μ with C_occ]
     *   2. (iν|λσ) → (ia|λσ)   [Contract ν with C_virt]
     *   3. (ia|λσ) → (ia|jσ)   [Contract λ with C_occ]
     *   4. (ia|jσ) → (ia|jb)   [Contract σ with C_virt]
     *   5. Rearrange to (ij|ab)
     * 
     * MEMORY: 3 intermediate tensors (temp1, temp2, temp3)
     *   - temp1: nocc × nbf × nbf × nbf
     *   - temp2: nocc × nvirt × nbf × nbf
     *   - temp3: nocc × nvirt × nocc × nbf
     * 
     * REFERENCE:
     *   Helgaker, Jørgensen, Olsen (2000)
     *   "Molecular Electronic-Structure Theory", Algorithm 9.5, pp. 322-323
     */
    
    // ========================================================================
    // STEP 1: (μν|λσ) → (iν|λσ)
    // Contract first index (μ) with C_occ to get occupied index i
    // ========================================================================
    
    Eigen::Tensor<double, 4> temp1(nocc, nbf, nbf, nbf);
    temp1.setZero();
    
    // temp1(i,ν,λ,σ) = Σ_μ C(μ,i) × ERI(μ,ν,λ,σ)
    // Cost: O(nbf^4 × nocc)
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
    
    // temp2(i,a,λ,σ) = Σ_ν C(ν,a) × temp1(i,ν,λ,σ)
    // Cost: O(nbf^3 × nocc × nvirt)
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
    
    // temp3(i,a,j,σ) = Σ_λ C(λ,j) × temp2(i,a,λ,σ)
    // Cost: O(nbf^2 × nocc^2 × nvirt)
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
    
    // temp4(i,a,j,b) = Σ_σ C(σ,b) × temp3(i,a,j,σ)
    // Cost: O(nbf × nocc^2 × nvirt^2)
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
    
    // eri_mo(i,j,a,b) = temp4(i,a,j,b)
    // Simple index reordering
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

Eigen::Tensor<double, 4> ERITransformer::transform_oooo_mixed(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_occ_A,
    const Eigen::MatrixXd& C_occ_B,
    int nbf,
    int nocc_A,
    int nocc_B
) {
    /**
     * Mixed-spin OOOO transformation: (μν|λσ)_AO → (kl|ij)_MO
     * 
     * FORMULA:
     *   (kl|ij)_MO = Σ_μνλσ C^α_μk C^α_νl (μν|λσ)_AO C^β_λi C^β_σj
     * 
     * 4-step quarter transform (same as OOVV but all occupied indices):
     *   Step 1: (μν|λσ) → (kν|λσ)   [Contract μ with α]
     *   Step 2: (kν|λσ) → (kl|λσ)   [Contract ν with α]
     *   Step 3: (kl|λσ) → (kl|iσ)   [Contract λ with β]
     *   Step 4: (kl|iσ) → (kl|ij)   [Contract σ with β]
     * 
     * COMPLEXITY: O(nbf^4 × nocc_A × nocc_B) ≈ O(N^5)
     */
    
    // Step 1: (μν|λσ) → (kν|λσ)
    Eigen::Tensor<double, 4> temp1(nocc_A, nbf, nbf, nbf);
    temp1.setZero();
    
    for (int k = 0; k < nocc_A; k++) {
        for (int nu = 0; nu < nbf; nu++) {
            for (int lam = 0; lam < nbf; lam++) {
                for (int sig = 0; sig < nbf; sig++) {
                    double val = 0.0;
                    for (int mu = 0; mu < nbf; mu++) {
                        val += C_occ_A(mu, k) * eri_ao(mu, nu, lam, sig);
                    }
                    temp1(k, nu, lam, sig) = val;
                }
            }
        }
    }
    
    // Step 2: (kν|λσ) → (kl|λσ)
    Eigen::Tensor<double, 4> temp2(nocc_A, nocc_A, nbf, nbf);
    temp2.setZero();
    
    for (int k = 0; k < nocc_A; k++) {
        for (int l = 0; l < nocc_A; l++) {
            for (int lam = 0; lam < nbf; lam++) {
                for (int sig = 0; sig < nbf; sig++) {
                    double val = 0.0;
                    for (int nu = 0; nu < nbf; nu++) {
                        val += C_occ_A(nu, l) * temp1(k, nu, lam, sig);
                    }
                    temp2(k, l, lam, sig) = val;
                }
            }
        }
    }
    
    // Step 3: (kl|λσ) → (kl|iσ)
    Eigen::Tensor<double, 4> temp3(nocc_A, nocc_A, nocc_B, nbf);
    temp3.setZero();
    
    for (int k = 0; k < nocc_A; k++) {
        for (int l = 0; l < nocc_A; l++) {
            for (int i = 0; i < nocc_B; i++) {
                for (int sig = 0; sig < nbf; sig++) {
                    double val = 0.0;
                    for (int lam = 0; lam < nbf; lam++) {
                        val += C_occ_B(lam, i) * temp2(k, l, lam, sig);
                    }
                    temp3(k, l, i, sig) = val;
                }
            }
        }
    }
    
    // Step 4: (kl|iσ) → (kl|ij)
    Eigen::Tensor<double, 4> eri_mo(nocc_A, nocc_A, nocc_B, nocc_B);
    eri_mo.setZero();
    
    for (int k = 0; k < nocc_A; k++) {
        for (int l = 0; l < nocc_A; l++) {
            for (int i = 0; i < nocc_B; i++) {
                for (int j = 0; j < nocc_B; j++) {
                    double val = 0.0;
                    for (int sig = 0; sig < nbf; sig++) {
                        val += C_occ_B(sig, j) * temp3(k, l, i, sig);
                    }
                    eri_mo(k, l, i, j) = val;
                }
            }
        }
    }
    
    return eri_mo;
}

Eigen::Tensor<double, 4> ERITransformer::transform_vvvv(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C_virt,
    int nbf,
    int nvirt
) {
    /**
     * VVVV transformation: (μν|λσ)_AO → (ab|cd)_MO
     * 
     * FORMULA:
     *   (ab|cd)_MO = Σ_μνλσ C_μa C_νb (μν|λσ)_AO C_λc C_σd
     * 
     * CRITICAL for complete MP3 - provides PP ladder term.
     * Without this term, E(3) diverges (too negative).
     * 
     * COST: O(nbf^4 × nvirt^4) - very expensive!
     *   - cc-pVDZ (nvirt=9): ~6,500 × nbf^4
     *   - cc-pVTZ (nvirt=28): ~614,000 × nbf^4 (!)
     * 
     * STORAGE: O(nvirt^4)
     *   - cc-pVDZ: 9^4 = 6,561 elements ≈ 52 KB
     *   - cc-pVTZ: 28^4 = 614,656 elements ≈ 4.9 MB
     * 
     * ALGORITHM: 4-step quarter transform (same as OOVV/OOOO)
     *   Step 1: (μν|λσ) → (aν|λσ)   [Contract μ]
     *   Step 2: (aν|λσ) → (ab|λσ)   [Contract ν]
     *   Step 3: (ab|λσ) → (ab|cσ)   [Contract λ]
     *   Step 4: (ab|cσ) → (ab|cd)   [Contract σ]
     */
    
    std::cout << "  Transforming <ab|cd> (vvvv) block (quarter transform)..." << std::flush;
    
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
    std::cout << "  VVVV storage: " << mem_mb << " MB\n";
    
    return eri_mo;
}

}} // namespace mshqc::integrals
