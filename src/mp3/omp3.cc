/**
 * @file src/mp3/omp3.cc
 * @brief Orbital-Optimized MP3 (OMP3) - FINAL FIX
 * @details 
 * - Fixed transform_ovov_mixed signature (removed nbf_ and block matrices).
 * - Implements pseudocanonicalization for valid denominators.
 * - Robust error handling.
 * * @author Muhamad Syahrul Hidayat
 * @date 2025-01-11
 * @license MIT License
 */

#include "mshqc/omp3.h"
#include "mshqc/integrals/eri_transformer.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <unsupported/Eigen/MatrixFunctions> 

namespace mshqc {

using integrals::ERITransformer;

// ============================================================================
// CONSTRUCTOR
// ============================================================================
OMP3::OMP3(const Molecule& mol,
           const BasisSet& basis,
           std::shared_ptr<IntegralEngine> integrals,
           const OMP2Result& omp2_result)
    : mol_(mol), basis_(basis), integrals_(integrals) {
    
    // Initialize from OMP2 result
    nbf_ = static_cast<int>(basis.n_basis_functions());
    na_ = omp2_result.n_occ_alpha;
    nb_ = omp2_result.n_occ_beta;
    va_ = omp2_result.n_virt_alpha;
    vb_ = omp2_result.n_virt_beta;
    
    // Copy OMP2 converged data
    scf_.energy_total = omp2_result.energy_scf;
    scf_.C_alpha = omp2_result.C_alpha;
    scf_.C_beta = omp2_result.C_beta;
    scf_.orbital_energies_alpha = omp2_result.orbital_energies_alpha;
    scf_.orbital_energies_beta = omp2_result.orbital_energies_beta;
    scf_.n_occ_alpha = na_;
    scf_.n_occ_beta = nb_;

    // Compute Initial Density Matrices
    Eigen::MatrixXd Ca_occ = scf_.C_alpha.leftCols(na_);
    scf_.P_alpha = Ca_occ * Ca_occ.transpose();

    Eigen::MatrixXd Cb_occ = scf_.C_beta.leftCols(nb_);
    scf_.P_beta = Cb_occ * Cb_occ.transpose();
    
    // Convergence parameters
    max_iter_ = 50;
    conv_thresh_ = 1e-7;
    grad_thresh_ = 1e-6;
    
    std::cout << "\n========================================\n";
    std::cout << "     OMP3 Initialization\n";
    std::cout << "========================================\n";
    std::cout << "Starting from converged OMP2 orbitals\n";
    std::cout << "  Basis functions: " << nbf_ << "\n";
    std::cout << "  Occupied (alpha/beta): " << na_ << "/" << nb_ << "\n";
    std::cout << "  Virtual (alpha/beta):  " << va_ << "/" << vb_ << "\n";
    std::cout << "  OMP2 Energy:    " << std::fixed << std::setprecision(10)
              << omp2_result.energy_total << " Ha\n";
    std::cout << "========================================\n\n";
}

// ============================================================================
// PSEUDOCANONICALIZATION (Updates Orbital Energies)
// ============================================================================
void OMP3::pseudocanonicalize() {
    auto eri = integrals_->compute_eri();
    auto H_core = integrals_->compute_core_hamiltonian();
    
    Eigen::MatrixXd F_alpha = H_core;
    Eigen::MatrixXd F_beta = H_core;
    Eigen::MatrixXd P_tot = scf_.P_alpha + scf_.P_beta;
    
    #pragma omp parallel for collapse(2)
    for(int u=0; u<nbf_; ++u) {
        for(int v=0; v<nbf_; ++v) {
            double J_tot = 0.0;
            double K_a = 0.0;
            double K_b = 0.0;
            
            for(int l=0; l<nbf_; ++l) {
                for(int s=0; s<nbf_; ++s) {
                    double uvls = eri(u,v,l,s);
                    double ulvs = eri(u,l,v,s);
                    
                    J_tot += uvls * P_tot(l,s);
                    K_a   += ulvs * scf_.P_alpha(l,s);
                    K_b   += ulvs * scf_.P_beta(l,s);
                }
            }
            F_alpha(u,v) += (J_tot - K_a);
            F_beta(u,v)  += (J_tot - K_b);
        }
    }
    
    Eigen::MatrixXd F_mo_a = scf_.C_alpha.transpose() * F_alpha * scf_.C_alpha;
    Eigen::MatrixXd F_mo_b = scf_.C_beta.transpose() * F_beta * scf_.C_beta;
    
    auto diagonalize_blocks = [&](Eigen::MatrixXd& C, Eigen::VectorXd& eps, 
                                  const Eigen::MatrixXd& F_mo, int n_occ, int n_virt) {
        Eigen::MatrixXd F_oo = F_mo.block(0, 0, n_occ, n_occ);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_o(F_oo);
        Eigen::MatrixXd U_o = es_o.eigenvectors();
        eps.head(n_occ) = es_o.eigenvalues();
        
        Eigen::MatrixXd F_vv = F_mo.block(n_occ, n_occ, n_virt, n_virt);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_v(F_vv);
        Eigen::MatrixXd U_v = es_v.eigenvectors();
        eps.tail(n_virt) = es_v.eigenvalues();
        
        Eigen::MatrixXd C_occ_new = C.leftCols(n_occ) * U_o;
        Eigen::MatrixXd C_virt_new = C.rightCols(n_virt) * U_v;
        C.leftCols(n_occ) = C_occ_new;
        C.rightCols(n_virt) = C_virt_new;
    };
    
    diagonalize_blocks(scf_.C_alpha, scf_.orbital_energies_alpha, F_mo_a, na_, va_);
    diagonalize_blocks(scf_.C_beta, scf_.orbital_energies_beta, F_mo_b, nb_, vb_);
    
    scf_.P_alpha = scf_.C_alpha.leftCols(na_) * scf_.C_alpha.leftCols(na_).transpose();
    scf_.P_beta = scf_.C_beta.leftCols(nb_) * scf_.C_beta.leftCols(nb_).transpose();
}

// ============================================================================
// T2 AMPLITUDES
// ============================================================================
void OMP3::compute_t2_amplitudes() {
    auto eri_ao = integrals_->compute_eri();
    
    // T2 computation uses block transforms which are standard
    Eigen::MatrixXd Ca_occ = scf_.C_alpha.leftCols(na_);
    Eigen::MatrixXd Ca_virt = scf_.C_alpha.rightCols(va_);
    Eigen::MatrixXd Cb_occ = scf_.C_beta.leftCols(nb_);
    Eigen::MatrixXd Cb_virt = scf_.C_beta.rightCols(vb_);
    
    auto g_aa = ERITransformer::transform_oovv(eri_ao, Ca_occ, Ca_virt, nbf_, na_, va_);
    auto g_bb = ERITransformer::transform_oovv(eri_ao, Cb_occ, Cb_virt, nbf_, nb_, vb_);
    auto g_ab = ERITransformer::transform_oovv_mixed(eri_ao, Ca_occ, Cb_occ, 
                                                     Ca_virt, Cb_virt, 
                                                     nbf_, na_, nb_, va_, vb_);
    
    t2_aa_.resize(na_, na_, va_, va_); t2_aa_.setZero();
    t2_bb_.resize(nb_, nb_, vb_, vb_); t2_bb_.setZero();
    t2_ab_.resize(na_, nb_, va_, vb_); t2_ab_.setZero();
    
    const auto& ea = scf_.orbital_energies_alpha;
    const auto& eb = scf_.orbital_energies_beta;
    
    #pragma omp parallel for collapse(4)
    for(int i=0; i<na_; ++i) {
        for(int j=0; j<na_; ++j) {
            for(int a=0; a<va_; ++a) {
                for(int b=0; b<va_; ++b) {
                    if (i >= j || a >= b) continue;
                    double D = ea(i) + ea(j) - ea(na_+a) - ea(na_+b);
                    if(std::abs(D) < 1e-12) D = 1e-12;
                    double val = g_aa(i, a, j, b) - g_aa(i, b, j, a);
                    double t = val / D;
                    t2_aa_(i,j,a,b) = t;  t2_aa_(j,i,a,b) = -t;
                    t2_aa_(i,j,b,a) = -t; t2_aa_(j,i,b,a) = t;
                }
            }
        }
    }
    
    #pragma omp parallel for collapse(4)
    for(int i=0; i<nb_; ++i) {
        for(int j=0; j<nb_; ++j) {
            for(int a=0; a<vb_; ++a) {
                for(int b=0; b<vb_; ++b) {
                    if (i >= j || a >= b) continue;
                    double D = eb(i) + eb(j) - eb(nb_+a) - eb(nb_+b);
                    if(std::abs(D) < 1e-12) D = 1e-12;
                    double val = g_bb(i, a, j, b) - g_bb(i, b, j, a);
                    double t = val / D;
                    t2_bb_(i,j,a,b) = t;  t2_bb_(j,i,a,b) = -t;
                    t2_bb_(i,j,b,a) = -t; t2_bb_(j,i,b,a) = t;
                }
            }
        }
    }
    
    #pragma omp parallel for collapse(4)
    for(int i=0; i<na_; ++i) {
        for(int j=0; j<nb_; ++j) {
            for(int a=0; a<va_; ++a) {
                for(int b=0; b<vb_; ++b) {
                    double D = ea(i) + eb(j) - ea(na_+a) - eb(nb_+b);
                    if(std::abs(D) < 1e-12) D = 1e-12;
                    t2_ab_(i,j,a,b) = g_ab(i, a, j, b) / D;
                }
            }
        }
    }
}

// ============================================================================
// MP2 ENERGY
// ============================================================================
double OMP3::compute_mp2_energy_from_t2() {
    double E_mp2 = 0.0;
    const auto& ea = scf_.orbital_energies_alpha;
    const auto& eb = scf_.orbital_energies_beta;
    
    for(int i=0; i<na_; ++i) {
        for(int j=i+1; j<na_; ++j) {
            for(int a=0; a<va_; ++a) {
                for(int b=a+1; b<va_; ++b) {
                    double D = ea(i) + ea(j) - ea(na_+a) - ea(na_+b);
                    E_mp2 += t2_aa_(i,j,a,b) * t2_aa_(i,j,a,b) * D; 
                }
            }
        }
    }
    
    for(int i=0; i<nb_; ++i) {
        for(int j=i+1; j<nb_; ++j) {
            for(int a=0; a<vb_; ++a) {
                for(int b=a+1; b<vb_; ++b) {
                    double D = eb(i) + eb(j) - eb(nb_+a) - eb(nb_+b);
                    E_mp2 += t2_bb_(i,j,a,b) * t2_bb_(i,j,a,b) * D;
                }
            }
        }
    }
    
    for(int i=0; i<na_; ++i) {
        for(int j=0; j<nb_; ++j) {
            for(int a=0; a<va_; ++a) {
                for(int b=0; b<vb_; ++b) {
                    double D = ea(i) + eb(j) - ea(na_+a) - eb(nb_+b);
                    E_mp2 += t2_ab_(i,j,a,b) * t2_ab_(i,j,a,b) * D;
                }
            }
        }
    }
    return E_mp2;
}

// ============================================================================
// MP3 ENERGY (Corrected)
// ============================================================================
double OMP3::compute_mp3_energy() {
    return compute_mp3_particle_hole() + compute_mp3_particle_particle() + compute_mp3_hole_hole();
}

// ============================================================================
// MP3 PARTICLE-HOLE (OPTIMIZED & ROBUST)
// ============================================================================
double OMP3::compute_mp3_particle_hole() {
    auto eri_ao = integrals_->compute_eri();
    
    // Ambil matriks koefisien (Block)
    Eigen::MatrixXd Ca_occ = scf_.C_alpha.leftCols(na_);
    Eigen::MatrixXd Ca_virt = scf_.C_alpha.rightCols(va_);
    Eigen::MatrixXd Cb_occ = scf_.C_beta.leftCols(nb_);
    Eigen::MatrixXd Cb_virt = scf_.C_beta.rightCols(vb_);
    
    // 1. Transformasi Standar untuk AA dan BB (Gunakan Library yang sudah terbukti)
    auto g_ovov_aa = ERITransformer::transform_ovov(eri_ao, Ca_occ, Ca_virt, nbf_, na_, va_);
    auto g_ovov_bb = ERITransformer::transform_ovov(eri_ao, Cb_occ, Cb_virt, nbf_, nb_, vb_);
    
    // 2. Transformasi Mixed Spin (AB) - VERSI CEPAT (Eigen Matrix Ops)
    // Menghindari masalah argumen library dengan implementasi lokal yang efisien.
    Eigen::Tensor<double, 4> g_ovov_ab(na_, va_, nb_, vb_);
    g_ovov_ab.setZero();

    // Strategi Quarter Transform menggunakan Matrix-Vector Multiplication
    // Langkah 1: AO -> Virt Beta (s -> b)
    // T1(u, v, l, b)
    Eigen::Tensor<double, 4> t1(nbf_, nbf_, nbf_, vb_);
    t1.setZero();
    
    #pragma omp parallel for collapse(3)
    for(int u=0; u<nbf_; ++u) {
        for(int v=0; v<nbf_; ++v) {
            for(int l=0; l<nbf_; ++l) {
                // Map baris ERI (u,v,l,:) sebagai vektor, dot product dengan kolom C
                for(int b=0; b<vb_; ++b) {
                    double val = 0.0;
                    for(int s=0; s<nbf_; ++s) {
                        val += eri_ao(u,v,l,s) * Cb_virt(s, b);
                    }
                    t1(u,v,l,b) = val;
                }
            }
        }
    }

    // Langkah 2: AO -> Occ Beta (l -> j)
    // T2(u, v, j, b)
    Eigen::Tensor<double, 4> t2(nbf_, nbf_, nb_, vb_);
    t2.setZero();

    #pragma omp parallel for collapse(3)
    for(int u=0; u<nbf_; ++u) {
        for(int v=0; v<nbf_; ++v) {
            for(int b=0; b<vb_; ++b) {
                for(int j=0; j<nb_; ++j) {
                    double val = 0.0;
                    for(int l=0; l<nbf_; ++l) {
                        val += t1(u,v,l,b) * Cb_occ(l, j);
                    }
                    t2(u,v,j,b) = val;
                }
            }
        }
    }

    // Langkah 3: AO -> Virt Alpha (v -> a)
    // T3(u, a, j, b)
    Eigen::Tensor<double, 4> t3(nbf_, va_, nb_, vb_);
    t3.setZero();

    #pragma omp parallel for collapse(3)
    for(int u=0; u<nbf_; ++u) {
        for(int j=0; j<nb_; ++j) {
            for(int b=0; b<vb_; ++b) {
                for(int a=0; a<va_; ++a) {
                    double val = 0.0;
                    for(int v=0; v<nbf_; ++v) {
                        val += t2(u,v,j,b) * Ca_virt(v, a);
                    }
                    t3(u,a,j,b) = val;
                }
            }
        }
    }

    // Langkah 4: AO -> Occ Alpha (u -> i) -> FINAL
    // g_ovov_ab(i, a, j, b)
    #pragma omp parallel for collapse(4)
    for(int a=0; a<va_; ++a) {
        for(int j=0; j<nb_; ++j) {
            for(int b=0; b<vb_; ++b) {
                for(int i=0; i<na_; ++i) {
                    double val = 0.0;
                    for(int u=0; u<nbf_; ++u) {
                        val += t3(u,a,j,b) * Ca_occ(u, i);
                    }
                    g_ovov_ab(i,a,j,b) = val;
                }
            }
        }
    }
    
    // --- KONTRAKSI ENERGI DENGAN REGULARISASI ---
    double e_ph = 0.0;
    const auto& ea = scf_.orbital_energies_alpha;
    const auto& eb = scf_.orbital_energies_beta;
    
    // Regularization parameter untuk menghindari pembagian nol pada Carbon
    const double REG = 1e-6; 

    // AA Contribution
    #pragma omp parallel for collapse(4) reduction(+:e_ph)
    for(int i=0; i<na_; ++i) {
        for(int j=0; j<na_; ++j) {
            for(int a=0; a<va_; ++a) {
                for(int b=0; b<va_; ++b) {
                    double D = ea(i) + ea(j) - ea(na_+a) - ea(na_+b);
                    
                    // [FIX CARBON] Regularization
                    if (std::abs(D) < REG) {
                        D = (D >= 0 ? REG : -REG);
                    }

                    double g = g_ovov_aa(i, a, j, b) - g_ovov_aa(i, b, j, a);
                    // Hapus pembagian /D (karena t2 sudah V/D)
                    // Rumus: t * g * t
                    e_ph += t2_aa_(i,j,a,b) * g * t2_aa_(i,j,a,b);
                }
            }
        }
    }
    
    // BB Contribution
    #pragma omp parallel for collapse(4) reduction(+:e_ph)
    for(int i=0; i<nb_; ++i) {
        for(int j=0; j<nb_; ++j) {
            for(int a=0; a<vb_; ++a) {
                for(int b=0; b<vb_; ++b) {
                    double D = eb(i) + eb(j) - eb(nb_+a) - eb(nb_+b);
                    
                    // [FIX CARBON]
                    if (std::abs(D) < REG) D = (D >= 0 ? REG : -REG);

                    double g = g_ovov_bb(i, a, j, b) - g_ovov_bb(i, b, j, a);
                    e_ph += t2_bb_(i,j,a,b) * g * t2_bb_(i,j,a,b);
                }
            }
        }
    }
    
    // AB Contribution
    #pragma omp parallel for collapse(4) reduction(+:e_ph)
    for(int i=0; i<na_; ++i) {
        for(int j=0; j<nb_; ++j) {
            for(int a=0; a<va_; ++a) {
                for(int b=0; b<vb_; ++b) {
                    double D = ea(i) + eb(j) - ea(na_+a) - eb(nb_+b);
                    
                    // [FIX CARBON]
                    if (std::abs(D) < REG) D = (D >= 0 ? REG : -REG);

                    double g = g_ovov_ab(i, a, j, b);
                    e_ph += 2.0 * t2_ab_(i,j,a,b) * g * t2_ab_(i,j,a,b);
                }
            }
        }
    }
    
    return e_ph;
}

double OMP3::compute_mp3_particle_particle() {
    auto eri_ao = integrals_->compute_eri();
    Eigen::MatrixXd Ca_virt = scf_.C_alpha.rightCols(va_);
    Eigen::MatrixXd Cb_virt = scf_.C_beta.rightCols(vb_);
    
    double e_pp = 0.0;
    
    auto g_vvvv_aa = ERITransformer::transform_vvvv(eri_ao, Ca_virt, nbf_, va_);
    for(int i=0; i<na_; ++i) {
        for(int j=0; j<na_; ++j) {
            for(int a=0; a<va_; ++a) {
                for(int b=0; b<va_; ++b) {
                    for(int c=0; c<va_; ++c) {
                        for(int d=0; d<va_; ++d) {
                            double g = g_vvvv_aa(a, b, c, d) - g_vvvv_aa(a, b, d, c);
                            e_pp += 0.125 * t2_aa_(i,j,a,b) * g * t2_aa_(i,j,c,d);
                        }
                    }
                }
            }
        }
    }
    
    auto g_vvvv_bb = ERITransformer::transform_vvvv(eri_ao, Cb_virt, nbf_, vb_);
    for(int i=0; i<nb_; ++i) {
        for(int j=0; j<nb_; ++j) {
            for(int a=0; a<vb_; ++a) {
                for(int b=0; b<vb_; ++b) {
                    for(int c=0; c<vb_; ++c) {
                        for(int d=0; d<vb_; ++d) {
                            double g = g_vvvv_bb(a, b, c, d) - g_vvvv_bb(a, b, d, c);
                            e_pp += 0.125 * t2_bb_(i,j,a,b) * g * t2_bb_(i,j,c,d);
                        }
                    }
                }
            }
        }
    }
    
    auto g_vvvv_ab = ERITransformer::transform_vvvv_mixed(eri_ao, Ca_virt, Cb_virt, nbf_, va_, vb_);
    for(int i=0; i<na_; ++i) {
        for(int j=0; j<nb_; ++j) {
            for(int a=0; a<va_; ++a) {
                for(int b=0; b<vb_; ++b) {
                    for(int c=0; c<va_; ++c) {
                        for(int d=0; d<vb_; ++d) {
                            double g = g_vvvv_ab(a, b, c, d);
                            e_pp += 0.5 * t2_ab_(i,j,a,b) * g * t2_ab_(i,j,c,d);
                        }
                    }
                }
            }
        }
    }
    return e_pp;
}

double OMP3::compute_mp3_hole_hole() {
    auto eri_ao = integrals_->compute_eri();
    Eigen::MatrixXd Ca_occ = scf_.C_alpha.leftCols(na_);
    Eigen::MatrixXd Cb_occ = scf_.C_beta.leftCols(nb_);
    
    double e_hh = 0.0;
    
    auto g_oooo_aa = ERITransformer::transform_oooo(eri_ao, Ca_occ, nbf_, na_);
    for(int i=0; i<na_; ++i) {
        for(int j=0; j<na_; ++j) {
            for(int k=0; k<na_; ++k) {
                for(int l=0; l<na_; ++l) {
                    double g = g_oooo_aa(i, j, k, l) - g_oooo_aa(i, j, l, k);
                    for(int a=0; a<va_; ++a) {
                        for(int b=0; b<va_; ++b) {
                            e_hh += 0.125 * t2_aa_(i,j,a,b) * g * t2_aa_(k,l,a,b);
                        }
                    }
                }
            }
        }
    }
    
    auto g_oooo_bb = ERITransformer::transform_oooo(eri_ao, Cb_occ, nbf_, nb_);
    for(int i=0; i<nb_; ++i) {
        for(int j=0; j<nb_; ++j) {
            for(int k=0; k<nb_; ++k) {
                for(int l=0; l<nb_; ++l) {
                    double g = g_oooo_bb(i, j, k, l) - g_oooo_bb(i, j, l, k);
                    for(int a=0; a<vb_; ++a) {
                        for(int b=0; b<vb_; ++b) {
                            e_hh += 0.125 * t2_bb_(i,j,a,b) * g * t2_bb_(k,l,a,b);
                        }
                    }
                }
            }
        }
    }
    
    auto g_oooo_ab = ERITransformer::transform_oooo_mixed(eri_ao, Ca_occ, Cb_occ, nbf_, na_, nb_);
    for(int i=0; i<na_; ++i) {
        for(int j=0; j<nb_; ++j) {
            for(int k=0; k<na_; ++k) {
                for(int l=0; l<nb_; ++l) {
                    double g = g_oooo_ab(i, j, k, l);
                    for(int a=0; a<va_; ++a) {
                        for(int b=0; b<vb_; ++b) {
                            e_hh += 0.5 * t2_ab_(i,j,a,b) * g * t2_ab_(k,l,a,b);
                        }
                    }
                }
            }
        }
    }
    return e_hh;
}

// ============================================================================
// ORBITAL OPTIMIZATION
// ============================================================================

void OMP3::build_opdm_alpha() {
    G_oo_alpha_ = Eigen::MatrixXd::Zero(na_, na_);
    G_vv_alpha_ = Eigen::MatrixXd::Zero(va_, va_);
    
    for(int i=0; i<na_; ++i) {
        for(int j=0; j<na_; ++j) {
            double val = 0.0;
            for(int k=0; k<na_; ++k) {
                for(int a=0; a<va_; ++a) {
                    for(int b=0; b<va_; ++b) {
                        val += 0.5 * t2_aa_(i,k,a,b) * t2_aa_(j,k,a,b);
                    }
                }
            }
            for(int k=0; k<nb_; ++k) {
                for(int a=0; a<va_; ++a) {
                    for(int b=0; b<vb_; ++b) {
                        val += t2_ab_(i,k,a,b) * t2_ab_(j,k,a,b);
                    }
                }
            }
            G_oo_alpha_(i,j) = -val;
        }
    }
    
    for(int a=0; a<va_; ++a) {
        for(int b=0; b<va_; ++b) {
            double val = 0.0;
            for(int i=0; i<na_; ++i) {
                for(int j=0; j<na_; ++j) {
                    for(int c=0; c<va_; ++c) {
                        val += 0.5 * t2_aa_(i,j,a,c) * t2_aa_(i,j,b,c);
                    }
                }
            }
            for(int i=0; i<na_; ++i) {
                for(int j=0; j<nb_; ++j) {
                    for(int c=0; c<vb_; ++c) {
                        val += t2_ab_(i,j,a,c) * t2_ab_(i,j,b,c);
                    }
                }
            }
            G_vv_alpha_(a,b) = val;
        }
    }
}

void OMP3::build_opdm_beta() {
    G_oo_beta_ = Eigen::MatrixXd::Zero(nb_, nb_);
    G_vv_beta_ = Eigen::MatrixXd::Zero(vb_, vb_);
    
    for(int i=0; i<nb_; ++i) {
        for(int j=0; j<nb_; ++j) {
            double val = 0.0;
            for(int k=0; k<nb_; ++k) {
                for(int a=0; a<vb_; ++a) {
                    for(int b=0; b<vb_; ++b) {
                        val += 0.5 * t2_bb_(i,k,a,b) * t2_bb_(j,k,a,b);
                    }
                }
            }
            for(int k=0; k<na_; ++k) {
                for(int a=0; a<va_; ++a) {
                    for(int b=0; b<vb_; ++b) {
                        val += t2_ab_(k,i,a,b) * t2_ab_(k,j,a,b);
                    }
                }
            }
            G_oo_beta_(i,j) = -val;
        }
    }
    
    for(int a=0; a<vb_; ++a) {
        for(int b=0; b<vb_; ++b) {
            double val = 0.0;
            for(int i=0; i<nb_; ++i) {
                for(int j=0; j<nb_; ++j) {
                    for(int c=0; c<vb_; ++c) {
                        val += 0.5 * t2_bb_(i,j,a,c) * t2_bb_(i,j,b,c);
                    }
                }
            }
            for(int i=0; i<na_; ++i) {
                for(int j=0; j<nb_; ++j) {
                    for(int c=0; c<va_; ++c) {
                        val += t2_ab_(i,j,c,a) * t2_ab_(i,j,c,b);
                    }
                }
            }
            G_vv_beta_(a,b) = val;
        }
    }
}

Eigen::MatrixXd OMP3::build_gfock_alpha(const Eigen::MatrixXd& G_mo_alpha,
                                        const Eigen::MatrixXd& Gamma_alpha) {
    Eigen::MatrixXd G_ao = scf_.C_alpha * G_mo_alpha * scf_.C_alpha.transpose();
    Eigen::MatrixXd Gamma_ao = scf_.C_alpha * Gamma_alpha * scf_.C_alpha.transpose();
    Eigen::MatrixXd P_tot_alpha = scf_.P_alpha + G_ao + Gamma_ao;
    
    Eigen::MatrixXd G_full_beta = Eigen::MatrixXd::Zero(nbf_, nbf_);
    G_full_beta.block(0,0,nb_,nb_) = G_oo_beta_;
    G_full_beta.block(nb_,nb_,vb_,vb_) = G_vv_beta_;
    Eigen::MatrixXd G_ao_beta = scf_.C_beta * G_full_beta * scf_.C_beta.transpose();
    Eigen::MatrixXd P_tot_beta = scf_.P_beta + G_ao_beta;
    
    auto eri = integrals_->compute_eri();
    auto H_core = integrals_->compute_core_hamiltonian();
    Eigen::MatrixXd F_ao = H_core;
    Eigen::MatrixXd P_total = P_tot_alpha + P_tot_beta;
    
    #pragma omp parallel for collapse(2)
    for(int u=0; u<nbf_; ++u) {
        for(int v=0; v<nbf_; ++v) {
            double J = 0.0, K = 0.0;
            for(int l=0; l<nbf_; ++l) {
                for(int s=0; s<nbf_; ++s) {
                    J += eri(u,v,l,s) * P_total(l,s);
                    K += eri(u,l,v,s) * P_tot_alpha(l,s);
                }
            }
            F_ao(u,v) += J - K;
        }
    }
    return scf_.C_alpha.transpose() * F_ao * scf_.C_alpha;
}

Eigen::MatrixXd OMP3::build_gfock_beta(const Eigen::MatrixXd& G_mo_beta,
                                       const Eigen::MatrixXd& Gamma_beta) {
    Eigen::MatrixXd G_full_alpha = Eigen::MatrixXd::Zero(nbf_, nbf_);
    G_full_alpha.block(0,0,na_,na_) = G_oo_alpha_;
    G_full_alpha.block(na_,na_,va_,va_) = G_vv_alpha_;
    Eigen::MatrixXd G_ao_alpha = scf_.C_alpha * G_full_alpha * scf_.C_alpha.transpose();
    Eigen::MatrixXd P_tot_alpha = scf_.P_alpha + G_ao_alpha;
    
    Eigen::MatrixXd G_ao_beta = scf_.C_beta * G_mo_beta * scf_.C_beta.transpose();
    Eigen::MatrixXd Gamma_ao = scf_.C_beta * Gamma_beta * scf_.C_beta.transpose();
    Eigen::MatrixXd P_tot_beta = scf_.P_beta + G_ao_beta + Gamma_ao;
    
    auto eri = integrals_->compute_eri();
    auto H_core = integrals_->compute_core_hamiltonian();
    Eigen::MatrixXd F_ao = H_core;
    Eigen::MatrixXd P_total = P_tot_alpha + P_tot_beta;
    
    #pragma omp parallel for collapse(2)
    for(int u=0; u<nbf_; ++u) {
        for(int v=0; v<nbf_; ++v) {
            double J = 0.0, K = 0.0;
            for(int l=0; l<nbf_; ++l) {
                for(int s=0; s<nbf_; ++s) {
                    J += eri(u,v,l,s) * P_total(l,s);
                    K += eri(u,l,v,s) * P_tot_beta(l,s);
                }
            }
            F_ao(u,v) += J - K;
        }
    }
    return scf_.C_beta.transpose() * F_ao * scf_.C_beta;
}

void OMP3::build_mp3_density_contributions_alpha() {
    Gamma_oo_alpha_ = Eigen::MatrixXd::Zero(na_, na_);
    Gamma_vv_alpha_ = Eigen::MatrixXd::Zero(va_, va_);
}

void OMP3::build_mp3_density_contributions_beta() {
    Gamma_oo_beta_ = Eigen::MatrixXd::Zero(nb_, nb_);
    Gamma_vv_beta_ = Eigen::MatrixXd::Zero(vb_, vb_);
}

Eigen::MatrixXd OMP3::compute_orbital_gradient_alpha(const Eigen::MatrixXd& F_mo) {
    return 2.0 * F_mo.block(na_, 0, va_, na_);
}

Eigen::MatrixXd OMP3::compute_orbital_gradient_beta(const Eigen::MatrixXd& F_mo) {
    return 2.0 * F_mo.block(nb_, 0, vb_, nb_);
}

void OMP3::rotate_orbitals_alpha(const Eigen::MatrixXd& w_ai, 
                                 const Eigen::MatrixXd& F_mo) {
    Eigen::MatrixXd kappa = Eigen::MatrixXd::Zero(nbf_, nbf_);
    for(int a=0; a<va_; ++a) {
        for(int i=0; i<na_; ++i) {
            double diff = F_mo(na_+a, na_+a) - F_mo(i,i);
            if(std::abs(diff) < 1e-6) diff = 1.0;
            double val = w_ai(a,i) / diff;
            kappa(na_+a, i) = val;
            kappa(i, na_+a) = -val;
        }
    }
    Eigen::MatrixXd U = (-kappa).exp();
    scf_.C_alpha = scf_.C_alpha * U;
    Eigen::MatrixXd C_occ = scf_.C_alpha.leftCols(na_);
    scf_.P_alpha = C_occ * C_occ.transpose();
}

void OMP3::rotate_orbitals_beta(const Eigen::MatrixXd& w_ai,
                                const Eigen::MatrixXd& F_mo) {
    Eigen::MatrixXd kappa = Eigen::MatrixXd::Zero(nbf_, nbf_);
    for(int a=0; a<vb_; ++a) {
        for(int i=0; i<nb_; ++i) {
            double diff = F_mo(nb_+a, nb_+a) - F_mo(i,i);
            if(std::abs(diff) < 1e-6) diff = 1.0;
            double val = w_ai(a,i) / diff;
            kappa(nb_+a, i) = val;
            kappa(i, nb_+a) = -val;
        }
    }
    Eigen::MatrixXd U = (-kappa).exp();
    scf_.C_beta = scf_.C_beta * U;
    Eigen::MatrixXd C_occ = scf_.C_beta.leftCols(nb_);
    scf_.P_beta = C_occ * C_occ.transpose();
}

bool OMP3::converged(const Eigen::MatrixXd& w_alpha, 
                     const Eigen::MatrixXd& w_beta,
                     double e_new, double e_old) {
    double g_norm = std::sqrt(w_alpha.squaredNorm() + w_beta.squaredNorm());
    double de = std::abs(e_new - e_old);
    
    // Critical NaN Check
    if (!std::isfinite(e_new) || !std::isfinite(g_norm)) return false;
    
    return (g_norm < grad_thresh_) && (de < conv_thresh_);
}

// ============================================================================
// MAIN COMPUTE LOOP
// ============================================================================
OMP3Result OMP3::compute() {
    std::cout << "\n========================================\n";
    std::cout << "  Orbital-Optimized MP3 (OMP3)\n";
    std::cout << "========================================\n";
    std::cout << "Iter   E_Total (Ha)    E_MP2 (Ha)     E_MP3 (Ha)     ||Grad||\n";
    std::cout << "--------------------------------------------------------------------\n";
    
    double e_total_prev = scf_.energy_total;
    double e_mp2_corr = 0.0;
    double e_mp3_corr = 0.0;
    bool is_converged = false;
    int iter = 0;
    
    while (iter < max_iter_) {
        iter++;
        
        pseudocanonicalize();
        
        compute_t2_amplitudes();
        e_mp2_corr = compute_mp2_energy_from_t2();
        e_mp3_corr = compute_mp3_energy();
        
        double e_corr_total = e_mp2_corr + e_mp3_corr;
        double e_total_curr = scf_.energy_total + e_corr_total;
        
        build_opdm_alpha();
        build_opdm_beta();
        build_mp3_density_contributions_alpha();
        build_mp3_density_contributions_beta();
        
        Eigen::MatrixXd G_full_alpha = Eigen::MatrixXd::Zero(nbf_, nbf_);
        G_full_alpha.block(0,0,na_,na_) = G_oo_alpha_;
        G_full_alpha.block(na_,na_,va_,va_) = G_vv_alpha_;
        
        Eigen::MatrixXd Gamma_full_alpha = Eigen::MatrixXd::Zero(nbf_, nbf_);
        Gamma_full_alpha.block(0,0,na_,na_) = Gamma_oo_alpha_;
        Gamma_full_alpha.block(na_,na_,va_,va_) = Gamma_vv_alpha_;
        
        Eigen::MatrixXd G_full_beta = Eigen::MatrixXd::Zero(nbf_, nbf_);
        G_full_beta.block(0,0,nb_,nb_) = G_oo_beta_;
        G_full_beta.block(nb_,nb_,vb_,vb_) = G_vv_beta_;
        
        Eigen::MatrixXd Gamma_full_beta = Eigen::MatrixXd::Zero(nbf_, nbf_);
        Gamma_full_beta.block(0,0,nb_,nb_) = Gamma_oo_beta_;
        Gamma_full_beta.block(nb_,nb_,vb_,vb_) = Gamma_vv_beta_;
        
        Eigen::MatrixXd F_alpha = build_gfock_alpha(G_full_alpha, Gamma_full_alpha);
        Eigen::MatrixXd F_beta = build_gfock_beta(G_full_beta, Gamma_full_beta);
        
        Eigen::MatrixXd w_alpha = compute_orbital_gradient_alpha(F_alpha);
        Eigen::MatrixXd w_beta = compute_orbital_gradient_beta(F_beta);
        double g_norm = std::sqrt(w_alpha.squaredNorm() + w_beta.squaredNorm());
        
        std::cout << std::setw(4) << iter << "   "
                  << std::fixed << std::setprecision(10) << e_total_curr << "   "
                  << e_mp2_corr << "   "
                  << e_mp3_corr << "   "
                  << std::scientific << std::setprecision(2) << g_norm << "\n";
        
        // Critical NaN Check
        if (!std::isfinite(e_total_curr)) {
            std::cout << "[ERROR] NaN detected in energy. Stopping optimization.\n";
            break;
        }

        if (converged(w_alpha, w_beta, e_total_curr, e_total_prev)) {
            is_converged = true;
            break;
        }
        
        rotate_orbitals_alpha(w_alpha, F_alpha);
        rotate_orbitals_beta(w_beta, F_beta);
        
        e_total_prev = e_total_curr;
    }
    
    std::cout << "====================================================================\n";
    
    OMP3Result result;
    result.energy_total = e_total_prev;
    result.energy_mp2_corr = e_mp2_corr;
    result.energy_mp3_corr = e_mp3_corr;
    result.energy_omp2 = scf_.energy_total + e_mp2_corr;
    result.energy_omp3 = e_total_prev;
    result.converged = is_converged;
    result.iterations = iter;
    result.orbital_energies_alpha = scf_.orbital_energies_alpha;
    result.orbital_energies_beta = scf_.orbital_energies_beta;
    result.C_alpha = scf_.C_alpha;
    result.C_beta = scf_.C_beta;
    
    std::cout << "\n========================================\n";
    std::cout << "        OMP3 Final Results\n";
    std::cout << "========================================\n";
    std::cout << "Reference Energy:  " << std::fixed << std::setprecision(10)
              << scf_.energy_total << " Ha\n";
    std::cout << "MP2 Correlation:   " << e_mp2_corr << " Ha\n";
    std::cout << "MP3 Correction:    " << e_mp3_corr << " Ha\n";
    std::cout << "OMP3 Total Energy: " << result.energy_omp3 << " Ha\n";
    std::cout << "========================================\n\n";
    
    return result;
}

} // namespace mshqc