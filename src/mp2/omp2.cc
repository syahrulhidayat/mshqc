/**
 * @file src/mp2/omp2.cc
 * @brief Orbital-Optimized MP2 (OMP2) - FIXED CRASH
 * @details Fixed argument order in transform_oovv_mixed causing assertion failure.
 */

#include "mshqc/mp2.h"
#include "mshqc/integrals/eri_transformer.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <unsupported/Eigen/MatrixFunctions> 

namespace mshqc {

using integrals::ERITransformer;

// ========================================================================
// CONSTRUCTOR
// ========================================================================
OMP2::OMP2(const Molecule& mol,
           const BasisSet& basis,
           std::shared_ptr<IntegralEngine> integrals,
           const SCFResult& scf_guess)
    : mol_(mol), basis_(basis), integrals_(integrals), scf_(scf_guess) {
    
    nbf_ = static_cast<int>(scf_.C_alpha.rows());
    na_ = scf_.n_occ_alpha;
    nb_ = scf_.n_occ_beta;
    va_ = nbf_ - na_;
    vb_ = nbf_ - nb_;
    
    max_iter_ = 50;
    conv_thresh_ = 1e-6;
    grad_thresh_ = 1e-5;
}

// ========================================================================
// COMPUTE T2 AMPLITUDES
// ========================================================================
void OMP2::compute_t2_amplitudes() {
    auto eri_ao = integrals_->compute_eri();

    Eigen::MatrixXd Ca_occ = scf_.C_alpha.leftCols(na_);
    Eigen::MatrixXd Ca_virt = scf_.C_alpha.rightCols(va_);
    Eigen::MatrixXd Cb_occ = scf_.C_beta.leftCols(nb_);
    Eigen::MatrixXd Cb_virt = scf_.C_beta.rightCols(vb_);

    // 1. Transform OOVV blocks (Same Spin)
    // Output: (i, a, j, b)
    auto g_aa = ERITransformer::transform_oovv(eri_ao, Ca_occ, Ca_virt, nbf_, na_, va_);
    auto g_bb = ERITransformer::transform_oovv(eri_ao, Cb_occ, Cb_virt, nbf_, nb_, vb_);
    
    // 2. Transform Mixed Spin Block (Alpha-Beta)
    // FIX: Urutan argumen harus (OccA, OccB, VirtA, VirtB) agar sesuai header!
    // Output tensor size: (na, va, nb, vb) -> (i, a, j, b)
    auto g_ab = ERITransformer::transform_oovv_mixed(
        eri_ao, Ca_occ, Cb_occ, Ca_virt, Cb_virt, nbf_, na_, nb_, va_, vb_
    ); 

    // Resize Storage
    t2_aa_ = Eigen::Tensor<double, 4>(na_, na_, va_, va_); t2_aa_.setZero();
    t2_bb_ = Eigen::Tensor<double, 4>(nb_, nb_, vb_, vb_); t2_bb_.setZero();
    t2_ab_ = Eigen::Tensor<double, 4>(na_, nb_, va_, vb_); t2_ab_.setZero();

    const auto& ea = scf_.orbital_energies_alpha;
    const auto& eb = scf_.orbital_energies_beta;

    // --- Alpha-Alpha ---
    #pragma omp parallel for collapse(4)
    for(int i=0; i<na_; ++i) {
        for(int j=0; j<na_; ++j) {
            for(int a=0; a<va_; ++a) {
                for(int b=0; b<va_; ++b) {
                    if (i >= j || a >= b) continue;

                    int am = na_ + a;
                    int bm = na_ + b;
                    double D = ea(i) + ea(j) - ea(am) - ea(bm);
                    
                    if(std::abs(D) < 1e-12) continue;

                    double val = g_aa(i, a, j, b) - g_aa(i, b, j, a);
                    double t = val / D;

                    t2_aa_(i,j,a,b) = t;  t2_aa_(j,i,a,b) = -t;
                    t2_aa_(i,j,b,a) = -t; t2_aa_(j,i,b,a) = t;
                }
            }
        }
    }

    // --- Beta-Beta ---
    #pragma omp parallel for collapse(4)
    for(int i=0; i<nb_; ++i) {
        for(int j=0; j<nb_; ++j) {
            for(int a=0; a<vb_; ++a) {
                for(int b=0; b<vb_; ++b) {
                    if (i >= j || a >= b) continue;

                    int am = nb_ + a;
                    int bm = nb_ + b;
                    double D = eb(i) + eb(j) - eb(am) - eb(bm);
                    
                    if(std::abs(D) < 1e-12) continue;

                    double val = g_bb(i, a, j, b) - g_bb(i, b, j, a);
                    double t = val / D;

                    t2_bb_(i,j,a,b) = t;  t2_bb_(j,i,a,b) = -t;
                    t2_bb_(i,j,b,a) = -t; t2_bb_(j,i,b,a) = t;
                }
            }
        }
    }

    // --- Alpha-Beta ---
    // g_ab akses: (i, a, j, b) sesuai output transform_oovv_mixed
    #pragma omp parallel for collapse(4)
    for(int i=0; i<na_; ++i) {
        for(int j=0; j<nb_; ++j) {
            for(int a=0; a<va_; ++a) {
                for(int b=0; b<vb_; ++b) {
                    int am = na_ + a;
                    int bm = nb_ + b;
                    double D = ea(i) + eb(j) - ea(am) - eb(bm);
                    
                    if(std::abs(D) < 1e-12) continue;

                    double val = g_ab(i, a, j, b);
                    t2_ab_(i,j,a,b) = val / D;
                }
            }
        }
    }
}

// ========================================================================
// COMPUTE MP2 ENERGY
// ========================================================================
double OMP2::compute_mp2_energy_from_t2() {
    double E_mp2 = 0.0;
    const auto& ea = scf_.orbital_energies_alpha;
    const auto& eb = scf_.orbital_energies_beta;

    // AA
    for(int i=0; i<na_; ++i) {
        for(int j=i+1; j<na_; ++j) {
            for(int a=0; a<va_; ++a) {
                for(int b=a+1; b<va_; ++b) {
                    double t = t2_aa_(i,j,a,b);
                    int am = na_ + a; int bm = na_ + b;
                    double D = ea(i) + ea(j) - ea(am) - ea(bm);
                    E_mp2 += t * t * D;
                }
            }
        }
    }

    // BB
    for(int i=0; i<nb_; ++i) {
        for(int j=i+1; j<nb_; ++j) {
            for(int a=0; a<vb_; ++a) {
                for(int b=a+1; b<vb_; ++b) {
                    double t = t2_bb_(i,j,a,b);
                    int am = nb_ + a; int bm = nb_ + b;
                    double D = eb(i) + eb(j) - eb(am) - eb(bm);
                    E_mp2 += t * t * D;
                }
            }
        }
    }

    // AB
    for(int i=0; i<na_; ++i) {
        for(int j=0; j<nb_; ++j) {
            for(int a=0; a<va_; ++a) {
                for(int b=0; b<vb_; ++b) {
                    double t = t2_ab_(i,j,a,b);
                    int am = na_ + a; int bm = nb_ + b;
                    double D = ea(i) + eb(j) - ea(am) - eb(bm);
                    E_mp2 += t * t * D;
                }
            }
        }
    }
    return E_mp2;
}

// ========================================================================
// OPDM BUILDERS
// ========================================================================
void OMP2::build_opdm_alpha() {
    G_oo_alpha_ = Eigen::MatrixXd::Zero(na_, na_);
    G_vv_alpha_ = Eigen::MatrixXd::Zero(va_, va_);

    // G_ij Alpha
    #pragma omp parallel for collapse(2)
    for(int i=0; i<na_; ++i) {
        for(int j=0; j<na_; ++j) {
            double val = 0.0;
            // AA
            for(int k=0; k<na_; ++k) {
                for(int a=0; a<va_; ++a) {
                    for(int b=0; b<va_; ++b) {
                        val += 0.5 * t2_aa_(i,k,a,b) * t2_aa_(j,k,a,b);
                    }
                }
            }
            // AB
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

    // G_ab Alpha
    #pragma omp parallel for collapse(2)
    for(int a=0; a<va_; ++a) {
        for(int b=0; b<va_; ++b) {
            double val = 0.0;
            // AA
            for(int i=0; i<na_; ++i) {
                for(int j=0; j<na_; ++j) {
                    for(int c=0; c<va_; ++c) {
                        val += 0.5 * t2_aa_(i,j,a,c) * t2_aa_(i,j,b,c);
                    }
                }
            }
            // AB
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

void OMP2::build_opdm_beta() {
    G_oo_beta_ = Eigen::MatrixXd::Zero(nb_, nb_);
    G_vv_beta_ = Eigen::MatrixXd::Zero(vb_, vb_);

    // G_ij Beta
    #pragma omp parallel for collapse(2)
    for(int i=0; i<nb_; ++i) {
        for(int j=0; j<nb_; ++j) {
            double val = 0.0;
            // BB
            for(int k=0; k<nb_; ++k) {
                for(int a=0; a<vb_; ++a) {
                    for(int b=0; b<vb_; ++b) {
                        val += 0.5 * t2_bb_(i,k,a,b) * t2_bb_(j,k,a,b);
                    }
                }
            }
            // AB
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

    // G_ab Beta
    #pragma omp parallel for collapse(2)
    for(int a=0; a<vb_; ++a) {
        for(int b=0; b<vb_; ++b) {
            double val = 0.0;
            // BB
            for(int i=0; i<nb_; ++i) {
                for(int j=0; j<nb_; ++j) {
                    for(int c=0; c<vb_; ++c) {
                        val += 0.5 * t2_bb_(i,j,a,c) * t2_bb_(i,j,b,c);
                    }
                }
            }
            // AB (c alpha, a beta)
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

Eigen::MatrixXd OMP2::build_opdm() {
    build_opdm_alpha();
    build_opdm_beta();
    return Eigen::MatrixXd::Zero(nbf_, nbf_); 
}

// ========================================================================
// GENERALIZED FOCK BUILD - FIXED DIMENSION BUG
// ========================================================================

Eigen::MatrixXd OMP2::build_gfock_alpha(const Eigen::MatrixXd& G_mo_alpha) {
    // 1. G_ao_alpha = C_alpha * G_mo_alpha * C_alpha^T
    Eigen::MatrixXd G_ao = scf_.C_alpha * G_mo_alpha * scf_.C_alpha.transpose();
    
    Eigen::MatrixXd P_tot_alpha = scf_.P_alpha + G_ao;
    
    // 3. Reconstruct P_tot_beta for Coulomb
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
                    double uvls = eri(u,v,l,s);
                    double ulvs = eri(u,l,v,s);
                    J += uvls * P_total(l,s);
                    K += ulvs * P_tot_alpha(l,s);
                }
            }
            F_ao(u,v) += J - K;
        }
    }
    return scf_.C_alpha.transpose() * F_ao * scf_.C_alpha;
}

Eigen::MatrixXd OMP2::build_gfock_beta(const Eigen::MatrixXd& G_mo_beta) {
    // Reconstruct full alpha correlation density matrix
    Eigen::MatrixXd G_full_alpha = Eigen::MatrixXd::Zero(nbf_, nbf_);
    G_full_alpha.block(0,0,na_,na_) = G_oo_alpha_;
    G_full_alpha.block(na_,na_,va_,va_) = G_vv_alpha_;
    
    Eigen::MatrixXd G_ao_alpha = scf_.C_alpha * G_full_alpha * scf_.C_alpha.transpose();
    Eigen::MatrixXd P_tot_alpha = scf_.P_alpha + G_ao_alpha;

    // Beta Density
    Eigen::MatrixXd G_ao_beta = scf_.C_beta * G_mo_beta * scf_.C_beta.transpose();
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
                    double uvls = eri(u,v,l,s);
                    double ulvs = eri(u,l,v,s);
                    J += uvls * P_total(l,s);
                    K += ulvs * P_tot_beta(l,s);
                }
            }
            F_ao(u,v) += J - K;
        }
    }
    return scf_.C_beta.transpose() * F_ao * scf_.C_beta;
}

// ------------------------------------------------------------------------
// ORBITAL ROTATION & MAIN LOOP
// ------------------------------------------------------------------------

Eigen::MatrixXd OMP2::compute_orbital_gradient_alpha(const Eigen::MatrixXd& F_mo) {
    return 2.0 * F_mo.block(na_, 0, va_, na_);
}

Eigen::MatrixXd OMP2::compute_orbital_gradient_beta(const Eigen::MatrixXd& F_mo) {
    return 2.0 * F_mo.block(nb_, 0, vb_, nb_);
}

void OMP2::rotate_orbitals_alpha(const Eigen::MatrixXd& w_ai, const Eigen::MatrixXd& F_mo) {
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

void OMP2::rotate_orbitals_beta(const Eigen::MatrixXd& w_ai, const Eigen::MatrixXd& F_mo) {
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

bool OMP2::converged(const Eigen::MatrixXd& w_alpha, const Eigen::MatrixXd& w_beta, double e_new, double e_old) {
    double g_norm = std::sqrt(w_alpha.squaredNorm() + w_beta.squaredNorm());
    double de = std::abs(e_new - e_old);
    return (g_norm < grad_thresh_) && (de < conv_thresh_);
}

OMP2Result OMP2::compute() {
    std::cout << "\n========================================\n";
    std::cout << "     Orbital-Optimized MP2 (OMP2)\n";
    std::cout << "========================================\n";

    bool is_open_shell = (na_ != nb_);
    if (is_open_shell) std::cout << "[INFO] U-OMP2 (Open-Shell)\n";
    else std::cout << "[INFO] R-OMP2 (Closed-Shell)\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Iter    E_Total (Ha)    E_Corr (Ha)     ||Grad||\n";
    std::cout << "----------------------------------------\n";

    double e_total_prev = scf_.energy_total; 
    double e_mp2_corr = 0.0;
    bool is_converged = false;
    int iter = 0;

    while (iter < max_iter_) {
        iter++;

        compute_t2_amplitudes();
        e_mp2_corr = compute_mp2_energy_from_t2();
        
        double e_total_curr = scf_.energy_total + e_mp2_corr;

        build_opdm_alpha();
        build_opdm_beta();

        Eigen::MatrixXd G_full_alpha = Eigen::MatrixXd::Zero(nbf_, nbf_);
        G_full_alpha.block(0,0,na_,na_) = G_oo_alpha_;
        G_full_alpha.block(na_,na_,va_,va_) = G_vv_alpha_;
        
        Eigen::MatrixXd G_full_beta = Eigen::MatrixXd::Zero(nbf_, nbf_);
        G_full_beta.block(0,0,nb_,nb_) = G_oo_beta_;
        G_full_beta.block(nb_,nb_,vb_,vb_) = G_vv_beta_;

        Eigen::MatrixXd F_alpha = build_gfock_alpha(G_full_alpha);
        Eigen::MatrixXd F_beta  = build_gfock_beta(G_full_beta);

        Eigen::MatrixXd w_alpha = compute_orbital_gradient_alpha(F_alpha);
        Eigen::MatrixXd w_beta  = compute_orbital_gradient_beta(F_beta);
        double g_norm = std::sqrt(w_alpha.squaredNorm() + w_beta.squaredNorm());

        std::cout << std::setw(4) << iter << "    "
                  << std::fixed << std::setprecision(8) << e_total_curr << "    "
                  << e_mp2_corr << "    "
                  << std::scientific << std::setprecision(2) << g_norm << "\n";

        if (converged(w_alpha, w_beta, e_total_curr, e_total_prev)) {
            is_converged = true;
            break;
        }

        rotate_orbitals_alpha(w_alpha, F_alpha);
        if (is_open_shell) {
            rotate_orbitals_beta(w_beta, F_beta);
        } else {
            scf_.C_beta = scf_.C_alpha;
        }

        e_total_prev = e_total_curr;
    }

    std::cout << "========================================\n";
    if(is_converged) std::cout << "[SUCCESS] OMP2 Optimization Converged.\n";
    else std::cout << "[WARN] OMP2 reached max iterations.\n";

    OMP2Result res;
    // 1. Data Energi
    res.energy_scf = scf_.energy_total;
    res.energy_total = e_total_prev;
    res.energy_mp2_corr = e_mp2_corr;
    res.converged = is_converged;
    res.iterations = iter;

    // ==========================================
    // 2. Data Orbital (INI YANG HARUS DITAMBAH)
    // ==========================================
    res.n_occ_alpha = na_;
    res.n_occ_beta  = nb_;
    res.n_virt_alpha = va_;
    res.n_virt_beta  = vb_;
    
    // Salin Matrix Orbital & Energi dari SCF internal OMP2
    res.orbital_energies_alpha = scf_.orbital_energies_alpha;
    res.orbital_energies_beta  = scf_.orbital_energies_beta;
    res.C_alpha = scf_.C_alpha;
    res.C_beta  = scf_.C_beta;
    // ==========================================
    
    
    return res;
}

} // namespace mshqc