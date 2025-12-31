/**
 * @file src/mp2/cholesky_omp2.cc
 * @brief Cholesky-Decomposed Orbital-Optimized MP2 (Cholesky-OMP2)
 * @details FIXED: Added Damping & Level Shift to prevent gradient explosion.
 */

#include "mshqc/cholesky_omp2.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <unsupported/Eigen/MatrixFunctions> 

namespace mshqc {

// ========================================================================
// CONSTRUCTOR 1: STANDARD (DECOMPOSE INTERNALLY)
// ========================================================================
CholeskyOMP2::CholeskyOMP2(const Molecule& mol,
                           const BasisSet& basis,
                           std::shared_ptr<IntegralEngine> integrals,
                           const SCFResult& scf_guess,
                           const CholeskyOMP2Config& config)
    : mol_(mol), basis_(basis), integrals_(integrals), scf_(scf_guess),
      config_(config)
{
    nbf_ = static_cast<int>(scf_.C_alpha.rows());
    na_ = scf_.n_occ_alpha;
    nb_ = scf_.n_occ_beta;
    va_ = nbf_ - na_;
    vb_ = nbf_ - nb_;

    // Decompose Internally
    if (config_.print_level > 0) 
        std::cout << "[CholeskyOMP2] Decomposing integrals (Thresh=" << config_.cholesky_threshold << ")...\n";
    
    internal_cholesky_ = std::make_unique<integrals::CholeskyERI>(config_.cholesky_threshold);
    auto eri = integrals_->compute_eri();
    internal_cholesky_->decompose(eri);
    
    // Set pointer to internal
    cholesky_ptr_ = internal_cholesky_.get();
}

// ========================================================================
// CONSTRUCTOR 2: REUSE EXTERNAL VECTORS
// ========================================================================
CholeskyOMP2::CholeskyOMP2(const Molecule& mol,
                           const BasisSet& basis,
                           std::shared_ptr<IntegralEngine> integrals,
                           const SCFResult& scf_guess,
                           const CholeskyOMP2Config& config,
                           const integrals::CholeskyERI& cholesky_vectors)
    : mol_(mol), basis_(basis), integrals_(integrals), scf_(scf_guess),
      config_(config), cholesky_ptr_(&cholesky_vectors) // Point to external
{
    nbf_ = static_cast<int>(scf_.C_alpha.rows());
    na_ = scf_.n_occ_alpha;
    nb_ = scf_.n_occ_beta;
    va_ = nbf_ - na_;
    vb_ = nbf_ - nb_;

    if (cholesky_ptr_->n_vectors() == 0) {
        std::cerr << "[ERROR] CholeskyOMP2 initialized with empty reused vectors!\n";
        exit(1);
    }
}

// ========================================================================
// HELPER: TRANSFORM CHOLESKY VECTORS TO MO BASIS
// ========================================================================
std::vector<Eigen::MatrixXd> CholeskyOMP2::transform_vectors(
    const Eigen::MatrixXd& C_occ, 
    const Eigen::MatrixXd& C_virt,
    int n_occ, int n_virt) 
{
    // Use pointer accessor
    const auto& L_ao = cholesky_ptr_->get_L_vectors();
    int n_vec = L_ao.size();
    
    std::vector<Eigen::MatrixXd> L_mo(n_vec);

    #pragma omp parallel for schedule(dynamic)
    for (int P = 0; P < n_vec; ++P) {
        Eigen::Map<const Eigen::MatrixXd> L_uv(L_ao[P].data(), nbf_, nbf_);
        Eigen::MatrixXd Temp = L_uv * C_virt;
        L_mo[P] = C_occ.transpose() * Temp;
    }
    return L_mo;
}

// ========================================================================
// COMPUTE T2 AMPLITUDES (CHOLESKY VERSION)
// ========================================================================
void CholeskyOMP2::compute_t2_amplitudes() {
    Eigen::MatrixXd Ca_occ = scf_.C_alpha.leftCols(na_);
    Eigen::MatrixXd Ca_virt = scf_.C_alpha.rightCols(va_);
    Eigen::MatrixXd Cb_occ = scf_.C_beta.leftCols(nb_);
    Eigen::MatrixXd Cb_virt = scf_.C_beta.rightCols(vb_);

    auto Q_alpha = transform_vectors(Ca_occ, Ca_virt, na_, va_);
    auto Q_beta  = transform_vectors(Cb_occ, Cb_virt, nb_, vb_);
    
    // Use pointer accessor
    int n_vec = cholesky_ptr_->n_vectors();

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

                    double iajb = 0.0, ibja = 0.0;
                    for(int P=0; P<n_vec; ++P) {
                        iajb += Q_alpha[P](i,a) * Q_alpha[P](j,b);
                        ibja += Q_alpha[P](i,b) * Q_alpha[P](j,a);
                    }

                    int am = na_ + a; int bm = na_ + b;
                    double D = ea(i) + ea(j) - ea(am) - ea(bm);
                    if(std::abs(D) < 1e-12) continue;

                    double t = (iajb - ibja) / D;
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

                    double iajb = 0.0, ibja = 0.0;
                    for(int P=0; P<n_vec; ++P) {
                        iajb += Q_beta[P](i,a) * Q_beta[P](j,b);
                        ibja += Q_beta[P](i,b) * Q_beta[P](j,a);
                    }

                    int am = nb_ + a; int bm = nb_ + b;
                    double D = eb(i) + eb(j) - eb(am) - eb(bm);
                    if(std::abs(D) < 1e-12) continue;

                    double t = (iajb - ibja) / D;
                    t2_bb_(i,j,a,b) = t;  t2_bb_(j,i,a,b) = -t;
                    t2_bb_(i,j,b,a) = -t; t2_bb_(j,i,b,a) = t;
                }
            }
        }
    }

    // --- Alpha-Beta ---
    #pragma omp parallel for collapse(4)
    for(int i=0; i<na_; ++i) {
        for(int j=0; j<nb_; ++j) {
            for(int a=0; a<va_; ++a) {
                for(int b=0; b<vb_; ++b) {
                    double iajb = 0.0;
                    for(int P=0; P<n_vec; ++P) {
                        iajb += Q_alpha[P](i,a) * Q_beta[P](j,b);
                    }
                    int am = na_ + a; int bm = nb_ + b;
                    double D = ea(i) + eb(j) - ea(am) - eb(bm);
                    if(std::abs(D) < 1e-12) continue;
                    t2_ab_(i,j,a,b) = iajb / D;
                }
            }
        }
    }
}

// ========================================================================
// ENERGY & DENSITY BUILDERS
// ========================================================================

double CholeskyOMP2::compute_mp2_energy_from_t2() {
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
                    E_mp2 += t * t * (ea(i) + ea(j) - ea(am) - ea(bm));
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
                    E_mp2 += t * t * (eb(i) + eb(j) - eb(am) - eb(bm));
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
                    E_mp2 += t * t * (ea(i) + eb(j) - ea(am) - eb(bm));
                }
            }
        }
    }
    return E_mp2;
}

void CholeskyOMP2::build_opdm_alpha() {
    G_oo_alpha_ = Eigen::MatrixXd::Zero(na_, na_);
    G_vv_alpha_ = Eigen::MatrixXd::Zero(va_, va_);
    // G_ij Alpha
    #pragma omp parallel for collapse(2)
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
    // G_ab Alpha
    #pragma omp parallel for collapse(2)
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

void CholeskyOMP2::build_opdm_beta() {
    G_oo_beta_ = Eigen::MatrixXd::Zero(nb_, nb_);
    G_vv_beta_ = Eigen::MatrixXd::Zero(vb_, vb_);
    #pragma omp parallel for collapse(2)
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
    #pragma omp parallel for collapse(2)
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

// ========================================================================
// GENERALIZED FOCK BUILD (CHOLESKY VERSION)
// ========================================================================
Eigen::MatrixXd CholeskyOMP2::build_gfock_from_density(
    const Eigen::MatrixXd& P_total_alpha, 
    const Eigen::MatrixXd& P_total_beta,
    bool return_alpha)
{
    auto H_core = integrals_->compute_core_hamiltonian();
    Eigen::MatrixXd P_tot = P_total_alpha + P_total_beta;
    Eigen::MatrixXd P_spin = return_alpha ? P_total_alpha : P_total_beta;
    
    Eigen::MatrixXd J_tot = Eigen::MatrixXd::Zero(nbf_, nbf_);
    Eigen::MatrixXd K_spin = Eigen::MatrixXd::Zero(nbf_, nbf_);
    
    // Use pointer accessor
    const auto& L_vecs = cholesky_ptr_->get_L_vectors();
    int n_vec = L_vecs.size();

    #pragma omp parallel
    {
        Eigen::MatrixXd J_local = Eigen::MatrixXd::Zero(nbf_, nbf_);
        Eigen::MatrixXd K_local = Eigen::MatrixXd::Zero(nbf_, nbf_);
        
        #pragma omp for schedule(dynamic)
        for (int P = 0; P < n_vec; ++P) {
            Eigen::Map<const Eigen::MatrixXd> L(L_vecs[P].data(), nbf_, nbf_);
            double scalar_J = (L.cwiseProduct(P_tot)).sum();
            J_local += scalar_J * L;
            K_local += L * P_spin * L; 
        }

        #pragma omp critical
        {
            J_tot += J_local;
            K_spin += K_local;
        }
    }
    Eigen::MatrixXd F_ao = H_core + J_tot - K_spin;
    const Eigen::MatrixXd& C = return_alpha ? scf_.C_alpha : scf_.C_beta;
    return C.transpose() * F_ao * C;
}

// ========================================================================
// ORBITAL ROTATION & MAIN LOOP (FIXED WITH DAMPING)
// ========================================================================

void CholeskyOMP2::rotate_orbitals_alpha(const Eigen::MatrixXd& w_ai, const Eigen::MatrixXd& F_mo) {
    // [FIX] Stabilitas: Damping & Level Shift
    double damping = 0.5; // Langkah konservatif untuk mencegah divergensi
    double shift = 0.2;   // Level shift untuk menghindari penyebut nol/kecil

    Eigen::MatrixXd kappa = Eigen::MatrixXd::Zero(nbf_, nbf_);
    for(int a=0; a<va_; ++a) {
        for(int i=0; i<na_; ++i) {
            // (E_virt - E_occ) + shift
            double diff = F_mo(na_+a, na_+a) - F_mo(i,i) + shift;
            
            // Proteksi pembagian
            if(std::abs(diff) < 1e-2) diff = (diff >= 0 ? 1e-2 : -1e-2);
            
            // Update langkah rotasi dengan damping
            double val = (w_ai(a,i) / diff) * damping;

            kappa(na_+a, i) = val;
            kappa(i, na_+a) = -val;
        }
    }
    scf_.C_alpha = scf_.C_alpha * (-kappa).exp();
    
    // Update density
    Eigen::MatrixXd C_occ = scf_.C_alpha.leftCols(na_);
    scf_.P_alpha = C_occ * C_occ.transpose();
}

void CholeskyOMP2::rotate_orbitals_beta(const Eigen::MatrixXd& w_ai, const Eigen::MatrixXd& F_mo) {
    // [FIX] Stabilitas: Damping & Level Shift
    double damping = 0.5;
    double shift = 0.2;

    Eigen::MatrixXd kappa = Eigen::MatrixXd::Zero(nbf_, nbf_);
    for(int a=0; a<vb_; ++a) {
        for(int i=0; i<nb_; ++i) {
            double diff = F_mo(nb_+a, nb_+a) - F_mo(i,i) + shift;
            if(std::abs(diff) < 1e-2) diff = (diff >= 0 ? 1e-2 : -1e-2);
            
            double val = (w_ai(a,i) / diff) * damping;

            kappa(nb_+a, i) = val;
            kappa(i, nb_+a) = -val;
        }
    }
    scf_.C_beta = scf_.C_beta * (-kappa).exp();
    
    // Update density
    Eigen::MatrixXd C_occ = scf_.C_beta.leftCols(nb_);
    scf_.P_beta = C_occ * C_occ.transpose();
}

OMP2Result CholeskyOMP2::compute() {
    if(config_.print_level > 0) {
        std::cout << "\n========================================\n";
        std::cout << "     Cholesky Orbital-Optimized MP2\n";
        std::cout << "     Vectors: " << cholesky_ptr_->n_vectors() << "\n";
        std::cout << "========================================\n";
    }

    double e_total_prev = scf_.energy_total;
    double e_mp2_corr = 0.0;
    bool is_converged = false;
    int iter = 0;

    // --- MAIN LOOP ---
    while (iter < config_.max_iterations) {
        iter++;
        compute_t2_amplitudes();
        e_mp2_corr = compute_mp2_energy_from_t2();
        double e_total_curr = scf_.energy_total + e_mp2_corr;

        build_opdm_alpha();
        build_opdm_beta();

        // Build Density
        Eigen::MatrixXd G_full_alpha = Eigen::MatrixXd::Zero(nbf_, nbf_);
        G_full_alpha.block(0,0,na_,na_) = G_oo_alpha_;
        G_full_alpha.block(na_,na_,va_,va_) = G_vv_alpha_;
        Eigen::MatrixXd P_tot_alpha = scf_.P_alpha + (scf_.C_alpha * G_full_alpha * scf_.C_alpha.transpose());

        Eigen::MatrixXd G_full_beta = Eigen::MatrixXd::Zero(nbf_, nbf_);
        G_full_beta.block(0,0,nb_,nb_) = G_oo_beta_;
        G_full_beta.block(nb_,nb_,vb_,vb_) = G_vv_beta_;
        Eigen::MatrixXd P_tot_beta = scf_.P_beta + (scf_.C_beta * G_full_beta * scf_.C_beta.transpose());

        // Build Gradient
        Eigen::MatrixXd F_alpha = build_gfock_from_density(P_tot_alpha, P_tot_beta, true);
        Eigen::MatrixXd F_beta  = build_gfock_from_density(P_tot_alpha, P_tot_beta, false);

        Eigen::MatrixXd w_alpha = 2.0 * F_alpha.block(na_, 0, va_, na_);
        Eigen::MatrixXd w_beta  = 2.0 * F_beta.block(nb_, 0, vb_, nb_);
        double g_norm = std::sqrt(w_alpha.squaredNorm() + w_beta.squaredNorm());

        if(config_.print_level > 0) {
            std::cout << std::setw(4) << iter << "    "
                      << std::fixed << std::setprecision(8) << e_total_curr << "    "
                      << e_mp2_corr << "    "
                      << std::scientific << std::setprecision(2) << g_norm << "\n";
        }

        double de = std::abs(e_total_curr - e_total_prev);
        if (g_norm < config_.gradient_threshold && de < config_.energy_threshold) {
            is_converged = true;
            break;
        }

        rotate_orbitals_alpha(w_alpha, F_alpha);
        if (na_ != nb_) {
            rotate_orbitals_beta(w_beta, F_beta);
        } else {
            scf_.C_beta = scf_.C_alpha;
            scf_.P_beta = scf_.P_alpha;
        }
        e_total_prev = e_total_curr;
    }

    OMP2Result res;
    res.energy_scf = scf_.energy_total; 
    res.energy_total = e_total_prev;
    res.energy_mp2_corr = e_mp2_corr;
    res.converged = is_converged;
    res.iterations = iter;
    
    // Orbital Data
    res.n_occ_alpha = na_;
    res.n_occ_beta = nb_;
    
    // [FIX] ISI DIMENSI VIRTUAL AGAR OMP3 TIDAK CRASH
    res.n_virt_alpha = va_;
    res.n_virt_beta = vb_;
    
    res.C_alpha = scf_.C_alpha;
    res.C_beta = scf_.C_beta;
    res.orbital_energies_alpha = scf_.orbital_energies_alpha;
    res.orbital_energies_beta = scf_.orbital_energies_beta;
    
    return res;
}

} // namespace mshqc