/**
 * @file src/mp2/mp2.cc
 * @brief Implementasi Terpadu BaseMP2, RMP2, dan UMP2
 */
extern "C" {
#include <cint.h>
}
#include <tblis/tblis.h>
#include "mshqc/symmetry/salc_builder.h"
#include "mshqc/mp2.h"
#include "mshqc/integrals/eri_transformer.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <omp.h>

namespace mshqc {
using integrals::ERITransformer;

// ============================================================================
// 1. BASE MP2 (Mesin Induk & Transformasi 3-Pusat)
// ============================================================================
BaseMP2::BaseMP2(const Molecule& mol, const BasisSet& basis, 
                 std::shared_ptr<IntegralEngine> integrals, 
                 const SCFResult& scf_guess,
                 const MP2Config& config,
                 std::shared_ptr<PointGroup> pg,
                 std::shared_ptr<PetiteList> pl)
    : mol_(mol), basis_(basis), integrals_(integrals), scf_(scf_guess),
      config_(config), pg_(pg), pl_(pl) 
{
    nbf_ = static_cast<int>(scf_.C_alpha.rows());
    nocc_a_ = scf_.n_occ_alpha;
    nocc_b_ = scf_.n_occ_beta;
    nvir_a_ = nbf_ - nocc_a_;
    nvir_b_ = nbf_ - nocc_b_;
    
    if (config_.eri_method == "df") config_.use_df = true;
}

void BaseMP2::transform_3center_mo() {
    if (scf_.L_mat.size() == 0) {
        throw std::runtime_error("FATAL: L_mat kosong! Mode DF/Cholesky butuh tensor dari SCF.");
    }
    
    int n_aux = scf_.L_mat.cols();
    
    B_ia_P_alpha_ = Eigen::MatrixXd::Zero(nocc_a_ * nvir_a_, n_aux);
    if (nocc_b_ > 0 && nvir_b_ > 0) {
        B_ia_P_beta_ = Eigen::MatrixXd::Zero(nocc_b_ * nvir_b_, n_aux);
    }
    
    const Eigen::MatrixXd& Ca_occ = scf_.C_alpha.leftCols(nocc_a_);
    const Eigen::MatrixXd& Ca_vir = scf_.C_alpha.rightCols(nvir_a_);
    
    #pragma omp parallel for schedule(dynamic)
    for (int P = 0; P < n_aux; ++P) {
        Eigen::Map<const Eigen::MatrixXd> B_AO(scf_.L_mat.col(P).data(), nbf_, nbf_);
        
        // Transformasi Alpha
        Eigen::MatrixXd B_MO_a = Ca_occ.transpose() * (B_AO * Ca_vir);
        
        // FIX: Manual Flattening (Mencegah mismatch Column-Major Eigen vs Row-Major MP2)
        for(int i = 0; i < nocc_a_; ++i) {
            for(int a = 0; a < nvir_a_; ++a) {
                B_ia_P_alpha_(i * nvir_a_ + a, P) = B_MO_a(i, a);
            }
        }
            
        // Transformasi Beta (jika Open-Shell)
        if (nocc_b_ > 0 && nvir_b_ > 0) {
            const Eigen::MatrixXd& Cb_occ = scf_.C_beta.leftCols(nocc_b_);
            const Eigen::MatrixXd& Cb_vir = scf_.C_beta.rightCols(nvir_b_);
            Eigen::MatrixXd B_MO_b = Cb_occ.transpose() * (B_AO * Cb_vir);
            
            for(int i = 0; i < nocc_b_; ++i) {
                for(int a = 0; a < nvir_b_; ++a) {
                    B_ia_P_beta_(i * nvir_b_ + a, P) = B_MO_b(i, a);
                }
            }
        }
    }
}

// ============================================================================
// 2. RMP2 (Restricted MP2)
// ============================================================================
namespace foundation {

void RMP2::transform_integrals() {
    if (config_.eri_method == "exact") {
        if (config_.print_level > 0) std::cout << "  [RMP2] Transformasi Exact OOVV (O(N^5))...\n";
        auto eri_ao = integrals_->compute_eri();
        const Eigen::MatrixXd& C_occ = scf_.C_alpha.leftCols(nocc_a_);
        const Eigen::MatrixXd& C_virt = scf_.C_alpha.rightCols(nvir_a_);
        
        auto eri_chemist = integrals::ERITransformer::transform_ovov(eri_ao, C_occ, C_virt, nbf_, nocc_a_, nvir_a_);
        Eigen::array<int, 4> shuffle_idxs = {0, 2, 1, 3}; // Chemist -> Physicist
        eri_mo_ = eri_chemist.shuffle(shuffle_idxs);
    } else {
        transform_3center_mo();
    }
}

void RMP2::compute_amplitudes_and_energy() {
    const Eigen::VectorXd& eps = scf_.orbital_energies_alpha;
    t2_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_a_, nvir_a_, nvir_a_);
    
    double e_corr_sum = 0.0;
    bool is_exact = (config_.eri_method == "exact");

    #pragma omp parallel for collapse(2) reduction(+:e_corr_sum) schedule(dynamic)
    for (int i = 0; i < nocc_a_; ++i) {
        for (int j = 0; j < nocc_a_; ++j) {
            if (i < n_frozen_ || j < n_frozen_) continue;
            double e_ij = eps(i) + eps(j);
            
            for (int a = 0; a < nvir_a_; ++a) {
                double den_a = e_ij - eps(nocc_a_ + a);
                for (int b = 0; b < nvir_a_; ++b) {
                    double denom = den_a - eps(nocc_a_ + b);
                    if (std::abs(denom) < 1e-12) {
                        t2_(i, j, a, b) = 0.0; 
                        continue;
                    }

                    double val_iajb = 0.0, val_ibja = 0.0;
                    
                    if (is_exact) {
                        val_iajb = eri_mo_(i, j, a, b);
                        val_ibja = eri_mo_(i, j, b, a);
                    } else {
                        // Dot Product Tensor DF/Cholesky (B_ia * B_jb) on the fly
                        int idx_ia = i * nvir_a_ + a;
                        int idx_jb = j * nvir_a_ + b;
                        int idx_ib = i * nvir_a_ + b;
                        int idx_ja = j * nvir_a_ + a;
                        
                        val_iajb = B_ia_P_alpha_.row(idx_ia).dot(B_ia_P_alpha_.row(idx_jb));
                        val_ibja = B_ia_P_alpha_.row(idx_ib).dot(B_ia_P_alpha_.row(idx_ja));
                    }

                    double t_val = val_iajb / denom;
                    t2_(i, j, a, b) = t_val;
                    e_corr_sum += t_val * (2.0 * val_iajb - val_ibja);
                }
            }
        }
    }
    e_corr_ = e_corr_sum;
}

MP2Result RMP2::compute() {
    auto t1 = std::chrono::high_resolution_clock::now();
    
    transform_integrals();
    compute_amplitudes_and_energy();
    
    auto t2 = std::chrono::high_resolution_clock::now();
    
    MP2Result result;
    result.energy_scf = scf_.energy_total;
    result.energy_mp2_corr = e_corr_;
    result.energy_total = scf_.energy_total + e_corr_;
    
    if (config_.print_level > 0) {
        std::cout << "\n=== RMP2 Results (" << config_.eri_method << ") ===\n";
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "RHF Energy:      " << std::setw(14) << result.energy_scf << " Ha\n";
        std::cout << "MP2 Correlation: " << std::setw(14) << result.energy_mp2_corr << " Ha\n";
        std::cout << "Total RMP2:      " << std::setw(14) << result.energy_total << " Ha\n";
        std::cout << "Time:            " << std::chrono::duration<double>(t2-t1).count() << " s\n";
    }
    
    return result;
}

} // namespace foundation

// ============================================================================
// 3. UMP2 (Unrestricted MP2)
// ============================================================================
void UMP2::transform_integrals() {
    if (config_.eri_method != "exact") {
        transform_3center_mo();
    }
}

double UMP2::compute_ss_alpha() {
    bool is_exact = (config_.eri_method == "exact");
    if (is_exact) {
        auto eri_ao = integrals_->compute_eri();
        const Eigen::MatrixXd& Ca_occ = scf_.C_alpha.leftCols(nocc_a_);
        const Eigen::MatrixXd& Ca_vir = scf_.C_alpha.rightCols(nvir_a_);
        auto eri_chem = integrals::ERITransformer::transform_ovov(eri_ao, Ca_occ, Ca_vir, nbf_, nocc_a_, nvir_a_);
        Eigen::array<int, 4> shuf = {0, 2, 1, 3};
        eri_aaaa_ = eri_chem.shuffle(shuf);
    }
    
    t2_aa_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_a_, nvir_a_, nvir_a_);
    const auto& eo = scf_.orbital_energies_alpha;
    const auto& ev = scf_.orbital_energies_alpha.tail(nvir_a_);
    double e_sum = 0.0;

    #pragma omp parallel for collapse(3) reduction(+:e_sum) schedule(dynamic)
    for(int b = 0; b < nvir_a_; ++b) {
        for(int a = 0; a < nvir_a_; ++a) {
            for(int j = 0; j < nocc_a_; ++j) {
                if (j < n_frozen_) continue;
                double den_partial = -ev(a) - ev(b) + eo(j);
                
                for(int i = 0; i < nocc_a_; ++i) {
                    if (i < n_frozen_) continue;
                    double den = den_partial + eo(i);
                    if (std::abs(den) < 1e-12) { t2_aa_(i, j, a, b) = 0.0; continue; }
                    
                    double val_iajb = 0.0, val_ibja = 0.0;
                    if (is_exact) {
                        val_iajb = eri_aaaa_(i, j, a, b);
                        val_ibja = eri_aaaa_(i, j, b, a);
                    } else {
                        int idx_ia = i * nvir_a_ + a;
                        int idx_jb = j * nvir_a_ + b;
                        int idx_ib = i * nvir_a_ + b;
                        int idx_ja = j * nvir_a_ + a;
                        val_iajb = B_ia_P_alpha_.row(idx_ia).dot(B_ia_P_alpha_.row(idx_jb));
                        val_ibja = B_ia_P_alpha_.row(idx_ib).dot(B_ia_P_alpha_.row(idx_ja));
                    }
                    
                    double val_num = val_iajb - val_ibja;
                    double val_t = val_num / den;
                    t2_aa_(i, j, a, b) = val_t;
                    e_sum += val_t * val_num;
                }
            }
        }
    }
    if (is_exact) eri_aaaa_.resize(0,0,0,0);
    return 0.25 * e_sum;
}

double UMP2::compute_ss_beta() {
    bool is_exact = (config_.eri_method == "exact");
    if (is_exact) {
        auto eri_ao = integrals_->compute_eri();
        const Eigen::MatrixXd& Cb_occ = scf_.C_beta.leftCols(nocc_b_);
        const Eigen::MatrixXd& Cb_vir = scf_.C_beta.rightCols(nvir_b_);
        auto eri_chem = integrals::ERITransformer::transform_ovov(eri_ao, Cb_occ, Cb_vir, nbf_, nocc_b_, nvir_b_);
        Eigen::array<int, 4> shuf = {0, 2, 1, 3};
        eri_bbbb_ = eri_chem.shuffle(shuf);
    }
    
    t2_bb_ = Eigen::Tensor<double, 4>(nocc_b_, nocc_b_, nvir_b_, nvir_b_);
    const auto& eo = scf_.orbital_energies_beta;
    const auto& ev = scf_.orbital_energies_beta.tail(nvir_b_);
    double e_sum = 0.0;

    #pragma omp parallel for collapse(3) reduction(+:e_sum) schedule(dynamic)
    for(int b = 0; b < nvir_b_; ++b) {
        for(int a = 0; a < nvir_b_; ++a) {
            for(int j = 0; j < nocc_b_; ++j) {
                if (j < n_frozen_) continue;
                double den_partial = -ev(a) - ev(b) + eo(j);
                
                for(int i = 0; i < nocc_b_; ++i) {
                    if (i < n_frozen_) continue;
                    double den = den_partial + eo(i);
                    if (std::abs(den) < 1e-12) { t2_bb_(i, j, a, b) = 0.0; continue; }
                    
                    double val_iajb = 0.0, val_ibja = 0.0;
                    if (is_exact) {
                        val_iajb = eri_bbbb_(i, j, a, b);
                        val_ibja = eri_bbbb_(i, j, b, a);
                    } else {
                        int idx_ia = i * nvir_b_ + a;
                        int idx_jb = j * nvir_b_ + b;
                        int idx_ib = i * nvir_b_ + b;
                        int idx_ja = j * nvir_b_ + a;
                        val_iajb = B_ia_P_beta_.row(idx_ia).dot(B_ia_P_beta_.row(idx_jb));
                        val_ibja = B_ia_P_beta_.row(idx_ib).dot(B_ia_P_beta_.row(idx_ja));
                    }
                    
                    double val_num = val_iajb - val_ibja;
                    double val_t = val_num / den;
                    t2_bb_(i, j, a, b) = val_t;
                    e_sum += val_t * val_num;
                }
            }
        }
    }
    if (is_exact) eri_bbbb_.resize(0,0,0,0);
    return 0.25 * e_sum;
}

double UMP2::compute_os() {
    bool is_exact = (config_.eri_method == "exact");
    if (is_exact) {
        auto eri_ao = integrals_->compute_eri();
        const Eigen::MatrixXd& Ca_occ = scf_.C_alpha.leftCols(nocc_a_);
        const Eigen::MatrixXd& Ca_vir = scf_.C_alpha.rightCols(nvir_a_);
        const Eigen::MatrixXd& Cb_occ = scf_.C_beta.leftCols(nocc_b_);
        const Eigen::MatrixXd& Cb_vir = scf_.C_beta.rightCols(nvir_b_);
        auto eri_chem = integrals::ERITransformer::transform_oovv_mixed(eri_ao, Ca_occ, Cb_occ, Ca_vir, Cb_vir, nbf_, nocc_a_, nocc_b_, nvir_a_, nvir_b_);
        Eigen::array<int, 4> shuf = {0, 2, 1, 3};
        eri_aabb_ = eri_chem.shuffle(shuf);
    }
    
    t2_ab_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_b_, nvir_a_, nvir_b_);
    const auto& eoa = scf_.orbital_energies_alpha;
    const auto& eva = scf_.orbital_energies_alpha.tail(nvir_a_);
    const auto& eob = scf_.orbital_energies_beta;
    const auto& evb = scf_.orbital_energies_beta.tail(nvir_b_);
    double e_sum = 0.0;

    #pragma omp parallel for collapse(3) reduction(+:e_sum) schedule(dynamic)
    for(int b = 0; b < nvir_b_; ++b) {
        for(int a = 0; a < nvir_a_; ++a) {
            for(int j = 0; j < nocc_b_; ++j) {
                if (j < n_frozen_) continue;
                double den_partial = -eva(a) - evb(b) + eob(j);
                
                for(int i = 0; i < nocc_a_; ++i) {
                    if (i < n_frozen_) continue;
                    double den = den_partial + eoa(i);
                    if (std::abs(den) < 1e-12) { t2_ab_(i, j, a, b) = 0.0; continue; }
                    
                    double val_num = 0.0;
                    if (is_exact) {
                        val_num = eri_aabb_(i, j, a, b);
                    } else {
                        int idx_ia = i * nvir_a_ + a;
                        int idx_jb = j * nvir_b_ + b;
                        val_num = B_ia_P_alpha_.row(idx_ia).dot(B_ia_P_beta_.row(idx_jb));
                    }
                    
                    double val_t = val_num / den;
                    t2_ab_(i, j, a, b) = val_t;
                    e_sum += val_t * val_num;
                }
            }
        }
    }
    if (is_exact) eri_aabb_.resize(0,0,0,0);
    return e_sum;
}

MP2Result UMP2::compute() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    transform_integrals();
    double e_ss_aa = compute_ss_alpha();
    double e_ss_bb = compute_ss_beta();
    double e_os    = compute_os();
    double e_corr  = e_ss_aa + e_ss_bb + e_os;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    MP2Result result;
    result.energy_scf = scf_.energy_total;
    result.energy_mp2_ss = e_ss_aa + e_ss_bb;
    result.energy_mp2_os = e_os;
    result.energy_mp2_corr = e_corr;
    result.energy_total = scf_.energy_total + e_corr;
    
    if(config_.print_level > 0) {
        std::cout << "\n=== UMP2 Results (" << config_.eri_method << ") ===\n";
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "SS Energy (AA):  " << std::setw(14) << e_ss_aa << " Ha\n";
        std::cout << "SS Energy (BB):  " << std::setw(14) << e_ss_bb << " Ha\n";
        std::cout << "OS Energy (AB):  " << std::setw(14) << e_os << " Ha\n";
        std::cout << "Correlation E:   " << std::setw(14) << e_corr << " Ha\n";
        std::cout << "Total Energy:    " << std::setw(14) << result.energy_total << " Ha\n";
        std::cout << "Time:            " << std::chrono::duration<double>(end_time - start_time).count() << " s\n";
    }
    return result;
}

T2Amplitudes UMP2::get_t2_amplitudes() const { 
    T2Amplitudes amps;
    amps.t2_aa = t2_aa_;
    amps.t2_bb = t2_bb_;
    amps.t2_ab = t2_ab_;
    return amps;
}

// ============================================================================
// 4. OMP2 (Orbital-Optimized MP2)
// ============================================================================



// Ganti deklarasi konstruktor menjadi seperti ini:
OMP2::OMP2(const Molecule& mol, const BasisSet& basis, 
           std::shared_ptr<IntegralEngine> integrals, 
           const SCFResult& scf_guess,
           const MP2Config& config,         
           std::shared_ptr<PointGroup> pg,
           std::shared_ptr<PetiteList> pl)  // <-- exact_2rdm dihapus
    : BaseMP2(mol, basis, integrals, scf_guess, config, pg, pl)
{
    na_  = nocc_a_; 
    nb_  = nocc_b_;
    va_  = nvir_a_; 
    vb_  = nvir_b_;
    n_frozen_ = 0; // Pastikan ini ada

    // Mencegah cetakan -inf di iterasi awal
    e_ss_ = 0.0;
    e_os_ = 0.0;
    
    max_iter_ = config_.max_iterations; 
    conv_thresh_ = config_.energy_threshold; 
   if (config_.gradient_threshold > 0.0) {
        grad_thresh_ = config_.gradient_threshold; 
    } else {
        grad_thresh_ = std::sqrt(config_.energy_threshold); 
    }

    if (pg_ && pl_) {
        symmetrizer_ = std::make_unique<BasisSymmetrizer>(basis_, *pg_, *pl_);
    }
    init_fast_integrals();
}
// ============================================================================
// ROBUST L-BFGS OPTIMIZER (Anti-Explosion & Strict Curvature)
// ============================================================================
struct OrbitalLBFGS {
    int m_max = 6;
    std::vector<Eigen::VectorXd> s_hist;
    std::vector<Eigen::VectorXd> y_hist;
    std::vector<double> rho_hist;
    
    Eigen::VectorXd g_prev;
    Eigen::VectorXd s_prev;
    bool is_first = true;

    void reset() {
        s_hist.clear(); y_hist.clear(); rho_hist.clear();
        is_first = true;
    }

    Eigen::VectorXd get_direction(const Eigen::VectorXd& g_curr, const Eigen::VectorXd& diag_H) {
        if (is_first) {
            g_prev = g_curr;
            is_first = false;
            // Preconditioned Steepest Descent untuk tebakan awal
            return -g_curr.cwiseQuotient(diag_H); 
        }

        Eigen::VectorXd y = g_curr - g_prev;
        Eigen::VectorXd s = s_prev; 
        double ys = y.dot(s);
        
        // STRICT CURVATURE CONDITION (KUNCI ANTI MELEDAK!)
        // L-BFGS hanya boleh mengingat langkah jika kelengkungannya positif tegas.
        if (ys > 1e-8) { 
            if ((int)s_hist.size() >= m_max) {
                s_hist.erase(s_hist.begin());
                y_hist.erase(y_hist.begin());
                rho_hist.erase(rho_hist.begin());
            }
            s_hist.push_back(s);
            y_hist.push_back(y);
            rho_hist.push_back(1.0 / ys);
        } else {
            // Jika masuk ke Saddle Point atau energi naik, BUANG MEMORI LAMA!
            reset();
            g_prev = g_curr;
            is_first = false;
            return -g_curr.cwiseQuotient(diag_H); // Fallback ke Steepest Descent yang aman
        }

        g_prev = g_curr;

        // Two-loop recursion L-BFGS
        Eigen::VectorXd q = g_curr;
        int k = s_hist.size();
        std::vector<double> alpha(k);

        for (int i = k - 1; i >= 0; --i) {
            alpha[i] = rho_hist[i] * s_hist[i].dot(q);
            q -= alpha[i] * y_hist[i];
        }

        // Terapkan Preconditioner (Diagonal Hessian)
        Eigen::VectorXd r = q.cwiseQuotient(diag_H);

        for (int i = 0; i < k; ++i) {
            double beta = rho_hist[i] * y_hist[i].dot(r);
            r += s_hist[i] * (alpha[i] - beta);
        }

        return -r; // Mengembalikan descent direction murni
    }
};
// ------------------------------------------------------------------
// INIT: LINEARIZED MEMORY & SCHWARTZ SCREENING
// ------------------------------------------------------------------
void OMP2::init_fast_integrals() {
    S_ = integrals_->compute_overlap();
    H_core_ = integrals_->compute_core_hamiltonian();
    if (config_.eri_method != "exact") return;

    J_val_.clear(); J_ind_.clear(); J_ptr_.clear();
    K_val_.clear(); K_ind_.clear(); K_ptr_.clear();
    row_map_.clear();
    
    long long est_nnz = (long long)(std::pow(nbf_, 4) * 0.15); 
    J_val_.reserve(est_nnz); J_ind_.reserve(est_nnz);
    K_val_.reserve(est_nnz); K_ind_.reserve(est_nnz);
    row_map_.reserve(nbf_ * nbf_ / 2);
    
    schwarz_ = Eigen::MatrixXd::Zero(nbf_, nbf_);
    J_ptr_.push_back(0);
    K_ptr_.push_back(0);

    auto ERI = integrals_->compute_eri(); 
    const double sparse_threshold = 1e-12;

    int nshells = basis_.n_shells();
    std::vector<int> shell_starts(nshells), shell_sizes(nshells);
    int offset = 0;
    for(int i=0; i<nshells; ++i) {
        shell_starts[i] = offset;
        shell_sizes[i] = basis_.shell(i).n_functions();
        offset += shell_sizes[i];
    }

    std::vector<std::pair<int, int>> shell_pairs;
    // PENTING: Kita paksa dense di sini, jangan pakai pl_
    for(int i=0; i<nshells; ++i) {
        for(int j=0; j<=i; ++j) {
            shell_pairs.push_back({i, j});
        }
    }

    for (const auto& pair_ij : shell_pairs) {
        int sh_i = pair_ij.first; int sh_j = pair_ij.second;
        int istart = shell_starts[sh_i]; int isize = shell_sizes[sh_i];
        int jstart = shell_starts[sh_j]; int jsize = shell_sizes[sh_j];

        for (int i = 0; i < isize; ++i) {
            for (int j = 0; j < jsize; ++j) {
                int mu = istart + i;
                int nu = jstart + j;
                
                if (sh_i == sh_j && nu > mu) continue;

                row_map_.push_back({mu, nu});
                double max_val_row = 0.0;
                
                for (int sig = 0; sig < nbf_; ++sig) {
                    for (int lam = 0; lam < nbf_; ++lam) {
                        int density_idx = lam + sig * nbf_; 
                        double vJ = ERI(mu, nu, lam, sig);
                        if (std::abs(vJ) > sparse_threshold) {
                            J_val_.push_back(vJ);
                            J_ind_.push_back(density_idx);
                            max_val_row = std::max(max_val_row, std::abs(vJ));
                        }
                        double vK = ERI(mu, lam, nu, sig);
                        if (std::abs(vK) > sparse_threshold) {
                            K_val_.push_back(vK);
                            K_ind_.push_back(density_idx);
                        }
                    }
                }
                J_ptr_.push_back(J_val_.size());
                K_ptr_.push_back(K_val_.size());
                
                schwarz_(mu, nu) = std::sqrt(max_val_row);
                schwarz_(nu, mu) = std::sqrt(max_val_row);
            }
        }
    }
}
// ------------------------------------------------------------------
// FAST FOCK BUILD (Using Precomputed Sparse Integrals)
// ------------------------------------------------------------------
void OMP2::build_fock_fast(const Eigen::MatrixXd& P_a, const Eigen::MatrixXd& P_b,
                           Eigen::MatrixXd& F_a, Eigen::MatrixXd& F_b)
{
    // ========================================================================
    // JALUR 1: DENSITY FITTING / CHOLESKY ROUTE
    // ========================================================================
    if (config_.eri_method != "exact") {
        Eigen::MatrixXd P_tot = P_a + P_b;
        Eigen::MatrixXd J_mat = Eigen::MatrixXd::Zero(nbf_, nbf_);
        Eigen::MatrixXd Ka_mat = Eigen::MatrixXd::Zero(nbf_, nbf_);
        Eigen::MatrixXd Kb_mat = Eigen::MatrixXd::Zero(nbf_, nbf_);
        
        int n_chol = scf_.L_mat.cols();
        
        #pragma omp parallel
        {
            Eigen::MatrixXd J_priv  = Eigen::MatrixXd::Zero(nbf_, nbf_);
            Eigen::MatrixXd Ka_priv = Eigen::MatrixXd::Zero(nbf_, nbf_);
            Eigen::MatrixXd Kb_priv = Eigen::MatrixXd::Zero(nbf_, nbf_);
            Eigen::MatrixXd Ta_buf(nbf_, nbf_), Tb_buf(nbf_, nbf_);
            
            #pragma omp for schedule(dynamic)
            for (int K = 0; K < n_chol; ++K) {
                // Ambil vektor L_K dari L_mat_
                Eigen::Map<const Eigen::MatrixXd> L_K(scf_.L_mat.col(K).data(), nbf_, nbf_);
                
                // Bangun Coulomb (J)
                double val_J = (L_K.cwiseProduct(P_tot)).sum();
                J_priv += val_J * L_K;
                
                // Bangun Exchange Alpha (Ka)
                Ta_buf.noalias() = L_K * P_a;
                Ka_priv.noalias() += Ta_buf * L_K;
                
                // Bangun Exchange Beta (Kb) jika open-shell
                if (nb_ > 0) {
                    Tb_buf.noalias() = L_K * P_b;
                    Kb_priv.noalias() += Tb_buf * L_K;
                }
            }
            
            #pragma omp critical
            {
                J_mat += J_priv;
                Ka_mat += Ka_priv;
                if (nb_ > 0) Kb_mat += Kb_priv;
            }
        }
        
        F_a = H_core_ + J_mat - Ka_mat;
        if (nb_ > 0) F_b = H_core_ + J_mat - Kb_mat;
        else F_b = F_a;
        
        return; // Keluar agar tidak mengeksekusi rute EXACT di bawah
    }

    // ========================================================================
    // JALUR 2: EXACT ROUTE (In-Core / Sparse)
    // ========================================================================
    Eigen::MatrixXd P_tot = P_a + P_b;
    double max_P = P_tot.cwiseAbs().maxCoeff(); 
    double threshold = 1e-9; 

    const double* __restrict__ p_dtot = P_tot.data();
    const double* __restrict__ p_da   = P_a.data();
    const double* __restrict__ p_db   = P_b.data();
    
    const double* __restrict__ Jv = J_val_.data(); 
    const int* __restrict__ Ji = J_ind_.data(); 
    const size_t* __restrict__ Jp = J_ptr_.data();
    const double* __restrict__ Kv = K_val_.data(); 
    const int* __restrict__ Ki = K_ind_.data(); 
    const size_t* __restrict__ Kp = K_ptr_.data();

    int n_threads = omp_get_max_threads();
    std::vector<Eigen::MatrixXd> Ga_priv(n_threads, Eigen::MatrixXd::Zero(nbf_, nbf_));
    std::vector<Eigen::MatrixXd> Gb_priv(n_threads, Eigen::MatrixXd::Zero(nbf_, nbf_));

    size_t n_rows = row_map_.size();
    
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t r = 0; r < n_rows; ++r) {
        int mu = row_map_[r].first;
        int nu = row_map_[r].second;
        
        if (schwarz_(mu, nu) * max_P < threshold) continue;

        size_t js = Jp[r]; size_t je = Jp[r+1];
        size_t ks = Kp[r]; size_t ke = Kp[r+1];

        double vj = 0.0;
        #pragma omp simd reduction(+:vj)
        for (size_t k = js; k < je; ++k) vj += Jv[k] * p_dtot[Ji[k]];

        double ka = 0.0, kb = 0.0;
        #pragma omp simd reduction(+:ka, kb)
        for (size_t k = ks; k < ke; ++k) {
            double v = Kv[k]; int idx = Ki[k];
            ka += v * p_da[idx];
            kb += v * p_db[idx];
        }
        
        int tid = omp_get_thread_num();
        Ga_priv[tid](mu, nu) += vj - ka;
        Gb_priv[tid](mu, nu) += vj - kb;
    }

    Eigen::MatrixXd G_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
    Eigen::MatrixXd G_b = Eigen::MatrixXd::Zero(nbf_, nbf_);

    for (int t = 0; t < n_threads; ++t) { G_a += Ga_priv[t]; G_b += Gb_priv[t]; }
    
    for (int i = 0; i < nbf_; ++i) {
        for (int j = 0; j < i; ++j) {
            G_a(j, i) = G_a(i, j); G_b(j, i) = G_b(i, j);
        }
    }

    F_a = H_core_ + G_a;
    F_b = H_core_ + G_b;
}
// ============================================================================
// CORE MP2 LOGIC
// ============================================================================

void OMP2::transform_integrals() {

    if (config_.eri_method != "exact") {
        scf_.irreps_alpha.assign(nbf_, 0);
        scf_.irreps_beta.assign(nbf_, 0);
    }
    if (config_.eri_method == "exact") {
        const auto& eri_ao = integrals_->compute_eri();
        const Eigen::MatrixXd& Ca_o = scf_.C_alpha.leftCols(na_);
        const Eigen::MatrixXd& Ca_v = scf_.C_alpha.rightCols(va_);
        const Eigen::MatrixXd& Cb_o = scf_.C_beta.leftCols(nb_);
        const Eigen::MatrixXd& Cb_v = scf_.C_beta.rightCols(vb_);

        auto occ_spaces_a = get_irrep_spaces(scf_.irreps_alpha, 0, na_);
        auto vir_spaces_a = get_irrep_spaces(scf_.irreps_alpha, na_, va_);

        g_aa_.clear(); g_bb_.clear(); g_ab_.clear();

        if (config_.print_level > 0) std::cout << "  [DEBUG] Memulai Transformasi Integral OOVV (GEMM Harvesting)..." << std::endl;
        
        g_aa_ = ERITransformer::transform_oovv_blocked(eri_ao, Ca_o, Ca_v, occ_spaces_a, vir_spaces_a, nbf_);
        
        bool is_restricted = (na_ == nb_ && va_ == vb_);
        if (is_restricted && nb_ > 0 && vb_ > 0) {
            g_bb_.allocate_block(0, 0, 0, 0, nb_, vb_, nb_, vb_);
            g_ab_.allocate_block(0, 0, 0, 0, na_, va_, nb_, vb_);
            auto* ptr_bb = g_bb_.get_block(0, 0, 0, 0);
            auto* ptr_ab = g_ab_.get_block(0, 0, 0, 0);
            ptr_bb->setZero(); ptr_ab->setZero();
            
            for (const auto& o1 : occ_spaces_a) {
                for (const auto& v1 : vir_spaces_a) {
                    for (const auto& o2 : occ_spaces_a) {
                        for (const auto& v2 : vir_spaces_a) {
                            if ((o1.id ^ v1.id ^ o2.id ^ v2.id) == 0) {
                                auto* g_blk = g_aa_.get_block(o1.id, v1.id, o2.id, v2.id);
                                if (g_blk) {
                                    for (int i=0; i<o1.size; ++i) {
                                        for (int a=0; a<v1.size; ++a) {
                                            for (int j=0; j<o2.size; ++j) {
                                                for (int b=0; b<v2.size; ++b) {
                                                    double val = (*g_blk)(i, a, j, b);
                                                    (*ptr_bb)(o1.offset+i, v1.offset+a, o2.offset+j, v2.offset+b) = val;
                                                    (*ptr_ab)(o1.offset+i, v1.offset+a, o2.offset+j, v2.offset+b) = val;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else if (!is_restricted && nb_ > 0 && vb_ > 0) {
            auto dense_bb = ERITransformer::transform_custom(eri_ao, Cb_o, Cb_v, Cb_o, Cb_v, nbf_, nb_, vb_, nb_, vb_);
            g_bb_.allocate_block(0, 0, 0, 0, nb_, vb_, nb_, vb_);
            *(g_bb_.get_block(0, 0, 0, 0)) = dense_bb;

            auto dense_ab = ERITransformer::transform_custom(eri_ao, Ca_o, Ca_v, Cb_o, Cb_v, nbf_, na_, va_, nb_, vb_);
            g_ab_.allocate_block(0, 0, 0, 0, na_, va_, nb_, vb_);
            *(g_ab_.get_block(0, 0, 0, 0)) = dense_ab;
        }
    } else {
        // ====================================================================
        // JALUR KILAT DENSITY FITTING & CHOLESKY
        // ====================================================================
        
        scf_.C_alpha = C_a_current_;
        scf_.C_beta = C_b_current_;
        transform_3center_mo(); 

        g_aa_.clear(); g_bb_.clear(); g_ab_.clear();

        // FIX: Gunakan blok ID 0 agar sinkron dengan fungsi T2 dan OPDM
        g_aa_.allocate_block(0, 0, 0, 0, na_, va_, na_, va_);
        auto* g_blk = g_aa_.get_block(0, 0, 0, 0);
        
        if (g_blk) {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < na_; ++i) {
                for (int a = 0; a < va_; ++a) {
                    int idx_ia = i * va_ + a;
                    for (int j = 0; j < na_; ++j) {
                        for (int b = 0; b < va_; ++b) {
                            int idx_jb = j * va_ + b;
                            (*g_blk)(i, a, j, b) = B_ia_P_alpha_.row(idx_ia).dot(B_ia_P_alpha_.row(idx_jb));
                        }
                    }
                }
            }
        }

        bool is_restricted = (na_ == nb_ && va_ == vb_);
        if (nb_ > 0 && vb_ > 0) {
            g_bb_.allocate_block(0, 0, 0, 0, nb_, vb_, nb_, vb_);
            auto* ptr_bb = g_bb_.get_block(0, 0, 0, 0);
            
            g_ab_.allocate_block(0, 0, 0, 0, na_, va_, nb_, vb_);
            auto* ptr_ab = g_ab_.get_block(0, 0, 0, 0);
            
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < nb_; ++i) {
                for (int a = 0; a < vb_; ++a) {
                    int idx_ia = i * vb_ + a;
                    for (int j = 0; j < nb_; ++j) {
                        for (int b = 0; b < vb_; ++b) {
                            int idx_jb = j * vb_ + b;
                            (*ptr_bb)(i, a, j, b) = B_ia_P_beta_.row(idx_ia).dot(B_ia_P_beta_.row(idx_jb));
                        }
                    }
                }
            }

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < na_; ++i) {
                for (int a = 0; a < va_; ++a) {
                    int idx_ia = i * va_ + a;
                    for (int j = 0; j < nb_; ++j) {
                        for (int b = 0; b < vb_; ++b) {
                            int idx_jb = j * vb_ + b;
                            (*ptr_ab)(i, a, j, b) = B_ia_P_alpha_.row(idx_ia).dot(B_ia_P_beta_.row(idx_jb));
                        }
                    }
                }
            }
        }
    }
}
void OMP2::pseudocanonicalize() {
    Eigen::MatrixXd F_ao_a, F_ao_b;
    build_fock_fast(scf_.P_alpha, scf_.P_beta, F_ao_a, F_ao_b);
    
    auto diag_block_by_irrep = [&](const Eigen::MatrixXd& F_ao, Eigen::MatrixXd& C, Eigen::VectorXd& eps, int nocc, int nvir, const std::vector<int>& irreps) {
        Eigen::MatrixXd F_mo = C.transpose() * F_ao * C;
        Eigen::MatrixXd U = Eigen::MatrixXd::Zero(nbf_, nbf_);
        
        auto occ_spaces = get_irrep_spaces(irreps, 0, nocc);
        auto vir_spaces = get_irrep_spaces(irreps, nocc, nvir);

        // Diagonalize Occupied blocks
        for (const auto& space : occ_spaces) {
            Eigen::MatrixXd F_sub = F_mo.block(space.offset, space.offset, space.size, space.size);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(F_sub);
            U.block(space.offset, space.offset, space.size, space.size) = es.eigenvectors();
            eps.segment(space.offset, space.size) = es.eigenvalues();
        }
        
        // Diagonalize Virtual blocks
        for (const auto& space : vir_spaces) {
            int off = nocc + space.offset;
            Eigen::MatrixXd F_sub = F_mo.block(off, off, space.size, space.size);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(F_sub);
            U.block(off, off, space.size, space.size) = es.eigenvectors();
            eps.segment(off, space.size) = es.eigenvalues();
        }
        C = C * U;
    };

    bool use_sym = (!scf_.irreps_alpha.empty() && scf_.irreps_alpha[0] != -1);
    
    if (use_sym) {
        diag_block_by_irrep(F_ao_a, scf_.C_alpha, scf_.orbital_energies_alpha, na_, va_, scf_.irreps_alpha);
        if (nb_ > 0 && vb_ > 0) {
            diag_block_by_irrep(F_ao_b, scf_.C_beta, scf_.orbital_energies_beta, nb_, vb_, scf_.irreps_beta);
        }
    } else {
        auto diag_block = [&](const Eigen::MatrixXd& F_ao, Eigen::MatrixXd& C, Eigen::VectorXd& eps, int nocc, int nvir) {
            Eigen::MatrixXd F_mo = C.transpose() * F_ao * C;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_o(F_mo.topLeftCorner(nocc, nocc));
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_v(F_mo.bottomRightCorner(nvir, nvir));
            Eigen::MatrixXd U = Eigen::MatrixXd::Zero(nbf_, nbf_);
            U.topLeftCorner(nocc, nocc) = es_o.eigenvectors();
            U.bottomRightCorner(nvir, nvir) = es_v.eigenvectors();
            C = C * U;
            eps.resize(nbf_);
            eps.head(nocc) = es_o.eigenvalues();
            eps.tail(nvir) = es_v.eigenvalues();
        };
        diag_block(F_ao_a, scf_.C_alpha, scf_.orbital_energies_alpha, na_, va_);
        if (nb_ > 0 && vb_ > 0) diag_block(F_ao_b, scf_.C_beta,  scf_.orbital_energies_beta,  nb_, vb_);
    }
    
    scf_.P_alpha = scf_.C_alpha.leftCols(na_) * scf_.C_alpha.leftCols(na_).transpose();
    if (nb_ > 0) scf_.P_beta  = scf_.C_beta.leftCols(nb_)  * scf_.C_beta.leftCols(nb_).transpose();
}
void OMP2::compute_t2_amplitudes() {
    t2_aa_.clear(); t2_bb_.clear(); t2_ab_.clear();
    int nf = n_frozen_;

    auto occ_spaces_a = get_irrep_spaces(scf_.irreps_alpha, 0, na_);
    auto vir_spaces_a = get_irrep_spaces(scf_.irreps_alpha, na_, va_);

    // 1. ALPHA-ALPHA (Hanya komputasi blok yang sah secara simetri & non-zero)
    for (const auto& o1 : occ_spaces_a) {
        if (o1.size == 0) continue; // <-- FILTER ZERO-SIZE
        for (const auto& v1 : vir_spaces_a) {
            if (v1.size == 0) continue; // <-- FILTER ZERO-SIZE
            for (const auto& o2 : occ_spaces_a) {
                if (o2.size == 0) continue; // <-- FILTER ZERO-SIZE
                for (const auto& v2 : vir_spaces_a) {
                    if (v2.size == 0) continue; // <-- FILTER ZERO-SIZE
                    
                    if ((o1.id ^ v1.id ^ o2.id ^ v2.id) == 0) {
                        auto* g_blk = g_aa_.get_block(o1.id, v1.id, o2.id, v2.id);
                        auto* g_blk_ex = g_aa_.get_block(o1.id, v2.id, o2.id, v1.id);
                        
                        // GUARD MUTLAK: Cegah Out-of-Bounds karena tabrakan dimensi
                        bool ex_valid = false;
                        if (g_blk_ex != nullptr) {
                            if (g_blk_ex->dimension(1) == (Eigen::Index)v2.size && 
                                g_blk_ex->dimension(3) == (Eigen::Index)v1.size) {
                                ex_valid = true;
                            }
                        }
                        
                        if (g_blk && ex_valid) { 
                            t2_aa_.allocate_block(o1.id, v1.id, o2.id, v2.id, o1.size, v1.size, o2.size, v2.size);
                            auto* t_blk = t2_aa_.get_block(o1.id, v1.id, o2.id, v2.id);
                            
                            if (t_blk) {              
                                t_blk->setZero(); // Wajib bersihkan sampah RAM
                            
                                for (int di = 0; di < o1.size; ++di) {
                                    int i_glb = o1.offset + di;
                                    if (i_glb < nf) continue; // FROZEN CORE SKIP
                                    for (int dj = 0; dj < o2.size; ++dj) {
                                        int j_glb = o2.offset + dj;
                                        if (j_glb < nf) continue; // FROZEN CORE SKIP
                                        
                                        double e_ij = scf_.orbital_energies_alpha(i_glb) + scf_.orbital_energies_alpha(j_glb);
                                        for (int da = 0; da < v1.size; ++da) {
                                            double den_a = e_ij - scf_.orbital_energies_alpha(na_ + v1.offset + da);
                                            for (int db = 0; db < v2.size; ++db) {
                                                double den = den_a - scf_.orbital_energies_alpha(na_ + v2.offset + db);
                                                double val = (*g_blk)(di, da, dj, db) - (*g_blk_ex)(di, db, dj, da);
                                                (*t_blk)(di, da, dj, db) = (std::abs(den) > 1e-12) ? val / den : 0.0;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. BETA & MIXED (Open-Shell Dense Fallback)
    if (nb_ > 0 && vb_ > 0) {
        auto* g_bb_blk = g_bb_.get_block(0,0,0,0);
        if (g_bb_blk) {
            // KOREKSI FATAL: T2_bb harus (Occ, Occ, Virt, Virt)
            t2_bb_.allocate_block(0,0,0,0, nb_, nb_, vb_, vb_);
            auto* t_bb_blk = t2_bb_.get_block(0,0,0,0);
           

            for(int i = nf; i < nb_; ++i) for(int j = nf; j < nb_; ++j) {
                double e_ij = scf_.orbital_energies_beta(i) + scf_.orbital_energies_beta(j);
                for(int a=0; a<vb_; ++a) {
                    double den_a = e_ij - scf_.orbital_energies_beta(nb_+a);
                    for(int b=0; b<vb_; ++b) {
                         double den = den_a - scf_.orbital_energies_beta(nb_+b);
                        double val = (*g_bb_blk)(i, a, j, b) - (*g_bb_blk)(i, b, j, a);
                        (*t_bb_blk)(i, j, a, b) = (std::abs(den) > 1e-12) ? val / den : 0.0;
                        
                    }
                }
            }
        }
        
        auto* g_ab_blk = g_ab_.get_block(0,0,0,0);
        if (g_ab_blk) {
            t2_ab_.allocate_block(0,0,0,0, na_, nb_, va_, vb_);
            auto* t_ab_blk = t2_ab_.get_block(0,0,0,0);
            for(int i = nf; i < na_; ++i) for(int j = nf; j < nb_; ++j) {
                double e_ij = scf_.orbital_energies_alpha(i) + scf_.orbital_energies_beta(j);
                for(int a=0; a<va_; ++a) {
                    double den_a = e_ij - scf_.orbital_energies_alpha(na_+a);
                    for(int b=0; b<vb_; ++b) {
                        double den = den_a - scf_.orbital_energies_beta(nb_+b);
                        (*t_ab_blk)(i, j, a, b) = (std::abs(den) > 1e-12) ? (*g_ab_blk)(i, a, j, b) / den : 0.0;
                    }
                }
            }
        }
    }
}
double OMP2::compute_mp2_energy() {
    double E_ss_aa = 0.0, E_ss_bb = 0.0, E_os = 0.0;
    int nf = n_frozen_;
    
    auto occ_spaces_a = get_irrep_spaces(scf_.irreps_alpha, 0, na_);
    auto vir_spaces_a = get_irrep_spaces(scf_.irreps_alpha, na_, va_);

    for (const auto& o1 : occ_spaces_a) {
        if (o1.size == 0) continue; // <-- FILTER ZERO-SIZE
        for (const auto& v1 : vir_spaces_a) {
            if (v1.size == 0) continue; // <-- FILTER ZERO-SIZE
            for (const auto& o2 : occ_spaces_a) {
                if (o2.size == 0) continue; // <-- FILTER ZERO-SIZE
                for (const auto& v2 : vir_spaces_a) {
                    if (v2.size == 0) continue; // <-- FILTER ZERO-SIZE
                    
                    if ((o1.id ^ v1.id ^ o2.id ^ v2.id) != 0) continue;

                    auto* t_blk = t2_aa_.get_block(o1.id, v1.id, o2.id, v2.id);
                    auto* g_blk = g_aa_.get_block(o1.id, v1.id, o2.id, v2.id);
                    auto* g_blk_ex = g_aa_.get_block(o1.id, v2.id, o2.id, v1.id);
                    
                    // GUARD DIMENSI EXCHANGE
                    bool ex_valid = false;
                    if (g_blk_ex != nullptr) {
                        if (g_blk_ex->dimension(1) == (Eigen::Index)v2.size && 
                            g_blk_ex->dimension(3) == (Eigen::Index)v1.size) {
                            ex_valid = true;
                        }
                    }

                    if (t_blk && g_blk && ex_valid) {
                        for (int di = 0; di < o1.size; ++di) {
                            if (o1.offset+di < nf) continue;
                            for (int dj = 0; dj < o2.size; ++dj) {
                                if (o2.offset+dj < nf) continue;
                                for (int da = 0; da < v1.size; ++da) {
                                    for (int db = 0; db < v2.size; ++db) {
                                        double g_val = (*g_blk)(di, da, dj, db) - (*g_blk_ex)(di, db, dj, da);
                                        E_ss_aa += (*t_blk)(di, da, dj, db) * g_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (nb_ > 0 && vb_ > 0) {
        auto* t_bb = t2_bb_.get_block(0,0,0,0);
        auto* g_bb = g_bb_.get_block(0,0,0,0);
        if (t_bb && g_bb) {
            for(int i=nf; i<nb_; ++i) for(int j=nf; j<nb_; ++j) for(int a=0; a<vb_; ++a) for(int b=0; b<vb_; ++b)
                E_ss_bb += (*t_bb)(i, j, a, b) * ((*g_bb)(i, a, j, b) - (*g_bb)(i, b, j, a));
        }
        auto* t_ab = t2_ab_.get_block(0,0,0,0);
        auto* g_ab = g_ab_.get_block(0,0,0,0);
        if (t_ab && g_ab) {
            for(int i=nf; i<na_; ++i) for(int j=nf; j<nb_; ++j) for(int a=0; a<va_; ++a) for(int b=0; b<vb_; ++b)
                E_os += (*t_ab)(i, j, a, b) * (*g_ab)(i, a, j, b);
        }
    }

    e_ss_ = 0.25 * E_ss_aa + 0.25 * E_ss_bb;
    e_os_ = E_os;
    return e_ss_ + e_os_;
}
void OMP2::build_opdm_alpha() {
    G_oo_alpha_ = Eigen::MatrixXd::Zero(na_, na_);
    G_vv_alpha_ = Eigen::MatrixXd::Zero(va_, va_);
    
    auto occ_spaces_a = get_irrep_spaces(scf_.irreps_alpha, 0, na_);
    auto vir_spaces_a = get_irrep_spaces(scf_.irreps_alpha, na_, va_);

    for (const auto& o1 : occ_spaces_a) {
        if (o1.size == 0) continue; // <-- FILTER ZERO-SIZE
        for (const auto& o2 : occ_spaces_a) {
            if (o2.size == 0) continue; // <-- FILTER ZERO-SIZE
            for (const auto& v1 : vir_spaces_a) {
                if (v1.size == 0) continue; // <-- FILTER ZERO-SIZE
                for (const auto& v2 : vir_spaces_a) {
                    if (v2.size == 0) continue; // <-- FILTER ZERO-SIZE
                    
                    if ((o1.id ^ v1.id ^ o2.id ^ v2.id) != 0) continue;

                    auto* t_blk = t2_aa_.get_block(o1.id, v1.id, o2.id, v2.id);
                    if (t_blk) {
                        for (int di = 0; di < o1.size; ++di) {
                            for (int dj = 0; dj < o1.size; ++dj) {
                                double val = 0.0;
                                for (int dk = 0; dk < o2.size; ++dk) {
                                    for (int da = 0; da < v1.size; ++da) {
                                        for (int db = 0; db < v2.size; ++db) {
                                            val += (*t_blk)(di, da, dk, db) * (*t_blk)(dj, da, dk, db);
                                        }
                                    }
                                }
                                G_oo_alpha_(o1.offset+di, o1.offset+dj) -= 0.5 * val;
                            }
                        }
                    }
                }
            }
        }
    }
    
    for (const auto& v1 : vir_spaces_a) {
        if (v1.size == 0) continue; // <-- FILTER ZERO-SIZE
        for (const auto& v2 : vir_spaces_a) {
            if (v2.size == 0) continue; // <-- FILTER ZERO-SIZE
            for (const auto& o1 : occ_spaces_a) {
                if (o1.size == 0) continue; // <-- FILTER ZERO-SIZE
                for (const auto& o2 : occ_spaces_a) {
                    if (o2.size == 0) continue; // <-- FILTER ZERO-SIZE
                    auto* t_blk = t2_aa_.get_block(o1.id, v1.id, o2.id, v2.id);
                    if (t_blk) {
                        for (int da = 0; da < v1.size; ++da) {
                            for (int db = 0; db < v1.size; ++db) {
                                double val = 0.0;
                                for (int di = 0; di < o1.size; ++di) {
                                    for (int dj = 0; dj < o2.size; ++dj) {
                                        for (int dc = 0; dc < v2.size; ++dc) {
                                            val += (*t_blk)(di, da, dj, dc) * (*t_blk)(di, db, dj, dc);
                                        }
                                    }
                                }
                                G_vv_alpha_(v1.offset+da, v1.offset+db) += 0.5 * val;
                            }
                        }
                    }
                }
            }
        }
    }

    if (nb_ > 0 && vb_ > 0) {
        auto* t_ab = t2_ab_.get_block(0,0,0,0);
        if (t_ab) {
            for(int i=0; i<na_; ++i) for(int j=0; j<na_; ++j) {
                double val = 0.0;
                for(int k=0; k<nb_; ++k) for(int a=0; a<va_; ++a) for(int b=0; b<vb_; ++b)
                    val += (*t_ab)(i, k, a, b) * (*t_ab)(j, k, a, b);
                G_oo_alpha_(i, j) -= val;
            }
            for(int a=0; a<va_; ++a) for(int c=0; c<va_; ++c) {
                double val = 0.0;
                for(int i=0; i<na_; ++i) for(int j=0; j<nb_; ++j) for(int b=0; b<vb_; ++b)
                    val += (*t_ab)(i, j, a, b) * (*t_ab)(i, j, c, b);
                G_vv_alpha_(a, c) += val;
            }
        }
    }
}

void OMP2::build_opdm_beta() {
    G_oo_beta_ = Eigen::MatrixXd::Zero(nb_, nb_);
    G_vv_beta_ = Eigen::MatrixXd::Zero(vb_, vb_);
    if (nb_ == 0 || vb_ == 0) return;
    
    auto* t_bb = t2_bb_.get_block(0,0,0,0);
    auto* t_ab = t2_ab_.get_block(0,0,0,0);
    
    if (t_bb) {
        for(int i=0; i<nb_; ++i) for(int j=0; j<nb_; ++j) {
            double val = 0.0;
            for(int k=0; k<nb_; ++k) for(int a=0; a<vb_; ++a) for(int b=0; b<vb_; ++b)
                val += (*t_bb)(i, k, a, b) * (*t_bb)(j, k, a, b);
            G_oo_beta_(i, j) -= 0.5 * val;
        }
        for(int a=0; a<vb_; ++a) for(int c=0; c<vb_; ++c) {
            double val = 0.0;
            for(int i=0; i<nb_; ++i) for(int j=0; j<nb_; ++j) for(int b=0; b<vb_; ++b)
                val += (*t_bb)(i, j, a, b) * (*t_bb)(i, j, c, b);
            G_vv_beta_(a, c) += 0.5 * val;
        }
    }
    if (t_ab) {
        for(int i=0; i<nb_; ++i) for(int j=0; j<nb_; ++j) {
            double val = 0.0;
            for(int k=0; k<na_; ++k) for(int a=0; a<va_; ++a) for(int b=0; b<vb_; ++b)
                val += (*t_ab)(k, i, a, b) * (*t_ab)(k, j, a, b);
            G_oo_beta_(i, j) -= val;
        }
        for(int a=0; a<vb_; ++a) for(int c=0; c<vb_; ++c) {
            double val = 0.0;
            for(int i=0; i<na_; ++i) for(int j=0; j<nb_; ++j) for(int b=0; b<va_; ++b)
                val += (*t_ab)(i, j, b, a) * (*t_ab)(i, j, b, c);
            G_vv_beta_(a, c) += val;
        }
    }
}
// ============================================================================
// 1. ALAM MIKRO (Menghitung Energi & Matrix Gradien)
// ============================================================================
double OMP2::execute_micro_iterations() {
    double old_energy = e_ss_ + e_os_;

    // Sinkronkan objek SCF dengan orbital terbaru hasil rotasi makro
    scf_.C_alpha = C_a_current_;
    scf_.C_beta  = C_b_current_;
    scf_.P_alpha = scf_.C_alpha.leftCols(na_) * scf_.C_alpha.leftCols(na_).transpose();
    scf_.P_beta  = scf_.C_beta.leftCols(nb_)  * scf_.C_beta.leftCols(nb_).transpose();

    // Canonicalize orbital pada blok occupied dan virtual untuk evaluasi MP2
    pseudocanonicalize();
    
    // Simpan kembali orbital yang sudah canonical ke state iterasi
    C_a_current_ = scf_.C_alpha;
    C_b_current_ = scf_.C_beta;

    // Transformasi integral & hitung T2
    transform_integrals();
    compute_t2_amplitudes();
    
    // Hitung energi korelasi MP2 (memperbarui e_ss_ dan e_os_)
    compute_mp2_energy();

    // Bangun matriks densitas korelasi (Relaxed 1-RDM)
    build_opdm_alpha();
    if (nb_ > 0) build_opdm_beta();

    return (e_ss_ + e_os_) - old_energy;
}
void OMP2::build_generalized_fock() {
    // 1. DENSITAS KORELASI PENUH (P_corr)
    Eigen::MatrixXd G_full_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
    G_full_a.block(0, 0, na_, na_) = G_oo_alpha_; 
    G_full_a.block(na_, na_, va_, va_) = G_vv_alpha_;
    Eigen::MatrixXd P_corr_a = scf_.C_alpha * G_full_a * scf_.C_alpha.transpose();
    
    Eigen::MatrixXd G_full_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    Eigen::MatrixXd P_corr_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    if (nb_ > 0) {
        G_full_b.block(0, 0, nb_, nb_) = G_oo_beta_;
        G_full_b.block(nb_, nb_, vb_, vb_) = G_vv_beta_;
        P_corr_b = scf_.C_beta * G_full_b * scf_.C_beta.transpose();
    }

    // 2. REFERENCE FOCK MATRIX (F_HF)
    Eigen::MatrixXd F_HF_ao_a, F_HF_ao_b;
    build_fock_fast(scf_.P_alpha, scf_.P_beta, F_HF_ao_a, F_HF_ao_b);
    Eigen::MatrixXd F_HF_mo_a = scf_.C_alpha.transpose() * F_HF_ao_a * scf_.C_alpha;
    
    Eigen::MatrixXd F_HF_mo_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    if (nb_ > 0) {
        F_HF_mo_b = scf_.C_beta.transpose() * F_HF_ao_b * scf_.C_beta;
    }

    // 3. RESPONSE POTENTIAL G[gamma]
    Eigen::MatrixXd G_gamma_ao_a, G_gamma_ao_b;
    build_fock_fast(P_corr_a, P_corr_b, G_gamma_ao_a, G_gamma_ao_b);
    G_gamma_ao_a -= H_core_;
    if (nb_ > 0) G_gamma_ao_b -= H_core_;
    
    Eigen::MatrixXd G_gamma_mo_a = scf_.C_alpha.transpose() * G_gamma_ao_a * scf_.C_alpha;
    
    Eigen::MatrixXd G_gamma_mo_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    if (nb_ > 0) {
        G_gamma_mo_b = scf_.C_beta.transpose() * G_gamma_ao_b * scf_.C_beta;
    }

    // 4. EXACT 2-RDM CONTRACTION (Z-Matrix)
    Eigen::MatrixXd Z_mat_a = Eigen::MatrixXd::Zero(va_, na_);
    Eigen::MatrixXd Z_mat_b = Eigen::MatrixXd::Zero(vb_, nb_);

    if (config_.eri_method == "exact") {
        Z_mat_a.setZero();
        Z_mat_b.setZero();

        if (scf_.irreps_alpha.empty()) scf_.irreps_alpha.assign(nbf_, 0);
        if (scf_.irreps_beta.empty())  scf_.irreps_beta.assign(nbf_, 0);

        if (config_.print_level > 0) {
            std::cout << "  [DEBUG] Memulai evaluasi Z-Vector O(N^5) MO-Driven (Irrep-Blocked TBLIS)..." << std::endl;
        }

        const auto& eri_ao = integrals_->compute_eri();
        const Eigen::MatrixXd& Ca_o = scf_.C_alpha.leftCols(na_);
        const Eigen::MatrixXd& Ca_v = scf_.C_alpha.rightCols(va_);
        const Eigen::MatrixXd& Cb_o = scf_.C_beta.leftCols(nb_);
        const Eigen::MatrixXd& Cb_v = scf_.C_beta.rightCols(vb_);

        bool has_beta = (nb_ > 0 && vb_ > 0 && !t2_bb_.blocks.empty() && !t2_ab_.blocks.empty());

        auto occ_spaces_a = get_irrep_spaces(scf_.irreps_alpha, 0, na_);
        auto vir_spaces_a = get_irrep_spaces(scf_.irreps_alpha, na_, va_);

        auto ovvv_blk = ERITransformer::transform_ovvv_blocked(eri_ao, Ca_o, Ca_v, occ_spaces_a, vir_spaces_a, nbf_);
        auto ooov_blk = ERITransformer::transform_ooov_blocked(eri_ao, Ca_o, Ca_v, occ_spaces_a, vir_spaces_a, nbf_);

        // =========================================================================
        // JALUR KHUSUS PURE R-OMP2: MENGHINDARI DENSE FALLBACK!
        // =========================================================================
        bool is_restricted = (na_ == nb_ && va_ == vb_);
        BlockedTensor4D T2_spatial;
        BlockedTensor4D* T2_ptr = &t2_aa_;

        if (is_restricted) {
            for (const auto& o1 : occ_spaces_a) {
                for (const auto& v1 : vir_spaces_a) {
                    for (const auto& o2 : occ_spaces_a) {
                        for (const auto& v2 : vir_spaces_a) {
                            if ((o1.id ^ v1.id ^ o2.id ^ v2.id) == 0) {
                                auto* g_blk = g_aa_.get_block(o1.id, v1.id, o2.id, v2.id);
                                auto* g_blk_ex = g_aa_.get_block(o1.id, v2.id, o2.id, v1.id);
                                if (g_blk && g_blk_ex) {
                                    T2_spatial.allocate_block(o1.id, v1.id, o2.id, v2.id, o1.size, v1.size, o2.size, v2.size);
                                    auto* T_blk = T2_spatial.get_block(o1.id, v1.id, o2.id, v2.id);
                                    
                                    for (int di = 0; di < o1.size; ++di) {
                                        if (o1.offset+di < n_frozen_) continue;
                                        for (int dj = 0; dj < o2.size; ++dj) {
                                            if (o2.offset+dj < n_frozen_) continue;
                                            double e_ij = scf_.orbital_energies_alpha(o1.offset+di) + scf_.orbital_energies_alpha(o2.offset+dj);
                                            for (int da = 0; da < v1.size; ++da) {
                                                double den_a = e_ij - scf_.orbital_energies_alpha(na_ + v1.offset+da);
                                                for (int db = 0; db < v2.size; ++db) {
                                                    double den = den_a - scf_.orbital_energies_alpha(na_ + v2.offset+db);
                                                    double v_dir = (*g_blk)(di, da, dj, db);
                                                    double v_ex = (*g_blk_ex)(di, db, dj, da);
                                                    (*T_blk)(di, da, dj, db) = (std::abs(den) > 1e-12) ? (2.0 * v_dir - v_ex) / den : 0.0;
                                                    
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            T2_ptr = &T2_spatial;
        }

       #pragma omp parallel
        {
            Eigen::MatrixXd Z_local = Eigen::MatrixXd::Zero(va_, na_);

            #pragma omp for schedule(dynamic)
            for (int s_i = 0; s_i < occ_spaces_a.size(); ++s_i) {
                const auto& o_i = occ_spaces_a[s_i];
                if (o_i.size == 0) continue; 

                for (const auto& v_a : vir_spaces_a) {
                    if (v_a.size == 0) continue; 
                    if ((o_i.id ^ v_a.id) != 0) continue; 

                    // KITA LANGSUNG LOOP j DAN b DI SINI! (Tanpa loop di dan da)
                    for (const auto& o_j : occ_spaces_a) {
                        if (o_j.size == 0) continue; 
                        
                        for (const auto& v_b : vir_spaces_a) {
                            if (v_b.size == 0) continue; 
                            
                            // --- 1. KONTRAKSI OVVV (TRUE TBLIS C-API) ---
                            for (const auto& v_c : vir_spaces_a) {
                                if (v_c.size == 0) continue; 
                                
                                if ((o_j.id ^ v_c.id ^ v_a.id ^ v_b.id) == 0) {
                                    auto* blk = ovvv_blk.get_block(o_j.id, v_c.id, v_a.id, v_b.id);
                                    auto* t_blk = T2_ptr->get_block(o_i.id, v_b.id, o_j.id, v_c.id);

                                    if (blk && t_blk) {
                                        tblis::tblis_tensor t_T, t_V, t_Z;
                                        
                                        tblis::len_type ni = o_i.size, na = v_a.size, nb = v_b.size, nj = o_j.size, nc = v_c.size;

                                        // Setup Tensor T (Dimensi: i, b, j, c) - POSISI DIPERBAIKI
                                        tblis::len_type len_T[] = {ni, nb, nj, nc};
                                        tblis::stride_type str_T[] = {1, ni, ni*nb, ni*nb*nj};
                                        tblis::tblis_init_tensor_d(&t_T, 4, len_T, t_blk->data(), str_T);

                                        // Setup Tensor V (Dimensi: j, c, a, b) - POSISI DIPERBAIKI
                                        tblis::len_type len_V[] = {nj, nc, na, nb};
                                        tblis::stride_type str_V[] = {1, nj, nj*nc, nj*nc*na};
                                        tblis::tblis_init_tensor_d(&t_V, 4, len_V, blk->data(), str_V);

                                        // Setup Tensor Output Z (Dimensi: i, a) - POSISI DIPERBAIKI
                                        Eigen::MatrixXd Z_temp = Eigen::MatrixXd::Zero(ni, na);
                                        tblis::len_type len_Z[] = {ni, na};
                                        tblis::stride_type str_Z[] = {1, ni};
                                        tblis::tblis_init_tensor_d(&t_Z, 2, len_Z, Z_temp.data(), str_Z);

                                        // MAGIC: Kalikan T(ibjc) * V(jcab) -> Z(ia) secara native
                                        tblis::tblis_tensor_mult(nullptr, nullptr, &t_T, "ibjc", &t_V, "jcab", &t_Z, "ia");

                                        for(int di=0; di<ni; ++di) {
                                            for(int da=0; da<na; ++da) {
                                                Z_local(v_a.offset + da, o_i.offset + di) += Z_temp(di, da);
                                            }
                                        }
                                    }
                                }
                            }
                            
                            // --- 2. KONTRAKSI OOOV (TRUE TBLIS C-API) ---
                            for (const auto& o_k : occ_spaces_a) {
                                if (o_k.size == 0) continue; 
                                
                                if ((o_i.id ^ o_j.id ^ o_k.id ^ v_b.id) == 0) {
                                    auto* blk = ooov_blk.get_block(o_j.id, o_i.id, o_k.id, v_b.id);
                                    auto* t_blk = T2_ptr->get_block(o_j.id, v_a.id, o_k.id, v_b.id);

                                    if (blk && t_blk) {
                                        tblis::tblis_tensor t_V, t_T, t_Z;
                                        tblis::len_type ni = o_i.size, na = v_a.size, nj = o_j.size, nk = o_k.size, nb = v_b.size;

                                        // Setup Tensor V (Dimensi: j, i, k, b) - POSISI DIPERBAIKI
                                        tblis::len_type len_V[] = {nj, ni, nk, nb};
                                        tblis::stride_type str_V[] = {1, nj, nj*ni, nj*ni*nk};
                                        tblis::tblis_init_tensor_d(&t_V, 4, len_V, blk->data(), str_V);

                                        // Setup Tensor T (Dimensi: j, a, k, b) - POSISI DIPERBAIKI
                                        tblis::len_type len_T[] = {nj, na, nk, nb};
                                        tblis::stride_type str_T[] = {1, nj, nj*na, nj*na*nk};
                                        tblis::tblis_init_tensor_d(&t_T, 4, len_T, t_blk->data(), str_T);

                                        // Setup Tensor Output Z (Dimensi: i, a) - POSISI DIPERBAIKI
                                        Eigen::MatrixXd Z_temp = Eigen::MatrixXd::Zero(ni, na);
                                        tblis::len_type len_Z[] = {ni, na};
                                        tblis::stride_type str_Z[] = {1, ni};
                                        tblis::tblis_init_tensor_d(&t_Z, 2, len_Z, Z_temp.data(), str_Z);

                                        // MAGIC: Kalikan V(jikb) * T(jakb) -> Z(ia) secara native
                                        tblis::tblis_tensor_mult(nullptr, nullptr, &t_V, "jikb", &t_T, "jakb", &t_Z, "ia");

                                        for(int di=0; di<ni; ++di) {
                                            for(int da=0; da<na; ++da) {
                                                Z_local(v_a.offset + da, o_i.offset + di) -= Z_temp(di, da); // PENGURANGAN
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            #pragma omp critical
            Z_mat_a += Z_local;
        }

        // --- DENSE FALLBACK HANYA UNTUK MOLEKUL OPEN-SHELL ---
        if (!is_restricted && has_beta) {
            auto ovvv_bb = ERITransformer::transform_custom(eri_ao, Cb_o, Cb_v, Cb_v, Cb_v, nbf_, nb_, vb_, vb_, vb_);
            auto ooov_bb = ERITransformer::transform_custom(eri_ao, Cb_o, Cb_o, Cb_o, Cb_v, nbf_, nb_, nb_, nb_, vb_);
            auto ovvv_ba_aa = ERITransformer::transform_custom(eri_ao, Cb_o, Cb_v, Ca_v, Ca_v, nbf_, nb_, vb_, va_, va_);
            auto ooov_aa_bb = ERITransformer::transform_custom(eri_ao, Ca_o, Ca_o, Cb_o, Cb_v, nbf_, na_, na_, nb_, vb_);
            auto ovvv_ab_bb = ERITransformer::transform_custom(eri_ao, Ca_o, Ca_v, Cb_v, Cb_v, nbf_, na_, va_, vb_, vb_);
            auto ooov_bb_aa = ERITransformer::transform_custom(eri_ao, Cb_o, Cb_o, Ca_o, Ca_v, nbf_, nb_, nb_, na_, va_);

            auto* t_bb_dense = t2_bb_.get_block(0,0,0,0);
            auto* t_ab_dense = t2_ab_.get_block(0,0,0,0);

            if (t_bb_dense && t_ab_dense) {
                #pragma omp parallel for
                for (int i = 0; i < nb_; ++i) {
                    for (int a = 0; a < vb_; ++a) {
                        if ((scf_.irreps_beta[i] ^ scf_.irreps_beta[nb_+a]) != 0) continue; 
                        double z1 = 0.0, z2 = 0.0;
                        for (int j = 0; j < nb_; ++j) {
                            for (int b = 0; b < vb_; ++b) {
                                for (int c = 0; c < vb_; ++c) z1 += (*t_bb_dense)(i, j, b, c) * ovvv_bb(j, c, a, b);
                                for (int k = 0; k < nb_; ++k) z2 += (*t_bb_dense)(j, k, a, b) * ooov_bb(j, i, k, b);
                            }
                        }
                        Z_mat_b(a, i) += z1 - z2;
                    }
                }
                
                #pragma omp parallel for
                for (int i = 0; i < na_; ++i) {
                    for (int a = 0; a < va_; ++a) {
                        if ((scf_.irreps_alpha[i] ^ scf_.irreps_alpha[na_+a]) != 0) continue; 
                        double z1 = 0.0, z2 = 0.0;
                        for (int j = 0; j < nb_; ++j) {
                            for (int b = 0; b < va_; ++b) {
                                for (int c = 0; c < vb_; ++c) z1 += (*t_ab_dense)(i, j, b, c) * ovvv_ba_aa(j, c, a, b);
                            }
                        }
                        for (int j = 0; j < na_; ++j) {
                            for (int b = 0; b < vb_; ++b) {
                                for (int k = 0; k < nb_; ++k) z2 += (*t_ab_dense)(j, k, a, b) * ooov_aa_bb(j, i, k, b);
                            }
                        }
                        Z_mat_a(a, i) += z1 - z2;
                    }
                }

                #pragma omp parallel for
                for (int i = 0; i < nb_; ++i) {
                    for (int a = 0; a < vb_; ++a) {
                        if ((scf_.irreps_beta[i] ^ scf_.irreps_beta[nb_+a]) != 0) continue; 
                        double z1 = 0.0, z2 = 0.0;
                        for (int j = 0; j < na_; ++j) {
                            for (int b = 0; b < vb_; ++b) {
                                for (int c = 0; c < va_; ++c) z1 += (*t_ab_dense)(j, i, c, b) * ovvv_ab_bb(j, c, a, b);
                            }
                        }
                        for (int j = 0; j < nb_; ++j) {
                            for (int b = 0; b < va_; ++b) {
                                for (int k = 0; k < na_; ++k) z2 += (*t_ab_dense)(k, j, b, a) * ooov_bb_aa(j, i, k, b);
                            }
                        }
                        Z_mat_b(a, i) += z1 - z2;
                    }
                }
            }
        } else if (is_restricted) {
            Z_mat_b = Z_mat_a; // Copy langsung dari Alpha
        }
        
        if (config_.print_level > 0) std::cout << "  [DEBUG] Evaluasi Z-Vector Irrep-Blocked Selesai!" << std::endl;
    } else {
        if (config_.print_level > 0) std::cout << "  [DEBUG] Memulai evaluasi Z-Vector O(N^4) Density Fitting/Cholesky..." << std::endl;
        
        bool is_restricted = (na_ == nb_ && va_ == vb_);
        int n_aux = scf_.L_mat.cols();
        
        // 1. FLATTENING TENSOR T2 (Alpha-Alpha, Beta-Beta, Alpha-Beta)
        Eigen::MatrixXd T2_aa = Eigen::MatrixXd::Zero(na_*va_, na_*va_);
        auto* t_aa_blk = t2_aa_.get_block(0,0,0,0);
        if (t_aa_blk) {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < na_; ++i) for (int a = 0; a < va_; ++a)
                for (int j = 0; j < na_; ++j) for (int b = 0; b < va_; ++b)
                    T2_aa(i*va_+a, j*va_+b) = (*t_aa_blk)(i, a, j, b);
        }

        Eigen::MatrixXd T2_ab = Eigen::MatrixXd::Zero(na_*va_, nb_*vb_);
        Eigen::MatrixXd T2_bb = Eigen::MatrixXd::Zero(nb_*vb_, nb_*vb_);
        
        if (nb_ > 0 && vb_ > 0) {
            auto* t_ab_blk = t2_ab_.get_block(0,0,0,0);
            if (t_ab_blk) {
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < na_; ++i) for (int a = 0; a < va_; ++a)
                    for (int j = 0; j < nb_; ++j) for (int b = 0; b < vb_; ++b)
                        T2_ab(i*va_+a, j*vb_+b) = (*t_ab_blk)(i, j, a, b);
            }
            auto* t_bb_blk = t2_bb_.get_block(0,0,0,0);
            if (t_bb_blk) {
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < nb_; ++i) for (int a = 0; a < vb_; ++a)
                    for (int j = 0; j < nb_; ++j) for (int b = 0; b < vb_; ++b)
                        T2_bb(i*vb_+a, j*vb_+b) = (*t_bb_blk)(i, j, a, b);
            }
        }

        // 2. BENTUK INTERMEDIET X (Matriks X_a dan X_b) O(N^4)
        Eigen::MatrixXd X_a = T2_aa * B_ia_P_alpha_;
        if (nb_ > 0 && vb_ > 0) X_a += T2_ab * B_ia_P_beta_;
        
        Eigen::MatrixXd X_b = Eigen::MatrixXd::Zero(nb_*vb_, n_aux);
        if (nb_ > 0 && vb_ > 0) X_b = T2_bb * B_ia_P_beta_ + T2_ab.transpose() * B_ia_P_alpha_;

        // 3. BANGUN TENSOR 3-PUSAT O-O dan V-V
        Eigen::MatrixXd B_oo_a = Eigen::MatrixXd::Zero(na_*na_, n_aux);
        Eigen::MatrixXd B_vv_a = Eigen::MatrixXd::Zero(va_*va_, n_aux);
        Eigen::MatrixXd B_oo_b = Eigen::MatrixXd::Zero(nb_*nb_, n_aux);
        Eigen::MatrixXd B_vv_b = Eigen::MatrixXd::Zero(vb_*vb_, n_aux);

        const Eigen::MatrixXd& Ca_o = scf_.C_alpha.leftCols(na_);
        const Eigen::MatrixXd& Ca_v = scf_.C_alpha.rightCols(va_);
        const Eigen::MatrixXd& Cb_o = scf_.C_beta.leftCols(nb_);
        const Eigen::MatrixXd& Cb_v = scf_.C_beta.rightCols(vb_);

        #pragma omp parallel
        {
            Eigen::MatrixXd priv_oo_a = Eigen::MatrixXd::Zero(na_*na_, n_aux);
            Eigen::MatrixXd priv_vv_a = Eigen::MatrixXd::Zero(va_*va_, n_aux);
            Eigen::MatrixXd priv_oo_b = Eigen::MatrixXd::Zero(nb_*nb_, n_aux);
            Eigen::MatrixXd priv_vv_b = Eigen::MatrixXd::Zero(vb_*vb_, n_aux);
            
            #pragma omp for schedule(dynamic)
            for (int P = 0; P < n_aux; ++P) {
                Eigen::Map<const Eigen::MatrixXd> B_AO(scf_.L_mat.col(P).data(), nbf_, nbf_);
                Eigen::MatrixXd MO_oo_a = Ca_o.transpose() * (B_AO * Ca_o);
                Eigen::MatrixXd MO_vv_a = Ca_v.transpose() * (B_AO * Ca_v);
                
                for(int i=0; i<na_; ++i) for(int j=0; j<na_; ++j) priv_oo_a(i*na_+j, P) = MO_oo_a(i, j);
                for(int a=0; a<va_; ++a) for(int b=0; b<va_; ++b) priv_vv_a(a*va_+b, P) = MO_vv_a(a, b);
                
                if (nb_ > 0 && vb_ > 0) {
                    Eigen::MatrixXd MO_oo_b = Cb_o.transpose() * (B_AO * Cb_o);
                    Eigen::MatrixXd MO_vv_b = Cb_v.transpose() * (B_AO * Cb_v);
                    for(int i=0; i<nb_; ++i) for(int j=0; j<nb_; ++j) priv_oo_b(i*nb_+j, P) = MO_oo_b(i, j);
                    for(int a=0; a<vb_; ++a) for(int b=0; b<vb_; ++b) priv_vv_b(a*vb_+b, P) = MO_vv_b(a, b);
                }
            }
            #pragma omp critical
            {
                B_oo_a += priv_oo_a; B_vv_a += priv_vv_a;
                if (nb_ > 0) { B_oo_b += priv_oo_b; B_vv_b += priv_vv_b; }
            }
        }

        // 4. KONTRAKSI AKHIR Z-VECTOR
        Z_mat_a.setZero();
        if (nb_ > 0) Z_mat_b.setZero();
        
        #pragma omp parallel
        {
            Eigen::MatrixXd Z_loc_a = Eigen::MatrixXd::Zero(va_, na_);
            Eigen::MatrixXd Z_loc_b = Eigen::MatrixXd::Zero(vb_, nb_);
            
            #pragma omp for schedule(dynamic)
            for (int P = 0; P < n_aux; ++P) {
                // Nol-copy Transpose berkat Column-Major Eigen
                Eigen::Map<const Eigen::MatrixXd> XT_a(X_a.col(P).data(), va_, na_);
                Eigen::Map<const Eigen::MatrixXd> V_a(B_vv_a.col(P).data(), va_, va_);
                Eigen::Map<const Eigen::MatrixXd> O_a(B_oo_a.col(P).data(), na_, na_);
                
                Z_loc_a.noalias() += V_a * XT_a - XT_a * O_a;
                
                if (nb_ > 0 && vb_ > 0) {
                    Eigen::Map<const Eigen::MatrixXd> XT_b(X_b.col(P).data(), vb_, nb_);
                    Eigen::Map<const Eigen::MatrixXd> V_b(B_vv_b.col(P).data(), vb_, vb_);
                    Eigen::Map<const Eigen::MatrixXd> O_b(B_oo_b.col(P).data(), nb_, nb_);
                    
                    Z_loc_b.noalias() += V_b * XT_b - XT_b * O_b;
                }
            }
            #pragma omp critical
            {
                Z_mat_a += Z_loc_a;
                if (nb_ > 0) Z_mat_b += Z_loc_b;
            }
        }
        
        if (is_restricted) Z_mat_b = Z_mat_a;
    }

    // 5. ASSEMBLE GENERALIZED FOCK MATRIX
    F_gen_a_ = F_HF_mo_a + G_gamma_mo_a;
    if (na_ > 0 && va_ > 0) {
        Eigen::MatrixXd F_vo_a = F_gen_a_.block(na_, 0, va_, na_);
        Eigen::MatrixXd L_sep_a = G_vv_alpha_ * F_vo_a - F_vo_a * G_oo_alpha_;
        
        F_gen_a_.block(na_, 0, va_, na_) += L_sep_a;
        F_gen_a_.block(0, na_, na_, va_) += L_sep_a.transpose();
        
        // FIX: Langsung tambahkan Z_mat_a (2-RDM Mutlak)
        F_gen_a_.block(na_, 0, va_, na_) += Z_mat_a;
        F_gen_a_.block(0, na_, na_, va_) += Z_mat_a.transpose();
    }

    F_gen_b_ = F_HF_mo_b + G_gamma_mo_b;
    if (nb_ > 0 && vb_ > 0) {
        Eigen::MatrixXd F_vo_b = F_gen_b_.block(nb_, 0, vb_, nb_);
        Eigen::MatrixXd L_sep_b = G_vv_beta_ * F_vo_b - F_vo_b * G_oo_beta_;
        
        F_gen_b_.block(nb_, 0, vb_, nb_) += L_sep_b;
        F_gen_b_.block(0, nb_, nb_, vb_) += L_sep_b.transpose();
        
        // FIX: Langsung tambahkan Z_mat_b (2-RDM Mutlak)
        F_gen_b_.block(nb_, 0, vb_, nb_) += Z_mat_b;
        F_gen_b_.block(0, nb_, nb_, vb_) += Z_mat_b.transpose();
    }

}

// ============================================================================
// DIAGONAL SOSCF / PRECONDITIONED NEWTON STEP
// ============================================================================
Eigen::VectorXd OMP2::compute_soscf_step() {
    int n_params = orbital_gradient_.size();
    if (n_params == 0) return Eigen::VectorXd::Zero(0);

    if (hessian_diag_.size() != n_params) hessian_diag_.resize(n_params);
    int idx = 0;
    
    // Level Shift untuk menjamin Hessian selalu Definit Positif (Super Stabil)
    double grad_norm = orbital_gradient_.norm();
    double level_shift = 0.5 + (grad_norm > 0.05 ? grad_norm * 2.0 : 0.0); 
    
    bool is_restricted = (na_ == nb_ && va_ == vb_);
    
    // 1. EVALUASI DIAGONAL HESSIAN ORBITAL EKSAK (H_ia,ia)
    for (int a = 0; a < va_; ++a) {
        for (int i = 0; i < na_; ++i) {
            double e_diff = scf_.orbital_energies_alpha(na_ + a) - scf_.orbital_energies_alpha(i);
            hessian_diag_(idx++) = 4.0 * std::abs(e_diff) + level_shift; 
        }
    }
    
    if (!is_restricted && nb_ > 0) {
        for (int a = 0; a < vb_; ++a) {
            for (int i = 0; i < nb_; ++i) {
                double e_diff = scf_.orbital_energies_beta(nb_ + a) - scf_.orbital_energies_beta(i);
                hessian_diag_(idx++) = 4.0 * std::abs(e_diff) + level_shift;
            }
        }
    }

    // 2. NEWTON-RAPHSON STEP ( kappa = - H^-1 * g )
    // Karena kita memakai aproksimasi Hessian diagonal, inversnya cukup dengan pembagian elemen-wise
    Eigen::VectorXd kappa = -orbital_gradient_.cwiseQuotient(hessian_diag_);

    // 3. TRUST-REGION RADIUS (Batasi agar step tidak meledak)
    double max_step = 0.15; // Jari-jari maksimum rotasi orbital
    double max_val = kappa.cwiseAbs().maxCoeff();
    if (max_val > max_step) {
        kappa *= (max_step / max_val);
    }

    return kappa;
}
// ============================================================================
// 3. ALAM MAKRO (Ekstraksi Gradien Vektor dengan Akselerasi DIIS)
// ============================================================================
void OMP2::execute_macro_iterations(DIIS& diis_a, DIIS& diis_b, int macro_iter) {
    build_generalized_fock();

    Eigen::MatrixXd F_ao_a = S_ * scf_.C_alpha * F_gen_a_ * scf_.C_alpha.transpose() * S_;
    Eigen::MatrixXd PS_a = scf_.P_alpha * S_;
    Eigen::MatrixXd SP_a = S_ * scf_.P_alpha;
    Eigen::MatrixXd Err_ao_a = F_ao_a * PS_a - SP_a * F_ao_a;

    diis_a.add_iteration(F_ao_a, Err_ao_a, scf_.P_alpha);
    
    // FIX: DIIS hanya diekstrapolasi jika iterasi > 0
    Eigen::MatrixXd F_ext_ao_a = (macro_iter < 1) ? F_ao_a : diis_a.extrapolate();
    F_gen_a_ = scf_.C_alpha.transpose() * F_ext_ao_a * scf_.C_alpha;

    bool is_restricted = (na_ == nb_ && va_ == vb_);
    if (!is_restricted && nb_ > 0) {
        Eigen::MatrixXd F_ao_b = S_ * scf_.C_beta * F_gen_b_ * scf_.C_beta.transpose() * S_;
        Eigen::MatrixXd PS_b = scf_.P_beta * S_;
        Eigen::MatrixXd SP_b = S_ * scf_.P_beta;
        Eigen::MatrixXd Err_ao_b = F_ao_b * PS_b - SP_b * F_ao_b;

        diis_b.add_iteration(F_ao_b, Err_ao_b, scf_.P_beta);
        
        // FIX: Sama untuk beta
        Eigen::MatrixXd F_ext_ao_b = (macro_iter < 1) ? F_ao_b : diis_b.extrapolate();
        F_gen_b_ = scf_.C_beta.transpose() * F_ext_ao_b * scf_.C_beta;
    } else if (is_restricted && nb_ > 0) {
        F_gen_b_ = F_gen_a_; 
    }
    int dim_a = va_ * na_;
    int dim_b = (is_restricted) ? 0 : (nb_ > 0 ? vb_ * nb_ : 0);
    int n_params = dim_a + dim_b;

    if (orbital_gradient_.size() != n_params) orbital_gradient_.resize(n_params);

    int idx = 0;
    bool use_sym = (!scf_.irreps_alpha.empty() && scf_.irreps_alpha[0] != -1);

    if (!is_restricted && nb_ > 0) {
        Eigen::MatrixXd wa = 2.0 * F_gen_a_.block(na_, 0, va_, na_);
        for (int a = 0; a < va_; ++a) {
            for (int i = 0; i < na_; ++i) {
                if (use_sym && (scf_.irreps_alpha[i] ^ scf_.irreps_alpha[na_ + a]) != 0) orbital_gradient_(idx++) = 0.0;
                else orbital_gradient_(idx++) = wa(a, i);
            }
        }
        Eigen::MatrixXd wb = 2.0 * F_gen_b_.block(nb_, 0, vb_, nb_);
        for (int b = 0; b < vb_; ++b) {
            for (int i = 0; i < nb_; ++i) {
                if (use_sym && (scf_.irreps_beta[i] ^ scf_.irreps_beta[nb_ + b]) != 0) orbital_gradient_(idx++) = 0.0;
                else orbital_gradient_(idx++) = wb(b, i);
            }
        }
    } else {
        Eigen::MatrixXd wa = 2.0 * F_gen_a_.block(na_, 0, va_, na_);
        Eigen::MatrixXd wb = 2.0 * F_gen_b_.block(nb_, 0, vb_, nb_);
        Eigen::MatrixXd w_sym = 0.5 * (wa + wb);

        for (int a = 0; a < va_; ++a) {
            for (int i = 0; i < na_; ++i) {
                if (use_sym && (scf_.irreps_alpha[i] ^ scf_.irreps_alpha[na_ + a]) != 0) orbital_gradient_(idx++) = 0.0;
                else orbital_gradient_(idx++) = w_sym(a, i);
            }
        }
    }
}

// ============================================================================
// 4. ROTASI ORBITAL EKSPOENENSIAL
// ============================================================================
void OMP2::apply_orbital_rotation(const Eigen::VectorXd& kappa) {
    if (kappa.norm() < 1e-12) return;

    bool is_restricted = (na_ == nb_ && va_ == vb_);
    int idx = 0;

    int n_mo_a = na_ + va_;
    Eigen::MatrixXd K_a = Eigen::MatrixXd::Zero(n_mo_a, n_mo_a);
    
    // Loop Unpack (HARUS SAMA: a di luar, i di dalam)
    for (int a = 0; a < va_; ++a) {
        for (int i = 0; i < na_; ++i) {
            double val = kappa(idx++);
            K_a(na_ + a, i) = val;
            K_a(i, na_ + a) = -val;
        }
    }
    C_a_current_ = C_a_current_ * K_a.exp();

    int n_mo_b = nb_ + vb_;
    Eigen::MatrixXd K_b = Eigen::MatrixXd::Zero(n_mo_b, n_mo_b);
    
    if (!is_restricted && nb_ > 0) {
        for (int a = 0; a < vb_; ++a) {
            for (int i = 0; i < nb_; ++i) {
                double val = kappa(idx++);
                K_b(nb_ + a, i) = val;
                K_b(i, nb_ + a) = -val;
            }
        }
    } else if (is_restricted && nb_ > 0) {
        K_b = K_a; // Jika restricted, copy kembar persis dari Alpha
    }

    if (nb_ > 0) {
        C_b_current_ = C_b_current_ * K_b.exp();
    }
}



MP2Result OMP2::compute() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    C_a_current_ = scf_.C_alpha;
    C_b_current_ = scf_.C_beta;
    
    double e_total_best = 1e99;
    double e_corr_best = 0.0;
    double e_total_last = 1e99;
    
    double current_step = 1.0;
    Eigen::MatrixXd C_a_last = scf_.C_alpha;
    Eigen::MatrixXd C_b_last = scf_.C_beta;
    Eigen::VectorXd last_kappa = Eigen::VectorXd::Zero(nbf_ * nbf_); 
    
    // Asumsi struct OrbitalLBFGS sudah Anda copy dari omp2.cc lama ke mp2.cc (lihat Langkah 4 di bawah)
    OrbitalLBFGS lbfgs_engine;
    DIIS diis_alpha(6);
    DIIS diis_beta(6);
    
    if(omp_get_thread_num() == 0) {
        std::cout << "\n========================================================\n";
        std::cout << "      Orbital-Optimized MP2 (OMP2)\n";
        std::cout << "      Macro-Micro Robust L-BFGS with Dynamic Shift\n";
        std::cout << "========================================================\n";
    }

    bool is_converged = false;
    int macro_iter = 0;

    while (macro_iter < config_.max_iterations) {
        execute_micro_iterations();
        
        Eigen::MatrixXd F_ao_a, F_ao_b;
        build_fock_fast(scf_.P_alpha, scf_.P_beta, F_ao_a, F_ao_b);
        
        double e_scf = 0.5 * (scf_.P_alpha.cwiseProduct(H_core_ + F_ao_a).sum() + 
                              scf_.P_beta.cwiseProduct(H_core_ + F_ao_b).sum()) 
                       + mol_.nuclear_repulsion_energy();
                       
        double e_mp2_corr = e_ss_ + e_os_;
        double e_tot = e_scf + e_mp2_corr;

        if (macro_iter > 0 && e_tot > e_total_last + 1e-7) {
            current_step *= 0.5; 
            lbfgs_engine.reset();
            diis_alpha.clear(); diis_beta.clear();
            
            C_a_current_ = C_a_last; C_b_current_ = C_b_last;
            scf_.C_alpha = C_a_last; scf_.C_beta  = C_b_last;
            
            Eigen::VectorXd actual_step = last_kappa * current_step;
            lbfgs_engine.s_prev = actual_step; 
            apply_orbital_rotation(actual_step);
            continue; 
        }

        current_step = std::min(1.0, current_step * 1.2); 
        e_total_last = e_tot;
        C_a_last = C_a_current_; C_b_last = C_b_current_;

        if (e_tot < e_total_best) { e_total_best = e_tot; e_corr_best = e_mp2_corr; }

        execute_macro_iterations(diis_alpha, diis_beta, macro_iter);
        double grad_norm = orbital_gradient_.norm();

        if(omp_get_thread_num() == 0) {
            std::cout << std::setw(4) << macro_iter << "    " 
                      << std::fixed << std::setprecision(8) << e_tot << "    "
                      << std::setprecision(8) << e_mp2_corr << "    " 
                      << std::scientific << std::setprecision(2) << grad_norm << "\n";
        }

        if (macro_iter > 0 && std::abs(e_total_last - e_tot) < conv_thresh_ && grad_norm < grad_thresh_) {
            is_converged = true; break;
        }

        int n_params = orbital_gradient_.size();
        Eigen::VectorXd diag_H(n_params);
        int idx = 0;
        
        double level_shift = (grad_norm > 0.1) ? 0.05 : 0.005;
        bool is_restricted = (na_ == nb_ && va_ == vb_);
        
        for (int a = 0; a < va_; ++a) {
            for (int i = 0; i < na_; ++i) {
                diag_H(idx++) = 4.0 * std::abs(scf_.orbital_energies_alpha(na_ + a) - scf_.orbital_energies_alpha(i)) + level_shift; 
            }
        }
        if (!is_restricted && nb_ > 0) {
            for (int a = 0; a < vb_; ++a) {
                for (int i = 0; i < nb_; ++i) {
                    diag_H(idx++) = 4.0 * std::abs(scf_.orbital_energies_beta(nb_ + a) - scf_.orbital_energies_beta(i)) + level_shift;
                }
            }
        }

        Eigen::VectorXd kappa = lbfgs_engine.get_direction(orbital_gradient_, diag_H);
        double max_val = kappa.cwiseAbs().maxCoeff();
        if (max_val > 0.35) kappa *= (0.35 / max_val);

        last_kappa = kappa;
        apply_orbital_rotation(kappa * current_step);
        macro_iter++;
    }

    MP2Result res;
    res.energy_total = e_total_best;
    res.energy_mp2_corr = e_corr_best;
    res.energy_mp2_ss = e_ss_; 
    res.energy_mp2_os = e_os_;
    res.energy_scf = e_total_best - e_corr_best; 
    res.converged = is_converged;
    res.iterations = macro_iter;
    res.C_alpha = C_a_last; 
    res.C_beta = C_b_last;
    res.orbital_energies_alpha = scf_.orbital_energies_alpha;
    res.orbital_energies_beta  = scf_.orbital_energies_beta;

    return res;
}

// Stubs for interface compatibility
void OMP2::reset_diis() {}
Eigen::MatrixXd OMP2::build_opdm() { return G_oo_alpha_ + G_oo_beta_; } 
Eigen::MatrixXd OMP2::extrapolate_diis(std::vector<Eigen::MatrixXd>&, std::vector<Eigen::MatrixXd>&) { return Eigen::MatrixXd(); }

} // namespace mshqc
