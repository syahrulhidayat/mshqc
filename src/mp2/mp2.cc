/**
 * @file src/mp2/mp2.cc
 * @brief Implementasi Terpadu BaseMP2, RMP2, dan UMP2
 */

#include <tblis/tblis.h>
#include "mshqc/symmetry/salc_builder.h"
#include "mshqc/mp2/mp2.h"
#include "mshqc/integrals/eri_transformer.h"
#include "mshqc/gradient/optimizer.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <omp.h>

namespace mshqc {
using integrals::ERITransformer;

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
BaseMP2::~BaseMP2() = default;

MP2Result BaseMP2::compute() {
    return MP2Result();
}
void BaseMP2::transform_3center_mo() {
    int n_aux = scf_.L_mat.cols();
    bool is_restricted = (nocc_a_ == nocc_b_ && nvir_a_ == nvir_b_ && scf_.n_occ_alpha == scf_.n_occ_beta);
    B_ia_P_alpha_ = Eigen::MatrixXd::Zero(nocc_a_ * nvir_a_, n_aux);

    const Eigen::MatrixXd& Ca_occ = scf_.C_alpha.leftCols(nocc_a_);
    const Eigen::MatrixXd& Ca_vir = scf_.C_alpha.rightCols(nvir_a_);
    Eigen::Map<const Eigen::MatrixXd> L_flat(scf_.L_mat.data(), nbf_, nbf_ * n_aux);

    Eigen::MatrixXd X_a = Ca_vir.transpose() * L_flat; 
    #pragma omp parallel for schedule(static)
    for (int P = 0; P < n_aux; ++P) {
        Eigen::Map<Eigen::MatrixXd> X_P(X_a.data() + P * nvir_a_ * nbf_, nvir_a_, nbf_);
        Eigen::MatrixXd B_MO_a = X_P * Ca_occ; 
        
        for (int i = 0; i < nocc_a_; ++i) {
            for (int a = 0; a < nvir_a_; ++a) {
                B_ia_P_alpha_(i * nvir_a_ + a, P) = B_MO_a(a, i);
            }
        }
    }

    if (!is_restricted && nocc_b_ > 0 && nvir_b_ > 0) {
        B_ia_P_beta_ = Eigen::MatrixXd::Zero(nocc_b_ * nvir_b_, n_aux);
        const Eigen::MatrixXd& Cb_occ = scf_.C_beta.leftCols(nocc_b_);
        const Eigen::MatrixXd& Cb_vir = scf_.C_beta.rightCols(nvir_b_);
        Eigen::MatrixXd X_b = Cb_vir.transpose() * L_flat;

        #pragma omp parallel for schedule(static)
        for (int P = 0; P < n_aux; ++P) {
            Eigen::Map<Eigen::MatrixXd> X_P_b(X_b.data() + P * nvir_b_ * nbf_, nvir_b_, nbf_);
            Eigen::MatrixXd B_MO_b = X_P_b * Cb_occ;

            for (int i = 0; i < nocc_b_; ++i) {
                for (int a = 0; a < nvir_b_; ++a) {
                    B_ia_P_beta_(i * nvir_b_ + a, P) = B_MO_b(a, i);
                }
            }
        }
    }
}

namespace foundation {

void RMP2::transform_integrals() {
    if (config_.eri_method == "exact") {
        if (config_.print_level > 0) std::cout << "  [RMP2] Transformasi Exact OOVV (O(N^5))...\n";
        auto eri_ao = integrals_->compute_eri();
        const Eigen::MatrixXd& C_occ = scf_.C_alpha.leftCols(nocc_a_);
        const Eigen::MatrixXd& C_virt = scf_.C_alpha.rightCols(nvir_a_);

        auto eri_chemist = integrals::ERITransformer::transform_ovov(eri_ao, C_occ, C_virt, nbf_, nocc_a_, nvir_a_);
        Eigen::array<int, 4> shuffle_idxs = {0, 2, 1, 3}; 
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
    result.t2_aa = t2_; 

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

} 

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

                   
                    double safe_den = (std::abs(den) < config_.level_shift) 
                                    ? std::copysign(config_.level_shift, den) 
                                    : den;

                   
                    double val_t = val_num / safe_den;
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
    result.t2_aa = t2_aa_;
    result.t2_bb = t2_bb_;
    result.t2_ab = t2_ab_;
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

OMP2::OMP2(const Molecule& mol, const BasisSet& basis, 
           std::shared_ptr<IntegralEngine> integrals, 
           const SCFResult& scf_guess,
           const MP2Config& config,         
           std::shared_ptr<PointGroup> pg,
           std::shared_ptr<PetiteList> pl)  
    : BaseMP2(mol, basis, integrals, scf_guess, config, pg, pl)
{
    na_  = nocc_a_; 
    nb_  = nocc_b_;
    va_  = nvir_a_; 
    vb_  = nvir_b_;
    n_frozen_ = 0; 

    e_ss_ = 0.0;
    e_os_ = 0.0;

    max_iter_ = config_.max_iterations; 
    conv_thresh_ = config_.energy_threshold; 
   if (config_.gradient_threshold > 0.0) {
        grad_thresh_ = config_.gradient_threshold; 
    } else {
        grad_thresh_ = std::sqrt(config_.energy_threshold); 
    }

    symmetrizer_ = nullptr;
    init_fast_integrals();
}
struct OrbitalLBFGS {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int m_max = 6;
    std::vector<Eigen::VectorXd> s_hist;
    std::vector<Eigen::VectorXd> y_hist;
    std::vector<double> rho_hist;
    std::array<double, 20> alpha_buffer;

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
            return -g_curr.cwiseQuotient(diag_H); 
        }

        Eigen::VectorXd y = g_curr - g_prev;
        Eigen::VectorXd s = s_prev; 
        double ys = y.dot(s);

        Eigen::VectorXd Bs = s.cwiseProduct(diag_H); 
        double sBs = s.dot(Bs);

        double theta = 1.0;
        if (ys < 0.2 * sBs) {
            theta = (0.8 * sBs) / (sBs - ys);
        }

        Eigen::VectorXd y_mod = theta * y + (1.0 - theta) * Bs;
        double ys_mod = y_mod.dot(s);

        if (ys_mod > 1e-12) { 
            if ((int)s_hist.size() >= m_max) {
                s_hist.erase(s_hist.begin());
                y_hist.erase(y_hist.begin());
                rho_hist.erase(rho_hist.begin());
            }
            s_hist.push_back(s);
            y_hist.push_back(y_mod); 
            rho_hist.push_back(1.0 / ys_mod);
        }

        g_prev = g_curr;

        Eigen::VectorXd q = g_curr;
        int k = s_hist.size();
        std::vector<double> alpha(k);

        for (int i = k - 1; i >= 0; --i) {
            alpha[i] = rho_hist[i] * s_hist[i].dot(q);
            q -= alpha[i] * y_hist[i];
        }

        Eigen::VectorXd r = q.cwiseQuotient(diag_H);

        for (int i = 0; i < k; ++i) {
            double beta = rho_hist[i] * y_hist[i].dot(r);
            r += s_hist[i] * (alpha[i] - beta);
        }

        return -r; 
    }
};


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
    const double sparse_threshold = 0;

    int nshells = basis_.n_shells();
    std::vector<int> shell_starts(nshells), shell_sizes(nshells);
    int offset = 0;
    for(int i=0; i<nshells; ++i) {
        shell_starts[i] = offset;
        shell_sizes[i] = basis_.shell(i).n_functions();
        offset += shell_sizes[i];
    }

    std::vector<std::pair<int, int>> shell_pairs;

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

void OMP2::transform_integrals() {
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);

    if (config_.eri_method == "exact") {
        const auto& eri_ao = integrals_->compute_eri();
        const Eigen::MatrixXd& Ca_o = scf_.C_alpha.leftCols(na_);
        const Eigen::MatrixXd& Ca_v = scf_.C_alpha.rightCols(va_);

        g_aa_.clear(); g_bb_.clear(); g_ab_.clear();

        if (config_.print_level > 0) std::cout << "  [DEBUG] Memulai Transformasi Integral OOVV (HPC Dense TBLIS)..." << std::endl;

        auto dense_aa = integrals::ERITransformer::transform_custom(eri_ao, Ca_o, Ca_v, Ca_o, Ca_v, nbf_, na_, va_, na_, va_);
        g_aa_.allocate_block(0, 0, 0, 0, na_, va_, na_, va_);
        *(g_aa_.get_block(0, 0, 0, 0)) = dense_aa;

        if (!is_restricted && nb_ > 0 && vb_ > 0) {
            const Eigen::MatrixXd& Cb_o = scf_.C_beta.leftCols(nb_);
            const Eigen::MatrixXd& Cb_v = scf_.C_beta.rightCols(vb_);

            auto dense_bb = integrals::ERITransformer::transform_custom(eri_ao, Cb_o, Cb_v, Cb_o, Cb_v, nbf_, nb_, vb_, nb_, vb_);
            g_bb_.allocate_block(0, 0, 0, 0, nb_, vb_, nb_, vb_);
            *(g_bb_.get_block(0, 0, 0, 0)) = dense_bb;

            auto dense_ab = integrals::ERITransformer::transform_custom(eri_ao, Ca_o, Ca_v, Cb_o, Cb_v, nbf_, na_, va_, nb_, vb_);
            g_ab_.allocate_block(0, 0, 0, 0, na_, va_, nb_, vb_);
            *(g_ab_.get_block(0, 0, 0, 0)) = dense_ab;
        }
    } else {
        scf_.C_alpha = C_a_current_;
        if (!is_restricted) scf_.C_beta = C_b_current_;
        
        transform_3center_mo(); 

        g_aa_.clear(); g_bb_.clear(); g_ab_.clear();

        g_aa_.allocate_block(0, 0, 0, 0, na_, va_, na_, va_);
        auto* g_blk = g_aa_.get_block(0, 0, 0, 0);

        if (g_blk) {
            Eigen::MatrixXd G_tmp_aa = B_ia_P_alpha_ * B_ia_P_alpha_.transpose();
            #pragma omp parallel for collapse(2) schedule(static)
            for (int j = 0; j < na_; ++j) {
                for (int b = 0; b < va_; ++b) {
                    int idx_jb = j * va_ + b;
                    for (int i = 0; i < na_; ++i) {
                        for (int a = 0; a < va_; ++a) {
                            int idx_ia = i * va_ + a;
                            (*g_blk)(i, a, j, b) = G_tmp_aa(idx_ia, idx_jb);
                        }
                    }
                }
            }
        }
        
        if (!is_restricted && nb_ > 0 && vb_ > 0) {
            g_bb_.allocate_block(0, 0, 0, 0, nb_, vb_, nb_, vb_);
            auto* ptr_bb = g_bb_.get_block(0, 0, 0, 0);

            g_ab_.allocate_block(0, 0, 0, 0, na_, va_, nb_, vb_);
            auto* ptr_ab = g_ab_.get_block(0, 0, 0, 0);

            Eigen::MatrixXd G_tmp_bb = B_ia_P_beta_ * B_ia_P_beta_.transpose();
            Eigen::MatrixXd G_tmp_ab = B_ia_P_alpha_ * B_ia_P_beta_.transpose();

            #pragma omp parallel for collapse(2) schedule(static)
            for (int j = 0; j < nb_; ++j) {
                for (int b = 0; b < vb_; ++b) {
                    int idx_jb = j * vb_ + b;
                    for (int i = 0; i < nb_; ++i) {
                        for (int a = 0; a < vb_; ++a) {
                            int idx_ia = i * vb_ + a;
                            (*ptr_bb)(i, a, j, b) = G_tmp_bb(idx_ia, idx_jb);
                        }
                    }
                }
            }

            #pragma omp parallel for collapse(2) schedule(static)
            for (int j = 0; j < nb_; ++j) {
                for (int b = 0; b < vb_; ++b) {
                    int idx_jb = j * vb_ + b;
                    for (int i = 0; i < na_; ++i) {
                        for (int a = 0; a < va_; ++a) {
                            int idx_ia = i * va_ + a;
                            (*ptr_ab)(i, a, j, b) = G_tmp_ab(idx_ia, idx_jb);
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

    auto diag_block = [&](const Eigen::MatrixXd& F_ao, Eigen::MatrixXd& C, Eigen::VectorXd& eps, int nocc, int nvir) {
     
        Eigen::MatrixXd C_occ = C.leftCols(nocc);
        Eigen::MatrixXd C_vir = C.rightCols(nvir);

        Eigen::MatrixXd F_oo(nocc, nocc);
        Eigen::MatrixXd F_vv(nvir, nvir);
        F_oo.noalias() = C_occ.transpose() * (F_ao * C_occ);
        F_vv.noalias() = C_vir.transpose() * (F_ao * C_vir);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_o(F_oo);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_v(F_vv);
        C.leftCols(nocc).noalias() = C_occ * es_o.eigenvectors();
        C.rightCols(nvir).noalias() = C_vir * es_v.eigenvectors();

        // 5. Pembaruan energi orbital
        eps.resize(nocc + nvir);
        eps.head(nocc) = es_o.eigenvalues();
        eps.tail(nvir) = es_v.eigenvalues();
    };
    diag_block(F_ao_a, scf_.C_alpha, scf_.orbital_energies_alpha, na_, va_);
    scf_.P_alpha.noalias() = scf_.C_alpha.leftCols(na_) * scf_.C_alpha.leftCols(na_).transpose();

    if (nb_ > 0 && vb_ > 0) {
        diag_block(F_ao_b, scf_.C_beta, scf_.orbital_energies_beta, nb_, vb_);
        scf_.P_beta.noalias() = scf_.C_beta.leftCols(nb_) * scf_.C_beta.leftCols(nb_).transpose();
    }
}


void OMP2::transform_3center_mo_cholesky() {
    int n_chol = scf_.L_mat.cols();
    B_ia_P_alpha_ = Eigen::MatrixXd::Zero(na_ * va_, n_chol);
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    bool has_beta = (!is_restricted && nb_ > 0 && vb_ > 0);
    if (has_beta) {
        B_ia_P_beta_ = Eigen::MatrixXd::Zero(nb_ * vb_, n_chol);
    }

    const Eigen::MatrixXd& Ca_occ = scf_.C_alpha.leftCols(na_);
    const Eigen::MatrixXd& Ca_vir = scf_.C_alpha.rightCols(va_);

    Eigen::MatrixXd Cb_occ, Cb_vir;
    if (has_beta) {
        Cb_occ = scf_.C_beta.leftCols(nb_);
        Cb_vir = scf_.C_beta.rightCols(vb_);
    }

    int chunk_size = 128; 

    #pragma omp parallel for schedule(dynamic)
    for (int P_start = 0; P_start < n_chol; P_start += chunk_size) {
        int P_end = std::min(n_chol, P_start + chunk_size);
        int P_size = P_end - P_start;

        Eigen::Map<const Eigen::MatrixXd> L_chunk(scf_.L_mat.col(P_start).data(), nbf_ * nbf_, P_size);
        Eigen::Map<const Eigen::MatrixXd> L_reshaped(L_chunk.data(), nbf_, nbf_ * P_size);

        Eigen::MatrixXd X_a = Ca_vir.transpose() * L_reshaped;

        for (int p = 0; p < P_size; ++p) {
            Eigen::Map<Eigen::MatrixXd> X_P(X_a.data() + p * va_ * nbf_, va_, nbf_);
            Eigen::MatrixXd B_MO_a = X_P * Ca_occ; 

            for (int i = 0; i < na_; ++i) {
                for (int a = 0; a < va_; ++a) {
                    B_ia_P_alpha_(i * va_ + a, P_start + p) = B_MO_a(a, i);
                }
            }
        }

        if (has_beta) {
            Eigen::MatrixXd X_b = Cb_vir.transpose() * L_reshaped;

            for (int p = 0; p < P_size; ++p) {
                Eigen::Map<Eigen::MatrixXd> X_P_b(X_b.data() + p * vb_ * nbf_, vb_, nbf_);
                Eigen::MatrixXd B_MO_b = X_P_b * Cb_occ; 

                for (int i = 0; i < nb_; ++i) {
                    for (int a = 0; a < vb_; ++a) {
                        B_ia_P_beta_(i * vb_ + a, P_start + p) = B_MO_b(a, i);
                    }
                }
            }
        }
    }
}

double OMP2::execute_micro_iterations() {
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    double old_energy = e_ss_ + e_os_;
    scf_.C_alpha = C_a_current_;
    scf_.C_beta  = C_b_current_;
    scf_.P_alpha = scf_.C_alpha.leftCols(na_) * scf_.C_alpha.leftCols(na_).transpose();
    scf_.P_beta  = scf_.C_beta.leftCols(nb_)  * scf_.C_beta.leftCols(nb_).transpose();

    pseudocanonicalize();

    C_a_current_ = scf_.C_alpha;
    C_b_current_ = scf_.C_beta;

    if (config_.eri_method == "cholesky") {
        transform_3center_mo_cholesky();
        compute_t2_and_energy_cholesky();

    } else if (config_.eri_method == "df") {
        transform_3center_mo(); 
        transform_integrals(); 

        compute_t2_amplitudes();
        compute_mp2_energy();
    } else {
        transform_integrals(); 

        compute_t2_amplitudes();
        compute_mp2_energy();
    }

    build_opdm_alpha();
    if (!is_restricted && nb_ > 0) {
        build_opdm_beta();
    } else if (is_restricted && nb_ > 0) {
        G_oo_beta_ = G_oo_alpha_; 
    }
    return (e_ss_ + e_os_) - old_energy;
}

void OMP2::execute_macro_iterations(DIIS& diis_a, DIIS& diis_b, int macro_iter) {
    build_generalized_fock();

    bool is_restricted = (na_ == nb_ && va_ == vb_);
    if (is_restricted && nb_ > 0) F_gen_b_ = F_gen_a_; 

    int dim_a = va_ * na_;
    int dim_b = (is_restricted) ? 0 : (nb_ > 0 ? vb_ * nb_ : 0);
    int n_params = dim_a + dim_b;

    if (orbital_gradient_.size() != n_params) orbital_gradient_.resize(n_params);

    int idx = 0;
    bool use_sym = (!scf_.irreps_alpha.empty() && scf_.irreps_alpha[0] != -1);
    
    if (!is_restricted && nb_ > 0) {
        Eigen::MatrixXd wa = 2.0 * F_gen_a_.block(na_, 0, va_, na_);
        for (int i = 0; i < na_; ++i) {
            for (int a = 0; a < va_; ++a) {
                if (use_sym && (scf_.irreps_alpha[i] ^ scf_.irreps_alpha[na_ + a]) != 0) orbital_gradient_(idx++) = 0.0;
                else orbital_gradient_(idx++) = wa(a, i);
            }
        }
        Eigen::MatrixXd wb = 2.0 * F_gen_b_.block(nb_, 0, vb_, nb_);
        for (int i = 0; i < nb_; ++i) {
            for (int b = 0; b < vb_; ++b) {
                if (use_sym && (scf_.irreps_beta[i] ^ scf_.irreps_beta[nb_ + b]) != 0) orbital_gradient_(idx++) = 0.0;
                else orbital_gradient_(idx++) = wb(b, i);
            }
        }
    } else {
        Eigen::MatrixXd wa = 2.0 * F_gen_a_.block(na_, 0, va_, na_);
        Eigen::MatrixXd wb = 2.0 * F_gen_b_.block(nb_, 0, vb_, nb_);
        Eigen::MatrixXd w_sym = wa + wb;
        
        for (int i = 0; i < na_; ++i) {
            for (int a = 0; a < va_; ++a) {
                if (use_sym && (scf_.irreps_alpha[i] ^ scf_.irreps_alpha[na_ + a]) != 0) orbital_gradient_(idx++) = 0.0;
                else orbital_gradient_(idx++) = w_sym(a, i);
            }
        }
    }
}
MP2Result OMP2::compute() {
    auto start_time = std::chrono::high_resolution_clock::now();

    C_a_current_ = scf_.C_alpha;
    C_b_current_ = scf_.C_beta;

    double e_total_best = 1e99;
    double e_corr_best = 0.0;
    double e_total_last = 1e99;
    double trust_radius = 0.15; 
    double expected_change = -1e-6;

    Eigen::MatrixXd C_a_last = scf_.C_alpha;
    Eigen::MatrixXd C_b_last = scf_.C_beta;
    Eigen::VectorXd last_kappa; 

    OrbitalLBFGS lbfgs_engine;
    DIIS diis_alpha(6);
    DIIS diis_beta(6);

    if(omp_get_thread_num() == 0) {
        std::cout << "\n========================================================\n";
        std::cout << "      Orbital-Optimized MP2 (OMP2)\n";
        if (config_.opt_method == "soscf") {
            std::cout << "      Trust-Region SOSCF (Approximate Hessian)\n";
        } else {
            std::cout << "      Macro-Micro Robust L-BFGS with Dynamic Shift\n";
        }
        std::cout << "========================================================\n";
    }

    bool is_converged = false;
    int macro_iter = 0;
    bool is_restricted = (na_ == nb_ && va_ == vb_);

    while (macro_iter < config_.max_iterations) {
        scf_.C_alpha = C_a_current_;
        scf_.C_beta  = C_b_current_;

        execute_micro_iterations();
        Eigen::MatrixXd F_ao_a, F_ao_b;
        build_fock_fast(scf_.P_alpha, scf_.P_beta, F_ao_a, F_ao_b);

        double e_scf = 0.5 * (scf_.P_alpha.cwiseProduct(H_core_ + F_ao_a).sum() + 
                              scf_.P_beta.cwiseProduct(H_core_ + F_ao_b).sum()) 
                       + mol_.nuclear_repulsion_energy();

        double e_mp2_corr = get_correlation_energy(); 
        double e_tot = e_scf + e_mp2_corr;

        if (macro_iter > 0) {
            double actual_change = e_tot - e_total_last;
            double rho = actual_change / expected_change; 

            if (actual_change > 1e-7) {
                C_a_current_ = C_a_last; 
                C_b_current_ = C_b_last;
                
                trust_radius *= 0.25; 
                if (trust_radius <= 1e-4) {
                    if (omp_get_thread_num() == 0) {
                        std::cout << "  [OMP2] Trust radius minimum tercapai. Konvergensi optimal berhasil dicapai.\n";
                    }
                    is_converged = true;
                    break; 
                }
                scf_.P_alpha = C_a_current_.leftCols(na_) * C_a_current_.leftCols(na_).transpose();
                if (!is_restricted && nb_ > 0) {
                    scf_.P_beta = C_b_current_.leftCols(nb_) * C_b_current_.leftCols(nb_).transpose();
                } else {
                    scf_.P_beta = scf_.P_alpha;
                }
                expected_change = -1e-6; 
                continue; 
                
            } else {
                if (rho > 0.75) {
                    trust_radius = std::min(0.25, trust_radius * 1.5); 
                } else if (rho < 0.25) {
                    trust_radius *= 0.5;
                }
                if (trust_radius <= 1e-5) {
                    if (omp_get_thread_num() == 0) {
                        std::cout << "  [OMP2] Trust radius minimum tercapai. Konvergensi optimal berhasil dicapai.\n";
                    }
                    is_converged = true;
                    break;
                }
            }
        }
      

        if (e_tot < e_total_best) { e_total_best = e_tot; e_corr_best = e_mp2_corr; }
        
        e_total_last = e_tot;
        C_a_last = C_a_current_; 
        C_b_last = C_b_current_;

        execute_macro_iterations(diis_alpha, diis_beta, macro_iter);
        double grad_norm = orbital_gradient_.norm();

        if(omp_get_thread_num() == 0) {
            std::cout << std::setw(4) << macro_iter << "    " 
                      << std::fixed << std::setprecision(8) << e_tot << "    "
                      << std::setprecision(8) << e_mp2_corr << "    " 
                      << std::scientific << std::setprecision(2) << grad_norm << "\n";
        }

        if (macro_iter > 0 && grad_norm < grad_thresh_ && std::abs(e_tot - e_total_last) < conv_thresh_) {
            is_converged = true; break;
        }

        int n_params = orbital_gradient_.size();
        Eigen::VectorXd diag_H(n_params);
        int idx = 0;
        double level_shift = (grad_norm > 0.1) ? 0.05 : 0.005;
        double spin_factor = is_restricted ? 4.0 : 2.0;
        
        for (int i = 0; i < na_; ++i) {             
            for (int a = 0; a < va_; ++a) {        
                double eps_diff = scf_.orbital_energies_alpha(na_ + a) - scf_.orbital_energies_alpha(i);
                double safe_diff = std::max(std::abs(eps_diff), 1e-4);
                double J_ia = 0.0;
                
                if (config_.eri_method != "exact") {
                    J_ia = B_ia_P_alpha_.row(i * va_ + a).squaredNorm(); 
                } else {
                    auto* g_blk = g_aa_.get_block(0, 0, 0, 0);
                    if (g_blk) J_ia = std::abs((*g_blk)(i, a, i, a));
                }
                double diag_J_a = spin_factor * J_ia;
                diag_H(idx++) = spin_factor * safe_diff + diag_J_a + level_shift;  
            }
        }
        if (!is_restricted && nb_ > 0) {
            for (int i = 0; i < nb_; ++i) {         
                for (int a = 0; a < vb_; ++a) {    
                    double eps_diff = scf_.orbital_energies_beta(nb_ + a) - scf_.orbital_energies_beta(i);
                    double safe_diff = std::max(std::abs(eps_diff), 1e-4);
                    double J_ia = 0.0;
                    if (config_.eri_method != "exact") {
                        J_ia = B_ia_P_beta_.row(i * vb_ + a).squaredNorm();
                    } else {
                        auto* g_blk = g_bb_.get_block(0, 0, 0, 0);
                        if (g_blk) J_ia = std::abs((*g_blk)(i, a, i, a));
                    }
                    diag_H(idx++) = 2.0 * safe_diff + 2.0 * J_ia + level_shift; // Ubah 0.0 menjadi 2.0 * J_ia
                }
            }
        }

        Eigen::VectorXd actual_step;

        if (config_.opt_method == "soscf") {
            mshqc::gradient::TrustRegionConfig tr_conf;
            tr_conf.micro_thresh = std::min(1e-4, grad_norm * 0.1); 
            mshqc::gradient::TrustRegionSOSCF soscf_engine(tr_conf);

            auto compute_hessian_vector = [&](const Eigen::VectorXd& p_vec) -> Eigen::VectorXd {
                bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
                Eigen::VectorXd Hp = diag_H.cwiseProduct(p_vec);
                
                int dim_a = va_ * na_;
                int dim_b = (is_restricted) ? 0 : (vb_ * nb_);
                double spin_factor = is_restricted ? 4.0 : 2.0;
                double ex_factor   = is_restricted ? 2.0 : 2.0;
                int n_aux = (config_.eri_method != "exact") ? scf_.L_mat.cols() : 0;

                if (config_.eri_method == "exact") {
                    auto* ptr_aa = g_aa_.get_block(0, 0, 0, 0);
                    auto* ptr_ab = (!is_restricted && dim_b > 0) ? g_ab_.get_block(0, 0, 0, 0) : nullptr;
                    Eigen::Map<const Eigen::MatrixXd> kappa_a(p_vec.data(), va_, na_);
                    Eigen::Map<const Eigen::MatrixXd> kappa_b(p_vec.data() + dim_a, vb_, nb_); 
                    // ========================================================
                    // 1. BLOK ALPHA-ALPHA (DAN KOPLING DARI BETA)
                    // ========================================================
                    if (ptr_aa && dim_a > 0) {
                        int idx_h = 0;
                        for (int i = 0; i < na_; ++i) {          
                            for (int a = 0; a < va_; ++a) {      
                                double off_diag = 0.0;
                                for (int j = 0; j < na_; ++j) {
                                    for (int b = 0; b < va_; ++b) {
                                        if (i == j && a == b) continue; 
                                        double coulomb = (*ptr_aa)(i, a, j, b);
                                        double exchange = (*ptr_aa)(i, b, j, a);
                                        off_diag += (spin_factor * coulomb - ex_factor * exchange) * kappa_a(b, j);
                                    }
                                }
                                if (ptr_ab) {
                                    for (int j = 0; j < nb_; ++j) {
                                        for (int b = 0; b < vb_; ++b) {
                                            double coulomb = (*ptr_ab)(i, a, j, b);
                                            off_diag += spin_factor * coulomb * kappa_b(b, j);
                                        }
                                    }
                                }
                                Hp(idx_h++) += off_diag;
                            }
                        }
                    }

                    // ========================================================
                    // 2. BLOK BETA-BETA (DAN KOPLING DARI ALPHA)
                    // ========================================================
                    if (!is_restricted && dim_b > 0) {
                        auto* ptr_bb = g_bb_.get_block(0, 0, 0, 0);
                        if (ptr_bb) {
                            int idx_h = dim_a; 
                            for (int i = 0; i < nb_; ++i) {          
                                for (int a = 0; a < vb_; ++a) {       
                                    double off_diag = 0.0;
                                    for (int j = 0; j < nb_; ++j) {
                                        for (int b = 0; b < vb_; ++b) {
                                            if (i == j && a == b) continue; 
                                            double coulomb = (*ptr_bb)(i, a, j, b);
                                            double exchange = (*ptr_bb)(i, b, j, a);
                                            off_diag += (spin_factor * coulomb - ex_factor * exchange) * kappa_b(b, j);
                                        }
                                    }
                                    if (ptr_ab) {
                                        for (int j = 0; j < na_; ++j) {
                                            for (int b = 0; b < va_; ++b) {
                                                double coulomb = (*ptr_ab)(j, b, i, a); 
                                                off_diag += spin_factor * coulomb * kappa_a(b, j);
                                            }
                                        }
                                    }
                                    Hp(idx_h++) += off_diag;
                                }
                            }
                        }
                    }
                } else {
                    // ========================================================
                    // 3. BLOK DENSITY FITTING (DF / CHOLESKY) - TOTAL DENSITY
                    // ========================================================
                    Eigen::Map<const Eigen::VectorXd> kappa_a_vec(p_vec.data(), dim_a);
                    Eigen::VectorXd v_P_total = B_ia_P_alpha_.transpose() * kappa_a_vec;
                    if (!is_restricted && dim_b > 0) {
                        Eigen::Map<const Eigen::VectorXd> kappa_b_vec(p_vec.data() + dim_a, dim_b);
                        v_P_total += B_ia_P_beta_.transpose() * kappa_b_vec;
                    }

                    if (dim_a > 0) {
                        Eigen::VectorXd Hp_J_a = B_ia_P_alpha_ * v_P_total;  
                        Eigen::Map<const Eigen::MatrixXd> K_mat(kappa_a_vec.data(), va_, na_);
                        Eigen::MatrixXd K_mat_T = K_mat.transpose(); 
                        Eigen::MatrixXd Hp_K_mat_a = Eigen::MatrixXd::Zero(va_, na_);
                        
                        int n_threads = omp_get_max_threads();
                        std::vector<Eigen::MatrixXd> local_H_a(n_threads, Eigen::MatrixXd::Zero(va_, na_));
                        
                        #pragma omp parallel
                        {
                            int tid = omp_get_thread_num();
                            Eigen::MatrixXd M_p(na_, na_); 
                            
                            #pragma omp for schedule(dynamic)
                            for (int P = 0; P < n_aux; ++P) {
                                Eigen::Map<const Eigen::MatrixXd> B_P(B_ia_P_alpha_.col(P).data(), va_, na_);
                                M_p.noalias() = K_mat_T * B_P; 
                                local_H_a[tid].noalias() += B_P * M_p;
                            }
                        }
                        for(int t = 0; t < n_threads; ++t) Hp_K_mat_a += local_H_a[t];
                        Eigen::VectorXd Hp_J_a_corrected = Hp_J_a;
                        int idx_a = 0;
                        for (int i = 0; i < na_; ++i) {
                            for (int a = 0; a < va_; ++a) {
                                double J_ia = B_ia_P_alpha_.row(i * va_ + a).squaredNorm();
                                Hp_J_a_corrected(idx_a) -= J_ia * kappa_a_vec(idx_a);
                                idx_a++;
                            }
                        }
                        Hp.head(dim_a) += spin_factor * Hp_J_a_corrected - ex_factor * Eigen::Map<Eigen::VectorXd>(Hp_K_mat_a.data(), dim_a);
                    }
                    
                    if (!is_restricted && dim_b > 0) {
                        Eigen::Map<const Eigen::VectorXd> kappa_b_vec(p_vec.data() + dim_a, dim_b);
                        Eigen::VectorXd Hp_J_b = B_ia_P_beta_ * v_P_total;
                        Eigen::Map<const Eigen::MatrixXd> K_mat_b(kappa_b_vec.data(), vb_, nb_);
                        Eigen::MatrixXd K_mat_T_b = K_mat_b.transpose(); 
                        Eigen::MatrixXd Hp_K_mat_b = Eigen::MatrixXd::Zero(vb_, nb_);
                        
                        int n_threads = omp_get_max_threads();
                        std::vector<Eigen::MatrixXd> local_H_b(n_threads, Eigen::MatrixXd::Zero(vb_, nb_));
                        
                        #pragma omp parallel
                        {
                            int tid = omp_get_thread_num();
                            Eigen::MatrixXd M_p_b(nb_, nb_);
                            
                            #pragma omp for schedule(dynamic)
                            for (int P = 0; P < n_aux; ++P) {
                                Eigen::Map<const Eigen::MatrixXd> B_P_b(B_ia_P_beta_.col(P).data(), vb_, nb_);
                                M_p_b.noalias() = K_mat_T_b * B_P_b;
                                local_H_b[tid].noalias() += B_P_b * M_p_b;
                            }
                        }
                        for(int t = 0; t < n_threads; ++t) Hp_K_mat_b += local_H_b[t];
                        Eigen::VectorXd Hp_J_b_corrected = Hp_J_b;
                        int idx_b = 0;
                        for (int i = 0; i < nb_; ++i) {
                            for (int a = 0; a < vb_; ++a) {
                                double J_ia = B_ia_P_beta_.row(i * vb_ + a).squaredNorm();
                                Hp_J_b_corrected(idx_b) -= J_ia * kappa_b_vec(idx_b);
                                idx_b++;
                            }
                        }
                        Hp.tail(dim_b) += spin_factor * Hp_J_b_corrected - ex_factor * Eigen::Map<Eigen::VectorXd>(Hp_K_mat_b.data(), dim_b);
                    }
                }
                bool use_sym = (!scf_.irreps_alpha.empty() && scf_.irreps_alpha[0] != -1);
                if (use_sym) {
                    int idx_sym = 0;
                    if (!is_restricted && dim_b > 0) {
                        for (int i = 0; i < na_; ++i) {
                            for (int a = 0; a < va_; ++a) {
                                if ((scf_.irreps_alpha[i] ^ scf_.irreps_alpha[na_ + a]) != 0) Hp(idx_sym) = 0.0;
                                idx_sym++;
                            }
                        }
                        for (int i = 0; i < nb_; ++i) {
                            for (int b = 0; b < vb_; ++b) {
                                if ((scf_.irreps_beta[i] ^ scf_.irreps_beta[nb_ + b]) != 0) Hp(idx_sym) = 0.0;
                                idx_sym++;
                            }
                        }
                    } else {
                        for (int i = 0; i < na_; ++i) {
                            for (int a = 0; a < va_; ++a) {
                                if ((scf_.irreps_alpha[i] ^ scf_.irreps_alpha[na_ + a]) != 0) Hp(idx_sym) = 0.0;
                                idx_sym++;
                            }
                        }
                    }
                }

                return Hp;
            };
            mshqc::gradient::TrustRegionResult step_info = soscf_engine.solve(orbital_gradient_, diag_H, trust_radius, compute_hessian_vector);
            actual_step = step_info.step;
            
            expected_change = step_info.predicted_energy_change;    
            if (expected_change >= 0.0) expected_change = -1e-6; 

        } else {
            Eigen::VectorXd kappa = lbfgs_engine.get_direction(orbital_gradient_, diag_H);
            if (kappa.dot(orbital_gradient_) > 0.0) {
                lbfgs_engine.reset();
                kappa = -orbital_gradient_.cwiseQuotient(diag_H); 
            }
            double step_norm = kappa.norm();
            if (step_norm > trust_radius) {
                actual_step = kappa * (trust_radius / step_norm);
            } else {
                actual_step = kappa;
            }
            lbfgs_engine.s_prev = actual_step;
        }

        apply_orbital_rotation(actual_step);

        scf_.P_alpha = C_a_current_.leftCols(na_) * C_a_current_.leftCols(na_).transpose();
        if (!is_restricted && nb_ > 0) scf_.P_beta = C_b_current_.leftCols(nb_) * C_b_current_.leftCols(nb_).transpose();
        else scf_.P_beta = scf_.P_alpha;

        last_kappa = actual_step;
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
void OMP2::reset_diis() {}
Eigen::MatrixXd OMP2::build_opdm() { return G_oo_alpha_ + G_oo_beta_; } 
Eigen::MatrixXd OMP2::extrapolate_diis(std::vector<Eigen::MatrixXd>&, std::vector<Eigen::MatrixXd>&) { return Eigen::MatrixXd(); }

} // namespace mshqc