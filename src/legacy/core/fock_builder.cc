// Copyright 2026 Muhamad Syahrul Hidayat
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file src/core/fock_builder.cc
 * @brief Fock Matrix Builder (HPC Edition - 4D Symmetry & CSAM)
 */

extern "C" {
#include <cint.h>
}

#include "mshqc/core/fock_builder.h"
#include "mshqc/basis.h"
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>

namespace mshqc {

FockBuilder::FockBuilder(std::shared_ptr<IntegralEngine> integrals,
                         const BasisSet& basis,
                         const Eigen::MatrixXd& H_core,
                         const Eigen::MatrixXd& schwarz)
    : integrals_(integrals), basis_(basis), nbasis_(basis.n_basis_functions()),
      H_core_(H_core), schwarz_(schwarz), is_first_iter_(true)
{
    G_alpha_accum_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    G_beta_accum_  = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
}

void FockBuilder::reset() {
    G_alpha_accum_.setZero();
    G_beta_accum_.setZero();
    is_first_iter_ = true;
}

void FockBuilder::set_petite_list(const PetiteList* list) {
    petite_list_ = list;
}

void FockBuilder::compute(const Eigen::MatrixXd& P_alpha, 
                          const Eigen::MatrixXd& P_beta,
                          Eigen::MatrixXd& F_alpha, 
                          Eigen::MatrixXd& F_beta) 
{
    // Deteksi jika ini adalah perhitungan RHF (P_beta kosong)
    bool is_rhf = (P_beta.size() == 0 || &P_alpha == &P_beta);

    Eigen::MatrixXd dP_a, dP_b;
    if (is_first_iter_) {
        dP_a = P_alpha; 
        if (!is_rhf) dP_b = P_beta;
        is_first_iter_ = false;
    } else {
        dP_a = P_alpha - P_alpha_old_; 
        if (!is_rhf) dP_b = P_beta - P_beta_old_;
    }
    P_alpha_old_ = P_alpha; 
    if (!is_rhf) P_beta_old_ = P_beta;

    // Untuk RHF, screening hanya bergantung pada dP_a
    double dP_max = dP_a.cwiseAbs().maxCoeff();
    if (!is_rhf) dP_max = std::max(dP_max, dP_b.cwiseAbs().maxCoeff());

    if (dP_max < 1e-15) {
        F_alpha = H_core_ + G_alpha_accum_;
        if (!is_rhf) F_beta  = H_core_ + G_beta_accum_;
        return;
    }

    int nshells = basis_.n_shells();
    double schwarz_max = schwarz_.maxCoeff();
    double threshold = 1e-13;

    std::vector<int> shell_offsets(nshells);
    std::vector<int> shell_sizes(nshells);
    auto map = basis_.shell_to_basis_function_map();
    for (int i = 0; i < nshells; ++i) {
        shell_offsets[i] = map[i];
        shell_sizes[i] = basis_.shell(i).n_functions();
    }

    // ========================================================================
    // CSAM (RADAR SIMETRI)
    // ========================================================================
    Eigen::MatrixXd P_max = Eigen::MatrixXd::Zero(nshells, nshells);
    for (int i = 0; i < nshells; ++i) {
        for (int j = 0; j <= i; ++j) {
            double max_val = 0.0;
            int start_i = shell_offsets[i], size_i = shell_sizes[i];
            int start_j = shell_offsets[j], size_j = shell_sizes[j];
            for (int m = 0; m < size_i; ++m) {
                for (int n = 0; n < size_j; ++n) {
                    double va = std::abs(dP_a(start_i + m, start_j + n));
                    double vb = is_rhf ? 0.0 : std::abs(dP_b(start_i + m, start_j + n));
                    max_val = std::max({max_val, va, vb});
                }
            }
            P_max(i, j) = max_val; 
            P_max(j, i) = max_val;
        }
    }

    // === MULAI BAGIAN BARU (PENGGANTI KERNEL & REDUKSI LAMA) ===
    int max_shell_dim = 1;
    for (int s : shell_sizes) max_shell_dim = std::max(max_shell_dim, s);
    const size_t max_quartet_sz = std::max<size_t>(
        64, static_cast<size_t>(max_shell_dim) * max_shell_dim * max_shell_dim * max_shell_dim
    );

    // ========================================================================
    // LAMBDA 1: KERNEL UHF (BUFFER RAW & TULIS SIMETRIS SATU ARAH)
    // ========================================================================
    auto process_shell_quartet = [&](Eigen::MatrixXd& Ga, Eigen::MatrixXd& Gb, double* buf,
                                     int M, int N, int P, int Q, double sym_weight) {
        if (!integrals_->compute_shell_block_raw(M, N, P, Q, buf)) return;

        int dM = shell_sizes[M], offM = shell_offsets[M];
        int dN = shell_sizes[N], offN = shell_offsets[N];
        int dP = shell_sizes[P], offP = shell_offsets[P];
        int dQ = shell_sizes[Q], offQ = shell_offsets[Q];

        double fac = sym_weight;
        if (M == N) fac *= 0.5;
        if (P == Q) fac *= 0.5;
        if (M == P && N == Q) fac *= 0.5;

        const double* I_ptr = buf;
        for (int q = 0; q < dQ; ++q) {
            int sig = offQ + q;
            for (int p = 0; p < dP; ++p) {
                int lam = offP + p;
                double pt_ls_2 = (dP_a(lam, sig) + dP_b(lam, sig)) * 2.0;
                double j_ls_acc = 0.0;
                for (int n = 0; n < dN; ++n) {
                    int nu = offN + n;
                    double dPa_ns = dP_a(nu, sig), dPa_nl = dP_a(nu, lam);
                    double dPb_ns = dP_b(nu, sig), dPb_nl = dP_b(nu, lam);
                    for (int m = 0; m < dM; ++m) {
                        int mu = offM + m;
                        double val = *I_ptr++;
                        if (std::abs(val) < 1e-14) continue;

                        double v = val * fac;
                        double j_val = v * pt_ls_2;
                        Ga(mu, nu) += j_val;
                        Gb(mu, nu) += j_val;
                        j_ls_acc += v * (dP_a(mu, nu) + dP_b(mu, nu)) * 2.0;

                        Ga(mu, lam) -= v * dPa_ns;
                        Ga(nu, lam) -= v * dP_a(mu, sig);
                        Ga(mu, sig) -= v * dPa_nl;
                        Ga(nu, sig) -= v * dP_a(mu, lam);

                        Gb(mu, lam) -= v * dPb_ns;
                        Gb(nu, lam) -= v * dP_b(mu, sig);
                        Gb(mu, sig) -= v * dPb_nl;
                        Gb(nu, sig) -= v * dP_b(mu, lam);
                    }
                }
                Ga(lam, sig) += j_ls_acc;
                Gb(lam, sig) += j_ls_acc;
            }
        }
    };

    // ========================================================================
    // LAMBDA 2: KERNEL RHF (BUFFER RAW & TULIS SIMETRIS SATU ARAH)
    // ========================================================================
    auto process_shell_quartet_rhf = [&](Eigen::MatrixXd& Ga, double* buf,
                                         int M, int N, int P, int Q, double sym_weight) {
        if (!integrals_->compute_shell_block_raw(M, N, P, Q, buf)) return;

        int dM = shell_sizes[M], offM = shell_offsets[M];
        int dN = shell_sizes[N], offN = shell_offsets[N];
        int dP = shell_sizes[P], offP = shell_offsets[P];
        int dQ = shell_sizes[Q], offQ = shell_offsets[Q];

        double fac = sym_weight;
        if (M == N) fac *= 0.5;
        if (P == Q) fac *= 0.5;
        if (M == P && N == Q) fac *= 0.5;

        const double* I_ptr = buf;
        for (int q = 0; q < dQ; ++q) {
            int sig = offQ + q;
            for (int p = 0; p < dP; ++p) {
                int lam = offP + p;
                double pt_ls_4 = dP_a(lam, sig) * 4.0;
                double j_ls_acc = 0.0;
                for (int n = 0; n < dN; ++n) {
                    int nu = offN + n;
                    double dPa_ns = dP_a(nu, sig), dPa_nl = dP_a(nu, lam);
                    for (int m = 0; m < dM; ++m) {
                        int mu = offM + m;
                        double val = *I_ptr++;
                        if (std::abs(val) < 1e-14) continue;

                        double v = val * fac;
                        Ga(mu, nu)  += v * pt_ls_4;
                        j_ls_acc    += v * dP_a(mu, nu) * 4.0;

                        Ga(mu, lam) -= v * dPa_ns;
                        Ga(nu, lam) -= v * dP_a(mu, sig);
                        Ga(mu, sig) -= v * dPa_nl;
                        Ga(nu, sig) -= v * dP_a(mu, lam);
                    }
                }
                Ga(lam, sig) += j_ls_acc;
            }
        }
    };

    // ========================================================================
    // TAHAP EKSEKUSI & REDUKSI PARALEL
    // ========================================================================
    Eigen::MatrixXd dGa = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    Eigen::MatrixXd dGb = is_rhf ? Eigen::MatrixXd() : Eigen::MatrixXd::Zero(nbasis_, nbasis_);

    #pragma omp parallel
    {
        std::vector<double> buf(max_quartet_sz);
        Eigen::MatrixXd Ga_loc = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
        Eigen::MatrixXd Gb_loc = is_rhf ? Eigen::MatrixXd() : Eigen::MatrixXd::Zero(nbasis_, nbasis_);

        if (petite_list_ && !petite_list_->get_unique_quartets().empty()) {
            const auto& uq = petite_list_->get_unique_quartets();
            #pragma omp for schedule(dynamic, 16) nowait
            for (size_t i = 0; i < uq.size(); ++i) {
                int M = uq[i].M, N = uq[i].N, P = uq[i].P, Q = uq[i].Q;
                double bound = schwarz_(M, N) * schwarz_(P, Q);
                if (bound * dP_max < threshold) continue;
                double local_P_max = std::max({P_max(M, N), P_max(P, Q), P_max(M, P), P_max(M, Q), P_max(N, P), P_max(N, Q)});
                if (bound * local_P_max < threshold) continue;

                if (is_rhf) process_shell_quartet_rhf(Ga_loc, buf.data(), M, N, P, Q, uq[i].weight);
                else        process_shell_quartet(Ga_loc, Gb_loc, buf.data(), M, N, P, Q, uq[i].weight);
            }
        } else {
            #pragma omp for schedule(dynamic, 1) nowait
            for (int M = 0; M < nshells; ++M) {
                for (int N = 0; N <= M; ++N) {
                    double bound_MN = schwarz_(M, N);
                    if (bound_MN * schwarz_max * dP_max < threshold) continue;
                    for (int P = 0; P <= M; ++P) {
                        int Q_max = (M == P) ? N : P;
                        for (int Q = 0; Q <= Q_max; ++Q) {
                            double bound = bound_MN * schwarz_(P, Q);
                            double local_P_max = std::max({P_max(M, N), P_max(P, Q), P_max(M, P), P_max(M, Q), P_max(N, P), P_max(N, Q)});
                            if (bound * local_P_max < threshold) continue;

                            if (is_rhf) process_shell_quartet_rhf(Ga_loc, buf.data(), M, N, P, Q, 1.0);
                            else        process_shell_quartet(Ga_loc, Gb_loc, buf.data(), M, N, P, Q, 1.0);
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            dGa.noalias() += Ga_loc;
            if (!is_rhf) dGb.noalias() += Gb_loc;
        }
    }

    G_alpha_accum_.noalias() += dGa + dGa.transpose();
    F_alpha = H_core_ + G_alpha_accum_;

    if (!is_rhf) {
        G_beta_accum_.noalias() += dGb + dGb.transpose();
        F_beta = H_core_ + G_beta_accum_;
    }
    // === AKHIR BAGIAN BARU ===
}

} // namespace mshqc