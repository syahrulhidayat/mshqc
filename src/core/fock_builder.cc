 // ==============================================================================
 // Copyright (c) 2026 Syahrul and mshqc contributors
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
 // ==============================================================================

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

    double dP_max = dP_a.cwiseAbs().maxCoeff();
    if (!is_rhf) dP_max = std::max(dP_max, dP_b.cwiseAbs().maxCoeff());

    if (dP_max < 1e-12) {
        F_alpha = H_core_ + G_alpha_accum_;
        if (!is_rhf) F_beta  = H_core_ + G_beta_accum_;
        return;
    }

    int nshells = basis_.n_shells();
    double schwarz_max = schwarz_.maxCoeff();
    double threshold = 1e-11;

    std::vector<int> shell_offsets(nshells);
    std::vector<int> shell_sizes(nshells);
    auto map = basis_.shell_to_basis_function_map();
    for(int i=0; i<nshells; ++i) {
        shell_offsets[i] = map[i];
        shell_sizes[i] = basis_.shell(i).n_functions();
    }

    Eigen::MatrixXd P_max = Eigen::MatrixXd::Zero(nshells, nshells);
    for (int i = 0; i < nshells; ++i) {
        for (int j = 0; j <= i; ++j) {
            double max_val = 0.0;
            int start_i = shell_offsets[i], size_i = shell_sizes[i];
            int start_j = shell_offsets[j], size_j = shell_sizes[j];
            for (int m = 0; m < size_i; ++m) {
                for (int n = 0; n < size_j; ++n) {
                    double va = std::abs(dP_a(start_i+m, start_j+n));
                    double vb = is_rhf ? 0.0 : std::abs(dP_b(start_i+m, start_j+n));
                    max_val = std::max({max_val, va, vb});
                }
            }
            P_max(i, j) = max_val; P_max(j, i) = max_val;
        }
    }

    int n_threads = omp_get_max_threads();
    std::vector<Eigen::MatrixXd> Ga_priv(n_threads, Eigen::MatrixXd::Zero(nbasis_, nbasis_));
    std::vector<Eigen::MatrixXd> Gb_priv(n_threads, Eigen::MatrixXd::Zero(nbasis_, nbasis_));

    auto process_shell_quartet = [&](int tid, int M, int N, int P, int Q, double sym_weight) {
        auto shell_ints = integrals_->compute_shell_block(M, N, P, Q);
        if (shell_ints.empty()) return;

        int dM = shell_sizes[M]; int offM = shell_offsets[M];
        int dN = shell_sizes[N]; int offN = shell_offsets[N];
        int dP = shell_sizes[P]; int offP = shell_offsets[P];
        int dQ = shell_sizes[Q]; int offQ = shell_offsets[Q];

        double fac = sym_weight;
        if (M == N) fac *= 0.5;
        if (P == Q) fac *= 0.5;
        if (M == P && N == Q) fac *= 0.5;

        const double* I_ptr = shell_ints.data();
        for (int q = 0; q < dQ; ++q) {
            int sig = offQ + q;
            for (int p = 0; p < dP; ++p) {
                int lam = offP + p;
                for (int n = 0; n < dN; ++n) {
                    int nu = offN + n;
                    for (int m = 0; m < dM; ++m) {
                        int mu = offM + m;
                        double val = *I_ptr++;
                        if (std::abs(val) < 1e-12) continue;

                        double v = val * fac;
                        double pt_ls = dP_a(lam, sig) + dP_b(lam, sig);
                        double pt_mn = dP_a(mu, nu) + dP_b(mu, nu);
                        double v2 = v * 2.0;

                        Ga_priv[tid](mu, nu) += v2 * pt_ls; Gb_priv[tid](mu, nu) += v2 * pt_ls;
                        Ga_priv[tid](nu, mu) += v2 * pt_ls; Gb_priv[tid](nu, mu) += v2 * pt_ls;
                        Ga_priv[tid](lam, sig) += v2 * pt_mn; Gb_priv[tid](lam, sig) += v2 * pt_mn;
                        Ga_priv[tid](sig, lam) += v2 * pt_mn; Gb_priv[tid](sig, lam) += v2 * pt_mn;

                        Ga_priv[tid](mu, lam) -= v * dP_a(nu, sig); Ga_priv[tid](lam, mu) -= v * dP_a(sig, nu);
                        Ga_priv[tid](nu, lam) -= v * dP_a(mu, sig); Ga_priv[tid](lam, nu) -= v * dP_a(sig, mu);
                        Ga_priv[tid](mu, sig) -= v * dP_a(nu, lam); Ga_priv[tid](sig, mu) -= v * dP_a(lam, nu);
                        Ga_priv[tid](nu, sig) -= v * dP_a(mu, lam); Ga_priv[tid](sig, nu) -= v * dP_a(lam, mu);

                        Gb_priv[tid](mu, lam) -= v * dP_b(nu, sig); Gb_priv[tid](lam, mu) -= v * dP_b(sig, nu);
                        Gb_priv[tid](nu, lam) -= v * dP_b(mu, sig); Gb_priv[tid](lam, nu) -= v * dP_b(sig, mu);
                        Gb_priv[tid](mu, sig) -= v * dP_b(nu, lam); Gb_priv[tid](sig, mu) -= v * dP_b(lam, nu);
                        Gb_priv[tid](nu, sig) -= v * dP_b(mu, lam); Gb_priv[tid](sig, nu) -= v * dP_b(lam, mu);
                    }
                }
            }
        }
    };

    auto process_shell_quartet_rhf = [&](int tid, int M, int N, int P, int Q, double sym_weight) {
        auto shell_ints = integrals_->compute_shell_block(M, N, P, Q);
        if (shell_ints.empty()) return;

        int dM = shell_sizes[M]; int offM = shell_offsets[M];
        int dN = shell_sizes[N]; int offN = shell_offsets[N];
        int dP = shell_sizes[P]; int offP = shell_offsets[P];
        int dQ = shell_sizes[Q]; int offQ = shell_offsets[Q];

        double fac = sym_weight;
        if (M == N) fac *= 0.5;
        if (P == Q) fac *= 0.5;
        if (M == P && N == Q) fac *= 0.5;

        const double* I_ptr = shell_ints.data();
        for (int q = 0; q < dQ; ++q) {
            int sig = offQ + q;
            for (int p = 0; p < dP; ++p) {
                int lam = offP + p;
                for (int n = 0; n < dN; ++n) {
                    int nu = offN + n;
                    for (int m = 0; m < dM; ++m) {
                        int mu = offM + m;
                        double val = *I_ptr++;
                        if (std::abs(val) < 1e-12) continue;

                        double v = val * fac;
                        double v2 = v * 2.0;

                        double pt_ls = dP_a(lam, sig) * 2.0;
                        double pt_mn = dP_a(mu, nu) * 2.0;

                        Ga_priv[tid](mu, nu) += v2 * pt_ls;
                        Ga_priv[tid](nu, mu) += v2 * pt_ls;
                        Ga_priv[tid](lam, sig) += v2 * pt_mn;
                        Ga_priv[tid](sig, lam) += v2 * pt_mn;

                        Ga_priv[tid](mu, lam) -= v * dP_a(nu, sig); Ga_priv[tid](lam, mu) -= v * dP_a(sig, nu);
                        Ga_priv[tid](nu, lam) -= v * dP_a(mu, sig); Ga_priv[tid](lam, nu) -= v * dP_a(sig, mu);
                        Ga_priv[tid](mu, sig) -= v * dP_a(nu, lam); Ga_priv[tid](sig, mu) -= v * dP_a(lam, nu);
                        Ga_priv[tid](nu, sig) -= v * dP_a(mu, lam); Ga_priv[tid](sig, nu) -= v * dP_a(lam, mu);
                    }
                }
            }
        }
    };

    if (petite_list_ && !petite_list_->get_unique_quartets().empty()) {
        const auto& unique_quartets = petite_list_->get_unique_quartets();

        #pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < unique_quartets.size(); ++i) {
            int tid = omp_get_thread_num();
            int M = unique_quartets[i].M; int N = unique_quartets[i].N;
            int P = unique_quartets[i].P; int Q = unique_quartets[i].Q;
            double weight = unique_quartets[i].weight;

            double bound = schwarz_(M, N) * schwarz_(P, Q);
            if (bound * dP_max < threshold) continue;

            double local_P_max = std::max({ P_max(M, N), P_max(P, Q), P_max(M, P), P_max(M, Q), P_max(N, P), P_max(N, Q) });
            if (bound * local_P_max < threshold) continue;

            if (is_rhf) process_shell_quartet_rhf(tid, M, N, P, Q, weight);
            else        process_shell_quartet(tid, M, N, P, Q, weight);
        }
    } else {
        #pragma omp parallel for schedule(dynamic, 1)
        for (int M = 0; M < nshells; ++M) {
            int tid = omp_get_thread_num();
            for (int N = 0; N <= M; ++N) {
                double bound_MN = schwarz_(M, N);
                if (bound_MN * schwarz_max * dP_max < threshold) continue;

                for (int P = 0; P <= M; ++P) {
                    int Q_max = (M == P) ? N : P;
                    for (int Q = 0; Q <= Q_max; ++Q) {
                        double bound = bound_MN * schwarz_(P, Q);
                        double local_P_max = std::max({ P_max(M, N), P_max(P, Q), P_max(M, P), P_max(M, Q), P_max(N, P), P_max(N, Q) });
                        if (bound * local_P_max < threshold) continue;

                        if (is_rhf) process_shell_quartet_rhf(tid, M, N, P, Q, 1.0);
                        else        process_shell_quartet(tid, M, N, P, Q, 1.0);
                    }
                }
            }
        }
    }

    for (int t = 0; t < n_threads; ++t) {
        G_alpha_accum_ += Ga_priv[t];
        if (!is_rhf) G_beta_accum_ += Gb_priv[t];
    }

    G_alpha_accum_ = 0.5 * (G_alpha_accum_ + G_alpha_accum_.transpose());
    F_alpha = H_core_ + G_alpha_accum_;

    if (!is_rhf) {
        G_beta_accum_ = 0.5 * (G_beta_accum_ + G_beta_accum_.transpose());
        F_beta  = H_core_ + G_beta_accum_;
    }
}

}
