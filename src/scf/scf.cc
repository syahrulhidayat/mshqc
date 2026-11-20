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
 * @file src/scf/scf.cc
 * @brief Unified SCF Engine (BaseSCF, RHF, UHF, ROHF) in a single file.
 * @details High-performance batched Level-3 BLAS Cholesky/DF Fock build,
 *          low-rank PSD density factorization for Iteration 1, and CSAM-screened In-Core Fock build.
 */

#include "mshqc/scf/scf.h"
#include "mshqc/scf/diis.h"
#include "mshqc/scf/sad.h"
#include "mshqc/symmetry/salc_builder.h"
#include "mshqc/integrals/df_eri.h"
#include "mshqc/integrals/packed_ao_utils.h"
#include "mshqc/utils/blas_thread_guard.h"
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include "mshqc/utils/hdf5_io.h"
#include <memory>
#include <vector>
#ifdef I
#undef I
#endif

namespace mshqc {

namespace {
// [INJECTED_HPC_V4_SCF]
template <typename Cfg>
inline double get_scf_memory_limit_gb(const Cfg& cfg) {
  if constexpr (requires { cfg.memory_limit_gb; }) {
    if (cfg.memory_limit_gb > 0.0) return cfg.memory_limit_gb;
  }
  if (const char* env_mem = std::getenv("MSHQC_MEMORY_GB")) {
    const double v = std::atof(env_mem);
    if (v >= 0.25) return v;
  }
  return 4.0;
}

template <typename Cfg>
inline double get_scf_level_shift(const Cfg& cfg) {
  if constexpr (requires { cfg.level_shift; }) {
    return cfg.level_shift;
  }
  return 0.0;
}


// Mengekstrak faktor berperingkat rendah C_tilde (P = C_tilde * C_tilde^T) dari matriks
// densitas simetris positif semidefinit (seperti P_sad) agar iterasi ke-1 Cholesky/DF
// dapat langsung memakai algoritma MO O(M * N_bf^2 * r) tanpa alokasi tensor 3D N_bf^2 * M.
Eigen::MatrixXd factorize_psd_density(const Eigen::MatrixXd& P, double tol = 1e-12) {
    const int n = P.rows();
    if (n == 0) return Eigen::MatrixXd(0, 0);

    // Cek apakah P murni diagonal (kasus standar SAD tanpa simetrisasi grup titik)
    double off_diag_norm = (P - P.diagonal().asDiagonal().toDenseMatrix()).cwiseAbs().maxCoeff();
    if (off_diag_norm < 1e-13) {
        std::vector<int> nz_cols;
        nz_cols.reserve(n);
        for (int i = 0; i < n; ++i) {
            if (P(i, i) > tol) nz_cols.push_back(i);
        }
        Eigen::MatrixXd C_fac = Eigen::MatrixXd::Zero(n, static_cast<int>(nz_cols.size()));
        for (size_t k = 0; k < nz_cols.size(); ++k) {
            int idx = nz_cols[k];
            C_fac(idx, static_cast<int>(k)) = std::sqrt(P(idx, idx));
        }
        return C_fac;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (P + P.transpose()));
    const Eigen::VectorXd& evals = es.eigenvalues();
    const Eigen::MatrixXd& evecs = es.eigenvectors();
    std::vector<int> pos_cols;
    pos_cols.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (evals(i) > tol) pos_cols.push_back(i);
    }
    Eigen::MatrixXd C_fac(n, static_cast<int>(pos_cols.size()));
    for (size_t k = 0; k < pos_cols.size(); ++k) {
        int idx = pos_cols[k];
        C_fac.col(static_cast<int>(k)) = std::sqrt(evals(idx)) * evecs.col(idx);
    }
    return C_fac;
}

// Kontraksi Exchange berkecepatan tinggi berbasis Batch Level-3 BLAS (DGEMM)
// untuk 1 kanal spin (RHF atau UHF saat P_alpha == P_beta).
// [INJECTED_PRIORITAS_2D_PACKED_AO]
void compute_exchange_cholesky_single(
    const double* L_ptr,
    int nbasis,
    int k_size,
    const Eigen::MatrixXd& C_occ,
    Eigen::MatrixXd& K_acc,
    bool is_packed = false)
{
  const int n_occ = static_cast<int>(C_occ.cols());
  if (n_occ <= 0 || k_size <= 0) return;

  const int batch_p = std::clamp(
      static_cast<int>((8ULL * 1024ULL * 1024ULL) /
                       std::max<size_t>(1, static_cast<size_t>(nbasis) * n_occ * sizeof(double))),
      1, 32);
  const size_t col_stride = is_packed
      ? static_cast<size_t>(integrals::ao_tri_size(nbasis))
      : static_cast<size_t>(nbasis) * nbasis;
  const int max_threads = omp_get_max_threads();
  std::vector<Eigen::MatrixXd> K_tls(max_threads, Eigen::MatrixXd::Zero(nbasis, nbasis));

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    Eigen::MatrixXd T_batch(nbasis, n_occ * batch_p);
    Eigen::MatrixXd L_buf;
    if (is_packed) L_buf.resize(nbasis, nbasis);
    int count_in_batch = 0;

    #pragma omp for schedule(static) nowait
    for (int P = 0; P < k_size; ++P) {
      const double* col_p = L_ptr + static_cast<size_t>(P) * col_stride;
      if (is_packed) {
        integrals::unpack_ao_column(col_p, nbasis, true, L_buf);
        T_batch.middleCols(count_in_batch * n_occ, n_occ).noalias() = L_buf * C_occ;
      } else {
        Eigen::Map<const Eigen::MatrixXd> L_P(col_p, nbasis, nbasis);
        T_batch.middleCols(count_in_batch * n_occ, n_occ).noalias() = L_P * C_occ;
      }
      count_in_batch++;

      if (count_in_batch == batch_p) {
        K_tls[tid].noalias() += T_batch * T_batch.transpose();
        count_in_batch = 0;
      }
    }
    if (count_in_batch > 0) {
      auto T_rem = T_batch.leftCols(count_in_batch * n_occ);
      K_tls[tid].noalias() += T_rem * T_rem.transpose();
    }
  }

  for (int t = 0; t < max_threads; ++t) {
    K_acc.noalias() += K_tls[t];
  }
}

// Kontraksi Exchange berkecepatan tinggi berbasis Batch Level-3 BLAS (DGEMM)
// untuk 2 kanal spin sekaligus (UHF / ROHF): membaca L_P hanya 1 kali per vektor P!
void compute_exchange_cholesky_dual(
    const double* L_ptr,
    int nbasis,
    int k_size,
    const Eigen::MatrixXd& Ca_occ,
    const Eigen::MatrixXd& Cb_occ,
    Eigen::MatrixXd& Ka_acc,
    Eigen::MatrixXd& Kb_acc,
    bool is_packed = false)
{
  const int na = static_cast<int>(Ca_occ.cols());
  const int nb = static_cast<int>(Cb_occ.cols());
  if ((na + nb) <= 0 || k_size <= 0) return;
  if (nb <= 0) {
    compute_exchange_cholesky_single(L_ptr, nbasis, k_size, Ca_occ, Ka_acc, is_packed);
    return;
  }

  Eigen::MatrixXd Cab_occ(nbasis, na + nb);
  Cab_occ.leftCols(na) = Ca_occ;
  Cab_occ.rightCols(nb) = Cb_occ;

  const int batch_p = std::clamp(
      static_cast<int>((8ULL * 1024ULL * 1024ULL) /
                       std::max<size_t>(1, static_cast<size_t>(nbasis) * (na + nb) * sizeof(double))),
      1, 32);
  const size_t col_stride = is_packed
      ? static_cast<size_t>(integrals::ao_tri_size(nbasis))
      : static_cast<size_t>(nbasis) * nbasis;
  const int max_threads = omp_get_max_threads();
  std::vector<Eigen::MatrixXd> Ka_tls(max_threads, Eigen::MatrixXd::Zero(nbasis, nbasis));
  std::vector<Eigen::MatrixXd> Kb_tls(max_threads, Eigen::MatrixXd::Zero(nbasis, nbasis));

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    Eigen::MatrixXd Tab_loc(nbasis, na + nb);
    Eigen::MatrixXd Ta_batch(nbasis, na * batch_p);
    Eigen::MatrixXd Tb_batch(nbasis, nb * batch_p);
    Eigen::MatrixXd L_buf;
    if (is_packed) L_buf.resize(nbasis, nbasis);
    int count_in_batch = 0;

    #pragma omp for schedule(static) nowait
    for (int P = 0; P < k_size; ++P) {
      const double* col_p = L_ptr + static_cast<size_t>(P) * col_stride;
      if (is_packed) {
        integrals::unpack_ao_column(col_p, nbasis, true, L_buf);
        Tab_loc.noalias() = L_buf * Cab_occ;
      } else {
        Eigen::Map<const Eigen::MatrixXd> L_P(col_p, nbasis, nbasis);
        Tab_loc.noalias() = L_P * Cab_occ;
      }
      Ta_batch.middleCols(count_in_batch * na, na) = Tab_loc.leftCols(na);
      Tb_batch.middleCols(count_in_batch * nb, nb) = Tab_loc.rightCols(nb);
      count_in_batch++;

      if (count_in_batch == batch_p) {
        Ka_tls[tid].noalias() += Ta_batch * Ta_batch.transpose();
        Kb_tls[tid].noalias() += Tb_batch * Tb_batch.transpose();
        count_in_batch = 0;
      }
    }
    if (count_in_batch > 0) {
      auto Ta_rem = Ta_batch.leftCols(count_in_batch * na);
      auto Tb_rem = Tb_batch.leftCols(count_in_batch * nb);
      Ka_tls[tid].noalias() += Ta_rem * Ta_rem.transpose();
      Kb_tls[tid].noalias() += Tb_rem * Tb_rem.transpose();
    }
  }

  for (int t = 0; t < max_threads; ++t) {
    Ka_acc.noalias() += Ka_tls[t];
    Kb_acc.noalias() += Kb_tls[t];
  }
}


// Ortonormalisasi simetris Lowdin C_orth = C_in * (C_in^T * S * C_in)^(-1/2)
// agar orbital terkonvergensi dari geometri tetangga x0 langsung valid pada metrik S(x0 +- h)
Eigen::MatrixXd orthonormalize_mo_guess(const Eigen::MatrixXd& C_in, const Eigen::MatrixXd& S) {
    const int n = static_cast<int>(S.rows());
    if (C_in.rows() != n || C_in.cols() != n || n == 0) return Eigen::MatrixXd();
    Eigen::MatrixXd M = C_in.transpose() * S * C_in;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (M + M.transpose()));
    if (es.info() != Eigen::Success) return Eigen::MatrixXd();
    const Eigen::VectorXd& evals = es.eigenvalues();
    if (evals.minCoeff() < 1e-10) return Eigen::MatrixXd();
    Eigen::VectorXd inv_sqrt = evals.array().rsqrt();
    Eigen::MatrixXd U = es.eigenvectors();
    return C_in * (U * inv_sqrt.asDiagonal() * U.transpose());
}


Eigen::MatrixXd compute_shell_pmax(
    int nshells,
    const std::vector<int>& shell_starts,
    const std::vector<int>& shell_sizes,
    const Eigen::MatrixXd& dP_a,
    const Eigen::MatrixXd* dP_b = nullptr)
{
    Eigen::MatrixXd P_max = Eigen::MatrixXd::Zero(nshells, nshells);
    for (int i = 0; i < nshells; ++i) {
        int start_i = shell_starts[i], size_i = shell_sizes[i];
        for (int j = 0; j <= i; ++j) {
            int start_j = shell_starts[j], size_j = shell_sizes[j];
            double max_val = 0.0;
            for (int m = 0; m < size_i; ++m) {
                for (int n = 0; n < size_j; ++n) {
                    double va = std::abs(dP_a(start_i + m, start_j + n));
                    double vb = dP_b ? std::abs((*dP_b)(start_i + m, start_j + n)) : 0.0;
                    if (va > max_val) max_val = va;
                    if (vb > max_val) max_val = vb;
                }
            }
            P_max(i, j) = max_val;
            P_max(j, i) = max_val;
        }
    }
    return P_max;
}

} // namespace

BaseSCF::BaseSCF(const Molecule& mol, const BasisSet& basis,
                 std::shared_ptr<IntegralEngine> integrals,
                 std::shared_ptr<PointGroup> pg,
                 std::shared_ptr<PetiteList> pl,
                 int n_alpha, int n_beta,
                 const SCFConfig& config)
    : mol_(mol), basis_(basis), integrals_(integrals),
      pg_(pg), pl_(pl), config_(config),
      nbasis_(basis.n_basis_functions()),
      n_alpha_(n_alpha), n_beta_(n_beta)
{
    if (pg_ && pl_) symmetrizer_ = std::make_unique<BasisSymmetrizer>(basis_, *pg_, *pl_);

    int nshells = basis_.n_shells();
    shell_starts_.resize(nshells);
    shell_sizes_.resize(nshells);
    int offset = 0;
    for (int i = 0; i < nshells; ++i) {
        shell_starts_[i] = offset;
        shell_sizes_[i] = basis_.shell(i).n_functions();
        offset += shell_sizes_[i];
    }

    if (config_.use_df) {
        BasisSet aux_basis(config_.aux_basis_name, mol_);
        BasisSet combined_basis = basis_;
        combined_basis.append(aux_basis);
        auto df_integrals = std::make_shared<IntegralEngine>(mol_, combined_basis);
        auto df_engine = std::make_shared<integrals::DensityFittingERI>(basis_, aux_basis, df_integrals, config_.df_threshold);

        const double mem_lim_gb = get_scf_memory_limit_gb(config_);
    const double est_df_packed_gb =
        static_cast<double>(integrals::ao_tri_size(nbasis_)) *
        static_cast<double>(aux_basis.n_basis_functions()) *
        static_cast<double>(sizeof(double)) / 1.0e9;
    bool use_disk = (config_.scf_type == "ooc") ||
                    (mem_lim_gb > 0.0 && est_df_packed_gb > mem_lim_gb * 0.55);
    if (use_disk && config_.print_level > 0) {
      std::cout << "  [SCF-HPC] Packed DF tensor (" << std::fixed << std::setprecision(2)
                << est_df_packed_gb << " GB) > budget; switching to HDF5 OOC mode.\n";
    }

    if (use_disk) config_.scf_type = "ooc";
    df_engine->compute(use_disk);

        if (!use_disk) {
            this->L_mat_ = std::move(df_engine->get_B_mat_mutable());
        } else {
            this->L_mat_.resize(0, df_engine->get_B_mat().cols());
        }
    }
}

void BaseSCF::init_integrals() {
    auto t_start = std::chrono::high_resolution_clock::now();

    S_ = integrals_->compute_overlap();
    if (S_.rows() == 0) throw std::runtime_error("CRITICAL: Empty Overlap Matrix.");
    H_ = integrals_->compute_kinetic() + integrals_->compute_nuclear();

    if (config_.print_level > 0) std::cout << "  [SCF] Computing Orthogonalizer (SALC)... ";
    if (symmetrizer_) {
        SalcBuilder salc(symmetrizer_.get());
        auto salc_data = salc.build_salc(S_);
        X_ = salc_data.first;
        salc_irreps_ = salc_data.second;
    } else {
        mshqc::utils::ScopedBlasThreads blas_s_mt(nbasis_, 96);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S_);
    const Eigen::VectorXd& s_evals = es.eigenvalues();
    const Eigen::MatrixXd& s_evecs = es.eigenvectors();
    int n_kept = 0;
    for (int i = 0; i < nbasis_; ++i) if (s_evals(i) >= 1e-6) ++n_kept;
    Eigen::MatrixXd U_k(nbasis_, n_kept);
    Eigen::VectorXd inv_sq(n_kept);
    int kc = 0;
    for (int i = 0; i < nbasis_; ++i) {
      if (s_evals(i) >= 1e-6) {
        U_k.col(kc) = s_evecs.col(i);
        inv_sq(kc) = 1.0 / std::sqrt(s_evals(i));
        ++kc;
      }
    }
    X_.resize(nbasis_, nbasis_);
    if (n_kept > 0) {
      Eigen::MatrixXd U_sc = U_k * inv_sq.asDiagonal();
      X_.noalias() = U_sc * U_k.transpose();
    } else {
      X_.setZero();
    }
        salc_irreps_.assign(nbasis_, 0);
    }
    if (config_.print_level > 0) std::cout << "Done.\n";

    if (config_.eri_method == "df") config_.use_df = true;
    if (!config_.use_df) {
        if (config_.eri_method == "cholesky") {
            init_integrals_cholesky();
        } else {
            schwarz_ = precompute_shell_schwarz();
            if (pl_ && pl_->get_unique_quartets().empty()) pl_->build(schwarz_, 1e-12);
            if (config_.scf_type == "incore") init_integrals_incore();
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t_end - t_start;
    if (config_.print_level > 0) {
        std::string mode_str = (config_.scf_type == "incore") ? "In-Core" : "Direct";
        std::cout << "  [SCF] Initialization Done. Mode: " << mode_str << " ("
                  << std::fixed << std::setprecision(3) << elapsed.count() << "s)\n";
    }
}

// [INJECTED_PRIORITAS_3]
Eigen::MatrixXd BaseSCF::precompute_shell_schwarz() {
    int nshells = basis_.n_shells();
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(nshells, nshells);

    int max_dim = 1;
    for (int s = 0; s < nshells; ++s) max_dim = std::max(max_dim, shell_sizes_[s]);
    const size_t buf_sz = std::max<size_t>(
        64, static_cast<size_t>(max_dim) * max_dim * max_dim * max_dim
    );

    #pragma omp parallel
    {
        std::vector<double> buf(buf_sz);
        #pragma omp for schedule(dynamic, 1) nowait
        for (int M = 0; M < nshells; ++M) {
            for (int N = 0; N <= M; ++N) {
                if (!integrals_->compute_shell_block_raw(M, N, M, N, buf.data())) continue;
                int sz = shell_sizes_[M] * shell_sizes_[N] * shell_sizes_[M] * shell_sizes_[N];
                double max_val = 0.0;
                for (int i = 0; i < sz; ++i) {
                    if (std::abs(buf[i]) > max_val) max_val = std::abs(buf[i]);
                }
                double s_val = std::sqrt(max_val);
                Q(M, N) = s_val;
                Q(N, M) = s_val;
            }
        }
    }
    return Q;
}

void BaseSCF::init_integrals_incore() {
    if (config_.print_level > 0) std::cout << "  [SCF] Building Optimized Shell-Block InCore Integrals... " << std::flush;

    incore_quartets_.clear();
    incore_eri_pool_.clear();

    double shell_cutoff = 1e-12;

    // [INJECTED_AUDIT_FIX_SCF_CLEANUP]
  if (pl_ && pl_->get_unique_quartets().empty()) pl_->build(schwarz_, 1e-12);

    size_t total_ints = 0;
    auto register_quartet = [&](int M, int N, int P, int Q, double weight) {
        if (schwarz_(M, N) * schwarz_(P, Q) < shell_cutoff) return;
        size_t q_size = static_cast<size_t>(shell_sizes_[M]) * shell_sizes_[N] * shell_sizes_[P] * shell_sizes_[Q];
        CachedQuartet cq;
        cq.M = M; cq.N = N; cq.P = P; cq.Q = Q;
        cq.weight = weight;
        cq.data_offset = total_ints;
        incore_quartets_.push_back(cq);
        total_ints += q_size;
    };

    if (pl_) {
        for (const auto& [M, N, P, Q, w] : pl_->get_unique_quartets()) register_quartet(M, N, P, Q, w);
    } else {
        int nshells = basis_.n_shells();
        for (int M = 0; M < nshells; ++M) {
            for (int N = 0; N <= M; ++N) {
                for (int P = 0; P <= M; ++P) {
                    int Q_max = (M == P) ? N : P;
                    for (int Q = 0; Q <= Q_max; ++Q) register_quartet(M, N, P, Q, 1.0);
                }
            }
        }
    }

    const double mem_lim_gb = get_scf_memory_limit_gb(config_);
  const double est_incore_gb =
      static_cast<double>(total_ints) * static_cast<double>(sizeof(double)) / 1.0e9;
  if (mem_lim_gb > 0.0 && est_incore_gb > mem_lim_gb * 0.70) {
    if (config_.print_level > 0) {
      std::cout << "  [SCF-HPC] Exact InCore ERI (" << std::fixed << std::setprecision(2)
                << est_incore_gb << " GB) > budget (" << (mem_lim_gb * 0.70)
                << " GB); auto-fallback to Direct SCF.\n";
    }
    incore_quartets_.clear();
    incore_quartets_.shrink_to_fit();
    config_.scf_type = "direct";
    return;
  }
  incore_eri_pool_.resize(total_ints, 0.0);

    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < incore_quartets_.size(); ++i) {
        const auto& cq = incore_quartets_[i];
        integrals_->compute_shell_block_raw(
            cq.M, cq.N, cq.P, cq.Q, &incore_eri_pool_[cq.data_offset]
        );
    }

    if (config_.print_level > 0) {
        std::cout << "Done. (Quartets: " << incore_quartets_.size()
                  << " | RAM: " << (incore_eri_pool_.size() * 8.0 / (1024.0 * 1024.0)) << " MB)\n";
    }
}

void BaseSCF::init_integrals_cholesky() {
    if (config_.print_level > 0) std::cout << "  [SCF] Decomposing Integrals (Cholesky)... " << std::flush;
    internal_cholesky_ = std::make_unique<integrals::CholeskyERI>(basis_, integrals_);
    internal_cholesky_->set_threshold(config_.cholesky_threshold);
    internal_cholesky_->compute();

    L_mat_ = std::move(internal_cholesky_->get_L_mat());
    internal_cholesky_.reset();

    if (config_.print_level > 0) std::cout << " Done. Rank: " << L_mat_.cols() << "\n";
}

Eigen::VectorXd BaseSCF::smear_electrons(const Eigen::VectorXd& eps, int target_electrons) {
    Eigen::VectorXd occ = Eigen::VectorXd::Zero(nbasis_);
    int electrons_left = target_electrons;
    double degen_tol = 1e-4;
    int i = 0;
    while (i < nbasis_ && electrons_left > 0) {
        int degen_count = 1;
        double current_e = eps(i);
        while (i + degen_count < nbasis_ && std::abs(eps(i + degen_count) - current_e) < degen_tol) degen_count++;
        int electrons_to_fill = std::min(electrons_left, degen_count);
        double fraction = (double)electrons_to_fill / (double)degen_count;
        for (int k = 0; k < degen_count; ++k) occ(i + k) = fraction;
        electrons_left -= electrons_to_fill;
        i += degen_count;
    }
    return occ;
}

void BaseSCF::solve_fock(const Eigen::MatrixXd& F, Eigen::MatrixXd& C, Eigen::VectorXd& eps) {
  mshqc::utils::ScopedBlasThreads blas_fock_mt(nbasis_, 96);
    Eigen::MatrixXd Fp = X_.transpose() * F * X_;
    if (symmetrizer_ && !salc_irreps_.empty()) {
        Eigen::MatrixXd C_salc = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
        eps.resize(nbasis_);
        int start_idx = 0;
        while (start_idx < nbasis_) {
            int current_irrep = salc_irreps_[start_idx];
            int size = 0;
            while (start_idx + size < nbasis_ && salc_irreps_[start_idx + size] == current_irrep) size++;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Fp.block(start_idx, start_idx, size, size));
            C_salc.block(start_idx, start_idx, size, size) = es.eigenvectors();
            eps.segment(start_idx, size) = es.eigenvalues();
            start_idx += size;
        }
        std::vector<std::pair<double, int>> sort_data(nbasis_);
        for (int i = 0; i < nbasis_; ++i) sort_data[i] = {eps(i), i};
        std::sort(sort_data.begin(), sort_data.end());
        Eigen::MatrixXd C_sorted(nbasis_, nbasis_);
        Eigen::VectorXd eps_sorted(nbasis_);
        for (int i = 0; i < nbasis_; ++i) {
            C_sorted.col(i) = C_salc.col(sort_data[i].second);
            eps_sorted(i) = sort_data[i].first;
        }
        C.noalias() = X_ * C_sorted;
        eps = eps_sorted;
    } else {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Fp);
        eps = es.eigenvalues();
        C.noalias() = X_ * es.eigenvectors();
    }
}

void BaseSCF::print_final(const SCFResult& r) {
    if (config_.print_level < 1) return;
    std::cout << "\n=== SCF Final Results ===\n";
    std::cout << "  Total Energy      : " << std::fixed << std::setprecision(10) << r.energy_total << " Ha\n";
    std::cout << "  Iterations        : " << r.iterations << "\n";
    std::cout << "  Converged         : " << (r.converged ? "Yes" : "No") << "\n";
    std::cout << "======================================\n";
}

SCFResult BaseSCF::compute() {
    init_integrals();
    initial_guess();
    DIIS diis_a(config_.diis_max_vectors);
    DIIS diis_b(config_.diis_max_vectors);

    if (config_.scf_type == "direct" && !config_.use_df && config_.eri_method != "cholesky") {
        fock_engine_ = std::make_unique<FockBuilder>(integrals_, basis_, H_, schwarz_);
        if (pl_) fock_engine_->set_petite_list(pl_.get());
    }

    bool converged = false;
    bool is_uhf = (n_alpha_ != n_beta_);
    if (config_.print_level > 0) {
        printf("\n Iter       Energy (Ha)        Delta E    Delta P\n");
    }

    for (iter_scf_ = 1; iter_scf_ <= config_.max_iterations; iter_scf_++) {
        energy_old_ = energy_;
        Eigen::MatrixXd Pa_snapshot = P_alpha_;
        Eigen::MatrixXd Pb_snapshot = P_beta_;

        if (symmetrizer_) {
            symmetrizer_->symmetrize(P_alpha_);
            if (is_uhf) symmetrizer_->symmetrize(P_beta_);
        }

        build_fock_matrix();

        if (symmetrizer_) {
            symmetrizer_->symmetrize(F_alpha_);
            if (is_uhf) symmetrizer_->symmetrize(F_beta_);
        }
        energy_ = compute_energy();

        Eigen::MatrixXd F_eff_a = F_alpha_;
        Eigen::MatrixXd F_eff_b = F_beta_;
        double err_rms = 0.0;

        if (iter_scf_ > 1) {
            mshqc::utils::ScopedBlasThreads blas_diis_mt(nbasis_, 96);
      Eigen::MatrixXd FPS_a = F_alpha_ * P_alpha_ * S_;
            Eigen::MatrixXd err_a = X_.transpose() * (FPS_a - FPS_a.transpose()) * X_;
            diis_a.add_iteration(F_alpha_, err_a);
            F_eff_a = diis_a.extrapolate();

            if (is_uhf) {
                Eigen::MatrixXd FPS_b = F_beta_ * P_beta_ * S_;
                Eigen::MatrixXd err_b = X_.transpose() * (FPS_b - FPS_b.transpose()) * X_;
                diis_b.add_iteration(F_beta_, err_b);
                F_eff_b = diis_b.extrapolate();
                err_rms = std::hypot(err_a.norm(), err_b.norm()) / nbasis_;
            } else {
                err_rms = err_a.norm() / nbasis_;
            }
        }

        // Adaptive AO-basis virtual level-shift: F_shift = F + sigma * (S - S*P*S)
    // Aktif hanya bila level_shift diset atau SCF atom berat/transisi berosilasi (iter > 15),
    // lalu otomatis dimatikan saat err_rms < 1e-4 agar solusi akhir 100% eksak.
    double active_shift = get_scf_level_shift(config_);
    if (active_shift <= 0.0 && iter_scf_ > 15 && err_rms > 1.0e-3) {
      active_shift = (iter_scf_ > 35) ? 0.50 : 0.20;
    }
    if (err_rms > 0.0 && err_rms < 1.0e-4) {
      active_shift = 0.0;
    }
    if (active_shift > 0.0 && iter_scf_ > 1) {
      Eigen::MatrixXd SPa = S_ * P_alpha_;
      F_eff_a.noalias() += active_shift * (S_ - SPa * S_);
      if (is_uhf) {
        Eigen::MatrixXd SPb = S_ * P_beta_;
        F_eff_b.noalias() += active_shift * (S_ - SPb * S_);
      }
    }

    solve_fock(F_eff_a, C_alpha_, eps_alpha_);
    if (is_uhf) {
      if (iter_scf_ == 1 && (F_eff_a - F_eff_b).cwiseAbs().maxCoeff() < 1e-14) {
        C_beta_ = C_alpha_;
        eps_beta_ = eps_alpha_;
      } else {
        solve_fock(F_eff_b, C_beta_, eps_beta_);
      }
    } else {
            C_beta_ = C_alpha_;
            eps_beta_ = eps_alpha_;
        }

        occ_numbers_alpha_.resize(0);
        occ_numbers_beta_.resize(0);
        update_densities();

        double dE = std::abs(energy_ - energy_old_);
    double dP = is_uhf
        ? std::hypot((P_alpha_ - Pa_snapshot).norm(), (P_beta_ - Pb_snapshot).norm()) / nbasis_
        : (P_alpha_ - Pa_snapshot).norm() / nbasis_;

    if (config_.print_level > 0) {
      printf(" %4d %18.10f %10.2e %10.2e\n", iter_scf_, energy(), dE, dP);
    }

    double e_tol = (config_.energy_threshold > 0.0) ? config_.energy_threshold : 1e-12;
    double p_tol = (config_.density_threshold > 0.0) ? config_.density_threshold : 1e-12;

    bool strict_conv = (dE < e_tol && dP < p_tol && err_rms < p_tol);
    bool plateau_conv = (iter_scf_ > 40 && dE < std::max(e_tol, 1e-12) && dP < std::max(p_tol, 1e-11) && err_rms < std::max(p_tol, 1e-11));
        if (iter_scf_ > 2 && (strict_conv || plateau_conv)) {
            converged = true;
            break;
        }
    }

    // Verifikasi keadaan dasar UHF khusus radikal poliatomik (mol_.n_atoms() > 2, misal NH2 2A1 -> 2B1)
    if (is_uhf && !config_.has_initial_guess && mol_.n_atoms() > 2 && n_beta_ > 0 && (!converged || nbasis_ <= 300)) {
        double best_energy = energy_;
        bool best_conv = converged;
        int best_iter = iter_scf_;
        Eigen::MatrixXd best_Ca = C_alpha_, best_Cb = C_beta_;
        Eigen::MatrixXd best_Pa = P_alpha_, best_Pb = P_beta_;
        Eigen::MatrixXd best_Fa = F_alpha_, best_Fb = F_beta_;
        Eigen::VectorXd best_ea = eps_alpha_, best_eb = eps_beta_;

        auto run_candidate_scf = [&](const Eigen::MatrixXd& trial_Ca, const Eigen::MatrixXd& trial_Cb) {
            C_alpha_ = trial_Ca;
            C_beta_  = trial_Cb;
            occ_numbers_alpha_.resize(0);
            occ_numbers_beta_.resize(0);
            update_densities();
            if (fock_engine_) fock_engine_->reset();

            DIIS trial_diis_a(config_.diis_max_vectors);
            DIIS trial_diis_b(config_.diis_max_vectors);
            bool trial_conv = false;
            energy_ = 0.0;

            for (iter_scf_ = 1; iter_scf_ <= config_.max_iterations; iter_scf_++) {
                energy_old_ = energy_;
                Eigen::MatrixXd Pa_snap = P_alpha_;
                Eigen::MatrixXd Pb_snap = P_beta_;

                if (symmetrizer_) {
                    symmetrizer_->symmetrize(P_alpha_);
                    symmetrizer_->symmetrize(P_beta_);
                }
                build_fock_matrix();
                if (symmetrizer_) {
                    symmetrizer_->symmetrize(F_alpha_);
                    symmetrizer_->symmetrize(F_beta_);
                }
                energy_ = compute_energy();

                if (iter_scf_ >= 3 && energy_ > best_energy + 0.05) {
                    break;
                }

                Eigen::MatrixXd F_eff_a = F_alpha_;
                Eigen::MatrixXd F_eff_b = F_beta_;
                double err_rms = 0.0;

                if (iter_scf_ > 1) {
                    Eigen::MatrixXd FPS_a = F_alpha_ * P_alpha_ * S_;
                    Eigen::MatrixXd err_a = X_.transpose() * (FPS_a - FPS_a.transpose()) * X_;
                    trial_diis_a.add_iteration(F_alpha_, err_a);
                    F_eff_a = trial_diis_a.extrapolate();

                    Eigen::MatrixXd FPS_b = F_beta_ * P_beta_ * S_;
                    Eigen::MatrixXd err_b = X_.transpose() * (FPS_b - FPS_b.transpose()) * X_;
                    trial_diis_b.add_iteration(F_beta_, err_b);
                    F_eff_b = trial_diis_b.extrapolate();
                    err_rms = std::hypot(err_a.norm(), err_b.norm()) / nbasis_;
                }

                solve_fock(F_eff_a, C_alpha_, eps_alpha_);
                solve_fock(F_eff_b, C_beta_, eps_beta_);
                occ_numbers_alpha_.resize(0);
                occ_numbers_beta_.resize(0);
                update_densities();

                double dE = std::abs(energy_ - energy_old_);
        double dP = std::hypot((P_alpha_ - Pa_snap).norm(), (P_beta_ - Pb_snap).norm()) / nbasis_;

        double e_tol = (config_.energy_threshold > 0.0) ? config_.energy_threshold : 1e-12;
        double p_tol = (config_.density_threshold > 0.0) ? config_.density_threshold : 1e-12;

                bool strict_conv = (dE < e_tol && dP < p_tol && err_rms < p_tol);
                bool plateau_conv = (iter_scf_ > 40 && dE < 1e-12 && dP < 1e-10 && err_rms < 1e-10);
                if (iter_scf_ > 2 && (strict_conv || plateau_conv)) {
                    trial_conv = true;
                    break;
                }
            }

            if (trial_conv && energy_ < best_energy - 1e-7) {
                best_energy = energy_;
                best_conv   = trial_conv;
                best_iter   = iter_scf_;
                best_Ca     = C_alpha_;
                best_Cb     = C_beta_;
                best_Pa     = P_alpha_;
                best_Pb     = P_beta_;
                best_Fa     = F_alpha_;
                best_Fb     = F_beta_;
                best_ea     = eps_alpha_;
                best_eb     = eps_beta_;
                return true;
            }
            return false;
        };

        Eigen::MatrixXd S_Ca_occ = S_ * best_Ca.leftCols(n_alpha_);
        std::vector<int> somo_beta_indices;
        for (int a = n_beta_; a < nbasis_; ++a) {
            double proj = (S_Ca_occ.transpose() * best_Cb.col(a)).squaredNorm();
            if (proj > 0.4) {
                somo_beta_indices.push_back(a);
            }
        }

        for (int a_idx : somo_beta_indices) {
            for (int i_off = 1; i_off <= std::min(2, n_beta_); ++i_off) {
                int i_idx = n_beta_ - i_off;
                if (i_off == 2 && std::abs(best_eb(n_beta_ - 1) - best_eb(n_beta_ - 2)) > 0.25) continue;
                Eigen::MatrixXd trial_Cb = best_Cb;
                trial_Cb.col(i_idx).swap(trial_Cb.col(a_idx));
                if (run_candidate_scf(best_Ca, trial_Cb)) {
                    break;
                }
            }
        }

        energy_    = best_energy;
        converged  = best_conv;
        iter_scf_  = best_iter;
        C_alpha_   = best_Ca;
        C_beta_    = best_Cb;
        P_alpha_   = best_Pa;
        P_beta_    = best_Pb;
        F_alpha_   = best_Fa;
        F_beta_    = best_Fb;
        eps_alpha_ = best_ea;
        eps_beta_  = best_eb;
    }

    int final_iter = iter_scf_;
    if (converged) {
        if (symmetrizer_) {
            symmetrizer_->symmetrize(P_alpha_);
            if (is_uhf) symmetrizer_->symmetrize(P_beta_);
        }
        // Gunakan iter_scf_ = 1 agar Fock terakhir dibangun secara eksak tanpa residu screening inkremental
        iter_scf_ = 1;
        build_fock_matrix();
        if (symmetrizer_) {
            symmetrizer_->symmetrize(F_alpha_);
            if (is_uhf) symmetrizer_->symmetrize(F_beta_);
        }
        energy_ = compute_energy();
        if (is_uhf && C_alpha_.rows() == nbasis_ && C_beta_.rows() == nbasis_) {
          auto canonicalize_uhf_subspace = [&](const Eigen::MatrixXd& F_ao, Eigen::MatrixXd& C,
                                               Eigen::VectorXd& eps, int nocc) {
            const int nvir = nbasis_ - nocc;
            eps.resize(nbasis_);
            if (nocc > 0) {
              Eigen::MatrixXd Co = C.leftCols(nocc);
              Eigen::MatrixXd Foo = Co.transpose() * F_ao * Co;
              Foo = 0.5 * (Foo + Foo.transpose()).eval();
              Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_o(Foo);
              C.leftCols(nocc).noalias() = Co * es_o.eigenvectors();
              eps.head(nocc) = es_o.eigenvalues();
            }
            if (nvir > 0) {
              Eigen::MatrixXd Cv = C.rightCols(nvir);
              Eigen::MatrixXd Fvv = Cv.transpose() * F_ao * Cv;
              Fvv = 0.5 * (Fvv + Fvv.transpose()).eval();
              Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_v(Fvv);
              C.rightCols(nvir).noalias() = Cv * es_v.eigenvectors();
              eps.tail(nvir) = es_v.eigenvalues();
            }
          };
          Eigen::MatrixXd Ca_prev = C_alpha_, Cb_prev = C_beta_;
      Eigen::VectorXd ea_prev = eps_alpha_, eb_prev = eps_beta_;
      solve_fock(F_alpha_, C_alpha_, eps_alpha_);
      solve_fock(F_beta_, C_beta_, eps_beta_);
      double ovlp_a = (n_alpha_ > 0) ? (Ca_prev.leftCols(n_alpha_).transpose() * S_ * C_alpha_.leftCols(n_alpha_)).squaredNorm() : 0.0;
      double ovlp_b = (n_beta_ > 0) ? (Cb_prev.leftCols(n_beta_).transpose() * S_ * C_beta_.leftCols(n_beta_)).squaredNorm() : 0.0;
      if (std::abs(ovlp_a - n_alpha_) > 1e-4 || std::abs(ovlp_b - n_beta_) > 1e-4) {
        C_alpha_ = Ca_prev; C_beta_ = Cb_prev;
        eps_alpha_ = ea_prev; eps_beta_ = eb_prev;
        canonicalize_uhf_subspace(F_alpha_, C_alpha_, eps_alpha_, n_alpha_);
        canonicalize_uhf_subspace(F_beta_, C_beta_, eps_beta_, n_beta_);
      }
      occ_numbers_alpha_.resize(0);
      occ_numbers_beta_.resize(0);
      update_densities();
        } else if (is_uhf) {
          solve_fock(F_alpha_, C_alpha_, eps_alpha_);
          solve_fock(F_beta_, C_beta_, eps_beta_);
          occ_numbers_alpha_.resize(0);
          occ_numbers_beta_.resize(0);
          update_densities();
        } else {
          solve_fock(F_alpha_, C_alpha_, eps_alpha_);
            Eigen::MatrixXd Pa_prev = P_alpha_;
            occ_numbers_alpha_ = smear_electrons(eps_alpha_, n_alpha_);
            C_beta_ = C_alpha_;
            eps_beta_ = eps_alpha_;
            occ_numbers_beta_ = occ_numbers_alpha_;
            update_densities();

            // Hanya bangun ulang Fock jika smearing mengubah matriks densitas
            if ((P_alpha_ - Pa_prev).cwiseAbs().maxCoeff() > 1e-12) {
                iter_scf_ = 1;
                build_fock_matrix();
                if (symmetrizer_) symmetrizer_->symmetrize(F_alpha_);
                F_beta_ = F_alpha_;
                energy_ = compute_energy();
            }
        }
    }

    std::vector<int> irreps_a(nbasis_, 0), irreps_b(nbasis_, 0);
    if (symmetrizer_) {
        if (config_.print_level > 0) std::cout << "\n  [Symmetry] Sorting Orbitals by Irrep...\n";
        irreps_a = symmetrizer_->assign_mo_irreps(C_alpha_);
        irreps_b = is_uhf ? symmetrizer_->assign_mo_irreps(C_beta_) : irreps_a;
    }

    if (config_.use_df && config_.scf_type != "ooc" && L_mat_.rows() == 0 && L_mat_.cols() > 0) {
        if (config_.print_level > 0) std::cout << "\n  [SCF] Reloading df_tensor.h5 to packed RAM (50% footprint)...\n";
        const int n_aux = static_cast<int>(L_mat_.cols());
        const int n_tri = integrals::ao_tri_size(nbasis_);
        L_mat_.resize(n_tri, n_aux);
        utils::HDF5TensorIO io("df_tensor.h5", utils::HDF5TensorIO::Mode::READ_ONLY);
        constexpr int kChunk = 64;
        for (int K0 = 0; K0 < n_aux; K0 += kChunk) {
            const int k_sz = std::min(kChunk, n_aux - K0);
            Eigen::MatrixXd sq_chunk(static_cast<Eigen::Index>(nbasis_) * nbasis_, k_sz);
            io.read_slice_4d("df_tensor", {K0, 0, 0, 0}, {k_sz, (long)nbasis_, (long)nbasis_, 1}, sq_chunk.data());
            #pragma omp parallel for schedule(static)
            for (int c = 0; c < k_sz; ++c) {
                const double* s_col = sq_chunk.col(c).data();
                double* d_col = L_mat_.col(K0 + c).data();
                for (int r = 0; r < nbasis_; ++r) {
                    const int base = r * (r + 1) / 2;
                    for (int q = 0; q <= r; ++q) {
                        d_col[base + q] = s_col[r + q * nbasis_];
                    }
                }
            }
        }
    }

    SCFResult r;
    r.energy_total = energy();
    r.iterations = final_iter;
    r.converged = converged;
    r.C_alpha = C_alpha_;
    r.C_beta = C_beta_;
    r.P_alpha = P_alpha_;
    r.P_beta = P_beta_;
    r.F_alpha = F_alpha_;
    r.F_beta = F_beta_;
    r.orbital_energies_alpha = eps_alpha_;
    r.orbital_energies_beta = eps_beta_;
    r.irreps_alpha = irreps_a;
    r.irreps_beta = irreps_b;
    r.n_occ_alpha = n_alpha_;
    r.n_occ_beta = n_beta_;
    r.L_mat = L_mat_;
    print_final(r);
    return r;
}

RHF::RHF(const Molecule& mol, const BasisSet& basis, std::shared_ptr<IntegralEngine> integrals, std::shared_ptr<PointGroup> pg, std::shared_ptr<PetiteList> pl, const SCFConfig& config)
    : BaseSCF(mol, basis, integrals, pg, pl, mol.n_electrons() / 2, mol.n_electrons() / 2, config) {
    if (mol.n_electrons() % 2 != 0) throw std::runtime_error("CRITICAL: RHF requires an even number of electrons.");
    G_J_accum_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    G_accum_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    P_old_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
}

void RHF::initial_guess() {
    if (config_.has_initial_guess && config_.init_C_alpha.rows() == nbasis_) {
        Eigen::MatrixXd C_orth = orthonormalize_mo_guess(config_.init_C_alpha, S_);
        if (C_orth.rows() == nbasis_) {
            C_alpha_ = std::move(C_orth);
            C_beta_ = C_alpha_;
            occ_numbers_alpha_.resize(0);
            update_densities();
            energy_ = 0.0;
            return;
        }
    }
    if (config_.print_level > 0) std::cout << "  [RHF] Generating Initial Guess (SAD)...\n";
    Eigen::MatrixXd P_sad = SADGuess::build(mol_, basis_);
    P_alpha_ = 0.5 * P_sad;
    P_beta_  = P_alpha_;
    C_alpha_.resize(0, 0);
    C_beta_.resize(0, 0);
    energy_ = 0.0;
}

void RHF::update_densities() {
    P_alpha_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    if (occ_numbers_alpha_.size() == nbasis_) {
        for (int i = 0; i < nbasis_; ++i) if (occ_numbers_alpha_(i) > 1e-12) P_alpha_ += occ_numbers_alpha_(i) * C_alpha_.col(i) * C_alpha_.col(i).transpose();
    } else {
        if (n_alpha_ > 0) P_alpha_.noalias() = C_alpha_.leftCols(n_alpha_) * C_alpha_.leftCols(n_alpha_).transpose();
    }
    P_beta_ = P_alpha_;
}

void RHF::build_fock_matrix() {
    if (iter_scf_ == 1) {
        G_accum_.setZero();
        G_J_accum_.setZero();
    }
    if (config_.eri_method == "cholesky" || config_.use_df) {
        Eigen::MatrixXd dP = (iter_scf_ == 1) ? P_alpha_ : (P_alpha_ - P_old_);
        if (iter_scf_ > 1 && dP.cwiseAbs().maxCoeff() < 1e-15) {
            F_alpha_ = H_ + G_accum_;
            F_beta_  = F_alpha_;
            return;
        }

        bool use_mo_alg = (C_alpha_.rows() == nbasis_) && (C_alpha_.cols() == nbasis_) && (occ_numbers_alpha_.size() == 0);
        Eigen::MatrixXd Ca_factor = use_mo_alg ? C_alpha_.leftCols(n_alpha_) : factorize_psd_density(P_alpha_);

        Eigen::MatrixXd J_mat  = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
        Eigen::MatrixXd K_acc  = Eigen::MatrixXd::Zero(nbasis_, nbasis_);

        int n_chol = L_mat_.cols();
        bool is_ooc = (L_mat_.rows() == 0 && n_chol > 0);

        std::unique_ptr<utils::HDF5TensorIO> io = nullptr;
        if (is_ooc) io = std::make_unique<utils::HDF5TensorIO>("df_tensor.h5", utils::HDF5TensorIO::Mode::READ_ONLY);

        int chunk_size = is_ooc ? 128 : n_chol;
        Eigen::Map<const Eigen::VectorXd> P_flat(P_alpha_.data(), nbasis_ * nbasis_);

        const bool is_packed = (!is_ooc) && integrals::is_packed_ao_matrix(L_mat_, nbasis_);
        if (!is_ooc) {
            J_mat = integrals::compute_coulomb_from_ao_factor(L_mat_, P_alpha_, nbasis_);
        }

        for (int K_start = 0; K_start < n_chol; K_start += chunk_size) {
            int K_end = std::min(n_chol, K_start + chunk_size);
            int k_size = K_end - K_start;

            Eigen::MatrixXd L_chunk;
            if (is_ooc) {
                L_chunk.resize(nbasis_ * nbasis_, k_size);
                io->read_slice_4d("df_tensor", {K_start, 0, 0, 0}, {k_size, (long)nbasis_, (long)nbasis_, 1}, L_chunk.data());

                Eigen::VectorXd X_J_chunk = L_chunk.transpose() * P_flat;
                Eigen::VectorXd J_flat_chunk = L_chunk * X_J_chunk;
                J_mat += Eigen::Map<Eigen::MatrixXd>(J_flat_chunk.data(), nbasis_, nbasis_);
            }

            const double* L_ptr = is_ooc ? L_chunk.data() : L_mat_.col(K_start).data();
            compute_exchange_cholesky_single(L_ptr, nbasis_, k_size, Ca_factor, K_acc, is_packed);
        }

        G_accum_ = 2.0 * J_mat - K_acc;
        G_accum_ = 0.5 * (G_accum_ + G_accum_.transpose());
        F_alpha_ = H_ + G_accum_;
    } else if (config_.scf_type == "incore") {
        Eigen::MatrixXd dP = (iter_scf_ == 1) ? P_alpha_ : (P_alpha_ - P_old_).eval();
        double max_dP = dP.cwiseAbs().maxCoeff();

        if (iter_scf_ > 1 && max_dP < 1e-12) {
            F_alpha_ = H_ + G_accum_;
        } else {
            int nshells = basis_.n_shells();
            Eigen::MatrixXd P_max = compute_shell_pmax(nshells, shell_starts_, shell_sizes_, dP);
            Eigen::MatrixXd dG = Eigen::MatrixXd::Zero(nbasis_, nbasis_);

            #pragma omp parallel
            {
                Eigen::MatrixXd G_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                #pragma omp for schedule(static) nowait
                for (size_t i = 0; i < incore_quartets_.size(); ++i) {
                    const auto& cq = incore_quartets_[i];
                    int M = cq.M; int N = cq.N; int P = cq.P; int Q = cq.Q;

                    double bound = schwarz_(M, N) * schwarz_(P, Q);
                    double local_P_max = std::max({P_max(M, N), P_max(P, Q), P_max(M, P), P_max(M, Q), P_max(N, P), P_max(N, Q)});
                    if (bound * local_P_max < 1e-14) continue;

                    double fac = cq.weight;
                    if (M == N) fac *= 0.5;
                    if (P == Q) fac *= 0.5;
                    if (M == P && N == Q) fac *= 0.5;

                    int dimM = shell_sizes_[M]; int offM = shell_starts_[M];
                    int dimN = shell_sizes_[N]; int offN = shell_starts_[N];
                    int dimP = shell_sizes_[P]; int offP = shell_starts_[P];
                    int dimQ = shell_sizes_[Q]; int offQ = shell_starts_[Q];

                    const double* I_ptr = &incore_eri_pool_[cq.data_offset];

                    for (int q = 0; q < dimQ; ++q) {
                        int sig = offQ + q;
                        for (int p = 0; p < dimP; ++p) {
                            int lam = offP + p;
                            double dP_ls_4 = dP(lam, sig) * 4.0;
                            for (int n = 0; n < dimN; ++n) {
                                int nu = offN + n;
                                double dP_ns = dP(nu, sig);
                                double dP_nl = dP(nu, lam);
                                for (int m = 0; m < dimM; ++m) {
                                    int mu = offM + m;
                                    double val = *I_ptr++;
                                    if (std::abs(val) < 1e-14) continue;

                                    double v = val * fac;
                                    G_local(mu, nu)   += v * dP_ls_4;
                                    G_local(lam, sig) += v * (dP(mu, nu) * 4.0);

                                    G_local(mu, lam) -= v * dP_ns;
                                    G_local(nu, lam) -= v * dP(mu, sig);
                                    G_local(mu, sig) -= v * dP_nl;
                                    G_local(nu, sig) -= v * dP(mu, lam);
                                }
                            }
                        }
                    }
                }
                #pragma omp critical
                { dG.noalias() += G_local; }
            }
            dG = (dG + dG.transpose()).eval();
            G_accum_ += dG;
            F_alpha_ = H_ + G_accum_;
        }
    } else {
        Eigen::MatrixXd P_empty, F_empty;
        fock_engine_->compute(P_alpha_, P_empty, F_alpha_, F_empty);
    }
    P_old_ = P_alpha_;
    F_beta_ = F_alpha_;
}

void UHF::build_fock_matrix() {
    if (iter_scf_ == 1) {
        G_accum_a_.setZero();
        G_accum_b_.setZero();
        G_J_accum_a_.setZero();
        G_J_accum_b_.setZero();
    }
    if (config_.eri_method == "cholesky" || config_.use_df) {
     if (iter_scf_ > 1) {
         double max_dP = std::max((P_alpha_ - P_alpha_old_).cwiseAbs().maxCoeff(),
                                  (P_beta_  - P_beta_old_).cwiseAbs().maxCoeff());
         if (max_dP < 1e-15) {
             F_alpha_ = H_ + G_accum_a_;
             F_beta_  = H_ + G_accum_b_;
             return;
         }
     }

     Eigen::MatrixXd P_tot = P_alpha_ + P_beta_;
     bool use_mo_alg = (C_alpha_.rows() == nbasis_) && (C_beta_.rows() == nbasis_);
     bool sym_spin = (P_alpha_ - P_beta_).cwiseAbs().maxCoeff() < 1e-14;

        Eigen::MatrixXd Ca_fac = use_mo_alg ? C_alpha_.leftCols(n_alpha_) : factorize_psd_density(P_alpha_);
        Eigen::MatrixXd Cb_fac = use_mo_alg ? C_beta_.leftCols(n_beta_)   : (sym_spin ? Eigen::MatrixXd(nbasis_, 0) : factorize_psd_density(P_beta_));

        Eigen::MatrixXd J_mat  = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
        Eigen::MatrixXd Ka_acc = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
        Eigen::MatrixXd Kb_acc = Eigen::MatrixXd::Zero(nbasis_, nbasis_);

        int n_chol = L_mat_.cols();
        bool is_ooc = (L_mat_.rows() == 0 && n_chol > 0);
        const bool is_packed = (!is_ooc) && integrals::is_packed_ao_matrix(L_mat_, nbasis_);

        std::unique_ptr<utils::HDF5TensorIO> io = nullptr;
        if (is_ooc) io = std::make_unique<utils::HDF5TensorIO>("df_tensor.h5", utils::HDF5TensorIO::Mode::READ_ONLY);

        int chunk_size = is_ooc ? 128 : n_chol;
        Eigen::Map<const Eigen::VectorXd> P_tot_flat(P_tot.data(), nbasis_ * nbasis_);

        if (!is_ooc) {
            J_mat = integrals::compute_coulomb_from_ao_factor(L_mat_, P_tot, nbasis_);
        }

        for (int K_start = 0; K_start < n_chol; K_start += chunk_size) {
            int K_end = std::min(n_chol, K_start + chunk_size);
            int k_size = K_end - K_start;

            Eigen::MatrixXd L_chunk;
            if (is_ooc) {
                L_chunk.resize(static_cast<Eigen::Index>(nbasis_) * nbasis_, k_size);
                io->read_slice_4d("df_tensor", {K_start, 0, 0, 0}, {k_size, (long)nbasis_, (long)nbasis_, 1}, L_chunk.data());

                Eigen::VectorXd X_J_chunk = L_chunk.transpose() * P_tot_flat;
                Eigen::VectorXd J_flat_chunk = L_chunk * X_J_chunk;
                J_mat.noalias() += Eigen::Map<Eigen::MatrixXd>(J_flat_chunk.data(), nbasis_, nbasis_);
            }

            const double* L_ptr = is_ooc ? L_chunk.data() : L_mat_.col(K_start).data();
            if (sym_spin && !use_mo_alg) {
                compute_exchange_cholesky_single(L_ptr, nbasis_, k_size, Ca_fac, Ka_acc, is_packed);
                Kb_acc = Ka_acc;
            } else {
                compute_exchange_cholesky_dual(L_ptr, nbasis_, k_size, Ca_fac, Cb_fac, Ka_acc, Kb_acc, is_packed);
            }
        }

        G_accum_a_ = 0.5 * ((J_mat - Ka_acc) + (J_mat - Ka_acc).transpose());
        G_accum_b_ = 0.5 * ((J_mat - Kb_acc) + (J_mat - Kb_acc).transpose());
        F_alpha_ = H_ + G_accum_a_;
        F_beta_  = H_ + G_accum_b_;
        P_alpha_old_ = P_alpha_;
        P_beta_old_  = P_beta_;
    } else if (config_.scf_type == "incore") {
        Eigen::MatrixXd P_tot = P_alpha_ + P_beta_;
        Eigen::MatrixXd Pa_k = (iter_scf_ == 1 && C_alpha_.rows() == nbasis_ && n_alpha_ > 0)
            ? (C_alpha_.leftCols(n_alpha_) * C_alpha_.leftCols(n_alpha_).transpose()).eval()
            : P_alpha_;
        Eigen::MatrixXd Pb_k = (iter_scf_ == 1 && C_beta_.rows() == nbasis_ && n_beta_ > 0)
            ? (C_beta_.leftCols(n_beta_) * C_beta_.leftCols(n_beta_).transpose()).eval()
            : P_beta_;

        Eigen::MatrixXd dP_tot, dPa, dPb;
        if (iter_scf_ == 1) {
            dP_tot = P_tot;
            dPa    = Pa_k;
            dPb    = Pb_k;
        } else {
            dP_tot = (P_tot - G_J_accum_a_).eval();
            dPa    = (Pa_k - P_alpha_old_).eval();
            dPb    = (Pb_k - P_beta_old_).eval();
        }
        double max_dP = std::max({dP_tot.cwiseAbs().maxCoeff(), dPa.cwiseAbs().maxCoeff(), dPb.cwiseAbs().maxCoeff()});

        if (iter_scf_ > 1 && max_dP < 1e-12) {
            F_alpha_ = H_ + G_accum_a_;
            F_beta_  = H_ + G_accum_b_;
        } else {
            int nshells = basis_.n_shells();
            Eigen::MatrixXd P_max = compute_shell_pmax(nshells, shell_starts_, shell_sizes_, dPa, &dPb);

            Eigen::MatrixXd dGa = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
            Eigen::MatrixXd dGb = Eigen::MatrixXd::Zero(nbasis_, nbasis_);

            #pragma omp parallel
            {
                Eigen::MatrixXd Ga_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                Eigen::MatrixXd Gb_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);

                #pragma omp for schedule(dynamic, 32) nowait
                for (size_t i = 0; i < incore_quartets_.size(); ++i) {
                    const auto& cq = incore_quartets_[i];
                    int M = cq.M; int N = cq.N; int P = cq.P; int Q = cq.Q;

                    double bound = schwarz_(M, N) * schwarz_(P, Q);
                    double local_P_max = std::max({P_max(M, N), P_max(P, Q), P_max(M, P), P_max(M, Q), P_max(N, P), P_max(N, Q)});
                    if (bound * local_P_max < 1e-14) continue;

                    double fac = cq.weight;
                    if (M == N) fac *= 0.5;
                    if (P == Q) fac *= 0.5;
                    if (M == P && N == Q) fac *= 0.5;

                    int dimM = shell_sizes_[M]; int offM = shell_starts_[M];
                    int dimN = shell_sizes_[N]; int offN = shell_starts_[N];
                    int dimP = shell_sizes_[P]; int offP = shell_starts_[P];
                    int dimQ = shell_sizes_[Q]; int offQ = shell_starts_[Q];

                    const double* I_ptr = &incore_eri_pool_[cq.data_offset];

                    for (int q = 0; q < dimQ; ++q) {
                        int sig = offQ + q;
                        const double* dPa_sig = dPa.col(sig).data();
                        const double* dPb_sig = dPb.col(sig).data();

                        for (int p = 0; p < dimP; ++p) {
                            int lam = offP + p;
                            const double* dPa_lam = dPa.col(lam).data();
                            const double* dPb_lam = dPb.col(lam).data();
                            double pt_ls_2 = dP_tot(lam, sig) * 2.0;
                            double j_ls_accum = 0.0;

                            for (int n = 0; n < dimN; ++n) {
                                int nu = offN + n;
                                const double* dP_tot_nu = dP_tot.col(nu).data();
                                double dPa_ns = dPa_sig[nu];
                                double dPa_nl = dPa_lam[nu];
                                double dPb_ns = dPb_sig[nu];
                                double dPb_nl = dPb_lam[nu];

                                double ka_nl_acc = 0.0, ka_ns_acc = 0.0;
                                double kb_nl_acc = 0.0, kb_ns_acc = 0.0;

                                for (int m = 0; m < dimM; ++m) {
                                    int mu = offM + m;
                                    double val = *I_ptr++;
                                    if (std::abs(val) < 1e-14) continue;

                                    double v = val * fac;
                                    double j_val = v * pt_ls_2;
                                    j_ls_accum += v * dP_tot_nu[mu];

                                    Ga_local(mu, nu)  += j_val;
                                    Gb_local(mu, nu)  += j_val;
                                    Ga_local(mu, lam) -= v * dPa_ns;
                                    Ga_local(mu, sig) -= v * dPa_nl;
                                    Gb_local(mu, lam) -= v * dPb_ns;
                                    Gb_local(mu, sig) -= v * dPb_nl;

                                    ka_nl_acc += v * dPa_sig[mu];
                                    ka_ns_acc += v * dPa_lam[mu];
                                    kb_nl_acc += v * dPb_sig[mu];
                                    kb_ns_acc += v * dPb_lam[mu];
                                }

                                Ga_local(nu, lam) -= ka_nl_acc;
                                Ga_local(nu, sig) -= ka_ns_acc;
                                Gb_local(nu, lam) -= kb_nl_acc;
                                Gb_local(nu, sig) -= kb_ns_acc;
                            }
                            Ga_local(lam, sig) += 2.0 * j_ls_accum;
                            Gb_local(lam, sig) += 2.0 * j_ls_accum;
                        }
                    }
                }
                #pragma omp critical
                {
                    dGa.noalias() += Ga_local;
                    dGb.noalias() += Gb_local;
                }
            }

            dGa = (dGa + dGa.transpose()).eval();
            dGb = (dGb + dGb.transpose()).eval();

            G_accum_a_ += dGa;
            G_accum_b_ += dGb;
            F_alpha_ = H_ + G_accum_a_;
            F_beta_  = H_ + G_accum_b_;
            G_J_accum_a_ = P_tot;
            P_alpha_old_ = Pa_k;
            P_beta_old_  = Pb_k;
        }
    } else {
        fock_engine_->compute(P_alpha_, P_beta_, F_alpha_, F_beta_);
        P_alpha_old_ = P_alpha_;
        P_beta_old_  = P_beta_;
    }
}

double RHF::compute_energy() { return P_alpha_.cwiseProduct(H_ + F_alpha_).sum(); }

UHF::UHF(const Molecule& mol, const BasisSet& basis, std::shared_ptr<IntegralEngine> integrals, std::shared_ptr<PointGroup> pg, std::shared_ptr<PetiteList> pl, int n_alpha, int n_beta, const SCFConfig& config)
    : BaseSCF(mol, basis, integrals, pg, pl, n_alpha, n_beta, config) {
    if (n_alpha < n_beta) throw std::runtime_error("UHF Error: High spin assumption violated.");

    G_accum_a_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    G_accum_b_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    G_J_accum_a_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    G_J_accum_b_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    P_alpha_old_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    P_beta_old_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
}

void UHF::initial_guess() {
    if (config_.has_initial_guess && config_.init_C_alpha.rows() == nbasis_) {
        Eigen::MatrixXd Ca_orth = orthonormalize_mo_guess(config_.init_C_alpha, S_);
        const Eigen::MatrixXd& Cb_src = (config_.init_C_beta.rows() == nbasis_) ? config_.init_C_beta : config_.init_C_alpha;
        Eigen::MatrixXd Cb_orth = orthonormalize_mo_guess(Cb_src, S_);
        if (Ca_orth.rows() == nbasis_ && Cb_orth.rows() == nbasis_) {
            C_alpha_ = std::move(Ca_orth);
            C_beta_ = std::move(Cb_orth);
            occ_numbers_alpha_.resize(0);
            occ_numbers_beta_.resize(0);
            update_densities();
            energy_ = 0.0;
            return;
        }
    }
    if (config_.print_level > 0) std::cout << "  [UHF] Generating Initial Guess (SAD)...\n";
    Eigen::MatrixXd P_sad = SADGuess::build(mol_, basis_);
    P_alpha_ = 0.5 * P_sad;
    P_beta_  = 0.5 * P_sad;

    if (mol_.n_atoms() == 1) {
        solve_fock(H_, C_alpha_, eps_alpha_);
        solve_fock(H_, C_beta_, eps_beta_);
        occ_numbers_alpha_ = smear_electrons(eps_alpha_, n_alpha_);
        occ_numbers_beta_  = smear_electrons(eps_beta_, n_beta_);
    } else {
        C_alpha_.resize(0, 0);
        C_beta_.resize(0, 0);
    }

    energy_ = 0.0;
}

void UHF::update_densities() {
    P_alpha_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    P_beta_  = Eigen::MatrixXd::Zero(nbasis_, nbasis_);

    if (n_alpha_ > 0) {
        P_alpha_.noalias() = C_alpha_.leftCols(n_alpha_) * C_alpha_.leftCols(n_alpha_).transpose();
    }
    if (n_beta_ > 0) {
        P_beta_.noalias() = C_beta_.leftCols(n_beta_) * C_beta_.leftCols(n_beta_).transpose();
    }
}

double UHF::compute_energy() { return 0.5 * (P_alpha_.cwiseProduct(H_ + F_alpha_).sum() + P_beta_.cwiseProduct(H_ + F_beta_).sum()); }

double UHF::compute_s2(const Eigen::MatrixXd& Ca, const Eigen::MatrixXd& Cb, const Eigen::MatrixXd& S) {
    double sz = 0.5 * (n_alpha_ - n_beta_);
    if (n_alpha_ == 0 || n_beta_ == 0) return sz * (sz + 1.0);
    return sz * (sz + 1.0) + n_beta_ - (Ca.leftCols(n_alpha_).transpose() * S * Cb.leftCols(n_beta_)).array().square().sum();
}

ROHF::ROHF(const Molecule& mol, const BasisSet& basis, std::shared_ptr<IntegralEngine> integrals, std::shared_ptr<PointGroup> pg, std::shared_ptr<PetiteList> pl, int n_alpha, int n_beta, const SCFConfig& config)
    : BaseSCF(mol, basis, integrals, pg, pl, n_alpha, n_beta, config) {
    if (n_alpha < n_beta) throw std::runtime_error("ROHF Error: Alpha must be >= Beta.");
    G_accum_a_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    G_accum_b_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    P_alpha_old_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    P_beta_old_  = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    G_J_accum_a_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    G_J_accum_b_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
}

void ROHF::initial_guess() {
    if (config_.has_initial_guess && config_.init_C_alpha.rows() == nbasis_) {
        Eigen::MatrixXd C_orth = orthonormalize_mo_guess(config_.init_C_alpha, S_);
        if (C_orth.rows() == nbasis_) {
            C_alpha_ = std::move(C_orth);
            C_beta_ = C_alpha_;
            occ_numbers_alpha_.resize(0);
            occ_numbers_beta_.resize(0);
            update_densities();
            energy_ = 0.0;
            return;
        }
    }
    if (config_.print_level > 0) std::cout << "  [ROHF] Generating Initial Guess (SAD)...\n";
    Eigen::MatrixXd P_sad = SADGuess::build(mol_, basis_);
    double n_tot = static_cast<double>(n_alpha_ + n_beta_);
    double frac_a = (n_tot > 0.0) ? (static_cast<double>(n_alpha_) / n_tot) : 0.5;
    double frac_b = (n_tot > 0.0) ? (static_cast<double>(n_beta_)  / n_tot) : 0.5;
    P_alpha_ = frac_a * P_sad;
    P_beta_  = frac_b * P_sad;
    C_alpha_.resize(0, 0);
    C_beta_.resize(0, 0);
    energy_ = 0.0;
}

void ROHF::update_densities() {
    P_alpha_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    P_beta_  = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    if (occ_numbers_alpha_.size() == nbasis_) {
        for (int i = 0; i < nbasis_; ++i) {
            if (occ_numbers_alpha_(i) > 1e-12) P_alpha_ += occ_numbers_alpha_(i) * C_alpha_.col(i) * C_alpha_.col(i).transpose();
            if (occ_numbers_beta_(i) > 1e-12)  P_beta_  += occ_numbers_beta_(i) * C_alpha_.col(i) * C_alpha_.col(i).transpose();
        }
    } else {
        if (n_alpha_ > 0) P_alpha_.noalias() = C_alpha_.leftCols(n_alpha_) * C_alpha_.leftCols(n_alpha_).transpose();
        if (n_beta_ > 0)  P_beta_.noalias()  = C_alpha_.leftCols(n_beta_) * C_alpha_.leftCols(n_beta_).transpose();
    }
}

void ROHF::build_fock_matrix() {
    if (iter_scf_ == 1) {
        G_accum_a_.setZero();
        G_accum_b_.setZero();
        G_J_accum_a_.setZero();
        G_J_accum_b_.setZero();
    }
    if (config_.eri_method == "cholesky" || config_.use_df) {
        Eigen::MatrixXd dPa = (iter_scf_ == 1) ? P_alpha_ : (P_alpha_ - P_alpha_old_);
        Eigen::MatrixXd dPb = (iter_scf_ == 1) ? P_beta_  : (P_beta_ - P_beta_old_);
        Eigen::MatrixXd dP_tot = dPa + dPb;

        if (iter_scf_ > 1 && dP_tot.cwiseAbs().maxCoeff() < 1e-12) {
            F_alpha_ = H_ + G_accum_a_;
            F_beta_  = H_ + G_accum_b_;
            return;
        }

        Eigen::MatrixXd P_tot = P_alpha_ + P_beta_;
        bool use_mo_alg = (C_alpha_.rows() == nbasis_) && (occ_numbers_alpha_.size() == 0);
        Eigen::MatrixXd Ca_factor = use_mo_alg ? C_alpha_.leftCols(n_alpha_) : factorize_psd_density(P_alpha_);
        Eigen::MatrixXd Cb_factor = use_mo_alg ? C_alpha_.leftCols(n_beta_)  : factorize_psd_density(P_beta_);

        Eigen::MatrixXd J_mat  = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
        Eigen::MatrixXd Ka_acc = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
        Eigen::MatrixXd Kb_acc = Eigen::MatrixXd::Zero(nbasis_, nbasis_);

        int n_chol = L_mat_.cols();
        bool is_ooc = (L_mat_.rows() == 0 && n_chol > 0);
        std::unique_ptr<utils::HDF5TensorIO> io = nullptr;
        if (is_ooc) io = std::make_unique<utils::HDF5TensorIO>("df_tensor.h5", utils::HDF5TensorIO::Mode::READ_ONLY);
        int chunk_size = is_ooc ? 128 : n_chol;

        Eigen::Map<const Eigen::VectorXd> P_tot_flat(P_tot.data(), nbasis_ * nbasis_);

        const bool is_packed = (!is_ooc) && integrals::is_packed_ao_matrix(L_mat_, nbasis_);
        if (!is_ooc) {
            J_mat = integrals::compute_coulomb_from_ao_factor(L_mat_, P_tot, nbasis_);
        }

        for (int K_start = 0; K_start < n_chol; K_start += chunk_size) {
            int K_end = std::min(n_chol, K_start + chunk_size);
            int k_size = K_end - K_start;

            Eigen::MatrixXd L_chunk;
            if (is_ooc) {
                L_chunk.resize(nbasis_ * nbasis_, k_size);
                io->read_slice_4d("df_tensor", {K_start, 0, 0, 0}, {k_size, (long)nbasis_, (long)nbasis_, 1}, L_chunk.data());

                Eigen::VectorXd X_J_chunk = L_chunk.transpose() * P_tot_flat;
                Eigen::VectorXd J_flat_chunk = L_chunk * X_J_chunk;
                J_mat += Eigen::Map<Eigen::MatrixXd>(J_flat_chunk.data(), nbasis_, nbasis_);
            }

            const double* L_ptr = is_ooc ? L_chunk.data() : L_mat_.col(K_start).data();
            compute_exchange_cholesky_dual(L_ptr, nbasis_, k_size, Ca_factor, Cb_factor, Ka_acc, Kb_acc, is_packed);
        }

        G_accum_a_ = 0.5 * ((J_mat - Ka_acc) + (J_mat - Ka_acc).transpose());
        G_accum_b_ = 0.5 * ((J_mat - Kb_acc) + (J_mat - Kb_acc).transpose());
        F_alpha_ = H_ + G_accum_a_;
        F_beta_  = H_ + G_accum_b_;
    } else if (config_.scf_type == "incore") {
        Eigen::MatrixXd dPa = (iter_scf_ == 1) ? P_alpha_ : (P_alpha_ - P_alpha_old_).eval();
        Eigen::MatrixXd dPb = (iter_scf_ == 1) ? P_beta_  : (P_beta_  - P_beta_old_).eval();
        Eigen::MatrixXd dP_tot = dPa + dPb;
        double max_dP = std::max(dPa.cwiseAbs().maxCoeff(), dPb.cwiseAbs().maxCoeff());

        if (iter_scf_ > 1 && max_dP < 1e-12) {
            F_alpha_ = H_ + G_accum_a_;
            F_beta_  = H_ + G_accum_b_;
        } else {
            int nshells = basis_.n_shells();
            Eigen::MatrixXd P_max = compute_shell_pmax(nshells, shell_starts_, shell_sizes_, dPa, &dPb);

            Eigen::MatrixXd dGa = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
            Eigen::MatrixXd dGb = Eigen::MatrixXd::Zero(nbasis_, nbasis_);

            #pragma omp parallel
            {
                Eigen::MatrixXd Ga_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                Eigen::MatrixXd Gb_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);

                #pragma omp for schedule(static) nowait
                for (size_t i = 0; i < incore_quartets_.size(); ++i) {
                    const auto& cq = incore_quartets_[i];
                    int M = cq.M; int N = cq.N; int P = cq.P; int Q = cq.Q;

                    double bound = schwarz_(M, N) * schwarz_(P, Q);
                    double local_P_max = std::max({P_max(M, N), P_max(P, Q), P_max(M, P), P_max(M, Q), P_max(N, P), P_max(N, Q)});
                    if (bound * local_P_max < 1e-14) continue;

                    double fac = cq.weight;
                    if (M == N) fac *= 0.5;
                    if (P == Q) fac *= 0.5;
                    if (M == P && N == Q) fac *= 0.5;

                    int dimM = shell_sizes_[M]; int offM = shell_starts_[M];
                    int dimN = shell_sizes_[N]; int offN = shell_starts_[N];
                    int dimP = shell_sizes_[P]; int offP = shell_starts_[P];
                    int dimQ = shell_sizes_[Q]; int offQ = shell_starts_[Q];

                    const double* I_ptr = &incore_eri_pool_[cq.data_offset];

                    for (int q = 0; q < dimQ; ++q) {
                        int sig = offQ + q;
                        for (int p = 0; p < dimP; ++p) {
                            int lam = offP + p;
                            double pt_ls_2 = dP_tot(lam, sig) * 2.0;
                            double j_ls_accum = 0.0;

                            for (int n = 0; n < dimN; ++n) {
                                int nu = offN + n;
                                double dPa_ns = dPa(nu, sig);
                                double dPa_nl = dPa(nu, lam);
                                double dPb_ns = dPb(nu, sig);
                                double dPb_nl = dPb(nu, lam);

                                for (int m = 0; m < dimM; ++m) {
                                    int mu = offM + m;
                                    double val = *I_ptr++;
                                    if (std::abs(val) < 1e-14) continue;

                                    double v = val * fac;
                                    double j_val = v * pt_ls_2;
                                    Ga_local(mu, nu) += j_val;
                                    Gb_local(mu, nu) += j_val;
                                    j_ls_accum += v * (dP_tot(mu, nu) * 2.0);

                                    Ga_local(mu, lam) -= v * dPa_ns;
                                    Ga_local(nu, lam) -= v * dPa(mu, sig);
                                    Ga_local(mu, sig) -= v * dPa_nl;
                                    Ga_local(nu, sig) -= v * dPa(mu, lam);

                                    Gb_local(mu, lam) -= v * dPb_ns;
                                    Gb_local(nu, lam) -= v * dPb(mu, sig);
                                    Gb_local(mu, sig) -= v * dPb_nl;
                                    Gb_local(nu, sig) -= v * dPb(mu, lam);
                                }
                            }
                            Ga_local(lam, sig) += j_ls_accum;
                            Gb_local(lam, sig) += j_ls_accum;
                        }
                    }
                }
                #pragma omp critical
                {
                    dGa.noalias() += Ga_local;
                    dGb.noalias() += Gb_local;
                }
            }

            dGa = (dGa + dGa.transpose()).eval();
            dGb = (dGb + dGb.transpose()).eval();

            G_accum_a_ += dGa;
            G_accum_b_ += dGb;
            F_alpha_ = H_ + G_accum_a_;
            F_beta_  = H_ + G_accum_b_;
        }
    } else {
        fock_engine_->compute(P_alpha_, P_beta_, F_alpha_, F_beta_);
    }
    P_alpha_old_ = P_alpha_;
    P_beta_old_  = P_beta_;
}

double ROHF::compute_energy() { return 0.5 * (P_alpha_.cwiseProduct(H_ + F_alpha_).sum() + P_beta_.cwiseProduct(H_ + F_beta_).sum()); }

SCFResult ROHF::compute() {
    if (n_alpha_ == n_beta_) { RHF rhf_engine(mol_, basis_, integrals_, pg_, pl_, config_); return rhf_engine.compute(); }
    init_integrals();
    initial_guess();
    if (config_.scf_type == "direct" && !config_.use_df && config_.eri_method != "cholesky") {
        fock_engine_ = std::make_unique<FockBuilder>(integrals_, basis_, H_, schwarz_);
        if (pl_) fock_engine_->set_petite_list(pl_.get());
    }

    DIIS diis(config_.diis_max_vectors);
    bool converged = false;
    if (config_.print_level > 0) printf("\n Iter        Energy (Ha)        Delta E    Delta P\n");

    for (iter_scf_ = 1; iter_scf_ <= config_.max_iterations; iter_scf_++) {
        energy_old_ = energy_;
        Eigen::MatrixXd P_tot_old = P_alpha_ + P_beta_;
        if (symmetrizer_) { symmetrizer_->symmetrize(P_alpha_); symmetrizer_->symmetrize(P_beta_); }

        build_fock_matrix();
        if (symmetrizer_) { symmetrizer_->symmetrize(F_alpha_); symmetrizer_->symmetrize(F_beta_); }
        energy_ = compute_energy();

        if (C_alpha_.rows() != nbasis_) {
            solve_fock(0.5 * (F_alpha_ + F_beta_), C_alpha_, eps_alpha_);
            C_beta_ = C_alpha_;
            eps_beta_ = eps_alpha_;
        }
        Eigen::MatrixXd Fa_mo = C_alpha_.transpose() * F_alpha_ * C_alpha_;
        Eigen::MatrixXd Fb_mo = C_alpha_.transpose() * F_beta_  * C_alpha_;
        Eigen::MatrixXd F_uni = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
        int n_c = n_beta_;
        int n_o = n_alpha_ - n_beta_;
        int n_v = nbasis_ - n_alpha_;
        double shift_o = 0.5;
        double shift_v = 1.0;

        if (n_c > 0) F_uni.topLeftCorner(n_c, n_c) = 0.5 * (Fa_mo.topLeftCorner(n_c, n_c) + Fb_mo.topLeftCorner(n_c, n_c));
        if (n_o > 0) {
            F_uni.block(n_c, n_c, n_o, n_o) = Fa_mo.block(n_c, n_c, n_o, n_o);
            for (int i = 0; i < n_o; ++i) F_uni(n_c + i, n_c + i) += shift_o;
        }
        if (n_v > 0) {
            F_uni.bottomRightCorner(n_v, n_v) = 0.5 * (Fa_mo.bottomRightCorner(n_v, n_v) + Fb_mo.bottomRightCorner(n_v, n_v));
            for (int i = 0; i < n_v; ++i) F_uni(n_alpha_ + i, n_alpha_ + i) += shift_v;
        }
        if (n_c > 0 && n_o > 0) {
            Eigen::MatrixXd blk = Fb_mo.block(0, n_c, n_c, n_o);
            F_uni.block(0, n_c, n_c, n_o) = blk;
            F_uni.block(n_c, 0, n_o, n_c) = blk.transpose();
        }
        if (n_c > 0 && n_v > 0) {
            Eigen::MatrixXd blk = 0.5 * (Fa_mo.block(0, n_alpha_, n_c, n_v) + Fb_mo.block(0, n_alpha_, n_c, n_v));
            F_uni.block(0, n_alpha_, n_c, n_v) = blk;
            F_uni.block(n_alpha_, 0, n_v, n_c) = blk.transpose();
        }
        if (n_o > 0 && n_v > 0) {
            Eigen::MatrixXd blk = Fa_mo.block(n_c, n_alpha_, n_o, n_v);
            F_uni.block(n_c, n_alpha_, n_o, n_v) = blk;
            F_uni.block(n_alpha_, n_c, n_v, n_o) = blk.transpose();
        }

        Eigen::MatrixXd SC = S_ * C_alpha_;
        Eigen::MatrixXd F_eff_ao = SC * F_uni * SC.transpose();
        Eigen::MatrixXd FPS = F_eff_ao * P_alpha_ * S_;
        diis.add_iteration(F_eff_ao, FPS - FPS.transpose());
        solve_fock((iter_scf_ <= 1) ? 0.7 * F_eff_ao + 0.3 * H_ : diis.extrapolate(), C_alpha_, eps_alpha_);
        C_beta_ = C_alpha_;
        eps_beta_ = eps_alpha_;
        occ_numbers_alpha_ = smear_electrons(eps_alpha_, n_alpha_);
        occ_numbers_beta_  = smear_electrons(eps_beta_, n_beta_);
        update_densities();

        double dE = std::abs(energy_ - energy_old_);
        double dP = ((P_alpha_ + P_beta_) - P_tot_old).norm() / nbasis_;
        if (config_.print_level > 0) printf(" %4d %18.10f %10.2e %10.2e\n", iter_scf_, energy(), dE, dP);
        if (dE < config_.energy_threshold && dP < config_.density_threshold) { converged = true; break; }
    }

    if (config_.use_df && config_.scf_type != "ooc" && L_mat_.rows() == 0 && L_mat_.cols() > 0) {
        if (config_.print_level > 0) std::cout << "\n  [SCF] Reloading df_tensor.h5 to packed RAM (50% footprint)...\n";
        const int n_aux = static_cast<int>(L_mat_.cols());
        const int n_tri = integrals::ao_tri_size(nbasis_);
        L_mat_.resize(n_tri, n_aux);
        utils::HDF5TensorIO io("df_tensor.h5", utils::HDF5TensorIO::Mode::READ_ONLY);
        constexpr int kChunk = 64;
        for (int K0 = 0; K0 < n_aux; K0 += kChunk) {
            const int k_sz = std::min(kChunk, n_aux - K0);
            Eigen::MatrixXd sq_chunk(static_cast<Eigen::Index>(nbasis_) * nbasis_, k_sz);
            io.read_slice_4d("df_tensor", {K0, 0, 0, 0}, {k_sz, (long)nbasis_, (long)nbasis_, 1}, sq_chunk.data());
            #pragma omp parallel for schedule(static)
            for (int c = 0; c < k_sz; ++c) {
                const double* s_col = sq_chunk.col(c).data();
                double* d_col = L_mat_.col(K0 + c).data();
                for (int r = 0; r < nbasis_; ++r) {
                    const int base = r * (r + 1) / 2;
                    for (int q = 0; q <= r; ++q) {
                        d_col[base + q] = s_col[r + q * nbasis_];
                    }
                }
            }
        }
    }

    SCFResult r;
    r.energy_total = energy();
    r.iterations = iter_scf_;
    r.converged = converged;
    r.C_alpha = C_alpha_;
    r.C_beta = C_beta_;
    r.P_alpha = P_alpha_;
    r.P_beta = P_beta_;
    r.F_alpha = F_alpha_;
    r.F_beta = F_beta_;
    r.orbital_energies_alpha = eps_alpha_;
    r.orbital_energies_beta = eps_beta_;
    r.n_occ_alpha = n_alpha_;
    r.n_occ_beta = n_beta_;
    r.L_mat = L_mat_;

    print_final(r);
    return r;
}

} // namespace mshqc