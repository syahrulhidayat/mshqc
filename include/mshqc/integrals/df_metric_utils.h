// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0
// [INJECTED_HPC_V2_DF_METRIC_LLT]
#pragma once

#include "mshqc/utils/blas_thread_guard.h"
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace mshqc {
namespace integrals {

// Faktor metrik DF universal M dengan sifat M * M^T = J^{-1}.
// - Jalur cepat (default): Cholesky J = L * L^T (dpotrf MT), M = L^{-T} (2x lebih hemat FLOPs, 10-30x lebih cepat).
// - Jalur aman (fallback otomatis): Eigendecomposition J^{-1/2} jika J hampir singular (basis difus aug-).
struct DFMetricFactor {
  bool use_cholesky = false;
  int n_aux = 0;
  Eigen::MatrixXd L_chol;      // lower triangular L jika use_cholesky == true
  Eigen::MatrixXd J_inv_sqrt;  // simetris J^{-1/2} jika use_cholesky == false

  static DFMetricFactor compute(const Eigen::MatrixXd& J_metric, double cutoff) {
    DFMetricFactor fac;
    fac.n_aux = static_cast<int>(J_metric.rows());
    if (fac.n_aux <= 0) return fac;

    const double rel_cond = (cutoff > 0.0 && cutoff <= 1e-10) ? cutoff : 1e-12;

    // 1. Coba Cholesky multithread (dpotrf via MKL / Eigen LLT)
    {
      mshqc::utils::ScopedBlasThreads blas_mt(fac.n_aux, 96);
      Eigen::LLT<Eigen::MatrixXd> llt(J_metric);
      if (llt.info() == Eigen::Success) {
        const Eigen::MatrixXd& L_raw = llt.matrixLLT();
        double max_d = 0.0;
        double min_d = 1e300;
        for (int i = 0; i < fac.n_aux; ++i) {
          const double d = L_raw(i, i);
          if (d > max_d) max_d = d;
          if (d < min_d) min_d = d;
        }
        const double max_eval_est = std::max(1.0, max_d * max_d);
        const double diag_tol = std::max(1e-13, rel_cond * max_eval_est);
        if (min_d > 0.0 && (min_d * min_d) >= diag_tol) {
          fac.use_cholesky = true;
          fac.L_chol = L_raw;
          return fac;
        }
      }
    }

    // 2. Fallback otomatis: Eigendecomposition J^{-1/2} untuk basis hampir linear-dependen
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    {
      mshqc::utils::ScopedBlasThreads blas_mt(fac.n_aux, 96);
      es.compute(J_metric);
    }
    const Eigen::VectorXd& evals = es.eigenvalues();
    const Eigen::MatrixXd& evecs = es.eigenvectors();
    const double max_eval = (evals.size() > 0) ? std::max(1.0, evals.maxCoeff()) : 1.0;
    const double eig_tol = std::max(1e-14, rel_cond * max_eval);

    int rank = 0;
    for (int i = 0; i < fac.n_aux; ++i) {
      if (evals(i) >= eig_tol) ++rank;
    }

    Eigen::MatrixXd U_kept(fac.n_aux, rank);
    Eigen::VectorXd inv_sq(rank);
    int c = 0;
    for (int i = 0; i < fac.n_aux; ++i) {
      if (evals(i) >= eig_tol) {
        U_kept.col(c) = evecs.col(i);
        inv_sq(c) = 1.0 / std::sqrt(evals(i));
        ++c;
      }
    }
    Eigen::MatrixXd U_sc = U_kept * inv_sq.asDiagonal();
    fac.J_inv_sqrt.resize(fac.n_aux, fac.n_aux);
    constexpr int tile = 64;
    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < fac.n_aux; r += tile) {
      const int rows = std::min(tile, fac.n_aux - r);
      fac.J_inv_sqrt.middleRows(r, rows).noalias() = U_sc.middleRows(r, rows) * U_kept.transpose();
    }
    fac.use_cholesky = false;
    return fac;
  }

  // Transformasi IN-PLACE: V -> B = V * M (hemat 50% Peak RAM, 2x lebih hemat FLOPs untuk Cholesky)
  void apply_V_to_B_inplace(Eigen::MatrixXd& V) const {
    const int n_rows = static_cast<int>(V.rows());
    if (n_rows <= 0 || n_aux <= 0) return;
    constexpr int tile = 256;

    #pragma omp parallel
    {
      Eigen::MatrixXd buf;
      Eigen::MatrixXd out_buf;
      #pragma omp for schedule(dynamic) nowait
      for (int r = 0; r < n_rows; r += tile) {
        const int len = std::min(tile, n_rows - r);
        buf = V.middleRows(r, len);
        if (use_cholesky) {
          // Selesaikan X * L^T = buf  =>  X = buf * L^{-T}
          L_chol.transpose().triangularView<Eigen::Upper>().solveInPlace<Eigen::OnTheRight>(buf);
          V.middleRows(r, len) = buf;
        } else {
          out_buf.resize(len, n_aux);
          out_buf.noalias() = buf * J_inv_sqrt;
          V.middleRows(r, len) = out_buf;
        }
      }
    }
  }

  // Transformasi IN-PLACE untuk gradien DF: Gamma -> Gamma_tilde = Gamma * M^T
  void apply_Gamma_to_tilde_inplace(Eigen::MatrixXd& Gamma) const {
    const int n_rows = static_cast<int>(Gamma.rows());
    if (n_rows <= 0 || n_aux <= 0) return;
    constexpr int tile = 256;

    #pragma omp parallel
    {
      Eigen::MatrixXd buf;
      Eigen::MatrixXd out_buf;
      #pragma omp for schedule(dynamic) nowait
      for (int r = 0; r < n_rows; r += tile) {
        const int len = std::min(tile, n_rows - r);
        buf = Gamma.middleRows(r, len);
        if (use_cholesky) {
          // Selesaikan X * L = buf  =>  X = buf * L^{-1} = buf * M^T
          L_chol.triangularView<Eigen::Lower>().solveInPlace<Eigen::OnTheRight>(buf);
          Gamma.middleRows(r, len) = buf;
        } else {
          out_buf.resize(len, n_aux);
          out_buf.noalias() = buf * J_inv_sqrt;
          Gamma.middleRows(r, len) = out_buf;
        }
      }
    }
  }

  // Transformasi metrik 2-pusat untuk gradien DF: gamma_tilde = M * gamma_2c * M^T
  Eigen::MatrixXd apply_gamma2c_to_tilde(const Eigen::MatrixXd& gamma_2c) const {
    if (n_aux <= 0) return Eigen::MatrixXd();
    if (use_cholesky) {
      mshqc::utils::ScopedBlasThreads blas_mt(n_aux, 96);
      Eigen::MatrixXd out = gamma_2c;
      L_chol.transpose().triangularView<Eigen::Upper>().solveInPlace<Eigen::OnTheLeft>(out);
      L_chol.triangularView<Eigen::Lower>().solveInPlace<Eigen::OnTheRight>(out);
      return 0.5 * (out + out.transpose());
    }
    Eigen::MatrixXd tmp(n_aux, n_aux), res(n_aux, n_aux);
    constexpr int tile = 64;
    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < n_aux; r += tile) {
      const int rows = std::min(tile, n_aux - r);
      tmp.middleRows(r, rows).noalias() = J_inv_sqrt.middleRows(r, rows) * gamma_2c;
    }
    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < n_aux; r += tile) {
      const int rows = std::min(tile, n_aux - r);
      res.middleRows(r, rows).noalias() = tmp.middleRows(r, rows) * J_inv_sqrt;
    }
    return 0.5 * (res + res.transpose());
  }
};

}  // namespace integrals
}  // namespace mshqc
