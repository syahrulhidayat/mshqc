#!/usr/bin/env python3
"""
inject_hpc_v2.py
----------------
Upgrade arsitektur MSHQC HPC v2 untuk skalabilitas molekul menengah-besar (N_bf >= 400)
dan atom berat (hingga Z = 86 + Frozen Core sejati pada MP2/OMP2/MP3):

  1. [NEW] include/mshqc/utils/blas_thread_guard.h   (Dynamic RAII MKL/OpenBLAS/Eigen MT guard)
  2. [NEW] include/mshqc/integrals/df_metric_utils.h (MT Cholesky LLT + auto Eigen fallback + in-place TRSM)
  3. [MOD] include/mshqc/integrals/df_eri.h          (Zero-copy get_B_mat_mutable())
  4. [MOD] include/mshqc/integrals/packed_ao_utils.h (ScopedBlasThreads untuk Coulomb/kontraksi besar)
  5. [MOD] src/integrals/df_eri.cc                   (LLT MT + in-place V_raw -> B_mat_, hemat 50% RAM)
  6. [MOD] src/scf/scf.cc                            (Zero-copy B_mat move + MT solve_fock/DIIS + UHF warmstart guard)
  7. [MOD] include/mshqc/mp2/mp2.h                   (n_frozen, freeze_core, compute_standard_frozen_core)
  8. [MOD] src/mp2/mp2.cc                            (Pair-batched O(V^2) RMP2/UMP2 + streaming OOM guard + FC OMP2)
  9. [MOD] src/mp/omp2_energy.cc                     (Pair-batched O(V^2) i<=j OMP2 tanpa matriks G_aa/G_bb/G_ab)
 10. [MOD] src/mp/omp_gradient.cc                    (Zero-T2 Z-vector/grad X_a/X_b + FC SOSCF mask + MT K.exp())
 11. [MOD] src/gradient/hpc_geom_opt.cc              (DFMetricFactor LLT + in-place Gamma_tilde + massa Z=1..86)
 12. [MOD] src/mp3/mp3.cc                            (Inherit config_.n_frozen di BaseMP3)
 13. [MOD] python/bindings.cc                        (Tabel unsur Z=1..86 + binding freeze_core/n_frozen/standard_frozen_core)
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str, dry_run: bool, is_new: bool = False) -> None:
    if dry_run:
        tag = "membuat" if is_new else "memperbarui"
        print(f"  [DRY-RUN] Akan {tag} {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not is_new and path.exists():
        bak = path.with_suffix(path.suffix + ".hpcv2.bak")
        if not bak.exists():
            shutil.copy2(path, bak)
    path.write_text(content, encoding="utf-8")
    print(f"  [OK] {'Dibuat' if is_new else 'Diperbarui'}: {path}")


def find_matching_brace(src: str, open_idx: int) -> int:
    depth = 0
    i = open_idx
    n = len(src)
    in_line_comment = False
    in_block_comment = False
    in_str = False
    in_char = False
    while i < n:
        ch = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
        elif in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 1
        elif in_str:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_str = False
        elif in_char:
            if ch == "\\":
                i += 1
            elif ch == "'":
                in_char = False
        else:
            if ch == "/" and nxt == "/":
                in_line_comment = True
                i += 1
            elif ch == "/" and nxt == "*":
                in_block_comment = True
                i += 1
            elif ch == '"':
                in_str = True
            elif ch == "'":
                in_char = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    raise ValueError("Tidak menemukan pasangan '}' yang cocok.")


def replace_function_body(src: str, sig_regex: str, new_func: str) -> str:
    m = re.search(sig_regex, src, flags=re.MULTILINE)
    if not m:
        raise RuntimeError(f"Signature tidak ditemukan: {sig_regex}")
    brace_open = src.find("{", m.end() - 1)
    if brace_open == -1:
        raise RuntimeError(f"Kurung buka tidak ditemukan: {sig_regex}")
    brace_close = find_matching_brace(src, brace_open)
    return src[: m.start()] + new_func + src[brace_close + 1 :]


# =============================================================================
# 1. NEW HEADER: include/mshqc/utils/blas_thread_guard.h
# =============================================================================
BLAS_GUARD_HEADER = r"""// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0
// [INJECTED_HPC_V2_BLAS_GUARD]
#pragma once

#include <Eigen/Core>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace mshqc {
namespace utils {

class ScopedBlasThreads {
 public:
  // Naikkan thread MKL/OpenBLAS/Eigen hanya di luar wilayah paralel OpenMP
  // dan hanya jika ukuran matriks >= threshold agar sistem kecil tanpa overhead.
  explicit ScopedBlasThreads(int problem_dim, int min_dim = 96) : active_(false) {
#ifdef _OPENMP
    if (omp_in_parallel()) return;
    const int target = omp_get_max_threads();
    if (target <= 1 || problem_dim < min_dim) return;
    set_backend_threads(target);
    active_ = true;
#else
    (void)problem_dim;
    (void)min_dim;
#endif
  }

  ~ScopedBlasThreads() {
    if (active_) {
      set_backend_threads(1);
    }
  }

  ScopedBlasThreads(const ScopedBlasThreads&) = delete;
  ScopedBlasThreads& operator=(const ScopedBlasThreads&) = delete;

 private:
  bool active_;

  using SetThreadsFn = void (*)(int);

  struct BackendHandles {
    SetThreadsFn mkl_set = nullptr;
    SetThreadsFn openblas_set = nullptr;

    BackendHandles() {
#if defined(_WIN32)
      const char* mkl_dlls[] = {"mkl_rt.2.dll", "mkl_rt.3.dll", "mkl_rt.dll"};
      for (const char* dll : mkl_dlls) {
        if (HMODULE h = GetModuleHandleA(dll)) {
          mkl_set = reinterpret_cast<SetThreadsFn>(GetProcAddress(h, "MKL_Set_Num_Threads"));
          if (mkl_set) break;
        }
      }
      const char* ob_dlls[] = {"libopenblas.dll", "openblas.dll"};
      for (const char* dll : ob_dlls) {
        if (HMODULE h = GetModuleHandleA(dll)) {
          openblas_set = reinterpret_cast<SetThreadsFn>(GetProcAddress(h, "openblas_set_num_threads"));
          if (openblas_set) break;
        }
      }
#elif defined(__linux__) || defined(__APPLE__)
      mkl_set = reinterpret_cast<SetThreadsFn>(dlsym(RTLD_DEFAULT, "MKL_Set_Num_Threads"));
      openblas_set = reinterpret_cast<SetThreadsFn>(dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
#endif
    }
  };

  static void set_backend_threads(int nt) {
    static const BackendHandles handles;
    if (handles.mkl_set) handles.mkl_set(nt);
    if (handles.openblas_set) handles.openblas_set(nt);
    Eigen::setNbThreads(nt);
  }
};

}  // namespace utils
}  // namespace mshqc
"""


# =============================================================================
# 2. NEW HEADER: include/mshqc/integrals/df_metric_utils.h
# =============================================================================
DF_METRIC_HEADER = r"""// Copyright 2026 Muhamad Syahrul Hidayat
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
"""


# =============================================================================
# 3. MOD: include/mshqc/integrals/df_eri.h
# =============================================================================
def patch_df_eri_h(root: Path, dry_run: bool) -> None:
    p = root / "include/mshqc/integrals/df_eri.h"
    src = read_text(p)
    marker = "get_B_mat_mutable"
    if marker in src:
        print(f"[SKIP] {p} sudah dipatch.")
        return
    src = re.sub(
        r"(const\s+Eigen::MatrixXd&\s+get_B_mat\(\)\s+const\s*\{\s*return\s+B_mat_;\s*\})",
        r"\1\n  Eigen::MatrixXd& get_B_mat_mutable() { return B_mat_; }",
        src,
        count=1,
    )
    write_text(p, src, dry_run)


# =============================================================================
# 4. MOD: include/mshqc/integrals/packed_ao_utils.h
# =============================================================================
def patch_packed_ao_utils(root: Path, dry_run: bool) -> None:
    p = root / "include/mshqc/integrals/packed_ao_utils.h"
    src = read_text(p)
    marker = "// [INJECTED_HPC_V2_PACKED_MT]"
    if marker in src:
        print(f"[SKIP] {p} sudah dipatch.")
        return

    src = src.replace(
        "#include <Eigen/Dense>",
        '#include <Eigen/Dense>\n#include "mshqc/utils/blas_thread_guard.h"\n' + marker,
        1,
    )
    src = re.sub(
        r"(inline\s+Eigen::VectorXd\s+contract_ao_factor_with_sym\s*\([^)]*\)\s*\{)",
        r"\1\n  mshqc::utils::ScopedBlasThreads blas_mt(nbf, 96);",
        src,
        count=1,
    )
    src = re.sub(
        r"(inline\s+Eigen::MatrixXd\s+compute_coulomb_from_ao_factor\s*\([^)]*\)\s*\{)",
        r"\1\n  mshqc::utils::ScopedBlasThreads blas_mt(nbf, 96);",
        src,
        count=1,
    )
    write_text(p, src, dry_run)


# =============================================================================
# 5. MOD: src/integrals/df_eri.cc
# =============================================================================
def patch_df_eri_cc(root: Path, dry_run: bool) -> None:
    p = root / "src/integrals/df_eri.cc"
    src = read_text(p)
    marker = "// [INJECTED_HPC_V2_DF_LLT_INPLACE]"
    if marker in src:
        print(f"[SKIP] {p} sudah dipatch.")
        return

    src = src.replace(
        '#include "mshqc/integrals/packed_ao_utils.h"',
        '#include "mshqc/integrals/packed_ao_utils.h"\n#include "mshqc/integrals/df_metric_utils.h"',
        1,
    )

    # Ganti langkah 2 (Eigen J^{-1/2}) & langkah 4 (B_mat = V_raw * J_inv_sqrt) dengan DFMetricFactor in-place
    step2_pat = re.compile(
        r"//\s*2\.\s*Dekomposisi Eigen J\^\(-1/2\).*?(?=//\s*3\.\s*Evaluasi Integral 3-Pusat)",
        flags=re.MULTILINE | re.DOTALL,
    )
    if not step2_pat.search(src):
        raise RuntimeError(f"Anchor langkah 2 tidak ditemukan di {p}")

    src = step2_pat.sub(
        f"{marker}\n"
        "  // 2. Faktorisasi Metrik 2-Pusat: Cholesky LLT Multithread (fallback Eigen otomatis)\n"
        "  DFMetricFactor df_metric = DFMetricFactor::compute(J_metric, cutoff_);\n"
        "  J_metric.resize(0, 0);\n\n  ",
        src,
        count=1,
    )

    step4_pat = re.compile(
        r"//\s*4\.\s*Kontraksi OpenMP-Tiled.*?(?=if\s*\(save_hdf5\))",
        flags=re.MULTILINE | re.DOTALL,
    )
    if not step4_pat.search(src):
        raise RuntimeError(f"Anchor langkah 4 tidak ditemukan di {p}")

    src = step4_pat.sub(
        "  // 4. Transformasi In-Place V_raw -> B_mat_ (hemat 50% Peak RAM & 2x lebih cepat)\n"
        "  df_metric.apply_V_to_B_inplace(V_raw);\n"
        "  B_mat_ = std::move(V_raw);\n\n  ",
        src,
        count=1,
    )

    write_text(p, src, dry_run)


# =============================================================================
# 6. MOD: src/scf/scf.cc
# =============================================================================
def patch_scf_cc(root: Path, dry_run: bool) -> None:
    p = root / "src/scf/scf.cc"
    src = read_text(p)
    marker = "// [INJECTED_HPC_V2_SCF_MT_AND_MOVE]"
    if marker in src:
        print(f"[SKIP] {p} sudah dipatch.")
        return

    src = src.replace(
        '#include "mshqc/integrals/packed_ao_utils.h"',
        '#include "mshqc/integrals/packed_ao_utils.h"\n#include "mshqc/utils/blas_thread_guard.h"\n' + marker,
        1,
    )

    # Zero-copy move B_mat_
    src = src.replace(
        "this->L_mat_ = std::move(df_engine->get_B_mat());",
        "this->L_mat_ = std::move(df_engine->get_B_mat_mutable());",
    )

    # MT BLAS pada diagonalisasi S_ (tanpa simetri)
    src = re.sub(
        r"(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>\s+es\(S_\);)",
        "mshqc::utils::ScopedBlasThreads blas_s_mt(nbasis_, 96);\n    \\1",
        src,
        count=1,
    )

    # MT BLAS pada solve_fock()
    src = re.sub(
        r"(void\s+BaseSCF::solve_fock\s*\([^)]*\)\s*\{)",
        r"\1\n  mshqc::utils::ScopedBlasThreads blas_fock_mt(nbasis_, 96);",
        src,
        count=1,
    )

    # MT BLAS pada DIIS error commutator (F*P*S)
    src = re.sub(
        r"(Eigen::MatrixXd\s+FPS_a\s*=\s*F_alpha_\s*\*\s*P_alpha_\s*\*\s*S_;)",
        "mshqc::utils::ScopedBlasThreads blas_diis_mt(nbasis_, 96);\n      \\1",
        src,
        count=1,
    )

    # Hindari stability-search berulang pada radikal poliatomik saat warm-start aktif
    src = src.replace(
        "if (is_uhf && mol_.n_atoms() > 2 && n_beta_ > 0) {",
        "if (is_uhf && !config_.has_initial_guess && mol_.n_atoms() > 2 && n_beta_ > 0) {",
        1,
    )

    write_text(p, src, dry_run)


# =============================================================================
# 7. MOD: include/mshqc/mp2/mp2.h
# =============================================================================
def patch_mp2_h(root: Path, dry_run: bool) -> None:
    p = root / "include/mshqc/mp2/mp2.h"
    src = read_text(p)
    marker = "// [INJECTED_HPC_V2_FROZEN_CORE]"
    if marker in src:
        print(f"[SKIP] {p} sudah dipatch.")
        return

    src = re.sub(
        r"(double\s+memory_limit_gb\s*=\s*4\.0;)",
        r"\1\n  " + marker + "\n  int n_frozen = 0;\n  bool freeze_core = false;",
        src,
        count=1,
    )

    helper_fn = r"""
// Jumlah orbital inti standar (konvensi gas mulia, identik dengan Psi4 freeze_core=true)
inline int compute_standard_frozen_core(const Molecule& mol) {
  int nf = 0;
  for (size_t i = 0; i < mol.n_atoms(); ++i) {
    const int Z = mol.atom(i).atomic_number;
    if (Z <= 2) nf += 0;
    else if (Z <= 10) nf += 1;
    else if (Z <= 18) nf += 5;
    else if (Z <= 36) nf += 9;
    else if (Z <= 54) nf += 18;
    else if (Z <= 71) nf += 27;
    else nf += 34;
  }
  return nf;
}

"""
    src = src.replace("struct MP2Config {", helper_fn + "struct MP2Config {", 1)
    write_text(p, src, dry_run)


# =============================================================================
# 8. MOD: src/mp2/mp2.cc
# =============================================================================
def patch_mp2_cc(root: Path, dry_run: bool) -> None:
    p = root / "src/mp2/mp2.cc"
    src = read_text(p)
    marker = "// [INJECTED_HPC_V2_BATCHED_MP2_AND_FC]"
    if marker in src:
        print(f"[SKIP] {p} sudah dipatch.")
        return

    # 8a. Zero-copy B_mat + auto n_frozen_ di BaseMP2
    src = src.replace(
        "scf_.L_mat = df_eng.get_B_mat();",
        "scf_.L_mat = std::move(df_eng.get_B_mat_mutable());",
    )
    src = re.sub(
        r"(nvir_a_\s*=\s*nbf_\s*-\s*nocc_a_;\s*\n[ \t]*nvir_b_\s*=\s*nbf_\s*-\s*nocc_b_;)",
        r"\1\n  " + marker + "\n"
        "  if (config_.n_frozen > 0) {\n"
        "    n_frozen_ = config_.n_frozen;\n"
        "  } else if (config_.freeze_core) {\n"
        "    n_frozen_ = compute_standard_frozen_core(mol_);\n"
        "  }\n"
        "  n_frozen_ = std::clamp(n_frozen_, 0, std::max(0, std::min(nocc_a_, nocc_b_ > 0 ? nocc_b_ : nocc_a_)));",
        src,
        count=1,
    )

    # 8b. Jangan reset n_frozen_ = 0 di konstruktor OMP2
    src = re.sub(
        r"^[ \t]*n_frozen_\s*=\s*0;[ \t]*\r?\n",
        "  // n_frozen_ diwarisi dari BaseMP2 (config_.n_frozen / config_.freeze_core)\n",
        src,
        count=1,
        flags=re.MULTILINE,
    )

    # 8c. Pair-batched O(V^2) i <= j + streaming fallback untuk RMP2
    new_rmp2 = r"""void RMP2::compute_amplitudes_and_energy() {
  const Eigen::VectorXd& eps = scf_.orbital_energies_alpha;
  const bool is_exact = (config_.eri_method == "exact");

  const size_t req_t2_bytes = static_cast<size_t>(nocc_a_) * nocc_a_ * nvir_a_ * nvir_a_ * sizeof(double);
  const size_t t2_budget_bytes = static_cast<size_t>(std::max(0.25, config_.memory_limit_gb * 0.5) * 1e9);
  const bool store_t2 = is_exact || (req_t2_bytes <= t2_budget_bytes);

  if (store_t2) {
    t2_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_a_, nvir_a_, nvir_a_);
    t2_.setZero();
  } else {
    t2_.resize(0, 0, 0, 0);
    if (config_.print_level > 0) {
      std::cout << " [RMP2] Streaming O(V^2) energy mode active (T2 "
                << (req_t2_bytes / (1024.0 * 1024.0)) << " MB > budget).\n";
    }
  }

  double e_corr_sum = 0.0;

  if (is_exact) {
    #pragma omp parallel for collapse(2) reduction(+:e_corr_sum) schedule(static)
    for (int i = n_frozen_; i < nocc_a_; ++i) {
      for (int j = n_frozen_; j < nocc_a_; ++j) {
        const double e_ij = eps(i) + eps(j);
        for (int a = 0; a < nvir_a_; ++a) {
          const double den_a = e_ij - eps(nocc_a_ + a);
          for (int b = 0; b < nvir_a_; ++b) {
            const double denom = den_a - eps(nocc_a_ + b);
            if (std::abs(denom) < 1e-12) continue;
            const double val_iajb = eri_mo_(i, j, a, b);
            const double val_ibja = eri_mo_(i, j, b, a);
            const double t_val = val_iajb / denom;
            t2_(i, j, a, b) = t_val;
            e_corr_sum += t_val * (2.0 * val_iajb - val_ibja);
          }
        }
      }
    }
  } else {
    // Pair-batched DF/CD MP2: O(V^2) RAM per thread & 2x lebih hemat FLOPs (i <= j)
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(static_cast<size_t>(nocc_a_ - n_frozen_) * (nocc_a_ - n_frozen_ + 1) / 2);
    for (int i = n_frozen_; i < nocc_a_; ++i) {
      for (int j = i; j < nocc_a_; ++j) pairs.emplace_back(i, j);
    }

    #pragma omp parallel reduction(+:e_corr_sum)
    {
      Eigen::MatrixXd G_ij(nvir_a_, nvir_a_);
      #pragma omp for schedule(dynamic, 1) nowait
      for (size_t p = 0; p < pairs.size(); ++p) {
        const int i = pairs[p].first;
        const int j = pairs[p].second;
        const double sym_w = (i == j) ? 1.0 : 2.0;
        const double e_ij = eps(i) + eps(j);

        G_ij.noalias() = B_ia_P_alpha_.middleRows(i * nvir_a_, nvir_a_)
                       * B_ia_P_alpha_.middleRows(j * nvir_a_, nvir_a_).transpose();

        for (int a = 0; a < nvir_a_; ++a) {
          const double den_a = e_ij - eps(nocc_a_ + a);
          for (int b = 0; b < nvir_a_; ++b) {
            const double denom = den_a - eps(nocc_a_ + b);
            if (std::abs(denom) < 1e-12) continue;
            const double val_iajb = G_ij(a, b);
            const double val_ibja = G_ij(b, a);
            const double t_val = val_iajb / denom;
            e_corr_sum += sym_w * t_val * (2.0 * val_iajb - val_ibja);
            if (store_t2) {
              t2_(i, j, a, b) = t_val;
              if (i != j) t2_(j, i, b, a) = t_val;
            }
          }
        }
      }
    }
  }
  e_corr_ = e_corr_sum;
}"""
    src = replace_function_body(src, r"void\s+RMP2::compute_amplitudes_and_energy\s*\(\)\s*\{", new_rmp2)

    # 8d. Pair-batched O(V^2) untuk UMP2 (AA, BB, AB)
    new_ump2_aa = r"""double UMP2::compute_ss_alpha() {
  const bool is_exact = (config_.eri_method == "exact");
  if (is_exact) {
    const auto& eri_ao = integrals_->compute_eri();
    const Eigen::MatrixXd& Ca_occ = scf_.C_alpha.leftCols(nocc_a_);
    const Eigen::MatrixXd& Ca_vir = scf_.C_alpha.rightCols(nvir_a_);
    auto eri_chem = integrals::ERITransformer::transform_custom(
        eri_ao, Ca_occ, Ca_vir, Ca_occ, Ca_vir, nbf_, nocc_a_, nvir_a_, nocc_a_, nvir_a_);
    Eigen::array<int, 4> shuf = {0, 2, 1, 3};
    eri_aaaa_ = eri_chem.shuffle(shuf);
  }

  const size_t req_bytes = static_cast<size_t>(nocc_a_) * nocc_a_ * nvir_a_ * nvir_a_ * sizeof(double);
  const size_t budget = static_cast<size_t>(std::max(0.25, config_.memory_limit_gb * 0.35) * 1e9);
  const bool store_t2 = is_exact || (req_bytes <= budget);
  if (store_t2) {
    t2_aa_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_a_, nvir_a_, nvir_a_);
    t2_aa_.setZero();
  } else {
    t2_aa_.resize(0, 0, 0, 0);
  }

  const auto& eo = scf_.orbital_energies_alpha;
  const auto& ev = scf_.orbital_energies_alpha.tail(nvir_a_);
  double e_sum = 0.0;

  if (is_exact) {
    #pragma omp parallel for collapse(3) reduction(+:e_sum) schedule(static)
    for (int b = 0; b < nvir_a_; ++b) {
      for (int a = 0; a < nvir_a_; ++a) {
        for (int j = n_frozen_; j < nocc_a_; ++j) {
          const double den_partial = -ev(a) - ev(b) + eo(j);
          for (int i = n_frozen_; i < nocc_a_; ++i) {
            const double den = den_partial + eo(i);
            if (std::abs(den) < 1e-12) continue;
            const double val_num = eri_aaaa_(i, j, a, b) - eri_aaaa_(i, j, b, a);
            const double safe_den = (config_.level_shift > 1e-12 && std::abs(den) < config_.level_shift)
                ? std::copysign(config_.level_shift, den) : den;
            const double val_t = val_num / safe_den;
            t2_aa_(i, j, a, b) = val_t;
            e_sum += val_t * val_num;
          }
        }
      }
    }
    eri_aaaa_.resize(0, 0, 0, 0);
  } else {
    std::vector<std::pair<int, int>> pairs;
    for (int i = n_frozen_; i < nocc_a_; ++i) {
      for (int j = i + 1; j < nocc_a_; ++j) pairs.emplace_back(i, j);
    }
    #pragma omp parallel reduction(+:e_sum)
    {
      Eigen::MatrixXd G_ij(nvir_a_, nvir_a_);
      #pragma omp for schedule(dynamic, 1) nowait
      for (size_t p = 0; p < pairs.size(); ++p) {
        const int i = pairs[p].first;
        const int j = pairs[p].second;
        const double e_ij = eo(i) + eo(j);
        G_ij.noalias() = B_ia_P_alpha_.middleRows(i * nvir_a_, nvir_a_)
                       * B_ia_P_alpha_.middleRows(j * nvir_a_, nvir_a_).transpose();
        for (int a = 0; a < nvir_a_; ++a) {
          const double den_a = e_ij - ev(a);
          for (int b = 0; b < nvir_a_; ++b) {
            const double den = den_a - ev(b);
            if (std::abs(den) < 1e-12) continue;
            const double val_num = G_ij(a, b) - G_ij(b, a);
            const double safe_den = (config_.level_shift > 1e-12 && std::abs(den) < config_.level_shift)
                ? std::copysign(config_.level_shift, den) : den;
            const double val_t = val_num / safe_den;
            e_sum += 2.0 * val_t * val_num;
            if (store_t2) {
              t2_aa_(i, j, a, b) = val_t;
              t2_aa_(j, i, a, b) = -val_t;
            }
          }
        }
      }
    }
  }
  return 0.25 * e_sum;
}"""
    src = replace_function_body(src, r"double\s+UMP2::compute_ss_alpha\s*\(\)\s*\{", new_ump2_aa)

    new_ump2_bb = r"""double UMP2::compute_ss_beta() {
  if (nocc_b_ == 0 || nvir_b_ == 0) return 0.0;
  const bool is_exact = (config_.eri_method == "exact");
  if (is_exact) {
    const auto& eri_ao = integrals_->compute_eri();
    const Eigen::MatrixXd& Cb_occ = scf_.C_beta.leftCols(nocc_b_);
    const Eigen::MatrixXd& Cb_vir = scf_.C_beta.rightCols(nvir_b_);
    auto eri_chem = integrals::ERITransformer::transform_custom(
        eri_ao, Cb_occ, Cb_vir, Cb_occ, Cb_vir, nbf_, nocc_b_, nvir_b_, nocc_b_, nvir_b_);
    Eigen::array<int, 4> shuf = {0, 2, 1, 3};
    eri_bbbb_ = eri_chem.shuffle(shuf);
  }

  const size_t req_bytes = static_cast<size_t>(nocc_b_) * nocc_b_ * nvir_b_ * nvir_b_ * sizeof(double);
  const size_t budget = static_cast<size_t>(std::max(0.25, config_.memory_limit_gb * 0.35) * 1e9);
  const bool store_t2 = is_exact || (req_bytes <= budget);
  if (store_t2) {
    t2_bb_ = Eigen::Tensor<double, 4>(nocc_b_, nocc_b_, nvir_b_, nvir_b_);
    t2_bb_.setZero();
  } else {
    t2_bb_.resize(0, 0, 0, 0);
  }

  const auto& eo = scf_.orbital_energies_beta;
  const auto& ev = scf_.orbital_energies_beta.tail(nvir_b_);
  double e_sum = 0.0;

  if (is_exact) {
    #pragma omp parallel for collapse(3) reduction(+:e_sum) schedule(static)
    for (int b = 0; b < nvir_b_; ++b) {
      for (int a = 0; a < nvir_b_; ++a) {
        for (int j = n_frozen_; j < nocc_b_; ++j) {
          const double den_partial = -ev(a) - ev(b) + eo(j);
          for (int i = n_frozen_; i < nocc_b_; ++i) {
            const double den = den_partial + eo(i);
            if (std::abs(den) < 1e-12) continue;
            const double val_num = eri_bbbb_(i, j, a, b) - eri_bbbb_(i, j, b, a);
            const double val_t = val_num / den;
            t2_bb_(i, j, a, b) = val_t;
            e_sum += val_t * val_num;
          }
        }
      }
    }
    eri_bbbb_.resize(0, 0, 0, 0);
  } else {
    std::vector<std::pair<int, int>> pairs;
    for (int i = n_frozen_; i < nocc_b_; ++i) {
      for (int j = i + 1; j < nocc_b_; ++j) pairs.emplace_back(i, j);
    }
    #pragma omp parallel reduction(+:e_sum)
    {
      Eigen::MatrixXd G_ij(nvir_b_, nvir_b_);
      #pragma omp for schedule(dynamic, 1) nowait
      for (size_t p = 0; p < pairs.size(); ++p) {
        const int i = pairs[p].first;
        const int j = pairs[p].second;
        const double e_ij = eo(i) + eo(j);
        G_ij.noalias() = B_ia_P_beta_.middleRows(i * nvir_b_, nvir_b_)
                       * B_ia_P_beta_.middleRows(j * nvir_b_, nvir_b_).transpose();
        for (int a = 0; a < nvir_b_; ++a) {
          const double den_a = e_ij - ev(a);
          for (int b = 0; b < nvir_b_; ++b) {
            const double den = den_a - ev(b);
            if (std::abs(den) < 1e-12) continue;
            const double val_num = G_ij(a, b) - G_ij(b, a);
            const double val_t = val_num / den;
            e_sum += 2.0 * val_t * val_num;
            if (store_t2) {
              t2_bb_(i, j, a, b) = val_t;
              t2_bb_(j, i, a, b) = -val_t;
            }
          }
        }
      }
    }
  }
  return 0.25 * e_sum;
}"""
    src = replace_function_body(src, r"double\s+UMP2::compute_ss_beta\s*\(\)\s*\{", new_ump2_bb)

    new_ump2_os = r"""double UMP2::compute_os() {
  if (nocc_a_ == 0 || nvir_a_ == 0 || nocc_b_ == 0 || nvir_b_ == 0) return 0.0;
  const bool is_exact = (config_.eri_method == "exact");
  if (is_exact) {
    const auto& eri_ao = integrals_->compute_eri();
    const Eigen::MatrixXd& Ca_occ = scf_.C_alpha.leftCols(nocc_a_);
    const Eigen::MatrixXd& Ca_vir = scf_.C_alpha.rightCols(nvir_a_);
    const Eigen::MatrixXd& Cb_occ = scf_.C_beta.leftCols(nocc_b_);
    const Eigen::MatrixXd& Cb_vir = scf_.C_beta.rightCols(nvir_b_);
    auto eri_chem = integrals::ERITransformer::transform_custom(
        eri_ao, Ca_occ, Ca_vir, Cb_occ, Cb_vir, nbf_, nocc_a_, nvir_a_, nocc_b_, nvir_b_);
    Eigen::array<int, 4> shuf = {0, 2, 1, 3};
    eri_aabb_ = eri_chem.shuffle(shuf);
  }

  const size_t req_bytes = static_cast<size_t>(nocc_a_) * nocc_b_ * nvir_a_ * nvir_b_ * sizeof(double);
  const size_t budget = static_cast<size_t>(std::max(0.25, config_.memory_limit_gb * 0.35) * 1e9);
  const bool store_t2 = is_exact || (req_bytes <= budget);
  if (store_t2) {
    t2_ab_ = Eigen::Tensor<double, 4>(nocc_a_, nocc_b_, nvir_a_, nvir_b_);
    t2_ab_.setZero();
  } else {
    t2_ab_.resize(0, 0, 0, 0);
  }

  const auto& eoa = scf_.orbital_energies_alpha;
  const auto& eva = scf_.orbital_energies_alpha.tail(nvir_a_);
  const auto& eob = scf_.orbital_energies_beta;
  const auto& evb = scf_.orbital_energies_beta.tail(nvir_b_);
  double e_sum = 0.0;

  if (is_exact) {
    #pragma omp parallel for collapse(3) reduction(+:e_sum) schedule(static)
    for (int b = 0; b < nvir_b_; ++b) {
      for (int a = 0; a < nvir_a_; ++a) {
        for (int j = n_frozen_; j < nocc_b_; ++j) {
          const double den_partial = -eva(a) - evb(b) + eob(j);
          for (int i = n_frozen_; i < nocc_a_; ++i) {
            const double den = den_partial + eoa(i);
            if (std::abs(den) < 1e-12) continue;
            const double val_num = eri_aabb_(i, j, a, b);
            const double val_t = val_num / den;
            t2_ab_(i, j, a, b) = val_t;
            e_sum += val_t * val_num;
          }
        }
      }
    }
    eri_aabb_.resize(0, 0, 0, 0);
  } else {
    std::vector<std::pair<int, int>> pairs;
    for (int i = n_frozen_; i < nocc_a_; ++i) {
      for (int j = n_frozen_; j < nocc_b_; ++j) pairs.emplace_back(i, j);
    }
    #pragma omp parallel reduction(+:e_sum)
    {
      Eigen::MatrixXd G_ij(nvir_a_, nvir_b_);
      #pragma omp for schedule(dynamic, 1) nowait
      for (size_t p = 0; p < pairs.size(); ++p) {
        const int i = pairs[p].first;
        const int j = pairs[p].second;
        const double e_ij = eoa(i) + eob(j);
        G_ij.noalias() = B_ia_P_alpha_.middleRows(i * nvir_a_, nvir_a_)
                       * B_ia_P_beta_.middleRows(j * nvir_b_, nvir_b_).transpose();
        for (int a = 0; a < nvir_a_; ++a) {
          const double den_a = e_ij - eva(a);
          for (int b = 0; b < nvir_b_; ++b) {
            const double den = den_a - evb(b);
            if (std::abs(den) < 1e-12) continue;
            const double val_num = G_ij(a, b);
            const double val_t = val_num / den;
            e_sum += val_t * val_num;
            if (store_t2) t2_ab_(i, j, a, b) = val_t;
          }
        }
      }
    }
  }
  return e_sum;
}"""
    src = replace_function_body(src, r"double\s+UMP2::compute_os\s*\(\)\s*\{", new_ump2_os)

    # 8e. Frozen-core safe OMP2::pseudocanonicalize()
    new_pseudo = r"""void OMP2::pseudocanonicalize() {
  build_fock_fast(scf_.P_alpha, scf_.P_beta, scf_.F_alpha, scf_.F_beta);

  auto diag_block = [&](const Eigen::MatrixXd& F_ao, Eigen::MatrixXd& C, Eigen::VectorXd& eps, int nocc, int nvir) {
    eps.resize(nocc + nvir);
    const int nf = std::clamp(n_frozen_, 0, nocc);
    if (nf > 0 && nf < nocc) {
      Eigen::MatrixXd C_frz = C.leftCols(nf);
      Eigen::MatrixXd F_ff = C_frz.transpose() * (F_ao * C_frz);
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_f(0.5 * (F_ff + F_ff.transpose()));
      C.leftCols(nf).noalias() = C_frz * es_f.eigenvectors();
      eps.head(nf) = es_f.eigenvalues();

      const int nact = nocc - nf;
      Eigen::MatrixXd C_act = C.middleCols(nf, nact);
      Eigen::MatrixXd F_aa = C_act.transpose() * (F_ao * C_act);
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_a(0.5 * (F_aa + F_aa.transpose()));
      C.middleCols(nf, nact).noalias() = C_act * es_a.eigenvectors();
      eps.segment(nf, nact) = es_a.eigenvalues();
    } else if (nocc > 0) {
      Eigen::MatrixXd C_occ = C.leftCols(nocc);
      Eigen::MatrixXd F_oo = C_occ.transpose() * (F_ao * C_occ);
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_o(0.5 * (F_oo + F_oo.transpose()));
      C.leftCols(nocc).noalias() = C_occ * es_o.eigenvectors();
      eps.head(nocc) = es_o.eigenvalues();
    }
    if (nvir > 0) {
      Eigen::MatrixXd C_vir = C.rightCols(nvir);
      Eigen::MatrixXd F_vv = C_vir.transpose() * (F_ao * C_vir);
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_v(0.5 * (F_vv + F_vv.transpose()));
      C.rightCols(nvir).noalias() = C_vir * es_v.eigenvectors();
      eps.tail(nvir) = es_v.eigenvalues();
    }
  };

  diag_block(scf_.F_alpha, scf_.C_alpha, scf_.orbital_energies_alpha, na_, va_);
  scf_.P_alpha.noalias() = scf_.C_alpha.leftCols(na_) * scf_.C_alpha.leftCols(na_).transpose();

  if (nb_ > 0 && vb_ > 0) {
    diag_block(scf_.F_beta, scf_.C_beta, scf_.orbital_energies_beta, nb_, vb_);
    scf_.P_beta.noalias() = scf_.C_beta.leftCols(nb_) * scf_.C_beta.leftCols(nb_).transpose();
  }
}"""
    src = replace_function_body(src, r"void\s+OMP2::pseudocanonicalize\s*\(\)\s*\{", new_pseudo)

    # 8f. Frozen-core gradient mask di OMP2::execute_macro_iterations
    src = src.replace(
        "if (use_sym && (scf_.irreps_alpha[i] ^ scf_.irreps_alpha[na_ + a]) != 0) orbital_gradient_(idx++) = 0.0;",
        "if (i < n_frozen_ || (use_sym && (scf_.irreps_alpha[i] ^ scf_.irreps_alpha[na_ + a]) != 0)) orbital_gradient_(idx++) = 0.0;",
    )
    src = src.replace(
        "if (use_sym && (scf_.irreps_beta[i] ^ scf_.irreps_beta[nb_ + b]) != 0) orbital_gradient_(idx++) = 0.0;",
        "if (i < n_frozen_ || (use_sym && (scf_.irreps_beta[i] ^ scf_.irreps_beta[nb_ + b]) != 0)) orbital_gradient_(idx++) = 0.0;",
    )

    write_text(p, src, dry_run)


# =============================================================================
# 9. MOD: src/mp/omp2_energy.cc (Pair-batched O(V^2) tanpa matriks G_aa/G_bb/G_ab)
# =============================================================================
def patch_omp2_energy(root: Path, dry_run: bool) -> None:
    p = root / "src/mp/omp2_energy.cc"
    src = read_text(p)
    marker = "// [INJECTED_HPC_V2_BATCHED_OMP2_ENERGY]"
    if marker in src:
        print(f"[SKIP] {p} sudah dipatch.")
        return

    new_fn = r"""// [INJECTED_HPC_V2_BATCHED_OMP2_ENERGY]
void OMP2::compute_t2_and_energy_cholesky() {
  double E_ss_aa = 0.0, E_ss_bb = 0.0, E_os = 0.0;
  const int nf = std::clamp(n_frozen_, 0, na_);
  const bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);

  t2_aa_.allocate_block(0, 0, 0, 0, na_, va_, na_, va_);
  auto* t_aa_blk = t2_aa_.get_block(0, 0, 0, 0);

  if (t_aa_blk) {
    t_aa_blk->setZero();
    const double* eps_a = scf_.orbital_energies_alpha.data();

    std::vector<std::pair<int, int>> pairs_aa;
    for (int i = nf; i < na_; ++i) {
      const int j_start = is_restricted ? i : (i + 1);
      for (int j = j_start; j < na_; ++j) pairs_aa.emplace_back(i, j);
    }

    #pragma omp parallel reduction(+:E_ss_aa)
    {
      Eigen::MatrixXd G_ij(va_, va_);
      #pragma omp for schedule(dynamic, 1) nowait
      for (size_t p = 0; p < pairs_aa.size(); ++p) {
        const int i = pairs_aa[p].first;
        const int j = pairs_aa[p].second;
        const double e_ij = eps_a[i] + eps_a[j];

        G_ij.noalias() = B_ia_P_alpha_.middleRows(i * va_, va_)
                       * B_ia_P_alpha_.middleRows(j * va_, va_).transpose();

        if (is_restricted) {
          const double w = (i == j) ? 1.0 : 2.0;
          for (int a = 0; a < va_; ++a) {
            const double den_a = e_ij - eps_a[na_ + a];
            for (int b = 0; b < va_; ++b) {
              const double den = den_a - eps_a[na_ + b];
              if (std::abs(den) <= 1e-12) continue;
              const double val_dir = G_ij(a, b);
              const double val_ex = G_ij(b, a);
              const double t_val = val_dir / den;
              E_ss_aa += w * t_val * (2.0 * val_dir - val_ex);
              (*t_aa_blk)(i, a, j, b) = t_val;
              if (i != j) (*t_aa_blk)(j, b, i, a) = t_val;
            }
          }
        } else {
          for (int a = 0; a < va_; ++a) {
            const double den_a = e_ij - eps_a[na_ + a];
            for (int b = 0; b < va_; ++b) {
              const double den = den_a - eps_a[na_ + b];
              if (std::abs(den) <= 1e-12) continue;
              const double diff = G_ij(a, b) - G_ij(b, a);
              const double t_val = diff / den;
              E_ss_aa += 2.0 * t_val * diff;
              (*t_aa_blk)(i, a, j, b) = t_val;
              (*t_aa_blk)(j, a, i, b) = -t_val;
            }
          }
        }
      }
    }
  }

  if (!is_restricted && nb_ > 0 && vb_ > 0) {
    const int nf_b = std::clamp(n_frozen_, 0, nb_);
    t2_bb_.allocate_block(0, 0, 0, 0, nb_, nb_, vb_, vb_);
    auto* t_bb_blk = t2_bb_.get_block(0, 0, 0, 0);
    if (t_bb_blk) t_bb_blk->setZero();

    t2_ab_.allocate_block(0, 0, 0, 0, na_, nb_, va_, vb_);
    auto* t_ab_blk = t2_ab_.get_block(0, 0, 0, 0);
    if (t_ab_blk) t_ab_blk->setZero();

    const double* eps_b = scf_.orbital_energies_beta.data();

    if (t_bb_blk) {
      std::vector<std::pair<int, int>> pairs_bb;
      for (int i = nf_b; i < nb_; ++i) {
        for (int j = i + 1; j < nb_; ++j) pairs_bb.emplace_back(i, j);
      }
      #pragma omp parallel reduction(+:E_ss_bb)
      {
        Eigen::MatrixXd G_ij(vb_, vb_);
        #pragma omp for schedule(dynamic, 1) nowait
        for (size_t p = 0; p < pairs_bb.size(); ++p) {
          const int i = pairs_bb[p].first;
          const int j = pairs_bb[p].second;
          const double e_ij = eps_b[i] + eps_b[j];
          G_ij.noalias() = B_ia_P_beta_.middleRows(i * vb_, vb_)
                         * B_ia_P_beta_.middleRows(j * vb_, vb_).transpose();
          for (int a = 0; a < vb_; ++a) {
            const double den_a = e_ij - eps_b[nb_ + a];
            for (int b = 0; b < vb_; ++b) {
              const double den = den_a - eps_b[nb_ + b];
              if (std::abs(den) <= 1e-12) continue;
              const double diff = G_ij(a, b) - G_ij(b, a);
              const double t_val = diff / den;
              E_ss_bb += 2.0 * t_val * diff;
              (*t_bb_blk)(i, j, a, b) = t_val;
              (*t_bb_blk)(j, i, a, b) = -t_val;
            }
          }
        }
      }
    }

    if (t_ab_blk) {
      const double* eps_a = scf_.orbital_energies_alpha.data();
      std::vector<std::pair<int, int>> pairs_ab;
      for (int i = nf; i < na_; ++i) {
        for (int j = nf_b; j < nb_; ++j) pairs_ab.emplace_back(i, j);
      }
      #pragma omp parallel reduction(+:E_os)
      {
        Eigen::MatrixXd G_ij(va_, vb_);
        #pragma omp for schedule(dynamic, 1) nowait
        for (size_t p = 0; p < pairs_ab.size(); ++p) {
          const int i = pairs_ab[p].first;
          const int j = pairs_ab[p].second;
          const double e_ij = eps_a[i] + eps_b[j];
          G_ij.noalias() = B_ia_P_alpha_.middleRows(i * va_, va_)
                         * B_ia_P_beta_.middleRows(j * vb_, vb_).transpose();
          for (int a = 0; a < va_; ++a) {
            const double den_a = e_ij - eps_a[na_ + a];
            for (int b = 0; b < vb_; ++b) {
              const double den = den_a - eps_b[nb_ + b];
              if (std::abs(den) <= 1e-12) continue;
              const double val_dir = G_ij(a, b);
              const double t_val = val_dir / den;
              E_os += t_val * val_dir;
              (*t_ab_blk)(i, j, a, b) = t_val;
            }
          }
        }
      }
    }
  }

  if (is_restricted) {
    e_ss_ = 0.0;
    e_os_ = E_ss_aa;
  } else {
    e_ss_ = 0.25 * E_ss_aa + 0.25 * E_ss_bb;
    e_os_ = E_os;
  }
}"""
    src = replace_function_body(src, r"void\s+OMP2::compute_t2_and_energy_cholesky\s*\(\)\s*\{", new_fn)
    write_text(p, src, dry_run)


# =============================================================================
# 10. MOD: src/mp/omp_gradient.cc (Zero-T2 Z-vector & grad + FC mask + MT K.exp)
# =============================================================================
def patch_omp_gradient(root: Path, dry_run: bool) -> None:
    p = root / "src/mp/omp_gradient.cc"
    src = read_text(p)
    marker = "// [INJECTED_HPC_V2_ZERO_T2_COPY]"
    if marker in src:
        print(f"[SKIP] {p} sudah dipatch.")
        return

    src = src.replace(
        '#include "mshqc/integrals/packed_ao_utils.h"',
        '#include "mshqc/integrals/packed_ao_utils.h"\n#include "mshqc/utils/blas_thread_guard.h"\n' + marker,
        1,
    )

    # 10a. Ganti evaluate_z_vector_cholesky agar bebas alokasi O^2 V^2 T2_aa/T2_bb/T2_ab
    new_zvec = r"""void OMP2::evaluate_z_vector_cholesky(Eigen::MatrixXd& Z_mat_a, Eigen::MatrixXd& Z_mat_b) {
  const int n_chol = static_cast<int>(scf_.L_mat.cols());
  Z_mat_a.setZero(va_, na_);
  const bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
  const bool has_beta = (!is_restricted && nb_ > 0 && vb_ > 0);
  if (has_beta) Z_mat_b.setZero(vb_, nb_);

  auto* t_aa_blk = t2_aa_.get_block(0, 0, 0, 0);
  auto* t_ab_blk = has_beta ? t2_ab_.get_block(0, 0, 0, 0) : nullptr;
  auto* t_bb_blk = has_beta ? t2_bb_.get_block(0, 0, 0, 0) : nullptr;

  const int nf_a = std::clamp(n_frozen_, 0, na_);
  const int nf_b = std::clamp(n_frozen_, 0, nb_);

  // Zero-T2-copy contraction: X_a (na*va x n_chol) & X_b (nb*vb x n_chol)
  Eigen::MatrixXd X_a = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(na_) * va_, n_chol);
  Eigen::MatrixXd X_b;
  if (has_beta) X_b = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(nb_) * vb_, n_chol);

  if (t_aa_blk || (has_beta && t_ab_blk)) {
    #pragma omp parallel
    {
      Eigen::MatrixXd T_aa_ij(va_, va_);
      Eigen::MatrixXd T_ab_ij;
      if (has_beta && t_ab_blk) T_ab_ij.resize(va_, vb_);

      #pragma omp for schedule(dynamic, 1) nowait
      for (int i = nf_a; i < na_; ++i) {
        auto X_ai = X_a.middleRows(static_cast<Eigen::Index>(i) * va_, va_);
        if (t_aa_blk) {
          for (int j = nf_a; j < na_; ++j) {
            if (is_restricted) {
              for (int a = 0; a < va_; ++a)
                for (int b = 0; b < va_; ++b)
                  T_aa_ij(a, b) = 2.0 * (*t_aa_blk)(i, a, j, b) - (*t_aa_blk)(i, b, j, a);
            } else {
              for (int a = 0; a < va_; ++a)
                for (int b = 0; b < va_; ++b)
                  T_aa_ij(a, b) = (*t_aa_blk)(i, a, j, b);
            }
            X_ai.noalias() += T_aa_ij * B_ia_P_alpha_.middleRows(static_cast<Eigen::Index>(j) * va_, va_);
          }
        }
        if (has_beta && t_ab_blk) {
          for (int jb = nf_b; jb < nb_; ++jb) {
            for (int a = 0; a < va_; ++a)
              for (int b = 0; b < vb_; ++b)
                T_ab_ij(a, b) = (*t_ab_blk)(i, jb, a, b);
            X_ai.noalias() += T_ab_ij * B_ia_P_beta_.middleRows(static_cast<Eigen::Index>(jb) * vb_, vb_);
          }
        }
      }
    }
  }

  if (has_beta && (t_bb_blk || t_ab_blk)) {
    #pragma omp parallel
    {
      Eigen::MatrixXd T_bb_ij(vb_, vb_);
      Eigen::MatrixXd T_ba_ij(vb_, va_);

      #pragma omp for schedule(dynamic, 1) nowait
      for (int ib = nf_b; ib < nb_; ++ib) {
        auto X_bi = X_b.middleRows(static_cast<Eigen::Index>(ib) * vb_, vb_);
        if (t_bb_blk) {
          for (int jb = nf_b; jb < nb_; ++jb) {
            for (int a = 0; a < vb_; ++a)
              for (int b = 0; b < vb_; ++b)
                T_bb_ij(a, b) = (*t_bb_blk)(ib, jb, a, b);
            X_bi.noalias() += T_bb_ij * B_ia_P_beta_.middleRows(static_cast<Eigen::Index>(jb) * vb_, vb_);
          }
        }
        if (t_ab_blk) {
          for (int ia = nf_a; ia < na_; ++ia) {
            for (int a = 0; a < va_; ++a)
              for (int b = 0; b < vb_; ++b)
                T_ba_ij(b, a) = (*t_ab_blk)(ia, ib, a, b);
            X_bi.noalias() += T_ba_ij * B_ia_P_alpha_.middleRows(static_cast<Eigen::Index>(ia) * va_, va_);
          }
        }
      }
    }
  }

  #pragma omp parallel
  {
    Eigen::MatrixXd Z_loc_a = Eigen::MatrixXd::Zero(va_, na_);
    Eigen::MatrixXd Z_loc_b;
    if (has_beta) Z_loc_b = Eigen::MatrixXd::Zero(vb_, nb_);

    #pragma omp for schedule(static)
    for (int P = 0; P < n_chol; ++P) {
      Eigen::Map<const Eigen::MatrixXd> B_oo_a(B_oo_P_alpha_.col(P).data(), na_, na_);
      Eigen::Map<const Eigen::MatrixXd> B_vv_a(B_vv_P_alpha_.col(P).data(), va_, va_);
      Eigen::Map<const Eigen::MatrixXd> X_ai(X_a.col(P).data(), va_, na_);
      Z_loc_a.noalias() += B_vv_a * X_ai - X_ai * B_oo_a;

      if (has_beta) {
        Eigen::Map<const Eigen::MatrixXd> B_oo_b(B_oo_P_beta_.col(P).data(), nb_, nb_);
        Eigen::Map<const Eigen::MatrixXd> B_vv_b(B_vv_P_beta_.col(P).data(), vb_, vb_);
        Eigen::Map<const Eigen::MatrixXd> X_bi(X_b.col(P).data(), vb_, nb_);
        Z_loc_b.noalias() += B_vv_b * X_bi - X_bi * B_oo_b;
      }
    }

    #pragma omp critical
    {
      Z_mat_a += Z_loc_a;
      if (has_beta) Z_mat_b += Z_loc_b;
    }
  }
}"""
    src = replace_function_body(src, r"void\s+OMP2::evaluate_z_vector_cholesky\s*\([^)]*\)\s*\{", new_zvec)

    # 10b. Masker frozen-core di compute_soscf_step (tepat sebelum return Hp;)
    fc_mask_code = (
        "    if (n_frozen_ > 0) {\n"
        "      const int nf_a = std::min(n_frozen_, na_);\n"
        "      for (int i = 0; i < nf_a; ++i)\n"
        "        for (int a = 0; a < va_; ++a)\n"
        "          Hp(i * va_ + a) = hessian_diag_(i * va_ + a) * p_vec(i * va_ + a);\n"
        "      if (!is_restricted && dim_b > 0) {\n"
        "        const int nf_b = std::min(n_frozen_, nb_);\n"
        "        for (int i = 0; i < nf_b; ++i)\n"
        "          for (int b = 0; b < vb_; ++b)\n"
        "            Hp(dim_a + i * vb_ + b) = hessian_diag_(dim_a + i * vb_ + b) * p_vec(dim_a + i * vb_ + b);\n"
        "      }\n"
        "    }\n"
        "    return Hp;"
    )
    src = re.sub(r"^[ \t]*return\s+Hp;", fc_mask_code, src, count=1, flags=re.MULTILINE)

    # 10c. ScopedBlasThreads pada K_a.exp() di apply_orbital_rotation
    src = src.replace(
        "C_a_current_ = C_a_current_ * K_a.exp();",
        "{\n    mshqc::utils::ScopedBlasThreads blas_exp_mt(n_mo_a, 96);\n    C_a_current_ = C_a_current_ * K_a.exp();\n  }",
        1,
    )
    src = src.replace(
        "C_b_current_ = C_b_current_ * K_b.exp();",
        "{\n      mshqc::utils::ScopedBlasThreads blas_exp_mt(n_mo_b, 96);\n      C_b_current_ = C_b_current_ * K_b.exp();\n    }",
        1,
    )

    # 10d. Zero-T2-copy X_a / X_b pada build_df_gradient_data()
    t2_grad_pat = re.compile(
        r"Eigen::MatrixXd\s+T2_aa\s*=\s*Eigen::MatrixXd::Zero\(na_\s*\*\s*va_,\s*na_\s*\*\s*va_\);.*?"
        r"parallel_gemm_ab\(T2_ab\.transpose\(\),\s*B_ia_P_alpha_,\s*X_b,\s*true\);\s*\n[ \t]*\}",
        flags=re.MULTILINE | re.DOTALL,
    )
    if not t2_grad_pat.search(src):
        raise RuntimeError(f"Blok T2_aa di build_df_gradient_data tidak ditemukan di {p}")

    zero_t2_grad = r"""const int nf_a = std::clamp(n_frozen_, 0, na_);
  const int nf_b = std::clamp(n_frozen_, 0, nb_);
  Eigen::MatrixXd X_a = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(na_) * va_, n_aux);
  Eigen::MatrixXd X_b;
  if (has_beta) X_b = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(nb_) * vb_, n_aux);

  if (t_aa_blk || (has_beta && t_ab_blk)) {
    #pragma omp parallel
    {
      Eigen::MatrixXd T_aa_ij(va_, va_);
      Eigen::MatrixXd T_ab_ij;
      if (has_beta && t_ab_blk) T_ab_ij.resize(va_, vb_);
      #pragma omp for schedule(dynamic, 1) nowait
      for (int i = nf_a; i < na_; ++i) {
        auto X_ai = X_a.middleRows(static_cast<Eigen::Index>(i) * va_, va_);
        if (t_aa_blk) {
          for (int j = nf_a; j < na_; ++j) {
            for (int a = 0; a < va_; ++a)
              for (int b = 0; b < va_; ++b)
                T_aa_ij(a, b) = is_restricted
                    ? (2.0 * (*t_aa_blk)(i, a, j, b) - (*t_aa_blk)(i, b, j, a))
                    : (*t_aa_blk)(i, a, j, b);
            X_ai.noalias() += T_aa_ij * B_ia_P_alpha_.middleRows(static_cast<Eigen::Index>(j) * va_, va_);
          }
        }
        if (has_beta && t_ab_blk) {
          for (int jb = nf_b; jb < nb_; ++jb) {
            for (int a = 0; a < va_; ++a)
              for (int b = 0; b < vb_; ++b)
                T_ab_ij(a, b) = (*t_ab_blk)(i, jb, a, b);
            X_ai.noalias() += T_ab_ij * B_ia_P_beta_.middleRows(static_cast<Eigen::Index>(jb) * vb_, vb_);
          }
        }
      }
    }
  }
  if (has_beta && (t_bb_blk || t_ab_blk)) {
    #pragma omp parallel
    {
      Eigen::MatrixXd T_bb_ij(vb_, vb_), T_ba_ij(vb_, va_);
      #pragma omp for schedule(dynamic, 1) nowait
      for (int ib = nf_b; ib < nb_; ++ib) {
        auto X_bi = X_b.middleRows(static_cast<Eigen::Index>(ib) * vb_, vb_);
        if (t_bb_blk) {
          for (int jb = nf_b; jb < nb_; ++jb) {
            for (int a = 0; a < vb_; ++a)
              for (int b = 0; b < vb_; ++b)
                T_bb_ij(a, b) = (*t_bb_blk)(ib, jb, a, b);
            X_bi.noalias() += T_bb_ij * B_ia_P_beta_.middleRows(static_cast<Eigen::Index>(jb) * vb_, vb_);
          }
        }
        if (t_ab_blk) {
          for (int ia = nf_a; ia < na_; ++ia) {
            for (int a = 0; a < va_; ++a)
              for (int b = 0; b < vb_; ++b)
                T_ba_ij(b, a) = (*t_ab_blk)(ia, ib, a, b);
            X_bi.noalias() += T_ba_ij * B_ia_P_alpha_.middleRows(static_cast<Eigen::Index>(ia) * va_, va_);
          }
        }
      }
    }
  }"""
    src = t2_grad_pat.sub(zero_t2_grad, src, count=1)

    write_text(p, src, dry_run)


# =============================================================================
# 11. MOD: src/gradient/hpc_geom_opt.cc (DFMetricFactor LLT + in-place Gamma + Z<=86)
# =============================================================================
def patch_hpc_geom_opt(root: Path, dry_run: bool) -> None:
    p = root / "src/gradient/hpc_geom_opt.cc"
    src = read_text(p)
    marker = "// [INJECTED_HPC_V2_DF_GRAD_LLT_AND_HEAVY_MASS]"
    if marker in src:
        print(f"[SKIP] {p} sudah dipatch.")
        return

    src = src.replace(
        '#include "mshqc/integrals/packed_ao_utils.h"',
        '#include "mshqc/integrals/packed_ao_utils.h"\n#include "mshqc/integrals/df_metric_utils.h"\n' + marker,
        1,
    )

    # 11a. Perluas tabel massa isotopik hingga Z=86 (logam transisi & unsur berat)
    new_mass_fn = r"""double get_exact_isotopic_mass(int Z) {
  static const std::map<int, double> m = {
    {1, 1.007825032}, {2, 4.002603254}, {3, 7.016003437}, {4, 9.01218307},
    {5, 11.00930536}, {6, 12.00000000}, {7, 14.003074004}, {8, 15.994914619},
    {9, 18.998403163}, {10, 19.99244018}, {11, 22.98976928}, {12, 23.98504170},
    {13, 26.98153853}, {14, 27.976926535}, {15, 30.973761998}, {16, 31.972071174},
    {17, 34.96885268}, {18, 39.96238312}, {19, 38.96370649}, {20, 39.96259086},
    {21, 44.95590828}, {22, 47.94794198}, {23, 50.94395704}, {24, 51.94050623},
    {25, 54.93804391}, {26, 55.93493633}, {27, 58.93319429}, {28, 57.93534241},
    {29, 62.92959772}, {30, 63.92914201}, {31, 68.9255735}, {32, 73.92117776},
    {33, 74.92159457}, {34, 79.9165218}, {35, 78.9183376}, {36, 83.91149773},
    {37, 84.91178974}, {38, 87.9056125}, {39, 88.9058403}, {40, 89.9046977},
    {41, 92.9063730}, {42, 97.9054048}, {43, 97.9072124}, {44, 101.9043441},
    {45, 102.9054980}, {46, 105.9034804}, {47, 106.9050916}, {48, 113.9033650},
    {49, 114.9038788}, {50, 119.9022016}, {51, 120.9038120}, {52, 129.9062227},
    {53, 126.9044719}, {54, 131.9041551}, {55, 132.9054519}, {56, 137.9052470},
    {72, 179.9465570}, {73, 180.9479958}, {74, 183.9509309}, {75, 186.9557501},
    {76, 191.9614770}, {77, 192.9629216}, {78, 194.9647917}, {79, 196.9665687},
    {80, 201.9706434}, {81, 204.9744278}, {82, 207.9766525}, {83, 208.9803991},
    {86, 222.0175782}
  };
  auto it = m.find(Z);
  return (it != m.end()) ? it->second : static_cast<double>(2.4 * Z);
}"""
    src = replace_function_body(src, r"double\s+get_exact_isotopic_mass\s*\(int\s+Z\)\s*\{", new_mass_fn)

    # 11b. Ganti compute_df_metric_inv_sqrt menjadi DFMetricFactor (sinkron 100% dengan df_eri.cc)
    new_metric_fn = r"""integrals::DFMetricFactor compute_df_metric_factor(
    const BasisSet& primary_basis,
    const BasisSet& aux_basis,
    const std::shared_ptr<IntegralEngine>& df_ints,
    double cutoff) {

  const int n_shells_prim = primary_basis.n_shells();
  const int n_shells_aux = aux_basis.n_shells();
  const int n_aux = aux_basis.n_basis_functions();

  std::vector<int> aux_starts(n_shells_aux), aux_sizes(n_shells_aux);
  int off = 0, max_a = 1;
  for (int i = 0; i < n_shells_aux; ++i) {
    aux_starts[i] = off;
    aux_sizes[i] = aux_basis.shell(i).n_functions();
    max_a = std::max(max_a, aux_sizes[i]);
    off += aux_sizes[i];
  }

  Eigen::MatrixXd J_metric = Eigen::MatrixXd::Zero(n_aux, n_aux);
  const size_t buf_sz = std::max<size_t>(64, static_cast<size_t>(max_a) * max_a);
  #pragma omp parallel
  {
    std::vector<double> buf(buf_sz);
    #pragma omp for schedule(dynamic, 1) nowait
    for (int R = 0; R < n_shells_aux; ++R) {
      const int abs_R = n_shells_prim + R, dim_R = aux_sizes[R], st_R = aux_starts[R];
      for (int S = 0; S <= R; ++S) {
        const int abs_S = n_shells_prim + S, dim_S = aux_sizes[S], st_S = aux_starts[S];
        if (!df_ints->compute_2c2e_block_raw(abs_R, abs_S, buf.data())) continue;
        for (int s = 0; s < dim_S; ++s) {
          for (int r = 0; r < dim_R; ++r) {
            const double val = buf[r + dim_R * s];
            J_metric(st_R + r, st_S + s) = val;
            if (st_R + r != st_S + s) J_metric(st_S + s, st_R + r) = val;
          }
        }
      }
    }
  }
  return integrals::DFMetricFactor::compute(J_metric, cutoff);
}"""
    src = replace_function_body(src, r"Eigen::MatrixXd\s+compute_df_metric_inv_sqrt\s*\([^)]*\)\s*\{", new_metric_fn)

    # 11c. Ganti perkalian Gamma_tilde & gamma_tilde menjadi in-place DFMetricFactor (hemat 50% RAM)
    grad_metric_pat = re.compile(
        r"Eigen::MatrixXd\s+J_inv_sqrt\s*=\s*compute_df_metric_inv_sqrt\(.*?"
        r"Gamma_tilde,\s*gamma_tilde\);",
        flags=re.MULTILINE | re.DOTALL,
    )
    if not grad_metric_pat.search(src):
        raise RuntimeError(f"Blok Gamma_tilde/gamma_tilde tidak ditemukan di {p}")

    new_grad_metric = r"""integrals::DFMetricFactor df_metric = compute_df_metric_factor(basis, aux_basis, df_ints, scf_conf.df_threshold);
  df_metric.apply_Gamma_to_tilde_inplace(Gamma_3c);
  Eigen::MatrixXd gamma_tilde = df_metric.apply_gamma2c_to_tilde(gamma_2c);

  Eigen::MatrixXd g_tot = compute_nuclear_repulsion_gradient(mol)
      + integrals->compute_1e_gradient(P_eff_ao, W_eff_ao)
      + df_ints->compute_df_2e_gradient(
          basis.n_shells(), aux_basis.n_shells(),
          nbf, aux_basis.n_basis_functions(),
          Gamma_3c, gamma_tilde);"""
    src = grad_metric_pat.sub(new_grad_metric, src, count=1)

    write_text(p, src, dry_run)


# =============================================================================
# 12. MOD: src/mp3/mp3.cc (Warisi config_.n_frozen di BaseMP3)
# =============================================================================
def patch_mp3_cc(root: Path, dry_run: bool) -> None:
    p = root / "src/mp3/mp3.cc"
    src = read_text(p)
    marker = "// [INJECTED_HPC_V2_MP3_CONFIG_FROZEN]"
    if marker in src:
        print(f"[SKIP] {p} sudah dipatch.")
        return

    src = re.sub(
        r"(n_aux_\s*=\s*scf_\.L_mat\.cols\(\);)",
        r"\1\n  " + marker + "\n  if (config_.n_frozen > 0) n_frozen_ = config_.n_frozen;",
        src,
        count=1,
    )
    write_text(p, src, dry_run)


# =============================================================================
# 13. MOD: python/bindings.cc (Tabel unsur Z=1..86 + binding freeze_core)
# =============================================================================
def patch_bindings_cc(root: Path, dry_run: bool) -> None:
    p = root / "python/bindings.cc"
    src = read_text(p)
    marker = "// [INJECTED_HPC_V2_HEAVY_ELEMENTS_AND_FC]"
    if marker in src:
        print(f"[SKIP] {p} sudah dipatch.")
        return

    # 13a. Perluas symbol_to_z hingga Z=86 (Rn)
    src = src.replace(
        '{"Sb",51},{"Te",52},{"I",53},{"Xe",54}',
        '{"Sb",51},{"Te",52},{"I",53},{"Xe",54},\n'
        '      {"Cs",55},{"Ba",56},{"La",57},{"Ce",58},{"Pr",59},{"Nd",60},{"Pm",61},{"Sm",62},{"Eu",63},{"Gd",64},\n'
        '      {"Tb",65},{"Dy",66},{"Ho",67},{"Er",68},{"Tm",69},{"Yb",70},{"Lu",71},\n'
        '      {"Hf",72},{"Ta",73},{"W",74},{"Re",75},{"Os",76},{"Ir",77},{"Pt",78},{"Au",79},{"Hg",80},\n'
        '      {"Tl",81},{"Pb",82},{"Bi",83},{"Po",84},{"At",85},{"Rn",86}',
        1,
    )

    # 13b. Ekspos n_frozen, freeze_core, memory_limit_gb pada MP2Config
    src = src.replace(
        '.def_rw("opt_method", &MP2Config::opt_method);',
        '.def_rw("opt_method", &MP2Config::opt_method)\n'
        f'    {marker}\n'
        '    .def_rw("n_frozen", &MP2Config::n_frozen)\n'
        '    .def_rw("freeze_core", &MP2Config::freeze_core)\n'
        '    .def_rw("memory_limit_gb", &MP2Config::memory_limit_gb);',
        1,
    )

    # 13c. Ekspos helper mshqc.standard_frozen_core(mol)
    src = src.replace(
        'm.def("hartree_to_kcal",',
        'm.def("standard_frozen_core", &mshqc::compute_standard_frozen_core, nb::arg("mol"),\n'
        '        "Compute standard noble-gas frozen core orbital count (Psi4 convention)");\n'
        '  m.def("hartree_to_kcal",',
        1,
    )

    write_text(p, src, dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description="MSHQC HPC v2 Production Scaling Injector")
    parser.add_argument("--root", type=Path, default=Path("."), help="Root direktori repositori mshqc")
    parser.add_argument("--dry-run", action="store_true", help="Simulasi tanpa mengubah file")
    args = parser.parse_args()

    root = args.root.resolve()
    print(f"=== MSHQC HPC v2 Production Scaling Injector ===\nRoot: {root}\n")

    print("-> [NEW] include/mshqc/utils/blas_thread_guard.h")
    write_text(root / "include/mshqc/utils/blas_thread_guard.h", BLAS_GUARD_HEADER, args.dry_run, is_new=True)

    print("-> [NEW] include/mshqc/integrals/df_metric_utils.h")
    write_text(root / "include/mshqc/integrals/df_metric_utils.h", DF_METRIC_HEADER, args.dry_run, is_new=True)

    patches = [
        ("include/mshqc/integrals/df_eri.h", patch_df_eri_h),
        ("include/mshqc/integrals/packed_ao_utils.h", patch_packed_ao_utils),
        ("src/integrals/df_eri.cc", patch_df_eri_cc),
        ("src/scf/scf.cc", patch_scf_cc),
        ("include/mshqc/mp2/mp2.h", patch_mp2_h),
        ("src/mp2/mp2.cc", patch_mp2_cc),
        ("src/mp/omp2_energy.cc", patch_omp2_energy),
        ("src/mp/omp_gradient.cc", patch_omp_gradient),
        ("src/gradient/hpc_geom_opt.cc", patch_hpc_geom_opt),
        ("src/mp3/mp3.cc", patch_mp3_cc),
        ("python/bindings.cc", patch_bindings_cc),
    ]

    for label, fn in patches:
        print(f"-> [MOD] {label}")
        fn(root, args.dry_run)

    print("\n=== SELESAI ===")
    print("Langkah berikutnya:")
    print("  1. Rebuild C++20 extension (pip install -e . --no-build-isolation atau cmake --build build).")
    print("  2. Jalankan verifikasi gradien analitik + simetri: python tests/verify_audit_and_scaling.py")
    print("  3. Stress-test molekul besar (Benzena cc-pVTZ 264 BF / Aspirin cc-pVDZ 222 BF / Naftalena cc-pVTZ 404 BF) + --with-psi4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())