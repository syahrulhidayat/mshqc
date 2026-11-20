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
 * @file src/mp3/mp3.cc
 * @brief Unified MP3 Implementation Powered by Native TBLIS & HPC Cholesky Factorization
 */

#include "mshqc/mp3/mp3.h"
#include "mshqc/mp3/omp3.h"
#include "mshqc/integrals/eri_transformer.h"
#include "mshqc/integrals/packed_ao_utils.h"
// [INJECTED_PRIORITAS_2D_PACKED_AO]
#include "mshqc/gradient/optimizer.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <omp.h>
#include <tblis/tblis.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace tblis {

inline size_t view_num_elements(const varray_view<double>& v) {
    const auto& len = v.lengths();
    if (len.empty()) return 0;
    size_t prod = 1;
    for (auto d : len) {
        if (d <= 0) return 0;
        prod *= static_cast<size_t>(d);
    }
    return prod;
}

#ifdef _WIN32
inline void build_offset_table(
    const std::vector<ptrdiff_t>& lengths,
    const std::vector<ptrdiff_t>& strides1,
    const std::vector<ptrdiff_t>& strides2,
    std::vector<ptrdiff_t>& off1,
    std::vector<ptrdiff_t>& off2)
{
    off1.assign(1, 0);
    off2.assign(1, 0);
    for (size_t d = 0; d < lengths.size(); ++d) {
        const ptrdiff_t L = lengths[d];
        const ptrdiff_t s1 = strides1[d];
        const ptrdiff_t s2 = strides2[d];
        const size_t prev = off1.size();
        std::vector<ptrdiff_t> next1(prev * static_cast<size_t>(L));
        std::vector<ptrdiff_t> next2(prev * static_cast<size_t>(L));
        for (size_t i = 0; i < prev; ++i) {
            const ptrdiff_t b1 = off1[i];
            const ptrdiff_t b2 = off2[i];
            const size_t base_idx = i * static_cast<size_t>(L);
            for (ptrdiff_t x = 0; x < L; ++x) {
                next1[base_idx + static_cast<size_t>(x)] = b1 + x * s1;
                next2[base_idx + static_cast<size_t>(x)] = b2 + x * s2;
            }
        }
        off1 = std::move(next1);
        off2 = std::move(next2);
    }
}

inline void win32_eigen_tensor_mult(
    double alpha,
    const varray_view<double>& A, const char* idxA,
    const varray_view<double>& B, const char* idxB,
    double beta,
    varray_view<double>& C, const char* idxC)
{
    const auto& lenA = A.lengths();
    const auto& strA = A.strides();
    const auto& lenB = B.lengths();
    const auto& strB = B.strides();
    const auto& lenC = C.lengths();
    const auto& strC = C.strides();

    const size_t ndimA = lenA.size();
    const size_t ndimB = lenB.size();
    const size_t ndimC = lenC.size();

    std::vector<ptrdiff_t> len_M, strA_M, strC_M;
    std::vector<ptrdiff_t> len_N, strB_N, strC_N;
    std::vector<ptrdiff_t> len_K, strA_K, strB_K;

    for (size_t dc = 0; dc < ndimC; ++dc) {
        const char ch = idxC[dc];
        bool in_A = false;
        for (size_t da = 0; da < ndimA; ++da) {
            if (idxA[da] == ch) {
                len_M.push_back(static_cast<ptrdiff_t>(lenC[dc]));
                strA_M.push_back(static_cast<ptrdiff_t>(strA[da]));
                strC_M.push_back(static_cast<ptrdiff_t>(strC[dc]));
                in_A = true;
                break;
            }
        }
        if (!in_A) {
            for (size_t db = 0; db < ndimB; ++db) {
                if (idxB[db] == ch) {
                    len_N.push_back(static_cast<ptrdiff_t>(lenC[dc]));
                    strB_N.push_back(static_cast<ptrdiff_t>(strB[db]));
                    strC_N.push_back(static_cast<ptrdiff_t>(strC[dc]));
                    break;
                }
            }
        }
    }

    for (size_t da = 0; da < ndimA; ++da) {
        const char ch = idxA[da];
        bool in_C = false;
        for (size_t dc = 0; dc < ndimC; ++dc) {
            if (idxC[dc] == ch) { in_C = true; break; }
        }
        if (!in_C) {
            for (size_t db = 0; db < ndimB; ++db) {
                if (idxB[db] == ch) {
                    len_K.push_back(static_cast<ptrdiff_t>(lenA[da]));
                    strA_K.push_back(static_cast<ptrdiff_t>(strA[da]));
                    strB_K.push_back(static_cast<ptrdiff_t>(strB[db]));
                    break;
                }
            }
        }
    }

    std::vector<ptrdiff_t> offA_M, offC_M, offB_N, offC_N, offA_K, offB_K;
    build_offset_table(len_M, strA_M, strC_M, offA_M, offC_M);
    build_offset_table(len_N, strB_N, strC_N, offB_N, offC_N);
    build_offset_table(len_K, strA_K, strB_K, offA_K, offB_K);

    const Eigen::Index M = static_cast<Eigen::Index>(offA_M.size());
    const Eigen::Index N = static_cast<Eigen::Index>(offB_N.size());
    const Eigen::Index K = static_cast<Eigen::Index>(offA_K.size());

    Eigen::MatrixXd MatA(M, K);
    const double* pA = A.data();
    double* dstA = MatA.data();
    for (Eigen::Index k = 0; k < K; ++k) {
        const ptrdiff_t baseA_k = offA_K[static_cast<size_t>(k)];
        double* colA = dstA + k * M;
        for (Eigen::Index m = 0; m < M; ++m) {
            colA[m] = pA[baseA_k + offA_M[static_cast<size_t>(m)]];
        }
    }

    Eigen::MatrixXd MatB(K, N);
    const double* pB = B.data();
    double* dstB = MatB.data();
    for (Eigen::Index n = 0; n < N; ++n) {
        const ptrdiff_t baseB_n = offB_N[static_cast<size_t>(n)];
        double* colB = dstB + n * K;
        for (Eigen::Index k = 0; k < K; ++k) {
            colB[k] = pB[offB_K[static_cast<size_t>(k)] + baseB_n];
        }
    }

    Eigen::MatrixXd MatC(M, N);
    MatC.noalias() = alpha * (MatA * MatB);

    double* pC = C.data();
    const double* srcC = MatC.data();
    for (Eigen::Index n = 0; n < N; ++n) {
        const ptrdiff_t baseC_n = offC_N[static_cast<size_t>(n)];
        const double* colC = srcC + n * M;
        if (beta == 0.0) {
            for (Eigen::Index m = 0; m < M; ++m) {
                pC[offC_M[static_cast<size_t>(m)] + baseC_n] = colC[m];
            }
        } else {
            for (Eigen::Index m = 0; m < M; ++m) {
                const ptrdiff_t idx = offC_M[static_cast<size_t>(m)] + baseC_n;
                pC[idx] = beta * pC[idx] + colC[m];
            }
        }
    }
}
#endif

template <typename T = double>
inline void safe_tblis_mult(
    T alpha,
    const varray_view<double>& A, const char* idxA,
    const varray_view<double>& B, const char* idxB,
    T beta,
    varray_view<double>& C, const char* idxC)
{
    const size_t szA = view_num_elements(A);
    const size_t szB = view_num_elements(B);
    const size_t szC = view_num_elements(C);

    if (szC == 0) return;
    if (szA == 0 || szB == 0) {
        Eigen::Map<Eigen::VectorXd> vecC(C.data(), static_cast<Eigen::Index>(szC));
        if (beta == T(0)) vecC.setZero();
        else vecC *= static_cast<double>(beta);
        return;
    }

    if (szC == 1 && szA == szB) {
        Eigen::Map<const Eigen::VectorXd> vecA(A.data(), static_cast<Eigen::Index>(szA));
        Eigen::Map<const Eigen::VectorXd> vecB(B.data(), static_cast<Eigen::Index>(szB));
        const double dot = vecA.dot(vecB);
        C.data()[0] = (beta == T(0)) ? (alpha * dot) : (beta * C.data()[0] + alpha * dot);
        return;
    }
    if (szA == 1 && szB == szC) {
        Eigen::Map<const Eigen::VectorXd> vecB(B.data(), static_cast<Eigen::Index>(szB));
        Eigen::Map<Eigen::VectorXd> vecC(C.data(), static_cast<Eigen::Index>(szC));
        const double scale = alpha * A.data()[0];
        if (beta == T(0)) vecC.noalias() = scale * vecB;
        else vecC.noalias() = beta * vecC + scale * vecB;
        return;
    }
    if (szB == 1 && szA == szC) {
        Eigen::Map<const Eigen::VectorXd> vecA(A.data(), static_cast<Eigen::Index>(szA));
        Eigen::Map<Eigen::VectorXd> vecC(C.data(), static_cast<Eigen::Index>(szC));
        const double scale = alpha * B.data()[0];
        if (beta == T(0)) vecC.noalias() = scale * vecA;
        else vecC.noalias() = beta * vecC + scale * vecA;
        return;
    }

#ifdef _WIN32
    win32_eigen_tensor_mult(static_cast<double>(alpha), A, idxA, B, idxB, static_cast<double>(beta), C, idxC);
#else
    tblis_tensor A_s(alpha, const_cast<varray_view<double>&>(A));
    tblis_tensor B_s(const_cast<varray_view<double>&>(B));
    tblis_tensor C_s(beta, C);
    tblis_tensor_mult(nullptr, nullptr, &A_s, idxA, &B_s, idxB, &C_s, idxC);
#endif
}

} // namespace tblis

#define mult safe_tblis_mult
namespace {

Eigen::MatrixXd parallel_gemm_abt_mp3(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    if (A.rows() <= 0 || B.rows() <= 0 || A.cols() <= 0 || B.cols() <= 0) {
        return Eigen::MatrixXd();
    }
    Eigen::MatrixXd C(A.rows(), B.rows());
    const int m = static_cast<int>(A.rows());
    constexpr int tile = 64;
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < m; i += tile) {
        int rows = std::min(tile, m - i);
        C.middleRows(i, rows).noalias() = A.middleRows(i, rows) * B.transpose();
    }
    return C;
}

tblis::varray_view<double> make_mat_view_4d_mp3(Eigen::MatrixXd& M, int d1, int d2, int d3, int d4) {
    using tblis::len_type;
    using tblis::stride_type;
    return tblis::varray_view<double>(
        {(len_type)d1, (len_type)d2, (len_type)d3, (len_type)d4},
        M.data(),
        {(stride_type)d2, 1, (stride_type)(d1 * d2 * d4), (stride_type)(d1 * d2)}
    );
}


void contract_vvvv_pairs_df(
    const Eigen::MatrixXd& B_vv_left, int nv_l,
    const Eigen::MatrixXd& B_vv_right, int nv_r,
    int n_aux,
    const std::vector<Eigen::MatrixXd>& T_pairs,
    std::vector<Eigen::MatrixXd>& W_pairs)
{
  const int n_pairs = static_cast<int>(T_pairs.size());
  if (n_pairs <= 0 || n_aux <= 0 || nv_l <= 0 || nv_r <= 0) return;

  const bool same_B = (&B_vv_left == &B_vv_right) && (nv_l == nv_r);
  const double flops_loop_p = 4.0 * static_cast<double>(n_pairs) * nv_l * nv_r * std::max(nv_l, nv_r) * n_aux;

  // Jalur 1: Adaptive Batched T+/T- BLAS-3 DGEMM untuk same-spin / restricted (hemat hingga 6-15x FLOPs, RAM <= 64 MB)
  if (same_B && nv_l >= 8) {
    const int V = nv_l;
    bool is_antisym = true;
    for (int k = 0; k < n_pairs && is_antisym; ++k) {
      if ((T_pairs[k] + T_pairs[k].transpose()).cwiseAbs().maxCoeff() > 1e-13) {
        is_antisym = false;
      }
    }
    const double V4 = static_cast<double>(V) * V * V * V;
    const double flops_batched = V4 * n_aux + (is_antisym ? 0.5 : 1.0) * V4 * n_pairs;

    if (flops_batched < 0.85 * flops_loop_p) {
      const int n_tri = V * (V + 1) / 2;
      const int n_strict = V * (V - 1) / 2;

      Eigen::MatrixXd T_plus;
      if (!is_antisym) {
        T_plus.resize(n_tri, n_pairs);
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n_pairs; ++k) {
          double* col = T_plus.col(k).data();
          const Eigen::MatrixXd& Tk = T_pairs[k];
          for (int e = 0; e < V; ++e) {
            const int base = e * (e + 1) / 2;
            for (int f = 0; f < e; ++f) col[base + f] = Tk(e, f) + Tk(f, e);
            col[base + e] = Tk(e, e);
          }
        }
      }

      Eigen::MatrixXd T_minus;
      if (n_strict > 0) {
        T_minus.resize(n_strict, n_pairs);
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n_pairs; ++k) {
          double* col = T_minus.col(k).data();
          const Eigen::MatrixXd& Tk = T_pairs[k];
          for (int e = 1; e < V; ++e) {
            const int base = e * (e - 1) / 2;
            for (int f = 0; f < e; ++f) col[base + f] = Tk(e, f) - Tk(f, e);
          }
        }
      }

      std::vector<int> pair_a(n_tri), pair_b(n_tri);
      for (int a = 0; a < V; ++a) {
        const int base = a * (a + 1) / 2;
        for (int b = 0; b <= a; ++b) {
          pair_a[base + b] = a;
          pair_b[base + b] = b;
        }
      }

      // Batasi ukuran buffer batch V+/V- <= 64 MB agar tetap O(V^4)-memory-free
      constexpr size_t kMaxBatchBytes = 64ULL * 1024ULL * 1024ULL;
      const size_t col_bytes = static_cast<size_t>((is_antisym ? 0 : n_tri) + n_strict) * sizeof(double);
      const int batch_ab = std::max(1, std::min(n_tri, static_cast<int>(kMaxBatchBytes / std::max<size_t>(1, col_bytes))));

      Eigen::MatrixXd V_plus_T;
      Eigen::MatrixXd V_minus_T;
      Eigen::MatrixXd W_plus_b;
      Eigen::MatrixXd W_minus_b;

      for (int p0 = 0; p0 < n_tri; p0 += batch_ab) {
        const int bsz = std::min(batch_ab, n_tri - p0);
        if (!is_antisym) V_plus_T.resize(n_tri, bsz);
        if (n_strict > 0) V_minus_T.resize(n_strict, bsz);

        #pragma omp parallel
        {
          Eigen::MatrixXd Ba_loc(V, n_aux), Bb_loc(V, n_aux), M_ab(V, V);
          int last_a = -1;
          #pragma omp for schedule(static) nowait
          for (int r = 0; r < bsz; ++r) {
            const int a = pair_a[p0 + r];
            const int b = pair_b[p0 + r];
            if (a != last_a) {
              Ba_loc = B_vv_left.middleRows(static_cast<Eigen::Index>(a) * V, V);
              last_a = a;
            }
            if (b == a) {
              M_ab.noalias() = Ba_loc * Ba_loc.transpose();
            } else {
              Bb_loc = B_vv_left.middleRows(static_cast<Eigen::Index>(b) * V, V);
              M_ab.noalias() = Ba_loc * Bb_loc.transpose();
            }

            if (!is_antisym) {
              double* vp = V_plus_T.col(r).data();
              for (int e = 0; e < V; ++e) {
                const int base = e * (e + 1) / 2;
                for (int f = 0; f < e; ++f) vp[base + f] = 0.5 * (M_ab(e, f) + M_ab(f, e));
                vp[base + e] = M_ab(e, e);
              }
            }
            if (n_strict > 0) {
              double* vm = V_minus_T.col(r).data();
              if (a == b) {
                std::fill(vm, vm + n_strict, 0.0);
              } else {
                for (int e = 1; e < V; ++e) {
                  const int base = e * (e - 1) / 2;
                  for (int f = 0; f < e; ++f) vm[base + f] = 0.5 * (M_ab(e, f) - M_ab(f, e));
                }
              }
            }
          }
        }

        if (!is_antisym) {
          W_plus_b.resize(bsz, n_pairs);
          constexpr int tile_r = 32;
          #pragma omp parallel for schedule(dynamic)
          for (int r0 = 0; r0 < bsz; r0 += tile_r) {
            const int rlen = std::min(tile_r, bsz - r0);
            W_plus_b.middleRows(r0, rlen).noalias() = V_plus_T.middleCols(r0, rlen).transpose() * T_plus;
          }
        }
        if (n_strict > 0) {
          W_minus_b.resize(bsz, n_pairs);
          constexpr int tile_r = 32;
          #pragma omp parallel for schedule(dynamic)
          for (int r0 = 0; r0 < bsz; r0 += tile_r) {
            const int rlen = std::min(tile_r, bsz - r0);
            W_minus_b.middleRows(r0, rlen).noalias() = V_minus_T.middleCols(r0, rlen).transpose() * T_minus;
          }
        }

        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n_pairs; ++k) {
          Eigen::MatrixXd& Wk = W_pairs[k];
          for (int r = 0; r < bsz; ++r) {
            const int a = pair_a[p0 + r];
            const int b = pair_b[p0 + r];
            const double wp = is_antisym ? 0.0 : W_plus_b(r, k);
            const double wm = (n_strict > 0) ? W_minus_b(r, k) : 0.0;
            if (a == b) {
              Wk(a, a) += wp;
            } else {
              Wk(a, b) += wp + wm;
              Wk(b, a) += wp - wm;
            }
          }
        }
      }
      return;
    }
  }

  // Jalur 2: Adaptive Batched BLAS-3 DGEMM untuk kanal opposite-spin (UHF AB) saat O_a*O_b besar
  if (!same_B && nv_l >= 8 && nv_r >= 8) {
    const double n_ab = static_cast<double>(nv_l) * nv_r;
    const double flops_batched_ab = 2.0 * n_ab * n_ab * n_aux + 2.0 * n_ab * n_ab * n_pairs;
    if (flops_batched_ab < 0.85 * flops_loop_p) {
      const int n_ef = nv_l * nv_r;
      Eigen::MatrixXd T_pack(n_ef, n_pairs);
      #pragma omp parallel for schedule(static)
      for (int k = 0; k < n_pairs; ++k) {
        std::copy(T_pairs[k].data(), T_pairs[k].data() + n_ef, T_pack.col(k).data());
      }

      constexpr size_t kMaxBatchBytes = 64ULL * 1024ULL * 1024ULL;
      const int batch_ab = std::max(1, std::min(n_ef, static_cast<int>(kMaxBatchBytes / std::max<size_t>(1, static_cast<size_t>(n_ef) * sizeof(double)))));

      Eigen::MatrixXd V_b_T;
      Eigen::MatrixXd W_b;
      for (int p0 = 0; p0 < n_ef; p0 += batch_ab) {
        const int bsz = std::min(batch_ab, n_ef - p0);
        V_b_T.resize(n_ef, bsz);

        #pragma omp parallel
        {
          Eigen::MatrixXd Ba_loc(nv_l, n_aux), Bb_loc(nv_r, n_aux), M_ab(nv_l, nv_r);
          int last_a = -1;
          #pragma omp for schedule(static) nowait
          for (int r = 0; r < bsz; ++r) {
            const int idx = p0 + r;
            const int a = idx / nv_r;
            const int b = idx % nv_r;
            if (a != last_a) {
              Ba_loc = B_vv_left.middleRows(static_cast<Eigen::Index>(a) * nv_l, nv_l);
              last_a = a;
            }
            Bb_loc = B_vv_right.middleRows(static_cast<Eigen::Index>(b) * nv_r, nv_r);
            M_ab.noalias() = Ba_loc * Bb_loc.transpose();
            std::copy(M_ab.data(), M_ab.data() + n_ef, V_b_T.col(r).data());
          }
        }

        W_b.resize(bsz, n_pairs);
        constexpr int tile_r = 32;
        #pragma omp parallel for schedule(dynamic)
        for (int r0 = 0; r0 < bsz; r0 += tile_r) {
          const int rlen = std::min(tile_r, bsz - r0);
          W_b.middleRows(r0, rlen).noalias() = V_b_T.middleCols(r0, rlen).transpose() * T_pack;
        }

        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n_pairs; ++k) {
          Eigen::MatrixXd& Wk = W_pairs[k];
          for (int r = 0; r < bsz; ++r) {
            const int idx = p0 + r;
            Wk(idx / nv_r, idx % nv_r) += W_b(r, k);
          }
        }
      }
      return;
    }
  }

  // Jalur 3: Loop-P original (optimal saat jumlah pasangan terisi n_pairs kecil dibanding V)
  if (n_pairs >= omp_get_max_threads()) {
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n_pairs; ++k) {
      Eigen::MatrixXd X_tmp(nv_l, nv_r);
      for (int P = 0; P < n_aux; ++P) {
        Eigen::Map<const Eigen::MatrixXd> B_Pl(B_vv_left.col(P).data(), nv_l, nv_l);
        Eigen::Map<const Eigen::MatrixXd> B_Pr(B_vv_right.col(P).data(), nv_r, nv_r);
        X_tmp.noalias() = T_pairs[k] * B_Pr;
        W_pairs[k].noalias() += B_Pl.transpose() * X_tmp;
      }
    }
  } else {
    #pragma omp parallel
    {
      std::vector<Eigen::MatrixXd> W_loc(n_pairs, Eigen::MatrixXd::Zero(nv_l, nv_r));
      Eigen::MatrixXd X_tmp(nv_l, nv_r);
      #pragma omp for schedule(static) nowait
      for (int P = 0; P < n_aux; ++P) {
        Eigen::Map<const Eigen::MatrixXd> B_Pl(B_vv_left.col(P).data(), nv_l, nv_l);
        Eigen::Map<const Eigen::MatrixXd> B_Pr(B_vv_right.col(P).data(), nv_r, nv_r);
        for (int k = 0; k < n_pairs; ++k) {
          X_tmp.noalias() = T_pairs[k] * B_Pr;
          W_loc[k].noalias() += B_Pl.transpose() * X_tmp;
        }
      }
      #pragma omp critical
      {
        for (int k = 0; k < n_pairs; ++k) W_pairs[k].noalias() += W_loc[k];
      }
    }
  }
}

} // namespace

namespace mshqc {

using integrals::ERITransformer;
using tblis::len_type;
using tblis::stride_type;
using tblis::varray_view;
BaseMP3::BaseMP3(const SCFResult& scf, const MP2Result& mp2, const MP2Config& config, std::shared_ptr<IntegralEngine> ints)
    : scf_(scf), mp2_(mp2), config_(config), ints_(ints) 
{
    std::transform(config_.eri_method.begin(), config_.eri_method.end(), config_.eri_method.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (config_.eri_method == "df") {
        config_.use_df = true;
    } else if (!config_.use_df && config_.eri_method != "cholesky") {
        config_.eri_method = "exact";
    }

    nbf_ = scf_.C_alpha.rows();
    no_a_ = scf_.n_occ_alpha; no_b_ = scf_.n_occ_beta;
    nv_a_ = nbf_ - no_a_;     nv_b_ = nbf_ - no_b_;
    n_aux_ = scf_.L_mat.cols();
    
    if (config_.n_frozen > 0) n_frozen_ = config_.n_frozen; 
    if (mp2_.C_alpha.rows() == nbf_) scf_.C_alpha = mp2_.C_alpha;
    if (mp2_.C_beta.rows() == nbf_)  scf_.C_beta  = mp2_.C_beta;
    if (mp2_.orbital_energies_alpha.size() == nbf_) scf_.orbital_energies_alpha = mp2_.orbital_energies_alpha;
    if (mp2_.orbital_energies_beta.size() == nbf_)  scf_.orbital_energies_beta  = mp2_.orbital_energies_beta;
    if (mp2_.energy_scf != 0.0) scf_.energy_total = mp2_.energy_scf;

    if (mp2_.t2_aa.size() > 0) t2_aa_ = mp2_.t2_aa;
    if (mp2_.t2_bb.size() > 0) t2_bb_ = mp2_.t2_bb;
    if (mp2_.t2_ab.size() > 0) t2_ab_ = mp2_.t2_ab;
}

double BaseMP3::tensor_dot(const Eigen::Tensor<double, 4>& A, const Eigen::Tensor<double, 4>& B) const {
    Eigen::Map<const Eigen::VectorXd> vecA(A.data(), A.size());
    Eigen::Map<const Eigen::VectorXd> vecB(B.data(), B.size());
    return vecA.dot(vecB);
}

#define TBLIS_VIEW_4D(name, t, d1, d2, d3, d4) \
    varray_view<double> name({(len_type)d1, (len_type)d2, (len_type)d3, (len_type)d4}, t.data(), \
    {1, (stride_type)d1, (stride_type)(d1*d2), (stride_type)(d1*d2*d3)})

#define TBLIS_VIEW_3D(name, ptr, d1, d2, d3) \
    varray_view<double> name({(len_type)d1, (len_type)d2, (len_type)d3}, ptr, \
    {1, (stride_type)d1, (stride_type)(d1*d2)})

#define TBLIS_VIEW_2D(name, ptr, d1, d2) \
    varray_view<double> name({(len_type)d1, (len_type)d2}, ptr, \
    {1, (stride_type)d1})

// [INJECTED_FINAL_OPT_RMP3_FROZEN_FIXED]
MP3Result RMP3::compute() {
    auto t_start = std::chrono::high_resolution_clock::now();
    if(omp_get_thread_num() == 0) std::cout << "\n=== RMP3 (Unified Native TBLIS - Cholesky) ===\n";

    const int f_a = std::clamp(n_frozen_, 0, no_a_);
  if (t2_aa_.size() == 0) {
    throw std::runtime_error("RMP3/UMP3 memerlukan tensor T2 penuh di RAM, tetapi MP2 sebelumnya berjalan dalam mode streaming O(V^2) karena batas memory_limit_gb. Naikkan memory_limit_gb atau aktifkan freeze_core=True.");
  }
    const int no_act = no_a_ - f_a;
    if (no_act <= 0 || nv_a_ <= 0) {
        MP3Result empty_res;
        empty_res.e_hf = scf_.energy_total;
        empty_res.e_mp2 = mp2_.energy_mp2_corr;
        empty_res.e_corr_total = empty_res.e_mp2;
        empty_res.e_total = empty_res.e_hf + empty_res.e_corr_total;
        return empty_res;
    }

    Eigen::Tensor<double, 4> t2_act_buf;
    Eigen::Tensor<double, 4>* t2_ptr = &t2_aa_;
    if (f_a > 0) {
        t2_act_buf = Eigen::Tensor<double, 4>(no_act, no_act, nv_a_, nv_a_);
        #pragma omp parallel for collapse(4) schedule(static)
        for (int i = 0; i < no_act; ++i)
            for (int j = 0; j < no_act; ++j)
                for (int a = 0; a < nv_a_; ++a)
                    for (int b = 0; b < nv_a_; ++b)
                        t2_act_buf(i, j, a, b) = t2_aa_(f_a + i, f_a + j, a, b);
        t2_ptr = &t2_act_buf;
    }
    Eigen::Tensor<double, 4>& t2_use = *t2_ptr;
    const int no_full_saved = no_a_;
    no_a_ = no_act;

    const Eigen::MatrixXd Co = scf_.C_alpha.middleCols(f_a, no_a_);
    const auto& Cv = scf_.C_alpha.rightCols(nv_a_);

    TBLIS_VIEW_4D(t_T, t2_use, no_a_, no_a_, nv_a_, nv_a_);
    Eigen::Tensor< double, 4 > W(no_a_, no_a_, nv_a_, nv_a_); W.setZero();
    TBLIS_VIEW_4D(t_W, W, no_a_, no_a_, nv_a_, nv_a_);

    const Eigen::Tensor<double, 4>* eri_ao_ptr = nullptr;
    if (config_.eri_method == "exact") eri_ao_ptr = &(ints_->compute_eri());

    Eigen::MatrixXd B_oo, B_ov, B_vv;
    if (config_.eri_method != "exact") {
        B_ov = (f_a > 0)
            ? mp2_.B_ia_P_alpha.bottomRows(static_cast<Eigen::Index>(no_a_) * nv_a_).eval()
            : mp2_.B_ia_P_alpha;

        auto build_B_fast = [&](const Eigen::MatrixXd& C1, const Eigen::MatrixXd& C2) {
            return integrals::transform_3c_ao_to_mo_block(scf_.L_mat, C1, C2, nbf_);
        };

        B_oo = build_B_fast(Co, Co);
        B_vv = build_B_fast(Cv, Cv);
    }

    // 1. Ladder term (vvvv) - AO-Driven Unique-Pair untuk Mode EXACT (0 MB V^4, identik dengan OMP3)
    if (config_.eri_method == "exact") {
        tblis::varray_view<double> t_eri_ao(
            {(len_type)nbf_, (len_type)nbf_, (len_type)nbf_, (len_type)nbf_},
            const_cast<double*>(eri_ao_ptr->data()),
            {1, (stride_type)nbf_, (stride_type)(nbf_*nbf_), (stride_type)(nbf_*nbf_*nbf_)}
        );
        int n_pairs_sym = no_a_ * (no_a_ + 1) / 2;
        Eigen::Tensor<double, 3> T_ao(n_pairs_sym, nbf_, nbf_);
        Eigen::Tensor<double, 3> W_ao(n_pairs_sym, nbf_, nbf_);
        W_ao.setZero();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < no_a_; ++i) {
            for (int j = i; j < no_a_; ++j) {
                int k = i * no_a_ - (i * (i - 1)) / 2 + (j - i);
                Eigen::MatrixXd Tij_vv(nv_a_, nv_a_);
                for (int e = 0; e < nv_a_; ++e)
                    for (int f = 0; f < nv_a_; ++f)
                        Tij_vv(e, f) = t2_use(i, j, e, f);
                Eigen::MatrixXd Tij_ao = Cv * Tij_vv * Cv.transpose();
                for (int mu = 0; mu < nbf_; ++mu)
                    for (int lam = 0; lam < nbf_; ++lam)
                        T_ao(k, mu, lam) = Tij_ao(mu, lam);
            }
        }

        TBLIS_VIEW_3D(t_Tao, T_ao.data(), n_pairs_sym, nbf_, nbf_);
        TBLIS_VIEW_3D(t_Wao, W_ao.data(), n_pairs_sym, nbf_, nbf_);
        tblis::mult<double>(1.0, t_Tao, "kpr", t_eri_ao, "pqrs", 0.0, t_Wao, "kqs");

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < no_a_; ++i) {
            for (int j = i; j < no_a_; ++j) {
                int k = i * no_a_ - (i * (i - 1)) / 2 + (j - i);
                Eigen::MatrixXd Wij_ao(nbf_, nbf_);
                for (int nu = 0; nu < nbf_; ++nu)
                    for (int sig = 0; sig < nbf_; ++sig)
                        Wij_ao(nu, sig) = W_ao(k, nu, sig);
                Eigen::MatrixXd Wij_vv = Cv.transpose() * Wij_ao * Cv;
                for (int a = 0; a < nv_a_; ++a) {
                    for (int b = 0; b < nv_a_; ++b) {
                        W(i, j, a, b) += Wij_vv(a, b);
                        if (i != j) W(j, i, b, a) += Wij_vv(a, b);
                    }
                }
            }
        }
    } else {
        int n_pairs_sym = no_a_ * (no_a_ + 1) / 2;
        std::vector<Eigen::MatrixXd> T_pairs(n_pairs_sym, Eigen::MatrixXd(nv_a_, nv_a_));
        std::vector<Eigen::MatrixXd> W_pairs(n_pairs_sym, Eigen::MatrixXd::Zero(nv_a_, nv_a_));
        int p_idx = 0;
        for (int i = 0; i < no_a_; ++i) {
            for (int j = i; j < no_a_; ++j) {
                for (int e = 0; e < nv_a_; ++e)
                    for (int f = 0; f < nv_a_; ++f)
                        T_pairs[p_idx](e, f) = t2_use(i, j, e, f);
                p_idx++;
            }
        }
        contract_vvvv_pairs_df(B_vv, nv_a_, B_vv, nv_a_, n_aux_, T_pairs, W_pairs);
        p_idx = 0;
        for (int i = 0; i < no_a_; ++i) {
            for (int j = i; j < no_a_; ++j) {
                for (int a = 0; a < nv_a_; ++a) {
                    for (int b = 0; b < nv_a_; ++b) {
                        W(i, j, a, b) += W_pairs[p_idx](a, b);
                        if (i != j) W(j, i, b, a) += W_pairs[p_idx](a, b);
                    }
                }
                p_idx++;
            }
        }
    }

    // 2 & 3. Ladder term (oooo) + Ring terms (ovov & oovv)
    if (config_.eri_method == "exact") {
        auto half_oo = ERITransformer::transform_half_bra(*eri_ao_ptr, Co, Co, nbf_, no_a_, no_a_);
        Eigen::Tensor<double, 4> V = ERITransformer::transform_finish_ket(half_oo, Co, Co, nbf_, no_a_, no_a_, no_a_, no_a_);
        Eigen::Tensor<double, 4> V_oovv = ERITransformer::transform_finish_ket(half_oo, Cv, Cv, nbf_, no_a_, no_a_, nv_a_, nv_a_);
        half_oo.resize(0, 0, 0, 0);
        Eigen::Tensor<double, 4> V_ovov = ERITransformer::transform_custom(*eri_ao_ptr, Co, Cv, Co, Cv, nbf_, no_a_, nv_a_, no_a_, nv_a_);

        TBLIS_VIEW_4D(t_V, V, no_a_, no_a_, no_a_, no_a_);
        TBLIS_VIEW_4D(t_Vovov, V_ovov, no_a_, nv_a_, no_a_, nv_a_);
        TBLIS_VIEW_4D(t_Voovv, V_oovv, no_a_, no_a_, nv_a_, nv_a_);

        tblis::mult<double>( 1.0, t_T, "mnab", t_V, "minj", 1.0, t_W, "ijab");
        tblis::mult<double>( 2.0, t_Vovov, "iakc", t_T, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>( 2.0, t_T, "ikac", t_Vovov, "kcjb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T, "ikca", t_Vovov, "kcjb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Vovov, "iakc", t_T, "kjbc", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikac", t_T, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T, "ikac", t_Voovv, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikbc", t_T, "kjac", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T, "ikcb", t_Voovv, "jkac", 1.0, t_W, "ijab");
    } else {
        Eigen::MatrixXd M_oooo = parallel_gemm_abt_mp3(B_oo, B_oo);
        Eigen::MatrixXd M_ovov = parallel_gemm_abt_mp3(B_ov, B_ov);
        Eigen::MatrixXd M_oovv = parallel_gemm_abt_mp3(B_oo, B_vv);

        auto t_V = make_mat_view_4d_mp3(M_oooo, no_a_, no_a_, no_a_, no_a_);
        auto t_Vovov = make_mat_view_4d_mp3(M_ovov, no_a_, nv_a_, no_a_, nv_a_);
        auto t_Voovv = make_mat_view_4d_mp3(M_oovv, no_a_, no_a_, nv_a_, nv_a_);

        tblis::mult<double>( 1.0, t_T, "mnab", t_V, "minj", 1.0, t_W, "ijab");
        tblis::mult<double>( 2.0, t_Vovov, "iakc", t_T, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>( 2.0, t_T, "ikac", t_Vovov, "kcjb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T, "ikca", t_Vovov, "kcjb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Vovov, "iakc", t_T, "kjbc", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikac", t_T, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T, "ikac", t_Voovv, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikbc", t_T, "kjac", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T, "ikcb", t_Voovv, "jkac", 1.0, t_W, "ijab");
    }

    double e_mp3 = 0.0;
    #pragma omp parallel for collapse(4) reduction(+:e_mp3)
    for (int i = 0; i < no_a_; ++i) {
        for (int j = 0; j < no_a_; ++j) {
            for (int a = 0; a < nv_a_; ++a) {
                for (int b = 0; b < nv_a_; ++b) {
                    e_mp3 += W(i, j, a, b) * (2.0 * t2_use(i, j, a, b) - t2_use(i, j, b, a));
                }
            }
        }
    }

    no_a_ = no_full_saved;

    MP3Result res;
    res.e_hf = scf_.energy_total;
    res.e_mp2 = mp2_.energy_mp2_corr;
    res.e3_aa = 0.0; res.e3_ab = 0.0; res.e_mp3 = e_mp3;
    res.e_corr_total = res.e_mp2 + res.e_mp3;
    res.e_total = res.e_hf + res.e_corr_total;

    auto t_end = std::chrono::high_resolution_clock::now();
    if(omp_get_thread_num() == 0) {
        std::cout << "  E_MP3        : " << std::fixed << std::setprecision(8) << res.e_mp3 << " Ha\n";
        std::cout << "  Total Energy : " << res.e_total << " Ha\n";
        std::cout << "  Time         : " << std::chrono::duration< double >(t_end - t_start).count() << " s\n";
    }
    return res;
}

MP3Result UMP3::compute() {
    auto t_start = std::chrono::high_resolution_clock::now();
    if(omp_get_thread_num() == 0) std::cout << "\n=== UMP3 (Unified Native TBLIS - Cholesky) ===\n";

    const int f_a = std::clamp(n_frozen_, 0, no_a_);
  if (t2_aa_.size() == 0) {
    throw std::runtime_error("RMP3/UMP3 memerlukan tensor T2 penuh di RAM, tetapi MP2 sebelumnya berjalan dalam mode streaming O(V^2) karena batas memory_limit_gb. Naikkan memory_limit_gb atau aktifkan freeze_core=True.");
  }
    const int f_b = std::clamp(n_frozen_, 0, no_b_);
    const int no_a_saved = no_a_;
    const int no_b_saved = no_b_;
    Eigen::Tensor<double, 4> t2_aa_saved, t2_bb_saved, t2_ab_saved;
    Eigen::MatrixXd Bov_a_saved, Bov_b_saved;

    if (f_a > 0 || f_b > 0) {
        // [INJECTED_AUDIT_FIX_UMP3_FROZEN_GUARD]
const int na_act = no_a_ - f_a;
        const int nb_act = no_b_ - f_b;
    if (na_act <= 0 || nv_a_ <= 0) {
      MP3Result empty_res;
      empty_res.e_hf = scf_.energy_total;
      empty_res.e_mp2 = mp2_.energy_mp2_corr;
      empty_res.e_corr_total = empty_res.e_mp2;
      empty_res.e_total = empty_res.e_hf + empty_res.e_corr_total;
      return empty_res;
    }
        t2_aa_saved = std::move(t2_aa_);
        t2_aa_ = Eigen::Tensor<double, 4>(na_act, na_act, nv_a_, nv_a_);
        #pragma omp parallel for collapse(4) schedule(static)
        for (int i = 0; i < na_act; ++i)
            for (int j = 0; j < na_act; ++j)
                for (int a = 0; a < nv_a_; ++a)
                    for (int b = 0; b < nv_a_; ++b)
                        t2_aa_(i, j, a, b) = t2_aa_saved(f_a + i, f_a + j, a, b);

        if (t2_bb_.size() > 0 && nb_act > 0) {
            t2_bb_saved = std::move(t2_bb_);
            t2_bb_ = Eigen::Tensor<double, 4>(nb_act, nb_act, nv_b_, nv_b_);
            #pragma omp parallel for collapse(4) schedule(static)
            for (int i = 0; i < nb_act; ++i)
                for (int j = 0; j < nb_act; ++j)
                    for (int a = 0; a < nv_b_; ++a)
                        for (int b = 0; b < nv_b_; ++b)
                            t2_bb_(i, j, a, b) = t2_bb_saved(f_b + i, f_b + j, a, b);
        }
        if (t2_ab_.size() > 0 && na_act > 0 && nb_act > 0) {
            t2_ab_saved = std::move(t2_ab_);
            t2_ab_ = Eigen::Tensor<double, 4>(na_act, nb_act, nv_a_, nv_b_);
            #pragma omp parallel for collapse(4) schedule(static)
            for (int i = 0; i < na_act; ++i)
                for (int j = 0; j < nb_act; ++j)
                    for (int a = 0; a < nv_a_; ++a)
                        for (int b = 0; b < nv_b_; ++b)
                            t2_ab_(i, j, a, b) = t2_ab_saved(f_a + i, f_b + j, a, b);
        }
        if (config_.eri_method != "exact") {
            if (mp2_.B_ia_P_alpha.rows() >= static_cast<Eigen::Index>(no_a_) * nv_a_) {
        Bov_a_saved = std::move(mp2_.B_ia_P_alpha);
        mp2_.B_ia_P_alpha = Bov_a_saved.bottomRows(static_cast<Eigen::Index>(na_act) * nv_a_);
      }
            if (Bov_b_saved.rows() == 0 && mp2_.B_ia_P_beta.rows() > 0 && nb_act > 0) {
                Bov_b_saved = std::move(mp2_.B_ia_P_beta);
                mp2_.B_ia_P_beta = Bov_b_saved.bottomRows(static_cast<Eigen::Index>(nb_act) * nv_b_);
            }
        }
        no_a_ = na_act;
        no_b_ = nb_act;
    }

    const Eigen::MatrixXd Cao = scf_.C_alpha.middleCols(f_a, no_a_); const auto& Cav = scf_.C_alpha.rightCols(nv_a_);
    const Eigen::MatrixXd Cbo = scf_.C_beta.middleCols(f_b, no_b_);  const auto& Cbv = scf_.C_beta.rightCols(nv_b_);
    double e3_aa = 0.0, e3_bb = 0.0, e3_ab = 0.0;

    Eigen::Tensor< double, 4 > dummy_bb, dummy_ab;
    double* ptr_bb = nullptr;
    double* ptr_ab = nullptr;

    if (no_b_ > 0 && nv_b_ > 0) {
        if (t2_bb_.size() == 0) {
            dummy_bb = Eigen::Tensor< double, 4 >(no_b_, no_b_, nv_b_, nv_b_);
            dummy_bb.setZero();
            ptr_bb = dummy_bb.data();
        } else {
            ptr_bb = t2_bb_.data();
        }

        if (t2_ab_.size() == 0) {
            dummy_ab = Eigen::Tensor< double, 4 >(no_a_, no_b_, nv_a_, nv_b_);
            dummy_ab.setZero();
            ptr_ab = dummy_ab.data();
        } else {
            ptr_ab = t2_ab_.data();
        }
    }
    
    TBLIS_VIEW_4D(t_Taa, t2_aa_, no_a_, no_a_, nv_a_, nv_a_);
    varray_view< double > t_Tbb({(len_type)no_b_, (len_type)no_b_, (len_type)nv_b_, (len_type)nv_b_}, 
        ptr_bb, {1, (stride_type)no_b_, (stride_type)(no_b_*no_b_), (stride_type)(no_b_*no_b_*nv_b_)});
    varray_view< double > t_Tab({(len_type)no_a_, (len_type)no_b_, (len_type)nv_a_, (len_type)nv_b_}, 
        ptr_ab, {1, (stride_type)no_a_, (stride_type)(no_a_*no_b_), (stride_type)(no_a_*no_b_*nv_a_)});

    Eigen::Tensor< double, 4 > Waa(no_a_, no_a_, nv_a_, nv_a_); TBLIS_VIEW_4D(t_Waa, Waa, no_a_, no_a_, nv_a_, nv_a_);
    Eigen::Tensor< double, 4 > Wbb(no_b_, no_b_, nv_b_, nv_b_); TBLIS_VIEW_4D(t_Wbb, Wbb, no_b_, no_b_, nv_b_, nv_b_);
    Eigen::Tensor< double, 4 > Wab(no_a_, no_b_, nv_a_, nv_b_); TBLIS_VIEW_4D(t_Wab, Wab, no_a_, no_b_, nv_a_, nv_b_);

    const Eigen::Tensor<double, 4>* eri_ao_ptr = nullptr;
    if (config_.eri_method == "exact") eri_ao_ptr = &(ints_->compute_eri());

    Eigen::MatrixXd B_oo_a, B_ov_a, B_vv_a;
    Eigen::MatrixXd B_oo_b, B_ov_b, B_vv_b;
    if (config_.eri_method != "exact") {
        B_ov_a = mp2_.B_ia_P_alpha; 
        if (no_b_ > 0 && nv_b_ > 0) B_ov_b = mp2_.B_ia_P_beta;

        auto build_B_fast = [&](const Eigen::MatrixXd& C1, const Eigen::MatrixXd& C2) {
            return integrals::transform_3c_ao_to_mo_block(scf_.L_mat, C1, C2, nbf_);
        };

        B_oo_a = build_B_fast(Cao, Cao);
        B_vv_a = build_B_fast(Cav, Cav);

        if (no_b_ > 0 && nv_b_ > 0) {
            B_oo_b = build_B_fast(Cbo, Cbo);
            B_vv_b = build_B_fast(Cbv, Cbv);
        }
    }

    // =========================================================================
    // 1. TAHAP LADDER (vvvv) - Batched Unique-Pair AO-Driven untuk Mode EXACT
    // =========================================================================
    if (config_.eri_method == "exact") {
        tblis::varray_view<double> t_eri_ao(
            {(len_type)nbf_, (len_type)nbf_, (len_type)nbf_, (len_type)nbf_},
            const_cast<double*>(eri_ao_ptr->data()),
            {1, (stride_type)nbf_, (stride_type)(nbf_*nbf_), (stride_type)(nbf_*nbf_*nbf_)}
        );
        const bool has_beta = (no_b_ > 0 && nv_b_ > 0);
        const int n_pairs_aa = no_a_ * (no_a_ - 1) / 2;
        const int n_pairs_bb = has_beta ? (no_b_ * (no_b_ - 1) / 2) : 0;
        const int n_pairs_ab = has_beta ? (no_a_ * no_b_) : 0;
        const int n_tot_pairs = n_pairs_aa + n_pairs_bb + n_pairs_ab;

        if (n_tot_pairs > 0) {
            Eigen::Tensor<double, 3> T_ao_batch(n_tot_pairs, nbf_, nbf_);
            Eigen::Tensor<double, 3> W_ao_batch(n_tot_pairs, nbf_, nbf_);
            W_ao_batch.setZero();

            int p_idx = 0;
            for (int i = 0; i < no_a_; ++i) {
                for (int j = i + 1; j < no_a_; ++j) {
                    Eigen::MatrixXd Tij_vv(nv_a_, nv_a_);
                    for (int e = 0; e < nv_a_; ++e)
                        for (int f = 0; f < nv_a_; ++f)
                            Tij_vv(e, f) = t2_aa_(i, j, e, f);
                    Eigen::MatrixXd Tij_ao = Cav * Tij_vv * Cav.transpose();
                    for (int mu = 0; mu < nbf_; ++mu)
                        for (int lam = 0; lam < nbf_; ++lam)
                            T_ao_batch(p_idx, mu, lam) = Tij_ao(mu, lam);
                    p_idx++;
                }
            }
            for (int i = 0; i < no_b_; ++i) {
                for (int j = i + 1; j < no_b_; ++j) {
                    Eigen::MatrixXd Tij_vv(nv_b_, nv_b_);
                    for (int e = 0; e < nv_b_; ++e)
                        for (int f = 0; f < nv_b_; ++f)
                            Tij_vv(e, f) = t2_bb_(i, j, e, f);
                    Eigen::MatrixXd Tij_ao = Cbv * Tij_vv * Cbv.transpose();
                    for (int mu = 0; mu < nbf_; ++mu)
                        for (int lam = 0; lam < nbf_; ++lam)
                            T_ao_batch(p_idx, mu, lam) = Tij_ao(mu, lam);
                    p_idx++;
                }
            }
            if (has_beta) {
                for (int i = 0; i < no_a_; ++i) {
                    for (int j = 0; j < no_b_; ++j) {
                        Eigen::MatrixXd Tij_vv(nv_a_, nv_b_);
                        for (int e = 0; e < nv_a_; ++e)
                            for (int f = 0; f < nv_b_; ++f)
                                Tij_vv(e, f) = t2_ab_(i, j, e, f);
                        Eigen::MatrixXd Tij_ao = Cav * Tij_vv * Cbv.transpose();
                        for (int mu = 0; mu < nbf_; ++mu)
                            for (int lam = 0; lam < nbf_; ++lam)
                                T_ao_batch(p_idx, mu, lam) = Tij_ao(mu, lam);
                        p_idx++;
                    }
                }
            }

            TBLIS_VIEW_3D(t_Tao_all, T_ao_batch.data(), n_tot_pairs, nbf_, nbf_);
            TBLIS_VIEW_3D(t_Wao_all, W_ao_batch.data(), n_tot_pairs, nbf_, nbf_);
            tblis::mult<double>(1.0, t_Tao_all, "kpr", t_eri_ao, "pqrs", 0.0, t_Wao_all, "kqs");

            p_idx = 0;
            double sum_aa = 0.0, sum_bb = 0.0, sum_ab = 0.0;
            for (int i = 0; i < no_a_; ++i) {
                for (int j = i + 1; j < no_a_; ++j) {
                    Eigen::MatrixXd Wij_ao(nbf_, nbf_);
                    for (int nu = 0; nu < nbf_; ++nu)
                        for (int sig = 0; sig < nbf_; ++sig)
                            Wij_ao(nu, sig) = W_ao_batch(p_idx, nu, sig);
                    Eigen::MatrixXd M_vv = Cav.transpose() * Wij_ao * Cav;
                    for (int a = 0; a < nv_a_; ++a)
                        for (int b = 0; b < nv_a_; ++b)
                            sum_aa += 2.0 * t2_aa_(i, j, a, b) * (M_vv(a, b) - M_vv(b, a));
                    p_idx++;
                }
            }
            for (int i = 0; i < no_b_; ++i) {
                for (int j = i + 1; j < no_b_; ++j) {
                    Eigen::MatrixXd Wij_ao(nbf_, nbf_);
                    for (int nu = 0; nu < nbf_; ++nu)
                        for (int sig = 0; sig < nbf_; ++sig)
                            Wij_ao(nu, sig) = W_ao_batch(p_idx, nu, sig);
                    Eigen::MatrixXd M_vv = Cbv.transpose() * Wij_ao * Cbv;
                    for (int a = 0; a < nv_b_; ++a)
                        for (int b = 0; b < nv_b_; ++b)
                            sum_bb += 2.0 * t2_bb_(i, j, a, b) * (M_vv(a, b) - M_vv(b, a));
                    p_idx++;
                }
            }
            if (has_beta) {
                for (int i = 0; i < no_a_; ++i) {
                    for (int j = 0; j < no_b_; ++j) {
                        Eigen::MatrixXd Wij_ao(nbf_, nbf_);
                        for (int nu = 0; nu < nbf_; ++nu)
                            for (int sig = 0; sig < nbf_; ++sig)
                                Wij_ao(nu, sig) = W_ao_batch(p_idx, nu, sig);
                        Eigen::MatrixXd M_vv = Cav.transpose() * Wij_ao * Cbv;
                        for (int a = 0; a < nv_a_; ++a)
                            for (int b = 0; b < nv_b_; ++b)
                                sum_ab += t2_ab_(i, j, a, b) * M_vv(a, b);
                        p_idx++;
                    }
                }
            }
            e3_aa += 0.125 * sum_aa;
            e3_bb += 0.125 * sum_bb;
            e3_ab += sum_ab;
        }
    } else {
        int n_pairs_aa = no_a_ * (no_a_ - 1) / 2;
        if (n_pairs_aa > 0) {
            std::vector<Eigen::MatrixXd> T_aa_pairs(n_pairs_aa, Eigen::MatrixXd(nv_a_, nv_a_));
            std::vector<Eigen::MatrixXd> W_aa_pairs(n_pairs_aa, Eigen::MatrixXd::Zero(nv_a_, nv_a_));
            int p_idx = 0;
            for (int i = 0; i < no_a_; ++i) {
                for (int j = i + 1; j < no_a_; ++j) {
                    for (int e = 0; e < nv_a_; ++e)
                        for (int f = 0; f < nv_a_; ++f)
                            T_aa_pairs[p_idx](e, f) = t2_aa_(i, j, e, f);
                    p_idx++;
                }
            }
            contract_vvvv_pairs_df(B_vv_a, nv_a_, B_vv_a, nv_a_, n_aux_, T_aa_pairs, W_aa_pairs);
            double sum_aa = 0.0;
            for (int k = 0; k < n_pairs_aa; ++k) {
                Eigen::MatrixXd W_asym = W_aa_pairs[k] - W_aa_pairs[k].transpose();
                sum_aa += 2.0 * (T_aa_pairs[k].cwiseProduct(W_asym)).sum();
            }
            e3_aa += 0.125 * sum_aa;
        }
        if (no_b_ > 0 && nv_b_ > 0) {
            int n_pairs_bb = no_b_ * (no_b_ - 1) / 2;
            if (n_pairs_bb > 0) {
                std::vector<Eigen::MatrixXd> T_bb_pairs(n_pairs_bb, Eigen::MatrixXd(nv_b_, nv_b_));
                std::vector<Eigen::MatrixXd> W_bb_pairs(n_pairs_bb, Eigen::MatrixXd::Zero(nv_b_, nv_b_));
                int p_idx = 0;
                for (int i = 0; i < no_b_; ++i) {
                    for (int j = i + 1; j < no_b_; ++j) {
                        for (int e = 0; e < nv_b_; ++e)
                            for (int f = 0; f < nv_b_; ++f)
                                T_bb_pairs[p_idx](e, f) = t2_bb_(i, j, e, f);
                        p_idx++;
                    }
                }
                contract_vvvv_pairs_df(B_vv_b, nv_b_, B_vv_b, nv_b_, n_aux_, T_bb_pairs, W_bb_pairs);
                double sum_bb = 0.0;
                for (int k = 0; k < n_pairs_bb; ++k) {
                    Eigen::MatrixXd W_asym = W_bb_pairs[k] - W_bb_pairs[k].transpose();
                    sum_bb += 2.0 * (T_bb_pairs[k].cwiseProduct(W_asym)).sum();
                }
                e3_bb += 0.125 * sum_bb;
            }
            int n_pairs_ab = no_a_ * no_b_;
            std::vector<Eigen::MatrixXd> T_ab_pairs(n_pairs_ab, Eigen::MatrixXd(nv_a_, nv_b_));
            std::vector<Eigen::MatrixXd> W_ab_pairs(n_pairs_ab, Eigen::MatrixXd::Zero(nv_a_, nv_b_));
            for (int i = 0; i < no_a_; ++i) {
                for (int j = 0; j < no_b_; ++j) {
                    int k = i * no_b_ + j;
                    for (int e = 0; e < nv_a_; ++e)
                        for (int f = 0; f < nv_b_; ++f)
                            T_ab_pairs[k](e, f) = t2_ab_(i, j, e, f);
                }
            }
            contract_vvvv_pairs_df(B_vv_a, nv_a_, B_vv_b, nv_b_, n_aux_, T_ab_pairs, W_ab_pairs);
            double sum_ab = 0.0;
            for (int k = 0; k < n_pairs_ab; ++k) {
                sum_ab += (T_ab_pairs[k].cwiseProduct(W_ab_pairs[k])).sum();
            }
            e3_ab += sum_ab;
        }
    }

    // =========================================================================
    // 2 & 3. TAHAP LADDER (oooo) & RING (ovov & oovv)
    // =========================================================================
    if (config_.eri_method == "exact") {
        auto half_oo_a = ERITransformer::transform_half_bra(*eri_ao_ptr, Cao, Cao, nbf_, no_a_, no_a_);
        auto half_ov_a = ERITransformer::transform_half_bra(*eri_ao_ptr, Cao, Cav, nbf_, no_a_, nv_a_);
        Eigen::Tensor<double, 4> half_oo_b;
        if (no_b_ > 0 && nv_b_ > 0) {
            half_oo_b = ERITransformer::transform_half_bra(*eri_ao_ptr, Cbo, Cbo, nbf_, no_b_, no_b_);
        }

        // --- 2. LADDER (oooo) EXACT ---
        if (no_a_ > 1) {
            auto Vaa = ERITransformer::transform_finish_ket(half_oo_a, Cao, Cao, nbf_, no_a_, no_a_, no_a_, no_a_);
            TBLIS_VIEW_4D(t_Vaa, Vaa, no_a_, no_a_, no_a_, no_a_);
            Waa.setZero();
            tblis::mult<double>( 1.0, t_Taa, "mnab", t_Vaa, "minj", 1.0, t_Waa, "ijab");
            tblis::mult<double>(-1.0, t_Taa, "mnab", t_Vaa, "mjni", 1.0, t_Waa, "ijab");
            e3_aa += 0.125 * tensor_dot(t2_aa_, Waa);
        }

        if (no_b_ > 0 && nv_b_ > 0) {
            if (no_b_ > 1) {
                auto Vbb = ERITransformer::transform_finish_ket(half_oo_b, Cbo, Cbo, nbf_, no_b_, no_b_, no_b_, no_b_);
                TBLIS_VIEW_4D(t_Vbb, Vbb, no_b_, no_b_, no_b_, no_b_);
                Wbb.setZero();
                tblis::mult<double>( 1.0, t_Tbb, "mnab", t_Vbb, "minj", 1.0, t_Wbb, "ijab");
                tblis::mult<double>(-1.0, t_Tbb, "mnab", t_Vbb, "mjni", 1.0, t_Wbb, "ijab");
                e3_bb += 0.125 * tensor_dot(t2_bb_, Wbb);
            }

            auto Vab = ERITransformer::transform_finish_ket(half_oo_a, Cbo, Cbo, nbf_, no_a_, no_a_, no_b_, no_b_);
            TBLIS_VIEW_4D(t_Vab, Vab, no_a_, no_a_, no_b_, no_b_);
            Wab.setZero();
            tblis::mult<double>(1.0, t_Tab, "mnab", t_Vab, "minj", 1.0, t_Wab, "ijab");
            e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
        }

        // --- 3. RING (ovov & oovv) EXACT ---
        auto ovov_aa = ERITransformer::transform_finish_ket(half_ov_a, Cao, Cav, nbf_, no_a_, nv_a_, no_a_, nv_a_);
        auto oovv_aa = ERITransformer::transform_finish_ket(half_oo_a, Cav, Cav, nbf_, no_a_, no_a_, nv_a_, nv_a_);
        TBLIS_VIEW_4D(t_ovov_aa, ovov_aa, no_a_, nv_a_, no_a_, nv_a_);
        TBLIS_VIEW_4D(t_oovv_aa, oovv_aa, no_a_, no_a_, nv_a_, nv_a_);

        Waa.setZero();
        if (no_a_ > 1) {
            tblis::mult<double>( 1.0, t_ovov_aa, "iakc", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
            tblis::mult<double>(-1.0, t_oovv_aa, "ikca", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
        }

        if (no_b_ > 0 && nv_b_ > 0) {
            auto ovov_bb    = ERITransformer::transform_custom(*eri_ao_ptr, Cbo, Cbv, Cbo, Cbv, nbf_, no_b_, nv_b_, no_b_, nv_b_);
            auto oovv_bb    = ERITransformer::transform_finish_ket(half_oo_b, Cbv, Cbv, nbf_, no_b_, no_b_, nv_b_, nv_b_);
            auto ovov_ab    = ERITransformer::transform_finish_ket(half_ov_a, Cbo, Cbv, nbf_, no_a_, nv_a_, no_b_, nv_b_);
            auto oovv_ab_ex = ERITransformer::transform_finish_ket(half_oo_a, Cbv, Cbv, nbf_, no_a_, no_a_, nv_b_, nv_b_);
            auto oovv_ba_ex = ERITransformer::transform_finish_ket(half_oo_b, Cav, Cav, nbf_, no_b_, no_b_, nv_a_, nv_a_);
            half_oo_a.resize(0, 0, 0, 0);
            half_oo_b.resize(0, 0, 0, 0);
            half_ov_a.resize(0, 0, 0, 0);

            TBLIS_VIEW_4D(t_ovov_bb, ovov_bb, no_b_, nv_b_, no_b_, nv_b_);
            TBLIS_VIEW_4D(t_oovv_bb, oovv_bb, no_b_, no_b_, nv_b_, nv_b_);
            TBLIS_VIEW_4D(t_ovov_ab, ovov_ab, no_a_, nv_a_, no_b_, nv_b_);
            TBLIS_VIEW_4D(t_oovv_ab_ex, oovv_ab_ex, no_a_, no_a_, nv_b_, nv_b_);
            TBLIS_VIEW_4D(t_oovv_ba_ex, oovv_ba_ex, no_b_, no_b_, nv_a_, nv_a_);

            if (no_a_ > 1) tblis::mult<double>(1.0, t_ovov_ab, "iakc", t_Tab, "jkbc", 1.0, t_Waa, "ijab");
            e3_aa += 1.0 * tensor_dot(t2_aa_, Waa);

            Wbb.setZero();
            if (no_b_ > 1) {
                tblis::mult<double>( 1.0, t_ovov_bb, "iakc", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
                tblis::mult<double>(-1.0, t_oovv_bb, "ikca", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
                tblis::mult<double>( 1.0, t_ovov_ab, "kcia", t_Tab, "kjcb", 1.0, t_Wbb, "ijab");
                e3_bb += 1.0 * tensor_dot(t2_bb_, Wbb);
            }

            Wab.setZero();
            tblis::mult<double>( 1.0, t_ovov_aa,    "iakc", t_Tab,          "kjcb", 1.0, t_Wab, "ijab");
            tblis::mult<double>(-1.0, t_oovv_aa,    "ikca", t_Tab,          "kjcb", 1.0, t_Wab, "ijab");
            tblis::mult<double>( 1.0, t_Tab,        "ikac", t_ovov_bb,    "kcjb", 1.0, t_Wab, "ijab");
            tblis::mult<double>(-1.0, t_Tab,        "ikac", t_oovv_bb,    "kjcb", 1.0, t_Wab, "ijab");
            if (no_a_ > 1) tblis::mult<double>(1.0, t_Taa,     "ikac", t_ovov_ab, "kcjb", 1.0, t_Wab, "ijab");
            if (no_b_ > 1) tblis::mult<double>(1.0, t_ovov_ab, "iakc", t_Tbb,     "kjcb", 1.0, t_Wab, "ijab");
            tblis::mult<double>(-1.0, t_oovv_ab_ex, "ikbc", t_Tab,          "kjac", 1.0, t_Wab, "ijab");
            tblis::mult<double>(-1.0, t_Tab,        "ikcb", t_oovv_ba_ex, "jkac", 1.0, t_Wab, "ijab");
            e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
        } else {
            e3_aa += 1.0 * tensor_dot(t2_aa_, Waa);
        }
    } else {
        // --- 2. LADDER (oooo) DF/CHOLESKY (ZERO-COPY) ---
        if (no_a_ > 1) {
            Eigen::MatrixXd M_oooo_aa = parallel_gemm_abt_mp3(B_oo_a, B_oo_a);
            auto t_Vaa = make_mat_view_4d_mp3(M_oooo_aa, no_a_, no_a_, no_a_, no_a_);
            Waa.setZero();
            tblis::mult<double>( 1.0, t_Taa, "mnab", t_Vaa, "minj", 1.0, t_Waa, "ijab");
            tblis::mult<double>(-1.0, t_Taa, "mnab", t_Vaa, "mjni", 1.0, t_Waa, "ijab");
            e3_aa += 0.125 * tensor_dot(t2_aa_, Waa);
        }

        if (no_b_ > 0 && nv_b_ > 0) {
            if (no_b_ > 1) {
                Eigen::MatrixXd M_oooo_bb = parallel_gemm_abt_mp3(B_oo_b, B_oo_b);
                auto t_Vbb = make_mat_view_4d_mp3(M_oooo_bb, no_b_, no_b_, no_b_, no_b_);
                Wbb.setZero();
                tblis::mult<double>( 1.0, t_Tbb, "mnab", t_Vbb, "minj", 1.0, t_Wbb, "ijab");
                tblis::mult<double>(-1.0, t_Tbb, "mnab", t_Vbb, "mjni", 1.0, t_Wbb, "ijab");
                e3_bb += 0.125 * tensor_dot(t2_bb_, Wbb);
            }

            Eigen::MatrixXd M_oooo_ab = parallel_gemm_abt_mp3(B_oo_a, B_oo_b);
            auto t_Vab = make_mat_view_4d_mp3(M_oooo_ab, no_a_, no_a_, no_b_, no_b_);
            Wab.setZero();
            tblis::mult<double>(1.0, t_Tab, "mnab", t_Vab, "minj", 1.0, t_Wab, "ijab");
            e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
        }

        // --- 3. RING (ovov & oovv) DF/CHOLESKY (ZERO-COPY) ---
        Eigen::MatrixXd M_ovov_aa = parallel_gemm_abt_mp3(B_ov_a, B_ov_a);
        Eigen::MatrixXd M_oovv_aa = parallel_gemm_abt_mp3(B_oo_a, B_vv_a);
        auto t_ovov_aa = make_mat_view_4d_mp3(M_ovov_aa, no_a_, nv_a_, no_a_, nv_a_);
        auto t_oovv_aa = make_mat_view_4d_mp3(M_oovv_aa, no_a_, no_a_, nv_a_, nv_a_);

        Waa.setZero();
        if (no_a_ > 1) {
            tblis::mult<double>( 1.0, t_ovov_aa, "iakc", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
            tblis::mult<double>(-1.0, t_oovv_aa, "ikca", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
        }

        if (no_b_ > 0 && nv_b_ > 0) {
            Eigen::MatrixXd M_ovov_bb    = parallel_gemm_abt_mp3(B_ov_b, B_ov_b);
            Eigen::MatrixXd M_oovv_bb    = parallel_gemm_abt_mp3(B_oo_b, B_vv_b);
            Eigen::MatrixXd M_ovov_ab    = parallel_gemm_abt_mp3(B_ov_a, B_ov_b);
            Eigen::MatrixXd M_oovv_ab_ex = parallel_gemm_abt_mp3(B_oo_a, B_vv_b);
            Eigen::MatrixXd M_oovv_ba_ex = parallel_gemm_abt_mp3(B_oo_b, B_vv_a);

            auto t_ovov_bb    = make_mat_view_4d_mp3(M_ovov_bb,    no_b_, nv_b_, no_b_, nv_b_);
            auto t_oovv_bb    = make_mat_view_4d_mp3(M_oovv_bb,    no_b_, no_b_, nv_b_, nv_b_);
            auto t_ovov_ab    = make_mat_view_4d_mp3(M_ovov_ab,    no_a_, nv_a_, no_b_, nv_b_);
            auto t_oovv_ab_ex = make_mat_view_4d_mp3(M_oovv_ab_ex, no_a_, no_a_, nv_b_, nv_b_);
            auto t_oovv_ba_ex = make_mat_view_4d_mp3(M_oovv_ba_ex, no_b_, no_b_, nv_a_, nv_a_);

            if (no_a_ > 1) tblis::mult<double>(1.0, t_ovov_ab, "iakc", t_Tab, "jkbc", 1.0, t_Waa, "ijab");
            e3_aa += 1.0 * tensor_dot(t2_aa_, Waa);

            Wbb.setZero();
            if (no_b_ > 1) {
                tblis::mult<double>( 1.0, t_ovov_bb, "iakc", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
                tblis::mult<double>(-1.0, t_oovv_bb, "ikca", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
                tblis::mult<double>( 1.0, t_ovov_ab, "kcia", t_Tab, "kjcb", 1.0, t_Wbb, "ijab");
                e3_bb += 1.0 * tensor_dot(t2_bb_, Wbb);
            }

            Wab.setZero();
            tblis::mult<double>( 1.0, t_ovov_aa,    "iakc", t_Tab,          "kjcb", 1.0, t_Wab, "ijab");
            tblis::mult<double>(-1.0, t_oovv_aa,    "ikca", t_Tab,          "kjcb", 1.0, t_Wab, "ijab");
            tblis::mult<double>( 1.0, t_Tab,        "ikac", t_ovov_bb,    "kcjb", 1.0, t_Wab, "ijab");
            tblis::mult<double>(-1.0, t_Tab,        "ikac", t_oovv_bb,    "kjcb", 1.0, t_Wab, "ijab");
            if (no_a_ > 1) tblis::mult<double>(1.0, t_Taa,     "ikac", t_ovov_ab, "kcjb", 1.0, t_Wab, "ijab");
            if (no_b_ > 1) tblis::mult<double>(1.0, t_ovov_ab, "iakc", t_Tbb,     "kjcb", 1.0, t_Wab, "ijab");
            tblis::mult<double>(-1.0, t_oovv_ab_ex, "ikbc", t_Tab,          "kjac", 1.0, t_Wab, "ijab");
            tblis::mult<double>(-1.0, t_Tab,        "ikcb", t_oovv_ba_ex, "jkac", 1.0, t_Wab, "ijab");
            e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
        } else {
            e3_aa += 1.0 * tensor_dot(t2_aa_, Waa);
        }
    }

    if (f_a > 0 || f_b > 0) {
        no_a_ = no_a_saved;
        no_b_ = no_b_saved;
        t2_aa_ = std::move(t2_aa_saved);
        if (t2_bb_saved.size() > 0) t2_bb_ = std::move(t2_bb_saved);
        if (t2_ab_saved.size() > 0) t2_ab_ = std::move(t2_ab_saved);
        if (Bov_a_saved.rows() > 0) mp2_.B_ia_P_alpha = std::move(Bov_a_saved);
        if (Bov_b_saved.rows() > 0) mp2_.B_ia_P_beta = std::move(Bov_b_saved);
    }

    MP3Result res;
    res.e_hf = scf_.energy_total;
    res.e_mp2 = mp2_.energy_mp2_corr;
    res.e3_aa = e3_aa; res.e3_bb = e3_bb; res.e3_ab = e3_ab;
    res.e_mp3 = e3_aa + e3_bb + e3_ab;
    res.e_corr_total = res.e_mp2 + res.e_mp3;
    res.e_total = res.e_hf + res.e_corr_total;

    auto t_end = std::chrono::high_resolution_clock::now();
    if(omp_get_thread_num() == 0) {
        std::cout << "  E_MP3        : " << std::fixed << std::setprecision(8) << res.e_mp3 << " Ha\n";
        std::cout << "  Total Energy : " << res.e_total << " Ha\n";
        std::cout << "  Time         : " << std::chrono::duration< double >(t_end - t_start).count() << " s\n";
    }
    return res;
}
ROMP3::ROMP3(const SCFResult& scf, const MP2Result& mp2,
             const MP2Config& config, std::shared_ptr<IntegralEngine> ints)
    : UMP3(scf, mp2, config, ints)
{
    // Gunakan orbital dan energi semikanonik yang sudah dihitung oleh ROMP2
    if (mp2_.C_alpha.rows() == nbf_ && mp2_.C_beta.rows() == nbf_) {
        scf_.C_alpha = mp2_.C_alpha;
        scf_.C_beta  = mp2_.C_beta;
    }
    if (mp2_.orbital_energies_alpha.size() == nbf_) {
        scf_.orbital_energies_alpha = mp2_.orbital_energies_alpha;
    }
    if (mp2_.orbital_energies_beta.size() == nbf_) {
        scf_.orbital_energies_beta = mp2_.orbital_energies_beta;
    }
    if (config_.eri_method != "exact" && no_a_ == no_b_ && mp2_.B_ia_P_beta.rows() == 0) {
        mp2_.B_ia_P_beta = mp2_.B_ia_P_alpha;
    }
}

MP3Result ROMP3::compute() {
    if (omp_get_thread_num() == 0 && config_.print_level > 0) {
        std::cout << "\n=== ROMP3 (Single-Shot Semicanonical TBLIS) ===\n";
    }
    MP3Result res = UMP3::compute();
    res.converged  = true;
    res.iterations = 1;
    res.C_alpha    = scf_.C_alpha;
    res.C_beta     = scf_.C_beta;
    res.n_occ_alpha = no_a_;
    res.n_occ_beta  = no_b_;
    return res;
}



double OMP3::get_correlation_energy() const {
    return e_ss_ + e_os_ + e_mp3_tot_;
}

double OMP3::execute_micro_iterations() {
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    scf_.C_alpha = C_a_current_;
    scf_.C_beta  = C_b_current_;
    scf_.P_alpha.noalias() = scf_.C_alpha.leftCols(na_) * scf_.C_alpha.leftCols(na_).transpose();
    scf_.P_beta.noalias()  = scf_.C_beta.leftCols(nb_)  * scf_.C_beta.leftCols(nb_).transpose();

    pseudocanonicalize();

    C_a_current_ = scf_.C_alpha;
    C_b_current_ = scf_.C_beta;

    if (config_.eri_method == "cholesky" || config_.eri_method == "df") {
        scf_.irreps_alpha.assign(nbf_, 0);
        if (!is_restricted && nb_ > 0) {
            scf_.irreps_beta.assign(nbf_, 0);
        }
        transform_3center_mo_cholesky();
        compute_t2_and_energy_cholesky();
    } else {
        transform_integrals();
        compute_t2_amplitudes();
        compute_mp2_energy();
    }

    compute_mp3_correction(); 
    
    // [INJECTED_HPC_V5_OMP3_HEAVY_RAM]
  L2_aa_ = std::move(t2_3rd_aa_);
  if (!is_restricted && nb_ > 0 && vb_ > 0) {
    L2_bb_ = std::move(t2_3rd_bb_);
    L2_ab_ = std::move(t2_3rd_ab_);
  }
    
    // Hanya dieksekusi 1x per iterasi makro setelah L2_aa_, L2_bb_, L2_ab_ diperbarui
    build_opdm_alpha();
    if (!is_restricted && nb_ > 0) {
        build_opdm_beta();
    } else if (is_restricted && nb_ > 0) {
        G_oo_beta_ = G_oo_alpha_;
        G_vv_beta_ = G_vv_alpha_;
    }

    return get_correlation_energy();
}

void OMP3::compute_mp3_correction() {
    if (na_ == 0 || va_ == 0) { e_mp3_tot_ = 0.0; return; }

    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    const Eigen::MatrixXd& Cao = scf_.C_alpha.leftCols(na_); 
    const Eigen::MatrixXd& Cav = scf_.C_alpha.rightCols(va_);
    const auto& ea = scf_.orbital_energies_alpha;
    
    auto* t2_aa_dense = t2_aa_.get_block(0,0,0,0);
    if(!t2_aa_dense) throw std::runtime_error("OMP3 missing T2 dense block.");
    Eigen::Tensor< double, 4 > T2_aa_ijab(na_, na_, va_, va_);
    
    #pragma omp parallel for collapse(4) schedule(static)
    for(int i = 0; i < na_; ++i) {
        for(int j = 0; j < na_; ++j) {
            for(int a = 0; a < va_; ++a) {
                for(int b = 0; b < va_; ++b) {
                    T2_aa_ijab(i,j,a,b) = (*t2_aa_dense)(i,a,j,b);
                }
            }
        }
    }

    // =========================================================================
    // IMPLEMENTASI INTEGRAL EKSAK (CACHE AO INTEGRAL)
    // =========================================================================
    if (config_.eri_method == "exact") {
        const auto& eri_ao_cached_local = integrals_->compute_eri();
        tblis::varray_view<double> t_eri_ao(
            {(len_type)nbf_, (len_type)nbf_, (len_type)nbf_, (len_type)nbf_},
            const_cast<double*>(eri_ao_cached_local.data()),
            {1, (stride_type)nbf_, (stride_type)(nbf_*nbf_), (stride_type)(nbf_*nbf_*nbf_)}
        );

        if (is_restricted) {
            t2_3rd_aa_ = Eigen::Tensor< double, 4 >(na_, na_, va_, va_);
            Eigen::Tensor< double, 4 > W(na_, na_, va_, va_); W.setZero();
            
            TBLIS_VIEW_4D(t_T, T2_aa_ijab, na_, na_, va_, va_);
            TBLIS_VIEW_4D(t_W, W, na_, na_, va_, va_);

            // AO-Driven Ladder hanya untuk pasangan unik i <= j
            int n_pairs_sym = na_ * (na_ + 1) / 2;
            Eigen::Tensor<double, 3> T_ao(n_pairs_sym, nbf_, nbf_);
            Eigen::Tensor<double, 3> W_ao(n_pairs_sym, nbf_, nbf_);
            W_ao.setZero();

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < na_; ++i) {
                for (int j = i; j < na_; ++j) {
                    int k = i * na_ - (i * (i - 1)) / 2 + (j - i);
                    Eigen::MatrixXd Tij_vv(va_, va_);
                    for (int e = 0; e < va_; ++e)
                        for (int f = 0; f < va_; ++f)
                            Tij_vv(e, f) = T2_aa_ijab(i, j, e, f);
                    Eigen::MatrixXd Tij_ao = Cav * Tij_vv * Cav.transpose();
                    for (int mu = 0; mu < nbf_; ++mu)
                        for (int lam = 0; lam < nbf_; ++lam)
                            T_ao(k, mu, lam) = Tij_ao(mu, lam);
                }
            }

            TBLIS_VIEW_3D(t_Tao, T_ao.data(), n_pairs_sym, nbf_, nbf_);
            TBLIS_VIEW_3D(t_Wao, W_ao.data(), n_pairs_sym, nbf_, nbf_);
            tblis::mult<double>(1.0, t_Tao, "kpr", t_eri_ao, "pqrs", 0.0, t_Wao, "kqs");

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < na_; ++i) {
                for (int j = i; j < na_; ++j) {
                    int k = i * na_ - (i * (i - 1)) / 2 + (j - i);
                    Eigen::MatrixXd Wij_ao(nbf_, nbf_);
                    for (int nu = 0; nu < nbf_; ++nu)
                        for (int sig = 0; sig < nbf_; ++sig)
                            Wij_ao(nu, sig) = W_ao(k, nu, sig);
                    Eigen::MatrixXd Wij_vv = Cav.transpose() * Wij_ao * Cav;
                    for (int a = 0; a < va_; ++a) {
                        for (int b = 0; b < va_; ++b) {
                            W(i, j, a, b) += Wij_vv(a, b);
                            if (i != j) W(j, i, b, a) += Wij_vv(a, b);
                        }
                    }
                }
            }

            auto half_oo_a = integrals::ERITransformer::transform_half_bra(eri_ao_cached_local, Cao, Cao, nbf_, na_, na_);
            Waa_ring_ = integrals::ERITransformer::transform_finish_ket(half_oo_a, Cao, Cao, nbf_, na_, na_, na_, na_);
            TBLIS_VIEW_4D(t_Voooo, Waa_ring_, na_, na_, na_, na_);
            tblis::mult< double >(1.0, t_T, "mnab", t_Voooo, "minj", 1.0, t_W, "ijab");

            Eigen::Tensor<double, 4>* g_blk_mp3 = g_aa_.get_block(0, 0, 0, 0);
            TBLIS_VIEW_4D(t_Vovov, (*g_blk_mp3), na_, va_, na_, va_);
            
            auto V_oovv = integrals::ERITransformer::transform_finish_ket(half_oo_a, Cav, Cav, nbf_, na_, na_, va_, va_);
            half_oo_a.resize(0, 0, 0, 0);
            TBLIS_VIEW_4D(t_Voovv, V_oovv, na_, na_, va_, va_);

            tblis::mult< double >(2.0,  t_Vovov, "iakc", t_T, "kjcb", 1.0, t_W, "ijab");
            tblis::mult< double >(-1.0, t_Voovv, "ikac", t_T, "kjcb", 1.0, t_W, "ijab");
            tblis::mult< double >(2.0,  t_T, "ikac", t_Vovov, "kcjb", 1.0, t_W, "ijab");
            tblis::mult< double >(-1.0, t_T, "ikac", t_Voovv, "kjcb", 1.0, t_W, "ijab");
            tblis::mult< double >(-1.0, t_T, "ikca", t_Vovov, "kcjb", 1.0, t_W, "ijab"); 
            tblis::mult< double >(-1.0, t_Vovov, "iakc", t_T, "kjbc", 1.0, t_W, "ijab");
            tblis::mult< double >(-1.0, t_Voovv, "ikbc", t_T, "kjac", 1.0, t_W, "ijab");
            tblis::mult< double >(-1.0, t_T, "ikcb", t_Voovv, "jkac", 1.0, t_W, "ijab"); 

            double e3_aa = 0.0;
            #pragma omp parallel for collapse(4) reduction(+:e3_aa) schedule(static)
            for (int i = 0; i < na_; ++i) {
                for (int j = 0; j < na_; ++j) {
                    for (int a = 0; a < va_; ++a) {
                        for (int b = 0; b < va_; ++b) {
                            if (i < n_frozen_ || j < n_frozen_) { t2_3rd_aa_(i, j, a, b) = 0.0; continue; }
              double D = ea(i) + ea(j) - ea(na_ + a) - ea(na_ + b);
              t2_3rd_aa_(i, j, a, b) = W(i, j, a, b) / D; 
                            e3_aa += W(i, j, a, b) * (2.0 * T2_aa_ijab(i, j, a, b) - T2_aa_ijab(i, j, b, a));
                        }
                    }
                }
            }
            e_mp3_tot_ = e3_aa;
            return;
        } else {
            const Eigen::MatrixXd& Cbo = scf_.C_beta.leftCols(nb_);
            const Eigen::MatrixXd& Cbv = scf_.C_beta.rightCols(vb_);
            const auto& eb = scf_.orbital_energies_beta;

            t2_3rd_aa_ = Eigen::Tensor< double, 4 >(na_, na_, va_, va_); t2_3rd_aa_.setZero();
            if (nb_ > 0 && vb_ > 0) {
                t2_3rd_bb_ = Eigen::Tensor< double, 4 >(nb_, nb_, vb_, vb_); t2_3rd_bb_.setZero();
                t2_3rd_ab_ = Eigen::Tensor< double, 4 >(na_, nb_, va_, vb_); t2_3rd_ab_.setZero();
            }

            Eigen::Tensor< double, 4 > Waa_ladder(na_, na_, va_, va_); Waa_ladder.setZero();
            Eigen::Tensor< double, 4 > Waa_ring(na_, na_, va_, va_);   Waa_ring.setZero();
            Eigen::Tensor< double, 4 > Wbb_ladder(nb_, nb_, vb_, vb_); if (nb_ > 0) Wbb_ladder.setZero();
            Eigen::Tensor< double, 4 > Wbb_ring(nb_, nb_, vb_, vb_);   if (nb_ > 0) Wbb_ring.setZero();
            Eigen::Tensor< double, 4 > Wab_ladder(na_, nb_, va_, vb_); if (nb_ > 0) Wab_ladder.setZero();
            Eigen::Tensor< double, 4 > Wab_ring(na_, nb_, va_, vb_);   if (nb_ > 0) Wab_ring.setZero();

            TBLIS_VIEW_4D(t_Taa, T2_aa_ijab, na_, na_, va_, va_);
            TBLIS_VIEW_4D(t_Waa_ladder, Waa_ladder, na_, na_, va_, va_);
            TBLIS_VIEW_4D(t_Waa_ring, Waa_ring, na_, na_, va_, va_);

            Eigen::Tensor< double, 4 >* t2_bb_dense_ptr = t2_bb_.get_block(0, 0, 0, 0);
            Eigen::Tensor< double, 4 > dummy_bb_local(nb_, nb_, vb_, vb_);
            if (!t2_bb_dense_ptr) {
                dummy_bb_local.setZero();
                t2_bb_dense_ptr = &dummy_bb_local;
            }
            
            Eigen::Tensor< double, 4 >* t2_ab_dense = t2_ab_.get_block(0, 0, 0, 0);
            Eigen::Tensor< double, 4 > dummy_ab(na_, nb_, va_, vb_);
            if (!t2_ab_dense) {
                dummy_ab.setZero();
                t2_ab_dense = &dummy_ab;
            }

            TBLIS_VIEW_4D(t_Tbb, (*t2_bb_dense_ptr), nb_, nb_, vb_, vb_);
            TBLIS_VIEW_4D(t_Wbb_ladder, Wbb_ladder, nb_, nb_, vb_, vb_);
            TBLIS_VIEW_4D(t_Wbb_ring, Wbb_ring, nb_, nb_, vb_, vb_);
            
            TBLIS_VIEW_4D(t_Tab, (*t2_ab_dense), na_, nb_, va_, vb_);
            TBLIS_VIEW_4D(t_Wab_ladder, Wab_ladder, na_, nb_, va_, vb_);
            TBLIS_VIEW_4D(t_Wab_ring, Wab_ring, na_, nb_, va_, vb_);

            // 1. Batched Unique-Pair AO-Driven VVVV Ladder (AA + BB + AB dalam 1x TBLIS mult, 0 MB V^4!)
            const bool has_beta = (nb_ > 0 && vb_ > 0);
            const int n_pairs_aa = na_ * (na_ - 1) / 2;
            const int n_pairs_bb = has_beta ? (nb_ * (nb_ - 1) / 2) : 0;
            const int n_pairs_ab = has_beta ? (na_ * nb_) : 0;
            const int n_tot_pairs = n_pairs_aa + n_pairs_bb + n_pairs_ab;

            if (n_tot_pairs > 0) {
                Eigen::Tensor<double, 3> T_ao_batch(n_tot_pairs, nbf_, nbf_);
                Eigen::Tensor<double, 3> W_ao_batch(n_tot_pairs, nbf_, nbf_);
                W_ao_batch.setZero();

                int p_idx = 0;
                for (int i = 0; i < na_; ++i) {
                    for (int j = i + 1; j < na_; ++j) {
                        Eigen::MatrixXd Tij_vv(va_, va_);
                        for (int e = 0; e < va_; ++e)
                            for (int f = 0; f < va_; ++f)
                                Tij_vv(e, f) = T2_aa_ijab(i, j, e, f);
                        Eigen::MatrixXd Tij_ao = Cav * Tij_vv * Cav.transpose();
                        for (int mu = 0; mu < nbf_; ++mu)
                            for (int lam = 0; lam < nbf_; ++lam)
                                T_ao_batch(p_idx, mu, lam) = Tij_ao(mu, lam);
                        p_idx++;
                    }
                }
                for (int i = 0; i < nb_; ++i) {
                    for (int j = i + 1; j < nb_; ++j) {
                        Eigen::MatrixXd Tij_vv(vb_, vb_);
                        for (int e = 0; e < vb_; ++e)
                            for (int f = 0; f < vb_; ++f)
                                Tij_vv(e, f) = (*t2_bb_dense_ptr)(i, j, e, f);
                        Eigen::MatrixXd Tij_ao = Cbv * Tij_vv * Cbv.transpose();
                        for (int mu = 0; mu < nbf_; ++mu)
                            for (int lam = 0; lam < nbf_; ++lam)
                                T_ao_batch(p_idx, mu, lam) = Tij_ao(mu, lam);
                        p_idx++;
                    }
                }
                if (has_beta) {
                    for (int i = 0; i < na_; ++i) {
                        for (int j = 0; j < nb_; ++j) {
                            Eigen::MatrixXd Tij_vv(va_, vb_);
                            for (int e = 0; e < va_; ++e)
                                for (int f = 0; f < vb_; ++f)
                                    Tij_vv(e, f) = (*t2_ab_dense)(i, j, e, f);
                            Eigen::MatrixXd Tij_ao = Cav * Tij_vv * Cbv.transpose();
                            for (int mu = 0; mu < nbf_; ++mu)
                                for (int lam = 0; lam < nbf_; ++lam)
                                    T_ao_batch(p_idx, mu, lam) = Tij_ao(mu, lam);
                            p_idx++;
                        }
                    }
                }

                TBLIS_VIEW_3D(t_Tao_all, T_ao_batch.data(), n_tot_pairs, nbf_, nbf_);
                TBLIS_VIEW_3D(t_Wao_all, W_ao_batch.data(), n_tot_pairs, nbf_, nbf_);
                tblis::mult<double>(1.0, t_Tao_all, "kpr", t_eri_ao, "pqrs", 0.0, t_Wao_all, "kqs");

                p_idx = 0;
                for (int i = 0; i < na_; ++i) {
                    for (int j = i + 1; j < na_; ++j) {
                        Eigen::MatrixXd Wij_ao(nbf_, nbf_);
                        for (int nu = 0; nu < nbf_; ++nu)
                            for (int sig = 0; sig < nbf_; ++sig)
                                Wij_ao(nu, sig) = W_ao_batch(p_idx, nu, sig);
                        Eigen::MatrixXd M_vv = Cav.transpose() * Wij_ao * Cav;
                        for (int a = 0; a < va_; ++a) {
                            for (int b = 0; b < va_; ++b) {
                                double val = 0.5 * (M_vv(a, b) - M_vv(b, a));
                                Waa_ladder(i, j, a, b) += val;
                                Waa_ladder(j, i, a, b) -= val;
                            }
                        }
                        p_idx++;
                    }
                }
                for (int i = 0; i < nb_; ++i) {
                    for (int j = i + 1; j < nb_; ++j) {
                        Eigen::MatrixXd Wij_ao(nbf_, nbf_);
                        for (int nu = 0; nu < nbf_; ++nu)
                            for (int sig = 0; sig < nbf_; ++sig)
                                Wij_ao(nu, sig) = W_ao_batch(p_idx, nu, sig);
                        Eigen::MatrixXd M_vv = Cbv.transpose() * Wij_ao * Cbv;
                        for (int a = 0; a < vb_; ++a) {
                            for (int b = 0; b < vb_; ++b) {
                                double val = 0.5 * (M_vv(a, b) - M_vv(b, a));
                                Wbb_ladder(i, j, a, b) += val;
                                Wbb_ladder(j, i, a, b) -= val;
                            }
                        }
                        p_idx++;
                    }
                }
                if (has_beta) {
                    for (int i = 0; i < na_; ++i) {
                        for (int j = 0; j < nb_; ++j) {
                            Eigen::MatrixXd Wij_ao(nbf_, nbf_);
                            for (int nu = 0; nu < nbf_; ++nu)
                                for (int sig = 0; sig < nbf_; ++sig)
                                    Wij_ao(nu, sig) = W_ao_batch(p_idx, nu, sig);
                            Eigen::MatrixXd M_vv = Cav.transpose() * Wij_ao * Cbv;
                            for (int a = 0; a < va_; ++a)
                                for (int b = 0; b < vb_; ++b)
                                    Wab_ladder(i, j, a, b) += M_vv(a, b);
                            p_idx++;
                        }
                    }
                }
            }

            // 2. Shared Half-Bra (Cao, Cao) untuk V_oooo_aa, V_oooo_ab, V_oovv_aa, V_oovv_ab_ex
            auto half_oo_a = integrals::ERITransformer::transform_half_bra(eri_ao_cached_local, Cao, Cao, nbf_, na_, na_);
            auto V_oooo_aa = integrals::ERITransformer::transform_finish_ket(half_oo_a, Cao, Cao, nbf_, na_, na_, na_, na_);
            TBLIS_VIEW_4D(t_Voooo_aa, V_oooo_aa, na_, na_, na_, na_);
            tblis::mult< double >(0.5, t_Taa, "mnab", t_Voooo_aa, "minj", 1.0, t_Waa_ladder, "ijab");
            tblis::mult< double >(-0.5, t_Taa, "mnab", t_Voooo_aa, "mjni", 1.0, t_Waa_ladder, "ijab");
            V_oooo_aa.resize(0, 0, 0, 0);

            auto V_oovv_aa = integrals::ERITransformer::transform_finish_ket(half_oo_a, Cav, Cav, nbf_, na_, na_, va_, va_);
            Eigen::Tensor< double, 4 >* g_aa_blk = g_aa_.get_block(0, 0, 0, 0);
            TBLIS_VIEW_4D(t_Vovov_aa, (*g_aa_blk), na_, va_, na_, va_);
            TBLIS_VIEW_4D(t_Voovv_aa, V_oovv_aa, na_, na_, va_, va_);

            tblis::mult< double >(1.0, t_Vovov_aa, "iakc", t_Taa, "kjcb", 1.0, t_Waa_ring, "ijab");
            tblis::mult< double >(-1.0, t_Voovv_aa, "ikac", t_Taa, "kjcb", 1.0, t_Waa_ring, "ijab");

            if (has_beta) {
                auto V_oooo_ab = integrals::ERITransformer::transform_finish_ket(half_oo_a, Cbo, Cbo, nbf_, na_, na_, nb_, nb_);
                TBLIS_VIEW_4D(t_Voooo_ab, V_oooo_ab, na_, na_, nb_, nb_);
                tblis::mult< double >(1.0, t_Tab, "mnab", t_Voooo_ab, "minj", 1.0, t_Wab_ladder, "ijab");
                V_oooo_ab.resize(0, 0, 0, 0);

                auto V_oovv_ab_ex = integrals::ERITransformer::transform_finish_ket(half_oo_a, Cbv, Cbv, nbf_, na_, na_, vb_, vb_);
                half_oo_a.resize(0, 0, 0, 0);

                // Shared Half-Bra (Cbo, Cbo) untuk V_oooo_bb, V_oovv_bb, V_oovv_ba_ex
                auto half_oo_b = integrals::ERITransformer::transform_half_bra(eri_ao_cached_local, Cbo, Cbo, nbf_, nb_, nb_);
                auto V_oooo_bb = integrals::ERITransformer::transform_finish_ket(half_oo_b, Cbo, Cbo, nbf_, nb_, nb_, nb_, nb_);
                TBLIS_VIEW_4D(t_Voooo_bb, V_oooo_bb, nb_, nb_, nb_, nb_);
                tblis::mult< double >(0.5, t_Tbb, "mnab", t_Voooo_bb, "minj", 1.0, t_Wbb_ladder, "ijab");
                tblis::mult< double >(-0.5, t_Tbb, "mnab", t_Voooo_bb, "mjni", 1.0, t_Wbb_ladder, "ijab");
                V_oooo_bb.resize(0, 0, 0, 0);

                auto V_oovv_bb    = integrals::ERITransformer::transform_finish_ket(half_oo_b, Cbv, Cbv, nbf_, nb_, nb_, vb_, vb_);
                auto V_oovv_ba_ex = integrals::ERITransformer::transform_finish_ket(half_oo_b, Cav, Cav, nbf_, nb_, nb_, va_, va_);
                half_oo_b.resize(0, 0, 0, 0);

                Eigen::Tensor< double, 4 >* g_bb_blk = g_bb_.get_block(0, 0, 0, 0);
                Eigen::Tensor< double, 4 >* g_ab_blk = g_ab_.get_block(0, 0, 0, 0);

                TBLIS_VIEW_4D(t_Vovov_bb_vw, (*g_bb_blk), nb_, vb_, nb_, vb_);
                TBLIS_VIEW_4D(t_Voovv_bb_vw, V_oovv_bb, nb_, nb_, vb_, vb_);
                TBLIS_VIEW_4D(t_Vovov_ab_vw, (*g_ab_blk), na_, va_, nb_, vb_);
                TBLIS_VIEW_4D(t_Voovv_ab_ex_vw, V_oovv_ab_ex, na_, na_, vb_, vb_);
                TBLIS_VIEW_4D(t_Voovv_ba_ex_vw, V_oovv_ba_ex, nb_, nb_, va_, va_);

                tblis::mult< double >(1.0, t_Vovov_ab_vw, "iakc", t_Tab, "jkbc", 1.0, t_Waa_ring, "ijab");
                tblis::mult< double >(1.0, t_Vovov_bb_vw, "iakc", t_Tbb, "kjcb", 1.0, t_Wbb_ring, "ijab");
                tblis::mult< double >(-1.0, t_Voovv_bb_vw, "ikac", t_Tbb, "kjcb", 1.0, t_Wbb_ring, "ijab");
                tblis::mult< double >(1.0, t_Vovov_ab_vw, "kcia", t_Tab, "kjcb", 1.0, t_Wbb_ring, "ijab"); 
                tblis::mult< double >(1.0,  t_Vovov_aa, "iakc", t_Tab, "kjcb", 1.0, t_Wab_ring, "ijab");
                tblis::mult< double >(-1.0, t_Voovv_aa, "ikac", t_Tab, "kjcb", 1.0, t_Wab_ring, "ijab");
                tblis::mult< double >(1.0,  t_Tab, "ikac", t_Vovov_bb_vw, "kcjb", 1.0, t_Wab_ring, "ijab");
                tblis::mult< double >(-1.0, t_Tab, "ikac", t_Voovv_bb_vw, "kjcb", 1.0, t_Wab_ring, "ijab");
                tblis::mult< double >(1.0,  t_Taa, "ikac", t_Vovov_ab_vw, "kcjb", 1.0, t_Wab_ring, "ijab");
                tblis::mult< double >(1.0,  t_Vovov_ab_vw, "iakc", t_Tbb, "kjcb", 1.0, t_Wab_ring, "ijab");
                tblis::mult< double >(-1.0, t_Voovv_ab_ex_vw, "ikbc", t_Tab, "kjac", 1.0, t_Wab_ring, "ijab");
                tblis::mult< double >(-1.0, t_Tab, "ikcb", t_Voovv_ba_ex_vw, "jkac", 1.0, t_Wab_ring, "ijab"); 
            }

            double e3_aa = 0.0, e3_bb = 0.0, e3_ab = 0.0;
            
            #pragma omp parallel for collapse(4) reduction(+:e3_aa) schedule(static)
            for (int i = 0; i < na_; ++i) {
                for (int j = 0; j < na_; ++j) {
                    for (int a = 0; a < va_; ++a) {
                        for (int b = 0; b < va_; ++b) {
                            double r_asym = Waa_ring(i, j, a, b) - Waa_ring(j, i, a, b) - Waa_ring(i, j, b, a) + Waa_ring(j, i, b, a);
                            double w_tot = Waa_ladder(i, j, a, b) + r_asym;
                            double D = ea(i) + ea(j) - ea(na_ + a) - ea(na_ + b);
                            t2_3rd_aa_(i, j, a, b) = w_tot / D;
                            e3_aa += 0.25 * T2_aa_ijab(i, j, a, b) * w_tot;
                        }
                    }
                }
            }

            if (has_beta) {
                #pragma omp parallel for collapse(4) reduction(+:e3_bb) schedule(static)
                for (int i = 0; i < nb_; ++i) {
                    for (int j = 0; j < nb_; ++j) {
                        for (int a = 0; a < vb_; ++a) {
                            for (int b = 0; b < vb_; ++b) {
                                double r_asym = Wbb_ring(i, j, a, b) - Wbb_ring(j, i, a, b) - Wbb_ring(i, j, b, a) + Wbb_ring(j, i, b, a);
                                double w_tot = Wbb_ladder(i, j, a, b) + r_asym;
                                double D = eb(i) + eb(j) - eb(nb_ + a) - eb(nb_ + b);
                                t2_3rd_bb_(i, j, a, b) = w_tot / D;
                                e3_bb += 0.25 * (*t2_bb_dense_ptr)(i, j, a, b) * w_tot;
                            }
                        }
                    }
                }

                #pragma omp parallel for collapse(4) reduction(+:e3_ab) schedule(static)
                for (int i = 0; i < na_; ++i) {
                    for (int j = 0; j < nb_; ++j) {
                        for (int a = 0; a < va_; ++a) {
                            for (int b = 0; b < vb_; ++b) {
                                double w_tot = Wab_ladder(i, j, a, b) + Wab_ring(i, j, a, b); 
                                double D = ea(i) + eb(j) - ea(na_ + a) - eb(nb_ + b);
                                t2_3rd_ab_(i, j, a, b) = w_tot / D;
                                e3_ab += 1.0 * (*t2_ab_dense)(i, j, a, b) * w_tot;
                            }
                        }
                    }
                }
            }
            
            e_mp3_tot_ = e3_aa + e3_bb + e3_ab;
            return;
        }
    }

    // =========================================================================
    // IMPLEMENTASI CHOLESKY & DENSITY FITTING (ZERO-COPY & PARALLEL GEMM)
    // =========================================================================
    int n_aux = scf_.L_mat.cols();
    const int nf_a = std::clamp(n_frozen_, 0, na_);
    const int nf_b = std::clamp(n_frozen_, 0, nb_);
    if (nf_a > 0 || nf_b > 0) {
        auto slice_boo = [&](const Eigen::MatrixXd& Boo, int n_full, int nf) {
        const int n_act = n_full - nf;
        Eigen::MatrixXd out(static_cast<Eigen::Index>(n_act) * n_act, n_aux);
        #pragma omp parallel for schedule(static)
        for (int P = 0; P < n_aux; ++P) {
            const double* s = Boo.col(P).data();
            double* d = out.col(P).data();
            for (int j = 0; j < n_act; ++j)
            for (int i = 0; i < n_act; ++i)
                d[i + j * n_act] = s[(nf + i) + (nf + j) * n_full];
        }
        return out;
        };

        if (is_restricted) {
        const int na_act = na_ - nf_a;
        t2_3rd_aa_ = Eigen::Tensor<double, 4>(na_, na_, va_, va_);
        t2_3rd_aa_.setZero();
        if (na_act <= 0 || va_ <= 0) { e_mp3_tot_ = 0.0; return; }

        Eigen::Tensor<double, 4> T_act(na_act, na_act, va_, va_);
        #pragma omp parallel for collapse(4) schedule(static)
        for (int i = 0; i < na_act; ++i)
            for (int j = 0; j < na_act; ++j)
            for (int a = 0; a < va_; ++a)
                for (int b = 0; b < va_; ++b)
                T_act(i, j, a, b) = T2_aa_ijab(nf_a + i, nf_a + j, a, b);

        Eigen::Tensor<double, 4> W(na_act, na_act, va_, va_);
        W.setZero();
        TBLIS_VIEW_4D(t_T, T_act, na_act, na_act, va_, va_);
        TBLIS_VIEW_4D(t_W, W, na_act, na_act, va_, va_);

        Eigen::MatrixXd B_ij_act = slice_boo(B_oo_P_alpha_, na_, nf_a);
        Eigen::MatrixXd B_ia_act = B_ia_P_alpha_.bottomRows(static_cast<Eigen::Index>(na_act) * va_);
        const Eigen::MatrixXd& B_ab_a = B_vv_P_alpha_;

        Eigen::MatrixXd M_oooo = parallel_gemm_abt_mp3(B_ij_act, B_ij_act);
        Eigen::MatrixXd M_ovov = parallel_gemm_abt_mp3(B_ia_act, B_ia_act);
        Eigen::MatrixXd M_oovv = parallel_gemm_abt_mp3(B_ij_act, B_ab_a);

        auto t_Voooo = make_mat_view_4d_mp3(M_oooo, na_act, na_act, na_act, na_act);
        auto t_Vovov = make_mat_view_4d_mp3(M_ovov, na_act, va_, na_act, va_);
        auto t_Voovv = make_mat_view_4d_mp3(M_oovv, na_act, na_act, va_, va_);

        tblis::mult<double>( 1.0, t_T, "mnab", t_Voooo, "minj", 1.0, t_W, "ijab");
        tblis::mult<double>( 2.0, t_Vovov, "iakc", t_T, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikac", t_T, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>( 2.0, t_T, "ikac", t_Vovov, "kcjb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T, "ikac", t_Voovv, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T, "ikca", t_Vovov, "kcjb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Vovov, "iakc", t_T, "kjbc", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikbc", t_T, "kjac", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T, "ikcb", t_Voovv, "jkac", 1.0, t_W, "ijab");

        const int n_pairs_sym = na_act * (na_act + 1) / 2;
        std::vector<Eigen::MatrixXd> T_pairs(n_pairs_sym, Eigen::MatrixXd(va_, va_));
        std::vector<Eigen::MatrixXd> W_pairs(n_pairs_sym, Eigen::MatrixXd::Zero(va_, va_));
        int p_idx = 0;
        for (int i = 0; i < na_act; ++i)
            for (int j = i; j < na_act; ++j) {
            for (int e = 0; e < va_; ++e)
                for (int f = 0; f < va_; ++f)
                T_pairs[p_idx](e, f) = T_act(i, j, e, f);
            ++p_idx;
            }
        contract_vvvv_pairs_df(B_ab_a, va_, B_ab_a, va_, n_aux, T_pairs, W_pairs);
        p_idx = 0;
        for (int i = 0; i < na_act; ++i)
            for (int j = i; j < na_act; ++j) {
            for (int a = 0; a < va_; ++a)
                for (int b = 0; b < va_; ++b) {
                W(i, j, a, b) += W_pairs[p_idx](a, b);
                if (i != j) W(j, i, b, a) += W_pairs[p_idx](a, b);
                }
            ++p_idx;
            }

        double e3_aa = 0.0;
        #pragma omp parallel for collapse(4) reduction(+:e3_aa) schedule(static)
        for (int i = 0; i < na_act; ++i)
            for (int j = 0; j < na_act; ++j)
            for (int a = 0; a < va_; ++a)
                for (int b = 0; b < va_; ++b) {
                const double D = ea(nf_a + i) + ea(nf_a + j) - ea(na_ + a) - ea(na_ + b);
                t2_3rd_aa_(nf_a + i, nf_a + j, a, b) = W(i, j, a, b) / D;
                e3_aa += W(i, j, a, b) * (2.0 * T_act(i, j, a, b) - T_act(i, j, b, a));
                }
        e_mp3_tot_ = e3_aa;
        return;
        } else {
        // Unrestricted DF/Cholesky OMP3 Frozen Core: zero out frozen rows di akhir secara aman
        // (tetap mempercepat vvvv via skip pasangan < nf)
        const int na_act = na_ - nf_a;
        const int nb_act = nb_ - nf_b;
        t2_3rd_aa_ = Eigen::Tensor<double, 4>(na_, na_, va_, va_); t2_3rd_aa_.setZero();
        if (nb_ > 0 && vb_ > 0) {
            t2_3rd_bb_ = Eigen::Tensor<double, 4>(nb_, nb_, vb_, vb_); t2_3rd_bb_.setZero();
            t2_3rd_ab_ = Eigen::Tensor<double, 4>(na_, nb_, va_, vb_); t2_3rd_ab_.setZero();
        }
        if (na_act <= 0 || va_ <= 0) { e_mp3_tot_ = 0.0; return; }

        double e3_aa = 0.0, e3_bb = 0.0, e3_ab = 0.0;
        Eigen::Tensor<double, 4> Taa_act(na_act, na_act, va_, va_);
        #pragma omp parallel for collapse(4) schedule(static)
        for (int i = 0; i < na_act; ++i)
            for (int j = 0; j < na_act; ++j)
            for (int a = 0; a < va_; ++a)
                for (int b = 0; b < va_; ++b)
                Taa_act(i, j, a, b) = T2_aa_ijab(nf_a + i, nf_a + j, a, b);

        const bool has_b_act = (nb_act > 0 && vb_ > 0);
        Eigen::Tensor<double, 4> Tbb_act, Tab_act;
        if (has_b_act) {
            auto* t2_bb_d = t2_bb_.get_block(0, 0, 0, 0);
            auto* t2_ab_d = t2_ab_.get_block(0, 0, 0, 0);
            Tbb_act = Eigen::Tensor<double, 4>(nb_act, nb_act, vb_, vb_); Tbb_act.setZero();
            Tab_act = Eigen::Tensor<double, 4>(na_act, nb_act, va_, vb_); Tab_act.setZero();
            if (t2_bb_d) {
            #pragma omp parallel for collapse(4) schedule(static)
            for (int i = 0; i < nb_act; ++i)
                for (int j = 0; j < nb_act; ++j)
                for (int a = 0; a < vb_; ++a)
                    for (int b = 0; b < vb_; ++b)
                    Tbb_act(i, j, a, b) = (*t2_bb_d)(nf_b + i, nf_b + j, a, b);
            }
            if (t2_ab_d) {
            #pragma omp parallel for collapse(4) schedule(static)
            for (int i = 0; i < na_act; ++i)
                for (int j = 0; j < nb_act; ++j)
                for (int a = 0; a < va_; ++a)
                    for (int b = 0; b < vb_; ++b)
                    Tab_act(i, j, a, b) = (*t2_ab_d)(nf_a + i, nf_b + j, a, b);
            }
        }

        Eigen::MatrixXd B_ij_a = slice_boo(B_oo_P_alpha_, na_, nf_a);
        Eigen::MatrixXd B_ia_a = B_ia_P_alpha_.bottomRows(static_cast<Eigen::Index>(na_act) * va_);
        const Eigen::MatrixXd& B_ab_a = B_vv_P_alpha_;

        Eigen::MatrixXd B_ij_b, B_ia_b;
        if (has_b_act) {
            B_ij_b = slice_boo(B_oo_P_beta_, nb_, nf_b);
            B_ia_b = B_ia_P_beta_.bottomRows(static_cast<Eigen::Index>(nb_act) * vb_);
        }

        Eigen::MatrixXd M_oooo_aa = parallel_gemm_abt_mp3(B_ij_a, B_ij_a);
        Eigen::MatrixXd M_ovov_aa = parallel_gemm_abt_mp3(B_ia_a, B_ia_a);
        Eigen::MatrixXd M_oovv_aa = parallel_gemm_abt_mp3(B_ij_a, B_ab_a);
        Eigen::MatrixXd M_ovov_ab;
        if (has_b_act) M_ovov_ab = parallel_gemm_abt_mp3(B_ia_a, B_ia_b);

        TBLIS_VIEW_4D(t_Taa, Taa_act, na_act, na_act, va_, va_);
        {
            Eigen::Tensor<double, 4> Waa_l(na_act, na_act, va_, va_); Waa_l.setZero();
            Eigen::Tensor<double, 4> Waa_r(na_act, na_act, va_, va_); Waa_r.setZero();
            TBLIS_VIEW_4D(t_Wl, Waa_l, na_act, na_act, va_, va_);
            TBLIS_VIEW_4D(t_Wr, Waa_r, na_act, na_act, va_, va_);
            auto t_Voooo = make_mat_view_4d_mp3(M_oooo_aa, na_act, na_act, na_act, na_act);
            auto t_Vovov = make_mat_view_4d_mp3(M_ovov_aa, na_act, va_, na_act, va_);
            auto t_Voovv = make_mat_view_4d_mp3(M_oovv_aa, na_act, na_act, va_, va_);
            if (na_act > 1) {
            tblis::mult<double>( 0.5, t_Taa, "mnab", t_Voooo, "minj", 1.0, t_Wl, "ijab");
            tblis::mult<double>(-0.5, t_Taa, "mnab", t_Voooo, "mjni", 1.0, t_Wl, "ijab");
            tblis::mult<double>( 1.0, t_Vovov, "iakc", t_Taa, "kjcb", 1.0, t_Wr, "ijab");
            tblis::mult<double>(-1.0, t_Voovv, "ikac", t_Taa, "kjcb", 1.0, t_Wr, "ijab");
            if (has_b_act) {
                auto t_Vovov_ab = make_mat_view_4d_mp3(M_ovov_ab, na_act, va_, nb_act, vb_);
                TBLIS_VIEW_4D(t_Tab, Tab_act, na_act, nb_act, va_, vb_);
                tblis::mult<double>(1.0, t_Vovov_ab, "iakc", t_Tab, "jkbc", 1.0, t_Wr, "ijab");
            }
            const int n_p = na_act * (na_act - 1) / 2;
            std::vector<Eigen::MatrixXd> Tp(n_p, Eigen::MatrixXd(va_, va_));
            std::vector<Eigen::MatrixXd> Wp(n_p, Eigen::MatrixXd::Zero(va_, va_));
            int pi = 0;
            for (int i = 0; i < na_act; ++i)
                for (int j = i + 1; j < na_act; ++j) {
                for (int e = 0; e < va_; ++e)
                    for (int f = 0; f < va_; ++f) Tp[pi](e, f) = Taa_act(i, j, e, f);
                ++pi;
                }
            contract_vvvv_pairs_df(B_ab_a, va_, B_ab_a, va_, n_aux, Tp, Wp);
            pi = 0;
            for (int i = 0; i < na_act; ++i)
                for (int j = i + 1; j < na_act; ++j) {
                for (int a = 0; a < va_; ++a)
                    for (int b = 0; b < va_; ++b) {
                    const double v = 0.5 * (Wp[pi](a, b) - Wp[pi](b, a));
                    Waa_l(i, j, a, b) += v;
                    Waa_l(j, i, a, b) -= v;
                    }
                ++pi;
                }
            #pragma omp parallel for collapse(4) reduction(+:e3_aa) schedule(static)
            for (int i = 0; i < na_act; ++i)
                for (int j = 0; j < na_act; ++j)
                for (int a = 0; a < va_; ++a)
                    for (int b = 0; b < va_; ++b) {
                    const double ra = Waa_r(i, j, a, b) - Waa_r(j, i, a, b) - Waa_r(i, j, b, a) + Waa_r(j, i, b, a);
                    const double wt = Waa_l(i, j, a, b) + ra;
                    const double D = ea(nf_a + i) + ea(nf_a + j) - ea(na_ + a) - ea(na_ + b);
                    t2_3rd_aa_(nf_a + i, nf_a + j, a, b) = wt / D;
                    e3_aa += 0.25 * Taa_act(i, j, a, b) * wt;
                    }
            }
        }

        if (has_b_act) {
            const auto& eb = scf_.orbital_energies_beta;
            const Eigen::MatrixXd& B_ab_b = B_vv_P_beta_;
            TBLIS_VIEW_4D(t_Tbb, Tbb_act, nb_act, nb_act, vb_, vb_);
            TBLIS_VIEW_4D(t_Tab, Tab_act, na_act, nb_act, va_, vb_);

            Eigen::MatrixXd M_oooo_bb = parallel_gemm_abt_mp3(B_ij_b, B_ij_b);
            Eigen::MatrixXd M_ovov_bb = parallel_gemm_abt_mp3(B_ia_b, B_ia_b);
            Eigen::MatrixXd M_oovv_bb = parallel_gemm_abt_mp3(B_ij_b, B_ab_b);

            if (nb_act > 1) {
            Eigen::Tensor<double, 4> Wbb_l(nb_act, nb_act, vb_, vb_); Wbb_l.setZero();
            Eigen::Tensor<double, 4> Wbb_r(nb_act, nb_act, vb_, vb_); Wbb_r.setZero();
            TBLIS_VIEW_4D(t_Wl, Wbb_l, nb_act, nb_act, vb_, vb_);
            TBLIS_VIEW_4D(t_Wr, Wbb_r, nb_act, nb_act, vb_, vb_);
            auto t_Voooo = make_mat_view_4d_mp3(M_oooo_bb, nb_act, nb_act, nb_act, nb_act);
            auto t_Vovov = make_mat_view_4d_mp3(M_ovov_bb, nb_act, vb_, nb_act, vb_);
            auto t_Voovv = make_mat_view_4d_mp3(M_oovv_bb, nb_act, nb_act, vb_, vb_);
            auto t_Vovov_ab = make_mat_view_4d_mp3(M_ovov_ab, na_act, va_, nb_act, vb_);
            tblis::mult<double>( 0.5, t_Tbb, "mnab", t_Voooo, "minj", 1.0, t_Wl, "ijab");
            tblis::mult<double>(-0.5, t_Tbb, "mnab", t_Voooo, "mjni", 1.0, t_Wl, "ijab");
            tblis::mult<double>( 1.0, t_Vovov, "iakc", t_Tbb, "kjcb", 1.0, t_Wr, "ijab");
            tblis::mult<double>(-1.0, t_Voovv, "ikac", t_Tbb, "kjcb", 1.0, t_Wr, "ijab");
            tblis::mult<double>( 1.0, t_Vovov_ab, "kcia", t_Tab, "kjcb", 1.0, t_Wr, "ijab");

            const int n_p = nb_act * (nb_act - 1) / 2;
            std::vector<Eigen::MatrixXd> Tp(n_p, Eigen::MatrixXd(vb_, vb_));
            std::vector<Eigen::MatrixXd> Wp(n_p, Eigen::MatrixXd::Zero(vb_, vb_));
            int pi = 0;
            for (int i = 0; i < nb_act; ++i)
                for (int j = i + 1; j < nb_act; ++j) {
                for (int e = 0; e < vb_; ++e)
                    for (int f = 0; f < vb_; ++f) Tp[pi](e, f) = Tbb_act(i, j, e, f);
                ++pi;
                }
            contract_vvvv_pairs_df(B_ab_b, vb_, B_ab_b, vb_, n_aux, Tp, Wp);
            pi = 0;
            for (int i = 0; i < nb_act; ++i)
                for (int j = i + 1; j < nb_act; ++j) {
                for (int a = 0; a < vb_; ++a)
                    for (int b = 0; b < vb_; ++b) {
                    const double v = 0.5 * (Wp[pi](a, b) - Wp[pi](b, a));
                    Wbb_l(i, j, a, b) += v;
                    Wbb_l(j, i, a, b) -= v;
                    }
                ++pi;
                }
            #pragma omp parallel for collapse(4) reduction(+:e3_bb) schedule(static)
            for (int i = 0; i < nb_act; ++i)
                for (int j = 0; j < nb_act; ++j)
                for (int a = 0; a < vb_; ++a)
                    for (int b = 0; b < vb_; ++b) {
                    const double ra = Wbb_r(i, j, a, b) - Wbb_r(j, i, a, b) - Wbb_r(i, j, b, a) + Wbb_r(j, i, b, a);
                    const double wt = Wbb_l(i, j, a, b) + ra;
                    const double D = eb(nf_b + i) + eb(nf_b + j) - eb(nb_ + a) - eb(nb_ + b);
                    t2_3rd_bb_(nf_b + i, nf_b + j, a, b) = wt / D;
                    e3_bb += 0.25 * Tbb_act(i, j, a, b) * wt;
                    }
            }

            {
            Eigen::Tensor<double, 4> Wab_l(na_act, nb_act, va_, vb_); Wab_l.setZero();
            Eigen::Tensor<double, 4> Wab_r(na_act, nb_act, va_, vb_); Wab_r.setZero();
            TBLIS_VIEW_4D(t_Wl, Wab_l, na_act, nb_act, va_, vb_);
            TBLIS_VIEW_4D(t_Wr, Wab_r, na_act, nb_act, va_, vb_);
            Eigen::MatrixXd M_oooo_ab = parallel_gemm_abt_mp3(B_ij_a, B_ij_b);
            auto t_Voooo_ab = make_mat_view_4d_mp3(M_oooo_ab, na_act, na_act, nb_act, nb_act);
            tblis::mult<double>(1.0, t_Tab, "mnab", t_Voooo_ab, "minj", 1.0, t_Wl, "ijab");

            auto t_Vovov_aa = make_mat_view_4d_mp3(M_ovov_aa, na_act, va_, na_act, va_);
            auto t_Voovv_aa = make_mat_view_4d_mp3(M_oovv_aa, na_act, na_act, va_, va_);
            auto t_Vovov_bb = make_mat_view_4d_mp3(M_ovov_bb, nb_act, vb_, nb_act, vb_);
            auto t_Voovv_bb = make_mat_view_4d_mp3(M_oovv_bb, nb_act, nb_act, vb_, vb_);
            auto t_Vovov_ab = make_mat_view_4d_mp3(M_ovov_ab, na_act, va_, nb_act, vb_);
            tblis::mult<double>( 1.0, t_Vovov_aa, "iakc", t_Tab, "kjcb", 1.0, t_Wr, "ijab");
            tblis::mult<double>(-1.0, t_Voovv_aa, "ikac", t_Tab, "kjcb", 1.0, t_Wr, "ijab");
            tblis::mult<double>( 1.0, t_Tab, "ikac", t_Vovov_bb, "kcjb", 1.0, t_Wr, "ijab");
            tblis::mult<double>(-1.0, t_Tab, "ikac", t_Voovv_bb, "kjcb", 1.0, t_Wr, "ijab");
            if (na_act > 1) tblis::mult<double>(1.0, t_Taa, "ikac", t_Vovov_ab, "kcjb", 1.0, t_Wr, "ijab");
            if (nb_act > 1) tblis::mult<double>(1.0, t_Vovov_ab, "iakc", t_Tbb, "kjcb", 1.0, t_Wr, "ijab");

            Eigen::MatrixXd M_oovv_ab_ex = parallel_gemm_abt_mp3(B_ij_a, B_ab_b);
            Eigen::MatrixXd M_oovv_ba_ex = parallel_gemm_abt_mp3(B_ij_b, B_ab_a);
            auto t_Voovv_ab_ex = make_mat_view_4d_mp3(M_oovv_ab_ex, na_act, na_act, vb_, vb_);
            auto t_Voovv_ba_ex = make_mat_view_4d_mp3(M_oovv_ba_ex, nb_act, nb_act, va_, va_);
            tblis::mult<double>(-1.0, t_Voovv_ab_ex, "ikbc", t_Tab, "kjac", 1.0, t_Wr, "ijab");
            tblis::mult<double>(-1.0, t_Tab, "ikcb", t_Voovv_ba_ex, "jkac", 1.0, t_Wr, "ijab");

            const int n_p = na_act * nb_act;
            std::vector<Eigen::MatrixXd> Tp(n_p, Eigen::MatrixXd(va_, vb_));
            std::vector<Eigen::MatrixXd> Wp(n_p, Eigen::MatrixXd::Zero(va_, vb_));
            for (int i = 0; i < na_act; ++i)
                for (int j = 0; j < nb_act; ++j)
                for (int e = 0; e < va_; ++e)
                    for (int f = 0; f < vb_; ++f)
                    Tp[i * nb_act + j](e, f) = Tab_act(i, j, e, f);
            contract_vvvv_pairs_df(B_ab_a, va_, B_ab_b, vb_, n_aux, Tp, Wp);
            for (int i = 0; i < na_act; ++i)
                for (int j = 0; j < nb_act; ++j)
                for (int a = 0; a < va_; ++a)
                    for (int b = 0; b < vb_; ++b)
                    Wab_l(i, j, a, b) += Wp[i * nb_act + j](a, b);

            #pragma omp parallel for collapse(4) reduction(+:e3_ab) schedule(static)
            for (int i = 0; i < na_act; ++i)
                for (int j = 0; j < nb_act; ++j)
                for (int a = 0; a < va_; ++a)
                    for (int b = 0; b < vb_; ++b) {
                    const double wt = Wab_l(i, j, a, b) + Wab_r(i, j, a, b);
                    const double D = ea(nf_a + i) + eb(nf_b + j) - ea(na_ + a) - eb(nb_ + b);
                    t2_3rd_ab_(nf_a + i, nf_b + j, a, b) = wt / D;
                    e3_ab += Tab_act(i, j, a, b) * wt;
                    }
            }
        }
        e_mp3_tot_ = e3_aa + e3_bb + e3_ab;
        return;
        }
    }

    const Eigen::MatrixXd& B_ij_a = B_oo_P_alpha_;
    const Eigen::MatrixXd& B_ab_a = B_vv_P_alpha_;

    if (is_restricted) {
        t2_3rd_aa_ = Eigen::Tensor< double, 4 >(na_, na_, va_, va_);
        Eigen::Tensor< double, 4 > W(na_, na_, va_, va_); W.setZero();
        
        TBLIS_VIEW_4D(t_T, T2_aa_ijab, na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_W, W, na_, na_, va_, va_);
      
        Eigen::MatrixXd M_oooo = parallel_gemm_abt_mp3(B_ij_a, B_ij_a);
        Eigen::MatrixXd M_ovov = parallel_gemm_abt_mp3(B_ia_P_alpha_, B_ia_P_alpha_);
        Eigen::MatrixXd M_oovv = parallel_gemm_abt_mp3(B_ij_a, B_ab_a);
        
        auto t_Voooo = make_mat_view_4d_mp3(M_oooo, na_, na_, na_, na_);
        auto t_Vovov = make_mat_view_4d_mp3(M_ovov, na_, va_, na_, va_);
        auto t_Voovv = make_mat_view_4d_mp3(M_oovv, na_, na_, va_, va_);

        tblis::mult< double >(1.0,  t_T, "mnab", t_Voooo, "minj", 1.0, t_W, "ijab");
        tblis::mult< double >(2.0,  t_Vovov, "iakc", t_T, "kjcb", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_Voovv, "ikac", t_T, "kjcb", 1.0, t_W, "ijab");
        tblis::mult< double >(2.0,  t_T, "ikac", t_Vovov, "kcjb", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_T, "ikac", t_Voovv, "kjcb", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_T, "ikca", t_Vovov, "kcjb", 1.0, t_W, "ijab"); 
        tblis::mult< double >(-1.0, t_Vovov, "iakc", t_T, "kjbc", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_Voovv, "ikbc", t_T, "kjac", 1.0, t_W, "ijab");
        tblis::mult< double >(-1.0, t_T, "ikcb", t_Voovv, "jkac", 1.0, t_W, "ijab"); 

        // Eksploitasi simetri W_ji(a,b) = W_ij(b,a): hanya hitung pasangan unik i <= j!
        int n_pairs_sym = na_ * (na_ + 1) / 2;
        std::vector<Eigen::MatrixXd> T_pairs(n_pairs_sym, Eigen::MatrixXd(va_, va_));
        std::vector<Eigen::MatrixXd> W_pairs(n_pairs_sym, Eigen::MatrixXd::Zero(va_, va_));

        int p_idx = 0;
        for (int i = 0; i < na_; ++i) {
            for (int j = i; j < na_; ++j) {
                for (int e = 0; e < va_; ++e)
                    for (int f = 0; f < va_; ++f)
                        T_pairs[p_idx](e, f) = T2_aa_ijab(i, j, e, f);
                p_idx++;
            }
        }

        contract_vvvv_pairs_df(B_ab_a, va_, B_ab_a, va_, n_aux, T_pairs, W_pairs);

        p_idx = 0;
        for (int i = 0; i < na_; ++i) {
            for (int j = i; j < na_; ++j) {
                for (int a = 0; a < va_; ++a) {
                    for (int b = 0; b < va_; ++b) {
                        W(i, j, a, b) += W_pairs[p_idx](a, b);
                        if (i != j) W(j, i, b, a) += W_pairs[p_idx](a, b);
                    }
                }
                p_idx++;
            }
        }

        double e3_aa = 0.0;
        #pragma omp parallel for collapse(4) reduction(+:e3_aa) schedule(static)
        for (int i = 0; i < na_; ++i) {
            for (int j = 0; j < na_; ++j) {
                for (int a = 0; a < va_; ++a) {
                    for (int b = 0; b < va_; ++b) {
                        double D = ea(i) + ea(j) - ea(na_+a) - ea(na_+b);
                        t2_3rd_aa_(i,j,a,b) = W(i,j,a,b) / D; 
                        e3_aa += W(i, j, a, b) * (2.0 * T2_aa_ijab(i, j, a, b) - T2_aa_ijab(i, j, b, a));
                    }
                }
            }
        }
        e_mp3_tot_ = e3_aa;
        return; 
    }

    t2_3rd_aa_ = Eigen::Tensor< double, 4 >(na_, na_, va_, va_); t2_3rd_aa_.setZero();
    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        t2_3rd_bb_ = Eigen::Tensor< double, 4 >(nb_, nb_, vb_, vb_); t2_3rd_bb_.setZero();
        t2_3rd_ab_ = Eigen::Tensor< double, 4 >(na_, nb_, va_, vb_); t2_3rd_ab_.setZero();
    }

    double e3_aa = 0.0, e3_bb = 0.0, e3_ab = 0.0;
    TBLIS_VIEW_4D(t_Taa, T2_aa_ijab, na_, na_, va_, va_);
    auto* t2_bb_dense_ptr = t2_bb_.get_block(0,0,0,0);
    Eigen::Tensor< double, 4 > dummy_bb_local;
    if (!t2_bb_dense_ptr && !is_restricted && nb_ > 0 && vb_ > 0) {
        dummy_bb_local = Eigen::Tensor< double, 4 >(nb_, nb_, vb_, vb_);
        dummy_bb_local.setZero();
        t2_bb_dense_ptr = &dummy_bb_local;
    }
    auto* t2_ab_dense = t2_ab_.get_block(0,0,0,0);
    Eigen::Tensor< double, 4 > dummy_ab;
    if (!t2_ab_dense && !is_restricted && nb_ > 0 && vb_ > 0) {
        dummy_ab = Eigen::Tensor< double, 4 >(na_, nb_, va_, vb_);
        dummy_ab.setZero();
        t2_ab_dense = &dummy_ab;
    }

    Eigen::MatrixXd M_oooo_aa = parallel_gemm_abt_mp3(B_ij_a, B_ij_a);
    Eigen::MatrixXd M_ovov_aa = parallel_gemm_abt_mp3(B_ia_P_alpha_, B_ia_P_alpha_);
    Eigen::MatrixXd M_oovv_aa = parallel_gemm_abt_mp3(B_ij_a, B_ab_a);
    Eigen::MatrixXd M_ovov_ab;
    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        M_ovov_ab = parallel_gemm_abt_mp3(B_ia_P_alpha_, B_ia_P_beta_);
    }

    {
        if (Waa_ladder_.size() == 0) Waa_ladder_.resize(na_, na_, va_, va_);
        if (Waa_ring_.size() == 0) Waa_ring_.resize(na_, na_, va_, va_);
        
        Waa_ladder_.setZero(); Waa_ring_.setZero();
        TBLIS_VIEW_4D(t_Waa_ladder, Waa_ladder_, na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_Waa_ring, Waa_ring_, na_, na_, va_, va_);
        
        auto t_Voooo = make_mat_view_4d_mp3(M_oooo_aa, na_, na_, na_, na_);
        auto t_Vovov = make_mat_view_4d_mp3(M_ovov_aa, na_, va_, na_, va_);
        auto t_Voovv = make_mat_view_4d_mp3(M_oovv_aa, na_, na_, va_, va_);

        tblis::mult< double >(0.5,  t_Taa, "mnab", t_Voooo, "minj", 1.0, t_Waa_ladder, "ijab");
        tblis::mult< double >(-0.5, t_Taa, "mnab", t_Voooo, "mjni", 1.0, t_Waa_ladder, "ijab");

        tblis::mult< double >(1.0,  t_Vovov, "iakc", t_Taa, "kjcb", 1.0, t_Waa_ring, "ijab");
        tblis::mult< double >(-1.0, t_Voovv, "ikac", t_Taa, "kjcb", 1.0, t_Waa_ring, "ijab");

        if (!is_restricted && nb_ > 0 && vb_ > 0) {
            auto t_Vovov_ab = make_mat_view_4d_mp3(M_ovov_ab, na_, va_, nb_, vb_);
            TBLIS_VIEW_4D(t_Tab, (*t2_ab_dense), na_, nb_, va_, vb_);
            tblis::mult< double >(1.0, t_Vovov_ab, "iakc", t_Tab, "jkbc", 1.0, t_Waa_ring, "ijab");
        }

        // Antisimetri AA: hanya hitung pasangan i < j!
        int n_pairs_aa = na_ * (na_ - 1) / 2;
        if (n_pairs_aa > 0) {
            std::vector<Eigen::MatrixXd> T_aa_pairs(n_pairs_aa, Eigen::MatrixXd(va_, va_));
            std::vector<Eigen::MatrixXd> W_aa_pairs(n_pairs_aa, Eigen::MatrixXd::Zero(va_, va_));
            int p_idx = 0;
            for (int i = 0; i < na_; ++i) {
                for (int j = i + 1; j < na_; ++j) {
                    for (int e = 0; e < va_; ++e)
                        for (int f = 0; f < va_; ++f)
                            T_aa_pairs[p_idx](e, f) = T2_aa_ijab(i, j, e, f);
                    p_idx++;
                }
            }

            contract_vvvv_pairs_df(B_ab_a, va_, B_ab_a, va_, n_aux, T_aa_pairs, W_aa_pairs);

            p_idx = 0;
            for (int i = 0; i < na_; ++i) {
                for (int j = i + 1; j < na_; ++j) {
                    for (int a = 0; a < va_; ++a) {
                        for (int b = 0; b < va_; ++b) {
                            double val = 0.5 * (W_aa_pairs[p_idx](a, b) - W_aa_pairs[p_idx](b, a));
                            Waa_ladder_(i, j, a, b) += val;
                            Waa_ladder_(j, i, a, b) -= val;
                        }
                    }
                    p_idx++;
                }
            }
        }

        #pragma omp parallel for collapse(4) reduction(+:e3_aa) schedule(static)
        for(int i = 0; i < na_; ++i) {
            for(int j = 0; j < na_; ++j) {
                for(int a = 0; a < va_; ++a) {
                    for(int b = 0; b < va_; ++b) {
                        double r_asym = Waa_ring_(i,j,a,b) - Waa_ring_(j,i,a,b) - Waa_ring_(i,j,b,a) + Waa_ring_(j,i,b,a);
                        double w_tot = Waa_ladder_(i,j,a,b) + r_asym;
                        double D = ea(i) + ea(j) - ea(na_+a) - ea(na_+b);
                        t2_3rd_aa_(i,j,a,b) = w_tot / D;
                        e3_aa += 0.25 * T2_aa_ijab(i,j,a,b) * w_tot;
                    }
                }
            }
        }
        Waa_ladder_.resize(0, 0, 0, 0);
        Waa_ring_.resize(0, 0, 0, 0);
        M_oooo_aa.resize(0, 0);
    }

    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        const auto& eb = scf_.orbital_energies_beta;
        const Eigen::MatrixXd& B_ij_b = B_oo_P_beta_;
        const Eigen::MatrixXd& B_ab_b = B_vv_P_beta_;
     
        TBLIS_VIEW_4D(t_Tbb, (*t2_bb_dense_ptr), nb_, nb_, vb_, vb_);
        TBLIS_VIEW_4D(t_Tab, (*t2_ab_dense), na_, nb_, va_, vb_);

        Eigen::MatrixXd M_oooo_bb = parallel_gemm_abt_mp3(B_ij_b, B_ij_b);
        Eigen::MatrixXd M_ovov_bb = parallel_gemm_abt_mp3(B_ia_P_beta_, B_ia_P_beta_);
        Eigen::MatrixXd M_oovv_bb = parallel_gemm_abt_mp3(B_ij_b, B_ab_b);

        {
            Eigen::Tensor< double, 4 > Wbb_ladder(nb_, nb_, vb_, vb_); Wbb_ladder.setZero();
            Eigen::Tensor< double, 4 > Wbb_ring(nb_, nb_, vb_, vb_); Wbb_ring.setZero();
            
            TBLIS_VIEW_4D(t_Wbb_ladder, Wbb_ladder, nb_, nb_, vb_, vb_);
            TBLIS_VIEW_4D(t_Wbb_ring, Wbb_ring, nb_, nb_, vb_, vb_);
            auto t_Voooo_bb = make_mat_view_4d_mp3(M_oooo_bb, nb_, nb_, nb_, nb_);
            auto t_Vovov_bb = make_mat_view_4d_mp3(M_ovov_bb, nb_, vb_, nb_, vb_);
            auto t_Voovv_bb = make_mat_view_4d_mp3(M_oovv_bb, nb_, nb_, vb_, vb_);
            auto t_Vovov_ab = make_mat_view_4d_mp3(M_ovov_ab, na_, va_, nb_, vb_);

            if (nb_ > 1) {
                tblis::mult< double >(0.5,  t_Tbb, "mnab", t_Voooo_bb, "minj", 1.0, t_Wbb_ladder, "ijab");
                tblis::mult< double >(-0.5, t_Tbb, "mnab", t_Voooo_bb, "mjni", 1.0, t_Wbb_ladder, "ijab");
         
                tblis::mult< double >(1.0,  t_Vovov_bb, "iakc", t_Tbb, "kjcb", 1.0, t_Wbb_ring, "ijab");
                tblis::mult< double >(-1.0, t_Voovv_bb, "ikac", t_Tbb, "kjcb", 1.0, t_Wbb_ring, "ijab");
                tblis::mult< double >(1.0,  t_Vovov_ab, "kcia", t_Tab, "kjcb", 1.0, t_Wbb_ring, "ijab"); 
            }

            // Antisimetri BB: hanya hitung jika nb_ > 1 untuk pasangan i < j!
            // Antisimetri BB: hanya hitung jika nb_ > 1 untuk pasangan i < j!
            int n_pairs_bb = nb_ * (nb_ - 1) / 2;
            if (n_pairs_bb > 0) {
                std::vector<Eigen::MatrixXd> T_bb_pairs(n_pairs_bb, Eigen::MatrixXd(vb_, vb_));
                std::vector<Eigen::MatrixXd> W_bb_pairs(n_pairs_bb, Eigen::MatrixXd::Zero(vb_, vb_));
                int p_idx = 0;
                for (int i = 0; i < nb_; ++i) {
                    for (int j = i + 1; j < nb_; ++j) {
                        for (int e = 0; e < vb_; ++e)
                            for (int f = 0; f < vb_; ++f)
                                T_bb_pairs[p_idx](e, f) = (*t2_bb_dense_ptr)(i, j, e, f);
                        p_idx++;
                    }
                }

                contract_vvvv_pairs_df(B_ab_b, vb_, B_ab_b, vb_, n_aux, T_bb_pairs, W_bb_pairs);

                p_idx = 0;
                for (int i = 0; i < nb_; ++i) {
                    for (int j = i + 1; j < nb_; ++j) {
                        for (int a = 0; a < vb_; ++a) {
                            for (int b = 0; b < vb_; ++b) {
                                double val = 0.5 * (W_bb_pairs[p_idx](a, b) - W_bb_pairs[p_idx](b, a));
                                Wbb_ladder(i, j, a, b) += val;
                                Wbb_ladder(j, i, a, b) -= val;
                            }
                        }
                        p_idx++;
                    }
                }
            }

            #pragma omp parallel for collapse(4) reduction(+:e3_bb) schedule(static)
            for(int i = 0; i < nb_; ++i) {
                for(int j = 0; j < nb_; ++j) {
                    for(int a = 0; a < vb_; ++a) {
                        for(int b = 0; b < vb_; ++b) {
                            double r_asym = Wbb_ring(i,j,a,b) - Wbb_ring(j,i,a,b) - Wbb_ring(i,j,b,a) + Wbb_ring(j,i,b,a);
                            double w_tot = Wbb_ladder(i,j,a,b) + r_asym;
                            double D = eb(i) + eb(j) - eb(nb_+a) - eb(nb_+b);
                            t2_3rd_bb_(i,j,a,b) = w_tot / D;
                            e3_bb += 0.25 * (*t2_bb_dense_ptr)(i,j,a,b) * w_tot;
                        }
                    }
                }
            }
        }

        {
            Eigen::Tensor< double, 4 > Wab_ladder(na_, nb_, va_, vb_); Wab_ladder.setZero();
            Eigen::Tensor< double, 4 > Wab_ring(na_, nb_, va_, vb_); Wab_ring.setZero();
            
            TBLIS_VIEW_4D(t_Wab_ladder, Wab_ladder, na_, nb_, va_, vb_);
            TBLIS_VIEW_4D(t_Wab_ring, Wab_ring, na_, nb_, va_, vb_);

            Eigen::MatrixXd M_oooo_ab = parallel_gemm_abt_mp3(B_ij_a, B_ij_b);
            auto t_Voooo_ab = make_mat_view_4d_mp3(M_oooo_ab, na_, na_, nb_, nb_);
            tblis::mult< double >(1.0, t_Tab, "mnab", t_Voooo_ab, "minj", 1.0, t_Wab_ladder, "ijab");

            auto t_Vovov_aa = make_mat_view_4d_mp3(M_ovov_aa, na_, va_, na_, va_);
            auto t_Voovv_aa = make_mat_view_4d_mp3(M_oovv_aa, na_, na_, va_, va_);
            tblis::mult< double >(1.0,  t_Vovov_aa, "iakc", t_Tab, "kjcb", 1.0, t_Wab_ring, "ijab");
            tblis::mult< double >(-1.0, t_Voovv_aa, "ikac", t_Tab, "kjcb", 1.0, t_Wab_ring, "ijab");

            auto t_Vovov_bb = make_mat_view_4d_mp3(M_ovov_bb, nb_, vb_, nb_, vb_);
            auto t_Voovv_bb = make_mat_view_4d_mp3(M_oovv_bb, nb_, nb_, vb_, vb_);
            tblis::mult< double >(1.0,  t_Tab, "ikac", t_Vovov_bb, "kcjb", 1.0, t_Wab_ring, "ijab"); 
            tblis::mult< double >(-1.0, t_Tab, "ikac", t_Voovv_bb, "kjcb", 1.0, t_Wab_ring, "ijab");

            auto t_Vovov_ab = make_mat_view_4d_mp3(M_ovov_ab, na_, va_, nb_, vb_);
            tblis::mult< double >(1.0,  t_Taa, "ikac", t_Vovov_ab, "kcjb", 1.0, t_Wab_ring, "ijab");
            if (nb_ > 1) {
                tblis::mult< double >(1.0,  t_Vovov_ab, "iakc", t_Tbb, "kjcb", 1.0, t_Wab_ring, "ijab");
            }

            Eigen::MatrixXd M_oovv_ab_ex = parallel_gemm_abt_mp3(B_ij_a, B_ab_b);
            Eigen::MatrixXd M_oovv_ba_ex = parallel_gemm_abt_mp3(B_ij_b, B_ab_a);
            auto t_Voovv_ab_ex = make_mat_view_4d_mp3(M_oovv_ab_ex, na_, na_, vb_, vb_);
            auto t_Voovv_ba_ex = make_mat_view_4d_mp3(M_oovv_ba_ex, nb_, nb_, va_, va_);

            tblis::mult< double >(-1.0, t_Voovv_ab_ex, "ikbc", t_Tab, "kjac", 1.0, t_Wab_ring, "ijab");
            tblis::mult< double >(-1.0, t_Tab, "ikcb", t_Voovv_ba_ex, "jkac", 1.0, t_Wab_ring, "ijab");
            M_oovv_ab_ex.resize(0, 0);
            M_oovv_ba_ex.resize(0, 0);

            int n_pairs_ab = na_ * nb_;
            std::vector<Eigen::MatrixXd> T_ab_pairs(n_pairs_ab, Eigen::MatrixXd(va_, vb_));
            std::vector<Eigen::MatrixXd> W_ab_pairs(n_pairs_ab, Eigen::MatrixXd::Zero(va_, vb_));
            for (int i = 0; i < na_; ++i) {
                for (int j = 0; j < nb_; ++j) {
                    int k = i * nb_ + j;
                    for (int e = 0; e < va_; ++e)
                        for (int f = 0; f < vb_; ++f)
                            T_ab_pairs[k](e, f) = (*t2_ab_dense)(i, j, e, f);
                }
            }

            contract_vvvv_pairs_df(B_ab_a, va_, B_ab_b, vb_, n_aux, T_ab_pairs, W_ab_pairs);

            for (int i = 0; i < na_; ++i) {
                for (int j = 0; j < nb_; ++j) {
                    int k = i * nb_ + j;
                    for (int a = 0; a < va_; ++a)
                        for (int b = 0; b < vb_; ++b)
                            Wab_ladder(i, j, a, b) += W_ab_pairs[k](a, b);
                }
            }

            #pragma omp parallel for collapse(4) reduction(+:e3_ab) schedule(static)
            for(int i = 0; i < na_; ++i) {
                for(int j = 0; j < nb_; ++j) {
                    for(int a = 0; a < va_; ++a) {
                        for(int b = 0; b < vb_; ++b) {
                            double w_tot = Wab_ladder(i,j,a,b) + Wab_ring(i,j,a,b); 
                            double D = ea(i) + eb(j) - ea(na_+a) - eb(nb_+b);
                            t2_3rd_ab_(i,j,a,b) = w_tot / D;
                            e3_ab += 1.0 * (*t2_ab_dense)(i,j,a,b) * w_tot;
                        }
                    }
                }
            }
        }
    }
    
    e_mp3_tot_ = e3_aa + e3_bb + e3_ab;
}
// [INJECTED_HPC_V4_MP3_STREAM_OPDM]
void OMP3::build_opdm_alpha() {
  if (L2_aa_.size() == 0) { OMP2::build_opdm_alpha(); return; }
  auto* t2_aa_dense = t2_aa_.get_block(0, 0, 0, 0);
  if (!t2_aa_dense || na_ <= 0 || va_ <= 0) return;

  const bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
  G_oo_alpha_ = Eigen::MatrixXd::Zero(na_, na_);
  G_vv_alpha_ = Eigen::MatrixXd::Zero(va_, va_);

  const int nf_a = std::clamp(n_frozen_, 0, na_);
  const int na_act = na_ - nf_a;
  if (na_act <= 0) return;

  const int max_t = omp_get_max_threads();
  std::vector<Eigen::MatrixXd> Goo_tls(max_t, Eigen::MatrixXd::Zero(na_act, na_act));
  std::vector<Eigen::MatrixXd> Gvv_tls(max_t, Eigen::MatrixXd::Zero(va_, va_));

  const int a_tile = std::clamp(
      static_cast<int>((4ULL * 1024ULL * 1024ULL) /
                       std::max<size_t>(1, static_cast<size_t>(na_act) * va_ * sizeof(double))),
      1, va_);

  // 1. AA (Zero-Copy Streaming: 0 byte tensor 4D baru, 2 DGEMM alih-alih 3 TBLIS mult!)
  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    Eigen::MatrixXd& Goo_loc = Goo_tls[tid];
    Eigen::MatrixXd& Gvv_loc = Gvv_tls[tid];

    Eigen::MatrixXd T_ij(va_, va_), L_ij(va_, va_), Tt_ij(va_, va_), Sum_t(va_, va_);

    const int n_ij = na_act * na_act;
    #pragma omp for schedule(dynamic, 1) nowait
    for (int p = 0; p < n_ij; ++p) {
      const int i = nf_a + p / na_act;
      const int j = nf_a + p % na_act;
      if (!is_restricted && i == j) continue;

      for (int b = 0; b < va_; ++b) {
        for (int a = 0; a < va_; ++a) {
          T_ij(a, b) = (*t2_aa_dense)(i, a, j, b);
          L_ij(a, b) = L2_aa_(i, j, a, b);
        }
      }
      if (is_restricted) {
        Tt_ij.noalias() = 2.0 * T_ij - T_ij.transpose();
        Sum_t.noalias() = Tt_ij + (2.0 * L_ij - L_ij.transpose());
        Gvv_loc.noalias() += T_ij * Sum_t.transpose() + L_ij * Tt_ij.transpose();
      } else {
        Sum_t.noalias() = T_ij + L_ij;
        Gvv_loc.noalias() += 0.5 * (T_ij * Sum_t.transpose() + L_ij * T_ij.transpose());
      }
    }

    Eigen::MatrixXd MatT(na_act, a_tile * va_);
    Eigen::MatrixXd MatL(na_act, a_tile * va_);
    Eigen::MatrixXd MatTt, MatSum;
    if (is_restricted) {
      MatTt.resize(na_act, a_tile * va_);
      MatSum.resize(na_act, a_tile * va_);
    } else {
      MatSum.resize(na_act, a_tile * va_);
    }

    #pragma omp for schedule(dynamic, 1) nowait
    for (int k = nf_a; k < na_; ++k) {
      for (int a0 = 0; a0 < va_; a0 += a_tile) {
        const int alen = std::min(a_tile, va_ - a0);
        const int cols = alen * va_;
        for (int da = 0; da < alen; ++da) {
          const int a = a0 + da;
          for (int b = 0; b < va_; ++b) {
            const int col = da * va_ + b;
            for (int ii = 0; ii < na_act; ++ii) {
              const int i = nf_a + ii;
              const double tv = (*t2_aa_dense)(i, a, k, b);
              const double lv = L2_aa_(i, k, a, b);
              MatT(ii, col) = tv;
              MatL(ii, col) = lv;
              if (is_restricted) {
                const double ttv = 2.0 * tv - (*t2_aa_dense)(i, b, k, a);
                const double ltv = 2.0 * lv - L2_aa_(i, k, b, a);
                MatTt(ii, col) = ttv;
                MatSum(ii, col) = ttv + ltv;
              } else {
                MatSum(ii, col) = tv + lv;
              }
            }
          }
        }
        if (is_restricted) {
          Goo_loc.noalias() -= MatT.leftCols(cols) * MatSum.leftCols(cols).transpose()
                             + MatL.leftCols(cols) * MatTt.leftCols(cols).transpose();
        } else {
          Goo_loc.noalias() -= 0.5 * (MatT.leftCols(cols) * MatSum.leftCols(cols).transpose()
                                    + MatL.leftCols(cols) * MatT.leftCols(cols).transpose());
        }
      }
    }
  }

  for (int t = 0; t < max_t; ++t) {
    G_oo_alpha_.block(nf_a, nf_a, na_act, na_act).noalias() += Goo_tls[t];
    G_vv_alpha_.noalias() += Gvv_tls[t];
  }

  if (is_restricted) return;

  // 2. AB (Unrestricted)
  auto* t2_ab_dense = t2_ab_.get_block(0, 0, 0, 0);
  const int nf_b = std::clamp(n_frozen_, 0, nb_);
  const int nb_act = nb_ - nf_b;
  if (nb_act > 0 && vb_ > 0 && t2_ab_dense && L2_ab_.size() > 0) {
    for (int t = 0; t < max_t; ++t) {
      Goo_tls[t].setZero();
      Gvv_tls[t].setZero();
    }
    const int a_tile_ab = std::clamp(
        static_cast<int>((4ULL * 1024ULL * 1024ULL) /
                         std::max<size_t>(1, static_cast<size_t>(na_act) * vb_ * sizeof(double))),
        1, va_);

    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      Eigen::MatrixXd& Goo_loc = Goo_tls[tid];
      Eigen::MatrixXd& Gvv_loc = Gvv_tls[tid];
      Eigen::MatrixXd T_ab(va_, vb_), L_ab(va_, vb_), Sum_ab(va_, vb_);

      const int n_ab = na_act * nb_act;
      #pragma omp for schedule(dynamic, 1) nowait
      for (int p = 0; p < n_ab; ++p) {
        const int i = nf_a + p / nb_act;
        const int j = nf_b + p % nb_act;
        for (int b = 0; b < vb_; ++b) {
          for (int a = 0; a < va_; ++a) {
            T_ab(a, b) = (*t2_ab_dense)(i, j, a, b);
            L_ab(a, b) = L2_ab_(i, j, a, b);
          }
        }
        Sum_ab.noalias() = T_ab + L_ab;
        Gvv_loc.noalias() += T_ab * Sum_ab.transpose() + L_ab * T_ab.transpose();
      }

      Eigen::MatrixXd MatT(na_act, a_tile_ab * vb_);
      Eigen::MatrixXd MatL(na_act, a_tile_ab * vb_);
      Eigen::MatrixXd MatSum(na_act, a_tile_ab * vb_);
      #pragma omp for schedule(dynamic, 1) nowait
      for (int k = nf_b; k < nb_; ++k) {
        for (int a0 = 0; a0 < va_; a0 += a_tile_ab) {
          const int alen = std::min(a_tile_ab, va_ - a0);
          const int cols = alen * vb_;
          for (int da = 0; da < alen; ++da) {
            const int a = a0 + da;
            for (int b = 0; b < vb_; ++b) {
              const int col = da * vb_ + b;
              for (int ii = 0; ii < na_act; ++ii) {
                const double tv = (*t2_ab_dense)(nf_a + ii, k, a, b);
                const double lv = L2_ab_(nf_a + ii, k, a, b);
                MatT(ii, col) = tv;
                MatL(ii, col) = lv;
                MatSum(ii, col) = tv + lv;
              }
            }
          }
          Goo_loc.noalias() -= MatT.leftCols(cols) * MatSum.leftCols(cols).transpose()
                             + MatL.leftCols(cols) * MatT.leftCols(cols).transpose();
        }
      }
    }

    for (int t = 0; t < max_t; ++t) {
      G_oo_alpha_.block(nf_a, nf_a, na_act, na_act).noalias() += Goo_tls[t];
      G_vv_alpha_.noalias() += Gvv_tls[t];
    }
  }
}

void OMP3::build_opdm_beta() {
  if (L2_bb_.size() == 0 && L2_ab_.size() == 0) {
    OMP2::build_opdm_beta();
    return;
  }
  const bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
  G_oo_beta_ = Eigen::MatrixXd::Zero(nb_, nb_);
  G_vv_beta_ = Eigen::MatrixXd::Zero(vb_, vb_);
  if (is_restricted || nb_ <= 0 || vb_ <= 0) return;

  const int nf_a = std::clamp(n_frozen_, 0, na_);
  const int nf_b = std::clamp(n_frozen_, 0, nb_);
  const int na_act = na_ - nf_a;
  const int nb_act = nb_ - nf_b;
  if (nb_act <= 0) return;

  auto* t2_bb_dense = t2_bb_.get_block(0, 0, 0, 0);
  auto* t2_ab_dense = t2_ab_.get_block(0, 0, 0, 0);

  const int max_t = omp_get_max_threads();
  std::vector<Eigen::MatrixXd> Goo_tls(max_t, Eigen::MatrixXd::Zero(nb_act, nb_act));
  std::vector<Eigen::MatrixXd> Gvv_tls(max_t, Eigen::MatrixXd::Zero(vb_, vb_));

  if (t2_bb_dense && L2_bb_.size() > 0 && nb_act > 1) {
    const int a_tile = std::clamp(
        static_cast<int>((4ULL * 1024ULL * 1024ULL) /
                         std::max<size_t>(1, static_cast<size_t>(nb_act) * vb_ * sizeof(double))),
        1, vb_);
    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      Eigen::MatrixXd& Goo_loc = Goo_tls[tid];
      Eigen::MatrixXd& Gvv_loc = Gvv_tls[tid];
      Eigen::MatrixXd T_bb(vb_, vb_), L_bb(vb_, vb_), Sum_bb(vb_, vb_);

      const int n_ij = nb_act * nb_act;
      #pragma omp for schedule(dynamic, 1) nowait
      for (int p = 0; p < n_ij; ++p) {
        const int i = nf_b + p / nb_act;
        const int j = nf_b + p % nb_act;
        if (i == j) continue;
        for (int b = 0; b < vb_; ++b) {
          for (int a = 0; a < vb_; ++a) {
            T_bb(a, b) = (*t2_bb_dense)(i, j, a, b);
            L_bb(a, b) = L2_bb_(i, j, a, b);
          }
        }
        Sum_bb.noalias() = T_bb + L_bb;
        Gvv_loc.noalias() += 0.5 * (T_bb * Sum_bb.transpose() + L_bb * T_bb.transpose());
      }

      Eigen::MatrixXd MatT(nb_act, a_tile * vb_);
      Eigen::MatrixXd MatL(nb_act, a_tile * vb_);
      Eigen::MatrixXd MatSum(nb_act, a_tile * vb_);
      #pragma omp for schedule(dynamic, 1) nowait
      for (int k = nf_b; k < nb_; ++k) {
        for (int a0 = 0; a0 < vb_; a0 += a_tile) {
          const int alen = std::min(a_tile, vb_ - a0);
          const int cols = alen * vb_;
          for (int da = 0; da < alen; ++da) {
            const int a = a0 + da;
            for (int b = 0; b < vb_; ++b) {
              const int col = da * vb_ + b;
              for (int ii = 0; ii < nb_act; ++ii) {
                const double tv = (*t2_bb_dense)(nf_b + ii, k, a, b);
                const double lv = L2_bb_(nf_b + ii, k, a, b);
                MatT(ii, col) = tv;
                MatL(ii, col) = lv;
                MatSum(ii, col) = tv + lv;
              }
            }
          }
          Goo_loc.noalias() -= 0.5 * (MatT.leftCols(cols) * MatSum.leftCols(cols).transpose()
                                    + MatL.leftCols(cols) * MatT.leftCols(cols).transpose());
        }
      }
    }
  }

  if (t2_ab_dense && L2_ab_.size() > 0 && na_act > 0 && va_ > 0) {
    const int b_tile = std::clamp(
        static_cast<int>((4ULL * 1024ULL * 1024ULL) /
                         std::max<size_t>(1, static_cast<size_t>(nb_act) * va_ * sizeof(double))),
        1, vb_);
    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      Eigen::MatrixXd& Goo_loc = Goo_tls[tid];
      Eigen::MatrixXd& Gvv_loc = Gvv_tls[tid];
      Eigen::MatrixXd T_ba(vb_, va_), L_ba(vb_, va_), Sum_ba(vb_, va_);

      const int n_ab = na_act * nb_act;
      #pragma omp for schedule(dynamic, 1) nowait
      for (int p = 0; p < n_ab; ++p) {
        const int i = nf_a + p / nb_act;
        const int j = nf_b + p % nb_act;
        for (int a = 0; a < va_; ++a) {
          for (int b = 0; b < vb_; ++b) {
            T_ba(b, a) = (*t2_ab_dense)(i, j, a, b);
            L_ba(b, a) = L2_ab_(i, j, a, b);
          }
        }
        Sum_ba.noalias() = T_ba + L_ba;
        Gvv_loc.noalias() += T_ba * Sum_ba.transpose() + L_ba * T_ba.transpose();
      }

      Eigen::MatrixXd MatT(nb_act, b_tile * va_);
      Eigen::MatrixXd MatL(nb_act, b_tile * va_);
      Eigen::MatrixXd MatSum(nb_act, b_tile * va_);
      #pragma omp for schedule(dynamic, 1) nowait
      for (int k = nf_a; k < na_; ++k) {
        for (int b0 = 0; b0 < vb_; b0 += b_tile) {
          const int blen = std::min(b_tile, vb_ - b0);
          const int cols = blen * va_;
          for (int db = 0; db < blen; ++db) {
            const int b = b0 + db;
            for (int a = 0; a < va_; ++a) {
              const int col = db * va_ + a;
              for (int jj = 0; jj < nb_act; ++jj) {
                const double tv = (*t2_ab_dense)(k, nf_b + jj, a, b);
                const double lv = L2_ab_(k, nf_b + jj, a, b);
                MatT(jj, col) = tv;
                MatL(jj, col) = lv;
                MatSum(jj, col) = tv + lv;
              }
            }
          }
          Goo_loc.noalias() -= MatT.leftCols(cols) * MatSum.leftCols(cols).transpose()
                             + MatL.leftCols(cols) * MatT.leftCols(cols).transpose();
        }
      }
    }
  }

  for (int t = 0; t < max_t; ++t) {
    G_oo_beta_.block(nf_b, nf_b, nb_act, nb_act).noalias() += Goo_tls[t];
    G_vv_beta_.noalias() += Gvv_tls[t];
  }
}

void OMP3::build_generalized_fock() {
  const bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);

  const Eigen::MatrixXd& Cao = scf_.C_alpha.leftCols(na_);
  const Eigen::MatrixXd& Cav = scf_.C_alpha.rightCols(va_);
  Eigen::MatrixXd Z_mat_a = Eigen::MatrixXd::Zero(va_, na_);

  const Eigen::MatrixXd& Cbo = scf_.C_beta.leftCols(nb_);
  const Eigen::MatrixXd& Cbv = scf_.C_beta.rightCols(vb_);
  Eigen::MatrixXd Z_mat_b = Eigen::MatrixXd::Zero(vb_, nb_);

  auto* t_aa_dense = t2_aa_.get_block(0, 0, 0, 0);
  if (!t_aa_dense) return;

  const int n_aux = static_cast<int>(scf_.L_mat.cols());
  const bool is_ex = (config_.eri_method == "exact");
  const int nf_a = std::clamp(n_frozen_, 0, na_);
  const int nf_b = std::clamp(n_frozen_, 0, nb_);
  const int na_act = na_ - nf_a;
  const int nb_act = nb_ - nf_b;
  const double f_ladd = is_restricted ? 0.25 : 0.125;

  if (!is_ex) {
    auto make_bov_view = [&](const Eigen::MatrixXd& B_ov, int no, int nv) {
      return varray_view<double>(
          {(len_type)no, (len_type)nv, (len_type)n_aux},
          const_cast<double*>(B_ov.data()),
          {(stride_type)nv, 1, (stride_type)(no * nv)}
      );
    };
    auto make_bvv_view = [&](const Eigen::MatrixXd& B_vv, int nv) {
      return varray_view<double>(
          {(len_type)nv, (len_type)nv, (len_type)n_aux},
          const_cast<double*>(B_vv.data()),
          {(stride_type)nv, 1, (stride_type)(nv * nv)}
      );
    };
    auto slice_boo_act = [&](const Eigen::MatrixXd& Boo, int n_full, int nf) -> Eigen::MatrixXd {
      const int n_act = n_full - nf;
      if (nf == 0) return Boo;
      Eigen::MatrixXd out(static_cast<Eigen::Index>(n_act) * n_act, n_aux);
      #pragma omp parallel for schedule(static)
      for (int P = 0; P < n_aux; ++P) {
        const double* s = Boo.col(P).data();
        double* d = out.col(P).data();
        for (int j = 0; j < n_act; ++j)
          for (int i = 0; i < n_act; ++i)
            d[i + j * n_act] = s[(nf + i) + (nf + j) * n_full];
      }
      return out;
    };

    auto accum_vvvv_ladder_act = [&](
        double scale,
        const Eigen::Tensor<double, 4>& T_bra, int no1, int no2, int nv1, int nv2,
        const Eigen::Tensor<double, 4>& T_ket,
        const Eigen::MatrixXd& B_ov_m, int nm, int nv_m,
        const Eigen::MatrixXd& B_vv_r, int nv_r,
        bool swap_ab_mode,
        Eigen::MatrixXd& Z_target_act)
    {
      if (nm <= 0 || no1 <= 0 || no2 <= 0 || nv1 <= 0 || nv2 <= 0 || n_aux <= 0) return;
      constexpr size_t kMaxChunkBytes = 64ULL * 1024ULL * 1024ULL;
      const size_t slice_bytes = static_cast<size_t>(nv_m) * nv_r * nv_r * sizeof(double);
      const int mb_max = (slice_bytes > 0)
          ? std::max(1, static_cast<int>(kMaxChunkBytes / slice_bytes))
          : nm;

      auto& T_bra_nc = const_cast<Eigen::Tensor<double, 4>&>(T_bra);
      auto& T_ket_nc = const_cast<Eigen::Tensor<double, 4>&>(T_ket);
      TBLIS_VIEW_4D(t_Tbra, T_bra_nc, no1, no2, nv1, nv2);
      TBLIS_VIEW_4D(t_Tket, T_ket_nc, no1, no2, nv1, nv2);

      for (int m0 = 0; m0 < nm; m0 += mb_max) {
        const int mb = std::min(mb_max, nm - m0);
        Eigen::MatrixXd B_chunk_buf;
        const Eigen::MatrixXd* B_chunk_ptr = &B_ov_m;
        if (m0 != 0 || mb != nm) {
          B_chunk_buf = B_ov_m.middleRows(static_cast<Eigen::Index>(m0) * nv_m,
                                          static_cast<Eigen::Index>(mb) * nv_m);
          B_chunk_ptr = &B_chunk_buf;
        }
        Eigen::MatrixXd M_chunk = parallel_gemm_abt_mp3(*B_chunk_ptr, B_vv_r);
        auto t_Vchunk = make_mat_view_4d_mp3(M_chunk, mb, nv_m, nv_r, nv_r);

        Eigen::Tensor<double, 4> Y_chunk(no1, no2, mb, nv_r);
        Y_chunk.setZero();
        TBLIS_VIEW_4D(t_Ychunk, Y_chunk, no1, no2, mb, nv_r);
        TBLIS_VIEW_2D(t_Zslice, Z_target_act.col(m0).data(), nv_m, mb);

        if (!swap_ab_mode) {
          tblis::mult<double>(1.0, t_Tbra, "ijab", t_Vchunk, "mafb", 0.0, t_Ychunk, "ijmf");
          tblis::mult<double>(scale, t_Tket, "ijxf", t_Ychunk, "ijmf", 1.0, t_Zslice, "xm");
        } else {
          tblis::mult<double>(1.0, t_Tbra, "ijab", t_Vchunk, "mbea", 0.0, t_Ychunk, "ijme");
          tblis::mult<double>(scale, t_Tket, "ijex", t_Ychunk, "ijme", 1.0, t_Zslice, "xm");
        }
      }
    };

    if (na_act > 0 && va_ > 0) {
      Eigen::MatrixXd Z_act_a = Eigen::MatrixXd::Zero(va_, na_act);
      TBLIS_VIEW_2D(t_Za, Z_act_a.data(), va_, na_act);

      Eigen::MatrixXd B_ij_a = slice_boo_act(B_oo_P_alpha_, na_, nf_a);
      Eigen::MatrixXd B_ia_a = (nf_a > 0)
          ? B_ia_P_alpha_.bottomRows(static_cast<Eigen::Index>(na_act) * va_).eval()
          : B_ia_P_alpha_;
      const Eigen::MatrixXd& B_ab_a = B_vv_P_alpha_;

      auto t_Bov_a = make_bov_view(B_ia_a, na_act, va_);
      auto t_Bvv_a = make_bvv_view(B_ab_a, va_);

      Eigen::MatrixXd M_ooov_aa = parallel_gemm_abt_mp3(B_ij_a, B_ia_a);
      auto t_Vooov_aa = make_mat_view_4d_mp3(M_ooov_aa, na_act, na_act, na_act, va_);

      {
        Eigen::Tensor<double, 4> Teff_aa(na_act, na_act, va_, va_);
        #pragma omp parallel for collapse(4) schedule(static)
        for (int i = 0; i < na_act; ++i)
          for (int j = 0; j < na_act; ++j)
            for (int a = 0; a < va_; ++a)
              for (int b = 0; b < va_; ++b) {
                const double t_val = (*t_aa_dense)(nf_a + i, a, nf_a + j, b);
                if (is_restricted) {
                  const double t_t = 2.0 * t_val - (*t_aa_dense)(nf_a + i, b, nf_a + j, a);
                  const double l_t = 2.0 * L2_aa_(nf_a + i, nf_a + j, a, b) - L2_aa_(nf_a + i, nf_a + j, b, a);
                  Teff_aa(i, j, a, b) = t_t + l_t;
                } else {
                  Teff_aa(i, j, a, b) = t_val + L2_aa_(nf_a + i, nf_a + j, a, b);
                }
              }
        TBLIS_VIEW_4D(t_Teff_aa, Teff_aa, na_act, na_act, va_, va_);
        Eigen::Tensor<double, 3> X_teff_a(na_act, va_, n_aux);
        X_teff_a.setZero();
        TBLIS_VIEW_3D(t_Xteff_a, X_teff_a.data(), na_act, va_, n_aux);
        tblis::mult<double>(1.0, t_Teff_aa, "ikbc", t_Bov_a, "kcP", 0.0, t_Xteff_a, "ibP");
        tblis::mult<double>(1.0, t_Bvv_a, "abP", t_Xteff_a, "ibP", 0.0, t_Za, "ai");
        tblis::mult<double>(-1.0, t_Vooov_aa, "jikc", t_Teff_aa, "jkac", 1.0, t_Za, "ai");
      }

      Eigen::Tensor<double, 4> T2_aa(na_act, na_act, va_, va_);
      Eigen::Tensor<double, 4> T2_tilde_aa;
      if (is_restricted) T2_tilde_aa.resize(na_act, na_act, va_, va_);
      #pragma omp parallel for collapse(4) schedule(static)
      for (int i = 0; i < na_act; ++i)
        for (int j = 0; j < na_act; ++j)
          for (int a = 0; a < va_; ++a)
            for (int b = 0; b < va_; ++b) {
              const double tv = (*t_aa_dense)(nf_a + i, a, nf_a + j, b);
              T2_aa(i, j, a, b) = tv;
              if (is_restricted) {
                T2_tilde_aa(i, j, a, b) = 2.0 * tv - (*t_aa_dense)(nf_a + i, b, nf_a + j, a);
              }
            }
      Eigen::Tensor<double, 4>& T2t_ref = is_restricted ? T2_tilde_aa : T2_aa;
      TBLIS_VIEW_4D(t_T2, T2_aa, na_act, na_act, va_, va_);
      TBLIS_VIEW_4D(t_T2t, T2t_ref, na_act, na_act, va_, va_);

      const bool has_b_act = (!is_restricted && nb_act > 0 && vb_ > 0);
      Eigen::Tensor<double, 4> T2_bb_act, T2_ab_act;
      if (has_b_act) {
        auto* t2_bb_d = t2_bb_.get_block(0, 0, 0, 0);
        auto* t2_ab_d = t2_ab_.get_block(0, 0, 0, 0);
        T2_bb_act = Eigen::Tensor<double, 4>(nb_act, nb_act, vb_, vb_); T2_bb_act.setZero();
        T2_ab_act = Eigen::Tensor<double, 4>(na_act, nb_act, va_, vb_); T2_ab_act.setZero();
        if (t2_bb_d) {
          #pragma omp parallel for collapse(4) schedule(static)
          for (int i = 0; i < nb_act; ++i)
            for (int j = 0; j < nb_act; ++j)
              for (int a = 0; a < vb_; ++a)
                for (int b = 0; b < vb_; ++b)
                  T2_bb_act(i, j, a, b) = (*t2_bb_d)(nf_b + i, nf_b + j, a, b);
        }
        if (t2_ab_d) {
          #pragma omp parallel for collapse(4) schedule(static)
          for (int i = 0; i < na_act; ++i)
            for (int j = 0; j < nb_act; ++j)
              for (int a = 0; a < va_; ++a)
                for (int b = 0; b < vb_; ++b)
                  T2_ab_act(i, j, a, b) = (*t2_ab_d)(nf_a + i, nf_b + j, a, b);
        }
      }

      accum_vvvv_ladder_act(
          -4.0 * f_ladd,
          T2t_ref, na_act, na_act, va_, va_, T2_aa,
          B_ia_a, na_act, va_, B_ab_a, va_,
          false, Z_act_a);

      {
        Eigen::Tensor<double, 4> G_oooo_aa(na_act, na_act, na_act, na_act);
        G_oooo_aa.setZero();
        TBLIS_VIEW_4D(t_Goooo_aa, G_oooo_aa, na_act, na_act, na_act, na_act);
        tblis::mult<double>(f_ladd, t_T2t, "ijab", t_T2, "mnab", 0.0, t_Goooo_aa, "minj");
        tblis::mult<double>(4.0, t_Goooo_aa, "minj", t_Vooov_aa, "njix", 1.0, t_Za, "xm");
      }

      {
        Eigen::Tensor<double, 4> G_ovov_aa(na_act, va_, na_act, va_);
        G_ovov_aa.setZero();
        TBLIS_VIEW_4D(t_Govov_aa, G_ovov_aa, na_act, va_, na_act, va_);
        if (is_restricted) {
          tblis::mult<double>( 0.50, t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Govov_aa, "iakc");
          tblis::mult<double>(-0.25, t_T2, "kjbc", t_T2t, "ijab", 1.0, t_Govov_aa, "iakc");
          tblis::mult<double>( 0.50, t_T2, "miea", t_T2t, "mkec", 1.0, t_Govov_aa, "iakc");
          tblis::mult<double>(-0.25, t_T2, "miae", t_T2t, "mkec", 1.0, t_Govov_aa, "iakc");
        } else {
          tblis::mult<double>(0.5, t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Govov_aa, "iakc");
          if (has_b_act) {
            TBLIS_VIEW_4D(t_Tab_tmp, T2_ab_act, na_act, nb_act, va_, vb_);
            tblis::mult<double>(0.5, t_Tab_tmp, "kjcb", t_Tab_tmp, "ijab", 1.0, t_Govov_aa, "iakc");
          }
        }
        Eigen::Tensor<double, 3> W_ov_a(na_act, va_, n_aux);
        W_ov_a.setZero();
        TBLIS_VIEW_3D(t_Wov_a, W_ov_a.data(), na_act, va_, n_aux);
        if (is_restricted) {
          tblis::mult<double>(1.0, t_Govov_aa, "makc", t_Bov_a, "kcP", 0.0, t_Wov_a, "maP");
          tblis::mult<double>(2.0, t_Bvv_a, "xaP", t_Wov_a, "maP", 1.0, t_Za, "xm");
          tblis::mult<double>(-2.0, t_Govov_aa, "ixkc", t_Vooov_aa, "imkc", 1.0, t_Za, "xm");
        } else {
          tblis::mult<double>(1.0, t_Govov_aa, "makc", t_Bov_a, "kcP", 0.0, t_Wov_a, "maP");
          tblis::mult<double>(1.0, t_Govov_aa, "iamc", t_Bov_a, "iaP", 1.0, t_Wov_a, "mcP");
          tblis::mult<double>(1.0, t_Bvv_a, "xaP", t_Wov_a, "maP", 1.0, t_Za, "xm");
          tblis::mult<double>(-1.0, t_Govov_aa, "ixkc", t_Vooov_aa, "imkc", 1.0, t_Za, "xm");
          tblis::mult<double>(-1.0, t_Govov_aa, "iakx", t_Vooov_aa, "kmia", 1.0, t_Za, "xm");
        }
      }

      {
        Eigen::Tensor<double, 4> G_oovv_aa(na_act, na_act, va_, va_);
        G_oovv_aa.setZero();
        TBLIS_VIEW_4D(t_Goovv_aa, G_oovv_aa, na_act, na_act, va_, va_);
        if (is_restricted) {
          tblis::mult<double>(-0.25, t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Goovv_aa, "ikac");
          tblis::mult<double>(-0.25, t_T2, "miea", t_T2t, "mkec", 1.0, t_Goovv_aa, "ikac");
          tblis::mult<double>(-0.50, t_T2, "kmec", t_T2t, "imea", 1.0, t_Goovv_aa, "ikac");
        } else {
          tblis::mult<double>(-0.5, t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Goovv_aa, "ikac");
          if (has_b_act) {
            TBLIS_VIEW_4D(t_Tab_tmp, T2_ab_act, na_act, nb_act, va_, vb_);
            tblis::mult<double>(-0.5, t_Tab_tmp, "kjcb", t_Tab_tmp, "ijab", 1.0, t_Goovv_aa, "ikac");
          }
        }
        Eigen::Tensor<double, 3> U_oo_a(na_act, na_act, n_aux);
        U_oo_a.setZero();
        TBLIS_VIEW_3D(t_Uoo_a, U_oo_a.data(), na_act, na_act, n_aux);
        tblis::mult<double>(1.0, t_Goovv_aa, "mkac", t_Bvv_a, "caP", 0.0, t_Uoo_a, "mkP");
        tblis::mult<double>(1.0, t_Bov_a, "kxP", t_Uoo_a, "mkP", 1.0, t_Za, "xm");
        tblis::mult<double>(1.0, t_Goovv_aa, "imac", t_Bvv_a, "acP", 0.0, t_Uoo_a, "imP");
        tblis::mult<double>(1.0, t_Bov_a, "ixP", t_Uoo_a, "imP", 1.0, t_Za, "xm");
        tblis::mult<double>(-1.0, t_Goovv_aa, "ikxc", t_Vooov_aa, "ikmc", 1.0, t_Za, "xm");
        tblis::mult<double>(-1.0, t_Goovv_aa, "ikax", t_Vooov_aa, "ikma", 1.0, t_Za, "xm");
      }

      M_ooov_aa.resize(0, 0);
      T2_tilde_aa.resize(0, 0, 0, 0);

      if (has_b_act) {
        Eigen::MatrixXd Z_act_b = Eigen::MatrixXd::Zero(vb_, nb_act);
        TBLIS_VIEW_2D(t_Zb, Z_act_b.data(), vb_, nb_act);
        TBLIS_VIEW_4D(t_T2_bb, T2_bb_act, nb_act, nb_act, vb_, vb_);
        TBLIS_VIEW_4D(t_T2_ab, T2_ab_act, na_act, nb_act, va_, vb_);

        Eigen::MatrixXd B_ij_b = slice_boo_act(B_oo_P_beta_, nb_, nf_b);
        Eigen::MatrixXd B_ia_b = (nf_b > 0)
            ? B_ia_P_beta_.bottomRows(static_cast<Eigen::Index>(nb_act) * vb_).eval()
            : B_ia_P_beta_;
        const Eigen::MatrixXd& B_ab_b = B_vv_P_beta_;

        auto t_Bov_b = make_bov_view(B_ia_b, nb_act, vb_);
        auto t_Bvv_b = make_bvv_view(B_ab_b, vb_);

        Eigen::MatrixXd M_ooov_bb = parallel_gemm_abt_mp3(B_ij_b, B_ia_b);
        Eigen::MatrixXd M_ooov_aa_bb = parallel_gemm_abt_mp3(B_ij_a, B_ia_b);
        Eigen::MatrixXd M_ooov_bb_aa = parallel_gemm_abt_mp3(B_ij_b, B_ia_a);

        auto t_Vooov_bb = make_mat_view_4d_mp3(M_ooov_bb, nb_act, nb_act, nb_act, vb_);
        auto t_Vooov_aa_bb = make_mat_view_4d_mp3(M_ooov_aa_bb, na_act, na_act, nb_act, vb_);
        auto t_Vooov_bb_aa = make_mat_view_4d_mp3(M_ooov_bb_aa, nb_act, nb_act, na_act, va_);

        {
          Eigen::Tensor<double, 3> X_b(nb_act, vb_, n_aux);
          X_b.setZero();
          TBLIS_VIEW_3D(t_Xb, X_b.data(), nb_act, vb_, n_aux);

          {
            Eigen::Tensor<double, 4> Teff_bb(nb_act, nb_act, vb_, vb_);
            #pragma omp parallel for collapse(4) schedule(static)
            for (int i = 0; i < nb_act; ++i)
              for (int j = 0; j < nb_act; ++j)
                for (int a = 0; a < vb_; ++a)
                  for (int b = 0; b < vb_; ++b)
                    Teff_bb(i, j, a, b) = T2_bb_act(i, j, a, b) + L2_bb_(nf_b + i, nf_b + j, a, b);
            TBLIS_VIEW_4D(t_Teff_bb, Teff_bb, nb_act, nb_act, vb_, vb_);
            tblis::mult<double>(1.0, t_Teff_bb, "ikbc", t_Bov_b, "kcP", 0.0, t_Xb, "ibP");
            tblis::mult<double>(-1.0, t_Vooov_bb, "jikc", t_Teff_bb, "jkac", 1.0, t_Zb, "ai");
          }

          {
            Eigen::Tensor<double, 4> Teff_ab(na_act, nb_act, va_, vb_);
            #pragma omp parallel for collapse(4) schedule(static)
            for (int i = 0; i < na_act; ++i)
              for (int j = 0; j < nb_act; ++j)
                for (int a = 0; a < va_; ++a)
                  for (int b = 0; b < vb_; ++b)
                    Teff_ab(i, j, a, b) = T2_ab_act(i, j, a, b) + L2_ab_(nf_a + i, nf_b + j, a, b);
            TBLIS_VIEW_4D(t_Teff_ab, Teff_ab, na_act, nb_act, va_, vb_);

            Eigen::Tensor<double, 3> X_ab_a(na_act, va_, n_aux);
            X_ab_a.setZero();
            TBLIS_VIEW_3D(t_Xab_a, X_ab_a.data(), na_act, va_, n_aux);
            tblis::mult<double>(1.0, t_Teff_ab, "ikbc", t_Bov_b, "kcP", 0.0, t_Xab_a, "ibP");
            tblis::mult<double>(1.0, t_Bvv_a, "abP", t_Xab_a, "ibP", 1.0, t_Za, "ai");
            tblis::mult<double>(-1.0, t_Vooov_aa_bb, "jikc", t_Teff_ab, "jkac", 1.0, t_Za, "ai");

            tblis::mult<double>(1.0, t_Teff_ab, "kicb", t_Bov_a, "kcP", 1.0, t_Xb, "ibP");
            tblis::mult<double>(-1.0, t_Vooov_bb_aa, "jikc", t_Teff_ab, "kjca", 1.0, t_Zb, "ai");
          }

          tblis::mult<double>(1.0, t_Bvv_b, "abP", t_Xb, "ibP", 1.0, t_Zb, "ai");
        }

        accum_vvvv_ladder_act(-0.5, T2_bb_act, nb_act, nb_act, vb_, vb_, T2_bb_act,
                              B_ia_b, nb_act, vb_, B_ab_b, vb_, false, Z_act_b);
        accum_vvvv_ladder_act(-1.0, T2_ab_act, na_act, nb_act, va_, vb_, T2_ab_act,
                              B_ia_a, na_act, va_, B_ab_b, vb_, false, Z_act_a);
        accum_vvvv_ladder_act(-1.0, T2_ab_act, na_act, nb_act, va_, vb_, T2_ab_act,
                              B_ia_b, nb_act, vb_, B_ab_a, va_, true, Z_act_b);

        {
          Eigen::Tensor<double, 4> G_oooo_bb(nb_act, nb_act, nb_act, nb_act); G_oooo_bb.setZero();
          Eigen::Tensor<double, 4> G_oooo_ab(na_act, na_act, nb_act, nb_act); G_oooo_ab.setZero();
          TBLIS_VIEW_4D(t_Goooo_bb, G_oooo_bb, nb_act, nb_act, nb_act, nb_act);
          TBLIS_VIEW_4D(t_Goooo_ab, G_oooo_ab, na_act, na_act, nb_act, nb_act);
          tblis::mult<double>(0.125, t_T2_bb, "ijab", t_T2_bb, "mnab", 0.0, t_Goooo_bb, "minj");
          tblis::mult<double>(0.500, t_T2_ab, "ijab", t_T2_ab, "mnab", 0.0, t_Goooo_ab, "minj");

          tblis::mult<double>(1.0, t_Goooo_ab, "minj", t_Vooov_bb_aa, "njix", 1.0, t_Za, "xm");
          tblis::mult<double>(1.0, t_Goooo_ab, "imnj", t_Vooov_bb_aa, "njix", 1.0, t_Za, "xm");
          tblis::mult<double>(1.0, t_Goooo_ab, "inmj", t_Vooov_aa_bb, "injx", 1.0, t_Zb, "xm");
          tblis::mult<double>(1.0, t_Goooo_ab, "injm", t_Vooov_aa_bb, "injx", 1.0, t_Zb, "xm");

          tblis::mult<double>(1.0, t_Goooo_bb, "minj", t_Vooov_bb, "njix", 1.0, t_Zb, "xm");
          tblis::mult<double>(1.0, t_Goooo_bb, "imnj", t_Vooov_bb, "njix", 1.0, t_Zb, "xm");
          tblis::mult<double>(1.0, t_Goooo_bb, "inmj", t_Vooov_bb, "injx", 1.0, t_Zb, "xm");
          tblis::mult<double>(1.0, t_Goooo_bb, "injm", t_Vooov_bb, "injx", 1.0, t_Zb, "xm");
        }

        {
          Eigen::Tensor<double, 4> G_ovov_bb(nb_act, vb_, nb_act, vb_); G_ovov_bb.setZero();
          TBLIS_VIEW_4D(t_Govov_bb, G_ovov_bb, nb_act, vb_, nb_act, vb_);
          tblis::mult<double>(0.5, t_T2_bb, "kjcb", t_T2_bb, "ijab", 0.0, t_Govov_bb, "iakc");
          tblis::mult<double>(0.5, t_T2_ab, "miea", t_T2_ab, "mkec", 1.0, t_Govov_bb, "iakc");

          Eigen::Tensor<double, 3> W_ov_b(nb_act, vb_, n_aux); W_ov_b.setZero();
          TBLIS_VIEW_3D(t_Wov_b, W_ov_b.data(), nb_act, vb_, n_aux);
          tblis::mult<double>(1.0, t_Govov_bb, "makc", t_Bov_b, "kcP", 0.0, t_Wov_b, "maP");
          tblis::mult<double>(1.0, t_Govov_bb, "iamc", t_Bov_b, "iaP", 1.0, t_Wov_b, "mcP");
          tblis::mult<double>(1.0, t_Bvv_b, "xaP", t_Wov_b, "maP", 1.0, t_Zb, "xm");
          tblis::mult<double>(-1.0, t_Govov_bb, "ixkc", t_Vooov_bb, "imkc", 1.0, t_Zb, "xm");
          tblis::mult<double>(-1.0, t_Govov_bb, "iakx", t_Vooov_bb, "kmia", 1.0, t_Zb, "xm");
        }

        {
          Eigen::Tensor<double, 4> G_ovov_ab(na_act, va_, nb_act, vb_); G_ovov_ab.setZero();
          TBLIS_VIEW_4D(t_Govov_ab, G_ovov_ab, na_act, va_, nb_act, vb_);
          tblis::mult<double>(1.0, t_T2, "miea", t_T2_ab, "mkec", 0.0, t_Govov_ab, "iakc");
          tblis::mult<double>(1.0, t_T2_bb, "kjcb", t_T2_ab, "ijab", 1.0, t_Govov_ab, "iakc");

          Eigen::Tensor<double, 3> W_ov_b(nb_act, vb_, n_aux); W_ov_b.setZero();
          TBLIS_VIEW_3D(t_Wov_b, W_ov_b.data(), nb_act, vb_, n_aux);
          tblis::mult<double>(1.0, t_Govov_ab, "iamc", t_Bov_a, "iaP", 0.0, t_Wov_b, "mcP");
          tblis::mult<double>(1.0, t_Bvv_b, "xaP", t_Wov_b, "maP", 1.0, t_Zb, "xm");

          Eigen::Tensor<double, 3> W_ab_a(na_act, va_, n_aux); W_ab_a.setZero();
          TBLIS_VIEW_3D(t_Wab_a, W_ab_a.data(), na_act, va_, n_aux);
          tblis::mult<double>(1.0, t_Govov_ab, "makc", t_Bov_b, "kcP", 0.0, t_Wab_a, "maP");
          tblis::mult<double>(1.0, t_Bvv_a, "xaP", t_Wab_a, "maP", 1.0, t_Za, "xm");

          tblis::mult<double>(-1.0, t_Govov_ab, "ixkc", t_Vooov_aa_bb, "imkc", 1.0, t_Za, "xm");
          tblis::mult<double>(-1.0, t_Govov_ab, "iakx", t_Vooov_bb_aa, "kmia", 1.0, t_Zb, "xm");
        }

        {
          Eigen::Tensor<double, 4> G_oovv_bb(nb_act, nb_act, vb_, vb_); G_oovv_bb.setZero();
          TBLIS_VIEW_4D(t_Goovv_bb, G_oovv_bb, nb_act, nb_act, vb_, vb_);
          tblis::mult<double>(-0.5, t_T2_bb, "kjcb", t_T2_bb, "ijab", 0.0, t_Goovv_bb, "ikac");
          tblis::mult<double>(-0.5, t_T2_ab, "miea", t_T2_ab, "mkec", 1.0, t_Goovv_bb, "ikac");

          Eigen::Tensor<double, 3> U_oo_b(nb_act, nb_act, n_aux); U_oo_b.setZero();
          TBLIS_VIEW_3D(t_Uoo_b, U_oo_b.data(), nb_act, nb_act, n_aux);
          tblis::mult<double>(1.0, t_Goovv_bb, "mkac", t_Bvv_b, "caP", 0.0, t_Uoo_b, "mkP");
          tblis::mult<double>(1.0, t_Bov_b, "kxP", t_Uoo_b, "mkP", 1.0, t_Zb, "xm");
          tblis::mult<double>(1.0, t_Goovv_bb, "imac", t_Bvv_b, "acP", 0.0, t_Uoo_b, "imP");
          tblis::mult<double>(1.0, t_Bov_b, "ixP", t_Uoo_b, "imP", 1.0, t_Zb, "xm");
          tblis::mult<double>(-1.0, t_Goovv_bb, "ikxc", t_Vooov_bb, "ikmc", 1.0, t_Zb, "xm");
          tblis::mult<double>(-1.0, t_Goovv_bb, "ikax", t_Vooov_bb, "ikma", 1.0, t_Zb, "xm");
        }

        {
          Eigen::Tensor<double, 4> G_oovv_ab_ex(na_act, na_act, vb_, vb_); G_oovv_ab_ex.setZero();
          TBLIS_VIEW_4D(t_Goovv_ab_ex, G_oovv_ab_ex, na_act, na_act, vb_, vb_);
          tblis::mult<double>(-0.5, t_T2_ab, "kmec", t_T2_ab, "imea", 0.0, t_Goovv_ab_ex, "ikac");

          Eigen::Tensor<double, 3> U_ab_ex(na_act, na_act, n_aux); U_ab_ex.setZero();
          TBLIS_VIEW_3D(t_Uab_ex, U_ab_ex.data(), na_act, na_act, n_aux);
          tblis::mult<double>(1.0, t_Goovv_ab_ex, "mkac", t_Bvv_b, "acP", 0.0, t_Uab_ex, "mkP");
          tblis::mult<double>(1.0, t_Bov_a, "kxP", t_Uab_ex, "mkP", 1.0, t_Za, "xm");
          tblis::mult<double>(1.0, t_Bov_a, "ixP", t_Uab_ex, "imP", 1.0, t_Za, "xm");
          tblis::mult<double>(-1.0, t_Goovv_ab_ex, "ikxc", t_Vooov_aa_bb, "ikmc", 1.0, t_Zb, "xm");
          tblis::mult<double>(-1.0, t_Goovv_ab_ex, "ikax", t_Vooov_aa_bb, "ikma", 1.0, t_Zb, "xm");
        }

        {
          Eigen::Tensor<double, 4> G_oovv_ba_ex(nb_act, nb_act, va_, va_); G_oovv_ba_ex.setZero();
          TBLIS_VIEW_4D(t_Goovv_ba_ex, G_oovv_ba_ex, nb_act, nb_act, va_, va_);
          tblis::mult<double>(-0.5, t_T2_ab, "mkce", t_T2_ab, "miae", 0.0, t_Goovv_ba_ex, "ikac");

          Eigen::Tensor<double, 3> U_ba_ex(nb_act, nb_act, n_aux); U_ba_ex.setZero();
          TBLIS_VIEW_3D(t_Uba_ex, U_ba_ex.data(), nb_act, nb_act, n_aux);
          tblis::mult<double>(1.0, t_Goovv_ba_ex, "mkac", t_Bvv_a, "acP", 0.0, t_Uba_ex, "mkP");
          tblis::mult<double>(1.0, t_Bov_b, "kxP", t_Uba_ex, "mkP", 1.0, t_Zb, "xm");
          tblis::mult<double>(1.0, t_Bov_b, "ixP", t_Uba_ex, "imP", 1.0, t_Zb, "xm");
          tblis::mult<double>(-1.0, t_Goovv_ba_ex, "ikxc", t_Vooov_bb_aa, "ikmc", 1.0, t_Za, "xm");
          tblis::mult<double>(-1.0, t_Goovv_ba_ex, "ikax", t_Vooov_bb_aa, "ikma", 1.0, t_Za, "xm");
        }

        Z_mat_b.rightCols(nb_act) = Z_act_b;
      }

      Z_mat_a.rightCols(na_act) = Z_act_a;
      if (is_restricted) Z_mat_b = Z_mat_a;
    }
  } else {
    Eigen::Tensor<double, 4> Teff_aa(na_, na_, va_, va_);
    Eigen::Tensor<double, 4> T2_aa(na_, na_, va_, va_);
    Eigen::Tensor<double, 4> T2_tilde_aa(na_, na_, va_, va_);
    #pragma omp parallel for collapse(4) schedule(static)
    for (int i = 0; i < na_; ++i)
      for (int j = 0; j < na_; ++j)
        for (int a = 0; a < va_; ++a)
          for (int b = 0; b < va_; ++b) {
            const double t_val = (*t_aa_dense)(i, a, j, b);
            T2_aa(i, j, a, b) = t_val;
            if (is_restricted) {
              const double t_tilde = 2.0 * t_val - (*t_aa_dense)(i, b, j, a);
              const double l_tilde = 2.0 * L2_aa_(i, j, a, b) - L2_aa_(i, j, b, a);
              Teff_aa(i, j, a, b) = t_tilde + l_tilde;
              T2_tilde_aa(i, j, a, b) = t_tilde;
            } else {
              Teff_aa(i, j, a, b) = t_val + L2_aa_(i, j, a, b);
              T2_tilde_aa(i, j, a, b) = t_val;
            }
          }
    TBLIS_VIEW_4D(t_Teff_aa, Teff_aa, na_, na_, va_, va_);
    TBLIS_VIEW_4D(t_T2, T2_aa, na_, na_, va_, va_);
    TBLIS_VIEW_4D(t_T2t, T2_tilde_aa, na_, na_, va_, va_);
    TBLIS_VIEW_2D(t_Zmat_a, Z_mat_a.data(), va_, na_);

    Eigen::Tensor<double, 4> G_ovov_aa(na_, va_, na_, va_); G_ovov_aa.setZero();
    Eigen::Tensor<double, 4> G_oovv_aa(na_, na_, va_, va_); G_oovv_aa.setZero();
    TBLIS_VIEW_4D(t_Govov_aa, G_ovov_aa, na_, va_, na_, va_);
    TBLIS_VIEW_4D(t_Goovv_aa, G_oovv_aa, na_, na_, va_, va_);

    if (is_restricted) {
      tblis::mult<double>( 0.50, t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Govov_aa, "iakc");
      tblis::mult<double>(-0.25, t_T2, "kjbc", t_T2t, "ijab", 1.0, t_Govov_aa, "iakc");
      tblis::mult<double>( 0.50, t_T2, "miea", t_T2t, "mkec", 1.0, t_Govov_aa, "iakc");
      tblis::mult<double>(-0.25, t_T2, "miae", t_T2t, "mkec", 1.0, t_Govov_aa, "iakc");
      tblis::mult<double>(-0.25, t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Goovv_aa, "ikac");
      tblis::mult<double>(-0.25, t_T2, "miea", t_T2t, "mkec", 1.0, t_Goovv_aa, "ikac");
      tblis::mult<double>(-0.50, t_T2, "kmec", t_T2t, "imea", 1.0, t_Goovv_aa, "ikac");
    } else {
      tblis::mult<double>( 0.5, t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Govov_aa, "iakc");
      tblis::mult<double>(-0.5, t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Goovv_aa, "ikac");
      if (nb_ > 0 && vb_ > 0) {
        auto* t2_ab_dense_tmp = t2_ab_.get_block(0, 0, 0, 0);
        if (t2_ab_dense_tmp) {
          TBLIS_VIEW_4D(t_T2_ab_tmp, (*t2_ab_dense_tmp), na_, nb_, va_, vb_);
          tblis::mult<double>( 0.5, t_T2_ab_tmp, "kjcb", t_T2_ab_tmp, "ijab", 1.0, t_Govov_aa, "iakc");
          tblis::mult<double>(-0.5, t_T2_ab_tmp, "kjcb", t_T2_ab_tmp, "ijab", 1.0, t_Goovv_aa, "ikac");
        }
      }
    }

    const auto& eri_ao_ref = integrals_->compute_eri();
    Eigen::Tensor<double, 4> half_ov_a = integrals::ERITransformer::transform_half_bra(eri_ao_ref, Cao, Cav, nbf_, na_, va_);
    Eigen::Tensor<double, 4> half_oo_a = integrals::ERITransformer::transform_half_bra(eri_ao_ref, Cao, Cao, nbf_, na_, na_);
    Eigen::Tensor<double, 4> V_ovvv_aa = integrals::ERITransformer::transform_finish_ket(half_ov_a, Cav, Cav, nbf_, na_, va_, va_, va_);
    Eigen::Tensor<double, 4> V_ooov_aa = integrals::ERITransformer::transform_finish_ket(half_oo_a, Cao, Cav, nbf_, na_, na_, na_, va_);
    if (is_restricted || nb_ == 0 || vb_ == 0) {
      half_ov_a.resize(0, 0, 0, 0);
      half_oo_a.resize(0, 0, 0, 0);
    }

    TBLIS_VIEW_4D(t_Vovvv_aa, V_ovvv_aa, na_, va_, va_, va_);
    TBLIS_VIEW_4D(t_Vooov_aa, V_ooov_aa, na_, na_, na_, va_);

    tblis::mult<double>( 1.0, t_Vovvv_aa, "kcab", t_Teff_aa, "ikbc", 0.0, t_Zmat_a, "ai");
    tblis::mult<double>(-1.0, t_Vooov_aa, "jikc", t_Teff_aa, "jkac", 1.0, t_Zmat_a, "ai");

    Eigen::Tensor<double, 4> Y_ooov_aa(na_, na_, na_, va_); Y_ooov_aa.setZero();
    TBLIS_VIEW_4D(t_Yooov_aa, Y_ooov_aa, na_, na_, na_, va_);
    tblis::mult<double>(1.0, t_T2t, "ijab", t_Vovvv_aa, "mafb", 0.0, t_Yooov_aa, "ijmf");
    tblis::mult<double>(-4.0 * f_ladd, t_T2, "ijxf", t_Yooov_aa, "ijmf", 1.0, t_Zmat_a, "xm");

    Eigen::Tensor<double, 4> G_oooo_aa(na_, na_, na_, na_); G_oooo_aa.setZero();
    TBLIS_VIEW_4D(t_Goooo_aa, G_oooo_aa, na_, na_, na_, na_);
    tblis::mult<double>(f_ladd, t_T2t, "ijab", t_T2, "mnab", 0.0, t_Goooo_aa, "minj");
    tblis::mult<double>(4.0, t_Goooo_aa, "minj", t_Vooov_aa, "njix", 1.0, t_Zmat_a, "xm");

    if (is_restricted) {
      tblis::mult<double>( 2.0, t_Govov_aa, "makc", t_Vovvv_aa, "kcxa", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>(-2.0, t_Govov_aa, "ixkc", t_Vooov_aa, "imkc", 1.0, t_Zmat_a, "xm");
    } else {
      tblis::mult<double>( 1.0, t_Govov_aa, "makc", t_Vovvv_aa, "kcxa", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>(-1.0, t_Govov_aa, "ixkc", t_Vooov_aa, "imkc", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>( 1.0, t_Govov_aa, "iamc", t_Vovvv_aa, "iaxc", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>(-1.0, t_Govov_aa, "iakx", t_Vooov_aa, "kmia", 1.0, t_Zmat_a, "xm");
    }

    tblis::mult<double>( 1.0, t_Goovv_aa, "mkac", t_Vovvv_aa, "kxca", 1.0, t_Zmat_a, "xm");
    tblis::mult<double>( 1.0, t_Goovv_aa, "imac", t_Vovvv_aa, "ixac", 1.0, t_Zmat_a, "xm");
    tblis::mult<double>(-1.0, t_Goovv_aa, "ikxc", t_Vooov_aa, "ikmc", 1.0, t_Zmat_a, "xm");
    tblis::mult<double>(-1.0, t_Goovv_aa, "ikax", t_Vooov_aa, "ikma", 1.0, t_Zmat_a, "xm");

    Teff_aa.resize(0, 0, 0, 0);
    T2_tilde_aa.resize(0, 0, 0, 0);
    G_ovov_aa.resize(0, 0, 0, 0);
    G_oovv_aa.resize(0, 0, 0, 0);

    if (!is_restricted && nb_ > 0 && vb_ > 0) {
      auto* t2_bb_dense = t2_bb_.get_block(0, 0, 0, 0);
      Eigen::Tensor<double, 4> dummy_bb;
      if (!t2_bb_dense) { dummy_bb = Eigen::Tensor<double, 4>(nb_, nb_, vb_, vb_); dummy_bb.setZero(); t2_bb_dense = &dummy_bb; }
      auto* t2_ab_dense = t2_ab_.get_block(0, 0, 0, 0);
      Eigen::Tensor<double, 4> dummy_ab;
      if (!t2_ab_dense) { dummy_ab = Eigen::Tensor<double, 4>(na_, nb_, va_, vb_); dummy_ab.setZero(); t2_ab_dense = &dummy_ab; }

      TBLIS_VIEW_2D(t_Zmat_b, Z_mat_b.data(), vb_, nb_);
      TBLIS_VIEW_4D(t_T2_bb, (*t2_bb_dense), nb_, nb_, vb_, vb_);
      TBLIS_VIEW_4D(t_T2_ab, (*t2_ab_dense), na_, nb_, va_, vb_);

      Eigen::Tensor<double, 4> Teff_bb(nb_, nb_, vb_, vb_);
      Eigen::Tensor<double, 4> Teff_ab(na_, nb_, va_, vb_);
      #pragma omp parallel for collapse(4) schedule(static)
      for (int i = 0; i < nb_; ++i)
        for (int j = 0; j < nb_; ++j)
          for (int a = 0; a < vb_; ++a)
            for (int b = 0; b < vb_; ++b)
              Teff_bb(i, j, a, b) = (*t2_bb_dense)(i, j, a, b) + L2_bb_(i, j, a, b);
      #pragma omp parallel for collapse(4) schedule(static)
      for (int i = 0; i < na_; ++i)
        for (int j = 0; j < nb_; ++j)
          for (int a = 0; a < va_; ++a)
            for (int b = 0; b < vb_; ++b)
              Teff_ab(i, j, a, b) = (*t2_ab_dense)(i, j, a, b) + L2_ab_(i, j, a, b);

      TBLIS_VIEW_4D(t_Teff_bb, Teff_bb, nb_, nb_, vb_, vb_);
      TBLIS_VIEW_4D(t_Teff_ab, Teff_ab, na_, nb_, va_, vb_);

      Eigen::Tensor<double, 4> G_oooo_bb(nb_, nb_, nb_, nb_); G_oooo_bb.setZero();
      Eigen::Tensor<double, 4> G_oooo_ab(na_, na_, nb_, nb_); G_oooo_ab.setZero();
      TBLIS_VIEW_4D(t_Goooo_bb, G_oooo_bb, nb_, nb_, nb_, nb_);
      TBLIS_VIEW_4D(t_Goooo_ab, G_oooo_ab, na_, na_, nb_, nb_);
      tblis::mult<double>(0.125, t_T2_bb, "ijab", t_T2_bb, "mnab", 0.0, t_Goooo_bb, "minj");
      tblis::mult<double>(0.500, t_T2_ab, "ijab", t_T2_ab, "mnab", 0.0, t_Goooo_ab, "minj");

      Eigen::Tensor<double, 4> G_ovov_bb(nb_, vb_, nb_, vb_); G_ovov_bb.setZero();
      Eigen::Tensor<double, 4> G_oovv_bb(nb_, nb_, vb_, vb_); G_oovv_bb.setZero();
      TBLIS_VIEW_4D(t_Govov_bb, G_ovov_bb, nb_, vb_, nb_, vb_);
      TBLIS_VIEW_4D(t_Goovv_bb, G_oovv_bb, nb_, nb_, vb_, vb_);
      tblis::mult<double>( 0.5, t_T2_bb, "kjcb", t_T2_bb, "ijab", 0.0, t_Govov_bb, "iakc");
      tblis::mult<double>( 0.5, t_T2_ab, "miea", t_T2_ab, "mkec", 1.0, t_Govov_bb, "iakc");
      tblis::mult<double>(-0.5, t_T2_bb, "kjcb", t_T2_bb, "ijab", 0.0, t_Goovv_bb, "ikac");
      tblis::mult<double>(-0.5, t_T2_ab, "miea", t_T2_ab, "mkec", 1.0, t_Goovv_bb, "ikac");

      Eigen::Tensor<double, 4> G_ovov_ab(na_, va_, nb_, vb_); G_ovov_ab.setZero();
      Eigen::Tensor<double, 4> G_oovv_ab_ex(na_, na_, vb_, vb_); G_oovv_ab_ex.setZero();
      Eigen::Tensor<double, 4> G_oovv_ba_ex(nb_, nb_, va_, va_); G_oovv_ba_ex.setZero();
      TBLIS_VIEW_4D(t_Govov_ab, G_ovov_ab, na_, va_, nb_, vb_);
      TBLIS_VIEW_4D(t_Goovv_ab_ex, G_oovv_ab_ex, na_, na_, vb_, vb_);
      TBLIS_VIEW_4D(t_Goovv_ba_ex, G_oovv_ba_ex, nb_, nb_, va_, va_);
      tblis::mult<double>( 1.0, t_T2, "miea", t_T2_ab, "mkec", 0.0, t_Govov_ab, "iakc");
      tblis::mult<double>( 1.0, t_T2_bb, "kjcb", t_T2_ab, "ijab", 1.0, t_Govov_ab, "iakc");
      tblis::mult<double>(-0.5, t_T2_ab, "kmec", t_T2_ab, "imea", 0.0, t_Goovv_ab_ex, "ikac");
      tblis::mult<double>(-0.5, t_T2_ab, "mkce", t_T2_ab, "miae", 0.0, t_Goovv_ba_ex, "ikac");

      Eigen::Tensor<double, 4> V_ovvv_ab = integrals::ERITransformer::transform_finish_ket(half_ov_a, Cbv, Cbv, nbf_, na_, va_, vb_, vb_);
      Eigen::Tensor<double, 4> V_ooov_aa_bb = integrals::ERITransformer::transform_finish_ket(half_oo_a, Cbo, Cbv, nbf_, na_, na_, nb_, vb_);
      half_ov_a.resize(0, 0, 0, 0);
      half_oo_a.resize(0, 0, 0, 0);

      auto half_ov_b = integrals::ERITransformer::transform_half_bra(eri_ao_ref, Cbo, Cbv, nbf_, nb_, vb_);
      Eigen::Tensor<double, 4> V_ovvv_bb = integrals::ERITransformer::transform_finish_ket(half_ov_b, Cbv, Cbv, nbf_, nb_, vb_, vb_, vb_);
      Eigen::Tensor<double, 4> V_ovvv_ba = integrals::ERITransformer::transform_finish_ket(half_ov_b, Cav, Cav, nbf_, nb_, vb_, va_, va_);
      half_ov_b.resize(0, 0, 0, 0);

      auto half_oo_b = integrals::ERITransformer::transform_half_bra(eri_ao_ref, Cbo, Cbo, nbf_, nb_, nb_);
      Eigen::Tensor<double, 4> V_ooov_bb = integrals::ERITransformer::transform_finish_ket(half_oo_b, Cbo, Cbv, nbf_, nb_, nb_, nb_, vb_);
      Eigen::Tensor<double, 4> V_ooov_bb_aa = integrals::ERITransformer::transform_finish_ket(half_oo_b, Cao, Cav, nbf_, nb_, nb_, na_, va_);
      half_oo_b.resize(0, 0, 0, 0);

      TBLIS_VIEW_4D(t_Vovvv_bb, V_ovvv_bb, nb_, vb_, vb_, vb_);
      TBLIS_VIEW_4D(t_Vooov_bb, V_ooov_bb, nb_, nb_, nb_, vb_);
      TBLIS_VIEW_4D(t_Vovvv_ab, V_ovvv_ab, na_, va_, vb_, vb_);
      TBLIS_VIEW_4D(t_Vovvv_ba, V_ovvv_ba, nb_, vb_, va_, va_);
      TBLIS_VIEW_4D(t_Vooov_aa_bb, V_ooov_aa_bb, na_, na_, nb_, vb_);
      TBLIS_VIEW_4D(t_Vooov_bb_aa, V_ooov_bb_aa, nb_, nb_, na_, va_);

      tblis::mult<double>( 1.0, t_Vovvv_ba, "kcab", t_Teff_ab, "ikbc", 1.0, t_Zmat_a, "ai");
      tblis::mult<double>(-1.0, t_Vooov_aa_bb, "jikc", t_Teff_ab, "jkac", 1.0, t_Zmat_a, "ai");

      tblis::mult<double>( 1.0, t_Vovvv_bb, "kcab", t_Teff_bb, "ikbc", 0.0, t_Zmat_b, "ai");
      tblis::mult<double>(-1.0, t_Vooov_bb, "jikc", t_Teff_bb, "jkac", 1.0, t_Zmat_b, "ai");
      tblis::mult<double>( 1.0, t_Vovvv_ab, "kcab", t_Teff_ab, "kicb", 1.0, t_Zmat_b, "ai");
      tblis::mult<double>(-1.0, t_Vooov_bb_aa, "jikc", t_Teff_ab, "kjca", 1.0, t_Zmat_b, "ai");

      Eigen::Tensor<double, 4> Y_ooov_bb(nb_, nb_, nb_, vb_); Y_ooov_bb.setZero();
      TBLIS_VIEW_4D(t_Yooov_bb, Y_ooov_bb, nb_, nb_, nb_, vb_);
      tblis::mult<double>( 1.0, t_T2_bb, "ijab", t_Vovvv_bb, "mafb", 0.0, t_Yooov_bb, "ijmf");
      tblis::mult<double>(-0.5, t_T2_bb, "ijxf", t_Yooov_bb, "ijmf", 1.0, t_Zmat_b, "xm");

      Eigen::Tensor<double, 4> Y_ooov_ab_a(na_, nb_, na_, vb_); Y_ooov_ab_a.setZero();
      TBLIS_VIEW_4D(t_Yooov_ab_a, Y_ooov_ab_a, na_, nb_, na_, vb_);
      tblis::mult<double>( 1.0, t_T2_ab, "ijab", t_Vovvv_ab, "mafb", 0.0, t_Yooov_ab_a, "ijmf");
      tblis::mult<double>(-1.0, t_T2_ab, "ijxf", t_Yooov_ab_a, "ijmf", 1.0, t_Zmat_a, "xm");

      Eigen::Tensor<double, 4> Y_ooov_ab_b(na_, nb_, nb_, va_); Y_ooov_ab_b.setZero();
      TBLIS_VIEW_4D(t_Yooov_ab_b, Y_ooov_ab_b, na_, nb_, nb_, va_);
      tblis::mult<double>( 1.0, t_T2_ab, "ijab", t_Vovvv_ba, "mbea", 0.0, t_Yooov_ab_b, "ijme");
      tblis::mult<double>(-1.0, t_T2_ab, "ijex", t_Yooov_ab_b, "ijme", 1.0, t_Zmat_b, "xm");

      tblis::mult<double>(1.0, t_Goooo_ab, "minj", t_Vooov_bb_aa, "njix", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>(1.0, t_Goooo_ab, "imnj", t_Vooov_bb_aa, "njix", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>(1.0, t_Goooo_ab, "inmj", t_Vooov_aa_bb, "injx", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(1.0, t_Goooo_ab, "injm", t_Vooov_aa_bb, "injx", 1.0, t_Zmat_b, "xm");

      tblis::mult<double>(1.0, t_Goooo_bb, "minj", t_Vooov_bb, "njix", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(1.0, t_Goooo_bb, "imnj", t_Vooov_bb, "njix", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(1.0, t_Goooo_bb, "inmj", t_Vooov_bb, "injx", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(1.0, t_Goooo_bb, "injm", t_Vooov_bb, "injx", 1.0, t_Zmat_b, "xm");

      tblis::mult<double>( 1.0, t_Govov_bb, "makc", t_Vovvv_bb, "kcxa", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(-1.0, t_Govov_bb, "ixkc", t_Vooov_bb, "imkc", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>( 1.0, t_Govov_bb, "iamc", t_Vovvv_bb, "iaxc", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(-1.0, t_Govov_bb, "iakx", t_Vooov_bb, "kmia", 1.0, t_Zmat_b, "xm");

      tblis::mult<double>( 1.0, t_Goovv_bb, "mkac", t_Vovvv_bb, "kxca", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>( 1.0, t_Goovv_bb, "imac", t_Vovvv_bb, "ixac", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(-1.0, t_Goovv_bb, "ikxc", t_Vooov_bb, "ikmc", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(-1.0, t_Goovv_bb, "ikax", t_Vooov_bb, "ikma", 1.0, t_Zmat_b, "xm");

      tblis::mult<double>( 1.0, t_Govov_ab, "makc", t_Vovvv_ba, "kcxa", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>(-1.0, t_Govov_ab, "ixkc", t_Vooov_aa_bb, "imkc", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>( 1.0, t_Goovv_ab_ex, "mkac", t_Vovvv_ab, "kxac", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>( 1.0, t_Goovv_ab_ex, "imac", t_Vovvv_ab, "ixac", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>(-1.0, t_Goovv_ba_ex, "ikxc", t_Vooov_bb_aa, "ikmc", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>(-1.0, t_Goovv_ba_ex, "ikax", t_Vooov_bb_aa, "ikma", 1.0, t_Zmat_a, "xm");

      tblis::mult<double>( 1.0, t_Govov_ab, "iamc", t_Vovvv_ab, "iaxc", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(-1.0, t_Govov_ab, "iakx", t_Vooov_bb_aa, "kmia", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>( 1.0, t_Goovv_ba_ex, "mkac", t_Vovvv_ba, "kxac", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>( 1.0, t_Goovv_ba_ex, "imac", t_Vovvv_ba, "ixac", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(-1.0, t_Goovv_ab_ex, "ikxc", t_Vooov_aa_bb, "ikmc", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(-1.0, t_Goovv_ab_ex, "ikax", t_Vooov_aa_bb, "ikma", 1.0, t_Zmat_b, "xm");
    } else if (is_restricted) {
      Z_mat_b = Z_mat_a;
    }
  }

  Eigen::MatrixXd G_full_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
  G_full_a.block(0, 0, na_, na_) = G_oo_alpha_;
  G_full_a.block(na_, na_, va_, va_) = G_vv_alpha_;
  Eigen::MatrixXd P_corr_a = scf_.C_alpha * G_full_a * scf_.C_alpha.transpose();

  Eigen::MatrixXd G_full_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
  Eigen::MatrixXd P_corr_b = Eigen::MatrixXd::Zero(nbf_, nbf_);

  if (is_restricted) {
    G_full_b = G_full_a;
    P_corr_b = P_corr_a;
  } else if (nb_ > 0) {
    G_full_b.block(0, 0, nb_, nb_) = G_oo_beta_;
    G_full_b.block(nb_, nb_, vb_, vb_) = G_vv_beta_;
    P_corr_b = scf_.C_beta * G_full_b * scf_.C_beta.transpose();
  }

  const Eigen::MatrixXd& F_HF_ao_a = scf_.F_alpha;
  const Eigen::MatrixXd& F_HF_ao_b = scf_.F_beta;
  Eigen::MatrixXd F_HF_mo_a = scf_.C_alpha.transpose() * F_HF_ao_a * scf_.C_alpha;

  Eigen::MatrixXd F_HF_mo_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
  if (nb_ > 0) F_HF_mo_b = scf_.C_beta.transpose() * F_HF_ao_b * scf_.C_beta;

  Eigen::MatrixXd G_gamma_ao_a, G_gamma_ao_b;
  build_fock_fast(P_corr_a, P_corr_b, G_gamma_ao_a, G_gamma_ao_b);
  G_gamma_ao_a -= H_core_;
  if (nb_ > 0) G_gamma_ao_b -= H_core_;

  Eigen::MatrixXd G_gamma_mo_a = scf_.C_alpha.transpose() * G_gamma_ao_a * scf_.C_alpha;
  Eigen::MatrixXd G_gamma_mo_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
  if (nb_ > 0) G_gamma_mo_b = scf_.C_beta.transpose() * G_gamma_ao_b * scf_.C_beta;

  F_gen_a_ = F_HF_mo_a + G_gamma_mo_a;
  if (na_ > 0 && va_ > 0) {
    Eigen::MatrixXd F_HF_vo_a = F_HF_mo_a.block(na_, 0, va_, na_);
    Eigen::MatrixXd L_sep_a = F_HF_vo_a * G_oo_alpha_ - G_vv_alpha_ * F_HF_vo_a;
    F_gen_a_.block(na_, 0, va_, na_) += L_sep_a + Z_mat_a;
    F_gen_a_.block(0, na_, na_, va_) += (L_sep_a + Z_mat_a).transpose();
  }

  F_gen_b_ = F_HF_mo_b + G_gamma_mo_b;
  if (nb_ > 0 && vb_ > 0) {
    Eigen::MatrixXd F_HF_vo_b = F_HF_mo_b.block(nb_, 0, vb_, nb_);
    Eigen::MatrixXd L_sep_b = F_HF_vo_b * G_oo_beta_ - G_vv_beta_ * F_HF_vo_b;
    F_gen_b_.block(nb_, 0, vb_, nb_) += L_sep_b + Z_mat_b;
    F_gen_b_.block(0, nb_, nb_, vb_) += (L_sep_b + Z_mat_b).transpose();
  } else if (is_restricted) {
    F_gen_b_ = F_gen_a_;
  }
}

void OMP3::build_hessian_diagonal(Eigen::VectorXd& diag_H, double grad_norm) {
    // PSI4 BEHAVIOR: Gunakan Hessian OMP2 polos, abaikan koreksi densitas MP3
    OMP2::build_hessian_diagonal(diag_H, grad_norm);
    
    /* -- COMMENT OUT MSHQC HEURISTIC --
    int idx = 0;
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    double spin_factor = is_restricted ? 4.0 : 2.0;

    for (int i = 0; i < na_; ++i) {
        for (int a = 0; a < va_; ++a) {
            double delta_density = std::abs(G_vv_alpha_(a, a)) + std::abs(G_oo_alpha_(i, i));
            diag_H(idx) += spin_factor * 1.5 * delta_density; 
            idx++;
        }
    }

    if (!is_restricted && nb_ > 0) {
        for (int i = 0; i < nb_; ++i) {
            for (int a = 0; a < vb_; ++a) {
                double delta_density = std::abs(G_vv_beta_(a, a)) + std::abs(G_oo_beta_(i, i));
                diag_H(idx) += 2.0 * 1.5 * delta_density;
                idx++;
            }
        }
    }
    ----------------------------------- */
}

void OMP3::debug_gradient_fd(int i_target, int a_target) {
    std::cout << "\n--- [DEBUG] Membedah Komponen Gradien OMP3 ---\n";
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    
    pseudocanonicalize(); 
    C_a_current_ = scf_.C_alpha;
    C_b_current_ = scf_.C_beta;
    transform_integrals();
    compute_t2_amplitudes();
    compute_mp2_energy();
    compute_mp3_correction();
    
    L2_aa_ = t2_3rd_aa_; 
    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        L2_bb_ = t2_3rd_bb_;
        L2_ab_ = t2_3rd_ab_;
    }
    
    build_opdm_alpha();
    if (!is_restricted && nb_ > 0) build_opdm_beta();
    else if (is_restricted) G_oo_beta_ = G_oo_alpha_;
    build_generalized_fock();

    double max_grad = -1.0;
    for(int i = 0; i < na_; ++i) {
        for(int a = 0; a < va_; ++a) {
            double val = std::abs(F_gen_a_(na_ + a, i));
            if (val > max_grad) {
                max_grad = val;
                i_target = i;
                a_target = a;
            }
        }
    }
    
    double grad_ana_tot = is_restricted ? -4.0 * F_gen_a_(na_ + a_target, i_target) : -2.0 * F_gen_a_(na_ + a_target, i_target);
    
    Eigen::MatrixXd F_ao_a, F_ao_b;
    build_fock_fast(scf_.P_alpha, scf_.P_beta, F_ao_a, F_ao_b);
    Eigen::MatrixXd F_mo_a = scf_.C_alpha.transpose() * F_ao_a * scf_.C_alpha;
    double grad_ana_hf = is_restricted ? -4.0 * F_mo_a(na_ + a_target, i_target) : -2.0 * F_mo_a(na_ + a_target, i_target);
    double grad_ana_corr = grad_ana_tot - grad_ana_hf;
    
    double theta = 1e-5;
    Eigen::MatrixXd C_a_orig = scf_.C_alpha;
    Eigen::MatrixXd C_b_orig = scf_.C_beta;
    Eigen::MatrixXd P_a_orig = scf_.P_alpha;
    Eigen::MatrixXd P_b_orig = scf_.P_beta;
    
    auto calc_energy_components = [&](double t, double& e_hf, double& e_mp2, double& e_mp3) {
        Eigen::MatrixXd U = Eigen::MatrixXd::Identity(nbf_, nbf_);
        U(i_target, na_ + a_target) = t;
        U(na_ + a_target, i_target) = -t;
        
        C_a_current_ = C_a_orig * U;
        if (!is_restricted) C_b_current_ = C_b_orig * U; 
        else C_b_current_ = C_a_current_;
        
        scf_.C_alpha = C_a_current_;
        scf_.C_beta = C_b_current_;
        scf_.P_alpha = scf_.C_alpha.leftCols(na_) * scf_.C_alpha.leftCols(na_).transpose();
        if (!is_restricted) scf_.P_beta = scf_.C_beta.leftCols(nb_) * scf_.C_beta.leftCols(nb_).transpose();
        else scf_.P_beta = scf_.P_alpha;
        
        pseudocanonicalize();
        C_a_current_ = scf_.C_alpha;
        C_b_current_ = scf_.C_beta;

        Eigen::MatrixXd F_ao_a_tmp, F_ao_b_tmp;
        build_fock_fast(scf_.P_alpha, scf_.P_beta, F_ao_a_tmp, F_ao_b_tmp);
        e_hf = 0.5 * (scf_.P_alpha.cwiseProduct(H_core_ + F_ao_a_tmp).sum() + 
                      scf_.P_beta.cwiseProduct(H_core_ + F_ao_b_tmp).sum()) 
               + mol_.nuclear_repulsion_energy();
                       
        transform_integrals(); 
        compute_t2_amplitudes();
        compute_mp2_energy();
        compute_mp3_correction();
        
        e_mp2 = e_ss_ + e_os_;
        e_mp3 = e_mp3_tot_;
    };

    double hf_plus, mp2_plus, mp3_plus;
    calc_energy_components(theta, hf_plus, mp2_plus, mp3_plus);
    
    double hf_minus, mp2_minus, mp3_minus;
    calc_energy_components(-theta, hf_minus, mp2_minus, mp3_minus);
    
    double g_num_hf  = (hf_plus - hf_minus) / (2.0 * theta);
    double g_num_mp2 = (mp2_plus - mp2_minus) / (2.0 * theta);
    double g_num_mp3 = (mp3_plus - mp3_minus) / (2.0 * theta);
    double g_num_corr = g_num_mp2 + g_num_mp3;
    double g_num_tot = g_num_hf + g_num_corr;

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Target Rotasi        : (i=" << i_target << " [Occ], a=" << a_target << " [Vir])\n";
    std::cout << "--- Finite Difference (NUMERIK) ---\n";
    std::cout << "Gradien HF Numerik   : " << std::scientific << g_num_hf << "\n";
    std::cout << "Gradien MP2 Numerik  : " << std::scientific << g_num_mp2 << "\n";
    std::cout << "Gradien MP3 Numerik  : " << std::scientific << g_num_mp3 << "\n";
    std::cout << "Korelasi Numerik     : " << std::scientific << g_num_corr << "\n";
    std::cout << "TOTAL Numerik        : " << std::scientific << g_num_tot << "\n";
    std::cout << "--- Rumus Analitik (KODE) ---\n";
    std::cout << "Gradien HF Analitik  : " << std::scientific << grad_ana_hf << "\n";
    std::cout << "Korelasi Analitik    : " << std::scientific << grad_ana_corr << "\n";
    std::cout << "TOTAL Analitik       : " << std::scientific << grad_ana_tot << "\n";
    std::cout << "--- EVALUASI SELISIH ---\n";
    std::cout << "Selisih HF           : " << std::abs(g_num_hf - grad_ana_hf) << "\n";
    std::cout << "Selisih Korelasi     : " << std::abs(g_num_corr - grad_ana_corr) << "\n";
    std::cout << "Selisih TOTAL        : " << std::abs(g_num_tot - grad_ana_tot) << "\n";
    std::cout << "-------------------------------------------------------------------\n";
    scf_.C_alpha = C_a_orig;
    scf_.C_beta  = C_b_orig;
    scf_.P_alpha = P_a_orig;
    scf_.P_beta  = P_b_orig;
    C_a_current_ = C_a_orig;
    C_b_current_ = C_b_orig;
}

MP3Result OMP3::compute_omp3() {
    MP2Result res2 = OMP2::compute();

    MP3Result res3;
    res3.converged = res2.converged;
    res3.iterations = res2.iterations;
    res3.e_hf = res2.energy_scf;
    res3.e_mp2 = e_ss_ + e_os_; 
    res3.e_mp3 = e_mp3_tot_;
    res3.e_corr_total = res3.e_mp2 + res3.e_mp3;
    res3.e_total = res3.e_hf + res3.e_corr_total;
    res3.C_alpha = res2.C_alpha;
    res3.C_beta = res2.C_beta;
    res3.n_occ_alpha = na_;
    res3.n_occ_beta = nb_;

    return res3;
}

} // namespace mshqc