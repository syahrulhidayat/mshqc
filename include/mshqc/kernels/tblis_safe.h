// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0
// [INJECTED_HYBRID_PHASE2_TBLIS_SAFE]
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <omp.h>
#include <tblis/tblis.h>
#include <vector>

namespace mshqc {
namespace kernels {

inline size_t view_num_elements(const tblis::varray_view<double>& v) {
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
    const tblis::varray_view<double>& A, const char* idxA,
    const tblis::varray_view<double>& B, const char* idxB,
    double beta,
    tblis::varray_view<double>& C, const char* idxC)
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

inline void safe_tblis_mult(
    double alpha,
    const tblis::varray_view<double>& A, const char* idxA,
    const tblis::varray_view<double>& B, const char* idxB,
    double beta,
    tblis::varray_view<double>& C, const char* idxC)
{
    const size_t szA = view_num_elements(A);
    const size_t szB = view_num_elements(B);
    const size_t szC = view_num_elements(C);
    if (szC == 0) return;

    if (szA == 0 || szB == 0) {
        Eigen::Map<Eigen::VectorXd> vecC(C.data(), static_cast<Eigen::Index>(szC));
        if (beta == 0.0) vecC.setZero();
        else vecC *= beta;
        return;
    }

#ifdef _WIN32
    win32_eigen_tensor_mult(alpha, A, idxA, B, idxB, beta, C, idxC);
#else
    tblis::tblis_tensor A_s(alpha, const_cast<tblis::varray_view<double>&>(A));
    tblis::tblis_tensor B_s(const_cast<tblis::varray_view<double>&>(B));
    tblis::tblis_tensor C_s(beta, C);
    tblis_tensor_mult(nullptr, nullptr, &A_s, idxA, &B_s, idxB, &C_s, idxC);
#endif
}

inline Eigen::MatrixXd parallel_gemm_abt(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    if (A.rows() <= 0 || B.rows() <= 0 || A.cols() <= 0 || B.cols() <= 0) {
        return Eigen::MatrixXd();
    }
    Eigen::MatrixXd C(A.rows(), B.rows());
    const int m = static_cast<int>(A.rows());
    constexpr int tile = 64;
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < m; i += tile) {
        const int rows = std::min(tile, m - i);
        C.middleRows(i, rows).noalias() = A.middleRows(i, rows) * B.transpose();
    }
    return C;
}

// Zero-copy 4D view atas M = B1 * B2^T berdimensi (d1*d2 x d3*d4)
inline tblis::varray_view<double> make_mat_view_4d(
    Eigen::MatrixXd& M, int d1, int d2, int d3, int d4)
{
    using tblis::len_type;
    using tblis::stride_type;
    return tblis::varray_view<double>(
        {(len_type)d1, (len_type)d2, (len_type)d3, (len_type)d4},
        M.data(),
        {(stride_type)d2, 1,
         (stride_type)(static_cast<ptrdiff_t>(d1) * d2 * d4),
         (stride_type)(static_cast<ptrdiff_t>(d1) * d2)});
}

// Zero-copy 4D view atas T2/W/L2 berdimensi (nv*nv x no*no) dengan indeks logis (i, j, a, b)
inline tblis::varray_view<double> make_t2_view_4d(
    const Eigen::MatrixXd& M, int no, int nv)
{
    using tblis::len_type;
    using tblis::stride_type;
    const ptrdiff_t nv2 = static_cast<ptrdiff_t>(nv) * nv;
    return tblis::varray_view<double>(
        {(len_type)no, (len_type)no, (len_type)nv, (len_type)nv},
        const_cast<double*>(M.data()),
        {(stride_type)(static_cast<ptrdiff_t>(no) * nv2),
         (stride_type)nv2,
         1,
         (stride_type)nv});
}

inline double& t2_at(Eigen::MatrixXd& M, int no, int nv, int i, int j, int a, int b) {
    return M(static_cast<Eigen::Index>(a) + static_cast<Eigen::Index>(b) * nv,
             static_cast<Eigen::Index>(i) * no + j);
}

inline double t2_at(const Eigen::MatrixXd& M, int no, int nv, int i, int j, int a, int b) {
    return M(static_cast<Eigen::Index>(a) + static_cast<Eigen::Index>(b) * nv,
             static_cast<Eigen::Index>(i) * no + j);
}

} // namespace kernels
} // namespace mshqc
