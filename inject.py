
import pathlib
import re

ROOT = pathlib.Path(".")

# ==============================================================================
# 1. BLOK SAFE_TBLIS_MULT DENGAN BYPASS NATIVE EIGEN GEMM KHUSUS WINDOWS (_WIN32)
# ==============================================================================
TBLIS_SAFE_BLOCK = """namespace tblis {

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

#define mult safe_tblis_mult"""

# Patch 1a: src/integrals/eri_transformer.cc
eri_path = ROOT / "src/integrals/eri_transformer.cc"
if eri_path.is_file():
    txt = eri_path.read_text(encoding="utf-8")
    pattern = r"#include\s*<tblis/tblis\.h>[\s\S]*?static\s+Eigen::Tensor<double,\s*4>\s+smart_transform_kernel\s*\("
    replacement = (
        "#include <tblis/tblis.h>\n\n"
        + TBLIS_SAFE_BLOCK
        + "\n\nnamespace mshqc {\nnamespace integrals {\n\nusing tblis::safe_tblis_mult;\n\nstatic Eigen::Tensor<double, 4> smart_transform_kernel("
    )
    new_txt, n = re.subn(pattern, replacement, txt, count=1)
    if n > 0:
        eri_path.write_text(new_txt, encoding="utf-8")
        print(f"[OK] Patched {eri_path}")

# Patch 1b: src/mp/omp_fock.cc
omp_fock_path = ROOT / "src/mp/omp_fock.cc"
if omp_fock_path.is_file():
    txt = omp_fock_path.read_text(encoding="utf-8")
    pattern = r"#include\s*<tblis/tblis\.h>[\s\S]*?namespace\s+mshqc\s*\{"
    replacement = (
        "#include <tblis/tblis.h>\n#include <iostream>\n\n"
        + TBLIS_SAFE_BLOCK
        + "\n\nnamespace mshqc {"
    )
    new_txt, n = re.subn(pattern, replacement, txt, count=1)
    if n > 0:
        omp_fock_path.write_text(new_txt, encoding="utf-8")
        print(f"[OK] Patched {omp_fock_path}")

# Patch 1c: src/mp3/mp3.cc
mp3_path = ROOT / "src/mp3/mp3.cc"
if mp3_path.is_file():
    txt = mp3_path.read_text(encoding="utf-8")
    pattern = r"#include\s*<tblis/tblis\.h>[\s\S]*?namespace\s+mshqc\s*\{"
    replacement = (
        "#include <tblis/tblis.h>\n#include <unsupported/Eigen/CXX11/Tensor>\n\n"
        + TBLIS_SAFE_BLOCK
        + "\n\nnamespace mshqc {"
    )
    new_txt, n = re.subn(pattern, replacement, txt, count=1)
    if n > 0:
        mp3_path.write_text(new_txt, encoding="utf-8")
        print(f"[OK] Patched {mp3_path}")
