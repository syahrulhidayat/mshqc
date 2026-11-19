// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0
// [INJECTED_PRIORITAS_2D_PACKED_AO]
#pragma once

#include <Eigen/Dense>
#include "mshqc/utils/blas_thread_guard.h"
#include <algorithm>
#include <vector>
#include <omp.h>

namespace mshqc {
namespace integrals {

inline int ao_tri_size(int nbf) {
    return nbf * (nbf + 1) / 2;
}

inline int ao_tri_index(int p, int q) {
    if (p < q) std::swap(p, q);
    return p * (p + 1) / 2 + q;
}

inline bool is_packed_ao_matrix(const Eigen::MatrixXd& L, int nbf) {
    return (nbf > 1) && (L.rows() == ao_tri_size(nbf));
}

inline bool has_valid_ao_factor(const Eigen::MatrixXd& L, int nbf) {
    if (L.cols() <= 0) return false;
    return (L.rows() == ao_tri_size(nbf)) || (L.rows() == static_cast<Eigen::Index>(nbf) * nbf);
}

// Membongkar 1 kolom vektor 3-pusat AO (packed atau full) menjadi matriks simetris nbf x nbf
inline void unpack_ao_column(const double* col_ptr, int nbf, bool packed, Eigen::MatrixXd& dst) {
    dst.resize(nbf, nbf);
    if (!packed) {
        std::copy(col_ptr, col_ptr + static_cast<size_t>(nbf) * nbf, dst.data());
        return;
    }
    for (int p = 0; p < nbf; ++p) {
        const int base = p * (p + 1) / 2;
        for (int q = 0; q < p; ++q) {
            const double val = col_ptr[base + q];
            dst(p, q) = val;
            dst(q, p) = val;
        }
        dst(p, p) = col_ptr[base + p];
    }
}

// Mengemas matriks simetris M (nbf x nbf) ke vektor segitiga berbobot-2 pada non-diagonal
inline Eigen::VectorXd pack_sym_weighted(const Eigen::MatrixXd& M) {
    const int nbf = static_cast<int>(M.rows());
    Eigen::VectorXd v(ao_tri_size(nbf));
    for (int p = 0; p < nbf; ++p) {
        const int base = p * (p + 1) / 2;
        for (int q = 0; q < p; ++q) {
            v(base + q) = M(p, q) + M(q, p);
        }
        v(base + p) = M(p, p);
    }
    return v;
}

// Membongkar vektor segitiga (tanpa bobot) ke matriks simetris nbf x nbf
inline Eigen::MatrixXd unpack_sym_vector(const Eigen::VectorXd& v, int nbf) {
    Eigen::MatrixXd M(nbf, nbf);
    for (int p = 0; p < nbf; ++p) {
        const int base = p * (p + 1) / 2;
        for (int q = 0; q < p; ++q) {
            const double val = v(base + q);
            M(p, q) = val;
            M(q, p) = val;
        }
        M(p, p) = v(base + p);
    }
    return M;
}

// Evaluasi c_P = sum_{mu,nu} L(mu,nu; P) * M(mu,nu) untuk matriks simetris M
inline Eigen::VectorXd contract_ao_factor_with_sym(
    const Eigen::MatrixXd& L,
    const Eigen::MatrixXd& M,
    int nbf)
{
  mshqc::utils::ScopedBlasThreads blas_mt(nbf, 96);
    if (is_packed_ao_matrix(L, nbf)) {
        Eigen::VectorXd m_tri = pack_sym_weighted(M);
        return L.transpose() * m_tri;
    }
    Eigen::Map<const Eigen::VectorXd> m_flat(M.data(), static_cast<Eigen::Index>(nbf) * nbf);
    return L.transpose() * m_flat;
}

// Evaluasi matriks Coulomb J(mu,nu) = sum_P L(mu,nu; P) * (L^T * P_tot)_P
inline Eigen::MatrixXd compute_coulomb_from_ao_factor(
    const Eigen::MatrixXd& L,
    const Eigen::MatrixXd& P_tot,
    int nbf)
{
  mshqc::utils::ScopedBlasThreads blas_mt(nbf, 96);
    if (is_packed_ao_matrix(L, nbf)) {
        Eigen::VectorXd p_tri = pack_sym_weighted(P_tot);
        Eigen::VectorXd x_j = L.transpose() * p_tri;
        Eigen::VectorXd j_tri = L * x_j;
        return unpack_sym_vector(j_tri, nbf);
    }
    Eigen::Map<const Eigen::VectorXd> p_flat(P_tot.data(), static_cast<Eigen::Index>(nbf) * nbf);
    Eigen::VectorXd x_j = L.transpose() * p_flat;
    Eigen::VectorXd j_flat = L * x_j;
    return Eigen::Map<Eigen::MatrixXd>(j_flat.data(), nbf, nbf);
}

// Evaluasi gamma_2c = 0.5 * (Gamma_full^T * L + L^T * Gamma_full) (mendukung L packed & full)
inline Eigen::MatrixXd compute_gamma2c_from_full_and_L(
    const Eigen::MatrixXd& Gamma_full,
    const Eigen::MatrixXd& L,
    int nbf)
{
    const int n_aux = static_cast<int>(L.cols());
    Eigen::MatrixXd g2(n_aux, n_aux);
    constexpr int tile_g = 64;

    if (is_packed_ao_matrix(L, nbf)) {
        const int n_tri = ao_tri_size(nbf);
        Eigen::MatrixXd G_tri_w(n_tri, n_aux);
        #pragma omp parallel for schedule(static)
        for (int P = 0; P < n_aux; ++P) {
            Eigen::Map<const Eigen::MatrixXd> G_P(Gamma_full.col(P).data(), nbf, nbf);
            double* dst = G_tri_w.col(P).data();
            for (int p = 0; p < nbf; ++p) {
                const int base = p * (p + 1) / 2;
                for (int q = 0; q < p; ++q) {
                    dst[base + q] = G_P(p, q) + G_P(q, p);
                }
                dst[base + p] = G_P(p, p);
            }
        }
        #pragma omp parallel for schedule(dynamic)
        for (int r = 0; r < n_aux; r += tile_g) {
            int rows = std::min(tile_g, n_aux - r);
            g2.middleRows(r, rows).noalias() = G_tri_w.middleCols(r, rows).transpose() * L;
        }
    } else {
        #pragma omp parallel for schedule(dynamic)
        for (int r = 0; r < n_aux; r += tile_g) {
            int rows = std::min(tile_g, n_aux - r);
            g2.middleRows(r, rows).noalias() = Gamma_full.middleCols(r, rows).transpose() * L;
        }
    }
    return 0.5 * (g2 + g2.transpose());
}

// Transformasi 3-indeks AO (packed atau full) -> MO B(x*d2 + y, P) secara chunked paralel
inline Eigen::MatrixXd transform_3c_ao_to_mo_block(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& C1,
    const Eigen::MatrixXd& C2,
    int nbf)
{
    const int d1 = static_cast<int>(C1.cols());
    const int d2 = static_cast<int>(C2.cols());
    const int n_aux = static_cast<int>(L_mat.cols());
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(d1) * d2, n_aux);
    if (d1 == 0 || d2 == 0 || n_aux == 0) return B;

    const bool packed = is_packed_ao_matrix(L_mat, nbf);
    constexpr int chunk_size = 32;

    #pragma omp parallel for schedule(dynamic)
    for (int P_start = 0; P_start < n_aux; P_start += chunk_size) {
        const int P_size = std::min(chunk_size, n_aux - P_start);
        Eigen::MatrixXd L_sq_buf;
        const double* L_sq_ptr = nullptr;

        if (packed) {
            L_sq_buf.resize(static_cast<Eigen::Index>(nbf) * nbf, P_size);
            for (int p = 0; p < P_size; ++p) {
                const double* src = L_mat.col(P_start + p).data();
                double* dst = L_sq_buf.col(p).data();
                for (int i = 0; i < nbf; ++i) {
                    const int base = i * (i + 1) / 2;
                    for (int j = 0; j < i; ++j) {
                        const double val = src[base + j];
                        dst[i + j * nbf] = val;
                        dst[j + i * nbf] = val;
                    }
                    dst[i + i * nbf] = src[base + i];
                }
            }
            L_sq_ptr = L_sq_buf.data();
        } else {
            L_sq_ptr = L_mat.col(P_start).data();
        }

        Eigen::Map<const Eigen::MatrixXd> L_res(L_sq_ptr, nbf, nbf * P_size);
        Eigen::MatrixXd temp1 = C1.transpose() * L_res;

        for (int p = 0; p < P_size; ++p) {
            Eigen::Map<const Eigen::MatrixXd> tmp_P(temp1.data() + static_cast<size_t>(p) * d1 * nbf, d1, nbf);
            Eigen::MatrixXd L_pq = tmp_P * C2;
            double* b_col = B.col(P_start + p).data();
            for (int x = 0; x < d1; ++x) {
                for (int y = 0; y < d2; ++y) {
                    b_col[x * d2 + y] = L_pq(x, y);
                }
            }
        }
    }
    return B;
}

} // namespace integrals
} // namespace mshqc
