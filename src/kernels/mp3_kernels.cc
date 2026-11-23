// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0
// [INJECTED_HYBRID_PHASE2_MP3_KERNELS_CC]

#include "mshqc/kernels/mp3_kernels.h"
#include "mshqc/kernels/tblis_safe.h"

#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdexcept>

namespace mshqc {
namespace kernels {

using tblis::len_type;
using tblis::stride_type;
using tblis::varray_view;

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

    // Jalur 1: Adaptive Batched T+/T- BLAS-3 DGEMM dengan pre-packed B_blocks (RAM <= 64 MB)
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

        if (flops_batched < 1.15 * flops_loop_p) {
            const int n_tri = V * (V + 1) / 2;
            const int n_strict = V * (V - 1) / 2;

            // Pre-pack V blok kontigu (V x n_aux) 1x untuk menghilangkan ratusan MB strided copy
            std::vector<Eigen::MatrixXd> B_blocks(V);
            #pragma omp parallel for schedule(static)
            for (int a = 0; a < V; ++a) {
                B_blocks[a] = B_vv_left.middleRows(static_cast<Eigen::Index>(a) * V, V);
            }

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

            constexpr size_t kMaxBatchBytes = 64ULL * 1024ULL * 1024ULL;
            const size_t col_bytes = static_cast<size_t>((is_antisym ? 0 : n_tri) + n_strict) * sizeof(double);
            const int batch_ab = std::max(1, std::min(n_tri, static_cast<int>(kMaxBatchBytes / std::max<size_t>(1, col_bytes))));

            Eigen::MatrixXd V_plus_T, V_minus_T, W_plus_b, W_minus_b;
            for (int p0 = 0; p0 < n_tri; p0 += batch_ab) {
                const int bsz = std::min(batch_ab, n_tri - p0);
                if (!is_antisym) V_plus_T.resize(n_tri, bsz);
                if (n_strict > 0) V_minus_T.resize(n_strict, bsz);

                #pragma omp parallel
                {
                    Eigen::MatrixXd M_ab(V, V);
                    #pragma omp for schedule(static) nowait
                    for (int r = 0; r < bsz; ++r) {
                        const int a = pair_a[p0 + r];
                        const int b = pair_b[p0 + r];
                        M_ab.noalias() = B_blocks[a] * B_blocks[b].transpose();
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

    // Jalur 2: Adaptive Batched AB DGEMM dengan pre-packed B_blocks_l & B_blocks_r
    if (!same_B && nv_l >= 8 && nv_r >= 8) {
        const double n_ab = static_cast<double>(nv_l) * nv_r;
        const double flops_batched_ab = 2.0 * n_ab * n_ab * n_aux + 2.0 * n_ab * n_ab * n_pairs;
        if (flops_batched_ab < 1.10 * flops_loop_p) {
            std::vector<Eigen::MatrixXd> B_blocks_l(nv_l), B_blocks_r(nv_r);
            #pragma omp parallel for schedule(static)
            for (int a = 0; a < nv_l; ++a) {
                B_blocks_l[a] = B_vv_left.middleRows(static_cast<Eigen::Index>(a) * nv_l, nv_l);
            }
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < nv_r; ++b) {
                B_blocks_r[b] = B_vv_right.middleRows(static_cast<Eigen::Index>(b) * nv_r, nv_r);
            }

            const int n_ef = nv_l * nv_r;
            Eigen::MatrixXd T_pack(n_ef, n_pairs);
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < n_pairs; ++k) {
                std::copy(T_pairs[k].data(), T_pairs[k].data() + n_ef, T_pack.col(k).data());
            }

            constexpr size_t kMaxBatchBytes = 64ULL * 1024ULL * 1024ULL;
            const int batch_ab = std::max(1, std::min(n_ef, static_cast<int>(
                kMaxBatchBytes / std::max<size_t>(1, static_cast<size_t>(n_ef) * sizeof(double)))));

            Eigen::MatrixXd V_b_T, W_b;
            for (int p0 = 0; p0 < n_ef; p0 += batch_ab) {
                const int bsz = std::min(batch_ab, n_ef - p0);
                V_b_T.resize(n_ef, bsz);

                #pragma omp parallel
                {
                    Eigen::MatrixXd M_ab(nv_l, nv_r);
                    #pragma omp for schedule(static) nowait
                    for (int r = 0; r < bsz; ++r) {
                        const int idx = p0 + r;
                        const int a = idx / nv_r;
                        const int b = idx % nv_r;
                        M_ab.noalias() = B_blocks_l[a] * B_blocks_r[b].transpose();
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

    // Jalur 3: P-parallel loop (baca B_vv 1x per thread, beban seimbang, batas TLS <= 64 MB)
    const size_t tls_bytes = static_cast<size_t>(omp_get_max_threads()) * n_pairs * nv_l * nv_r * sizeof(double);
    if (n_aux >= omp_get_max_threads() && tls_bytes <= 64ULL * 1024ULL * 1024ULL) {
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
    } else {
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
    }
}

Eigen::MatrixXd eval_pp_ladder_rhf_df(
    const Eigen::MatrixXd& T2,
    const Eigen::MatrixXd& B_vv,
    int n_occ,
    int n_vir)
{
    const int n_aux = static_cast<int>(B_vv.cols());
    Eigen::MatrixXd W_pp = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(n_vir) * n_vir,
        static_cast<Eigen::Index>(n_occ) * n_occ);
    if (n_occ <= 0 || n_vir <= 0 || n_aux <= 0) return W_pp;

    const int n_pairs = n_occ * (n_occ + 1) / 2;
    std::vector<Eigen::MatrixXd> T_pairs(n_pairs, Eigen::MatrixXd(n_vir, n_vir));
    std::vector<Eigen::MatrixXd> W_pairs(n_pairs, Eigen::MatrixXd::Zero(n_vir, n_vir));

    int idx = 0;
    for (int i = 0; i < n_occ; ++i) {
        for (int j = i; j < n_occ; ++j) {
            Eigen::Map<const Eigen::MatrixXd> T_ij(
                T2.col(static_cast<Eigen::Index>(i) * n_occ + j).data(), n_vir, n_vir);
            T_pairs[idx++] = T_ij;
        }
    }

    contract_vvvv_pairs_df(B_vv, n_vir, B_vv, n_vir, n_aux, T_pairs, W_pairs);

    idx = 0;
    for (int i = 0; i < n_occ; ++i) {
        for (int j = i; j < n_occ; ++j) {
            Eigen::Map<Eigen::MatrixXd> dst_ij(
                W_pp.col(static_cast<Eigen::Index>(i) * n_occ + j).data(), n_vir, n_vir);
            dst_ij = W_pairs[idx];
            if (i != j) {
                Eigen::Map<Eigen::MatrixXd> dst_ji(
                    W_pp.col(static_cast<Eigen::Index>(j) * n_occ + i).data(), n_vir, n_vir);
                dst_ji = W_pairs[idx].transpose();
            }
            ++idx;
        }
    }
    return W_pp;
}

std::pair<double, Eigen::MatrixXd> eval_rmp3_df(
    const Eigen::MatrixXd& T2,
    const Eigen::MatrixXd& B_oo,
    const Eigen::MatrixXd& B_ov,
    const Eigen::MatrixXd& B_vv,
    const Eigen::VectorXd& eps_occ,
    const Eigen::VectorXd& eps_vir,
    const Eigen::MatrixXd& W_pp_override)
{
    const int no = static_cast<int>(eps_occ.size());
    const int nv = static_cast<int>(eps_vir.size());
    if (no <= 0 || nv <= 0) {
        return {0.0, Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(nv) * nv,
                                           static_cast<Eigen::Index>(no) * no)};
    }

    // 1. Particle-Particle Ladder bebas O(V^4) (adaptive 3-path)
    Eigen::MatrixXd W = (W_pp_override.rows() == T2.rows() && W_pp_override.cols() == T2.cols())
        ? W_pp_override
        : eval_pp_ladder_rhf_df(T2, B_vv, no, nv);

    auto t_T = make_t2_view_4d(T2, no, nv);
    auto t_W = make_t2_view_4d(W, no, nv);

    // 2. Hole-Hole Ladder + Halved Particle-Hole Ring via zero-copy TBLIS
    Eigen::MatrixXd M_oooo = parallel_gemm_abt(B_oo, B_oo);
    Eigen::MatrixXd M_ovov = parallel_gemm_abt(B_ov, B_ov);
    Eigen::MatrixXd M_oovv = parallel_gemm_abt(B_oo, B_vv);

    auto t_Voooo = make_mat_view_4d(M_oooo, no, no, no, no);
    auto t_Vovov = make_mat_view_4d(M_ovov, no, nv, no, nv);
    auto t_Voovv = make_mat_view_4d(M_oovv, no, no, nv, nv);

    safe_tblis_mult(1.0, t_T, "mnab", t_Voooo, "minj", 1.0, t_W, "ijab");

    {
        Eigen::MatrixXd T_tilde(T2.rows(), T2.cols());
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < no * no; ++idx) {
            Eigen::Map<const Eigen::MatrixXd> T_ij(T2.col(idx).data(), nv, nv);
            Eigen::Map<Eigen::MatrixXd> Tt_ij(T_tilde.col(idx).data(), nv, nv);
            Tt_ij = 2.0 * T_ij - T_ij.transpose();
        }
        auto t_Tt = make_t2_view_4d(T_tilde, no, nv);

        Eigen::MatrixXd W_ring = Eigen::MatrixXd::Zero(W.rows(), W.cols());
        auto t_Wr = make_t2_view_4d(W_ring, no, nv);

        safe_tblis_mult( 1.0, t_Vovov, "iakc", t_Tt, "kjcb", 0.0, t_Wr, "ijab");
        safe_tblis_mult(-1.0, t_Voovv, "ikac", t_T,  "kjcb", 1.0, t_Wr, "ijab");
        safe_tblis_mult(-1.0, t_Voovv, "ikbc", t_T,  "kjac", 1.0, t_Wr, "ijab");

        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < no; ++i) {
            for (int j = 0; j < no; ++j) {
                Eigen::Map<Eigen::MatrixXd> W_ij(
                    W.col(static_cast<Eigen::Index>(i) * no + j).data(), nv, nv);
                Eigen::Map<const Eigen::MatrixXd> Wr_ij(
                    W_ring.col(static_cast<Eigen::Index>(i) * no + j).data(), nv, nv);
                Eigen::Map<const Eigen::MatrixXd> Wr_ji(
                    W_ring.col(static_cast<Eigen::Index>(j) * no + i).data(), nv, nv);
                W_ij.noalias() += Wr_ij + Wr_ji.transpose();
            }
        }
    }

    Eigen::MatrixXd L2(W.rows(), W.cols());
    double e_mp3 = 0.0;

    #pragma omp parallel for collapse(2) reduction(+:e_mp3) schedule(static)
    for (int i = 0; i < no; ++i) {
        for (int j = 0; j < no; ++j) {
            const double eij = eps_occ(i) + eps_occ(j);
            const Eigen::Index col = static_cast<Eigen::Index>(i) * no + j;
            Eigen::Map<const Eigen::MatrixXd> T_ij(T2.col(col).data(), nv, nv);
            Eigen::Map<const Eigen::MatrixXd> W_ij(W.col(col).data(), nv, nv);
            Eigen::Map<Eigen::MatrixXd> L_ij(L2.col(col).data(), nv, nv);
            for (int b = 0; b < nv; ++b) {
                const double eij_b = eij - eps_vir(b);
                for (int a = 0; a < nv; ++a) {
                    const double D = eij_b - eps_vir(a);
                    const double wv = W_ij(a, b);
                    L_ij(a, b) = (std::abs(D) > 1e-12) ? (wv / D) : 0.0;
                    e_mp3 += wv * (2.0 * T_ij(a, b) - T_ij(b, a));
                }
            }
        }
    }

    return {e_mp3, std::move(L2)};
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> eval_omp3_1rdm_rhf(
    const Eigen::MatrixXd& T2,
    const Eigen::MatrixXd& L2,
    int n_occ,
    int n_vir)
{
    Eigen::MatrixXd G_oo = Eigen::MatrixXd::Zero(n_occ, n_occ);
    Eigen::MatrixXd G_vv = Eigen::MatrixXd::Zero(n_vir, n_vir);
    if (n_occ <= 0 || n_vir <= 0) return {std::move(G_oo), std::move(G_vv)};

    const int max_t = omp_get_max_threads();
    std::vector<Eigen::MatrixXd> Gvv_tls(max_t, Eigen::MatrixXd::Zero(n_vir, n_vir));

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        Eigen::MatrixXd& Gvv_loc = Gvv_tls[tid];
        Eigen::MatrixXd Tt_ij(n_vir, n_vir), Sum_t(n_vir, n_vir);

        #pragma omp for schedule(static)
        for (int idx = 0; idx < n_occ * n_occ; ++idx) {
            Eigen::Map<const Eigen::MatrixXd> T_ij(T2.col(idx).data(), n_vir, n_vir);
            Eigen::Map<const Eigen::MatrixXd> L_ij(L2.col(idx).data(), n_vir, n_vir);
            Tt_ij.noalias() = 2.0 * T_ij - T_ij.transpose();
            Sum_t.noalias() = Tt_ij + (2.0 * L_ij - L_ij.transpose());
            Gvv_loc.noalias() += T_ij * Sum_t.transpose() + L_ij * Tt_ij.transpose();
        }
    }
    for (int t = 0; t < max_t; ++t) G_vv.noalias() += Gvv_tls[t];

    Eigen::MatrixXd Tt_all(T2.rows(), T2.cols());
    Eigen::MatrixXd Sum_all(T2.rows(), T2.cols());
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < n_occ * n_occ; ++idx) {
        Eigen::Map<const Eigen::MatrixXd> T_ij(T2.col(idx).data(), n_vir, n_vir);
        Eigen::Map<const Eigen::MatrixXd> L_ij(L2.col(idx).data(), n_vir, n_vir);
        Eigen::Map<Eigen::MatrixXd> Tt_ij(Tt_all.col(idx).data(), n_vir, n_vir);
        Eigen::Map<Eigen::MatrixXd> Sum_ij(Sum_all.col(idx).data(), n_vir, n_vir);
        Tt_ij.noalias() = 2.0 * T_ij - T_ij.transpose();
        Sum_ij.noalias() = Tt_ij + (2.0 * L_ij - L_ij.transpose());
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n_occ; ++i) {
        for (int j = 0; j < n_occ; ++j) {
            double val = 0.0;
            for (int k = 0; k < n_occ; ++k) {
                const Eigen::Index ik = static_cast<Eigen::Index>(i) * n_occ + k;
                const Eigen::Index jk = static_cast<Eigen::Index>(j) * n_occ + k;
                val += T2.col(ik).dot(Sum_all.col(jk)) + L2.col(ik).dot(Tt_all.col(jk));
            }
            G_oo(i, j) = -val;
        }
    }

    G_oo = 0.5 * (G_oo + G_oo.transpose());
    G_vv = 0.5 * (G_vv + G_vv.transpose());
    return {std::move(G_oo), std::move(G_vv)};
}

Eigen::MatrixXd eval_omp3_zvec_rhf_df(
    const Eigen::MatrixXd& T2,
    const Eigen::MatrixXd& L2,
    const Eigen::MatrixXd& B_oo,
    const Eigen::MatrixXd& B_ov,
    const Eigen::MatrixXd& B_vv,
    int n_occ,
    int n_vir,
    bool include_vvvv)
{
    const int n_aux = static_cast<int>(B_ov.cols());
    Eigen::MatrixXd Z_vo = Eigen::MatrixXd::Zero(n_vir, n_occ);
    if (n_occ <= 0 || n_vir <= 0 || n_aux <= 0) return Z_vo;

    varray_view<double> t_Za({(len_type)n_vir, (len_type)n_occ}, Z_vo.data(), {1, (stride_type)n_vir});
    varray_view<double> t_Bov(
        {(len_type)n_occ, (len_type)n_vir, (len_type)n_aux},
        const_cast<double*>(B_ov.data()),
        {(stride_type)n_vir, 1, (stride_type)(static_cast<ptrdiff_t>(n_occ) * n_vir)});
    varray_view<double> t_Bvv(
        {(len_type)n_vir, (len_type)n_vir, (len_type)n_aux},
        const_cast<double*>(B_vv.data()),
        {(stride_type)n_vir, 1, (stride_type)(static_cast<ptrdiff_t>(n_vir) * n_vir)});

    Eigen::MatrixXd M_ooov = parallel_gemm_abt(B_oo, B_ov);
    auto t_Vooov = make_mat_view_4d(M_ooov, n_occ, n_occ, n_occ, n_vir);

    auto t_T2 = make_t2_view_4d(T2, n_occ, n_vir);

    Eigen::MatrixXd T2_tilde(T2.rows(), T2.cols());
    Eigen::MatrixXd T_eff(T2.rows(), T2.cols());
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < n_occ * n_occ; ++idx) {
        Eigen::Map<const Eigen::MatrixXd> T_ij(T2.col(idx).data(), n_vir, n_vir);
        Eigen::Map<const Eigen::MatrixXd> L_ij(L2.col(idx).data(), n_vir, n_vir);
        Eigen::Map<Eigen::MatrixXd> Tt_ij(T2_tilde.col(idx).data(), n_vir, n_vir);
        Eigen::Map<Eigen::MatrixXd> Te_ij(T_eff.col(idx).data(), n_vir, n_vir);
        Tt_ij.noalias() = 2.0 * T_ij - T_ij.transpose();
        Te_ij.noalias() = Tt_ij + (2.0 * L_ij - L_ij.transpose());
    }
    auto t_T2t = make_t2_view_4d(T2_tilde, n_occ, n_vir);
    auto t_Teff = make_t2_view_4d(T_eff, n_occ, n_vir);

    // 1. Kontribusi Amplitudo Gabungan T_eff = T_tilde + L_tilde
    {
        Eigen::Tensor<double, 3> X_teff(n_occ, n_vir, n_aux);
        X_teff.setZero();
        varray_view<double> t_X(
            {(len_type)n_occ, (len_type)n_vir, (len_type)n_aux},
            X_teff.data(),
            {1, (stride_type)n_occ, (stride_type)(static_cast<ptrdiff_t>(n_occ) * n_vir)});
        safe_tblis_mult( 1.0, t_Teff, "ikbc", t_Bov, "kcP", 0.0, t_X, "ibP");
        safe_tblis_mult( 1.0, t_Bvv,  "abP",  t_X,   "ibP", 0.0, t_Za, "ai");
        safe_tblis_mult(-1.0, t_Vooov, "jikc", t_Teff, "jkac", 1.0, t_Za, "ai");
    }
    T_eff.resize(0, 0);

    // 2. Kontribusi Particle-Particle Ladder bebas O(V^4) via chunked Y_ooov (<= 64 MB)
    if (include_vvvv) {
        constexpr size_t kMaxChunkBytes = 64ULL * 1024ULL * 1024ULL;
        const size_t slice_bytes = static_cast<size_t>(n_vir) * n_vir * n_vir * sizeof(double);
        const int mb_max = (slice_bytes > 0)
            ? std::max(1, static_cast<int>(kMaxChunkBytes / slice_bytes))
            : n_occ;

        for (int m0 = 0; m0 < n_occ; m0 += mb_max) {
            const int mb = std::min(mb_max, n_occ - m0);
            Eigen::MatrixXd B_chunk_buf;
            const Eigen::MatrixXd* B_chunk_ptr = &B_ov;
            if (m0 != 0 || mb != n_occ) {
                B_chunk_buf = B_ov.middleRows(
                    static_cast<Eigen::Index>(m0) * n_vir,
                    static_cast<Eigen::Index>(mb) * n_vir);
                B_chunk_ptr = &B_chunk_buf;
            }
            Eigen::MatrixXd M_chunk = parallel_gemm_abt(*B_chunk_ptr, B_vv);
            auto t_Vchunk = make_mat_view_4d(M_chunk, mb, n_vir, n_vir, n_vir);

            Eigen::Tensor<double, 4> Y_chunk(n_occ, n_occ, mb, n_vir);
            Y_chunk.setZero();
            varray_view<double> t_Y(
                {(len_type)n_occ, (len_type)n_occ, (len_type)mb, (len_type)n_vir},
                Y_chunk.data(),
                {1, (stride_type)n_occ,
                 (stride_type)(static_cast<ptrdiff_t>(n_occ) * n_occ),
                 (stride_type)(static_cast<ptrdiff_t>(n_occ) * n_occ * mb)});
            varray_view<double> t_Zslice(
                {(len_type)n_vir, (len_type)mb},
                Z_vo.col(m0).data(),
                {1, (stride_type)n_vir});

            safe_tblis_mult( 1.0, t_T2t, "ijab", t_Vchunk, "mafb", 0.0, t_Y, "ijmf");
            safe_tblis_mult(-1.0, t_T2,  "ijxf", t_Y,      "ijmf", 1.0, t_Zslice, "xm");
        }
    }

    // 3. Kontribusi Hole-Hole Ladder (OOOO)
    {
        Eigen::Tensor<double, 4> G_oooo(n_occ, n_occ, n_occ, n_occ);
        G_oooo.setZero();
        const ptrdiff_t o2 = static_cast<ptrdiff_t>(n_occ) * n_occ;
        varray_view<double> t_Goooo(
            {(len_type)n_occ, (len_type)n_occ, (len_type)n_occ, (len_type)n_occ},
            G_oooo.data(),
            {1, (stride_type)n_occ, (stride_type)o2, (stride_type)(o2 * n_occ)});
        safe_tblis_mult(0.25, t_T2t,   "ijab", t_T2,    "mnab", 0.0, t_Goooo, "minj");
        safe_tblis_mult(4.0,  t_Goooo, "minj", t_Vooov, "njix", 1.0, t_Za,    "xm");
    }

    // 4. Kontribusi Particle-Hole Ring (OVOV)
    {
        Eigen::Tensor<double, 4> G_ovov(n_occ, n_vir, n_occ, n_vir);
        G_ovov.setZero();
        const ptrdiff_t ov = static_cast<ptrdiff_t>(n_occ) * n_vir;
        varray_view<double> t_Govov(
            {(len_type)n_occ, (len_type)n_vir, (len_type)n_occ, (len_type)n_vir},
            G_ovov.data(),
            {1, (stride_type)n_occ, (stride_type)ov, (stride_type)(ov * n_occ)});
        safe_tblis_mult(0.50, t_T2t, "kjcb", t_T2t, "ijab", 0.0, t_Govov, "iakc");

        Eigen::Tensor<double, 3> W_ov(n_occ, n_vir, n_aux);
        W_ov.setZero();
        varray_view<double> t_Wov(
            {(len_type)n_occ, (len_type)n_vir, (len_type)n_aux},
            W_ov.data(),
            {1, (stride_type)n_occ, (stride_type)ov});
        safe_tblis_mult( 1.0, t_Govov, "makc", t_Bov,   "kcP",  0.0, t_Wov, "maP");
        safe_tblis_mult( 2.0, t_Bvv,   "xaP",  t_Wov,   "maP",  1.0, t_Za,  "xm");
        safe_tblis_mult(-2.0, t_Govov, "ixkc", t_Vooov, "imkc", 1.0, t_Za,  "xm");
    }

    // 5. Kontribusi Particle-Hole Ring Exchange (OOVV)
    {
        const ptrdiff_t o2 = static_cast<ptrdiff_t>(n_occ) * n_occ;
        Eigen::Tensor<double, 4> G_oovv(n_occ, n_occ, n_vir, n_vir);
        G_oovv.setZero();
        varray_view<double> t_Goovv(
            {(len_type)n_occ, (len_type)n_occ, (len_type)n_vir, (len_type)n_vir},
            G_oovv.data(),
            {1, (stride_type)n_occ, (stride_type)o2, (stride_type)(o2 * n_vir)});

        {
            Eigen::Tensor<double, 4> G_t1(n_occ, n_occ, n_vir, n_vir);
            G_t1.setZero();
            varray_view<double> t_Gt1(
                {(len_type)n_occ, (len_type)n_occ, (len_type)n_vir, (len_type)n_vir},
                G_t1.data(),
                {1, (stride_type)n_occ, (stride_type)o2, (stride_type)(o2 * n_vir)});
            safe_tblis_mult(1.0, t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Gt1, "ikac");
            #pragma omp parallel for collapse(4) schedule(static)
            for (int i = 0; i < n_occ; ++i)
                for (int k = 0; k < n_occ; ++k)
                    for (int a = 0; a < n_vir; ++a)
                        for (int c = 0; c < n_vir; ++c)
                            G_oovv(i, k, a, c) = -0.25 * (G_t1(i, k, a, c) + G_t1(k, i, c, a));
        }
        safe_tblis_mult(-0.50, t_T2, "kmec", t_T2t, "imea", 1.0, t_Goovv, "ikac");

        Eigen::Tensor<double, 3> U_oo(n_occ, n_occ, n_aux);
        U_oo.setZero();
        varray_view<double> t_Uoo(
            {(len_type)n_occ, (len_type)n_occ, (len_type)n_aux},
            U_oo.data(),
            {1, (stride_type)n_occ, (stride_type)o2});

        safe_tblis_mult( 1.0, t_Goovv, "mkac", t_Bvv,   "caP",  0.0, t_Uoo, "mkP");
        safe_tblis_mult( 1.0, t_Bov,   "kxP",  t_Uoo,   "mkP",  1.0, t_Za,  "xm");
        safe_tblis_mult( 1.0, t_Goovv, "imac", t_Bvv,   "acP",  0.0, t_Uoo, "imP");
        safe_tblis_mult( 1.0, t_Bov,   "ixP",  t_Uoo,   "imP",  1.0, t_Za,  "xm");
        safe_tblis_mult(-1.0, t_Goovv, "ikxc", t_Vooov, "ikmc", 1.0, t_Za,  "xm");
        safe_tblis_mult(-1.0, t_Goovv, "ikax", t_Vooov, "ikma", 1.0, t_Za,  "xm");
    }

    return Z_vo;
}

Eigen::MatrixXd apply_mo_df_hessian_rhf(
    const Eigen::MatrixXd& kappa_vo,
    const Eigen::VectorXd& eps_occ,
    const Eigen::VectorXd& eps_vir,
    const Eigen::MatrixXd& B_oo,
    const Eigen::MatrixXd& B_ov,
    const Eigen::MatrixXd& B_vv)
{
    const int no = static_cast<int>(eps_occ.size());
    const int nv = static_cast<int>(eps_vir.size());
    const int n_aux = static_cast<int>(B_ov.cols());
    if (kappa_vo.rows() != nv || kappa_vo.cols() != no) {
        throw std::invalid_argument("apply_mo_df_hessian_rhf: dimensi kappa_vo harus (nv x no)");
    }

    Eigen::MatrixXd Hp_vo(nv, no);
    for (int i = 0; i < no; ++i) {
        for (int a = 0; a < nv; ++a) {
            const double diff = std::max(std::abs(eps_vir(a) - eps_occ(i)), 1e-4);
            Hp_vo(a, i) = 4.0 * diff * kappa_vo(a, i);
        }
    }

    // J-response: 4 * sum_P B_ov^P * (4 * (B_ov^P . kappa_vo))
    Eigen::Map<const Eigen::VectorXd> k_vec(kappa_vo.data(), static_cast<Eigen::Index>(nv) * no);
    Eigen::VectorXd w_aux = 4.0 * (B_ov.transpose() * k_vec);
    Eigen::VectorXd j_flat = B_ov * w_aux;
    Eigen::MatrixXd H_kappa_vo = Eigen::Map<const Eigen::MatrixXd>(j_flat.data(), nv, no);

    // K-response: O(N_aux * O^2 * V + N_aux * O * V^2)
    Eigen::MatrixXd kappa_ov = kappa_vo.transpose(); // (no x nv)
    const int max_t = omp_get_max_threads();
    std::vector<Eigen::MatrixXd> K_tls(max_t, Eigen::MatrixXd::Zero(nv, no));

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        Eigen::MatrixXd T_vo(nv, no), T_oo2(no, no);
        #pragma omp for schedule(static) nowait
        for (int P = 0; P < n_aux; ++P) {
            Eigen::Map<const Eigen::MatrixXd> Boo(B_oo.col(P).data(), no, no);
            Eigen::Map<const Eigen::MatrixXd> Bvo(B_ov.col(P).data(), nv, no);
            Eigen::Map<const Eigen::MatrixXd> Bvv(B_vv.col(P).data(), nv, nv);
            T_vo.noalias() = kappa_vo * Boo;
            K_tls[tid].noalias() += Bvv * T_vo;
            T_oo2.noalias() = kappa_ov * Bvo;
            K_tls[tid].noalias() += Bvo * T_oo2;
        }
    }
    for (int t = 0; t < max_t; ++t) H_kappa_vo.noalias() -= K_tls[t];

    Hp_vo.noalias() += 4.0 * H_kappa_vo;
    return Hp_vo;
}


Eigen::MatrixXd eval_mo_gcorr_rhf_df(
    const Eigen::MatrixXd& B_oo,
    const Eigen::MatrixXd& B_ov,
    const Eigen::MatrixXd& B_vv,
    const Eigen::MatrixXd& g_oo,
    const Eigen::MatrixXd& g_vv,
    int n_occ,
    int n_vir)
{
    const int n_aux = static_cast<int>(B_ov.cols());
    Eigen::MatrixXd G_vo = Eigen::MatrixXd::Zero(n_vir, n_occ);
    if (n_occ <= 0 || n_vir <= 0 || n_aux <= 0) return G_vo;
    if (B_oo.rows() != static_cast<Eigen::Index>(n_occ) * n_occ ||
        B_ov.rows() != static_cast<Eigen::Index>(n_occ) * n_vir ||
        B_vv.rows() != static_cast<Eigen::Index>(n_vir) * n_vir ||
        B_oo.cols() != n_aux || B_vv.cols() != n_aux) {
        throw std::invalid_argument("eval_mo_gcorr_rhf_df: dimensi B_oo/B_ov/B_vv tidak cocok dengan (n_occ, n_vir)");
    }

    // 1. Coulomb ruang MO: 2 * sum_P B^P_vo * [Tr(B^P_oo g_oo) + Tr(B^P_vv g_vv)]
    Eigen::Map<const Eigen::VectorXd> goo_v(g_oo.data(), static_cast<Eigen::Index>(n_occ) * n_occ);
    Eigen::Map<const Eigen::VectorXd> gvv_v(g_vv.data(), static_cast<Eigen::Index>(n_vir) * n_vir);
    Eigen::VectorXd c_aux = B_oo.transpose() * goo_v + B_vv.transpose() * gvv_v;
    Eigen::VectorXd j_flat = 2.0 * (B_ov * c_aux);
    G_vo = Eigen::Map<const Eigen::MatrixXd>(j_flat.data(), n_vir, n_occ);

    // 2. Exchange ruang MO: -sum_P [B^P_vo (g_oo B^P_oo) + B^P_vv (g_vv B^P_vo)]
    const int max_t = omp_get_max_threads();
    std::vector<Eigen::MatrixXd> K_tls(max_t, Eigen::MatrixXd::Zero(n_vir, n_occ));

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        Eigen::MatrixXd& K_loc = K_tls[tid];
        Eigen::MatrixXd T_oo(n_occ, n_occ), T_vo(n_vir, n_occ);
        #pragma omp for schedule(static) nowait
        for (int P = 0; P < n_aux; ++P) {
            Eigen::Map<const Eigen::MatrixXd> Boo(B_oo.col(P).data(), n_occ, n_occ);
            Eigen::Map<const Eigen::MatrixXd> Bvo(B_ov.col(P).data(), n_vir, n_occ);
            Eigen::Map<const Eigen::MatrixXd> Bvv(B_vv.col(P).data(), n_vir, n_vir);
            T_oo.noalias() = g_oo * Boo;
            K_loc.noalias() += Bvo * T_oo;
            T_vo.noalias() = g_vv * Bvo;
            K_loc.noalias() += Bvv * T_vo;
        }
    }
    for (int t = 0; t < max_t; ++t) G_vo.noalias() -= K_tls[t];
    return G_vo;
}

} // namespace kernels
} // namespace mshqc
