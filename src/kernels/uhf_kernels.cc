// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0
// [INJECTED_HYBRID_PHASE3_UHF_KERNELS_CC]

#include "mshqc/kernels/uhf_kernels.h"
#include "mshqc/kernels/mp3_kernels.h"
#include "mshqc/kernels/primitives.h"
#include "mshqc/kernels/tblis_safe.h"
#include "mshqc/integrals/packed_ao_utils.h"

#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdexcept>
#include <vector>

namespace mshqc {
namespace kernels {

using tblis::len_type;
using tblis::stride_type;
using tblis::varray_view;

namespace {

inline varray_view<double> make_t2_view_4d_rect(
    const Eigen::MatrixXd& M, int no1, int no2, int nv1, int nv2)
{
    const ptrdiff_t nv12 = static_cast<ptrdiff_t>(nv1) * nv2;
    return varray_view<double>(
        {(len_type)no1, (len_type)no2, (len_type)nv1, (len_type)nv2},
        const_cast<double*>(M.data()),
        {(stride_type)(static_cast<ptrdiff_t>(no2) * nv12),
         (stride_type)nv12,
         1,
         (stride_type)nv1});
}

inline varray_view<double> make_bov_view_3d(const Eigen::MatrixXd& B_ov, int no, int nv, int n_aux) {
    return varray_view<double>(
        {(len_type)no, (len_type)nv, (len_type)n_aux},
        const_cast<double*>(B_ov.data()),
        {(stride_type)nv, 1, (stride_type)(static_cast<ptrdiff_t>(no) * nv)});
}

inline varray_view<double> make_bvv_view_3d(const Eigen::MatrixXd& B_vv, int nv, int n_aux) {
    return varray_view<double>(
        {(len_type)nv, (len_type)nv, (len_type)n_aux},
        const_cast<double*>(B_vv.data()),
        {(stride_type)nv, 1, (stride_type)(static_cast<ptrdiff_t>(nv) * nv)});
}

inline void accum_vvvv_ladder_uhf_chunked(
    double scale,
    const Eigen::MatrixXd& T_bra, int no1, int no2, int nv1, int nv2,
    const Eigen::MatrixXd& T_ket,
    const Eigen::MatrixXd& B_ov_m, int nm, int nv_m,
    const Eigen::MatrixXd& B_vv_r, int nv_r,
    bool swap_ab_mode,
    Eigen::MatrixXd& Z_target)
{
    const int n_aux = static_cast<int>(B_vv_r.cols());
    if (nm <= 0 || no1 <= 0 || no2 <= 0 || nv1 <= 0 || nv2 <= 0 || n_aux <= 0) return;

    constexpr size_t kMaxChunkBytes = 64ULL * 1024ULL * 1024ULL;
    const size_t slice_bytes = static_cast<size_t>(nv_m) * nv_r * nv_r * sizeof(double);
    const int mb_max = (slice_bytes > 0)
        ? std::max(1, static_cast<int>(kMaxChunkBytes / slice_bytes))
        : nm;

    auto t_Tbra = make_t2_view_4d_rect(T_bra, no1, no2, nv1, nv2);
    auto t_Tket = make_t2_view_4d_rect(T_ket, no1, no2, nv1, nv2);

    for (int m0 = 0; m0 < nm; m0 += mb_max) {
        const int mb = std::min(mb_max, nm - m0);
        Eigen::MatrixXd B_chunk_buf;
        const Eigen::MatrixXd* B_chunk_ptr = &B_ov_m;
        if (m0 != 0 || mb != nm) {
            B_chunk_buf = B_ov_m.middleRows(
                static_cast<Eigen::Index>(m0) * nv_m,
                static_cast<Eigen::Index>(mb) * nv_m);
            B_chunk_ptr = &B_chunk_buf;
        }
        Eigen::MatrixXd M_chunk = parallel_gemm_abt(*B_chunk_ptr, B_vv_r);
        auto t_Vchunk = make_mat_view_4d(M_chunk, mb, nv_m, nv_r, nv_r);

        Eigen::Tensor<double, 4> Y_chunk(no1, no2, mb, nv_r);
        Y_chunk.setZero();
        varray_view<double> t_Y(
            {(len_type)no1, (len_type)no2, (len_type)mb, (len_type)nv_r},
            Y_chunk.data(),
            {1, (stride_type)no1,
             (stride_type)(static_cast<ptrdiff_t>(no1) * no2),
             (stride_type)(static_cast<ptrdiff_t>(no1) * no2 * mb)});
        varray_view<double> t_Zslice(
            {(len_type)nv_m, (len_type)mb},
            Z_target.col(m0).data(),
            {1, (stride_type)nv_m});

        if (!swap_ab_mode) {
            safe_tblis_mult(1.0,   t_Tbra, "ijab", t_Vchunk, "mafb", 0.0, t_Y,      "ijmf");
            safe_tblis_mult(scale, t_Tket, "ijxf", t_Y,      "ijmf", 1.0, t_Zslice, "xm");
        } else {
            safe_tblis_mult(1.0,   t_Tbra, "ijab", t_Vchunk, "mbea", 0.0, t_Y,      "ijme");
            safe_tblis_mult(scale, t_Tket, "ijex", t_Y,      "ijme", 1.0, t_Zslice, "xm");
        }
    }
}

} // namespace


std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> build_jk_3c_uhf(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& C_occ_a,
    const Eigen::MatrixXd& C_occ_b,
    int nbf)
{
    if (!integrals::has_valid_ao_factor(L_mat, nbf)) {
        throw std::invalid_argument("build_jk_3c_uhf: dimensi L_mat tidak cocok dengan nbf");
    }
    const int na = static_cast<int>(C_occ_a.cols());
    const int nb = static_cast<int>(C_occ_b.cols());
    const int n_aux = static_cast<int>(L_mat.cols());

    Eigen::MatrixXd P_tot = Eigen::MatrixXd::Zero(nbf, nbf);
    if (na > 0) P_tot.noalias() += C_occ_a * C_occ_a.transpose();
    if (nb > 0) P_tot.noalias() += C_occ_b * C_occ_b.transpose();
    Eigen::MatrixXd J = build_coulomb_3c(L_mat, P_tot, nbf);

    Eigen::MatrixXd Ka = Eigen::MatrixXd::Zero(nbf, nbf);
    Eigen::MatrixXd Kb = Eigen::MatrixXd::Zero(nbf, nbf);
    if (n_aux == 0 || (na == 0 && nb == 0)) {
        return {std::move(J), std::move(Ka), std::move(Kb)};
    }

    const bool packed = integrals::is_packed_ao_matrix(L_mat, nbf);
    const int n_tot_o = na + nb;
    Eigen::MatrixXd Cab_o(nbf, n_tot_o);
    if (na > 0) Cab_o.leftCols(na) = C_occ_a;
    if (nb > 0) Cab_o.rightCols(nb) = C_occ_b;

    const int max_o = std::max(1, std::max(na, nb));
    const int batch_p = std::clamp(
        static_cast<int>((8ULL * 1024ULL * 1024ULL) /
            std::max<size_t>(1, static_cast<size_t>(nbf) * static_cast<size_t>(max_o) * sizeof(double))),
        1, 32);

    const int max_threads = omp_get_max_threads();
    std::vector<Eigen::MatrixXd> Ka_tls(na > 0 ? max_threads : 0, Eigen::MatrixXd::Zero(nbf, nbf));
    std::vector<Eigen::MatrixXd> Kb_tls(nb > 0 ? max_threads : 0, Eigen::MatrixXd::Zero(nbf, nbf));

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        Eigen::MatrixXd L_P_buf;
        if (packed) L_P_buf.resize(nbf, nbf);
        Eigen::MatrixXd Tab_loc(nbf, n_tot_o);
        Eigen::MatrixXd Ta_batch, Tb_batch;
        if (na > 0) Ta_batch.resize(nbf, static_cast<Eigen::Index>(na) * batch_p);
        if (nb > 0) Tb_batch.resize(nbf, static_cast<Eigen::Index>(nb) * batch_p);
        int cnt = 0;

        #pragma omp for schedule(static) nowait
        for (int P = 0; P < n_aux; ++P) {
            const double* col_p = L_mat.col(P).data();
            if (packed) integrals::unpack_ao_column(col_p, nbf, true, L_P_buf);
            Eigen::Map<const Eigen::MatrixXd> L_P(packed ? L_P_buf.data() : col_p, nbf, nbf);

            Tab_loc.noalias() = L_P * Cab_o;
            if (na > 0) Ta_batch.middleCols(static_cast<Eigen::Index>(cnt) * na, na) = Tab_loc.leftCols(na);
            if (nb > 0) Tb_batch.middleCols(static_cast<Eigen::Index>(cnt) * nb, nb) = Tab_loc.rightCols(nb);
            ++cnt;

            if (cnt == batch_p) {
                if (na > 0) Ka_tls[tid].noalias() += Ta_batch * Ta_batch.transpose();
                if (nb > 0) Kb_tls[tid].noalias() += Tb_batch * Tb_batch.transpose();
                cnt = 0;
            }
        }
        if (cnt > 0) {
            if (na > 0) {
                auto Ta_rem = Ta_batch.leftCols(static_cast<Eigen::Index>(cnt) * na);
                Ka_tls[tid].noalias() += Ta_rem * Ta_rem.transpose();
            }
            if (nb > 0) {
                auto Tb_rem = Tb_batch.leftCols(static_cast<Eigen::Index>(cnt) * nb);
                Kb_tls[tid].noalias() += Tb_rem * Tb_rem.transpose();
            }
        }
    }

    for (int t = 0; t < max_threads; ++t) {
        if (na > 0) Ka.noalias() += Ka_tls[t];
        if (nb > 0) Kb.noalias() += Kb_tls[t];
    }
    if (na > 0) Ka = 0.5 * (Ka + Ka.transpose());
    if (nb > 0) Kb = 0.5 * (Kb + Kb.transpose());
    return {std::move(J), std::move(Ka), std::move(Kb)};
}


std::tuple<double, double, double, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> eval_ump2_df(
    const Eigen::MatrixXd& B_ov_a,
    const Eigen::MatrixXd& B_ov_b,
    const Eigen::VectorXd& eps_oa,
    const Eigen::VectorXd& eps_va,
    const Eigen::VectorXd& eps_ob,
    const Eigen::VectorXd& eps_vb)
{
    const int na = static_cast<int>(eps_oa.size());
    const int va = static_cast<int>(eps_va.size());
    const int nb = static_cast<int>(eps_ob.size());
    const int vb = static_cast<int>(eps_vb.size());

    Eigen::MatrixXd T2_aa = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(va) * va, static_cast<Eigen::Index>(na) * na);
    Eigen::MatrixXd T2_bb = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(vb) * vb, static_cast<Eigen::Index>(nb) * nb);
    Eigen::MatrixXd T2_ab = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(va) * vb, static_cast<Eigen::Index>(na) * nb);

    double e_aa = 0.0, e_bb = 0.0, e_ab = 0.0;

    auto eval_ss = [](const Eigen::MatrixXd& B_ov, const Eigen::VectorXd& eo,
                      const Eigen::VectorXd& ev, Eigen::MatrixXd& T2) -> double {
        const int no = static_cast<int>(eo.size());
        const int nv = static_cast<int>(ev.size());
        if (no <= 1 || nv <= 0) return 0.0;
        double e_tot = 0.0;
        #pragma omp parallel
        {
            Eigen::MatrixXd I_ij(nv, nv);
            double loc = 0.0;
            #pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < no; ++i) {
                auto B_i = B_ov.middleRows(static_cast<Eigen::Index>(i) * nv, nv);
                for (int j = i + 1; j < no; ++j) {
                    auto B_j = B_ov.middleRows(static_cast<Eigen::Index>(j) * nv, nv);
                    I_ij.noalias() = B_i * B_j.transpose();
                    const double eij = eo(i) + eo(j);
                    Eigen::Map<Eigen::MatrixXd> T_ij(
                        T2.col(static_cast<Eigen::Index>(i) * no + j).data(), nv, nv);
                    Eigen::Map<Eigen::MatrixXd> T_ji(
                        T2.col(static_cast<Eigen::Index>(j) * no + i).data(), nv, nv);
                    for (int b = 0; b < nv; ++b) {
                        const double eij_b = eij - ev(b);
                        for (int a = 0; a < nv; ++a) {
                            const double denom = eij_b - ev(a);
                            const double val = I_ij(a, b) - I_ij(b, a);
                            const double t = (std::abs(denom) > 1e-12) ? (val / denom) : 0.0;
                            T_ij(a, b) = t;
                            T_ji(a, b) = -t;
                            loc += 0.5 * t * val; // 2 (for i<j + j>i) * 0.25 = 0.5
                        }
                    }
                }
            }
            #pragma omp atomic
            e_tot += loc;
        }
        return e_tot;
    };

    e_aa = eval_ss(B_ov_a, eps_oa, eps_va, T2_aa);
    e_bb = eval_ss(B_ov_b, eps_ob, eps_vb, T2_bb);

    if (na > 0 && nb > 0 && va > 0 && vb > 0) {
        #pragma omp parallel
        {
            Eigen::MatrixXd I_ij(va, vb);
            double loc = 0.0;
            #pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < na; ++i) {
                auto B_i = B_ov_a.middleRows(static_cast<Eigen::Index>(i) * va, va);
                for (int j = 0; j < nb; ++j) {
                    auto B_j = B_ov_b.middleRows(static_cast<Eigen::Index>(j) * vb, vb);
                    I_ij.noalias() = B_i * B_j.transpose();
                    const double eij = eps_oa(i) + eps_ob(j);
                    Eigen::Map<Eigen::MatrixXd> T_ij(
                        T2_ab.col(static_cast<Eigen::Index>(i) * nb + j).data(), va, vb);
                    for (int b = 0; b < vb; ++b) {
                        const double eij_b = eij - eps_vb(b);
                        for (int a = 0; a < va; ++a) {
                            const double denom = eij_b - eps_va(a);
                            const double v = I_ij(a, b);
                            const double t = (std::abs(denom) > 1e-12) ? (v / denom) : 0.0;
                            T_ij(a, b) = t;
                            loc += t * v;
                        }
                    }
                }
            }
            #pragma omp atomic
            e_ab += loc;
        }
    }

    return {e_aa, e_bb, e_ab, std::move(T2_aa), std::move(T2_bb), std::move(T2_ab)};
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> eval_omp2_1rdm_uhf(
    const Eigen::MatrixXd& T2_aa,
    const Eigen::MatrixXd& T2_bb,
    const Eigen::MatrixXd& T2_ab,
    int na, int va, int nb, int vb,
    double c_os, double c_ss)
{
    Eigen::MatrixXd g_oo_a = Eigen::MatrixXd::Zero(na, na);
    Eigen::MatrixXd g_vv_a = Eigen::MatrixXd::Zero(va, va);
    Eigen::MatrixXd g_oo_b = Eigen::MatrixXd::Zero(nb, nb);
    Eigen::MatrixXd g_vv_b = Eigen::MatrixXd::Zero(vb, vb);

    if (na > 1 && va > 0) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < na; ++i) {
            for (int j = 0; j < na; ++j) {
                double val = 0.0;
                for (int k = 0; k < na; ++k) {
                    val += T2_aa.col(static_cast<Eigen::Index>(i) * na + k)
                               .dot(T2_aa.col(static_cast<Eigen::Index>(j) * na + k));
                }
                g_oo_a(i, j) -= 0.5 * c_ss * val;
            }
        }
        #pragma omp parallel
        {
            Eigen::MatrixXd loc = Eigen::MatrixXd::Zero(va, va);
            #pragma omp for schedule(static)
            for (int idx = 0; idx < na * na; ++idx) {
                Eigen::Map<const Eigen::MatrixXd> T_ij(T2_aa.col(idx).data(), va, va);
                loc.noalias() += T_ij * T_ij.transpose();
            }
            #pragma omp critical
            g_vv_a.noalias() += 0.5 * c_ss * loc;
        }
    }

    if (nb > 1 && vb > 0) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < nb; ++i) {
            for (int j = 0; j < nb; ++j) {
                double val = 0.0;
                for (int k = 0; k < nb; ++k) {
                    val += T2_bb.col(static_cast<Eigen::Index>(i) * nb + k)
                               .dot(T2_bb.col(static_cast<Eigen::Index>(j) * nb + k));
                }
                g_oo_b(i, j) -= 0.5 * c_ss * val;
            }
        }
        #pragma omp parallel
        {
            Eigen::MatrixXd loc = Eigen::MatrixXd::Zero(vb, vb);
            #pragma omp for schedule(static)
            for (int idx = 0; idx < nb * nb; ++idx) {
                Eigen::Map<const Eigen::MatrixXd> T_ij(T2_bb.col(idx).data(), vb, vb);
                loc.noalias() += T_ij * T_ij.transpose();
            }
            #pragma omp critical
            g_vv_b.noalias() += 0.5 * c_ss * loc;
        }
    }

    if (na > 0 && nb > 0 && va > 0 && vb > 0) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < na; ++i) {
            for (int j = 0; j < na; ++j) {
                double val = 0.0;
                for (int kb = 0; kb < nb; ++kb) {
                    val += T2_ab.col(static_cast<Eigen::Index>(i) * nb + kb)
                               .dot(T2_ab.col(static_cast<Eigen::Index>(j) * nb + kb));
                }
                g_oo_a(i, j) -= c_os * val;
            }
        }
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < nb; ++i) {
            for (int j = 0; j < nb; ++j) {
                double val = 0.0;
                for (int ka = 0; ka < na; ++ka) {
                    val += T2_ab.col(static_cast<Eigen::Index>(ka) * nb + i)
                               .dot(T2_ab.col(static_cast<Eigen::Index>(ka) * nb + j));
                }
                g_oo_b(i, j) -= c_os * val;
            }
        }
        #pragma omp parallel
        {
            Eigen::MatrixXd loc_a = Eigen::MatrixXd::Zero(va, va);
            Eigen::MatrixXd loc_b = Eigen::MatrixXd::Zero(vb, vb);
            #pragma omp for schedule(static)
            for (int idx = 0; idx < na * nb; ++idx) {
                Eigen::Map<const Eigen::MatrixXd> T_ij(T2_ab.col(idx).data(), va, vb);
                loc_a.noalias() += T_ij * T_ij.transpose();
                loc_b.noalias() += T_ij.transpose() * T_ij;
            }
            #pragma omp critical
            {
                g_vv_a.noalias() += c_os * loc_a;
                g_vv_b.noalias() += c_os * loc_b;
            }
        }
    }

    g_oo_a = 0.5 * (g_oo_a + g_oo_a.transpose());
    g_vv_a = 0.5 * (g_vv_a + g_vv_a.transpose());
    g_oo_b = 0.5 * (g_oo_b + g_oo_b.transpose());
    g_vv_b = 0.5 * (g_vv_b + g_vv_b.transpose());
    return {std::move(g_oo_a), std::move(g_vv_a), std::move(g_oo_b), std::move(g_vv_b)};
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> eval_omp2_zvec_uhf_df(
    const Eigen::MatrixXd& T2_aa,
    const Eigen::MatrixXd& T2_bb,
    const Eigen::MatrixXd& T2_ab,
    const Eigen::MatrixXd& B_oo_a,
    const Eigen::MatrixXd& B_ov_a,
    const Eigen::MatrixXd& B_vv_a,
    const Eigen::MatrixXd& B_oo_b,
    const Eigen::MatrixXd& B_ov_b,
    const Eigen::MatrixXd& B_vv_b,
    int na, int va, int nb, int vb,
    double c_os, double c_ss)
{
    const int n_aux = static_cast<int>(B_ov_a.cols());
    Eigen::MatrixXd Z_a = Eigen::MatrixXd::Zero(va, na);
    Eigen::MatrixXd Z_b = Eigen::MatrixXd::Zero(vb, nb);
    if (n_aux <= 0) return {std::move(Z_a), std::move(Z_b)};

    Eigen::MatrixXd X_a = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(na) * va, n_aux);
    Eigen::MatrixXd X_b = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(nb) * vb, n_aux);

    if (na > 0 && va > 0) {
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < na; ++i) {
            auto X_ai = X_a.middleRows(static_cast<Eigen::Index>(i) * va, va);
            if (na > 1) {
                for (int j = 0; j < na; ++j) {
                    Eigen::Map<const Eigen::MatrixXd> T_ij(
                        T2_aa.col(static_cast<Eigen::Index>(i) * na + j).data(), va, va);
                    X_ai.noalias() += c_ss * (T_ij * B_ov_a.middleRows(static_cast<Eigen::Index>(j) * va, va));
                }
            }
            if (nb > 0 && vb > 0) {
                for (int jb = 0; jb < nb; ++jb) {
                    Eigen::Map<const Eigen::MatrixXd> T_ij(
                        T2_ab.col(static_cast<Eigen::Index>(i) * nb + jb).data(), va, vb);
                    X_ai.noalias() += c_os * (T_ij * B_ov_b.middleRows(static_cast<Eigen::Index>(jb) * vb, vb));
                }
            }
        }
    }

    if (nb > 0 && vb > 0) {
        #pragma omp parallel for schedule(dynamic, 1)
        for (int ib = 0; ib < nb; ++ib) {
            auto X_bi = X_b.middleRows(static_cast<Eigen::Index>(ib) * vb, vb);
            if (nb > 1) {
                for (int jb = 0; jb < nb; ++jb) {
                    Eigen::Map<const Eigen::MatrixXd> T_ij(
                        T2_bb.col(static_cast<Eigen::Index>(ib) * nb + jb).data(), vb, vb);
                    X_bi.noalias() += c_ss * (T_ij * B_ov_b.middleRows(static_cast<Eigen::Index>(jb) * vb, vb));
                }
            }
            if (na > 0 && va > 0) {
                for (int ia = 0; ia < na; ++ia) {
                    Eigen::Map<const Eigen::MatrixXd> T_ij(
                        T2_ab.col(static_cast<Eigen::Index>(ia) * nb + ib).data(), va, vb);
                    X_bi.noalias() += c_os * (T_ij.transpose() * B_ov_a.middleRows(static_cast<Eigen::Index>(ia) * va, va));
                }
            }
        }
    }

    #pragma omp parallel
    {
        Eigen::MatrixXd Z_loc_a = Eigen::MatrixXd::Zero(va, na);
        Eigen::MatrixXd Z_loc_b = Eigen::MatrixXd::Zero(vb, nb);
        #pragma omp for schedule(static)
        for (int P = 0; P < n_aux; ++P) {
            if (na > 0 && va > 0) {
                Eigen::Map<const Eigen::MatrixXd> Boo(B_oo_a.col(P).data(), na, na);
                Eigen::Map<const Eigen::MatrixXd> Bvv(B_vv_a.col(P).data(), va, va);
                Eigen::Map<const Eigen::MatrixXd> X_P(X_a.col(P).data(), va, na);
                Z_loc_a.noalias() += Bvv * X_P - X_P * Boo;
            }
            if (nb > 0 && vb > 0) {
                Eigen::Map<const Eigen::MatrixXd> Boo(B_oo_b.col(P).data(), nb, nb);
                Eigen::Map<const Eigen::MatrixXd> Bvv(B_vv_b.col(P).data(), vb, vb);
                Eigen::Map<const Eigen::MatrixXd> X_P(X_b.col(P).data(), vb, nb);
                Z_loc_b.noalias() += Bvv * X_P - X_P * Boo;
            }
        }
        #pragma omp critical
        {
            Z_a.noalias() += Z_loc_a;
            Z_b.noalias() += Z_loc_b;
        }
    }
    return {std::move(Z_a), std::move(Z_b)};
}

std::tuple<double, double, double, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> eval_ump3_df(
    const Eigen::MatrixXd& T2_aa,
    const Eigen::MatrixXd& T2_bb,
    const Eigen::MatrixXd& T2_ab,
    const Eigen::MatrixXd& B_oo_a,
    const Eigen::MatrixXd& B_ov_a,
    const Eigen::MatrixXd& B_vv_a,
    const Eigen::MatrixXd& B_oo_b,
    const Eigen::MatrixXd& B_ov_b,
    const Eigen::MatrixXd& B_vv_b,
    const Eigen::VectorXd& eps_oa,
    const Eigen::VectorXd& eps_va,
    const Eigen::VectorXd& eps_ob,
    const Eigen::VectorXd& eps_vb)
{
    const int na = static_cast<int>(eps_oa.size());
    const int va = static_cast<int>(eps_va.size());
    const int nb = static_cast<int>(eps_ob.size());
    const int vb = static_cast<int>(eps_vb.size());
    const int n_aux = static_cast<int>(B_ov_a.cols());
    const bool has_beta = (nb > 0 && vb > 0);

    Eigen::MatrixXd L2_aa = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(va) * va, static_cast<Eigen::Index>(na) * na);
    Eigen::MatrixXd L2_bb = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(vb) * vb, static_cast<Eigen::Index>(nb) * nb);
    Eigen::MatrixXd L2_ab = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(va) * vb, static_cast<Eigen::Index>(na) * nb);
    double e3_aa = 0.0, e3_bb = 0.0, e3_ab = 0.0;
    if (na <= 0 || va <= 0 || n_aux <= 0) {
        return {0.0, 0.0, 0.0, std::move(L2_aa), std::move(L2_bb), std::move(L2_ab)};
    }

    auto t_Taa = make_t2_view_4d(T2_aa, na, va);
    auto t_Tbb = make_t2_view_4d(T2_bb, nb, vb);
    auto t_Tab = make_t2_view_4d_rect(T2_ab, na, nb, va, vb);

    Eigen::MatrixXd M_oooo_aa = parallel_gemm_abt(B_oo_a, B_oo_a);
    Eigen::MatrixXd M_ovov_aa = parallel_gemm_abt(B_ov_a, B_ov_a);
    Eigen::MatrixXd M_oovv_aa = parallel_gemm_abt(B_oo_a, B_vv_a);
    Eigen::MatrixXd M_ovov_ab;
    if (has_beta) M_ovov_ab = parallel_gemm_abt(B_ov_a, B_ov_b);

    // 1. Channel AA
    if (na > 1) {
        Eigen::MatrixXd Waa_l = Eigen::MatrixXd::Zero(T2_aa.rows(), T2_aa.cols());
        Eigen::MatrixXd Waa_r = Eigen::MatrixXd::Zero(T2_aa.rows(), T2_aa.cols());
        auto t_Wl = make_t2_view_4d(Waa_l, na, va);
        auto t_Wr = make_t2_view_4d(Waa_r, na, va);
        auto t_Voooo = make_mat_view_4d(M_oooo_aa, na, na, na, na);
        auto t_Vovov = make_mat_view_4d(M_ovov_aa, na, va, na, va);
        auto t_Voovv = make_mat_view_4d(M_oovv_aa, na, na, va, va);

        safe_tblis_mult( 0.5, t_Taa, "mnab", t_Voooo, "minj", 1.0, t_Wl, "ijab");
        safe_tblis_mult(-0.5, t_Taa, "mnab", t_Voooo, "mjni", 1.0, t_Wl, "ijab");
        safe_tblis_mult( 1.0, t_Vovov, "iakc", t_Taa, "kjcb", 1.0, t_Wr, "ijab");
        safe_tblis_mult(-1.0, t_Voovv, "ikac", t_Taa, "kjcb", 1.0, t_Wr, "ijab");
        if (has_beta) {
            auto t_Vovov_ab = make_mat_view_4d(M_ovov_ab, na, va, nb, vb);
            safe_tblis_mult(1.0, t_Vovov_ab, "iakc", t_Tab, "jkbc", 1.0, t_Wr, "ijab");
        }

        const int n_p = na * (na - 1) / 2;
        std::vector<Eigen::MatrixXd> Tp(n_p, Eigen::MatrixXd(va, va));
        std::vector<Eigen::MatrixXd> Wp(n_p, Eigen::MatrixXd::Zero(va, va));
        int pi = 0;
        for (int i = 0; i < na; ++i)
            for (int j = i + 1; j < na; ++j) {
                Eigen::Map<const Eigen::MatrixXd> Tij(T2_aa.col(static_cast<Eigen::Index>(i) * na + j).data(), va, va);
                Tp[pi++] = Tij;
            }
        contract_vvvv_pairs_df(B_vv_a, va, B_vv_a, va, n_aux, Tp, Wp);
        pi = 0;
        for (int i = 0; i < na; ++i)
            for (int j = i + 1; j < na; ++j) {
                Eigen::MatrixXd W_asym = 0.5 * (Wp[pi] - Wp[pi].transpose());
                Eigen::Map<Eigen::MatrixXd> dst_ij(Waa_l.col(static_cast<Eigen::Index>(i) * na + j).data(), va, va);
                Eigen::Map<Eigen::MatrixXd> dst_ji(Waa_l.col(static_cast<Eigen::Index>(j) * na + i).data(), va, va);
                dst_ij.noalias() += W_asym;
                dst_ji.noalias() -= W_asym;
                ++pi;
            }

        #pragma omp parallel for collapse(2) reduction(+:e3_aa) schedule(static)
        for (int i = 0; i < na; ++i) {
            for (int j = 0; j < na; ++j) {
                const double eij = eps_oa(i) + eps_oa(j);
                for (int b = 0; b < va; ++b) {
                    const double eij_b = eij - eps_va(b);
                    for (int a = 0; a < va; ++a) {
                        const double ra = t2_at(Waa_r, na, va, i, j, a, b)
                                        - t2_at(Waa_r, na, va, j, i, a, b)
                                        - t2_at(Waa_r, na, va, i, j, b, a)
                                        + t2_at(Waa_r, na, va, j, i, b, a);
                        const double wt = t2_at(Waa_l, na, va, i, j, a, b) + ra;
                        const double D = eij_b - eps_va(a);
                        t2_at(L2_aa, na, va, i, j, a, b) = (std::abs(D) > 1e-12) ? (wt / D) : 0.0;
                        e3_aa += 0.25 * t2_at(T2_aa, na, va, i, j, a, b) * wt;
                    }
                }
            }
        }
    }
    M_oooo_aa.resize(0, 0);

    // 2. Channels BB & AB
    if (has_beta) {
        Eigen::MatrixXd M_oooo_bb = parallel_gemm_abt(B_oo_b, B_oo_b);
        Eigen::MatrixXd M_ovov_bb = parallel_gemm_abt(B_ov_b, B_ov_b);
        Eigen::MatrixXd M_oovv_bb = parallel_gemm_abt(B_oo_b, B_vv_b);

        if (nb > 1) {
            Eigen::MatrixXd Wbb_l = Eigen::MatrixXd::Zero(T2_bb.rows(), T2_bb.cols());
            Eigen::MatrixXd Wbb_r = Eigen::MatrixXd::Zero(T2_bb.rows(), T2_bb.cols());
            auto t_Wl = make_t2_view_4d(Wbb_l, nb, vb);
            auto t_Wr = make_t2_view_4d(Wbb_r, nb, vb);
            auto t_Voooo = make_mat_view_4d(M_oooo_bb, nb, nb, nb, nb);
            auto t_Vovov = make_mat_view_4d(M_ovov_bb, nb, vb, nb, vb);
            auto t_Voovv = make_mat_view_4d(M_oovv_bb, nb, nb, vb, vb);
            auto t_Vovov_ab = make_mat_view_4d(M_ovov_ab, na, va, nb, vb);

            safe_tblis_mult( 0.5, t_Tbb, "mnab", t_Voooo, "minj", 1.0, t_Wl, "ijab");
            safe_tblis_mult(-0.5, t_Tbb, "mnab", t_Voooo, "mjni", 1.0, t_Wl, "ijab");
            safe_tblis_mult( 1.0, t_Vovov, "iakc", t_Tbb, "kjcb", 1.0, t_Wr, "ijab");
            safe_tblis_mult(-1.0, t_Voovv, "ikac", t_Tbb, "kjcb", 1.0, t_Wr, "ijab");
            safe_tblis_mult( 1.0, t_Vovov_ab, "kcia", t_Tab, "kjcb", 1.0, t_Wr, "ijab");

            const int n_p = nb * (nb - 1) / 2;
            std::vector<Eigen::MatrixXd> Tp(n_p, Eigen::MatrixXd(vb, vb));
            std::vector<Eigen::MatrixXd> Wp(n_p, Eigen::MatrixXd::Zero(vb, vb));
            int pi = 0;
            for (int i = 0; i < nb; ++i)
                for (int j = i + 1; j < nb; ++j) {
                    Eigen::Map<const Eigen::MatrixXd> Tij(T2_bb.col(static_cast<Eigen::Index>(i) * nb + j).data(), vb, vb);
                    Tp[pi++] = Tij;
                }
            contract_vvvv_pairs_df(B_vv_b, vb, B_vv_b, vb, n_aux, Tp, Wp);
            pi = 0;
            for (int i = 0; i < nb; ++i)
                for (int j = i + 1; j < nb; ++j) {
                    Eigen::MatrixXd W_asym = 0.5 * (Wp[pi] - Wp[pi].transpose());
                    Eigen::Map<Eigen::MatrixXd> dst_ij(Wbb_l.col(static_cast<Eigen::Index>(i) * nb + j).data(), vb, vb);
                    Eigen::Map<Eigen::MatrixXd> dst_ji(Wbb_l.col(static_cast<Eigen::Index>(j) * nb + i).data(), vb, vb);
                    dst_ij.noalias() += W_asym;
                    dst_ji.noalias() -= W_asym;
                    ++pi;
                }

            #pragma omp parallel for collapse(2) reduction(+:e3_bb) schedule(static)
            for (int i = 0; i < nb; ++i) {
                for (int j = 0; j < nb; ++j) {
                    const double eij = eps_ob(i) + eps_ob(j);
                    for (int b = 0; b < vb; ++b) {
                        const double eij_b = eij - eps_vb(b);
                        for (int a = 0; a < vb; ++a) {
                            const double ra = t2_at(Wbb_r, nb, vb, i, j, a, b)
                                            - t2_at(Wbb_r, nb, vb, j, i, a, b)
                                            - t2_at(Wbb_r, nb, vb, i, j, b, a)
                                            + t2_at(Wbb_r, nb, vb, j, i, b, a);
                            const double wt = t2_at(Wbb_l, nb, vb, i, j, a, b) + ra;
                            const double D = eij_b - eps_vb(a);
                            t2_at(L2_bb, nb, vb, i, j, a, b) = (std::abs(D) > 1e-12) ? (wt / D) : 0.0;
                            e3_bb += 0.25 * t2_at(T2_bb, nb, vb, i, j, a, b) * wt;
                        }
                    }
                }
            }
        }
        M_oooo_bb.resize(0, 0);

        // Channel AB
        {
            Eigen::MatrixXd Wab_l = Eigen::MatrixXd::Zero(T2_ab.rows(), T2_ab.cols());
            Eigen::MatrixXd Wab_r = Eigen::MatrixXd::Zero(T2_ab.rows(), T2_ab.cols());
            auto t_Wl = make_t2_view_4d_rect(Wab_l, na, nb, va, vb);
            auto t_Wr = make_t2_view_4d_rect(Wab_r, na, nb, va, vb);

            Eigen::MatrixXd M_oooo_ab = parallel_gemm_abt(B_oo_a, B_oo_b);
            auto t_Voooo_ab = make_mat_view_4d(M_oooo_ab, na, na, nb, nb);
            safe_tblis_mult(1.0, t_Tab, "mnab", t_Voooo_ab, "minj", 1.0, t_Wl, "ijab");

            auto t_Vovov_aa = make_mat_view_4d(M_ovov_aa, na, va, na, va);
            auto t_Voovv_aa = make_mat_view_4d(M_oovv_aa, na, na, va, va);
            auto t_Vovov_bb = make_mat_view_4d(M_ovov_bb, nb, vb, nb, vb);
            auto t_Voovv_bb = make_mat_view_4d(M_oovv_bb, nb, nb, vb, vb);
            auto t_Vovov_ab = make_mat_view_4d(M_ovov_ab, na, va, nb, vb);

            safe_tblis_mult( 1.0, t_Vovov_aa, "iakc", t_Tab, "kjcb", 1.0, t_Wr, "ijab");
            safe_tblis_mult(-1.0, t_Voovv_aa, "ikac", t_Tab, "kjcb", 1.0, t_Wr, "ijab");
            safe_tblis_mult( 1.0, t_Tab, "ikac", t_Vovov_bb, "kcjb", 1.0, t_Wr, "ijab");
            safe_tblis_mult(-1.0, t_Tab, "ikac", t_Voovv_bb, "kjcb", 1.0, t_Wr, "ijab");
            if (na > 1) safe_tblis_mult(1.0, t_Taa, "ikac", t_Vovov_ab, "kcjb", 1.0, t_Wr, "ijab");
            if (nb > 1) safe_tblis_mult(1.0, t_Vovov_ab, "iakc", t_Tbb, "kjcb", 1.0, t_Wr, "ijab");

            Eigen::MatrixXd M_oovv_ab_ex = parallel_gemm_abt(B_oo_a, B_vv_b);
            Eigen::MatrixXd M_oovv_ba_ex = parallel_gemm_abt(B_oo_b, B_vv_a);
            auto t_Voovv_ab_ex = make_mat_view_4d(M_oovv_ab_ex, na, na, vb, vb);
            auto t_Voovv_ba_ex = make_mat_view_4d(M_oovv_ba_ex, nb, nb, va, va);
            safe_tblis_mult(-1.0, t_Voovv_ab_ex, "ikbc", t_Tab, "kjac", 1.0, t_Wr, "ijab");
            safe_tblis_mult(-1.0, t_Tab, "ikcb", t_Voovv_ba_ex, "jkac", 1.0, t_Wr, "ijab");

            const int n_p = na * nb;
            std::vector<Eigen::MatrixXd> Tp(n_p, Eigen::MatrixXd(va, vb));
            std::vector<Eigen::MatrixXd> Wp(n_p, Eigen::MatrixXd::Zero(va, vb));
            for (int k = 0; k < n_p; ++k) {
                Eigen::Map<const Eigen::MatrixXd> Tij(T2_ab.col(k).data(), va, vb);
                Tp[k] = Tij;
            }
            contract_vvvv_pairs_df(B_vv_a, va, B_vv_b, vb, n_aux, Tp, Wp);
            for (int k = 0; k < n_p; ++k) {
                Eigen::Map<Eigen::MatrixXd> dst(Wab_l.col(k).data(), va, vb);
                dst.noalias() += Wp[k];
            }

            #pragma omp parallel for collapse(2) reduction(+:e3_ab) schedule(static)
            for (int i = 0; i < na; ++i) {
                for (int j = 0; j < nb; ++j) {
                    const double eij = eps_oa(i) + eps_ob(j);
                    const Eigen::Index col = static_cast<Eigen::Index>(i) * nb + j;
                    Eigen::Map<const Eigen::MatrixXd> Tij(T2_ab.col(col).data(), va, vb);
                    Eigen::Map<const Eigen::MatrixXd> Wl(Wab_l.col(col).data(), va, vb);
                    Eigen::Map<const Eigen::MatrixXd> Wr(Wab_r.col(col).data(), va, vb);
                    Eigen::Map<Eigen::MatrixXd> Lij(L2_ab.col(col).data(), va, vb);
                    for (int b = 0; b < vb; ++b) {
                        const double eij_b = eij - eps_vb(b);
                        for (int a = 0; a < va; ++a) {
                            const double wt = Wl(a, b) + Wr(a, b);
                            const double D = eij_b - eps_va(a);
                            Lij(a, b) = (std::abs(D) > 1e-12) ? (wt / D) : 0.0;
                            e3_ab += Tij(a, b) * wt;
                        }
                    }
                }
            }
        }
    }

    return {e3_aa, e3_bb, e3_ab, std::move(L2_aa), std::move(L2_bb), std::move(L2_ab)};
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> eval_omp3_1rdm_uhf(
    const Eigen::MatrixXd& T2_aa,
    const Eigen::MatrixXd& T2_bb,
    const Eigen::MatrixXd& T2_ab,
    const Eigen::MatrixXd& L2_aa,
    const Eigen::MatrixXd& L2_bb,
    const Eigen::MatrixXd& L2_ab,
    int na, int va, int nb, int vb)
{
    Eigen::MatrixXd g_oo_a = Eigen::MatrixXd::Zero(na, na);
    Eigen::MatrixXd g_vv_a = Eigen::MatrixXd::Zero(va, va);
    Eigen::MatrixXd g_oo_b = Eigen::MatrixXd::Zero(nb, nb);
    Eigen::MatrixXd g_vv_b = Eigen::MatrixXd::Zero(vb, vb);

    auto accum_ss = [](const Eigen::MatrixXd& T2, const Eigen::MatrixXd& L2,
                       int no, int nv, Eigen::MatrixXd& g_oo, Eigen::MatrixXd& g_vv) {
        if (no <= 1 || nv <= 0) return;
        Eigen::MatrixXd Sum = T2 + L2;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < no; ++i) {
            for (int j = 0; j < no; ++j) {
                double val = 0.0;
                for (int k = 0; k < no; ++k) {
                    const Eigen::Index ik = static_cast<Eigen::Index>(i) * no + k;
                    const Eigen::Index jk = static_cast<Eigen::Index>(j) * no + k;
                    val += T2.col(ik).dot(Sum.col(jk)) + L2.col(ik).dot(T2.col(jk));
                }
                g_oo(i, j) -= 0.5 * val;
            }
        }
        #pragma omp parallel
        {
            Eigen::MatrixXd loc = Eigen::MatrixXd::Zero(nv, nv);
            #pragma omp for schedule(static)
            for (int idx = 0; idx < no * no; ++idx) {
                Eigen::Map<const Eigen::MatrixXd> Tij(T2.col(idx).data(), nv, nv);
                Eigen::Map<const Eigen::MatrixXd> Lij(L2.col(idx).data(), nv, nv);
                Eigen::Map<const Eigen::MatrixXd> Sij(Sum.col(idx).data(), nv, nv);
                loc.noalias() += Tij * Sij.transpose() + Lij * Tij.transpose();
            }
            #pragma omp critical
            g_vv.noalias() += 0.5 * loc;
        }
    };

    accum_ss(T2_aa, L2_aa, na, va, g_oo_a, g_vv_a);
    accum_ss(T2_bb, L2_bb, nb, vb, g_oo_b, g_vv_b);

    if (na > 0 && nb > 0 && va > 0 && vb > 0) {
        Eigen::MatrixXd Sum_ab = T2_ab + L2_ab;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < na; ++i) {
            for (int j = 0; j < na; ++j) {
                double val = 0.0;
                for (int kb = 0; kb < nb; ++kb) {
                    const Eigen::Index ik = static_cast<Eigen::Index>(i) * nb + kb;
                    const Eigen::Index jk = static_cast<Eigen::Index>(j) * nb + kb;
                    val += T2_ab.col(ik).dot(Sum_ab.col(jk)) + L2_ab.col(ik).dot(T2_ab.col(jk));
                }
                g_oo_a(i, j) -= val;
            }
        }
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < nb; ++i) {
            for (int j = 0; j < nb; ++j) {
                double val = 0.0;
                for (int ka = 0; ka < na; ++ka) {
                    const Eigen::Index ki = static_cast<Eigen::Index>(ka) * nb + i;
                    const Eigen::Index kj = static_cast<Eigen::Index>(ka) * nb + j;
                    val += T2_ab.col(ki).dot(Sum_ab.col(kj)) + L2_ab.col(ki).dot(T2_ab.col(kj));
                }
                g_oo_b(i, j) -= val;
            }
        }
        #pragma omp parallel
        {
            Eigen::MatrixXd loc_a = Eigen::MatrixXd::Zero(va, va);
            Eigen::MatrixXd loc_b = Eigen::MatrixXd::Zero(vb, vb);
            #pragma omp for schedule(static)
            for (int idx = 0; idx < na * nb; ++idx) {
                Eigen::Map<const Eigen::MatrixXd> Tij(T2_ab.col(idx).data(), va, vb);
                Eigen::Map<const Eigen::MatrixXd> Lij(L2_ab.col(idx).data(), va, vb);
                Eigen::Map<const Eigen::MatrixXd> Sij(Sum_ab.col(idx).data(), va, vb);
                loc_a.noalias() += Tij * Sij.transpose() + Lij * Tij.transpose();
                loc_b.noalias() += Tij.transpose() * Sij + Lij.transpose() * Tij;
            }
            #pragma omp critical
            {
                g_vv_a.noalias() += loc_a;
                g_vv_b.noalias() += loc_b;
            }
        }
    }

    g_oo_a = 0.5 * (g_oo_a + g_oo_a.transpose());
    g_vv_a = 0.5 * (g_vv_a + g_vv_a.transpose());
    g_oo_b = 0.5 * (g_oo_b + g_oo_b.transpose());
    g_vv_b = 0.5 * (g_vv_b + g_vv_b.transpose());
    return {std::move(g_oo_a), std::move(g_vv_a), std::move(g_oo_b), std::move(g_vv_b)};
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> eval_omp3_zvec_uhf_df(
    const Eigen::MatrixXd& T2_aa,
    const Eigen::MatrixXd& T2_bb,
    const Eigen::MatrixXd& T2_ab,
    const Eigen::MatrixXd& L2_aa,
    const Eigen::MatrixXd& L2_bb,
    const Eigen::MatrixXd& L2_ab,
    const Eigen::MatrixXd& B_oo_a,
    const Eigen::MatrixXd& B_ov_a,
    const Eigen::MatrixXd& B_vv_a,
    const Eigen::MatrixXd& B_oo_b,
    const Eigen::MatrixXd& B_ov_b,
    const Eigen::MatrixXd& B_vv_b,
    int na, int va, int nb, int vb)
{
    const int n_aux = static_cast<int>(B_ov_a.cols());
    Eigen::MatrixXd Z_a = Eigen::MatrixXd::Zero(va, na);
    Eigen::MatrixXd Z_b = Eigen::MatrixXd::Zero(vb, nb);
    if (na <= 0 || va <= 0 || n_aux <= 0) return {std::move(Z_a), std::move(Z_b)};

    const bool has_beta = (nb > 0 && vb > 0);
    varray_view<double> t_Za({(len_type)va, (len_type)na}, Z_a.data(), {1, (stride_type)va});
    auto t_Bov_a = make_bov_view_3d(B_ov_a, na, va, n_aux);
    auto t_Bvv_a = make_bvv_view_3d(B_vv_a, va, n_aux);
    auto t_T2_aa = make_t2_view_4d(T2_aa, na, va);
    auto t_T2_ab = make_t2_view_4d_rect(T2_ab, na, nb, va, vb);

    Eigen::MatrixXd M_ooov_aa = parallel_gemm_abt(B_oo_a, B_ov_a);
    auto t_Vooov_aa = make_mat_view_4d(M_ooov_aa, na, na, na, va);

    // 1. T_eff_aa
    {
        Eigen::MatrixXd Teff_aa = T2_aa + L2_aa;
        auto t_Teff = make_t2_view_4d(Teff_aa, na, va);
        Eigen::Tensor<double, 3> X_a(na, va, n_aux);
        X_a.setZero();
        varray_view<double> t_Xa(
            {(len_type)na, (len_type)va, (len_type)n_aux}, X_a.data(),
            {1, (stride_type)na, (stride_type)(static_cast<ptrdiff_t>(na) * va)});
        safe_tblis_mult( 1.0, t_Teff, "ikbc", t_Bov_a, "kcP", 0.0, t_Xa, "ibP");
        safe_tblis_mult( 1.0, t_Bvv_a, "abP", t_Xa,    "ibP", 0.0, t_Za, "ai");
        safe_tblis_mult(-1.0, t_Vooov_aa, "jikc", t_Teff, "jkac", 1.0, t_Za, "ai");
    }

    // 2. AA VVVV ladder (chunked Y_ooov)
    accum_vvvv_ladder_uhf_chunked(-0.5, T2_aa, na, na, va, va, T2_aa, B_ov_a, na, va, B_vv_a, va, false, Z_a);

    // 3. AA OOOO ladder
    {
        const ptrdiff_t o2 = static_cast<ptrdiff_t>(na) * na;
        Eigen::Tensor<double, 4> G_oooo(na, na, na, na);
        G_oooo.setZero();
        varray_view<double> t_G(
            {(len_type)na, (len_type)na, (len_type)na, (len_type)na}, G_oooo.data(),
            {1, (stride_type)na, (stride_type)o2, (stride_type)(o2 * na)});
        safe_tblis_mult(0.125, t_T2_aa, "ijab", t_T2_aa, "mnab", 0.0, t_G,  "minj");
        safe_tblis_mult(4.0,   t_G,     "minj", t_Vooov_aa, "njix", 1.0, t_Za, "xm");
    }

    // 4. AA OVOV ring
    {
        const ptrdiff_t ov = static_cast<ptrdiff_t>(na) * va;
        Eigen::Tensor<double, 4> G_ovov(na, va, na, va);
        G_ovov.setZero();
        varray_view<double> t_G(
            {(len_type)na, (len_type)va, (len_type)na, (len_type)va}, G_ovov.data(),
            {1, (stride_type)na, (stride_type)ov, (stride_type)(ov * na)});
        safe_tblis_mult(0.5, t_T2_aa, "kjcb", t_T2_aa, "ijab", 0.0, t_G, "iakc");
        if (has_beta) safe_tblis_mult(0.5, t_T2_ab, "kjcb", t_T2_ab, "ijab", 1.0, t_G, "iakc");

        Eigen::Tensor<double, 3> W_ov(na, va, n_aux);
        W_ov.setZero();
        varray_view<double> t_W(
            {(len_type)na, (len_type)va, (len_type)n_aux}, W_ov.data(),
            {1, (stride_type)na, (stride_type)ov});
        safe_tblis_mult( 1.0, t_G, "makc", t_Bov_a, "kcP", 0.0, t_W, "maP");
        safe_tblis_mult( 1.0, t_G, "iamc", t_Bov_a, "iaP", 1.0, t_W, "mcP");
        safe_tblis_mult( 1.0, t_Bvv_a, "xaP", t_W, "maP", 1.0, t_Za, "xm");
        safe_tblis_mult(-1.0, t_G, "ixkc", t_Vooov_aa, "imkc", 1.0, t_Za, "xm");
        safe_tblis_mult(-1.0, t_G, "iakx", t_Vooov_aa, "kmia", 1.0, t_Za, "xm");
    }

    // 5. AA OOVV ring
    {
        const ptrdiff_t o2 = static_cast<ptrdiff_t>(na) * na;
        Eigen::Tensor<double, 4> G_oovv(na, na, va, va);
        G_oovv.setZero();
        varray_view<double> t_G(
            {(len_type)na, (len_type)na, (len_type)va, (len_type)va}, G_oovv.data(),
            {1, (stride_type)na, (stride_type)o2, (stride_type)(o2 * va)});
        safe_tblis_mult(-0.5, t_T2_aa, "kjcb", t_T2_aa, "ijab", 0.0, t_G, "ikac");
        if (has_beta) safe_tblis_mult(-0.5, t_T2_ab, "kjcb", t_T2_ab, "ijab", 1.0, t_G, "ikac");

        Eigen::Tensor<double, 3> U_oo(na, na, n_aux);
        U_oo.setZero();
        varray_view<double> t_U(
            {(len_type)na, (len_type)na, (len_type)n_aux}, U_oo.data(),
            {1, (stride_type)na, (stride_type)o2});
        safe_tblis_mult( 1.0, t_G, "mkac", t_Bvv_a, "caP", 0.0, t_U, "mkP");
        safe_tblis_mult( 1.0, t_Bov_a, "kxP", t_U, "mkP", 1.0, t_Za, "xm");
        safe_tblis_mult( 1.0, t_G, "imac", t_Bvv_a, "acP", 0.0, t_U, "imP");
        safe_tblis_mult( 1.0, t_Bov_a, "ixP", t_U, "imP", 1.0, t_Za, "xm");
        safe_tblis_mult(-1.0, t_G, "ikxc", t_Vooov_aa, "ikmc", 1.0, t_Za, "xm");
        safe_tblis_mult(-1.0, t_G, "ikax", t_Vooov_aa, "ikma", 1.0, t_Za, "xm");
    }
    M_ooov_aa.resize(0, 0);

    // 6. Beta & Alpha-Beta cross contributions
    if (has_beta) {
        varray_view<double> t_Zb({(len_type)vb, (len_type)nb}, Z_b.data(), {1, (stride_type)vb});
        auto t_Bov_b = make_bov_view_3d(B_ov_b, nb, vb, n_aux);
        auto t_Bvv_b = make_bvv_view_3d(B_vv_b, vb, n_aux);
        auto t_T2_bb = make_t2_view_4d(T2_bb, nb, vb);

        Eigen::MatrixXd M_ooov_bb    = parallel_gemm_abt(B_oo_b, B_ov_b);
        Eigen::MatrixXd M_ooov_aa_bb = parallel_gemm_abt(B_oo_a, B_ov_b);
        Eigen::MatrixXd M_ooov_bb_aa = parallel_gemm_abt(B_oo_b, B_ov_a);

        auto t_Vooov_bb    = make_mat_view_4d(M_ooov_bb,    nb, nb, nb, vb);
        auto t_Vooov_aa_bb = make_mat_view_4d(M_ooov_aa_bb, na, na, nb, vb);
        auto t_Vooov_bb_aa = make_mat_view_4d(M_ooov_bb_aa, nb, nb, na, va);

        // T_eff_bb & T_eff_ab
        {
            Eigen::Tensor<double, 3> X_b(nb, vb, n_aux);
            X_b.setZero();
            varray_view<double> t_Xb(
                {(len_type)nb, (len_type)vb, (len_type)n_aux}, X_b.data(),
                {1, (stride_type)nb, (stride_type)(static_cast<ptrdiff_t>(nb) * vb)});

            Eigen::MatrixXd Teff_bb = T2_bb + L2_bb;
            auto t_Teff_bb = make_t2_view_4d(Teff_bb, nb, vb);
            safe_tblis_mult( 1.0, t_Teff_bb, "ikbc", t_Bov_b, "kcP", 0.0, t_Xb, "ibP");
            safe_tblis_mult(-1.0, t_Vooov_bb, "jikc", t_Teff_bb, "jkac", 1.0, t_Zb, "ai");

            Eigen::MatrixXd Teff_ab = T2_ab + L2_ab;
            auto t_Teff_ab = make_t2_view_4d_rect(Teff_ab, na, nb, va, vb);
            Eigen::Tensor<double, 3> X_ab_a(na, va, n_aux);
            X_ab_a.setZero();
            varray_view<double> t_Xab_a(
                {(len_type)na, (len_type)va, (len_type)n_aux}, X_ab_a.data(),
                {1, (stride_type)na, (stride_type)(static_cast<ptrdiff_t>(na) * va)});

            safe_tblis_mult( 1.0, t_Teff_ab, "ikbc", t_Bov_b, "kcP", 0.0, t_Xab_a, "ibP");
            safe_tblis_mult( 1.0, t_Bvv_a,   "abP",  t_Xab_a, "ibP", 1.0, t_Za,    "ai");
            safe_tblis_mult(-1.0, t_Vooov_aa_bb, "jikc", t_Teff_ab, "jkac", 1.0, t_Za, "ai");

            safe_tblis_mult( 1.0, t_Teff_ab, "kicb", t_Bov_a, "kcP", 1.0, t_Xb, "ibP");
            safe_tblis_mult(-1.0, t_Vooov_bb_aa, "jikc", t_Teff_ab, "kjca", 1.0, t_Zb, "ai");

            safe_tblis_mult( 1.0, t_Bvv_b, "abP", t_Xb, "ibP", 1.0, t_Zb, "ai");
        }

        // VVVV ladders (BB, AB->a, AB->b)
        accum_vvvv_ladder_uhf_chunked(-0.5, T2_bb, nb, nb, vb, vb, T2_bb, B_ov_b, nb, vb, B_vv_b, vb, false, Z_b);
        accum_vvvv_ladder_uhf_chunked(-1.0, T2_ab, na, nb, va, vb, T2_ab, B_ov_a, na, va, B_vv_b, vb, false, Z_a);
        accum_vvvv_ladder_uhf_chunked(-1.0, T2_ab, na, nb, va, vb, T2_ab, B_ov_b, nb, vb, B_vv_a, va, true,  Z_b);

        // OOOO ladders (BB & AB)
        {
            const ptrdiff_t ob2 = static_cast<ptrdiff_t>(nb) * nb;
            const ptrdiff_t oa2 = static_cast<ptrdiff_t>(na) * na;
            Eigen::Tensor<double, 4> G_bb(nb, nb, nb, nb); G_bb.setZero();
            Eigen::Tensor<double, 4> G_ab(na, na, nb, nb); G_ab.setZero();
            varray_view<double> t_Gbb({(len_type)nb, (len_type)nb, (len_type)nb, (len_type)nb}, G_bb.data(),
                                      {1, (stride_type)nb, (stride_type)ob2, (stride_type)(ob2 * nb)});
            varray_view<double> t_Gab({(len_type)na, (len_type)na, (len_type)nb, (len_type)nb}, G_ab.data(),
                                      {1, (stride_type)na, (stride_type)oa2, (stride_type)(oa2 * nb)});
            safe_tblis_mult(0.125, t_T2_bb, "ijab", t_T2_bb, "mnab", 0.0, t_Gbb, "minj");
            safe_tblis_mult(0.500, t_T2_ab, "ijab", t_T2_ab, "mnab", 0.0, t_Gab, "minj");

            safe_tblis_mult(1.0, t_Gab, "minj", t_Vooov_bb_aa, "njix", 1.0, t_Za, "xm");
            safe_tblis_mult(1.0, t_Gab, "imnj", t_Vooov_bb_aa, "njix", 1.0, t_Za, "xm");
            safe_tblis_mult(1.0, t_Gab, "inmj", t_Vooov_aa_bb, "injx", 1.0, t_Zb, "xm");
            safe_tblis_mult(1.0, t_Gab, "injm", t_Vooov_aa_bb, "injx", 1.0, t_Zb, "xm");

            safe_tblis_mult(1.0, t_Gbb, "minj", t_Vooov_bb, "njix", 1.0, t_Zb, "xm");
            safe_tblis_mult(1.0, t_Gbb, "imnj", t_Vooov_bb, "njix", 1.0, t_Zb, "xm");
            safe_tblis_mult(1.0, t_Gbb, "inmj", t_Vooov_bb, "injx", 1.0, t_Zb, "xm");
            safe_tblis_mult(1.0, t_Gbb, "injm", t_Vooov_bb, "injx", 1.0, t_Zb, "xm");
        }

        // OVOV BB
        {
            const ptrdiff_t ov = static_cast<ptrdiff_t>(nb) * vb;
            Eigen::Tensor<double, 4> G_bb(nb, vb, nb, vb); G_bb.setZero();
            varray_view<double> t_G({(len_type)nb, (len_type)vb, (len_type)nb, (len_type)vb}, G_bb.data(),
                                    {1, (stride_type)nb, (stride_type)ov, (stride_type)(ov * nb)});
            safe_tblis_mult(0.5, t_T2_bb, "kjcb", t_T2_bb, "ijab", 0.0, t_G, "iakc");
            safe_tblis_mult(0.5, t_T2_ab, "miea", t_T2_ab, "mkec", 1.0, t_G, "iakc");

            Eigen::Tensor<double, 3> W_b(nb, vb, n_aux); W_b.setZero();
            varray_view<double> t_W({(len_type)nb, (len_type)vb, (len_type)n_aux}, W_b.data(),
                                    {1, (stride_type)nb, (stride_type)ov});
            safe_tblis_mult( 1.0, t_G, "makc", t_Bov_b, "kcP", 0.0, t_W, "maP");
            safe_tblis_mult( 1.0, t_G, "iamc", t_Bov_b, "iaP", 1.0, t_W, "mcP");
            safe_tblis_mult( 1.0, t_Bvv_b, "xaP", t_W, "maP", 1.0, t_Zb, "xm");
            safe_tblis_mult(-1.0, t_G, "ixkc", t_Vooov_bb, "imkc", 1.0, t_Zb, "xm");
            safe_tblis_mult(-1.0, t_G, "iakx", t_Vooov_bb, "kmia", 1.0, t_Zb, "xm");
        }

        // OVOV AB
        {
            const ptrdiff_t ova = static_cast<ptrdiff_t>(na) * va;
            const ptrdiff_t ovb = static_cast<ptrdiff_t>(nb) * vb;
            Eigen::Tensor<double, 4> G_ab(na, va, nb, vb); G_ab.setZero();
            varray_view<double> t_G({(len_type)na, (len_type)va, (len_type)nb, (len_type)vb}, G_ab.data(),
                                    {1, (stride_type)na, (stride_type)ova, (stride_type)(ova * nb)});
            safe_tblis_mult(1.0, t_T2_aa, "miea", t_T2_ab, "mkec", 0.0, t_G, "iakc");
            safe_tblis_mult(1.0, t_T2_bb, "kjcb", t_T2_ab, "ijab", 1.0, t_G, "iakc");

            Eigen::Tensor<double, 3> W_b(nb, vb, n_aux); W_b.setZero();
            varray_view<double> t_Wb({(len_type)nb, (len_type)vb, (len_type)n_aux}, W_b.data(),
                                     {1, (stride_type)nb, (stride_type)ovb});
            safe_tblis_mult(1.0, t_G, "iamc", t_Bov_a, "iaP", 0.0, t_Wb, "mcP");
            safe_tblis_mult(1.0, t_Bvv_b, "xaP", t_Wb, "maP", 1.0, t_Zb, "xm");

            Eigen::Tensor<double, 3> W_a(na, va, n_aux); W_a.setZero();
            varray_view<double> t_Wa({(len_type)na, (len_type)va, (len_type)n_aux}, W_a.data(),
                                     {1, (stride_type)na, (stride_type)ova});
            safe_tblis_mult(1.0, t_G, "makc", t_Bov_b, "kcP", 0.0, t_Wa, "maP");
            safe_tblis_mult(1.0, t_Bvv_a, "xaP", t_Wa, "maP", 1.0, t_Za, "xm");

            safe_tblis_mult(-1.0, t_G, "ixkc", t_Vooov_aa_bb, "imkc", 1.0, t_Za, "xm");
            safe_tblis_mult(-1.0, t_G, "iakx", t_Vooov_bb_aa, "kmia", 1.0, t_Zb, "xm");
        }

        // OOVV BB
        {
            const ptrdiff_t o2 = static_cast<ptrdiff_t>(nb) * nb;
            Eigen::Tensor<double, 4> G_bb(nb, nb, vb, vb); G_bb.setZero();
            varray_view<double> t_G({(len_type)nb, (len_type)nb, (len_type)vb, (len_type)vb}, G_bb.data(),
                                    {1, (stride_type)nb, (stride_type)o2, (stride_type)(o2 * vb)});
            safe_tblis_mult(-0.5, t_T2_bb, "kjcb", t_T2_bb, "ijab", 0.0, t_G, "ikac");
            safe_tblis_mult(-0.5, t_T2_ab, "miea", t_T2_ab, "mkec", 1.0, t_G, "ikac");

            Eigen::Tensor<double, 3> U_b(nb, nb, n_aux); U_b.setZero();
            varray_view<double> t_U({(len_type)nb, (len_type)nb, (len_type)n_aux}, U_b.data(),
                                    {1, (stride_type)nb, (stride_type)o2});
            safe_tblis_mult( 1.0, t_G, "mkac", t_Bvv_b, "caP", 0.0, t_U, "mkP");
            safe_tblis_mult( 1.0, t_Bov_b, "kxP", t_U, "mkP", 1.0, t_Zb, "xm");
            safe_tblis_mult( 1.0, t_G, "imac", t_Bvv_b, "acP", 0.0, t_U, "imP");
            safe_tblis_mult( 1.0, t_Bov_b, "ixP", t_U, "imP", 1.0, t_Zb, "xm");
            safe_tblis_mult(-1.0, t_G, "ikxc", t_Vooov_bb, "ikmc", 1.0, t_Zb, "xm");
            safe_tblis_mult(-1.0, t_G, "ikax", t_Vooov_bb, "ikma", 1.0, t_Zb, "xm");
        }

        // OOVV AB_ex & BA_ex
        {
            const ptrdiff_t oa2 = static_cast<ptrdiff_t>(na) * na;
            Eigen::Tensor<double, 4> G_ab_ex(na, na, vb, vb); G_ab_ex.setZero();
            varray_view<double> t_G({(len_type)na, (len_type)na, (len_type)vb, (len_type)vb}, G_ab_ex.data(),
                                    {1, (stride_type)na, (stride_type)oa2, (stride_type)(oa2 * vb)});
            safe_tblis_mult(-0.5, t_T2_ab, "kmec", t_T2_ab, "imea", 0.0, t_G, "ikac");

            Eigen::Tensor<double, 3> U_a(na, na, n_aux); U_a.setZero();
            varray_view<double> t_U({(len_type)na, (len_type)na, (len_type)n_aux}, U_a.data(),
                                    {1, (stride_type)na, (stride_type)oa2});
            safe_tblis_mult( 1.0, t_G, "mkac", t_Bvv_b, "acP", 0.0, t_U, "mkP");
            safe_tblis_mult( 1.0, t_Bov_a, "kxP", t_U, "mkP", 1.0, t_Za, "xm");
            safe_tblis_mult( 1.0, t_Bov_a, "ixP", t_U, "imP", 1.0, t_Za, "xm");
            safe_tblis_mult(-1.0, t_G, "ikxc", t_Vooov_aa_bb, "ikmc", 1.0, t_Zb, "xm");
            safe_tblis_mult(-1.0, t_G, "ikax", t_Vooov_aa_bb, "ikma", 1.0, t_Zb, "xm");
        }
        {
            const ptrdiff_t ob2 = static_cast<ptrdiff_t>(nb) * nb;
            Eigen::Tensor<double, 4> G_ba_ex(nb, nb, va, va); G_ba_ex.setZero();
            varray_view<double> t_G({(len_type)nb, (len_type)nb, (len_type)va, (len_type)va}, G_ba_ex.data(),
                                    {1, (stride_type)nb, (stride_type)ob2, (stride_type)(ob2 * va)});
            safe_tblis_mult(-0.5, t_T2_ab, "mkce", t_T2_ab, "miae", 0.0, t_G, "ikac");

            Eigen::Tensor<double, 3> U_b(nb, nb, n_aux); U_b.setZero();
            varray_view<double> t_U({(len_type)nb, (len_type)nb, (len_type)n_aux}, U_b.data(),
                                    {1, (stride_type)nb, (stride_type)ob2});
            safe_tblis_mult( 1.0, t_G, "mkac", t_Bvv_a, "acP", 0.0, t_U, "mkP");
            safe_tblis_mult( 1.0, t_Bov_b, "kxP", t_U, "mkP", 1.0, t_Zb, "xm");
            safe_tblis_mult( 1.0, t_Bov_b, "ixP", t_U, "imP", 1.0, t_Zb, "xm");
            safe_tblis_mult(-1.0, t_G, "ikxc", t_Vooov_bb_aa, "ikmc", 1.0, t_Za, "xm");
            safe_tblis_mult(-1.0, t_G, "ikax", t_Vooov_bb_aa, "ikma", 1.0, t_Za, "xm");
        }
    }

    return {std::move(Z_a), std::move(Z_b)};
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> apply_mo_df_hessian_uhf(
    const Eigen::MatrixXd& kappa_vo_a,
    const Eigen::MatrixXd& kappa_vo_b,
    const Eigen::VectorXd& eps_oa,
    const Eigen::VectorXd& eps_va,
    const Eigen::VectorXd& eps_ob,
    const Eigen::VectorXd& eps_vb,
    const Eigen::MatrixXd& B_oo_a,
    const Eigen::MatrixXd& B_ov_a,
    const Eigen::MatrixXd& B_vv_a,
    const Eigen::MatrixXd& B_oo_b,
    const Eigen::MatrixXd& B_ov_b,
    const Eigen::MatrixXd& B_vv_b)
{
    const int na = static_cast<int>(eps_oa.size());
    const int va = static_cast<int>(eps_va.size());
    const int nb = static_cast<int>(eps_ob.size());
    const int vb = static_cast<int>(eps_vb.size());
    const int n_aux = static_cast<int>(B_ov_a.cols());

    Eigen::MatrixXd Hp_a = Eigen::MatrixXd::Zero(va, na);
    Eigen::MatrixXd Hp_b = Eigen::MatrixXd::Zero(vb, nb);

    for (int i = 0; i < na; ++i)
        for (int a = 0; a < va; ++a)
            Hp_a(a, i) = 2.0 * std::max(std::abs(eps_va(a) - eps_oa(i)), 1e-4) * kappa_vo_a(a, i);

    for (int i = 0; i < nb; ++i)
        for (int a = 0; a < vb; ++a)
            Hp_b(a, i) = 2.0 * std::max(std::abs(eps_vb(a) - eps_ob(i)), 1e-4) * kappa_vo_b(a, i);

    if (n_aux <= 0) return {std::move(Hp_a), std::move(Hp_b)};

    Eigen::VectorXd w_aux = Eigen::VectorXd::Zero(n_aux);
    if (na > 0 && va > 0) {
        Eigen::Map<const Eigen::VectorXd> ka_vec(kappa_vo_a.data(), static_cast<Eigen::Index>(va) * na);
        w_aux.noalias() += 2.0 * (B_ov_a.transpose() * ka_vec);
    }
    if (nb > 0 && vb > 0) {
        Eigen::Map<const Eigen::VectorXd> kb_vec(kappa_vo_b.data(), static_cast<Eigen::Index>(vb) * nb);
        w_aux.noalias() += 2.0 * (B_ov_b.transpose() * kb_vec);
    }

    Eigen::MatrixXd Hk_a = Eigen::MatrixXd::Zero(va, na);
    Eigen::MatrixXd Hk_b = Eigen::MatrixXd::Zero(vb, nb);
    if (na > 0 && va > 0) {
        Eigen::VectorXd ja = B_ov_a * w_aux;
        Hk_a = Eigen::Map<const Eigen::MatrixXd>(ja.data(), va, na);
    }
    if (nb > 0 && vb > 0) {
        Eigen::VectorXd jb = B_ov_b * w_aux;
        Hk_b = Eigen::Map<const Eigen::MatrixXd>(jb.data(), vb, nb);
    }

    Eigen::MatrixXd ka_ov = kappa_vo_a.transpose();
    Eigen::MatrixXd kb_ov = kappa_vo_b.transpose();
    const int max_t = omp_get_max_threads();
    std::vector<Eigen::MatrixXd> Ka_tls(max_t, Eigen::MatrixXd::Zero(va, na));
    std::vector<Eigen::MatrixXd> Kb_tls(max_t, Eigen::MatrixXd::Zero(vb, nb));

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        Eigen::MatrixXd Ta_vo(va, na), Ta2(na, na);
        Eigen::MatrixXd Tb_vo(vb, nb), Tb2(nb, nb);
        #pragma omp for schedule(static) nowait
        for (int P = 0; P < n_aux; ++P) {
            if (na > 0 && va > 0) {
                Eigen::Map<const Eigen::MatrixXd> Boo(B_oo_a.col(P).data(), na, na);
                Eigen::Map<const Eigen::MatrixXd> Bvo(B_ov_a.col(P).data(), va, na);
                Eigen::Map<const Eigen::MatrixXd> Bvv(B_vv_a.col(P).data(), va, va);
                Ta_vo.noalias() = kappa_vo_a * Boo;
                Ka_tls[tid].noalias() += Bvv * Ta_vo;
                Ta2.noalias() = ka_ov * Bvo;
                Ka_tls[tid].noalias() += Bvo * Ta2;
            }
            if (nb > 0 && vb > 0) {
                Eigen::Map<const Eigen::MatrixXd> Boo(B_oo_b.col(P).data(), nb, nb);
                Eigen::Map<const Eigen::MatrixXd> Bvo(B_ov_b.col(P).data(), vb, nb);
                Eigen::Map<const Eigen::MatrixXd> Bvv(B_vv_b.col(P).data(), vb, vb);
                Tb_vo.noalias() = kappa_vo_b * Boo;
                Kb_tls[tid].noalias() += Bvv * Tb_vo;
                Tb2.noalias() = kb_ov * Bvo;
                Kb_tls[tid].noalias() += Bvo * Tb2;
            }
        }
    }

    for (int t = 0; t < max_t; ++t) {
        if (na > 0 && va > 0) Hk_a.noalias() -= Ka_tls[t];
        if (nb > 0 && vb > 0) Hk_b.noalias() -= Kb_tls[t];
    }

    Hp_a.noalias() += 2.0 * Hk_a;
    Hp_b.noalias() += 2.0 * Hk_b;
    return {std::move(Hp_a), std::move(Hp_b)};
}


std::pair<Eigen::MatrixXd, Eigen::MatrixXd> eval_mo_gcorr_uhf_df(
    const Eigen::MatrixXd& B_oo_a,
    const Eigen::MatrixXd& B_ov_a,
    const Eigen::MatrixXd& B_vv_a,
    const Eigen::MatrixXd& B_oo_b,
    const Eigen::MatrixXd& B_ov_b,
    const Eigen::MatrixXd& B_vv_b,
    const Eigen::MatrixXd& g_oo_a,
    const Eigen::MatrixXd& g_vv_a,
    const Eigen::MatrixXd& g_oo_b,
    const Eigen::MatrixXd& g_vv_b,
    int na, int va, int nb, int vb)
{
    const bool has_a = (na > 0 && va > 0);
    const bool has_b = (nb > 0 && vb > 0);
    const int n_aux = has_a ? static_cast<int>(B_ov_a.cols()) : (has_b ? static_cast<int>(B_ov_b.cols()) : 0);
    Eigen::MatrixXd Ga_vo = Eigen::MatrixXd::Zero(va, na);
    Eigen::MatrixXd Gb_vo = Eigen::MatrixXd::Zero(vb, nb);
    if (n_aux <= 0 || (!has_a && !has_b)) return {std::move(Ga_vo), std::move(Gb_vo)};

    // 1. Coulomb terkopel alpha + beta di ruang MO
    Eigen::VectorXd c_aux = Eigen::VectorXd::Zero(n_aux);
    if (has_a) {
        Eigen::Map<const Eigen::VectorXd> goa(g_oo_a.data(), static_cast<Eigen::Index>(na) * na);
        Eigen::Map<const Eigen::VectorXd> gva(g_vv_a.data(), static_cast<Eigen::Index>(va) * va);
        c_aux.noalias() += B_oo_a.transpose() * goa + B_vv_a.transpose() * gva;
    }
    if (has_b) {
        Eigen::Map<const Eigen::VectorXd> gob(g_oo_b.data(), static_cast<Eigen::Index>(nb) * nb);
        Eigen::Map<const Eigen::VectorXd> gvb(g_vv_b.data(), static_cast<Eigen::Index>(vb) * vb);
        c_aux.noalias() += B_oo_b.transpose() * gob + B_vv_b.transpose() * gvb;
    }
    if (has_a) {
        Eigen::VectorXd ja = B_ov_a * c_aux;
        Ga_vo = Eigen::Map<const Eigen::MatrixXd>(ja.data(), va, na);
    }
    if (has_b) {
        Eigen::VectorXd jb = B_ov_b * c_aux;
        Gb_vo = Eigen::Map<const Eigen::MatrixXd>(jb.data(), vb, nb);
    }

    // 2. Exchange terpisah spin di ruang MO (zero AO unpacking)
    const int max_t = omp_get_max_threads();
    std::vector<Eigen::MatrixXd> Ka_tls(has_a ? max_t : 0, Eigen::MatrixXd::Zero(va, na));
    std::vector<Eigen::MatrixXd> Kb_tls(has_b ? max_t : 0, Eigen::MatrixXd::Zero(vb, nb));

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        Eigen::MatrixXd Ta_oo, Ta_vo, Tb_oo, Tb_vo;
        if (has_a) { Ta_oo.resize(na, na); Ta_vo.resize(va, na); }
        if (has_b) { Tb_oo.resize(nb, nb); Tb_vo.resize(vb, nb); }
        #pragma omp for schedule(static) nowait
        for (int P = 0; P < n_aux; ++P) {
            if (has_a) {
                Eigen::Map<const Eigen::MatrixXd> Boo(B_oo_a.col(P).data(), na, na);
                Eigen::Map<const Eigen::MatrixXd> Bvo(B_ov_a.col(P).data(), va, na);
                Eigen::Map<const Eigen::MatrixXd> Bvv(B_vv_a.col(P).data(), va, va);
                Ta_oo.noalias() = g_oo_a * Boo;
                Ka_tls[tid].noalias() += Bvo * Ta_oo;
                Ta_vo.noalias() = g_vv_a * Bvo;
                Ka_tls[tid].noalias() += Bvv * Ta_vo;
            }
            if (has_b) {
                Eigen::Map<const Eigen::MatrixXd> Boo(B_oo_b.col(P).data(), nb, nb);
                Eigen::Map<const Eigen::MatrixXd> Bvo(B_ov_b.col(P).data(), vb, nb);
                Eigen::Map<const Eigen::MatrixXd> Bvv(B_vv_b.col(P).data(), vb, vb);
                Tb_oo.noalias() = g_oo_b * Boo;
                Kb_tls[tid].noalias() += Bvo * Tb_oo;
                Tb_vo.noalias() = g_vv_b * Bvo;
                Kb_tls[tid].noalias() += Bvv * Tb_vo;
            }
        }
    }
    for (int t = 0; t < max_t; ++t) {
        if (has_a) Ga_vo.noalias() -= Ka_tls[t];
        if (has_b) Gb_vo.noalias() -= Kb_tls[t];
    }
    return {std::move(Ga_vo), std::move(Gb_vo)};
}

} // namespace kernels
} // namespace mshqc
