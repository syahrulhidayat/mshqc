// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0
#include "mshqc/integrals/eri_transformer.h"
#include "mshqc/kernels/tblis_safe.h"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdexcept>
#include <vector>

namespace mshqc {
namespace integrals {

using kernels::safe_tblis_mult;
using tblis::len_type;
using tblis::stride_type;
using tblis::varray_view;

namespace {

inline varray_view<double> make_mat_view(const Eigen::MatrixXd& M) {
    return varray_view<double>(
        {(len_type)M.rows(), (len_type)M.cols()},
        const_cast<double*>(M.data()),
        {1, (stride_type)M.rows()});
}

inline varray_view<double> make_eri_view(const ERITensor& eri_ao, int nbf) {
    const stride_type n = static_cast<stride_type>(nbf);
    return varray_view<double>(
        {(len_type)nbf, (len_type)nbf, (len_type)nbf, (len_type)nbf},
        const_cast<double*>(eri_ao.data()),
        {1, n, n * n, n * n * n});
}

void build_jk_4d_fused_rhf_impl(
    const ERITensor& eri_ao,
    const Eigen::MatrixXd& P_a,
    int nbf,
    Eigen::MatrixXd& J_out,
    Eigen::MatrixXd& K_out)
{
    const Eigen::Index n = static_cast<Eigen::Index>(nbf);
    const Eigen::Index n2 = n * n;
    const int n_tri = nbf * (nbf + 1) / 2;
    std::vector<int> pair_s(n_tri), pair_n(n_tri);
    int idx = 0;
    for (int s = 0; s < nbf; ++s) {
        for (int col_n = s; col_n < nbf; ++col_n) {
            pair_s[idx] = s;
            pair_n[idx] = col_n;
            ++idx;
        }
    }

    const int max_t = omp_get_max_threads();
    std::vector<Eigen::MatrixXd> J_tls(max_t, Eigen::MatrixXd::Zero(nbf, nbf));
    std::vector<Eigen::MatrixXd> K_tls(max_t, Eigen::MatrixXd::Zero(nbf, nbf));
    const double* eri_ptr = eri_ao.data();

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        Eigen::MatrixXd& J_loc = J_tls[tid];
        Eigen::MatrixXd& K_loc = K_tls[tid];
        #pragma omp for schedule(static) nowait
        for (int k = 0; k < n_tri; ++k) {
            const int s = pair_s[k];
            const int col_n = pair_n[k];
            const Eigen::Index off = (static_cast<Eigen::Index>(s) * n + col_n) * n2;
            Eigen::Map<const Eigen::MatrixXd> V_ns(eri_ptr + off, nbf, nbf);
            const double p_ns = P_a(col_n, s);
            if (col_n == s) {
                if (std::abs(p_ns) > 1.0e-16) {
                    J_loc.noalias() += (2.0 * p_ns) * V_ns;
                }
                K_loc.col(s).noalias() += V_ns * P_a.col(s);
            } else {
                if (std::abs(p_ns) > 1.0e-16) {
                    J_loc.noalias() += (4.0 * p_ns) * V_ns;
                }
                K_loc.col(col_n).noalias() += V_ns * P_a.col(s);
                K_loc.col(s).noalias() += V_ns * P_a.col(col_n);
            }
        }
    }

    J_out = Eigen::MatrixXd::Zero(nbf, nbf);
    K_out = Eigen::MatrixXd::Zero(nbf, nbf);
    for (int t = 0; t < max_t; ++t) {
        J_out.noalias() += J_tls[t];
        K_out.noalias() += K_tls[t];
    }
    J_out = 0.5 * (J_out + J_out.transpose());
    K_out = 0.5 * (K_out + K_out.transpose());
}

void build_jk_4d_fused_uhf_impl(
    const ERITensor& eri_ao,
    const Eigen::MatrixXd& P_a,
    const Eigen::MatrixXd& P_b,
    int nbf,
    Eigen::MatrixXd& J_out,
    Eigen::MatrixXd& Ka_out,
    Eigen::MatrixXd& Kb_out)
{
    const Eigen::Index n = static_cast<Eigen::Index>(nbf);
    const Eigen::Index n2 = n * n;
    const int n_tri = nbf * (nbf + 1) / 2;
    std::vector<int> pair_s(n_tri), pair_n(n_tri);
    int idx = 0;
    for (int s = 0; s < nbf; ++s) {
        for (int col_n = s; col_n < nbf; ++col_n) {
            pair_s[idx] = s;
            pair_n[idx] = col_n;
            ++idx;
        }
    }

    const Eigen::MatrixXd P_tot = P_a + P_b;
    const int max_t = omp_get_max_threads();
    std::vector<Eigen::MatrixXd> J_tls(max_t, Eigen::MatrixXd::Zero(nbf, nbf));
    std::vector<Eigen::MatrixXd> Ka_tls(max_t, Eigen::MatrixXd::Zero(nbf, nbf));
    std::vector<Eigen::MatrixXd> Kb_tls(max_t, Eigen::MatrixXd::Zero(nbf, nbf));
    const double* eri_ptr = eri_ao.data();

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        Eigen::MatrixXd& J_loc = J_tls[tid];
        Eigen::MatrixXd& Ka_loc = Ka_tls[tid];
        Eigen::MatrixXd& Kb_loc = Kb_tls[tid];
        #pragma omp for schedule(static) nowait
        for (int k = 0; k < n_tri; ++k) {
            const int s = pair_s[k];
            const int col_n = pair_n[k];
            const Eigen::Index off = (static_cast<Eigen::Index>(s) * n + col_n) * n2;
            Eigen::Map<const Eigen::MatrixXd> V_ns(eri_ptr + off, nbf, nbf);
            const double pt = P_tot(col_n, s);
            if (col_n == s) {
                if (std::abs(pt) > 1.0e-16) {
                    J_loc.noalias() += pt * V_ns;
                }
                Ka_loc.col(s).noalias() += V_ns * P_a.col(s);
                Kb_loc.col(s).noalias() += V_ns * P_b.col(s);
            } else {
                if (std::abs(pt) > 1.0e-16) {
                    J_loc.noalias() += (2.0 * pt) * V_ns;
                }
                Ka_loc.col(col_n).noalias() += V_ns * P_a.col(s);
                Ka_loc.col(s).noalias() += V_ns * P_a.col(col_n);
                Kb_loc.col(col_n).noalias() += V_ns * P_b.col(s);
                Kb_loc.col(s).noalias() += V_ns * P_b.col(col_n);
            }
        }
    }

    J_out = Eigen::MatrixXd::Zero(nbf, nbf);
    Ka_out = Eigen::MatrixXd::Zero(nbf, nbf);
    Kb_out = Eigen::MatrixXd::Zero(nbf, nbf);
    for (int t = 0; t < max_t; ++t) {
        J_out.noalias() += J_tls[t];
        Ka_out.noalias() += Ka_tls[t];
        Kb_out.noalias() += Kb_tls[t];
    }
    J_out = 0.5 * (J_out + J_out.transpose());
    Ka_out = 0.5 * (Ka_out + Ka_out.transpose());
    Kb_out = 0.5 * (Kb_out + Kb_out.transpose());
}

Eigen::Tensor<double, 4> smart_transform_kernel(
    const ERITensor& eri_ao,
    const Eigen::MatrixXd& C1,
    const Eigen::MatrixXd& C2,
    const Eigen::MatrixXd& C3,
    const Eigen::MatrixXd& C4,
    int nbf, int n1, int n2, int n3, int n4)
{
    auto t_eri = make_eri_view(eri_ao, nbf);
    auto t_C1 = make_mat_view(C1);
    auto t_C2 = make_mat_view(C2);
    auto t_C3 = make_mat_view(C3);
    auto t_C4 = make_mat_view(C4);

    Eigen::Tensor<double, 4> T1(n1, nbf, nbf, nbf);
    T1.setZero();
    varray_view<double> v_T1(
        {(len_type)n1, (len_type)nbf, (len_type)nbf, (len_type)nbf},
        T1.data(), {1, (stride_type)n1, (stride_type)(n1 * nbf), (stride_type)(n1 * nbf * nbf)});
    safe_tblis_mult(1.0, t_eri, "abcd", t_C1, "ae", 0.0, v_T1, "ebcd");

    Eigen::Tensor<double, 4> T2(n1, n2, nbf, nbf);
    T2.setZero();
    varray_view<double> v_T2(
        {(len_type)n1, (len_type)n2, (len_type)nbf, (len_type)nbf},
        T2.data(), {1, (stride_type)n1, (stride_type)(n1 * n2), (stride_type)(n1 * n2 * nbf)});
    safe_tblis_mult(1.0, v_T1, "ebcd", t_C2, "bf", 0.0, v_T2, "efcd");

    Eigen::Tensor<double, 4> T3(n1, n2, n3, nbf);
    T3.setZero();
    varray_view<double> v_T3(
        {(len_type)n1, (len_type)n2, (len_type)n3, (len_type)nbf},
        T3.data(), {1, (stride_type)n1, (stride_type)(n1 * n2), (stride_type)(n1 * n2 * n3)});
    safe_tblis_mult(1.0, v_T2, "efcd", t_C3, "cg", 0.0, v_T3, "efgd");

    Eigen::Tensor<double, 4> out(n1, n2, n3, n4);
    out.setZero();
    varray_view<double> v_out(
        {(len_type)n1, (len_type)n2, (len_type)n3, (len_type)n4},
        out.data(), {1, (stride_type)n1, (stride_type)(n1 * n2), (stride_type)(n1 * n2 * n3)});
    safe_tblis_mult(1.0, v_T3, "efgd", t_C4, "dh", 0.0, v_out, "efgh");
    return out;
}

// Optimal O(O N^4) quarter-transformation contracting both occupied indices first: (i a | j b)
Eigen::Tensor<double, 4> optimal_oovv_mixed_kernel(
    const ERITensor& eri_ao,
    const Eigen::MatrixXd& C_o1, const Eigen::MatrixXd& C_v1,
    const Eigen::MatrixXd& C_o2, const Eigen::MatrixXd& C_v2,
    int nbf, int no1, int nv1, int no2, int nv2)
{
    auto t_eri = make_eri_view(eri_ao, nbf);
    auto t_Co1 = make_mat_view(C_o1);
    auto t_Cv1 = make_mat_view(C_v1);
    auto t_Co2 = make_mat_view(C_o2);
    auto t_Cv2 = make_mat_view(C_v2);

    Eigen::Tensor<double, 4> T1(no1, nbf, nbf, nbf);
    T1.setZero();
    varray_view<double> v_T1(
        {(len_type)no1, (len_type)nbf, (len_type)nbf, (len_type)nbf},
        T1.data(), {1, (stride_type)no1, (stride_type)(no1 * nbf), (stride_type)(no1 * nbf * nbf)});
    safe_tblis_mult(1.0, t_eri, "abcd", t_Co1, "ae", 0.0, v_T1, "ebcd");

    Eigen::Tensor<double, 4> T2(no1, nbf, no2, nbf);
    T2.setZero();
    varray_view<double> v_T2(
        {(len_type)no1, (len_type)nbf, (len_type)no2, (len_type)nbf},
        T2.data(), {1, (stride_type)no1, (stride_type)(no1 * nbf), (stride_type)(no1 * nbf * no2)});
    safe_tblis_mult(1.0, v_T1, "ebcd", t_Co2, "cf", 0.0, v_T2, "ebfd");

    Eigen::Tensor<double, 4> T3(no1, nv1, no2, nbf);
    T3.setZero();
    varray_view<double> v_T3(
        {(len_type)no1, (len_type)nv1, (len_type)no2, (len_type)nbf},
        T3.data(), {1, (stride_type)no1, (stride_type)(no1 * nv1), (stride_type)(no1 * nv1 * no2)});
    safe_tblis_mult(1.0, v_T2, "ebfd", t_Cv1, "bg", 0.0, v_T3, "egfd");

    Eigen::Tensor<double, 4> out(no1, nv1, no2, nv2);
    out.setZero();
    varray_view<double> v_out(
        {(len_type)no1, (len_type)nv1, (len_type)no2, (len_type)nv2},
        out.data(), {1, (stride_type)no1, (stride_type)(no1 * nv1), (stride_type)(no1 * nv1 * no2)});
    safe_tblis_mult(1.0, v_T3, "egfd", t_Cv2, "dh", 0.0, v_out, "egfh");
    return out;
}

} // namespace

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> build_jk_4d_density_uhf(
    const ERITensor& eri_ao,
    const Eigen::MatrixXd& P_alpha,
    const Eigen::MatrixXd& P_beta)
{
    const int nbf = static_cast<int>(eri_ao.dimension(0));
    if (P_alpha.rows() != nbf || P_alpha.cols() != nbf ||
        P_beta.rows() != nbf || P_beta.cols() != nbf) {
        throw std::invalid_argument("build_jk_4d_density_uhf: dimensi P tidak cocok dengan eri_ao");
    }
    Eigen::MatrixXd J, Ka, Kb;
    build_jk_4d_fused_uhf_impl(eri_ao, P_alpha, P_beta, nbf, J, Ka, Kb);
    return {std::move(J), std::move(Ka), std::move(Kb)};
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> build_jk_4d_rhf(
    const ERITensor& eri_ao,
    const Eigen::MatrixXd& C_occ)
{
    const int nbf = static_cast<int>(eri_ao.dimension(0));
    if (C_occ.rows() != nbf) {
        throw std::invalid_argument("build_jk_4d_rhf: dimensi C_occ tidak cocok dengan eri_ao");
    }
    Eigen::MatrixXd P_a = (C_occ.cols() > 0)
        ? Eigen::MatrixXd(C_occ * C_occ.transpose())
        : Eigen::MatrixXd::Zero(nbf, nbf);
    Eigen::MatrixXd J, K_mat;
    build_jk_4d_fused_rhf_impl(eri_ao, P_a, nbf, J, K_mat);
    return {std::move(J), std::move(K_mat)};
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> build_jk_4d_uhf(
    const ERITensor& eri_ao,
    const Eigen::MatrixXd& C_occ_a,
    const Eigen::MatrixXd& C_occ_b)
{
    const int nbf = static_cast<int>(eri_ao.dimension(0));
    Eigen::MatrixXd Pa = (C_occ_a.cols() > 0)
        ? Eigen::MatrixXd(C_occ_a * C_occ_a.transpose())
        : Eigen::MatrixXd::Zero(nbf, nbf);
    Eigen::MatrixXd Pb = (C_occ_b.cols() > 0)
        ? Eigen::MatrixXd(C_occ_b * C_occ_b.transpose())
        : Eigen::MatrixXd::Zero(nbf, nbf);
    return build_jk_4d_density_uhf(eri_ao, Pa, Pb);
}

Eigen::MatrixXd transform_eri_4d(
    const ERITensor& eri_ao,
    const Eigen::MatrixXd& C1,
    const Eigen::MatrixXd& C2,
    const Eigen::MatrixXd& C3,
    const Eigen::MatrixXd& C4)
{
    const int nbf = static_cast<int>(eri_ao.dimension(0));
    const int n1 = static_cast<int>(C1.cols());
    const int n2 = static_cast<int>(C2.cols());
    const int n3 = static_cast<int>(C3.cols());
    const int n4 = static_cast<int>(C4.cols());
    if (C1.rows() != nbf || C2.rows() != nbf || C3.rows() != nbf || C4.rows() != nbf) {
        throw std::invalid_argument("transform_eri_4d: baris C tidak cocok dengan nbf");
    }
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(n1) * n2, static_cast<Eigen::Index>(n3) * n4);
    if (n1 == 0 || n2 == 0 || n3 == 0 || n4 == 0) return out;

    Eigen::Tensor<double, 4> mo = (n1 <= n2 && n3 <= n4 && n1 < nbf && n3 < nbf)
        ? optimal_oovv_mixed_kernel(eri_ao, C1, C2, C3, C4, nbf, n1, n2, n3, n4)
        : smart_transform_kernel(eri_ao, C1, C2, C3, C4, nbf, n1, n2, n3, n4);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            const Eigen::Index row = static_cast<Eigen::Index>(i) * n2 + j;
            for (int k = 0; k < n3; ++k) {
                const Eigen::Index col_base = static_cast<Eigen::Index>(k) * n4;
                for (int l = 0; l < n4; ++l) {
                    out(row, col_base + l) = mo(i, j, k, l);
                }
            }
        }
    }
    return out;
}

std::pair<double, double> eval_rmp2_exact(
    const ERITensor& eri_ao,
    const Eigen::MatrixXd& C_occ,
    const Eigen::MatrixXd& C_vir,
    const Eigen::VectorXd& eps_occ,
    const Eigen::VectorXd& eps_vir)
{
    const int nbf = static_cast<int>(eri_ao.dimension(0));
    const int no = static_cast<int>(C_occ.cols());
    const int nv = static_cast<int>(C_vir.cols());
    if (no == 0 || nv == 0) return {0.0, 0.0};

    Eigen::Tensor<double, 4> g = optimal_oovv_mixed_kernel(
        eri_ao, C_occ, C_vir, C_occ, C_vir, nbf, no, nv, no, nv);

    double e_os = 0.0;
    double e_ss = 0.0;
    #pragma omp parallel for collapse(2) reduction(+:e_os, e_ss) schedule(static)
    for (int i = 0; i < no; ++i) {
        for (int j = 0; j < no; ++j) {
            const double eij = eps_occ(i) + eps_occ(j);
            for (int a = 0; a < nv; ++a) {
                const double eija = eij - eps_vir(a);
                for (int b = 0; b < nv; ++b) {
                    const double denom = eija - eps_vir(b);
                    if (std::abs(denom) > 1.0e-12) {
                        const double v_ab = g(i, a, j, b);
                        const double v_ba = g(i, b, j, a);
                        const double t_ab = v_ab / denom;
                        e_os += t_ab * v_ab;
                        e_ss += t_ab * (v_ab - v_ba);
                    }
                }
            }
        }
    }
    return {e_os, e_ss};
}

std::tuple<double, double, double> eval_ump2_exact(
    const ERITensor& eri_ao,
    const Eigen::MatrixXd& Ca_occ,
    const Eigen::MatrixXd& Ca_vir,
    const Eigen::MatrixXd& Cb_occ,
    const Eigen::MatrixXd& Cb_vir,
    const Eigen::VectorXd& eps_oa,
    const Eigen::VectorXd& eps_va,
    const Eigen::VectorXd& eps_ob,
    const Eigen::VectorXd& eps_vb)
{
    const int nbf = static_cast<int>(eri_ao.dimension(0));
    const int na = static_cast<int>(Ca_occ.cols());
    const int va = static_cast<int>(Ca_vir.cols());
    const int nb = static_cast<int>(Cb_occ.cols());
    const int vb = static_cast<int>(Cb_vir.cols());

    double e_aa = 0.0, e_bb = 0.0, e_ab = 0.0;

    if (na > 1 && va > 0) {
        Eigen::Tensor<double, 4> g_aa = optimal_oovv_mixed_kernel(
            eri_ao, Ca_occ, Ca_vir, Ca_occ, Ca_vir, nbf, na, va, na, va);
        #pragma omp parallel for collapse(2) reduction(+:e_aa) schedule(static)
        for (int i = 0; i < na; ++i) {
            for (int j = 0; j < na; ++j) {
                const double eij = eps_oa(i) + eps_oa(j);
                for (int a = 0; a < va; ++a) {
                    const double eija = eij - eps_va(a);
                    for (int b = 0; b < va; ++b) {
                        const double denom = eija - eps_va(b);
                        if (std::abs(denom) > 1.0e-12) {
                            const double diff = g_aa(i, a, j, b) - g_aa(i, b, j, a);
                            e_aa += 0.25 * (diff * diff) / denom;
                        }
                    }
                }
            }
        }
    }

    if (nb > 1 && vb > 0) {
        Eigen::Tensor<double, 4> g_bb = optimal_oovv_mixed_kernel(
            eri_ao, Cb_occ, Cb_vir, Cb_occ, Cb_vir, nbf, nb, vb, nb, vb);
        #pragma omp parallel for collapse(2) reduction(+:e_bb) schedule(static)
        for (int i = 0; i < nb; ++i) {
            for (int j = 0; j < nb; ++j) {
                const double eij = eps_ob(i) + eps_ob(j);
                for (int a = 0; a < vb; ++a) {
                    const double eija = eij - eps_vb(a);
                    for (int b = 0; b < vb; ++b) {
                        const double denom = eija - eps_vb(b);
                        if (std::abs(denom) > 1.0e-12) {
                            const double diff = g_bb(i, a, j, b) - g_bb(i, b, j, a);
                            e_bb += 0.25 * (diff * diff) / denom;
                        }
                    }
                }
            }
        }
    }

    if (na > 0 && va > 0 && nb > 0 && vb > 0) {
        Eigen::Tensor<double, 4> g_ab = optimal_oovv_mixed_kernel(
            eri_ao, Ca_occ, Ca_vir, Cb_occ, Cb_vir, nbf, na, va, nb, vb);
        #pragma omp parallel for collapse(2) reduction(+:e_ab) schedule(static)
        for (int i = 0; i < na; ++i) {
            for (int j = 0; j < nb; ++j) {
                const double eij = eps_oa(i) + eps_ob(j);
                for (int a = 0; a < va; ++a) {
                    const double eija = eij - eps_va(a);
                    for (int b = 0; b < vb; ++b) {
                        const double denom = eija - eps_vb(b);
                        if (std::abs(denom) > 1.0e-12) {
                            const double v = g_ab(i, a, j, b);
                            e_ab += (v * v) / denom;
                        }
                    }
                }
            }
        }
    }

    return {e_aa, e_bb, e_ab};
}

Eigen::MatrixXd factorize_eri_4d_exact(const ERITensor& eri_ao, double threshold) {
    const int nbf = static_cast<int>(eri_ao.dimension(0));
    if (nbf <= 0) return Eigen::MatrixXd(0, 0);
    const int n_tri = nbf * (nbf + 1) / 2;
    const double tau = std::max(threshold, 1.0e-14);

    std::vector<int> tri_p(n_tri), tri_q(n_tri);
    for (int p = 0; p < nbf; ++p) {
        const int base = p * (p + 1) / 2;
        for (int q = 0; q <= p; ++q) {
            tri_p[base + q] = p;
            tri_q[base + q] = q;
        }
    }

    Eigen::VectorXd D(n_tri);
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < n_tri; ++idx) {
        const int p = tri_p[idx];
        const int q = tri_q[idx];
        D(idx) = std::max(0.0, eri_ao(p, q, p, q));
    }

    int est_rank = std::min(n_tri, std::max(64, nbf * 8));
    Eigen::MatrixXd L_tri(n_tri, est_rank);

    constexpr int kMaxBlock = 32;
    constexpr double kSpanFactor = 0.05;
    int iter = 0;
    std::vector<int> cand_idx;
    cand_idx.reserve(n_tri);

    while (iter < n_tri) {
        int max_piv = -1;
        const double D_max = D.maxCoeff(&max_piv);
        if (D_max < tau || max_piv < 0) break;

        const double span_cut = std::max(tau, kSpanFactor * D_max);
        cand_idx.clear();
        for (int idx = 0; idx < n_tri; ++idx) {
            if (D(idx) >= span_cut) cand_idx.push_back(idx);
        }
        const int q_cnt = std::min(kMaxBlock, static_cast<int>(cand_idx.size()));
        if (q_cnt <= 0) break;

        std::partial_sort(cand_idx.begin(), cand_idx.begin() + q_cnt, cand_idx.end(),
                          [&D](int a, int b) { return D(a) > D(b); });
        cand_idx.resize(q_cnt);

        if (iter + q_cnt > L_tri.cols()) {
            L_tri.conservativeResize(
                Eigen::NoChange, std::min(n_tri, std::max(iter + q_cnt, static_cast<int>(L_tri.cols()) * 2)));
        }

        Eigen::MatrixXd V_panel(n_tri, q_cnt);
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < q_cnt; ++c) {
            const int piv = cand_idx[c];
            const int pp = tri_p[piv];
            const int qq = tri_q[piv];
            double* dst = V_panel.col(c).data();
            for (int r = 0; r < n_tri; ++r) {
                dst[r] = eri_ao(tri_p[r], tri_q[r], pp, qq);
            }
        }

        if (iter > 0) {
            Eigen::MatrixXd L_piv_T(iter, q_cnt);
            for (int c = 0; c < q_cnt; ++c) {
                L_piv_T.col(c) = L_tri.row(cand_idx[c]).head(iter).transpose();
            }
            Eigen::Map<const Eigen::MatrixXd> L_prev(L_tri.data(), n_tri, iter);
            constexpr int kRowTile = 256;
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < n_tri; r += kRowTile) {
                const int rlen = std::min(kRowTile, n_tri - r);
                V_panel.middleRows(r, rlen).noalias() -= L_prev.middleRows(r, rlen) * L_piv_T;
            }
        }

        std::vector<char> used(q_cnt, 0);
        int accepted = 0;
        for (int step = 0; step < q_cnt; ++step) {
            int best_c = -1;
            double best_d = -1.0;
            for (int c = 0; c < q_cnt; ++c) {
                if (used[c]) continue;
                const double dv = V_panel(cand_idx[c], c);
                if (dv > best_d) {
                    best_d = dv;
                    best_c = c;
                }
            }
            if (best_c < 0) break;
            if (best_d < tau) {
                if (accepted == 0) D(cand_idx[best_c]) = 0.0;
                break;
            }
            if (accepted > 0 && best_d < span_cut) break;

            used[best_c] = 1;
            const int piv = cand_idx[best_c];
            const double inv_sqrt = 1.0 / std::sqrt(best_d);
            L_tri.col(iter).noalias() = V_panel.col(best_c) * inv_sqrt;
            D.array() -= L_tri.col(iter).array().square();
            D(piv) = 0.0;
            D = D.cwiseMax(0.0);

            #pragma omp parallel for schedule(static)
            for (int c = 0; c < q_cnt; ++c) {
                if (used[c]) continue;
                const double proj = L_tri(cand_idx[c], iter);
                V_panel.col(c).noalias() -= proj * L_tri.col(iter);
            }
            ++iter;
            ++accepted;
        }
    }

    L_tri.conservativeResize(Eigen::NoChange, iter);
    return L_tri;
}

} // namespace integrals
} // namespace mshqc
