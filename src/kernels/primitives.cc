// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0
// [INJECTED_THIN_CORE_STAGE5_PRIMITIVES_CC]

#include "mshqc/kernels/primitives.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdexcept>
#include <vector>

#include "mshqc/integrals/df_eri.h"
#include "mshqc/integrals/df_metric_utils.h"
#include "mshqc/integrals/packed_ao_utils.h"
#include "mshqc/utils/blas_thread_guard.h"

namespace mshqc {
namespace kernels {

namespace {

inline const double* unpack_ao_chunk(
    const Eigen::MatrixXd& L_mat,
    int nbf,
    bool packed,
    int P_start,
    int P_size,
    Eigen::MatrixXd& L_sq_buf)
{
    if (!packed) {
        return L_mat.col(P_start).data();
    }
    for (int p = 0; p < P_size; ++p) {
        const double* s = L_mat.col(P_start + p).data();
        double* d = L_sq_buf.col(p).data();
        for (int i = 0; i < nbf; ++i) {
            const int base = i * (i + 1) / 2;
            for (int j = 0; j < i; ++j) {
                const double v = s[base + j];
                d[i + j * nbf] = v;
                d[j + i * nbf] = v;
            }
            d[i + i * nbf] = s[base + i];
        }
    }
    return L_sq_buf.data();
}

} // namespace

Eigen::MatrixXd compute_df_ao_factor(
    const Molecule& mol,
    const BasisSet& primary_basis,
    const BasisSet& aux_basis,
    double cutoff,
    bool save_hdf5,
    bool verbose)
{
    BasisSet combined = primary_basis;
    combined.append(aux_basis);
    auto df_ints = std::make_shared<IntegralEngine>(mol, combined);
    integrals::DensityFittingERI df_eng(primary_basis, aux_basis, df_ints, cutoff);
    df_eng.set_verbose(verbose);
    df_eng.compute(save_hdf5);
    return std::move(df_eng.get_B_mat_mutable());
}

SafeDFEngine::SafeDFEngine(
    const Molecule& mol,
    const BasisSet& primary_basis,
    const BasisSet& aux_basis,
    double cutoff)
    : mol_(mol),
      primary_(primary_basis),
      aux_(aux_basis),
      combined_(primary_basis),
      cutoff_(cutoff)
{
    combined_.append(aux_);
    df_ints_ = std::make_shared<IntegralEngine>(mol_, combined_);
    df_eng_ = std::make_unique<integrals::DensityFittingERI>(primary_, aux_, df_ints_, cutoff_);
}

SafeDFEngine::~SafeDFEngine() = default;

void SafeDFEngine::compute(bool save_hdf5) { df_eng_->compute(save_hdf5); }
const Eigen::MatrixXd& SafeDFEngine::get_B_mat() const { return df_eng_->get_B_mat(); }
void SafeDFEngine::set_verbose(bool v) { df_eng_->set_verbose(v); }
void SafeDFEngine::set_hdf5_path(const std::string& path) { df_eng_->set_hdf5_path(path); }
const std::string& SafeDFEngine::get_hdf5_path() const { return df_eng_->get_hdf5_path(); }

Eigen::MatrixXd build_coulomb_3c(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& P_tot,
    int nbf)
{
    if (!integrals::has_valid_ao_factor(L_mat, nbf)) {
        throw std::invalid_argument("build_coulomb_3c: dimensi L_mat tidak cocok dengan nbf");
    }
    return integrals::compute_coulomb_from_ao_factor(L_mat, P_tot, nbf);
}

Eigen::MatrixXd build_exchange_3c_occ(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& C_occ,
    int nbf)
{
    if (!integrals::has_valid_ao_factor(L_mat, nbf)) {
        throw std::invalid_argument("build_exchange_3c_occ: dimensi L_mat tidak cocok dengan nbf");
    }
    const int n_aux = static_cast<int>(L_mat.cols());
    const int n_occ = static_cast<int>(C_occ.cols());
    Eigen::MatrixXd K_tot = Eigen::MatrixXd::Zero(nbf, nbf);
    if (n_occ == 0 || n_aux == 0) return K_tot;

    const bool packed = integrals::is_packed_ao_matrix(L_mat, nbf);
    const int batch_p = std::clamp(
        static_cast<int>((8ULL * 1024ULL * 1024ULL) /
            std::max<size_t>(1, static_cast<size_t>(nbf) * static_cast<size_t>(n_occ) * sizeof(double))),
        1, 32);

    const int max_threads = omp_get_max_threads();
    std::vector<Eigen::MatrixXd> K_tls(max_threads, Eigen::MatrixXd::Zero(nbf, nbf));

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        Eigen::MatrixXd& K_loc = K_tls[tid];
        Eigen::MatrixXd L_P_buf;
        if (packed) L_P_buf.resize(nbf, nbf);
        Eigen::MatrixXd T_batch(nbf, static_cast<Eigen::Index>(n_occ) * batch_p);
        int cnt = 0;

        #pragma omp for schedule(static) nowait
        for (int P = 0; P < n_aux; ++P) {
            const double* col_p = L_mat.col(P).data();
            if (packed) integrals::unpack_ao_column(col_p, nbf, true, L_P_buf);
            Eigen::Map<const Eigen::MatrixXd> L_P(packed ? L_P_buf.data() : col_p, nbf, nbf);
            T_batch.middleCols(static_cast<Eigen::Index>(cnt) * n_occ, n_occ).noalias() = L_P * C_occ;
            ++cnt;
            if (cnt == batch_p) {
                K_loc.noalias() += T_batch * T_batch.transpose();
                cnt = 0;
            }
        }
        if (cnt > 0) {
            auto T_rem = T_batch.leftCols(static_cast<Eigen::Index>(cnt) * n_occ);
            K_loc.noalias() += T_rem * T_rem.transpose();
        }
    }

    for (int t = 0; t < max_threads; ++t) {
        K_tot.noalias() += K_tls[t];
    }
    return 0.5 * (K_tot + K_tot.transpose());
}

Eigen::MatrixXd build_exchange_3c_diag(
    const Eigen::MatrixXd& L_mat,
    const Eigen::VectorXd& d_diag,
    int nbf)
{
    if (!integrals::has_valid_ao_factor(L_mat, nbf)) {
        throw std::invalid_argument("build_exchange_3c_diag: dimensi L_mat tidak cocok dengan nbf");
    }
    if (d_diag.size() != nbf) {
        throw std::invalid_argument("build_exchange_3c_diag: panjang d_diag harus sama dengan nbf");
    }

    const int n_aux = static_cast<int>(L_mat.cols());
    Eigen::MatrixXd K_tot = Eigen::MatrixXd::Zero(nbf, nbf);
    if (n_aux == 0) return K_tot;

    std::vector<int> nz_idx;
    std::vector<double> nz_sqrt;
    nz_idx.reserve(nbf);
    nz_sqrt.reserve(nbf);
    for (int i = 0; i < nbf; ++i) {
        if (d_diag(i) > 1.0e-12) {
            nz_idx.push_back(i);
            nz_sqrt.push_back(std::sqrt(d_diag(i)));
        }
    }
    const int r = static_cast<int>(nz_idx.size());
    if (r == 0) return K_tot;

    const bool packed = integrals::is_packed_ao_matrix(L_mat, nbf);
    const int batch_p = std::clamp(
        static_cast<int>((8ULL * 1024ULL * 1024ULL) /
            std::max<size_t>(1, static_cast<size_t>(nbf) * static_cast<size_t>(r) * sizeof(double))),
        1, 32);

    const int max_threads = omp_get_max_threads();
    std::vector<Eigen::MatrixXd> K_tls(max_threads, Eigen::MatrixXd::Zero(nbf, nbf));

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        Eigen::MatrixXd& K_loc = K_tls[tid];
        Eigen::MatrixXd L_P_buf;
        if (packed) L_P_buf.resize(nbf, nbf);
        Eigen::MatrixXd T_batch(nbf, static_cast<Eigen::Index>(r) * batch_p);
        int cnt = 0;

        #pragma omp for schedule(static) nowait
        for (int P = 0; P < n_aux; ++P) {
            const double* col_p = L_mat.col(P).data();
            if (packed) integrals::unpack_ao_column(col_p, nbf, true, L_P_buf);
            Eigen::Map<const Eigen::MatrixXd> L_P(packed ? L_P_buf.data() : col_p, nbf, nbf);

            const Eigen::Index base_col = static_cast<Eigen::Index>(cnt) * r;
            for (int k = 0; k < r; ++k) {
                T_batch.col(base_col + k).noalias() = nz_sqrt[k] * L_P.col(nz_idx[k]);
            }
            ++cnt;
            if (cnt == batch_p) {
                K_loc.noalias() += T_batch * T_batch.transpose();
                cnt = 0;
            }
        }
        if (cnt > 0) {
            auto T_rem = T_batch.leftCols(static_cast<Eigen::Index>(cnt) * r);
            K_loc.noalias() += T_rem * T_rem.transpose();
        }
    }

    for (int t = 0; t < max_threads; ++t) {
        K_tot.noalias() += K_tls[t];
    }
    return 0.5 * (K_tot + K_tot.transpose());
}

Eigen::MatrixXd build_exchange_3c_sym(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& D_sym,
    int nbf)
{
    if (!integrals::has_valid_ao_factor(L_mat, nbf)) {
        throw std::invalid_argument("build_exchange_3c_sym: dimensi L_mat tidak cocok dengan nbf");
    }
    const int n_aux = static_cast<int>(L_mat.cols());
    Eigen::MatrixXd K_tot = Eigen::MatrixXd::Zero(nbf, nbf);
    if (n_aux == 0) return K_tot;

    const bool packed = integrals::is_packed_ao_matrix(L_mat, nbf);
    #pragma omp parallel
    {
        Eigen::MatrixXd K_loc = Eigen::MatrixXd::Zero(nbf, nbf);
        Eigen::MatrixXd L_P(nbf, nbf);
        Eigen::MatrixXd tmp(nbf, nbf);

        #pragma omp for schedule(static)
        for (int P = 0; P < n_aux; ++P) {
            integrals::unpack_ao_column(L_mat.col(P).data(), nbf, packed, L_P);
            tmp.noalias() = L_P * D_sym;
            K_loc.noalias() += tmp * L_P;
        }

        #pragma omp critical
        {
            K_tot.noalias() += K_loc;
        }
    }
    return 0.5 * (K_tot + K_tot.transpose());
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> build_jk_3c_rhf(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& C_occ,
    int nbf)
{
    Eigen::MatrixXd P_tot = 2.0 * (C_occ * C_occ.transpose());
    Eigen::MatrixXd J = build_coulomb_3c(L_mat, P_tot, nbf);
    Eigen::MatrixXd K_mat = build_exchange_3c_occ(L_mat, C_occ, nbf);
    return {std::move(J), std::move(K_mat)};
}

Eigen::MatrixXd transform_3c_mo(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& C1,
    const Eigen::MatrixXd& C2,
    int nbf)
{
    if (!integrals::has_valid_ao_factor(L_mat, nbf)) {
        throw std::invalid_argument("transform_3c_mo: dimensi L_mat tidak cocok dengan nbf");
    }
    return integrals::transform_3c_ao_to_mo_block(L_mat, C1, C2, nbf);
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> transform_3c_mo_blocks(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& C_occ,
    const Eigen::MatrixXd& C_vir,
    int nbf)
{
    if (!integrals::has_valid_ao_factor(L_mat, nbf)) {
        throw std::invalid_argument("transform_3c_mo_blocks: dimensi L_mat tidak cocok dengan nbf");
    }
    const int no = static_cast<int>(C_occ.cols());
    const int nv = static_cast<int>(C_vir.cols());
    const int n_aux = static_cast<int>(L_mat.cols());

    Eigen::MatrixXd B_oo(static_cast<Eigen::Index>(no) * no, n_aux);
    Eigen::MatrixXd B_ov(static_cast<Eigen::Index>(no) * nv, n_aux);
    Eigen::MatrixXd B_vv(static_cast<Eigen::Index>(nv) * nv, n_aux);
    if (n_aux == 0 || (no == 0 && nv == 0)) {
        return {std::move(B_oo), std::move(B_ov), std::move(B_vv)};
    }

    const bool packed = integrals::is_packed_ao_matrix(L_mat, nbf);
    const size_t bytes_per_p = std::max<size_t>(1, static_cast<size_t>(nbf) * nbf * sizeof(double));
    const int chunk_size = std::clamp(
        static_cast<int>((8ULL * 1024ULL * 1024ULL) / bytes_per_p), 1, 32);

    #pragma omp parallel
    {
        Eigen::MatrixXd L_sq_buf;
        if (packed) L_sq_buf.resize(static_cast<Eigen::Index>(nbf) * nbf, chunk_size);
        Eigen::MatrixXd B_oo_tmp(no, no);
        Eigen::MatrixXd B_ov_tmp(nv, no);
        Eigen::MatrixXd B_vv_tmp(nv, nv);

        #pragma omp for schedule(dynamic, 1) nowait
        for (int P_start = 0; P_start < n_aux; P_start += chunk_size) {
            const int P_size = std::min(chunk_size, n_aux - P_start);
            const double* L_sq_ptr = unpack_ao_chunk(L_mat, nbf, packed, P_start, P_size, L_sq_buf);

            Eigen::Map<const Eigen::MatrixXd> L_res(
                L_sq_ptr, nbf, static_cast<Eigen::Index>(nbf) * P_size);
            Eigen::MatrixXd X_occ, X_vir;
            if (no > 0) X_occ.noalias() = C_occ.transpose() * L_res;
            if (nv > 0) X_vir.noalias() = C_vir.transpose() * L_res;

            for (int p = 0; p < P_size; ++p) {
                if (no > 0) {
                    Eigen::Map<const Eigen::MatrixXd> X_Po(
                        X_occ.data() + static_cast<size_t>(p) * no * nbf, no, nbf);
                    B_oo_tmp.noalias() = X_Po * C_occ;
                    B_oo.col(P_start + p) = Eigen::Map<const Eigen::VectorXd>(
                        B_oo_tmp.data(), static_cast<Eigen::Index>(no) * no);
                }
                if (nv > 0) {
                    Eigen::Map<const Eigen::MatrixXd> X_Pv(
                        X_vir.data() + static_cast<size_t>(p) * nv * nbf, nv, nbf);
                    if (no > 0) {
                        B_ov_tmp.noalias() = X_Pv * C_occ;
                        B_ov.col(P_start + p) = Eigen::Map<const Eigen::VectorXd>(
                            B_ov_tmp.data(), static_cast<Eigen::Index>(nv) * no);
                    }
                    B_vv_tmp.noalias() = X_Pv * C_vir;
                    B_vv.col(P_start + p) = Eigen::Map<const Eigen::VectorXd>(
                        B_vv_tmp.data(), static_cast<Eigen::Index>(nv) * nv);
                }
            }
        }
    }
    return {std::move(B_oo), std::move(B_ov), std::move(B_vv)};
}

std::tuple<
    Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd,
    Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
transform_3c_mo_blocks_uhf(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& Ca_occ,
    const Eigen::MatrixXd& Ca_vir,
    const Eigen::MatrixXd& Cb_occ,
    const Eigen::MatrixXd& Cb_vir,
    int nbf)
{
    if (!integrals::has_valid_ao_factor(L_mat, nbf)) {
        throw std::invalid_argument("transform_3c_mo_blocks_uhf: dimensi L_mat tidak cocok dengan nbf");
    }
    const int na = static_cast<int>(Ca_occ.cols());
    const int va = static_cast<int>(Ca_vir.cols());
    const int nb = static_cast<int>(Cb_occ.cols());
    const int vb = static_cast<int>(Cb_vir.cols());
    const int n_aux = static_cast<int>(L_mat.cols());

    const bool has_b = (nb > 0 && vb > 0);
    Eigen::MatrixXd B_oo_a(static_cast<Eigen::Index>(na) * na, n_aux);
    Eigen::MatrixXd B_ov_a(static_cast<Eigen::Index>(na) * va, n_aux);
    Eigen::MatrixXd B_vv_a(static_cast<Eigen::Index>(va) * va, n_aux);
    Eigen::MatrixXd B_oo_b(has_b ? static_cast<Eigen::Index>(nb) * nb : 0, n_aux);
    Eigen::MatrixXd B_ov_b(has_b ? static_cast<Eigen::Index>(nb) * vb : 0, n_aux);
    Eigen::MatrixXd B_vv_b(has_b ? static_cast<Eigen::Index>(vb) * vb : 0, n_aux);
    if (n_aux == 0) {
        return {std::move(B_oo_a), std::move(B_ov_a), std::move(B_vv_a),
                std::move(B_oo_b), std::move(B_ov_b), std::move(B_vv_b)};
    }

    const bool packed = integrals::is_packed_ao_matrix(L_mat, nbf);
    const size_t bytes_per_p = std::max<size_t>(1, static_cast<size_t>(nbf) * nbf * sizeof(double));
    const int chunk_size = std::clamp(
        static_cast<int>((8ULL * 1024ULL * 1024ULL) / bytes_per_p), 1, 32);

    #pragma omp parallel
    {
        Eigen::MatrixXd L_sq_buf;
        if (packed) L_sq_buf.resize(static_cast<Eigen::Index>(nbf) * nbf, chunk_size);
        Eigen::MatrixXd Boo_a(na, na), Bov_a(va, na), Bvv_a(va, va);
        Eigen::MatrixXd Boo_b, Bov_b, Bvv_b;
        if (has_b) {
            Boo_b.resize(nb, nb);
            Bov_b.resize(vb, nb);
            Bvv_b.resize(vb, vb);
        }

        #pragma omp for schedule(dynamic, 1) nowait
        for (int P_start = 0; P_start < n_aux; P_start += chunk_size) {
            const int P_size = std::min(chunk_size, n_aux - P_start);
            const double* L_sq_ptr = unpack_ao_chunk(L_mat, nbf, packed, P_start, P_size, L_sq_buf);

            Eigen::Map<const Eigen::MatrixXd> L_res(
                L_sq_ptr, nbf, static_cast<Eigen::Index>(nbf) * P_size);
            Eigen::MatrixXd Xa_o, Xa_v;
            if (na > 0) Xa_o.noalias() = Ca_occ.transpose() * L_res;
            if (va > 0) Xa_v.noalias() = Ca_vir.transpose() * L_res;

            for (int p = 0; p < P_size; ++p) {
                if (na > 0) {
                    Eigen::Map<const Eigen::MatrixXd> Xpo(
                        Xa_o.data() + static_cast<size_t>(p) * na * nbf, na, nbf);
                    Boo_a.noalias() = Xpo * Ca_occ;
                    B_oo_a.col(P_start + p) = Eigen::Map<const Eigen::VectorXd>(
                        Boo_a.data(), static_cast<Eigen::Index>(na) * na);
                }
                if (va > 0) {
                    Eigen::Map<const Eigen::MatrixXd> Xpv(
                        Xa_v.data() + static_cast<size_t>(p) * va * nbf, va, nbf);
                    if (na > 0) {
                        Bov_a.noalias() = Xpv * Ca_occ;
                        B_ov_a.col(P_start + p) = Eigen::Map<const Eigen::VectorXd>(
                            Bov_a.data(), static_cast<Eigen::Index>(va) * na);
                    }
                    Bvv_a.noalias() = Xpv * Ca_vir;
                    B_vv_a.col(P_start + p) = Eigen::Map<const Eigen::VectorXd>(
                        Bvv_a.data(), static_cast<Eigen::Index>(va) * va);
                }
            }

            if (has_b) {
                Eigen::MatrixXd Xb_o = Cb_occ.transpose() * L_res;
                Eigen::MatrixXd Xb_v = Cb_vir.transpose() * L_res;
                for (int p = 0; p < P_size; ++p) {
                    Eigen::Map<const Eigen::MatrixXd> Xpo(
                        Xb_o.data() + static_cast<size_t>(p) * nb * nbf, nb, nbf);
                    Eigen::Map<const Eigen::MatrixXd> Xpv(
                        Xb_v.data() + static_cast<size_t>(p) * vb * nbf, vb, nbf);
                    Boo_b.noalias() = Xpo * Cb_occ;
                    Bov_b.noalias() = Xpv * Cb_occ;
                    Bvv_b.noalias() = Xpv * Cb_vir;
                    B_oo_b.col(P_start + p) = Eigen::Map<const Eigen::VectorXd>(
                        Boo_b.data(), static_cast<Eigen::Index>(nb) * nb);
                    B_ov_b.col(P_start + p) = Eigen::Map<const Eigen::VectorXd>(
                        Bov_b.data(), static_cast<Eigen::Index>(vb) * nb);
                    B_vv_b.col(P_start + p) = Eigen::Map<const Eigen::VectorXd>(
                        Bvv_b.data(), static_cast<Eigen::Index>(vb) * vb);
                }
            }
        }
    }
    return {std::move(B_oo_a), std::move(B_ov_a), std::move(B_vv_a),
            std::move(B_oo_b), std::move(B_ov_b), std::move(B_vv_b)};
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd> semicanonicalize_subspaces(
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXd& F_ao,
    int n_occ)
{
    const int n_vir = static_cast<int>(C.cols()) - n_occ;
    if (n_occ < 0 || n_vir < 0) {
        throw std::invalid_argument("semicanonicalize_subspaces: n_occ di luar rentang");
    }

    Eigen::MatrixXd C_out = C;
    Eigen::VectorXd eps(n_occ + n_vir);

    if (n_occ > 0) {
        Eigen::MatrixXd C_o = C_out.leftCols(n_occ);
        Eigen::MatrixXd F_oo = C_o.transpose() * F_ao * C_o;
        F_oo = 0.5 * (F_oo + F_oo.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_o(F_oo);
        eps.head(n_occ) = es_o.eigenvalues();
        C_out.leftCols(n_occ) = C_o * es_o.eigenvectors();
    }

    if (n_vir > 0) {
        Eigen::MatrixXd C_v = C_out.rightCols(n_vir);
        Eigen::MatrixXd F_vv = C_v.transpose() * F_ao * C_v;
        F_vv = 0.5 * (F_vv + F_vv.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_v(F_vv);
        eps.tail(n_vir) = es_v.eigenvalues();
        C_out.rightCols(n_vir) = C_v * es_v.eigenvectors();
    }

    Eigen::MatrixXd F_vo = C_out.rightCols(n_vir).transpose() * F_ao * C_out.leftCols(n_occ);
    return {std::move(C_out), std::move(eps), std::move(F_vo)};
}

std::tuple<double, double, Eigen::MatrixXd> eval_rmp2_df(
    const Eigen::MatrixXd& B_ov,
    const Eigen::VectorXd& eps_occ,
    const Eigen::VectorXd& eps_vir)
{
    const int no = static_cast<int>(eps_occ.size());
    const int nv = static_cast<int>(eps_vir.size());
    if (B_ov.rows() != static_cast<Eigen::Index>(no) * nv) {
        throw std::invalid_argument("eval_rmp2_df: B_ov.rows() harus sama dengan no * nv");
    }

    Eigen::MatrixXd T2(static_cast<Eigen::Index>(nv) * nv, static_cast<Eigen::Index>(no) * no);
    double e_os = 0.0;
    double e_ss = 0.0;

    #pragma omp parallel
    {
        Eigen::MatrixXd I_ij(nv, nv);
        double loc_os = 0.0;
        double loc_ss = 0.0;

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < no; ++i) {
            auto B_i = B_ov.middleRows(static_cast<Eigen::Index>(i) * nv, nv);
            for (int j = i; j < no; ++j) {
                auto B_j = B_ov.middleRows(static_cast<Eigen::Index>(j) * nv, nv);
                I_ij.noalias() = B_i * B_j.transpose();

                const double eij = eps_occ(i) + eps_occ(j);
                const double sym_w = (i == j) ? 1.0 : 2.0;

                Eigen::Map<Eigen::MatrixXd> T_ij(
                    T2.col(static_cast<Eigen::Index>(i) * no + j).data(), nv, nv);

                for (int b = 0; b < nv; ++b) {
                    const double eij_b = eij - eps_vir(b);
                    for (int a = 0; a < nv; ++a) {
                        const double denom = eij_b - eps_vir(a);
                        const double v_ab = I_ij(a, b);
                        const double t_ab = (std::abs(denom) > 1e-12) ? (v_ab / denom) : 0.0;
                        T_ij(a, b) = t_ab;
                        loc_os += sym_w * t_ab * v_ab;
                        loc_ss += sym_w * t_ab * (v_ab - I_ij(b, a));
                    }
                }
                if (i != j) {
                    Eigen::Map<Eigen::MatrixXd> T_ji(
                        T2.col(static_cast<Eigen::Index>(j) * no + i).data(), nv, nv);
                    T_ji = T_ij.transpose();
                }
            }
        }

        #pragma omp atomic
        e_os += loc_os;
        #pragma omp atomic
        e_ss += loc_ss;
    }

    return {e_os, e_ss, std::move(T2)};
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> eval_omp2_1rdm_rhf(
    const Eigen::MatrixXd& T2,
    int n_occ,
    int n_vir,
    double c_os,
    double c_ss)
{
    const double c_same = c_os + c_ss;
    Eigen::MatrixXd T_eff(T2.rows(), T2.cols());

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < n_occ * n_occ; ++idx) {
        Eigen::Map<const Eigen::MatrixXd> T_ij(T2.col(idx).data(), n_vir, n_vir);
        Eigen::Map<Eigen::MatrixXd> Te_ij(T_eff.col(idx).data(), n_vir, n_vir);
        Te_ij = c_same * T_ij - c_ss * T_ij.transpose();
    }

    Eigen::MatrixXd gamma_oo = Eigen::MatrixXd::Zero(n_occ, n_occ);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n_occ; ++i) {
        for (int j = 0; j < n_occ; ++j) {
            double val = 0.0;
            for (int k = 0; k < n_occ; ++k) {
                val += T2.col(static_cast<Eigen::Index>(i) * n_occ + k)
                         .dot(T_eff.col(static_cast<Eigen::Index>(j) * n_occ + k));
            }
            gamma_oo(i, j) = -val;
        }
    }

    Eigen::MatrixXd gamma_vv = Eigen::MatrixXd::Zero(n_vir, n_vir);
    #pragma omp parallel
    {
        Eigen::MatrixXd gvv_loc = Eigen::MatrixXd::Zero(n_vir, n_vir);
        #pragma omp for schedule(static)
        for (int idx = 0; idx < n_occ * n_occ; ++idx) {
            Eigen::Map<const Eigen::MatrixXd> T_ij(T2.col(idx).data(), n_vir, n_vir);
            Eigen::Map<const Eigen::MatrixXd> Te_ij(T_eff.col(idx).data(), n_vir, n_vir);
            gvv_loc.noalias() += T_ij * Te_ij.transpose();
        }
        #pragma omp critical
        {
            gamma_vv.noalias() += gvv_loc;
        }
    }

    gamma_oo = 0.5 * (gamma_oo + gamma_oo.transpose());
    gamma_vv = 0.5 * (gamma_vv + gamma_vv.transpose());
    return {std::move(gamma_oo), std::move(gamma_vv)};
}

Eigen::MatrixXd eval_omp2_zvec_rhf_df(
    const Eigen::MatrixXd& T2,
    const Eigen::MatrixXd& B_oo,
    const Eigen::MatrixXd& B_ov,
    const Eigen::MatrixXd& B_vv,
    int n_occ,
    int n_vir,
    double c_os,
    double c_ss)
{
    const int n_aux = static_cast<int>(B_ov.cols());
    const double c_same = c_os + c_ss;
    Eigen::MatrixXd X_all = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(n_occ) * n_vir, n_aux);

    #pragma omp parallel
    {
        Eigen::MatrixXd T_eff_ij(n_vir, n_vir);
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < n_occ; ++i) {
            auto X_ai = X_all.middleRows(static_cast<Eigen::Index>(i) * n_vir, n_vir);
            for (int j = 0; j < n_occ; ++j) {
                Eigen::Map<const Eigen::MatrixXd> T_ij(
                    T2.col(static_cast<Eigen::Index>(i) * n_occ + j).data(), n_vir, n_vir);
                T_eff_ij = c_same * T_ij - c_ss * T_ij.transpose();
                X_ai.noalias() += T_eff_ij * B_ov.middleRows(
                    static_cast<Eigen::Index>(j) * n_vir, n_vir);
            }
        }
    }

    Eigen::MatrixXd Z_vo = Eigen::MatrixXd::Zero(n_vir, n_occ);
    #pragma omp parallel
    {
        Eigen::MatrixXd Z_loc = Eigen::MatrixXd::Zero(n_vir, n_occ);
        #pragma omp for schedule(static)
        for (int P = 0; P < n_aux; ++P) {
            Eigen::Map<const Eigen::MatrixXd> B_oo_P(B_oo.col(P).data(), n_occ, n_occ);
            Eigen::Map<const Eigen::MatrixXd> B_vv_P(B_vv.col(P).data(), n_vir, n_vir);
            Eigen::Map<const Eigen::MatrixXd> X_P(X_all.col(P).data(), n_vir, n_occ);
            Z_loc.noalias() += B_vv_P * X_P;
            Z_loc.noalias() -= X_P * B_oo_P;
        }
        #pragma omp critical
        {
            Z_vo.noalias() += Z_loc;
        }
    }
    return Z_vo;
}

Eigen::VectorXd contract_ao_sym(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& M_sym,
    int nbf)
{
    if (!integrals::has_valid_ao_factor(L_mat, nbf)) {
        throw std::invalid_argument("contract_ao_sym: dimensi L_mat tidak cocok dengan nbf");
    }
    return integrals::contract_ao_factor_with_sym(L_mat, M_sym, nbf);
}

Eigen::MatrixXd compute_gamma2c(
    const Eigen::MatrixXd& Gamma_full,
    const Eigen::MatrixXd& L_mat,
    int nbf)
{
    if (!integrals::has_valid_ao_factor(L_mat, nbf)) {
        throw std::invalid_argument("compute_gamma2c: dimensi L_mat tidak cocok dengan nbf");
    }
    return integrals::compute_gamma2c_from_full_and_L(Gamma_full, L_mat, nbf);
}

Eigen::MatrixXd eval_df_nuclear_gradient(
    const Molecule& mol,
    const BasisSet& primary_basis,
    const BasisSet& aux_basis,
    const Eigen::MatrixXd& P_eff_ao,
    const Eigen::MatrixXd& W_eff_ao,
    Eigen::MatrixXd Gamma_3c,
    const Eigen::MatrixXd& gamma_2c,
    double cutoff)
{
    const int natoms = static_cast<int>(mol.n_atoms());
    Eigen::MatrixXd g_tot = Eigen::MatrixXd::Zero(natoms, 3);

    for (int a = 0; a < natoms; ++a) {
        const double Za = static_cast<double>(mol.atom(a).atomic_number);
        const Eigen::Vector3d ra(mol.atom(a).x, mol.atom(a).y, mol.atom(a).z);
        for (int b = a + 1; b < natoms; ++b) {
            const double Zb = static_cast<double>(mol.atom(b).atomic_number);
            const Eigen::Vector3d rb(mol.atom(b).x, mol.atom(b).y, mol.atom(b).z);
            const Eigen::Vector3d rab = ra - rb;
            const double r2 = rab.squaredNorm();
            const Eigen::Vector3d f = (-Za * Zb / (r2 * std::sqrt(r2))) * rab;
            g_tot.row(a) += f.transpose();
            g_tot.row(b) -= f.transpose();
        }
    }

    IntegralEngine prim_ints(mol, primary_basis);
    g_tot.noalias() += prim_ints.compute_1e_gradient(P_eff_ao, W_eff_ao);

    BasisSet combined = primary_basis;
    combined.append(aux_basis);
    IntegralEngine df_ints(mol, combined);

    const int n_shells_prim = static_cast<int>(primary_basis.n_shells());
    const int n_shells_aux = static_cast<int>(aux_basis.n_shells());
    const int nbf = static_cast<int>(primary_basis.n_basis_functions());
    const int n_aux = static_cast<int>(aux_basis.n_basis_functions());

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
                if (!df_ints.compute_2c2e_block_raw(abs_R, abs_S, buf.data())) continue;
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

    auto df_metric = integrals::DFMetricFactor::compute(J_metric, cutoff);
    df_metric.apply_Gamma_to_tilde_inplace(Gamma_3c);
    Eigen::MatrixXd gamma_tilde = df_metric.apply_gamma2c_to_tilde(gamma_2c);

    g_tot.noalias() += df_ints.compute_df_2e_gradient(
        n_shells_prim, n_shells_aux, nbf, n_aux, Gamma_3c, gamma_tilde);
    return g_tot;
}


std::tuple<double, Eigen::MatrixXd, Eigen::MatrixXd> eval_lt_sos_intermediates_rhf_df(
    const Eigen::MatrixXd& B_ov,
    const Eigen::VectorXd& d_ov,
    const Eigen::VectorXd& weights,
    const Eigen::VectorXd& nodes,
    double c_os)
{
    const Eigen::Index m = B_ov.rows();
    const Eigen::Index n_aux = B_ov.cols();
    const int nq = static_cast<int>(weights.size());
    if (d_ov.size() != m || nodes.size() != weights.size()) {
        throw std::invalid_argument("eval_lt_sos_intermediates_rhf_df: dimensi input tidak cocok");
    }

    Eigen::MatrixXd X_conn = Eigen::MatrixXd::Zero(m, n_aux);
    Eigen::MatrixXd X_deriv = Eigen::MatrixXd::Zero(m, n_aux);
    if (m == 0 || n_aux == 0 || nq == 0) {
        return {0.0, std::move(X_conn), std::move(X_deriv)};
    }

    double e_sum = 0.0;
    #pragma omp parallel reduction(+:e_sum)
    {
        Eigen::VectorXd exp_half(m);
        Eigen::MatrixXd B_w(m, n_aux);
        Eigen::MatrixXd X_q(n_aux, n_aux);
        Eigen::MatrixXd Y_q(m, n_aux);
        Eigen::MatrixXd Xc_loc = Eigen::MatrixXd::Zero(m, n_aux);
        Eigen::MatrixXd Xd_loc = Eigen::MatrixXd::Zero(m, n_aux);

        #pragma omp for schedule(static) nowait
        for (int q = 0; q < nq; ++q) {
            const double wq = weights(q);
            const double tq = nodes(q);
            exp_half = (-0.5 * tq * d_ov.array()).exp();
            B_w = B_ov.array().colwise() * exp_half.array();
            X_q.noalias() = B_w.transpose() * B_w;
            e_sum += wq * X_q.squaredNorm();

            Y_q.noalias() = B_w * X_q;
            Y_q.array().colwise() *= exp_half.array();
            Xc_loc.noalias() -= (c_os * wq) * Y_q;
            Xd_loc.noalias() -= (c_os * wq * tq) * Y_q;
        }

        #pragma omp critical
        {
            X_conn.noalias() += Xc_loc;
            X_deriv.noalias() += Xd_loc;
        }
    }
    return {-c_os * e_sum, std::move(X_conn), std::move(X_deriv)};
}

std::tuple<double, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
eval_lt_sos_intermediates_uhf_df(
    const Eigen::MatrixXd& B_ov_a,
    const Eigen::MatrixXd& B_ov_b,
    const Eigen::VectorXd& da,
    const Eigen::VectorXd& db,
    const Eigen::VectorXd& weights,
    const Eigen::VectorXd& nodes,
    double c_os)
{
    const Eigen::Index ma = B_ov_a.rows();
    const Eigen::Index mb = B_ov_b.rows();
    const Eigen::Index n_aux = B_ov_a.cols();
    const int nq = static_cast<int>(weights.size());
    if (da.size() != ma || db.size() != mb || B_ov_b.cols() != n_aux || nodes.size() != weights.size()) {
        throw std::invalid_argument("eval_lt_sos_intermediates_uhf_df: dimensi input tidak cocok");
    }

    Eigen::MatrixXd Xc_a = Eigen::MatrixXd::Zero(ma, n_aux);
    Eigen::MatrixXd Xd_a = Eigen::MatrixXd::Zero(ma, n_aux);
    Eigen::MatrixXd Xc_b = Eigen::MatrixXd::Zero(mb, n_aux);
    Eigen::MatrixXd Xd_b = Eigen::MatrixXd::Zero(mb, n_aux);
    if (ma == 0 || mb == 0 || n_aux == 0 || nq == 0) {
        return {0.0, std::move(Xc_a), std::move(Xd_a), std::move(Xc_b), std::move(Xd_b)};
    }

    double e_sum = 0.0;
    #pragma omp parallel reduction(+:e_sum)
    {
        Eigen::VectorXd ea_half(ma), eb_half(mb);
        Eigen::MatrixXd Ba_w(ma, n_aux), Bb_w(mb, n_aux);
        Eigen::MatrixXd Xa_q(n_aux, n_aux), Xb_q(n_aux, n_aux);
        Eigen::MatrixXd Ya_q(ma, n_aux), Yb_q(mb, n_aux);
        Eigen::MatrixXd Xca_loc = Eigen::MatrixXd::Zero(ma, n_aux);
        Eigen::MatrixXd Xda_loc = Eigen::MatrixXd::Zero(ma, n_aux);
        Eigen::MatrixXd Xcb_loc = Eigen::MatrixXd::Zero(mb, n_aux);
        Eigen::MatrixXd Xdb_loc = Eigen::MatrixXd::Zero(mb, n_aux);

        #pragma omp for schedule(static) nowait
        for (int q = 0; q < nq; ++q) {
            const double wq = weights(q);
            const double tq = nodes(q);
            ea_half = (-0.5 * tq * da.array()).exp();
            eb_half = (-0.5 * tq * db.array()).exp();
            Ba_w = B_ov_a.array().colwise() * ea_half.array();
            Bb_w = B_ov_b.array().colwise() * eb_half.array();
            Xa_q.noalias() = Ba_w.transpose() * Ba_w;
            Xb_q.noalias() = Bb_w.transpose() * Bb_w;
            e_sum += wq * Xa_q.cwiseProduct(Xb_q).sum();

            Ya_q.noalias() = Ba_w * Xb_q;
            Ya_q.array().colwise() *= ea_half.array();
            Yb_q.noalias() = Bb_w * Xa_q;
            Yb_q.array().colwise() *= eb_half.array();

            Xca_loc.noalias() -= (c_os * wq) * Ya_q;
            Xda_loc.noalias() -= (c_os * wq * tq) * Ya_q;
            Xcb_loc.noalias() -= (c_os * wq) * Yb_q;
            Xdb_loc.noalias() -= (c_os * wq * tq) * Yb_q;
        }

        #pragma omp critical
        {
            Xc_a.noalias() += Xca_loc;
            Xd_a.noalias() += Xda_loc;
            Xc_b.noalias() += Xcb_loc;
            Xd_b.noalias() += Xdb_loc;
        }
    }
    return {-c_os * e_sum, std::move(Xc_a), std::move(Xd_a), std::move(Xc_b), std::move(Xd_b)};
}

} // namespace kernels
} // namespace mshqc
