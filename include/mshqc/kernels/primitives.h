// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0
// [INJECTED_HYBRID_PHASE2_PRIMITIVES]
#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <memory>
#include <omp.h>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "mshqc/basis.h"
#include "mshqc/core/molecule.h"
#include "mshqc/integrals/df_eri.h"
#include "mshqc/integrals/packed_ao_utils.h"
#include "mshqc/ints/integrals.h"
#include "mshqc/utils/blas_thread_guard.h"

namespace mshqc {
namespace kernels {

// 0. RAII-safe Density Fitting AO 3-index factor builder (auto-combines primary + aux basis)
inline Eigen::MatrixXd compute_df_ao_factor(
    const Molecule& mol,
    const BasisSet& primary_basis,
    const BasisSet& aux_basis,
    double cutoff = 1e-10,
    bool save_hdf5 = false,
    bool verbose = false)
{
    BasisSet combined = primary_basis;
    combined.append(aux_basis);
    auto df_ints = std::make_shared<IntegralEngine>(mol, combined);
    integrals::DensityFittingERI df_eng(primary_basis, aux_basis, df_ints, cutoff);
    df_eng.set_verbose(verbose);
    df_eng.compute(save_hdf5);
    return std::move(df_eng.get_B_mat_mutable());
}

// Safe RAII class for Python bindings (prevents dangling BasisSet/IntegralEngine references)
class SafeDFEngine {
public:
    SafeDFEngine(const Molecule& mol,
                 const BasisSet& primary_basis,
                 const BasisSet& aux_basis,
                 double cutoff = 1e-10)
        : mol_(mol),
          primary_(primary_basis),
          aux_(aux_basis),
          combined_(primary_basis),
          cutoff_(cutoff)
    {
        combined_.append(aux_);
        df_ints_ = std::make_shared<IntegralEngine>(mol_, combined_);
        df_eng_ = std::make_unique<integrals::DensityFittingERI>(
            primary_, aux_, df_ints_, cutoff_);
    }

    void compute(bool save_hdf5 = false) {
        df_eng_->compute(save_hdf5);
    }

    const Eigen::MatrixXd& get_B_mat() const {
        return df_eng_->get_B_mat();
    }

    void set_verbose(bool v) { df_eng_->set_verbose(v); }
    void set_hdf5_path(const std::string& path) { df_eng_->set_hdf5_path(path); }
    const std::string& get_hdf5_path() const { return df_eng_->get_hdf5_path(); }

private:
    Molecule mol_;
    BasisSet primary_;
    BasisSet aux_;
    BasisSet combined_;
    double cutoff_;
    std::shared_ptr<IntegralEngine> df_ints_;
    std::unique_ptr<integrals::DensityFittingERI> df_eng_;
};

// 1. Coulomb J[P_tot] dari faktor 3-indeks AO (packed atau full)
inline Eigen::MatrixXd build_coulomb_3c(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& P_tot,
    int nbf)
{
    if (!integrals::has_valid_ao_factor(L_mat, nbf)) {
        throw std::invalid_argument("build_coulomb_3c: dimensi L_mat tidak cocok dengan nbf");
    }
    return integrals::compute_coulomb_from_ao_factor(L_mat, P_tot, nbf);
}

// 2. Exchange K[C_occ * C_occ^T] berorde O(N_aux * N_occ * N_bf^2)
inline Eigen::MatrixXd build_exchange_3c_occ(
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

    #pragma omp parallel
    {
        Eigen::MatrixXd K_loc = Eigen::MatrixXd::Zero(nbf, nbf);
        Eigen::MatrixXd L_P(nbf, nbf);
        Eigen::MatrixXd T_P(nbf, n_occ);

        #pragma omp for schedule(static)
        for (int P = 0; P < n_aux; ++P) {
            integrals::unpack_ao_column(L_mat.col(P).data(), nbf, packed, L_P);
            T_P.noalias() = L_P * C_occ;
            K_loc.noalias() += T_P * T_P.transpose();
        }

        #pragma omp critical
        {
            K_tot.noalias() += K_loc;
        }
    }
    return 0.5 * (K_tot + K_tot.transpose());
}

// 3. Exchange K[D_sym] untuk matriks densitas simetris umum (mis. P_corr pada OMP2/OMP3)
inline Eigen::MatrixXd build_exchange_3c_sym(
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

// 4. Paket cepat (J[2*C*C^T], K[C*C^T]) untuk RHF sehingga F = H_core + J - K
inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd> build_jk_3c_rhf(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& C_occ,
    int nbf)
{
    Eigen::MatrixXd P_tot = 2.0 * (C_occ * C_occ.transpose());
    Eigen::MatrixXd J = build_coulomb_3c(L_mat, P_tot, nbf);
    Eigen::MatrixXd K = build_exchange_3c_occ(L_mat, C_occ, nbf);
    return {std::move(J), std::move(K)};
}

// 5. Transformasi 3-indeks AO -> MO: B(x*d2 + y, P)
inline Eigen::MatrixXd transform_3c_mo(
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

// 6. Semi-kanonikalisasi sub-ruang Occ-Occ dan Vir-Vir
inline std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd> semicanonicalize_subspaces(
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

// 7. Evaluasi amplitudo T2 dan energi OS/SS RMP2 dari B_ov (no*nv x n_aux)
inline std::tuple<double, double, Eigen::MatrixXd> eval_rmp2_df(
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

// 8. 1-RDM korelasi OMP2 RHF per kanal spin alpha, mendukung spin-scaling (c_os, c_ss)
inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd> eval_omp2_1rdm_rhf(
    const Eigen::MatrixXd& T2,
    int n_occ,
    int n_vir,
    double c_os = 1.0,
    double c_ss = 1.0)
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

// 9. Vektor-Z OMP2 RHF via faktor 3-indeks (Pers. 2.71g), mendukung SCS-OMP2
inline Eigen::MatrixXd eval_omp2_zvec_rhf_df(
    const Eigen::MatrixXd& T2,
    const Eigen::MatrixXd& B_oo,
    const Eigen::MatrixXd& B_ov,
    const Eigen::MatrixXd& B_vv,
    int n_occ,
    int n_vir,
    double c_os = 1.0,
    double c_ss = 1.0)
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

} // namespace kernels
} // namespace mshqc
