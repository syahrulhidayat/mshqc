// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0
// [INJECTED_THIN_CORE_STAGE5_PRIMITIVES_H]
#pragma once

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "mshqc/basis.h"
#include "mshqc/core/molecule.h"
#include "mshqc/ints/integrals.h"

namespace mshqc {
namespace integrals {
class DensityFittingERI;
}

namespace kernels {

// 0. RAII-safe Density Fitting AO 3-index factor builder (auto-combines primary + aux basis)
Eigen::MatrixXd compute_df_ao_factor(
    const Molecule& mol,
    const BasisSet& primary_basis,
    const BasisSet& aux_basis,
    double cutoff = 1e-10,
    bool save_hdf5 = false,
    bool verbose = false);

// Safe RAII wrapper for Python bindings (prevents dangling BasisSet/IntegralEngine references)
class SafeDFEngine {
public:
    SafeDFEngine(const Molecule& mol,
                 const BasisSet& primary_basis,
                 const BasisSet& aux_basis,
                 double cutoff = 1e-10);
    ~SafeDFEngine();

    void compute(bool save_hdf5 = false);
    const Eigen::MatrixXd& get_B_mat() const;
    void set_verbose(bool v);
    void set_hdf5_path(const std::string& path);
    const std::string& get_hdf5_path() const;

private:
    Molecule mol_;
    BasisSet primary_;
    BasisSet aux_;
    BasisSet combined_;
    double cutoff_;
    std::shared_ptr<IntegralEngine> df_ints_;
    std::unique_ptr<integrals::DensityFittingERI> df_eng_;
};

// 1. Coulomb & Exchange 3-index AO contractions
Eigen::MatrixXd build_coulomb_3c(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& P_tot,
    int nbf);

Eigen::MatrixXd build_exchange_3c_occ(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& C_occ,
    int nbf);

// Fast diagonal-density exchange K[diag(d)] for SAD Iteration-0 (avoids dense L_P * C_sad GEMM)
Eigen::MatrixXd build_exchange_3c_diag(
    const Eigen::MatrixXd& L_mat,
    const Eigen::VectorXd& d_diag,
    int nbf);

Eigen::MatrixXd build_exchange_3c_sym(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& D_sym,
    int nbf);

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> build_jk_3c_rhf(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& C_occ,
    int nbf);

// 2. AO -> MO 3-index transforms (single-block and fused 1-pass multi-block)
Eigen::MatrixXd transform_3c_mo(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& C1,
    const Eigen::MatrixXd& C2,
    int nbf);

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> transform_3c_mo_blocks(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& C_occ,
    const Eigen::MatrixXd& C_vir,
    int nbf);

std::tuple<
    Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd,
    Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
transform_3c_mo_blocks_uhf(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& Ca_occ,
    const Eigen::MatrixXd& Ca_vir,
    const Eigen::MatrixXd& Cb_occ,
    const Eigen::MatrixXd& Cb_vir,
    int nbf);

// 3. Subspace semicanonicalization & RHF DF-MP2 / DF-OMP2 kernels
std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd> semicanonicalize_subspaces(
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXd& F_ao,
    int n_occ);

std::tuple<double, double, Eigen::MatrixXd> eval_rmp2_df(
    const Eigen::MatrixXd& B_ov,
    const Eigen::VectorXd& eps_occ,
    const Eigen::VectorXd& eps_vir);

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> eval_omp2_1rdm_rhf(
    const Eigen::MatrixXd& T2,
    int n_occ,
    int n_vir,
    double c_os = 1.0,
    double c_ss = 1.0);

Eigen::MatrixXd eval_omp2_zvec_rhf_df(
    const Eigen::MatrixXd& T2,
    const Eigen::MatrixXd& B_oo,
    const Eigen::MatrixXd& B_ov,
    const Eigen::MatrixXd& B_vv,
    int n_occ,
    int n_vir,
    double c_os = 1.0,
    double c_ss = 1.0);

// 4. Thin-Core Analytic DF Nuclear Gradient contractions (libcint ip* derivatives)
Eigen::VectorXd contract_ao_sym(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& M_sym,
    int nbf);

Eigen::MatrixXd compute_gamma2c(
    const Eigen::MatrixXd& Gamma_full,
    const Eigen::MatrixXd& L_mat,
    int nbf);

Eigen::MatrixXd eval_df_nuclear_gradient(
    const Molecule& mol,
    const BasisSet& primary_basis,
    const BasisSet& aux_basis,
    const Eigen::MatrixXd& P_eff_ao,
    const Eigen::MatrixXd& W_eff_ao,
    Eigen::MatrixXd Gamma_3c,
    const Eigen::MatrixXd& gamma_2c,
    double cutoff = 1e-12);

} // namespace kernels
} // namespace mshqc
