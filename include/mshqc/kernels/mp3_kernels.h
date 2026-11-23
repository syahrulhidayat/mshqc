// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0
// [INJECTED_HYBRID_PHASE2_MP3_KERNELS]
#pragma once

#include <Eigen/Dense>
#include <utility>
#include <vector>

namespace mshqc {
namespace kernels {

// Adaptive 3-path O(V^4)-memory-free particle-particle ladder contraction (<= 64 MB buffer)
void contract_vvvv_pairs_df(
    const Eigen::MatrixXd& B_vv_left, int nv_l,
    const Eigen::MatrixXd& B_vv_right, int nv_r,
    int n_aux,
    const std::vector<Eigen::MatrixXd>& T_pairs,
    std::vector<Eigen::MatrixXd>& W_pairs);

// RHF occupied-pair sliced (i <= j) particle-particle ladder using adaptive 3-path kernel
Eigen::MatrixXd eval_pp_ladder_rhf_df(
    const Eigen::MatrixXd& T2,
    const Eigen::MatrixXd& B_vv,
    int n_occ,
    int n_vir,
    bool include_vvvv = true);

// Complete RHF DF/Cholesky MP3 energy E^(3) and second-order amplitude L2 = W / D
std::pair<double, Eigen::MatrixXd> eval_rmp3_df(
    const Eigen::MatrixXd& T2,
    const Eigen::MatrixXd& B_oo,
    const Eigen::MatrixXd& B_ov,
    const Eigen::MatrixXd& B_vv,
    const Eigen::VectorXd& eps_occ,
    const Eigen::VectorXd& eps_vir,
    const Eigen::MatrixXd& W_pp_override = Eigen::MatrixXd());

// Streaming zero-copy RHF OMP3 1-RDM (gamma_oo, gamma_vv)
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> eval_omp3_1rdm_rhf(
    const Eigen::MatrixXd& T2,
    const Eigen::MatrixXd& L2,
    int n_occ,
    int n_vir);

// O(V^4)-memory-free RHF OMP3 Z-vector via chunked Y_ooov intermediate (<= 64 MB)
Eigen::MatrixXd eval_omp3_zvec_rhf_df(
    const Eigen::MatrixXd& T2,
    const Eigen::MatrixXd& L2,
    const Eigen::MatrixXd& B_oo,
    const Eigen::MatrixXd& B_ov,
    const Eigen::MatrixXd& B_vv,
    int n_occ,
    int n_vir);

// MO-space DF approximate orbital Hessian-vector product H * kappa for RHF TR-SOSCF
Eigen::MatrixXd apply_mo_df_hessian_rhf(
    const Eigen::MatrixXd& kappa_vo,
    const Eigen::VectorXd& eps_occ,
    const Eigen::VectorXd& eps_vir,
    const Eigen::MatrixXd& B_oo,
    const Eigen::MatrixXd& B_ov,
    const Eigen::MatrixXd& B_vv);

// MO-space correlated 1-RDM G[P_corr]_vo contractions (zero AO unpacking)
Eigen::MatrixXd eval_mo_gcorr_rhf_df(
    const Eigen::MatrixXd& B_oo,
    const Eigen::MatrixXd& B_ov,
    const Eigen::MatrixXd& B_vv,
    const Eigen::MatrixXd& g_oo,
    const Eigen::MatrixXd& g_vv,
    int n_occ,
    int n_vir);

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
    int na, int va, int nb, int vb);

} // namespace kernels
} // namespace mshqc
