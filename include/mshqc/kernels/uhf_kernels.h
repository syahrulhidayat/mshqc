// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0
// [INJECTED_HYBRID_PHASE3_UHF_KERNELS_H]
#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <utility>

namespace mshqc {
namespace kernels {

// 1. Coulomb total J[P_a + P_b] dan Exchange (K_a, K_b) untuk UHF 3-indeks
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> build_jk_3c_uhf(
    const Eigen::MatrixXd& L_mat,
    const Eigen::MatrixXd& C_occ_a,
    const Eigen::MatrixXd& C_occ_b,
    int nbf);

// 2. Evaluasi amplitudo & energi UMP2 DF (e_aa, e_bb, e_ab, T2_aa, T2_bb, T2_ab)
//    Layout:
//      T2_aa: (va*va x na*na), col i*na+j, row a+b*va (antisimetri)
//      T2_bb: (vb*vb x nb*nb), col i*nb+j, row a+b*vb (antisimetri)
//      T2_ab: (va*vb x na*nb), col i*nb+j, row a+b*va
std::tuple<double, double, double, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> eval_ump2_df(
    const Eigen::MatrixXd& B_ov_a,
    const Eigen::MatrixXd& B_ov_b,
    const Eigen::VectorXd& eps_oa,
    const Eigen::VectorXd& eps_va,
    const Eigen::VectorXd& eps_ob,
    const Eigen::VectorXd& eps_vb);

// 3. 1-RDM korelasi UHF OMP2 (g_oo_a, g_vv_a, g_oo_b, g_vv_b), mendukung spin-scaling
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> eval_omp2_1rdm_uhf(
    const Eigen::MatrixXd& T2_aa,
    const Eigen::MatrixXd& T2_bb,
    const Eigen::MatrixXd& T2_ab,
    int na, int va, int nb, int vb,
    double c_os = 1.0,
    double c_ss = 1.0);

// 4. Vektor-Z UHF OMP2 via faktor 3-indeks DF (Z_vo_a, Z_vo_b), mendukung spin-scaling
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
    double c_os = 1.0,
    double c_ss = 1.0);

// 5. Evaluasi UMP3 DF bebas O(V^4): (e3_aa, e3_bb, e3_ab, L2_aa, L2_bb, L2_ab)
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
    const Eigen::VectorXd& eps_vb);

// 6. 1-RDM korelasi UHF OMP3 (g_oo_a, g_vv_a, g_oo_b, g_vv_b)
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> eval_omp3_1rdm_uhf(
    const Eigen::MatrixXd& T2_aa,
    const Eigen::MatrixXd& T2_bb,
    const Eigen::MatrixXd& T2_ab,
    const Eigen::MatrixXd& L2_aa,
    const Eigen::MatrixXd& L2_bb,
    const Eigen::MatrixXd& L2_ab,
    int na, int va, int nb, int vb);

// 7. Vektor-Z UHF OMP3 bebas O(V^4) via chunked Y_ooov (Z_vo_a, Z_vo_b)
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
    int na, int va, int nb, int vb);

// 8. Produk Hessian-vektor MO-DF terkopel (alpha & beta) untuk TR-SOSCF UHF
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
    const Eigen::MatrixXd& B_vv_b);

} // namespace kernels
} // namespace mshqc
