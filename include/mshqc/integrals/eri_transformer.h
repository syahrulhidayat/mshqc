// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0
#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <utility>
#include "mshqc/ints/integrals.h"

namespace mshqc {
namespace integrals {

// 1. Direct Exact 4-Index Coulomb & Exchange Builders (OpenMP column-tiled BLAS + TBLIS)
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> build_jk_4d_rhf(
    const ERITensor& eri_ao,
    const Eigen::MatrixXd& C_occ);

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> build_jk_4d_uhf(
    const ERITensor& eri_ao,
    const Eigen::MatrixXd& C_occ_a,
    const Eigen::MatrixXd& C_occ_b);

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> build_jk_4d_density_uhf(
    const ERITensor& eri_ao,
    const Eigen::MatrixXd& P_alpha,
    const Eigen::MatrixXd& P_beta);

// 2. TBLIS-Powered Exact 4-Index AO -> MO Transformation -> matrix (n1*n2 x n3*n4) in row-major pair order
Eigen::MatrixXd transform_eri_4d(
    const ERITensor& eri_ao,
    const Eigen::MatrixXd& C1,
    const Eigen::MatrixXd& C2,
    const Eigen::MatrixXd& C3,
    const Eigen::MatrixXd& C4);

// 3. Direct O(O N^4) TBLIS Exact MP2 Energies (zero Cholesky approximation)
std::pair<double, double> eval_rmp2_exact(
    const ERITensor& eri_ao,
    const Eigen::MatrixXd& C_occ,
    const Eigen::MatrixXd& C_vir,
    const Eigen::VectorXd& eps_occ,
    const Eigen::VectorXd& eps_vir);

std::tuple<double, double, double> eval_ump2_exact(
    const ERITensor& eri_ao,
    const Eigen::MatrixXd& Ca_occ,
    const Eigen::MatrixXd& Ca_vir,
    const Eigen::MatrixXd& Cb_occ,
    const Eigen::MatrixXd& Cb_vir,
    const Eigen::VectorXd& eps_oa,
    const Eigen::VectorXd& eps_va,
    const Eigen::VectorXd& eps_ob,
    const Eigen::VectorXd& eps_vb);

// 4. OpenMP + Blocked Level-3 BLAS Packed Lower-Triangular Cholesky directly on in-memory ERITensor
Eigen::MatrixXd factorize_eri_4d_exact(
    const ERITensor& eri_ao,
    double threshold = 1.0e-12);

} // namespace integrals
} // namespace mshqc
