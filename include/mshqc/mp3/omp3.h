/**
 * @file include/mshqc/mp3/omp3.h
 * @brief Orbital-Optimized MP3 (OMP3) Engine
 * @details Menggunakan arsitektur "share engine", mewarisi mesin SCF, DIIS, 
 * dan optimasi orbital dari modul OMP2.
 */

#pragma once

#include "mshqc/mp2/mp2.h"
#include "mshqc/mp3/mp3.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

namespace mshqc {

class OMP3 : public OMP2 {
public:
    using OMP2::OMP2;
    virtual ~OMP3() = default;
    MP3Result compute_omp3();

protected:
    double execute_micro_iterations() override;
    void build_generalized_fock() override;
    double get_correlation_energy() const override;

    void build_opdm_alpha() override;
    void build_opdm_beta() override;

    void build_hessian_diagonal(Eigen::VectorXd& diag_H, double grad_norm) override;
    void debug_gradient_fd(int i_target, int a_target);

    double e_mp3_aa_ = 0.0;
    double e_mp3_bb_ = 0.0;
    double e_mp3_ab_ = 0.0;
    double e_mp3_tot_ = 0.0;

    Eigen::Tensor<double, 4> t2_3rd_aa_, t2_3rd_bb_, t2_3rd_ab_;
    Eigen::Tensor<double, 4> L2_aa_, L2_bb_, L2_ab_;
    Eigen::Tensor<double, 4> Waa_ladder_;
    Eigen::Tensor<double, 4> Waa_ring_;

    Eigen::Tensor<double, 4> Gamma_vvvv_aa, Gamma_oooo_aa, Gamma_ovov_aa;
    Eigen::Tensor<double, 4> Gamma_vvvv_bb, Gamma_oooo_bb, Gamma_ovov_bb, Gamma_ovov_ab;
    Eigen::Tensor<double, 4> T2_tilde_aa, L2_tilde_aa;

    void compute_mp3_correction();
};

} // namespace mshqc
