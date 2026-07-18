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
    // Mewarisi konstruktor OMP2 secara otomatis. 
    // Ini menghemat ratusan baris kode dan mencegah error mismatch parameter di file .cc
    using OMP2::OMP2;

    virtual ~OMP3() = default;

    // Fungsi eksekusi utama OMP3
    MP3Result compute_omp3();

protected:
    // OVERRIDE: Menghubungkan kontrol iterasi dan energi ke mesin OMP2
    double execute_micro_iterations() override;
    void build_generalized_fock() override;
    double get_correlation_energy() const override;

    // OVERRIDE: Membangun Matriks Densitas Satu-Partikel (OPDM) khusus MP3
    void build_opdm_alpha() override;
    void build_opdm_beta() override;

    void build_hessian_diagonal(Eigen::VectorXd& diag_H, double grad_norm) override;

    // Penyimpanan Internal Khusus Orde-3 (Energi)
    double e_mp3_aa_ = 0.0;
    double e_mp3_bb_ = 0.0;
    double e_mp3_ab_ = 0.0;
    double e_mp3_tot_ = 0.0;

    // Penyimpanan Amplitudo Orde-3 dan Amplitudo Lagrangian (Lambda)
    Eigen::Tensor<double, 4> t2_3rd_aa_, t2_3rd_bb_, t2_3rd_ab_;
    Eigen::Tensor<double, 4> L2_aa_, L2_bb_, L2_ab_;
    Eigen::Tensor<double, 4> Waa_ladder_;
    Eigen::Tensor<double, 4> Waa_ring_;

    // Evaluasi Energi Koreksi MP3
    void compute_mp3_correction();
};

} // namespace mshqc
