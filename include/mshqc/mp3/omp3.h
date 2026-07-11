#pragma once
#include "mshqc/mp2/mp2.h"
#include "mshqc/mp3/mp3.h"

namespace mshqc {

class OMP3 : public OMP2 {
public:
    OMP3(const Molecule& mol, const BasisSet& basis, 
         std::shared_ptr<IntegralEngine> integrals, 
         const SCFResult& scf_guess,
         const MP2Config& config,         
         std::shared_ptr<PointGroup> pg = nullptr,
         std::shared_ptr<PetiteList> pl = nullptr);

    // Fungsi eksekusi utama OMP3
    MP3Result compute_omp3();

protected:
    // OVERRIDE: Menghubungkan mesin OMP2 ke OMP3
    double execute_micro_iterations() override;
    void build_generalized_fock() override;
    double get_correlation_energy() const override;

    // OVERRIDE: Matriks Densitas Satu-Partikel (OPDM) OMP3
    void build_opdm_alpha() override;
    void build_opdm_beta() override;

    // Penyimpanan Internal Khusus Orde-3
    double e_mp3_aa_ = 0.0;
    double e_mp3_bb_ = 0.0;
    double e_mp3_ab_ = 0.0;
    double e_mp3_tot_ = 0.0;

    Eigen::Tensor<double, 4> t2_3rd_aa_, t2_3rd_bb_, t2_3rd_ab_;
    Eigen::Tensor<double, 4> L2_aa_, L2_bb_, L2_ab_;

    // Evaluasi Energi Koreksi MP3
    void compute_mp3_correction();
};

} // namespace mshqc