/**
 * @file rmp2.cc
 * @brief Implementation of Restricted MP2
 * (FIXED: Manual Reconstruction using std::max to recover sparse integrals)
 */

#include "mshqc/foundation/rmp2.h"
#include "mshqc/integrals/eri_transformer.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm> // Wajib ada untuk std::max

namespace mshqc {
namespace foundation {

RMP2::RMP2(const SCFResult& rhf_result,
           const BasisSet& basis,
           std::shared_ptr<IntegralEngine> integrals)
    : rhf_(rhf_result), basis_(basis), integrals_(integrals) {
    
    nbf_ = basis_.n_basis_functions();
    nocc_ = rhf_.n_occ_alpha;
    nvirt_ = nbf_ - nocc_;
}

void RMP2::transform_integrals_ao_to_mo() {
    std::cout << "  Transforming integrals to MO basis...\n";
    
    auto eri_ao = integrals_->compute_eri();
    using integrals::ERITransformer;
    
    const Eigen::MatrixXd& C = rhf_.C_alpha;
    Eigen::MatrixXd C_occ = C.leftCols(nocc_);
    Eigen::MatrixXd C_virt = C.rightCols(nvirt_);
    
    // 1. Ambil Raw Tensor (ia|jb) dari Transformer
    // Ini mungkin sparse (bolong-bolong)
    auto eri_chemist = ERITransformer::transform_ovov(
        eri_ao, C_occ, C_virt, nbf_, nocc_, nvirt_
    );
    
    // 2. Siapkan Tensor Tujuan <ij|ab>
    eri_mo_ = Eigen::Tensor<double, 4>(nocc_, nocc_, nvirt_, nvirt_);
    
    std::cout << "  Reconstructing <ij|ab> (Physicist) from (ia|jb) (Chemist)...\n";
    
    // 3. Rekonstruksi Manual (Brute Force Recovery)
    // Kita tidak pakai .shuffle() bawaan karena data mungkin hilang.
    // Kita cek kedua kemungkinan posisi data: (ia|jb) dan (jb|ia).
    
    #pragma omp parallel for collapse(4)
    for (int i = 0; i < nocc_; i++) {
        for (int j = 0; j < nocc_; j++) {
            for (int a = 0; a < nvirt_; a++) {
                for (int b = 0; b < nvirt_; b++) {
                    
                    // Chemist Notation: (i, a, j, b)
                    // Symmetry: (ia|jb) == (jb|ia)
                    
                    double val1 = eri_chemist(i, a, j, b);
                    double val2 = eri_chemist(j, b, i, a); // Tukar pasang elektron
                    
                    // AMBIL YANG ADA ISINYA (Maksimum Magnitudo)
                    // Jika val1 = 0 dan val2 ada isinya, kita ambil val2.
                    // Jika keduanya ada isinya, nilainya pasti sama, jadi aman.
                    double val_final = (std::abs(val1) > std::abs(val2)) ? val1 : val2;
                    
                    // Simpan ke Physicist Notation <ij|ab> -> (i, j, a, b)
                    eri_mo_(i, j, a, b) = val_final;
                }
            }
        }
    }
    
    std::cout << "  Reconstruction complete.\n";
}

void RMP2::compute_t2_amplitudes() {
    const Eigen::VectorXd& eps = rhf_.orbital_energies_alpha;
    t2_ = Eigen::Tensor<double, 4>(nocc_, nocc_, nvirt_, nvirt_);
    
    #pragma omp parallel for collapse(4)
    for (int i = 0; i < nocc_; i++) {
        for (int j = 0; j < nocc_; j++) {
            for (int a = 0; a < nvirt_; a++) {
                for (int b = 0; b < nvirt_; b++) {
                    double denom = eps(i) + eps(j) - eps(nocc_ + a) - eps(nocc_ + b);
                    if (std::abs(denom) < 1e-12) continue;
                    
                    t2_(i, j, a, b) = eri_mo_(i, j, a, b) / denom;
                }
            }
        }
    }
}

double RMP2::compute_correlation_energy() {
    double e_mp2 = 0.0;
    
    #pragma omp parallel for reduction(+:e_mp2) collapse(4)
    for (int i = 0; i < nocc_; i++) {
        for (int j = 0; j < nocc_; j++) {
            for (int a = 0; a < nvirt_; a++) {
                for (int b = 0; b < nvirt_; b++) {
                    
                    double dir = eri_mo_(i, j, a, b); 
                    double ex  = eri_mo_(i, j, b, a); 
                    double t   = t2_(i, j, a, b);
                    
                    e_mp2 += (2.0 * dir - ex) * t;
                }
            }
        }
    }
    return e_mp2;
}

RMP2Result RMP2::compute() {
    std::cout << "\n=== RMP2 Calculation (Manual Reconstruction) ===\n"; // Cek log ini nanti
    std::cout << "Basis functions: " << nbf_ << "\n";
    std::cout << "Occupied: " << nocc_ << ", Virtual: " << nvirt_ << "\n";
    
    transform_integrals_ao_to_mo();
    compute_t2_amplitudes();
    double e_corr = compute_correlation_energy();
    
    RMP2Result result;
    result.e_rhf = rhf_.energy_total;
    result.e_corr = e_corr;
    result.e_total = rhf_.energy_total + e_corr;
    result.n_occ = nocc_;
    result.n_virt = nvirt_;
    result.t2 = t2_;
    
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "RHF Energy:      " << std::setw(14) << result.e_rhf << " Ha\n";
    std::cout << "MP2 Correlation: " << std::setw(14) << result.e_corr << " Ha\n";
    std::cout << "Total RMP2:      " << std::setw(14) << result.e_total << " Ha\n";
    
    return result;
}

const Eigen::Tensor<double, 4>& RMP2::get_t2_amplitudes() const {
    return t2_;
}

} // namespace foundation
} // namespace mshqc