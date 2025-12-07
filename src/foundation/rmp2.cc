/**
 * @file rmp2.cc
 * @brief Implementation of Restricted MP2 for closed-shell systems
 * 
 * THEORY REFERENCES:
 *   - C. Møller & M. S. Plesset, Phys. Rev. 46, 618 (1934)
 *   - A. Szabo & N. S. Ostlund, "Modern Quantum Chemistry" (1996), Ch. 6
*//*
 * @brief Implementation of Restricted MP2 using Efficient Integral Transformation
 * @author Muhamad Sahrul Hidayat
 */
/**
 */

#include "mshqc/foundation/rmp2.h"
#include "mshqc/integrals/eri_transformer.h" // Include transformer
#include <iostream>
#include <iomanip>
#include <cmath>

namespace mshqc {
namespace foundation {

RMP2::RMP2(const SCFResult& rhf_result,
           const BasisSet& basis,
           std::shared_ptr<IntegralEngine> integrals)
    : rhf_(rhf_result), basis_(basis), integrals_(integrals) {
    
    nbf_ = basis_.n_basis_functions();
    nocc_ = rhf_.n_occ_alpha;
    nvirt_ = nbf_ - nocc_;
    
    // Safety check for closed-shell
    if (rhf_.n_occ_alpha != rhf_.n_occ_beta) {
        throw std::runtime_error("RMP2 Error: Reference must be closed-shell RHF.");
    }
}

void RMP2::transform_integrals_ao_to_mo() {
    std::cout << "  Transforming integrals to MO basis (Quarter Transform O(N^5))...\n";
    
    // Gunakan Integral Engine
    auto eri_ao = integrals_->compute_eri();
    
    // Gunakan ERITransformer yang sudah kita buat
    using integrals::ERITransformer;
    
    // RHF hanya punya satu set koefisien C (Alpha = Beta)
    const Eigen::MatrixXd& C = rhf_.C_alpha;
    
    Eigen::MatrixXd C_occ = C.leftCols(nocc_);
    Eigen::MatrixXd C_virt = C.rightCols(nvirt_);
    
    // Transformasi <ij|ab> menggunakan Quarter Transform
    // Karena RHF: OccAlpha == OccBeta, VirtAlpha == VirtBeta
    // Kita panggil transform_oovv sekali saja.
    
    // Output dari transformer: (i, a, j, b) -> Chemist Notation
    auto eri_chemist = ERITransformer::transform_oovv_quarter(
        eri_ao, C_occ, C_virt, nbf_, nocc_, nvirt_
    );
    
    // RMP2 butuh Physicist Notation <ij|ab>
    // Kita lakukan Shuffle: (i, a, j, b) -> (i, j, a, b)
    // Indeks: 0->0, 1->2, 2->1, 3->3
    Eigen::array<int, 4> shuf = {0, 2, 1, 3};
    eri_mo_ = eri_chemist.shuffle(shuf);
    
    std::cout << "  Transformation complete.\n";
}

void RMP2::compute_t2_amplitudes() {
    const Eigen::VectorXd& eps = rhf_.orbital_energies_alpha;
    
    t2_ = Eigen::Tensor<double, 4>(nocc_, nocc_, nvirt_, nvirt_);
    
    for (int i = 0; i < nocc_; i++) {
        for (int j = 0; j < nocc_; j++) {
            for (int a = 0; a < nvirt_; a++) {
                for (int b = 0; b < nvirt_; b++) {
                    // Denominator: ei + ej - ea - eb
                    double denom = eps(i) + eps(j) - eps(nocc_ + a) - eps(nocc_ + b);
                    
                    if (std::abs(denom) < 1e-12) continue;
                    
                    // RMP2 Amplitude: <ij|ab> / D
                    t2_(i, j, a, b) = eri_mo_(i, j, a, b) / denom;
                }
            }
        }
    }
}

double RMP2::compute_correlation_energy() {
    // E(2) = sum_ijab (2<ij|ab> - <ij|ba>) * t_ij^ab
    double e_mp2 = 0.0;
    
    for (int i = 0; i < nocc_; i++) {
        for (int j = 0; j < nocc_; j++) {
            for (int a = 0; a < nvirt_; a++) {
                for (int b = 0; b < nvirt_; b++) {
                    double dir = eri_mo_(i, j, a, b);   // <ij|ab>
                    double ex  = eri_mo_(i, j, b, a);   // <ij|ba>
                    double t   = t2_(i, j, a, b);
                    
                    e_mp2 += (2.0 * dir - ex) * t;
                }
            }
        }
    }
    return e_mp2;
}

RMP2Result RMP2::compute() {
    std::cout << "\n=== RMP2 Calculation (Efficient) ===\n";
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