/**
 * @file rmp3.cc
 * @brief Implementation of Restricted Møller-Plesset 3rd-order perturbation theory
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 * @license MIT
 */

#include "mshqc/foundation/rmp3.h"
#include "mshqc/integrals/eri_transformer.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace mshqc {
namespace foundation {

RMP3::RMP3(const SCFResult& rhf_result,
           const RMP2Result& rmp2_result,
           const BasisSet& basis,
           std::shared_ptr<IntegralEngine> integrals)
    : rhf_(rhf_result), rmp2_(rmp2_result), basis_(basis), integrals_(integrals) {
    
    nbf_ = basis.n_basis_functions();
    nocc_ = rmp2_.n_occ;
    nvirt_ = rmp2_.n_virt;
}

void RMP3::build_fock_mo() {}

RMP3Result RMP3::compute() {
    std::cout << "\n====================================\n";
    std::cout << "  Restricted MP3 (Pople Formula)\n";
    std::cout << "====================================\n";
    
    using namespace mshqc::integrals;
    auto eri_ao = integrals_->compute_eri();
    const auto& C_occ = rhf_.C_alpha.leftCols(nocc_);
    const auto& C_virt = rhf_.C_alpha.rightCols(nvirt_);
    const auto& eps = rhf_.orbital_energies_alpha;
    const auto& t2_1 = rmp2_.t2;

    // 1. Transformasi Integral
    std::cout << "  1. Transforming VVVV block...\n";
    auto I_vvvv = ERITransformer::transform_vvvv(eri_ao, C_virt, nbf_, nvirt_);

    std::cout << "  2. Transforming OOOO block...\n";
    auto I_oooo = ERITransformer::transform_oooo(eri_ao, C_occ, nbf_, nocc_);

    std::cout << "  3. Transforming OVOV block (The only one needed for mixed terms)...\n";
    // I_ovov(i, a, j, b) corresponds to (ia|jb)
    auto I_ovov = ERITransformer::transform_ovov(eri_ao, C_occ, C_virt, nbf_, nocc_, nvirt_);

    t2_2_ = Eigen::Tensor<double, 4>(nocc_, nocc_, nvirt_, nvirt_);
    t2_2_.setZero();

    std::cout << "  4. Computing T2(2) Amplitudes...\n";
    
    #pragma omp parallel for collapse(4)
    for (int i = 0; i < nocc_; i++) {
        for (int j = 0; j < nocc_; j++) {
            for (int a = 0; a < nvirt_; a++) {
                for (int b = 0; b < nvirt_; b++) {
                    
                    double denom = eps(i) + eps(j) - eps(nocc_ + a) - eps(nocc_ + b);
                    if (std::abs(denom) < 1e-12) continue;
                    
                    double val = 0.0;

                    // A. Particle-Particle (VVVV)
                    for (int c = 0; c < nvirt_; c++) {
                        for (int d = 0; d < nvirt_; d++) {
                            // Direct: <ab|cd> -> (ac|bd) -> I(a,c,b,d)
                            double dir = I_vvvv(a, c, b, d);
                            
                            // Exchange: <ab|dc> -> (ad|bc) -> I(a,d,b,c)
                            double ex = I_vvvv(a, d, b, c);
                            
                            // Masukkan term (Direct - Exchange)
                            val += 0.5 * (dir - ex) * t2_1(i, j, c, d);
                        }
                    }

                    // B. Hole-Hole (OOOO)
                    for (int k = 0; k < nocc_; k++) {
                        for (int l = 0; l < nocc_; l++) {
                            // Direct: <kl|ij> -> (ki|lj) -> I(k,i,l,j)
                            double dir = I_oooo(k, i, l, j);
                            
                            // Exchange: <kl|ji> -> (kj|li) -> I(k,j,l,i)
                            double ex = I_oooo(k, j, l, i);
                            
                            // Masukkan term (Direct - Exchange)
                            val += 0.5 * (dir - ex) * t2_1(k, l, a, b);
                        }
                    }

                    // C. Particle-Hole / Ring (OVOV Only!)
                    // Reference: Pople et al. (1977) Eq. 15
                    // Term 1: Sum_kc [ 2(jb|kc) - (kb|jc) ] * t_ik^ac
                    // Term 2: Sum_kc [ 2(ia|kc) - (ka|ic) ] * t_jk^bc (Permutation i<->j, a<->b)
                    
                    // C. Particle-Hole / Ring (OVOV Only!)
                    // Reference: Pople et al. (1977) Eq. 15
                    // [CRITICAL FIX]: Term ini biasanya memiliki tanda NEGATIF dalam sum amplitudo
                    // relatif terhadap definisi integral positif.
                    
                    for (int k = 0; k < nocc_; k++) {
                        for (int c = 0; c < nvirt_; c++) {
                            // Mapping (p q | r s) -> I_ovov(p, r, q, s)
                            
                            double int_jbkc = I_ovov(j, b, k, c); // (jb|kc)
                            double int_kbjc = I_ovov(k, b, j, c); // (kb|jc)
                            double int_iakc = I_ovov(i, a, k, c); // (ia|kc)
                            double int_kaic = I_ovov(k, a, i, c); // (ka|ic)
                            
                            double t_ikac = t2_1(i, k, a, c);
                            double t_jkbc = t2_1(j, k, b, c);
                            
                            // FIX: Ganti '+=' menjadi '-='
                            // Diagramatik MP3 Ring term memiliki faktor topologi (-1)
                            val -= (2.0 * int_jbkc - int_kbjc) * t_ikac;
                            val -= (2.0 * int_iakc - int_kaic) * t_jkbc;
                        }
                    }
                    
                    t2_2_(i, j, a, b) = val / denom;
                }
            }
        }
    }

    std::cout << "  5. Computing Energy...\n";
    double e_mp3 = 0.0;
    
    #pragma omp parallel for reduction(+:e_mp3) collapse(4)
    for (int i = 0; i < nocc_; i++) {
        for (int j = 0; j < nocc_; j++) {
            for (int a = 0; a < nvirt_; a++) {
                for (int b = 0; b < nvirt_; b++) {
                    // Energy uses (ia|jb) which is I_ovov(i, a, j, b)
                    double J = I_ovov(i, a, j, b); 
                    double K = I_ovov(i, b, j, a); 
                    
                    e_mp3 += (2.0 * J - K) * t2_2_(i, j, a, b);
                }
            }
        }
    }

    RMP3Result result;
    result.e_rhf = rmp2_.e_rhf;
    result.e_mp2 = rmp2_.e_corr;
    result.e_mp3 = e_mp3;
    result.e_corr_total = rmp2_.e_corr + e_mp3;
    result.e_total = rmp2_.e_rhf + result.e_corr_total;
    result.n_occ = nocc_;
    result.n_virt = nvirt_;
    result.t2_1 = t2_1;
    result.t2_2 = t2_2_;
    
    std::cout << std::setprecision(8);
    std::cout << "RHF Energy:       " << result.e_rhf << "\n";
    std::cout << "MP2 Correlation:  " << result.e_mp2 << "\n";
    std::cout << "MP3 Correction:   " << e_mp3 << "\n";
    std::cout << "Total Correlation:" << result.e_corr_total << "\n";
    
    return result;
}

// Deprecated
void RMP3::transform_integrals_ao_to_mo() {}
void RMP3::compute_t2_second_order() {}
double RMP3::compute_third_order_energy() { return 0.0; }
const Eigen::Tensor<double, 4>& RMP3::get_t2_second_order() const { return t2_2_; }

} // namespace foundation
} // namespace mshqc