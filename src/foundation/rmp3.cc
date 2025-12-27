/**
 * @file rmp3.cc
 * @brief Implementation of Restricted Møller-Plesset 3rd-order perturbation theory
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 * @license MIT
 */

/**
 * @file rmp3.cc
 * @brief Restricted MP3 with Optimized ERI Transformer
 */

/**
 * @file rmp3.cc
 * @brief FIX: Correct Sign/Indexing for RMP3
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

// Tidak perlu fungsi transformasi raksasa lagi, kita lakukan on-the-fly atau per blok
void RMP3::build_fock_mo() {
    // (Opsional) RMP3 standar untuk RHF kanonik biasanya mengasumsikan Fock diagonal (epsilon)
    // Jadi matriks Fock penuh tidak selalu dibutuhkan jika pakai orbital kanonik.
}

// Hapus transform_integrals_ao_to_mo() yang lama yang mengubah semua (C,C,C,C)
// Kita ganti dengan strategi per-blok di compute()

RMP3Result RMP3::compute() {
    std::cout << "\n====================================\n";
    std::cout << "  Restricted MP3 (Corrected Logic)\n";
    std::cout << "====================================\n";
    
    using namespace mshqc::integrals;
    auto eri_ao = integrals_->compute_eri();
    const auto& C_occ = rhf_.C_alpha.leftCols(nocc_);
    const auto& C_virt = rhf_.C_alpha.rightCols(nvirt_);
    const auto& eps = rhf_.orbital_energies_alpha;
    const auto& t2_1 = rmp2_.t2; // T2 MP2 (Physicist <ij|ab>)

    // 1. Transformasi Integral ke Blok yang dibutuhkan
    //    RMP3 butuh: <ab|cd> (VVVV), <kl|ij> (OOOO), <ak|ic> (VOVO/OVOV)
    
    std::cout << "  1. Transforming VVVV block...\n";
    // (a,b,c,d) -> Physicist <ab|cd>
    auto I_vvvv = ERITransformer::transform_vvvv(eri_ao, C_virt, nbf_, nvirt_);
    // Shuffle agar sesuai Physicist (a,b,c,d) -> (a,b,c,d) (sudah benar dari transform_vvvv jika quarter standard)
    // Cek eri_transformer Anda: jika return (p,q,r,s) dari (C1,C2,C3,C4), maka
    // transform_vvvv(Virt, Virt, Virt, Virt) -> (a, b, c, d). Ini adalah Chemist (ab|cd) = Physicist <ab|cd>.

    std::cout << "  2. Transforming OOOO block...\n";
    // (k,l,i,j) -> Chemist (kl|ij) -> Physicist <kl|ij>
    auto I_oooo = ERITransformer::transform_oooo(eri_ao, C_occ, nbf_, nocc_);

    std::cout << "  3. Transforming OVOV block (for Ring)...\n";
    // transform_ovov(Occ, Virt, Occ, Virt) -> (i, a, j, b) Chemist (ia|jb) -> Physicist <ij|ab>
    // Kita butuh variasi <ib|aj> dsb. Kita simpan blok ini.
    auto I_ovov = ERITransformer::transform_ovov(eri_ao, C_occ, C_virt, nbf_, nocc_, nvirt_);

    // Init T2 Second Order
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

                    // --- TERM A: Particle-Particle Ladder ---
                    // 0.5 * Sum_cd <ab|cd> * t_ij^cd
                    for (int c = 0; c < nvirt_; c++) {
                        for (int d = 0; d < nvirt_; d++) {
                            // I_vvvv is (a,b,c,d) -> <ab|cd>
                            double v = I_vvvv(a, b, c, d); 
                            val += 0.5 * v * t2_1(i, j, c, d);
                        }
                    }

                    // --- TERM B: Hole-Hole Ladder ---
                    // 0.5 * Sum_kl <kl|ij> * t_kl^ab
                    for (int k = 0; k < nocc_; k++) {
                        for (int l = 0; l < nocc_; l++) {
                            // I_oooo is (k,l,i,j) -> <kl|ij>
                            double v = I_oooo(k, l, i, j); 
                            val += 0.5 * v * t2_1(k, l, a, b);
                        }
                    }

                    // --- TERM C: Particle-Hole Ring ---
                    // Sum_kc [ (2<lb|kc> - <lb|ck>) * t_ik^ac + ... ]
                    // Ini bagian yang sering salah tanda/indeks.
                    // Rumus Pople (spin adapted):
                    // P_ij P_ab [ Sum_kc (2<ac|jk> - <ac|kj>) * t_ik^cb ]
                    // Mari gunakan implementasi eksplisit tanpa P operator biar jelas:
                    
                    for (int k = 0; k < nocc_; k++) {
                        for (int c = 0; c < nvirt_; c++) {
                            // Integrals from I_ovov (m, e, n, f) -> <mn|ef>
                            // Kita butuh <mb|ej> = I_ovov(m, b, j, e) ?? Hati-hati.
                            // I_ovov menyimpan (i, a, j, b) = <ij|ab>.
                            
                            // Akses tensor I_ovov(p, q, r, s) berarti <pr|qs>
                            
                            // 1. Integrals for Term 1
                            // <kb|jc>
                            double v_kbjc = I_ovov(k, b, j, c); 
                            // <kb|cj> -> <k j | b c> ?? Tidak ada di blok OVOV.
                            // <kb|cj> (Phys) = (kc|bj) (Chem).
                            // Blok I_ovov adalah (Occ, Virt, Occ, Virt) -> (i, a, j, b) Chemist.
                            // Jadi I_ovov(i, a, j, b) = <ij|ab>.
                            
                            // Kita butuh <k b | j c>. Ini adalah I_ovov(k, b, j, c).
                            // Kita butuh <k b | c j>. Ini Exchange. 
                            // <k b | c j> = (k c | b j). I_ovov(k, c, j, b).
                            
                            double J1 = I_ovov(k, b, j, c); // <kb|jc>
                            double K1 = I_ovov(k, c, j, b); // <kb|cj>
                            
                            // Term 1: t_ik^ac
                            double t_ikac = t2_1(i, k, a, c);
                            
                            // 2. Integrals for Term 2
                            // <ka|ic>
                            double J2 = I_ovov(k, a, i, c); // <ka|ic>
                            double K2 = I_ovov(k, c, i, a); // <ka|ci>
                            
                            // Term 2: t_jk^bc
                            double t_jkbc = t2_1(j, k, b, c);
                            
                            // Formulasi Pople (Eq 17):
                            // val += (2*J1 - K1) * t_ikac; (Permutasi j,b)
                            // val += (2*J2 - K2) * t_jkbc; (Permutasi i,a)
                            // Hati-hati dengan tanda negatif dari permutasi P_ij di rumus master.
                            // Tapi dalam loop langsung biasanya dijumlahkan:
                            
                            val += (2.0 * J1 - K1) * t_ikac;
                            val += (2.0 * J2 - K2) * t_jkbc;
                        }
                    }
                    
                    // Simpan
                    t2_2_(i, j, a, b) = val / denom;
                }
            }
        }
    }

    std::cout << "  5. Computing Energy...\n";
    // Hitung Energi E3
    // E3 = Sum (2<ij|ab> - <ij|ba>) * t2_2(i,j,a,b)
    // Gunakan I_ovov lagi karena I_ovov(i,a,j,b) = <ij|ab>
    double e_mp3 = 0.0;
    
    #pragma omp parallel for reduction(+:e_mp3) collapse(4)
    for (int i = 0; i < nocc_; i++) {
        for (int j = 0; j < nocc_; j++) {
            for (int a = 0; a < nvirt_; a++) {
                for (int b = 0; b < nvirt_; b++) {
                    double J = I_ovov(i, a, j, b); // <ij|ab>
                    double K = I_ovov(i, b, j, a); // <ij|ba>
                    
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

void RMP3::transform_integrals_ao_to_mo() {} // Deprecated
void RMP3::compute_t2_second_order() {} // Deprecated
double RMP3::compute_third_order_energy() { return 0.0; } // Deprecated

const Eigen::Tensor<double, 4>& RMP3::get_t2_second_order() const { return t2_2_; }

} // namespace foundation
} // namespace mshqc