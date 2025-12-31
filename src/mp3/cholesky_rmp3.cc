/**
 * @file cholesky_rmp3.cc
 * @brief Implementation of Cholesky-decomposed Restricted MP3
 * Reuses Cholesky vectors to avoid full ERI construction.
 * @author Muhamad Syahrul Hidayat
 * @date 2025-12-31
 */

#include "mshqc/cholesky_rmp3.h" // Pastikan path include ini benar
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

namespace mshqc {
namespace foundation {

// Constructor: Menerima hasil CholeskyRMP2 untuk menggunakan kembali vektor
CholeskyRMP3::CholeskyRMP3(const SCFResult& rhf_result,
                           const CholeskyRMP2Result& crmp2_result,
                           const BasisSet& basis)
    : rhf_(rhf_result), 
      crmp2_(crmp2_result), 
      basis_(basis) {
    
    nbf_ = basis.n_basis_functions();
    nocc_ = crmp2_.n_occ;
    nvirt_ = crmp2_.n_virt;
    n_chol_vectors_ = crmp2_.n_chol_vectors;
}

// Helper untuk mengubah vektor AO (L_uv^K) menjadi vektor MO (L_pq^K)
Eigen::MatrixXd CholeskyRMP3::transform_subspace(
    const std::vector<Eigen::VectorXd>& ao_vectors,
    const Eigen::MatrixXd& C_left,  // Koefisien orbital pertama 
    const Eigen::MatrixXd& C_right, // Koefisien orbital kedua 
    int dim1, int dim2) {

    // 1. Unpack AO vectors ke dalam Matrix (N^2 x M)
    Eigen::MatrixXd L_ao(nbf_ * nbf_, n_chol_vectors_);
    
    #pragma omp parallel for
    for (int k = 0; k < n_chol_vectors_; k++) {
        L_ao.col(k) = ao_vectors[k];
    }

    // 2. Transformasi: L_pq^K = sum_uv C_up * C_vq * L_uv^K
    Eigen::MatrixXd L_mo(dim1 * dim2, n_chol_vectors_);
    L_mo.setZero();

    // Loop over Cholesky vectors (K)
    #pragma omp parallel for
    for (int k = 0; k < n_chol_vectors_; k++) {
        // Rekonstruksi L_ao^K menjadi matriks NxN
        Eigen::MatrixXd L_K_mat(nbf_, nbf_);
        for (int i = 0; i < nbf_; i++) {
            for (int j = 0; j < nbf_; j++) {
                // Asumsi penyimpanan row-major (i * nbf + j)
                L_K_mat(i, j) = ao_vectors[k](i * nbf_ + j);
            }
        }

        // Transformasi paruh pertama: M = C_left^T * L_ao
        Eigen::MatrixXd Half = C_left.transpose() * L_K_mat; 

        // Transformasi paruh kedua: Final = Half * C_right
        Eigen::MatrixXd Final = Half * C_right; // (dim1 x dim2)

        // Flatten kembali ke kolom output
        for (int p = 0; p < dim1; p++) {
            for (int q = 0; q < dim2; q++) {
                L_mo(p * dim2 + q, k) = Final(p, q);
            }
        }
    }

    return L_mo;
}

RMP3Result CholeskyRMP3::compute() {
    std::cout << "\n==============================================\n";
    std::cout << "  Cholesky-Restricted MP3 (Reuse Vectors)\n";
    std::cout << "==============================================\n";
    
    // 1. Persiapan Data
    const auto& C_occ = rhf_.C_alpha.leftCols(nocc_);
    const auto& C_virt = rhf_.C_alpha.rightCols(nvirt_);
    const auto& eps = rhf_.orbital_energies_alpha;
    const auto& t2_1 = crmp2_.t2; // Amplitudo dari MP2
    const auto& ao_vecs = crmp2_.chol_vectors; // REUSE: Vektor AO dari RMP2

    std::cout << "  [Step 1] Transforming Cholesky Vectors to MO Basis...\n";
    std::cout << "           Vectors: " << n_chol_vectors_ << "\n";

    // 2. Transformasi Vektor Cholesky ke Basis MO
    Eigen::MatrixXd L_VV = transform_subspace(ao_vecs, C_virt, C_virt, nvirt_, nvirt_);
    Eigen::MatrixXd L_OO = transform_subspace(ao_vecs, C_occ, C_occ, nocc_, nocc_);
    Eigen::MatrixXd L_OV = transform_subspace(ao_vecs, C_occ, C_virt, nocc_, nvirt_);

    std::cout << "           Done transforming vectors.\n";

    // 3. Inisialisasi T2 Order-2
    t2_2_ = Eigen::Tensor<double, 4>(nocc_, nocc_, nvirt_, nvirt_);
    t2_2_.setZero();

    std::cout << "  [Step 2] Computing T2(2) Amplitudes (On-the-fly reconstruction)...\n";
    
    #pragma omp parallel for collapse(2) schedule(dynamic)
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
                            double dir = L_VV.row(a * nvirt_ + c).dot(L_VV.row(b * nvirt_ + d));
                            double ex = L_VV.row(a * nvirt_ + d).dot(L_VV.row(b * nvirt_ + c));
                            val += 0.5 * (dir - ex) * t2_1(i, j, c, d);
                        }
                    }

                    // B. Hole-Hole (OOOO)
                    for (int k = 0; k < nocc_; k++) {
                        for (int l = 0; l < nocc_; l++) {
                            double dir = L_OO.row(k * nocc_ + i).dot(L_OO.row(l * nocc_ + j));
                            double ex = L_OO.row(k * nocc_ + j).dot(L_OO.row(l * nocc_ + i));
                            val += 0.5 * (dir - ex) * t2_1(k, l, a, b);
                        }
                    }

                    // C. Particle-Hole / Ring (OVOV)
                    for (int k = 0; k < nocc_; k++) {
                        for (int c = 0; c < nvirt_; c++) {
                            // L_OV row index: (occ_idx * nvirt + virt_idx)
                            auto L_jb = L_OV.row(j * nvirt_ + b);
                            auto L_kc = L_OV.row(k * nvirt_ + c);
                            auto L_kb = L_OV.row(k * nvirt_ + b);
                            auto L_jc = L_OV.row(j * nvirt_ + c);
                            
                            auto L_ia = L_OV.row(i * nvirt_ + a);
                            auto L_ka = L_OV.row(k * nvirt_ + a);
                            auto L_ic = L_OV.row(i * nvirt_ + c);

                            double int_jbkc = L_jb.dot(L_kc);
                            double int_kbjc = L_kb.dot(L_jc);
                            double int_iakc = L_ia.dot(L_kc);
                            double int_kaic = L_ka.dot(L_ic);
                            
                            double t_ikac = t2_1(i, k, a, c);
                            double t_jkbc = t2_1(j, k, b, c);
                            
                            val -= (2.0 * int_jbkc - int_kbjc) * t_ikac;
                            val -= (2.0 * int_iakc - int_kaic) * t_jkbc;
                        }
                    }
                    
                    t2_2_(i, j, a, b) = val / denom;
                }
            }
        }
    }

    std::cout << "  [Step 3] Computing MP3 Energy...\n";
    double e_mp3 = 0.0;
    
    #pragma omp parallel for reduction(+:e_mp3) collapse(4)
    for (int i = 0; i < nocc_; i++) {
        for (int j = 0; j < nocc_; j++) {
            for (int a = 0; a < nvirt_; a++) {
                for (int b = 0; b < nvirt_; b++) {
                    double J = L_OV.row(i * nvirt_ + a).dot(L_OV.row(j * nvirt_ + b));
                    double K = L_OV.row(i * nvirt_ + b).dot(L_OV.row(j * nvirt_ + a));
                    
                    e_mp3 += (2.0 * J - K) * t2_2_(i, j, a, b);
                }
            }
        }
    }

    // === FIX HERE ===
    RMP3Result result;
    
    // Gunakan crmp2_.e_rhf karena struct SCFResult mungkin tidak memiliki member 'energy'
    // dan kita sudah memastikan CholeskyRMP2Result memiliki e_rhf.
    result.e_rhf = crmp2_.e_rhf; 
    
    result.e_mp2 = crmp2_.e_corr;
    result.e_mp3 = e_mp3;
    result.e_corr_total = result.e_mp2 + e_mp3;
    result.e_total = result.e_rhf + result.e_corr_total;
    result.n_occ = nocc_;
    result.n_virt = nvirt_;
    result.t2_1 = t2_1;
    result.t2_2 = t2_2_;
    
    std::cout << std::setprecision(8);
    std::cout << "----------------------------------------------\n";
    std::cout << "ANALYSIS Cholesky-MP3\n";
    std::cout << "----------------------------------------------\n";
    std::cout << "[INFO] MP2 Correlation : " << result.e_mp2 << " Ha\n";
    std::cout << "[INFO] MP3 Correction  : " << e_mp3 << " Ha\n";
    std::cout << "[INFO] Total Correlation: " << result.e_corr_total << " Ha\n";
    std::cout << "[INFO] Total Energy    : " << result.e_total << " Ha\n";
    std::cout << "==============================================\n";
    
    return result;
}

} // namespace foundation
} // namespace mshqc