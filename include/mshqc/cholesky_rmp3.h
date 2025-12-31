/**
 * @file cholesky_rmp3.h
 * @brief Cholesky-decomposed Restricted MP3 with Vector Reuse
 * * Implementation of RMP3 energy correction using Cholesky Decomposition vectors.
 * Designed to be run strictly AFTER Cholesky-RMP2 to reuse the decomposed vectors,
 * avoiding the expensive O(N^4) ERI construction and decomposition steps.
 * * ADVANTAGES:
 * 1. Memory Efficiency: Stores O(N^2 M) vectors instead of O(N^4) tensors.
 * 2. Computational Speed: Reuses vectors from MP2 step (no re-decomposition).
 * 3. On-the-fly Assembly: Constructs (pq|rs) integrals only when needed in loops.
 * * @author Muhamad Sahrul Hidayat
 * @date 2025-12-31
 * @license MIT
 */

#ifndef MSHQC_FOUNDATION_CHOLESKY_RMP3_H
#define MSHQC_FOUNDATION_CHOLESKY_RMP3_H

#include "mshqc/foundation/rmp3.h"         // Untuk struct RMP3Result
#include "mshqc/cholesky_rmp2.h" // 
#include "mshqc/scf.h"
#include "mshqc/basis.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace mshqc {
namespace foundation {

/**
 * @brief Cholesky-Decomposed RMP3 Solver
 * * Menghitung koreksi energi MP3 menggunakan vektor Cholesky yang sudah ada.
 * * Alur Logika:
 * 1. Menerima vector AO (L_uv^K) dari hasil Cholesky-MP2.
 * 2. Mentransformasi L_uv^K menjadi L_pq^K (MO basis) untuk ruang:
 * - Virtual-Virtual (VV)
 * - Occupied-Occupied (OO)
 * - Occupied-Virtual (OV)
 * 3. Menghitung amplitudo T2^(2) menggunakan rekonstruksi integral on-the-fly:
 * (pq|rs) ≈ Σ_K L_pq^K * L_rs^K
 */
class CholeskyRMP3 {
public:
    /**
     * @brief Constructor
     * * @param rhf_result Hasil perhitungan SCF/RHF.
     * @param crmp2_result Hasil perhitungan Cholesky-MP2 (harus mengandung vektor Cholesky).
     * @param basis Basis set yang digunakan.
     */
    CholeskyRMP3(const SCFResult& rhf_result,
                 const CholeskyRMP2Result& crmp2_result,
                 const BasisSet& basis);

    /**
     * @brief Menjalankan perhitungan Cholesky-MP3
     * * @return RMP3Result Struct berisi energi MP2, MP3, Total, dan amplitudo T2.
     */
    RMP3Result compute();

private:
    // References to input data
    const SCFResult& rhf_;
    const CholeskyRMP2Result& crmp2_;
    const BasisSet& basis_;

    // Dimensions
    int nbf_;            // Basis functions
    int nocc_;           // Occupied orbitals
    int nvirt_;          // Virtual orbitals
    int n_chol_vectors_; // Jumlah vektor Cholesky (Rank M)

    // Storage for second-order amplitudes
    Eigen::Tensor<double, 4> t2_2_;

    /**
     * @brief Helper: Transformasi vektor Cholesky dari basis AO ke sub-ruang MO tertentu.
     * * L_pq^K = Σ_uv C_up * C_vq * L_uv^K
     * * @param ao_vectors Vector Cholesky dalam basis AO (dari MP2).
     * @param C_left Matriks koefisien untuk indeks pertama (p).
     * @param C_right Matriks koefisien untuk indeks kedua (q).
     * @param dim1 Dimensi ruang pertama (misal: nocc atau nvirt).
     * @param dim2 Dimensi ruang kedua.
     * @return Eigen::MatrixXd Matriks (dim1*dim2 x M) berisi vektor MO.
     */
    Eigen::MatrixXd transform_subspace(
        const std::vector<Eigen::VectorXd>& ao_vectors,
        const Eigen::MatrixXd& C_left,
        const Eigen::MatrixXd& C_right,
        int dim1, int dim2
    );
};

} // namespace foundation
} // namespace mshqc

#endif // MSHQC_FOUNDATION_CHOLESKY_RMP3_H