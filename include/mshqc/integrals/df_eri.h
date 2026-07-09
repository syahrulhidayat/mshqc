/**
 * @file include/mshqc/integrals/df_eri.h
 * @brief Density Fitting (Resolution of the Identity) Tensor Generator
 */

#pragma once

#include "mshqc/basis.h"
#include "mshqc/ints/integrals.h"
#include <Eigen/Dense>

namespace mshqc {
namespace integrals {

class DensityFittingERI {
public:
    // Constructor menerima Basis Set Primer, Basis Set Aux, dan Engine Libcint
    DensityFittingERI(const BasisSet& primary_basis, 
                      const BasisSet& aux_basis, 
                      std::shared_ptr<IntegralEngine> integrals,
                      double cutoff = 1e-10);

    // Fungsi utama untuk membentuk tensor B_{ij}^P
    void compute();

    // Mengambil tensor hasil B_mat (Dimensi: n_primary_pairs x n_aux)
    const Eigen::MatrixXd& get_B_mat() const { return B_mat_; }

private:
    const BasisSet* primary_basis_;
    const BasisSet* aux_basis_;
    std::shared_ptr<IntegralEngine> integrals_;

    int n_primary_;
    int n_aux_;
    bool is_computed_;

    // Tensor akhir B_{ij}^P, disimpan sebagai matriks 2D untuk efisiensi DGEMM
    // Baris = (i, j) dari primary basis, Kolom = P dari aux basis
    Eigen::MatrixXd B_mat_; 

    // Fungsi internal untuk menghitung J^{-1/2}
    Eigen::MatrixXd compute_J_inv_half();
    double cutoff_;
};

} // namespace integrals
} // namespace mshqc