/**
 * @file src/ci/ci_utils.cc
 * @brief Utilities for CI calculations (Hamiltonian build, etc.)
 */

#include "mshqc/ci/ci_utils.h"
#include "mshqc/ci/slater_condon.h"
#include <iostream>

namespace mshqc {
namespace ci {
/*
Eigen::MatrixXd build_hamiltonian(const std::vector<Determinant>& dets, 
                                  const CIIntegrals& integrals) {
    int n_det = dets.size();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_det, n_det);
    
    // Gunakan SlaterCondon rules untuk menghitung elemen matriks <D_i|H|D_j>
    // Ini bisa diparallelkan dengan OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_det; ++i) {
        for (int j = i; j < n_det; ++j) {
            double elem = evaluate_matrix_element(dets[i], dets[j], integrals);
            H(i, j) = elem;
            H(j, i) = elem; // Hermitian
        }
    }
    
    return H;
}
*/
} // namespace ci
} // namespace mshqc