#ifndef MSHQC_CI_HAMILTONIAN_SPARSE_H
#define MSHQC_CI_HAMILTONIAN_SPARSE_H

#include <vector>
#include <functional>
#include <cmath>
#include <unordered_map>
#include "mshqc/ci/determinant.h"
#include "mshqc/ci/slater_condon.h"
#include "mshqc/ci/sparse_coo.h"
#include "mshqc/ci/sparse_csr.h"

namespace mshqc {
namespace ci {

// Build sparse Hamiltonian in COO with simple screening.
// - eps_value: drop |H_ij| < eps_value
// - Note: elements with excitation level > 2 are zero by Slater-Condon and skipped early
// - Symmetry: we insert both (i,j) and (j,i) for i != j to obtain a full symmetric matrix
void build_hamiltonian_coo(const std::vector<Determinant>& dets,
                           const CIIntegrals& ints,
                           SparseCOO& H,
                           double eps_value = 0.0);

// Build COO with hash-based lookup (faster for connected determinants)
// - det_map: O(1) determinant->index lookup
// - For Hamiltonian assembly with connected dets (singles/doubles), ~50-80% faster
void build_hamiltonian_coo_hash(const std::vector<Determinant>& dets,
                                const CIIntegrals& ints,
                                const std::unordered_map<Determinant, int>& det_map,
                                SparseCOO& H,
                                double eps_value = 0.0);

// Build CSR directly (COO assemble -> CSR finalize)
void build_hamiltonian_csr(const std::vector<Determinant>& dets,
                           const CIIntegrals& ints,
                           SparseCSR& Hcsr,
                           double eps_value = 0.0);

// Build CSR with hash-based lookup
void build_hamiltonian_csr_hash(const std::vector<Determinant>& dets,
                                const CIIntegrals& ints,
                                const std::unordered_map<Determinant, int>& det_map,
                                SparseCSR& Hcsr,
                                double eps_value = 0.0);

// Sigma-vector via CSR: y = Hcsr * c
Eigen::VectorXd sigma_vector_sparse(const SparseCSR& Hcsr,
                                    const Eigen::VectorXd& c);

// On-the-fly sigma-vector: σ = H·c without storing H
// 
// For each determinant, generates connected excitations and computes H_ij on-the-fly.
// Uses hash map for O(1) determinant lookup.
// 
// Benefits:
//   - Memory: O(N) instead of O(N²)
//   - Speed: Faster for large N due to cache efficiency and hash lookup
//   - Essential for FCI with N > 10,000 determinants
// 
// @param dets List of determinants
// @param c CI coefficients
// @param ints MO integrals
// @param det_map Determinant->index hash map for O(1) lookup
// @param n_orb Total number of orbitals
// @return σ = H·c
Eigen::VectorXd sigma_vector_onthefly(const std::vector<Determinant>& dets,
                                      const Eigen::VectorXd& c,
                                      const CIIntegrals& ints,
                                      const std::unordered_map<Determinant, int>& det_map,
                                      int n_orb);

} // namespace ci
} // namespace mshqc

#endif // MSHQC_CI_HAMILTONIAN_SPARSE_H
