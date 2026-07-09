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
#ifdef I
#undef I
#endif

namespace mshqc {
namespace ci {













void build_hamiltonian_coo(const std::vector<Determinant>& dets,
                           const CIIntegrals& ints,
                           SparseCOO& H,
                           double eps_value = 0.0);










void build_hamiltonian_coo_hash(const std::vector<Determinant>& dets,
                                const CIIntegrals& ints,
                                const std::unordered_map<Determinant, int>& det_map,
                                SparseCOO& H,
                                double eps_value = 0.0);




void build_hamiltonian_csr(const std::vector<Determinant>& dets,
                           const CIIntegrals& ints,
                           SparseCSR& Hcsr,
                           double eps_value = 0.0);




void build_hamiltonian_csr_hash(const std::vector<Determinant>& dets,
                                const CIIntegrals& ints,
                                const std::unordered_map<Determinant, int>& det_map,
                                SparseCSR& Hcsr,
                                double eps_value = 0.0);




Eigen::VectorXd sigma_vector_sparse(const SparseCSR& Hcsr,
                                    const Eigen::VectorXd& c);

















































Eigen::VectorXd sigma_vector_onthefly(const std::vector<Determinant>& dets,
                                      const Eigen::VectorXd& c,
                                      const CIIntegrals& ints,
                                      const std::unordered_map<Determinant, int>& det_map,
                                      int n_orb);

} 


} 



#endif 


