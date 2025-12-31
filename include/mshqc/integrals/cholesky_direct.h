// ============================================================================
// FILE: cholesky_direct.h
// ============================================================================
#ifndef MSHQC_INTEGRALS_CHOLESKY_DIRECT_H
#define MSHQC_INTEGRALS_CHOLESKY_DIRECT_H

#include <libint2.hpp>
#include <Eigen/Dense>
#include <vector>

namespace mshqc {
namespace integrals {

class DirectCholesky {
public:
    // Constructor
    DirectCholesky(const std::vector<libint2::Shell>& shells, double threshold = 1e-6);

    // Main computation
    void compute();

    // Getters
    const std::vector<Eigen::VectorXd>& get_L() const { return L_vectors_; }
    int rank() const { return L_vectors_.size(); }
    int n_basis() const { return n_basis_; }

private:
    const std::vector<libint2::Shell>& shells_;
    std::vector<std::vector<size_t>> shell2bf_;
    double threshold_;
    int n_basis_;
    
    std::vector<Eigen::VectorXd> L_vectors_;
    Eigen::VectorXd diag_;

    // Helper methods
    double compute_eri(int mu, int nu, int lam, int sig, libint2::Engine& engine);
    Eigen::VectorXd compute_column(int pivot_index, int max_l, int max_nprim);
};

} // namespace integrals
} // namespace mshqc

#endif