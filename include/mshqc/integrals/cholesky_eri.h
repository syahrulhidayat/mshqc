/**
 * @file cholesky_eri.h
 * @brief Cholesky Decomposition for Electron Repulsion Integrals
 * 
 * THEORY REFERENCES:
 * - N. H. F. Beebe & J. Linderberg, Int. J. Quantum Chem. **12**, 683 (1977)
 * - H. Koch et al., J. Chem. Phys. **118**, 9481 (2003)
 * - F. Aquilante et al., J. Chem. Phys. **129**, 024113 (2008)
 * 
 * @author Muhamad Syahrul Hidayat (Agent 2)
 * @date 2025-11-16
 * @license MIT License
 */

#ifndef MSHQC_INTEGRALS_CHOLESKY_ERI_H
#define MSHQC_INTEGRALS_CHOLESKY_ERI_H

#include <vector>
#include <utility>
#include <memory>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace mshqc {
namespace integrals {

struct CholeskyDecompositionResult {
    int n_vectors = 0;
    int n_basis = 0;
    double threshold = 0.0;
    double max_error = 0.0;
    double rms_error = 0.0;
    double compression_ratio = 0.0;
    bool converged = false;
};

class CholeskyERI {
public:
    explicit CholeskyERI(double threshold = 1e-6);
    ~CholeskyERI() = default;
    
    // Main methods
    CholeskyDecompositionResult decompose(
        const Eigen::Tensor<double, 4>& eri_full
    );
    
    double reconstruct(int i, int j, int a, int b) const;
    Eigen::Tensor<double, 4> reconstruct_full() const;
    
    // Getters
    int n_vectors() const { return n_vectors_; }
    int n_basis() const { return n_basis_; }
    double threshold() const { return threshold_; }
    bool is_decomposed() const { return decomposed_; }
    double compression_ratio() const;
    size_t storage_bytes() const;
    double max_reconstruction_error() const { return max_error_; }
    double rms_reconstruction_error() const { return rms_error_; }
    
    // Utilities
    std::pair<double, double> validate_reconstruction(
        const Eigen::Tensor<double, 4>& eri_exact
    );
    void print_statistics(bool verbose = false) const;
    void reset();
    
    const std::vector<Eigen::VectorXd>& get_L_vectors() const {
        return L_vectors_;
    }
    // Clear existing vectors (for re-decomposition)
    void clear() {
        L_vectors_.clear();
    }

    // Add a vector manually (for on-the-fly algorithm)
    void add_vector(const Eigen::VectorXd& vec) {
        L_vectors_.push_back(vec);
    }

private:
    std::vector<Eigen::VectorXd> L_vectors_;
    int n_basis_ = 0;
    int n_vectors_ = 0;
    double threshold_ = 1e-6;
    bool decomposed_ = false;
    double max_error_ = 0.0;
    double rms_error_ = 0.0;
    
    inline int composite_index(int i, int j) const {
        return i * n_basis_ + j;
    }
    
    int find_pivot(const Eigen::VectorXd& D) const;
    Eigen::VectorXd compute_new_vector(
        const Eigen::Tensor<double, 4>& eri_full,
        const Eigen::VectorXd& D,
        int pivot_ij
    ) const;
};

} // namespace integrals
} // namespace mshqc

#endif // MSHQC_INTEGRALS_CHOLESKY_ERI_H