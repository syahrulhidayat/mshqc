#include "mshqc/ci/hamiltonian_sparse.h"
#include "mshqc/ci/ci_utils.h"
#include <unordered_map>

namespace mshqc {
namespace ci {

void build_hamiltonian_coo(const std::vector<Determinant>& dets,
                           const CIIntegrals& ints,
                           SparseCOO& H,
                           double eps_value) {
    const int n = static_cast<int>(dets.size());
    H.resize(n, n);
    // Heuristic reserve: diagonal + 8 off-diagonals per row (adjust later via profiling)
    H.reserve(static_cast<std::size_t>(n) * 9);

    // Diagonal terms
    for (int i = 0; i < n; ++i) {
        const double vii = diagonal_element(dets[i], ints);
        if (std::abs(vii) >= eps_value) {
            H.add(i, i, vii);
        }
    }

    // Off-diagonals (upper triangle) with early excitation screening
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            auto exc = find_excitation(dets[i], dets[j]);
            if (exc.level > 2) continue; // Slater-Condon zero

            const double vij = hamiltonian_element(dets[i], dets[j], ints);
            if (std::abs(vij) < eps_value) continue;

            // Symmetric insert
            H.add(i, j, vij);
            H.add(j, i, vij);
        }
    }
}

void build_hamiltonian_csr(const std::vector<Determinant>& dets,
                           const CIIntegrals& ints,
                           SparseCSR& Hcsr,
                           double eps_value) {
    SparseCOO coo;
    build_hamiltonian_coo(dets, ints, coo, eps_value);
    std::vector<SparseCOO::Index> rp;
    std::vector<SparseCOO::Index> ci;
    std::vector<SparseCOO::Scalar> vv;
    coo.finalize_to_csr(rp, ci, vv, true);
    Hcsr.set(coo.n_rows(), coo.n_cols(),
             std::vector<SparseCSR::Index>(rp.begin(), rp.end()),
             std::vector<SparseCSR::Index>(ci.begin(), ci.end()),
             std::vector<SparseCSR::Scalar>(vv.begin(), vv.end()));
}

void build_hamiltonian_coo_hash(const std::vector<Determinant>& dets,
                                const CIIntegrals& ints,
                                const std::unordered_map<Determinant, int>& det_map,
                                SparseCOO& H,
                                double eps_value) {
    // NOTE: For full Hamiltonian assembly, hash doesn't provide significant benefit
    // since we still need O(N²) determinant pair evaluation.
    // 
    // Hash is most useful for:
    //   - On-the-fly sigma-vector (generate connected dets, lookup if exists)
    //   - Selected CI where only subset of dets are included
    // 
    // For now, just use the standard approach.
    // Future optimization: implement on-the-fly sigma for large FCI.
    
    (void)det_map;  // Suppress unused parameter warning
    build_hamiltonian_coo(dets, ints, H, eps_value);
}

void build_hamiltonian_csr_hash(const std::vector<Determinant>& dets,
                                const CIIntegrals& ints,
                                const std::unordered_map<Determinant, int>& det_map,
                                SparseCSR& Hcsr,
                                double eps_value) {
    SparseCOO coo;
    build_hamiltonian_coo_hash(dets, ints, det_map, coo, eps_value);
    std::vector<SparseCOO::Index> rp;
    std::vector<SparseCOO::Index> ci;
    std::vector<SparseCOO::Scalar> vv;
    coo.finalize_to_csr(rp, ci, vv, true);
    Hcsr.set(coo.n_rows(), coo.n_cols(),
             std::vector<SparseCSR::Index>(rp.begin(), rp.end()),
             std::vector<SparseCSR::Index>(ci.begin(), ci.end()),
             std::vector<SparseCSR::Scalar>(vv.begin(), vv.end()));
}

Eigen::VectorXd sigma_vector_sparse(const SparseCSR& Hcsr,
                                    const Eigen::VectorXd& c) {
    std::vector<double> x(static_cast<size_t>(c.size()));
    for (int i = 0; i < c.size(); ++i) x[static_cast<size_t>(i)] = c(i);
    std::vector<double> y;
    Hcsr.matvec(x, y);
    Eigen::VectorXd sigma(Hcsr.n_rows());
    for (int i = 0; i < sigma.size(); ++i) sigma(i) = y[static_cast<size_t>(i)];
    return sigma;
}

} // namespace ci
} // namespace mshqc
