#ifndef MSHQC_MP2_CHOLESKY_OMP2_H
#define MSHQC_MP2_CHOLESKY_OMP2_H

#include <memory>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/integrals/cholesky_eri.h"
#include "mshqc/scf.h" 
#include "mshqc/mp2.h" // Pastikan OMP2Result ada di sini

namespace mshqc {

/**
 * @brief Configuration for Cholesky OMP2
 * Defined to match Python bindings requirements.
 */
struct CholeskyOMP2Config {
    int max_iterations = 50;
    double energy_threshold = 1e-6;
    double gradient_threshold = 1e-5;
    double cholesky_threshold = 1e-6;
    int print_level = 1;
};

/**
 * @class CholeskyOMP2
 * @brief Orbital-Optimized MP2 using Cholesky Decomposition (Reuse Strategy).
 */
class CholeskyOMP2 {
public:
    /**
     * @brief Constructor 1: Standard (Decompose internally)
     */
    CholeskyOMP2(const Molecule& mol,
                 const BasisSet& basis,
                 std::shared_ptr<IntegralEngine> integrals,
                 const SCFResult& scf_guess,
                 const CholeskyOMP2Config& config = CholeskyOMP2Config());

    /**
     * @brief Constructor 2: Reuse Vectors (Efficient)
     */
    CholeskyOMP2(const Molecule& mol,
                 const BasisSet& basis,
                 std::shared_ptr<IntegralEngine> integrals,
                 const SCFResult& scf_guess,
                 const CholeskyOMP2Config& config,
                 const integrals::CholeskyERI& cholesky_vectors);

    OMP2Result compute();

private:
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    SCFResult scf_;
    
    // Config Storage
    CholeskyOMP2Config config_;

    // Jika kita decompose sendiri, kita butuh storage sendiri
    std::unique_ptr<integrals::CholeskyERI> internal_cholesky_;
    
    // Reference ke cholesky yang aktif (bisa internal atau external)
    const integrals::CholeskyERI* cholesky_ptr_;

    int nbf_;
    int na_, nb_;
    int va_, vb_;

    // T2 Storage
    Eigen::Tensor<double, 4> t2_aa_;
    Eigen::Tensor<double, 4> t2_bb_;
    Eigen::Tensor<double, 4> t2_ab_;

    // OPDM Storage
    Eigen::MatrixXd G_oo_alpha_, G_vv_alpha_;
    Eigen::MatrixXd G_oo_beta_, G_vv_beta_;

    // --- Private Methods ---
    std::vector<Eigen::MatrixXd> transform_vectors(const Eigen::MatrixXd& C_occ, 
                                                   const Eigen::MatrixXd& C_virt,
                                                   int n_occ, int n_virt);

    void compute_t2_amplitudes();
    double compute_mp2_energy_from_t2();
    void build_opdm_alpha();
    void build_opdm_beta();

    Eigen::MatrixXd build_gfock_from_density(const Eigen::MatrixXd& P_total_alpha, 
                                             const Eigen::MatrixXd& P_total_beta,
                                             bool return_alpha);

    void rotate_orbitals_alpha(const Eigen::MatrixXd& w_ai, const Eigen::MatrixXd& F_mo);
    void rotate_orbitals_beta(const Eigen::MatrixXd& w_ai, const Eigen::MatrixXd& F_mo);
};

} // namespace mshqc

#endif // MSHQC_MP2_CHOLESKY_OMP2_H