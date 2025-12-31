/**
 * @file mshqc/cholesky_rmp2.h
 * @brief Cholesky-Decomposed Restricted MP2 (RMP2)
 * @details Efficient RMP2 implementation reusing Cholesky vectors.
 * Generates amplitudes compatible with standard RMP3.
 * @author Muhamad Syahrul Hidayat
 * @date 2025-01-11
 */

#ifndef MSHQC_CHOLESKY_RMP2_H
#define MSHQC_CHOLESKY_RMP2_H

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/foundation/rmp2.h" // Untuk foundation::RMP2Result
#include "mshqc/integrals/cholesky_eri.h"

namespace mshqc {
namespace foundation {

/**
 * @brief Extended result struct for Cholesky-MP2
 * Stores Cholesky vectors specifically for passing to MP3
 */
struct CholeskyRMP2Result {
    // Basic Energy Results
    double e_rhf;
    double e_corr;      // MP2 Correlation energy
    double e_total;     // Total Energy
    
    // Amplitudes (Required for MP3)
    Eigen::Tensor<double, 4> t2; 

    // Cholesky Data (CRITICAL for Reuse in MP3)
    // Menyimpan vektor L dalam basis AO agar MP3 tidak perlu dekomposisi ulang
    std::vector<Eigen::VectorXd> chol_vectors; 
    int n_chol_vectors;
    
    // Dimensions
    int n_occ;
    int n_virt;
};

} // namespace foundation

/**
 * @brief Configuration for Cholesky-RMP2
 */
struct CholeskyRMP2Config {
    double cholesky_threshold = 1e-6;
    int print_level = 1;
};

/**
 * @class CholeskyRMP2
 * @brief Efficient RMP2 using Cholesky Decomposition
 */
class CholeskyRMP2 {
public:
    /**
     * @brief Constructor 1: Standard (Decompose Internally)
     */
    CholeskyRMP2(const Molecule& mol,
                 const BasisSet& basis,
                 std::shared_ptr<IntegralEngine> integrals,
                 const SCFResult& rhf_result,
                 const CholeskyRMP2Config& config = CholeskyRMP2Config());

    /**
     * @brief Constructor 2: Reuse Vectors (High Efficiency)
     */
    CholeskyRMP2(const Molecule& mol,
                 const BasisSet& basis,
                 std::shared_ptr<IntegralEngine> integrals,
                 const SCFResult& rhf_result,
                 const CholeskyRMP2Config& config,
                 const integrals::CholeskyERI& existing_cholesky);

    /**
     * @brief Compute RMP2 energy and amplitudes
     * @return CholeskyRMP2Result (containing vectors for MP3 reuse)
     */
    foundation::CholeskyRMP2Result compute();

private:
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    SCFResult rhf_;
    CholeskyRMP2Config config_;

    // Cholesky Handling
    const integrals::CholeskyERI* cholesky_ptr_;
    std::unique_ptr<integrals::CholeskyERI> internal_cholesky_;

    // Dimensions
    int nbf_;
    int nocc_;
    int nvirt_;

    // Amplitudes
    Eigen::Tensor<double, 4> t2_;

    // Helper: Transform AO vectors to MO (ia) block
    std::vector<Eigen::MatrixXd> transform_vectors();
};

} // namespace mshqc

#endif // MSHQC_CHOLESKY_RMP2_H