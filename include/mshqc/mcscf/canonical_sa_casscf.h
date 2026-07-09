#ifndef MSHQC_MCSCF_CANONICAL_SA_CASSCF_H
#define MSHQC_MCSCF_CANONICAL_SA_CASSCF_H

#include "mshqc/mcscf/active_space.h"
#include "mshqc/mcscf/cholesky_sa_casscf.h" 

#include "mshqc/core/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/ints/integrals.h"
#include "mshqc/scf/scf.h" 
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace mcscf {

/**
 * @class CanonicalSACASSCF
 * @brief State-Averaged CASSCF using Exact Integrals (High Performance)
 * @details Optimized with Stepwise Transformation and 8-Fold Symmetry Fock Build.
 */
class CanonicalSACASSCF {
public:
    CanonicalSACASSCF(const Molecule& mol,
                      const BasisSet& basis,
                      std::shared_ptr<IntegralEngine> integrals,
                      const ActiveSpace& active_space,
                      const SACASConfig& config);

    SACASResult compute(const Eigen::MatrixXd& initial_orbitals);
    SACASResult compute(const SCFResult& initial_guess);

private:
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    ActiveSpace active_space_;
    SACASConfig config_;

    

    Eigen::MatrixXd schwarz_;
    double max_schwarz_ = 0.0;
    inline size_t idx2(size_t i, size_t j) const {
    return (i >= j) ? (i * (i + 1) / 2 + j) : (j * (j + 1) / 2 + i);
}

    

    

    

    Eigen::Tensor<double, 4> transform_integrals_stepwise(const Eigen::MatrixXd& C_mo);

    

    

    Eigen::MatrixXd compute_generalized_fock_optimized(
        const Eigen::MatrixXd& P_avg_mo, 
        const Eigen::MatrixXd& C_mo
    );

    Eigen::VectorXd compute_orbital_gradient(
        const Eigen::MatrixXd& F_gen, 
        const Eigen::MatrixXd& C_mo
    ) const;
    
    Eigen::MatrixXd apply_rotation(const Eigen::MatrixXd& C, const Eigen::VectorXd& kappa) const;
};

} 

} 


#endif