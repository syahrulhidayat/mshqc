/**
 * @file canonical_sa_caspt2.h
 * @brief Canonical State-Averaged CASPT2 (Full 4-index Transformation)
 * @details
 * Performs standard O(N^5) transformation of AO ERIs to MO basis.
 * Computes PT2 energy corrections using exact MO integrals.
 * Reference implementation for benchmarking Cholesky approximations.
 * * @author Muhamad Sahrul Hidayat
 */

#ifndef MSHQC_CANONICAL_SA_CASPT2_H
#define MSHQC_CANONICAL_SA_CASPT2_H

#include "mshqc/mcscf/canonical_sa_casscf.h"
#include "mshqc/mcscf/cholesky_sa_caspt2.h" 

#include "mshqc/ints/integrals.h"
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#ifdef I
#undef I
#endif
namespace mshqc {
namespace mcscf {

class CanonicalSACASPT2 {
public:
    CanonicalSACASPT2(const SACASResult& result,
                      std::shared_ptr<IntegralEngine> integrals,
                      const BasisSet& basis,
                      const ActiveSpace& active_space,
                      const CASPT2Config& config);

    CASPT2Result compute();

private:
    SACASResult cas_res_;
    std::shared_ptr<IntegralEngine> integrals_;
    const BasisSet& basis_;
    ActiveSpace active_space_;
    CASPT2Config config_;
    
    int n_inact_, n_act_, n_virt_, nbasis_;

    

    

    

    Eigen::Tensor<double, 4> transform_integrals_to_mo() const;

    

    

    double compute_state_pt2(int state_idx, 
                             const Eigen::Tensor<double, 4>& mo_eri,
                             const Eigen::VectorXd& eps,
                             PT2Amplitudes* amps = nullptr); 

};

} 

} 


#endif