/**
 * @file canonical_sa_caspt3.h
 * @brief Canonical State-Averaged CASPT3 (Reference Implementation)
 * @details
 * Performs O(N^6) exact computation of CASPT3 energy using full MO integrals.
 * Used to benchmark the Cholesky-CASPT3 approximation.
 * LOGIC MATCHED with CholeskySACASPT3 (Factors 0.5 for Ladder, Sign Corrected).
 * @author Muhamad Sahrul Hidayat
 */

#ifndef MSHQC_CANONICAL_SA_CASPT3_H
#define MSHQC_CANONICAL_SA_CASPT3_H

#include "mshqc/mcscf/canonical_sa_caspt2.h"
#include "mshqc/mcscf/cholesky_sa_caspt3.h" // Reuse Config & Result structs
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#ifdef I
#undef I
#endif
namespace mshqc {
namespace mcscf {

class CanonicalSACASPT3 {
public:
    CanonicalSACASPT3(const SACASResult& result,
                      std::shared_ptr<IntegralEngine> integrals,
                      const BasisSet& basis,
                      const ActiveSpace& active_space,
                      const CASPT3Config& config);

    CASPT3Result compute();

private:
    SACASResult cas_res_;
    std::shared_ptr<IntegralEngine> integrals_;
    const BasisSet& basis_;
    ActiveSpace active_space_;
    CASPT3Config config_;
    
    int n_inact_, n_act_, n_virt_, nbasis_;

    // --- Helpers ---
    // Transform full AO -> MO integrals (pq|rs)
    Eigen::Tensor<double, 4> transform_integrals_to_mo() const;

    // The O(N^6) Contractor (Exact Integrals)
    double compute_state_pt3(int state_idx,
                             const PT2Amplitudes& amps,
                             const Eigen::Tensor<double, 4>& mo_eri,
                             const Eigen::VectorXd& eps) const;
};

} // namespace mcscf
} // namespace mshqc

#endif