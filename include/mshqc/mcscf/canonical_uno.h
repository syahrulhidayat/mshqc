#ifndef MSHQC_MCSCF_CANONICAL_UNO_H
#define MSHQC_MCSCF_CANONICAL_UNO_H

#include "mshqc/scf.h"
#include "mshqc/integrals.h"
#include "mshqc/mcscf/uno_result.h" 


#include <memory>
#include <vector>
#include <Eigen/Dense>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace mcscf {




/**
 * @class CanonicalUNO
 * @brief UNO Generator using Canonical Orthogonalization (S^-1/2)
 * Robust for large/diffuse basis sets where S is near-singular.
 */
class CanonicalUNO {
public:
    CanonicalUNO(const SCFResult& uhf_result,
                 std::shared_ptr<IntegralEngine> integrals,
                 int n_basis);

    UNOResult compute();
    void print_report(double threshold = 0.02) const;
    void save_orbitals(const std::string& filename) const;

private:
    SCFResult uhf_res_;
    std::shared_ptr<IntegralEngine> integrals_;
    int nbasis_;
    UNOResult result_;
    bool computed_ = false;

    

    double calculate_entropy(const Eigen::VectorXd& n) const;
    void analyze_active_space(double threshold);
};

} 

} 


#endif 
