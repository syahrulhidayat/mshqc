#pragma once

#include "mshqc/basis.h"
#include "mshqc/ints/integrals.h"
#include <Eigen/Dense>

namespace mshqc {
namespace integrals {

class DensityFittingERI {
public:
        DensityFittingERI(const BasisSet& primary_basis,
                      const BasisSet& aux_basis,
                      std::shared_ptr<IntegralEngine> integrals,
                      double cutoff = 1e-10);

    void compute();

    const Eigen::MatrixXd& get_B_mat() const { return B_mat_; }

private:
    const BasisSet* primary_basis_;
    const BasisSet* aux_basis_;
    std::shared_ptr<IntegralEngine> integrals_;

    int n_primary_;
    int n_aux_;
    bool is_computed_;
    Eigen::MatrixXd B_mat_;
    Eigen::MatrixXd compute_J_inv_half();
    double cutoff_;
};

}
}
