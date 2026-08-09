 // ==============================================================================
 // Copyright (c) 2026 Syahrul and mshqc contributors
 //
 // Licensed under the Apache License, Version 2.0 (the "License");
 // you may not use this file except in compliance with the License.
 // You may obtain a copy of the License at
 //
 //     http://www.apache.org/licenses/LICENSE-2.0
 //
 // Unless required by applicable law or agreed to in writing, software
 // distributed under the License is distributed on an "AS IS" BASIS,
 // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 // See the License for the specific language governing permissions and
 // limitations under the License.
 // ==============================================================================

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
