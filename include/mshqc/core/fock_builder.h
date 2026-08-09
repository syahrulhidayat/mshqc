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

#ifndef MSHQC_CORE_FOCK_BUILDER_H
#define MSHQC_CORE_FOCK_BUILDER_H

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "mshqc/ints/integrals.h"

#include "mshqc/symmetry/petite_list.h"
#ifdef I
#undef I
#endif

namespace mshqc {

class FockBuilder {
public:

    FockBuilder(std::shared_ptr<IntegralEngine> integrals,
                const BasisSet& basis,
                const Eigen::MatrixXd& H_core,
                const Eigen::MatrixXd& schwarz);

    void reset();
    void set_petite_list(const PetiteList* list);

    void compute(const Eigen::MatrixXd& P_alpha,
                 const Eigen::MatrixXd& P_beta,
                 Eigen::MatrixXd& F_alpha,
                 Eigen::MatrixXd& F_beta);

private:
    std::shared_ptr<IntegralEngine> integrals_;

    const BasisSet& basis_;
    int nbasis_;
    Eigen::MatrixXd H_core_;
    Eigen::MatrixXd schwarz_;

    Eigen::MatrixXd G_alpha_accum_;
    Eigen::MatrixXd G_beta_accum_;

    bool is_first_iter_;
    Eigen::MatrixXd P_alpha_old_;
    Eigen::MatrixXd P_beta_old_;

    const PetiteList* petite_list_ = nullptr;
};

}

#endif
