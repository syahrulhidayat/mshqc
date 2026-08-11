// ==============================================================================
// Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
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

















#ifndef MSHQC_DFMP2_H
#define MSHQC_DFMP2_H

#include "mshqc/scf/scf.h"
#include "mshqc/basis.h"
#include "mshqc/ints/integrals.h"
#include <Eigen/Dense>
#ifdef I
#undef I
#endif

namespace mshqc {

struct DFMP2Result {
    double e_ss;

    double e_os;

    double e_corr;

    double e_total;

};

class DFMP2 {
public:

    DFMP2(const SCFResult& rohf_result,
          const BasisSet& basis,
          const BasisSet& aux_basis,
          std::shared_ptr<IntegralEngine> integrals);

    DFMP2Result compute();

private:
    const SCFResult& rohf_;
    const BasisSet& basis_;
    const BasisSet& aux_basis_;
    std::shared_ptr<IntegralEngine> integrals_;

    int nbf_;

    int naux_;

    int nocc_;

    int nvir_;

    Eigen::MatrixXd B_ia_;

    Eigen::MatrixXd J_;

    Eigen::MatrixXd J_inv_;

    void compute_metric();

    void transform_3center();

    double compute_ss_energy();
    double compute_os_energy();
};

}

#endif
