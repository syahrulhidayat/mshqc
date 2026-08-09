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

#pragma once

#include <Eigen/Dense>
#include <vector>
#include "mshqc/symmetry/molecule_sym.h"

namespace mshqc {

class SalcBuilder {
public:

    SalcBuilder(BasisSymmetrizer* sym);

    std::pair<Eigen::MatrixXd, std::vector<int>> build_salc(const Eigen::MatrixXd& S);

private:

    BasisSymmetrizer* sym_;
};

}
