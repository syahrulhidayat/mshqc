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


#ifndef MSHQC_SYMMETRY_MOLECULE_SYM_H
#define MSHQC_SYMMETRY_MOLECULE_SYM_H

#include "mshqc/basis.h"
#include "mshqc/symmetry/point_group.h"
#include "mshqc/symmetry/petite_list.h"
#include <map>
#include <Eigen/Dense>
#include <vector>
#ifdef I
#undef I
#endif

namespace mshqc {

class BasisSymmetrizer {
public:
    BasisSymmetrizer(const BasisSet& basis, const PointGroup& pg, const PetiteList& pl);

    void symmetrize(Eigen::MatrixXd& F) const;
    std::vector<int> assign_mo_irreps(const Eigen::MatrixXd& C, double threshold = 1e-5) const;
    const std::vector<Eigen::MatrixXd>& get_R_ao() const { return R_ao_; }

private:
    const BasisSet& basis_;
    const PointGroup& pg_;
    const PetiteList& pl_;

    std::vector<std::vector<int>> shell_map_;
    std::vector<Eigen::MatrixXd> R_ao_;

    void build_map_and_matrices();
};

}

#endif
