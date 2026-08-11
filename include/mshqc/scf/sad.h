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


#ifndef MSHQC_SAD_H
#define MSHQC_SAD_H

#include "mshqc/core/molecule.h"
#include "mshqc/basis.h"
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <map>
#ifdef I
#undef I
#endif

namespace mshqc {

enum class SadBasisType {
    MINIMAL,
    DOUBLE_ZETA,
    UNKNOWN
};

class SADGuess {
public:

    static Eigen::MatrixXd build(const Molecule& mol, const BasisSet& basis);

private:

    static Eigen::MatrixXd get_atomic_density(int Z, SadBasisType btype, int n_bf);
};

}

#endif
