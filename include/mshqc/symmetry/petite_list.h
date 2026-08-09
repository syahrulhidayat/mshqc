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

#ifndef MSHQC_SYMMETRY_PETITE_LIST_H
#define MSHQC_SYMMETRY_PETITE_LIST_H

#include "mshqc/basis.h"
#include "mshqc/symmetry/point_group.h"
#include <vector>
#ifdef I
#undef I
#endif

namespace mshqc {

struct UniqueShellPair {
    int p;
    int q;
    double weight;
};

struct UniqueShellQuartet {
    int M;
    int N;
    int P;
    int Q;
    double weight;
};

class PetiteList {
public:
    PetiteList(const BasisSet& basis, const PointGroup& pg);

    void build();

    const std::vector<UniqueShellPair>& get_unique_pairs() const { return unique_pairs_; }
    const std::vector<UniqueShellQuartet>& get_unique_quartets() const { return unique_quartets_; }

private:
    const BasisSet& basis_;
    const PointGroup& pg_;

    std::vector<UniqueShellPair> unique_pairs_;
    std::vector<UniqueShellQuartet> unique_quartets_;

    int find_shell_at(const Eigen::Vector3d& pos, int original_shell_idx) const;
};

}

#endif
