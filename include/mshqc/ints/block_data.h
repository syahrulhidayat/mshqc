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

#ifndef MSHQC_INTS_BLOCK_DATA_H
#define MSHQC_INTS_BLOCK_DATA_H

#include "mshqc/basis.h"
#include <vector>
#include <cmath>
#include <algorithm>
#ifdef I
#undef I
#endif

namespace mshqc {

struct ShellBlock {
    int start_shell;

    int end_shell;

    int start_basis;

    int size_basis;

    double max_schwarz;

};

inline std::vector<ShellBlock> make_shell_blocks(const BasisSet& basis, int target_size = 32) {
    std::vector<ShellBlock> blocks;
    int nshells = basis.n_shells();

    auto shell2bf = basis.shell_to_basis_function_map();

    int current_start = 0;
    int current_basis_count = 0;
    int basis_offset_start = 0;

    for (int i = 0; i < nshells; ++i) {
        int shell_dim = basis.shell(i).n_functions();

        if (current_basis_count == 0) {
            basis_offset_start = shell2bf[i];
        }

        current_basis_count += shell_dim;

        if (current_basis_count >= target_size || i == nshells - 1) {
            ShellBlock block;
            block.start_shell = current_start;
            block.end_shell   = i + 1;

            block.start_basis = basis_offset_start;
            block.size_basis  = current_basis_count;
            block.max_schwarz = 1.0;

            blocks.push_back(block);

            current_start = i + 1;
            current_basis_count = 0;
        }
    }
    return blocks;
}

}

#endif
