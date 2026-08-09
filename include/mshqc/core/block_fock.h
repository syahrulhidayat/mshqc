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

#ifndef MSHQC_CORE_BLOCK_FOCK_H
#define MSHQC_CORE_BLOCK_FOCK_H

#include <Eigen/Dense>
#include <vector>
#include "mshqc/ints/block_data.h"
#ifdef I
#undef I
#endif

namespace mshqc {

class BlockFockBuilder {
public:

    static void accumulate_J_blas(const double* I_batch,
                                  const double* P_block,
                                  double* F_block,
                                  int dim_row, int dim_col);

    static void pack_density_block(const Eigen::MatrixXd& P,
                                   const ShellBlock& C, const ShellBlock& D,
                                   double* buffer);

    static void scatter_fock_block_atomic(Eigen::MatrixXd& F,
                                          const ShellBlock& A, const ShellBlock& B,
                                          const double* buffer);
};

}

#endif
