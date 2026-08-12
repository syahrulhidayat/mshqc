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


#include "mshqc/integrals/cholesky_eri.h"
#include "mshqc/JIT/ExecutionManager.h"
#include "mshqc/compiler/Frontend/GraphBuilder.h"
#include "mshqc/Runtime/MemRefUtils.h"
#include "mshqc/Runtime/AlignedAllocator.h"
#include <iostream>
#include <cmath>

namespace mshqc {
namespace integrals {

void CholeskyERI::decompose_direct() {

    static jit::ExecutionManager jit_mgr;
    static bool is_chol_compiled = false;
    const std::string kernel_update = "cholesky_update_kernel";

    if (!is_chol_compiled) {
        compiler::GraphBuilder builder;
        builder.initializeModule(kernel_update);

        builder.emitContractOp(
            {n_basis_ * n_basis_, 200},
            {200},
            {n_basis_ * n_basis_},
            "pk,k->p"
        );

        jit_mgr.compileAndCache(kernel_update, builder.getModule());
        is_chol_compiled = true;
    }

    /*
    if (iter > 0) {
        runtime::StridedMemRefType<double, 2> memref_L_store = runtime::makeMemRef2D(L_store_aligned, npair, iter);
        runtime::StridedMemRefType<double, 1> memref_L_pivot = runtime::makeMemRef1D(L_pivot_aligned);
        runtime::StridedMemRefType<double, 1> memref_col_buf = runtime::makeMemRef1D(col_buf_aligned);

        void* args[] = { &memref_L_store, &memref_L_pivot, &memref_col_buf };

        try {
            jit_mgr.execute(kernel_update, "contract_kernel", args);
        } catch(...) { std::abort(); }
    }
    */
}

}
}
