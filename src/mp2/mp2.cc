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


#include "mshqc/mp2/mp2.h"
#include "mshqc/JIT/ExecutionManager.h"
#include "mshqc/JIT/MP2MLIRBuilder.h"
#include "mshqc/Runtime/MemRefUtils.h"
#include "mshqc/Runtime/AlignedAllocator.h"
#include <iostream>
#include <vector>

namespace mshqc {
namespace foundation {

void OMP2::transform_integrals() {
    static jit::ExecutionManager jit_mgr;
    static bool is_compiled = false;
    const std::string kernel_name = "mp2_transform_kernel";

    if (!is_compiled) {
        mlir::MLIRContext context;
        jit::MP2MLIRBuilder builder(&context);

        builder.buildQuarterTransformGraph(nbf_, va_);
        builder.optimizeAndLower();

        jit_mgr.compileAndCache(kernel_name, builder.getModule());
        is_compiled = true;
    }

    std::vector<double, runtime::AlignedAllocator<double, 64>> eri_quarter(nbf_ * nbf_ * nbf_ * va_, 0.0);

    runtime::StridedMemRefType<double, 4> memref_eri_ao;
    memref_eri_ao.allocatedPtr = const_cast<double*>(integrals_->compute_eri().data());
    memref_eri_ao.alignedPtr = memref_eri_ao.allocatedPtr;
    memref_eri_ao.offset = 0;
    for(int i=0; i<4; i++) memref_eri_ao.sizes[i] = nbf_;
    memref_eri_ao.strides[3] = 1; memref_eri_ao.strides[2] = nbf_;
    memref_eri_ao.strides[1] = nbf_*nbf_; memref_eri_ao.strides[0] = nbf_*nbf_*nbf_;

    auto memref_C = runtime::makeMemRef2D(scf_.C_alpha.rightCols(va_), nbf_, va_);

    runtime::StridedMemRefType<double, 4> memref_eri_q;
    memref_eri_q.allocatedPtr = eri_quarter.data();
    memref_eri_q.alignedPtr = eri_quarter.data();
    memref_eri_q.offset = 0;
    memref_eri_q.sizes[0] = memref_eri_q.sizes[1] = memref_eri_q.sizes[2] = nbf_; memref_eri_q.sizes[3] = va_;
    memref_eri_q.strides[3] = 1; memref_eri_q.strides[2] = va_;
    memref_eri_q.strides[1] = nbf_ * va_; memref_eri_q.strides[0] = nbf_ * nbf_ * va_;

    void* args[] = { &memref_eri_ao, &memref_C, &memref_eri_q };

    try {
        jit_mgr.execute(kernel_name, "mp2_quarter_transform", args);
    } catch(const std::exception& e) {
        std::cerr << "[FATAL] MSHQC JIT Trap: Eksekusi MP2 Quarter-Transform Gagal: " << e.what() << "\n";
        std::abort();
    }
}

}
}
