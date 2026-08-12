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


#include "mshqc/mp3/omp3.h"
#include "mshqc/JIT/ExecutionManager.h"
#include "mshqc/compiler/Frontend/GraphBuilder.h"
#include "mshqc/Runtime/MemRefUtils.h"
#include "mshqc/Runtime/AlignedAllocator.h"
#include <iostream>

namespace mshqc {

double OMP3::get_correlation_energy() const {
    return e_ss_ + e_os_ + e_mp3_tot_;
}

double OMP3::execute_micro_iterations() {
    OMP2::execute_micro_iterations();
    compute_mp3_correction();

    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    L2_aa_ = t2_3rd_aa_;
    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        L2_bb_ = t2_3rd_bb_;
        L2_ab_ = t2_3rd_ab_;
    }

    build_opdm_alpha();
    if (!is_restricted && nb_ > 0) {
        build_opdm_beta();
    } else if (is_restricted && nb_ > 0) {
        G_oo_beta_ = G_oo_alpha_;
    }

    return get_correlation_energy();
}

void OMP3::compute_mp3_correction() {
    if (na_ == 0 || va_ == 0) { e_mp3_tot_ = 0.0; return; }

    auto& jit_mgr = jit::ExecutionManager::getInstance();
    static bool is_mp3_ladder_compiled = false;
    const std::string kernel_name = "mp3_ladder_kernel";

    if (!is_mp3_ladder_compiled) {
        compiler::GraphBuilder builder;
        builder.initializeModule(kernel_name);
        builder.emitContractOp(
            {na_, na_, va_, va_},
            {na_, na_, na_, na_},
            {na_, na_, va_, va_},
            "mnab,minj->ijab"
        );
        jit_mgr.compileAndCache(kernel_name, builder.getModule());
        is_mp3_ladder_compiled = true;
    }

    auto* t2_aa_dense = t2_aa_.get_block(0,0,0,0);
    if (!t2_aa_dense) throw std::runtime_error("OMP3 missing T2 dense block.");

    runtime::StridedMemRefType<double, 4> memref_T2;
    memref_T2.allocatedPtr = t2_aa_dense->data();
    memref_T2.alignedPtr = memref_T2.allocatedPtr;
    memref_T2.offset = 0;
    memref_T2.sizes[0] = memref_T2.sizes[1] = na_;
    memref_T2.sizes[2] = memref_T2.sizes[3] = va_;
    memref_T2.strides[3] = 1; memref_T2.strides[2] = va_;
    memref_T2.strides[1] = va_ * va_; memref_T2.strides[0] = na_ * va_ * va_;

    std::vector<double, runtime::AlignedAllocator<double, 64>> V_oooo_buf(na_*na_*na_*na_, 0.0);
    runtime::StridedMemRefType<double, 4> memref_V;
    memref_V.allocatedPtr = V_oooo_buf.data();
    memref_V.alignedPtr = memref_V.allocatedPtr;
    memref_V.offset = 0;
    memref_V.sizes[0] = memref_V.sizes[1] = memref_V.sizes[2] = memref_V.sizes[3] = na_;
    memref_V.strides[3] = 1; memref_V.strides[2] = na_;
    memref_V.strides[1] = na_ * na_; memref_V.strides[0] = na_ * na_ * na_;

    std::vector<double, runtime::AlignedAllocator<double, 64>> W_ladder_buf(na_*na_*va_*va_, 0.0);
    runtime::StridedMemRefType<double, 4> memref_W;
    memref_W.allocatedPtr = W_ladder_buf.data();
    memref_W.alignedPtr = memref_W.allocatedPtr;
    memref_W.offset = 0;
    memref_W.sizes[0] = memref_W.sizes[1] = na_;
    memref_W.sizes[2] = memref_W.sizes[3] = va_;
    memref_W.strides[3] = 1; memref_W.strides[2] = va_;
    memref_W.strides[1] = va_ * va_; memref_W.strides[0] = na_ * va_ * va_;

    void* args[] = { &memref_T2, &memref_V, &memref_W };

    try {
        jit_mgr.execute(kernel_name, "contract_kernel", args);
    } catch(const std::exception& e) {
        std::cerr << "[FATAL] MSHQC JIT Trap: Eksekusi MP3 Ladder Contraction Gagal: " << e.what() << "\n";
        std::abort();
    }
}

}
