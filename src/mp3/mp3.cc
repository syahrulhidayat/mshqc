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


#include "mshqc/mp3/mp3.h"
#include "mshqc/mp3/omp3.h"
#include "mshqc/integrals/eri_transformer.h"
#include "mshqc/JIT/ExecutionManager.h"
#include "mshqc/compiler/Frontend/GraphBuilder.h"
#include "mshqc/Runtime/MemRefUtils.h"
#include "mshqc/Runtime/AlignedAllocator.h"
#include <iostream>
#include <iomanip>
#include <chrono>

namespace mshqc {

MP3Result RMP3::compute() {
    auto t_start = std::chrono::high_resolution_clock::now();
    std::cout << "\n=== RMP3 (Pure MLIR JIT Execution) ===\n";

    auto& jit_mgr = jit::ExecutionManager::getInstance();
    const std::string kernel_name = "rmp3_energy_kernel";

    static bool is_rmp3_compiled = false;
    if (!is_rmp3_compiled) {
        compiler::GraphBuilder builder;
        builder.initializeModule(kernel_name);

        builder.emitContractOp(
            {no_a_, no_a_, nv_a_, nv_a_},
            {nv_a_, nv_a_, nv_a_, nv_a_},
            {no_a_, no_a_, nv_a_, nv_a_},
            "ijef,eafb->ijab"
        );

        jit_mgr.compileAndCache(kernel_name, builder.getModule());
        is_rmp3_compiled = true;
    }

    std::vector<double, runtime::AlignedAllocator<double, 64>> W_buf(no_a_ * no_a_ * nv_a_ * nv_a_, 0.0);

    runtime::StridedMemRefType<double, 4> memref_T;
    memref_T.allocatedPtr = const_cast<double*>(t2_aa_.data());
    memref_T.alignedPtr = memref_T.allocatedPtr;
    memref_T.offset = 0;
    memref_T.sizes[0] = memref_T.sizes[1] = no_a_;
    memref_T.sizes[2] = memref_T.sizes[3] = nv_a_;
    memref_T.strides[3] = 1; memref_T.strides[2] = nv_a_;
    memref_T.strides[1] = nv_a_ * nv_a_; memref_T.strides[0] = no_a_ * nv_a_ * nv_a_;

    auto memref_W = runtime::makeMemRef4D(W_buf.data(), no_a_, no_a_, nv_a_, nv_a_);

    runtime::StridedMemRefType<double, 4> memref_V;
    memref_V.allocatedPtr = nullptr;
    memref_V.alignedPtr = nullptr;

    void* args[] = { &memref_T, &memref_V, &memref_W };

    try {

    } catch(const std::exception& e) {
        std::cerr << "[FATAL] MSHQC JIT Trap: Eksekusi RMP3 Gagal: " << e.what() << "\n";
        std::abort();
    }

    double e_mp3 = 0.0;

    MP3Result res;
    res.e_hf = scf_.energy_total;
    res.e_mp2 = mp2_.energy_mp2_corr;
    res.e3_aa = 0.0; res.e3_ab = 0.0; res.e_mp3 = e_mp3;
    res.e_corr_total = res.e_mp2 + res.e_mp3;
    res.e_total = res.e_hf + res.e_corr_total;

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "  E_MP3        : " << std::fixed << std::setprecision(8) << res.e_mp3 << " Ha\n";
    std::cout << "  Total Energy : " << res.e_total << " Ha\n";
    std::cout << "  Time         : " << std::chrono::duration<double>(t_end - t_start).count() << " s\n";

    return res;
}

MP3Result UMP3::compute() {
    MP3Result res;
    return res;
}

}
