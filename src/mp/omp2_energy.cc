

#include "mshqc/mp2/mp2.h"
#include "mshqc/JIT/ExecutionManager.h"
#include "mshqc/compiler/Frontend/GraphBuilder.h"
#include "mshqc/Runtime/MemRefUtils.h"
#include "mshqc/Runtime/AlignedAllocator.h"
#include <omp.h>
#include <iostream>

namespace mshqc {

double OMP2::compute_mp2_energy() {
    double E_corr = 0.0;
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    const std::string kernel_name = "mp2_energy_kernel";

    auto& jit_mgr = jit::ExecutionManager::getInstance();

    static bool is_compiled = false;
    if (!is_compiled) {
        compiler::GraphBuilder builder;
        builder.initializeModule(kernel_name);

        builder.emitContractOp(
            {na_, va_, na_, va_},
            {na_, va_, na_, va_},
            {1},
            "iajb,iajb->e"
        );
        jit_mgr.compileAndCache(kernel_name, builder.getModule());
        is_compiled = true;
    }

    if (is_restricted) {

        auto* t_blk = t2_aa_.get_block(0, 0, 0, 0);
        auto* g_blk = g_aa_.get_block(0, 0, 0, 0);

        if (t_blk && g_blk) {
            std::vector<double, runtime::AlignedAllocator<double, 64>> e_buf(1, 0.0);

            runtime::StridedMemRefType<double, 4> memref_T;
            memref_T.allocatedPtr = const_cast<double*>(t_blk->data());
            memref_T.alignedPtr = memref_T.allocatedPtr;

            runtime::StridedMemRefType<double, 4> memref_G;
            memref_G.allocatedPtr = const_cast<double*>(g_blk->data());
            memref_G.alignedPtr = memref_G.allocatedPtr;

            auto memref_E = runtime::makeMemRef1D(e_buf);
            void* args[] = { &memref_T, &memref_G, &memref_E };

            try {
                jit_mgr.execute(kernel_name, "contract_kernel", args);
                E_corr += e_buf[0];
            } catch(const std::exception& e) {
                std::cerr << "[FATAL] MSHQC JIT Trap: Eksekusi Energi MP2 Gagal: " << e.what() << "\n";
                std::abort();
            }
        }

        e_ss_ = 0.0;
        e_os_ = E_corr;
        return E_corr;
    } else {

        return 0.0;
    }
}

}
