// ==============================================================================
// MSHQC - Pure MLIR JIT Accelerated OMP2 Correlation Density (1-RDM)
// ==============================================================================

#include "mshqc/mp2/mp2.h"
#include "mshqc/integrals/eri_transformer.h"
#include "mshqc/JIT/ExecutionManager.h"
#include "mshqc/compiler/Frontend/GraphBuilder.h"
#include "mshqc/Runtime/MemRefUtils.h"
#include "mshqc/Runtime/AlignedAllocator.h"
#include <omp.h>
#include <iostream>

namespace mshqc {

// ... [Implementasi build_fock_fast tetap dipertahankan dengan delegasi JIT jika diperlukan] ...

void OMP2::build_opdm_alpha() {
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    
    G_oo_alpha_ = Eigen::MatrixXd::Zero(na_, na_);
    G_vv_alpha_ = Eigen::MatrixXd::Zero(va_, va_);

    auto* t_aa_blk = t2_aa_.get_block(0,0,0,0);
    if (!t_aa_blk) return;

    static jit::ExecutionManager jit_mgr;
    static bool is_opdm_compiled = false;
    const std::string kernel_oo = "omp2_opdm_oo_kernel";
    const std::string kernel_vv = "omp2_opdm_vv_kernel";

    // 1. Konstruksi Graf MLIR Linalg untuk Evaluasi 1-RDM OMP2
    if (!is_opdm_compiled) {
        compiler::GraphBuilder b_oo, b_vv;
        
        b_oo.initializeModule(kernel_oo);
        // Goo(i, j) = -0.5 * sum_{a,k,b} T(i, a, k, b) * T(j, a, k, b)
        b_oo.emitContractOp(
            {na_, va_, na_, va_}, 
            {na_, va_, na_, va_}, 
            {na_, na_}, 
            "iakb,jakb->ij"
        );
        jit_mgr.compileAndCache(kernel_oo, b_oo.getModule().get());

        b_vv.initializeModule(kernel_vv);
        // Gvv(a, b) = 0.5 * sum_{i,j,c} T(i, a, j, c) * T(i, b, j, c)
        b_vv.emitContractOp(
            {na_, va_, na_, va_}, 
            {na_, va_, na_, va_}, 
            {va_, va_}, 
            "iajc,ibjc->ab"
        );
        jit_mgr.compileAndCache(kernel_vv, b_vv.getModule().get());
        
        is_opdm_compiled = true;
    }

    // 2. Persiapan C-ABI MemRef Descriptor
    runtime::StridedMemRefType<double, 4> memref_T2;
    memref_T2.allocatedPtr = const_cast<double*>(t_aa_blk->data());
    memref_T2.alignedPtr = memref_T2.allocatedPtr;
    memref_T2.offset = 0;
    memref_T2.sizes[0] = memref_T2.sizes[2] = na_;
    memref_T2.sizes[1] = memref_T2.sizes[3] = va_;
    memref_T2.strides[3] = 1; memref_T2.strides[2] = va_;
    memref_T2.strides[1] = na_ * va_; memref_T2.strides[0] = va_ * na_ * va_;

    std::vector<double, runtime::AlignedAllocator<double, 64>> Goo_buf(na_ * na_, 0.0);
    std::vector<double, runtime::AlignedAllocator<double, 64>> Gvv_buf(va_ * va_, 0.0);
    
    auto memref_Goo = runtime::makeMemRef2D(Goo_buf, na_, na_);
    auto memref_Gvv = runtime::makeMemRef2D(Gvv_buf, va_, va_);

    void* args_oo[] = { &memref_T2, &memref_T2, &memref_Goo };
    void* args_vv[] = { &memref_T2, &memref_T2, &memref_Gvv };

    // 3. Eksekusi Hardware Kernel
    try {
        jit_mgr.execute(kernel_oo, "contract_kernel", args_oo);
        jit_mgr.execute(kernel_vv, "contract_kernel", args_vv);
    } catch(const std::exception& e) {
        std::cerr << "[FATAL] MSHQC JIT Trap: Eksekusi OMP2 1-RDM Gagal: " << e.what() << "\n";
        std::abort();
    }

    // 4. Akumulasi Hasil (Scaling)
    for (int i = 0; i < na_; ++i) {
        for (int j = 0; j < na_; ++j) {
            G_oo_alpha_(i, j) = -0.5 * Goo_buf[i * na_ + j];
        }
    }
    for (int a = 0; a < va_; ++a) {
        for (int b = 0; b < va_; ++b) {
            G_vv_alpha_(a, b) = 0.5 * Gvv_buf[a * va_ + b];
        }
    }

    // ... [Implementasi untuk komponen G_oo_beta dan G_vv_beta dieksekusi dengan kerangka JIT serupa] ...
}

// ... [Implementasi build_opdm_beta dan build_generalized_fock] ...

} // namespace mshqc
