// ==============================================================================
// MSHQC - Pure MLIR JIT Accelerated Self-Consistent Field (SCF)
// ==============================================================================

#include "mshqc/scf/scf.h"
#include "mshqc/Runtime/MemRefUtils.h"
#include "mshqc/Runtime/AlignedAllocator.h"
#include "mshqc/JIT/ExecutionManager.h"
#include "mshqc/JIT/FockMLIRBuilder.h"
#include <iostream>

namespace mshqc {

void BaseSCF::build_fock_matrix() {
    static jit::ExecutionManager jit_mgr;
    static bool is_kernel_compiled = false;
    const std::string kernel_name = "fock_build_kernel";

    // Kompilasi LLVM IR hanya pada iterasi pertama (JIT Caching)
    if (!is_kernel_compiled) {
        mlir::MLIRContext context;
        jit::FockMLIRBuilder builder(&context);
        
        builder.buildGraph(nbasis_);
        builder.optimizeAndLower();
        
        jit_mgr.compileAndCache(kernel_name, builder.getModule());
        is_kernel_compiled = true;
    }

    Eigen::MatrixXd dP = (iter_scf_ == 1) ? P_alpha_ : (P_alpha_ - P_old_);
    if (dP.cwiseAbs().maxCoeff() < 1e-11) {
        F_alpha_ = H_ + G_accum_;
        return;
    }

    // Pemetaan Memori C++ ke MemRef C-ABI
    auto memref_P = runtime::makeMemRef2D(dP, nbasis_, nbasis_);
    auto memref_G = runtime::makeMemRef2D(G_accum_, nbasis_, nbasis_);
    
    runtime::StridedMemRefType<double, 4> memref_ERI;
    memref_ERI.allocatedPtr = const_cast<double*>(integrals_->compute_eri().data());
    memref_ERI.alignedPtr = memref_ERI.allocatedPtr;
    memref_ERI.offset = 0;
    memref_ERI.sizes[0] = memref_ERI.sizes[1] = memref_ERI.sizes[2] = memref_ERI.sizes[3] = nbasis_;
    memref_ERI.strides[3] = 1;
    memref_ERI.strides[2] = nbasis_;
    memref_ERI.strides[1] = nbasis_ * nbasis_;
    memref_ERI.strides[0] = nbasis_ * nbasis_ * nbasis_;

    void* args[] = { &memref_P, &memref_ERI, &memref_G };

    try {
        jit_mgr.execute(kernel_name, "compute_fock_j_k", args);
    } catch(const std::exception& e) {
        std::cerr << "[FATAL] MSHQC JIT Trap: Eksekusi SCF Fock Gagal: " << e.what() << "\n";
        std::abort();
    }

    F_alpha_ = H_ + G_accum_;
    P_old_ = P_alpha_;
}

// ... Implementasi fungsi BaseSCF lainnya (compute, init, dll.) tetap dipertahankan ...

} // namespace mshqc
