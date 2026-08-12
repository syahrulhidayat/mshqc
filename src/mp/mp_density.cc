

#include "mshqc/mp/mp_density.h"
#include "mshqc/JIT/ExecutionManager.h"
#include "mshqc/compiler/Frontend/GraphBuilder.h"
#include "mshqc/Runtime/MemRefUtils.h"
#include "mshqc/Runtime/AlignedAllocator.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace mshqc {
namespace mp {

void MPDensityMatrix::add_t2_contribution(
    Eigen::MatrixXd& opdm,
    const Eigen::Tensor<double, 4>& t2_aa,
    const Eigen::Tensor<double, 4>& t2_bb,
    const Eigen::Tensor<double, 4>& t2_ab,
    int n_occ_alpha, int n_occ_beta,
    int n_virt_alpha, int n_virt_beta) {

    static jit::ExecutionManager jit_mgr;
    static bool is_t2_rdm_compiled = false;
    const std::string kernel_vv = "t2_rdm_vv_kernel";
    const std::string kernel_oo = "t2_rdm_oo_kernel";

    if (!is_t2_rdm_compiled) {
        compiler::GraphBuilder builder_vv, builder_oo;

        builder_vv.initializeModule(kernel_vv);

        builder_vv.emitContractOp(
            {n_occ_alpha, n_occ_alpha, n_virt_alpha, n_virt_alpha},
            {n_occ_alpha, n_occ_alpha, n_virt_alpha, n_virt_alpha},
            {n_virt_alpha, n_virt_alpha},
            "ijab,ijcb->ac"
        );
        jit_mgr.compileAndCache(kernel_vv, builder_vv.getModule());

        builder_oo.initializeModule(kernel_oo);

        builder_oo.emitContractOp(
            {n_occ_alpha, n_occ_alpha, n_virt_alpha, n_virt_alpha},
            {n_occ_alpha, n_occ_alpha, n_virt_alpha, n_virt_alpha},
            {n_occ_alpha, n_occ_alpha},
            "ijab,kjab->ki"
        );
        jit_mgr.compileAndCache(kernel_oo, builder_oo.getModule());

        is_t2_rdm_compiled = true;
    }

    std::vector<double, runtime::AlignedAllocator<double, 64>> vv_blk(n_virt_alpha * n_virt_alpha, 0.0);
    std::vector<double, runtime::AlignedAllocator<double, 64>> oo_blk(n_occ_alpha * n_occ_alpha, 0.0);

    runtime::StridedMemRefType<double, 4> memref_T2;
    memref_T2.allocatedPtr = const_cast<double*>(t2_aa.data());
    memref_T2.alignedPtr = memref_T2.allocatedPtr;
    memref_T2.offset = 0;
    memref_T2.sizes[0] = memref_T2.sizes[1] = n_occ_alpha;
    memref_T2.sizes[2] = memref_T2.sizes[3] = n_virt_alpha;
    memref_T2.strides[3] = 1; memref_T2.strides[2] = n_virt_alpha;
    memref_T2.strides[1] = n_virt_alpha * n_virt_alpha; memref_T2.strides[0] = n_occ_alpha * n_virt_alpha * n_virt_alpha;

    auto memref_vv = runtime::makeMemRef2D(vv_blk, n_virt_alpha, n_virt_alpha);
    auto memref_oo = runtime::makeMemRef2D(oo_blk, n_occ_alpha, n_occ_alpha);

    void* args_vv[] = { &memref_T2, &memref_T2, &memref_vv };
    void* args_oo[] = { &memref_T2, &memref_T2, &memref_oo };

    try {
        jit_mgr.execute(kernel_vv, "contract_kernel", args_vv);
        jit_mgr.execute(kernel_oo, "contract_kernel", args_oo);
    } catch(const std::exception& e) {
        std::cerr << "[FATAL] MSHQC JIT Trap: Eksekusi T2 1-RDM Contraction Gagal: " << e.what() << "\n";
        std::abort();
    }

    for (int a = 0; a < n_virt_alpha; ++a) {
        for (int c = 0; c < n_virt_alpha; ++c) {
            opdm(n_occ_alpha + a, n_occ_alpha + c) += 0.5 * vv_blk[a * n_virt_alpha + c];
        }
    }
    for (int k = 0; k < n_occ_alpha; ++k) {
        for (int i = 0; i < n_occ_alpha; ++i) {
            opdm(k, i) -= 0.5 * oo_blk[k * n_occ_alpha + i];
        }
    }

}

}
}
