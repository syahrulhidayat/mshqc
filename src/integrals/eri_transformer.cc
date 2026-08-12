// ==============================================================================
// MSHQC - Pure MLIR JIT Accelerated AO-to-MO Integral Transformation
// ==============================================================================

#include "mshqc/integrals/eri_transformer.h"
#include "mshqc/JIT/ExecutionManager.h"
#include "mshqc/compiler/Frontend/GraphBuilder.h"
#include "mshqc/Runtime/MemRefUtils.h"
#include "mshqc/Runtime/AlignedAllocator.h"
#include <iostream>

namespace mshqc {
namespace integrals {

// Mengganti smart_transform_kernel yang berbasis tblis menjadi pure MLIR-JIT
static Eigen::Tensor<double, 4> smart_transform_kernel_jit(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C1, const Eigen::MatrixXd& C2,
    const Eigen::MatrixXd& C3, const Eigen::MatrixXd& C4,
    int nbf, int n1, int n2, int n3, int n4) {

    static jit::ExecutionManager jit_mgr;
    static bool is_eri_compiled = false;
    const std::string kernel_name = "eri_transform_kernel_full";

    // 1. Konstruksi Graf Fusi 4-Langkah O(N^5)
    if (!is_eri_compiled) {
        compiler::GraphBuilder builder;
        builder.initializeModule(kernel_name);
        
        // Translasi ekuivalen dari 4 tahap tblis::mult dengan loop fusion potensial
        // Tahap 1: abcd, ae -> ebcd
        // Tahap 2: ebcd, bf -> efcd
        // Tahap 3: efcd, cg -> efgd
        // Tahap 4: efgd, dh -> efgh (hasil akhir)
        
        // Catatan: Builder disederhanakan untuk mengemisikan operasi fusi penuh
        // Di back-end, ini akan diturunkan menjadi operasi Linalg dan di-tile
        builder.emitContractOp(
            {nbf, nbf, nbf, nbf}, 
            {nbf, n1}, 
            {n1, n2, n3, n4}, // Langsung memetakan ke dimensi hasil akhir
            "abcd,ae,bf,cg,dh->efgh" // Full contraction string
        );
        
        jit_mgr.compileAndCache(kernel_name, builder.getModule());
        is_eri_compiled = true;
    }

    // 2. Alokasi 64-byte aligned untuk memori target MO Integrals
    std::vector<double, runtime::AlignedAllocator<double, 64>> eri_mo_buf(n1 * n2 * n3 * n4, 0.0);
    
    // 3. Pemetaan Memori Fisik ke MLIR C-Interface ABI
    runtime::StridedMemRefType<double, 4> memref_eri_ao;
    memref_eri_ao.allocatedPtr = const_cast<double*>(eri_ao.data());
    memref_eri_ao.alignedPtr = memref_eri_ao.allocatedPtr;
    memref_eri_ao.offset = 0;
    memref_eri_ao.sizes[0] = memref_eri_ao.sizes[1] = memref_eri_ao.sizes[2] = memref_eri_ao.sizes[3] = nbf;
    memref_eri_ao.strides[3] = 1; memref_eri_ao.strides[2] = nbf;
    memref_eri_ao.strides[1] = nbf * nbf; memref_eri_ao.strides[0] = nbf * nbf * nbf;

    // MemRef Koefisien C1-C4 (Diasumsikan ditransformasi via runtime::makeMemRef2D)
    auto memref_C1 = runtime::makeMemRef2D(const_cast<Eigen::MatrixXd&>(C1), nbf, n1);
    auto memref_C2 = runtime::makeMemRef2D(const_cast<Eigen::MatrixXd&>(C2), nbf, n2);
    auto memref_C3 = runtime::makeMemRef2D(const_cast<Eigen::MatrixXd&>(C3), nbf, n3);
    auto memref_C4 = runtime::makeMemRef2D(const_cast<Eigen::MatrixXd&>(C4), nbf, n4);

    runtime::StridedMemRefType<double, 4> memref_eri_mo;
    memref_eri_mo.allocatedPtr = eri_mo_buf.data();
    memref_eri_mo.alignedPtr = memref_eri_mo.allocatedPtr;
    memref_eri_mo.offset = 0;
    memref_eri_mo.sizes[0] = n1; memref_eri_mo.sizes[1] = n2;
    memref_eri_mo.sizes[2] = n3; memref_eri_mo.sizes[3] = n4;
    memref_eri_mo.strides[3] = 1; memref_eri_mo.strides[2] = n4;
    memref_eri_mo.strides[1] = n3 * n4; memref_eri_mo.strides[0] = n2 * n3 * n4;

    void* args[] = { &memref_eri_ao, &memref_C1, &memref_C2, &memref_C3, &memref_C4, &memref_eri_mo };

    // 4. Eksekusi Hardware Kernel
    try {
        jit_mgr.execute(kernel_name, "contract_kernel", args);
    } catch(const std::exception& e) {
        std::cerr << "[FATAL] MSHQC JIT Trap: Eksekusi Transformasi Integral Gagal: " << e.what() << "\n";
        std::abort();
    }

    // Pemetaan kembali ke Eigen::Tensor untuk konsistensi API sementara
    Eigen::Tensor<double, 4> result(n1, n2, n3, n4);
    std::copy(eri_mo_buf.begin(), eri_mo_buf.end(), result.data());
    return result;
}

Eigen::Tensor<double, 4> ERITransformer::transform_custom(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C1, const Eigen::MatrixXd& C2,
    const Eigen::MatrixXd& C3, const Eigen::MatrixXd& C4,
    int nbf, int n1, int n2, int n3, int n4) {
    return smart_transform_kernel_jit(eri_ao, C1, C2, C3, C4, nbf, n1, n2, n3, n4);
}

// ... [Delegasi fungsi transformasi spesifik (OOVV, OVVV, dll) menggunakan transform_custom tetap dipertahankan] ...

} // namespace integrals
} // namespace mshqc
