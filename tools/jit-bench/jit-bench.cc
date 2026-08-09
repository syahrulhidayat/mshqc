#include <x86intrin.h>
#include <cstdlib>
#include "mshqc/compiler/JIT/ExecutionEngine.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include <iostream>
#include <vector>
#include <chrono>

struct MemRef3D {
    double *allocated;
    double *aligned;
    intptr_t offset;
    intptr_t sizes[3];
    intptr_t strides[3];
};

struct MemRef4D {
    double *allocated;
    double *aligned;
    intptr_t offset;
    intptr_t sizes[4];
    intptr_t strides[4];
};

extern "C" {
    void memrefCopy(int64_t elemSize, void* unrankedDescriptorSrc, void* unrankedDescriptorDst) {

        return;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <lowered_mlir_file>\n";
        return 1;
    }

    volatile void* bypass_dce = (void*)memrefCopy;

    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::omp::OpenMPDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();

    auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
    if (!module) {
        std::cerr << "[FATAL] Gagal memparsing graf LLVM IR.\n";
        return 1;
    }

    auto engineOrErr = mshqc::compiler::MshqcJIT::create(*module);
    if (!engineOrErr) {
        std::cerr << "[FATAL] Gagal menginisialisasi arsitektur JIT.\n";
        return 1;
    }
    auto engine = std::move(*engineOrErr);

    size_t lhs_size = 10 * 20 * 30;
    size_t rhs_size = 10 * 20 * 30;

    size_t lhs_bytes = (lhs_size * sizeof(double) + 63) & ~63;
    size_t rhs_bytes = (rhs_size * sizeof(double) + 63) & ~63;

    double* lhs_ptr = static_cast<double*>(std::aligned_alloc(64, lhs_bytes));
    double* rhs_ptr = static_cast<double*>(std::aligned_alloc(64, rhs_bytes));

    for(size_t i = 0; i < lhs_size; ++i) lhs_ptr[i] = 1.25;
    for(size_t i = 0; i < rhs_size; ++i) rhs_ptr[i] = 2.50;

    MemRef3D lhs = {lhs_ptr, lhs_ptr, 0, {10, 20, 30}, {600, 30, 1}};
    MemRef3D rhs = {rhs_ptr, rhs_ptr, 0, {10, 20, 30}, {600, 30, 1}};

    size_t res_size = 10 * 20 * 10 * 20;
    size_t res_bytes = (res_size * sizeof(double) + 63) & ~63;
    double* res_ptr = static_cast<double*>(std::aligned_alloc(64, res_bytes));

    MemRef4D res = {res_ptr, res_ptr, 0, {10, 20, 10, 20}, {4000, 200, 20, 1}};

    std::vector<void*> args = {
        &lhs,
        &rhs,
        &res
    };

    std::cout << "[INFO] Memulai eksekusi JIT runtime...\n";

    unsigned int dummy;
    uint64_t start_cycles = __rdtscp(&dummy);

    if (auto err = engine->invoke("test_mp2_contraction", args)) {
        std::cerr << "[FATAL] Terjadi interupsi pada eksekusi JIT runtime.\n";
        return 1;
    }

    uint64_t end_cycles = __rdtscp(&dummy);
    uint64_t total_cycles = end_cycles - start_cycles;

    std::cout << "[METRIK] Total CPU Cycles terkonsumsi : " << total_cycles << "\n";
    if (res.aligned != nullptr) {
        std::cout << "[METRIK] Pointer L-Value berhasil dipetakan ke alamat fisik: "
                  << res.aligned << "\n";
    }

    return 0;
}
