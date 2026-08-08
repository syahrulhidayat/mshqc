#include "mshqc/compiler/JIT/ExecutionEngine.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include <iostream>
#include <vector>
#include <chrono>

// ABI Pemetaan Memori 3D (Sesuai ekspansi LLVM IR arg0-arg8)
struct MemRef3D {
    double *allocated;
    double *aligned;
    intptr_t offset;
    intptr_t sizes[3];
    intptr_t strides[3];
};

// ABI Pemetaan Memori 4D (Sesuai luaran struct return value)
struct MemRef4D {
    double *allocated;
    double *aligned;
    intptr_t offset;
    intptr_t sizes[4];
    intptr_t strides[4];
};

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <lowered_mlir_file>\n";
        return 1;
    }

    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    
    // 1. Parsing file LLVM IR murni
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
    if (!module) {
        std::cerr << "[FATAL] Gagal memparsing graf LLVM IR.\n";
        return 1;
    }

    // 2. Konstruksi Mesin JIT (Optimasi Hardware O3 & AVX)
    auto engineOrErr = mshqc::compiler::MshqcJIT::create(*module);
    if (!engineOrErr) {
        std::cerr << "[FATAL] Gagal menginisialisasi arsitektur JIT.\n";
        return 1;
    }
    auto engine = std::move(*engineOrErr);

    // 3. Persiapan Memori Fisik C++
    size_t lhs_size = 10 * 20 * 30;
    size_t rhs_size = 10 * 20 * 30;
    std::vector<double> lhs_data(lhs_size, 1.25);
    std::vector<double> rhs_data(rhs_size, 2.50);

    MemRef3D lhs = {lhs_data.data(), lhs_data.data(), 0, {10, 20, 30}, {600, 30, 1}};
    MemRef3D rhs = {rhs_data.data(), rhs_data.data(), 0, {10, 20, 30}, {600, 30, 1}};
    MemRef4D res = {nullptr, nullptr, 0, {0,0,0,0}, {0,0,0,0}};

    // Array pointer (Packed ABI) yang dituntut oleh OrcJIT
    std::vector<void*> args = {&lhs, &rhs, &res};

    // 4. Eksekusi JIT & Profiling Latensi
    std::cout << "[INFO] Memulai eksekusi JIT runtime...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    if (auto err = engine->invoke("test_mp2_contraction", args)) {
        std::cerr << "[FATAL] Terjadi interupsi pada eksekusi JIT runtime.\n";
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    std::cout << "[METRIK] Latensi eksekusi kontraksi murni : " << diff.count() << " detik.\n";
    if (res.aligned != nullptr) {
        std::cout << "[METRIK] Pointer L-Value berhasil dipetakan ke alamat fisik: " 
                  << res.aligned << "\n";
    }

    return 0;
}
