// ==============================================================================
// Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
// ==============================================================================
// MSHQC - Compiler Pass: Comprehensive Bufferization & Deallocation
// ==============================================================================

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mshqc/compiler/Passes/Passes.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mshqc {
namespace compiler {

#define GEN_PASS_DEF_BUFFERIZE
#include "Passes.h.inc"

namespace {

struct BufferizePass : public impl::BufferizeBase<BufferizePass> {
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::bufferization::BufferizationDialect,
                        mlir::memref::MemRefDialect,
                        mlir::linalg::LinalgDialect,
                        mlir::tensor::TensorDialect,
                        mlir::func::FuncDialect>();
        
        mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    }

    void runOnOperation() override {
        mlir::ModuleOp module = getOperation();
        mlir::MLIRContext *context = &getContext();

        // 1. Konfigurasi One-Shot Bufferize
        mlir::bufferization::OneShotBufferizationOptions options;
        options.bufferizeFunctionBoundaries = true;

        // 2. Deklarasi BufferizationState (Default Constructor: 0 argumen)
        mlir::bufferization::BufferizationState state;

        // 3. Eksekusi Bufferisasi Terpadu
        if (mlir::failed(mlir::bufferization::runOneShotModuleBufferize(module, options, state))) {
            signalPassFailure();
            return;
        }

        // 4. Terapkan Buffer Deallocation secara Global
        mlir::PassManager pm(context);
        
        pm.addPass(mlir::bufferization::createOwnershipBasedBufferDeallocationPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());

        if (mlir::failed(pm.run(module))) {
            signalPassFailure();
        }
    }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> createBufferizePass() {
    return std::make_unique<BufferizePass>();
}

} // namespace compiler
} // namespace mshqc
