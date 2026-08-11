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

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "mshqc/compiler/Passes/Passes.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"


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
        mlir::bufferization::OneShotBufferizationOptions options;
        options.bufferizeFunctionBoundaries = true;

        mlir::ModuleOp module = getOperation();
        mlir::bufferization::BufferizationState bufferizationState;

        if (mlir::failed(mlir::bufferization::runOneShotModuleBufferize(
                module, options, bufferizationState))) {
            signalPassFailure();
        }
    }
};

}

std::unique_ptr<mlir::Pass> createBufferizePass() {
    return std::make_unique<BufferizePass>();
}

}
}
