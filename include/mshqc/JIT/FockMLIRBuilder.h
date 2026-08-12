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


#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

namespace mshqc {
namespace jit {

class FockMLIRBuilder {
public:
    FockMLIRBuilder(mlir::MLIRContext* ctx) : context_(ctx), builder_(ctx) {
        context_->loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect>();
        module_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
    }

    void buildGraph(int64_t nbasis) {

    }

    void optimizeAndLower() {
        mlir::PassManager pm(context_);

        mlir::linalg::LinalgTilingOptions tilingOptions;
        tilingOptions.setTileSizes({64, 64, 64});
        pm.addPass(mlir::createLinalgStrategyTilePass("scf_build_fock", tilingOptions));

        pm.addPass(mlir::bufferization::createBufferDeallocationPass());

        if (mlir::failed(pm.run(module_))) {
            throw std::runtime_error("MLIR Pass Manager failed to optimize Fock graph.");
        }
    }

    mlir::OwningOpRef<mlir::ModuleOp>& getModule() { return module_; }

private:
    mlir::MLIRContext* context_;
    mlir::OpBuilder builder_;
    mlir::OwningOpRef<mlir::ModuleOp> module_;
};

}
}
