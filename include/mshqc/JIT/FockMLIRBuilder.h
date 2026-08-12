

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
