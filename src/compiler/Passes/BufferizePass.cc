#include "mshqc/compiler/Passes/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mshqc {
namespace compiler {

#define GEN_PASS_DEF_BUFFERIZE
#include "Passes.h.inc"

namespace {
struct BufferizePass : public impl::BufferizeBase<BufferizePass> {
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::bufferization::BufferizationDialect,
                        mlir::memref::MemRefDialect>();
    }

    void runOnOperation() override {
        mlir::bufferization::OneShotBufferizationOptions options;
        options.allowReturnAllocs = true;
        options.bufferizeFunctionBoundaries = true;
        options.setFunctionBoundaryTypeConversion(mlir::bufferization::LayoutMapOption::IdentityExt);

        if (mlir::failed(mlir::bufferization::runOneShotModuleBufferize(getOperation(), options))) {
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
