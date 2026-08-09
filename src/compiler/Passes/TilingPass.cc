#include "mshqc/compiler/Passes/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mshqc {
namespace compiler {

#define GEN_PASS_DEF_LINALGTILING
#include "Passes.h.inc"

namespace {

struct LinalgTilingPass : public impl::LinalgTilingBase<LinalgTilingPass> {
    using LinalgTilingBase::LinalgTilingBase;
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::scf::SCFDialect, mlir::linalg::LinalgDialect>, mlir::vector::VectorDialect>();
    }
    void runOnOperation() override {
        mlir::func::FuncOp funcOp = getOperation();
        mlir::IRRewriter rewriter(&getContext());

        llvm::SmallVector<mlir::linalg::LinalgOp, 4> targetOps;
        funcOp.walk([&](mlir::linalg::LinalgOp op) {
            if (!op->hasAttr("tiled") && mlir::isa<mlir::linalg::GenericOp>(op)) {
                targetOps.push_back(op);
            }
        });

        for (auto op : targetOps) {
            llvm::SmallVector<int64_t> opTiles;

            for (auto iterType : op.getIteratorTypesArray()) {
                if (iterType == mlir::utils::IteratorType::parallel) {
                    opTiles.push_back(32);
                } else {
                    opTiles.push_back(8);
                }
            } else {
                    opTiles.push_back(8);
                }
            }

            mlir::linalg::LinalgTilingOptions tilingOptions;
            tilingOptions.setTileSizes(opTiles);
            tilingOptions.setLoopType(mlir::linalg::LinalgTilingLoopType::ParallelLoops);

            rewriter.setInsertionPoint(op);
            mlir::FailureOr<mlir::linalg::TiledLinalgOp> tilingResult =
                mlir::linalg::tileLinalgOp(rewriter, op, tilingOptions);

            if (mlir::succeeded(tilingResult)) {
                tilingResult->op->setAttr("tiled", rewriter.getUnitAttr());
                if (!tilingResult->tensorResults.empty()) {
                    rewriter.replaceOp(op, tilingResult->tensorResults);
                } else {
                    rewriter.eraseOp(op);
                }

                rewriter.setInsertionPoint(tilingResult->op);
                (void)mlir::linalg::vectorize(rewriter, tilingResult->op);
            }
        }

    }
};
}

std::unique_ptr<mlir::Pass> createLinalgTilingPass() {
    return std::make_unique<LinalgTilingPass>();
}

}
}
