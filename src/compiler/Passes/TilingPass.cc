#include "mshqc/compiler/Passes/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mshqc {
namespace compiler {

#define GEN_PASS_DEF_LINALGTILING
#include "Passes.h.inc"

namespace {

struct GenericVectorizationPattern : public mlir::OpInterfaceRewritePattern<mlir::linalg::LinalgOp> {
    using OpInterfaceRewritePattern<mlir::linalg::LinalgOp>::OpInterfaceRewritePattern;
    mlir::LogicalResult matchAndRewrite(mlir::linalg::LinalgOp op, mlir::PatternRewriter &rewriter) const override {
        if (mlir::failed(mlir::linalg::vectorize(rewriter, op))) {
            return mlir::failure();
        }
        return mlir::success();
    }
};

struct LinalgTilingPass : public impl::LinalgTilingBase<LinalgTilingPass> {
    using LinalgTilingBase::LinalgTilingBase;
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::scf::SCFDialect, mlir::linalg::LinalgDialect, mlir::vector::VectorDialect>();
    }
    void runOnOperation() override {
        mlir::func::FuncOp funcOp = getOperation();
        mlir::IRRewriter rewriter(&getContext());

        llvm::SmallVector<int64_t> baseTiles(tileSizes.begin(), tileSizes.end());
        if (baseTiles.empty()) {
            baseTiles = {8, 8, 8, 8};
        }

        llvm::SmallVector<mlir::linalg::LinalgOp, 4> targetOps;
        funcOp.walk([&](mlir::linalg::LinalgOp op) {
            if (!op->hasAttr("tiled")) {
                targetOps.push_back(op);
            }
        });

        for (auto op : targetOps) {
            llvm::SmallVector<int64_t> opTiles = baseTiles;
            unsigned numLoops = op.getNumLoops();
            if (numLoops < opTiles.size()) {
                opTiles.resize(numLoops);
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
            }
        }

        mlir::RewritePatternSet vectorPatterns(&getContext());
        vectorPatterns.add<GenericVectorizationPattern>(&getContext());
        if (mlir::failed(mlir::applyPatternsGreedily(getOperation(), std::move(vectorPatterns)))) {
            signalPassFailure();
        }
    }
};
}

std::unique_ptr<mlir::Pass> createLinalgTilingPass() {
    return std::make_unique<LinalgTilingPass>();
}

}
}
