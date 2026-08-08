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

namespace {
struct GenericVectorizationPattern : public mlir::OpInterfaceRewritePattern<mlir::linalg::LinalgOp> {
    using OpInterfaceRewritePattern<mlir::linalg::LinalgOp>::OpInterfaceRewritePattern;
    mlir::LogicalResult matchAndRewrite(mlir::linalg::LinalgOp op, mlir::PatternRewriter &rewriter) const override {
        // Pemanggilan utilitas MLIR Vectorize modern
        if (mlir::failed(mlir::linalg::vectorize(rewriter, op))) {
            return mlir::failure();
        }
        return mlir::success();
    }
};
} // namespace

struct LinalgTilingPass : public impl::LinalgTilingBase<LinalgTilingPass> {
    using LinalgTilingBase::LinalgTilingBase;

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::scf::SCFDialect, mlir::linalg::LinalgDialect, mlir::vector::VectorDialect>();
    }

    void runOnOperation() override {
        mlir::func::FuncOp funcOp = getOperation();
        mlir::IRRewriter rewriter(&getContext());

        llvm::SmallVector<int64_t> tiles(tileSizes.begin(), tileSizes.end());
        if (tiles.empty()) {
            tiles = {32, 32, 32, 32};
        }

        mlir::linalg::LinalgTilingOptions tilingOptions;
        tilingOptions.setTileSizes(tiles);

        funcOp.walk([&](mlir::linalg::LinalgOp op) {
            if (op->hasAttr("tiled")) return mlir::WalkResult::advance();

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
            return mlir::WalkResult::advance();
        });

        // Pemaksaan vektorisasi Linalg ke representasi intrinsik Vector Dialect
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
