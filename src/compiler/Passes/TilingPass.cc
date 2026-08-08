#include "mshqc/compiler/Passes/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mshqc {
namespace compiler {

#define GEN_PASS_DEF_LINALGTILING
#include "Passes.h.inc"

namespace {
struct LinalgTilingPass : public impl::LinalgTilingBase<LinalgTilingPass> {
    using LinalgTilingBase::LinalgTilingBase;

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::scf::SCFDialect, mlir::linalg::LinalgDialect>();
    }

    void runOnOperation() override {
        mlir::func::FuncOp funcOp = getOperation();
        mlir::IRRewriter rewriter(&getContext());

        // Mengambil ukuran memori dari CLI mshqc-opt, fallback ke 32x32x32x32
        llvm::SmallVector<int64_t> tiles = tileSizes;
        if (tiles.empty()) {
            tiles = {32, 32, 32, 32}; 
        }

        mlir::linalg::LinalgTilingOptions tilingOptions;
        tilingOptions.setTileSizes(tiles);

        funcOp.walk([&](mlir::linalg::GenericOp op) {
            if (op->hasAttr("tiled")) return mlir::WalkResult::advance();

            rewriter.setInsertionPoint(op);
            mlir::FailureOr<mlir::linalg::TilingResult> tilingResult = 
                mlir::linalg::tileLinalgOp(rewriter, op, tilingOptions);
            
            if (mlir::succeeded(tilingResult)) {
                if (!tilingResult->tiledOps.empty()) {
                    tilingResult->tiledOps.front()->setAttr("tiled", rewriter.getUnitAttr());
                }
                rewriter.replaceOp(op, tilingResult->replacements);
            }
            return mlir::WalkResult::advance();
        });
    }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> createLinalgTilingPass() {
    return std::make_unique<LinalgTilingPass>();
}

} // namespace compiler
} // namespace mshqc
