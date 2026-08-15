// ==============================================================================
// Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
// ==============================================================================

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "mshqc/compiler/Passes/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace mshqc {
namespace compiler {

#define GEN_PASS_DEF_LINALGTILING
#include "Passes.h.inc"

namespace {
struct LinalgTilingPass : public impl::LinalgTilingBase<LinalgTilingPass> {
    using LinalgTilingBase::LinalgTilingBase;

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::scf::SCFDialect, mlir::linalg::LinalgDialect, mlir::affine::AffineDialect>();
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
            llvm::SmallVector<int64_t> l2Tiles;
            for (auto iterType : op.getIteratorTypesArray()) {
                if (iterType == mlir::utils::IteratorType::parallel) {
                    l2Tiles.push_back(32); 
                } else if (iterType == mlir::utils::IteratorType::reduction) {
                    l2Tiles.push_back(16); 
                } else {
                    l2Tiles.push_back(0);
                }
            }

            mlir::linalg::LinalgTilingOptions l2Options;
            l2Options.setTileSizes(l2Tiles);
            l2Options.setLoopType(mlir::linalg::LinalgTilingLoopType::Loops);
            
            rewriter.setInsertionPoint(op);
            mlir::FailureOr<mlir::linalg::TiledLinalgOp> l2Result = 
                mlir::linalg::tileLinalgOp(rewriter, op, l2Options);

            if (mlir::succeeded(l2Result)) {
                llvm::SmallVector<int64_t> l1Tiles;
                for (auto iterType : l2Result->op.getIteratorTypesArray()) {
                    if (iterType == mlir::utils::IteratorType::parallel) {
                        l1Tiles.push_back(4); // Preservasi kapasitas register AVX
                    } else if (iterType == mlir::utils::IteratorType::reduction) {
                        l1Tiles.push_back(4);
                    } else {
                        l1Tiles.push_back(0);
                    }
                }

                mlir::linalg::LinalgTilingOptions l1Options;
                l1Options.setTileSizes(l1Tiles);
                l1Options.setLoopType(mlir::linalg::LinalgTilingLoopType::Loops);

                rewriter.setInsertionPoint(l2Result->op);
                mlir::FailureOr<mlir::linalg::TiledLinalgOp> l1Result =
                    mlir::linalg::tileLinalgOp(rewriter, l2Result->op, l1Options);

                if (mlir::succeeded(l1Result)) {
                    l1Result->op->setAttr("tiled", rewriter.getUnitAttr());
                    // SUBSTITUSI SSA LEVEL 1 (Bottom-Up)
                    rewriter.replaceOp(l2Result->op, l1Result->tensorResults);
                }
                // SUBSTITUSI SSA LEVEL 2 (Menjaga root node tetap hidup untuk L1 builder)
                rewriter.replaceOp(op, l2Result->tensorResults);
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
