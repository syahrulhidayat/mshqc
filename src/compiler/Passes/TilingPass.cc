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

#include "mshqc/compiler/Passes/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

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
                    opTiles.push_back(0);
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