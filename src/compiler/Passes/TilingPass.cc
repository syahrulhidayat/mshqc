#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

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
            // Level 1: L2 Cache Macro-tiling (Parameter: 128)
            llvm::SmallVector<int64_t> l2Tiles;
            for (auto iterType : op.getIteratorTypesArray()) {
                if (iterType == mlir::utils::IteratorType::parallel) {
                    l2Tiles.push_back(128); 
                } else {
                    l2Tiles.push_back(0);
                }
            }

            mlir::linalg::LinalgTilingOptions l2Options;
            l2Options.setTileSizes(l2Tiles);
            l2Options.setLoopType(mlir::linalg::LinalgTilingLoopType::ParallelLoops);

            rewriter.setInsertionPoint(op);
            mlir::FailureOr<mlir::linalg::TiledLinalgOp> l2Result = 
                mlir::linalg::tileLinalgOp(rewriter, op, l2Options);

            if (mlir::succeeded(l2Result)) {
                // Level 2: L1 Cache Micro-tiling (Parameter: 32)
                llvm::SmallVector<int64_t> l1Tiles;
                for (auto iterType : l2Result->op.getIteratorTypesArray()) {
                    if (iterType == mlir::utils::IteratorType::parallel) {
                        l1Tiles.push_back(32);
                    } else {
                        l1Tiles.push_back(0);
                    }
                }

                mlir::linalg::LinalgTilingOptions l1Options;
                l1Options.setTileSizes(l1Tiles);
                l1Options.setLoopType(mlir::linalg::LinalgTilingLoopType::ParallelLoops);

                rewriter.setInsertionPoint(l2Result->op);
                mlir::FailureOr<mlir::linalg::TiledLinalgOp> l1Result = 
                    mlir::linalg::tileLinalgOp(rewriter, l2Result->op, l1Options);

                if (mlir::succeeded(l1Result)) {
                    l1Result->op->setAttr("tiled", rewriter.getUnitAttr());
                    
                    if (!l1Result->tensorResults.empty()) {
                        rewriter.replaceOp(l2Result->op, l1Result->tensorResults);
                    } else {
                        rewriter.eraseOp(l2Result->op);
                    }
                    
                    if (!l2Result->tensorResults.empty()) {
                        rewriter.replaceOp(op, l2Result->tensorResults);
                    } else {
                        rewriter.eraseOp(op);
                    }
                }
            }
        }
    }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> createLinalgTilingPass() {
    return std::make_unique<LinalgTilingPass>();
}

} // namespace compiler
} // namespace mshqc
