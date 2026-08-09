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
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mshqc {
namespace compiler {

#define GEN_PASS_DEF_LINALGVECTORIZE
#include "Passes.h.inc"

namespace {
struct LinalgVectorizePass : public impl::LinalgVectorizeBase<LinalgVectorizePass> {
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::linalg::LinalgDialect, mlir::vector::VectorDialect>();
    }

    void runOnOperation() override {
        mlir::func::FuncOp funcOp = getOperation();
        mlir::IRRewriter rewriter(&getContext());

        llvm::SmallVector<mlir::linalg::LinalgOp, 4> targetOps;
        funcOp.walk([&](mlir::linalg::LinalgOp op) {

            if (op->hasAttr("tiled")) {
                targetOps.push_back(op);
            }
        });

        for (auto op : targetOps) {
            rewriter.setInsertionPoint(op);
            (void)mlir::linalg::vectorize(rewriter, op);
        }
    }
};
}

std::unique_ptr<mlir::Pass> createLinalgVectorizePass() {
    return std::make_unique<LinalgVectorizePass>();
}

}
}
