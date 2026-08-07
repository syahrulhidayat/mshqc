#include "mshqc/compiler/Passes/Passes.h"
#include "mshqc/compiler/Dialect/MshqcDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mshqc {
namespace compiler {

#define GEN_PASS_DEF_LOWERTOLINALG
#include "mshqc/compiler/Passes/Passes.h.inc"

namespace {
struct ContractOpLowering : public mlir::OpRewritePattern<ContractOp> {
    using OpRewritePattern<ContractOp>::OpRewritePattern;
    mlir::LogicalResult matchAndRewrite(ContractOp op, mlir::PatternRewriter &rewriter) const override {

        return mlir::success();
    }
};

struct LowerToLinalgPass : public impl::LowerToLinalgBase<LowerToLinalgPass> {
    void runOnOperation() override {
        mlir::ConversionTarget target(getContext());
        target.addLegalDialect<mlir::linalg::LinalgDialect, mlir::func::FuncDialect>();
        target.addIllegalOp<ContractOp>();

        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<ContractOpLowering>(&getContext());

        if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
}

std::unique_ptr<mlir::Pass> createLowerToLinalgPass() {
    return std::make_unique<LowerToLinalgPass>();
}

}
}
