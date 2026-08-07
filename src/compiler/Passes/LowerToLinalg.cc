#include "mshqc/compiler/Passes/Passes.h"
#include "mshqc/compiler/Dialect/MshqcDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct ContractOpLowering : public OpRewritePattern<mshqc::compiler::ContractOp> {
    using OpRewritePattern<mshqc::compiler::ContractOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mshqc::compiler::ContractOp op,
                                  PatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        Value lhs = op.getLhs();
        Value rhs = op.getRhs();

        auto lhsType = ::llvm::cast<ShapedType>(lhs.getType());
        auto rhsType = ::llvm::cast<ShapedType>(rhs.getType());
        auto resultType = ::llvm::cast<ShapedType>(op.getResult().getType());

        SmallVector<AffineMap, 3> indexingMaps;
        SmallVector<utils::IteratorType, 3> iteratorTypes;

        return success();
    }
};

struct LowerToLinalgPass : public PassWrapper<LowerToLinalgPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToLinalgPass)

    StringRef getArgument() const final { return "mshqc-lower-to-linalg"; }
    StringRef getDescription() const final { return "Lower mshqc dialect to linalg operations."; }

    void runOnOperation() override {
        ConversionTarget target(getContext());

        target.addLegalDialect<linalg::LinalgDialect, affine::AffineDialect, func::FuncDialect>();

        target.addIllegalOp<mshqc::compiler::ContractOp>();

        RewritePatternSet patterns(&getContext());
        patterns.add<ContractOpLowering>(&getContext());

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

}

namespace mshqc {
namespace compiler {

std::unique_ptr<mlir::Pass> createLowerToLinalgPass() {
    return std::make_unique<LowerToLinalgPass>();
}

}
}
