#include "mshqc/compiler/Passes/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mshqc {
namespace compiler {

#define GEN_PASS_DEF_LOWERTOLLVM
#include "Passes.h.inc"

namespace {
struct LowerToLLVMPass : public impl::LowerToLLVMBase<LowerToLLVMPass> {
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
    }

    void runOnOperation() override {
        mlir::LLVMConversionTarget target(getContext());
        target.addLegalOp<mlir::ModuleOp>();

        mlir::LLVMTypeConverter typeConverter(&getContext());
        mlir::RewritePatternSet patterns(&getContext());

        // Rantai translasi graf dari tingkat tinggi ke abstraksi mesin (LLVM IR)
        mlir::populateAffineToStdConversionPatterns(patterns);
        mlir::populateSCFToControlFlowConversionPatterns(patterns);
        mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
        mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
        mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);

        auto module = getOperation();
        if (mlir::failed(mlir::applyFullConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
    return std::make_unique<LowerToLLVMPass>();
}

} // namespace compiler
} // namespace mshqc
