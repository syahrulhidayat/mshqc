#!/usr/bin/env python3
from pathlib import Path

def inject_passes_header(base_dir: Path):
    path = base_dir / "include/mshqc/compiler/Passes/Passes.h"
    content = """#ifndef MSHQC_COMPILER_PASSES_H_
#define MSHQC_COMPILER_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mshqc {
namespace compiler {

// Deklarasi fungsi pembuat pass (Lowering dari Mshqc ke Linalg)
std::unique_ptr<mlir::Pass> createLowerToLinalgPass();

} // namespace compiler
} // namespace mshqc

#endif // MSHQC_COMPILER_PASSES_H_
"""
    with open(path, 'w') as f: f.write(content)
    print(f"[UPDATED] {path}")

def inject_lowering_source(base_dir: Path):
    path = base_dir / "src/compiler/Passes/LowerToLinalg.cc"
    content = """#include "mshqc/compiler/Passes/Passes.h"
#include "mshqc/compiler/Dialect/MshqcDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

// =============================================================================
// Rewrite Pattern: Mshqc.Contract -> Linalg.Generic
// =============================================================================
struct ContractOpLowering : public OpRewritePattern<mshqc::compiler::ContractOp> {
    using OpRewritePattern<mshqc::compiler::ContractOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mshqc::compiler::ContractOp op,
                                  PatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        Value lhs = op.getLhs();
        Value rhs = op.getRhs();
        
        // Ekstraksi tipe tensor
        auto lhsType = lhs.getType().cast<ShapedType>();
        auto rhsType = rhs.getType().cast<ShapedType>();
        auto resultType = op.getType().cast<ShapedType>();

        // Dalam implementasi penuh, string einsum_eq akan di-parsing di sini 
        // untuk menghasilkan AffineMap yang memetakan indeks kontraksi.
        // Untuk tahap ini, kita membuat representasi pemetaan (AffineMap) kosong
        // sebagai placeholder struktur Linalg Generic.
        
        SmallVector<AffineMap, 3> indexingMaps; // lhs, rhs, result
        SmallVector<utils::IteratorType, 3> iteratorTypes; // parallel, reduction
        
        // TODO: Generate indexing maps berdasarkan op.getEinsumEq()
        
        // Membangun operasi Linalg Generic pengganti Mshqc.Contract
        /*
        auto linalgOp = rewriter.create<linalg::GenericOp>(
            loc,
            TypeRange{resultType},
            ValueRange{lhs, rhs},
            ValueRange{}, // init tensors
            indexingMaps,
            iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
                // Implementasi MAC (Multiply-Accumulate)
                Value mul = nestedBuilder.create<arith::MulFOp>(nestedLoc, args[0], args[1]);
                Value add = nestedBuilder.create<arith::AddFOp>(nestedLoc, mul, args[2]);
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
            }
        );
        rewriter.replaceOp(op, linalgOp.getResults());
        */
        
        // Sementara di-pass untuk mencegah kompilasi terhenti karena logic AffineMap belum ada
        return success();
    }
};

// =============================================================================
// Pass Registration
// =============================================================================
struct LowerToLinalgPass : public PassWrapper<LowerToLinalgPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToLinalgPass)

    StringRef getArgument() const final { return "mshqc-lower-to-linalg"; }
    StringRef getDescription() const final { return "Lower mshqc dialect to linalg operations."; }

    void runOnOperation() override {
        ConversionTarget target(getContext());
        
        // Menentukan dialect apa saja yang sah (legal) setelah lowering
        target.addLegalDialect<linalg::LinalgDialect, affine::AffineDialect, func::FuncDialect>();
        
        // MshqcContract tidak lagi sah, harus diubah
        target.addIllegalOp<mshqc::compiler::ContractOp>();

        RewritePatternSet patterns(&getContext());
        patterns.add<ContractOpLowering>(&getContext());

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // end anonymous namespace

namespace mshqc {
namespace compiler {

std::unique_ptr<mlir::Pass> createLowerToLinalgPass() {
    return std::make_unique<LowerToLinalgPass>();
}

} // namespace compiler
} // namespace mshqc
"""
    with open(path, 'w') as f: f.write(content)
    print(f"[UPDATED] {path}")

if __name__ == "__main__":
    base_dir = Path.cwd()
    print("Menginjeksi MLIR Lowering Pass (Mshqc -> Linalg)...")
    inject_passes_header(base_dir)
    inject_lowering_source(base_dir)
    print("Selesai. Silakan kompilasi ulang.")