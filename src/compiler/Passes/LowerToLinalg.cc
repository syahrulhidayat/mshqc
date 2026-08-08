#include "mshqc/compiler/Passes/Passes.h"
#include "mshqc/compiler/Dialect/MshqcDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mshqc {
namespace compiler {

#define GEN_PASS_DEF_LOWERTOLINALG
#include "Passes.h.inc"

namespace {
struct ContractOpLowering : public mlir::OpRewritePattern<ContractOp> {
    using OpRewritePattern<ContractOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ContractOp op, mlir::PatternRewriter &rewriter) const override {
        mlir::Location loc = op.getLoc();
        mlir::Value lhs = op.getLhs();
        mlir::Value rhs = op.getRhs();
        
        auto lhsType = ::llvm::cast<mlir::ShapedType>(lhs.getType());
        auto rhsType = ::llvm::cast<mlir::ShapedType>(rhs.getType());
        auto resType = ::llvm::cast<mlir::ShapedType>(op.getResult().getType());

        // Inisialisasi tensor output dengan nol (0.0) menggunakan arith.constant dan linalg.fill
        mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getFloatAttr(resType.getElementType(), 0.0));
        mlir::Value initTensor = rewriter.create<mlir::tensor::EmptyOp>(
            loc, resType.getShape(), resType.getElementType());
        mlir::Value filledTensor = rewriter.create<mlir::linalg::FillOp>(
            loc, mlir::ValueRange{zero}, mlir::ValueRange{initTensor}).result();

        // Scaffold pemetaan iterasi: 
        // Untuk tahap ini, kita mengimplementasikan fallback matmul generik.
        // Pada iterasi lanjut, einsum_eq akan memandu AffineMap secara presisi.
        llvm::SmallVector<mlir::utils::IteratorType, 3> iteratorTypes(
            resType.getRank(), mlir::utils::IteratorType::parallel);
        
        // Asumsi Rank: Jika ini OMP2 (iaP, jbP -> iajb), maka rank iterasi total > rank result.
        // Implementasi sementara menambahkan iterasi reduction di akhir.
        iteratorTypes.push_back(mlir::utils::IteratorType::reduction);

        // TODO: Konstruksi AffineMap dinamis berbasis string einsum
        mlir::MLIRContext* ctx = rewriter.getContext();
        mlir::AffineMap lhsMap = mlir::AffineMap::getMultiDimIdentityMap(lhsType.getRank(), ctx);
        mlir::AffineMap rhsMap = mlir::AffineMap::getMultiDimIdentityMap(rhsType.getRank(), ctx);
        mlir::AffineMap resMap = mlir::AffineMap::getMultiDimIdentityMap(resType.getRank(), ctx);
        
        // Membangun operasi Linalg Generic pengganti Mshqc.Contract
        auto linalgOp = rewriter.create<mlir::linalg::GenericOp>(
            loc,
            mlir::TypeRange{resType},
            mlir::ValueRange{lhs, rhs},
            mlir::ValueRange{filledTensor},
            llvm::ArrayRef<mlir::AffineMap>{lhsMap, rhsMap, resMap},
            iteratorTypes,
            [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc, mlir::ValueRange args) {
                // Implementasi MAC (Multiply-Accumulate) Fused di level elemen
                mlir::Value mul = nestedBuilder.create<mlir::arith::MulFOp>(nestedLoc, args[0], args[1]);
                mlir::Value add = nestedBuilder.create<mlir::arith::AddFOp>(nestedLoc, mul, args[2]);
                nestedBuilder.create<mlir::linalg::YieldOp>(nestedLoc, add);
            }
        );

        rewriter.replaceOp(op, linalgOp.getResults());
        return mlir::success();
    }
};

struct LowerToLinalgPass : public impl::LowerToLinalgBase<LowerToLinalgPass> {
    void runOnOperation() override {
        mlir::ConversionTarget target(getContext());
        target.addLegalDialect<mlir::linalg::LinalgDialect, 
                               mlir::func::FuncDialect,
                               mlir::tensor::TensorDialect,
                               mlir::arith::ArithDialect>();
        
        target.addIllegalOp<ContractOp>();

        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<ContractOpLowering>(&getContext());

        if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> createLowerToLinalgPass() {
    return std::make_unique<LowerToLinalgPass>();
}

} // namespace compiler
} // namespace mshqc
