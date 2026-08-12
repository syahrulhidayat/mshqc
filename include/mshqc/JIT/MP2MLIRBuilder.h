

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

namespace mshqc {
namespace jit {

class MP2MLIRBuilder {
public:
    MP2MLIRBuilder(mlir::MLIRContext* ctx) : context_(ctx), builder_(ctx) {
        context_->loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect>();
        module_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
    }

    void buildQuarterTransformGraph(int64_t nbasis, int64_t nvirt) {
        auto f64 = builder_.getF64Type();

        auto memref_eri_ao = mlir::MemRefType::get({nbasis, nbasis, nbasis, nbasis}, f64);
        auto memref_c_mo   = mlir::MemRefType::get({nbasis, nvirt}, f64);
        auto memref_eri_q  = mlir::MemRefType::get({nbasis, nbasis, nbasis, nvirt}, f64);

        auto funcType = builder_.getFunctionType({memref_eri_ao, memref_c_mo, memref_eri_q}, {});
        auto funcOp = mlir::func::FuncOp::create(builder_.getUnknownLoc(), "mp2_quarter_transform", funcType);
        module_.push_back(funcOp);

        mlir::Block* entryBlock = funcOp.addEntryBlock();
        builder_.setInsertionPointToStart(entryBlock);

        auto eri_ao_arg = entryBlock->getArgument(0);
        auto c_mo_arg   = entryBlock->getArgument(1);
        auto eri_q_arg  = entryBlock->getArgument(2);

        llvm::SmallVector<mlir::AffineMap, 3> indexingMaps = {
            mlir::AffineMap::parse("(mu, nu, lam, sig, b) -> (mu, nu, lam, sig)", context_),
            mlir::AffineMap::parse("(mu, nu, lam, sig, b) -> (sig, b)", context_),
            mlir::AffineMap::parse("(mu, nu, lam, sig, b) -> (mu, nu, lam, b)", context_)
        };

        llvm::SmallVector<llvm::StringRef, 5> iteratorTypes = {
            "parallel", "parallel", "parallel", "reduction", "parallel"
        };

        auto genericOp = builder_.create<mlir::linalg::GenericOp>(
            builder_.getUnknownLoc(),
            mlir::TypeRange{},
            mlir::ValueRange{eri_ao_arg, c_mo_arg},
            mlir::ValueRange{eri_q_arg},
            indexingMaps,
            iteratorTypes,
            [&](mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange args) {
                auto mul = b.create<mlir::arith::MulFOp>(loc, args[0], args[1]);
                auto add = b.create<mlir::arith::AddFOp>(loc, mul, args[2]);
                b.create<mlir::linalg::YieldOp>(loc, add.getResult());
            }
        );

        builder_.create<mlir::func::ReturnOp>(builder_.getUnknownLoc());
    }

    void optimizeAndLower() {
        mlir::PassManager pm(context_);

        mlir::linalg::LinalgTilingOptions tilingOptions;
        tilingOptions.setTileSizes({32, 32, 32, 32, 32});
        pm.addPass(mlir::createLinalgStrategyTilePass("mp2_quarter_transform", tilingOptions));

        pm.addPass(mlir::bufferization::createBufferDeallocationPass());

        if (mlir::failed(pm.run(module_))) {
            throw std::runtime_error("MLIR Pass Manager failed to optimize MP2 graph.");
        }
    }

    mlir::OwningOpRef<mlir::ModuleOp>& getModule() { return module_; }

private:
    mlir::MLIRContext* context_;
    mlir::OpBuilder builder_;
    mlir::OwningOpRef<mlir::ModuleOp> module_;
};

}
}
