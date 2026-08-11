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

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "mshqc/compiler/Passes/Passes.h"
#include "mshqc/compiler/Dialect/MshqcDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"
#include <string>
#include <vector>
#include <map>

namespace mshqc {
namespace compiler {

#define GEN_PASS_DEF_LOWERTOLINALG
#include "Passes.h.inc"

namespace {

static mlir::LogicalResult parseEinsumToAffine(
    mlir::MLIRContext* ctx, llvm::StringRef einsum_eq,
    mlir::AffineMap& lhsMap, mlir::AffineMap& rhsMap, mlir::AffineMap& resMap,
    llvm::SmallVectorImpl<mlir::utils::IteratorType>& iterTypes) {

    std::string eq = einsum_eq.str();

    eq.erase(std::remove_if(eq.begin(), eq.end(), ::isspace), eq.end());

    auto arrow_pos = eq.find("->");
    if (arrow_pos == std::string::npos) return mlir::failure();

    std::string lhs_rhs_str = eq.substr(0, arrow_pos);
    std::string res_str = eq.substr(arrow_pos + 2);

    auto comma_pos = lhs_rhs_str.find(",");
    if (comma_pos == std::string::npos) return mlir::failure();

    std::string lhs_str = lhs_rhs_str.substr(0, comma_pos);
    std::string rhs_str = lhs_rhs_str.substr(comma_pos + 1);

    std::map<char, int> char_to_dim;
    int current_dim = 0;

    auto register_chars = [&](const std::string& str) {
        for (char c : str) {
            if (char_to_dim.find(c) == char_to_dim.end()) {
                char_to_dim[c] = current_dim++;
            }
        }
    };

    register_chars(lhs_str);
    register_chars(rhs_str);
    register_chars(res_str);

    int num_loops = current_dim;

    for (int i = 0; i < num_loops; ++i) {
        char loop_char = ' ';
        for (auto const& [key, val] : char_to_dim) {
            if (val == i) loop_char = key;
        }
        if (res_str.find(loop_char) == std::string::npos) {
            iterTypes.push_back(mlir::utils::IteratorType::reduction);
        } else {
            iterTypes.push_back(mlir::utils::IteratorType::parallel);
        }
    }

    auto build_map = [&](const std::string& str) -> mlir::AffineMap {
        llvm::SmallVector<mlir::AffineExpr, 4> exprs;
        for (char c : str) {
            exprs.push_back(mlir::getAffineDimExpr(char_to_dim[c], ctx));
        }
        return mlir::AffineMap::get(num_loops, 0, exprs, ctx);
    };

    lhsMap = build_map(lhs_str);
    rhsMap = build_map(rhs_str);
    resMap = build_map(res_str);

    return mlir::success();
}

struct ContractOpLowering : public mlir::OpRewritePattern<ContractOp> {
    using OpRewritePattern<ContractOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ContractOp op, mlir::PatternRewriter &rewriter) const override {
        mlir::Location loc = op.getLoc();
        mlir::Value lhs = op.getLhs();
        mlir::Value rhs = op.getRhs();

        auto lhsType = ::llvm::cast<mlir::ShapedType>(lhs.getType());
        auto rhsType = ::llvm::cast<mlir::ShapedType>(rhs.getType());
        auto resType = ::llvm::cast<mlir::ShapedType>(op.getResult().getType());

        mlir::Value filledTensor = op.getOuts(); // Pemetaan Zero-copy langsung ke memori fisik C++

        llvm::SmallVector<mlir::utils::IteratorType, 5> iteratorTypes;
        mlir::AffineMap lhsMap, rhsMap, resMap;

        if (mlir::failed(parseEinsumToAffine(rewriter.getContext(), op.getEinsumEq(), lhsMap, rhsMap, resMap, iteratorTypes))) {
            return op.emitError("Failed to parse einsum equation into AffineMaps");
        }

        auto linalgOp = rewriter.create<mlir::linalg::GenericOp>(
            loc,
            mlir::TypeRange{resType},
            mlir::ValueRange{lhs, rhs},
            mlir::ValueRange{filledTensor},
            llvm::ArrayRef<mlir::AffineMap>{lhsMap, rhsMap, resMap},
            iteratorTypes,
            [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc, mlir::ValueRange args) {
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
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::linalg::LinalgDialect,
                        mlir::func::FuncDialect,
                        mlir::tensor::TensorDialect,
                        mlir::arith::ArithDialect>();
    }

    void runOnOperation() override {
        mlir::MLIRContext *ctx = &getContext();
        ctx->getOrLoadDialect<mlir::linalg::LinalgDialect>();
        ctx->getOrLoadDialect<mlir::tensor::TensorDialect>();
        ctx->getOrLoadDialect<mlir::arith::ArithDialect>();

        mlir::ConversionTarget target(*ctx);
        target.addLegalDialect<mlir::linalg::LinalgDialect,
                               mlir::func::FuncDialect,
                               mlir::tensor::TensorDialect,
                               mlir::arith::ArithDialect>();

        target.addIllegalOp<ContractOp>();

        mlir::RewritePatternSet patterns(ctx);
        patterns.add<ContractOpLowering>(ctx);

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
