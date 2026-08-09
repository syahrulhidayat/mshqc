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

#include "mshqc/compiler/Frontend/GraphBuilder.h"
#include "mshqc/compiler/Dialect/MshqcDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;

namespace mshqc {
namespace compiler {

GraphBuilder::GraphBuilder() : builder(&context) {
    context.getOrLoadDialect<MshqcDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
}

GraphBuilder::~GraphBuilder() = default;

void GraphBuilder::initializeModule(const std::string& functionName) {
    Location loc = builder.getUnknownLoc();
    module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module->getBody());
}

mlir::Value GraphBuilder::emitContractOp(const std::vector<int64_t>& lhsShape,
                                         const std::vector<int64_t>& rhsShape,
                                         const std::string& einsum_eq) {
    Location loc = builder.getUnknownLoc();
    auto f64Type = builder.getF64Type();

    auto lhsTensorType = RankedTensorType::get(lhsShape, f64Type);
    auto rhsTensorType = RankedTensorType::get(rhsShape, f64Type);
    auto resTensorType = RankedTensorType::get({lhsShape[0], lhsShape[1], rhsShape[0], rhsShape[1]}, f64Type);

    auto funcType = builder.getFunctionType({lhsTensorType, rhsTensorType}, {resTensorType});
    auto funcOp = builder.create<func::FuncOp>(loc, builder.getStringAttr("contract_kernel"), funcType);

    Block* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);

    Value lhsArg = entryBlock->getArgument(0);
    Value rhsArg = entryBlock->getArgument(1);

    auto contractNode = builder.create<mshqc::compiler::ContractOp>(
        loc, resTensorType, lhsArg, rhsArg, builder.getStringAttr(einsum_eq)
    );

    builder.create<func::ReturnOp>(loc, contractNode.getResult());
    return contractNode.getResult();
}

bool GraphBuilder::verifyGraph() {
    return succeeded(verify(module.get()));
}

}
}
