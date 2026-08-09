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

    auto contractNode = mshqc::compiler::ContractOp::create(
        loc, resTensorType, lhsArg, rhsArg, builder.getStringAttr(einsum_eq)
    );

    builder.create<func::ReturnOp>(loc, mlir::ValueRange{contractNode.getResult()});
    return contractNode.getResult();
}

bool GraphBuilder::verifyGraph() {
    return succeeded(verify(module.get()));
}

}
}
