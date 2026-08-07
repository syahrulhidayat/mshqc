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

    auto funcType = builder.getFunctionType(std::nullopt, std::nullopt);
    auto funcOp = builder.create<func::FuncOp>(loc, functionName, funcType);

    Block* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);
}

mlir::Value GraphBuilder::emitContractOp(const std::vector<int64_t>& lhsShape,
                                         const std::vector<int64_t>& rhsShape,
                                         const std::string& einsum_eq) {
    Location loc = builder.getUnknownLoc();

    auto f64Type = builder.getF64Type();
    auto lhsTensorType = RankedTensorType::get(lhsShape, f64Type);
    auto rhsTensorType = RankedTensorType::get(rhsShape, f64Type);

    Value dummyLhs = builder.create<mshqc::compiler::ContractOp>(loc, lhsTensorType, Value(), Value(), builder.getStringAttr(einsum_eq)).getResult();

    return dummyLhs;
}

bool GraphBuilder::verifyGraph() {
    return succeeded(verify(module.get()));
}

}
}
