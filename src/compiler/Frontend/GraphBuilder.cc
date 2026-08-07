#include "mshqc/compiler/Frontend/GraphBuilder.h"
#include "mshqc/compiler/Dialect/MshqcDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;

namespace mshqc {
namespace compiler {

GraphBuilder::GraphBuilder() : builder(&context) {
    // Memuat Mshqc Dialect ke dalam memori kompilator
    context.getOrLoadDialect<MshqcDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
}

GraphBuilder::~GraphBuilder() = default;

void GraphBuilder::initializeModule(const std::string& functionName) {
    Location loc = builder.getUnknownLoc();
    module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module->getBody());

    // Membangun prototipe fungsi (sementara menggunakan arity nol untuk scaffolding)
    auto funcType = builder.getFunctionType(std::nullopt, std::nullopt);
    auto funcOp = builder.create<func::FuncOp>(loc, functionName, funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);
}

mlir::Value GraphBuilder::emitContractOp(const std::vector<int64_t>& lhsShape,
                                         const std::vector<int64_t>& rhsShape,
                                         const std::string& einsum_eq) {
    Location loc = builder.getUnknownLoc();
    
    // Alokasi tipe tensor memori
    auto f64Type = builder.getF64Type();
    auto lhsTensorType = RankedTensorType::get(lhsShape, f64Type);
    auto rhsTensorType = RankedTensorType::get(rhsShape, f64Type);
    
    // (Dummy) Nilai tensor kosong untuk konstruksi graf.
    // Di tahap produksi, ini akan dipetakan langsung dari memori Eigen via pointer.
    Value dummyLhs = builder.create<mshqc::compiler::ContractOp>(loc, lhsTensorType, Value(), Value(), builder.getStringAttr(einsum_eq)).getResult();
    
    return dummyLhs; // Representasi nilai hasil return
}

bool GraphBuilder::verifyGraph() {
    return succeeded(verify(module.get()));
}

} // namespace compiler
} // namespace mshqc
