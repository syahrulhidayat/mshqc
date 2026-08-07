#!/usr/bin/env python3
from pathlib import Path
import re

def refactor_graph_builder(base_dir: Path):
    """Menyelaraskan GraphBuilder agar menghasilkan parameter fungsi (Block Arguments) yang valid di MLIR."""
    gb_path = base_dir / "src/compiler/Frontend/GraphBuilder.cc"
    if not gb_path.exists(): return

    content = """#include "mshqc/compiler/Frontend/GraphBuilder.h"
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

    // Membuat fungsi dinamis sesuai ukuran tensor untuk menampung argumen memori
    auto funcType = builder.getFunctionType({lhsTensorType, rhsTensorType}, {resTensorType});
    auto funcOp = builder.create<func::FuncOp>(loc, "contract_kernel", funcType);
    
    Block* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);

    // Menggunakan Block Arguments sebagai nilai aktual (bukan pointer kosong)
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

} // namespace compiler
} // namespace mshqc
"""
    with open(gb_path, 'w') as f:
        f.write(content)
    print(f"[REFACTORED] {gb_path.name} dioptimasi dengan pemetaan Block Argument.")

def inject_mp2_shadow_execution(base_dir: Path):
    """Menyisipkan Graph Builder ke dalam logika komputasi amplitudo MP2."""
    mp2_path = base_dir / "src/mp2/mp2.cc"
    if not mp2_path.exists(): return

    with open(mp2_path, 'r') as f:
        content = f.read()

    # Injeksi header jika belum ada
    if "GraphBuilder.h" not in content:
        content = content.replace(
            '#include "mshqc/mp2/mp2.h"',
            '#include "mshqc/mp2/mp2.h"\n#include "mshqc/compiler/Frontend/GraphBuilder.h"'
        )

    # Deteksi blok komputasi MP2 untuk injeksi AST MLIR
    target_block = "void RMP2::compute_amplitudes_and_energy() {"
    if target_block in content and "GraphBuilder mlir_builder;" not in content:
        mlir_logic = """
    // Konstruksi MLIR AST (Shadow Execution)
    #ifdef MSHQC_ENABLE_MLIR
    mshqc::compiler::GraphBuilder mlir_builder;
    mlir_builder.initializeModule("rmp2_amplitude_module");
    int64_t n_aux = B_ia_P_alpha_.cols();
    std::vector<int64_t> lhs_shape = {nocc_a_, nvir_a_, n_aux};
    std::vector<int64_t> rhs_shape = {nocc_a_, nvir_a_, n_aux};
    mlir_builder.emitContractOp(lhs_shape, rhs_shape, "iaP,jbP->iajb");
    if (!mlir_builder.verifyGraph()) {
        std::cerr << "[CRITICAL] MLIR Graph Semantic Verification Failed.\\n";
    }
    #endif
"""
        content = content.replace(target_block, target_block + mlir_logic)
        with open(mp2_path, 'w') as f:
            f.write(content)
        print(f"[REFACTORED] Shadow execution AST MLIR ditambahkan pada {mp2_path.name}.")

if __name__ == "__main__":
    base_directory = Path.cwd()
    print("[INFO] Memulai sinkronisasi Frontend MLIR...")
    refactor_graph_builder(base_directory)
    inject_mp2_shadow_execution(base_directory)
    print("[SUCCESS] Kompilator siap memproses C++ AST ke format DAG.")