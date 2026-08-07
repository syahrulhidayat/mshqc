#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mshqc/compiler/Dialect/MshqcDialect.h"
#include "mshqc/compiler/Passes/Passes.h"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    
    // Register dialek standar LLVM/MLIR
    mlir::registerAllDialects(registry);
    
    // Register dialek khusus mshqc
    registry.insert<mshqc::compiler::MshqcDialect>();

    // Register passes kustom mshqc
    mshqc::compiler::registerPasses();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "MSHQC Modular Optimizer Driver\n", registry));
}
