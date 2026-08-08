#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mshqc/compiler/Dialect/MshqcDialect.h"
#include "mshqc/compiler/Passes/Passes.h"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;

    mlir::registerAllDialects(registry);
    mlir::registerAllPasses();

    registry.insert<mshqc::compiler::MshqcDialect>();

    mshqc::compiler::registerMshqcPasses();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "MSHQC Modular Optimizer Driver\n", registry));
}
