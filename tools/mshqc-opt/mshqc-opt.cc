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
