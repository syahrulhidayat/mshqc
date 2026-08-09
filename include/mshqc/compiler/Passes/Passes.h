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

#ifndef MSHQC_COMPILER_PASSES_H_
#define MSHQC_COMPILER_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <memory>

namespace mshqc {
namespace compiler {

std::unique_ptr<mlir::Pass> createLowerToLinalgPass();

std::unique_ptr<mlir::Pass> createBufferizePass();

#define GEN_PASS_DECL
#include "Passes.h.inc"

std::unique_ptr<mlir::Pass> createLinalgTilingPass();

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

}
}

#endif
