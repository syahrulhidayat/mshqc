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

#ifndef MSHQC_COMPILER_JIT_EXECUTIONENGINE_H
#define MSHQC_COMPILER_JIT_EXECUTIONENGINE_H

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <string>

namespace mshqc {
namespace compiler {

class MshqcJIT {
public:

    static llvm::Expected<std::unique_ptr<MshqcJIT>> create(mlir::ModuleOp module);

    llvm::Error invoke(llvm::StringRef name, llvm::MutableArrayRef<void *> args);

private:
    MshqcJIT(std::unique_ptr<mlir::ExecutionEngine> engine);
    std::unique_ptr<mlir::ExecutionEngine> engine_;
};

}
}

#endif
