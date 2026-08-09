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

#include "mshqc/compiler/JIT/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"

namespace mshqc {
namespace compiler {

MshqcJIT::MshqcJIT(std::unique_ptr<mlir::ExecutionEngine> engine)
    : engine_(std::move(engine)) {}

llvm::Expected<std::unique_ptr<MshqcJIT>> MshqcJIT::create(mlir::ModuleOp module) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    mlir::ExecutionEngineOptions engineOptions;

    std::vector<llvm::StringRef> sharedLibs = {"libomp.so"};
    engineOptions.sharedLibPaths = sharedLibs;
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
    engineOptions.transformer = mlir::makeOptimizingTransformer(3, 0, nullptr);

    mlir::registerBuiltinDialectTranslation(*module->getContext());
    mlir::registerLLVMDialectTranslation(*module->getContext());
    mlir::registerOpenMPDialectTranslation(*module->getContext());

    auto engine = mlir::ExecutionEngine::create(module, engineOptions);
    if (!engine) {
        return engine.takeError();
    }

    return std::unique_ptr<MshqcJIT>(new MshqcJIT(std::move(*engine)));
}

llvm::Error MshqcJIT::invoke(llvm::StringRef name, llvm::MutableArrayRef<void *> args) {
    auto expectedFPtr = engine_->lookupPacked(name);
    if (!expectedFPtr) {
        return expectedFPtr.takeError();
    }

    void (*fn)(void **) = *expectedFPtr;
    fn(args.data());

    return llvm::Error::success();
}

}
}