#include "mshqc/compiler/JIT/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/TargetSelect.h"

namespace mshqc {
namespace compiler {

MshqcJIT::MshqcJIT(std::unique_ptr<mlir::ExecutionEngine> engine)
    : engine_(std::move(engine)) {}

llvm::Expected<std::unique_ptr<MshqcJIT>> MshqcJIT::create(mlir::ModuleOp module) {

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = mlir::makeOptimizingTransformer(
        3, 0, nullptr);

    mlir::registerLLVMDialectTranslation(*module->getContext());

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
