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
