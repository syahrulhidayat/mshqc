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

#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

}
}

#endif
