#ifndef MSHQC_COMPILER_PASSES_H_
#define MSHQC_COMPILER_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mshqc {
namespace compiler {

// Deklarasi fungsi pembuat pass (Lowering dari Mshqc ke Linalg)
std::unique_ptr<mlir::Pass> createLowerToLinalgPass();

} // namespace compiler
} // namespace mshqc

#endif // MSHQC_COMPILER_PASSES_H_
