#include "mshqc/compiler/Dialect/MshqcDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mshqc::compiler;

// Inklusi definisi Dialect yang di-generate oleh TableGen
#include "MshqcDialect.cpp.inc"

// Inisialisasi Dialect ke dalam Context MLIR
void MshqcDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "MshqcOps.cpp.inc"
    >();
}

// Inklusi implementasi operasi yang di-generate oleh TableGen
#define GET_OP_CLASSES
#include "MshqcOps.cpp.inc"
