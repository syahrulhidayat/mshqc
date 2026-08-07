#ifndef MSHQC_COMPILER_DIALECT_MSHQCDIALECT_H_
#define MSHQC_COMPILER_DIALECT_MSHQCDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include deklarasi Dialect yang di-generate oleh TableGen
#include "MshqcDialect.h.inc"

// Deklarasi Operasi (Ops) yang di-generate oleh TableGen
#define GET_OP_CLASSES
#include "MshqcOps.h.inc"

#endif // MSHQC_COMPILER_DIALECT_MSHQCDIALECT_H_
