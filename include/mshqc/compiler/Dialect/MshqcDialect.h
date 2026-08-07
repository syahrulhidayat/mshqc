#ifndef MSHQC_COMPILER_DIALECT_MSHQCDIALECT_H_
#define MSHQC_COMPILER_DIALECT_MSHQCDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "MshqcDialect.h.inc"

#define GET_OP_CLASSES
#include "MshqcOps.h.inc"

#endif
