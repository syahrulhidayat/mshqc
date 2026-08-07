# CMakeLists.txt untuk mshqc MLIR/LLVM Backend

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/mshqc/compiler/Dialect/MshqcOps.td)
mlir_tablegen(MshqcOps.h.inc -gen-op-decls)
mlir_tablegen(MshqcOps.cpp.inc -gen-op-defs)
mlir_tablegen(MshqcDialect.h.inc -gen-dialect-decls)
mlir_tablegen(MshqcDialect.cpp.inc -gen-dialect-defs)

add_custom_target(MshqcCompilerIncGen DEPENDS MshqcOps.h.inc MshqcOps.cpp.inc MshqcDialect.h.inc MshqcDialect.cpp.inc)

add_library(mshqc_compiler STATIC
    Dialect/MshqcDialect.cc
    Dialect/MshqcOps.cc
    Passes/FusionPass.cc
    Passes/LowerToLinalg.cc
    Frontend/GraphBuilder.cc
    JIT/ExecutionEngine.cc
)

add_dependencies(mshqc_compiler MshqcCompilerIncGen)

target_link_libraries(mshqc_compiler
    PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRAffineDialect
    MLIRExecutionEngine
    MLIRTargetLLVMIRExport
    ${llvm_libs}
)
