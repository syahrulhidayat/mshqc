 # ==============================================================================
 # Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
 #
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 #     http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 # ==============================================================================

# CMakeLists.txt untuk mshqc MLIR/LLVM Backend

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/mshqc/compiler/Dialect/MshqcOps.td)
mlir_tablegen(MshqcOps.h.inc -gen-op-decls)
mlir_tablegen(MshqcOps.cpp.inc -gen-op-defs)
mlir_tablegen(MshqcDialect.h.inc -gen-dialect-decls)
mlir_tablegen(MshqcDialect.cpp.inc -gen-dialect-defs)

add_custom_target(MshqcCompilerIncGen DEPENDS MshqcOps.h.inc MshqcOps.cpp.inc MshqcDialect.h.inc MshqcDialect.cpp.inc)


# Generate Pass headers
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/mshqc/compiler/Passes/Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Mshqc)
add_custom_target(MshqcPassIncGen DEPENDS Passes.h.inc)

add_library(mshqc_compiler STATIC
    Dialect/MshqcDialect.cc
    Dialect/MshqcOps.cc
    Passes/FusionPass.cc
    Passes/LowerToLinalg.cc
    Passes/TilingPass.cc
    Passes/BufferizePass.cc
    Passes/LowerToLLVMPass.cc
    Passes/VectorizePass.cc
    Frontend/GraphBuilder.cc
    JIT/ExecutionEngine.cc
)

add_dependencies(mshqc_compiler MshqcCompilerIncGen)
add_dependencies(mshqc_compiler MshqcPassIncGen)

target_link_libraries(mshqc_compiler
    PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRSCFUtils
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRLinalgTransforms
    MLIRTensorTransforms
    MLIRFuncTransforms
    MLIRLLVMDialect
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMCommonConversion
    MLIRSCFToControlFlow
    MLIRMemRefToLLVM
    MLIRMemRefTransforms
    MLIRFuncToLLVM
    MLIRControlFlowToLLVM
    MLIRArithToLLVM
    MLIRAffineToStandard
    MLIRTensorDialect
    MLIRVectorDialect
    MLIRVectorToLLVM
    MLIRAffineDialect
    MLIRExecutionEngine
    MLIRTargetLLVMIRExport
    ${llvm_libs}
)

# ==============================================================================
# Resolusi Header & TableGen Artefacts
# ==============================================================================
target_include_directories(mshqc_compiler PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
)
