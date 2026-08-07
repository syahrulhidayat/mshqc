#!/usr/bin/env python3
import os
import sys
import re
from pathlib import Path

def create_directories_and_stubs(base_dir: Path):
    """Membangun arsitektur direktori compiler dan file header/source kosong dengan include guards."""
    
    dirs = {
        "include/mshqc/compiler/Dialect": ["MshqcDialect.h", "MshqcOps.td"],
        "include/mshqc/compiler/Passes": ["PassDetail.h", "Passes.h"],
        "include/mshqc/compiler/JIT": ["ExecutionEngine.h"],
        "src/compiler/Dialect": ["MshqcDialect.cc", "MshqcOps.cc"],
        "src/compiler/Passes": ["FusionPass.cc", "LowerToLinalg.cc"],
        "src/compiler/JIT": ["ExecutionEngine.cc"]
    }

    print("[INFO] Membangun struktur direktori dan file stub...")
    for dir_path, files in dirs.items():
        full_dir = base_dir / dir_path
        full_dir.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            file_path = full_dir / file
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    if file.endswith('.h'):
                        guard = f"MSHQC_COMPILER_{file.replace('.', '_').upper()}_"
                        f.write(f"#ifndef {guard}\n#define {guard}\n\n#endif // {guard}\n")
                    elif file.endswith('.td'):
                        f.write("// MLIR TableGen definitions for mshqc Dialect\n")
                    else:
                        f.write(f"// TODO: Implementasi modul {file}\n")
                print(f"  [CREATED] {file_path}")
            else:
                print(f"  [EXISTS]  {file_path}")

def inject_root_cmake(base_dir: Path):
    """Menginjeksi konfigurasi LLVM/MLIR ke dalam root CMakeLists.txt tanpa duplikasi."""
    cmake_path = base_dir / "CMakeLists.txt"
    
    if not cmake_path.exists():
        print("[ERROR] Root CMakeLists.txt tidak ditemukan.")
        sys.exit(1)

    with open(cmake_path, 'r') as f:
        content = f.read()

    if "find_package(LLVM" in content and "find_package(MLIR" in content:
        print("[INFO] Konfigurasi LLVM/MLIR sudah ada di root CMakeLists.txt. Melewati injeksi.")
        return

    injection_block = """
# ==============================================================================
# LLVM and MLIR Compiler Infrastructure (Injected Auto)
# ==============================================================================
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs support core orcjit native)

add_subdirectory(src/compiler)
"""
    # Mencari titik injeksi yang aman (sebelum add_subdirectory(src) jika ada, atau di akhir file)
    if "add_subdirectory(src)" in content:
        content = content.replace("add_subdirectory(src)", injection_block + "\nadd_subdirectory(src)")
    else:
        content += "\n" + injection_block

    with open(cmake_path, 'w') as f:
        f.write(content)
    print("[MODIFIED] Root CMakeLists.txt berhasil diinjeksi.")

def create_compiler_cmake(base_dir: Path):
    """Membuat CMakeLists.txt untuk target mshqc_compiler."""
    compiler_cmake_path = base_dir / "src/compiler/CMakeLists.txt"
    
    if compiler_cmake_path.exists():
        print("[INFO] src/compiler/CMakeLists.txt sudah ada. Melewati pembuatan.")
        return

    cmake_content = """# CMakeLists.txt untuk mshqc MLIR/LLVM Backend

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
    JIT/ExecutionEngine.cc
)

add_dependencies(mshqc_compiler MshqcCompilerIncGen)

target_link_libraries(mshqc_compiler
    PUBLIC
    MLIRIR
    MLIRLinalgDialect
    MLIRAffineDialect
    MLIRExecutionEngine
    MLIRTargetLLVMIRExport
    ${llvm_libs}
)
"""
    with open(compiler_cmake_path, 'w') as f:
        f.write(cmake_content)
    print("[CREATED] src/compiler/CMakeLists.txt")

if __name__ == "__main__":
    base_directory = Path.cwd()
    print(f"Memulai injeksi arsitektur ML Compiler di: {base_directory}")
    
    create_directories_and_stubs(base_directory)
    inject_root_cmake(base_directory)
    create_compiler_cmake(base_directory)
    
    print("\n[SUCCESS] Proses scaffolding selesai.")
    print("Jalankan 'cmake -B build' untuk memverifikasi resolusi dependensi LLVM/MLIR.")