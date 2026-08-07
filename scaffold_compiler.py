#!/usr/bin/env python3
import os
from pathlib import Path

def setup_tools_directory(base_dir: Path):
    """Membangun tool CLI mshqc-opt untuk debugging IR terisolasi."""
    tools_dir = base_dir / "tools" / "mshqc-opt"
    tools_dir.mkdir(parents=True, exist_ok=True)

    # 1. mshqc-opt.cc
    opt_cc = tools_dir / "mshqc-opt.cc"
    opt_cc_content = """#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mshqc/compiler/Dialect/MshqcDialect.h"
#include "mshqc/compiler/Passes/Passes.h"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    
    // Register dialek standar LLVM/MLIR
    mlir::registerAllDialects(registry);
    
    // Register dialek khusus mshqc
    registry.insert<mshqc::compiler::MshqcDialect>();

    // Register passes kustom mshqc
    mshqc::compiler::registerPasses();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "MSHQC Modular Optimizer Driver\\n", registry));
}
"""
    with open(opt_cc, 'w') as f: f.write(opt_cc_content)

    # 2. tools/CMakeLists.txt
    tools_cmake = base_dir / "tools" / "CMakeLists.txt"
    tools_cmake_content = """# CMakeLists untuk tools MLIR
add_subdirectory(mshqc-opt)
"""
    with open(tools_cmake, 'w') as f: f.write(tools_cmake_content)

    # 3. tools/mshqc-opt/CMakeLists.txt
    opt_cmake = tools_dir / "CMakeLists.txt"
    opt_cmake_content = """get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(mshqc-opt mshqc-opt.cc)
llvm_update_compile_flags(mshqc-opt)

target_link_libraries(mshqc-opt PRIVATE
    mshqc_compiler
    ${dialect_libs}
    MLIROptLib
    MLIRPass
)
"""
    with open(opt_cmake, 'w') as f: f.write(opt_cmake_content)

def setup_pass_tablegen(base_dir: Path):
    """Memigrasikan deklarasi C++ Pass manual menjadi TableGen."""
    passes_td = base_dir / "include/mshqc/compiler/Passes/Passes.td"
    passes_td_content = """#ifndef MSHQC_PASSES
#define MSHQC_PASSES

include "mlir/Pass/PassBase.td"

def LowerToLinalg : Pass<"mshqc-lower-to-linalg", "::mlir::func::FuncOp"> {
    let summary = "Lower mshqc dialect to linalg operations.";
    let description = [{
        Transformasi analitik dari graf tingkat tinggi (mshqc.contract) 
        menuju representasi aljabar linear standar (linalg.generic) 
        sebelum dilakukan loop fusion dan bufferization.
    }];
    let constructor = "mshqc::compiler::createLowerToLinalgPass()";
}

#endif // MSHQC_PASSES
"""
    with open(passes_td, 'w') as f: f.write(passes_td_content)

    passes_h = base_dir / "include/mshqc/compiler/Passes/Passes.h"
    passes_h_content = """#ifndef MSHQC_COMPILER_PASSES_H_
#define MSHQC_COMPILER_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <memory>

namespace mshqc {
namespace compiler {

// Generate deklarasi pass dari TableGen
#define GEN_PASS_DECL
#include "mshqc/compiler/Passes/Passes.h.inc"

// Generate fungsi registrasi pass untuk mshqc-opt
#define GEN_PASS_REGISTRATION
#include "mshqc/compiler/Passes/Passes.h.inc"

} // namespace compiler
} // namespace mshqc

#endif // MSHQC_COMPILER_PASSES_H_
"""
    with open(passes_h, 'w') as f: f.write(passes_h_content)

    lowering_cc = base_dir / "src/compiler/Passes/LowerToLinalg.cc"
    lowering_cc_content = """#include "mshqc/compiler/Passes/Passes.h"
#include "mshqc/compiler/Dialect/MshqcDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mshqc {
namespace compiler {

// Konstruksi Base Class otomatis dari TableGen
#define GEN_PASS_DEF_LOWERTOLINALG
#include "mshqc/compiler/Passes/Passes.h.inc"

namespace {
struct ContractOpLowering : public mlir::OpRewritePattern<ContractOp> {
    using OpRewritePattern<ContractOp>::OpRewritePattern;
    mlir::LogicalResult matchAndRewrite(ContractOp op, mlir::PatternRewriter &rewriter) const override {
        // Blok lowering (linalg.generic) akan dieksekusi di fase berikutnya
        return mlir::success();
    }
};

struct LowerToLinalgPass : public impl::LowerToLinalgBase<LowerToLinalgPass> {
    void runOnOperation() override {
        mlir::ConversionTarget target(getContext());
        target.addLegalDialect<mlir::linalg::LinalgDialect, mlir::func::FuncDialect>();
        target.addIllegalOp<ContractOp>();

        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<ContractOpLowering>(&getContext());

        if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> createLowerToLinalgPass() {
    return std::make_unique<LowerToLinalgPass>();
}

} // namespace compiler
} // namespace mshqc
"""
    with open(lowering_cc, 'w') as f: f.write(lowering_cc_content)

def update_cmake_targets(base_dir: Path):
    """Merutekan ulang graph kompilasi CMake agar mencakup TableGen Passes dan direktori Tools."""
    root_cmake = base_dir / "CMakeLists.txt"
    if root_cmake.exists():
        with open(root_cmake, 'r') as f:
            content = f.read()
        if "add_subdirectory(tools)" not in content:
            content = content.replace("add_subdirectory(src/compiler)", "add_subdirectory(src/compiler)\nadd_subdirectory(tools)")
            with open(root_cmake, 'w') as f: f.write(content)

    compiler_cmake = base_dir / "src/compiler/CMakeLists.txt"
    if compiler_cmake.exists():
        with open(compiler_cmake, 'r') as f:
            content = f.read()
        
        # Tambahkan instruksi kompilasi TableGen untuk Pass
        if "Passes.h.inc" not in content:
            tablegen_pass = """
# Generate Pass headers
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/mshqc/compiler/Passes/Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Mshqc)
add_custom_target(MshqcPassIncGen DEPENDS Passes.h.inc)
add_dependencies(mshqc_compiler MshqcPassIncGen)
"""
            content = content.replace("add_library(mshqc_compiler", tablegen_pass + "\nadd_library(mshqc_compiler")
            with open(compiler_cmake, 'w') as f: f.write(content)

if __name__ == "__main__":
    base_directory = Path.cwd()
    print("[INFO] Mengeksekusi restrukturisasi arsitektur kompilator mshqc...")
    setup_tools_directory(base_directory)
    setup_pass_tablegen(base_directory)
    update_cmake_targets(base_directory)
    print("[SUCCESS] Direktori tools/ dan infrastruktur Passes.td berhasil diimplementasikan.")