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

def LinalgTiling : Pass<"mshqc-linalg-tiling", "::mlir::func::FuncOp"> {
    let summary = "Apply loop tiling to Linalg operations for L1/L2 cache locality.";
    let constructor = "mshqc::compiler::createLinalgTilingPass()";
    let options = [
        ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes for loop nest">
    ];
}

def LowerToLLVM : Pass<"mshqc-lower-to-llvm", "::mlir::ModuleOp"> {
    let summary = "Lower all intermediate dialects (Linalg, SCF, MemRef, Arith, Func) to LLVM IR dialect.";
    let constructor = "mshqc::compiler::createLowerToLLVMPass()";
}

def Bufferize : Pass<"mshqc-bufferize", "::mlir::ModuleOp"> {
    let summary = "Menerjemahkan semantik Tensor abstrak menjadi semantik MemRef fisik (Pointer).";
    let constructor = "mshqc::compiler::createBufferizePass()";
}