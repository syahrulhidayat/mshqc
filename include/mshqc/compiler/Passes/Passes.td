#ifndef MSHQC_PASSES
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
