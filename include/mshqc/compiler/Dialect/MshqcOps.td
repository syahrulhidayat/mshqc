#ifndef MSHQC_OPS
#define MSHQC_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Mshqc_Dialect : Dialect {
    let name = "mshqc";
    let summary = "A high-performance dialect for Quantum Chemistry Tensor Contractions";
    let description = [{
        Dialect khusus untuk merepresentasikan persamaan kimia kuantum (SCF, MP2, MP3, CCSD).
        Operasi pada dialect ini dirancang untuk mempertahankan informasi semantik
        tingkat tinggi guna memungkinkan optimasi loop fusion dan manajemen cache locality
        secara analitik sebelum diturunkan (lowered) ke Linalg/Affine.
    }];
    let cppNamespace = "::mshqc::compiler";
}

class Mshqc_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mshqc_Dialect, mnemonic, traits>;

def Mshqc_ContractOp : Mshqc_Op<"contract", [Pure]> {
    let summary = "Quantum chemistry tensor contraction operation";
    let description = [{
        Mewakili operasi einsum tingkat tinggi yang mengeksekusi kontraksi antar
        tensor multidimensi. Menggantikan abstraksi TBLIS/Eigen murni untuk
        memungkinkan analisis cost-model pada memory footprint.
    }];

    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, StrAttr:$einsum_eq);
    let results = (outs AnyTensor:$result);
}

#endif
