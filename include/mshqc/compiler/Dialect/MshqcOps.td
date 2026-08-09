 // ==============================================================================
 // Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
 //
 // Licensed under the Apache License, Version 2.0 (the "License");
 // you may not use this file except in compliance with the License.
 // You may obtain a copy of the License at
 //
 //     http://www.apache.org/licenses/LICENSE-2.0
 //
 // Unless required by applicable law or agreed to in writing, software
 // distributed under the License is distributed on an "AS IS" BASIS,
 // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 // See the License for the specific language governing permissions and
 // limitations under the License.
 // ==============================================================================

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
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}

#endif
