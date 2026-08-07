#!/usr/bin/env python3
import os
from pathlib import Path

def inject_tablegen(base_dir: Path):
    """Mendefinisikan Mshqc Dialect dan operasi tensor dasar di TableGen."""
    td_path = base_dir / "include/mshqc/compiler/Dialect/MshqcOps.td"
    content = """// MLIR TableGen definitions for mshqc Dialect
#ifndef MSHQC_OPS
#define MSHQC_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// =============================================================================
// Dialect Definition
// =============================================================================
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

// =============================================================================
// Base Operation Class
// =============================================================================
class Mshqc_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mshqc_Dialect, mnemonic, traits>;

// =============================================================================
// Operations
// =============================================================================
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

#endif // MSHQC_OPS
"""
    with open(td_path, 'w') as f:
        f.write(content)
    print(f"[INJECTED] {td_path}")

def inject_headers(base_dir: Path):
    """Menulis header C++ untuk Dialect yang mengaitkan file auto-generated dari CMake."""
    h_path = base_dir / "include/mshqc/compiler/Dialect/MshqcDialect.h"
    content = """#ifndef MSHQC_COMPILER_DIALECT_MSHQCDIALECT_H_
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
"""
    with open(h_path, 'w') as f:
        f.write(content)
    print(f"[INJECTED] {h_path}")

def inject_sources(base_dir: Path):
    """Menulis source C++ untuk registrasi Dialect ke dalam MLIR Context."""
    cc_path = base_dir / "src/compiler/Dialect/MshqcDialect.cc"
    content = """#include "mshqc/compiler/Dialect/MshqcDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mshqc::compiler;

// Inklusi definisi Dialect yang di-generate oleh TableGen
#include "MshqcDialect.cpp.inc"

// Inisialisasi Dialect ke dalam Context MLIR
void MshqcDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "MshqcOps.cpp.inc"
    >();
}

// Inklusi implementasi operasi yang di-generate oleh TableGen
#define GET_OP_CLASSES
#include "MshqcOps.cpp.inc"
"""
    with open(cc_path, 'w') as f:
        f.write(content)
    print(f"[INJECTED] {cc_path}")

if __name__ == "__main__":
    base_directory = Path.cwd()
    print("Memulai injeksi definisi MLIR Dialect (TableGen & C++)...")
    inject_tablegen(base_directory)
    inject_headers(base_directory)
    inject_sources(base_directory)
    print("\n[SUCCESS] Dialect Mshqc telah didefinisikan.")
    print("Jalankan ulang 'cmake --build build -j' untuk memicu TableGen (mlir-tblgen) men-generate file *.inc.")