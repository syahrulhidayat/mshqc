#!/usr/bin/env python3
import re
from pathlib import Path

def patch_mlir_api(base_dir: Path):
    """Menyesuaikan sintaks Mshqc MLIR C++ dengan LLVM API modern."""
    
    # 1. Refaktor LowerToLinalg.cc (Memperbaiki error casting RTTI)
    linalg_path = base_dir / "src/compiler/Passes/LowerToLinalg.cc"
    if linalg_path.exists():
        with open(linalg_path, 'r') as f:
            content = f.read()
        
        # Ganti metode .cast internal dengan global llvm::cast
        content = re.sub(
            r'lhs\.getType\(\)\.cast<ShapedType>\(\)', 
            r'::llvm::cast<ShapedType>(lhs.getType())', 
            content
        )
        content = re.sub(
            r'rhs\.getType\(\)\.cast<ShapedType>\(\)', 
            r'::llvm::cast<ShapedType>(rhs.getType())', 
            content
        )
        # ContractOp mengambil getResult() sebelum getType()
        content = re.sub(
            r'op\.getType\(\)\.cast<ShapedType>\(\)', 
            r'::llvm::cast<ShapedType>(op.getResult().getType())', 
            content
        )
        
        with open(linalg_path, 'w') as f:
            f.write(content)
        print(f"[PATCHED] llvm::cast API diterapkan pada {linalg_path.name}")

    # 2. Refaktor GraphBuilder.cc (Memperbaiki error konversi tipe)
    builder_path = base_dir / "src/compiler/Frontend/GraphBuilder.cc"
    if builder_path.exists():
        with open(builder_path, 'r') as f:
            content = f.read()
        
        # Ganti std::nullopt dengan TypeRange{} untuk representasi arity nol
        content = content.replace(
            "builder.getFunctionType(std::nullopt, std::nullopt)", 
            "builder.getFunctionType(TypeRange{}, TypeRange{})"
        )
        
        with open(builder_path, 'w') as f:
            f.write(content)
        print(f"[PATCHED] Resolusi mlir::TypeRange diterapkan pada {builder_path.name}")

if __name__ == "__main__":
    base_directory = Path.cwd()
    print("[INFO] Memulai modernisasi MLIR C++ API...")
    patch_mlir_api(base_directory)
    print("[SUCCESS] Skrip selesai. Kode siap dikompilasi.")