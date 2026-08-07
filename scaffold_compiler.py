#!/usr/bin/env python3
from pathlib import Path

def patch_dialect_headers(base_dir: Path):
    """Menambahkan MLIR Builtin Types dan Interface ke dalam Dialect Header."""
    h_path = base_dir / "include/mshqc/compiler/Dialect/MshqcDialect.h"
    
    if not h_path.exists():
        print(f"[ERROR] File {h_path} tidak ditemukan pada arsitektur direktori.")
        return

    with open(h_path, 'r') as f:
        content = f.read()

    if "BuiltinTypes.h" not in content:
        # Blok header MLIR yang wajib ada sebelum file .inc di-load
        missing_headers = """#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
"""
        # Injeksi persis sebelum '#include "MshqcDialect.h.inc"'
        target_str = '#include "MshqcDialect.h.inc"'
        content = content.replace(target_str, missing_headers + "\n" + target_str)
        
        with open(h_path, 'w') as f:
            f.write(content)
        print("[PATCHED] Header MLIR tingkat rendah telah ditambahkan ke MshqcDialect.h.")
    else:
        print("[INFO] Header MLIR sudah tersedia.")

if __name__ == "__main__":
    base_directory = Path.cwd()
    print("[INFO] Memulai sinkronisasi pointer header MLIR...")
    patch_dialect_headers(base_directory)
    print("[SUCCESS] Silakan commit dan evaluasi ulang pipeline kompilator.")