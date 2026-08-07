#!/usr/bin/env python3
from pathlib import Path

def patch_compiler_includes(base_dir: Path):
    """Menginjeksi ruting header absolut untuk modul MLIR Compiler."""
    cmake_path = base_dir / "src/compiler/CMakeLists.txt"
    
    if not cmake_path.exists():
        print(f"[ERROR] {cmake_path} tidak ditemukan.")
        return

    with open(cmake_path, 'r') as f:
        content = f.read()

    # Injeksi target_include_directories jika belum ada
    if "target_include_directories(mshqc_compiler" not in content:
        injection = """
# ==============================================================================
# Resolusi Header & TableGen Artefacts
# ==============================================================================
target_include_directories(mshqc_compiler PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
)
"""
        with open(cmake_path, 'w') as f:
            f.write(content + injection)
        print("[PATCHED] src/compiler/CMakeLists.txt telah dikalibrasi. Resolusi header terbuka.")
    else:
        print("[INFO] Ruting header sudah terkonfigurasi.")

if __name__ == "__main__":
    base_directory = Path.cwd()
    print("[INFO] Mengeksekusi penambalan memori include pada CMake...")
    patch_compiler_includes(base_directory)
    print("[SUCCESS] Silakan commit dan push ulang ke GitHub Actions.")