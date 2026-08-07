#!/usr/bin/env python3
import os
import re
import shutil
from pathlib import Path

def purge_integration_module(base_dir: Path):
    """Menghapus adapter integrasi MP-CI yang tidak lagi valid."""
    integration_src = base_dir / "src/integration"
    integration_inc = base_dir / "include/mshqc/integration"
    
    for d in [integration_src, integration_inc]:
        if d.exists() and d.is_dir():
            shutil.rmtree(d)
            print(f"[REMOVED] Direktori adapter {d} telah dihapus.")

    cmake_path = base_dir / "CMakeLists.txt"
    if cmake_path.exists():
        with open(cmake_path, 'r') as f:
            content = f.read()
        content = re.sub(r'file\(GLOB_RECURSE SRC_INTEGRATION[^\n]+\n', '', content)
        content = content.replace('${SRC_INTEGRATION} ', '').replace('${SRC_INTEGRATION}', '')
        with open(cmake_path, 'w') as f:
            f.write(content)
        print("[PATCHED] SRC_INTEGRATION dihapus dari graf CMake.")

def patch_foundation_headers(base_dir: Path):
    """Menghapus referensi objek CI dari modul foundation (opdm dan fcidump)."""
    foundation_inc = base_dir / "include/mshqc/foundation"
    foundation_src = base_dir / "src/foundation"
    
    files_to_patch = []
    if foundation_inc.exists():
        files_to_patch.extend(foundation_inc.rglob("*.h"))
    if foundation_src.exists():
        files_to_patch.extend(foundation_src.rglob("*.cc"))

    for filepath in files_to_patch:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        with open(filepath, 'w') as f:
            for line in lines:
                # Abaikan baris yang memuat instruksi import CI atau objek Determinant
                if "mshqc/ci/" in line or "ci::" in line or "Determinant" in line or "FCI" in line:
                    continue
                f.write(line)
        print(f"[CLEANED] File {filepath.name} telah dipurifikasi dari pointer CI.")

if __name__ == "__main__":
    base_directory = Path.cwd()
    print("[INFO] Memulai resolusi dangling dependencies...")
    purge_integration_module(base_directory)
    patch_foundation_headers(base_directory)
    print("[SUCCESS] Memori telah dibersihkan. Silakan push ulang ke GitHub Actions.")