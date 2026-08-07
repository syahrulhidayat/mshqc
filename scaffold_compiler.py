#!/usr/bin/env python3
import os
import re
import shutil
from pathlib import Path

def prune_directories(base_dir: Path):
    """Menghapus direktori sumber dan header ci dan mcscf secara rekursif."""
    dirs_to_remove = [
        base_dir / "src/ci",
        base_dir / "src/mcscf",
        base_dir / "include/mshqc/ci",
        base_dir / "include/mshqc/mcscf"
    ]
    for d in dirs_to_remove:
        if d.exists() and d.is_dir():
            shutil.rmtree(d)
            print(f"[REMOVED] Direktori {d} telah dihapus dari sistem memori.")

def patch_cmake(base_dir: Path):
    """Menghapus referensi SRC_CI dan SRC_MCSCF dari CMakeLists.txt."""
    cmake_path = base_dir / "CMakeLists.txt"
    if not cmake_path.exists():
        return

    with open(cmake_path, 'r') as f:
        content = f.read()

    # Menghapus instruksi GLOB_RECURSE
    content = re.sub(r'file\(GLOB_RECURSE SRC_MCSCF[^\n]+\n', '', content)
    content = re.sub(r'file\(GLOB_RECURSE SRC_CI[^\n]+\n', '', content)
    
    # Menghapus injeksi variabel dari array ALL_SOURCES
    content = content.replace('${SRC_MCSCF} ', '').replace('${SRC_CI} ', '')
    content = content.replace('${SRC_MCSCF}', '').replace('${SRC_CI}', '')
    
    with open(cmake_path, 'w') as f:
        f.write(content)
    print(f"[PATCHED] {cmake_path} berhasil direstrukturisasi.")

def patch_bindings(base_dir: Path):
    """Menghapus header C++ dan blok nanobind untuk modul ci dan mcscf dari bindings.cc."""
    bindings_path = base_dir / "python/bindings.cc"
    if not bindings_path.exists():
        return

    with open(bindings_path, 'r') as f:
        content = f.read()

    # 1. Hapus direktif #include untuk modul ci dan mcscf
    content = re.sub(r'#include "mshqc/(ci|mcscf)/.*?\n', '', content)
    
    # 2. Hapus deklarasi namespace
    content = re.sub(r'using namespace mshqc::mcscf;\n', '', content)

    # 3. Ekstraksi dan pemotongan blok Python bindings secara presisi
    # Batas awal pemotongan: blok CI
    match_ci_start = re.search(r'nb::class_<ci::Determinant>\(m, "Determinant"\)', content)
    # Batas akhir pemotongan: blok Gradient (harus dipertahankan)
    match_grad_start = re.search(r'nb::class_<gradient::GradientResult>\(m, "GradientResult"\)', content)

    if match_ci_start and match_grad_start:
        content = content[:match_ci_start.start()] + content[match_grad_start.start():]
    
    # 4. Hapus secara eksplisit objek PT2Amplitudes di akhir file (karena terikat pada modul CASPT2)
    content = re.sub(r'\s*nb::class_<PT2Amplitudes>\(m, "PT2Amplitudes"\).*?t2_semi2\);', '', content, flags=re.DOTALL)

    with open(bindings_path, 'w') as f:
        f.write(content)
    print(f"[PATCHED] {bindings_path} telah dibersihkan dari pointer yang putus.")

if __name__ == "__main__":
    base_directory = Path.cwd()
    print(f"[INFO] Memulai proses purifikasi direktori dan linking di {base_directory}...")
    
    prune_directories(base_directory)
    patch_cmake(base_directory)
    patch_bindings(base_directory)
    
    print("[SUCCESS] Operasi eradikasi modul selesai. Pipeline C++ siap dikompilasi ulang.")