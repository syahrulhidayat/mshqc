import os
import shutil
import re

def clean_dead_code():
    print("=== MSHQC DEAD CODE CLEANER ===")
    
    # 1. HAPUS FOLDER FISIK
    folders_to_delete = [
        "src/foundation",
        "include/mshqc/foundation"
    ]
    
    for folder in folders_to_delete:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"[OK] Direktori terhapus: {folder}")
        else:
            print(f"[SKIP] Direktori sudah tidak ada: {folder}")

    # 2. BERSIHKAN CMAKELISTS.TXT
    cmake_file = "CMakeLists.txt"
    if os.path.exists(cmake_file):
        with open(cmake_file, 'r') as f:
            cmake_content = f.read()
        
        # Hapus baris file(GLOB_RECURSE SRC_FOUNDATION ...)
        cmake_content = re.sub(r'file\(GLOB_RECURSE SRC_FOUNDATION[^\)]+\)\n?', '', cmake_content)
        
        # Hapus variabel ${SRC_FOUNDATION} dari list ALL_SOURCES
        cmake_content = cmake_content.replace('${SRC_FOUNDATION} ', '')
        cmake_content = cmake_content.replace('${SRC_FOUNDATION}', '')
        
        with open(cmake_file, 'w') as f:
            f.write(cmake_content)
        print(f"[OK] Variabel dibersihkan dari: {cmake_file}")

    # 3. BERSIHKAN PYTHON/BINDINGS.CC
    binding_file = "python/bindings.cc"
    if os.path.exists(binding_file):
        with open(binding_file, 'r') as f:
            binding_content = f.read()
            
        # Hapus include headers yang mati
        binding_content = re.sub(r'#include "mshqc/foundation/fcidump\.h"\n?', '', binding_content)
        binding_content = re.sub(r'#include "mshqc/foundation/wavefunction\.h"\n?', '', binding_content)
        
        # Hapus blok binding fungsi export_fcidump (regex multiline)
        fcidump_pattern = r'\s*m\.def\("export_fcidump"[\s\S]*?"Export SCF and Integral results to standard FCIDUMP format"\);'
        binding_content = re.sub(fcidump_pattern, '', binding_content)
        
        with open(binding_file, 'w') as f:
            f.write(binding_content)
        print(f"[OK] Bindings dihapus dari: {binding_file}")

    # 4. HAPUS CACHE BUILD
    cache_folders = ["build", "dist", "python/mshqc.egg-info"]
    for folder in cache_folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"[OK] Cache terhapus: {folder}")

if __name__ == "__main__":
    clean_dead_code()
    print("===============================")
    print("Pembersihan selesai! Codebase Anda sekarang lebih ringan.")
    print("Silakan jalankan ulang: python setup.py install")