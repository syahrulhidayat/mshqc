import os
import shutil
from pathlib import Path

# 1. PETA PERPINDAHAN FILE (Dari -> Ke)
# Disesuaikan dengan struktur yang ada di src/
FILE_MOVES = {
    "include/mshqc/molecule.h": "include/mshqc/core/molecule.h",
    
    # SCF & DIIS
    "include/mshqc/scf.h": "include/mshqc/scf/scf.h",
    "include/mshqc/diis.h": "include/mshqc/scf/diis.h",
    "include/mshqc/soscf.h": "include/mshqc/scf/soscf.h",
    "include/mshqc/sad.h": "include/mshqc/scf/sad.h",
    
    # MP2
    "include/mshqc/mp2.h": "include/mshqc/mp2/mp2.h",
    "include/mshqc/dfmp2.h": "include/mshqc/mp2/dfmp2.h",
    "include/mshqc/ump2.h": "include/mshqc/mp2/ump2.h",
    
    # MP3
    "include/mshqc/mp3.h": "include/mshqc/mp3/mp3.h",
    "include/mshqc/omp3.h": "include/mshqc/mp3/omp3.h",
    "include/mshqc/ump3.h": "include/mshqc/mp3/ump3.h",
    "include/mshqc/ump3_kernels.h": "include/mshqc/mp3/ump3_kernels.h",
    "include/mshqc/ump3_memory.h": "include/mshqc/mp3/ump3_memory.h",
    
    # Integrals
    "include/mshqc/integrals.h": "include/mshqc/ints/integrals.h",
    
    # Hierarki & lainnya (Bisa kamu tambahkan sendiri jika ada yang terlewat)
    "include/mshqc/mpn_hierarchy.h": "include/mshqc/mp/mpn_hierarchy.h",
    "include/mshqc/spherical_integration.h": "include/mshqc/integrals/spherical_integration.h",
    "include/mshqc/spherical_transformer.h": "include/mshqc/integrals/spherical_transformer.h",
}

def main():
    base_dir = Path.cwd()
    
    # Buat mapping untuk string #include (contoh: "mshqc/scf.h" -> "mshqc/scf/scf.h")
    include_map = {}
    for old_path, new_path in FILE_MOVES.items():
        old_inc = old_path.replace("include/", "")
        new_inc = new_path.replace("include/", "")
        include_map[old_inc] = new_inc

    # 2. PINDAHKAN FILE
    print("Mulai memindahkan file...")
    for old_rel, new_rel in FILE_MOVES.items():
        old_path = base_dir / old_rel
        new_path = base_dir / new_rel
        
        if old_path.exists():
            # Pastikan direktori tujuan ada
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))
            print(f"✅ Dipindahkan: {old_rel} -> {new_rel}")
        else:
            print(f"⚠️ Dilewati: {old_rel} (File tidak ditemukan)")

    # 3. UPDATE #INCLUDE DI SEMUA FILE .cc DAN .h
    print("\nMulai memperbarui #include...")
    
    # Cari semua file di include/ dan src/ dan python/bindings.cc
    target_extensions = ('.h', '.cc', '.cpp', '.hpp')
    files_to_check = []
    
    for folder in ['include', 'src', 'python']:
        folder_path = base_dir / folder
        if folder_path.exists():
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(target_extensions):
                        files_to_check.append(Path(root) / file)

    updated_count = 0
    for file_path in files_to_check:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            for old_inc, new_inc in include_map.items():
                # Replace string include di dalam file
                # Ini akan mengubah #include "mshqc/scf.h" menjadi #include "mshqc/scf/scf.h"
                content = content.replace(old_inc, new_inc)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"🔄 Diperbarui: {file_path.relative_to(base_dir)}")
                updated_count += 1
                
        except Exception as e:
            print(f"❌ Gagal membaca/menulis {file_path.relative_to(base_dir)}: {e}")

    print(f"\nSelesai! {updated_count} file telah diperbarui #include-nya.")

if __name__ == "__main__":
    main()