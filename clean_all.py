import os
import shutil
import re

def remove_comments_and_format(file_path):
    # Baca isi file
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Regex cerdas: Cocokkan string literal DULU agar tidak ikut terhapus, 
    # baru cocokkan komentar block /* ... */ dan komentar baris // ...
    pattern = r'("(?:\\[\s\S]|[^"])*"|\'(?:\\[\s\S]|[^\'])*\')|(/\*[\s\S]*?\*/|//[^\r\n]*)'
    regex = re.compile(pattern)
    
    def replacer(match):
        if match.group(2) is not None:
            # Jika yang cocok adalah grup 2 (komentar), hapus (return string kosong)
            return "" 
        else:
            # Jika yang cocok adalah grup 1 (teks string), biarkan saja
            return match.group(1) 

    # 1. Hapus semua komentar
    cleaned_text = regex.sub(replacer, text)

    # 2. Rapihkan whitespace & baris renggang
    lines = cleaned_text.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Hapus spasi kosong yang tidak berguna di akhir baris
        stripped_line = line.rstrip()
        
        if stripped_line:
            # Jika baris ada isinya, masukkan ke list
            formatted_lines.append(stripped_line)
        elif formatted_lines and formatted_lines[-1] != "":
            # Jika baris kosong, HANYA masukkan jika baris sebelumnya tidak kosong 
            # (Ini akan merapatkan jarak renggang menjadi max 1 baris kosong)
            formatted_lines.append("")

    # Gabungkan kembali dan pastikan diakhiri 1 baris baru (standar POSIX)
    final_text = '\n'.join(formatted_lines).strip() + '\n'

    # Tulis kembali ke file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(final_text)

def main():
    print("=== MSHQC ULTIMATE CLEANER & FORMATTER ===")
    
    # ---------------------------------------------------------
    # 1. HAPUS FOLDER DEAD CODE
    # ---------------------------------------------------------
    print("\n[1/4] Menghapus folder dead code...")
    folders_to_delete = [
        "src/foundation",
        "include/mshqc/foundation",
        "include/mshqc/validation"
    ]
    for folder in folders_to_delete:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"      [TERHAPUS] {folder}")
            
    # ---------------------------------------------------------
    # 2. BERSIHKAN CMAKELISTS & BINDINGS
    # ---------------------------------------------------------
    print("\n[2/4] Membersihkan CMakeLists.txt & bindings.cc...")
    
    # Bersihkan CMakeLists.txt
    cmake_file = "CMakeLists.txt"
    if os.path.exists(cmake_file):
        with open(cmake_file, 'r') as f:
            cmake_content = f.read()
        
        cmake_content = re.sub(r'file\(GLOB_RECURSE SRC_FOUNDATION[^\)]+\)\n?', '', cmake_content)
        cmake_content = cmake_content.replace('${SRC_FOUNDATION} ', '')
        cmake_content = cmake_content.replace('${SRC_FOUNDATION}', '')
        
        with open(cmake_file, 'w') as f:
            f.write(cmake_content)
        print(f"      [BERSIH] {cmake_file}")

    # Bersihkan bindings.cc
    binding_file = "python/bindings.cc"
    if os.path.exists(binding_file):
        with open(binding_file, 'r') as f:
            binding_content = f.read()
            
        binding_content = re.sub(r'#include "mshqc/foundation/fcidump\.h"\n?', '', binding_content)
        binding_content = re.sub(r'#include "mshqc/foundation/wavefunction\.h"\n?', '', binding_content)
        
        fcidump_pattern = r'\s*m\.def\("export_fcidump"[\s\S]*?"Export SCF and Integral results to standard FCIDUMP format"\);'
        binding_content = re.sub(fcidump_pattern, '', binding_content)
        
        with open(binding_file, 'w') as f:
            f.write(binding_content)
        print(f"      [BERSIH] {binding_file}")

    # ---------------------------------------------------------
    # 3. STRIP COMMENTS & FORMAT SOURCE CODE
    # ---------------------------------------------------------
    print("\n[3/4] Menghapus komentar dan merapihkan baris kode...")
    target_dirs = ["src", "include"]
    processed_files = 0
    
    for d in target_dirs:
        for root, dirs, files in os.walk(d):
            for file in files:
                if file.endswith((".cc", ".h", ".cpp")):
                    file_path = os.path.join(root, file)
                    remove_comments_and_format(file_path)
                    processed_files += 1
    
    print(f"      [SELESAI] {processed_files} file berhasil di-strip dan diformat ulang.")

    # ---------------------------------------------------------
    # 4. HAPUS CACHE KOMPILASI
    # ---------------------------------------------------------
    print("\n[4/4] Menghapus cache kompilasi Python/CMake...")
    cache_folders = ["build", "dist", "python/mshqc.egg-info", "releases"]
    for folder in cache_folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"      [TERHAPUS] {folder}")

    print("\n==================================================")
    print("PEMBERSIHAN TOTAL SELESAI!")
    print("Codebase Anda sekarang ramping, tanpa komentar, dan rapih.")
    print("Jalankan ulang: python setup.py install")
    print("==================================================")

if __name__ == "__main__":
    main()