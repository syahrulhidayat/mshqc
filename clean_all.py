import os
import re
from pathlib import Path

def clean_cpp_comments(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return False

    header = ""
    body = content

    # Identifikasi batas akhir dari blok lisensi secara empiris menggunakan indeks string
    marker = "Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors"
    if marker in content:
        marker_idx = content.find(marker)
        # Cari garis pembatas '// =====' yang menutup blok lisensi setelah teks copyright
        end_border_idx = content.find("// =====", marker_idx)
        if end_border_idx != -1:
            # Cari karakter newline (\n) untuk menangkap keseluruhan baris pembatas
            newline_idx = content.find("\n", end_border_idx)
            if newline_idx != -1:
                split_point = newline_idx + 1
                header = content[:split_point]
                body = content[split_point:]

    # Tokenizer leksikal: mengamankan string ("..."), char ('.'), block comment (/*...*/)
    # Target eliminasi eksklusif: //[^\n]* (tidak akan menyentuh makro preprosesor #)
    token_pattern = re.compile(r'("(?:\\[\s\S]|[^"])*"|\'(?:\\[\s\S]|[^\'])*\'|/\*.*?\*/)|//[^\n]*')
    
    cleaned_body = token_pattern.sub(lambda m: m.group(1) if m.group(1) else '', body)
    
    # Resolusi trailing whitespace untuk mencegah akumulasi baris kosong kotor
    cleaned_body = re.sub(r'[ \t]+$', '', cleaned_body, flags=re.MULTILINE)

    new_content = header + cleaned_body

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

def main():
    # Pemindaian dibatasi pada direktori komputasi dan infrastruktur compiler
    target_dirs = ['src', 'include', 'python', 'tools', 'test']
    valid_extensions = {'.cc', '.cpp', '.h', '.hpp'}
    
    root = Path(__file__).parent.resolve()
    modified = 0

    for tdir in target_dirs:
        dir_path = root / tdir
        if not dir_path.exists():
            continue
            
        for file_path in dir_path.rglob('*'):
            if file_path.suffix in valid_extensions:
                if clean_cpp_comments(file_path):
                    print(f"Cleaned: {file_path.relative_to(root)}")
                    modified += 1
                    
    print(f"\nExecution complete. Total C++ source files optimized: {modified}")

if __name__ == "__main__":
    main()