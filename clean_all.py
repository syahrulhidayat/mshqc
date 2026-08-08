import os
import re
from pathlib import Path

def remove_comments(text):
    """
    Menghapus komentar C/C++ secara aman.
    Menggunakan regex untuk melewati string literal dan menghapus komentar.
    """
    # Regex ini mencocokkan:
    # 1. Komentar single line (//...)
    # 2. Komentar multi line (/*...*/)
    # 3. String literal ("...")
    # 4. Char literal ('...')
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return ""  # Jika ini komentar, hapus (return string kosong)
        else:
            return s   # Jika ini string/char literal, biarkan
            
    return re.sub(pattern, replacer, text)

def format_whitespaces(text):
    """
    Merapikan spasi berlebih dan baris kosong yang renggang.
    """
    # Pisahkan berdasarkan baris
    lines = text.split('\n')
    
    # Hapus spasi di akhir setiap baris (trailing whitespace)
    cleaned_lines = [line.rstrip() for line in lines]
    
    # Gabungkan kembali
    text = '\n'.join(cleaned_lines)
    
    # Ubah 3 atau lebih baris kosong (newline berturut-turut) menjadi 2 newline (1 baris kosong)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Hapus baris kosong di awal file
    text = text.lstrip('\n')
    
    return text

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_text = f.read()
            
        # Proses 1: Hapus Komentar
        no_comments = remove_comments(original_text)
        
        # Proses 2: Rapikan kode yang renggang
        final_text = format_whitespaces(no_comments)
        
        # Tulis ulang hanya jika ada perubahan
        if original_text != final_text:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(final_text)
            print(f"Beres: {filepath}")
            
    except Exception as e:
        print(f"Error memproses {filepath}: {e}")

def main():
    # Tentukan direktori target (src dan include)
    base_dir = Path(__file__).parent.parent.absolute() if "tools" in __file__ else Path.cwd()
    
    target_dirs = [
        base_dir / 'src',
        base_dir / 'include'
    ]
    
    target_extensions = {'.cc', '.h', '.cpp', '.hpp', '.td'}
    
    print("Memulai pembersihan komentar dan merapikan baris...")
    
    for directory in target_dirs:
        if not directory.exists():
            print(f"Peringatan: Direktori {directory} tidak ditemukan.")
            continue
            
        # Gunakan rglob untuk mencari file secara rekursif
        for file_path in directory.rglob('*'):
            if file_path.suffix in target_extensions:
                process_file(file_path)
                
    print("Selesai.")

if __name__ == '__main__':
    main()