import os
import re
from pathlib import Path

def remove_comments(text):
    """
    Menghapus komentar C/C++ secara aman.
    Menggunakan regex untuk melewati string literal dan menghapus komentar.
    """
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return ""  # Hapus komentar
        else:
            return s   # Biarkan string/char literal
            
    return re.sub(pattern, replacer, text)

def format_whitespaces(text):
    """
    Merapikan spasi berlebih dan baris kosong yang renggang.
    """
    lines = text.split('\n')
    cleaned_lines = [line.rstrip() for line in lines]
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.lstrip('\n')
    
    return text

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_text = f.read()
            
        no_comments = remove_comments(original_text)
        final_text = format_whitespaces(no_comments)
        
        if original_text != final_text:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(final_text)
            print(f"Beres: {filepath}")
            
    except Exception as e:
        print(f"Error memproses {filepath}: {e}")

def main():
    # Tentukan direktori basis proyek
    base_dir = Path(__file__).parent.parent.absolute() if "tools" in __file__ else Path.cwd()
    
    # Menambahkan 'tools' ke dalam target direktori pembersihan
    target_dirs = [
        base_dir / 'src',
        base_dir / 'include',
        base_dir / 'tools'
    ]
    
    target_extensions = {'.cc', '.h', '.cpp', '.hpp', '.td'}
    
    print("Memulai pembersihan komentar dan merapikan baris...")
    
    for directory in target_dirs:
        if not directory.exists():
            print(f"Peringatan: Direktori {directory} tidak ditemukan.")
            continue
            
        for file_path in directory.rglob('*'):
            if file_path.suffix in target_extensions:
                process_file(file_path)
                
    print("Selesai.")

if __name__ == '__main__':
    main()