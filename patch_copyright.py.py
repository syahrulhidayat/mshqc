import os
from pathlib import Path

# Metadata Lisensi
COPYRIGHT_HOLDER = "2026 Muhamd Syahrul Hidayat and mshqc contributors"
LICENSE_TEMPLATE = """ {comment} ==============================================================================
 {comment} Copyright (c) {copyright}
 {comment}
 {comment} Licensed under the Apache License, Version 2.0 (the "License");
 {comment} you may not use this file except in compliance with the License.
 {comment} You may obtain a copy of the License at
 {comment}
 {comment}     http://www.apache.org/licenses/LICENSE-2.0
 {comment}
 {comment} Unless required by applicable law or agreed to in writing, software
 {comment} distributed under the License is distributed on an "AS IS" BASIS,
 {comment} WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 {comment} See the License for the specific language governing permissions and
 {comment} limitations under the License.
 {comment} ==============================================================================\n\n"""

# Konfigurasi Target Ekstensi dan Format Komentar
COMMENT_STYLES = {
    '.cc': '//',
    '.h': '//',
    '.td': '//',
    '.py': '#',
    '.cmake': '#',
    '.txt': '#' # Untuk CMakeLists.txt
}

TARGET_DIRS = ['src', 'include', 'python', 'test', 'tools', 'cmake']

def generate_header(ext):
    comment_char = COMMENT_STYLES.get(ext)
    if not comment_char:
        return None
    return LICENSE_TEMPLATE.format(comment=comment_char, copyright=COPYRIGHT_HOLDER)

def inject_license():
    base_dir = Path(__file__).parent.resolve()
    processed_count = 0
    skipped_count = 0

    for target in TARGET_DIRS:
        target_path = base_dir / target
        if not target_path.exists():
            continue

        for filepath in target_path.rglob('*'):
            if not filepath.is_file():
                continue
            
            # Khusus untuk CMakeLists.txt yang tidak memiliki ekstensi
            ext = filepath.suffix
            if filepath.name == 'CMakeLists.txt':
                ext = '.txt'

            header = generate_header(ext)
            if not header:
                continue

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Pengecekan idempotensi
            if "Licensed under the Apache License" in content:
                skipped_count += 1
                continue

            # Menangani shebang pada file Python
            if content.startswith('#!'):
                lines = content.split('\n', 1)
                new_content = lines[0] + '\n' + header + (lines[1] if len(lines) > 1 else '')
            else:
                new_content = header + content

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            processed_count += 1
            print(f"Injected: {filepath.relative_to(base_dir)}")

    print(f"\n[Injection Complete] Processed: {processed_count} files | Skipped (Already Licensed): {skipped_count} files")

if __name__ == "__main__":
    inject_license()