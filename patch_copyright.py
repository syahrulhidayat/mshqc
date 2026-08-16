import os
import shutil
from pathlib import Path

# Template Lisensi Apache 2.0
LICENSE_TEMPLATE = """{c} Copyright 2026 Muhamad Syahrul Hidayat
{c}
{c} Licensed under the Apache License, Version 2.0 (the "License");
{c} you may not use this file except in compliance with the License.
{c} You may obtain a copy of the License at
{c}
{c}     http://www.apache.org/licenses/LICENSE-2.0
{c}
{c} Unless required by applicable law or agreed to in writing, software
{c} distributed under the License is distributed on an "AS IS" BASIS,
{c} WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
{c} See the License for the specific language governing permissions and
{c} limitations under the License.

"""

def get_comment_syntax(suffix: str, filename: str) -> str:
    """Mengembalikan sintaks komentar berdasarkan ekstensi berkas."""
    if suffix in [".cc", ".h", ".cpp", ".hpp", ".c"]:
        return "//"
    elif suffix in [".py", ".sh"] or filename == "CMakeLists.txt":
        return "#"
    return ""

def prune_directories(base_path: Path):
    """Menghapus direktori modul CI dan MCSCF beserta header terkait."""
    targets = [
        base_path / "src" / "ci",
        base_path / "src" / "mcscf",
        base_path / "include" / "mshqc" / "ci",
        base_path / "include" / "mshqc" / "mcscf"
    ]
    
    for target in targets:
        if target.exists() and target.is_dir():
            shutil.rmtree(target)
            print(f"[I/O] Deleted directory: {target.relative_to(base_path)}")

def inject_license(base_path: Path):
    """Menginjeksi header lisensi pada awal berkas sumber jika belum ada."""
    valid_extensions = {".cc", ".h", ".py", ".sh"}
    valid_filenames = {"CMakeLists.txt"}
    
    # Pengecualian untuk direktori build/lingkungan eksternal
    exclude_dirs = {".git", "build", "mshqc.egg-info", "__pycache__", "releases", "include/eigen-3.4.0"}
    
    injected_count = 0
    
    for filepath in base_path.rglob("*"):
        if not filepath.is_file():
            continue
            
        # Lewati jika berada di dalam direktori yang dikecualikan
        if any(ex_dir in filepath.parts for ex_dir in exclude_dirs):
            continue
            
        suffix = filepath.suffix
        filename = filepath.name
        
        if suffix in valid_extensions or filename in valid_filenames:
            comment_char = get_comment_syntax(suffix, filename)
            if not comment_char:
                continue
                
            header = LICENSE_TEMPLATE.format(c=comment_char)
            
            try:
                content = filepath.read_text(encoding="utf-8")
                
                # Evaluasi deterministik untuk mencegah injeksi ganda
                if "Licensed under the Apache License" not in content:
                    # Pertahankan shebang (#!/bin/bash atau #!/usr/bin/env python) di baris pertama
                    if content.startswith("#!"):
                        lines = content.splitlines(keepends=True)
                        new_content = lines[0] + "\n" + header + "".join(lines[1:])
                    else:
                        new_content = header + content
                        
                    filepath.write_text(new_content, encoding="utf-8")
                    injected_count += 1
            except UnicodeDecodeError:
                print(f"[ERROR] Binary or non-UTF-8 file skipped: {filepath.relative_to(base_path)}")
                
    print(f"[I/O] Successfully injected Apache 2.0 license into {injected_count} files.")

if __name__ == "__main__":
    root_dir = Path.cwd()
    print("=== MSHQC Prune and License Injection Utility ===")
    prune_directories(root_dir)
    inject_license(root_dir)
    print("=== Operation Complete ===")