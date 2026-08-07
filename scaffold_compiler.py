#!/usr/bin/env python3
import os
from pathlib import Path

def patch_unified_headers(base_dir: Path):
    """Me-rutekan ulang include yang usang ke arsitektur MP2/MP3 terpadu."""
    src_dir = base_dir / "src"
    if not src_dir.exists():
        return

    # Pindai seluruh file C++
    for filepath in src_dir.rglob("*.cc"):
        with open(filepath, 'r') as f:
            content = f.read()

        modified = False
        
        # Rutekan ulang rmp2.h ke mp2.h
        if '#include "mshqc/foundation/rmp2.h"' in content:
            content = content.replace('#include "mshqc/foundation/rmp2.h"', '#include "mshqc/mp2/mp2.h"')
            modified = True
            
        # Rutekan ulang rmp3.h ke mp3.h
        if '#include "mshqc/foundation/rmp3.h"' in content:
            content = content.replace('#include "mshqc/foundation/rmp3.h"', '#include "mshqc/mp3/mp3.h"')
            modified = True

        if modified:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"[PATCHED] Header ruting diperbaiki pada: {filepath.name}")

if __name__ == "__main__":
    base_directory = Path.cwd()
    print("[INFO] Memulai perbaikan ruting header untuk Unified MP2/MP3 Engine...")
    patch_unified_headers(base_directory)
    print("[SUCCESS] Pointer statis C++ telah disejajarkan.")