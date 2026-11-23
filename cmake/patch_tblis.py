#!/usr/bin/env python3
"""Cross-platform compatibility patches for TBLIS v1.2.0 (GCC 15 / Clang 21 / ARM64 / glibc)."""
from pathlib import Path
import sys

if not Path("configure").is_file():
    sys.exit("Error: jalankan patch_tblis.py dari direktori root sumber TBLIS.")

p = Path("src/external/catch/catch.hpp")
p.write_text(p.read_text(encoding="utf-8").replace("SIGSTKSZ", "8192"), encoding="utf-8")

for rel in (
    "src/external/marray/include/short_vector.hpp",
    "src/external/marray/include/range.hpp",
    "src/util/thread.h",
):
    fp = Path(rel)
    txt = fp.read_text(encoding="utf-8")
    if "#include <limits>" not in txt:
        fp.write_text("#include <limits>\n" + txt, encoding="utf-8")

tp = Path("src/external/tci/tci/task_set.c")
t_txt = tp.read_text(encoding="utf-8")
if "#include <stdlib.h>" not in t_txt:
    tp.write_text("#include <stdlib.h>\n" + t_txt, encoding="utf-8")

ep = Path("src/external/marray/include/expression.hpp")
e_txt = ep.read_text(encoding="utf-8")
if "__x86_64__" not in e_txt:
    ep.write_text(
        e_txt.replace(
            "#include <x86intrin.h>",
            "#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))\n"
            "#include <x86intrin.h>\n"
            "#else\n"
            "#undef __AVX512F__\n"
            "#undef __AVX__\n"
            "#undef __SSE2__\n"
            "#endif",
        ),
        encoding="utf-8",
    )

cp = Path("src/util/cpuid.hpp")
if cp.exists():
    txt = cp.read_text(encoding="utf-8")
    txt = txt.replace("enum {VENDOR_ARM, VENDOR_UNKNOWN}", "enum {VENDOR_ARM, VENDOR_UNKNOWN};")
    txt = txt.replace("enum {MODEL_ARMV7, MODEL_ARMV8, MODEL_UNKNOWN}", "enum {MODEL_ARMV7, MODEL_ARMV8, MODEL_UNKNOWN};")
    cp.write_text(txt, encoding="utf-8")

cx = Path("src/util/cpuid.cxx")
if cx.exists():
    cx.write_text(
        cx.read_text(encoding="utf-8").replace(
            "strtoi(ptno, pos, 16)",
            "((void)pos, (int)strtol(ptno.c_str(), NULL, 16))",
        ),
        encoding="utf-8",
    )
print("[OK] TBLIS v1.2.0 source patched successfully.")
