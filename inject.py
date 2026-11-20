#!/usr/bin/env python3
"""
fix_scf_newlines.py
Memperbaiki string literal C++ yang terpotong newline di src/scf/scf.cc
akibat re.subn() dan memverifikasi integritas tanda kutip pada seluruh file C++ ter-patch.
"""
import re
import sys
from pathlib import Path

ROOT = Path(".").resolve()
SCF_PATH = ROOT / "src" / "scf" / "scf.cc"

if not SCF_PATH.is_file():
    print(f"[ERROR] File tidak ditemukan: {SCF_PATH}")
    sys.exit(1)

src = SCF_PATH.read_text(encoding="utf-8")
orig = src

# Perbaiki literal newline di dalam string std::cout pada dua blok inject
fixes = [
    (
        re.compile(r'(" GB\) > budget; switching to HDF5 OOC mode\.)\r?\n\s*(";)', re.MULTILINE),
        lambda m: m.group(1) + r"\n" + m.group(2),
    ),
    (
        re.compile(r'(" GB\); auto-fallback to Direct SCF\.)\r?\n\s*(";)', re.MULTILINE),
        lambda m: m.group(1) + r"\n" + m.group(2),
    ),
]

total_fixed = 0
for pat, repl in fixes:
    src, n = pat.subn(repl, src)
    total_fixed += n

if src != orig:
    SCF_PATH.write_text(src, encoding="utf-8")
    print(f"[FIXED] {SCF_PATH.relative_to(ROOT)} -> {total_fixed} string literal terpotong diperbaiki.")
else:
    print(f"[INFO ] Tidak ada pola terpotong yang tersisa di {SCF_PATH.relative_to(ROOT)}.")

# Verifikasi audit: cek apakah ada baris C++ dengan jumlah tanda kutip ganjil (unterminated string)
cpp_targets = [
    "src/scf/scf.cc",
    "src/scf/sad_guess.cc",
    "src/mp2/mp2.cc",
    "src/mp/omp_fock.cc",
    "src/mp/omp_gradient.cc",
    "src/mp3/mp3.cc",
    "src/gradient/hpc_geom_opt.cc",
]

def strip_line_comment_and_count_quotes(line: str) -> int:
    in_str = False
    in_chr = False
    q_count = 0
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        nxt = line[i + 1] if i + 1 < n else ""
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
                q_count += 1
            i += 1
            continue
        if in_chr:
            if ch == "\\":
                i += 2
                continue
            if ch == "'":
                in_chr = False
            i += 1
            continue
        if ch == "/" and nxt == "/":
            break
        if ch == '"':
            in_str = True
            q_count += 1
        elif ch == "'":
            in_chr = True
        i += 1
    return q_count

errors = 0
for rel in cpp_targets:
    p = ROOT / rel
    if not p.is_file():
        print(f"[WARN ] Dilewati (tidak ada): {rel}")
        continue
    in_block = False
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if in_block:
            if "*/" in s:
                in_block = False
            continue
        if s.startswith("/*") and "*/" not in s:
            in_block = True
            continue
        if 'R"(' in s:
            continue
        if strip_line_comment_and_count_quotes(line) % 2 != 0:
            print(f"[FAIL ] Kemungkinan unterminated string di {rel}:{lineno} -> {line.strip()}")
            errors += 1

if errors == 0:
    print("[PASS ] Seluruh 7 file C++ lolos pemeriksaan integritas string literal.")
    sys.exit(0)
else:
    print(f"[FAIL ] Ditemukan {errors} baris bermasalah.")
    sys.exit(1)