#!/usr/bin/env python3
"""
Full Bab IV Benchmark Runner (Mode Gabungan: Simetri Grup Titik Abelian + Permutasi 8-Lipat).
Menghasilkan format tabel identik dengan benchmarks/run_bab4_v2.py (LIVE TEST 1/5..5/5)
serta otomatis mencetak ringkasan komparasi terhadap baseline C1 + 8-Lipat jika tersedia.

Usage:
  python benchmarks/run_bab4_abelian_v2.py
  python benchmarks/run_bab4_abelian_v2.py --quick --skip-ram
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def _sum_suite_times(ref_suite: Dict[str, Any]) -> Dict[str, float]:
    totals = {"EXACT": 0.0, "CHOLESKY": 0.0, "DF": 0.0, "ALL": 0.0}
    for key, entry in ref_suite.items():
        mode = key.split("|")[1] if "|" in key else "ALL"
        for mkey, val in entry.items():
            if isinstance(val, dict) and "t" in val:
                dt = float(val["t"])
                if mode in totals:
                    totals[mode] += dt
                totals["ALL"] += dt
    return totals


def _print_head_to_head_summary(base_path: Path, abel_path: Path) -> None:
    if not base_path.is_file() or not abel_path.is_file():
        return
    base = json.loads(base_path.read_text(encoding="utf-8"))
    abel = json.loads(abel_path.read_text(encoding="utf-8"))

    t_base = _sum_suite_times(base.get("ref_suite", {}))
    t_abel = _sum_suite_times(abel.get("ref_suite", {}))

    max_de = 0.0
    n_checked = 0
    for k, b_ent in base.get("ref_suite", {}).items():
        a_ent = abel.get("ref_suite", {}).get(k, {})
        for m, b_val in b_ent.items():
            a_val = a_ent.get(m)
            if isinstance(b_val, dict) and isinstance(a_val, dict) and "e" in b_val and "e" in a_val:
                max_de = max(max_de, abs(float(b_val["e"]) - float(a_val["e"])))
                n_checked += 1

    print("
" + "=" * 96, flush=True)
    print(" [PERBANDINGAN LANGSUNG BAB IV] C1 + 8-FOLD (Default) vs ABELIAN + 8-FOLD", flush=True)
    print("=" * 96, flush=True)
    print(f"| {'Mode Integral':<16} | {'C1 + 8-Fold (s)':>16} | {'Abelian + 8-Fold (s)':>20} | {'Rasio C1/Abelian':>18} |", flush=True)
    print(f"|{'-'*18}|{'-'*18}|{'-'*22}|{'-'*20}|", flush=True)
    for m in ("EXACT", "CHOLESKY", "DF", "ALL"):
        tb, ta = t_base[m], t_abel[m]
        if tb > 0.0 or ta > 0.0:
            ratio = tb / max(ta, 1e-9)
            lbl = "TOTAL SEMUA MODE" if m == "ALL" else m
            print(f"| {lbl:<16} | {tb:>16.3f} | {ta:>20.3f} | {ratio:>16.2f} x |", flush=True)
    print("=" * 96, flush=True)
    print(
        f" Paritas Energi ({n_checked} evaluasi metode): max|E_C1 - E_Abelian| = {max_de:.2e} Ha",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bab IV Full Suite Runner (Abelian + 8-Fold Symmetry)")
    parser.add_argument("--out", default="mshqc_bab4_abelian_v2_raw.json")
    parser.add_argument("--log", default="12_bab4_abelian_v2_full_log.txt")
    parser.add_argument("--baseline-json", default="mshqc_bab4_v2_raw.json")
    parser.add_argument("--core5", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-ram", action="store_true")
    args = parser.parse_args()

    runner = Path(__file__).resolve().parent / "run_bab4_v2.py"
    cmd = [
        sys.executable, str(runner),
        "--use-symmetry",
        "--out", args.out,
        "--log", args.log,
    ]
    if args.core5:
        cmd.append("--core5")
    if args.quick:
        cmd.append("--quick")
    if args.skip_ram:
        cmd.append("--skip-ram")

    subprocess.run(cmd, check=True)
    _print_head_to_head_summary(Path(args.baseline_json), Path(args.out))


if __name__ == "__main__":
    main()
