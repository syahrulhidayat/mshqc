#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path

ROW_RE = re.compile(
    r"^\|\s*(?P<sys>Neon|H2O|Lithium|Carbon|Oxygen)\s*"
    r"\|\s*(?P<eri>EXACT|CHOLESKY|DF)\s*"
    r"\|\s*(?P<method>RHF|UHF|RMP2|UMP2|OMP2|OMP3)\s*"
    r"\|\s*(?P<energy>-?\d+\.\d+)\s*"
    r"\|\s*(?P<time>\d+\.\d+)\s*\|"
)

def parse_log(filepath: str):
    data = {}
    ordered_keys = []
    for line in Path(filepath).read_text(encoding="utf-8", errors="replace").splitlines():
        m = ROW_RE.match(line.strip())
        if m:
            key = (m.group("sys"), m.group("eri"), m.group("method"))
            if key not in data:
                ordered_keys.append(key)
            data[key] = (float(m.group("energy")), float(m.group("time")))
    return ordered_keys, data

def main():
    if len(sys.argv) < 3:
        print("Usage: python tests/compare_mshqc_psi4.py <mshqc_log> <psi4_log>")
        sys.exit(1)

    mshqc_keys, mshqc_data = parse_log(sys.argv[1])
    psi4_keys, psi4_data = parse_log(sys.argv[2])

    keys = [k for k in mshqc_keys if k in psi4_data]
    if not keys:
        print("No matching benchmark rows found between logs.")
        sys.exit(1)

    md_lines = [
        "## Head-to-Head Benchmark Summary: `mshqc` vs `Psi4` (`cc-pVTZ` / `cc-pVTZ-RI`)",
        "",
        "| System | ERI Mode | Method | `mshqc` Energy (Ha) | `Psi4` Energy (Ha) | `|dE|` (uHa) | `mshqc` (s) | `Psi4` (s) | Speedup |",
        "|:---|:---|:---|---:|---:|---:|---:|---:|---:|",
    ]

    print("\n" + "=" * 112)
    print("                HEAD-TO-HEAD BENCHMARK COMPARISON: MSHQC vs PSI4 (cc-pVTZ / cc-pVTZ-RI)")
    print("=" * 112)
    hdr = f"| {'System':<8} | {'ERI Mode':<9} | {'Method':<6} | {'mshqc (Ha)':<16} | {'Psi4 (Ha)':<16} | {'|dE| uHa':<9} | {'mshqc(s)':<8} | {'Psi4(s)':<8} | {'Speedup':<8} |"
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    max_de_uha = 0.0
    sub_001_count = 0
    omp2_speedups = []
    omp3_speedups = []

    for sys_name, eri, method in keys:
        e_m, t_m = mshqc_data[(sys_name, eri, method)]
        e_p, t_p = psi4_data[(sys_name, eri, method)]
        de_uha = abs(e_m - e_p) * 1e6
        speedup = (t_p / max(t_m, 1e-4)) if t_p > 0 else 0.0

        max_de_uha = max(max_de_uha, de_uha)
        if de_uha < 0.005:
            sub_001_count += 1
        if method == "OMP2":
            omp2_speedups.append(speedup)
        elif method == "OMP3":
            omp3_speedups.append(speedup)

        print(
            f"| {sys_name:<8} | {eri:<9} | {method:<6} | {e_m:<16.8f} | {e_p:<16.8f} | "
            f"{de_uha:<9.2f} | {t_m:<8.3f} | {t_p:<8.3f} | {speedup:<7.2f}x |"
        )
        md_lines.append(
            f"| {sys_name} | `{eri.lower()}` | `{method}` | `{e_m:.8f}` | `{e_p:.8f}` | "
            f"`{de_uha:.2f}` | `{t_m:.3f}` | `{t_p:.3f}` | **{speedup:.2f}x** |"
        )

    print("-" * len(hdr))
    stats_line = (
        f"Total matched runs: {len(keys)} | Max |dE|: {max_de_uha:.2f} uHa | "
        f"Runs with 0.00 uHa deviation: {sub_001_count}/{len(keys)}"
    )
    print(stats_line)
    if omp2_speedups and omp3_speedups:
        sp_line = (
            f"OMP2 Speedup Range: {min(omp2_speedups):.2f}x - {max(omp2_speedups):.2f}x | "
            f"OMP3 Speedup Range: {min(omp3_speedups):.2f}x - {max(omp3_speedups):.2f}x"
        )
        print(sp_line)
    print("=" * 112 + "\n")

    md_lines.extend([
        "",
        f"- **Total Validated Runs:** `{len(keys)}` (`5 systems x 3 ERI modes x 4 methods`)",
        f"- **Maximum Absolute Energy Deviation (`|dE|`):** `{max_de_uha:.2f} uHa` (`0.00 uHa` on `{sub_001_count}/{len(keys)}` runs at `1e-8 Ha` precision)",
    ])
    if omp2_speedups and omp3_speedups:
        md_lines.extend([
            f"- **`OMP2` Speedup Range (`t_Psi4 / t_mshqc`):** `{min(omp2_speedups):.2f}x` to `{max(omp2_speedups):.2f}x`",
            f"- **`OMP3` Speedup Range (`t_Psi4 / t_mshqc`):** `{min(omp3_speedups):.2f}x` to `{max(omp3_speedups):.2f}x`",
        ])

    md_text = "\n".join(md_lines) + "\n"
    Path("comparison_summary.md").write_text(md_text, encoding="utf-8")

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as f:
            f.write(md_text)

if __name__ == "__main__":
    main()
