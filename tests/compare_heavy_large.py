#!/usr/bin/env python3
"""
tests/compare_heavy_large.py
----------------------------
Perbandingan Head-to-Head mshqc vs Psi4 dari mshqc_heavy_large_raw.json dan psi4_heavy_large_raw.json.
Menghasilkan laporan terminal & test_heavy_large_validation.txt
"""

import sys
import json
import pathlib


def main():
    mshqc_path = sys.argv[1] if len(sys.argv) > 1 else "mshqc_heavy_large_raw.json"
    psi4_path = sys.argv[2] if len(sys.argv) > 2 else "psi4_heavy_large_raw.json"

    m_data = json.loads(pathlib.Path(mshqc_path).read_text(encoding="utf-8"))
    p_data = json.loads(pathlib.Path(psi4_path).read_text(encoding="utf-8"))

    p_map = {c["label"]: c for c in p_data["cases"]}

    lines = []
    lines.append("# Head-to-Head Benchmark: Heavy Atoms (Z=18..36) & Large Molecules (mshqc vs Psi4)")
    lines.append(
        f"Threads: mshqc={m_data.get('n_threads')}, Psi4={p_data.get('n_threads')} (v{p_data.get('version', 'unknown')}) | "
        f"Smoke test mshqc.run() Benzene DF: {m_data['smoke_test']['t_s']:.3f}s (E_MP2={m_data['smoke_test']['e_mp2']:.8f} Ha)\n"
    )

    hdr = (
        f"| {'Sistem':<19} | {'Nbf':>4} | {'FC':>2} | "
        f"{'mRHF(s)':>7} | {'pSCF(s)':>7} | {'|dE_SCF|uH':>10} | "
        f"{'mOMP2(s)':>8} | {'pOMP2(s)':>8} | {'Spd2':>6} | {'|dE2|uH':>8} | "
        f"{'mOMP3(s)':>8} | {'pOMP3(s)':>8} | {'Spd3':>6} | {'|dE3|uH':>8} | "
        f"{'PeakMB(m/p)':>13} |"
    )
    sep = "-" * len(hdr)
    lines.append(hdr)
    lines.append(sep)

    matched = 0
    max_de_scf = 0.0
    max_de_omp2 = 0.0
    max_de_omp3 = 0.0

    for m in m_data["cases"]:
        label = m["label"]
        if label not in p_map:
            continue
        p = p_map[label]
        matched += 1

        de_scf_uha = abs(m["e_scf"] - p["e_scf"]) * 1e6
        de2_uha = abs(m["e_omp2"] - p["e_omp2"]) * 1e6
        spd2 = f"{p['t_omp2'] / max(m['t_omp2'], 1e-6):>5.2f}x"

        max_de_scf = max(max_de_scf, de_scf_uha)
        max_de_omp2 = max(max_de_omp2, de2_uha)

        if m["t_omp3"] is not None and p["t_omp3"] is not None:
            de3_uha = abs(m["e_omp3"] - p["e_omp3"]) * 1e6
            max_de_omp3 = max(max_de_omp3, de3_uha)
            m3_str = f"{m['t_omp3']:>8.3f}"
            p3_str = f"{p['t_omp3']:>8.3f}"
            spd3_str = f"{p['t_omp3'] / max(m['t_omp3'], 1e-6):>5.2f}x"
            de3_str = f"{de3_uha:>8.2f}"
        else:
            m3_str, p3_str, spd3_str, de3_str = "    skip", "    skip", "  skip", "    skip"

        peak_str = f"{m['peak_mb']:.0f}/{p['peak_mb']:.0f}"

        row = (
            f"| {label:<19} | {m['nbf']:>4d} | {m['n_frozen']:>2d} | "
            f"{m['t_scf']:>7.3f} | {p['t_scf']:>7.3f} | {de_scf_uha:>10.2f} | "
            f"{m['t_omp2']:>8.3f} | {p['t_omp2']:>8.3f} | {spd2:>6} | {de2_uha:>8.2f} | "
            f"{m3_str} | {p3_str} | {spd3_str} | {de3_str} | "
            f"{peak_str:>13} |"
        )
        lines.append(row)

    lines.append(sep)
    lines.append(f"\nTotal Matched Heavy/Large Runs: {matched}")
    lines.append(f"Max |dE_SCF|  : {max_de_scf:.3f} uHa")
    lines.append(f"Max |dE_OMP2| : {max_de_omp2:.3f} uHa")
    lines.append(f"Max |dE_OMP3| : {max_de_omp3:.3f} uHa")

    report = "\n".join(lines) + "\n"
    print(report)
    pathlib.Path("test_heavy_large_validation.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()