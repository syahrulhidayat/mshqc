#!/usr/bin/env python3
# tests/compare_heavy_large.py
# Perbandingan Head-to-Head mshqc vs Psi4 dari mshqc_heavy_large_raw.json dan psi4_heavy_large_raw.json.
# Memisahkan metrik All-Electron (apple-to-apple penuh) vs Frozen-Core (strict FC mshqc vs Psi4 dfocc
# relaxed core-valence/core-virtual) serta melaporkan status FAIL Psi4 secara transparan.

import sys
import json
import pathlib


def main():
    mshqc_path = sys.argv[1] if len(sys.argv) > 1 else "mshqc_heavy_large_raw.json"
    psi4_path = sys.argv[2] if len(sys.argv) > 2 else "psi4_heavy_large_raw.json"

    m_data = json.loads(pathlib.Path(mshqc_path).read_text(encoding="utf-8"))
    p_data = json.loads(pathlib.Path(psi4_path).read_text(encoding="utf-8"))

    def norm_lbl(s: str) -> str:
        return s.replace("188 BF", "180 BF")

    p_map = {norm_lbl(c["label"]): c for c in p_data["cases"]}

    lines = []
    lines.append("# Head-to-Head Benchmark: Heavy Atoms (Z=18..36) & Large Molecules (mshqc vs Psi4)")
    lines.append(
        f"Threads: mshqc={m_data.get('n_threads')}, Psi4={p_data.get('n_threads')} (v{p_data.get('version', 'unknown')}) | "
        f"Smoke test mshqc.run() Benzene DF: {m_data['smoke_test']['t_s']:.3f}s (E_MP2={m_data['smoke_test']['e_mp2']:.8f} Ha)\n"
    )

    hdr = (
        f"| {'Sistem':<19} | {'Nbf':>4} | {'FC':>2} | "
        f"{'mRHF(s)':>7} | {'pSCF(s)':>7} | {'|dE_SCF|uH':>10} | "
        f"{'mOMP2(s)':>8} | {'pOMP2(s)':>8} | {'Spd2':>7} | {'|dE2|uH':>8} | "
        f"{'mOMP3(s)':>8} | {'pOMP3(s)':>8} | {'Spd3':>7} | {'|dE3|uH':>8} | "
        f"{'PeakMB(m/p)':>13} |"
    )
    sep = "-" * len(hdr)
    lines.append(hdr)
    lines.append(sep)

    matched = 0
    max_de_scf = 0.0
    max_de2_ae, max_de3_ae = 0.0, 0.0
    max_de2_fc, max_de3_fc = 0.0, 0.0
    psi4_fails = 0

    for m in m_data["cases"]:
        lbl = norm_lbl(m["label"])
        if lbl not in p_map:
            continue
        p = p_map[lbl]
        matched += 1
        is_fc = (m["n_frozen"] > 0)

        de_scf_uha = abs(m["e_scf"] - p["e_scf"]) * 1e6
        max_de_scf = max(max_de_scf, de_scf_uha)

        if m["t_omp2"] is not None and p["t_omp2"] is not None:
            de2_uha = abs(m["e_omp2"] - p["e_omp2"]) * 1e6
            if is_fc:
                max_de2_fc = max(max_de2_fc, de2_uha)
            else:
                max_de2_ae = max(max_de2_ae, de2_uha)
            m2_str = f"{m['t_omp2']:>8.3f}"
            p2_str = f"{p['t_omp2']:>8.3f}"
            spd2 = f"{p['t_omp2'] / max(m['t_omp2'], 1e-6):>6.2f}x"
            de2_str = f"{de2_uha:>8.2f}"
        else:
            m2_str = f"{m['t_omp2']:>8.3f}" if m["t_omp2"] is not None else "    FAIL"
            p2_str = f"{p['t_omp2']:>8.3f}" if p["t_omp2"] is not None else "    FAIL"
            spd2, de2_str = "    N/A", "    FAIL"

        if m["t_omp3"] is not None and p["t_omp3"] is not None:
            de3_uha = abs(m["e_omp3"] - p["e_omp3"]) * 1e6
            if is_fc:
                max_de3_fc = max(max_de3_fc, de3_uha)
            else:
                max_de3_ae = max(max_de3_ae, de3_uha)
            m3_str = f"{m['t_omp3']:>8.3f}"
            p3_str = f"{p['t_omp3']:>8.3f}"
            spd3_str = f"{p['t_omp3'] / max(m['t_omp3'], 1e-6):>6.2f}x"
            de3_str = f"{de3_uha:>8.2f}"
        elif m["t_omp3"] is not None and p["t_omp3"] is None:
            psi4_fails += 1
            m3_str = f"{m['t_omp3']:>8.3f}"
            p3_str, spd3_str, de3_str = "    FAIL", "    N/A", "    FAIL"
        else:
            m3_str, p3_str, spd3_str, de3_str = "    skip", "    skip", "   skip", "    skip"

        peak_str = f"{m['peak_mb']:.0f}/{p['peak_mb']:.0f}"

        row = (
            f"| {lbl:<19} | {m['nbf']:>4d} | {m['n_frozen']:>2d} | "
            f"{m['t_scf']:>7.3f} | {p['t_scf']:>7.3f} | {de_scf_uha:>10.2f} | "
            f"{m2_str} | {p2_str} | {spd2} | {de2_str} | "
            f"{m3_str} | {p3_str} | {spd3_str} | {de3_str} | "
            f"{peak_str:>13} |"
        )
        lines.append(row)

    lines.append(sep)
    lines.append(f"\nTotal Matched Heavy/Large Runs : {matched} (Psi4 OMP3 convergence failures: {psi4_fails})")
    lines.append(f"Max |dE_SCF|  (All Cases)      : {max_de_scf:.3f} uHa")
    lines.append(f"Max |dE_OMP2| (All-Electron)   : {max_de2_ae:.3f} uHa  [exact apple-to-apple]")
    lines.append(f"Max |dE_OMP3| (All-Electron)   : {max_de3_ae:.3f} uHa  [exact apple-to-apple]")
    lines.append(f"Max |dE_OMP2| (Frozen-Core)    : {max_de2_fc:.3f} uHa  [strict FC mshqc vs Psi4 relaxed core-act/vir]")
    lines.append(f"Max |dE_OMP3| (Frozen-Core)    : {max_de3_fc:.3f} uHa  [strict FC mshqc vs Psi4 relaxed core-act/vir]")

    report = "\n".join(lines) + "\n"
    print(report)
    pathlib.Path("test_heavy_large_validation.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
