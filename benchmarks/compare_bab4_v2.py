#!/usr/bin/env python3
"""
Komparator Head-to-Head Bab IV v2: mshqc v2.0.0 (C1+8-Fold & opsional Abelian+8-Fold) vs Psi4.
Memvalidasi seluruh 150 konfigurasi inti Bab IV (energi, <S^2>, iterasi, speedup, RAM)
serta metode ekstensi OMP2.5 secara terpisah.

Usage:
  python benchmarks/compare_bab4_v2.py \
    --mshqc mshqc_bab4_v2_raw.json \
    --psi4 psi4_bab4_v2_raw.json \
    --mshqc-abelian mshqc_bab4_abelian_v2_raw.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ALL_SYSTEM_ORDER = ["Neon", "H2O", "Lithium", "Carbon", "Oxygen", "OH", "O2"]
ERI_MODE_ORDER = ["EXACT", "CHOLESKY", "DF"]
IS_UHF_MAP = {
    "Neon": False, "H2O": False, "Lithium": True,
    "Carbon": True, "Oxygen": True, "OH": True, "O2": True,
}


class TeeRecorder:
    def __init__(self) -> None:
        self.terminal = sys.__stdout__
        self.chunks: List[str] = []

    def write(self, msg: str) -> None:
        self.terminal.write(msg)
        self.terminal.flush()
        self.chunks.append(msg)

    def flush(self) -> None:
        self.terminal.flush()

    def get_text(self) -> str:
        return "".join(self.chunks)


def core_method_pairs(is_uhf: bool) -> List[Tuple[str, str]]:
    pairs = [("UHF" if is_uhf else "RHF", "SCF")]
    if is_uhf:
        pairs.append(("ROHF", "ROHF"))
    pairs.extend([
        ("UMP2" if is_uhf else "RMP2", "MP2"),
        ("UMP3" if is_uhf else "RMP3", "MP3"),
    ])
    if is_uhf:
        pairs.extend([("ROMP2", "ROMP2"), ("ROMP3", "ROMP3")])
    pairs.extend([("OMP2", "OMP2"), ("OMP3", "OMP3")])
    return pairs


def print_table_1(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    print("\n" + "=" * 112, flush=True)
    print(" 1. TABEL KOMPRESI TENSOR ERI AO & VEKTOR CHOLESKY AKTUAL (tau_CD = 1e-9)", flush=True)
    print("=" * 112, flush=True)
    print(f"| {'Sistem':<8} | {'Basis':<8} | {'N_bf':<5} | {'N_aux(DF)':<9} | {'M_chol':<7} | {'Exact (MB)':<11} | {'Chol (MB)':<10} | {'DF (MB)':<9} | {'Exact/Chol':<10} | {'Exact/DF':<9} |", flush=True)
    print(f"|{'-'*10}|{'-'*10}|{'-'*7}|{'-'*11}|{'-'*9}|{'-'*13}|{'-'*12}|{'-'*11}|{'-'*12}|{'-'*11}|", flush=True)
    for r in rows:
        mc = r.get("mem_chol_packed", r.get("mem_chol", 0.0))
        md = r.get("mem_df_packed", r.get("mem_df", 0.0))
        print(
            f"| {r['sys']:<8} | {r['basis']:<8} | {r['nbf']:<5} | {r['naux']:<9} | {r['m_chol']:<7} | "
            f"{r['mem_exact']:<11.3f} | {mc:<10.3f} | {md:<9.3f} | "
            f"{r['r_chol']:<8.2f} x | {r['r_df']:<7.2f} x |",
            flush=True,
        )


def print_table_2(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    print("\n" + "=" * 116, flush=True)
    print(" 2. TABEL ELIMINASI HAMBATAN MEMORI O(V^4) PADA OMP3: KONVENSIONAL vs MSHQC v2.0 (DF MODE)", flush=True)
    print("=" * 116, flush=True)
    print(f"| {'Sistem':<7} | {'Basis':<8} | {'(O, V, N_aux)':<15} | {'Konvensional V_vvvv + Gamma_vvvv (MB)':<37} | {'MSHQC B_3c + T2 + Y_ooov (MB)':<30} | {'Reduksi Memori MO':<17} |", flush=True)
    print(f"|{'-'*9}|{'-'*10}|{'-'*17}|{'-'*39}|{'-'*32}|{'-'*19}|", flush=True)
    for r in rows:
        ov_str = f"({r['oa']},{r['va']},{r['naux']})"
        print(
            f"| {r['sys']:<7} | {r['basis']:<8} | {ov_str:<15} | "
            f"{r['mem_conv_v4']:<37.3f} | {r['mem_mshqc']:<30.3f} | {r['ratio']:<15.2f} x |",
            flush=True,
        )


def print_table_3(mshqc_db: Dict[str, Any], psi4_db: Dict[str, Any]) -> None:
    print("\n" + "=" * 144, flush=True)
    print(" 3. TABEL PEMULIHAN SIMETRI SPIN <S^2> SEBELUM (UHF) DAN SESUDAH OPTIMISASI ORBITAL: MSHQC v2.0 vs PSI4", flush=True)
    print("=" * 144, flush=True)
    print(
        f"| {'Sistem':<7} | {'Mode':<8} | {'Ideal':<5} | {'<S^2> UHF (mshqc/Psi4)':<23} | "
        f"{'<S^2> OMP2 (mshqc/Psi4)':<23} | {'It OMP2(m/P)':<12} | {'Red OMP2':<8} | "
        f"{'<S^2> OMP3 (mshqc/Psi4)':<23} | {'It OMP3(m/P)':<12} | {'Red OMP3':<8} |",
        flush=True,
    )
    print(f"|{'-'*9}|{'-'*10}|{'-'*7}|{'-'*25}|{'-'*25}|{'-'*14}|{'-'*10}|{'-'*25}|{'-'*14}|{'-'*10}|", flush=True)

    cd_bug_notes: List[str] = []
    for sys_name in ["Lithium", "Carbon", "Oxygen", "OH", "O2"]:
        for mode in ERI_MODE_ORDER:
            m_ent = mshqc_db.get(f"{sys_name}|{mode}", {})
            p_ent = psi4_db.get(f"{sys_name}|{mode}", {})
            if not m_ent:
                continue
            s_ideal = float(m_ent["s_ideal"])
            s2_uhf = float(m_ent["SCF"]["s2"])
            s2_o2 = float(m_ent["OMP2"]["s2"])
            s2_o3 = float(m_ent["OMP3"]["s2"])
            it_o2 = int(m_ent["OMP2"].get("iter", 0))
            it_o3 = int(m_ent["OMP3"].get("iter", 0))

            d_uhf = max(0.0, s2_uhf - s_ideal)
            red_o2 = 100.0 * (d_uhf - max(0.0, s2_o2 - s_ideal)) / d_uhf if d_uhf > 1e-12 else 0.0
            red_o3 = 100.0 * (d_uhf - max(0.0, s2_o3 - s_ideal)) / d_uhf if d_uhf > 1e-12 else 0.0

            p_s2_u = float(p_ent.get("SCF", {}).get("s2", float("nan")))
            p_s2_2 = float(p_ent.get("OMP2", {}).get("s2", float("nan")))
            p_it_2 = int(p_ent.get("OMP2", {}).get("iter", 0))
            p_s2_3 = float(p_ent.get("OMP3", {}).get("s2", float("nan")))
            p_it_3 = int(p_ent.get("OMP3", {}).get("iter", 0))

            if mode == "CHOLESKY" and not math.isnan(p_s2_2) and abs(p_s2_2 - p_s2_u) < 1e-6 and abs(s2_o2 - s2_uhf) > 1e-4:
                cd_bug_notes.append(f"{sys_name} (Psi4 CHOLESKY OMP2 melaporkan <S^2> UHF={p_s2_2:.6f}, mshqc={s2_o2:.6f})")

            print(
                f"| {sys_name:<7} | {mode:<8} | {s_ideal:<5.2f} | {f'{s2_uhf:.6f} / {p_s2_u:.6f}':<23} | "
                f"{f'{s2_o2:.6f} / {p_s2_2:.6f}':<23} | {f'{it_o2:>2} / {p_it_2:>2} it':<12} | {red_o2:>6.2f} % | "
                f"{f'{s2_o3:.6f} / {p_s2_3:.6f}':<23} | {f'{it_o3:>2} / {p_it_3:>2} it':<12} | {red_o3:>6.2f} % |",
                flush=True,
            )
    if cd_bug_notes:
        print(" [CATATAN AUDIT PSI4] Deteksi bug upstream dfocc cd_omp2_manager (wavefunction tidak disalin kembali):", flush=True)
        for note in cd_bug_notes:
            print(f"   - {note}", flush=True)


def print_table_4(traces: Dict[str, Any], fd_summary: Dict[str, Any]) -> None:
    if not traces and not fd_summary:
        return
    print("\n" + "=" * 108, flush=True)
    print(" 4. JEJAK KONVERGENSI TR-SOSCF & VERIFIKASI GRADIEN ORBITAL ANALITIK vs FD", flush=True)
    print("=" * 108, flush=True)
    for key, td in traces.items():
        for meth in ("OMP2", "OMP3"):
            steps = td.get(meth, [])
            if steps:
                print(
                    f"  [TRACE {key:<10} {meth}] {len(steps)} langkah | "
                    f"E0={steps[0]['energy_total']:.8f} -> Ef={steps[-1]['energy_total']:.8f} | "
                    f"g_rms={steps[-1]['grad_rms']:.2e}",
                    flush=True,
                )
    for key, fd in fd_summary.items():
        print(
            f"  [FD {key:<12}] Ana={fd['ana']:+.10e} | FD={fd['fd']:+.10e} | |dw|={fd['err']:.2e}",
            flush=True,
        )


def print_table_5_and_md(
    mshqc_db: Dict[str, Any],
    psi4_db: Dict[str, Any],
    abel_db: Optional[Dict[str, Any]],
    md_out: Path,
) -> Tuple[float, float, float, float, int]:
    print("\n" + "=" * 124, flush=True)
    print(" 5A. TABEL VALIDASI HEAD-TO-HEAD 150 KONFIGURASI INTI BAB IV: MSHQC v2.0 (C1+8F) vs PSI4", flush=True)
    print("=" * 124, flush=True)
    print(
        f"| {'Sistem':<8} | {'Mode ERI':<9} | {'Metode':<6} | {'E_mshqc (Ha)':<16} | "
        f"{'E_Psi4 (Ha)':<16} | {'|dE| uHa':<10} | {'mshqc(s)':<9} | {'Psi4(s)':<9} | {'Speedup':<9} |",
        flush=True,
    )
    print(f"|{'-'*10}|{'-'*11}|{'-'*8}|{'-'*18}|{'-'*18}|{'-'*12}|{'-'*11}|{'-'*11}|{'-'*11}|", flush=True)

    md_lines = [
        "### Bab IV Head-to-Head Comparison: `mshqc v2.0.0` vs `Psi4` (`cc-pVTZ` / `cc-pVTZ-RI`)",
        "",
        "| Mode ERI | Configs | Max `|dE|` (uHa) | `mshqc C1+8F` (s) | `Psi4` (s) | Speedup (`Psi4/mshqc`) |",
        "|:---|---:|---:|---:|---:|---:|",
    ]

    tot_m, tot_p, max_de = 0.0, 0.0, 0.0
    n_matched = 0
    mode_stats = {m: {"m": 0.0, "p": 0.0, "n": 0, "max_de": 0.0} for m in ERI_MODE_ORDER}

    for sys_name in ALL_SYSTEM_ORDER:
        is_uhf = IS_UHF_MAP[sys_name]
        for mode in ERI_MODE_ORDER:
            m_ent = mshqc_db.get(f"{sys_name}|{mode}", {})
            p_ent = psi4_db.get(f"{sys_name}|{mode}", {})
            if not m_ent:
                continue
            for m_lbl, key in core_method_pairs(is_uhf):
                if key not in m_ent:
                    continue
                em, tm = float(m_ent[key]["e"]), float(m_ent[key]["t"])
                p_info = p_ent.get(key, {})
                ep = float(p_info["e"]) if "e" in p_info else float("nan")
                tp = float(p_info.get("t", 0.0))

                de_uha = abs(em - ep) * 1e6 if not math.isnan(ep) else float("nan")
                sp = (tp / max(tm, 1e-4)) if tp > 0 else 0.0

                tot_m += tm
                tot_p += tp
                mode_stats[mode]["m"] += tm
                mode_stats[mode]["p"] += tp
                if not math.isnan(de_uha):
                    n_matched += 1
                    mode_stats[mode]["n"] += 1
                    max_de = max(max_de, de_uha)
                    mode_stats[mode]["max_de"] = max(mode_stats[mode]["max_de"], de_uha)

                ep_s = f"{ep:.8f}" if not math.isnan(ep) else "-"
                de_s = f"{de_uha:.2f}" if not math.isnan(de_uha) else "-"
                print(
                    f"| {sys_name:<8} | {mode:<9} | {m_lbl:<6} | {em:<16.8f} | "
                    f"{ep_s:>16} | {de_s:>10} | {tm:<9.3f} | {tp:<9.3f} | {sp:<7.2f} x |",
                    flush=True,
                )

    tot_sp = (tot_p / max(tot_m, 1e-4)) if tot_p > 0 else 0.0
    print(f"|{'-'*10}|{'-'*11}|{'-'*8}|{'-'*18}|{'-'*18}|{'-'*12}|{'-'*11}|{'-'*11}|{'-'*11}|", flush=True)
    for mode in ERI_MODE_ORDER:
        st = mode_stats[mode]
        if st["n"] == 0 and st["m"] == 0.0:
            continue
        sp_m = (st["p"] / max(st["m"], 1e-4)) if st["p"] > 0 else 0.0
        print(
            f"| {'SUBTOTAL':<8} | {mode:<9} | {'ALL':<6} | {'-':<16} | {'-':<16} | "
            f"{st['max_de']:>10.2f} | {st['m']:<9.3f} | {st['p']:<9.3f} | {sp_m:<7.2f} x |",
            flush=True,
        )
        md_lines.append(
            f"| `{mode}` | {st['n']} | `{st['max_de']:.2f}` | `{st['m']:.3f}` | `{st['p']:.3f}` | **{sp_m:.2f}x** |"
        )
    print(f"|{'-'*10}|{'-'*11}|{'-'*8}|{'-'*18}|{'-'*18}|{'-'*12}|{'-'*11}|{'-'*11}|{'-'*11}|", flush=True)
    print(
        f"| {'TOTAL':<8} | {'ALL MODES':<9} | {'ALL':<6} | {'-':<16} | {'-':<16} | "
        f"{max_de:>10.2f} | {tot_m:<9.3f} | {tot_p:<9.3f} | {tot_sp:<7.2f} x |",
        flush=True,
    )
    print("=" * 124, flush=True)

    md_lines.append(
        f"| **TOTAL** | **{n_matched}** | **`{max_de:.2f}`** | **`{tot_m:.3f}`** | **`{tot_p:.3f}`** | **{tot_sp:.2f}x** |"
    )

    # 5B: Validasi tambahan OMP2.5 (jika tersedia pada kedua JSON)
    omp25_rows = []
    for sys_name in ALL_SYSTEM_ORDER:
        for mode in ERI_MODE_ORDER:
            m_o = mshqc_db.get(f"{sys_name}|{mode}", {}).get("OMP2.5")
            p_o = psi4_db.get(f"{sys_name}|{mode}", {}).get("OMP2.5")
            if m_o and p_o:
                omp25_rows.append((sys_name, mode, m_o, p_o))
    if omp25_rows:
        print("\n" + "=" * 124, flush=True)
        print(" 5B. VALIDASI METODE EKSTENSI OMP2.5: MSHQC v2.0 vs PSI4", flush=True)
        print("=" * 124, flush=True)
        print(
            f"| {'Sistem':<8} | {'Mode ERI':<9} | {'Metode':<6} | {'E_mshqc (Ha)':<16} | "
            f"{'E_Psi4 (Ha)':<16} | {'|dE| uHa':<10} | {'mshqc(s)':<9} | {'Psi4(s)':<9} | {'Speedup':<9} |",
            flush=True,
        )
        print(f"|{'-'*10}|{'-'*11}|{'-'*8}|{'-'*18}|{'-'*18}|{'-'*12}|{'-'*11}|{'-'*11}|{'-'*11}|", flush=True)
        tm25, tp25, max25 = 0.0, 0.0, 0.0
        for sys_name, mode, m_o, p_o in omp25_rows:
            em, tm = float(m_o["e"]), float(m_o["t"])
            ep, tp = float(p_o["e"]), float(p_o["t"])
            de = abs(em - ep) * 1e6
            sp = tp / max(tm, 1e-4)
            tm25 += tm
            tp25 += tp
            max25 = max(max25, de)
            print(
                f"| {sys_name:<8} | {mode:<9} | {'OMP2.5':<6} | {em:<16.8f} | "
                f"{ep:<16.8f} | {de:>10.2f} | {tm:<9.3f} | {tp:<9.3f} | {sp:<7.2f} x |",
                flush=True,
            )
        sp25 = tp25 / max(tm25, 1e-4)
        print("=" * 124, flush=True)
        print(f" Total OMP2.5 ({len(omp25_rows)} kasus): max|dE| = {max25:.2f} uHa | mshqc = {tm25:.3f} s | Psi4 = {tp25:.3f} s | Speedup = {sp25:.2f}x", flush=True)
        md_lines.extend([
            "",
            f"- **OMP2.5 Extended Parity ({len(omp25_rows)} runs):** max `|dE| = {max25:.2f} uHa`, `mshqc = {tm25:.3f} s` vs `Psi4 = {tp25:.3f} s` (**{sp25:.2f}x**)",
        ])

    # 5C: Ringkasan 3-Arah (C1+8F vs Abelian+8F vs Psi4)
    if abel_db:
        print("\n" + "=" * 116, flush=True)
        print(" 5C. RINGKASAN 3-ARAH (150 KONFIGURASI INTI): MSHQC (C1+8F) vs MSHQC (ABELIAN+8F) vs PSI4", flush=True)
        print("=" * 116, flush=True)
        print(
            f"| {'Mode ERI':<10} | {'mshqc C1+8F (s)':>15} | {'mshqc Abel+8F (s)':>17} | "
            f"{'Psi4 (s)':>11} | {'Psi4/C1':>9} | {'Psi4/Abel':>10} | {'max|dE_Ab-P4|':>13} |",
            flush=True,
        )
        print(f"|{'-'*12}|{'-'*17}|{'-'*19}|{'-'*13}|{'-'*11}|{'-'*12}|{'-'*15}|", flush=True)
        tot_ab = 0.0
        max_de_ab = 0.0
        for mode in ERI_MODE_ORDER:
            t_ab_m, de_ab_m = 0.0, 0.0
            for sys_name in ALL_SYSTEM_ORDER:
                a_ent = abel_db.get(f"{sys_name}|{mode}", {})
                p_ent = psi4_db.get(f"{sys_name}|{mode}", {})
                for _, key in core_method_pairs(IS_UHF_MAP[sys_name]):
                    if key in a_ent:
                        t_ab_m += float(a_ent[key]["t"])
                        if key in p_ent:
                            de_ab_m = max(de_ab_m, abs(float(a_ent[key]["e"]) - float(p_ent[key]["e"])) * 1e6)
            tot_ab += t_ab_m
            max_de_ab = max(max_de_ab, de_ab_m)
            tm_m, tp_m = mode_stats[mode]["m"], mode_stats[mode]["p"]
            if tm_m > 0.0 or t_ab_m > 0.0:
                print(
                    f"| {mode:<10} | {tm_m:>15.3f} | {t_ab_m:>17.3f} | {tp_m:>11.3f} | "
                    f"{tp_m/max(tm_m,1e-4):>8.2f}x | {tp_m/max(t_ab_m,1e-4):>9.2f}x | {de_ab_m:>10.2f} uHa |",
                    flush=True,
                )
        print(f"|{'-'*12}|{'-'*17}|{'-'*19}|{'-'*13}|{'-'*11}|{'-'*12}|{'-'*15}|", flush=True)
        print(
            f"| {'TOTAL':<10} | {tot_m:>15.3f} | {tot_ab:>17.3f} | {tot_p:>11.3f} | "
            f"{tot_p/max(tot_m,1e-4):>8.2f}x | {tot_p/max(tot_ab,1e-4):>9.2f}x | {max_de_ab:>10.2f} uHa |",
            flush=True,
        )
        print("=" * 116, flush=True)
        md_lines.append(
            f"- **3-Way Core Total:** `mshqc C1+8F = {tot_m:.3f} s` (**{tot_p/max(tot_m,1e-4):.2f}x**) | "
            f"`mshqc Abel+8F = {tot_ab:.3f} s` (**{tot_p/max(tot_ab,1e-4):.2f}x**) | `Psi4 = {tot_p:.3f} s`"
        )

    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return tot_m, tot_p, tot_sp, max_de, n_matched


def print_table_6(mshqc_ram: Dict[str, Any], psi4_ram: Dict[str, Any]) -> None:
    if not mshqc_ram:
        return
    print("\n" + "=" * 126, flush=True)
    print(" 6. PENGUKURAN PEAK RSS (VmHWM) & NET WORKING RAM TERKALIBRASI: MSHQC v2.0 vs PSI4", flush=True)
    print("=" * 126, flush=True)
    print(
        f"| {'Sistem':<7} | {'Basis':<7} | {'Mode ERI':<8} | {'Metode':<6} | "
        f"{'mshqc Peak(MB)':<14} | {'mshqc Net(MB)':<13} | {'Psi4 Peak(MB)':<13} | {'Psi4 Net(MB)':<12} | {'Rasio Peak':<10} | {'Rasio Net':<10} |",
        flush=True,
    )
    print(f"|{'-'*9}|{'-'*9}|{'-'*10}|{'-'*8}|{'-'*16}|{'-'*15}|{'-'*15}|{'-'*14}|{'-'*12}|{'-'*12}|", flush=True)
    for key, res_m in mshqc_ram.items():
        sys_name, b_name, mode, method = key.split("|")
        res_p = psi4_ram.get(key, {})
        mp, mn = float(res_m.get("peak_hwm", 0.0)), float(res_m.get("net_ram", 0.0))
        pp, pn = float(res_p.get("peak_hwm", 0.0)), float(res_p.get("net_ram", 0.0))
        r_peak = (pp / mp) if (mp > 0 and pp > 0) else 0.0
        r_net = (pn / mn) if (mn > 0 and pn > 0) else 0.0
        print(
            f"| {sys_name:<7} | {b_name:<7} | {mode:<8} | {method:<6} | "
            f"{mp:<14.2f} | {mn:<13.2f} | {pp:<13.2f} | {pn:<12.2f} | "
            f"{r_peak:<8.2f} x | {r_net:<8.2f} x |",
            flush=True,
        )
    print("=" * 126, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bab IV v2 Head-to-Head Comparator (mshqc vs Psi4)")
    parser.add_argument("--mshqc", default="mshqc_bab4_v2_raw.json")
    parser.add_argument("--psi4", default="psi4_bab4_v2_raw.json")
    parser.add_argument("--mshqc-abelian", default="mshqc_bab4_abelian_v2_raw.json")
    parser.add_argument("--out-txt", default="14_bab4_v2_compare_psi4_log.txt")
    parser.add_argument("--out-md", default="comparison_bab4_v2_summary.md")
    parser.add_argument("--tol-uha", type=float, default=5.0, help="Batas fail paritas energi (uHa)")
    args = parser.parse_args()

    m_path, p_path = Path(args.mshqc), Path(args.psi4)
    if not m_path.is_file() or not p_path.is_file():
        sys.exit(f"File tidak ditemukan: mshqc={m_path} | psi4={p_path}")

    m_pay = json.loads(m_path.read_text(encoding="utf-8"))
    p_pay = json.loads(p_path.read_text(encoding="utf-8"))
    a_path = Path(args.mshqc_abelian) if args.mshqc_abelian else None
    a_pay = json.loads(a_path.read_text(encoding="utf-8")) if (a_path and a_path.is_file()) else None

    rec = TeeRecorder()
    sys.stdout = rec
    try:
        print(f"Engine: {m_pay.get('engine')} vs {p_pay.get('engine')} (EXACT_SYM={p_pay.get('exact_symmetry', 'auto')})", flush=True)
        print_table_1(m_pay.get("table1", []))
        print_table_2(m_pay.get("table2", []))
        print_table_3(m_pay.get("ref_suite", {}), p_pay.get("ref_suite", {}))
        print_table_4(m_pay.get("traces", {}), m_pay.get("fd_summary", {}))
        tot_m, tot_p, tot_sp, max_de, n_matched = print_table_5_and_md(
            m_pay.get("ref_suite", {}),
            p_pay.get("ref_suite", {}),
            a_pay.get("ref_suite", {}) if a_pay else None,
            Path(args.out_md),
        )
        print_table_6(m_pay.get("ram_bench", {}), p_pay.get("ram_bench", {}))
        print(
            f"\n[RINGKASAN BAB IV v2] Matched={n_matched} | max|dE|={max_de:.2f} uHa | "
            f"mshqc={tot_m:.3f} s vs Psi4={tot_p:.3f} s | Speedup={tot_sp:.2f}x",
            flush=True,
        )
    finally:
        sys.stdout = rec.terminal

    Path(args.out_txt).write_text(rec.get_text(), encoding="utf-8")
    if n_matched > 0 and max_de > args.tol_uha:
        sys.exit(f"Gagal ambang paritas energi: max|dE|={max_de:.2f} uHa > {args.tol_uha:.2f} uHa")


if __name__ == "__main__":
    main()
