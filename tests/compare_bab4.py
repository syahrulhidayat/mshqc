#!/usr/bin/env python3
"""
Komparator Murni Bab IV Skripsi: MSHQC vs Psi4 (Zero QC Runtime Overhead)
Usage: python compare_bab4.py [mshqc_bab4_raw.json] [psi4_bab4_raw.json]
"""

import os
import sys
import json
import math
from pathlib import Path

ALL_SYSTEM_ORDER = ["Neon", "H2O", "Lithium", "Carbon", "Oxygen", "OH", "O2"]
ERI_MODE_ORDER = ["EXACT", "CHOLESKY", "DF"]
IS_UHF_MAP = {
    "Neon": False, "H2O": False, "Lithium": True,
    "Carbon": True, "Oxygen": True, "OH": True, "O2": True,
}


def print_table_1(table1_rows):
    print("\n" + "=" * 112)
    print("  1. TABEL KOMPRESI TENSOR ERI AO & VEKTOR CHOLESKY AKTUAL (MSHQC vs PSI4, tau_CD = 1e-9)")
    print("=" * 112)
    print(f"| {'Sistem':<8} | {'Basis':<8} | {'N_bf':<5} | {'N_aux(DF)':<9} | {'M_chol':<7} | {'Exact (MB)':<11} | {'Chol (MB)':<10} | {'DF (MB)':<9} | {'Exact/Chol':<10} | {'Exact/DF':<9} |")
    print(f"|{'-'*10}|{'-'*10}|{'-'*7}|{'-'*11}|{'-'*9}|{'-'*13}|{'-'*12}|{'-'*11}|{'-'*12}|{'-'*11}|")
    for r in table1_rows:
        print(
            f"| {r['sys']:<8} | {r['basis']:<8} | {r['nbf']:<5} | {r['naux']:<9} | {r['m_chol']:<7} | "
            f"{r['mem_exact']:<11.3f} | {r['mem_chol']:<10.3f} | {r['mem_df']:<9.3f} | "
            f"{r['r_chol']:<8.2f} x | {r['r_df']:<7.2f} x |"
        )


def print_table_2(table2_rows):
    print("\n" + "=" * 116)
    print("  2. TABEL ELIMINASI HAMBATAN MEMORI O(V^4) PADA OMP3: KONVENSIONAL vs MSHQC (DF MODE)")
    print("=" * 116)
    print(f"| {'Sistem':<7} | {'Basis':<8} | {'(O, V, N_aux)':<15} | {'Konvensional V_vvvv + Gamma_vvvv (MB)':<37} | {'MSHQC B_3c + T2 + Y_ooov (MB)':<30} | {'Reduksi Memori MO':<17} |")
    print(f"|{'-'*9}|{'-'*10}|{'-'*17}|{'-'*39}|{'-'*32}|{'-'*19}|")
    for r in table2_rows:
        ov_str = f"({r['oa']},{r['va']},{r['naux']})"
        print(
            f"| {r['sys']:<7} | {r['basis']:<8} | {ov_str:<15} | "
            f"{r['mem_conv_v4']:<37.3f} | {r['mem_mshqc']:<30.3f} | {r['ratio']:<15.2f} x |"
        )


def print_table_3(mshqc_db, psi4_db):
    print("\n" + "=" * 144)
    print("  3. TABEL PEMULIHAN SIMETRI SPIN <S^2> SEBELUM (UHF) DAN SESUDAH OPTIMISASI ORBITAL: MSHQC vs PSI4 (cc-pVTZ)")
    print("=" * 144)
    print(
        f"| {'Sistem':<7} | {'Mode':<5} | {'Ideal':<5} | {'<S^2> UHF (mshqc/Psi4)':<23} | "
        f"{'<S^2> OMP2 (mshqc/Psi4)':<23} | {'It OMP2(m/P)':<12} | {'Red OMP2':<8} | "
        f"{'<S^2> OMP3 (mshqc/Psi4)':<23} | {'It OMP3(m/P)':<12} | {'Red OMP3':<8} |"
    )
    print(f"|{'-'*9}|{'-'*7}|{'-'*7}|{'-'*25}|{'-'*25}|{'-'*14}|{'-'*10}|{'-'*25}|{'-'*14}|{'-'*10}|")

    for sys_name in ["Lithium", "Carbon", "Oxygen", "OH", "O2"]:
        for mode in ["EXACT", "DF"]:
            m_entry = mshqc_db.get(f"{sys_name}|{mode}", {})
            p_entry = psi4_db.get(f"{sys_name}|{mode}", {})
            if not m_entry:
                continue

            s_ideal = m_entry["s_ideal"]
            s2_uhf  = m_entry["SCF"]["s2"]
            s2_omp2 = m_entry["OMP2"]["s2"]
            s2_omp3 = m_entry["OMP3"]["s2"]
            it_omp2 = m_entry["OMP2"]["iter"]
            it_omp3 = m_entry["OMP3"]["iter"]

            d_uhf = max(0.0, s2_uhf - s_ideal)
            red_omp2 = 100.0 * (d_uhf - max(0.0, s2_omp2 - s_ideal)) / d_uhf if d_uhf > 1e-12 else 0.0
            red_omp3 = 100.0 * (d_uhf - max(0.0, s2_omp3 - s_ideal)) / d_uhf if d_uhf > 1e-12 else 0.0

            p_s2_uhf = p_entry.get("SCF", {}).get("s2", float("nan"))
            p_s2_o2  = p_entry.get("OMP2", {}).get("s2", float("nan"))
            p_it_o2  = p_entry.get("OMP2", {}).get("iter", 0)
            p_s2_o3  = p_entry.get("OMP3", {}).get("s2", float("nan"))
            p_it_o3  = p_entry.get("OMP3", {}).get("iter", 0)

            print(
                f"| {sys_name:<7} | {mode:<5} | {s_ideal:<5.2f} | {f'{s2_uhf:.6f} / {p_s2_uhf:.6f}':<23} | "
                f"{f'{s2_omp2:.6f} / {p_s2_o2:.6f}':<23} | {f'{it_omp2:>2} / {p_it_o2:>2} it':<12} | {red_omp2:>6.2f} % | "
                f"{f'{s2_omp3:.6f} / {p_s2_o3:.6f}':<23} | {f'{it_omp3:>2} / {p_it_o3:>2} it':<12} | {red_omp3:>6.2f} % |"
            )


def print_table_4(traces, fd_logs):
    print("\n" + "=" * 108)
    print("  4. JEJAK ITERASI PENUH TRUST-REGION SOSCF & VERIFIKASI GRADIEN (cc-pVTZ, EXACT & DF)")
    print("=" * 108)
    for sys_name, mode in [("H2O", "DF"), ("Oxygen", "DF")]:
        print(f"\n>>> Log Iterasi Lengkap TR-SOSCF: {sys_name} (cc-pVTZ, Mode={mode}) <<<")
        sys.stdout.write(traces.get(f"{sys_name}|{mode}", ""))

    print("\n>>> Uji Verifikasi Finite-Difference (C++ print_level=3) <<<")
    for sys_name in ["Neon", "H2O", "Carbon"]:
        print(f"\n[{sys_name}] Menguji FD C++ internal:")
        sys.stdout.write(fd_logs.get(sys_name, ""))


def print_table_5_and_build_md(mshqc_db, psi4_db):
    print("\n" + "=" * 124)
    print("  5A. TABEL VALIDASI HEAD-TO-HEAD METODE YANG SAMA: MSHQC vs PSI4 (HF, MP2, MP3, OMP2, OMP3 | cc-pVTZ)")
    print("=" * 124)
    print(
        f"| {'Sistem':<8} | {'Mode ERI':<9} | {'Metode':<6} | {'E_mshqc (Ha)':<16} | "
        f"{'E_Psi4 (Ha)':<16} | {'|dE| uHa':<10} | {'mshqc(s)':<9} | {'Psi4(s)':<9} | {'Speedup':<9} |"
    )
    print(f"|{'-'*10}|{'-'*11}|{'-'*8}|{'-'*18}|{'-'*18}|{'-'*12}|{'-'*11}|{'-'*11}|{'-'*11}|")

    md_lines = [
        "## Bab IV Head-to-Head Benchmark Summary: `mshqc` vs `Psi4` (`cc-pVTZ` / `cc-pVTZ-RI`)",
        "",
        "| System | ERI Mode | Method | `mshqc` Energy (Ha) | `Psi4` Energy (Ha) | `|dE|` (uHa) | `mshqc` (s) | `Psi4` (s) | Speedup |",
        "|:---|:---|:---|---:|---:|---:|---:|---:|---:|",
    ]

    tot_mshqc_s, tot_psi4_s, max_de_all = 0.0, 0.0, 0.0
    mode_times = {m: {"mshqc": 0.0, "psi4": 0.0} for m in ERI_MODE_ORDER}

    for sys_name in ALL_SYSTEM_ORDER:
        is_uhf = IS_UHF_MAP[sys_name]
        method_pairs = [
            ("UHF" if is_uhf else "RHF", "SCF"),
            ("UMP2" if is_uhf else "RMP2", "MP2"),
            ("UMP3" if is_uhf else "RMP3", "MP3"),
            ("OMP2", "OMP2"),
            ("OMP3", "OMP3"),
        ]
        for mode in ERI_MODE_ORDER:
            m_entry = mshqc_db.get(f"{sys_name}|{mode}", {})
            p_entry = psi4_db.get(f"{sys_name}|{mode}", {})
            if not m_entry:
                continue
            for m_label, key in method_pairs:
                em, tm = m_entry[key]["e"], m_entry[key]["t"]
                p_info = p_entry.get(key, {})
                ep, tp = p_info.get("e", float("nan")), p_info.get("t", 0.0)
                de_uha = abs(em - ep) * 1e6 if not math.isnan(ep) else float("nan")
                sp = (tp / max(tm, 1e-4)) if tp > 0 else 0.0

                tot_mshqc_s += tm
                tot_psi4_s += tp
                mode_times[mode]["mshqc"] += tm
                mode_times[mode]["psi4"] += tp
                if not math.isnan(de_uha):
                    max_de_all = max(max_de_all, de_uha)

                print(
                    f"| {sys_name:<8} | {mode:<9} | {m_label:<6} | {em:<16.8f} | "
                    f"{ep:<16.8f} | {de_uha:<10.2f} | {tm:<9.3f} | {tp:<9.3f} | {sp:<7.2f} x |"
                )
                md_lines.append(
                    f"| {sys_name} | `{mode.lower()}` | `{m_label}` | `{em:.8f}` | `{ep:.8f}` | "
                    f"`{de_uha:.2f}` | `{tm:.3f}` | `{tp:.3f}` | **{sp:.2f}x** |"
                )

    tot_sp = (tot_psi4_s / max(tot_mshqc_s, 1e-4)) if tot_psi4_s > 0 else 0.0
    print(f"|{'-'*10}|{'-'*11}|{'-'*8}|{'-'*18}|{'-'*18}|{'-'*12}|{'-'*11}|{'-'*11}|{'-'*11}|")
    for mode in ERI_MODE_ORDER:
        tm_m, tp_m = mode_times[mode]["mshqc"], mode_times[mode]["psi4"]
        sp_m = (tp_m / max(tm_m, 1e-4)) if tp_m > 0 else 0.0
        print(f"| {'SUBTOTAL':<8} | {mode:<9} | {'ALL':<6} | {'-':<16} | {'-':<16} | {'-':<10} | {tm_m:<9.3f} | {tp_m:<9.3f} | {sp_m:<7.2f} x |")
    print(f"|{'-'*10}|{'-'*11}|{'-'*8}|{'-'*18}|{'-'*18}|{'-'*12}|{'-'*11}|{'-'*11}|{'-'*11}|")
    print(f"| {'TOTAL':<8} | {'ALL MODES':<9} | {'ALL':<6} | {'-':<16} | {'-':<16} | {max_de_all:<10.2f} | {tot_mshqc_s:<9.3f} | {tot_psi4_s:<9.3f} | {tot_sp:<7.2f} x |")
    print("=" * 124)
    print(
        f"  Total Waktu Solver Tabel 5A (7 Sistem x 3 Mode x 5 Metode = 105 Run):\n"
        f"  -> mshqc = {tot_mshqc_s:.3f} s | Psi4 = {tot_psi4_s:.3f} s | Speedup Kumulatif = {tot_sp:.2f}x"
    )

    md_lines.append(
        f"| **TOTAL** | **ALL** | **ALL** | **—** | **—** | **Max: `{max_de_all:.2f}`** | "
        f"**`{tot_mshqc_s:.3f}`** | **`{tot_psi4_s:.3f}`** | **`{tot_sp:.2f}x`** |"
    )
    Path("comparison_bab4_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return tot_mshqc_s, tot_psi4_s, tot_sp


def print_table_6(mshqc_ram, psi4_ram):
    print("\n" + "=" * 126)
    print("  6. PENGUKURAN PEAK RSS (VmHWM) & NET WORKING RAM TERKALIBRASI SETELAH WARM-UP: MSHQC vs PSI4")
    print("=" * 126)
    print(
        f"| {'Sistem':<7} | {'Basis':<7} | {'Mode ERI':<8} | {'Metode':<6} | "
        f"{'mshqc Peak(MB)':<14} | {'mshqc Net(MB)':<13} | {'Psi4 Peak(MB)':<13} | {'Psi4 Net(MB)':<12} | {'Rasio Peak':<10} | {'Rasio Net':<10} |"
    )
    print(f"|{'-'*9}|{'-'*9}|{'-'*10}|{'-'*8}|{'-'*16}|{'-'*15}|{'-'*15}|{'-'*14}|{'-'*12}|{'-'*12}|")

    for key, res_m in mshqc_ram.items():
        sys_name, b_name, mode, method = key.split("|")
        res_p = psi4_ram.get(key, {})
        mp, mn = res_m.get("peak_hwm", 0.0), res_m.get("net_ram", 0.0)
        pp, pn = res_p.get("peak_hwm", 0.0), res_p.get("net_ram", 0.0)
        r_peak = (pp / mp) if mp > 0 and pp > 0 else 0.0
        r_net  = (pn / mn) if mn > 0 and pn > 0 else 0.0

        print(
            f"| {sys_name:<7} | {b_name:<7} | {mode:<8} | {method:<6} | "
            f"{mp:<14.2f} | {mn:<13.2f} | {pp:<13.2f} | {pn:<12.2f} | "
            f"{r_peak:<8.2f} x | {r_net:<8.2f} x |"
        )


def main():
    mshqc_path = sys.argv[1] if len(sys.argv) > 1 else "mshqc_bab4_raw.json"
    psi4_path  = sys.argv[2] if len(sys.argv) > 2 else "psi4_bab4_raw.json"

    if not os.path.exists(mshqc_path) or not os.path.exists(psi4_path):
        print(f"Usage: python {sys.argv[0]} <mshqc_bab4_raw.json> <psi4_bab4_raw.json>")
        sys.exit(1)

    mshqc_payload = json.loads(Path(mshqc_path).read_text(encoding="utf-8"))
    psi4_payload  = json.loads(Path(psi4_path).read_text(encoding="utf-8"))

    print_table_1(mshqc_payload.get("table1", []))
    print_table_2(mshqc_payload.get("table2", []))
    print_table_3(mshqc_payload.get("ref_suite", {}), psi4_payload.get("ref_suite", {}))
    print_table_4(mshqc_payload.get("traces", {}), mshqc_payload.get("fd_logs", {}))
    tot_m, tot_p, sp = print_table_5_and_build_md(mshqc_payload.get("ref_suite", {}), psi4_payload.get("ref_suite", {}))
    print_table_6(mshqc_payload.get("ram_bench", {}), psi4_payload.get("ram_bench", {}))

    print("\n" + "=" * 126)
    print("  RINGKASAN AKHIR BENCHMARK BAB IV (MSHQC vs PSI4 | cc-pVTZ / cc-pVTZ-RI)")
    print("=" * 126)
    print(f"  Total Waktu Solver (105 Run) : mshqc = {tot_m:.3f} s | Psi4 = {tot_p:.3f} s | Speedup Kumulatif = {sp:.2f}x")
    print(f"  Wall-Clock Runner Terpisah   : mshqc = {mshqc_payload.get('wall_time_s', 0.0):.2f} s | Psi4 = {psi4_payload.get('wall_time_s', 0.0):.2f} s")
    print("=" * 126)


if __name__ == "__main__":
    main()