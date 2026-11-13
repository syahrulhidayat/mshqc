#!/usr/bin/env python3
"""
Komparator Head-to-Head Optimisasi Geometri & Vibrasi: MSHQC vs Psi4
Usage: python tests/compare_geom.py [mshqc_geom_raw.json] [psi4_geom_raw.json]
Output: test_geom_validation.txt
"""

import sys
import json
from pathlib import Path


def main():
    m_path = Path(sys.argv[1] if len(sys.argv) > 1 else "mshqc_geom_raw.json")
    p_path = Path(sys.argv[2] if len(sys.argv) > 2 else "psi4_geom_raw.json")

    if not m_path.is_file():
        print(f"File {m_path} tidak ditemukan.")
        sys.exit(1)

    m_db = json.loads(m_path.read_text(encoding="utf-8"))
    p_db = json.loads(p_path.read_text(encoding="utf-8")) if p_path.is_file() else {}

    out_lines = []

    def emit(s: str = ""):
        print(s, flush=True)
        out_lines.append(s)

    emit("\n" + "=" * 138)
    emit(" 1. TABEL KOMPARASI GEOMETRI EKUILIBRIUM (R_eq), ENERGI OPTIMUM (E_opt), DAN WAKTU EKSEKUSI: MSHQC vs PSI4 (DF MODE)")
    emit("=" * 138)
    hdr = (
        f"| {'System':<6} | {'Basis':<7} | {'Method':<6} | {'R_eq mshqc (A)':<14} | {'R_eq Psi4 (A)':<13} | "
        f"{'|dR| (mA)':<9} | {'E_opt mshqc (Ha)':<16} | {'E_opt Psi4 (Ha)':<16} | {'|dE| (uHa)':<10} | "
        f"{'mshqc (s)':<9} | {'Psi4 (s)':<8} | {'Speedup':<7} |"
    )
    sep = "-" * len(hdr)
    emit(sep)
    emit(hdr)
    emit(sep)

    max_dr_ma = 0.0
    max_de_uha = 0.0
    tot_t_m = 0.0
    tot_t_p = 0.0
    matched = 0

    for key, m_row in m_db.items():
        p_row = p_db.get(key)
        sys_name = m_row["system"]
        basis = m_row["basis"]
        method = m_row["method"]
        r_m = m_row["r_eq"]
        e_m = m_row["e_opt"]
        t_m = m_row["time_opt_s"]

        if p_row:
            r_p = p_row["r_eq"]
            e_p = p_row["e_opt"]
            t_p = p_row["time_opt_s"]
            dr_ma = abs(r_m - r_p) * 1e3
            de_uha = abs(e_m - e_p) * 1e6
            sp = t_p / max(t_m, 1e-4) if t_p > 0 else 0.0

            max_dr_ma = max(max_dr_ma, dr_ma)
            max_de_uha = max(max_de_uha, de_uha)
            tot_t_m += t_m
            tot_t_p += t_p
            matched += 1

            emit(
                f"| {sys_name:<6} | {basis:<7} | {method:<6} | {r_m:<14.6f} | {r_p:<13.6f} | "
                f"{dr_ma:<9.3f} | {e_m:<16.10f} | {e_p:<16.10f} | {de_uha:<10.2f} | "
                f"{t_m:<9.3f} | {t_p:<8.3f} | {sp:<6.2f}x |"
            )
        else:
            emit(
                f"| {sys_name:<6} | {basis:<7} | {method:<6} | {r_m:<14.6f} | {'N/A':<13} | "
                f"{'N/A':<9} | {e_m:<16.10f} | {'N/A':<16} | {'N/A':<10} | "
                f"{t_m:<9.3f} | {'N/A':<8} | {'N/A':<7} |"
            )

    emit(sep)
    if matched > 0:
        tot_sp = tot_t_p / max(tot_t_m, 1e-4)
        emit(
            f"| {'TOTAL':<6} | {'ALL':<7} | {'ALL':<6} | {'-':<14} | {'-':<13} | "
            f"{max_dr_ma:<9.3f} | {'-':<16} | {'-':<16} | {max_de_uha:<10.2f} | "
            f"{tot_t_m:<9.3f} | {tot_t_p:<8.3f} | {tot_sp:<6.2f}x |"
        )
        emit(sep)
        emit(
            f"Total Matched Geometry Runs: {matched} | Max |dR|: {max_dr_ma:.3f} mA | "
            f"Max |dE|: {max_de_uha:.2f} uHa | Cumulative Speedup: {tot_sp:.2f}x\n"
        )

    emit("=" * 106)
    emit(" 2. TABEL DIAGNOSTIK KONVERGENSI GEOMETRI, FREKUENSI VIBRASI HARMONIK, DAN ZPE (MSHQC C++20 ENGINE)")
    emit("=" * 106)
    hdr2 = (
        f"| {'System':<6} | {'Basis':<7} | {'Method':<6} | {'Iter':<4} | {'Grad RMS (Ha/Bohr)':<18} | "
        f"{'Angle (deg)':<11} | {'Max Freq (cm-1)':<15} | {'ZPE (kcal/mol)':<14} | {'Status':<8} |"
    )
    sep2 = "-" * len(hdr2)
    emit(sep2)
    emit(hdr2)
    emit(sep2)
    for _, m_row in m_db.items():
        ang_s = f"{m_row['angle_eq']:.4f}" if m_row["angle_eq"] > 0 else "   -   "
        status = "CONV" if m_row["converged"] else "WARN"
        emit(
            f"| {m_row['system']:<6} | {m_row['basis']:<7} | {m_row['method']:<6} | {m_row['iterations']:<4d} | "
            f"{m_row['g_rms']:<18.2e} | {ang_s:<11} | {m_row['max_freq_cm1']:<15.2f} | "
            f"{m_row['zpe_kcal']:<14.4f} | {status:<8} |"
        )
    emit(sep2 + "\n")

    Path("test_geom_validation.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
