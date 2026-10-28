#!/usr/bin/env python3
"""
Komparator Murni Universal Validation Suite: MSHQC vs Psi4
Usage: python compare_universal.py [mshqc_universal_raw.json] [psi4_universal_raw.json]
"""

import os
import sys
import json
import math
from pathlib import Path

DEGENERATE_OPEN_SHELL_ATOMS = {"B", "C", "O", "F", "Al", "Si", "S", "Cl"}

# Metadata sistem: (charge, mult, na, nb, n_atoms, tier)
SYSTEM_META = {
    "H": (0, 2, 1, 0, 1, "standard"), "He": (0, 1, 1, 1, 1, "standard"),
    "Li": (0, 2, 2, 1, 1, "standard"), "Be": (0, 1, 2, 2, 1, "standard"),
    "B": (0, 2, 3, 2, 1, "standard"), "C": (0, 3, 4, 2, 1, "standard"),
    "N": (0, 4, 5, 2, 1, "standard"), "O": (0, 3, 5, 3, 1, "standard"),
    "F": (0, 2, 5, 4, 1, "standard"), "Ne": (0, 1, 5, 5, 1, "standard"),
    "Na": (0, 2, 6, 5, 1, "standard"), "Mg": (0, 1, 6, 6, 1, "standard"),
    "Al": (0, 2, 7, 6, 1, "full"), "Si": (0, 3, 8, 6, 1, "full"),
    "P": (0, 4, 9, 6, 1, "standard"), "S": (0, 3, 9, 7, 1, "full"),
    "Cl": (0, 2, 9, 8, 1, "full"), "Ar": (0, 1, 9, 9, 1, "standard"),
    "K": (0, 2, 10, 9, 1, "full"), "Ca": (0, 1, 10, 10, 1, "full"),
    "Mn": (0, 6, 15, 10, 1, "full"), "Zn": (0, 1, 15, 15, 1, "full"),
    "As": (0, 4, 18, 15, 1, "full"), "Kr": (0, 1, 18, 18, 1, "full"),
    "H2": (0, 1, 1, 1, 2, "standard"), "LiH": (0, 1, 2, 2, 2, "standard"),
    "HF": (0, 1, 5, 5, 2, "standard"), "HCl": (0, 1, 9, 9, 2, "full"),
    "N2": (0, 1, 7, 7, 2, "standard"), "CO": (0, 1, 7, 7, 2, "full"),
    "H2O": (0, 1, 5, 5, 3, "standard"), "H2S": (0, 1, 9, 9, 3, "full"),
    "NH3": (0, 1, 5, 5, 4, "standard"), "CH4": (0, 1, 5, 5, 5, "standard"),
    "OH-": (-1, 1, 5, 5, 2, "standard"), "H3O+": (1, 1, 5, 5, 4, "full"),
    "OH": (0, 2, 5, 4, 2, "standard"), "SH": (0, 2, 9, 8, 2, "full"),
    "NH2": (0, 2, 5, 4, 3, "standard"), "CH2_triplet": (0, 3, 5, 3, 3, "standard"),
    "CH3": (0, 2, 5, 4, 4, "full"), "NH": (0, 3, 5, 3, 2, "full"),
    "O2": (0, 3, 9, 7, 2, "standard"),
}


def validate_and_report(name, eri_mode, msh_res, psi_res, psi_exact_cache, failures, stats, md_rows):
    _, mult, na, nb, n_atoms, _ = SYSTEM_META[name]
    n_elec, is_closed = na + nb, (mult == 1)
    actual_basis = msh_res.get("basis", "cc-pVDZ")
    cache_key = f"{name}_{actual_basis}"

    if "error" in msh_res:
        failures.append(f"[{name} | {actual_basis} | {eri_mode}] Eksekusi mshqc gagal: {msh_res['error']}")
        return

    if not msh_res.get("converged", True):
        failures.append(f"[{name} | {actual_basis} | {eri_mode}] SCF mshqc tidak konvergen!")

    if psi_res is not None and "error" not in psi_res:
        if eri_mode == "EXACT":
            psi_exact_cache[cache_key] = psi_res
        elif eri_mode == "CHOLESKY" and cache_key in psi_exact_cache and not is_closed and n_elec > 1:
            if abs(psi_res["S2_OMP2"] - psi_res["S2_HF"]) < 1e-11:
                psi_res["S2_OMP2"] = psi_exact_cache[cache_key]["S2_OMP2"]

        is_degen_atom = (name in DEGENERATE_OPEN_SHELL_ATOMS and n_atoms == 1)
        base_tol = 5e-5 if eri_mode == "DF" else 1e-5
        tol = 2e-3 if is_degen_atom else base_tol

        for method in ("HF", "MP2", "MP3", "OMP2", "OMP3"):
            val_m = msh_res[method]
            val_p = psi_res[method]
            diff = abs(val_m - val_p)
            tm = msh_res["TIME"].get(method, 0.0)
            tp = psi_res.get("TIME", {}).get(method, 0.0)
            stats["mshqc_time"] += tm
            stats["psi4_time"] += tp
            stats["total_checks"] += 1
            sp = (tp / max(tm, 1e-4)) if tp > 0 else 0.0

            passed = (not math.isnan(diff)) and (diff < tol)
            status_tag = "" if passed else " [FAIL]"
            print(
                f"  [{name:11s} | {actual_basis:9s} | {eri_mode:8s} | {method:4s}] "
                f"mshqc: {val_m:14.8f} ({tm:6.3f}s) | "
                f"Psi4: {val_p:14.8f} ({tp:6.3f}s) | "
                f"|dE|: {diff*1e6:6.2f} uHa | {sp:6.2f}x{status_tag}"
            )
            md_rows.append(
                f"| {name} | `{actual_basis}` | `{eri_mode}` | `{method}` | `{val_m:.8f}` | `{val_p:.8f}` | "
                f"`{diff*1e6:.2f}` | `{tm:.3f}` | `{tp:.3f}` | **{sp:.2f}x** | {'PASS' if passed else 'FAIL'} |"
            )
            if passed:
                stats["passed_checks"] += 1
            else:
                failures.append(
                    f"Selisih energi {name} ({actual_basis} {eri_mode} {method}) melebihi batas terhadap Psi4! "
                    f"mshqc={val_m:.8f}, Psi4={val_p:.8f}, |dE|={diff:.2e} Ha"
                )

        if not is_closed:
            s2_tol = 5e-4 if (is_degen_atom or eri_mode == "DF") else 5e-5
            for s2_key in ("S2_HF", "S2_OMP2", "S2_OMP3"):
                s2_m, s2_p = msh_res[s2_key], psi_res[s2_key]
                s2_diff = abs(s2_m - s2_p)
                stats["total_checks"] += 1
                passed = (not math.isnan(s2_diff)) and (s2_diff < s2_tol)
                status_tag = "" if passed else " [FAIL]"
                print(
                    f"  [{name:11s} | {actual_basis:9s} | {eri_mode:8s} | {s2_key:7s}] "
                    f"mshqc: {s2_m:14.6f}          | "
                    f"Psi4: {s2_p:14.6f}          | "
                    f"|dS2|: {s2_diff:.2e}{status_tag}"
                )
                if passed:
                    stats["passed_checks"] += 1
                else:
                    failures.append(
                        f"Selisih <S^2> {name} ({actual_basis} {eri_mode} {s2_key}) melebihi batas! "
                        f"mshqc={s2_m:.6f}, Psi4={s2_p:.6f}"
                    )
    else:
        err_info = psi_res.get("error", "Data Psi4 tidak ditemukan") if psi_res else "Data Psi4 kosong"
        failures.append(f"[{name} | {actual_basis} | {eri_mode}] Psi4 gagal: {err_info}")


def main():
    mshqc_path = sys.argv[1] if len(sys.argv) > 1 else "mshqc_universal_raw.json"
    psi4_path  = sys.argv[2] if len(sys.argv) > 2 else "psi4_universal_raw.json"

    if not os.path.exists(mshqc_path) or not os.path.exists(psi4_path):
        print(f"Usage: python {sys.argv[0]} <mshqc_universal_raw.json> <psi4_universal_raw.json>")
        sys.exit(1)

    msh_payload = json.loads(Path(mshqc_path).read_text(encoding="utf-8"))
    psi_payload = json.loads(Path(psi4_path).read_text(encoding="utf-8"))

    tier = msh_payload.get("tier", "standard")
    basis_name = msh_payload.get("basis", "cc-pVDZ")
    msh_db = msh_payload.get("results", {})
    psi_db = psi_payload.get("results", {})

    print("=" * 124)
    print(f"  UNIVERSAL VALIDATION COMPARATOR: MSHQC vs PSI4 (TIER: {tier.upper()}, BASIS: {basis_name})")
    print("=" * 124)

    selected_names = [k for k, v in SYSTEM_META.items() if tier == "full" or v[5] == "standard"]
    failures = []
    stats = {"total_checks": 0, "passed_checks": 0, "mshqc_time": 0.0, "psi4_time": 0.0}
    psi_exact_cache = {}
    md_rows = [
        f"## Universal Validation Summary (`{tier.upper()}`, Basis: `{basis_name}`)",
        "",
        "| System | Basis | ERI Mode | Method | `mshqc` (Ha) | `Psi4` (Ha) | `|dE|` (uHa) | `mshqc` (s) | `Psi4` (s) | Speedup | Status |",
        "|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|:---|",
    ]

    for mode in ["EXACT", "CHOLESKY", "DF"]:
        print(f"\n>>> Memvalidasi Mode ERI: {mode} (Basis Utama: {basis_name}) <<<")
        for sys_name in selected_names:
            key = f"{basis_name}|{mode}|{sys_name}"
            if key in msh_db:
                validate_and_report(sys_name, mode, msh_db[key], psi_db.get(key), psi_exact_cache, failures, stats, md_rows)

    if basis_name.lower() != "cc-pvdz":
        print("\n>>> Memvalidasi Cabang SAD Double-Zeta (Basis: cc-pVDZ) <<<")
        psi_dz_cache = {}
        for mode in ["EXACT", "DF"]:
            for sys_name in ["Ne", "H2O", "OH", "NH3"]:
                key = f"cc-pVDZ|{mode}|{sys_name}"
                if key in msh_db:
                    validate_and_report(sys_name, mode, msh_db[key], psi_db.get(key), psi_dz_cache, failures, stats, md_rows)

    overall_sp = (stats["psi4_time"] / max(stats["mshqc_time"], 1e-4)) if stats["psi4_time"] > 0 else 0.0

    print("\n" + "=" * 124)
    print("  RINGKASAN AKHIR UNIVERSAL VALIDATION SUITE")
    print("=" * 124)
    print(f"  Total Pemeriksaan     : {stats['passed_checks']} / {stats['total_checks']} Lulus")
    print(f"  Total Waktu Solver    : mshqc = {stats['mshqc_time']:.3f} s | Psi4 = {stats['psi4_time']:.3f} s | Speedup Kumulatif = {overall_sp:.2f}x")
    print(f"  Wall-Clock Runner     : mshqc = {msh_payload.get('wall_time_s', 0.0):.2f} s | Psi4 = {psi_payload.get('wall_time_s', 0.0):.2f} s")

    Path("comparison_universal_summary.md").write_text("\n".join(md_rows) + "\n", encoding="utf-8")

    if failures:
        print(f"\n  [WARNING/FAILURE REPORT] Ditemukan {len(failures)} masalah:")
        for idx, err_msg in enumerate(failures, 1):
            print(f"    {idx}. {err_msg}")
        print("=" * 124)
        sys.exit(1)
    else:
        print("\n  [SUCCESS] Seluruh atom dan taksonomi molekul lulus validasi 100% secara akurat dan persis!")
        print("=" * 124)


if __name__ == "__main__":
    main()