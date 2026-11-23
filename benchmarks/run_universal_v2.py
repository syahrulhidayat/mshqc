#!/usr/bin/env python3
"""
mshqc v2.0 Pure Thin-Core Universal Validation Runner (43 sistem: atom, molekul, ion, radikal).
Menggunakan 100% API v2.0 (`MSHQCCalculator`, mode CHOLESKY & DF, isolated subprocess worker).

Usage:
  python benchmarks/run_universal_v2.py --tier quick --basis cc-pVDZ
  python benchmarks/run_universal_v2.py --tier standard --basis cc-pVDZ --extended
  python benchmarks/run_universal_v2.py --tier full --basis cc-pVDZ
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List

CPU_AVAIL = os.cpu_count() or 4
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", str(min(6, CPU_AVAIL))))
os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("TBLIS_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

import mshqc

ANG_TO_BOHR = 1.8897261246257702
CC_PV_SUPPORTED_Z = set(range(1, 19)) | {20} | set(range(31, 37))

SYSTEMS = {
    "H":           (0, 2, 1, 0, [(1, "H", 0.0, 0.0, 0.0)], "standard"),
    "He":          (0, 1, 1, 1, [(2, "He", 0.0, 0.0, 0.0)], "standard"),
    "Li":          (0, 2, 2, 1, [(3, "Li", 0.0, 0.0, 0.0)], "standard"),
    "Be":          (0, 1, 2, 2, [(4, "Be", 0.0, 0.0, 0.0)], "standard"),
    "B":           (0, 2, 3, 2, [(5, "B", 0.0, 0.0, 0.0)], "standard"),
    "C":           (0, 3, 4, 2, [(6, "C", 0.0, 0.0, 0.0)], "standard"),
    "N":           (0, 4, 5, 2, [(7, "N", 0.0, 0.0, 0.0)], "standard"),
    "O":           (0, 3, 5, 3, [(8, "O", 0.0, 0.0, 0.0)], "standard"),
    "F":           (0, 2, 5, 4, [(9, "F", 0.0, 0.0, 0.0)], "standard"),
    "Ne":          (0, 1, 5, 5, [(10, "Ne", 0.0, 0.0, 0.0)], "standard"),
    "Na":          (0, 2, 6, 5, [(11, "Na", 0.0, 0.0, 0.0)], "standard"),
    "Mg":          (0, 1, 6, 6, [(12, "Mg", 0.0, 0.0, 0.0)], "standard"),
    "Al":          (0, 2, 7, 6, [(13, "Al", 0.0, 0.0, 0.0)], "full"),
    "Si":          (0, 3, 8, 6, [(14, "Si", 0.0, 0.0, 0.0)], "full"),
    "P":           (0, 4, 9, 6, [(15, "P", 0.0, 0.0, 0.0)], "standard"),
    "S":           (0, 3, 9, 7, [(16, "S", 0.0, 0.0, 0.0)], "full"),
    "Cl":          (0, 2, 9, 8, [(17, "Cl", 0.0, 0.0, 0.0)], "full"),
    "Ar":          (0, 1, 9, 9, [(18, "Ar", 0.0, 0.0, 0.0)], "standard"),
    "K":           (0, 2, 10, 9, [(19, "K", 0.0, 0.0, 0.0)], "full"),
    "Ca":          (0, 1, 10, 10, [(20, "Ca", 0.0, 0.0, 0.0)], "full"),
    "Mn":          (0, 6, 15, 10, [(25, "Mn", 0.0, 0.0, 0.0)], "full"),
    "Zn":          (0, 1, 15, 15, [(30, "Zn", 0.0, 0.0, 0.0)], "full"),
    "As":          (0, 4, 18, 15, [(33, "As", 0.0, 0.0, 0.0)], "full"),
    "Kr":          (0, 1, 18, 18, [(36, "Kr", 0.0, 0.0, 0.0)], "full"),
    "H2":          (0, 1, 1, 1, [(1, "H", 0.0, 0.0, -0.37 * ANG_TO_BOHR), (1, "H", 0.0, 0.0, 0.37 * ANG_TO_BOHR)], "standard"),
    "LiH":         (0, 1, 2, 2, [(3, "Li", 0.0, 0.0, -0.7975 * ANG_TO_BOHR), (1, "H", 0.0, 0.0, 0.7975 * ANG_TO_BOHR)], "standard"),
    "HF":          (0, 1, 5, 5, [(9, "F", 0.0, 0.0, 0.0), (1, "H", 0.0, 0.0, 0.9168 * ANG_TO_BOHR)], "standard"),
    "HCl":         (0, 1, 9, 9, [(17, "Cl", 0.0, 0.0, 0.0), (1, "H", 0.0, 0.0, 1.2746 * ANG_TO_BOHR)], "full"),
    "N2":          (0, 1, 7, 7, [(7, "N", 0.0, 0.0, -0.5488 * ANG_TO_BOHR), (7, "N", 0.0, 0.0, 0.5488 * ANG_TO_BOHR)], "standard"),
    "CO":          (0, 1, 7, 7, [(6, "C", 0.0, 0.0, -0.5642 * ANG_TO_BOHR), (8, "O", 0.0, 0.0, 0.5642 * ANG_TO_BOHR)], "full"),
    "H2O":         (0, 1, 5, 5, [(8, "O", 0.0, 0.0, 0.1173 * ANG_TO_BOHR), (1, "H", 0.0, 0.7572 * ANG_TO_BOHR, -0.4692 * ANG_TO_BOHR), (1, "H", 0.0, -0.7572 * ANG_TO_BOHR, -0.4692 * ANG_TO_BOHR)], "standard"),
    "H2S":         (0, 1, 9, 9, [(16, "S", 0.0, 0.0, 0.1030 * ANG_TO_BOHR), (1, "H", 0.0, 0.9616 * ANG_TO_BOHR, -0.8240 * ANG_TO_BOHR), (1, "H", 0.0, -0.9616 * ANG_TO_BOHR, -0.8240 * ANG_TO_BOHR)], "full"),
    "NH3":         (0, 1, 5, 5, [(7, "N", 0.0, 0.0, 0.1165 * ANG_TO_BOHR), (1, "H", 0.0, 0.9397 * ANG_TO_BOHR, -0.2718 * ANG_TO_BOHR), (1, "H", 0.8138 * ANG_TO_BOHR, -0.46985 * ANG_TO_BOHR, -0.2718 * ANG_TO_BOHR), (1, "H", -0.8138 * ANG_TO_BOHR, -0.46985 * ANG_TO_BOHR, -0.2718 * ANG_TO_BOHR)], "standard"),
    "CH4":         (0, 1, 5, 5, [(6, "C", 0.0, 0.0, 0.0), (1, "H", 0.629118 * ANG_TO_BOHR, 0.629118 * ANG_TO_BOHR, 0.629118 * ANG_TO_BOHR), (1, "H", -0.629118 * ANG_TO_BOHR, -0.629118 * ANG_TO_BOHR, 0.629118 * ANG_TO_BOHR), (1, "H", -0.629118 * ANG_TO_BOHR, 0.629118 * ANG_TO_BOHR, -0.629118 * ANG_TO_BOHR), (1, "H", 0.629118 * ANG_TO_BOHR, -0.629118 * ANG_TO_BOHR, -0.629118 * ANG_TO_BOHR)], "standard"),
    "OH-":         (-1, 1, 5, 5, [(8, "O", 0.0, 0.0, 0.0), (1, "H", 0.0, 0.0, 0.9640 * ANG_TO_BOHR)], "standard"),
    "H3O+":        (1, 1, 5, 5, [(8, "O", 0.0, 0.0, 0.0650 * ANG_TO_BOHR), (1, "H", 0.0, 0.9570 * ANG_TO_BOHR, -0.2100 * ANG_TO_BOHR), (1, "H", 0.8288 * ANG_TO_BOHR, -0.4785 * ANG_TO_BOHR, -0.2100 * ANG_TO_BOHR), (1, "H", -0.8288 * ANG_TO_BOHR, -0.4785 * ANG_TO_BOHR, -0.2100 * ANG_TO_BOHR)], "full"),
    "OH":          (0, 2, 5, 4, [(8, "O", 0.0, 0.0, 0.0), (1, "H", 0.0, 0.0, 0.9697 * ANG_TO_BOHR)], "standard"),
    "SH":          (0, 2, 9, 8, [(16, "S", 0.0, 0.0, 0.0), (1, "H", 0.0, 0.0, 1.3409 * ANG_TO_BOHR)], "full"),
    "NH2":         (0, 2, 5, 4, [(7, "N", 0.0, 0.0, 0.1415 * ANG_TO_BOHR), (1, "H", 0.0, 0.8015 * ANG_TO_BOHR, -0.5060 * ANG_TO_BOHR), (1, "H", 0.0, -0.8015 * ANG_TO_BOHR, -0.5060 * ANG_TO_BOHR)], "standard"),
    "CH2_triplet": (0, 3, 5, 3, [(6, "C", 0.0, 0.0, 0.1050 * ANG_TO_BOHR), (1, "H", 0.0, 0.9920 * ANG_TO_BOHR, -0.3150 * ANG_TO_BOHR), (1, "H", 0.0, -0.9920 * ANG_TO_BOHR, -0.3150 * ANG_TO_BOHR)], "standard"),
    "CH3":         (0, 2, 5, 4, [(6, "C", 0.0, 0.0, 0.0), (1, "H", 0.0, 1.0790 * ANG_TO_BOHR, 0.0), (1, "H", 0.934442 * ANG_TO_BOHR, -0.5395 * ANG_TO_BOHR, 0.0), (1, "H", -0.934442 * ANG_TO_BOHR, -0.5395 * ANG_TO_BOHR, 0.0)], "full"),
    "NH":          (0, 3, 5, 3, [(7, "N", 0.0, 0.0, 0.0), (1, "H", 0.0, 0.0, 1.0362 * ANG_TO_BOHR)], "full"),
    "O2":          (0, 3, 9, 7, [(8, "O", 0.0, 0.0, -0.6035 * ANG_TO_BOHR), (8, "O", 0.0, 0.0, 0.6035 * ANG_TO_BOHR)], "standard"),
}
QUICK_SUBSET = ["H", "He", "H2O", "OH", "NH2", "OH-"]


def select_basis_for_system(spec, requested_basis: str) -> str:
    if requested_basis.lower().startswith("cc-pv"):
        if any(z not in CC_PV_SUPPORTED_Z for z, _, _, _, _ in spec[4]):
            return "def2-SVP" if "dz" in requested_basis.lower() else "def2-TZVP"
    return requested_basis


def run_mshqc_system(spec, actual_basis: str, eri_mode: str, extended: bool = False) -> Dict[str, Any]:
    charge, mult, na, nb, atoms, _ = spec
    n_elec = na + nb
    aux_basis = f"{actual_basis.lower()}-ri"

    mol = mshqc.Molecule(charge=charge, multiplicity=mult)
    for _, sym, x, y, zc in atoms:
        mol.add_atom(sym, x, y, zc, units="bohr")

    calc = mshqc.MSHQCCalculator(
        mol, basis_name=actual_basis, eri_method=eri_mode.lower(),
        aux_basis_name=aux_basis, threshold=1e-9,
    )
    t0 = time.perf_counter()
    scf_res = calc.run_scf()
    t_hf = time.perf_counter() - t0
    s2_hf = float(getattr(scf_res, "s2", 0.0))

    out: Dict[str, Any] = {
        "basis": actual_basis,
        "converged": bool(scf_res.converged),
        "HF": float(scf_res.energy_total),
        "S2_HF": s2_hf,
        "TIME": {"HF": float(t_hf)},
    }

    if n_elec > 1:
        methods_plan = [
            ("MP2", lambda: calc.run_mp2(scf_res)),
            ("MP3", lambda: calc.run_mp3(scf_res)),
            ("OMP2", lambda: calc.run_omp2(scf_res, grad_thresh=1e-5)),
            ("OMP3", lambda: calc.run_omp3(scf_res, grad_thresh=1e-5)),
        ]
        if extended:
            methods_plan.extend([
                ("LT-SOS-OMP2", lambda: calc.run_lt_sos_omp2(scf_res, grad_thresh=1e-5)),
                ("kappa-OOMP2", lambda: calc.run_kappa_omp2(scf_res, grad_thresh=1e-5)),
                ("OMP2.5", lambda: calc.run_omp25(scf_res, grad_thresh=1e-5)),
            ])
        for name, fn in methods_plan:
            t0 = time.perf_counter()
            r = fn()
            dt = time.perf_counter() - t0
            out[name] = float(r.energy_total)
            out["TIME"][name] = float(dt)
            if name in ("OMP2", "OMP3", "kappa-OOMP2"):
                out[f"S2_{name}"] = float(getattr(r, "s2", 0.0))
    else:
        for name in ("MP2", "MP3", "OMP2", "OMP3"):
            out[name] = float(scf_res.energy_total)
            out["TIME"][name] = 0.0
        out["S2_OMP2"] = out["S2_OMP3"] = s2_hf

    gc.collect()
    return out


def run_isolated_worker(sys_name: str, actual_basis: str, eri_mode: str, extended: bool, tmp_json: str) -> Dict[str, Any]:
    if os.path.exists(tmp_json):
        os.remove(tmp_json)
    cmd = [
        sys.executable, "-u", os.path.abspath(__file__),
        "--single-system", sys_name,
        "--single-basis", actual_basis,
        "--single-mode", eri_mode,
        "--single-out", tmp_json,
    ]
    if extended:
        cmd.append("--extended")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode == 0 and os.path.exists(tmp_json):
        with open(tmp_json, "r", encoding="utf-8") as f:
            return json.load(f)
    if proc.returncode < 0:
        try:
            sig_name = signal.Signals(-proc.returncode).name
        except Exception:
            sig_name = f"SIG({-proc.returncode})"
        return {"error": f"Terminated by native signal {sig_name}", "basis": actual_basis}
    tail = (proc.stdout or "").strip().splitlines()[-3:]
    return {"error": f"Worker exit {proc.returncode}: {' | '.join(tail)}", "basis": actual_basis}


def main() -> None:
    parser = argparse.ArgumentParser(description="mshqc v2.0 Universal Validation Runner (CHOLESKY & DF)")
    parser.add_argument("--tier", choices=["quick", "standard", "full"], default="standard")
    parser.add_argument("--basis", default="cc-pVDZ")
    parser.add_argument("--modes", nargs="+", default=None, help="Mode ERI (default: DF saat quick, CHOLESKY DF saat standard/full)")
    parser.add_argument("--extended", action="store_true", help="Sertakan LT-SOS-OMP2, kappa-OOMP2, OMP2.5")
    parser.add_argument("--out", default="mshqc_universal_v2_raw.json")
    parser.add_argument("--single-system", default=None)
    parser.add_argument("--single-basis", default=None)
    parser.add_argument("--single-mode", default=None)
    parser.add_argument("--single-out", default=None)
    args = parser.parse_args()

    if args.single_system:
        res = run_mshqc_system(SYSTEMS[args.single_system], args.single_basis, args.single_mode, extended=args.extended)
        if args.single_out:
            with open(args.single_out, "w", encoding="utf-8") as f:
                json.dump(res, f)
        sys.exit(0)

    modes = args.modes or (["DF"] if args.tier == "quick" else ["CHOLESKY", "DF"])
    if args.tier == "quick":
        selected = {k: SYSTEMS[k] for k in QUICK_SUBSET}
    elif args.tier == "standard":
        selected = {k: v for k, v in SYSTEMS.items() if v[5] == "standard"}
    else:
        selected = SYSTEMS

    t_start = time.perf_counter()
    print("=" * 112, flush=True)
    print(
        f" MSHQC v{mshqc.__version__} UNIVERSAL VALIDATION RUNNER "
        f"(TIER: {args.tier.upper()}, BASIS: {args.basis}, MODES: {','.join(modes)}, THREADS: {N_THREADS})",
        flush=True,
    )
    print("=" * 112, flush=True)

    results_db: Dict[str, Any] = {}
    failures: List[str] = []
    tmp_worker = f"{args.out}.worker_tmp.json"

    for mode in modes:
        print(f"\n>>> [MSHQC v2.0] Mode ERI: {mode} (Basis Utama: {args.basis}) <<<", flush=True)
        for sys_name, spec in selected.items():
            actual_basis = select_basis_for_system(spec, args.basis)
            key = f"{args.basis}|{mode}|{sys_name}"
            res = run_isolated_worker(sys_name, actual_basis, mode, args.extended, tmp_worker)
            results_db[key] = res
            if "error" not in res and res.get("converged", False):
                t_sum = sum(res["TIME"].values())
                print(
                    f"  [PASS] {sys_name:<11} ({actual_basis:<9} | {mode:<8}) -> "
                    f"HF: {res['HF']:12.6f} | MP2: {res['MP2']:12.6f} | MP3: {res['MP3']:12.6f} | "
                    f"OMP2: {res['OMP2']:12.6f} | OMP3: {res['OMP3']:14.8f} Ha ({t_sum:.3f} s)",
                    flush=True,
                )
            else:
                msg = res.get("error", "SCF did not converge")
                failures.append(f"{key}: {msg}")
                print(f"  [FAIL] {sys_name:<11} ({actual_basis:<9} | {mode:<8}) -> {msg}", flush=True)

    if os.path.exists(tmp_worker):
        os.remove(tmp_worker)

    wall_s = time.perf_counter() - t_start
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "engine": f"mshqc-v{mshqc.__version__}",
            "tier": args.tier,
            "basis": args.basis,
            "modes": modes,
            "threads": N_THREADS,
            "wall_time_s": wall_s,
            "n_passed": len(results_db) - len(failures),
            "n_failed": len(failures),
            "failures": failures,
            "results": results_db,
        }, f, indent=2)

    print(
        f"\n[SELESAI] Universal v2.0: {len(results_db) - len(failures)}/{len(results_db)} PASS "
        f"-> {args.out} ({wall_s:.2f} s)",
        flush=True,
    )
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
