# ==============================================================================
# Git Commit: fix(benchmark): fix indentation error in post-HF loop and remove duplicate imports
# ==============================================================================

#!/usr/bin/env python3
"""
Simplified PSI4 Universal Validation Runner (JOSS-Compliant Fair Benchmark, Clean Wfn State & Real-Time Output)
Output: psi4_universal_raw.json
"""

import os
import gc
import sys
import time
import json
import math
import shutil
import signal
import tempfile
import argparse
import subprocess
import numpy as np

try:
    os.sched_setaffinity(0, set(range(os.cpu_count() or 4)))
except Exception:
    pass

CPU_AVAIL = os.cpu_count() or 4
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", str(min(4, CPU_AVAIL))))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import psi4

ANG_TO_BOHR = 1.8897261246257702
CC_PV_SUPPORTED_Z = set(range(1, 19)) | {20} | set(range(31, 37))

SYSTEMS = {
    "H":  (0, 2, 1, 0, [(1,  "H",  0.0, 0.0, 0.0)], "standard"),
    "He": (0, 1, 1, 1, [(2,  "He", 0.0, 0.0, 0.0)], "standard"),
    "Li": (0, 2, 2, 1, [(3,  "Li", 0.0, 0.0, 0.0)], "standard"),
    "Be": (0, 1, 2, 2, [(4,  "Be", 0.0, 0.0, 0.0)], "standard"),
    "B":  (0, 2, 3, 2, [(5,  "B",  0.0, 0.0, 0.0)], "standard"),
    "C":  (0, 3, 4, 2, [(6,  "C",  0.0, 0.0, 0.0)], "standard"),
    "N":  (0, 4, 5, 2, [(7,  "N",  0.0, 0.0, 0.0)], "standard"),
    "O":  (0, 3, 5, 3, [(8,  "O",  0.0, 0.0, 0.0)], "standard"),
    "F":  (0, 2, 5, 4, [(9,  "F",  0.0, 0.0, 0.0)], "standard"),
    "Ne": (0, 1, 5, 5, [(10, "Ne", 0.0, 0.0, 0.0)], "standard"),
    "Na": (0, 2, 6, 5, [(11, "Na", 0.0, 0.0, 0.0)], "standard"),
    "Mg": (0, 1, 6, 6, [(12, "Mg", 0.0, 0.0, 0.0)], "standard"),
    "Al": (0, 2, 7, 6, [(13, "Al", 0.0, 0.0, 0.0)], "full"),
    "Si": (0, 3, 8, 6, [(14, "Si", 0.0, 0.0, 0.0)], "full"),
    "P":  (0, 4, 9, 6, [(15, "P",  0.0, 0.0, 0.0)], "standard"),
    "S":  (0, 3, 9, 7, [(16, "S",  0.0, 0.0, 0.0)], "full"),
    "Cl": (0, 2, 9, 8, [(17, "Cl", 0.0, 0.0, 0.0)], "full"),
    "Ar": (0, 1, 9, 9, [(18, "Ar", 0.0, 0.0, 0.0)], "standard"),
    "K":  (0, 2, 10, 9,  [(19, "K",  0.0, 0.0, 0.0)], "full"),
    "Ca": (0, 1, 10, 10, [(20, "Ca", 0.0, 0.0, 0.0)], "full"),
    "Mn": (0, 6, 15, 10, [(25, "Mn", 0.0, 0.0, 0.0)], "full"),
    "Zn": (0, 1, 15, 15, [(30, "Zn", 0.0, 0.0, 0.0)], "full"),
    "As": (0, 4, 18, 15, [(33, "As", 0.0, 0.0, 0.0)], "full"),
    "Kr": (0, 1, 18, 18, [(36, "Kr", 0.0, 0.0, 0.0)], "full"),
    "H2": (0, 1, 1, 1, [(1, "H", 0.0, 0.0, -0.37 * ANG_TO_BOHR), (1, "H", 0.0, 0.0, 0.37 * ANG_TO_BOHR)], "standard"),
    "LiH": (0, 1, 2, 2, [(3, "Li", 0.0, 0.0, -0.7975 * ANG_TO_BOHR), (1, "H", 0.0, 0.0, 0.7975 * ANG_TO_BOHR)], "standard"),
    "HF": (0, 1, 5, 5, [(9, "F", 0.0, 0.0, 0.0), (1, "H", 0.0, 0.0, 0.9168 * ANG_TO_BOHR)], "standard"),
    "HCl": (0, 1, 9, 9, [(17, "Cl", 0.0, 0.0, 0.0), (1, "H", 0.0, 0.0, 1.2746 * ANG_TO_BOHR)], "full"),
    "N2": (0, 1, 7, 7, [(7, "N", 0.0, 0.0, -0.5488 * ANG_TO_BOHR), (7, "N", 0.0, 0.0, 0.5488 * ANG_TO_BOHR)], "standard"),
    "CO": (0, 1, 7, 7, [(6, "C", 0.0, 0.0, -0.5642 * ANG_TO_BOHR), (8, "O", 0.0, 0.0, 0.5642 * ANG_TO_BOHR)], "full"),
    "H2O": (0, 1, 5, 5, [
        (8, "O", 0.0, 0.0, 0.117300 * ANG_TO_BOHR),
        (1, "H", 0.0, 0.757200 * ANG_TO_BOHR, -0.469200 * ANG_TO_BOHR),
        (1, "H", 0.0, -0.757200 * ANG_TO_BOHR, -0.469200 * ANG_TO_BOHR),
    ], "standard"),
    "H2S": (0, 1, 9, 9, [
        (16, "S", 0.0, 0.0, 0.103000 * ANG_TO_BOHR),
        (1, "H", 0.0, 0.961600 * ANG_TO_BOHR, -0.824000 * ANG_TO_BOHR),
        (1, "H", 0.0, -0.961600 * ANG_TO_BOHR, -0.824000 * ANG_TO_BOHR),
    ], "full"),
    "NH3": (0, 1, 5, 5, [
        (7, "N", 0.0, 0.0, 0.116500 * ANG_TO_BOHR),
        (1, "H", 0.0, 0.939700 * ANG_TO_BOHR, -0.271800 * ANG_TO_BOHR),
        (1, "H", 0.813800 * ANG_TO_BOHR, -0.469850 * ANG_TO_BOHR, -0.271800 * ANG_TO_BOHR),
        (1, "H", -0.813800 * ANG_TO_BOHR, -0.469850 * ANG_TO_BOHR, -0.271800 * ANG_TO_BOHR),
    ], "standard"),
    "CH4": (0, 1, 5, 5, [
        (6, "C", 0.0, 0.0, 0.0),
        (1, "H", 0.629118 * ANG_TO_BOHR, 0.629118 * ANG_TO_BOHR, 0.629118 * ANG_TO_BOHR),
        (1, "H", -0.629118 * ANG_TO_BOHR, -0.629118 * ANG_TO_BOHR, 0.629118 * ANG_TO_BOHR),
        (1, "H", -0.629118 * ANG_TO_BOHR, 0.629118 * ANG_TO_BOHR, -0.629118 * ANG_TO_BOHR),
        (1, "H", 0.629118 * ANG_TO_BOHR, -0.629118 * ANG_TO_BOHR, -0.629118 * ANG_TO_BOHR),
    ], "standard"),
    "OH-": (-1, 1, 5, 5, [(8, "O", 0.0, 0.0, 0.0), (1, "H", 0.0, 0.0, 0.9640 * ANG_TO_BOHR)], "standard"),
    "H3O+": (1, 1, 5, 5, [
        (8, "O", 0.0, 0.0, 0.065000 * ANG_TO_BOHR),
        (1, "H", 0.0, 0.957000 * ANG_TO_BOHR, -0.210000 * ANG_TO_BOHR),
        (1, "H", 0.828800 * ANG_TO_BOHR, -0.478500 * ANG_TO_BOHR, -0.210000 * ANG_TO_BOHR),
        (1, "H", -0.828800 * ANG_TO_BOHR, -0.478500 * ANG_TO_BOHR, -0.210000 * ANG_TO_BOHR),
    ], "full"),
    "OH": (0, 2, 5, 4, [(8, "O", 0.0, 0.0, 0.0), (1, "H", 0.0, 0.0, 0.9697 * ANG_TO_BOHR)], "standard"),
    "SH": (0, 2, 9, 8, [(16, "S", 0.0, 0.0, 0.0), (1, "H", 0.0, 0.0, 1.3409 * ANG_TO_BOHR)], "full"),
    "NH2": (0, 2, 5, 4, [
        (7, "N", 0.0, 0.0, 0.141500 * ANG_TO_BOHR),
        (1, "H", 0.0, 0.801500 * ANG_TO_BOHR, -0.506000 * ANG_TO_BOHR),
        (1, "H", 0.0, -0.801500 * ANG_TO_BOHR, -0.506000 * ANG_TO_BOHR),
    ], "standard"),
    "CH2_triplet": (0, 3, 5, 3, [
        (6, "C", 0.0, 0.0, 0.105000 * ANG_TO_BOHR),
        (1, "H", 0.0, 0.992000 * ANG_TO_BOHR, -0.315000 * ANG_TO_BOHR),
        (1, "H", 0.0, -0.992000 * ANG_TO_BOHR, -0.315000 * ANG_TO_BOHR),
    ], "standard"),
    "CH3": (0, 2, 5, 4, [
        (6, "C", 0.0, 0.0, 0.0),
        (1, "H", 0.0, 1.079000 * ANG_TO_BOHR, 0.0),
        (1, "H", 0.934442 * ANG_TO_BOHR, -0.539500 * ANG_TO_BOHR, 0.0),
        (1, "H", -0.934442 * ANG_TO_BOHR, -0.539500 * ANG_TO_BOHR, 0.0),
    ], "full"),
    "NH": (0, 3, 5, 3, [(7, "N", 0.0, 0.0, 0.0), (1, "H", 0.0, 0.0, 1.0362 * ANG_TO_BOHR)], "full"),
    "O2": (0, 3, 9, 7, [(8, "O", 0.0, 0.0, -0.6035 * ANG_TO_BOHR), (8, "O", 0.0, 0.0, 0.6035 * ANG_TO_BOHR)], "standard"),
}

def select_basis_for_system(spec, requested_basis: str) -> str:
    if requested_basis.lower().startswith("cc-pv"):
        if any(z not in CC_PV_SUPPORTED_Z for z, _, _, _, _ in spec[4]):
            return "def2-SVP" if "dz" in requested_basis.lower() else "def2-TZVP"
    return requested_basis

def compute_s2_from_wfn(wfn, S_mat: np.ndarray, na: int, nb: int, is_closed: bool) -> float:
    if is_closed or na == nb:
        return 0.0
    sz = 0.5 * (na - nb)
    ideal = float(sz * (sz + 1.0))
    if na == 0 or nb == 0:
        return ideal
    try:
        ca = np.array(wfn.Ca().to_array(dense=True), dtype=np.float64, copy=True)[:, :na]
        cb = np.array(wfn.Cb().to_array(dense=True), dtype=np.float64, copy=True)[:, :nb]
        sab = ca.T @ S_mat @ cb
        val = float(ideal + nb - np.sum(sab ** 2))
        if not math.isfinite(val):
            raise ValueError("Non-finite S2 value computed")
        return val
    except Exception:
        return ideal

def snapshot_scf_wfn(wfn) -> dict:
    return {
        "Ca": wfn.Ca().clone(),
        "Cb": wfn.Cb().clone(),
        "Da": wfn.Da().clone(),
        "Db": wfn.Db().clone(),
        "Fa": wfn.Fa().clone(),
        "Fb": wfn.Fb().clone(),
        "ea": wfn.epsilon_a().clone(),
        "eb": wfn.epsilon_b().clone(),
    }

def restore_scf_wfn(wfn, snap: dict):
    wfn.Ca().copy(snap["Ca"])
    wfn.Cb().copy(snap["Cb"])
    wfn.Da().copy(snap["Da"])
    wfn.Db().copy(snap["Db"])
    wfn.Fa().copy(snap["Fa"])
    wfn.Fb().copy(snap["Fb"])
    wfn.epsilon_a().copy(snap["ea"])
    wfn.epsilon_b().copy(snap["eb"])

def assert_finite(name: str, val: float):
    if val is None or not math.isfinite(float(val)):
        raise RuntimeError(f"Psi4 returned non-finite energy ({val}) for {name}")

def run_psi4_system(spec, actual_basis: str, eri_mode: str) -> dict:
    charge, mult, na, nb, atoms, _ = spec
    n_elec, is_closed = na + nb, (mult == 1)
    mode_lower = eri_mode.lower()
    aux_basis = f"{actual_basis.lower()}-ri"
    ref_type = "rhf" if is_closed else "uhf"

    has_ca = any(z == 20 for z, _, _, _, _ in atoms)
    mp2_aux = "def2-svp-ri" if (has_ca and aux_basis.startswith("cc-pv")) else aux_basis
    scf_aux = "def2-universal-jkfit" if (has_ca and aux_basis.startswith("cc-pv")) else aux_basis

    scratch_dir = tempfile.mkdtemp(prefix="psi4_scr_")
    try:
        psi4.core.be_quiet()
        psi4.set_num_threads(N_THREADS)
        psi4.set_memory("4 GB")
        psi4.core.IOManager.shared_object().set_default_path(scratch_dir)

        geom_lines = [
            f"{charge} {mult}",
            "units bohr",
            "symmetry c1",
            "no_reorient",
            "no_com",
        ]
        for _, sym, x, y, zc in atoms:
            geom_lines.append(f"{sym} {x:.12f} {y:.12f} {zc:.12f}")
        mol = psi4.geometry("\n".join(geom_lines))

        base_opts = {
            "basis": actual_basis,
            "reference": ref_type,
            "e_convergence": 1e-10,
            "d_convergence": 1e-8,
            "maxiter": 200,
            "freeze_core": False,
        }
        if mode_lower == "exact":
            base_opts["scf_type"] = "pk"
            base_opts["mp2_type"] = "conv"
        elif mode_lower == "cholesky":
            base_opts["scf_type"] = "cd"
            base_opts["cholesky_tolerance"] = 1e-9
            base_opts["mp2_type"] = "df"
            base_opts["df_basis_mp2"] = mp2_aux
            base_opts["df_basis_cc"] = mp2_aux
        else:  # df
            base_opts["scf_type"] = "df"
            base_opts["mp2_type"] = "df"
            base_opts["df_basis_scf"] = scf_aux
            base_opts["df_basis_mp2"] = mp2_aux
            base_opts["df_basis_cc"] = mp2_aux

        # ... (Bagian geometri dan base_opts tetap sama) ...
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.clean_variables()
        psi4.set_options(base_opts)

        scf_converged = True
        t0_hf = time.perf_counter()
        try:
            e_hf, scf_wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
            t_hf_baseline = time.perf_counter() - t0_hf
        except psi4.SCFConvergenceError:
            psi4.core.clean()
            psi4.set_options({"maxiter": 400})
            t0_hf = time.perf_counter()
            e_hf, scf_wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
            t_hf_baseline = time.perf_counter() - t0_hf

        assert_finite("HF", e_hf)
        S_mat = np.array(scf_wfn.S().to_array(dense=True), dtype=np.float64, copy=True)
        s2_hf = compute_s2_from_wfn(scf_wfn, S_mat, na, nb, is_closed)

        psi_res = {
            "basis": actual_basis,
            "converged": bool(scf_converged),
            "HF": float(e_hf),
            "S2_HF": float(s2_hf),
            "TIME": {"HF": float(t_hf_baseline), "MP2": 0.0, "MP3": 0.0, "OMP2": 0.0, "OMP3": 0.0},
        }

        if n_elec > 1:
            post_hf_opts = dict(base_opts)
            post_hf_opts.update({
                "e_convergence": 1e-9,
                "r_convergence": 1e-5,
                "rms_mograd_convergence": 1e-5,
                "max_mograd_convergence": 1e-4,
                "mo_maxiter": 50,
            })

            def run_isolated_post_hf(method_label: str):
                # Putus dependensi I/O: Eksekusi pipeline penuh secara terisolasi tanpa memori wfn iterasi sebelumnya
                psi4.core.clean()
                psi4.core.clean_variables()
                psi4.core.clean_options()
                
                opts = dict(post_hf_opts)
                psi4.set_options(opts)

                t0 = time.perf_counter()
                # Eksekusi full graph dari objek molekul, bukan referensi scf_wfn
                e_val, wfn_out = psi4.energy(method_label, molecule=mol, return_wfn=True)
                t_total = time.perf_counter() - t0
                
                assert_finite(method_label.upper(), e_val)
                t_post_hf = max(0.0, t_total - t_hf_baseline)
                return float(e_val), wfn_out, float(t_post_hf)

            e_mp2, mp2_wfn, t_mp2 = run_isolated_post_hf("mp2")
            del mp2_wfn

            e_mp3, mp3_wfn, t_mp3 = run_isolated_post_hf("mp3")
            del mp3_wfn

            e_omp2, omp2_wfn, t_omp2 = run_isolated_post_hf("omp2")
            s2_omp2 = compute_s2_from_wfn(omp2_wfn, S_mat, na, nb, is_closed)
            del omp2_wfn

            e_omp3, omp3_wfn, t_omp3 = run_isolated_post_hf("omp3")
            s2_omp3 = compute_s2_from_wfn(omp3_wfn, S_mat, na, nb, is_closed)
            del omp3_wfn

            psi_res.update({
                "MP2": e_mp2,
                "MP3": e_mp3,
                "OMP2": e_omp2,
                "OMP3": e_omp3,
                "S2_OMP2": s2_omp2,
                "S2_OMP3": s2_omp3,
            })
            psi_res["TIME"].update({
                "MP2": t_mp2,
                "MP3": t_mp3,
                "OMP2": t_omp2,
                "OMP3": t_omp3,
            })
        else:
            for m in ("MP2", "MP3", "OMP2", "OMP3"):
                psi_res[m] = float(e_hf)
            psi_res["S2_OMP2"] = psi_res["S2_OMP3"] = float(s2_hf)

        del scf_wfn, mol
        psi4.core.clean()
        gc.collect()
        return psi_res
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

def run_gdb_backtrace(sys_name: str, actual_basis: str, eri_mode: str):
    gdb_bin = shutil.which("gdb")
    if not gdb_bin:
        print(f"  [GDB] Binary 'gdb' tidak ditemukan di PATH; lewati pengambilan stacktrace.", flush=True)
        return

    print("\n" + "!" * 110, flush=True)
    print(f"  [GDB CRASH DIAGNOSIS] Menjalankan gdb --args + bt full untuk: {sys_name} ({actual_basis} | {eri_mode})", flush=True)
    print("!" * 110, flush=True)

    cmd = [
        gdb_bin, "-batch",
        "-ex", "set pagination off",
        "-ex", "handle SIGPIPE nostop noprint pass",
        "-ex", "run",
        "-ex", "bt full",
        "-ex", "info registers",
        "-ex", "thread apply all bt full",
        "-ex", "quit",
        "--args", sys.executable, "-u", os.path.abspath(__file__),
        "--single-system", sys_name,
        "--single-basis", actual_basis,
        "--single-mode", eri_mode,
    ]
    subprocess.run(cmd, check=False)
    print("!" * 110 + "\n", flush=True)

def run_isolated_worker(sys_name: str, actual_basis: str, eri_mode: str, tmp_json: str, debug_nan: bool = False) -> dict:
    if os.path.exists(tmp_json):
        os.remove(tmp_json)

    cmd = [
        sys.executable, "-u", os.path.abspath(__file__),
        "--single-system", sys_name,
        "--single-basis", actual_basis,
        "--single-mode", eri_mode,
        "--single-out", tmp_json,
    ]
    if debug_nan:
        cmd.append("--debug-nan")

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    if proc.returncode == 0 and os.path.exists(tmp_json):
        with open(tmp_json, "r", encoding="utf-8") as f:
            return json.load(f)

    if proc.returncode < 0:
        sig_num = -proc.returncode
        try:
            sig_name = signal.Signals(sig_num).name
        except Exception:
            sig_name = f"SIG({sig_num})"
        err_msg = f"Terminated by C++ native signal {sig_name} (exit {proc.returncode})"
        run_gdb_backtrace(sys_name, actual_basis, eri_mode)
    else:
        tail_out = (proc.stdout or "").strip().splitlines()[-3:]
        err_msg = f"Worker exit {proc.returncode}: {' | '.join(tail_out)}"

    return {"error": err_msg, "basis": actual_basis}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified PSI4 Universal Validation Runner")
    parser.add_argument("--tier", choices=["standard", "full"], default="standard")
    parser.add_argument("--basis", default="cc-pVDZ")
    parser.add_argument("--out", default="psi4_universal_raw.json")
    parser.add_argument("--threads", "-t", type=int, default=N_THREADS, help="Jumlah thread OpenMP/MKL")
    parser.add_argument("--debug-nan", action="store_true", default=False, help="Aktifkan floating point trap untuk deteksi NaN/Inf")
    parser.add_argument("--single-system", default=None)
    parser.add_argument("--single-basis", default=None)
    parser.add_argument("--single-mode", default=None)
    parser.add_argument("--single-out", default=None)

    args, unknown_args = parser.parse_known_args()

    if args.debug_nan:
        np.seterr(all="raise")

    if args.threads:
        N_THREADS = args.threads
        os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
        os.environ["MKL_NUM_THREADS"] = str(N_THREADS)

    if args.single_system:
        spec = SYSTEMS[args.single_system]
        res = run_psi4_system(spec, args.single_basis, args.single_mode)
        if args.single_out:
            with open(args.single_out, "w", encoding="utf-8") as f:
                json.dump(res, f, allow_nan=False)
        sys.exit(0)

    t_start = time.perf_counter()
    print("=" * 110, flush=True)
    print(f"  PSI4 UNIVERSAL VALIDATION RUNNER (TIER: {args.tier.upper()}, BASIS: {args.basis}, THREADS: {N_THREADS})", flush=True)
    print("=" * 110, flush=True)

    selected_systems = {k: v for k, v in SYSTEMS.items() if args.tier == "full" or v[5] == "standard"}
    results_db = {}
    tmp_worker_json = f"{args.out}.worker_tmp.json"

    def save_checkpoint():
        wall_time = time.perf_counter() - t_start
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "engine": "psi4",
                "tier": args.tier,
                "basis": args.basis,
                "threads": N_THREADS,
                "wall_time_s": wall_time,
                "results": results_db,
            }, f, indent=2, allow_nan=False)

    for mode in ["EXACT", "CHOLESKY", "DF"]:
        print(f"\n>>> [PSI4] Mode ERI: {mode} (Basis Utama: {args.basis}) <<<", flush=True)
        for sys_name, spec in selected_systems.items():
            actual_basis = select_basis_for_system(spec, args.basis)
            key = f"{args.basis}|{mode}|{sys_name}"
            res = run_isolated_worker(sys_name, actual_basis, mode, tmp_worker_json)
            results_db[key] = res
            if "error" not in res:
                t_sum = sum(res["TIME"].values())
                print(
                    f"  [PSI4]  {sys_name:<11} ({actual_basis:<9} | {mode:<8}) -> "
                    f"HF: {res['HF']:12.6f} | MP2: {res['MP2']:12.6f} | "
                    f"MP3: {res['MP3']:12.6f} | OMP2: {res['OMP2']:12.6f} | "
                    f"OMP3: {res['OMP3']:14.8f} Ha ({t_sum:.3f} s)",
                    flush=True,
                )
            else:
                print(f"  [FAIL]  {sys_name:<11} ({actual_basis:<9} | {mode:<8}) -> Error: {res['error']}", flush=True)
            save_checkpoint()

    if args.basis.lower() != "cc-pvdz":
        print("\n>>> [PSI4] Cabang SAD Double-Zeta (Basis: cc-pVDZ) <<<", flush=True)
        for mode in ["EXACT", "DF"]:
            for sys_name in ["Ne", "H2O", "OH", "NH3"]:
                spec = SYSTEMS[sys_name]
                actual_basis = select_basis_for_system(spec, "cc-pVDZ")
                key = f"cc-pVDZ|{mode}|{sys_name}"
                res = run_isolated_worker(sys_name, actual_basis, mode, tmp_worker_json)
                results_db[key] = res
                if "error" not in res:
                    t_sum = sum(res["TIME"].values())
                    print(
                        f"  [PSI4]  {sys_name:<11} ({actual_basis:<9} | {mode:<8}) -> "
                        f"HF: {res['HF']:12.6f} | MP2: {res['MP2']:12.6f} | "
                        f"MP3: {res['MP3']:12.6f} | OMP2: {res['OMP2']:12.6f} | "
                        f"OMP3: {res['OMP3']:14.8f} Ha ({t_sum:.3f} s)",
                        flush=True,
                    )
                else:
                    print(f"  [FAIL]  {sys_name:<11} ({actual_basis:<9} | {mode:<8}) -> Error: {res['error']}", flush=True)
                save_checkpoint()

    if os.path.exists(tmp_worker_json):
        os.remove(tmp_worker_json)

    wall_time = time.perf_counter() - t_start
    save_checkpoint()
    print(f"\n[SELESAI] Hasil PSI4 Universal disimpan ke: {args.out} ({wall_time:.2f} s)", flush=True)