#!/usr/bin/env python3
"""
Runner Mandiri Psi4 untuk Universal Validation Suite (Max Performance & RHF Cb Guard)
Output: psi4_universal_raw.json
"""

import os
import sys
import time
import json
import ctypes
import tempfile
import argparse

try:
    os.sched_setaffinity(0, set(range(os.cpu_count() or 4)))
except Exception:
    pass

CPU_AVAIL = os.cpu_count() or 4
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", str(min(6, CPU_AVAIL))))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

SHM_DIR = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
os.environ["PSI_SCRATCH"] = SHM_DIR

import numpy as np
import psi4

psi4.set_memory("8 GB")
psi4.set_num_threads(N_THREADS)
psi4.core.be_quiet()


def _load_blas_setters():
    setters = []
    for lib_name in ("libmkl_rt.so", "libmkl_rt.so.2", "libopenblas.so", "libopenblas.so.0", None):
        try:
            lib = ctypes.CDLL(lib_name)
        except Exception:
            continue
        for fn_name in ("MKL_Set_Num_Threads", "openblas_set_num_threads"):
            if hasattr(lib, fn_name):
                fn = getattr(lib, fn_name)
                fn.argtypes = [ctypes.c_int]
                fn.restype = None
                setters.append(fn)
    return setters


_BLAS_SETTERS = _load_blas_setters()


def set_blas_threads(num_t: int):
    for fn in _BLAS_SETTERS:
        try:
            fn(int(num_t))
        except Exception:
            pass


ANG_TO_BOHR = 1.8897261246257702
CC_PV_SUPPORTED_Z = set(range(1, 19)) | {20} | set(range(31, 37))
DEGENERATE_OPEN_SHELL_ATOMS = {"B", "C", "O", "F", "Al", "Si", "S", "Cl"}

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


def select_basis_for_system(spec, requested_basis):
    if requested_basis.lower().startswith("cc-pv"):
        if any(z not in CC_PV_SUPPORTED_Z for z, _, _, _, _ in spec[4]):
            return "def2-SVP" if "dz" in requested_basis.lower() else "def2-TZVP"
    return requested_basis


def snapshot_wfn(wfn):
    snap = {"e": float(wfn.energy()), "Ca": wfn.Ca().clone(), "Da": wfn.Da().clone(), "Fa": wfn.Fa().clone(), "ea": wfn.epsilon_a().clone()}
    if wfn.Cb() is not None:
        snap.update({"Cb": wfn.Cb().clone(), "Db": wfn.Db().clone(), "Fb": wfn.Fb().clone(), "eb": wfn.epsilon_b().clone()})
    return snap


def restore_wfn(wfn, snap):
    wfn.set_energy(snap["e"])
    for k, getter in (("Ca", wfn.Ca), ("Da", wfn.Da), ("Fa", wfn.Fa), ("ea", wfn.epsilon_a)):
        getter().copy(snap[k])
    if "Cb" in snap and wfn.Cb() is not None:
        for k, getter in (("Cb", wfn.Cb), ("Db", wfn.Db), ("Fb", wfn.Fb), ("eb", wfn.epsilon_b)):
            getter().copy(snap[k])


def calc_s2(wfn, is_closed: bool):
    if is_closed:
        return 0.0
    na, nb = wfn.nalpha(), wfn.nbeta()
    if na == nb:
        return 0.0
    sz = 0.5 * (na - nb)
    ideal = sz * (sz + 1.0)
    if na == 0 or nb == 0 or wfn.Cb() is None:
        return float(ideal)
    try:
        Ca = np.asarray(wfn.Ca_subset("AO", "OCC"))
        Cb = np.asarray(wfn.Cb_subset("AO", "OCC"))
        S  = np.asarray(wfn.mintshelper().ao_overlap())
        return float(ideal + nb - np.sum((Ca.T @ S @ Cb) ** 2))
    except Exception:
        return float(ideal)


def run_psi4_system(name, spec, basis_name, eri_mode):
    charge, mult, na, nb, atoms, _ = spec
    n_elec = na + nb
    is_closed = (mult == 1)

    scf_type_map = {"EXACT": "pk", "CHOLESKY": "cd", "DF": "df"}
    mp_type_map  = {"EXACT": "conv", "CHOLESKY": "cd", "DF": "df"}

    psi4.core.clean()
    psi4.core.clean_options()

    use_c1 = (eri_mode in ("CHOLESKY", "DF")) or (mult != 1 and len(atoms) > 1) or (name in DEGENERATE_OPEN_SHELL_ATOMS)
    mol_lines = [f"{charge} {mult}", "units bohr"] + (["symmetry c1"] if use_c1 else []) + ["no_reorient", "no_com"]
    mol_lines += [f"{sym} {x:.12f} {y:.12f} {zc:.12f}" for _, sym, x, y, zc in atoms]
    mol = psi4.geometry("\n".join(mol_lines))

    opts = {
        "basis": basis_name,
        "reference": "rhf" if is_closed else "uhf",
        "scf_type": scf_type_map[eri_mode],
        "mp2_type": mp_type_map[eri_mode],
        "mp_type": mp_type_map[eri_mode],
        "cc_type": mp_type_map[eri_mode],
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
        "r_convergence": 1e-6,
        "rms_mograd_convergence": 1e-5,
        "max_mograd_convergence": 1e-4,
        "freeze_core": False,
        "maxiter": 200,
        "mo_maxiter": 50,
        "cachelevel": 2,
        "read_scf_3index": False,
    }
    if eri_mode == "CHOLESKY":
        opts["cholesky_tolerance"] = 1e-9
    elif eri_mode == "DF":
        aux_b = basis_name.lower() + "-ri"
        opts.update({"df_basis_scf": aux_b, "df_basis_mp2": aux_b, "df_basis_cc": aux_b})

    psi4.set_options(opts)

    set_blas_threads(1)
    t0 = time.perf_counter()
    e_hf, wfn_hf = psi4.energy("scf", molecule=mol, return_wfn=True)
    t_hf = time.perf_counter() - t0

    res = {
        "basis": basis_name,
        "HF": float(e_hf),
        "S2_HF": calc_s2(wfn_hf, is_closed),
        "TIME": {"HF": float(t_hf), "MP2": 0.0, "MP3": 0.0, "OMP2": 0.0, "OMP3": 0.0},
    }

    if n_elec <= 1:
        for m in ("MP2", "MP3", "OMP2", "OMP3"):
            res[m] = float(e_hf)
        res["S2_OMP2"] = res["S2_OMP3"] = res["S2_HF"]
        psi4.core.clean()
        return res

    hf_snap = snapshot_wfn(wfn_hf)
    if eri_mode in ("CHOLESKY", "DF"):
        psi4.set_options({"scf_type": "pk", "read_scf_3index": False})

    set_blas_threads(N_THREADS)
    try:
        for m_key, m_call in (("MP2", "mp2"), ("MP3", "mp3")):
            t0 = time.perf_counter()
            res[m_key] = float(psi4.energy(m_call, ref_wfn=wfn_hf, molecule=mol))
            res["TIME"][m_key] = float(time.perf_counter() - t0)
            psi4.core.clean()
            restore_wfn(wfn_hf, hf_snap)

        for m_key, m_call in (("OMP2", "omp2"), ("OMP3", "omp3")):
            t0 = time.perf_counter()
            try:
                e_val, wfn_opt = psi4.energy(m_call, ref_wfn=wfn_hf, molecule=mol, return_wfn=True)
                res[m_key] = float(e_val)
                res[f"S2_{m_key}"] = calc_s2(wfn_opt, is_closed)
            except Exception:
                res[m_key] = float("nan")
                res[f"S2_{m_key}"] = float("nan")
            res["TIME"][m_key] = float(time.perf_counter() - t0)
            psi4.core.clean()
            if m_key == "OMP2":
                restore_wfn(wfn_hf, hf_snap)
    finally:
        set_blas_threads(1)

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Psi4 Universal Validation Standalone Runner")
    parser.add_argument("--tier", choices=["standard", "full"], default="standard")
    parser.add_argument("--basis", default="cc-pVDZ")
    parser.add_argument("--out", default="psi4_universal_raw.json")
    args = parser.parse_args()

    t_start = time.perf_counter()
    print("=" * 110, flush=True)
    print(f"  PSI4 UNIVERSAL VALIDATION RUNNER (TIER: {args.tier.upper()}, BASIS: {args.basis}, THREADS: {N_THREADS})", flush=True)
    print("=" * 110, flush=True)

    selected_systems = {k: v for k, v in SYSTEMS.items() if args.tier == "full" or v[5] == "standard"}
    results_db = {}

    for mode in ["EXACT", "CHOLESKY", "DF"]:
        print(f"\n>>> [PSI4] Mode ERI: {mode} (Basis Utama: {args.basis}) <<<", flush=True)
        for sys_name, spec in selected_systems.items():
            actual_basis = select_basis_for_system(spec, args.basis)
            key = f"{args.basis}|{mode}|{sys_name}"
            try:
                res = run_psi4_system(sys_name, spec, actual_basis, mode)
                results_db[key] = res
                t_sum = sum(res["TIME"].values())
                print(f"  [PSI4]  {sys_name:<11} ({actual_basis:<9} | {mode:<8}) -> OMP3: {res['OMP3']:14.8f} Ha ({t_sum:.3f} s)", flush=True)
            except Exception as exc:
                results_db[key] = {"error": str(exc), "basis": actual_basis}
                print(f"  [FAIL]  {sys_name:<11} ({actual_basis:<9} | {mode:<8}) -> Error: {exc}", flush=True)

    if args.basis.lower() != "cc-pvdz":
        print("\n>>> [PSI4] Cabang SAD Double-Zeta (Basis: cc-pVDZ) <<<", flush=True)
        for mode in ["EXACT", "DF"]:
            for sys_name in ["Ne", "H2O", "OH", "NH3"]:
                spec = SYSTEMS[sys_name]
                actual_basis = select_basis_for_system(spec, "cc-pVDZ")
                key = f"cc-pVDZ|{mode}|{sys_name}"
                try:
                    results_db[key] = run_psi4_system(sys_name, spec, actual_basis, mode)
                except Exception as exc:
                    results_db[key] = {"error": str(exc), "basis": actual_basis}

    wall_time = time.perf_counter() - t_start
    payload = {
        "engine": "psi4",
        "tier": args.tier,
        "basis": args.basis,
        "threads": N_THREADS,
        "wall_time_s": wall_time,
        "results": results_db,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[SELESAI] Hasil Psi4 Universal disimpan ke: {args.out} ({wall_time:.2f} s)", flush=True)