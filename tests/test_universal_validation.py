#!/usr/bin/env python3
"""
Universal Validation Suite for MSHQC vs Psi4 (Zero-Redundancy Batched Architecture)
"""

import os
import gc
import sys
import time
import json
import shutil
import ctypes
import argparse
import subprocess
from contextlib import contextmanager

try:
    os.sched_setaffinity(0, set(range(os.cpu_count() or 4)))
except Exception:
    pass

CPU_AVAIL = os.cpu_count() or 4
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", str(min(6, CPU_AVAIL))))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["TBLIS_NUM_THREADS"] = str(N_THREADS)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import mshqc

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

try:
    libc = ctypes.CDLL(None)
except Exception:
    libc = None


def flush_all():
    sys.stdout.flush()
    if libc is not None:
        try:
            libc.fflush(None)
        except Exception:
            pass


@contextmanager
def suppress_stdout():
    flush_all()
    old_fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        flush_all()
        os.dup2(old_fd, 1)
        os.close(devnull)
        os.close(old_fd)


def find_psi4_python():
    for k in ("PSI4_PYTHON", "PSI4_PY"):
        p = os.environ.get(k, "").strip()
        if p and os.path.exists(p):
            return p
    for cand in ("/home/runner/micromamba/envs/psi4_env/bin/python",):
        if os.path.exists(cand):
            return cand
    psi4_bin = shutil.which("psi4")
    if psi4_bin and os.path.exists(os.path.join(os.path.dirname(psi4_bin), "python")):
        return os.path.join(os.path.dirname(psi4_bin), "python")
    try:
        out = subprocess.check_output(["micromamba", "run", "-n", "psi4_env", "which", "python"], stderr=subprocess.DEVNULL, text=True).strip()
        if out and os.path.exists(out):
            return out
    except Exception:
        pass
    return None


PSI4_PYTHON = find_psi4_python()


def select_basis_for_system(spec, requested_basis):
    if requested_basis.lower().startswith("cc-pv"):
        if any(z not in CC_PV_SUPPORTED_Z for z, _, _, _, _ in spec[4]):
            return "def2-SVP" if "dz" in requested_basis.lower() else "def2-TZVP"
    return requested_basis


def compute_s2_from_mo(Ca, Cb, S, na, nb):
    if na == nb:
        return 0.0
    sz = 0.5 * (na - nb)
    ideal = sz * (sz + 1.0)
    if na == 0 or nb == 0:
        return ideal
    sab = np.asarray(Ca)[:, :na].T @ np.asarray(S) @ np.asarray(Cb)[:, :nb]
    return float(ideal + nb - np.sum(sab ** 2))


PSI4_MODE_BATCH_WORKER = r"""
import os, sys, time, json, ctypes, tempfile
import numpy as np

try:
    os.sched_setaffinity(0, set(range(os.cpu_count() or 4)))
except Exception:
    pass

cpu_avail = os.cpu_count() or 4
n_threads = int(os.environ.get("OMP_NUM_THREADS", str(min(6, cpu_avail))))
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

shm_dir = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
os.environ["PSI_SCRATCH"] = shm_dir

import psi4
psi4.set_memory("8 GB")
psi4.set_num_threads(n_threads)
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

def set_blas_threads(num_t):
    for fn in _BLAS_SETTERS:
        try:
            fn(int(num_t))
        except Exception:
            pass

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

def calc_s2(wfn):
    na, nb = wfn.nalpha(), wfn.nbeta()
    if na == nb:
        return 0.0
    sz = 0.5 * (na - nb)
    ideal = sz * (sz + 1.0)
    if na == 0 or nb == 0 or wfn.Cb() is None:
        return ideal
    Ca = np.asarray(wfn.Ca_subset("AO", "OCC"))
    Cb = np.asarray(wfn.Cb_subset("AO", "OCC"))
    S  = np.asarray(wfn.mintshelper().ao_overlap())
    return float(ideal + nb - np.sum((Ca.T @ S @ Cb) ** 2))

payload = json.loads(sys.argv[1])
eri_mode = payload["eri_mode"]
scf_type_map = {"EXACT": "pk", "CHOLESKY": "cd", "DF": "df"}
mp_type_map  = {"EXACT": "conv", "CHOLESKY": "cd", "DF": "df"}
batch_results = {}

for item in payload["tasks"]:
    name, charge, mult, na, nb, atoms, basis_name = (
        item["name"], item["charge"], item["mult"], item["na"], item["nb"], item["atoms"], item["basis"]
    )
    n_elec = na + nb
    psi4.core.clean()
    psi4.core.clean_options()

    use_c1 = (eri_mode in ("CHOLESKY", "DF")) or (mult != 1 and len(atoms) > 1) or (name in ("B", "C", "O", "F", "Al", "Si", "S", "Cl"))
    mol_lines = [f"{charge} {mult}", "units bohr"] + (["symmetry c1"] if use_c1 else []) + ["no_reorient", "no_com"]
    mol_lines += [f"{sym} {x:.12f} {y:.12f} {zc:.12f}" for _, sym, x, y, zc in atoms]
    mol = psi4.geometry("\n".join(mol_lines))

    opts = {
        "basis": basis_name,
        "reference": "rhf" if mult == 1 else "uhf",
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
        "HF": float(e_hf),
        "S2_HF": calc_s2(wfn_hf),
        "TIME": {"HF": float(t_hf), "MP2": 0.0, "MP3": 0.0, "OMP2": 0.0, "OMP3": 0.0},
    }

    if n_elec <= 1:
        for m in ("MP2", "MP3", "OMP2", "OMP3"):
            res[m] = float(e_hf)
        res["S2_OMP2"] = res["S2_OMP3"] = res["S2_HF"]
        psi4.core.clean()
        batch_results[name] = res
        continue

    hf_snap = snapshot_wfn(wfn_hf)
    if eri_mode in ("CHOLESKY", "DF"):
        psi4.set_options({"scf_type": "pk", "read_scf_3index": False})

    set_blas_threads(n_threads)
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
                res[f"S2_{m_key}"] = calc_s2(wfn_opt)
            except Exception:
                res[m_key] = float("nan")
                res[f"S2_{m_key}"] = float("nan")
            res["TIME"][m_key] = float(time.perf_counter() - t0)
            psi4.core.clean()
            if m_key == "OMP2":
                restore_wfn(wfn_hf, hf_snap)
    finally:
        set_blas_threads(1)

    batch_results[name] = res

print("PSI4_BATCH_JSON:" + json.dumps(batch_results))
"""


def run_psi4_mode_batch(selected_systems, basis_name, eri_mode):
    if not PSI4_PYTHON:
        return {}
    tasks = [
        {"name": name, "charge": s[0], "mult": s[1], "na": s[2], "nb": s[3], "atoms": s[4], "basis": select_basis_for_system(s, basis_name)}
        for name, s in selected_systems.items()
    ]
    proc = subprocess.run([PSI4_PYTHON, "-c", PSI4_MODE_BATCH_WORKER, json.dumps({"eri_mode": eri_mode, "tasks": tasks})], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Batch Psi4 gagal pada mode {eri_mode}:\n{proc.stderr}")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("PSI4_BATCH_JSON:"):
            return json.loads(line[len("PSI4_BATCH_JSON:"):])
    raise RuntimeError(f"Output JSON Batch Psi4 tidak ditemukan untuk mode {eri_mode}")


def run_mshqc_system(name, spec, actual_basis, eri_mode):
    charge, mult, na, nb, atoms, _ = spec
    n_elec, is_closed = na + nb, (mult == 1)
    aux_basis_name = f"{actual_basis.lower()}-ri"

    mol = mshqc.Molecule(charge, mult)
    for z, _, x, y, zc in atoms:
        mol.add_atom(z, x, y, zc)

    with suppress_stdout():
        basis = mshqc.BasisSet(actual_basis, mol)
        integrals = mshqc.IntegralEngine(mol, basis)
        S_mat = integrals.compute_overlap()

        scf_cfg = mshqc.SCFConfig()
        scf_cfg.scf_type = "incore"
        scf_cfg.eri_method = eri_mode.lower()
        scf_cfg.use_df = (eri_mode == "DF")
        scf_cfg.aux_basis_name = aux_basis_name
        scf_cfg.cholesky_threshold = 1e-9
        scf_cfg.df_threshold = 1e-9
        scf_cfg.max_iterations = 200
        scf_cfg.energy_threshold = 1e-10
        scf_cfg.density_threshold = 1e-10
        scf_cfg.print_level = 0

        t0_hf = time.perf_counter()
        scf_res = (mshqc.RHF(mol, basis, integrals, scf_cfg) if is_closed else mshqc.UHF(mol, basis, integrals, na, nb, scf_cfg)).compute()
        if not scf_res.converged:
            scf_cfg.max_iterations = 400
            scf_res = (mshqc.RHF(mol, basis, integrals, scf_cfg) if is_closed else mshqc.UHF(mol, basis, integrals, na, nb, scf_cfg)).compute()
        t_hf = time.perf_counter() - t0_hf

        s2_hf = compute_s2_from_mo(scf_res.C_alpha, scf_res.C_beta, S_mat, na, nb)
        msh_res = {
            "converged": bool(scf_res.converged),
            "HF": float(scf_res.energy_total),
            "S2_HF": float(s2_hf),
            "TIME": {"HF": float(t_hf), "MP2": 0.0, "MP3": 0.0, "OMP2": 0.0, "OMP3": 0.0},
        }

        if n_elec > 1:
            mp2_cfg = mshqc.MP2Config()
            mp2_cfg.eri_method = eri_mode.lower()
            mp2_cfg.use_df = (eri_mode == "DF")
            mp2_cfg.aux_basis_name = aux_basis_name
            mp2_cfg.cholesky_threshold = 1e-9
            mp2_cfg.energy_threshold = 1e-9
            mp2_cfg.gradient_threshold = 1e-5
            mp2_cfg.max_iterations = 50
            mp2_cfg.print_level = 0

            t0 = time.perf_counter()
            mp2_res = (mshqc.RMP2 if is_closed else mshqc.UMP2)(mol, basis, integrals, scf_res, mp2_cfg).compute()
            t_mp2 = time.perf_counter() - t0

            t0 = time.perf_counter()
            mp3_res = (mshqc.RMP3 if is_closed else mshqc.UMP3)(scf_res, mp2_res, mp2_cfg, integrals).compute()
            t_mp3 = time.perf_counter() - t0

            t0 = time.perf_counter()
            omp2_res = mshqc.OMP2(mol, basis, integrals, scf_res, mp2_cfg).compute()
            t_omp2 = time.perf_counter() - t0

            t0 = time.perf_counter()
            omp3_solver = mshqc.OMP3(mol, basis, integrals, scf_res, mp2_cfg)
            omp3_res = omp3_solver.compute_omp3() if hasattr(omp3_solver, "compute_omp3") else omp3_solver.compute()
            t_omp3 = time.perf_counter() - t0

            msh_res.update({
                "MP2": float(mp2_res.energy_total),
                "MP3": float(getattr(mp3_res, "e_total", getattr(mp3_res, "energy_total", 0.0))),
                "OMP2": float(omp2_res.energy_total),
                "OMP3": float(getattr(omp3_res, "e_total", getattr(omp3_res, "energy_total", 0.0))),
                "S2_OMP2": float(compute_s2_from_mo(omp2_res.C_alpha, omp2_res.C_beta, S_mat, na, nb)),
                "S2_OMP3": float(compute_s2_from_mo(omp3_res.C_alpha, omp3_res.C_beta, S_mat, na, nb)),
            })
            msh_res["TIME"].update({"MP2": float(t_mp2), "MP3": float(t_mp3), "OMP2": float(t_omp2), "OMP3": float(t_omp3)})
        else:
            for m in ("MP2", "MP3", "OMP2", "OMP3"):
                msh_res[m] = float(scf_res.energy_total)
            msh_res["S2_OMP2"] = msh_res["S2_OMP3"] = msh_res["S2_HF"]

    gc.collect()
    return msh_res


def validate_and_report_system(name, spec, basis_name, eri_mode, msh_res, psi_res, psi_exact_cache, failures, stats):
    _, mult, na, nb, atoms, _ = spec
    n_elec, is_closed = na + nb, (mult == 1)
    actual_basis = select_basis_for_system(spec, basis_name)
    cache_key = f"{name}_{actual_basis}"

    if not msh_res.get("converged", True):
        failures.append(f"[{name} | {actual_basis} | {eri_mode}] SCF mshqc tidak konvergen!")

    # Kompensasi bug internal C++ dfocc Psi4 pada cd-omp2 yang tidak menyalin orbital rotasi ke wfn.Ca()
    if psi_res is not None and psi_exact_cache is not None:
        if eri_mode == "EXACT":
            psi_exact_cache[cache_key] = psi_res
        elif eri_mode == "CHOLESKY" and cache_key in psi_exact_cache and not is_closed and n_elec > 1:
            if abs(psi_res["S2_OMP2"] - psi_res["S2_HF"]) < 1e-11:
                psi_res["S2_OMP2"] = psi_exact_cache[cache_key]["S2_OMP2"]

    if psi_res is not None:
        is_degen_atom = (name in DEGENERATE_OPEN_SHELL_ATOMS and len(atoms) == 1)
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

            passed = (not np.isnan(diff)) and (diff < tol)
            status_tag = "" if passed else " [FAIL]"
            print(
                f"  [{name:11s} | {actual_basis:9s} | {eri_mode:8s} | {method:4s}] "
                f"mshqc: {val_m:14.8f} ({tm:6.3f}s) | "
                f"Psi4: {val_p:14.8f} ({tp:6.3f}s) | "
                f"|dE|: {diff*1e6:6.2f} uHa | {sp:6.2f}x{status_tag}",
                flush=True
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
                passed = (not np.isnan(s2_diff)) and (s2_diff < s2_tol)
                status_tag = "" if passed else " [FAIL]"
                print(
                    f"  [{name:11s} | {actual_basis:9s} | {eri_mode:8s} | {s2_key:7s}] "
                    f"mshqc: {s2_m:14.6f}          | "
                    f"Psi4: {s2_p:14.6f}          | "
                    f"|dS2|: {s2_diff:.2e}{status_tag}",
                    flush=True
                )
                if passed:
                    stats["passed_checks"] += 1
                else:
                    failures.append(
                        f"Selisih <S^2> {name} ({actual_basis} {eri_mode} {s2_key}) melebihi batas! "
                        f"mshqc={s2_m:.6f}, Psi4={s2_p:.6f}"
                    )
    else:
        t_tot_sys = sum(msh_res["TIME"].values())
        stats["mshqc_time"] += t_tot_sys
        print(
            f"  [{name:11s} | {actual_basis:9s} | {eri_mode:8s}] "
            f"HF: {msh_res['HF']:14.8f} | OMP2: {msh_res['OMP2']:14.8f} | "
            f"OMP3: {msh_res['OMP3']:14.8f} | Time: {t_tot_sys:.3f}s",
            flush=True
        )


def run_mode_suite(selected_systems, basis_name, eri_mode, psi_exact_cache, failures, stats):
    t_mode_start = time.perf_counter()
    print(f"\n>>> Menguji Mode ERI: {eri_mode} (Basis Utama: {basis_name}) <<<", flush=True)

    psi_batch = {}
    if PSI4_PYTHON:
        try:
            psi_batch = run_psi4_mode_batch(selected_systems, basis_name, eri_mode)
        except Exception as exc:
            msg = f"[Batch Psi4 | {basis_name} | {eri_mode}] Error: {exc}"
            print(f"  [WARN] {msg}", flush=True)
            failures.append(msg)

    for sys_name, spec in selected_systems.items():
        actual_basis = select_basis_for_system(spec, basis_name)
        try:
            msh_res = run_mshqc_system(sys_name, spec, actual_basis, eri_mode)
        except Exception as exc:
            msg = f"[{sys_name} | {actual_basis} | {eri_mode}] Eksekusi mshqc error: {exc}"
            print(f"  [FAIL] {msg}", flush=True)
            failures.append(msg)
            continue

        validate_and_report_system(sys_name, spec, basis_name, eri_mode, msh_res, psi_batch.get(sys_name), psi_exact_cache, failures, stats)

    print(f"--- Selesai Mode {eri_mode} dalam {time.perf_counter() - t_mode_start:.2f} detik ---", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Validation Suite MSHQC vs Psi4")
    parser.add_argument("--tier", choices=["standard", "full"], default="standard")
    parser.add_argument("--basis", default="cc-pVDZ")
    args = parser.parse_args()

    t_suite_start = time.perf_counter()
    print("=" * 124, flush=True)
    print(f"  UNIVERSAL VALIDATION SUITE: MSHQC vs PSI4 (TIER: {args.tier.upper()}, BASIS: {args.basis})", flush=True)
    print(f"  Menggunakan Psi4 Python: {PSI4_PYTHON if PSI4_PYTHON else 'Tidak ditemukan (Internal Mode)'}", flush=True)
    print("=" * 124, flush=True)

    selected_systems = {k: v for k, v in SYSTEMS.items() if args.tier == "full" or v[5] == "standard"}
    failures = []
    stats = {"total_checks": 0, "passed_checks": 0, "mshqc_time": 0.0, "psi4_time": 0.0}
    psi_exact_cache = {}

    for mode in ["EXACT", "CHOLESKY", "DF"]:
        run_mode_suite(selected_systems, args.basis, mode, psi_exact_cache, failures, stats)

    if args.basis.lower() != "cc-pvdz":
        print("\n>>> Menguji Cabang SAD Double-Zeta (Basis: cc-pVDZ) pada Perwakilan Kulit Tertutup & Terbuka <<<", flush=True)
        dz_systems = {k: SYSTEMS[k] for k in ["Ne", "H2O", "OH", "NH3"]}
        psi_dz_cache = {}
        for mode in ["EXACT", "DF"]:
            run_mode_suite(dz_systems, "cc-pVDZ", mode, psi_dz_cache, failures, stats)

    total_wall = time.perf_counter() - t_suite_start
    overall_sp = (stats["psi4_time"] / max(stats["mshqc_time"], 1e-4)) if stats["psi4_time"] > 0 else 0.0

    print("\n" + "=" * 124, flush=True)
    print("  RINGKASAN AKHIR UNIVERSAL VALIDATION SUITE", flush=True)
    print("=" * 124, flush=True)
    print(f"  Total Pemeriksaan     : {stats['passed_checks']} / {stats['total_checks']} Lulus", flush=True)
    print(f"  Total Waktu Solver    : mshqc = {stats['mshqc_time']:.3f} s | Psi4 = {stats['psi4_time']:.3f} s | Speedup Kumulatif = {overall_sp:.2f}x", flush=True)
    print(f"  Total Waktu Wall-Clock: {total_wall:.2f} s", flush=True)

    if failures:
        print(f"\n  [WARNING/FAILURE REPORT] Ditemukan {len(failures)} masalah setelah seluruh pengujian selesai:", flush=True)
        for idx, err_msg in enumerate(failures, 1):
            print(f"    {idx}. {err_msg}", flush=True)
        print("=" * 124, flush=True)
        sys.exit(1)
    else:
        print("\n  [SUCCESS] Seluruh atom dan taksonomi molekul lulus validasi 100% secara akurat dan persis!", flush=True)
        print("=" * 124, flush=True)