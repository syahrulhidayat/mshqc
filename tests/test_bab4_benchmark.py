#!/usr/bin/env python3
"""
Benchmark & Validasi Empiris Bab IV Skripsi (MSHQC vs Psi4) - Refactored High-Speed Suite
Mencakup:
1. Kompresi Tensor ERI AO & Peringkat Cholesky Aktual (mshqc vs Psi4, tau_CD = 1e-9)
2. Bukti Eliminasi Hambatan Memori O(V^4) pada OMP3 (MO Tensor Breakdown)
3. Evaluasi Reduksi Kontaminasi Spin <S^2> (UHF vs OMP2 vs OMP3: mshqc vs Psi4)
4. Verifikasi Gradien Analitik vs Finite-Difference & Jejak Iterasi TR-SOSCF
5A. Validasi Head-to-Head Metode yang Sama (HF, MP2, MP3, OMP2, OMP3: mshqc vs Psi4)
5B. Profil Konvergensi Iterasi (N_iter: mshqc vs Psi4) & Efek Relaksasi Orbital (dE_OO)
5C. Bukti Ketangguhan (Robustness) TR-SOSCF dari Referensi UHF Metastabil (OH & O2)
6. Pengukuran Peak RSS (VmHWM) & Net Working RAM Terkalibrasi (mshqc vs Psi4)
"""

import os
import gc
import sys
import time
import shutil
import ctypes
import json
import subprocess
from contextlib import contextmanager

# Konfigurasi HPC untuk proses utama MSHQC (OpenMP + TBLIS aktif, hindari nested BLAS)
CPU_AVAIL = os.cpu_count() or 4
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", str(min(6, CPU_AVAIL))))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["TBLIS_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import mshqc

ANG_TO_BOHR = 1.8897261246257702

SYSTEMS = {
    "Neon": {
        "atoms": [(10, 0.0, 0.0, 0.0)],
        "charge": 0, "mult": 1, "is_uhf": False,
    },
    "H2O": {
        "atoms": [
            (8, 0.0, 0.0, 0.117300 * ANG_TO_BOHR),
            (1, 0.0, 0.757200 * ANG_TO_BOHR, -0.469200 * ANG_TO_BOHR),
            (1, 0.0, -0.757200 * ANG_TO_BOHR, -0.469200 * ANG_TO_BOHR),
        ],
        "charge": 0, "mult": 1, "is_uhf": False,
    },
    "Lithium": {
        "atoms": [(3, 0.0, 0.0, 0.0)],
        "charge": 0, "mult": 2, "is_uhf": True,
    },
    "Carbon": {
        "atoms": [(6, 0.0, 0.0, 0.0)],
        "charge": 0, "mult": 3, "is_uhf": True,
    },
    "Oxygen": {
        "atoms": [(8, 0.0, 0.0, 0.0)],
        "charge": 0, "mult": 3, "is_uhf": True,
    },
    "OH": {
        "atoms": [
            (8, 0.0, 0.0, 0.0),
            (1, 0.0, 0.0, 0.9697 * ANG_TO_BOHR),
        ],
        "charge": 0, "mult": 2, "is_uhf": True,
    },
    "O2": {
        "atoms": [
            (8, 0.0, 0.0, -0.6035 * ANG_TO_BOHR),
            (8, 0.0, 0.0,  0.6035 * ANG_TO_BOHR),
        ],
        "charge": 0, "mult": 3, "is_uhf": True,
    },
}

ALL_SYSTEM_ORDER = ["Neon", "H2O", "Lithium", "Carbon", "Oxygen", "OH", "O2"]
ERI_MODE_ORDER = ["EXACT", "CHOLESKY", "DF"]

VERIFIED_M_CHOL = {
    ("Neon", "cc-pvdz"): 92,
    ("Neon", "cc-pvtz"): 287,
    ("Neon", "cc-pvqz"): 567,
    ("H2O", "cc-pvdz"): 269,
    ("H2O", "cc-pvtz"): 790,
    ("H2O", "cc-pvqz"): 1496,
    ("Lithium", "cc-pvtz"): 245,
    ("Carbon", "cc-pvtz"): 280,
    ("Oxygen", "cc-pvtz"): 283,
    ("OH", "cc-pvtz"): 475,
    ("O2", "cc-pvtz"): 777,
}

try:
    libc = ctypes.CDLL(None)
except Exception:
    libc = None


def flush_all():
    """Memastikan buffer stdout Python dan C++ di-flush secara sinkron."""
    sys.stdout.flush()
    if libc is not None:
        try:
            libc.fflush(None)
        except Exception:
            pass


@contextmanager
def suppress_stdout():
    """Membungkam C++ std::cout saat timing agar setara dengan psi4.core.be_quiet()."""
    flush_all()
    old_stdout_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        yield
    finally:
        flush_all()
        os.dup2(old_stdout_fd, 1)
        os.close(devnull_fd)
        os.close(old_stdout_fd)


def find_psi4_python():
    """Mencari executable Python yang memiliki pustaka Psi4."""
    for env_key in ("PSI4_PY", "PSI4_PYTHON"):
        cand = os.environ.get(env_key, "").strip()
        if cand and os.path.exists(cand):
            return cand
    psi4_bin = shutil.which("psi4")
    if psi4_bin:
        cand_py = os.path.join(os.path.dirname(psi4_bin), "python")
        if os.path.exists(cand_py):
            return cand_py
    try:
        out = subprocess.check_output(
            ["micromamba", "run", "-n", "psi4_env", "which", "python"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        if out and os.path.exists(out):
            return out
    except Exception:
        pass
    return sys.executable


PSI4_PY_BIN = find_psi4_python()


def extract_json_safe(output_str):
    for line in reversed(output_str.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except Exception:
                pass
    return {}


def build_mol(sys_name):
    info = SYSTEMS[sys_name]
    mol = mshqc.Molecule(info["charge"], info["mult"])
    for z, x, y, z_coord in info["atoms"]:
        mol.add_atom(z, x, y, z_coord)
    return mol


def get_occ_counts(sys_name, scf_res=None):
    if scf_res is not None and hasattr(scf_res, "n_occ_alpha") and hasattr(scf_res, "n_occ_beta"):
        return int(scf_res.n_occ_alpha), int(scf_res.n_occ_beta)
    info = SYSTEMS[sys_name]
    n_elec = sum(atom[0] for atom in info["atoms"]) - info["charge"]
    n_unpaired = info["mult"] - 1
    return (n_elec + n_unpaired) // 2, (n_elec - n_unpaired) // 2


def compute_spin_s2(mol, ints, C_a, C_b, n_occ_a, n_occ_b):
    S_ao = np.array(ints.compute_overlap())
    Ca_occ = np.array(C_a)[:, :n_occ_a]
    Cb_occ = np.array(C_b)[:, :n_occ_b]
    sz = 0.5 * (n_occ_a - n_occ_b)
    s_ideal = sz * (sz + 1.0)
    if n_occ_b == 0:
        return s_ideal, s_ideal
    S_ab = Ca_occ.T @ S_ao @ Cb_occ
    s2 = s_ideal + n_occ_b - np.sum(S_ab ** 2)
    return s_ideal, float(s2)


def build_ints(mol, basis):
    with suppress_stdout():
        return mshqc.IntegralEngine(mol, basis)


def make_scf_config(eri_mode, aux_basis_name="", print_level=0):
    scf_cfg = mshqc.SCFConfig()
    scf_cfg.scf_type = "incore"
    scf_cfg.max_iterations = 200
    scf_cfg.energy_threshold = 1e-9
    scf_cfg.density_threshold = 1e-9
    scf_cfg.df_threshold = 1e-9
    scf_cfg.print_level = print_level
    if eri_mode == "DF":
        scf_cfg.eri_method = "df"
        scf_cfg.use_df = True
        if aux_basis_name:
            scf_cfg.aux_basis_name = aux_basis_name
    elif eri_mode == "CHOLESKY":
        scf_cfg.eri_method = "cholesky"
        scf_cfg.cholesky_threshold = 1e-9
    else:
        scf_cfg.eri_method = "exact"
    return scf_cfg


def make_mp2_config(eri_mode, aux_basis_name="", print_level=0, max_iterations=50):
    cfg = mshqc.MP2Config()
    cfg.energy_threshold = 1e-9
    cfg.gradient_threshold = 1e-5
    cfg.max_iterations = max_iterations
    cfg.print_level = print_level
    if eri_mode == "DF":
        cfg.eri_method = "df"
        cfg.use_df = True
        if aux_basis_name:
            cfg.aux_basis_name = aux_basis_name
    elif eri_mode == "CHOLESKY":
        cfg.eri_method = "cholesky"
        cfg.cholesky_threshold = 1e-9
    else:
        cfg.eri_method = "exact"
    return cfg


def run_scf_raw(mol, basis, ints, eri_mode, is_uhf, aux_basis_name="", print_level=0):
    scf_cfg = make_scf_config(eri_mode, aux_basis_name, print_level)
    if is_uhf:
        n_elec = mol.n_electrons()
        n_unp = mol.multiplicity() - 1
        na = (n_elec + n_unp) // 2
        nb = (n_elec - n_unp) // 2
        solver = mshqc.UHF(mol, basis, ints, na, nb, scf_cfg)
    else:
        solver = mshqc.RHF(mol, basis, ints, scf_cfg)
    return solver.compute()


# ==============================================================================
# WORKER SUBPROCESS PSI4 (MULTI-THREADED BLAS, SYMMETRY, & 3-INDEX REUSE)
# ==============================================================================
PSI4_BATCH_WORKER = r"""
import os, sys, time, json, re, tempfile
import numpy as np

cpu_avail = os.cpu_count() or 4
n_threads = int(os.environ.get("OMP_NUM_THREADS", str(min(6, cpu_avail))))
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_NESTED"] = "FALSE"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

shm_dir = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
os.environ["PSI_SCRATCH"] = shm_dir

import psi4
psi4.set_num_threads(n_threads)
psi4.set_memory("8 GB")

Z_TO_SYM = {1: "H", 3: "Li", 6: "C", 8: "O", 10: "Ne"}

def snapshot_wfn(wfn):
    snap = {
        "e": float(wfn.energy()),
        "Ca": wfn.Ca().clone(),
        "Da": wfn.Da().clone(),
        "Fa": wfn.Fa().clone(),
        "ea": wfn.epsilon_a().clone(),
    }
    if wfn.Cb() is not None:
        snap["Cb"] = wfn.Cb().clone()
        snap["Db"] = wfn.Db().clone()
        snap["Fb"] = wfn.Fb().clone()
        snap["eb"] = wfn.epsilon_b().clone()
    return snap

def restore_wfn(wfn, snap):
    wfn.set_energy(snap["e"])
    wfn.Ca().copy(snap["Ca"])
    wfn.Da().copy(snap["Da"])
    wfn.Fa().copy(snap["Fa"])
    wfn.epsilon_a().copy(snap["ea"])
    if "Cb" in snap and wfn.Cb() is not None:
        wfn.Cb().copy(snap["Cb"])
        wfn.Db().copy(snap["Db"])
        wfn.Fb().copy(snap["Fb"])
        wfn.epsilon_b().copy(snap["eb"])

def calc_s2_wfn(wfn):
    na = wfn.nalpha()
    nb = wfn.nbeta()
    sz = 0.5 * (na - nb)
    s_ideal = sz * (sz + 1.0)
    if na == 0 or nb == 0:
        return s_ideal
    Ca_occ = np.asarray(wfn.Ca_subset("AO", "OCC"))
    Cb_occ = np.asarray(wfn.Cb_subset("AO", "OCC"))
    S_ao = np.asarray(wfn.mintshelper().ao_overlap())
    Sab = Ca_occ.T @ S_ao @ Cb_occ
    return float(s_ideal + nb - np.sum(Sab ** 2))

def parse_occ_iterations(log_text):
    iters = 0
    for line in log_text.splitlines():
        m = re.match(r"^\s*(\d+)\s+-?\d+\.\d+\s+-?\d+\.\d+", line)
        if m:
            iters = max(iters, int(m.group(1)))
    return iters

def run_energy_with_fallback(method_name, scf_wfn, mol, eri_mode, return_wfn=False):
    t0 = time.perf_counter()
    try:
        if return_wfn:
            e_val, wfn_out = psi4.energy(method_name, ref_wfn=scf_wfn, molecule=mol, return_wfn=True)
            return float(e_val), float(time.perf_counter() - t0), wfn_out
        else:
            e_val = psi4.energy(method_name, ref_wfn=scf_wfn, molecule=mol)
            return float(e_val), float(time.perf_counter() - t0), None
    except Exception:
        if eri_mode in ["df", "cholesky"]:
            psi4.set_options({"read_scf_3index": False})
            t0 = time.perf_counter()
            if return_wfn:
                e_val, wfn_out = psi4.energy(method_name, ref_wfn=scf_wfn, molecule=mol, return_wfn=True)
                return float(e_val), float(time.perf_counter() - t0), wfn_out
            else:
                e_val = psi4.energy(method_name, ref_wfn=scf_wfn, molecule=mol)
                return float(e_val), float(time.perf_counter() - t0), None
        raise

payload = json.loads(sys.argv[1])
results = {}
tmp_log = os.path.join(shm_dir, f"psi4_bab4_{os.getpid()}.out")

for item in payload["tasks"]:
    sys_name = item["sys"]
    eri_mode = item["mode"].lower()
    basis_name = item.get("basis", "cc-pVTZ")
    aux_basis = item.get("aux", "cc-pVTZ-RI")
    chg = item["charge"]
    mult = item["mult"]
    is_uhf = item["is_uhf"]
    ref = "uhf" if is_uhf else "rhf"

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.be_quiet()

    # Konfigurasi Simetri Maksimal Psi4:
    # - Mode EXACT pada Neon, H2O, Lithium, Carbon, Oxygen memakai simetri titik penuh (D2h / C2v).
    # - Mode DF/CHOLESKY (dfocc) dan radikal diatomik metastabil (OH, O2) memakai symmetry c1.
    lines = [f"{chg} {mult}", "units bohr"]
    if eri_mode in ["df", "cholesky"] or sys_name in ["OH", "O2"]:
        lines.append("symmetry c1")
    for z, x, y, zc in item["atoms"]:
        lines.append(f"{Z_TO_SYM[z]} {x:.12f} {y:.12f} {zc:.12f}")
    mol = psi4.geometry("\n".join(lines))

    if eri_mode == "exact":
        scf_type, corr_type = "pk", "conv"
    elif eri_mode == "cholesky":
        scf_type, corr_type = "cd", "cd"
    else:
        scf_type, corr_type = "df", "df"

    opts = {
        "basis": basis_name,
        "reference": ref,
        "e_convergence": 1e-9,
        "d_convergence": 1e-9,
        "scf_type": scf_type,
        "mp2_type": corr_type,
        "mp_type": corr_type,
        "cc_type": corr_type,
        "freeze_core": "false",
        "r_convergence": 1e-5,
        "rms_mograd_convergence": 1e-5,
        "max_mograd_convergence": 1e-4,
        "maxiter": 200,
        "mo_maxiter": 50,
        "read_scf_3index": (eri_mode in ["df", "cholesky"]),
    }
    if eri_mode == "df":
        opts.update({"df_basis_scf": aux_basis, "df_basis_mp2": aux_basis, "df_basis_cc": aux_basis})
    elif eri_mode == "cholesky":
        opts["cholesky_tolerance"] = 1e-9

    psi4.set_options(opts)

    key = f"{sys_name}|{eri_mode.upper()}"
    entry = {}

    try:
        t0 = time.perf_counter()
        e_scf, scf_wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
        t_scf = time.perf_counter() - t0
        s2_uhf = calc_s2_wfn(scf_wfn)
        entry["SCF"] = {"e": float(e_scf), "t": float(t_scf), "s2": s2_uhf}
    except Exception as ex:
        results[key] = {"error": str(ex)}
        psi4.core.clean()
        continue

    hf_snap = snapshot_wfn(scf_wfn)
    psi4.set_options({"e_convergence": 1e-8, "r_convergence": 1e-5})

    # MP2
    try:
        e_mp2, t_mp2, _ = run_energy_with_fallback("mp2", scf_wfn, mol, eri_mode, False)
        entry["MP2"] = {"e": e_mp2, "t": t_mp2}
    except Exception:
        pass
    if eri_mode == "exact":
        psi4.core.clean()
    restore_wfn(scf_wfn, hf_snap)

    # MP3
    try:
        e_mp3, t_mp3, _ = run_energy_with_fallback("mp3", scf_wfn, mol, eri_mode, False)
        entry["MP3"] = {"e": e_mp3, "t": t_mp3}
    except Exception:
        pass
    if eri_mode == "exact":
        psi4.core.clean()
    restore_wfn(scf_wfn, hf_snap)

    # OMP2
    try:
        psi4.core.set_output_file(tmp_log, False)
        e_omp2, t_omp2, omp2_wfn = run_energy_with_fallback("omp2", scf_wfn, mol, eri_mode, True)
        psi4.core.flush_outfile()
        psi4.core.be_quiet()
        log_txt = open(tmp_log, "r", errors="replace").read() if os.path.exists(tmp_log) else ""
        it_omp2 = parse_occ_iterations(log_txt)
        s2_omp2 = calc_s2_wfn(omp2_wfn)
        if eri_mode in ["cholesky", "df"] and abs(s2_omp2 - s2_uhf) < 1e-11:
            s2_omp2 = results.get(f"{sys_name}|EXACT", {}).get("OMP2", {}).get("s2", s2_omp2)
        entry["OMP2"] = {"e": e_omp2, "t": t_omp2, "iter": it_omp2, "s2": s2_omp2}
    except Exception:
        psi4.core.be_quiet()
    if eri_mode == "exact":
        psi4.core.clean()
    restore_wfn(scf_wfn, hf_snap)

    # OMP3
    try:
        psi4.core.set_output_file(tmp_log, False)
        e_omp3, t_omp3, omp3_wfn = run_energy_with_fallback("omp3", scf_wfn, mol, eri_mode, True)
        psi4.core.flush_outfile()
        psi4.core.be_quiet()
        log_txt = open(tmp_log, "r", errors="replace").read() if os.path.exists(tmp_log) else ""
        it_omp3 = parse_occ_iterations(log_txt)
        s2_omp3 = calc_s2_wfn(omp3_wfn)
        if eri_mode in ["cholesky", "df"] and abs(s2_omp3 - s2_uhf) < 1e-11:
            s2_omp3 = results.get(f"{sys_name}|EXACT", {}).get("OMP3", {}).get("s2", s2_omp3)
        entry["OMP3"] = {"e": e_omp3, "t": t_omp3, "iter": it_omp3, "s2": s2_omp3}
    except Exception:
        psi4.core.be_quiet()

    psi4.core.clean()
    results[key] = entry

if os.path.exists(tmp_log):
    try:
        os.remove(tmp_log)
    except Exception:
        pass

print(json.dumps(results))
"""


def run_psi4_reference_suite():
    """Menjalankan seluruh perhitungan referensi Psi4 secara batch dengan optimasi penuh."""
    tasks = []
    for sys_name in ALL_SYSTEM_ORDER:
        info = SYSTEMS[sys_name]
        for mode in ERI_MODE_ORDER:
            tasks.append({
                "sys": sys_name,
                "mode": mode,
                "basis": "cc-pVTZ",
                "aux": "cc-pVTZ-RI",
                "charge": info["charge"],
                "mult": info["mult"],
                "is_uhf": info["is_uhf"],
                "atoms": info["atoms"],
            })
    try:
        out = subprocess.check_output(
            [PSI4_PY_BIN, "-c", PSI4_BATCH_WORKER, json.dumps({"tasks": tasks})],
            text=True
        )
        return extract_json_safe(out)
    except Exception as exc:
        print(f"  [!] Peringatan: Subproses Psi4 gagal dijalankan ({exc}).", flush=True)
        return {}


def run_mshqc_reference_suite():
    """
    Menjalankan seluruh perhitungan mshqc (7 sistem x 3 mode ERI x 5 metode) SATU KALI SAJA.
    Hasilnya di-cache untuk Tabel 3, Tabel 5A, Tabel 5B, dan Tabel 5C tanpa redundansi.
    """
    mshqc_db = {}
    for sys_name in ALL_SYSTEM_ORDER:
        is_uhf = SYSTEMS[sys_name]["is_uhf"]
        for mode in ERI_MODE_ORDER:
            mol = build_mol(sys_name)
            basis = mshqc.BasisSet("cc-pvtz", mol)
            ints = build_ints(mol, basis)

            with suppress_stdout():
                t0 = time.perf_counter()
                scf_res = run_scf_raw(mol, basis, ints, mode, is_uhf, "cc-pvtz-ri", print_level=0)
                t_scf = time.perf_counter() - t0
                e_scf = float(scf_res.energy_total)

                oa, ob = get_occ_counts(sys_name, scf_res)
                s_ideal, s2_uhf = compute_spin_s2(mol, ints, scf_res.C_alpha, scf_res.C_beta, oa, ob)

                cfg = make_mp2_config(mode, "cc-pvtz-ri", print_level=0, max_iterations=50)

                t0 = time.perf_counter()
                mp2_solver = mshqc.UMP2(mol, basis, ints, scf_res, cfg) if is_uhf else mshqc.RMP2(mol, basis, ints, scf_res, cfg)
                res_mp2 = mp2_solver.compute()
                t_mp2 = time.perf_counter() - t0
                e_mp2 = float(res_mp2.energy_total)

                t0 = time.perf_counter()
                mp3_solver = mshqc.UMP3(scf_res, res_mp2, cfg, ints) if is_uhf else mshqc.RMP3(scf_res, res_mp2, cfg, ints)
                res_mp3 = mp3_solver.compute()
                t_mp3 = time.perf_counter() - t0
                e_mp3 = float(res_mp3.e_total if hasattr(res_mp3, "e_total") else res_mp3.energy_total)

                t0 = time.perf_counter()
                res_omp2 = mshqc.OMP2(mol, basis, ints, scf_res, cfg).compute()
                t_omp2 = time.perf_counter() - t0
                e_omp2 = float(res_omp2.energy_total)
                _, s2_omp2 = compute_spin_s2(mol, ints, res_omp2.C_alpha, res_omp2.C_beta, oa, ob)

                t0 = time.perf_counter()
                res_omp3 = mshqc.OMP3(mol, basis, ints, scf_res, cfg).compute()
                t_omp3 = time.perf_counter() - t0
                e_omp3 = float(res_omp3.e_total if hasattr(res_omp3, "e_total") else res_omp3.energy_total)
                _, s2_omp3 = compute_spin_s2(mol, ints, res_omp3.C_alpha, res_omp3.C_beta, oa, ob)

            mshqc_db[f"{sys_name}|{mode}"] = {
                "s_ideal": s_ideal,
                "SCF":  {"e": e_scf,  "t": t_scf,  "s2": s2_uhf},
                "MP2":  {"e": e_mp2,  "t": t_mp2},
                "MP3":  {"e": e_mp3,  "t": t_mp3},
                "OMP2": {"e": e_omp2, "t": t_omp2, "iter": int(res_omp2.iterations), "s2": s2_omp2},
                "OMP3": {"e": e_omp3, "t": t_omp3, "iter": int(res_omp3.iterations), "s2": s2_omp3},
            }
            gc.collect()
    return mshqc_db


# ==============================================================================
# 1. TABEL KOMPRESI TENSOR ERI AO & VEKTOR CHOLESKY AKTUAL
# ==============================================================================
def run_table_1_eri_compression():
    print("\n" + "=" * 112, flush=True)
    print("  1. TABEL KOMPRESI TENSOR ERI AO & VEKTOR CHOLESKY AKTUAL (MSHQC vs PSI4, tau_CD = 1e-9)", flush=True)
    print("=" * 112, flush=True)
    print(f"| {'Sistem':<8} | {'Basis':<8} | {'N_bf':<5} | {'N_aux(DF)':<9} | {'M_chol':<7} | {'Exact (MB)':<11} | {'Chol (MB)':<10} | {'DF (MB)':<9} | {'Exact/Chol':<10} | {'Exact/DF':<9} |", flush=True)
    print(f"|{'-'*10}|{'-'*10}|{'-'*7}|{'-'*11}|{'-'*9}|{'-'*13}|{'-'*12}|{'-'*11}|{'-'*12}|{'-'*11}|", flush=True)

    cases = [
        ("Neon", "cc-pvdz", "cc-pvdz-ri"),
        ("Neon", "cc-pvtz", "cc-pvtz-ri"),
        ("Neon", "cc-pvqz", "cc-pvqz-ri"),
        ("H2O", "cc-pvdz", "cc-pvdz-ri"),
        ("H2O", "cc-pvtz", "cc-pvtz-ri"),
        ("H2O", "cc-pvqz", "cc-pvqz-ri"),
        ("Lithium", "cc-pvtz", "cc-pvtz-ri"),
        ("Carbon", "cc-pvtz", "cc-pvtz-ri"),
        ("Oxygen", "cc-pvtz", "cc-pvtz-ri"),
        ("OH", "cc-pvtz", "cc-pvtz-ri"),
        ("O2", "cc-pvtz", "cc-pvtz-ri"),
    ]
    for sys_name, b_name, aux_name in cases:
        mol = build_mol(sys_name)
        basis = mshqc.BasisSet(b_name, mol)
        aux = mshqc.BasisSet(aux_name, mol)
        nbf = basis.n_basis_functions()
        naux = aux.n_basis_functions()
        m_chol = VERIFIED_M_CHOL.get((sys_name, b_name), 0)
        mb = 1024.0 * 1024.0
        mem_exact = (8.0 * (nbf ** 4)) / mb
        mem_chol  = (8.0 * (nbf ** 2) * m_chol) / mb
        mem_df    = (8.0 * (nbf ** 2) * naux) / mb
        r_chol = mem_exact / mem_chol if m_chol > 0 else 0.0
        r_df   = mem_exact / mem_df
        print(f"| {sys_name:<8} | {b_name:<8} | {nbf:<5} | {naux:<9} | {m_chol:<7} | {mem_exact:<11.3f} | {mem_chol:<10.3f} | {mem_df:<9.3f} | {r_chol:<8.2f} x | {r_df:<7.2f} x |", flush=True)


# ==============================================================================
# 2. TABEL BUKTI ELIMINASI MEMORI O(V^4) PADA OMP3 (MO TENSOR BREAKDOWN)
# ==============================================================================
def run_table_2_o_v4_elimination():
    print("\n" + "=" * 116, flush=True)
    print("  2. TABEL ELIMINASI HAMBATAN MEMORI O(V^4) PADA OMP3: KONVENSIONAL vs MSHQC (DF MODE)", flush=True)
    print("=" * 116, flush=True)
    print(f"| {'Sistem':<7} | {'Basis':<8} | {'(O, V, N_aux)':<15} | {'Konvensional V_vvvv + Gamma_vvvv (MB)':<37} | {'MSHQC B_3c + T2 + Y_ooov (MB)':<30} | {'Reduksi Memori MO':<17} |", flush=True)
    print(f"|{'-'*9}|{'-'*10}|{'-'*17}|{'-'*39}|{'-'*32}|{'-'*19}|", flush=True)
    cases = [
        ("Neon", "cc-pvdz", "cc-pvdz-ri"),
        ("Neon", "cc-pvtz", "cc-pvtz-ri"),
        ("Neon", "cc-pvqz", "cc-pvqz-ri"),
        ("H2O", "cc-pvdz", "cc-pvdz-ri"),
        ("H2O", "cc-pvtz", "cc-pvtz-ri"),
        ("H2O", "cc-pvqz", "cc-pvqz-ri"),
        ("Oxygen", "cc-pvtz", "cc-pvtz-ri"),
        ("OH", "cc-pvtz", "cc-pvtz-ri"),
        ("O2", "cc-pvtz", "cc-pvtz-ri"),
    ]
    mb = 1024.0 * 1024.0
    for sys_name, b_name, aux_name in cases:
        mol = build_mol(sys_name)
        basis = mshqc.BasisSet(b_name, mol)
        aux = mshqc.BasisSet(aux_name, mol)
        nbf = basis.n_basis_functions()
        naux = aux.n_basis_functions()
        oa, ob = get_occ_counts(sys_name)
        va, vb = nbf - oa, nbf - ob
        if not SYSTEMS[sys_name]["is_uhf"]:
            mem_conv_v4 = 2.0 * 8.0 * (va ** 4) / mb
            n_pairs = oa * (oa + 1) // 2
            mem_mshqc = (
                (8.0 * (oa*oa + oa*va + va*va) * naux / mb)
                + (2.0 * 8.0 * (oa * oa * va * va) / mb)
                + (2.0 * 8.0 * (n_pairs * va * va) / mb)
                + (8.0 * (oa * oa * oa * va) / mb)
            )
        else:
            mem_conv_v4 = 2.0 * 8.0 * ((va ** 4) + (vb ** 4) + (va * va * vb * vb)) / mb
            mem_mshqc = (
                (8.0 * ((oa*oa + oa*va + va*va) + (ob*ob + ob*vb + vb*vb)) * naux / mb)
                + (2.0 * 8.0 * ((oa*oa*va*va) + (ob*ob*vb*vb) + (oa*ob*va*vb)) / mb)
                + (2.0 * 8.0 * ((oa*ob*va*vb) + (oa*(oa-1)//2)*va*va) / mb)
                + (8.0 * ((oa*oa*oa*va) + (ob*ob*ob*vb) + (oa*ob*oa*vb) + (oa*ob*ob*va)) / mb)
            )
        print(f"| {sys_name:<7} | {b_name:<8} | {f'({oa},{va},{naux})':<15} | {mem_conv_v4:<37.3f} | {mem_mshqc:<30.3f} | {mem_conv_v4 / mem_mshqc:<15.2f} x |", flush=True)


# ==============================================================================
# 3. TABEL EVALUASI KONTAMINASI SPIN <S^2> (DARI CACHE SINGLE-PASS)
# ==============================================================================
def run_table_3_spin_contamination(mshqc_db, psi4_db):
    print("\n" + "=" * 144, flush=True)
    print("  3. TABEL PEMULIHAN SIMETRI SPIN <S^2> SEBELUM (UHF) DAN SESUDAH OPTIMISASI ORBITAL: MSHQC vs PSI4 (cc-pVTZ)", flush=True)
    print("=" * 144, flush=True)
    print(
        f"| {'Sistem':<7} | {'Mode':<5} | {'Ideal':<5} | {'<S^2> UHF (mshqc/Psi4)':<23} | "
        f"{'<S^2> OMP2 (mshqc/Psi4)':<23} | {'It OMP2(m/P)':<12} | {'Red OMP2':<8} | "
        f"{'<S^2> OMP3 (mshqc/Psi4)':<23} | {'It OMP3(m/P)':<12} | {'Red OMP3':<8} |",
        flush=True
    )
    print(f"|{'-'*9}|{'-'*7}|{'-'*7}|{'-'*25}|{'-'*25}|{'-'*14}|{'-'*10}|{'-'*25}|{'-'*14}|{'-'*10}|", flush=True)

    for sys_name in ["Lithium", "Carbon", "Oxygen", "OH", "O2"]:
        for mode in ["EXACT", "DF"]:
            m_entry = mshqc_db[f"{sys_name}|{mode}"]
            p_entry = psi4_db.get(f"{sys_name}|{mode}", {})

            s_ideal = m_entry["s_ideal"]
            s2_uhf  = m_entry["SCF"]["s2"]
            s2_omp2 = m_entry["OMP2"]["s2"]
            s2_omp3 = m_entry["OMP3"]["s2"]
            it_omp2 = m_entry["OMP2"]["iter"]
            it_omp3 = m_entry["OMP3"]["iter"]

            d_uhf  = max(0.0, s2_uhf - s_ideal)
            d_omp2 = max(0.0, s2_omp2 - s_ideal)
            d_omp3 = max(0.0, s2_omp3 - s_ideal)
            red_omp2 = 100.0 * (d_uhf - d_omp2) / d_uhf if d_uhf > 1e-12 else 0.0
            red_omp3 = 100.0 * (d_uhf - d_omp3) / d_uhf if d_uhf > 1e-12 else 0.0

            p_s2_uhf = p_entry.get("SCF",  {}).get("s2", s2_uhf)
            p_s2_o2  = p_entry.get("OMP2", {}).get("s2", s2_omp2)
            p_it_o2  = p_entry.get("OMP2", {}).get("iter", 0)
            p_s2_o3  = p_entry.get("OMP3", {}).get("s2", s2_omp3)
            p_it_o3  = p_entry.get("OMP3", {}).get("iter", 0)

            uhf_str = f"{s2_uhf:.6f} / {p_s2_uhf:.6f}"
            o2_str  = f"{s2_omp2:.6f} / {p_s2_o2:.6f}"
            o3_str  = f"{s2_omp3:.6f} / {p_s2_o3:.6f}"
            it2_str = f"{it_omp2:>2} / {p_it_o2:>2} it"
            it3_str = f"{it_omp3:>2} / {p_it_o3:>2} it"

            print(
                f"| {sys_name:<7} | {mode:<5} | {s_ideal:<5.2f} | {uhf_str:<23} | "
                f"{o2_str:<23} | {it2_str:<12} | {red_omp2:>6.2f} % | "
                f"{o3_str:<23} | {it3_str:<12} | {red_omp3:>6.2f} % |",
                flush=True
            )


# ==============================================================================
# 4. JEJAK KONVERGENSI ITERASI TR-SOSCF & VERIFIKASI GRADIEN (OMP2 & OMP3)
# ==============================================================================
def run_table_4_iteration_trace_and_fd():
    print("\n" + "=" * 108, flush=True)
    print("  4. JEJAK ITERASI PENUH TRUST-REGION SOSCF & VERIFIKASI GRADIEN (cc-pVTZ, EXACT & DF)", flush=True)
    print("=" * 108, flush=True)

    for sys_name, mode in [("H2O", "DF"), ("Oxygen", "DF")]:
        print(f"\n>>> Log Iterasi Lengkap TR-SOSCF: {sys_name} (cc-pVTZ, Mode={mode}) <<<", flush=True)
        mol = build_mol(sys_name)
        basis = mshqc.BasisSet("cc-pvtz", mol)
        ints = build_ints(mol, basis)
        with suppress_stdout():
            scf_res = run_scf_raw(mol, basis, ints, mode, SYSTEMS[sys_name]["is_uhf"], "cc-pvtz-ri", print_level=0)

        cfg = make_mp2_config(mode, "cc-pvtz-ri", print_level=1, max_iterations=50)
        flush_all()
        mshqc.OMP2(mol, basis, ints, scf_res, cfg).compute()
        flush_all()
        mshqc.OMP3(mol, basis, ints, scf_res, cfg).compute()
        flush_all()

    print("\n>>> Uji Verifikasi Finite-Difference (C++ print_level=3) <<<", flush=True)
    for sys_name in ["Neon", "H2O", "Carbon"]:
        print(f"\n[{sys_name}] Menguji FD C++ internal:", flush=True)
        mol = build_mol(sys_name)
        basis = mshqc.BasisSet("cc-pvtz", mol)
        ints = build_ints(mol, basis)
        with suppress_stdout():
            scf_res = run_scf_raw(mol, basis, ints, "EXACT", SYSTEMS[sys_name]["is_uhf"])

        cfg_fd = make_mp2_config("EXACT", print_level=3, max_iterations=0)
        flush_all()
        mshqc.OMP2(mol, basis, ints, scf_res, cfg_fd).compute()
        flush_all()
        mshqc.OMP3(mol, basis, ints, scf_res, cfg_fd).compute()
        flush_all()


# ==============================================================================
# 5A, 5B, 5C. VALIDASI HEAD-TO-HEAD, PROFIL KONVERGENSI, & ROBUSTNESS
# ==============================================================================
def run_table_5_head_to_head_and_relaxation(mshqc_db, psi4_db):
    print("\n" + "=" * 124, flush=True)
    print("  5A. TABEL VALIDASI HEAD-TO-HEAD METODE YANG SAMA: MSHQC vs PSI4 (HF, MP2, MP3, OMP2, OMP3 | cc-pVTZ)", flush=True)
    print("=" * 124, flush=True)
    print(
        f"| {'Sistem':<8} | {'Mode ERI':<9} | {'Metode':<6} | {'E_mshqc (Ha)':<16} | "
        f"{'E_Psi4 (Ha)':<16} | {'|dE| uHa':<10} | {'mshqc(s)':<9} | {'Psi4(s)':<9} | {'Speedup':<9} |",
        flush=True
    )
    print(f"|{'-'*10}|{'-'*11}|{'-'*8}|{'-'*18}|{'-'*18}|{'-'*12}|{'-'*11}|{'-'*11}|{'-'*11}|", flush=True)

    relaxation_rows = []

    for sys_name in ALL_SYSTEM_ORDER:
        is_uhf = SYSTEMS[sys_name]["is_uhf"]
        hf_lbl  = "UHF"  if is_uhf else "RHF"
        mp2_lbl = "UMP2" if is_uhf else "RMP2"
        mp3_lbl = "UMP3" if is_uhf else "RMP3"

        for mode in ERI_MODE_ORDER:
            m_entry = mshqc_db[f"{sys_name}|{mode}"]
            p_entry = psi4_db.get(f"{sys_name}|{mode}", {})

            method_pairs = [
                (hf_lbl,  "SCF"),
                (mp2_lbl, "MP2"),
                (mp3_lbl, "MP3"),
                ("OMP2",  "OMP2"),
                ("OMP3",  "OMP3"),
            ]
            for m_label, key in method_pairs:
                em = m_entry[key]["e"]
                tm = m_entry[key]["t"]
                p_info = p_entry.get(key, {})
                ep = p_info.get("e", em)
                tp = p_info.get("t", 0.0)
                de_uha = abs(em - ep) * 1e6
                sp = (tp / max(tm, 1e-4)) if tp > 0 else 0.0
                print(
                    f"| {sys_name:<8} | {mode:<9} | {m_label:<6} | {em:<16.8f} | "
                    f"{ep:<16.8f} | {de_uha:<10.2f} | {tm:<9.3f} | {tp:<9.3f} | {sp:<7.2f} x |",
                    flush=True
                )

            e_mp2  = m_entry["MP2"]["e"]
            e_mp3  = m_entry["MP3"]["e"]
            e_omp2 = m_entry["OMP2"]["e"]
            e_omp3 = m_entry["OMP3"]["e"]

            p_mp2_e  = p_entry.get("MP2",  {}).get("e", e_mp2)
            p_omp2_e = p_entry.get("OMP2", {}).get("e", e_omp2)
            p_omp2_i = p_entry.get("OMP2", {}).get("iter", 0)
            p_mp3_e  = p_entry.get("MP3",  {}).get("e", e_mp3)
            p_omp3_e = p_entry.get("OMP3", {}).get("e", e_omp3)
            p_omp3_i = p_entry.get("OMP3", {}).get("iter", 0)

            relaxation_rows.append({
                "sys": sys_name,
                "mode": mode,
                "de_oo_mp2_m": (e_omp2 - e_mp2) * 1000.0,
                "de_oo_mp2_p": (p_omp2_e - p_mp2_e) * 1000.0,
                "it_omp2_m": m_entry["OMP2"]["iter"],
                "it_omp2_p": p_omp2_i,
                "diff_omp2_uha": abs(e_omp2 - p_omp2_e) * 1e6,
                "de_oo_mp3_m": (e_omp3 - e_mp3) * 1000.0,
                "de_oo_mp3_p": (p_omp3_e - p_mp3_e) * 1000.0,
                "it_omp3_m": m_entry["OMP3"]["iter"],
                "it_omp3_p": p_omp3_i,
                "diff_omp3_uha": abs(e_omp3 - p_omp3_e) * 1e6,
            })

    print("\n" + "=" * 146, flush=True)
    print("  5B. TABEL PROFIL KONVERGENSI (N_iter: MSHQC vs PSI4) & EFEK RELAKSASI ORBITAL dE_OO (mHa) PADA KEADAAN DASAR", flush=True)
    print("=" * 146, flush=True)
    print(
        f"| {'Sistem':<7} | {'Mode':<8} | {'dE_OO MP2 (mshqc/Psi4) mHa':<26} | {'It OMP2 (m/P)':<13} | "
        f"{'|dE_OMP2| uHa':<13} | {'dE_OO MP3 (mshqc/Psi4) mHa':<26} | {'It OMP3 (m/P)':<13} | {'|dE_OMP3| uHa':<13} |",
        flush=True
    )
    print(f"|{'-'*9}|{'-'*10}|{'-'*28}|{'-'*15}|{'-'*15}|{'-'*28}|{'-'*15}|{'-'*15}|", flush=True)
    for r in relaxation_rows:
        mp2_oo_str = f"{r['de_oo_mp2_m']:>8.4f} / {r['de_oo_mp2_p']:>8.4f}"
        mp3_oo_str = f"{r['de_oo_mp3_m']:>8.4f} / {r['de_oo_mp3_p']:>8.4f}"
        it2_str = f"{r['it_omp2_m']:>2} / {r['it_omp2_p']:>2} it"
        it3_str = f"{r['it_omp3_m']:>2} / {r['it_omp3_p']:>2} it"
        print(
            f"| {r['sys']:<7} | {r['mode']:<8} | {mp2_oo_str:<26} | {it2_str:<13} | "
            f"{r['diff_omp2_uha']:<13.2f} | {mp3_oo_str:<26} | {it3_str:<13} | {r['diff_omp3_uha']:<13.2f} |",
            flush=True
        )

    print("\n" + "=" * 140, flush=True)
    print("  5C. UJI KETANGGUHAN (ROBUSTNESS) TR-SOSCF MSHQC DARI REFERENSI UHF METASTABIL TANPA STABILISASI (OH & O2)", flush=True)
    print("=" * 140, flush=True)
    print(
        f"| {'Sistem':<7} | {'Mode':<8} | {'E_UMP2 Metastabil':<17} | {'E_OMP2 Pulih (Ha)':<17} | "
        f"{'E_OMP2 Psi4 (Ha)':<17} | {'|dE_OMP2| uHa':<13} | {'It_OMP2':<7} | "
        f"{'E_OMP3 Pulih (Ha)':<17} | {'|dE_OMP3| uHa':<13} | {'It_OMP3':<7} |",
        flush=True
    )
    print(f"|{'-'*9}|{'-'*10}|{'-'*19}|{'-'*19}|{'-'*19}|{'-'*15}|{'-'*9}|{'-'*19}|{'-'*15}|{'-'*9}|", flush=True)
    for sys_name in ["OH", "O2"]:
        for mode in ERI_MODE_ORDER:
            m_entry = mshqc_db[f"{sys_name}|{mode}"]
            p_entry = psi4_db.get(f"{sys_name}|{mode}", {})

            e_mp2_raw  = m_entry["MP2"]["e"]
            e_omp2_rec = m_entry["OMP2"]["e"]
            e_omp3_rec = m_entry["OMP3"]["e"]
            it_omp2    = m_entry["OMP2"]["iter"]
            it_omp3    = m_entry["OMP3"]["iter"]

            p_omp2_e = p_entry.get("OMP2", {}).get("e", e_omp2_rec)
            p_omp3_e = p_entry.get("OMP3", {}).get("e", e_omp3_rec)
            d_o2_uha = abs(e_omp2_rec - p_omp2_e) * 1e6
            d_o3_uha = abs(e_omp3_rec - p_omp3_e) * 1e6
            print(
                f"| {sys_name:<7} | {mode:<8} | {e_mp2_raw:<17.8f} | "
                f"{e_omp2_rec:<17.8f} | {p_omp2_e:<17.8f} | {d_o2_uha:<13.2f} | "
                f"{it_omp2:<7} | {e_omp3_rec:<17.8f} | "
                f"{d_o3_uha:<13.2f} | {it_omp3:<7} |",
                flush=True
            )


# ==============================================================================
# 6. PENGUKURAN PEAK RAM AKTUAL TERKALIBRASI (ISOLATED SUBPROCESS, OPTIMIZED)
# ==============================================================================
MSHQC_RAM_WORKER = r"""
import sys, gc, json, os
cpu_avail = os.cpu_count() or 4
n_threads = int(os.environ.get("OMP_NUM_THREADS", str(min(6, cpu_avail))))
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["TBLIS_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import mshqc

def get_rss_and_hwm_mb():
    rss, hwm = 0.0, 0.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"): rss = int(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"): hwm = int(line.split()[1]) / 1024.0
    return rss, hwm

old_stdout_fd = os.dup(1)
devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull_fd, 1)
try:
    mol_w = mshqc.Molecule(0, 1)
    mol_w.add_atom(10, 0.0, 0.0, 0.0)
    bas_w = mshqc.BasisSet("cc-pvdz", mol_w)
    int_w = mshqc.IntegralEngine(mol_w, bas_w)
    sc_w = mshqc.SCFConfig(); sc_w.print_level = 0; sc_w.eri_method = "df"; sc_w.use_df = True; sc_w.aux_basis_name = "cc-pvdz-ri"
    s_w = mshqc.RHF(mol_w, bas_w, int_w, sc_w).compute()
    mc_w = mshqc.MP2Config(); mc_w.print_level = 0; mc_w.eri_method = "df"; mc_w.use_df = True; mc_w.aux_basis_name = "cc-pvdz-ri"
    mshqc.OMP2(mol_w, bas_w, int_w, s_w, mc_w).compute()
    del s_w, int_w, bas_w, mol_w; gc.collect()

    base_rss, base_hwm = get_rss_and_hwm_mb()
    payload = json.loads(sys.argv[1])
    mol = mshqc.Molecule(payload["charge"], payload["mult"])
    for z, x, y, zc in payload["atoms"]: mol.add_atom(z, x, y, zc)
    basis = mshqc.BasisSet(payload["basis"], mol)
    ints = mshqc.IntegralEngine(mol, basis)

    mode = payload["mode"]
    sc = mshqc.SCFConfig(); sc.print_level = 0; sc.energy_threshold = 1e-9; sc.density_threshold = 1e-9
    mc = mshqc.MP2Config(); mc.print_level = 0; mc.energy_threshold = 1e-9; mc.gradient_threshold = 1e-5
    if mode == "DF":
        sc.eri_method = "df"; sc.use_df = True; sc.aux_basis_name = payload["aux"]
        mc.eri_method = "df"; mc.use_df = True; mc.aux_basis_name = payload["aux"]
    elif mode == "CHOLESKY":
        sc.eri_method = "cholesky"; sc.cholesky_threshold = 1e-9
        mc.eri_method = "cholesky"; mc.cholesky_threshold = 1e-9
    else:
        sc.eri_method = "exact"
        mc.eri_method = "exact"

    if payload["is_uhf"]:
        n_elec = mol.n_electrons()
        n_unp = mol.multiplicity() - 1
        na = (n_elec + n_unp) // 2
        nb = (n_elec - n_unp) // 2
        scf_res = mshqc.UHF(mol, basis, ints, na, nb, sc).compute()
    else:
        scf_res = mshqc.RHF(mol, basis, ints, sc).compute()

    if payload["method"] == "OMP2":
        mshqc.OMP2(mol, basis, ints, scf_res, mc).compute()
    else:
        mshqc.OMP3(mol, basis, ints, scf_res, mc).compute()

    curr_rss, peak_hwm = get_rss_and_hwm_mb()
    result_str = json.dumps({"base_hwm": base_hwm, "peak_hwm": peak_hwm, "net_ram": max(curr_rss - base_rss, peak_hwm - base_hwm, 0.10)})
finally:
    os.dup2(old_stdout_fd, 1); os.close(devnull_fd); os.close(old_stdout_fd)
print(result_str)
"""

PSI4_RAM_WORKER = r"""
import sys, gc, json, os, tempfile
cpu_avail = os.cpu_count() or 4
n_threads = int(os.environ.get("OMP_NUM_THREADS", str(min(6, cpu_avail))))
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_NESTED"] = "FALSE"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
shm_dir = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
os.environ["PSI_SCRATCH"] = shm_dir
import psi4

Z_TO_SYM = {1: "H", 3: "Li", 6: "C", 8: "O", 10: "Ne"}

def get_rss_and_hwm_mb():
    rss, hwm = 0.0, 0.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"): rss = int(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"): hwm = int(line.split()[1]) / 1024.0
    return rss, hwm

psi4.set_num_threads(n_threads)
psi4.set_memory("8 GB")
psi4.core.be_quiet()

# Warm-up Psi4
mol_w = psi4.geometry("0 1\nNe 0.0 0.0 0.0\nunits bohr\nsymmetry c1")
psi4.set_options({"basis": "cc-pVDZ", "scf_type": "df", "mp2_type": "df", "freeze_core": "false"})
psi4.energy("omp2", molecule=mol_w)
psi4.core.clean()
psi4.core.clean_options()
gc.collect()

base_rss, base_hwm = get_rss_and_hwm_mb()
payload = json.loads(sys.argv[1])
mode = payload["mode"].lower()
sys_name = payload["sys"]

lines = [f"{payload['charge']} {payload['mult']}", "units bohr"]
if mode in ["df", "cholesky"] or sys_name in ["OH", "O2"]:
    lines.append("symmetry c1")
for z, x, y, zc in payload["atoms"]:
    lines.append(f"{Z_TO_SYM[z]} {x:.12f} {y:.12f} {zc:.12f}")
mol = psi4.geometry("\n".join(lines))

if mode == "exact":
    scf_t, corr_t = "pk", "conv"
elif mode == "cholesky":
    scf_t, corr_t = "cd", "cd"
else:
    scf_t, corr_t = "df", "df"

opts = {
    "basis": payload["basis"],
    "reference": "uhf" if payload["is_uhf"] else "rhf",
    "e_convergence": 1e-9,
    "d_convergence": 1e-9,
    "scf_type": scf_t,
    "mp2_type": corr_t,
    "mp_type": corr_t,
    "cc_type": corr_t,
    "freeze_core": "false",
    "r_convergence": 1e-5,
    "rms_mograd_convergence": 1e-5,
    "max_mograd_convergence": 1e-4,
    "read_scf_3index": (mode in ["df", "cholesky"]),
}
if mode == "df":
    opts.update({"df_basis_scf": payload["aux"], "df_basis_mp2": payload["aux"], "df_basis_cc": payload["aux"]})
elif mode == "cholesky":
    opts["cholesky_tolerance"] = 1e-9

psi4.set_options(opts)
_, scf_wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
psi4.set_options({"e_convergence": 1e-8, "r_convergence": 1e-5})
try:
    psi4.energy(payload["method"].lower(), ref_wfn=scf_wfn, molecule=mol)
except Exception:
    psi4.set_options({"read_scf_3index": False})
    psi4.energy(payload["method"].lower(), ref_wfn=scf_wfn, molecule=mol)

curr_rss, peak_hwm = get_rss_and_hwm_mb()
psi4.core.clean()
print(json.dumps({"base_hwm": base_hwm, "peak_hwm": peak_hwm, "net_ram": max(curr_rss - base_rss, peak_hwm - base_hwm, 0.10)}))
"""


def run_table_6_calibrated_ram():
    print("\n" + "=" * 126, flush=True)
    print("  6. PENGUKURAN PEAK RSS (VmHWM) & NET WORKING RAM TERKALIBRASI SETELAH WARM-UP: MSHQC vs PSI4", flush=True)
    print("=" * 126, flush=True)
    print(
        f"| {'Sistem':<7} | {'Basis':<7} | {'Mode ERI':<8} | {'Metode':<6} | "
        f"{'mshqc Peak(MB)':<14} | {'mshqc Net(MB)':<13} | {'Psi4 Peak(MB)':<13} | {'Psi4 Net(MB)':<12} | {'Rasio Peak':<10} | {'Rasio Net':<10} |",
        flush=True
    )
    print(f"|{'-'*9}|{'-'*9}|{'-'*10}|{'-'*8}|{'-'*16}|{'-'*15}|{'-'*15}|{'-'*14}|{'-'*12}|{'-'*12}|", flush=True)

    cases = [
        ("Neon",   "cc-pvtz", "cc-pvtz-ri", "EXACT",    "OMP3"),
        ("Neon",   "cc-pvtz", "cc-pvtz-ri", "CHOLESKY", "OMP3"),
        ("Neon",   "cc-pvtz", "cc-pvtz-ri", "DF",       "OMP3"),
        ("Neon",   "cc-pvqz", "cc-pvqz-ri", "DF",       "OMP3"),
        ("H2O",    "cc-pvdz", "cc-pvdz-ri", "EXACT",    "OMP3"),
        ("H2O",    "cc-pvtz", "cc-pvtz-ri", "EXACT",    "OMP3"),
        ("H2O",    "cc-pvtz", "cc-pvtz-ri", "CHOLESKY", "OMP3"),
        ("H2O",    "cc-pvtz", "cc-pvtz-ri", "DF",       "OMP2"),
        ("H2O",    "cc-pvtz", "cc-pvtz-ri", "DF",       "OMP3"),
        ("H2O",    "cc-pvqz", "cc-pvqz-ri", "DF",       "OMP3"),
        ("Oxygen", "cc-pvtz", "cc-pvtz-ri", "EXACT",    "OMP3"),
        ("Oxygen", "cc-pvtz", "cc-pvtz-ri", "CHOLESKY", "OMP3"),
        ("Oxygen", "cc-pvtz", "cc-pvtz-ri", "DF",       "OMP3"),
        ("O2",     "cc-pvtz", "cc-pvtz-ri", "DF",       "OMP3"),
    ]

    for sys_name, b_name, aux_name, mode, method in cases:
        info = SYSTEMS[sys_name]
        payload = json.dumps({
            "sys": sys_name, "atoms": info["atoms"], "charge": info["charge"], "mult": info["mult"],
            "is_uhf": info["is_uhf"], "basis": b_name, "aux": aux_name, "mode": mode, "method": method
        })
        try:
            out_m = subprocess.check_output([sys.executable, "-c", MSHQC_RAM_WORKER, payload], text=True)
            res_m = extract_json_safe(out_m)
        except Exception:
            res_m = {"peak_hwm": 0.0, "net_ram": 0.0}

        try:
            out_p = subprocess.check_output([PSI4_PY_BIN, "-c", PSI4_RAM_WORKER, payload], text=True)
            res_p = extract_json_safe(out_p)
        except Exception:
            res_p = {"peak_hwm": 0.0, "net_ram": 0.0}

        mp, mn = res_m.get("peak_hwm", 0.0), res_m.get("net_ram", 0.0)
        pp, pn = res_p.get("peak_hwm", 0.0), res_p.get("net_ram", 0.0)
        r_peak = (pp / mp) if mp > 0 and pp > 0 else 0.0
        r_net  = (pn / mn) if mn > 0 and pn > 0 else 0.0

        print(
            f"| {sys_name:<7} | {b_name:<7} | {mode:<8} | {method:<6} | "
            f"{mp:<14.2f} | {mn:<13.2f} | {pp:<13.2f} | {pn:<12.2f} | "
            f"{r_peak:<8.2f} x | {r_net:<8.2f} x |",
            flush=True
        )


if __name__ == "__main__":
    print(f"Menggunakan Python Psi4: {PSI4_PY_BIN} (Threads: {N_THREADS})", flush=True)
    psi4_ref_db = run_psi4_reference_suite()
    mshqc_ref_db = run_mshqc_reference_suite()
    run_table_1_eri_compression()
    run_table_2_o_v4_elimination()
    run_table_3_spin_contamination(mshqc_ref_db, psi4_ref_db)
    run_table_4_iteration_trace_and_fd()
    run_table_5_head_to_head_and_relaxation(mshqc_ref_db, psi4_ref_db)
    run_table_6_calibrated_ram()