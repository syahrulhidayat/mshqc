#!/usr/bin/env python3
"""
Benchmark & Validasi Empiris Bab IV Skripsi (MSHQC vs Psi4) - Max Performance & Zero Redundancy
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

SYSTEMS = {
    "Neon":    {"atoms": [(10, 0.0, 0.0, 0.0)], "charge": 0, "mult": 1, "is_uhf": False},
    "H2O":     {"atoms": [(8, 0.0, 0.0, 0.117300 * ANG_TO_BOHR), (1, 0.0, 0.757200 * ANG_TO_BOHR, -0.469200 * ANG_TO_BOHR), (1, 0.0, -0.757200 * ANG_TO_BOHR, -0.469200 * ANG_TO_BOHR)], "charge": 0, "mult": 1, "is_uhf": False},
    "Lithium": {"atoms": [(3, 0.0, 0.0, 0.0)], "charge": 0, "mult": 2, "is_uhf": True},
    "Carbon":  {"atoms": [(6, 0.0, 0.0, 0.0)], "charge": 0, "mult": 3, "is_uhf": True},
    "Oxygen":  {"atoms": [(8, 0.0, 0.0, 0.0)], "charge": 0, "mult": 3, "is_uhf": True},
    "OH":      {"atoms": [(8, 0.0, 0.0, 0.0), (1, 0.0, 0.0, 0.9697 * ANG_TO_BOHR)], "charge": 0, "mult": 2, "is_uhf": True},
    "O2":      {"atoms": [(8, 0.0, 0.0, -0.6035 * ANG_TO_BOHR), (8, 0.0, 0.0, 0.6035 * ANG_TO_BOHR)], "charge": 0, "mult": 3, "is_uhf": True},
}

ALL_SYSTEM_ORDER = ["Neon", "H2O", "Lithium", "Carbon", "Oxygen", "OH", "O2"]
ERI_MODE_ORDER = ["EXACT", "CHOLESKY", "DF"]

TABLE1_CASES = [
    ("Neon", "cc-pvdz", "cc-pvdz-ri"), ("Neon", "cc-pvtz", "cc-pvtz-ri"), ("Neon", "cc-pvqz", "cc-pvqz-ri"),
    ("H2O", "cc-pvdz", "cc-pvdz-ri"), ("H2O", "cc-pvtz", "cc-pvtz-ri"), ("H2O", "cc-pvqz", "cc-pvqz-ri"),
    ("Lithium", "cc-pvtz", "cc-pvtz-ri"), ("Carbon", "cc-pvtz", "cc-pvtz-ri"),
    ("Oxygen", "cc-pvtz", "cc-pvtz-ri"), ("OH", "cc-pvtz", "cc-pvtz-ri"), ("O2", "cc-pvtz", "cc-pvtz-ri"),
]

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


@contextmanager
def capture_c_stdout():
    """Menangkap stdout C++ ke dalam string buffer tanpa menampilkannya ke layar."""
    flush_all()
    old_fd = os.dup(1)
    r_fd, w_fd = os.pipe()
    buf = []
    try:
        os.dup2(w_fd, 1)
        os.close(w_fd)
        yield buf
    finally:
        flush_all()
        os.dup2(old_fd, 1)
        os.close(old_fd)
        chunks = []
        while True:
            data = os.read(r_fd, 4096)
            if not data:
                break
            chunks.append(data.decode("utf-8", errors="replace"))
        os.close(r_fd)
        buf.append("".join(chunks))


def find_psi4_python():
    for k in ("PSI4_PY", "PSI4_PYTHON"):
        cand = os.environ.get(k, "").strip()
        if cand and os.path.exists(cand):
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
    for z, x, y, zc in info["atoms"]:
        mol.add_atom(z, x, y, zc)
    return mol


def get_occ_counts(sys_name, scf_res=None):
    if scf_res is not None and hasattr(scf_res, "n_occ_alpha") and hasattr(scf_res, "n_occ_beta"):
        return int(scf_res.n_occ_alpha), int(scf_res.n_occ_beta)
    info = SYSTEMS[sys_name]
    n_elec = sum(a[0] for a in info["atoms"]) - info["charge"]
    n_unp = info["mult"] - 1
    return (n_elec + n_unp) // 2, (n_elec - n_unp) // 2


def compute_spin_s2(ints, C_a, C_b, n_occ_a, n_occ_b):
    S_ao = np.array(ints.compute_overlap())
    Ca_occ, Cb_occ = np.array(C_a)[:, :n_occ_a], np.array(C_b)[:, :n_occ_b]
    sz = 0.5 * (n_occ_a - n_occ_b)
    s_ideal = sz * (sz + 1.0)
    if n_occ_b == 0:
        return s_ideal, s_ideal
    return s_ideal, float(s_ideal + n_occ_b - np.sum((Ca_occ.T @ S_ao @ Cb_occ) ** 2))


def make_scf_config(eri_mode, aux_basis_name="", print_level=0):
    cfg = mshqc.SCFConfig()
    cfg.scf_type = "incore"
    cfg.max_iterations = 200
    cfg.energy_threshold = 1e-9
    cfg.density_threshold = 1e-9
    cfg.df_threshold = 1e-9
    cfg.print_level = print_level
    cfg.eri_method = eri_mode.lower()
    cfg.use_df = (eri_mode == "DF")
    if eri_mode == "DF" and aux_basis_name:
        cfg.aux_basis_name = aux_basis_name
    elif eri_mode == "CHOLESKY":
        cfg.cholesky_threshold = 1e-9
    return cfg


def make_mp2_config(eri_mode, aux_basis_name="", print_level=0, max_iterations=50):
    cfg = mshqc.MP2Config()
    cfg.energy_threshold = 1e-9
    cfg.gradient_threshold = 1e-5
    cfg.max_iterations = max_iterations
    cfg.print_level = print_level
    cfg.eri_method = eri_mode.lower()
    cfg.use_df = (eri_mode == "DF")
    if eri_mode == "DF" and aux_basis_name:
        cfg.aux_basis_name = aux_basis_name
    elif eri_mode == "CHOLESKY":
        cfg.cholesky_threshold = 1e-9
    return cfg


def run_scf_raw(mol, basis, ints, eri_mode, is_uhf, aux_basis_name="", print_level=0):
    cfg = make_scf_config(eri_mode, aux_basis_name, print_level)
    if is_uhf:
        na, nb = (mol.n_electrons() + mol.multiplicity() - 1) // 2, (mol.n_electrons() - mol.multiplicity() + 1) // 2
        return mshqc.UHF(mol, basis, ints, na, nb, cfg).compute()
    return mshqc.RHF(mol, basis, ints, cfg).compute()


PSI4_BATCH_WORKER = r"""
import os, sys, time, json, re, ctypes, tempfile
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
psi4.set_num_threads(n_threads)
psi4.set_memory("8 GB")

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

Z_TO_SYM = {1: "H", 3: "Li", 6: "C", 8: "O", 10: "Ne"}

def snapshot_wfn(wfn):
    snap = {"e": float(wfn.energy()), "Ca": wfn.Ca().clone(), "Da": wfn.Da().clone(), "Fa": wfn.Fa().clone(), "ea": wfn.epsilon_a().clone()}
    if wfn.Cb() is not None:
        snap.update({"Cb": wfn.Cb().clone(), "Db": wfn.Db().clone(), "Fb": wfn.Fb().clone(), "eb": wfn.epsilon_b().clone()})
    return snap

def restore_wfn(wfn, snap):
    wfn.set_energy(snap["e"])
    for k, g in (("Ca", wfn.Ca), ("Da", wfn.Da), ("Fa", wfn.Fa), ("ea", wfn.epsilon_a)):
        g().copy(snap[k])
    if "Cb" in snap and wfn.Cb() is not None:
        for k, g in (("Cb", wfn.Cb), ("Db", wfn.Db), ("Fb", wfn.Fb), ("eb", wfn.epsilon_b)):
            g().copy(snap[k])

def calc_s2_wfn(wfn):
    na, nb = wfn.nalpha(), wfn.nbeta()
    sz = 0.5 * (na - nb)
    s_ideal = sz * (sz + 1.0)
    if na == 0 or nb == 0:
        return s_ideal
    Ca = np.asarray(wfn.Ca_subset("AO", "OCC"))
    Cb = np.asarray(wfn.Cb_subset("AO", "OCC"))
    S  = np.asarray(wfn.mintshelper().ao_overlap())
    return float(s_ideal + nb - np.sum((Ca.T @ S @ Cb) ** 2))

def parse_occ_iterations(log_text):
    iters = 0
    for line in log_text.splitlines():
        m = re.match(r"^\s*(\d+)\s+-?\d+\.\d+\s+-?\d+\.\d+", line)
        if m:
            iters = max(iters, int(m.group(1)))
    return iters

payload = json.loads(sys.argv[1])
results = {"runs": {}, "m_chol": {}}
tmp_log = os.path.join(shm_dir, f"psi4_bab4_{os.getpid()}.out")

# 1. Hitung peringkat Cholesky aktual secara dinamis untuk Tabel 1 (tanpa hardcode)
psi4.core.be_quiet()
for c_item in payload.get("chol_cases", []):
    s_name, b_name = c_item["sys"], c_item["basis"]
    lines = [f"{c_item['charge']} {c_item['mult']}", "units bohr", "symmetry c1"]
    lines += [f"{Z_TO_SYM[z]} {x:.12f} {y:.12f} {zc:.12f}" for z, x, y, zc in c_item["atoms"]]
    mol_c = psi4.geometry("\n".join(lines))
    bas_c = psi4.core.BasisSet.build(mol_c, "ORBITAL", b_name)
    mints_c = psi4.core.MintsHelper(bas_c)
    L_mat = mints_c.cd_ints(1e-9)
    results["m_chol"][f"{s_name}|{b_name}"] = int(L_mat.shape[0])
    del L_mat, mints_c, bas_c, mol_c
    psi4.core.clean()

# 2. Jalankan suite referensi utama
for item in payload["tasks"]:
    sys_name, eri_mode = item["sys"], item["mode"].lower()
    basis_name, aux_basis = item.get("basis", "cc-pVTZ"), item.get("aux", "cc-pVTZ-RI")
    chg, mult, is_uhf = item["charge"], item["mult"], item["is_uhf"]

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.be_quiet()

    lines = [f"{chg} {mult}", "units bohr"]
    if eri_mode in ["df", "cholesky"] or sys_name == "OH":
        lines.append("symmetry c1")
    lines += [f"{Z_TO_SYM[z]} {x:.12f} {y:.12f} {zc:.12f}" for z, x, y, zc in item["atoms"]]
    mol = psi4.geometry("\n".join(lines))

    scf_type = "pk" if eri_mode == "exact" else ("cd" if eri_mode == "cholesky" else "df")
    corr_type = "conv" if eri_mode == "exact" else ("cd" if eri_mode == "cholesky" else "df")

    opts = {
        "basis": basis_name,
        "reference": "uhf" if is_uhf else "rhf",
        "e_convergence": 1e-9,
        "d_convergence": 1e-9,
        "scf_type": scf_type,
        "mp2_type": corr_type,
        "mp_type": corr_type,
        "cc_type": corr_type,
        "freeze_core": "false",
        "r_convergence": 1e-6,
        "rms_mograd_convergence": 1e-5,
        "max_mograd_convergence": 1e-4,
        "maxiter": 200,
        "mo_maxiter": 50,
        "cachelevel": 2,
        "read_scf_3index": False,
    }
    if eri_mode == "df":
        opts.update({"df_basis_scf": aux_basis, "df_basis_mp2": aux_basis, "df_basis_cc": aux_basis})
    elif eri_mode == "cholesky":
        opts["cholesky_tolerance"] = 1e-9

    psi4.set_options(opts)
    key = f"{sys_name}|{eri_mode.upper()}"
    entry = {}

    set_blas_threads(1)
    try:
        t0 = time.perf_counter()
        e_scf, scf_wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
        t_scf = time.perf_counter() - t0
        entry["SCF"] = {"e": float(e_scf), "t": float(t_scf), "s2": calc_s2_wfn(scf_wfn)}
    except Exception as ex:
        results["runs"][key] = {"error": str(ex)}
        psi4.core.clean()
        continue

    hf_snap = snapshot_wfn(scf_wfn)
    if eri_mode in ["cholesky", "df"]:
        psi4.set_options({"scf_type": "pk", "read_scf_3index": False})

    # Aktifkan multithreading BLAS penuh untuk seluruh metode pasca-HF (occ & dfocc)
    set_blas_threads(n_threads)
    need_iter_log = is_uhf and eri_mode in ["exact", "df"]
    try:
        for m_key, m_call in (("MP2", "mp2"), ("MP3", "mp3")):
            try:
                t0 = time.perf_counter()
                e_val = psi4.energy(m_call, ref_wfn=scf_wfn, molecule=mol)
                t_val = time.perf_counter() - t0
                entry[m_key] = {"e": float(e_val), "t": float(t_val)}
            except Exception:
                pass
            psi4.core.clean()
            restore_wfn(scf_wfn, hf_snap)

        for m_key, m_call in (("OMP2", "omp2"), ("OMP3", "omp3")):
            try:
                if need_iter_log:
                    psi4.core.set_output_file(tmp_log, False)
                t0 = time.perf_counter()
                e_val, opt_wfn = psi4.energy(m_call, ref_wfn=scf_wfn, molecule=mol, return_wfn=True)
                t_val = time.perf_counter() - t0
                it_cnt = 0
                if need_iter_log:
                    psi4.core.flush_outfile()
                    psi4.core.be_quiet()
                    log_txt = open(tmp_log, "r", errors="replace").read() if os.path.exists(tmp_log) else ""
                    it_cnt = parse_occ_iterations(log_txt)
                entry[m_key] = {"e": float(e_val), "t": float(t_val), "iter": it_cnt, "s2": calc_s2_wfn(opt_wfn)}
            except Exception:
                psi4.core.be_quiet()
            psi4.core.clean()
            if m_key == "OMP2":
                restore_wfn(scf_wfn, hf_snap)
    finally:
        set_blas_threads(1)

    results["runs"][key] = entry

if os.path.exists(tmp_log):
    try:
        os.remove(tmp_log)
    except Exception:
        pass

print(json.dumps(results))
"""


def run_psi4_reference_suite():
    tasks = [
        {"sys": s, "mode": m, "basis": "cc-pVTZ", "aux": "cc-pVTZ-RI", **SYSTEMS[s]}
        for s in ALL_SYSTEM_ORDER for m in ERI_MODE_ORDER
    ]
    chol_cases = [
        {"sys": s, "basis": b, **SYSTEMS[s]}
        for s, b, _ in TABLE1_CASES
    ]
    try:
        out = subprocess.check_output(
            [PSI4_PY_BIN, "-c", PSI4_BATCH_WORKER, json.dumps({"tasks": tasks, "chol_cases": chol_cases})],
            text=True
        )
        data = extract_json_safe(out)
        return data.get("runs", {}), data.get("m_chol", {})
    except Exception as exc:
        print(f"  [!] Peringatan: Subproses Psi4 gagal dijalankan ({exc}).", flush=True)
        return {}, {}


def run_mshqc_reference_suite():
    mshqc_db = {}
    traces = {}
    for sys_name in ALL_SYSTEM_ORDER:
        is_uhf = SYSTEMS[sys_name]["is_uhf"]
        for mode in ERI_MODE_ORDER:
            mol = build_mol(sys_name)
            basis = mshqc.BasisSet("cc-pvtz", mol)
            with suppress_stdout():
                ints = mshqc.IntegralEngine(mol, basis)
                t0 = time.perf_counter()
                scf_res = run_scf_raw(mol, basis, ints, mode, is_uhf, "cc-pvtz-ri", print_level=0)
                t_scf = time.perf_counter() - t0
                e_scf = float(scf_res.energy_total)

                oa, ob = get_occ_counts(sys_name, scf_res)
                s_ideal, s2_uhf = compute_spin_s2(ints, scf_res.C_alpha, scf_res.C_beta, oa, ob)

                cfg = make_mp2_config(mode, "cc-pvtz-ri", print_level=0, max_iterations=50)

                t0 = time.perf_counter()
                res_mp2 = (mshqc.UMP2 if is_uhf else mshqc.RMP2)(mol, basis, ints, scf_res, cfg).compute()
                t_mp2 = time.perf_counter() - t0
                e_mp2 = float(res_mp2.energy_total)

                t0 = time.perf_counter()
                res_mp3 = (mshqc.UMP3 if is_uhf else mshqc.RMP3)(scf_res, res_mp2, cfg, ints).compute()
                t_mp3 = time.perf_counter() - t0
                e_mp3 = float(getattr(res_mp3, "e_total", getattr(res_mp3, "energy_total", 0.0)))

                t0 = time.perf_counter()
                res_omp2 = mshqc.OMP2(mol, basis, ints, scf_res, cfg).compute()
                t_omp2 = time.perf_counter() - t0
                e_omp2 = float(res_omp2.energy_total)
                _, s2_omp2 = compute_spin_s2(ints, res_omp2.C_alpha, res_omp2.C_beta, oa, ob)

                t0 = time.perf_counter()
                res_omp3 = mshqc.OMP3(mol, basis, ints, scf_res, cfg).compute()
                t_omp3 = time.perf_counter() - t0
                e_omp3 = float(getattr(res_omp3, "e_total", getattr(res_omp3, "energy_total", 0.0)))
                _, s2_omp3 = compute_spin_s2(ints, res_omp3.C_alpha, res_omp3.C_beta, oa, ob)

            # Tangkap jejak iterasi TR-SOSCF menggunakan objek scf_res & ints yang sudah ada (tanpa hitung ulang SCF)
            if (sys_name, mode) in (("H2O", "DF"), ("Oxygen", "DF")):
                cfg_tr = make_mp2_config(mode, "cc-pvtz-ri", print_level=1, max_iterations=50)
                with capture_c_stdout() as buf:
                    mshqc.OMP2(mol, basis, ints, scf_res, cfg_tr).compute()
                    mshqc.OMP3(mol, basis, ints, scf_res, cfg_tr).compute()
                traces[(sys_name, mode)] = buf[0]

            mshqc_db[f"{sys_name}|{mode}"] = {
                "s_ideal": s_ideal,
                "SCF":  {"e": e_scf,  "t": t_scf,  "s2": s2_uhf},
                "MP2":  {"e": e_mp2,  "t": t_mp2},
                "MP3":  {"e": e_mp3,  "t": t_mp3},
                "OMP2": {"e": e_omp2, "t": t_omp2, "iter": int(res_omp2.iterations), "s2": s2_omp2},
                "OMP3": {"e": e_omp3, "t": t_omp3, "iter": int(res_omp3.iterations), "s2": s2_omp3},
            }
            del res_mp2, res_mp3, res_omp2, res_omp3, scf_res, ints, basis, mol
            gc.collect()
    return mshqc_db, traces


def run_table_1_eri_compression(chol_db):
    print("\n" + "=" * 112, flush=True)
    print("  1. TABEL KOMPRESI TENSOR ERI AO & VEKTOR CHOLESKY AKTUAL (MSHQC vs PSI4, tau_CD = 1e-9)", flush=True)
    print("=" * 112, flush=True)
    print(f"| {'Sistem':<8} | {'Basis':<8} | {'N_bf':<5} | {'N_aux(DF)':<9} | {'M_chol':<7} | {'Exact (MB)':<11} | {'Chol (MB)':<10} | {'DF (MB)':<9} | {'Exact/Chol':<10} | {'Exact/DF':<9} |", flush=True)
    print(f"|{'-'*10}|{'-'*10}|{'-'*7}|{'-'*11}|{'-'*9}|{'-'*13}|{'-'*12}|{'-'*11}|{'-'*12}|{'-'*11}|", flush=True)

    mb = 1024.0 * 1024.0
    for sys_name, b_name, aux_name in TABLE1_CASES:
        mol = build_mol(sys_name)
        nbf = mshqc.BasisSet(b_name, mol).n_basis_functions()
        naux = mshqc.BasisSet(aux_name, mol).n_basis_functions()
        m_chol = chol_db.get(f"{sys_name}|{b_name}", 0)
        mem_exact = (8.0 * (nbf ** 4)) / mb
        mem_chol  = (8.0 * (nbf ** 2) * m_chol) / mb
        mem_df    = (8.0 * (nbf ** 2) * naux) / mb
        r_chol = mem_exact / mem_chol if m_chol > 0 else 0.0
        r_df   = mem_exact / mem_df
        print(f"| {sys_name:<8} | {b_name:<8} | {nbf:<5} | {naux:<9} | {m_chol:<7} | {mem_exact:<11.3f} | {mem_chol:<10.3f} | {mem_df:<9.3f} | {r_chol:<8.2f} x | {r_df:<7.2f} x |", flush=True)


def run_table_2_o_v4_elimination():
    print("\n" + "=" * 116, flush=True)
    print("  2. TABEL ELIMINASI HAMBATAN MEMORI O(V^4) PADA OMP3: KONVENSIONAL vs MSHQC (DF MODE)", flush=True)
    print("=" * 116, flush=True)
    print(f"| {'Sistem':<7} | {'Basis':<8} | {'(O, V, N_aux)':<15} | {'Konvensional V_vvvv + Gamma_vvvv (MB)':<37} | {'MSHQC B_3c + T2 + Y_ooov (MB)':<30} | {'Reduksi Memori MO':<17} |", flush=True)
    print(f"|{'-'*9}|{'-'*10}|{'-'*17}|{'-'*39}|{'-'*32}|{'-'*19}|", flush=True)

    cases = [c for c in TABLE1_CASES if c[0] not in ("Lithium", "Carbon")]
    mb = 1024.0 * 1024.0
    for sys_name, b_name, aux_name in cases:
        mol = build_mol(sys_name)
        nbf = mshqc.BasisSet(b_name, mol).n_basis_functions()
        naux = mshqc.BasisSet(aux_name, mol).n_basis_functions()
        oa, ob = get_occ_counts(sys_name)
        va, vb = nbf - oa, nbf - ob
        if not SYSTEMS[sys_name]["is_uhf"]:
            mem_conv_v4 = 2.0 * 8.0 * (va ** 4) / mb
            n_pairs = oa * (oa + 1) // 2
            mem_mshqc = (8.0 * ((oa*oa + oa*va + va*va) * naux + 2.0 * oa*oa*va*va + 2.0 * n_pairs*va*va + oa*oa*oa*va)) / mb
        else:
            mem_conv_v4 = 2.0 * 8.0 * ((va ** 4) + (vb ** 4) + (va * va * vb * vb)) / mb
            mem_mshqc = (
                8.0 * (
                    ((oa*oa + oa*va + va*va) + (ob*ob + ob*vb + vb*vb)) * naux
                    + 2.0 * ((oa*oa*va*va) + (ob*ob*vb*vb) + (oa*ob*va*vb))
                    + 2.0 * ((oa*ob*va*vb) + (oa*(oa-1)//2)*va*va)
                    + ((oa*oa*oa*va) + (ob*ob*ob*vb) + (oa*ob*oa*vb) + (oa*ob*ob*va))
                )
            ) / mb
        print(f"| {sys_name:<7} | {b_name:<8} | {f'({oa},{va},{naux})':<15} | {mem_conv_v4:<37.3f} | {mem_mshqc:<30.3f} | {mem_conv_v4 / mem_mshqc:<15.2f} x |", flush=True)


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
            s2_uhf, s2_omp2, s2_omp3 = m_entry["SCF"]["s2"], m_entry["OMP2"]["s2"], m_entry["OMP3"]["s2"]
            it_omp2, it_omp3 = m_entry["OMP2"]["iter"], m_entry["OMP3"]["iter"]

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
                f"{f'{s2_omp3:.6f} / {p_s2_o3:.6f}':<23} | {f'{it_omp3:>2} / {p_it_o3:>2} it':<12} | {red_omp3:>6.2f} % |",
                flush=True
            )


def run_table_4_iteration_trace_and_fd(traces):
    print("\n" + "=" * 108, flush=True)
    print("  4. JEJAK ITERASI PENUH TRUST-REGION SOSCF & VERIFIKASI GRADIEN (cc-pVTZ, EXACT & DF)", flush=True)
    print("=" * 108, flush=True)

    for sys_name, mode in [("H2O", "DF"), ("Oxygen", "DF")]:
        print(f"\n>>> Log Iterasi Lengkap TR-SOSCF: {sys_name} (cc-pVTZ, Mode={mode}) <<<", flush=True)
        sys.stdout.write(traces.get((sys_name, mode), ""))
        flush_all()

    print("\n>>> Uji Verifikasi Finite-Difference (C++ print_level=3) <<<", flush=True)
    for sys_name in ["Neon", "H2O", "Carbon"]:
        print(f"\n[{sys_name}] Menguji FD C++ internal:", flush=True)
        mol = build_mol(sys_name)
        basis = mshqc.BasisSet("cc-pvtz", mol)
        with suppress_stdout():
            ints = mshqc.IntegralEngine(mol, basis)
            scf_res = run_scf_raw(mol, basis, ints, "EXACT", SYSTEMS[sys_name]["is_uhf"])

        cfg_fd = make_mp2_config("EXACT", print_level=3, max_iterations=0)
        flush_all()
        mshqc.OMP2(mol, basis, ints, scf_res, cfg_fd).compute()
        flush_all()
        mshqc.OMP3(mol, basis, ints, scf_res, cfg_fd).compute()
        flush_all()


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

    tot_mshqc_s, tot_psi4_s, max_de_all = 0.0, 0.0, 0.0
    mode_times = {m: {"mshqc": 0.0, "psi4": 0.0} for m in ERI_MODE_ORDER}

    for sys_name in ALL_SYSTEM_ORDER:
        is_uhf = SYSTEMS[sys_name]["is_uhf"]
        method_pairs = [
            ("UHF" if is_uhf else "RHF", "SCF"),
            ("UMP2" if is_uhf else "RMP2", "MP2"),
            ("UMP3" if is_uhf else "RMP3", "MP3"),
            ("OMP2", "OMP2"),
            ("OMP3", "OMP3"),
        ]
        for mode in ERI_MODE_ORDER:
            m_entry = mshqc_db[f"{sys_name}|{mode}"]
            p_entry = psi4_db.get(f"{sys_name}|{mode}", {})
            for m_label, key in method_pairs:
                em, tm = m_entry[key]["e"], m_entry[key]["t"]
                p_info = p_entry.get(key, {})
                ep, tp = p_info.get("e", float("nan")), p_info.get("t", 0.0)
                de_uha = abs(em - ep) * 1e6
                sp = (tp / max(tm, 1e-4)) if tp > 0 else 0.0

                tot_mshqc_s += tm
                tot_psi4_s += tp
                mode_times[mode]["mshqc"] += tm
                mode_times[mode]["psi4"] += tp
                if not np.isnan(de_uha):
                    max_de_all = max(max_de_all, de_uha)

                print(
                    f"| {sys_name:<8} | {mode:<9} | {m_label:<6} | {em:<16.8f} | "
                    f"{ep:<16.8f} | {de_uha:<10.2f} | {tm:<9.3f} | {tp:<9.3f} | {sp:<7.2f} x |",
                    flush=True
                )

    tot_sp = (tot_psi4_s / max(tot_mshqc_s, 1e-4)) if tot_psi4_s > 0 else 0.0
    print(f"|{'-'*10}|{'-'*11}|{'-'*8}|{'-'*18}|{'-'*18}|{'-'*12}|{'-'*11}|{'-'*11}|{'-'*11}|", flush=True)
    for mode in ERI_MODE_ORDER:
        tm_m, tp_m = mode_times[mode]["mshqc"], mode_times[mode]["psi4"]
        sp_m = (tp_m / max(tm_m, 1e-4)) if tp_m > 0 else 0.0
        print(f"| {'SUBTOTAL':<8} | {mode:<9} | {'ALL':<6} | {'-':<16} | {'-':<16} | {'-':<10} | {tm_m:<9.3f} | {tp_m:<9.3f} | {sp_m:<7.2f} x |", flush=True)
    print(f"|{'-'*10}|{'-'*11}|{'-'*8}|{'-'*18}|{'-'*18}|{'-'*12}|{'-'*11}|{'-'*11}|{'-'*11}|", flush=True)
    print(f"| {'TOTAL':<8} | {'ALL MODES':<9} | {'ALL':<6} | {'-':<16} | {'-':<16} | {max_de_all:<10.2f} | {tot_mshqc_s:<9.3f} | {tot_psi4_s:<9.3f} | {tot_sp:<7.2f} x |", flush=True)
    print("=" * 124, flush=True)
    print(
        f"  Total Waktu Solver Tabel 5A (7 Sistem x 3 Mode x 5 Metode = 105 Run):\n"
        f"  -> mshqc = {tot_mshqc_s:.3f} s | Psi4 = {tot_psi4_s:.3f} s | Speedup Kumulatif = {tot_sp:.2f}x",
        flush=True
    )


MSHQC_RAM_WORKER = r"""
import sys, gc, json, os
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
os.environ["TBLIS_NUM_THREADS"] = str(n_threads)
import mshqc

def get_rss_and_hwm_mb():
    rss, hwm = 0.0, 0.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"): rss = int(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"): hwm = int(line.split()[1]) / 1024.0
    return rss, hwm

old_fd = os.dup(1)
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 1)
try:
    mol_w = mshqc.Molecule(0, 1); mol_w.add_atom(2, 0.0, 0.0, 0.0)
    bas_w = mshqc.BasisSet("cc-pvdz", mol_w)
    int_w = mshqc.IntegralEngine(mol_w, bas_w); int_w.compute_overlap()
    del int_w, bas_w, mol_w
    gc.collect()

    base_rss, base_hwm = get_rss_and_hwm_mb()
    payload = json.loads(sys.argv[1])
    mol = mshqc.Molecule(payload["charge"], payload["mult"])
    for z, x, y, zc in payload["atoms"]: mol.add_atom(z, x, y, zc)
    basis = mshqc.BasisSet(payload["basis"], mol)
    ints = mshqc.IntegralEngine(mol, basis)

    mode = payload["mode"]
    sc = mshqc.SCFConfig(); sc.print_level = 0; sc.energy_threshold = 1e-9; sc.density_threshold = 1e-9
    mc = mshqc.MP2Config(); mc.print_level = 0; mc.energy_threshold = 1e-9; mc.gradient_threshold = 1e-5
    sc.eri_method = mc.eri_method = mode.lower()
    sc.use_df = mc.use_df = (mode == "DF")
    if mode == "DF":
        sc.aux_basis_name = mc.aux_basis_name = payload["aux"]
    elif mode == "CHOLESKY":
        sc.cholesky_threshold = mc.cholesky_threshold = 1e-9

    if payload["is_uhf"]:
        na, nb = (mol.n_electrons() + mol.multiplicity() - 1) // 2, (mol.n_electrons() - mol.multiplicity() + 1) // 2
        scf_res = mshqc.UHF(mol, basis, ints, na, nb, sc).compute()
    else:
        scf_res = mshqc.RHF(mol, basis, ints, sc).compute()
    gc.collect()

    (mshqc.OMP2 if payload["method"] == "OMP2" else mshqc.OMP3)(mol, basis, ints, scf_res, mc).compute()
    curr_rss, peak_hwm = get_rss_and_hwm_mb()
    result_str = json.dumps({"base_hwm": base_hwm, "peak_hwm": peak_hwm, "net_ram": max(peak_hwm - base_hwm, curr_rss - base_rss, 0.10)})
finally:
    os.dup2(old_fd, 1); os.close(devnull); os.close(old_fd)
print(result_str)
"""

PSI4_RAM_WORKER = r"""
import sys, gc, json, os, ctypes, tempfile, shutil
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

base_shm = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
scratch_sub = tempfile.mkdtemp(prefix="psi4_ram_", dir=base_shm)
os.environ["PSI_SCRATCH"] = scratch_sub

import psi4

Z_TO_SYM = {1: "H", 3: "Li", 6: "C", 8: "O", 10: "Ne"}

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

def get_rss_and_hwm_mb():
    rss, hwm = 0.0, 0.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"): rss = int(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"): hwm = int(line.split()[1]) / 1024.0
    return rss, hwm

def get_scratch_mb(folder):
    total_b = 0
    if os.path.isdir(folder):
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath):
                try:
                    total_b += os.path.getsize(fpath)
                except Exception:
                    pass
    return total_b / (1024.0 * 1024.0)

try:
    psi4.set_num_threads(n_threads)
    psi4.set_memory("8 GB")
    psi4.core.be_quiet()

    mol_w = psi4.geometry("0 1\nH 0.0 0.0 0.0\nH 0.0 0.0 1.4\nunits bohr\nsymmetry c1")
    bas_w = psi4.core.BasisSet.build(mol_w, "ORBITAL", "cc-pVDZ")
    mints_w = psi4.core.MintsHelper(bas_w); mints_w.ao_overlap()
    del mints_w, bas_w, mol_w
    psi4.core.clean(); gc.collect()

    base_rss, base_hwm = get_rss_and_hwm_mb()
    payload = json.loads(sys.argv[1])
    mode, sys_name = payload["mode"].lower(), payload["sys"]

    lines = [f"{payload['charge']} {payload['mult']}", "units bohr"]
    if mode in ["df", "cholesky"] or sys_name == "OH":
        lines.append("symmetry c1")
    lines += [f"{Z_TO_SYM[z]} {x:.12f} {y:.12f} {zc:.12f}" for z, x, y, zc in payload["atoms"]]
    mol = psi4.geometry("\n".join(lines))

    scf_t = "pk" if mode == "exact" else ("cd" if mode == "cholesky" else "df")
    corr_t = "conv" if mode == "exact" else ("cd" if mode == "cholesky" else "df")

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
        "r_convergence": 1e-6,
        "rms_mograd_convergence": 1e-5,
        "max_mograd_convergence": 1e-4,
        "cachelevel": 2,
        "read_scf_3index": False,
    }
    if mode == "df":
        opts.update({"df_basis_scf": payload["aux"], "df_basis_mp2": payload["aux"], "df_basis_cc": payload["aux"]})
    elif mode == "cholesky":
        opts["cholesky_tolerance"] = 1e-9

    psi4.set_options(opts)
    set_blas_threads(1)
    _, scf_wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
    if mode in ["cholesky", "df"]:
        psi4.set_options({"scf_type": "pk", "read_scf_3index": False})

    set_blas_threads(n_threads)
    psi4.energy(payload["method"].lower(), ref_wfn=scf_wfn, molecule=mol)
    set_blas_threads(1)

    curr_rss, peak_hwm = get_rss_and_hwm_mb()
    scratch_ram = get_scratch_mb(scratch_sub)
    psi4.core.clean()

    total_peak = peak_hwm + scratch_ram
    net_working = max(peak_hwm - base_hwm, curr_rss - base_rss, 0.10) + scratch_ram
    print(json.dumps({"base_hwm": base_hwm, "peak_hwm": total_peak, "net_ram": net_working}))
finally:
    shutil.rmtree(scratch_sub, ignore_errors=True)
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
        payload = json.dumps({"sys": sys_name, "basis": b_name, "aux": aux_name, "mode": mode, "method": method, **SYSTEMS[sys_name]})
        try:
            res_m = extract_json_safe(subprocess.check_output([sys.executable, "-c", MSHQC_RAM_WORKER, payload], text=True))
        except Exception:
            res_m = {"peak_hwm": 0.0, "net_ram": 0.0}
        try:
            res_p = extract_json_safe(subprocess.check_output([PSI4_PY_BIN, "-c", PSI4_RAM_WORKER, payload], text=True))
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
    t_bab4_start = time.perf_counter()
    print(f"Menggunakan Python Psi4: {PSI4_PY_BIN}", flush=True)
    psi4_ref_db, chol_db = run_psi4_reference_suite()
    mshqc_ref_db, tr_traces = run_mshqc_reference_suite()
    run_table_1_eri_compression(chol_db)
    run_table_2_o_v4_elimination()
    run_table_3_spin_contamination(mshqc_ref_db, psi4_ref_db)
    run_table_4_iteration_trace_and_fd(tr_traces)
    run_table_5_head_to_head_and_relaxation(mshqc_ref_db, psi4_ref_db)
    run_table_6_calibrated_ram()

    total_mshqc_solver = sum(mshqc_ref_db[k][m]["t"] for k in mshqc_ref_db for m in ("SCF", "MP2", "MP3", "OMP2", "OMP3"))
    total_psi4_solver = sum(psi4_ref_db.get(k, {}).get(m, {}).get("t", 0.0) for k in mshqc_ref_db for m in ("SCF", "MP2", "MP3", "OMP2", "OMP3"))
    cum_speedup = (total_psi4_solver / max(total_mshqc_solver, 1e-4)) if total_psi4_solver > 0 else 0.0
    total_wall_bab4 = time.perf_counter() - t_bab4_start

    print("\n" + "=" * 126, flush=True)
    print("  RINGKASAN AKHIR BENCHMARK BAB IV (MSHQC vs PSI4 | cc-pVTZ / cc-pVTZ-RI)", flush=True)
    print("=" * 126, flush=True)
    print(f"  Total Waktu Solver (105 Run) : mshqc = {total_mshqc_solver:.3f} s | Psi4 = {total_psi4_solver:.3f} s | Speedup Kumulatif = {cum_speedup:.2f}x", flush=True)
    print(f"  Total Waktu Wall-Clock Suite : {total_wall_bab4:.2f} s", flush=True)
    print("=" * 126, flush=True)