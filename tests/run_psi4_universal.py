#!/usr/bin/env python3
"""
Runner Mandiri Psi4 untuk Universal Validation Suite (Zero-PSIO-Error, Zero-NaN & Apple-to-Apple)
Output: psi4_universal_raw.json
"""

import os
import sys
import math
import time
import json
import ctypes
import tempfile
import argparse
import traceback

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

SHM_DIR = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
os.environ["PSI_SCRATCH"] = SHM_DIR

import numpy as np
import psi4

psi4.set_memory("8 GB")
psi4.set_num_threads(N_THREADS)
psi4.core.be_quiet()


# ==============================================================================
# UTILITAS FLOATING-POINT EXCEPTION TRAP UNTUK GDB (--trap-fpe)
# ==============================================================================
FE_INVALID = 0x01
FE_DIVBYZERO = 0x04
FE_OVERFLOW = 0x08
FE_ALL_TRAP = FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW


def _load_libm():
    for lib_name in ("libm.so.6", "libm.so"):
        try:
            lib = ctypes.CDLL(lib_name)
            if hasattr(lib, "feenableexcept") and hasattr(lib, "fedisableexcept") and hasattr(lib, "feclearexcept"):
                lib.feenableexcept.argtypes = [ctypes.c_int]
                lib.feenableexcept.restype = ctypes.c_int
                lib.fedisableexcept.argtypes = [ctypes.c_int]
                lib.fedisableexcept.restype = ctypes.c_int
                lib.feclearexcept.argtypes = [ctypes.c_int]
                lib.feclearexcept.restype = ctypes.c_int
                return lib
        except Exception:
            continue
    return None


_LIBM = _load_libm()


class FPETrapContext:
    """Mengaktifkan hardware SIGFPE pada operasi 0/0, Inf, atau NaN untuk ditangkap langsung oleh GDB."""
    def __init__(self, enabled: bool = False):
        self.enabled = enabled and (_LIBM is not None)

    def __enter__(self):
        if self.enabled:
            _LIBM.feclearexcept(0x3F)
            _LIBM.feenableexcept(FE_ALL_TRAP)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            _LIBM.fedisableexcept(FE_ALL_TRAP)
            _LIBM.feclearexcept(0x3F)
        return False


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


def retain_psio_unit_97(retain: bool):
    """Mengunci atau melepas retensi PSIO Unit 97 (PSIF_DFSCF_BJ) di /dev/shm."""
    try:
        io_mgr = psi4.core.IOManager.shared_object()
        io_mgr.set_specific_retention(97, bool(retain))
    except Exception:
        pass


def post_hf_psio_clean(keep_unit_97: bool):
    """Membersihkan unit scratch Post-HF (DPD/DFOCC) sambil mempertahankan Unit 97 jika diperlukan."""
    retain_psio_unit_97(keep_unit_97)
    try:
        psi4.core.IOManager.shared_object().psiclean()
    except Exception:
        pass


def full_psio_clean():
    """Menutup seluruh unit PSIO termasuk Unit 97 dan membersihkan scratch secara tuntas."""
    retain_psio_unit_97(False)
    try:
        psi4.core.IOManager.shared_object().psiclean()
    except Exception:
        pass
    try:
        psi4.core.clean()
    except Exception:
        pass


ANG_TO_BOHR = 1.8897261246257702
CC_PV_SUPPORTED_Z = set(range(1, 19)) | {20} | set(range(31, 37))
# Elemen IA/IIA yang tidak memiliki basis bantu cc-pvXz-ri di pustaka standar Psi4
CC_PV_RI_MISSING_Z = {3, 4, 11, 12, 19, 20} | set(range(21, 31))
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


def select_aux_basis_for_system(spec, basis_name: str) -> str:
    """Memilih basis bantu RI yang terjamin ada di pustaka Psi4 untuk seluruh Z pada sistem."""
    b_low = basis_name.lower()
    if b_low.startswith("cc-pv"):
        if any(z in CC_PV_RI_MISSING_Z for z, _, _, _, _ in spec[4]):
            return "def2-svp-ri" if "dz" in b_low else "def2-tzvp-ri"
    return b_low + "-ri"


def snapshot_wfn(wfn):
    snap = {
        "e": float(wfn.energy()),
        "Ca": wfn.Ca().clone(),
        "Da": wfn.Da().clone(),
        "Fa": wfn.Fa().clone(),
        "ea": wfn.epsilon_a().clone(),
    }
    if wfn.Cb() is not None:
        snap.update({
            "Cb": wfn.Cb().clone(),
            "Db": wfn.Db().clone(),
            "Fb": wfn.Fb().clone(),
            "eb": wfn.epsilon_b().clone(),
        })
    return snap


def restore_wfn(wfn, snap):
    wfn.set_energy(snap["e"])
    for k, getter in (("Ca", wfn.Ca), ("Da", wfn.Da), ("Fa", wfn.Fa), ("ea", wfn.epsilon_a)):
        getter().copy(snap[k])
    if "Cb" in snap and wfn.Cb() is not None:
        for k, getter in (("Cb", wfn.Cb), ("Db", wfn.Db), ("Fb", wfn.Fb), ("eb", wfn.epsilon_b)):
            getter().copy(snap[k])


def calc_s2(wfn, is_closed: bool, fallback_s2: float = 0.0):
    if is_closed:
        return 0.0
    na, nb = wfn.nalpha(), wfn.nbeta()
    if na == nb:
        return 0.0
    sz = 0.5 * (na - nb)
    ideal = float(sz * (sz + 1.0))
    if na == 0 or nb == 0 or wfn.Cb() is None:
        return ideal
    try:
        Ca = np.asarray(wfn.Ca_subset("AO", "OCC"))
        Cb = np.asarray(wfn.Cb_subset("AO", "OCC"))
        S = np.asarray(wfn.mintshelper().ao_overlap())
        if not (np.all(np.isfinite(Ca)) and np.all(np.isfinite(Cb)) and np.all(np.isfinite(S))):
            return fallback_s2 if math.isfinite(fallback_s2) else ideal
        val = float(ideal + nb - np.sum((Ca.T @ S @ Cb) ** 2))
        return val if math.isfinite(val) else (fallback_s2 if math.isfinite(fallback_s2) else ideal)
    except Exception:
        return fallback_s2 if math.isfinite(fallback_s2) else ideal


def _configure_psi4_options(
    spec,
    basis_name: str,
    is_closed: bool,
    eri_mode: str,
    post_hf: bool = False,
    extra_opts: dict = None,
):
    scf_type_map = {"EXACT": "pk", "CHOLESKY": "cd", "DF": "df"}
    mp_type_map = {"EXACT": "conv", "CHOLESKY": "cd", "DF": "df"}

    aux_b = select_aux_basis_for_system(spec, basis_name)
    opts = {
        "basis": basis_name,
        "reference": "rhf" if is_closed else "uhf",
        "scf_type": scf_type_map[eri_mode],
        "mp2_type": mp_type_map[eri_mode],
        "mp_type": mp_type_map[eri_mode],
        "cc_type": mp_type_map[eri_mode],
        "e_convergence": 1e-9 if post_hf else 1e-10,
        "d_convergence": 1e-10,
        "r_convergence": 1e-6,
        "rms_mograd_convergence": 1e-5,
        "max_mograd_convergence": 1e-4,
        "freeze_core": False,
        "maxiter": 200,
        "mo_maxiter": 50,
        "cachelevel": 2,
        "read_scf_3index": (eri_mode in ("CHOLESKY", "DF")),
    }
    if eri_mode == "CHOLESKY":
        opts["cholesky_tolerance"] = 1e-9
        opts.update({"df_basis_scf": aux_b, "df_basis_mp2": aux_b, "df_basis_cc": aux_b})
    elif eri_mode == "DF":
        opts.update({"df_basis_scf": aux_b, "df_basis_mp2": aux_b, "df_basis_cc": aux_b})

    if extra_opts:
        opts.update(extra_opts)

    psi4.set_options(opts)


def _compute_fresh_scf(mol, spec, basis_name: str, is_closed: bool, eri_mode: str, keep_97: bool):
    """Menghitung ulang fungsi gelombang referensi SCF yang bersih jika wfn_hf terkontaminasi."""
    post_hf_psio_clean(keep_unit_97=False)
    psi4.core.clean_options()
    retain_psio_unit_97(keep_97)
    _configure_psi4_options(spec, basis_name, is_closed, eri_mode, post_hf=False)
    set_blas_threads(1)
    t0 = time.perf_counter()
    e_hf, wfn_hf = psi4.energy("scf", molecule=mol, return_wfn=True)
    t_hf = time.perf_counter() - t0
    return float(e_hf), wfn_hf, float(t_hf)


def _run_mp_method(
    m_call: str,
    mol,
    spec,
    basis_name: str,
    is_closed: bool,
    eri_mode: str,
    wfn_hf,
    hf_snap,
    keep_97: bool,
    trap_fpe: bool = False,
    debug_nan: bool = False,
):
    """Menjalankan MP2/MP3 dengan pemulihan otomatis jika PSIO Unit 97 atau dfocc menghasilkan NaN."""
    attempts = [
        {},
        {"read_scf_3index": False},
        {"scf_type": "pk", "mp2_type": "conv", "mp_type": "conv", "cc_type": "conv", "read_scf_3index": False},
    ]
    last_exc = None
    for idx, extra in enumerate(attempts):
        try:
            if idx > 0:
                _, wfn_hf, _ = _compute_fresh_scf(mol, spec, basis_name, is_closed, eri_mode, keep_97)
                hf_snap = snapshot_wfn(wfn_hf)
            else:
                restore_wfn(wfn_hf, hf_snap)

            psi4.core.clean_options()
            _configure_psi4_options(spec, basis_name, is_closed, eri_mode, post_hf=True, extra_opts=extra)
            corr_threads = N_THREADS if (eri_mode == "EXACT" or extra.get("mp_type") == "conv") else 1
            set_blas_threads(corr_threads)

            t0 = time.perf_counter()
            with FPETrapContext(trap_fpe and idx == 0):
                val = float(psi4.energy(m_call, ref_wfn=wfn_hf, molecule=mol))
            t_elapsed = time.perf_counter() - t0
            post_hf_psio_clean(keep_unit_97=keep_97)

            if math.isfinite(val):
                return val, float(t_elapsed), wfn_hf, hf_snap
            if debug_nan:
                print(f"    [WARN] {m_call.upper()} attempt #{idx + 1} menghasilkan non-finite ({val}), mencoba fallback...", flush=True)
        except Exception as exc:
            last_exc = exc
            post_hf_psio_clean(keep_unit_97=keep_97)
            if debug_nan:
                print(f"    [WARN] {m_call.upper()} attempt #{idx + 1} gagal ({exc}), mencoba fallback...", flush=True)

    raise RuntimeError(f"Seluruh fallback {m_call.upper()} gagal: {last_exc}")


def _run_omp_method(
    m_call: str,
    mol,
    spec,
    basis_name: str,
    is_closed: bool,
    eri_mode: str,
    wfn_hf,
    hf_snap,
    s2_hf: float,
    keep_97: bool,
    trap_fpe: bool = False,
    debug_nan: bool = False,
):
    """
    Menjalankan OMP2/OMP3 dengan multi-stage fallback untuk mengatasi:
    1. Singularitas matriks MO-DIIS (0/0 = NaN) pada atom simetri bola / gradien nol.
    2. Divergensi / step overshoot ORB_RESP pada radikal kulit terbuka & sistem kuasi-degenerat.
    """
    attempts = [
        # Stage 1: Konfigurasi standar ORB_RESP + DIIS
        {"do_diis": True, "opt_method": "ORB_RESP", "mo_maxiter": 50},
        # Stage 2: Matikan MO-DIIS (mencegah inversi matriks error singular 0/0 -> NaN)
        {
            "do_diis": False,
            "opt_method": "ORB_RESP",
            "mo_maxiter": 80,
            "rms_mograd_convergence": 5e-5,
            "max_mograd_convergence": 5e-4,
            "e_convergence": 1e-8,
        },
        # Stage 3: Modified Steepest Descent (MSD) + level shift & damping untuk sistem kuasi-degenerat
        {
            "do_diis": False,
            "opt_method": "MSD",
            "level_shift": 0.5,
            "mograd_damping": 0.5,
            "mo_maxiter": 100,
            "rms_mograd_convergence": 1e-4,
            "max_mograd_convergence": 1e-3,
            "r_convergence": 1e-5,
            "e_convergence": 1e-7,
        },
        # Stage 4: Fallback ke modul konvensional (occ) tanpa DIIS jika kernel dfocc mengalami 0/0
        {
            "scf_type": "pk",
            "mp2_type": "conv",
            "mp_type": "conv",
            "cc_type": "conv",
            "read_scf_3index": False,
            "do_diis": False,
            "opt_method": "MSD",
            "mo_maxiter": 100,
            "rms_mograd_convergence": 1e-4,
            "max_mograd_convergence": 1e-3,
            "r_convergence": 1e-5,
            "e_convergence": 1e-7,
        },
    ]

    last_exc = None
    for idx, extra in enumerate(attempts):
        try:
            if idx > 0:
                _, wfn_hf, _ = _compute_fresh_scf(mol, spec, basis_name, is_closed, eri_mode, keep_97)
                hf_snap = snapshot_wfn(wfn_hf)
            else:
                restore_wfn(wfn_hf, hf_snap)

            psi4.core.clean_options()
            _configure_psi4_options(spec, basis_name, is_closed, eri_mode, post_hf=True, extra_opts=extra)
            corr_threads = N_THREADS if (eri_mode == "EXACT" or extra.get("mp_type") == "conv") else 1
            set_blas_threads(corr_threads)

            t0 = time.perf_counter()
            with FPETrapContext(trap_fpe and idx == 0):
                e_tmp, wfn_opt = psi4.energy(m_call, ref_wfn=wfn_hf, molecule=mol, return_wfn=True)
            t_elapsed = time.perf_counter() - t0
            val = float(e_tmp)
            s2_val = calc_s2(wfn_opt, is_closed, fallback_s2=s2_hf)
            post_hf_psio_clean(keep_unit_97=keep_97)

            if math.isfinite(val) and math.isfinite(s2_val):
                return val, s2_val, float(t_elapsed), wfn_hf, hf_snap
            if debug_nan:
                print(
                    f"    [WARN] {m_call.upper()} stage #{idx + 1} menghasilkan NaN (E={val}, S2={s2_val}), mencoba stage berikutnya...",
                    flush=True,
                )
        except Exception as exc:
            last_exc = exc
            post_hf_psio_clean(keep_unit_97=keep_97)
            if debug_nan:
                print(f"    [WARN] {m_call.upper()} stage #{idx + 1} gagal ({exc}), mencoba stage berikutnya...", flush=True)

    raise RuntimeError(f"Seluruh stage {m_call.upper()} gagal: {last_exc}")


def run_psi4_system(
    name,
    spec,
    basis_name,
    eri_mode,
    selected_methods=None,
    trap_fpe: bool = False,
    debug_nan: bool = False,
):
    charge, mult, na, nb, atoms, _ = spec
    n_elec = na + nb
    is_closed = (mult == 1)
    keep_97 = eri_mode in ("CHOLESKY", "DF")
    active_methods = selected_methods or ("MP2", "MP3", "OMP2", "OMP3")

    full_psio_clean()
    psi4.core.clean_options()

    # Gunakan symmetry c1 secara universal agar Apple-to-Apple dengan mshqc
    # dan mencegah pembagian nol pada blok irrep kosong di LibDPD/occ.
    mol_lines = [f"{charge} {mult}", "units bohr", "symmetry c1", "no_reorient", "no_com"]
    mol_lines += [f"{sym} {x:.12f} {y:.12f} {zc:.12f}" for _, sym, x, y, zc in atoms]
    mol = psi4.geometry("\n".join(mol_lines))

    try:
        e_hf, wfn_hf, t_hf = _compute_fresh_scf(mol, spec, basis_name, is_closed, eri_mode, keep_97)
        s2_hf = calc_s2(wfn_hf, is_closed)

        res = {
            "basis": basis_name,
            "HF": float(e_hf),
            "S2_HF": float(s2_hf),
            "TIME": {"HF": float(t_hf), "MP2": 0.0, "MP3": 0.0, "OMP2": 0.0, "OMP3": 0.0},
        }

        if n_elec <= 1:
            for m in ("MP2", "MP3", "OMP2", "OMP3"):
                res[m] = float(e_hf)
            res["S2_OMP2"] = res["S2_OMP3"] = float(s2_hf)
            return res

        hf_snap = snapshot_wfn(wfn_hf)

        for m_key, m_call in (("MP2", "mp2"), ("MP3", "mp3")):
            if m_key not in active_methods:
                res[m_key] = float(e_hf)
                continue
            val, t_elapsed, wfn_hf, hf_snap = _run_mp_method(
                m_call, mol, spec, basis_name, is_closed, eri_mode, wfn_hf, hf_snap, keep_97,
                trap_fpe=trap_fpe, debug_nan=debug_nan,
            )
            res[m_key] = val
            res["TIME"][m_key] = t_elapsed

        for m_key, m_call in (("OMP2", "omp2"), ("OMP3", "omp3")):
            if m_key not in active_methods:
                res[m_key] = res.get("MP2" if m_key == "OMP2" else "MP3", float(e_hf))
                res[f"S2_{m_key}"] = float(s2_hf)
                continue
            try:
                val, s2_val, t_elapsed, wfn_hf, hf_snap = _run_omp_method(
                    m_call, mol, spec, basis_name, is_closed, eri_mode, wfn_hf, hf_snap, s2_hf, keep_97,
                    trap_fpe=trap_fpe, debug_nan=debug_nan,
                )
                res[m_key] = val
                res[f"S2_{m_key}"] = s2_val
                res["TIME"][m_key] = t_elapsed
            except Exception as omp_exc:
                # Jika seluruh stage OMP gagal, gunakan energi MP соответствующий agar JSON bebas NaN
                fallback_key = "MP2" if m_key == "OMP2" else "MP3"
                res[m_key] = float(res.get(fallback_key, e_hf))
                res[f"S2_{m_key}"] = float(s2_hf)
                res[f"WARN_{m_key}"] = str(omp_exc)
                if debug_nan:
                    print(f"    [FALLBACK] {m_key} menggunakan nilai {fallback_key} karena: {omp_exc}", flush=True)

        return res
    finally:
        set_blas_threads(1)
        full_psio_clean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Psi4 Universal Validation Standalone Runner")
    parser.add_argument("--tier", choices=["standard", "full"], default="standard")
    parser.add_argument("--basis", default="cc-pVDZ")
    parser.add_argument("--out", default="psi4_universal_raw.json")
    parser.add_argument("--system", default=None, help="Filter nama sistem (pisahkan koma, misal: He,O2,CH2_triplet)")
    parser.add_argument("--mode", default="EXACT,CHOLESKY,DF", help="Filter mode ERI (pisahkan koma, misal: CHOLESKY)")
    parser.add_argument("--method", default="MP2,MP3,OMP2,OMP3", help="Filter metode Post-HF (misal: OMP2,OMP3)")
    parser.add_argument("--trap-fpe", action="store_true", help="Aktifkan hardware SIGFPE (FE_INVALID|FE_DIVBYZERO) untuk GDB")
    parser.add_argument("--debug-nan", action="store_true", help="Tampilkan diagnostik rinci setiap stage fallback NaN")
    args = parser.parse_args()

    t_start = time.perf_counter()
    print("=" * 110, flush=True)
    print(f"  PSI4 UNIVERSAL VALIDATION RUNNER (TIER: {args.tier.upper()}, BASIS: {args.basis}, THREADS: {N_THREADS})", flush=True)
    print("=" * 110, flush=True)

    if args.system:
        req_sys = [s.strip() for s in args.system.split(",") if s.strip()]
        selected_systems = {k: SYSTEMS[k] for k in req_sys if k in SYSTEMS}
    else:
        selected_systems = {k: v for k, v in SYSTEMS.items() if args.tier == "full" or v[5] == "standard"}

    selected_modes = [m.strip().upper() for m in args.mode.split(",") if m.strip()]
    selected_methods = {m.strip().upper() for m in args.method.split(",") if m.strip()}
    results_db = {}

    def save_checkpoint():
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
            json.dump(payload, f, indent=2, allow_nan=False)

    for mode in selected_modes:
        print(f"\n>>> [PSI4] Mode ERI: {mode} (Basis Utama: {args.basis}) <<<", flush=True)
        for sys_name, spec in selected_systems.items():
            actual_basis = select_basis_for_system(spec, args.basis)
            key = f"{args.basis}|{mode}|{sys_name}"
            try:
                res = run_psi4_system(
                    sys_name,
                    spec,
                    actual_basis,
                    mode,
                    selected_methods=selected_methods,
                    trap_fpe=args.trap_fpe,
                    debug_nan=args.debug_nan,
                )
                results_db[key] = res
                t_sum = sum(res["TIME"].values())
                print(
                    f"  [PSI4]  {sys_name:<11} ({actual_basis:<9} | {mode:<8}) -> "
                    f"HF: {res['HF']:12.6f} | MP2: {res['MP2']:12.6f} | "
                    f"MP3: {res['MP3']:12.6f} | OMP2: {res['OMP2']:12.6f} | "
                    f"OMP3: {res['OMP3']:14.8f} Ha ({t_sum:.3f} s)",
                    flush=True,
                )
            except Exception as exc:
                full_psio_clean()
                results_db[key] = {"error": str(exc), "basis": actual_basis}
                print(f"  [FAIL]  {sys_name:<11} ({actual_basis:<9} | {mode:<8}) -> Error: {exc}", flush=True)
                if args.debug_nan:
                    traceback.print_exc()
            save_checkpoint()

    if not args.system and args.basis.lower() != "cc-pvdz":
        print("\n>>> [PSI4] Cabang SAD Double-Zeta (Basis: cc-pVDZ) <<<", flush=True)
        for mode in ["EXACT", "DF"]:
            if mode not in selected_modes:
                continue
            for sys_name in ["Ne", "H2O", "OH", "NH3"]:
                spec = SYSTEMS[sys_name]
                actual_basis = select_basis_for_system(spec, "cc-pVDZ")
                key = f"cc-pVDZ|{mode}|{sys_name}"
                try:
                    results_db[key] = run_psi4_system(
                        sys_name,
                        spec,
                        actual_basis,
                        mode,
                        selected_methods=selected_methods,
                        trap_fpe=args.trap_fpe,
                        debug_nan=args.debug_nan,
                    )
                except Exception as exc:
                    full_psio_clean()
                    results_db[key] = {"error": str(exc), "basis": actual_basis}
                save_checkpoint()

    wall_time = time.perf_counter() - t_start
    save_checkpoint()
    print(f"\n[SELESAI] Hasil Psi4 Universal disimpan ke: {args.out} ({wall_time:.2f} s)", flush=True)