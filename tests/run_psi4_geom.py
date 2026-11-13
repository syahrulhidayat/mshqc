#!/usr/bin/env python3
"""
Runner Mandiri Psi4 untuk Benchmark Optimisasi Geometri & Vibrasi Multi-Sistem
Output: psi4_geom_raw.json, psi4_geom_test.txt
"""

import os
import sys
import time
import json
import logging
import tempfile
import contextlib
from pathlib import Path
import numpy as np

CPU_AVAIL = os.cpu_count() or 4
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", str(min(4, CPU_AVAIL))))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

SHM_DIR = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
os.environ["PSI_SCRATCH"] = SHM_DIR

import psi4

psi4.set_num_threads(N_THREADS)
psi4.set_memory("4 GB")
psi4.core.set_output_file("NUL" if os.name == "nt" else "/dev/null", False)

BOHR_TO_ANG = 0.529177210903

GEOM_SYSTEMS = {
    "H2": {
        "charge": 0, "mult": 1, "ref": "rhf", "basis": "cc-pVTZ",
        "atoms": [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.80)],
        "is_triatomic": False,
        "methods": ["rhf", "rmp2", "rmp3", "omp2", "omp3"],
    },
    "HF": {
        "charge": 0, "mult": 1, "ref": "rhf", "basis": "cc-pVDZ",
        "atoms": [("F", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.97)],
        "is_triatomic": False,
        "methods": ["rhf", "rmp2", "rmp3", "omp2", "omp3"],
    },
    "OH": {
        "charge": 0, "mult": 2, "ref": "uhf", "basis": "cc-pVDZ",
        "atoms": [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.03)],
        "is_triatomic": False,
        "methods": ["uhf", "ump2", "ump3", "omp2", "omp3"],
    },
    "NH": {
        "charge": 0, "mult": 3, "ref": "uhf", "basis": "cc-pVDZ",
        "atoms": [("N", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.08)],
        "is_triatomic": False,
        "methods": ["uhf", "ump2", "ump3", "omp2", "omp3"],
    },
    "H2O": {
        "charge": 0, "mult": 1, "ref": "rhf", "basis": "cc-pVDZ",
        "atoms": [
            ("O", 0.0,  0.0000,  0.1250),
            ("H", 0.0,  0.7800, -0.4900),
            ("H", 0.0, -0.7800, -0.4900),
        ],
        "is_triatomic": True,
        "methods": ["rhf", "rmp2", "omp2", "omp3"],
    },
}


@contextlib.contextmanager
def suppress_all_output():
    logging.disable(logging.CRITICAL)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)
        logging.disable(logging.NOTSET)


def compute_angle_deg(coords_ang: np.ndarray) -> float:
    if coords_ang.shape[0] < 3:
        return 0.0
    v1 = coords_ang[1] - coords_ang[0]
    v2 = coords_ang[2] - coords_ang[0]
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def map_psi4_call(m: str):
    m_low = m.lower()
    if m_low in ("rhf", "uhf"):
        return "scf", False, None
    if m_low in ("rmp2", "ump2"):
        return "mp2", False, None
    if m_low in ("rmp3", "ump3"):
        return "mp3", False, "energy"
    if m_low == "omp2":
        return "omp2", True, None
    if m_low == "omp3":
        return "omp3", True, "energy"
    return m_low, False, "energy"


def main():
    db = {}
    lines = []

    hdr = (
        f"| {'System':<6} | {'Basis':<7} | {'Method':<8} | {'R_eq (A)':<10} | {'Angle (deg)':<11} | "
        f"{'E_opt (Ha)':<16} | {'Time (s)':<8} |"
    )
    sep = "-" * len(hdr)
    print("\n" + "=" * len(hdr), flush=True)
    print(" PSI4 REFERENCE GEOMETRY OPTIMIZATION BENCHMARK SUITE (DF MODE)", flush=True)
    print("=" * len(hdr), flush=True)
    print(sep, flush=True)
    print(hdr, flush=True)
    print(sep, flush=True)
    lines.extend([sep, hdr, sep])

    for sys_name, spec in GEOM_SYSTEMS.items():
        basis = spec["basis"]
        aux_basis = f"{basis}-RI"

        for m in spec["methods"]:
            p4_name, is_oo, dertype = map_psi4_call(m)
            geom_lines = [f"{spec['charge']} {spec['mult']}", "units angstrom", "symmetry c1"]
            for sym, x, y, z in spec["atoms"]:
                geom_lines.append(f"{sym} {x:.8f} {y:.8f} {z:.8f}")
            geom_str = "\n".join(geom_lines)

            with suppress_all_output():
                psi4.core.clean()
                psi4.core.clean_options()
                psi4.core.set_output_file("NUL" if os.name == "nt" else "/dev/null", False)

                p4_mol = psi4.geometry(geom_str)
                opts = {
                    "basis": basis,
                    "df_basis_scf": aux_basis,
                    "df_basis_mp2": aux_basis,
                    "df_basis_cc": aux_basis,
                    "reference": spec["ref"],
                    "scf_type": "df",
                    "mp2_type": "df",
                    "mp_type": "df",
                    "cc_type": "df",
                    "freeze_core": "false",
                    "orb_opt": "true" if is_oo else "false",
                    "hess_type": "approx_diag",
                    "e_convergence": 1e-10,
                    "d_convergence": 1e-10,
                    "r_convergence": 1e-9,
                    "rms_mograd_convergence": 1e-8,
                    "max_mograd_convergence": 1e-7,
                    "g_convergence": "gau_verytight",
                }
                psi4.set_options(opts)

                t0 = time.perf_counter()
                if dertype is not None:
                    e_opt = float(psi4.optimize(p4_name, molecule=p4_mol, dertype=dertype))
                else:
                    e_opt = float(psi4.optimize(p4_name, molecule=p4_mol))
                t_opt = time.perf_counter() - t0

                coords_ang = np.asarray(p4_mol.geometry()) * BOHR_TO_ANG
                r_eq = float(np.linalg.norm(coords_ang[1] - coords_ang[0]))
                angle_eq = compute_angle_deg(coords_ang) if spec["is_triatomic"] else 0.0

            key = f"{sys_name}|{m.upper()}"
            db[key] = {
                "system": sys_name,
                "basis": basis,
                "method": m.upper(),
                "r_eq": r_eq,
                "angle_eq": angle_eq,
                "e_opt": e_opt,
                "time_opt_s": t_opt,
            }

            ang_str = f"{angle_eq:.4f}" if spec["is_triatomic"] else "   -   "
            row_str = (
                f"| {sys_name:<6} | {basis:<7} | {m.upper():<8} | {r_eq:<10.6f} | {ang_str:<11} | "
                f"{e_opt:<16.10f} | {t_opt:<8.3f} |"
            )
            print(row_str, flush=True)
            lines.append(row_str)

    print(sep + "\n", flush=True)
    lines.append(sep)

    Path("psi4_geom_raw.json").write_text(json.dumps(db, indent=2), encoding="utf-8")
    Path("psi4_geom_test.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
