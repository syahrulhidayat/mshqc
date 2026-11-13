#!/usr/bin/env python3
import os
import sys
import time
import logging
import contextlib
import numpy as np
import mshqc


@contextlib.contextmanager
def suppress_all_output():
    """Membungkam seluruh stdout/stderr level OS dan logger Python (optking/psi4)."""
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


def main():
    try:
        import psi4
        has_psi4 = True
    except ImportError:
        has_psi4 = False

    rows = []

    for method in ("omp2", "omp3"):
        mol = mshqc.Molecule(charge=0, multiplicity=2)
        mol.add_atom("O", 0.0, 0.0, 0.0, units="angstrom")
        mol.add_atom("H", 0.0, 0.0, 1.03, units="angstrom")

        opt_res = mol.optimize_geometry(
            method=method, eri_mode="df", basis="cc-pVDZ", verbose=False
        )
        r_mshqc = opt_res.bond_lengths_angstrom["O1-H2"]
        e_mshqc = opt_res.final_energy
        t_mshqc = opt_res.wall_time_sec

        vib_res = mol.frequencies(
            method=method, eri_mode="df", basis="cc-pVDZ", verbose=False
        )
        freq_mshqc = float(vib_res.frequencies_cm1[0])

        r_p4 = float("nan")
        e_p4 = float("nan")
        t_p4 = float("nan")

        if has_psi4:
            with suppress_all_output():
                psi4.core.clean()
                psi4.core.clean_options()
                psi4.core.set_output_file(os.devnull, False)
                psi4.core.be_quiet()

                p4_mol = psi4.geometry("""
                0 2
                O 0.0 0.0 0.0
                H 0.0 0.0 1.03
                symmetry c1
                """)

                psi4.set_options({
                    "basis": "cc-pVDZ",
                    "reference": "uhf",
                    "scf_type": "df",
                    "orb_opt": "true",
                    "e_convergence": 1e-10,
                    "d_convergence": 1e-9,
                    "g_convergence": "gau_verytight",
                })

                t0 = time.perf_counter()
                if method == "omp3":
                    e_p4 = float(psi4.optimize(method, molecule=p4_mol, dertype="energy"))
                else:
                    e_p4 = float(psi4.optimize(method, molecule=p4_mol))
                t_p4 = time.perf_counter() - t0

                c_p4 = np.asarray(p4_mol.geometry()) * 0.529177210903
                r_p4 = float(np.linalg.norm(c_p4[1] - c_p4[0]))

        rows.append({
            "method": f"UHF-{method.upper()}",
            "r_m": r_mshqc,
            "r_p": r_p4,
            "dr_ma": abs(r_mshqc - r_p4) * 1e3 if has_psi4 else float("nan"),
            "e_m": e_mshqc,
            "e_p": e_p4,
            "de_uha": abs(e_mshqc - e_p4) * 1e6 if has_psi4 else float("nan"),
            "freq": freq_mshqc,
            "t_m": t_mshqc,
            "t_p": t_p4,
            "speedup": (t_p4 / max(t_mshqc, 1e-4)) if (has_psi4 and t_p4 > 0) else float("nan"),
        })

    print("\n" + "=" * 126)
    print(" BENCHMARK OPTIMISASI GEOMETRI & FREKUENSI VIBRASI RADIKAL HIDROKSIL (.OH, 2Pi | cc-pVDZ / DF)")
    print("=" * 126)
    hdr = (
        f"| {'Method':<8} | {'R_eq mshqc (A)':<14} | {'R_eq Psi4 (A)':<13} | {'|dR| (mA)':<9} | "
        f"{'E_opt mshqc (Ha)':<16} | {'E_opt Psi4 (Ha)':<16} | {'|dE| (uHa)':<10} | "
        f"{'Freq (cm^-1)':<12} | {'mshqc (s)':<9} | {'Psi4 (s)':<8} | {'Speedup':<7} |"
    )
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    for r in rows:
        if has_psi4:
            print(
                f"| {r['method']:<8} | {r['r_m']:<14.6f} | {r['r_p']:<13.6f} | {r['dr_ma']:<9.3f} | "
                f"{r['e_m']:<16.10f} | {r['e_p']:<16.10f} | {r['de_uha']:<10.2f} | "
                f"{r['freq']:<12.2f} | {r['t_m']:<9.2f} | {r['t_p']:<8.2f} | {r['speedup']:<6.2f}x |"
            )
        else:
            print(
                f"| {r['method']:<8} | {r['r_m']:<14.6f} | {'N/A':<13} | {'N/A':<9} | "
                f"{r['e_m']:<16.10f} | {'N/A':<16} | {'N/A':<10} | "
                f"{r['freq']:<12.2f} | {r['t_m']:<9.2f} | {'N/A':<8} | {'N/A':<7} |"
            )
    print(sep + "\n")


if __name__ == "__main__":
    main()
