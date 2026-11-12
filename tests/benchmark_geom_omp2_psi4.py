#!/usr/bin/env python3
import time
import numpy as np
import mshqc

def main():
    print("\n>>> [1] Menguji Optimisasi Geometri Radikal Hidroksil (.OH, Doublet, UHF-OMP2 & OMP3)")
    for method in ("omp2", "omp3"):
        mol = mshqc.Molecule(charge=0, multiplicity=2)
        mol.add_atom("O", 0.0, 0.0, 0.0, units="angstrom")
        mol.add_atom("H", 0.0, 0.0, 1.03, units="angstrom")  # Tebakan awal renggang (1.03 A)

        opt_res = mol.optimize_geometry(method=method, eri_mode="df", basis="cc-pVDZ")
        r_mshqc = opt_res.bond_lengths_angstrom["O1-H2"]

        vib_res = mol.frequencies(method=method, eri_mode="df", basis="cc-pVDZ")
        print(f"[{method.upper()}] R_eq(O-H) = {r_mshqc:.6f} A | E_opt = {opt_res.final_energy:.10f} Ha | Freq = {vib_res.frequencies_cm1[0]:.2f} cm^-1")

        try:
            import psi4
            psi4.core.be_quiet()
            p4_mol = psi4.geometry('''
            0 2
            O 0.0 0.0 0.0
            H 0.0 0.0 1.03
            symmetry c1
            ''')
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
            e_p4 = psi4.optimize(method, molecule=p4_mol)
            t_p4 = time.perf_counter() - t0
            c_p4 = np.asarray(p4_mol.geometry()) * 0.529177210903
            r_p4 = float(np.linalg.norm(c_p4[1] - c_p4[0]))
            print(f"  vs Psi4: R_eq = {r_p4:.6f} A (|dR| = {abs(r_mshqc - r_p4)*1e3:.3f} mA) | "
                  f"|dE| = {abs(opt_res.final_energy - e_p4)*1e6:.2f} uHa | "
                  f"mshqc: {opt_res.wall_time_sec:.2f}s vs Psi4: {t_p4:.2f}s")
        except ImportError:
            pass

if __name__ == "__main__":
    main()
