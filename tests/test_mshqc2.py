#!/usr/bin/env python3
import time
import mshqc

def run_benchmark():
    print("==========================================================================================================")
    print("  MSHQC BENCHMARK | COMPARISON: EXACT vs CHOLESKY vs DF (HF, MP2, OMP2, OMP3)")
    print("  Base: cc-pVTZ | Aux Base: cc-pVTZ-RI | Chol Tol: 1e-09 | DF Tol: 1e-09")
    print("==========================================================================================================")

    # Daftar sistem molekul yang akan diuji
    systems = ["Neon", "H2O", "Lithium", "Carbon", "Oxygen"]
    modes = ["exact", "cholesky", "df"]
    
    results = [] # Untuk menyimpan data tabel akhir

    for sys_name in systems:
        for mode in modes:
            # 1. Setup Molekul & Ref (RHF/UHF)
            if sys_name == "Neon":
                mol = mshqc.Molecule(charge=0, multiplicity=1)
                mol.add_atom("Ne", 0.0, 0.0, 0.0, units="angstrom")
                ref_type = "R"
            elif sys_name == "H2O":
                mol = mshqc.Molecule(charge=0, multiplicity=1)
                mol.add_atom("O", 0.0, 0.0, 0.1173, units="angstrom")
                mol.add_atom("H", 0.0, 0.7572, -0.4692, units="angstrom")
                mol.add_atom("H", 0.0, -0.7572, -0.4692, units="angstrom")
                ref_type = "R"
            elif sys_name == "Lithium":
                mol = mshqc.Molecule(charge=0, multiplicity=2)
                mol.add_atom("Li", 0.0, 0.0, 0.0, units="angstrom")
                ref_type = "U"
            elif sys_name == "Carbon":
                mol = mshqc.Molecule(charge=0, multiplicity=3)
                mol.add_atom("C", 0.0, 0.0, 0.0, units="angstrom")
                ref_type = "U"
            elif sys_name == "Oxygen":
                mol = mshqc.Molecule(charge=0, multiplicity=3)
                mol.add_atom("O", 0.0, 0.0, 0.0, units="angstrom")
                ref_type = "U"

            print(f"Executando: {sys_name:<9}-> {ref_type}HF, {ref_type}MP2, OMP2, & OMP3 (Integral: {mode.upper()})")

            # 2. Load Basis Set & Integral
            basis = mshqc.BasisSet("cc-pVTZ", mol)
            integrals = mshqc.IntegralEngine(mol, basis)

            # 3. Eksekusi HF (Catat Waktu)
            scf_conf = mshqc.SCFConfig(eri_method=mode, threshold=1e-9)
            t0 = time.time()
            if ref_type == "R":
                scf_res = mshqc.RHF(mol, basis, integrals, scf_conf).compute()
            else:
                scf_res = mshqc.UHF(mol, basis, integrals, scf_conf).compute()
            t_hf = time.time() - t0
            results.append((sys_name, mode.upper(), f"{ref_type}HF", scf_res.energy_total, t_hf))

            # 4. Eksekusi MP2 (Catat Waktu)
            mp2_conf = mshqc.MP2Config(eri_method=mode, threshold=1e-9, gradient_threshold=1e-5)
            t0 = time.time()
            if ref_type == "R":
                mp2_res = mshqc.RMP2(mol, basis, integrals, scf_res, mp2_conf).compute()
            else:
                mp2_res = mshqc.UMP2(mol, basis, integrals, scf_res, mp2_conf).compute()
            t_mp2 = time.time() - t0
            results.append((sys_name, mode.upper(), f"{ref_type}MP2", mp2_res.energy_total, t_mp2))

            # 5. Eksekusi OMP2 (Catat Waktu)
            t0 = time.time()
            omp2_res = mshqc.OMP2(mol, basis, integrals, scf_res, mp2_conf).compute()
            t_omp2 = time.time() - t0
            results.append((sys_name, mode.upper(), "OMP2", omp2_res.energy_total, t_omp2))

            # 6. Eksekusi OMP3 (Catat Waktu)
            t0 = time.time()
            omp3_res = mshqc.OMP3(mol, basis, integrals, scf_res, mp2_conf).compute()
            t_omp3 = time.time() - t0
            results.append((sys_name, mode.upper(), "OMP3", omp3_res.energy_total, t_omp3))

    # Print Tabel Hasil Akhir
    print("\n===============================================================================")
    print("                     RESULTADO FINAL DO BENCHMARK MSHQC")
    print("===============================================================================")
    print("-" * 67)
    print(f"| {'Sistema':<8} | {'ERI Mode':<10} | {'Método':<8} | {'Energia (Ha)':<16} | {'Tempo (s)'} |")
    print("-" * 67)
    for r in results:
        print(f"| {r[0]:<8} | {r[1]:<10} | {r[2]:<8} | {r[3]:<16.8f} | {r[4]:.3f}     |")
    print("-" * 67)

if __name__ == "__main__":
    run_benchmark()