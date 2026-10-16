import os
import time

# =====================================================================
# 1. KONFIGURASI HPC & LINGKUNGAN (Setara dengan MSHQC)
# =====================================================================
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TBLIS_NUM_THREADS"] = "6"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import psi4

# Set thread & memori untuk Psi4 API
psi4.set_num_threads(6)
psi4.set_memory("8 GB")
psi4.core.be_quiet()

# ==============================================================================
# 2. KONFIGURASI GLOBAL PSI4
# ==============================================================================
BASIS_NAME = "cc-pVTZ"
AUX_BASIS  = "cc-pVTZ-RI"
THRESHOLD  = 1e-9
CHOLESKY_THRESH = 1e-9
ANG_TO_BOHR = 1.88972612462577

ERI_MODES = ["exact", "cholesky", "df"]

Z_TO_SYM = {1: "H", 3: "Li", 6: "C", 8: "O", 10: "Ne"}

SYSTEMS = [
    ("Neon",     [(10, 0.0, 0.0, 0.0)], 0, 1, 'rhf'),
    ("H2O",      [(8, 0.0, 0.0, 0.1173 * ANG_TO_BOHR), 
                  (1, 0.0, 0.7572 * ANG_TO_BOHR, -0.4692 * ANG_TO_BOHR), 
                  (1, 0.0, -0.7572 * ANG_TO_BOHR, -0.4692 * ANG_TO_BOHR)], 0, 1, 'rhf'),
    ("Lithium",  [(3, 0.0, 0.0, 0.0)], 0, 2, 'uhf'),
    ("Carbon",   [(6, 0.0, 0.0, 0.0)], 0, 3, 'uhf'),
    ("Oxygen",   [(8, 0.0, 0.0, 0.0)], 0, 3, 'uhf'),
]

RESULTS_DB = []

def build_psi4_mol(atoms, chg, mult, eri_mode="exact"):
    lines = [f"{chg} {mult}", "units bohr"]
    if eri_mode in ["df", "cholesky"]:
        lines.append("symmetry c1")  # Wajib di Psi4 agar modul DFOCC (DF/CD-OMP2 & OMP3) bisa berjalan!
    for z, x, y, z_c in atoms:
        lines.append(f"{Z_TO_SYM[z]} {x:.12f} {y:.12f} {z_c:.12f}")
    return psi4.geometry("\n".join(lines))

def run_psi4_bench(sys_name, atoms, chg, mult, ref, basis_name, eri_mode):
    psi4.core.clean()
    psi4.core.clean_options()
    
    mol = build_psi4_mol(atoms, chg, mult)

    # Mapping tipe integral antara MSHQC dan Psi4
    if eri_mode == "exact":
        scf_type = "pk"      # In-core exact integrals di Psi4
        mp2_type = "conv"    # Conventional 4-index ERI untuk MP2/OMP2/OMP3
        cc_type  = "conv"
    elif eri_mode == "cholesky":
        scf_type = "cd"      # Cholesky Decomposition
        mp2_type = "cd"
        cc_type  = "cd"
    elif eri_mode == "df":
        scf_type = "df"      # Density Fitting (RI)
        mp2_type = "df"
        cc_type  = "df"

    opts = {
        "basis": basis_name,
        "reference": ref,
        "e_convergence": THRESHOLD,
        "d_convergence": THRESHOLD,
        "scf_type": scf_type,
        "mp2_type": mp2_type,
        "cc_type": cc_type,             # Mengontrol tipe ERI pada modul OCC untuk OMP3
        "freeze_core": "false",         # WAJIB: MSHQC mengkorelasikan semua elektron
        "r_convergence": 1e-5,          # Setara dengan gradient_threshold = 1e-5
        "rms_mograd_convergence": 1e-5, # Threshold konvergensi gradien orbital OMP2 & OMP3
        "max_mograd_convergence": 1e-4,
    }

    if eri_mode == "df":
        opts["df_basis_scf"] = AUX_BASIS
        opts["df_basis_mp2"] = AUX_BASIS
        opts["df_basis_cc"]  = AUX_BASIS
    elif eri_mode == "cholesky":
        opts["cholesky_tolerance"] = CHOLESKY_THRESH

    psi4.set_options(opts)

    # --- 1. EKSEKUSI SCF (RHF / UHF) ---
    t0_scf = time.perf_counter()
    try:
        e_scf, scf_wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
        t_scf = time.perf_counter() - t0_scf
    except Exception as e:
        print(f"  [!] SCF gagal untuk {sys_name} ({eri_mode.upper()}): {e}")
        return None, 0.0, None, None, 0.0, None, 0.0, None, 0.0

    # --- 2. EKSEKUSI MP2 (RMP2 / UMP2) ---
    mp2_label = "RMP2" if ref == 'rhf' else "UMP2"
    t0_mp2 = time.perf_counter()
    try:
        e_mp2 = psi4.energy("mp2", ref_wfn=scf_wfn, molecule=mol)
        t_mp2 = time.perf_counter() - t0_mp2
    except Exception as e:
        print(f"  [X] Error pada {mp2_label} di {sys_name} ({eri_mode}): {e}")
        e_mp2, t_mp2 = None, 0.0

    # --- 3. EKSEKUSI OMP2 ---
    t0_omp2 = time.perf_counter()
    try:
        e_omp2 = psi4.energy("omp2", ref_wfn=scf_wfn, molecule=mol)
        t_omp2 = time.perf_counter() - t0_omp2
    except Exception as e:
        print(f"  [X] Error pada OMP2 di {sys_name} ({eri_mode}): {e}")
        e_omp2, t_omp2 = None, 0.0

    # --- 4. EKSEKUSI OMP3 ---
    t0_omp3 = time.perf_counter()
    try:
        e_omp3 = psi4.energy("omp3", ref_wfn=scf_wfn, molecule=mol)
        t_omp3 = time.perf_counter() - t0_omp3
    except Exception as e:
        print(f"  [X] Error pada OMP3 di {sys_name} ({eri_mode}): {e}")
        e_omp3, t_omp3 = None, 0.0

    psi4.core.clean()
    return e_scf, t_scf, mp2_label, e_mp2, t_mp2, e_omp2, t_omp2, e_omp3, t_omp3

def main():
    print("==========================================================================================================")
    print(f"  PSI4 BENCHMARK (APPLE-TO-APPLE) | EXACT vs CHOLESKY vs DF (HF, MP2, OMP2, OMP3)")
    print(f"  Base: {BASIS_NAME} | Aux Base: {AUX_BASIS} | Chol Tol: {CHOLESKY_THRESH} | Freeze Core: False")
    print("==========================================================================================================")

    for sys_name, atoms, chg, mult, ref in SYSTEMS:
        ref_label = ref.upper()
        mp2_lbl = "RMP2" if ref == 'rhf' else "UMP2"
        
        for eri_mode in ERI_MODES:
            print(f"Executando: {sys_name:<8} -> {ref_label}, {mp2_lbl}, OMP2, & OMP3 (Integral: {eri_mode.upper()})")
            
            e_scf, t_scf, mp2_label, e_mp2, t_mp2, e_omp2, t_omp2, e_omp3, t_omp3 = run_psi4_bench(
                sys_name, atoms, chg, mult, ref, BASIS_NAME, eri_mode
            )
            
            if e_scf is not None:
                RESULTS_DB.append({
                    "sys": sys_name, "eri": eri_mode.upper(), "method": ref_label,
                    "psi4_e": e_scf, "psi4_t": t_scf
                })
            if e_mp2 is not None:
                RESULTS_DB.append({
                    "sys": sys_name, "eri": eri_mode.upper(), "method": mp2_label,
                    "psi4_e": e_mp2, "psi4_t": t_mp2
                })
            if e_omp2 is not None:
                RESULTS_DB.append({
                    "sys": sys_name, "eri": eri_mode.upper(), "method": "OMP2",
                    "psi4_e": e_omp2, "psi4_t": t_omp2
                })
            if e_omp3 is not None:
                RESULTS_DB.append({
                    "sys": sys_name, "eri": eri_mode.upper(), "method": "OMP3",
                    "psi4_e": e_omp3, "psi4_t": t_omp3
                })
                
            time.sleep(0.5)

    print("\n===============================================================================")
    print(f"                      RESULTADO FINAL DO BENCHMARK PSI4")
    print("===============================================================================")
    header = f"| {'Sistema':<8} | {'ERI Mode':<10} | {'Método':<8} | {'Energia (Ha)':<16} | {'Tempo (s)':<8} |"
    print("-" * len(header)); print(header); print("-" * len(header))

    for row in RESULTS_DB:
        sys = row['sys']; eri = row['eri']; met = row['method']
        p_e = row['psi4_e']; p_t = row['psi4_t']
        print(f"| {sys:<8} | {eri:<10} | {met:<8} | {p_e:<16.8f} | {p_t:<8.3f} |")
    print("-" * len(header))

if __name__ == "__main__":
    main()