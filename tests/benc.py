import os
import time

# =====================================================================
# 1. KONFIGURASI HPC & LINGKUNGAN (Setara 6 Thread dengan MSHQC)
# =====================================================================
os.environ["OMP_NUM_THREADS"] = "6"
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
    """Membangun objek molekul Psi4 dalam satuan Bohr."""
    lines = [f"{chg} {mult}", "units bohr"]
    if eri_mode in ["df", "cholesky"]:
        # Wajib di Psi4: modul DFOCC (CD/DF untuk MP2, OMP2, OMP3) hanya mendukung simetri C1
        lines.append("symmetry c1")
    for z, x, y, z_c in atoms:
        lines.append(f"{Z_TO_SYM[z]} {x:.12f} {y:.12f} {z_c:.12f}")
    return psi4.geometry("\n".join(lines))

def snapshot_wfn(wfn):
    """Menyimpan salinan matriks orbital & Fock HF sebelum dimodifikasi in-place oleh OMP2."""
    snap = {
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
    """Mengembalikan matriks orbital & Fock HF murni untuk perhitungan berikutnya."""
    wfn.Ca().copy(snap["Ca"])
    wfn.Da().copy(snap["Da"])
    wfn.Fa().copy(snap["Fa"])
    wfn.epsilon_a().copy(snap["ea"])
    if "Cb" in snap and wfn.Cb() is not None:
        wfn.Cb().copy(snap["Cb"])
        wfn.Db().copy(snap["Db"])
        wfn.Fb().copy(snap["Fb"])
        wfn.epsilon_b().copy(snap["eb"])

def run_psi4_bench(sys_name, atoms, chg, mult, ref, basis_name, eri_mode):
    psi4.core.clean()
    psi4.core.clean_options()
    
    mol = build_psi4_mol(atoms, chg, mult, eri_mode)

    # Mapping tipe integral antara MSHQC dan Psi4
    if eri_mode == "exact":
        scf_type  = "pk"     # In-core exact integrals di Psi4
        corr_type = "conv"   # Conventional 4-index ERI (modul OCC)
    elif eri_mode == "cholesky":
        scf_type  = "cd"     # Cholesky Decomposition
        corr_type = "cd"     # Cholesky ERI (modul DFOCC dengan DO_CD=True)
    elif eri_mode == "df":
        scf_type  = "df"     # Density Fitting (RI)
        corr_type = "df"     # DF ERI (modul DFMP2 untuk MP2, DFOCC untuk OMP2/OMP3)

    opts = {
        "basis": basis_name,
        "reference": ref,
        "e_convergence": THRESHOLD,
        "d_convergence": THRESHOLD,
        "scf_type": scf_type,
        "mp2_type": corr_type,          # Mengontrol ERI (conv/cd/df) untuk MP2 & OMP2
        "mp_type": corr_type,           # Mengontrol ERI (conv/cd/df) untuk MP3 & OMP3
        "cc_type": corr_type,
        "freeze_core": "false",         # WAJIB: Korelasikan semua elektron
        "r_convergence": 1e-5,
        "rms_mograd_convergence": 1e-5,
        "max_mograd_convergence": 1e-4,
        "read_scf_3index": False,
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

    hf_snap = snapshot_wfn(scf_wfn)

    # PENTING: Setelah scf_wfn didapat, ubah scf_type ke 'pk' agar run_dfocc()
    # di proc.py tidak memaksa READ_SCF_3INDEX = True (mencari Unit 97 yang sudah dihapus).
    # mp2_type dan mp_type tetap 'cd' atau 'df' sehingga DFOCC tetap menjalankan CD/DF-OMP2 & OMP3!
    if eri_mode in ["cholesky", "df"]:
        psi4.set_options({
            "scf_type": "pk",
            "read_scf_3index": False,
        })

    # --- 2. EKSEKUSI MP2 (RMP2 / UMP2) ---
    mp2_label = "RMP2" if ref == 'rhf' else "UMP2"
    t0_mp2 = time.perf_counter()
    try:
        e_mp2 = psi4.energy("mp2", ref_wfn=scf_wfn, molecule=mol)
        t_mp2 = time.perf_counter() - t0_mp2
    except Exception as e:
        print(f"  [X] Error pada {mp2_label} di {sys_name} ({eri_mode}): {e}")
        e_mp2, t_mp2 = None, 0.0

    psi4.core.clean()
    restore_wfn(scf_wfn, hf_snap)

    # --- 3. EKSEKUSI OMP2 ---
    t0_omp2 = time.perf_counter()
    try:
        e_omp2 = psi4.energy("omp2", ref_wfn=scf_wfn, molecule=mol)
        t_omp2 = time.perf_counter() - t0_omp2
    except Exception as e:
        print(f"  [X] Error pada OMP2 di {sys_name} ({eri_mode}): {e}")
        e_omp2, t_omp2 = None, 0.0

    psi4.core.clean()
    restore_wfn(scf_wfn, hf_snap)

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
    print(f"  Basis: {BASIS_NAME} | Aux Basis: {AUX_BASIS} | Chol Tol: {CHOLESKY_THRESH} | Freeze Core: False")
    print("==========================================================================================================")

    for sys_name, atoms, chg, mult, ref in SYSTEMS:
        ref_label = ref.upper()
        mp2_lbl = "RMP2" if ref == 'rhf' else "UMP2"
        
        for eri_mode in ERI_MODES:
            print(f"Menjalankan: {sys_name:<8} -> {ref_label}, {mp2_lbl}, OMP2, & OMP3 (Integral: {eri_mode.upper()})")
            
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
    print(f"                        HASIL AKHIR BENCHMARK PSI4")
    print("===============================================================================")
    header = f"| {'Sistem':<8} | {'Mode ERI':<10} | {'Metode':<8} | {'Energi (Ha)':<16} | {'Waktu (s)':<8} |"
    print("-" * len(header)); print(header); print("-" * len(header))

    for row in RESULTS_DB:
        sys = row['sys']; eri = row['eri']; met = row['method']
        p_e = row['psi4_e']; p_t = row['psi4_t']
        print(f"| {sys:<8} | {eri:<10} | {met:<8} | {p_e:<16.8f} | {p_t:<8.3f} |")
    print("-" * len(header))

if __name__ == "__main__":
    main()
