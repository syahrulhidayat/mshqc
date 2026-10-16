import os
import time
import mshqc

# =====================================================================
# 1. KONFIGURASI HPC & LINGKUNGAN
# =====================================================================
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TBLIS_NUM_THREADS"] = "6" 
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# ==============================================================================
# 2. KONFIGURASI GLOBAL MSHQC
# ==============================================================================
BASIS_NAME = "cc-pVTZ"
AUX_BASIS  = "cc-pVTZ-RI"
THRESHOLD  = 1e-9
CHOLESKY_THRESH = 1e-9
DF_THRESH  = 1e-9
ANG_TO_BOHR = 1.88972612462577 
SCF_MODE = "incore"

ERI_MODES = ["exact", "cholesky", "df"]
# ==============================================================================

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

def run_mshqc_bench(sys_name, atoms, chg, mult, ref, basis_name, eri_mode):
    raw_mol = mshqc.Molecule()
    for z, x, y, z_c in atoms: raw_mol.add_atom(z, x, y, z_c)
    raw_mol.set_charge(chg); raw_mol.set_multiplicity(mult)
    
    pg = mshqc.PointGroup(raw_mol)
    pg.detect()
    aligned_mol = pg.get_aligned_molecule()

    try:
        basis = mshqc.BasisSet(basis_name, aligned_mol)
        ints = mshqc.IntegralEngine(aligned_mol, basis)
        pl = mshqc.PetiteList(basis, pg)    
        pl.build()
    except Exception as e:
        print(f"  [X] Error Inisialisasi di {sys_name}: {e}")
        return None, 0.0, None, None, 0.0, None, 0.0, None, 0.0

    # --- 1. SETUP SCF CONFIG ---
    scf_conf = mshqc.SCFConfig()
    scf_conf.energy_threshold = THRESHOLD
    scf_conf.density_threshold = THRESHOLD
    scf_conf.print_level = 0
    scf_conf.scf_type = SCF_MODE
    
    scf_conf.eri_method = eri_mode
    if eri_mode == "df":
        scf_conf.use_df = True
        scf_conf.aux_basis_name = AUX_BASIS
        scf_conf.df_threshold = DF_THRESH
    else:
        scf_conf.use_df = False
        if eri_mode == "cholesky":
            scf_conf.cholesky_threshold = CHOLESKY_THRESH

    n_elec = aligned_mol.n_electrons() - chg
    n_a = (n_elec + mult - 1) // 2
    n_b = n_elec - n_a

    if ref == 'rhf': scf_solver = mshqc.RHF(aligned_mol, basis, ints, pg, pl, scf_conf)
    elif ref == 'uhf': scf_solver = mshqc.UHF(aligned_mol, basis, ints, pg, pl, n_a, n_b, scf_conf)
    else: scf_solver = mshqc.ROHF(aligned_mol, basis, ints, pg, pl, n_a, n_b, scf_conf)
    
    t0_scf = time.perf_counter()
    scf_res = scf_solver.compute()
    t_scf = time.perf_counter() - t0_scf
    
    if not scf_res.converged:
        print(f"  [!] SCF gagal konvergen untuk {sys_name} ({eri_mode.upper()}).")
        return None, 0.0, None, None, 0.0, None, 0.0, None, 0.0
        
    e_scf = scf_res.energy_total

    # --- 2. SETUP CORRELATION CONFIG (RMP2 / UMP2) ---
    corr_conf = mshqc.MP2Config()
    corr_conf.scf_type = SCF_MODE
    corr_conf.print_level = 0
    corr_conf.energy_threshold = THRESHOLD
    
    corr_conf.eri_method = eri_mode
    if eri_mode == "df":
        corr_conf.use_df = True
        corr_conf.aux_basis_name = AUX_BASIS
        corr_conf.df_threshold = DF_THRESH
    else:
        corr_conf.use_df = False
        if eri_mode == "cholesky":
            corr_conf.cholesky_threshold = CHOLESKY_THRESH
    
    # === Eksekusi RMP2 / UMP2 ===
    t0_mp2 = time.perf_counter()
    mp2_label = "RMP2" if ref == 'rhf' else "UMP2"
    try:
        if ref == 'rhf':
            mp2_solver = mshqc.RMP2(aligned_mol, basis, ints, scf_res, corr_conf, pg, pl)
        else:
            mp2_solver = mshqc.UMP2(aligned_mol, basis, ints, scf_res, corr_conf, pg, pl)
            
        mp2_res = mp2_solver.compute()
        t_mp2 = time.perf_counter() - t0_mp2
        e_mp2 = mp2_res.energy_total
    except Exception as e:
        print(f"  [X] Error pada perhitungan {mp2_label} di {sys_name} ({eri_mode}): {e}")
        e_mp2, t_mp2 = None, 0.0

    # --- 3. SETUP & EKSEKUSI OMP2 ---
    corr_conf.gradient_threshold = 1e-5
    corr_conf.opt_method = "soscf" 
    
    t0_omp2 = time.perf_counter()
    try:
        omp2_solver = mshqc.OMP2(aligned_mol, basis, ints, scf_res, corr_conf, pg, pl)
        omp2_res = omp2_solver.compute()
        t_omp2 = time.perf_counter() - t0_omp2
        e_omp2 = omp2_res.energy_total
    except Exception as e:
        print(f"  [X] Error pada perhitungan OMP2 di {sys_name} ({eri_mode}): {e}")
        e_omp2, t_omp2 = None, 0.0

    # --- 4. SETUP & EKSEKUSI OMP3 ---
    t0_omp3 = time.perf_counter()
    try:
        omp3_solver = mshqc.OMP3(aligned_mol, basis, ints, scf_res, corr_conf, pg, pl)
        omp3_res = omp3_solver.compute()
        t_omp3 = time.perf_counter() - t0_omp3
        e_omp3 = omp3_res.e_total
    except Exception as e:
        print(f"  [X] Error pada perhitungan OMP3 di {sys_name} ({eri_mode}): {e}")
        e_omp3, t_omp3 = None, 0.0

    return e_scf, t_scf, mp2_label, e_mp2, t_mp2, e_omp2, t_omp2, e_omp3, t_omp3


def main():
    print("==========================================================================================================")
    print(f"  MSHQC BENCHMARK | COMPARISON: EXACT vs CHOLESKY vs DF (HF, MP2, OMP2, OMP3)")
    print(f"  Base: {BASIS_NAME} | Aux Base: {AUX_BASIS} | Chol Tol: {CHOLESKY_THRESH} | DF Tol: {DF_THRESH}")
    print("==========================================================================================================")

    for sys_name, atoms, chg, mult, ref in SYSTEMS:
        ref_label = ref.upper()
        mp2_lbl = "RMP2" if ref == 'rhf' else "UMP2"
        
        for eri_mode in ERI_MODES:
            print(f"Executando: {sys_name:<8} -> {ref_label}, {mp2_lbl}, OMP2, & OMP3 (Integral: {eri_mode.upper()})")
            
            e_scf, t_scf, mp2_label, e_mp2, t_mp2, e_omp2, t_omp2, e_omp3, t_omp3 = run_mshqc_bench(
                sys_name, atoms, chg, mult, ref, BASIS_NAME, eri_mode
            )
            
            if e_scf is not None:
                RESULTS_DB.append({
                    "sys": sys_name, "eri": eri_mode.upper(), "method": ref_label,
                    "mshqc_e": e_scf, "mshqc_t": t_scf
                })
            if e_mp2 is not None:
                RESULTS_DB.append({
                    "sys": sys_name, "eri": eri_mode.upper(), "method": mp2_label,
                    "mshqc_e": e_mp2, "mshqc_t": t_mp2
                })
            if e_omp2 is not None:
                RESULTS_DB.append({
                    "sys": sys_name, "eri": eri_mode.upper(), "method": "OMP2",
                    "mshqc_e": e_omp2, "mshqc_t": t_omp2
                })
            if e_omp3 is not None:
                RESULTS_DB.append({
                    "sys": sys_name, "eri": eri_mode.upper(), "method": "OMP3",
                    "mshqc_e": e_omp3, "mshqc_t": t_omp3
                })
                
            time.sleep(0.5)

    print("\n===============================================================================")
    print(f"                      RESULTADO FINAL DO BENCHMARK MSHQC")
    print("===============================================================================")
    header = f"| {'Sistema':<8} | {'ERI Mode':<10} | {'Método':<8} | {'Energia (Ha)':<16} | {'Tempo (s)':<8} |"
    print("-" * len(header)); print(header); print("-" * len(header))

    for row in RESULTS_DB:
        sys = row['sys']; eri = row['eri']; met = row['method']
        m_e = row['mshqc_e']; m_t = row['mshqc_t']
        print(f"| {sys:<8} | {eri:<10} | {met:<8} | {m_e:<16.8f} | {m_t:<8.3f} |")
    print("-" * len(header))

if __name__ == "__main__":
    main()