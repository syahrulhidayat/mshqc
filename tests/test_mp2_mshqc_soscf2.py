import os
import multiprocessing

# ==============================================================================
# INJEKSI THREAD OPENMP (WAJIB PALING ATAS)
# ==============================================================================
N_CORES = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TBLIS_NUM_THREADS"] = "1" # Kunci TBLIS agar tidak tabrakan dengan OMP MSHQC

import mshqc
import time

# ==============================================================================
# KONFIGURASI GLOBAL MSHQC
# ==============================================================================
BASIS_NAME = "cc-pVTZ"
AUX_BASIS  = "cc-pVTZ-RI"
THRESHOLD  = 1e-9
ANG_TO_BOHR = 1.88972612462577 

# --- SAKLAR UTAMA ---
ERI_METHOD = "df" 
SCF_MODE = "direct"
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

def run_mshqc_bench(sys_name, atoms, chg, mult, ref, basis_name, mp2_algo):
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
        print(f"Init Error pada {sys_name}: {e}")
        return None, 0.0

    # --- 1. SETUP SCF CONFIG ---
    scf_conf = mshqc.SCFConfig()
    scf_conf.energy_threshold = THRESHOLD
    scf_conf.density_threshold = THRESHOLD
    scf_conf.print_level = 0
    scf_conf.scf_type = SCF_MODE
    scf_conf.eri_method = ERI_METHOD
    
    if ERI_METHOD == "df":
        scf_conf.use_df = True
        scf_conf.aux_basis_name = AUX_BASIS
        scf_conf.df_threshold = 1e-9
    elif ERI_METHOD == "cholesky":
        scf_conf.cholesky_threshold = 1e-9

    n_elec = aligned_mol.n_electrons() - chg
    n_a = (n_elec + mult - 1) // 2
    n_b = n_elec - n_a

    if ref == 'rhf': scf_solver = mshqc.RHF(aligned_mol, basis, ints, pg, pl, scf_conf)
    elif ref == 'uhf': scf_solver = mshqc.UHF(aligned_mol, basis, ints, pg, pl, n_a, n_b, scf_conf)
    else: scf_solver = mshqc.ROHF(aligned_mol, basis, ints, pg, pl, n_a, n_b, scf_conf)
    
    scf_res = scf_solver.compute()
    if not scf_res.converged:
        print(f"SCF gagal konvergen pada {sys_name}.")
        return None, 0.0

    # --- 2. SETUP MP2 CONFIG ---
    mp2_conf = mshqc.MP2Config()
    mp2_conf.scf_type = SCF_MODE
    mp2_conf.eri_method = ERI_METHOD
    mp2_conf.use_df = (ERI_METHOD == "df")
    mp2_conf.aux_basis_name = AUX_BASIS
    mp2_conf.print_level = 0
    mp2_conf.energy_threshold = THRESHOLD
    mp2_conf.opt_method = "soscf" 
    mp2_conf.gradient_threshold = THRESHOLD
    

    t0 = time.time()
    try:
        if mp2_algo == 'rmp2':
            mp2_solver = mshqc.RMP2(aligned_mol, basis, ints, scf_res, mp2_conf, pg, pl)
        elif mp2_algo == 'ump2':
            mp2_solver = mshqc.UMP2(aligned_mol, basis, ints, scf_res, mp2_conf, pg, pl)
        elif mp2_algo == 'omp2':
            mp2_solver = mshqc.OMP2(aligned_mol, basis, ints, scf_res, mp2_conf, pg, pl)
        
        mp2_res = mp2_solver.compute()
        t_mp2 = time.time() - t0
        return mp2_res.energy_total, t_mp2
    except Exception as e:
        print(f"Error MSHQC MP2 pada {sys_name} ({mp2_algo}): {e}")
        return None, 0.0

def main():
    print("=========================================================================================")
    print(f"  MSHQC ONLY | PERFORMANCE PROFILING TEST")
    print(f"  Basis: {BASIS_NAME} | Integral: {ERI_METHOD.upper()} | SCF Mode: {SCF_MODE.upper()}")
    print("=========================================================================================")

    for sys_name, atoms, chg, mult, ref in SYSTEMS:
        metode_aktif = ['rmp2', 'omp2'] if mult == 1 else ['ump2', 'omp2']
            
        for algo in metode_aktif:
            print(f"Mengeksekusi: {sys_name:<8} -> {algo.upper()}")
            
            m_e, m_t = run_mshqc_bench(sys_name, atoms, chg, mult, ref, BASIS_NAME, algo)
            
            if m_e is not None:
                RESULTS_DB.append({
                    "sys": sys_name, "method": algo.upper(),
                    "mshqc_e": m_e, "mshqc_t": m_t
                })
        time.sleep(1)

    print("\n============================================================")
    print(f"                 HASIL AKHIR MSHQC BENCHMARK")
    print("============================================================")
    header = f"| {'System':<8} | {'Method':<8} | {'Energy (Ha)':<16} | {'Time (s)':<8} |"
    print("-" * len(header)); print(header); print("-" * len(header))

    for row in RESULTS_DB:
        sys = row['sys']; met = row['method']
        m_e = row['mshqc_e']; m_t = row['mshqc_t']
        print(f"| {sys:<8} | {met:<8} | {m_e:<16.8f} | {m_t:<8.3f} |")
    print("-" * len(header))

if __name__ == "__main__":
    main()