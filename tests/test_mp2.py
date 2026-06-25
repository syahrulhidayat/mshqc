import os
import multiprocessing

# ==============================================================================
# INJEKSI THREAD OPENMP (WAJIB PALING ATAS)
# ==============================================================================
N_CORES = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = N_CORES
os.environ["OPENBLAS_NUM_THREADS"] = N_CORES
os.environ["MKL_NUM_THREADS"] = N_CORES

import mshqc
import psi4
import time
import gc

# ==============================================================================
# KONFIGURASI GLOBAL
# ==============================================================================
BASIS_NAME = "cc-pVTZ"
AUX_BASIS  = "cc-pVTZ-RI"
THRESHOLD  = 1e-9
ANG_TO_BOHR = 1.88972612462577 

# --- SAKLAR UTAMA (UBAH DI SINI) ---
# Pilihan: 'exact', 'df', 'cholesky'
ERI_METHOD = "df" 
# Pilihan: 'incore', 'direct'
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

ATOM_MAP = {1: 'H', 2: 'He', 3: 'Li', 6: 'C', 8: 'O', 10: 'Ne'}
RESULTS_DB = []

def run_psi4_bench(sys_name, atoms, chg, mult, basis, ref, mp2_algo):
    psi4.core.clean()
    
    # ---> INI KUNCI UNTUK MENGHENINGKAN PSI4 <---
    # Semua "data sampah" akan dialihkan (dibuang) ke file ini,
    # sehingga terminal Anda akan bersih 100%.
    psi4.core.set_output_file('psi4_silent.out', False)
    
    mol_str = f"{chg} {mult}\nunits bohr\n" 
    for z, x, y, z_c in atoms: 
        mol_str += f"{ATOM_MAP.get(z, 'X')} {x:.10f} {y:.10f} {z_c:.10f}\n"
    psi4.geometry(mol_str)
    
    psi4_scf_type = 'df' if ERI_METHOD == 'df' else ('pk' if SCF_MODE == 'incore' else 'direct')
    psi4_mp2_type = 'df' if ERI_METHOD == 'df' else 'conv'

    opts = {
        'basis': basis,
        'reference': ref,
        'scf_type': psi4_scf_type,
        'mp2_type': psi4_mp2_type,
        'e_convergence': THRESHOLD,
        'd_convergence': THRESHOLD,
        'freeze_core': 'False',
        'puream': True,
        'print': 0 # Pastikan opsi print internal Psi4 mati
    }
    
    if ERI_METHOD == 'df':
        opts['df_basis_scf'] = AUX_BASIS
        opts['df_basis_mp2'] = AUX_BASIS
    elif ERI_METHOD == 'cholesky':
        opts['cholesky_tolerance'] = 1e-9
        
    psi4.set_options(opts)

    psi4_target = 'omp2' if mp2_algo == 'omp2' else 'mp2'

    res = {"status": "FAIL", "energy": 0.0, "time": 0.0}
    try:
        t0 = time.time()
        e = psi4.energy(psi4_target)
        res["time"] = time.time() - t0
        res["energy"] = e
        res["status"] = "OK"
    except Exception as e: 
        print(f"Psi4 Error pada {sys_name} ({psi4_target}): {e}")
    return res

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

    
    
    # -------------------------------------------------------------------------
    # KUNCI APPLES-TO-APPLES: Samakan threshold OMP2 dengan d_convergence Psi4
    # -------------------------------------------------------------------------
    mp2_conf.energy_threshold = THRESHOLD
    #mp2_conf.gradient_threshold = THRESHOLD
    # ------------------------------------------------------------------------
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
    psi4.set_memory('4 GB')
    print("=============================================================================================")
    print(f"  MSHQC vs PSI4 | MP2 SUITE BENCHMARK")
    print(f"  Basis: {BASIS_NAME} | Integral: {ERI_METHOD.upper()} | SCF Mode: {SCF_MODE.upper()}")
    print("=============================================================================================")

    for sys_name, atoms, chg, mult, ref in SYSTEMS:
        # Tentukan metode MP2 yang valid untuk sistem ini
        metode_aktif = []
        if mult == 1:
            metode_aktif.extend(['rmp2', 'omp2'])
        else:
            metode_aktif.extend(['ump2', 'omp2'])
            
        for algo in metode_aktif:
            print(f"Mengeksekusi: {sys_name:<8} -> {algo.upper()}")
            
            m_e, m_t = run_mshqc_bench(sys_name, atoms, chg, mult, ref, BASIS_NAME, algo)
            psi_res  = run_psi4_bench(sys_name, atoms, chg, mult, BASIS_NAME, ref, algo)
            
            if m_e is not None:
                RESULTS_DB.append({
                    "sys": sys_name, "method": algo.upper(),
                    "mshqc_e": m_e, "mshqc_t": m_t,
                    "psi4_e": psi_res['energy'], "psi4_t": psi_res['time'],
                    "status": "OK" if psi_res['status'] == "OK" else "PSI_FAIL"
                })
                
        # Thermal Control (Opsional)
        time.sleep(2)

    print("\n\n=============================================================================================")
    print(f"                                      HASIL AKHIR BENCHMARK")
    print("=============================================================================================")
    header = f"| {'System':<8} | {'Method':<8} | {'MSHQC (Ha)':<16} | {'Psi4 (Ha)':<16} | {'Diff (mHa)':<11} | {'T_MSH':<6} | {'T_Psi':<6} |"
    print("-" * len(header)); print(header); print("-" * len(header))

    for row in RESULTS_DB:
        sys = row['sys']; met = row['method']
        m_e = row['mshqc_e']; p_e = row['psi4_e']
        m_t = row['mshqc_t']; p_t = row['psi4_t']

        if row['status'] == "OK":
            diff = (m_e - p_e) * 1000.0
            diff_str = f"\033[92m{diff:<11.5f}\033[0m" if abs(diff) < 1e-4 else f"\033[91m{diff:<11.5f}\033[0m"
            print(f"| {sys:<8} | {met:<8} | {m_e:<16.8f} | {p_e:<16.8f} | {diff_str} | {m_t:<6.3f} | {p_t:<6.3f} |")
        else:
            print(f"| {sys:<8} | {met:<8} | {m_e:<16.8f} | {'FAILED':<16} | {'-':<11} | {m_t:<6.3f} | {p_t:<6.3f} |")
    print("-" * len(header))

if __name__ == "__main__":
    main()