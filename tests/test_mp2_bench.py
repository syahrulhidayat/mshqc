import os
import multiprocessing
import time
import gc
import resource

# =====================================================================
# 1. KONFIGURASI ENVIRONMENT
# =====================================================================
N_CORES = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TBLIS_NUM_THREADS"] = "1" 
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import mshqc
import psi4

# =====================================================================
# 2. PARAMETER GLOBAL
# =====================================================================
BASIS_NAME  = "cc-pVTZ"
AUX_BASIS   = "cc-pVTZ-RI"
THRESHOLD   = 1e-9
ANG_TO_BOHR = 1.88972612462577 
ERI_METHOD  = "df" 
SCF_MODE    = "direct"

# Referensi menggunakan UHF untuk open-shell pada OMP2
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

# =====================================================================
# KERNEL PSI4 (Parser Log Bersih khusus OMP2) - DIPERBARUI
# =====================================================================
def run_psi4_omp2(sys_name, atoms, chg, mult, basis, ref):
    psi4.core.clean()
    psi4.set_num_threads(4)
    
    log_file = f"psi4_{sys_name}_omp2_temp.log"
    psi4.core.set_output_file(log_file, False)
    
    mol_str = f"{chg} {mult}\nunits bohr\n" 
    for z, x, y, z_c in atoms: 
        mol_str += f"{ATOM_MAP.get(z, 'X')} {x:.10f} {y:.10f} {z_c:.10f}\n"
    psi4.geometry(mol_str)
    
    opts = {
        'basis': basis,
        'reference': ref,
        'scf_type': 'df',
        'mp2_type': 'df',
        'df_basis_scf': AUX_BASIS,
        'df_basis_mp2': AUX_BASIS,
        'e_convergence': THRESHOLD,
        'd_convergence': THRESHOLD,
        'freeze_core': 'False',   # CRITICAL! Matikan frozen core
        'puream': True,
        'print': 2  # Naikkan level print agar Psi4 lebih cerewet soal iterasi OMP2
    }
    psi4.set_options(opts)

    print(f"\n[PSI4] Menjalankan OMP2 pada {sys_name}...")
    t0 = time.time()
    e = psi4.energy('omp2')
    t = time.time() - t0
    
    # --- PARSER LOG PSI4 YANG LEBIH FLEKSIBEL ---
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        printing = False
        print("-" * 60)
        print("  Iterasi Psi4 (Orbital Optimization):")
        
        for line in lines:
            # Mencari penanda awal tabel iterasi OCC/OMP2
            if "Iter       Total Energy        Delta E" in line or "Orbital Optimization Iterations" in line:
                printing = True
                print("  " + line.strip())
                continue
                
            # Berhenti nge-print jika iterasi selesai atau ada summary
            if printing and ("==> Post-Iterations" in line or "Optimization converged" in line or "Energy and wave" in line):
                break
            
            # Print baris iterasi jika sedang dalam mode printing
            if printing:
                # Hanya print baris yang terlihat seperti data numerik iterasi atau header
                if line.strip() and ("---" in line or any(char.isdigit() for char in line)):
                    print("  " + line.rstrip())
                    
        print("-" * 60)
        
        if os.path.exists(log_file):
            os.remove(log_file)
            
    except Exception as err:
        print(f"  [Gagal memparsing log Psi4: {err}]")

    return e, t

# =====================================================================
# KERNEL MSHQC (OMP2)
# =====================================================================
def run_mshqc_omp2(sys_name, atoms, chg, mult, ref, basis_name):
    raw_mol = mshqc.Molecule()
    for z, x, y, z_c in atoms: raw_mol.add_atom(z, x, y, z_c)
    raw_mol.set_charge(chg); raw_mol.set_multiplicity(mult)
    
    pg = mshqc.PointGroup(raw_mol)
    pg.detect()
    aligned_mol = pg.get_aligned_molecule()

    basis = mshqc.BasisSet(basis_name, aligned_mol)
    ints = mshqc.IntegralEngine(aligned_mol, basis)
    pl = mshqc.PetiteList(basis, pg)    
    pl.build()

    # 1. Setup SCF (HF) - Mode Senyap (Silent)
    scf_conf = mshqc.SCFConfig()
    scf_conf.energy_threshold = THRESHOLD
    scf_conf.density_threshold = THRESHOLD
    scf_conf.scf_type = SCF_MODE
    scf_conf.eri_method = ERI_METHOD
    scf_conf.use_df = True
    scf_conf.aux_basis_name = AUX_BASIS
    scf_conf.print_level = 0  # <--- Bisu untuk fokus ke OMP2
    
    n_elec = aligned_mol.n_electrons() - chg
    n_a = (n_elec + mult - 1) // 2
    n_b = n_elec - n_a

    if ref == 'rhf': scf_solver = mshqc.RHF(aligned_mol, basis, ints, pg, pl, scf_conf)
    elif ref == 'uhf': scf_solver = mshqc.UHF(aligned_mol, basis, ints, pg, pl, n_a, n_b, scf_conf)
    else: scf_solver = mshqc.ROHF(aligned_mol, basis, ints, pg, pl, n_a, n_b, scf_conf)
    
    scf_res = scf_solver.compute()
    if not scf_res.converged:
        print(f"SCF gagal konvergen pada {sys_name}.")
        return 0.0, 0.0

    # 2. Setup OMP2 - Mode Berisik (Verbose)
    mp2_conf = mshqc.MP2Config()
    mp2_conf.scf_type = SCF_MODE
    mp2_conf.eri_method = ERI_METHOD
    mp2_conf.use_df = True
    mp2_conf.aux_basis_name = AUX_BASIS
    mp2_conf.energy_threshold = THRESHOLD
    mp2_conf.opt_method = "soscf"
    mp2_conf.print_level = 2  # <--- Aktifkan log iterasi OMP2
    
    mp2_solver = mshqc.OMP2(aligned_mol, basis, ints, scf_res, mp2_conf, pg, pl)
    
    print(f"\n[MSHQC] Menjalankan OMP2 pada {sys_name}...")
    t0 = time.time()
    mp2_res = mp2_solver.compute()
    t = time.time() - t0 # Waktu murni untuk OMP2, SCF tidak dihitung

    return mp2_res.energy_total, t

# =====================================================================
# FUNGSI PROFILING
# =====================================================================
def get_resource_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_ram_mb = usage.ru_maxrss / 1024.0 
    return max_ram_mb, usage.ru_utime, usage.ru_stime

# =====================================================================
# MAIN LOOP
# =====================================================================
def main():
    psi4.set_memory('4 GB')
    print("\n" + "=" * 80)
    print(f"  BENCHMARK OMP2 BERSIH")
    print(f"  Basis: {BASIS_NAME} | Integral: {ERI_METHOD.upper()} | Threads: {os.environ['OMP_NUM_THREADS']}")
    print("=" * 80)

    for sys_name, atoms, chg, mult, ref in SYSTEMS:
        print("\n" + "#" * 60)
        print(f"  SISTEM: {sys_name.upper()} | METODE: OMP2 (Ref: {ref.upper()})")
        print("#" * 60)
        
        gc.collect() 
        m_e, m_t = run_mshqc_omp2(sys_name, atoms, chg, mult, ref, BASIS_NAME)
        
        gc.collect()
        p_e, p_t = run_psi4_omp2(sys_name, atoms, chg, mult, BASIS_NAME, ref)
        
        RESULTS_DB.append({
            "sys": sys_name, "method": "OMP2",
            "mshqc_e": m_e, "mshqc_t": m_t,
            "psi4_e": p_e, "psi4_t": p_t
        })
        time.sleep(1)

    print("\n\n" + "=" * 80)
    print("  HASIL AKHIR BENCHMARK OMP2")
    print("=" * 80)
    
    header = f"| {'System':<8} | {'Method':<6} | {'MSHQC (Ha)':<16} | {'Psi4 (Ha)':<16} | {'Diff (mHa)':<11} | {'T_MSH':<6} | {'T_Psi':<6} |"
    print("-" * len(header)); print(header); print("-" * len(header))

    for row in RESULTS_DB:
        sys = row['sys']; met = row['method']
        m_e = row['mshqc_e']; p_e = row['psi4_e']
        m_t = row['mshqc_t']; p_t = row['psi4_t']

        diff = (m_e - p_e) * 1000.0
        diff_str = f"\033[92m{diff:<11.5f}\033[0m" if abs(diff) < 1e-4 else f"\033[91m{diff:<11.5f}\033[0m"
        print(f"| {sys:<8} | {met:<6} | {m_e:<16.8f} | {p_e:<16.8f} | {diff_str} | {m_t:<6.3f} | {p_t:<6.3f} |")
    
    print("-" * len(header))
    
    max_ram, u_cpu, s_cpu = get_resource_usage()
    print("\n[PROFILING HARDWARE]")
    print(f"Puncak Penggunaan RAM (Max RSS) : {max_ram:.2f} MB")
    print(f"Total Waktu CPU (User Mode)     : {u_cpu:.2f} Detik")
    print(f"Total Waktu CPU (System Mode)   : {s_cpu:.2f} Detik")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()