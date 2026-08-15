 # ==============================================================================
 # Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
 #
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 #     http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 # ==============================================================================

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

# Perhatikan: Sistem open-shell sekarang menggunakan 'rohf'
SYSTEMS = [
    ("Neon",     [(10, 0.0, 0.0, 0.0)], 0, 1, 'rhf'),
    ("H2O",      [(8, 0.0, 0.0, 0.1173 * ANG_TO_BOHR), 
                  (1, 0.0, 0.7572 * ANG_TO_BOHR, -0.4692 * ANG_TO_BOHR), 
                  (1, 0.0, -0.7572 * ANG_TO_BOHR, -0.4692 * ANG_TO_BOHR)], 0, 1, 'rhf'),
    ("Lithium",  [(3, 0.0, 0.0, 0.0)], 0, 2, 'rohf'),
    ("Carbon",   [(6, 0.0, 0.0, 0.0)], 0, 3, 'rohf'),
    ("Oxygen",   [(8, 0.0, 0.0, 0.0)], 0, 3, 'rohf'),
]
ATOM_MAP = {1: 'H', 2: 'He', 3: 'Li', 6: 'C', 8: 'O', 10: 'Ne'}
RESULTS_DB = []

# =====================================================================
# KERNEL PSI4 (Parser Log Bersih)
# =====================================================================
def run_psi4_hf(sys_name, atoms, chg, mult, basis, ref):
    psi4.core.clean()
    psi4.set_num_threads(4)
    
    # Simpan output ke file sementara agar tidak mengotori terminal
    log_file = f"psi4_{sys_name}_temp.log"
    psi4.core.set_output_file(log_file, False)
    
    mol_str = f"{chg} {mult}\nunits bohr\n" 
    for z, x, y, z_c in atoms: 
        mol_str += f"{ATOM_MAP.get(z, 'X')} {x:.10f} {y:.10f} {z_c:.10f}\n"
    psi4.geometry(mol_str)
    
    opts = {
        'basis': basis,
        'reference': ref,
        'scf_type': 'df',
        'df_basis_scf': AUX_BASIS,
        'e_convergence': THRESHOLD,
        'd_convergence': THRESHOLD,
        'puream': True,
        'print': 1  # Cukup level 1
    }
    psi4.set_options(opts)

    print(f"\n[PSI4] Menjalankan {ref.upper()} pada {sys_name}...")
    t0 = time.time()
    e = psi4.energy('scf')
    t = time.time() - t0
    
    # --- PARSER LOG PSI4 (Hanya Ambil Iterasi) ---
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        printing = False
        print("-" * 60)
        print("  Iterasi Psi4:")
        for line in lines:
            if "==> Iterations <==" in line:
                printing = True
                continue
            if "Energy and wave function converged." in line or "==> Post-Iterations <==" in line:
                if printing:
                    break
            if printing and line.strip():
                print("  " + line.rstrip())
        print("-" * 60)
        
        # Hapus file temporary
        if os.path.exists(log_file):
            os.remove(log_file)
            
    except Exception as err:
        print(f"  [Gagal memparsing log Psi4: {err}]")

    return e, t

# =====================================================================
# KERNEL MSHQC (Sesuai dengan ROHF)
# =====================================================================
def run_mshqc_hf(sys_name, atoms, chg, mult, ref, basis_name):
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

    scf_conf = mshqc.SCFConfig()
    scf_conf.energy_threshold = THRESHOLD
    scf_conf.density_threshold = THRESHOLD
    scf_conf.scf_type = SCF_MODE
    scf_conf.eri_method = ERI_METHOD
    scf_conf.use_df = True
    scf_conf.aux_basis_name = AUX_BASIS
    
    # Aktifkan log iterasi mshqc
    scf_conf.print_level = 2 

    n_elec = aligned_mol.n_electrons() - chg
    n_a = (n_elec + mult - 1) // 2
    n_b = n_elec - n_a

    # Inisialisasi solver berdasarkan tipe referensi
    if ref == 'rhf': 
        scf_solver = mshqc.RHF(aligned_mol, basis, ints, pg, pl, scf_conf)
    elif ref == 'uhf': 
        scf_solver = mshqc.UHF(aligned_mol, basis, ints, pg, pl, n_a, n_b, scf_conf)
    elif ref == 'rohf': 
        # Memastikan solver ROHF dipanggil
        scf_solver = mshqc.ROHF(aligned_mol, basis, ints, pg, pl, n_a, n_b, scf_conf)
    else:
        raise ValueError("Metode tidak dikenali.")
    
    print(f"\n[MSHQC] Menjalankan {ref.upper()} pada {sys_name}...")
    t0 = time.time()
    scf_res = scf_solver.compute()
    t = time.time() - t0
    
    if not scf_res.converged:
        print(f"SCF gagal konvergen pada {sys_name}.")
        return 0.0, 0.0

    return scf_res.energy_total, t

# =====================================================================
# FUNGSI PROFILING
# =====================================================================
def get_resource_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_ram_mb = usage.ru_maxrss / 1024.0 
    user_cpu_time = usage.ru_utime
    sys_cpu_time = usage.ru_stime
    return max_ram_mb, user_cpu_time, sys_cpu_time

# =====================================================================
# MAIN LOOP
# =====================================================================
def main():
    psi4.set_memory('4 GB')
    print("\n" + "=" * 80)
    print(f"  BENCHMARK SCF BERSIH (RHF & ROHF)")
    print(f"  Basis: {BASIS_NAME} | Integral: {ERI_METHOD.upper()} | Threads: {os.environ['OMP_NUM_THREADS']}")
    print("=" * 80)

    for sys_name, atoms, chg, mult, ref in SYSTEMS:
        print("\n" + "#" * 60)
        print(f"  SISTEM: {sys_name.upper()} | METODE: {ref.upper()}")
        print("#" * 60)
        
        # 1. Eksekusi MSHQC
        gc.collect() 
        m_e, m_t = run_mshqc_hf(sys_name, atoms, chg, mult, ref, BASIS_NAME)
        
        # 2. Eksekusi Psi4
        gc.collect()
        p_e, p_t = run_psi4_hf(sys_name, atoms, chg, mult, BASIS_NAME, ref)
        
        # 3. Simpan Hasil
        RESULTS_DB.append({
            "sys": sys_name, "method": ref.upper(),
            "mshqc_e": m_e, "mshqc_t": m_t,
            "psi4_e": p_e, "psi4_t": p_t
        })
        time.sleep(1)

    # Menampilkan Laporan Akhir
    print("\n\n" + "=" * 80)
    print("  HASIL AKHIR BENCHMARK HARTREE-FOCK")
    print("=" * 80)
    
    header = f"| {'System':<8} | {'Ref':<5} | {'MSHQC (Ha)':<16} | {'Psi4 (Ha)':<16} | {'Diff (mHa)':<11} | {'T_MSH':<6} | {'T_Psi':<6} |"
    print("-" * len(header)); print(header); print("-" * len(header))

    for row in RESULTS_DB:
        sys = row['sys']; met = row['method']
        m_e = row['mshqc_e']; p_e = row['psi4_e']
        m_t = row['mshqc_t']; p_t = row['psi4_t']

        diff = (m_e - p_e) * 1000.0
        diff_str = f"\033[92m{diff:<11.5f}\033[0m" if abs(diff) < 1e-4 else f"\033[91m{diff:<11.5f}\033[0m"
        print(f"| {sys:<8} | {met:<5} | {m_e:<16.8f} | {p_e:<16.8f} | {diff_str} | {m_t:<6.3f} | {p_t:<6.3f} |")
    
    print("-" * len(header))
    
    # Pengukuran Resource Hardware
    max_ram, u_cpu, s_cpu = get_resource_usage()
    print("\n[PROFILING HARDWARE]")
    print(f"Puncak Penggunaan RAM (Max RSS) : {max_ram:.2f} MB")
    print(f"Total Waktu CPU (User Mode)     : {u_cpu:.2f} Detik")
    print(f"Total Waktu CPU (System Mode)   : {s_cpu:.2f} Detik")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()