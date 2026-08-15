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
import psutil
import time
import gc

# ==============================================================================
# INJEKSI THREAD OPENMP (WAJIB PALING ATAS)
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "6" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TBLIS_NUM_THREADS"] = "1" 
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import mshqc

# ==============================================================================
# KONFIGURASI GLOBAL
# ==============================================================================
BASIS_NAME = "cc-pVTZ"
AUX_BASIS  = "cc-pVTZ-RI"
THRESHOLD  = 1e-9
ANG_TO_BOHR = 1.88972612462577 

ERI_METHOD = "df" 
SCF_MODE = "direct"
# ==============================================================================

# Sistem yang diuji. RMP2 dan RMP3 akan otomatis tereksekusi pada molekul 'rhf'.
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

def get_current_ram_mb():
    """Mendapatkan penggunaan RAM (RSS) proses saat ini dalam Megabytes"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_mshqc_only(sys_name, atoms, chg, mult, ref, basis_name):
    # Setup Molekul
    raw_mol = mshqc.Molecule()
    for z, x, y, z_c in atoms: raw_mol.add_atom(z, x, y, z_c)
    raw_mol.set_charge(chg); raw_mol.set_multiplicity(mult)
    
    # Deteksi Simetri
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
        return

    # Konfigurasi SCF
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

    n_elec = aligned_mol.n_electrons() - chg
    n_a = (n_elec + mult - 1) // 2
    n_b = n_elec - n_a

    # Penentuan Tipe Referensi
    if ref == 'rhf':
        scf_solver = mshqc.RHF(aligned_mol, basis, ints, pg, pl, scf_conf)
        fam = 'RMP'
    elif ref == 'uhf':
        scf_solver = mshqc.UHF(aligned_mol, basis, ints, pg, pl, n_a, n_b, scf_conf)
        fam = 'UMP'
    else:
        scf_solver = mshqc.ROHF(aligned_mol, basis, ints, pg, pl, n_a, n_b, scf_conf)
        fam = 'RMP' # Fallback
    
    scf_res = scf_solver.compute()
    if not scf_res.converged:
        print(f"SCF tidak konvergen pada {sys_name}")
        return

    # Konfigurasi MP2/MP3
    mp2_conf = mshqc.MP2Config()
    mp2_conf.scf_type = SCF_MODE
    mp2_conf.eri_method = ERI_METHOD
    mp2_conf.use_df = (ERI_METHOD == "df")
    mp2_conf.aux_basis_name = AUX_BASIS
    mp2_conf.print_level = 0
    mp2_conf.energy_threshold = THRESHOLD
    mp2_conf.gradient_threshold = THRESHOLD
    mp2_conf.opt_method = "soscf" 

    try:
        # ----------------------------------------------------
        # 1. MP2 (RMP2 / UMP2)
        # ----------------------------------------------------
        gc.collect()
        mem_start = get_current_ram_mb()
        t0 = time.time()
        
        if fam == 'RMP': mp2_solver = mshqc.RMP2(aligned_mol, basis, ints, scf_res, mp2_conf, pg, pl)
        elif fam == 'UMP': mp2_solver = mshqc.UMP2(aligned_mol, basis, ints, scf_res, mp2_conf, pg, pl)
        
        mp2_res = mp2_solver.compute()
        
        t1 = time.time()
        mem_end = get_current_ram_mb()
        
        e_mp2 = getattr(mp2_res, 'e_total', getattr(mp2_res, 'energy_total', 0.0))
        RESULTS_DB.append({
            "sys": sys_name, "method": f"{fam}2",
            "energy": e_mp2, "time": t1 - t0, "ram": max(0.0, mem_end - mem_start)
        })

        # ----------------------------------------------------
        # 2. MP3 (RMP3 / UMP3) -> Menggunakan hasil MP2
        # ----------------------------------------------------
        gc.collect()
        mem_start = get_current_ram_mb()
        t0 = time.time()
        
        if fam == 'RMP': mp3_solver = mshqc.RMP3(scf_res, mp2_res, mp2_conf, ints)
        elif fam == 'UMP': mp3_solver = mshqc.UMP3(scf_res, mp2_res, mp2_conf, ints)
        
        mp3_res = mp3_solver.compute()
        
        t1 = time.time()
        mem_end = get_current_ram_mb()
        
        e_mp3 = getattr(mp3_res, 'e_total', getattr(mp3_res, 'energy_total', 0.0))
        RESULTS_DB.append({
            "sys": sys_name, "method": f"{fam}3",
            "energy": e_mp3, "time": t1 - t0, "ram": max(0.0, mem_end - mem_start)
        })

        # ----------------------------------------------------
        # 3. OMP3 (Orbital-Optimized MP3)
        # ----------------------------------------------------
        gc.collect()
        mem_start = get_current_ram_mb()
        t0 = time.time()
        
        # OMP3 memakan argumen awal yang mirip dengan MP2 (termasuk tebakan scf_res)
        omp3_solver = mshqc.OMP3(aligned_mol, basis, ints, scf_res, mp2_conf, pg, pl)
        omp3_res = omp3_solver.compute()
        
        t1 = time.time()
        mem_end = get_current_ram_mb()
        
        e_omp3 = getattr(omp3_res, 'e_total', getattr(omp3_res, 'energy_total', 0.0))
        RESULTS_DB.append({
            "sys": sys_name, "method": "OMP3",
            "energy": e_omp3, "time": t1 - t0, "ram": max(0.0, mem_end - mem_start)
        })

    except Exception as e:
        print(f"Error MSHQC MPn pada {sys_name}: {e}")
def main():
    print("===============================================================================")
    print(f"  MSHQC ONLY | KORELASI ELEKTRON (MP2, MP3, OMP3) BENCHMARK SUITE")
    print(f"  Basis: {BASIS_NAME} | Integral: {ERI_METHOD.upper()} | SCF Mode: {SCF_MODE.upper()}")
    print("===============================================================================")

    for sys_name, atoms, chg, mult, ref in SYSTEMS:
        print(f"Mengeksekusi: {sys_name:<8} ...")
        run_mshqc_only(sys_name, atoms, chg, mult, ref, BASIS_NAME)
        time.sleep(0.5)

    # --- TABEL HASIL ---
    print("\n\n" + "="*82)
    print(f"{'LAPORAN PERFORMA MSHQC':^82}")
    print("="*82)
    
    header = f"| {'System':<8} | {'Method':<7} | {'Energy (Ha)':<18} | {'Time (s)':<11} | {'RAM (MB)':<10} |"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for row in RESULTS_DB:
        sys = row['sys']; met = row['method']
        e = row['energy']; t = row['time']; ram = row['ram']
        print(f"| {sys:<8} | {met:<7} | {e:<18.8f} | {t:<11.3f} | {ram:<10.2f} |")
            
    print("-" * len(header))

if __name__ == "__main__":
    main()