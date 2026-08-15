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
import psutil
import time
import gc

# ==============================================================================
# INJEKSI THREAD OPENMP (WAJIB PALING ATAS)
# ==============================================================================
N_CORES = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = "6" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TBLIS_NUM_THREADS"] = "1" 
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import mshqc
import psi4

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

def get_current_ram_mb():
    """Mendapatkan penggunaan RAM (RSS) proses saat ini dalam Megabytes"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_psi4_bench(sys_name, atoms, chg, mult, basis, ref, algo):
    psi4.core.clean()
    psi4.core.set_output_file('psi4_silent.out', False)
    
    mol_str = f"{chg} {mult}\nunits bohr\n" 
    for z, x, y, z_c in atoms: 
        mol_str += f"{ATOM_MAP.get(z, 'X')} {x:.10f} {y:.10f} {z_c:.10f}\n"
    psi4.geometry(mol_str)
    
    psi4_scf_type = 'df' if ERI_METHOD == 'df' else ('pk' if SCF_MODE == 'incore' else 'direct')
    psi4_mp_type = 'df' if ERI_METHOD == 'df' else 'conv'

    opts = {
        'basis': basis,
        'reference': ref,
        'scf_type': psi4_scf_type,
        'mp2_type': psi4_mp_type,
        'e_convergence': THRESHOLD,
        'd_convergence': THRESHOLD,
        'freeze_core': 'False',
        'puream': True,
        'print': 0 
    }
    
    if ERI_METHOD == 'df':
        opts['df_basis_scf'] = AUX_BASIS
        opts['df_basis_mp2'] = AUX_BASIS
    elif ERI_METHOD == 'cholesky':
        opts['cholesky_tolerance'] = 1e-9
        
    psi4.set_options(opts)

    psi4_target = algo.replace('r', '').replace('u', '')

    res = {"status": "FAIL", "energy": 0.0, "time": 0.0, "ram": 0.0}
    try:
        gc.collect()
        mem_start = get_current_ram_mb()
        t0 = time.time()
        
        e = psi4.energy(psi4_target)
        
        t1 = time.time()
        mem_end = get_current_ram_mb()
        
        res["time"] = t1 - t0
        res["energy"] = e
        res["ram"] = max(0.0, mem_end - mem_start)
        res["status"] = "OK"
    except Exception as e: 
        print(f"Psi4 Error pada {sys_name} ({psi4_target}): {e}")
    return res

def run_mshqc_combined(sys_name, atoms, chg, mult, ref, basis_name, fam):
    """Menjalankan MP2 & MP3 MSHQC secara sekuensial agar tidak redundan"""
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
        return None

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

    if ref == 'rhf': scf_solver = mshqc.RHF(aligned_mol, basis, ints, pg, pl, scf_conf)
    elif ref == 'uhf': scf_solver = mshqc.UHF(aligned_mol, basis, ints, pg, pl, n_a, n_b, scf_conf)
    else: scf_solver = mshqc.ROHF(aligned_mol, basis, ints, pg, pl, n_a, n_b, scf_conf)
    
    scf_res = scf_solver.compute()
    if not scf_res.converged:
        return None

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
        # --- MP2 ---
        gc.collect()
        mem0 = get_current_ram_mb()
        t0 = time.time()
        
        if fam == 'RMP': mp2_solver = mshqc.RMP2(aligned_mol, basis, ints, scf_res, mp2_conf, pg, pl)
        elif fam == 'UMP': mp2_solver = mshqc.UMP2(aligned_mol, basis, ints, scf_res, mp2_conf, pg, pl)
        elif fam == 'OMP': mp2_solver = mshqc.OMP2(aligned_mol, basis, ints, scf_res, mp2_conf, pg, pl)
        
        mp2_res = mp2_solver.compute()
        
        t1 = time.time()
        mem1 = get_current_ram_mb()
        
        # --- MP3 ---
        # Melanjutkan ke MP3 dengan hasil mp2_res sebelumnya
        if fam == 'RMP': mp3_solver = mshqc.RMP3(scf_res, mp2_res, mp2_conf, ints)
        elif fam == 'UMP': mp3_solver = mshqc.UMP3(scf_res, mp2_res, mp2_conf, ints)
        elif fam == 'OMP': mp3_solver = mshqc.OMP3(scf_res, mp2_res, mp2_conf, ints)
        
        mp3_res = mp3_solver.compute()
        
        t2 = time.time()
        mem2 = get_current_ram_mb()
        
        e_mp2 = getattr(mp2_res, 'e_total', getattr(mp2_res, 'energy_total', 0.0))
        e_mp3 = getattr(mp3_res, 'e_total', getattr(mp3_res, 'energy_total', 0.0))
        
        return {
            'mp2_e': e_mp2, 'mp2_t': t1 - t0, 'mp2_ram': max(0.0, mem1 - mem0),
            'mp3_e': e_mp3, 'mp3_t': t2 - t0, 'mp3_ram': max(0.0, mem2 - mem0) 
            # Note: Waktu MP3 dihitung sejak awal proses korelasi elektron, sejalan dengan metode Psi4
        }

    except Exception as e:
        print(f"Error MSHQC MPn pada {sys_name} ({fam}): {e}")
        return None

def main():
    psi4.set_memory('4 GB')
    print("=============================================================================================")
    print(f"  MSHQC vs PSI4 | KORELASI ELEKTRON (MP2 & MP3) BENCHMARK SUITE")
    print(f"  Basis: {BASIS_NAME} | Integral: {ERI_METHOD.upper()} | SCF Mode: {SCF_MODE.upper()}")
    print("=============================================================================================")

    for sys_name, atoms, chg, mult, ref in SYSTEMS:
        families = ['RMP', 'OMP'] if mult == 1 else ['UMP', 'OMP']
            
        for fam in families:
            print(f"Mengeksekusi: {sys_name:<8} -> {fam}2 & {fam}3 (Single Pass) ...")
            
            # --- MSHQC EKSEKUSI GABUNGAN ---
            m_res = run_mshqc_combined(sys_name, atoms, chg, mult, ref, BASIS_NAME, fam)
            
            if m_res is not None:
                # --- PSI4 EKSEKUSI TERPISAH ---
                p2_res = run_psi4_bench(sys_name, atoms, chg, mult, BASIS_NAME, ref, f"{fam}2".lower())
                p3_res = run_psi4_bench(sys_name, atoms, chg, mult, BASIS_NAME, ref, f"{fam}3".lower())
                
                # Simpan Hasil MP2
                RESULTS_DB.append({
                    "sys": sys_name, "method": f"{fam}2",
                    "mshqc_e": m_res['mp2_e'], "mshqc_t": m_res['mp2_t'], "mshqc_ram": m_res['mp2_ram'],
                    "psi4_e": p2_res['energy'], "psi4_t": p2_res['time'], "psi4_ram": p2_res['ram'],
                    "status": "OK" if p2_res['status'] == "OK" else "PSI_FAIL"
                })
                
                # Simpan Hasil MP3
                RESULTS_DB.append({
                    "sys": sys_name, "method": f"{fam}3",
                    "mshqc_e": m_res['mp3_e'], "mshqc_t": m_res['mp3_t'], "mshqc_ram": m_res['mp3_ram'],
                    "psi4_e": p3_res['energy'], "psi4_t": p3_res['time'], "psi4_ram": p3_res['ram'],
                    "status": "OK" if p3_res['status'] == "OK" else "PSI_FAIL"
                })
                
        time.sleep(1)

    # --- TABEL HASIL (DESAIN BARU) ---
    print("\n\n" + "="*148)
    print(f"{'HASIL AKHIR BENCHMARK KOMPREHENSIF':^148}")
    print("="*148)
    
    # Kolom baru mencakup E_MSH dan E_Psi
    header = f"| {'System':<8} | {'Method':<6} | {'E_MSH (Ha)':<14} | {'E_Psi (Ha)':<14} | {'dE (mHa)':<10} | {'T_MSH (s)':<9} | {'T_Psi (s)':<9} | {'dT (s)':<9} | {'RAM_MSH':<7} | {'RAM_Psi':<7} | {'dRAM (MB)':<9} |"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for row in RESULTS_DB:
        sys = row['sys']; met = row['method']
        m_e = row['mshqc_e']; p_e = row['psi4_e']
        m_t = row['mshqc_t']; p_t = row['psi4_t']
        m_ram = row['mshqc_ram']; p_ram = row['psi4_ram']

        if row['status'] == "OK":
            # Kalkulasi Selisih (Delta)
            diff_e = (m_e - p_e) * 1000.0
            diff_t = m_t - p_t
            diff_ram = m_ram - p_ram
            
            # Format Warna
            e_str = f"\033[92m{diff_e:<10.5f}\033[0m" if abs(diff_e) < 1e-4 else f"\033[91m{diff_e:<10.5f}\033[0m"
            t_str = f"\033[92m{diff_t:<9.3f}\033[0m" if diff_t <= 0 else f"\033[93m{diff_t:<9.3f}\033[0m"
            ram_str = f"\033[92m{diff_ram:<9.2f}\033[0m" if diff_ram <= 0 else f"\033[93m{diff_ram:<9.2f}\033[0m"
            
            print(f"| {sys:<8} | {met:<6} | {m_e:<14.8f} | {p_e:<14.8f} | {e_str} | {m_t:<9.3f} | {p_t:<9.3f} | {t_str} | {m_ram:<7.2f} | {p_ram:<7.2f} | {ram_str} |")
        else:
            print(f"| {sys:<8} | {met:<6} | {'FAILED':<14} | {'-':<14} | {'-':<10} | {m_t:<9.3f} | {'-':<9} | {'-':<9} | {m_ram:<7.2f} | {'-':<7} | {'-':<9} |")
            
    print("-" * len(header))

if __name__ == "__main__":
    main()