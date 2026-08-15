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

    # Memetakan rmp2/ump2 -> mp2 dan rmp3/ump3 -> mp3
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

def main():
    psi4.set_memory('4 GB')
    print("=========================================================================")
    print(f"  PSI4 ONLY | ELECTRON CORRELATION (MP2 & MP3) BENCHMARK SUITE")
    print(f"  Basis: {BASIS_NAME} | Integral: {ERI_METHOD.upper()} | SCF Mode: {SCF_MODE.upper()}")
    print("=========================================================================")

    for sys_name, atoms, chg, mult, ref in SYSTEMS:
        families = ['RMP', 'OMP'] if mult == 1 else ['UMP', 'OMP']
            
        for fam in families:
            for order in [2, 3]:
                method_name = f"{fam}{order}"
                print(f"Mengeksekusi: {sys_name:<8} -> {method_name} ...")
                
                p_res = run_psi4_bench(sys_name, atoms, chg, mult, BASIS_NAME, ref, method_name.lower())
                
                RESULTS_DB.append({
                    "sys": sys_name, 
                    "method": method_name,
                    "energy": p_res['energy'], 
                    "time": p_res['time'], 
                    "ram": p_res['ram'],
                    "status": p_res['status']
                })
                
        time.sleep(0.5)

    # --- TABEL HASIL AKHIR ---
    print("\n\n" + "="*77)
    print(f"{'HASIL AKHIR BENCHMARK PSI4':^77}")
    print("="*77)
    
    header = f"| {'System':<8} | {'Method':<6} | {'Energy (Ha)':<18} | {'Time (s)':<10} | {'RAM (MB)':<10} | {'Status':<8} |"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for row in RESULTS_DB:
        sys = row['sys']
        met = row['method']
        e = row['energy']
        t = row['time']
        ram = row['ram']
        stat = row['status']

        if stat == "OK":
            print(f"| {sys:<8} | {met:<6} | {e:<18.10f} | {t:<10.3f} | {ram:<10.2f} | {stat:<8} |")
        else:
            print(f"| {sys:<8} | {met:<6} | {'FAILED':<18} | {'-':<10} | {'-':<10} | {stat:<8} |")
            
    print("-" * len(header))

if __name__ == "__main__":
    main()