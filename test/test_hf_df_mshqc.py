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
import mshqc

# ==============================================================================
# INJEKSI THREAD OPENMP (WAJIB PALING ATAS)
# ==============================================================================
N_CORES = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TBLIS_NUM_THREADS"] = "1" 

# ==============================================================================
# KONFIGURASI GLOBAL MSHQC
# ==============================================================================
BASIS_NAME = "cc-pVTZ"
AUX_BASIS  = "cc-pVTZ-RI"
THRESHOLD  = 1e-9
ANG_TO_BOHR = 1.88972612462577 

# --- SAKLAR UTAMA ---
ERI_METHOD = "df"        # Menggunakan Density Fitting
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

def run_mshqc_scf(sys_name, atoms, chg, mult, ref, basis_name):
    raw_mol = mshqc.Molecule()
    for z, x, y, z_c in atoms: raw_mol.add_atom(z, x, y, z_c)
    raw_mol.set_charge(chg)
    raw_mol.set_multiplicity(mult)
    
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
    
    # Setup konfigurasi khusus DF
    if ERI_METHOD == "df":
        scf_conf.use_df = True
        scf_conf.aux_basis_name = AUX_BASIS
        scf_conf.df_threshold = 1e-9

    n_elec = aligned_mol.n_electrons() - chg
    n_a = (n_elec + mult - 1) // 2
    n_b = n_elec - n_a

    if ref == 'rhf': 
        scf_solver = mshqc.RHF(aligned_mol, basis, ints, pg, pl, scf_conf)
    elif ref == 'uhf': 
        scf_solver = mshqc.UHF(aligned_mol, basis, ints, pg, pl, n_a, n_b, scf_conf)
    else: 
        scf_solver = mshqc.ROHF(aligned_mol, basis, ints, pg, pl, n_a, n_b, scf_conf)
    
    t0 = time.time()
    try:
        scf_res = scf_solver.compute()
        t_scf = time.time() - t0
        
        if not scf_res.converged:
            print(f"SCF gagal konvergen pada {sys_name}.")
            return None, 0.0
            
        return scf_res.energy_total, t_scf
    except Exception as e:
        print(f"Error MSHQC SCF pada {sys_name}: {e}")
        return None, 0.0


def main():
    print("=========================================================================================")
    print(f"  MSHQC ONLY | SCF PERFORMANCE PROFILING TEST")
    print(f"  Basis: {BASIS_NAME} | Integral: {ERI_METHOD.upper()} | SCF Mode: {SCF_MODE.upper()}")
    print("=========================================================================================")

    for sys_name, atoms, chg, mult, ref in SYSTEMS:
        print(f"Mengeksekusi: {sys_name:<8} -> {ref.upper()}")
        
        scf_e, scf_t = run_mshqc_scf(sys_name, atoms, chg, mult, ref, BASIS_NAME)
        
        if scf_e is not None:
            RESULTS_DB.append({
                "sys": sys_name, 
                "method": ref.upper(),
                "scf_e": scf_e, 
                "scf_t": scf_t
            })
        time.sleep(0.5)

    print("\n============================================================")
    print(f"                 HASIL AKHIR MSHQC BENCHMARK")
    print("============================================================")
    header = f"| {'System':<8} | {'Method':<8} | {'SCF Energy (Ha)':<18} | {'Time (s)':<8} |"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for row in RESULTS_DB:
        sys = row['sys']
        met = row['method']
        s_e = row['scf_e']
        s_t = row['scf_t']
        print(f"| {sys:<8} | {met:<8} | {s_e:<18.8f} | {s_t:<8.3f} |")
    print("-" * len(header))

if __name__ == "__main__":
    main()
    os._exit(0)