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

# =====================================================================
# 1. KONFIGURASI HPC & LINGKUNGAN
# =====================================================================
N_CORES = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TBLIS_NUM_THREADS"] = "1" 
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["EIGEN_NUM_THREADS"] = "1"
import mshqc
import time

# =====================================================================
# 2. PARAMETER GLOBAL (KHUSUS EKSAK & MP2)
# =====================================================================
BASIS_NAME = "cc-pVTZ"
THRESHOLD  = 1e-9
ANG_TO_BOHR = 1.88972612462577 

ERI_METHOD = "exact" # Metode diubah menjadi eksak (Exact 4-center)
SCF_MODE = "incore"

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

    # ---------------------------------------------------------
    # Setup Konfigurasi SCF
    # ---------------------------------------------------------
    scf_conf = mshqc.SCFConfig()
    scf_conf.energy_threshold = THRESHOLD
    scf_conf.density_threshold = THRESHOLD
    scf_conf.print_level = 0
    scf_conf.scf_type = SCF_MODE
    scf_conf.eri_method = ERI_METHOD
    scf_conf.use_df = False

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

    # ---------------------------------------------------------
    # Setup Konfigurasi MP2
    # ---------------------------------------------------------
    mp2_conf = mshqc.MP2Config()
    mp2_conf.scf_type = SCF_MODE
    mp2_conf.eri_method = ERI_METHOD
    mp2_conf.use_df = False
    mp2_conf.print_level = 0
    mp2_conf.energy_threshold = THRESHOLD
    mp2_conf.gradient_threshold = THRESHOLD
    mp2_conf.opt_method = "soscf" 

    t0 = time.time()
    try:
        if mp2_algo == 'rmp2': mp2_solver = mshqc.RMP2(aligned_mol, basis, ints, scf_res, mp2_conf, pg, pl)
        elif mp2_algo == 'ump2': mp2_solver = mshqc.UMP2(aligned_mol, basis, ints, scf_res, mp2_conf, pg, pl)
        elif mp2_algo == 'omp2': mp2_solver = mshqc.OMP2(aligned_mol, basis, ints, scf_res, mp2_conf, pg, pl)
        
        mp2_res = mp2_solver.compute()
        t_mp2 = time.time() - t0
        
        return mp2_res.energy_total, t_mp2
        
    except Exception as e:
        print(f"Error MSHQC MP2 pada {sys_name} ({mp2_algo}): {e}")
        return None, 0.0

def main():
    print("=========================================================================================")
    print(f"  MSHQC ONLY | PERFORMANCE PROFILING TEST (EXACT - MP2 ONLY)")
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
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for row in RESULTS_DB:
        sys = row['sys']
        met = row['method']
        m_e = row['mshqc_e']
        m_t = row['mshqc_t']
        print(f"| {sys:<8} | {met:<8} | {m_e:<16.8f} | {m_t:<8.3f} |")
    print("-" * len(header))

if __name__ == "__main__":
    main()