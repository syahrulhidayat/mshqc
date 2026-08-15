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

# --- CONFIG THREADS ---
N_CORES = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TBLIS_NUM_THREADS"] = "1" 
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import mshqc
import time

BASIS_NAME = "cc-pVTZ"
AUX_BASIS  = "cc-pVTZ-RI"
THRESHOLD  = 1e-9
ANG_TO_BOHR = 1.88972612462577 

ERI_METHOD = "df" 
SCF_MODE = "direct"

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
        return None, None, None, None

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
        return None, None, None, None

    mp2_conf = mshqc.MP2Config()
    mp2_conf.scf_type = SCF_MODE
    mp2_conf.eri_method = ERI_METHOD
    mp2_conf.use_df = (ERI_METHOD == "df")
    mp2_conf.aux_basis_name = AUX_BASIS
    mp2_conf.print_level = 0
    mp2_conf.energy_threshold = THRESHOLD
    mp2_conf.gradient_threshold = 1e-4
    mp2_conf.opt_method = "soscf" 

    try:
        # ==================== RUN MP2 ====================
        t0 = time.time()
        if mp2_algo == 'rmp2': mp2_solver = mshqc.RMP2(aligned_mol, basis, ints, scf_res, mp2_conf, pg, pl)
        elif mp2_algo == 'ump2': mp2_solver = mshqc.UMP2(aligned_mol, basis, ints, scf_res, mp2_conf, pg, pl)
        elif mp2_algo == 'omp2': mp2_solver = mshqc.OMP2(aligned_mol, basis, ints, scf_res, mp2_conf, pg, pl)
        
        mp2_res = mp2_solver.compute()
        t_mp2 = time.time() - t0

        # ==================== RUN MP3 ====================
        t1 = time.time()
        mp3_algo = mp2_algo.replace('2', '3') 
        if mp3_algo == 'rmp3': mp3_solver = mshqc.RMP3(scf_res, mp2_res, mp2_conf, ints)
        elif mp3_algo == 'ump3': mp3_solver = mshqc.UMP3(scf_res, mp2_res, mp2_conf, ints)
        elif mp3_algo == 'omp3': mp3_solver = mshqc.OMP3(scf_res, mp2_res, mp2_conf, ints)
        
        mp3_res = mp3_solver.compute()
        t_mp3 = time.time() - t1
        
        # Ekstrak energi dengan aman (antisipasi perbedaan penamaan atribut C++)
        e_mp2 = getattr(mp2_res, 'e_total', getattr(mp2_res, 'energy_total', 0.0))
        e_mp3 = getattr(mp3_res, 'e_total', getattr(mp3_res, 'energy_total', 0.0))

        return e_mp2, t_mp2, e_mp3, t_mp3
        
    except Exception as e:
        print(f"Error MSHQC MP2/MP3 pada {sys_name} ({mp2_algo}): {e}")
        return None, None, None, None

def main():
    print("=========================================================================================")
    print(f"  MSHQC ONLY | PERFORMANCE PROFILING TEST")
    print(f"  Basis: {BASIS_NAME} | Integral: {ERI_METHOD.upper()} | SCF Mode: {SCF_MODE.upper()}")
    print("=========================================================================================")

    for sys_name, atoms, chg, mult, ref in SYSTEMS:
        metode_aktif = ['rmp2', 'omp2'] if mult == 1 else ['ump2', 'omp2']
            
        for algo in metode_aktif:
            print(f"Mengeksekusi: {sys_name:<8} -> {algo.upper()} & {algo.upper().replace('2', '3')}")
            
            mp2_e, mp2_t, mp3_e, mp3_t = run_mshqc_bench(sys_name, atoms, chg, mult, ref, BASIS_NAME, algo)
            
            if mp2_e is not None:
                # Simpan hasil MP2 ke database
                RESULTS_DB.append({
                    "sys": sys_name, "method": algo.upper(),
                    "mshqc_e": mp2_e, "mshqc_t": mp2_t
                })
                # Simpan hasil MP3 ke database
                RESULTS_DB.append({
                    "sys": sys_name, "method": algo.upper().replace('2', '3'),
                    "mshqc_e": mp3_e, "mshqc_t": mp3_t
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