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
import mshqc

# ==============================================================================
# KONFIGURASI
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "6" 
BASIS_NAME = "cc-pVTZ"
AUX_BASIS  = "cc-pVTZ-RI"
THRESHOLD  = 1e-9
SCF_MODE   = "direct"
ERI_METHOD = "df" 

SYSTEMS = [
    ("Neon",     [(10, 0.0, 0.0, 0.0)], 0, 1, 'rhf'),
    ("H2O",      [(8, 0.0, 0.0, 0.2217), (1, 0.0, 1.430, -0.886), (1, 0.0, -1.430, -0.886)], 0, 1, 'rhf'),
    ("Lithium",  [(3, 0.0, 0.0, 0.0)], 0, 2, 'uhf'),
    ("Oxygen",   [(8, 0.0, 0.0, 0.0)], 0, 3, 'uhf'),
]

ATOM_MAP = {1: 'H', 3: 'Li', 6: 'C', 8: 'O', 10: 'Ne'}

def get_current_ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def run_mshqc_benchmark(sys_name, atoms, chg, mult, ref, fam):
    """Benchmark murni untuk MSHQC (MP2 & MP3)"""
    raw_mol = mshqc.Molecule()
    for z, x, y, z_c in atoms: raw_mol.add_atom(z, x, y, z_c)
    raw_mol.set_charge(chg); raw_mol.set_multiplicity(mult)
    
    pg = mshqc.PointGroup(raw_mol)
    pg.detect()
    aligned_mol = pg.get_aligned_molecule()

    basis = mshqc.BasisSet(BASIS_NAME, aligned_mol)
    ints = mshqc.IntegralEngine(aligned_mol, basis)
    pl = mshqc.PetiteList(basis, pg); pl.build()

    scf_conf = mshqc.SCFConfig()
    scf_conf.eri_method = ERI_METHOD
    if ERI_METHOD == "df":
        scf_conf.use_df = True
        scf_conf.aux_basis_name = AUX_BASIS

    n_elec = aligned_mol.n_electrons() - chg
    n_a = (n_elec + mult - 1) // 2
    n_b = n_elec - n_a

    # SCF
    if ref == 'rhf': solver = mshqc.RHF(aligned_mol, basis, ints, pg, pl, scf_conf)
    else: solver = mshqc.UHF(aligned_mol, basis, ints, pg, pl, n_a, n_b, scf_conf)
    scf_res = solver.compute()

    # MPn Config
    mp_conf = mshqc.MP2Config()
    mp_conf.use_df = (ERI_METHOD == "df")
    mp_conf.aux_basis_name = AUX_BASIS

    # Execution
    gc.collect()
    mem_start = get_current_ram_mb()
    t0 = time.time()
    
    # Pilih Solver
    if fam == 'RMP': 
        mp2 = mshqc.RMP2(aligned_mol, basis, ints, scf_res, mp_conf, pg, pl)
        mp3 = mshqc.RMP3(scf_res, mp2.compute(), mp_conf, ints)
    else: 
        mp2 = mshqc.UMP2(aligned_mol, basis, ints, scf_res, mp_conf, pg, pl)
        mp3 = mshqc.UMP3(scf_res, mp2.compute(), mp_conf, ints)
    
    res_mp3 = mp3.compute()
    t_total = time.time() - t0
    ram_usage = get_current_ram_mb() - mem_start
    
    return res_mp3.e_total, t_total, ram_usage

def main():
    print(f"{'MSHQC MP3 Benchmark':^60}")
    print("="*60)
    print(f"{'System':<10} | {'Method':<6} | {'Energy (Ha)':<15} | {'Time (s)':<8} | {'RAM (MB)':<8}")
    print("-"*60)

    for sys_name, atoms, chg, mult, ref in SYSTEMS:
        fam = 'RMP' if mult == 1 else 'UMP'
        try:
            e, t, ram = run_mshqc_benchmark(sys_name, atoms, chg, mult, ref, fam)
            print(f"{sys_name:<10} | {fam+'3':<6} | {e:<15.8f} | {t:<8.3f} | {ram:<8.2f}")
        except Exception as e:
            print(f"{sys_name:<10} | {fam+'3':<6} | {'FAILED':<15} | {'-':<8} | {'-':<8}")

if __name__ == "__main__":
    main()