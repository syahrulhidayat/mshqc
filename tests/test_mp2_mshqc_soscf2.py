import os
import sys
import multiprocessing
import threading
import psutil
import time
import gc

# ==============================================================================
# INJEKSI THREAD (Dibatasi 6 untuk menguji utilitas multicore)
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["TBLIS_ARCH"] = "x86_64"

import mshqc

# ==============================================================================
# KELAS PEMANTAU RESOURCE (Background Polling)
# ==============================================================================
class ResourceMonitor:
    def __init__(self):
        self.keep_measuring = True
        self.peak_ram = 0
        self.cpu_samples = []

    def measure(self):
        p = psutil.Process(os.getpid())
        while self.keep_measuring:
            try:
                # Menangkap RSS (Resident Set Size) RAM sebenarnya dari OS (C/C++ mem)
                self.peak_ram = max(self.peak_ram, p.memory_info().rss)
                # Di Linux, 6 Thread CPU akan menghasilkan nilai maksimal ~600%
                self.cpu_samples.append(p.cpu_percent(interval=None))
            except:
                pass
            time.sleep(0.01) # Polling super cepat (10ms) agar tidak terlewat

    def __enter__(self):
        self.keep_measuring = True
        self.peak_ram = 0
        self.cpu_samples = []
        psutil.Process(os.getpid()).cpu_percent(interval=None) # Inisialisasi
        self.thread = threading.Thread(target=self.measure)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.keep_measuring = False
        self.thread.join()

    def get_metrics(self):
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        peak_ram_mb = self.peak_ram / (1024 * 1024)
        return peak_ram_mb, avg_cpu

# ==============================================================================
# KONFIGURASI GLOBAL
# ==============================================================================
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
        return None, 0.0, 0.0, 0.0

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
        return None, 0.0, 0.0, 0.0

    mp2_conf = mshqc.MP2Config()
    mp2_conf.scf_type = SCF_MODE
    mp2_conf.eri_method = ERI_METHOD
    mp2_conf.use_df = (ERI_METHOD == "df")
    mp2_conf.aux_basis_name = AUX_BASIS
    mp2_conf.print_level = 0
    mp2_conf.energy_threshold = THRESHOLD
    mp2_conf.opt_method = "soscf"

    try:
        if mp2_algo == 'rmp2':
            mp2_solver = mshqc.RMP2(aligned_mol, basis, ints, scf_res, mp2_conf, pg, pl)
        elif mp2_algo == 'ump2':
            mp2_solver = mshqc.UMP2(aligned_mol, basis, ints, scf_res, mp2_conf, pg, pl)
        elif mp2_algo == 'omp2':
            mp2_solver = mshqc.OMP2(aligned_mol, basis, ints, scf_res, mp2_conf, pg, pl)
        
        # Mulai mengukur sumber daya tepat sebelum mesin MP2 dieksekusi
        t0 = time.time()
        with ResourceMonitor() as monitor:
            mp2_res = mp2_solver.compute()
            
        t_mp2 = time.time() - t0
        peak_ram, avg_cpu = monitor.get_metrics()
        
        return mp2_res.energy_total, t_mp2, peak_ram, avg_cpu
        
    except Exception as e:
        print(f"Error MSHQC MP2 pada {sys_name} ({mp2_algo}): {e}")
        return None, 0.0, 0.0, 0.0

def main():
    print("=========================================================================================")
    print(f"  MSHQC ONLY | PERFORMANCE PROFILING TEST")
    print(f"  Basis: {BASIS_NAME} | Integral: {ERI_METHOD.upper()} | SCF Mode: {SCF_MODE.upper()}")
    print("=========================================================================================")

    for sys_name, atoms, chg, mult, ref in SYSTEMS:
        metode_aktif = ['rmp2', 'omp2'] if mult == 1 else ['ump2', 'omp2']
            
        for algo in metode_aktif:
            print(f"Mengeksekusi: {sys_name:<8} -> {algo.upper()}")
            m_e, m_t, peak_ram, avg_cpu = run_mshqc_bench(sys_name, atoms, chg, mult, ref, BASIS_NAME, algo)
            
            if m_e is not None:
                RESULTS_DB.append({
                    "sys": sys_name, "method": algo.upper(),
                    "mshqc_e": m_e, "mshqc_t": m_t,
                    "ram": peak_ram, "cpu": avg_cpu
                })
        time.sleep(1)

    print("\n=========================================================================================")
    print(f"                               HASIL AKHIR BENCHMARK MSHQC")
    print("=========================================================================================")
    header = f"| {'System':<8} | {'Method':<8} | {'MSHQC (Ha)':<16} | {'Time (s)':<8} | {'RAM (MB)':<8} | {'CPU (%)':<7} |"
    print("-" * len(header)); print(header); print("-" * len(header))

    for row in RESULTS_DB:
        print(f"| {row['sys']:<8} | {row['method']:<8} | {row['mshqc_e']:<16.8f} | {row['mshqc_t']:<8.3f} | {row['ram']:<8.1f} | {row['cpu']:<7.1f} |")
    print("-" * len(header))

if __name__ == "__main__":
    main()
    os._exit(0)  