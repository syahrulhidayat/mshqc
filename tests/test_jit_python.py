import os
import time
import mshqc

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["TBLIS_ARCH"] = "x86_64"

def validate_jit_engine():
    print("[METRIK] Menginisialisasi Lingkungan HPC MSHQC...")
    
    ANG_TO_BOHR = 1.88972612462577
    
    # 1. Definisi Molekul (Terkoreksi ke Satuan Bohr)
    raw_mol = mshqc.Molecule()
    raw_mol.add_atom(8, 0.0, 0.0, 0.119262 * ANG_TO_BOHR)
    raw_mol.add_atom(1, 0.0, 0.763239 * ANG_TO_BOHR, -0.477047 * ANG_TO_BOHR)
    raw_mol.add_atom(1, 0.0, -0.763239 * ANG_TO_BOHR, -0.477047 * ANG_TO_BOHR)
    
    # 2. Kalibrasi Simetri
    pg = mshqc.PointGroup(raw_mol, 1e-6)
    pg.detect()
    aligned_mol = pg.get_aligned_molecule()
    
    # 3. Alokasi Dependensi (Sinkronisasi dengan Benchmark cc-pVTZ)
    basis = mshqc.BasisSet("cc-pVTZ", aligned_mol, "data/basis")
    pl = mshqc.PetiteList(basis, pg)
    pl.build()
    integrals = mshqc.IntegralEngine(aligned_mol, basis)
    
    # 4. Eksekusi SCF
    scf_config = mshqc.SCFConfig()
    scf_config.scf_type = "direct"
    scf_config.eri_method = "df" 
    scf_config.use_df = True    
    scf_config.aux_basis_name = "cc-pVTZ-RI" 
    scf_config.print_level = 0
    
    scf_engine = mshqc.RHF(aligned_mol, basis, integrals, pg, pl, scf_config)
    scf_result = scf_engine.compute()
    print(f"[METRIK] Konvergensi SCF Energi: {scf_result.energy_total:.8f} Ha")
    
    # 5. Eksekusi RMP2
    mp2_config = mshqc.MP2Config()
    mp2_config.eri_method = "df" 
    mp2_config.use_df = True
    mp2_config.aux_basis_name = "cc-pVTZ-RI"
    mp2_config.print_level = 0 
    
    start_time = time.perf_counter()
    mp2_engine = mshqc.RMP2(aligned_mol, basis, integrals, scf_result, mp2_config, pg, pl)
    mp2_result = mp2_engine.compute()
    end_time = time.perf_counter()
    
    print(f"\n[METRIK] Total Energi RMP2: {mp2_result.energy_total:.8f} Ha")
    print(f"[METRIK] Energi Korelasi: {mp2_result.energy_mp2_corr:.8f} Ha")
    print(f"[METRIK] Waktu Eksekusi Python-to-C++: {end_time - start_time:.4f} detik")

if __name__ == "__main__":
    validate_jit_engine()
    os._exit(0)