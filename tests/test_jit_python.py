import os
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["TBLIS_ARCH"] = "x86_64"
import mshqc
import time

def validate_jit_engine():
    print("[METRIK] Menginisialisasi Lingkungan HPC MSHQC...")
    
    # 1. Definisi Molekul Mentah (H2O)
    raw_mol = mshqc.Molecule()
    raw_mol.add_atom(8, 0.0, 0.0, 0.119262)
    raw_mol.add_atom(1, 0.0, 0.763239, -0.477047)
    raw_mol.add_atom(1, 0.0, -0.763239, -0.477047)
    
    # 2. Kalibrasi Simetri dan Ekstraksi Aligned Molecule
    pg = mshqc.PointGroup(raw_mol, 1e-6)
    pg.detect()
    
    # KUNCI RESOLUSI SIGSEGV: Gunakan aligned_mol untuk seluruh inisialisasi downstream
    aligned_mol = pg.get_aligned_molecule()
    
    # 3. Alokasi Dependensi Objek
    basis = mshqc.BasisSet("sto-3g", aligned_mol, "data/basis")
    
    pl = mshqc.PetiteList(basis, pg)
    pl.build()
    
    integrals = mshqc.IntegralEngine(aligned_mol, basis)
    
    # 4. Eksekusi SCF (Tebakan Awal)
    scf_config = mshqc.SCFConfig()
    scf_config.scf_type = "direct"
    scf_config.print_level = 0
    scf_engine = mshqc.RHF(aligned_mol, basis, integrals, pg, pl, scf_config)
    scf_result = scf_engine.compute()
    
    print(f"[METRIK] Konvergensi SCF Energi: {scf_result.energy_total:.8f} Ha")
    
    # 5. Eksekusi RMP2 (Pemicu JIT MLIR)
    mp2_config = mshqc.MP2Config()
    mp2_config.eri_method = "exact" 
    mp2_config.print_level = 0 
    
    print("[METRIK] Mengeksekusi Modul RMP2 dengan MLIR JIT Backend...")
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