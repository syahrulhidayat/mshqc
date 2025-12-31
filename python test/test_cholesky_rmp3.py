import time
import numpy as np
import os
import mshqc

# ==============================================================================
# HELPER FUNCTIONS (Extended for MP3)
# ==============================================================================
def print_header(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def angstrom_to_bohr(val):
    return val * 1.8897259886

def analyze_results(rhf, mp2, mp3, label):
    """Analisis Komprehensif RHF -> MP2 -> MP3"""
    print("-" * 60)
    print(f" FINAL ANALYSIS: {label}")
    print("-" * 60)
    
    # 1. RHF
    print(f" [SCF] Energy       : {rhf.energy_total:14.8f} Ha")
    
    # 2. MP2
    print(f" [MP2] Correlation  : {mp2.e_corr:14.8f} Ha")
    print(f" [MP2] Total        : {mp2.e_total:14.8f} Ha")
    
    # 3. MP3
    print(f" [MP3] Correction   : {mp3.e_mp3:14.8f} Ha")
    print(f" [MP3] Total Corr   : {mp3.e_mp2 + mp3.e_mp3:14.8f} Ha")
    print(f" [MP3] Total Energy : {mp3.e_total:14.8f} Ha")
    
    # 4. Check Physics (MP3 correction is usually smaller than MP2)
    ratio = abs(mp3.e_mp3 / mp2.e_corr)
    print(f" [CHK] MP3/MP2 Ratio: {ratio:.2%}")
    if ratio < 0.5:
        print(" [PASS] Convergence behavior looks normal (MP3 < MP2).")
    else:
        print(" [WARN] Perturbation series might be oscillating or slow converging.")

# ==============================================================================
# TEST 1: NEON ATOM
# ==============================================================================
def test_neon_full_pipeline():
    print_header("TEST 1: Neon Atom (Ne) - Full RHF -> RMP2 -> RMP3 Pipeline")
    
    mol = mshqc.Molecule()
    mol.add_atom(10, 0.0, 0.0, 0.0) 
    mol.set_charge(0)
    mol.set_multiplicity(1) 
    
    basis_name = "cc-pVTZ"
    basis_dir = os.environ.get("MSHQC_DATA_DIR", "data") + "/basis"
    basis = mshqc.BasisSet(basis_name, mol, basis_dir)
    integrals = mshqc.IntegralEngine(mol, basis)
    
    # 1. Decompose
    print(" > [Step 1] Decomposing Integrals...")
    t0 = time.time()
    chol = mshqc.CholeskyERI(1e-6)
    chol.decompose(integrals.compute_eri())
    print(f"   Vectors: {chol.n_vectors()} (Time: {time.time()-t0:.4f}s)")
    
    # 2. RHF
    print("\n > [Step 2] Running Cholesky-RHF...")
    conf = mshqc.CholeskyRHFConfig()
    conf.print_level = 0
    rhf = mshqc.CholeskyRHF(mol, basis, integrals, conf, chol)
    rhf_res = rhf.compute()
    print(f"   Done RHF. E = {rhf_res.energy_total:.8f} Ha")
    
    # 3. RMP2 (Generates T2 and passes vectors)
    print("\n > [Step 3] Running Cholesky-RMP2 (Prepares T2(1))...")
    mp2_conf = mshqc.CholeskyRMP2Config()
    mp2_conf.print_level = 0
    rmp2 = mshqc.CholeskyRMP2(mol, basis, integrals, rhf_res, mp2_conf, chol)
    mp2_res = rmp2.compute()
    print(f"   Done RMP2. E_Corr = {mp2_res.e_corr:.8f} Ha")

    # 4. RMP3 (THE NEW PART)
    print("\n > [Step 4] Running Cholesky-RMP3 (Reuse Vectors from Step 3)...")
    # Perhatikan: Kita passing mp2_res yang sekarang berisi vektor Cholesky!
    t0 = time.time()
    rmp3 = mshqc.CholeskyRMP3(rhf_res, mp2_res, basis)
    mp3_res = rmp3.compute()
    print(f"   Done RMP3 ({time.time()-t0:.4f} s).")
    
    analyze_results(rhf_res, mp2_res, mp3_res, "Neon Atom")

# ==============================================================================
# TEST 2: WATER MOLECULE (H2O)
# ==============================================================================
def test_water_full_pipeline():
    print_header("TEST 2: Water (H2O) - Full Pipeline Benchmark")
    
    mol = mshqc.Molecule()
    mol.add_atom(8, 0.0, 0.0, 0.0)
    mol.add_atom(1, 0.0, -1.430, 1.107) # Geometry in Bohr approx
    mol.add_atom(1, 0.0, 1.430, 1.107)
    
    mol.set_charge(0)
    mol.set_multiplicity(1)
    
    basis_name = "cc-pVDZ"
    basis_dir = os.environ.get("MSHQC_DATA_DIR", "data") + "/basis"
    basis = mshqc.BasisSet(basis_name, mol, basis_dir)
    integrals = mshqc.IntegralEngine(mol, basis)
    
    # 1. Decompose
    print(" > [Step 1] Decomposing Integrals (Threshold=1e-5)...")
    t0 = time.time()
    chol = mshqc.CholeskyERI(1e-5)
    chol.decompose(integrals.compute_eri())
    print(f"   Vectors: {chol.n_vectors()} (Time: {time.time()-t0:.4f}s)")
    
    # 2. RHF
    print("\n > [Step 2] Running Cholesky-RHF...")
    rhf = mshqc.CholeskyRHF(mol, basis, integrals, mshqc.CholeskyRHFConfig(), chol)
    rhf_res = rhf.compute()
    print(f"   Done RHF. E = {rhf_res.energy_total:.8f} Ha")
    
    # 3. RMP2
    print("\n > [Step 3] Running Cholesky-RMP2...")
    mp2_conf = mshqc.CholeskyRMP2Config()
    mp2_conf.print_level = 0
    rmp2 = mshqc.CholeskyRMP2(mol, basis, integrals, rhf_res, mp2_conf, chol)
    mp2_res = rmp2.compute()
    print(f"   Done RMP2. E_Corr = {mp2_res.e_corr:.8f} Ha")
    
    # 4. RMP3
    print("\n > [Step 4] Running Cholesky-RMP3...")
    t0 = time.time()
    
    # Logika Reuse Vector terjadi di sini. 
    # Constructor CholeskyRMP3 menerima mp2_res yang membawa vektor Cholesky.
    # Tidak ada ERI calculation yang terjadi di sini.
    rmp3 = mshqc.CholeskyRMP3(rhf_res, mp2_res, basis)
    mp3_res = rmp3.compute()
    
    print(f"   Done RMP3 ({time.time()-t0:.4f} s).")
    
    analyze_results(rhf_res, mp2_res, mp3_res, "Water (H2O)")

if __name__ == "__main__":
    try:
        test_neon_full_pipeline()
        test_water_full_pipeline()
        print("\n" + "="*80)
        print(" SUCCESS: Cholesky-RMP3 pipeline validated.")
        print(" All steps (Decomp -> RHF -> RMP2 -> RMP3) completed.")
        print("="*80)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Test crashed: {e}")
        import traceback
        traceback.print_exc()