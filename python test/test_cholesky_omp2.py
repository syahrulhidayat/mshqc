import mshqc
from mshqc.calculators import MSHQCCalculator
import time
import os

def print_header(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def test_water_omp3_pipeline():
    """
    TEST 1: Water Molecule (H2O)
    Pipeline: Cholesky-ROHF -> Cholesky-OMP2 -> Cholesky-OMP3
    Goal: Validasi alur orbital optimization bertingkat.
    """
    print_header("TEST 1: H2O | Flow: Chol-ROHF -> OMP2 -> OMP3")
    
    # 1. Setup Molekul & Basis
    mol = mshqc.Molecule(0, 1)
    mol.add_atom(8, 0.00000000,  0.00000000,  0.11730000) # O
    mol.add_atom(1, 0.00000000,  0.75720000, -0.46920000) # H
    mol.add_atom(1, 0.00000000, -0.75720000, -0.46920000) # H
    
    basis_name = "cc-pVTZ"
    print(f"System: H2O | Basis: {basis_name}")
    
    calc = MSHQCCalculator(mol, basis_name)
    
    # 2. Decompose Cholesky (Manual untuk Reuse)
    print("\n[Step 0] Pre-computing Cholesky Vectors...")
    threshold = 1e-5
    chol_vectors = mshqc.CholeskyERI(threshold)
    t0 = time.time()
    eri = calc.integrals.compute_eri()
    chol_vectors.decompose(eri)
    print(f"  > Decomposed {chol_vectors.n_vectors()} vectors in {time.time()-t0:.4f} s")

    # 3. Reference SCF (ROHF)
    print("\n[Step 1] Running Cholesky-ROHF...")
    rohf_config = mshqc.CholeskyROHFConfig()
    rohf_config.cholesky_threshold = threshold
    rohf_config.print_level = 0
    
    t0 = time.time()
    scf_res = calc.run_cholesky_rohf(config=rohf_config, existing_cholesky=chol_vectors)
    print(f"  > SCF Energy: {scf_res.energy_total:.8f} Ha ({time.time()-t0:.4f} s)")

    # 4. OMP2 (Orbital Opt Level 1)
    print("\n[Step 2] Running Cholesky-OMP2...")
    omp2_config = mshqc.CholeskyOMP2Config()
    omp2_config.max_iterations = 20
    omp2_config.energy_threshold = 1e-6
    omp2_config.cholesky_threshold = threshold
    omp2_config.print_level = 0 # Minimal output
    
    t0 = time.time()
    omp2_res = calc.run_cholesky_omp2(
        scf_res, 
        config=omp2_config, 
        existing_cholesky=chol_vectors
    )
    print(f"  > OMP2 Energy: {omp2_res.energy_total:.8f} Ha ({time.time()-t0:.4f} s)")
    print(f"  > OMP2 Converged: {omp2_res.converged}")

    # 5. OMP3 (Orbital Opt Level 2)
    #    Menggunakan hasil OMP2 (orbital yang sudah dioptimasi) sebagai tebakan awal
    print("\n[Step 3] Running Cholesky-OMP3...")
    omp3_config = mshqc.CholeskyOMP3Config()
    omp3_config.max_iterations = 20
    omp3_config.energy_threshold = 1e-6
    omp3_config.cholesky_threshold = threshold
    omp3_config.print_level = 1 # Tampilkan iterasi
    
    t0 = time.time()
    omp3_res = calc.run_cholesky_omp3(
        omp2_res,  # Pass result OMP2 di sini!
        config=omp3_config,
        existing_cholesky=chol_vectors
    )
    
    print("-" * 40)
    print(f"OMP3 Done in {time.time()-t0:.4f} s")
    print(f"Final OMP3 Energy: {omp3_res.energy_total:.8f} Ha")
    print("-" * 40)

    if omp3_res.converged:
        print("[PASS] OMP3 Pipeline Converged Successfully.")
    else:
        print("[FAIL] OMP3 did not converge.")


def test_neon_full_reuse_omp3():
    """
    TEST 2: Neon Atom (Heavy Basis)
    Tujuan: Menunjukkan efisiensi reuse vector pada basis besar.
    """
    print_header("TEST 2: Neon (Ne) | High-Level Reuse Pipeline")
    
    mol = mshqc.Molecule(0, 1)
    mol.add_atom(10, 0.0, 0.0, 0.0)
    
    basis_name = "cc-pCVQZ"
    print(f"System: Ne Atom | Basis: {basis_name}")
    calc = MSHQCCalculator(mol, basis_name)
    
    # --- 1. Decompose ---
    print("\n[Step 1] Cholesky Decomposition...")
    chol = mshqc.CholeskyERI(1e-5)
    t0 = time.time()
    chol.decompose(calc.integrals.compute_eri())
    t_decomp = time.time() - t0
    print(f"  > Vectors: {chol.n_vectors()} | Time: {t_decomp:.4f} s")
    
    # --- 2. SCF ---
    print("\n[Step 2] SCF (Reuse Vectors)...")
    t0 = time.time()
    scf_res = calc.run_cholesky_rohf(
        config=mshqc.CholeskyROHFConfig(), 
        existing_cholesky=chol
    )
    t_scf = time.time() - t0
    print(f"  > SCF Time: {t_scf:.4f} s")
    
    # --- 3. OMP2 ---
    print("\n[Step 3] OMP2 (Reuse Vectors)...")
    omp2_conf = mshqc.CholeskyOMP2Config()
    omp2_conf.print_level = 0
    t0 = time.time()
    omp2_res = calc.run_cholesky_omp2(scf_res, config=omp2_conf, existing_cholesky=chol)
    t_omp2 = time.time() - t0
    print(f"  > OMP2 Time: {t_omp2:.4f} s | E = {omp2_res.energy_total:.8f} Ha")
    
    # --- 4. OMP3 ---
    print("\n[Step 4] OMP3 (Reuse Vectors)...")
    omp3_conf = mshqc.CholeskyOMP3Config()
    omp3_conf.max_iterations = 20
    t0 = time.time()
    omp3_res = calc.run_cholesky_omp3(omp2_res, config=omp3_conf, existing_cholesky=chol)
    t_omp3 = time.time() - t0
    
    print("\n--- Efficiency Summary ---")
    print(f"Decomposition : {t_decomp:.4f} s (One-time cost)")
    print(f"SCF Init Cost : ~0.0 s")
    print(f"OMP2 Init Cost: ~0.0 s")
    print(f"OMP3 Init Cost: ~0.0 s")
    print(f"Total Compute : {t_scf + t_omp2 + t_omp3:.4f} s")
    
    print("\nFinal Energies:")
    print(f"  E(SCF) : {scf_res.energy_total:.8f}")
    print(f"  E(OMP2): {omp2_res.energy_total:.8f}")
    print(f"  E(OMP3): {omp3_res.energy_total:.8f}")

if __name__ == "__main__":
    try:
        test_water_omp3_pipeline()
        test_neon_full_reuse_omp3()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()