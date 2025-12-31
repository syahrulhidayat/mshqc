import mshqc
import time
import sys

def main():
    print("=================================================================")
    print("  MSHQC BENCHMARK: STANDARD vs CHOLESKY (Li Atom)")
    print("=================================================================")

    # 1. SETUP SYSTEM
    mol = mshqc.Molecule()
    mol.add_atom(3, 0.0, 0.0, 0.0) # Li
    mol.set_multiplicity(2)        # Doublet
    mol.set_charge(0)

    # Gunakan basis set besar agar terasa bedanya
    basis_name = "cc-pVQZ"
    print(f"  Molecule: Lithium (Li)")
    print(f"  Basis   : {basis_name}")

    try:
        basis = mshqc.BasisSet(basis_name, mol)
    except Exception as e:
        print(f"Error loading basis {basis_name}: {e}")
        return

    integrals = mshqc.IntegralEngine(mol, basis)
    nbasis = basis.n_basis_functions()
    
    # Li (3e) -> 2 alpha, 1 beta
    n_alpha = 2
    n_beta = 1
    
    print(f"  N Basis : {nbasis}\n")

    # ========================================================================
    # PART A: STANDARD METHODS (FULL ERI)
    # ========================================================================
    print(">>> RUNNING STANDARD METHODS (Full O(N^4) Integrals)...")

    # A1. Standard UHF
    print("  1. Standard UHF... ", end="", flush=True)
    uhf_cfg = mshqc.SCFConfig()
    uhf_cfg.print_level = 0 # Silent
    
    uhf_std = mshqc.UHF(mol, basis, integrals, n_alpha, n_beta, uhf_cfg)
    
    t0 = time.time()
    res_uhf_std = uhf_std.compute()
    t_std_uhf = time.time() - t0
    
    e_std_uhf = res_uhf_std.energy_total
    print(f"Done ({t_std_uhf:.4f}s)")

    # A2. Standard UMP2
    print("  2. Standard UMP2... ", end="", flush=True)
    ump2_std = mshqc.UMP2(res_uhf_std, basis, integrals)
    
    t0 = time.time()
    res_ump2_std = ump2_std.compute()
    t_std_mp2 = time.time() - t0
    
    e_std_mp2 = res_ump2_std.e_corr_total
    print(f"Done ({t_std_mp2:.4f}s)")

    # A3. Standard UMP3
    print("  3. Standard UMP3... ", end="", flush=True)
    # Standard UMP3 butuh hasil UMP2
    ump3_std = mshqc.UMP3(res_uhf_std, res_ump2_std, basis, integrals)
    
    t0 = time.time()
    res_ump3_std = ump3_std.compute()
    t_std_mp3 = time.time() - t0
    
    # PERHATIAN: Di C++ struct membernya 'e_mp3', di Python binding 'e_mp3_corr'
    e_std_mp3 = res_ump3_std.e_mp3_corr 
    e_std_total = res_ump3_std.e_total
    print(f"Done ({t_std_mp3:.4f}s)")


    # ========================================================================
    # PART B: CHOLESKY METHODS (APPROXIMATE)
    # ========================================================================
    print("\n>>> RUNNING CHOLESKY METHODS (Reusable Vectors)...")
    threshold = 1e-6

    # B0. Decomposition (Once)
    print("  0. Decomposing Integrals... ", end="", flush=True)
    t0 = time.time()
    
    chol_obj = mshqc.CholeskyERI(threshold)
    eri_tensor = integrals.compute_eri()
    chol_obj.decompose(eri_tensor)
    
    # Ambil vektor untuk di-inject ke UHF
    L_vecs = chol_obj.get_L_vectors()
    
    t_chol_decomp = time.time() - t0
    print(f"Done ({t_chol_decomp:.4f}s, {len(L_vecs)} vecs)")

    # B1. Cholesky UHF
    print("  1. Cholesky UHF... ", end="", flush=True)
    c_uhf_cfg = mshqc.CholeskyUHFConfig()
    c_uhf_cfg.cholesky_threshold = threshold
    c_uhf_cfg.print_level = 0
    
    uhf_chol = mshqc.CholeskyUHF(mol, basis, integrals, n_alpha, n_beta, c_uhf_cfg)
    uhf_chol.set_cholesky_vectors(L_vecs) # INJECT: Hindari hitung ulang!
    
    t0 = time.time()
    res_uhf_chol = uhf_chol.compute()
    t_chol_uhf = time.time() - t0
    
    e_chol_uhf = res_uhf_chol.energy_total
    print(f"Done ({t_chol_uhf:.4f}s)")

    # B2. Cholesky UMP2
    print("  2. Cholesky UMP2... ", end="", flush=True)
    c_mp2_cfg = mshqc.CholeskyUMP2Config()
    c_mp2_cfg.cholesky_threshold = threshold
    c_mp2_cfg.print_level = 0

    # Pass object chol_obj (Python wrapper untuk CholeskyERI)
    ump2_chol = mshqc.CholeskyUMP2(res_uhf_chol, basis, integrals, c_mp2_cfg, chol_obj)
    
    t0 = time.time()
    res_ump2_chol = ump2_chol.compute()
    t_chol_mp2 = time.time() - t0
    
    e_chol_mp2 = res_ump2_chol.e_corr_total
    print(f"Done ({t_chol_mp2:.4f}s)")

    # B3. Cholesky UMP3
    print("  3. Cholesky UMP3... ", end="", flush=True)
    c_mp3_cfg = mshqc.CholeskyUMP3Config()
    c_mp3_cfg.cholesky_threshold = threshold
    c_mp3_cfg.print_level = 0

    # Reuse UMP2 object
    ump3_chol = mshqc.CholeskyUMP3(ump2_chol, c_mp3_cfg)
    
    t0 = time.time()
    res_ump3_chol = ump3_chol.compute()
    t_chol_mp3 = time.time() - t0
    
    e_chol_mp3 = res_ump3_chol.e_mp3_total
    e_chol_total = res_uhf_chol.energy_total + e_chol_mp2 + e_chol_mp3
    print(f"Done ({t_chol_mp3:.4f}s)")


    # ========================================================================
    # PART C: COMPARISON TABLES
    # ========================================================================
    print("\n")
    print("="*74)
    print(f"  FINAL COMPARISON RESULTS ({basis_name})")
    print("="*74)
    
    # Table 1: Energy
    print("  [ACCURACY] Energy Comparison (Hartree)")
    print("  " + "-"*70)
    print(f"  {'Method':<10}| {'Standard (Ref)':<20} | {'Cholesky (Approx)':<20} | {'Difference':<15}")
    print("  " + "-"*70)
    
    def print_row(name, std, chol):
        diff = abs(std - chol)
        print(f"  {name:<10}| {std:<20.8f} | {chol:<20.8f} | {diff:<15.2e}")

    print_row("UHF", e_std_uhf, e_chol_uhf)
    print_row("MP2 Corr", e_std_mp2, e_chol_mp2)
    print_row("MP3 Corr", e_std_mp3, e_chol_mp3)
    print("  " + "-"*70)
    print_row("TOTAL", e_std_total, e_chol_total)
    print("  " + "-"*70 + "\n")

    # Table 2: Timing
    print("  [PERFORMANCE] Time Comparison (Seconds)")
    print("  " + "-"*60)
    print(f"  {'Step':<10}| {'Standard':<12} | {'Cholesky':<12} | {'Speedup':<10}")
    print("  " + "-"*60)

    def print_time(name, t_std, t_chol):
        speedup = t_std / t_chol if t_chol > 0 else 0
        print(f"  {name:<10}| {t_std:<12.4f} | {t_chol:<12.4f} | {speedup:<10.2f} x")

    print_time("UHF", t_std_uhf, t_chol_uhf)
    print_time("MP2", t_std_mp2, t_chol_mp2)
    print_time("MP3", t_std_mp3, t_chol_mp3)
    
    t_total_std = t_std_uhf + t_std_mp2 + t_std_mp3
    t_total_chol = t_chol_decomp + t_chol_uhf + t_chol_mp2 + t_chol_mp3
    
    print("  " + "-"*60)
    print(f"  {'Decomp':<10}| {'-':<12} | {t_chol_decomp:<12.4f} | {'-':<10}")
    print("  " + "-"*60)
    print_time("TOTAL", t_total_std, t_total_chol)
    print("  " + "-"*60)

if __name__ == "__main__":
    main()