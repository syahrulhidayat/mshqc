# File: /home/syahrul/mshqc/bench_li_caspt3.py
import time
import os
import sys
import mshqc

# --- KONFIGURASI BENCHMARK ---
TARGET_FROZEN = 1  # 1s (Core)
TARGET_ACTIVE = 4  # 2s + 2p (Valence)
CHOLESKY_THRESH = 1e-8

def run_benchmark(basis_name):
    print(f"\n{'='*80}")
    print(f"  RUNNING BASIS SET: {basis_name}")
    print(f"  Config: Frozen={TARGET_FROZEN}, Active={TARGET_ACTIVE}")
    print(f"{'-'*80}")
    
    results = {'name': basis_name}
    
    # 1. SETUP MOLECULE & BASIS
    # -------------------------
    try:
        mol = mshqc.Molecule()
        mol.add_atom(3, 0.0, 0.0, 0.0) # Li
        mol.set_multiplicity(2)        # Doublet
        mol.set_charge(0)
        
        # Auto-detect basis path
        basis_dir = os.path.join(os.environ.get('MSHQC_DATA_DIR', 'data'), 'basis')
        basis = mshqc.BasisSet(basis_name, mol, basis_dir)
        # Gunakan urutan langsung tanpa nama parameter# Install dependencies via conda-forge

        integrals = mshqc.IntegralEngine(mol, basis)
        
        n_basis = basis.n_basis_functions()
        results['n_basis'] = n_basis
        print(f"  > Basis Functions: {n_basis}")
        
    except Exception as e:
        print(f"  [ERROR] Init failed: {e}")
        return None

    # 2. CHOLESKY DECOMPOSITION
    # -------------------------
    print("  > [1/6] Cholesky Decomposition... ", end="", flush=True)
    t0 = time.time()
    
    chol_engine = mshqc.CholeskyERI(CHOLESKY_THRESH)
    eri_tensor = integrals.compute_eri()
    chol_engine.decompose(eri_tensor)
    L_vectors = chol_engine.get_L_vectors()
    
    results['t_chol'] = time.time() - t0
    print(f"Done ({results['t_chol']:.4f}s)")

    # 3. UHF CALCULATION
    # ------------------
    print("  > [2/6] UHF Calculation... ", end="", flush=True)
    t0 = time.time()
    
    uhf_config = mshqc.CholeskyUHFConfig()
    uhf_config.cholesky_threshold = CHOLESKY_THRESH
    uhf_config.print_level = 0
    
    # N_alpha=2, N_beta=1 for Li
    uhf = mshqc.CholeskyUHF(mol, basis, integrals, 2, 1, uhf_config)
    uhf.set_cholesky_vectors(L_vectors)
    uhf_res = uhf.compute()
    
    results['t_uhf'] = time.time() - t0
    results['e_uhf'] = uhf_res.energy_total
    print(f"Done ({results['t_uhf']:.4f}s)")

    # 4. UNO GENERATION
    # -----------------
    print("  > [3/6] UNO Generation... ", end="", flush=True)
    uno_gen = mshqc.CholeskyUNO(uhf_res, integrals, n_basis)
    uno_res = uno_gen.compute()
    print("Done.")

    # 5. SS-CASSCF
    # ------------
    print("  > [4/6] SS-CASSCF... ", end="", flush=True)
    t0 = time.time()
    
    active_space = mshqc.ActiveSpace.CAS_Frozen(TARGET_FROZEN, TARGET_ACTIVE, n_basis, 3)
    
    sa_config = mshqc.SACASConfig()
    sa_config.set_equal_weights(1) # 1 State = SS-CASSCF
    sa_config.max_iter = 50
    sa_config.print_level = 0
    
    casscf = mshqc.CholeskySACASSCF(mol, basis, integrals, active_space, sa_config, L_vectors)
    casscf_res = casscf.compute(uno_res.C_uno)
    
    results['t_cas'] = time.time() - t0
    results['e_cas'] = casscf_res.state_energies[0]
    print(f"Done ({results['t_cas']:.4f}s)")

    # 6. SS-CASPT2
    # ------------
    print("  > [5/6] SS-CASPT2... ", end="", flush=True)
    t0 = time.time()
    
    pt2_config = mshqc.CASPT2Config()
    pt2_config.shift = 0.0
    
    pt2 = mshqc.CholeskySACASPT2(casscf_res, L_vectors, n_basis, active_space, pt2_config)
    pt2_out = pt2.compute()
    
    results['t_pt2'] = time.time() - t0
    results['e_pt2'] = pt2_out.e_pt2[0]
    print(f"Done ({results['t_pt2']:.4f}s)")

    # 7. SS-CASPT3
    # ------------
    print("  > [6/6] SS-CASPT3... ", end="", flush=True)
    t0 = time.time()
    
    pt3_config = mshqc.CASPT3Config()
    pt3_config.shift = 0.0
    
    pt3 = mshqc.CholeskySACASPT3(casscf_res, L_vectors, n_basis, active_space, pt3_config)
    pt3_out = pt3.compute()
    
    results['t_pt3'] = time.time() - t0
    results['e_pt3'] = pt3_out.e_pt3[0]
    
    # Total Energy Calculation
    results['e_total'] = results['e_cas'] + results['e_pt2'] + results['e_pt3']
    print(f"Done ({results['t_pt3']:.4f}s)")
    
    return results

def main():
    print("="*100)
    print("  MSHQC PYTHON BENCHMARK: Li SS-CASPT3")
    print("="*100)

    # Daftar Basis Set (Sesuai kode C++)
    basis_list = [
        "cc-pCVTZ",
        "cc-pCVQZ",
        # "cc-pCV5Z",     # Uncomment jika RAM > 16GB
        "aug-cc-pCVTZ",
         "aug-cc-pCVQZ",
          "cc-pCV5Z"                # Uncomment jika ingin test lebih berat
    ]

    all_results = []
    
    # Loop Benchmark
    for basis in basis_list:
        res = run_benchmark(basis)
        if res:
            all_results.append(res)
    
    # Sort by N Basis
    all_results.sort(key=lambda x: x['n_basis'])

    # --- FINAL TABLE ---
    print("\n\n")
    print("="*120)
    print("  FINAL SUMMARY: Lithium Ground State")
    print("="*120)
    
    # Header
    header = (f"  {'Basis Set':<15} | {'NBas':>4} | {'E(Total)':>14} | "
              f"{'E(PT2)':>10} | {'E(PT3)':>10} | "
              f"{'T(Chol)':>7} | {'T(UHF)':>7} | {'T(CAS)':>7} | {'T(PT2)':>7} | {'T(PT3)':>7} | {'Tot(s)':>7}")
    print(header)
    print("-" * 120)

    for r in all_results:
        t_total = r['t_chol'] + r['t_uhf'] + r['t_cas'] + r['t_pt2'] + r['t_pt3']
        
        row = (f"  {r['name']:<15} | {r['n_basis']:>4} | {r['e_total']:>14.8f} | "
               f"{r['e_pt2']:>10.6f} | {r['e_pt3']:>10.6f} | "
               f"{r['t_chol']:>7.3f} | {r['t_uhf']:>7.3f} | {r['t_cas']:>7.3f} | "
               f"{r['t_pt2']:>7.3f} | {r['t_pt3']:>7.3f} | {t_total:>7.3f}")
        print(row)
        
    print("="*120)

if __name__ == "__main__":
    main()