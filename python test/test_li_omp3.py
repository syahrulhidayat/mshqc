# File: bench_omp_period2.py
import mshqc
import time
import sys

# --- KONFIGURASI ATOM ---
# Konfigurasi elektron (Alpha/Beta) untuk Ground State
ATOM_CONFIGS = {
    'He': {'z': 2, 'na': 1, 'nb': 1, 'mult': 1}, # Singlet (Closed)
    'Li': {'z': 3, 'na': 2, 'nb': 1, 'mult': 2}, # Doublet (Open)
    'Be': {'z': 4, 'na': 2, 'nb': 2, 'mult': 1}, # Singlet (Closed, quasi-degenerate)
    'B':  {'z': 5, 'na': 3, 'nb': 2, 'mult': 2}, # Doublet (Open)
    'C':  {'z': 6, 'na': 4, 'nb': 2, 'mult': 3}, # Triplet (Open - High Spin)
    'N':  {'z': 7, 'na': 5, 'nb': 2, 'mult': 4}, # Quartet (Open - High Spin)
    'O':  {'z': 8, 'na': 5, 'nb': 3, 'mult': 3}, # Triplet (Open)
    'F':  {'z': 9, 'na': 5, 'nb': 4, 'mult': 2}, # Doublet (Open)
}

def run_omp_benchmark(atom_label, basis_name="cc-pVQZ"):
    cfg = ATOM_CONFIGS[atom_label]
    
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: Atom {atom_label} (Z={cfg['z']}) | Basis: {basis_name}")
    print(f"  State: N_alpha={cfg['na']}, N_beta={cfg['nb']} (Mult={cfg['mult']})")
    print(f"{'='*60}")

    results = {'atom': atom_label}

    # 1. SETUP
    try:
        mol = mshqc.Molecule()
        mol.add_atom(cfg['z'], 0.0, 0.0, 0.0)
        
        # Load Basis
        basis = mshqc.BasisSet(basis_name, mol)
        results['n_basis'] = basis.n_basis_functions()
        print(f"  > Basis Functions: {results['n_basis']}")

        # Compute Integrals
        t0 = time.time()
        integrals = mshqc.IntegralEngine(mol, basis)
        print(f"  > Integrals computed ({time.time()-t0:.4f} s)")

        # SCF Config
        scf_config = mshqc.SCFConfig()
        scf_config.energy_threshold = 1e-10
        scf_config.print_level = 0
        
    except Exception as e:
        print(f"  [ERROR] Setup failed: {e}")
        return None

    # 2. ROHF (Restricted Open-Shell HF)
    print("  > [1/3] Running ROHF...", end="", flush=True)
    t0 = time.time()
    try:
        rohf = mshqc.ROHF(mol, basis, cfg['na'], cfg['nb'], scf_config)
        rohf_res = rohf.run()
        results['e_rohf'] = rohf_res.energy_total
        results['t_rohf'] = time.time() - t0
        print(f" Done. E = {results['e_rohf']:.8f} Ha ({results['t_rohf']:.4f} s)")
    except Exception as e:
        print(f" FAILED: {e}")
        return None

    # 3. OMP2 (Orbital-Optimized MP2)
    # Note: max_iter=1 berarti "ROHF-MP2" (non-iterative orbitals). 
    # Ubah >1 jika ingin relaksasi orbital penuh.
    print("  > [2/3] Running OMP2...", end="", flush=True)
    t0 = time.time()
    try:
        omp2 = mshqc.OMP2(mol, basis, integrals, rohf_res)
        omp2.set_max_iterations(1) 
        omp2_res = omp2.compute()
        results['e_omp2'] = omp2_res.energy_total
        results['t_omp2'] = time.time() - t0
        print(f" Done. E = {results['e_omp2']:.8f} Ha ({results['t_omp2']:.4f} s)")
    except Exception as e:
        print(f" FAILED: {e}")
        return None

    # 4. OMP3 (Orbital-Optimized MP3)
    print("  > [3/3] Running OMP3...", end="", flush=True)
    t0 = time.time()
    try:
        # Biasanya OMP3 mengambil hasil OMP2 sebagai referensi/start
        omp3 = mshqc.OMP3(mol, basis, integrals, omp2_res)
        omp3.set_max_iterations(1)
        omp3_res = omp3.compute()
        results['e_omp3'] = omp3_res.energy_total
        results['t_omp3'] = time.time() - t0
        print(f" Done. E = {results['e_omp3']:.8f} Ha ({results['t_omp3']:.4f} s)")
    except Exception as e:
        print(f" FAILED: {e}")
        return None

    return results

def main():
    print("##########################################################")
    print("  MSHQC BENCHMARK SUITE: ROHF -> OMP2 -> OMP3")
    print("  Target: Period 2 Atoms (Open & Closed Shell)")
    print("##########################################################")

    # Daftar atom yang akan di-test
    # Gunakan cc-pVQZ untuk akurasi tinggi, atau ganti cc-pVTZ biar lebih cepat
    target_atoms = ['He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F']
    basis_set = "cc-pVTZ" 

    all_results = []

    for atom in target_atoms:
        res = run_omp_benchmark(atom, basis_set)
        if res:
            all_results.append(res)

    # --- FINAL SUMMARY TABLE ---
    print("\n\n")
    print("="*110)
    print(f"  FINAL SUMMARY: OMP Benchmark ({basis_set})")
    print("="*110)
    print(f"  {'Atom':<4} | {'NBas':>4} | {'E(ROHF)':>14} | {'E(OMP2)':>14} | {'E(OMP3)':>14} | {'Time(s)':>8}")
    print("-" * 110)

    for r in all_results:
        total_time = r['t_rohf'] + r['t_omp2'] + r['t_omp3']
        print(f"  {r['atom']:<4} | {r['n_basis']:>4} | "
              f"{r['e_rohf']:>14.8f} | {r['e_omp2']:>14.8f} | {r['e_omp3']:>14.8f} | "
              f"{total_time:>8.2f}")
    
    print("="*110)

if __name__ == "__main__":
    main()