# File: /home/syahrul/mshqc/bench_period2_casscf.py
import time
import os
import sys
import mshqc

# --- KONFIGURASI BENCHMARK ---
CHOLESKY_THRESH = 1e-8

# Konfigurasi Fisik Atom (Ground State)
# Active Space Strategy (Valence Only):
# - Period 1 (H, He): Frozen=0, Active=5 (1s + 2s + 3x2p) 
# - Period 2 (Li-C): Frozen=1 (1s), Active=4 (2s + 3x2p valence shell)
ATOM_CONFIGS = {
    'H':  {'z': 1, 'mult': 2, 'na': 1, 'nb': 0, 'frozen': 0, 'active': 5}, # Doublet
    'He': {'z': 2, 'mult': 1, 'na': 1, 'nb': 1, 'frozen': 0, 'active': 5}, # Singlet
    'Li': {'z': 3, 'mult': 2, 'na': 2, 'nb': 1, 'frozen': 1, 'active': 4}, # Doublet
    'Be': {'z': 4, 'mult': 1, 'na': 2, 'nb': 2, 'frozen': 1, 'active': 4}, # Singlet
    'B':  {'z': 5, 'mult': 2, 'na': 3, 'nb': 2, 'frozen': 1, 'active': 4}, # Doublet
    'C':  {'z': 6, 'mult': 3, 'na': 4, 'nb': 2, 'frozen': 1, 'active': 4}, # Triplet
}

def run_variational_test(atom_label, basis_name):
    cfg = ATOM_CONFIGS[atom_label]
    
    print(f"\n{'='*80}")
    print(f"  VARIATIONAL TEST: Atom={atom_label} (Z={cfg['z']}) | Basis={basis_name}")
    print(f"  Config : Mult={cfg['mult']}, Frozen={cfg['frozen']}, Active={cfg['active']}")
    print(f"{'-'*80}")
    
    results = {'atom': atom_label, 'basis': basis_name}
    
    # 1. SETUP MOLECULE & BASIS
    try:
        mol = mshqc.Molecule()
        mol.add_atom(cfg['z'], 0.0, 0.0, 0.0) 
        mol.set_multiplicity(cfg['mult'])
        mol.set_charge(0)
        
        basis_dir = os.path.join(os.environ.get('MSHQC_DATA_DIR', 'data'), 'basis')
        basis = mshqc.BasisSet(basis_name, mol, basis_dir)
        integrals = mshqc.IntegralEngine(mol, basis)
        
        n_basis = basis.n_basis_functions()
        results['n_basis'] = n_basis
        print(f"  > Basis Functions: {n_basis}")
        
    except Exception as e:
        print(f"  [ERROR] Init failed: {e}")
        return None

    # 2. CHOLESKY DECOMPOSITION
    print("  > [1/4] Cholesky Decomposition... ", end="", flush=True)
    t0 = time.time()
    
    chol_engine = mshqc.CholeskyERI(CHOLESKY_THRESH)
    eri_tensor = integrals.compute_eri()
    chol_engine.decompose(eri_tensor)
    L_vectors = chol_engine.get_L_vectors()
    
    results['t_chol'] = time.time() - t0
    print(f"Done ({results['t_chol']:.4f}s)")

    # 3. UHF CALCULATION (Starting Guess)
    print("  > [2/4] UHF Calculation... ", end="", flush=True)
    t0 = time.time()
    
    uhf_config = mshqc.CholeskyUHFConfig()
    uhf_config.cholesky_threshold = CHOLESKY_THRESH
    uhf_config.print_level = 0
    
    uhf = mshqc.CholeskyUHF(mol, basis, integrals, cfg['na'], cfg['nb'], uhf_config)
    uhf.set_cholesky_vectors(L_vectors)
    uhf_res = uhf.compute()
    
    results['t_uhf'] = time.time() - t0
    results['e_uhf'] = uhf_res.energy_total
    print(f"Done ({results['t_uhf']:.4f}s)")

    # 4. UNO GENERATION (Natural Orbitals)
    print("  > [3/4] UNO Generation... ", end="", flush=True)
    uno_gen = mshqc.CholeskyUNO(uhf_res, integrals, n_basis)
    uno_res = uno_gen.compute()
    print("Done.")

    # 5. SS-CASSCF (Optimization)
    print("  > [4/4] SS-CASSCF... ", end="", flush=True)
    t0 = time.time()
    
    active_space = mshqc.ActiveSpace.CAS_Frozen(cfg['frozen'], cfg['active'], n_basis, cfg['z'])
    
    sa_config = mshqc.SACASConfig()
    sa_config.set_equal_weights(1) # SS-CASSCF
    sa_config.max_iter = 100       # Berikan iterasi lebih banyak untuk konvergensi ketat
    sa_config.print_level = 0
    
    casscf = mshqc.CholeskySACASSCF(mol, basis, integrals, active_space, sa_config, L_vectors)
    casscf_res = casscf.compute(uno_res.C_uno)
    
    results['t_cas'] = time.time() - t0
    results['e_cas'] = casscf_res.state_energies[0] # Energi Variasional Akhir
    print(f"Done ({results['t_cas']:.4f}s)")
    print(f"  > Final E_CAS: {results['e_cas']:.8f} Ha")
    
    return results

def main():
    print("="*100)
    print("  MSHQC BENCHMARK: Variational Only (UHF -> UNO -> CASSCF)")
    print("="*100)

    atoms_to_test = ['H', 'He', 'Li', 'Be', 'B', 'C']
    
    basis_list = [
        "cc-pVTZ",      # Standard valence basis
        "cc-pCVTZ"      # Core-valence basis (penting jika nanti pakai PT2)
    ]

    all_results = []
    
    for atom in atoms_to_test:
        for basis in basis_list:
            res = run_variational_test(atom, basis)
            if res:
                all_results.append(res)
    
    # --- FINAL TABLE ---
    print("\n\n")
    print("="*110)
    print("  FINAL SUMMARY: Reference Wavefunction Quality")
    print("="*110)
    
    header = (f"  {'Atom':<4} | {'Basis Set':<12} | {'NBas':>4} | "
              f"{'E(UHF)':>14} | {'E(CASSCF)':>14} | "
              f"{'T(UHF)':>7} | {'T(CAS)':>7} | {'Total(s)':>8}")
    print(header)
    print("-" * 110)

    for r in all_results:
        t_tot = r['t_chol'] + r['t_uhf'] + r['t_cas']
        
        row = (f"  {r['atom']:<4} | {r['basis']:<12} | {r['n_basis']:>4} | "
               f"{r['e_uhf']:>14.8f} | {r['e_cas']:>14.8f} | "
               f"{r['t_uhf']:>7.3f} | {r['t_cas']:>7.3f} | {t_tot:>8.3f}")
        print(row)
        
    print("="*110)

if __name__ == "__main__":
    main()