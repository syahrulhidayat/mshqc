# File: /home/syahrul/mshqc/bench_period2_full.py
import time
import os
import sys
import mshqc

# --- KONFIGURASI BENCHMARK ---
CHOLESKY_THRESH = 1e-8
PT_SHIFT = 0.25  # Shift level (0.0 untuk raw perturbation, 0.25 untuk IPEA standar)

# Konfigurasi Fisik Atom (Ground State)
ATOM_CONFIGS = {
    'H':  {'z': 1, 'mult': 2, 'na': 1, 'nb': 0, 'frozen': 0, 'active': 5}, # Doublet
    'He': {'z': 2, 'mult': 1, 'na': 1, 'nb': 1, 'frozen': 0, 'active': 5}, # Singlet
    'Li': {'z': 3, 'mult': 2, 'na': 2, 'nb': 1, 'frozen': 1, 'active': 4}, # Doublet
    'Be': {'z': 4, 'mult': 1, 'na': 2, 'nb': 2, 'frozen': 1, 'active': 4}, # Singlet
    'B':  {'z': 5, 'mult': 2, 'na': 3, 'nb': 2, 'frozen': 1, 'active': 4}, # Doublet
    'C':  {'z': 6, 'mult': 3, 'na': 4, 'nb': 2, 'frozen': 1, 'active': 4}, # Triplet
}

def run_full_benchmark(atom_label, basis_name):
    cfg = ATOM_CONFIGS[atom_label]
    
    print(f"\n{'='*80}")
    print(f"  FULL BENCHMARK: Atom={atom_label} (Z={cfg['z']}) | Basis={basis_name}")
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
    print("  > [1/6] Cholesky Decomposition... ", end="", flush=True)
    t0 = time.time()
    
    chol_engine = mshqc.CholeskyERI(CHOLESKY_THRESH)
    eri_tensor = integrals.compute_eri()
    chol_engine.decompose(eri_tensor)
    L_vectors = chol_engine.get_L_vectors()
    
    results['t_chol'] = time.time() - t0
    print(f"Done ({results['t_chol']:.4f}s)")

    # 3. UHF CALCULATION
    print("  > [2/6] UHF Calculation... ", end="", flush=True)
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

    # 4. UNO GENERATION
    print("  > [3/6] UNO Generation... ", end="", flush=True)
    uno_gen = mshqc.CholeskyUNO(uhf_res, integrals, n_basis)
    uno_res = uno_gen.compute()
    print("Done.")

    # 5. SS-CASSCF
    print("  > [4/6] SS-CASSCF... ", end="", flush=True)
    t0 = time.time()
    
    active_space = mshqc.ActiveSpace.CAS_Frozen(cfg['frozen'], cfg['active'], n_basis, cfg['z'])
    
    sa_config = mshqc.SACASConfig()
    sa_config.set_equal_weights(1)
    sa_config.max_iter = 100
    sa_config.print_level = 0
    
    casscf = mshqc.CholeskySACASSCF(mol, basis, integrals, active_space, sa_config, L_vectors)
    casscf_res = casscf.compute(uno_res.C_uno)
    
    results['t_cas'] = time.time() - t0
    results['e_cas'] = casscf_res.state_energies[0]
    print(f"Done ({results['t_cas']:.4f}s)")

    # 6. CASPT2
    print("  > [5/6] CASPT2... ", end="", flush=True)
    t0 = time.time()
    
    pt2_config = mshqc.CASPT2Config()
    pt2_config.shift = PT_SHIFT
    pt2_config.print_level = 0
    pt2_config.export_amplitudes = True # Penting untuk PT3
    
    pt2 = mshqc.CholeskySACASPT2(casscf_res, L_vectors, n_basis, active_space, pt2_config)
    pt2_res = pt2.compute()
    
    results['t_pt2'] = time.time() - t0
    results['e_pt2'] = pt2_res.e_pt2[0] # Energi korelasi PT2 saja
    print(f"Done ({results['t_pt2']:.4f}s)")

    # 7. CASPT3
    print("  > [6/6] CASPT3... ", end="", flush=True)
    t0 = time.time()
    
    pt3_config = mshqc.CASPT3Config()
    pt3_config.shift = PT_SHIFT
    pt3_config.print_level = 0
    
    # PT3 sekarang mengambil amplitudo dari PT2 result secara otomatis (jika dihandle di C++)
    # atau menghitung ulang jika diperlukan, tergantung binding.
    # Asumsi: Binding Python menerima argumen yang sama.
    pt3 = mshqc.CholeskySACASPT3(casscf_res, L_vectors, n_basis, active_space, pt3_config)
    pt3_res = pt3.compute()
    
    results['t_pt3'] = time.time() - t0
    results['e_pt3'] = pt3_res.e_pt3[0] # Energi korelasi PT3 saja
    
    # Total Energy (Variasional + PT2 + PT3)
    results['e_total'] = results['e_cas'] + results['e_pt2'] + results['e_pt3']
    print(f"Done ({results['t_pt3']:.4f}s)")
    print(f"  > Final E_Total: {results['e_total']:.8f} Ha")
    
    return results

def main():
    print("="*120)
    print("  MSHQC FULL BENCHMARK: UHF -> CASSCF -> PT2 -> PT3")
    print("="*120)

    atoms_to_test = ['H', 'He', 'Li', 'Be', 'B', 'C']
    
    basis_list = [
        "cc-pVTZ",      
        "cc-pCVTZ",
        "cc-pCVQZ"     
    ]

    all_results = []
    
    for atom in atoms_to_test:
        for basis in basis_list:
            res = run_full_benchmark(atom, basis)
            if res:
                all_results.append(res)
    
    # --- FINAL TABLE ---
    print("\n\n")
    print("="*135)
    print("  FINAL SUMMARY: Ground State Energies (Hartree)")
    print("="*135)
    
    header = (f"  {'Atom':<4} | {'Basis':<9} | "
              f"{'E(CAS)':>12} | {'E(PT2)':>10} | {'E(PT3)':>10} | {'E(Total)':>12} | "
              f"{'T(CAS)':>6} | {'T(PT2)':>6} | {'T(PT3)':>6} | {'Tot(s)':>6}")
    print(header)
    print("-" * 135)

    for r in all_results:
        t_tot = r['t_chol'] + r['t_uhf'] + r['t_cas'] + r['t_pt2'] + r['t_pt3']
        
        row = (f"  {r['atom']:<4} | {r['basis']:<9} | "
               f"{r['e_cas']:>12.6f} | {r['e_pt2']:>10.6f} | {r['e_pt3']:>10.6f} | {r['e_total']:>12.6f} | "
               f"{r['t_cas']:>6.3f} | {r['t_pt2']:>6.3f} | {r['t_pt3']:>6.3f} | {t_tot:>6.3f}")
        print(row)
        
    print("="*135)

if __name__ == "__main__":
    main()