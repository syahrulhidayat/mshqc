import mshqc
import time
import numpy as np

def run_benchmark(basis_name, n_frozen, n_active_orb, n_states, desc):
    print("\n" + "="*90)
    print(f"  BENCHMARK: {basis_name} (Li Atom)")
    print(f"  Type     : {desc}")
    print("="*90)

    # 1. Setup
    mol = mshqc.Molecule()
    mol.add_atom(3, 0.0, 0.0, 0.0)
    mol.set_multiplicity(2)
    mol.set_charge(0)

    try:
        basis = mshqc.BasisSet(basis_name, mol)
    except:
        print("[SKIP] Basis failed.")
        return

    integrals = mshqc.IntegralEngine(mol, basis)
    n_basis = basis.n_basis_functions()
    print(f"  N Basis: {n_basis} | Frozen={n_frozen}, Active={n_active_orb}")

    # [PREP 1] Cholesky
    print("\n  [PREP 1] Computing Cholesky Decomposition... ", end="", flush=True)
    t0 = time.time()
    chol_engine = mshqc.CholeskyERI(1e-8)
    eri_tensor = integrals.compute_eri()
    chol_engine.decompose(eri_tensor)
    L_vectors = chol_engine.get_L_vectors()
    print(f"Done. ({time.time()-t0:.4f}s)")

    # [PREP 2] Optimized UHF
    print("  [PREP 2] Running UHF... ", end="", flush=True)
    t0 = time.time()
    uhf_conf = mshqc.CholeskyUHFConfig()
    uhf_conf.cholesky_threshold = 1e-8
    uhf_conf.max_iterations = 100  # Sekarang bisa diset!
    uhf_conf.print_level = 0

    uhf = mshqc.CholeskyUHF(mol, basis, integrals, 2, 1, uhf_conf)
    uhf.set_cholesky_vectors(L_vectors) # Inject vector
    uhf_res = uhf.compute()
    print(f"Done. ({time.time()-t0:.4f}s)")
    print(f"    > UHF Energy: {uhf_res.energy_total:.8f} Ha")

    # [PREP 3] UNO
    print("  [PREP 3] Generating UNO... ", end="", flush=True)
    uno_gen = mshqc.CholeskyUNO(uhf_res, integrals, n_basis)
    uno_res = uno_gen.compute()
    print("Done.")

    # [EXECUTE] SA-CASSCF -> PT2 -> PT3
    
    # Auto-Shift Logic
    shift = 0.0 # Standard
    
    # Limit Active Space
    n_act_actual = min(n_active_orb, n_basis - n_frozen)
    active_space = mshqc.ActiveSpace.CAS_Frozen(n_frozen, n_act_actual, n_basis, 3)

    # 1. SA-CASSCF
    print(f"\n  [STEP 1] SA-CASSCF (Frz={n_frozen}, Act={n_act_actual})... ", end="", flush=True)
    sa_conf = mshqc.SACASConfig()
    sa_conf.set_equal_weights(n_states)
    sa_conf.print_level = 0
    sa_conf.max_iter = 100
    
    t0 = time.time()
    casscf = mshqc.CholeskySACASSCF(mol, basis, integrals, active_space, sa_conf, L_vectors)
    sa_res = casscf.compute(uno_res.C_uno)
    print(f"Done. ({time.time()-t0:.2f}s)")

    # Report SA-CASSCF
    print("\n  >>> RESULT: SA-CASSCF Energies <<<")
    print("  " + "-"*60)
    print("  State |    Energy (Ha)    |   Exc (eV)  | Weight ")
    print("  " + "-"*60)
    e0_cas = sa_res.state_energies[0]
    for i in range(n_states):
        e = sa_res.state_energies[i]
        exc = (e - e0_cas) * 27.2114
        # Mengakses vector weights dari config
        w = sa_conf.weights[i] 
        print(f"    {i:<3} | {e:.8f} | {exc:>11.4f} | {w:.4f}")
    print("  " + "-"*60)

    # 2. CASPT2
    print("\n  [STEP 2] CASPT2... ", end="", flush=True)
    pt2_conf = mshqc.CASPT2Config()
    pt2_conf.shift = shift
    pt2_conf.print_level = 0
    
    t0 = time.time()
    # Pastikan binding support passing L_vectors sebagai list
    pt2 = mshqc.CholeskySACASPT2(sa_res, L_vectors, n_basis, active_space, pt2_conf)
    pt2_res = pt2.compute()
    print(f"Done. ({time.time()-t0:.2f}s)")

    # 3. CASPT3
    print("\n  [STEP 3] CASPT3... ", end="", flush=True)
    pt3_conf = mshqc.CASPT3Config()
    pt3_conf.shift = shift
    pt3_conf.zero_thresh = 1e-8 # Sekarang bisa diset!
    pt3_conf.print_level = 0
    
    t0 = time.time()
    pt3 = mshqc.CholeskySACASPT3(sa_res, L_vectors, n_basis, active_space, pt3_conf)
    pt3_res = pt3.compute()
    print(f"Done. ({time.time()-t0:.2f}s)")

    # FINAL TABLE
    print("\n")
    print(f"  >>> FINAL RESULTS TABLE: {basis_name} <<<")
    print("-" * 100)
    print("  St |   E(CASSCF)    |    E(PT2)     |    E(PT3)     |   E(Total)    |  Exc (Ha)   |  Exc (eV)")
    print("-" * 100)

    if pt3_res.e_pt3:
        e0_tot = 0.0
        for i in range(n_states):
            e_cas = sa_res.state_energies[i]
            e_pt2 = pt2_res.e_pt2[i]
            e_pt3 = pt3_res.e_pt3[i]
            e_tot = e_cas + e_pt2 + e_pt3
            
            if i == 0: e0_tot = e_tot
            
            exc_ha = e_tot - e0_tot
            exc_ev = exc_ha * 27.2114
            
            print(f"   {i:<2}| {e_cas:.6f} | {e_pt2:13.6f} | {e_pt3:13.6f} | {e_tot:.7f} | {exc_ha:.5f}     | {exc_ev:.4f}")
    print("-" * 100 + "\n")

def main():
    print("=================================================================")
    print("  MSHQC PYTHON BENCHMARK: SA-CASPT3")
    print("=================================================================")
    
    # PERBAIKAN: Gunakan () bukan {}
    configs = [
        ("cc-pCVTZ",   1,   30,   15, "Standard Basis (Shift=0.0)"),
        ("cc-pCVQZ",   1,   30,   15, "Standard Basis (Shift=0.0)"),
        ("aug-cc-pCVTZ",   1,   30,   15, "Diffuse Basis (Shift=0.0)"),
        ("aug-cc-pCVQZ",   1,   30,   15, "Diffuse Basis (Shift=0.0)")
    ]
    
    for c in configs:
        run_benchmark(*c)

if __name__ == "__main__":
    main()