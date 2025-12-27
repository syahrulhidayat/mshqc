import mshqc
import time
import sys

def run_analysis(atomic_number, atom_name, basis_name):
    """
    Menjalankan analisis RHF -> RMP2 -> RMP3 untuk atom closed-shell.
    """
    print("\n" + "="*60)
    print(f" PROCESSING: {atom_name} (Z={atomic_number})")
    print(f" BASIS SET:  {basis_name}")
    print("="*60)

    # 1. Define Molecule
    mol = mshqc.Molecule()
    mol.add_atom(atomic_number, 0.0, 0.0, 0.0)
    mol.set_charge(0)
    
    # PENTING: Multiplicity harus 1 (Singlet/Closed Shell) untuk RMP
    mol.set_multiplicity(1) 
    
    # 2. Define Basis Set
    try:
        # Basis set akan dicari di folder data/basis/
        basis = mshqc.BasisSet(basis_name, mol)
    except Exception as e:
        print(f"[ERROR] Gagal memuat basis set '{basis_name}': {e}")
        return

    n_elec = mol.n_electrons()
    print(f"System: {atom_name} atom ({n_elec} electrons)")
    print(f"Basis functions: {basis.n_basis_functions()}\n")

    # Cek apakah sistem benar-benar closed shell (jumlah elektron genap)
    if n_elec % 2 != 0:
        print(f"[WARNING] Atom {atom_name} memiliki {n_elec} elektron (Ganjil).")
        print("          Metode RHF/RMP mungkin tidak cocok. Gunakan UHF/UMP.")
        return

    # 3. Integrals Engine
    integrals = mshqc.IntegralEngine(mol, basis)

    # 4. Run RHF (Restricted Hartree-Fock)
    scf_config = mshqc.SCFConfig()
    scf_config.energy_threshold = 1e-10
    scf_config.max_iterations = 100
    
    print("--- Step 1: Hartree-Fock (RHF) ---")
    t0 = time.time()
    rhf = mshqc.RHF(mol, basis, integrals, scf_config)
    rhf_res = rhf.compute()
    t_rhf = time.time() - t0
    
    print(f"Energy (RHF): {rhf_res.energy_total:.8f} Ha")
    print(f"Time: {t_rhf:.4f} s")
    
    if not rhf_res.converged:
        print("[STOP] RHF did not converge! Skipping post-HF methods.")
        return
    
    # 5. Run RMP2 (Restricted MP2)
    print("\n--- Step 2: RMP2 (Second-Order Correlation) ---")
    t0 = time.time()
    rmp2 = mshqc.RMP2(rhf_res, basis, integrals)
    rmp2_res = rmp2.compute()
    t_mp2 = time.time() - t0
    
    print(f"Correlation:  {rmp2_res.e_corr:.8f} Ha")
    print(f"Total (MP2):  {rmp2_res.e_total:.8f} Ha")
    print(f"Time: {t_mp2:.4f} s")
    
    # 6. Run RMP3 (Restricted MP3)
    print("\n--- Step 3: RMP3 (Third-Order Correction) ---")
    t0 = time.time()
    rmp3 = mshqc.RMP3(rhf_res, rmp2_res, basis, integrals)
    rmp3_res = rmp3.compute()
    t_mp3 = time.time() - t0
    
    print(f"MP3 Corr:     {rmp3_res.e_mp3:.8f} Ha")
    print(f"Total (MP3):  {rmp3_res.e_total:.8f} Ha")
    print(f"Time: {t_mp3:.4f} s")

    # Summary Singkat
    print("-" * 60)
    print(f"FINAL SUMMARY ({atom_name} / {basis_name})")
    print(f"  E(RHF)  : {rhf_res.energy_total:.8f} Ha")
    print(f"  E(MP2)  : {rmp2_res.e_total:.8f} Ha")
    print(f"  E(MP3)  : {rmp3_res.e_total:.8f} Ha")
    print(f"  Total Time: {t_rhf + t_mp2 + t_mp3:.4f} s")
    print("-" * 60)

def main():
    print("==========================================================")
    print("   MSHQC EXTENDED BENCHMARK: CLOSED-SHELL ATOMS")
    print("   Method: RHF -> RMP2 -> RMP3")
    print("==========================================================\n")

    try:
        # --- 1. Helium (He, Z=2) ---
        # Sistem paling sederhana, konvergensi sangat cepat.
        # Menggunakan basis Triple-Z (VTZ).
        run_analysis(2, "Helium", "cc-pVQZ")
        
        # --- 2. Beryllium (Be, Z=4) ---
        # Logam alkali tanah, closed shell (2s2).
        # Menggunakan basis Triple-Z (VTZ).
        run_analysis(4, "Beryllium", "cc-pVQZ")
        
        # --- 3. Neon (Ne, Z=10) ---
        # Gas mulia, closed shell (2p6).
        # Menggunakan basis Triple-Z (VTZ).
        run_analysis(10, "Neon", "cc-pVQZ")
    
        
    
    
        
    except KeyboardInterrupt:
        print("\n[ABORT] Calculation stopped by user.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()