import sys
import mshqc

# Mapping Symbol ke Atomic Number (Z)
ATOMIC_NUMBERS = {"H": 1, "Li": 3}

def get_electron_occupancy(total_charge, multiplicity, atomic_numbers):
    """Menghitung jumlah elektron alpha dan beta."""
    total_electrons = sum(atomic_numbers) - total_charge
    # Multiplicity = 2S + 1  -> S = (M-1)/2
    # N_alpha - N_beta = 2S  -> N_alpha - N_beta = M - 1
    # N_alpha + N_beta = N_total
    
    n_excess_spin = multiplicity - 1
    if (total_electrons - n_excess_spin) % 2 != 0:
        raise ValueError("Multiplisitas tidak cocok dengan jumlah elektron (Ganjil/Genap error).")
    
    n_beta = (total_electrons - n_excess_spin) // 2
    n_alpha = total_electrons - n_beta
    return n_alpha, n_beta

def run_calculation(atom_label, coordinates, charge, multiplicity, basis_name):
    print(f"\n{'='*50}\nHitung: {atom_label} | Basis: {basis_name} | Mult: {multiplicity}\n{'='*50}")

    # --- 1. SETUP MOLECULE ---
    try:
        mol = mshqc.Molecule(charge, multiplicity)
        z_val = ATOMIC_NUMBERS[atom_label]
        mol.add_atom(z_val, *coordinates)
    except Exception as e:
        print(f"Error Molecule: {e}")
        return

    # --- 2. SETUP BASIS SET & INTEGRALS ---
    # PERBAIKAN: Urutan argumen basis_name DULU, baru mol
    try:
        # Konstruktor yang benar: BasisSet(basis_name, mol, basis_dir)
        basis = mshqc.BasisSet(basis_name, mol)  # ✅ DIPERBAIKI
        
        # IntegralEngine tetap sama
        integrals = mshqc.IntegralEngine(mol, basis)
    except Exception as e:
        print(f"Error Setup Basis/Integrals: {e}\nCek apakah file basis ada di 'data/basis/{basis_name}.gbs'")
        return

    # --- 3. HITUNG JUMLAH ELEKTRON (Alpha/Beta) ---
    try:
        n_alpha, n_beta = get_electron_occupancy(charge, multiplicity, [z_val])
        print(f"Info: Z={z_val}, Charge={charge} -> n_alpha={n_alpha}, n_beta={n_beta}")
    except Exception as e:
        print(f"Error Hitung Elektron: {e}")
        return

    # --- 4. UHF CALCULATION ---
    print("--- Start UHF ---")
    try:
        # UHF(mol, basis, integrals, n_alpha, n_beta, config)
        uhf_calc = mshqc.UHF(mol, basis, integrals, n_alpha, n_beta)
        
        uhf_result = uhf_calc.compute()  # Jalankan SCF
        uhf_energy = uhf_result.energy_total
        print(f"E(UHF) : {uhf_energy:.8f} Hartree")
    except Exception as e:
        print(f"Error UHF: {e}")
        return

    # --- 5. UMP2 CALCULATION ---
    print("--- Start UMP2 ---")
    try:
        # UMP2(uhf_result, basis, integrals)
        ump2_calc = mshqc.UMP2(uhf_result, basis, integrals)
        ump2_result = ump2_calc.compute()
        print(f"E(UMP2): {ump2_result.e_total:.8f} Hartree (Corr: {ump2_result.e_corr_total:.8f})")
    except Exception as e:
        print(f"Error UMP2: {e}")
        return

    # --- 6. UMP3 CALCULATION ---
    print("--- Start UMP3 ---")
    try:
        # UMP3(uhf_result, ump2_result, basis, integrals)
        ump3_calc = mshqc.UMP3(uhf_result, ump2_result, basis, integrals)
        ump3_result = ump3_calc.compute()
        print(f"E(UMP3): {ump3_result.e_total:.8f} Hartree")
        print(f"  - E(UHF)      : {ump3_result.e_uhf:.8f}")
        print(f"  - E(MP2 corr) : {ump3_result.e_mp2:.8f}")
        print(f"  - E(MP3 corr) : {ump3_result.e_mp3_corr:.8f}")
        print(f"  - E(Total corr): {ump3_result.e_corr_total:.8f}")
    except Exception as e:
        print(f"Error UMP3: {e}")

# --- MAIN BLOCK ---
if __name__ == "__main__":
    coords = (0.0, 0.0, 0.0)
    
    # Litium (3e, Doublet -> 2 alpha, 1 beta)
    run_calculation("Li", coords, 0, 2, "cc-pVDZ")
    
    # Hidrogen (1e, Doublet -> 1 alpha, 0 beta)
    run_calculation("H", coords, 0, 2, "cc-pVDZ")
