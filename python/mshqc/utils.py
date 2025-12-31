"""
Utility Functions for MSHQC
File: /home/syahrul/mshqc/python/mshqc/utils.py
"""

import time
import os


def quick_calculation(element, basis="cc-pVTZ", method="ump3",
                     charge=0, multiplicity=None, basis_dir=None):
    """
    Quick single-atom calculation
    
    Args:
        element: str or int, element symbol or atomic number
        basis: str, basis set name
        method: str, calculation method
        charge: int, molecular charge
        multiplicity: int, spin multiplicity
        basis_dir: str, optional basis directory path
    
    Returns:
        Result object
    
    Example:
        >>> from mshqc.utils import quick_calculation
        >>> result = quick_calculation("Li", "cc-pVTZ", "ump3")
    """

    # Import di dalam fungsi untuk menghindari circular import
    import mshqc
    from .calculators import MSHQCCalculator
    
    # Get basis directory
    if basis_dir is None:
        if 'MSHQC_DATA_DIR' in os.environ:
            basis_dir = os.path.join(os.environ['MSHQC_DATA_DIR'], 'basis')
        else:
            basis_dir = "data/basis"
    
    mol = mshqc.Molecule()
    
    if isinstance(element, str):
        # Tabel Periodik Lengkap (Z=1 sampai Z=118)
        symbol_to_z = {
            'H': 1, 'He': 2,
            'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
            'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
            'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
            'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
            'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
            'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103,
            'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
        }
        
        # Ubah default value: Jika tidak ditemukan, raise Error daripada return 3 (Lithium)
        if element not in symbol_to_z:
            raise ValueError(f"Unknown element symbol: {element}")
        z = symbol_to_z[element]
        mol.add_atom(z, 0.0, 0.0, 0.0)
    else:
        mol.add_atom(element, 0.0, 0.0, 0.0)
    
    mol.set_charge(charge)
    
    if multiplicity is None:
        n_elec = mol.n_electrons()
        multiplicity = 2 if n_elec % 2 == 1 else 1
    
    mol.set_multiplicity(multiplicity)
    
    calc = MSHQCCalculator(mol, basis, basis_dir=basis_dir)
    
    method = method.lower()
    
    # =========================================================================
    #  Logic Khusus Cholesky Pipeline (OMP2 & OMP3)
    # =========================================================================
    if method in ["chomp2", "cholesky_omp2", "chomp3", "cholesky_omp3"]:
        # Tentukan jika target akhirnya adalah OMP3
        target_method = "omp3" if "3" in method else "omp2"
        print(f"\n--- Running Quick Cholesky-{target_method.upper()} for {element} ---")
        
        # 1. Decompose ERI (Sekali di awal)
        threshold = 1e-6
        print("1. Decomposing Integrals...")
        # Import class Cholesky dari module mshqc
        chol = mshqc.CholeskyERI(threshold)
        eri = calc.integrals.compute_eri()
        chol.decompose(eri)
        
        # 2. Run Reference SCF (Menggunakan Vektor Cholesky yang sama)
        print("2. Running Reference SCF (Vector Reuse)...")
        
        # Hitung jumlah elektron manual untuk konfigurasi
        n_total = mol.n_electrons()
        na = (n_total + multiplicity - 1) // 2
        nb = n_total - na
        
        if multiplicity == 1:
        
            print("   > Using Cholesky-ROHF (Closed Shell)")
            rohf_conf = mshqc.CholeskyROHFConfig()
            rohf_conf.cholesky_threshold = threshold
            rohf_conf.print_level = 0 # Silent
            
            c_scf = mshqc.CholeskyROHF(mol, calc.basis, calc.integrals, 
                                       na, nb, rohf_conf, chol)
        else:
            # Open-Shell: Gunakan Cholesky-UHF
            print("   > Using Cholesky-UHF (Open Shell)")
            uhf_conf = mshqc.CholeskyUHFConfig()
            uhf_conf.cholesky_threshold = threshold
            uhf_conf.print_level = 0 # Silent
            
            c_scf = mshqc.CholeskyUHF(mol, calc.basis, calc.integrals, 
                                      na, nb, uhf_conf)
            c_scf.set_cholesky_vectors(chol.get_L_vectors())

        scf_res = c_scf.compute()
        print(f"   SCF Done. E = {scf_res.energy_total:.8f} Ha")

        # 3. Run Cholesky-OMP2 (Reuse Vektor lagi!)
        print("3. Running Orbital Optimization (OMP2)...")
        # Panggil helper yang sudah kita buat di calculators.py
        omp2_res = calc.run_cholesky_omp2(
            scf_res, 
            existing_cholesky=chol  # <--- KUNCI: Pass objek chol yang sudah decompose
        )
        
        # [BARU] 4. Run Cholesky-OMP3 jika diminta
        if target_method == "omp3":
            print("4. Running Orbital Optimization (OMP3)...")
            omp3_res = calc.run_cholesky_omp3(
                omp2_res, 
                existing_cholesky=chol # Reuse vector lagi!
            )
            return omp3_res
            
        return omp2_res

    # 1. Run SCF (Prioritas RHF jika Closed-Shell, kecuali user minta UHF)
    scf_res = None
    if multiplicity == 1 and method not in ["uhf", "ump2", "ump3"]:
        scf_res = calc.run_rhf()
    else:
        scf_res = calc.run_uhf()
        
    if method in ["scf", "rhf", "uhf"]:
        return scf_res

    # 2. Run Post-SCF
    if "mp2" in method:
        if "omp" in method: # OMP2
            return calc.run_omp2(scf_res)
        elif multiplicity == 1 and "u" not in method: # RMP2 (Default Singlet)
            return calc.run_rmp2(scf_res)
        else: # UMP2
            return calc.run_ump2(scf_res)
            
    if "mp3" in method:
        # Run MP2 dulu sebagai prasyarat
        if "omp" in method: # OMP3
            mp2_res = calc.run_omp2(scf_res)
            return calc.run_omp3(mp2_res)
        elif multiplicity == 1 and "u" not in method: # RMP3
            rmp2_res = calc.run_rmp2(scf_res)
            return calc.run_rmp3(scf_res, rmp2_res)
        else: # UMP3
            ump2_res = calc.run_ump2(scf_res)
            return calc.run_ump3(scf_res, ump2_res)
        
    if method in ["ch-rmp2", "cholesky_rmp2"]:
        # Pastikan SCF-nya adalah RHF (Closed Shell)
        if multiplicity != 1:
             raise ValueError("Cholesky-RMP2 requires closed-shell singlet (multiplicity=1)")
        
        print(f"\n--- Running Quick Cholesky-RMP2 for {element} ---")
        
        # 1. Decompose
        print("1. Decomposing Integrals...")
        chol = mshqc.CholeskyERI(1e-6)
        eri = calc.integrals.compute_eri()
        chol.decompose(eri)
        
        # 2. RHF
        print("2. Running Cholesky-RHF...")
        rhf_conf = mshqc.CholeskyRHFConfig()
        rhf_conf.print_level = 0
        rhf_solver = mshqc.CholeskyRHF(mol, calc.basis, calc.integrals, rhf_conf, chol)
        rhf_res = rhf_solver.compute()
        print(f"   RHF Done. E = {rhf_res.energy_total:.8f} Ha")
        
        # 3. RMP2
        print("3. Running Cholesky-RMP2...")
        return calc.run_cholesky_rmp2(rhf_res, existing_cholesky=chol)
    
    # ... Di dalam fungsi quick_calculation ...

    if method in ["ch-rmp3", "cholesky_rmp3"]:
        # Pastikan Closed Shell
        if multiplicity != 1:
             raise ValueError("Cholesky-RMP3 requires closed-shell singlet (multiplicity=1)")
        
        print(f"\n--- Running Quick Cholesky-RMP3 for {element} ---")
        
        # 1. Decompose
        print("1. Decomposing Integrals...")
        chol = mshqc.CholeskyERI(1e-6)
        eri = calc.integrals.compute_eri()
        chol.decompose(eri)
        
        # 2. RHF (Reuse Vector)
        print("2. Running Cholesky-RHF...")
        rhf_conf = mshqc.CholeskyRHFConfig()
        rhf_conf.print_level = 0
        rhf_solver = mshqc.CholeskyRHF(mol, calc.basis, calc.integrals, rhf_conf, chol)
        rhf_res = rhf_solver.compute()
        print(f"   RHF Done. E = {rhf_res.energy_total:.8f} Ha")
        
        # 3. RMP2 (Reuse Vector) -> Return CholeskyRMP2Result
        print("3. Running Cholesky-RMP2...")
        rmp2_conf = mshqc.CholeskyRMP2Config()
        rmp2_solver = mshqc.CholeskyRMP2(mol, calc.basis, calc.integrals, rhf_res, rmp2_conf, chol)
        crmp2_res = rmp2_solver.compute()
        print(f"   RMP2 Done. E_Corr = {crmp2_res.e_corr:.8f} Ha")

        # 4. RMP3 (Reuse Vector dari crmp2_res)
        print("4. Running Cholesky-RMP3...")
        # Gunakan helper calculator atau langsung object
        rmp3_res = calc.run_cholesky_rmp3(rhf_res, crmp2_res)
        
        return rmp3_res

    raise ValueError(f"Unknown method: {method}")

    
    
def benchmark_basis_sets(element, basis_list, method="ump3"):
    """
    Benchmark multiple basis sets for single element
    
    Args:
        element: str or int
        basis_list: list of basis set names
        method: calculation method
    
    Returns:
        dict with results for each basis
    
    Example:
        >>> from mshqc.utils import benchmark_basis_sets
        >>> results = benchmark_basis_sets("Li",
        ...     ["cc-pVDZ", "cc-pVTZ", "cc-pVQZ"], "ump3")
    """
    results = {}
    
    print("="*70)
    print(f"BASIS SET BENCHMARK: {element} ({method.upper()})")
    print("="*70)
    
    for basis in basis_list:
        print(f"\nRunning {basis}...")
        t0 = time.time()
        
        try:
            result = quick_calculation(element, basis, method)
            elapsed = time.time() - t0
            
            results[basis] = {
                'result': result,
                'time': elapsed,
                'success': True
            }
            
       # Get energy (Gunakan getattr agar Pylance tidak error)
            if hasattr(result, 'e_total'):
                energy = getattr(result, 'e_total')
            elif hasattr(result, 'energy_total'):
                energy = getattr(result, 'energy_total')
            else:
                energy = getattr(result, 'energy')
            
            results[basis]['energy'] = energy
            
            print(f"  Success: E = {energy:.10f} Ha ({elapsed:.2f}s)")
            
        except Exception as e:
            print(f"  Failed: {str(e)}")
            results[basis] = {
                'success': False,
                'error': str(e)
            }
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Basis':<20} {'Energy (Ha)':<20} {'Time (s)':<15}")
    print("-"*70)
    
    for basis in basis_list:
        if results[basis]['success']:
            print(f"{basis:<20} {results[basis]['energy']:<20.10f} "
                  f"{results[basis]['time']:<15.2f}")
        else:
            print(f"{basis:<20} {'FAILED':<20} {'-':<15}")
    
    print("="*70)
    
    return results


def compare_methods(molecule, basis, methods=None):
    """
    Compare multiple methods on same system
    
    Args:
        molecule: Molecule object
        basis: str, basis set name
        methods: list of method names
    
    Returns:
        dict with comparison
    
    Example:
        >>> import mshqc
        >>> mol = mshqc.Molecule()
        >>> mol.add_atom(3, 0, 0, 0)
        >>> from mshqc.utils import compare_methods
        >>> compare_methods(mol, "cc-pVTZ", ["uhf", "ump2", "ump3"])
    """
    if methods is None:
        methods = ["uhf", "ump2", "ump3"]
    
    from .session import MSHQCSession
    session = MSHQCSession(molecule, basis)
    results = session.run_complete_analysis(methods)
    
    return results