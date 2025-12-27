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
        symbol_to_z = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5,
            'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10
        }
        z = symbol_to_z.get(element, 3)
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