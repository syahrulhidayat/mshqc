"""
MSHQC Session - Complete Workflow Manager

File: /home/syahrul/mshqc/python/mshqc/session.py
"""

from .calculators import MSHQCCalculator, MCSCFCalculator
import mshqc

class MSHQCSession:
    """Complete MSHQC calculation session"""
    
    def __init__(self, molecule, basis_name, method="auto"):
        """
        Initialize MSHQC session
        
        Args:
            molecule: Molecule object
            basis_name: str, basis set name
            method: str, calculation method
        """
        self.molecule = molecule
        self.basis_name = basis_name
        self.method = method
        
        # Initialize calculators
        self.scf_calc = MSHQCCalculator(molecule, basis_name)
        self.mcscf_calc = MCSCFCalculator(molecule, basis_name)
        # self.ci_calc dinonaktifkan sementara
        
        self.results = {}
    
    def run_complete_analysis(self, methods=None):
        """
        Run complete multi-method analysis
        
        Args:
            methods: list of methods to run
        
        Returns:
            dict with all results
        """
        if methods is None:
            methods = ["scf", "mp2", "mp3"]
        
        print("="*70)
        print(f"MSHQC Complete Analysis")
        print(f"Basis: {self.basis_name}")
        print("="*70)
        
        # Run SCF
        if "scf" in methods or "uhf" in methods or "all" in methods:
            print("\n>>> Running UHF...")
            self.results['uhf'] = self.scf_calc.run_uhf()
        
        # Run MP2
        if "mp2" in methods or "ump2" in methods or "all" in methods:
            if 'uhf' not in self.results:
                self.results['uhf'] = self.scf_calc.run_uhf()
            print("\n>>> Running UMP2...")
            self.results['ump2'] = self.scf_calc.run_ump2(
                self.results['uhf']
            )
        
        # Run MP3
        if "mp3" in methods or "ump3" in methods or "all" in methods:
            if 'ump2' not in self.results:
                if 'uhf' not in self.results:
                    self.results['uhf'] = self.scf_calc.run_uhf()
                self.results['ump2'] = self.scf_calc.run_ump2(
                    self.results['uhf']
                )
            print("\n>>> Running UMP3...")
            self.results['ump3'] = self.scf_calc.run_ump3(
                self.results['uhf'], self.results['ump2']
            )
        
        # Block CISD dinonaktifkan
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print summary of all computed energies"""
        print("\n" + "="*70)
        print("ENERGY SUMMARY")
        print("="*70)
        print(f"{'Method':<20} {'Energy (Ha)':<20} {'dE (kcal/mol)':<15}")
        print("-"*70)
        
        ref_energy = None
        if 'uhf' in self.results:
            e = self.results['uhf'].energy_total
            print(f"{'UHF':<20} {e:<20.10f} {'-':<15}")
            ref_energy = e
        
        if 'ump2' in self.results:
            e = self.results['ump2'].e_total
            delta = (e - ref_energy) * 627.509 if ref_energy else 0
            print(f"{'UMP2':<20} {e:<20.10f} {delta:<15.4f}")
        
        if 'ump3' in self.results:
            e = self.results['ump3'].e_total
            delta = (e - ref_energy) * 627.509 if ref_energy else 0
            print(f"{'UMP3':<20} {e:<20.10f} {delta:<15.4f}")
        
        print("="*70)