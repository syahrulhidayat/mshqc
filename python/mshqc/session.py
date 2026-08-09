 # ==============================================================================
 # Copyright (c) 2026 Syahrul and mshqc contributors
 #
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 #     http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 # ==============================================================================

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
        
        

        self.scf_calc = MSHQCCalculator(molecule, basis_name)
        self.mcscf_calc = MCSCFCalculator(molecule, basis_name)
        

        
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
        
        

        if "scf" in methods or "uhf" in methods or "all" in methods:
            print("\n>>> Running UHF...")
            self.results['uhf'] = self.scf_calc.run_uhf()
        
        

        if "mp2" in methods or "ump2" in methods or "all" in methods:
            if 'uhf' not in self.results:
                self.results['uhf'] = self.scf_calc.run_uhf()
            print("\n>>> Running UMP2...")
            self.results['ump2'] = self.scf_calc.run_ump2(
                self.results['uhf']
            )
        
        

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
        
        

        
        

        self._print_summary()
        
        return self.results
    
    

    

    

    def run_cholesky_workflow(self, threshold=1e-6, scf_type="uhf"):
        """
        Run complete Cholesky-based workflow:
        Decomp -> SCF (UHF/ROHF/RHF) -> Cholesky-OMP2 -> Cholesky-OMP3
        
        Args:
            threshold (float): Cholesky accuracy (e.g. 1e-6)
            scf_type (str): "uhf", "rohf", or "rhf"
        """
        print("="*70)
        print(f"MSHQC Cholesky Pipeline (Threshold={threshold}, SCF={scf_type.upper()})")
        print("="*70)
        
        

        print("\n>>> Step 1: Cholesky Decomposition...")
        import time
        t0 = time.time()
        
        

        engine = self.scf_calc.integrals 
        chol = mshqc.CholeskyERI(threshold)
        chol.decompose(engine.compute_eri())
        
        print(f"    Done ({time.time()-t0:.2f}s). Vectors: {chol.n_vectors()}")
        

        self.results['cholesky_eri'] = chol
        
        

        n_elec = self.molecule.n_electrons()
        mult = self.molecule.multiplicity()
        n_alpha = (n_elec + mult - 1) // 2
        n_beta = n_elec - n_alpha
        
        

        print(f"\n>>> Step 2: Cholesky-{scf_type.upper()}...")
        t0 = time.time()
        
        

        if scf_type.lower() == "rhf":
            

            rhf_conf = mshqc.CholeskyRHFConfig()
            rhf_conf.cholesky_threshold = threshold
            rhf_conf.print_level = 0
            
            

            c_scf = mshqc.CholeskyRHF(self.molecule, self.scf_calc.basis, engine,
                                      rhf_conf, chol)

        elif scf_type.lower() == "rohf":
            

            rohf_conf = mshqc.CholeskyROHFConfig()
            rohf_conf.cholesky_threshold = threshold
            rohf_conf.print_level = 0
            
            c_scf = mshqc.CholeskyROHF(self.molecule, self.scf_calc.basis, engine,
                                       n_alpha, n_beta, rohf_conf, chol)
        else:
            

            uhf_conf = mshqc.CholeskyUHFConfig()
            uhf_conf.cholesky_threshold = threshold
            uhf_conf.print_level = 0
            
            c_scf = mshqc.CholeskyUHF(self.molecule, self.scf_calc.basis, engine,
                                      n_alpha, n_beta, uhf_conf)
            c_scf.set_cholesky_vectors(chol.get_L_vectors())
        
        scf_res = c_scf.compute()
        print(f"    Done ({time.time()-t0:.2f}s). E = {scf_res.energy_total:.8f} Ha")
        self.results['scf'] = scf_res
        
        

        print("\n>>> Step 3: Cholesky-OMP2...")
        
        

        omp2_res = self.scf_calc.run_cholesky_omp2(
            scf_res, 
            existing_cholesky=chol  

        )
        self.results['omp2'] = omp2_res
        
        

        print("\n>>> Step 4: Cholesky-OMP3...")
        omp3_res = self.scf_calc.run_cholesky_omp3(
            omp2_res, 
            existing_cholesky=chol
        )
        self.results['omp3'] = omp3_res
        
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

    


    def run_benchmark_caspt2(self, n_states=3, n_active_elec=4, n_active_orb=3):
        """
        Run Benchmark: Canonical (Exact) vs Cholesky (Approx) SA-CASPT2
        """
        print("="*80)
        print("  MSHQC BENCHMARK: Canonical vs Cholesky SA-CASPT2")
        print("="*80)
        
        

        res_can = self.mcscf_calc.run_canonical_sa_caspt2_pipeline(
            n_states, n_active_elec, n_active_orb
        )
        
        

        

        n_frozen = self.molecule.n_electrons() - n_active_elec
        
        res_chol = self.mcscf_calc.run_sa_caspt3_pipeline(
            n_frozen=n_frozen,
            n_active_orb=n_active_orb,
            n_states=n_states
        )
        
        

        print("\n" + "="*65)
        print("  COMPARISON SUMMARY (Total Energy)")
        print("="*65)
        print(f"{'State':<6} {'Canonical (Ref)':<20} {'Cholesky (Approx)':<20} {'Diff (mHa)':<12}")
        print("-" * 65)
        
        for i in range(n_states):
            

            e_can = res_can['pt2_result'].e_total[i]
            
            

            

            

            

            

            
            

            

            
            e_chol = 0.0 

            if 'final_energies' in res_chol:
                 

                 pass

            

            print(f"{i:<6} {e_can:<20.8f} {'(Check Logs)':<20} {'-':<12}")
            
        return {"canonical": res_can, "cholesky": res_chol}
    


    def run_benchmark_caspt3(self, n_states=3, n_active_elec=4, n_active_orb=3):
        """
        Run Benchmark: Canonical vs Cholesky SA-CASPT3
        """
        print("="*80)
        print("  MSHQC BENCHMARK: Canonical vs Cholesky SA-CASPT3")
        print("="*80)
        
        

        res_can = self.mcscf_calc.run_canonical_sa_caspt3_pipeline(
            n_states, n_active_elec, n_active_orb
        )
        
        

        

        n_frozen = self.molecule.n_electrons() - n_active_elec
        
        res_chol = self.mcscf_calc.run_sa_caspt3_pipeline(
            n_frozen=n_frozen,
            n_active_orb=n_active_orb,
            n_states=n_states
        )
        
        

        print("\n" + "="*80)
        print("  COMPARISON SUMMARY (Total: CAS + PT2 + PT3)")
        print("="*80)
        print(f"{'State':<6} {'Canonical (Ref)':<20} {'Cholesky (Approx)':<20} {'Diff (mHa)':<12}")
        print("-" * 80)
        
        for i in range(n_states):
            

            e_cas = res_can['sacas_result'].state_energies[i]
            e_pt2 = res_can['pt2_result'].e_pt2[i]
            e_pt3 = res_can['pt3_result'].e_pt3[i]
            e_can_total = e_cas + e_pt2 + e_pt3
            
            

            e_chol_total = res_chol['final_energies'][i]
            
            diff = (e_can_total - e_chol_total) * 1000.0
            print(f"{i:<6} {e_can_total:<20.8f} {e_chol_total:<20.8f} {diff:<12.4f}")
            
        return {"canonical": res_can, "cholesky": res_chol}