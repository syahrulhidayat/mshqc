"""
High-Level Calculator Interfaces for MSHQC
File: /home/syahrul/mshqc/python/mshqc/calculators.py
"""

import numpy as np
import time
import os
from typing import Dict, Optional

# Import langsung dari package mshqc
import mshqc

class MSHQCCalculator:
    """High-level interface untuk SCF dan MP calculations"""
    
    def __init__(self, molecule, basis_name: str, basis_dir: Optional[str] = None):
        """
        Initialize calculator
        """
        self.molecule = molecule
        self.basis_name = basis_name
        
        # Logika penentuan path basis
        target_dir = basis_dir
        if target_dir is None:
            if 'MSHQC_DATA_DIR' in os.environ:
                target_dir = os.path.join(os.environ['MSHQC_DATA_DIR'], 'basis')
            else:
                target_dir = "data/basis"
        
        self.basis_dir = target_dir
        
        # Create basis and integrals using target_dir (dijamin string)
        self.basis = mshqc.BasisSet(basis_name, molecule, target_dir)
        self.integrals = mshqc.IntegralEngine(molecule, self.basis)
        self.n_basis = self.basis.n_basis_functions()
        
    # --- SCF Methods ---

    def run_uhf(self, n_alpha=None, n_beta=None, config=None):
        """Run UHF calculation"""
        if n_alpha is None:
            n_alpha = (self.molecule.n_electrons() + 
                      self.molecule.multiplicity() - 1) // 2
        if n_beta is None:
            n_beta = self.molecule.n_electrons() - n_alpha
        if config is None:
            config = mshqc.SCFConfig()
            
        print(f"  Running UHF ({self.n_basis} basis functions)...")
        t0 = time.time()
        uhf = mshqc.UHF(self.molecule, self.basis, self.integrals,
                       n_alpha, n_beta, config)
        res = uhf.compute()
        print(f"  Done ({time.time() - t0:.2f}s). E: {res.energy_total:.8f} Ha")
        return res
    
    def run_rhf(self, config=None):
        """Run RHF calculation (Closed Shell)"""
        if config is None:
            config = mshqc.SCFConfig()
            
        print(f"  Running RHF ({self.n_basis} basis functions)...")
        t0 = time.time()
        rhf = mshqc.RHF(self.molecule, self.basis, self.integrals, config)
        res = rhf.compute()
        print(f"  Done ({time.time() - t0:.2f}s). E: {res.energy_total:.8f} Ha")
        return res
    
    # --- Unrestricted MP Methods (UMP) ---

    def run_ump2(self, scf_result):
        """Run UMP2 calculation"""
        print("  Running UMP2...")
        t0 = time.time()
        ump2 = mshqc.UMP2(scf_result, self.basis, self.integrals)
        res = ump2.compute()
        print(f"  Done ({time.time() - t0:.2f}s). E_Corr: {res.e_corr_total:.8f}")
        return res
    
    def run_ump3(self, scf_result, ump2_result):
        """Run UMP3 calculation"""
        print("  Running UMP3...")
        t0 = time.time()
        ump3 = mshqc.UMP3(scf_result, ump2_result, 
                         self.basis, self.integrals)
        res = ump3.compute()
        print(f"  Done ({time.time() - t0:.2f}s). E_MP3: {res.e_mp3_corr:.8f}")
        return res
    def run_cholesky_ump2(self, uhf_result, threshold=1e-6):
        """Run Fast Cholesky-UMP2"""
        print(f"  Running Cholesky-UMP2 (Threshold={threshold})...")
        t0 = time.time()
        
        # 1. Setup Config
        config = mshqc.CholeskyUMP2Config()
        config.cholesky_threshold = threshold
        config.print_level = 1
        
        # 2. Decompose (atau reuse jika sudah ada di session)
        # Di sini kita decompose baru untuk simplicitas
        print("    > Decomposing Integrals...")
        chol = mshqc.CholeskyERI(threshold)
        eri = self.integrals.compute_eri()
        chol.decompose(eri)
        
        # 3. Run Compute
        # Perhatikan: Konstruktor butuh objek CholeskyERI
        ump2 = mshqc.CholeskyUMP2(uhf_result, self.basis, self.integrals, config, chol)
        res = ump2.compute()
        
        print(f"  Done ({time.time() - t0:.2f}s). E_Corr: {res.e_corr_total:.8f}")
        return res
    # --- Restricted MP Methods (RMP - NEW) ---

    def run_rmp2(self, rhf_result):
        """Run RMP2 calculation (Closed Shell)"""
        print("  Running RMP2...")
        t0 = time.time()
        rmp2 = mshqc.RMP2(rhf_result, self.basis, self.integrals)
        res = rmp2.compute()
        print(f"  Done ({time.time() - t0:.2f}s). E_Corr: {res.e_corr:.8f}")
        return res
    
    def run_rmp3(self, rhf_result, rmp2_result):
        """Run RMP3 calculation (Closed Shell)"""
        print("  Running RMP3...")
        t0 = time.time()
        rmp3 = mshqc.RMP3(rhf_result, rmp2_result, 
                         self.basis, self.integrals)
        res = rmp3.compute()
        print(f"  Done ({time.time() - t0:.2f}s). E_MP3: {res.e_mp3:.8f}")
        return res

    # --- Orbital Optimized MP Methods (OMP - NEW) ---

    def run_omp2(self, scf_result, max_iter=50):
        """Run Orbital-Optimized MP2"""
        print("  Running OMP2 (Orbital Optimized)...")
        t0 = time.time()
        omp2 = mshqc.OMP2(self.molecule, self.basis, self.integrals, scf_result)
        omp2.set_max_iterations(max_iter)
        res = omp2.compute()
        print(f"  Done ({time.time() - t0:.2f}s). E_OMP2: {res.energy_total:.8f}")
        return res
    
    def run_omp3(self, omp2_result, max_iter=50):
        """Run Orbital-Optimized MP3"""
        print("  Running OMP3 (Orbital Optimized)...")
        t0 = time.time()
        omp3 = mshqc.OMP3(self.molecule, self.basis, self.integrals, omp2_result)
        omp3.set_max_iterations(max_iter)
        res = omp3.compute()
        print(f"  Done ({time.time() - t0:.2f}s). E_OMP3: {res.energy_total:.8f}")
        return res
    
    # --- Cholesky Pipeline ---

    def run_cholesky_pipeline(self, n_alpha=None, n_beta=None, threshold=1e-6):
        """Run complete Cholesky pipeline"""
        results = {}
        
        print("Step 1: Cholesky Decomposition...")
        t0 = time.time()
        chol = mshqc.CholeskyERI(threshold)
        eri_tensor = self.integrals.compute_eri()
        chol.decompose(eri_tensor)
        results['cholesky_time'] = time.time() - t0
        results['n_cholesky_vectors'] = chol.n_vectors()
        
        if n_alpha is None:
            n_alpha = (self.molecule.n_electrons() + 
                      self.molecule.multiplicity() - 1) // 2
        if n_beta is None:
            n_beta = self.molecule.n_electrons() - n_alpha
            
        print("Step 2: Cholesky UHF...")
        t0 = time.time()
        uhf_config = mshqc.CholeskyUHFConfig()
        uhf_config.cholesky_threshold = threshold
        uhf_config.print_level = 0
        
        uhf = mshqc.CholeskyUHF(self.molecule, self.basis, self.integrals,
                               n_alpha, n_beta, uhf_config)
        uhf.set_cholesky_vectors(chol.get_L_vectors())
        uhf_result = uhf.compute()
        results['uhf_time'] = time.time() - t0
        results['uhf_result'] = uhf_result
        
        return results
    

    def run_cholesky_rohf(self, n_alpha=None, n_beta=None, config=None, existing_cholesky=None):
        """
        Run Cholesky-ROHF calculation
        
        Args:
            n_alpha (int): Jumlah elektron alpha (opsional, auto-detect dari molekul)
            n_beta (int): Jumlah elektron beta (opsional)
            config (CholeskyROHFConfig): Konfigurasi SCF
            existing_cholesky (CholeskyERI): Objek Cholesky dari perhitungan sebelumnya (untuk reuse)
        """
        # 1. Ambil total elektron dari molekul
        n_total = self.molecule.n_electrons()
        
        # 2. Jika n_alpha tidak diberikan, hitung berdasarkan multiplisitas
        if n_alpha is None:
            mult = self.molecule.multiplicity()
            n_alpha = (n_total + mult - 1) // 2
            
        # 3. Jika n_beta tidak diberikan (walaupun n_alpha ada), hitung sisanya
        # [FIX] Ini mengatasi error Pylance dimana n_beta bisa tertinggal sebagai None
        if n_beta is None:
            n_beta = n_total - n_alpha
            
        # 4. Inisialisasi config default jika kosong
        if config is None:
            config = mshqc.CholeskyROHFConfig()
            
        print(f"\nRunning Cholesky-ROHF (Na={n_alpha}, Nb={n_beta})...")
        t0 = time.time()
        
        # 5. Pilih konstruktor yang sesuai (Reuse vs Standard)
        if existing_cholesky:
            # Mode Reuse Vectors (Cepat)
            scf = mshqc.CholeskyROHF(
                self.molecule, self.basis, self.integrals,
                n_alpha, n_beta, config, existing_cholesky
            )
        else:
            # Mode Standard (Decompose from scratch)
            scf = mshqc.CholeskyROHF(
                self.molecule, self.basis, self.integrals,
                n_alpha, n_beta, config
            )
            
        # 6. Jalankan komputasi
        result = scf.compute()
        print(f"Done in {time.time()-t0:.2f}s. E = {result.energy_total:.8f} Ha")
        return result
    

    # Tambahkan di dalam class MSHQCCalculator

    def run_cholesky_rhf(self, config=None, existing_cholesky=None):
        """
        Run Cholesky-RHF calculation (Closed-Shell)
        
        Args:
            config (CholeskyRHFConfig): Konfigurasi SCF
            existing_cholesky (CholeskyERI): Objek Cholesky untuk reuse vectors
        """
        if config is None:
            config = mshqc.CholeskyRHFConfig()
            
        print(f"\nRunning Cholesky-RHF...")
        t0 = time.time()
        
        if existing_cholesky:
            # Mode Reuse Vectors
            scf = mshqc.CholeskyRHF(
                self.molecule, self.basis, self.integrals,
                config, existing_cholesky
            )
        else:
            # Mode Standard
            scf = mshqc.CholeskyRHF(
                self.molecule, self.basis, self.integrals,
                config
            )
            
        result = scf.compute()
        print(f"Done in {time.time()-t0:.2f}s. E = {result.energy_total:.8f} Ha")
        return result
    
    def run_cholesky_omp2(self, scf_result, config=None, existing_cholesky=None):
        """
        Run Cholesky Orbital-Optimized MP2 (OMP2)
        
        Args:
            scf_result: Hasil SCF (UHF/ROHF) sebagai tebakan awal orbital
            config: Konfigurasi CholeskyOMP2Config
            existing_cholesky: Objek CholeskyERI (untuk reuse vektor)
        """
        if config is None:
            config = mshqc.CholeskyOMP2Config()
            
        print(f"\nRunning Cholesky-OMP2...")
        t0 = time.time()
        
        if existing_cholesky:
            # Mode Reuse Vectors (Sangat Cepat)
            omp2 = mshqc.CholeskyOMP2(
                self.molecule, self.basis, self.integrals,
                scf_result, config, existing_cholesky
            )
        else:
            # Mode Standard (Decompose sendiri)
            omp2 = mshqc.CholeskyOMP2(
                self.molecule, self.basis, self.integrals,
                scf_result, config
            )
            
        result = omp2.compute()
        print(f"Done in {time.time()-t0:.2f}s.")
        print(f"  => E(SCF) : {result.energy_scf:.8f} Ha")
        print(f"  => E(OMP2): {result.energy_total:.8f} Ha")
        return result
    # ... (Di dalam class MSHQCCalculator, setelah run_cholesky_omp2) ...

    def run_cholesky_omp3(self, omp2_result, config=None, existing_cholesky=None):
        """
        Run Cholesky Orbital-Optimized MP3 (OMP3)
        
        Args:
            omp2_result: Hasil OMP2Result (sebagai tebakan awal)
            config: Konfigurasi CholeskyOMP3Config
            existing_cholesky: Objek CholeskyERI (WAJIB ADA untuk OMP3 ini)
        """
        if config is None:
            config = mshqc.CholeskyOMP3Config()
            
        if existing_cholesky is None:
            raise ValueError("Cholesky-OMP3 requires existing Cholesky vectors used in previous steps.")

        print(f"\nRunning Cholesky-OMP3...")
        t0 = time.time()
        
        omp3 = mshqc.CholeskyOMP3(
            self.molecule, self.basis, self.integrals,
            omp2_result, config, existing_cholesky
        )
            
        result = omp3.compute()
        print(f"Done in {time.time()-t0:.2f}s.")
        print(f"  => E(Total): {result.energy_total:.8f} Ha")
        return result



# ... (di dalam class MSHQCCalculator, misal setelah run_rmp2) ...

    def run_cholesky_rmp2(self, rhf_result, config=None, existing_cholesky=None):
        """
        Run Cholesky-Restricted MP2 (RMP2)
        
        Args:
            rhf_result: Hasil RHF (SCFResult)
            config: Konfigurasi CholeskyRMP2Config
            existing_cholesky: Objek CholeskyERI (untuk reuse vektor)
        """
        if config is None:
            config = mshqc.CholeskyRMP2Config()
            
        print(f"\nRunning Cholesky-RMP2...")
        t0 = time.time()
        
        if existing_cholesky:
            # Mode Reuse Vectors (Sangat Cepat)
            mp2 = mshqc.CholeskyRMP2(
                self.molecule, self.basis, self.integrals,
                rhf_result, config, existing_cholesky
            )
        else:
            # Mode Standard (Decompose sendiri)
            mp2 = mshqc.CholeskyRMP2(
                self.molecule, self.basis, self.integrals,
                rhf_result, config
            )
            
        result = mp2.compute()
        print(f"Done in {time.time()-t0:.2f}s.")
        print(f"  => E(Corr): {result.e_corr:.8f} Ha")
        print(f"  => E(Total): {result.e_total:.8f} Ha")
        return result
    
    # Letakkan setelah run_cholesky_rmp2

    def run_cholesky_rmp3(self, rhf_result, crmp2_result):
        """
        Run Cholesky-Restricted MP3 (RMP3)
        
        Args:
            rhf_result: Hasil RHF (SCFResult)
            crmp2_result: Hasil CholeskyRMP2Result (WAJIB dari run_cholesky_rmp2)
                          Ini membawa vektor Cholesky yang akan direuse.
        """
        print(f"\nRunning Cholesky-RMP3 (Reuse Vectors)...")
        t0 = time.time()
        
        # Inisialisasi solver
        rmp3 = mshqc.CholeskyRMP3(rhf_result, crmp2_result, self.basis)
            
        result = rmp3.compute()
        print(f"Done in {time.time()-t0:.2f}s.")
        print(f"  => E(MP3 Corr): {result.e_mp3:.8f} Ha")
        print(f"  => E(Total)   : {result.e_total:.8f} Ha")
        return result

class MCSCFCalculator:
    """High-level interface untuk MCSCF calculations"""
    
    def __init__(self, molecule, basis_name: str, basis_dir: Optional[str] = None):
        """Initialize MCSCF calculator"""
        self.molecule = molecule
        
        # Logika penentuan path basis
        target_dir = basis_dir
        if target_dir is None:
            if 'MSHQC_DATA_DIR' in os.environ:
                target_dir = os.path.join(os.environ['MSHQC_DATA_DIR'], 'basis')
            else:
                target_dir = "data/basis"
        
        # Create basis using target_dir (dijamin string)
        self.basis = mshqc.BasisSet(basis_name, molecule, target_dir)
        self.integrals = mshqc.IntegralEngine(molecule, self.basis)
        self.n_basis = self.basis.n_basis_functions()
        
    def run_casscf(self, n_active_elec: int, n_active_orb: int,
                   scf_guess=None):
        """Run standard CASSCF"""
        active_space = mshqc.ActiveSpace.CAS(
            n_active_elec, n_active_orb,
            self.n_basis, self.molecule.n_electrons()
        )
        
        casscf = mshqc.CASSCF(self.molecule, self.basis,
                             self.integrals, active_space)
        # Set default iterations
        casscf.set_max_iterations(50)
        
        if scf_guess is not None:
            return casscf.compute(scf_guess)
        else:
            calc = MSHQCCalculator(self.molecule, self.basis.name())
            uhf_result = calc.run_uhf()
            return casscf.compute(uhf_result)

    def run_standard_caspt2(self, casscf_result):
        """Run Standard CASPT2 (Non-Cholesky)"""
        print("  Running Standard CASPT2...")
        t0 = time.time()
        # Menggunakan class CASPT2 standar (bukan Cholesky)
        caspt2 = mshqc.CASPT2(self.molecule, self.basis, self.integrals, casscf_result)
        res = caspt2.compute()
        print(f"  Done ({time.time() - t0:.2f}s). E_Total: {res.e_total:.8f}")
        return res
    
    def run_sa_caspt3_pipeline(self, n_frozen: int, n_active_orb: int,
                               n_states: int, threshold=1e-8, shift=0.0):
        """Run complete SA-CASPT3 pipeline"""
        results = {}
        
        print("="*70)
        print(f"SA-CASPT3 Pipeline: {n_states} states")
        print("="*70)
        
        # Cholesky
        print("\n[1/6] Cholesky Decomposition...")
        t0 = time.time()
        chol = mshqc.CholeskyERI(threshold)
        eri_tensor = self.integrals.compute_eri()
        chol.decompose(eri_tensor)
        L_vectors = chol.get_L_vectors()
        results['t_chol'] = time.time() - t0
        
        # UHF
        print("[2/6] Cholesky UHF...")
        t0 = time.time()
        n_alpha = (self.molecule.n_electrons() + 
                  self.molecule.multiplicity() - 1) // 2
        n_beta = self.molecule.n_electrons() - n_alpha
        
        uhf_config = mshqc.CholeskyUHFConfig()
        uhf_config.cholesky_threshold = threshold
        uhf_config.print_level = 0
        
        uhf = mshqc.CholeskyUHF(self.molecule, self.basis, self.integrals,
                               n_alpha, n_beta, uhf_config)
        uhf.set_cholesky_vectors(L_vectors)
        uhf_result = uhf.compute()
        results['t_uhf'] = time.time() - t0
        
        # UNO
        print("[3/6] UNO Generation...")
        t0 = time.time()
        uno_gen = mshqc.CholeskyUNO(uhf_result, self.integrals, self.n_basis)
        uno_result = uno_gen.compute()
        results['t_uno'] = time.time() - t0
        
        # SA-CASSCF
        print(f"[4/6] SA-CASSCF ({n_states} states)...")
        t0 = time.time()
        active_space = mshqc.ActiveSpace.CAS_Frozen(
            n_frozen, n_active_orb, self.n_basis,
            self.molecule.n_electrons()
        )
        
        sa_config = mshqc.SACASConfig()
        sa_config.set_equal_weights(n_states)
        sa_config.max_iter = 100
        
        sa_casscf = mshqc.CholeskySACASSCF(
            self.molecule, self.basis, self.integrals,
            active_space, sa_config, L_vectors
        )
        sa_result = sa_casscf.compute(uno_result.C_uno)
        results['t_casscf'] = time.time() - t0
        
        # CASPT2
        print("[5/6] SA-CASPT2...")
        t0 = time.time()
        pt2_config = mshqc.CASPT2Config()
        pt2_config.shift = shift
        
        pt2_solver = mshqc.CholeskySACASPT2(
            sa_result, L_vectors, self.n_basis,
            active_space, pt2_config
        )
        pt2_result = pt2_solver.compute()
        results['t_pt2'] = time.time() - t0
        
        # CASPT3
        print("[6/6] SA-CASPT3...")
        t0 = time.time()
        pt3_config = mshqc.CASPT3Config()
        pt3_config.shift = shift
        
        pt3_solver = mshqc.CholeskySACASPT3(
            sa_result, L_vectors, self.n_basis,
            active_space, pt3_config
        )
        pt3_result = pt3_solver.compute()
        results['t_pt3'] = time.time() - t0
        
        # Final energies
        results['final_energies'] = []
        
        for i in range(n_states):
            # Binding e_pt2/e_pt3 mengembalikan List[float]
            e_pt2_val = pt2_result.e_pt2[i]
            e_pt3_val = pt3_result.e_pt3[i]
            
            e_total = sa_result.state_energies[i] + e_pt2_val + e_pt3_val
            results['final_energies'].append(e_total)
        
        return results