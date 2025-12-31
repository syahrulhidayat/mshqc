# File: /home/syahrul/mshqc/python/mshqc/_mshqc.pyi
import numpy as np
from typing import List, Optional, Tuple, overload, Any, Union

# ==========================================
# Core Classes
# ==========================================

class ERITensor:
    def size(self) -> int: ...
    def dimension(self, index: int) -> int: ...

class Atom:
    atomic_number: int
    x: float
    y: float
    z: float
    def __init__(self, atomic_number: int, x: float, y: float, z: float) -> None: ...
    def position(self) -> List[float]: ...

class Molecule:
    # Perbaikan Overload di sini
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, charge: int, multiplicity: int) -> None: ...
    
    def __init__(self, *args, **kwargs) -> None: ... # Catch-all implementation for type checker
    
    @overload
    def add_atom(self, Z: int, x: float, y: float, z: float) -> None: ...
    @overload
    def add_atom(self, atom: Atom) -> None: ...
    
    def n_atoms(self) -> int: ...
    def atom(self, i: int) -> Atom: ...
    def total_nuclear_charge(self) -> float: ...
    def n_electrons(self) -> int: ...
    def charge(self) -> int: ...
    def set_charge(self, q: int) -> None: ...
    def multiplicity(self) -> int: ...
    def set_multiplicity(self, m: int) -> None: ...
    def nuclear_repulsion_energy(self) -> float: ...
    def atoms(self) -> List[Atom]: ...

class BasisSet:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, basis_name: str, mol: Molecule, basis_dir: str = "data/basis") -> None: ...
    
    def __init__(self, *args, **kwargs) -> None: ...
    
    def n_basis_functions(self) -> int: ...
    def name(self) -> str: ...
    def read_gbs(self, basis_file: str, mol: Molecule) -> None: ...
    def add_shell(self, shell: Shell) -> None: ...
    def n_shells(self) -> int: ...
    def shell(self, i: int) -> Shell: ...
class AngularMomentum:
    S: int
    P: int
    D: int
    F: int
    G: int
    H: int

class GaussianPrimitive:
    exponent: float
    coefficient: float
    def __init__(self, exponent: float, coefficient: float) -> None: ...

class Shell:
    def __init__(self, am: AngularMomentum, center: int, center_pos: List[float]) -> None: ...
    @property
    def angular_momentum(self) -> AngularMomentum: ...
    @property
    def center(self) -> int: ...
    @property
    def primitives(self) -> List[GaussianPrimitive]: ...

class CholeskyDecompositionResult:
    n_vectors: int
    n_basis: int
    threshold: float
    compression_ratio: float
    converged: bool

class IntegralEngine:
    def __init__(self, mol: Molecule, basis: BasisSet) -> None: ...
    def compute_eri(self) -> ERITensor: ...
    def compute_overlap(self) -> np.ndarray: ...
    def compute_kinetic(self) -> np.ndarray: ...
    def compute_nuclear(self) -> np.ndarray: ...
    def compute_eri_diagonal(self) -> np.ndarray: ...
    def compute_eri_column(self, pivot_index: int) -> np.ndarray: ...
    
class CholeskyERI:
    def __init__(self, threshold: float = 1e-6) -> None: ...
    def decompose(self, eri_tensor: ERITensor) -> None: ...
    def get_L_vectors(self) -> List[np.ndarray]: ...
    def reconstruct(self) -> ERITensor: ...
    def n_vectors(self) -> int: ...
    def threshold(self) -> float: ...

# ==========================================
# SCF / UHF
# ==========================================

class CholeskyUHFConfig:
    cholesky_threshold: float
    print_level: int
    max_iterations: int        
    energy_threshold: float     
    def __init__(self) -> None: ...

class CholeskyUHF:
    def __init__(self, molecule: Molecule, basis: BasisSet, 
                 integrals: IntegralEngine, n_alpha: int, n_beta: int,
                 config: CholeskyUHFConfig = ...) -> None: ...
    def set_cholesky_vectors(self, vectors: List[np.ndarray]) -> None: ...
    def compute(self) -> SCFResult: ...
class SCFConfig:
    max_iterations: int
    energy_threshold: float
    density_threshold: float
    diis_threshold: float
    diis_max_vectors: int
    print_level: int
    level_shift: float
    def __init__(self) -> None: ...

# Update SCFResult agar lengkap
class SCFResult:
    energy_electronic: float
    energy_nuclear: float
    energy_total: float
    orbital_energies_alpha: np.ndarray
    orbital_energies_beta: np.ndarray
    C_alpha: np.ndarray
    C_beta: np.ndarray
    P_alpha: np.ndarray
    P_beta: np.ndarray
    F_alpha: np.ndarray
    F_beta: np.ndarray
    iterations: int
    converged: bool
    gradient_norm: float
    n_occ_alpha: int
    n_occ_beta: int
    def __init__(self) -> None: ...

class UHF:
    def __init__(self, molecule: Molecule, basis: BasisSet, 
                 integrals: IntegralEngine, n_alpha: int, n_beta: int, 
                 config: SCFConfig = ...) -> None: ...
    def compute(self) -> SCFResult: ...

class RHF:
    def __init__(self, molecule: Molecule, basis: BasisSet, 
                 integrals: IntegralEngine, config: SCFConfig = ...) -> None: ...
    def compute(self) -> SCFResult: ...

class ROHF:
    def __init__(self, molecule: Molecule, basis: BasisSet, 
                 n_alpha: int, n_beta: int, config: SCFConfig = ...) -> None: ...
    def run(self) -> SCFResult: ...


# File: /home/syahrul/mshqc/python/mshqc/_mshqc.pyi

class CholeskyROHFConfig(SCFConfig):
    cholesky_threshold: float
    screen_exchange: bool
    def __init__(self) -> None: ...

class CholeskyROHF:
    # Constructor 1: Standard
    @overload
    def __init__(self, molecule: Molecule, basis: BasisSet, 
                 integrals: IntegralEngine, n_alpha: int, n_beta: int,
                 config: CholeskyROHFConfig = ...) -> None: ...
    
    # Constructor 2: Reuse Vectors
    @overload
    def __init__(self, molecule: Molecule, basis: BasisSet, 
                 integrals: IntegralEngine, n_alpha: int, n_beta: int,
                 config: CholeskyROHFConfig, 
                 existing_cholesky: CholeskyERI) -> None: ...

    def compute(self) -> SCFResult: ...

class CholeskyRHFConfig(SCFConfig):
    cholesky_threshold: float
    screen_exchange: bool
    def __init__(self) -> None: ...

class CholeskyRHF:
    # Constructor 1: Standard
    @overload
    def __init__(self, molecule: Molecule, basis: BasisSet, 
                 integrals: IntegralEngine,
                 config: CholeskyRHFConfig = ...) -> None: ...
    
    # Constructor 2: Reuse Vectors
    @overload
    def __init__(self, molecule: Molecule, basis: BasisSet, 
                 integrals: IntegralEngine,
                 config: CholeskyRHFConfig, 
                 existing_cholesky: CholeskyERI) -> None: ...

    def compute(self) -> SCFResult: ...
    def energy(self) -> float: ...



class CholeskyOMP2Config:
    max_iterations: int
    energy_threshold: float
    gradient_threshold: float
    cholesky_threshold: float
    print_level: int
    def __init__(self) -> None: ...

class CholeskyOMP2:
    # Constructor 1: Standard
    @overload
    def __init__(self, molecule: Molecule, basis: BasisSet, 
                 integrals: IntegralEngine, scf_guess: SCFResult,
                 config: CholeskyOMP2Config = ...) -> None: ...
    
    # Constructor 2: Reuse Vectors
    @overload
    def __init__(self, molecule: Molecule, basis: BasisSet, 
                 integrals: IntegralEngine, scf_guess: SCFResult,
                 config: CholeskyOMP2Config, 
                 existing_cholesky: CholeskyERI) -> None: ...

    def compute(self) -> OMP2Result: ...
# ... (Di bagian bawah file, dekat definisi OMP/MP lainnya) ...

class CholeskyOMP3Config:
    max_iterations: int
    energy_threshold: float
    gradient_threshold: float
    cholesky_threshold: float
    print_level: int
    def __init__(self) -> None: ...

class CholeskyOMP3Result:
    energy_scf: float
    energy_mp2_corr: float
    energy_mp3_corr: float
    energy_total: float
    converged: bool
    iterations: int
    C_alpha: np.ndarray
    C_beta: np.ndarray
    orbital_energies_alpha: np.ndarray
    orbital_energies_beta: np.ndarray
    def __init__(self) -> None: ...

class CholeskyOMP3:
    def __init__(self, mol: Molecule, basis: BasisSet, 
                 integrals: IntegralEngine, 
                 omp2_guess: OMP2Result, 
                 config: CholeskyOMP3Config,
                 cholesky_vectors: CholeskyERI) -> None: ...
    def compute(self) -> CholeskyOMP3Result: ...

# ... (kode sebelumnya: CholeskyOMP3, dll) ...

class CholeskyRMP2Config:
    cholesky_threshold: float
    print_level: int
    def __init__(self) -> None: ...

# [PERBAIKAN 1]: Definisi Class CholeskyRMP2Result DITAMBAHKAN DI SINI
class CholeskyRMP2Result:
    e_rhf: float
    e_corr: float
    e_total: float
    n_chol_vectors: int
    def __init__(self) -> None: ...

class CholeskyRMP2:
    # Constructor 1: Standard
    @overload
    def __init__(self, molecule: Molecule, basis: BasisSet, 
                 integrals: IntegralEngine, rhf_result: SCFResult,
                 config: CholeskyRMP2Config = ...) -> None: ...
    
    # Constructor 2: Reuse Vectors
    @overload
    def __init__(self, molecule: Molecule, basis: BasisSet, 
                 integrals: IntegralEngine, rhf_result: SCFResult,
                 config: CholeskyRMP2Config, 
                 existing_cholesky: CholeskyERI) -> None: ...
    
    def __init__(self, *args, **kwargs) -> None: ...

    # [PERBAIKAN 2]: Hapus duplikasi compute. Gunakan satu saja yang benar.
    # def compute(self) -> RMP2Result: ...  <-- HAPUS INI (Salah Tipe)
    def compute(self) -> CholeskyRMP2Result: ... 

# [PERBAIKAN 3]: Pastikan CholeskyRMP3 menggunakan CholeskyRMP2Result yang sudah didefinisikan di atas
class CholeskyRMP3:
    def __init__(self, rhf_result: SCFResult, 
                 crmp2_result: CholeskyRMP2Result, 
                 basis: BasisSet) -> None: ...
    def compute(self) -> RMP3Result: ...

# ... (sisa kode MCSCF dll) ...
# ==========================================
# MCSCF / CASSCF
# ==========================================

class ActiveSpace:
    def __init__(self) -> None: ...
    
    # Fungsi ini yang kamu panggil di python, tapi belum ada di binding C++ (lihat langkah 2)
    @staticmethod
    def CAS_Frozen(n_frozen: int, n_active_orb: int, n_basis: int, n_electrons: int) -> ActiveSpace: ...
    
    @staticmethod
    def CAS(n_elec: int, n_orb: int, n_total_orb: int, n_total_elec: int) -> ActiveSpace: ...
    def n_inactive(self) -> int: ...
    def n_active(self) -> int: ...
    def n_virtual(self) -> int: ...
    def n_elec_active(self) -> int: ...

class UNOResult:
    C_uno: np.ndarray
    def __init__(self) -> None: ...
    occupations: np.ndarray
    entropy: float
    active_indices: List[int]

class CholeskyUNO:
    def __init__(self, uhf_res: SCFResult, integrals: IntegralEngine, n_basis: int) -> None: ...
    def compute(self) -> UNOResult: ...
    def print_report(self, threshold: float = 0.02) -> None: ...
    def save_orbitals(self, filename: str) -> None: ...

class SACASConfig:
    n_states: int
    max_iter: int
    cholesky_thresh: float
    weights: List[float]        
    e_thresh: float              
    grad_thresh: float           
    print_level: int             
    rotation_damping: float      
    shift: float                 
    def __init__(self) -> None: ...
    def set_equal_weights(self, n_states: int) -> None: ...

class SACASResult:
    state_energies: List[float]
    def __init__(self) -> None: ...
    e_avg: float
    C_mo: np.ndarray
    orbital_energies: np.ndarray
    converged: bool

class CholeskySACASSCF:
    @overload
    def __init__(self, mol: Molecule, basis: BasisSet, integrals: IntegralEngine, 
                 active_space: ActiveSpace, config: SACASConfig) -> None: ...
    @overload
    def __init__(self, mol: Molecule, basis: BasisSet, integrals: IntegralEngine, 
                 active_space: ActiveSpace, config: SACASConfig, 
                 L_vectors: List[np.ndarray]) -> None: ...
                 
    def __init__(self, *args, **kwargs) -> None: ...
                 
    def compute(self, initial_guess: np.ndarray) -> SACASResult: ...
class CASResult:
    e_casscf: float
    e_nuclear: float
    n_iterations: int
    converged: bool
    C_mo: np.ndarray
    orbital_energies: np.ndarray
    ci_coeffs: np.ndarray
    determinants: List[Any]
    n_determinants: int
    active_space: ActiveSpace
    def __init__(self) -> None: ...

class CASSCF:
    def __init__(self, mol: Molecule, basis: BasisSet, integrals: IntegralEngine, active_space: ActiveSpace) -> None: ...
    
    # Update metode compute untuk menerima SCFResult atau numpy array
    def compute(self, initial_guess: Union[SCFResult, np.ndarray]) -> CASResult: ...
    
    # Tambahkan metode setter ini
    def set_max_iterations(self, n: int) -> None: ...
    def set_energy_threshold(self, t: float) -> None: ...
    def set_gradient_threshold(self, t: float) -> None: ...
    def set_ci_solver(self, solver_type: str) -> None: ...

class CASPT2Result1:
    e_casscf: float
    e_pt2: float
    e_total: float
    converged: bool
    status_message: str
    def __init__(self) -> None: ...

class CASPT2:
    def __init__(self, mol: Molecule, basis: BasisSet, integrals: IntegralEngine, casscf_result: CASResult) -> None: ...
    def compute(self) -> CASPT2Result1: ...

class CholeskyCASSCF:
    @overload
    def __init__(self, mol: Molecule, basis: BasisSet, integrals: IntegralEngine, active_space: ActiveSpace) -> None: ...
    @overload
    def __init__(self, mol: Molecule, basis: BasisSet, integrals: IntegralEngine, active_space: ActiveSpace, vectors: List[np.ndarray]) -> None: ...
    def compute(self) -> CASResult: ...

# ==========================================
# PT2 / PT3
# ==========================================

class CASPT2Config:
    shift: float
    print_level: int
    def __init__(self) -> None: ...
    export_amplitudes: bool

class CASPT2Result:
    e_pt2: List[float]
    def __init__(self) -> None: ...
    e_total: List[float]

# [UPDATE DALAM class CholeskySACASPT2]

class CholeskySACASPT2:
    def __init__(self, result: SACASResult, L_vectors: List[np.ndarray], n_basis: int, active_space: ActiveSpace, config: CASPT2Config) -> None: ...
    
    # Hapus argumen opsional jika ada, jadikan tanpa argumen
    def compute(self) -> CASPT2Result: ...

class CASPT3Config:
    shift: float
    print_level: int
    zero_thresh: float
    def __init__(self) -> None: ...

class CASPT3Result:
    e_pt3: List[float]
    def __init__(self) -> None: ...
    e_cas: float
    e_pt2: float
    e_total: float

class CholeskySACASPT3:
    def __init__(self, sacas_res: SACASResult, L_vectors: List[np.ndarray], 
                 n_basis: int, active_space: ActiveSpace, config: CASPT3Config) -> None: ...
    def compute(self) -> CASPT3Result: ...

# --- MP2 / MP3 ---
class OMP2:
    def __init__(self, mol: Molecule, basis: BasisSet, integrals: IntegralEngine, scf_guess: SCFResult) -> None: ...
    def compute(self) -> OMP2Result: ...
    def set_max_iterations(self, n: int) -> None: ...
    def set_convergence_threshold(self, t: float) -> None: ...
    def set_gradient_threshold(self, t: float) -> None: ...

class OMP2Result:
    energy_scf: float
    energy_mp2_ss: float
    energy_mp2_os: float
    energy_mp2_corr: float
    energy_total: float
    converged: bool          
    iterations: int
    n_occ_alpha: int
    n_occ_beta: int
    n_virt_alpha: int
    n_virt_beta: int
    def __init__(self) -> None: ...

class OMP3Result:
    energy_total: float
    energy_mp3_corr: float
    def __init__(self) -> None: ...

class OMP3:
    def __init__(self, mol: Molecule, basis: BasisSet, integrals: IntegralEngine, omp2_res: OMP2Result) -> None: ...
    def compute(self) -> OMP3Result: ...
    def set_max_iterations(self, n: int) -> None: ...
    def set_convergence_threshold(self, t: float) -> None: ...

class UMP2Result:
    e_corr_ss_aa: float
    e_corr_ss_bb: float
    e_corr_os: float
    e_corr_total: float
    e_total: float
    def __init__(self) -> None: ...

class UMP3Result:
    e_uhf: float
    e_mp2: float
    e_mp3_corr: float
    e_corr_total: float
    e_total: float
    e3_aa: float
    e3_bb: float
    e3_ab: float
    def __init__(self) -> None: ...

class UMP2:
    def __init__(self, uhf_result: SCFResult, basis: BasisSet, integrals: IntegralEngine) -> None: ...
    def compute(self) -> UMP2Result: ...

class UMP3:
    def __init__(self, uhf_result: SCFResult, ump2_result: UMP2Result, basis: BasisSet, integrals: IntegralEngine) -> None: ...
    def compute(self) -> UMP3Result: ...

# --- Cholesky MP2 / MP3 ---

class CholeskyUMP2Config:
    cholesky_threshold: float
    use_on_the_fly: bool
    validate_energy: bool
    print_level: int
    def __init__(self) -> None: ...

class CholeskyUMP2Result:
    e_corr_ss_aa: float
    e_corr_ss_bb: float
    e_corr_os: float
    e_corr_total: float
    e_total: float
    n_cholesky_vectors: int
    compression_ratio: float
    memory_mb: float
    time_cholesky_s: float
    time_transform_s: float
    time_energy_s: float

class CholeskyUMP2:
    def __init__(self, scf_res: SCFResult, basis: BasisSet, integrals: IntegralEngine, 
                 config: CholeskyUMP2Config, chol_eri: CholeskyERI) -> None: ...
    def compute(self) -> CholeskyUMP2Result: ...
    def compute_t2_amplitudes(self) -> None: ...
    def get_cholesky(self) -> CholeskyERI: ...

class CholeskyUMP3Config:
    cholesky_threshold: float
    use_ump2_result: bool
    store_intermediates: bool
    print_level: int
    def __init__(self) -> None: ...

class CholeskyUMP3Result:
    e_mp2_total: float
    e_mp3_ss_aa: float
    e_mp3_ss_bb: float
    e_mp3_os: float
    e_mp3_total: float
    e_corr_total: float
    e_total: float
    n_cholesky_vectors: int
    time_mp3_s: float

class CholeskyUMP3:
    @overload
    def __init__(self, scf_res: SCFResult, basis: BasisSet, integrals: IntegralEngine, config: CholeskyUMP3Config) -> None: ...
    @overload
    def __init__(self, ump2_solver: CholeskyUMP2, config: CholeskyUMP3Config) -> None: ...
    def compute(self) -> CholeskyUMP3Result: ...
    def initialize_cholesky(self) -> None: ...
    def transform_cholesky_vectors(self) -> None: ...
    
# --- Missing RMP Classes from bindings.cc  ---

class RMP2Result:
    e_corr: float
    e_rhf: float
    e_total: float
    def __init__(self) -> None: ...

class RMP3Result:
    e_mp2: float
    e_mp3: float
    e_total: float
    def __init__(self) -> None: ...

class RMP2:
    def __init__(self, rhf_result: SCFResult, basis: BasisSet, integrals: IntegralEngine) -> None: ...
    def compute(self) -> RMP2Result: ...

class RMP3:
    def __init__(self, rhf_result: SCFResult, rmp2_result: RMP2Result, basis: BasisSet, integrals: IntegralEngine) -> None: ...
    def compute(self) -> RMP3Result: ...

    # --- Gradient & Optimization ---

class GradientResult:
    energy: float
    gradient: np.ndarray
    def __init__(self) -> None: ...

class OptConfig:
    max_iterations: int
    def __init__(self) -> None: ...

class OptResult:
    converged: bool
    n_iterations: int
    final_energy: float
    def __init__(self) -> None: ...

# --- Utility Functions ---

def bohr_to_angstrom(bohr: float) -> float: ...
def angstrom_to_bohr(angstrom: float) -> float: ...
def hartree_to_ev(hartree: float) -> float: ...
def hartree_to_kcal(hartree: float) -> float: ...