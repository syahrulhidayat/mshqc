# Copyright 2026 Muhamad Syahrul Hidayat
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

"""Type stubs for compiled C++20 extension module `mshqc._mshqc` (hybrid-v2)."""

from typing import Dict, List, Optional, Tuple, overload
import numpy as np

class ERITensor:
    def size(self) -> int: ...
    def dimension(self, index: int) -> int: ...
    def shape(self) -> Tuple[int, int, int, int]: ...

class Atom:
    atomic_number: int
    x: float
    y: float
    z: float
    def __init__(self, atomic_number: int, x: float, y: float, z: float) -> None: ...
    def position(self) -> List[float]: ...

class GeomOptConfig:
    method: str
    eri_method: str
    basis_name: str
    aux_basis_name: str
    reference: str
    opt_method: str
    max_iter: int
    energy_thresh: float
    orb_grad_thresh: float
    g_rms_tol: float
    g_max_tol: float
    dx_rms_tol: float
    dx_max_tol: float
    trust_radius_init: float
    trust_radius_max: float
    trust_radius_min: float
    fd_step_bohr: float
    use_analytic_gradient: bool
    project_trans_rot: bool
    verbose: bool
    xyz_traj_file: str
    freeze_core: bool
    n_frozen: int
    memory_limit_gb: float
    def __init__(self) -> None: ...

class GeomOptResult:
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def method(self) -> str: ...
    @property
    def eri_method(self) -> str: ...
    @property
    def basis_name(self) -> str: ...
    @property
    def initial_energy(self) -> float: ...
    @property
    def final_energy(self) -> float: ...
    @property
    def final_g_rms(self) -> float: ...
    @property
    def final_g_max(self) -> float: ...
    @property
    def wall_time_sec(self) -> float: ...
    @property
    def coordinates_bohr(self) -> np.ndarray: ...
    @property
    def coordinates_angstrom(self) -> np.ndarray: ...
    @property
    def gradient_ha_bohr(self) -> np.ndarray: ...
    @property
    def energy_history(self) -> List[float]: ...
    @property
    def g_rms_history(self) -> List[float]: ...
    @property
    def bond_lengths_angstrom(self) -> Dict[str, float]: ...

class VibThermoResult:
    @property
    def frequencies_cm1(self) -> List[float]: ...
    @property
    def normal_modes(self) -> np.ndarray: ...
    @property
    def hessian_ha_bohr2(self) -> np.ndarray: ...
    @property
    def electronic_energy_ha(self) -> float: ...
    @property
    def zpe_hartree(self) -> float: ...
    @property
    def zpe_kcal_mol(self) -> float: ...
    @property
    def thermal_energy_ha(self) -> float: ...
    @property
    def enthalpy_ha(self) -> float: ...
    @property
    def gibbs_free_energy_ha(self) -> float: ...
    @property
    def temperature_k(self) -> float: ...
    @property
    def n_imaginary(self) -> int: ...

class PESScanPoint:
    @property
    def r_angstrom(self) -> float: ...
    @property
    def r_bohr(self) -> float: ...
    @property
    def energy_ha(self) -> float: ...

class HPCGeometryOptimizer:
    @staticmethod
    def optimize_molecule(mol: Molecule, config: GeomOptConfig = ...) -> GeomOptResult: ...
    @staticmethod
    def compute_nuclear_gradient_cpp(mol: Molecule, config: GeomOptConfig = ...) -> np.ndarray: ...
    @staticmethod
    def compute_frequencies(
        mol: Molecule,
        config: GeomOptConfig = ...,
        temperature_k: float = 298.15,
        pressure_atm: float = 1.0,
    ) -> VibThermoResult: ...
    @staticmethod
    def scan_bond_pes(
        mol: Molecule,
        atom_i: int,
        atom_j: int,
        distances_angstrom: List[float],
        config: GeomOptConfig = ...,
    ) -> List[PESScanPoint]: ...

class Molecule:
    def __init__(self, charge: int = 0, multiplicity: int = 1) -> None: ...
    @overload
    def add_atom(self, Z: int, x: float, y: float, z: float, units: str = "bohr") -> None: ...
    @overload
    def add_atom(self, symbol: str, x: float, y: float, z: float, units: str = "bohr") -> None: ...
    @overload
    def add_atom(self, atom: Atom) -> None: ...
    def n_atoms(self) -> int: ...
    def atom(self, i: int) -> Atom: ...
    def atoms(self) -> List[Atom]: ...
    def total_nuclear_charge(self) -> int: ...
    def n_electrons(self) -> int: ...
    def charge(self) -> int: ...
    def set_charge(self, q: int) -> None: ...
    def multiplicity(self) -> int: ...
    def set_multiplicity(self, m: int) -> None: ...
    def nuclear_repulsion_energy(self) -> float: ...
    def set_atom_position(self, i: int, x: float, y: float, z: float) -> None: ...
    def optimize_geometry(
        self,
        method: str = "omp2",
        eri_mode: str = "df",
        eri_method: str = "",
        basis: str = "cc-pVDZ",
        aux_basis: str = "",
        reference: str = "auto",
        max_iter: int = 40,
        g_convergence: str = "",
        g_rms_tol: float = 1.0e-6,
        g_max_tol: float = 2.0e-6,
        dx_rms_tol: float = 4.0e-6,
        dx_max_tol: float = 6.0e-6,
        energy_thresh: float = 1.0e-12,
        orb_grad_thresh: float = 1.0e-10,
        fd_step_bohr: float = 5.0e-4,
        use_analytic_gradient: bool = True,
        verbose: bool = False,
    ) -> GeomOptResult: ...
    def frequencies(
        self,
        method: str = "omp2",
        eri_mode: str = "df",
        eri_method: str = "",
        basis: str = "cc-pVDZ",
        reference: str = "auto",
        temperature_k: float = 298.15,
        verbose: bool = True,
    ) -> VibThermoResult: ...
    def gradient(
        self,
        method: str = "omp2",
        eri_mode: str = "df",
        eri_method: str = "",
        basis: str = "cc-pVDZ",
        aux_basis: str = "",
        reference: str = "auto",
        fd_step_bohr: float = 5.0e-4,
        use_analytic_gradient: bool = True,
    ) -> np.ndarray: ...
    def scan_pes(
        self,
        atom_i: int,
        atom_j: int,
        distances_angstrom: List[float],
        method: str = "omp2",
        eri_mode: str = "df",
        basis: str = "cc-pVDZ",
    ) -> List[PESScanPoint]: ...

class AngularMomentum:
    S: int
    P: int
    D: int
    F: int
    G: int
    H: int
    I: int
    K: int

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

class BasisSet:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, basis_name: str, mol: Molecule, basis_dir: str = "data/basis") -> None: ...
    def read_gbs(self, basis_file: str, mol: Molecule) -> None: ...
    def add_shell(self, shell: Shell) -> None: ...
    def n_shells(self) -> int: ...
    def n_basis_functions(self) -> int: ...
    def n_basis(self) -> int: ...
    def shell(self, i: int) -> Shell: ...
    def name(self) -> str: ...

class IntegralEngine:
    def __init__(self, mol: Molecule, basis: BasisSet) -> None: ...
    def compute_overlap(self) -> np.ndarray: ...
    def compute_kinetic(self) -> np.ndarray: ...
    def compute_nuclear(self) -> np.ndarray: ...
    def compute_eri(self) -> ERITensor: ...
    def compute_eri_diagonal(self) -> np.ndarray: ...
    def compute_eri_column(self, pivot_index: int) -> np.ndarray: ...

class CholeskyDecompositionResult:
    @property
    def n_vectors(self) -> int: ...
    @property
    def n_basis(self) -> int: ...
    @property
    def threshold(self) -> float: ...
    @property
    def compression_ratio(self) -> float: ...
    @property
    def converged(self) -> bool: ...

class CholeskyERI:
    @overload
    def __init__(self, threshold: float = 1e-9) -> None: ...
    @overload
    def __init__(self, basis: BasisSet, integrals: IntegralEngine) -> None: ...
    @overload
    def __init__(self, mol: Molecule, basis: BasisSet, threshold: float = 1e-9) -> None: ...
    def set_threshold(self, threshold: float) -> None: ...
    def set_print_level(self, level: int) -> None: ...
    def n_vectors(self) -> int: ...
    def n_cholesky_vectors(self) -> int: ...
    def threshold(self) -> float: ...
    def compute(self) -> CholeskyDecompositionResult: ...
    def decompose(self, eri: ERITensor) -> CholeskyDecompositionResult: ...
    def get_L_mat(self) -> np.ndarray: ...
    def reconstruct(self) -> np.ndarray: ...
    def reconstruct_full(self) -> ERITensor: ...
    def is_decomposed(self) -> bool: ...
    def decomposed(self) -> bool: ...

class SymOpType:
    Identity: int
    Rotation: int
    Reflection: int
    Inversion: int
    ImproperRotation: int

class SymmetryOperation:
    type: SymOpType
    order: int
    matrix: np.ndarray
    name: str

class PointGroup:
    def __init__(self, mol: Molecule, tolerance: float = 1e-6) -> None: ...
    def detect(self) -> None: ...
    def get_symbol(self) -> str: ...
    def get_order(self) -> int: ...
    def get_aligned_molecule(self) -> Molecule: ...
    def get_operations(self) -> List[SymmetryOperation]: ...

class UniqueShellPair:
    p: int
    q: int
    weight: float

class PetiteList:
    def __init__(self, basis: BasisSet, pg: PointGroup) -> None: ...
    @overload
    def build(self) -> None: ...
    @overload
    def build(self, schwarz: np.ndarray, cutoff: float = 1e-12) -> None: ...
    def get_unique_pairs(self) -> List[UniqueShellPair]: ...

class BasisSymmetrizer:
    def __init__(self, basis: BasisSet, pg: PointGroup, pl: PetiteList) -> None: ...
    def symmetrize(self, S: np.ndarray) -> np.ndarray: ...

class DIIS:
    def __init__(self, max_vectors: int = 8) -> None: ...
    def clear(self) -> None: ...
    def add_iteration(self, F: np.ndarray, err: np.ndarray) -> None: ...
    def extrapolate(self) -> np.ndarray: ...

class ShellPair:
    sh_a: int
    sh_b: int
    max_val: float
    center: Tuple[float, float, float]

class Screening:
    def __init__(self, basis: BasisSet) -> None: ...
    def reset(self) -> None: ...
    def compute(self, integrals: IntegralEngine) -> None: ...
    def print_stats(self, threshold: float) -> None: ...
    def get_significant_pairs(self, threshold: float) -> List[ShellPair]: ...
    def is_significant(self, sh_a: int, sh_b: int, sh_c: int, sh_d: int, threshold: float) -> bool: ...
    def get_schwarz_val(self, sh_a: int, sh_b: int) -> float: ...
    def max_schwarz(self) -> float: ...

class SCFConfig:
    scf_type: str
    eri_method: str
    cholesky_threshold: float
    df_threshold: float
    max_iterations: int
    energy_threshold: float
    density_threshold: float
    diis_threshold: float
    diis_max_vectors: int
    print_level: int
    use_df: bool
    aux_basis_name: str
    memory_limit_gb: float
    level_shift: float
    def __init__(
        self,
        eri_method: str = "exact",
        threshold: float = 1e-9,
        aux_basis_name: str = "",
        scf_type: str = "incore",
        print_level: int = 0,
        max_iterations: int = 100,
    ) -> None: ...

class SCFResult:
    energy_electronic: float
    energy_nuclear: float
    energy_total: float
    @property
    def e_total(self) -> float: ...
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
    n_occ_alpha: int
    n_occ_beta: int
    @property
    def L_mat(self) -> np.ndarray: ...
    def __init__(self) -> None: ...

class RHF:
    @overload
    def __init__(
        self,
        molecule: Molecule,
        basis: BasisSet,
        integrals: IntegralEngine,
        pg: Optional[PointGroup],
        pl: Optional[PetiteList],
        config: SCFConfig = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        molecule: Molecule,
        basis: BasisSet,
        integrals: IntegralEngine,
        config: SCFConfig = ...,
        pg: Optional[PointGroup] = None,
        pl: Optional[PetiteList] = None,
    ) -> None: ...
    def compute(self) -> SCFResult: ...
    def energy(self) -> float: ...

class UHF:
    @overload
    def __init__(
        self,
        molecule: Molecule,
        basis: BasisSet,
        integrals: IntegralEngine,
        pg: Optional[PointGroup],
        pl: Optional[PetiteList],
        n_alpha: int,
        n_beta: int,
        config: SCFConfig = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        molecule: Molecule,
        basis: BasisSet,
        integrals: IntegralEngine,
        n_alpha: int,
        n_beta: int,
        config: SCFConfig = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        molecule: Molecule,
        basis: BasisSet,
        integrals: IntegralEngine,
        config: SCFConfig = ...,
        pg: Optional[PointGroup] = None,
        pl: Optional[PetiteList] = None,
    ) -> None: ...
    def compute(self) -> SCFResult: ...
    def energy(self) -> float: ...

class ROHF(UHF): ...

class MP2Config:
    scf_type: str
    eri_method: str
    use_df: bool
    aux_basis_name: str
    df_threshold: float
    cholesky_threshold: float
    max_iterations: int
    energy_threshold: float
    gradient_threshold: float
    print_level: int
    opt_method: str
    n_frozen: int
    freeze_core: bool
    memory_limit_gb: float
    def __init__(
        self,
        eri_method: str = "exact",
        threshold: float = 1e-9,
        gradient_threshold: float = 1e-5,
        opt_method: str = "soscf",
        aux_basis_name: str = "",
        scf_type: str = "incore",
        print_level: int = 0,
        max_iterations: int = 50,
    ) -> None: ...

class MP2Result:
    energy_scf: float
    energy_mp2_ss: float
    energy_mp2_os: float
    energy_mp2_corr: float
    energy_total: float
    energy_mp2_singles: float
    @property
    def e_total(self) -> float: ...
    @property
    def e_corr_total(self) -> float: ...
    converged: bool
    iterations: int
    n_occ_alpha: int
    n_occ_beta: int
    n_virt_alpha: int
    n_virt_beta: int
    C_alpha: np.ndarray
    C_beta: np.ndarray
    orbital_energies_alpha: np.ndarray
    orbital_energies_beta: np.ndarray
    def __init__(self) -> None: ...

class RMP2:
    def __init__(
        self,
        mol: Molecule,
        basis: BasisSet,
        integrals: IntegralEngine,
        scf_guess: SCFResult,
        config: MP2Config = ...,
        pg: Optional[PointGroup] = None,
        pl: Optional[PetiteList] = None,
    ) -> None: ...
    def compute(self) -> MP2Result: ...
    def set_frozen_core(self, n_frozen: int) -> None: ...

class UMP2(RMP2): ...
class ROMP2(UMP2): ...
class OMP2(RMP2): ...

class MP3Result:
    e_hf: float
    e_mp2: float
    e_mp3: float
    e3_aa: float
    e3_bb: float
    e3_ab: float
    e_corr_total: float
    e_total: float
    @property
    def energy_total(self) -> float: ...
    @property
    def energy_scf(self) -> float: ...
    @property
    def energy_mp2_corr(self) -> float: ...
    @property
    def energy_mp3_corr(self) -> float: ...
    converged: bool
    iterations: int
    C_alpha: np.ndarray
    C_beta: np.ndarray
    n_occ_alpha: int
    n_occ_beta: int
    def __init__(self) -> None: ...

class RMP3:
    @overload
    def __init__(
        self,
        scf_guess: SCFResult,
        mp2_guess: MP2Result,
        config: MP2Config,
        integrals: IntegralEngine,
    ) -> None: ...
    @overload
    def __init__(
        self,
        mol: Molecule,
        basis: BasisSet,
        integrals: IntegralEngine,
        scf_guess: SCFResult,
        config: MP2Config = ...,
        pg: Optional[PointGroup] = None,
        pl: Optional[PetiteList] = None,
    ) -> None: ...
    def compute(self) -> MP3Result: ...
    def set_frozen_core(self, n_frozen: int) -> None: ...

class UMP3(RMP3): ...
class ROMP3(UMP3): ...

class OMP3:
    def __init__(
        self,
        mol: Molecule,
        basis: BasisSet,
        integrals: IntegralEngine,
        scf_guess: SCFResult,
        config: MP2Config = ...,
        pg: Optional[PointGroup] = None,
        pl: Optional[PetiteList] = None,
    ) -> None: ...
    def compute(self) -> MP3Result: ...
    def set_frozen_core(self, n_frozen: int) -> None: ...

def bohr_to_angstrom(bohr: float) -> float: ...
def angstrom_to_bohr(angstrom: float) -> float: ...
def hartree_to_ev(hartree: float) -> float: ...
def hartree_to_kcal(hartree: float) -> float: ...
def standard_frozen_core(mol: Molecule) -> int: ...

class kernels:
    class DensityFittingERI:
        def __init__(
            self,
            mol: Molecule,
            primary_basis: BasisSet,
            aux_basis: BasisSet,
            cutoff: float = 1e-10,
        ) -> None: ...
        def compute(self, save_hdf5: bool = False) -> None: ...
        def get_B_mat(self) -> np.ndarray: ...
        def set_verbose(self, verbose: bool) -> None: ...
        def set_hdf5_path(self, path: str) -> None: ...
        def get_hdf5_path(self) -> str: ...

    class FockBuilder:
        def __init__(
            self,
            integrals: IntegralEngine,
            basis: BasisSet,
            H_core: np.ndarray,
            schwarz: np.ndarray,
        ) -> None: ...
        def reset(self) -> None: ...
        def compute(
            self, P_alpha: np.ndarray, P_beta: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]: ...

    @staticmethod
    def compute_df_ao_factor(
        mol: Molecule,
        primary_basis: BasisSet,
        aux_basis: BasisSet,
        cutoff: float = 1e-10,
        save_hdf5: bool = False,
        verbose: bool = False,
    ) -> np.ndarray: ...
    @staticmethod
    def is_packed_ao(L_mat: np.ndarray, nbf: int) -> bool: ...
    @staticmethod
    def build_coulomb_3c(L_mat: np.ndarray, P_tot: np.ndarray, nbf: int) -> np.ndarray: ...
    @staticmethod
    def build_exchange_3c_occ(L_mat: np.ndarray, C_occ: np.ndarray, nbf: int) -> np.ndarray: ...
    @staticmethod
    def build_exchange_3c_sym(L_mat: np.ndarray, D_sym: np.ndarray, nbf: int) -> np.ndarray: ...
    @staticmethod
    def build_jk_3c_rhf(
        L_mat: np.ndarray, C_occ: np.ndarray, nbf: int
    ) -> Tuple[np.ndarray, np.ndarray]: ...
    @staticmethod
    def build_jk_3c_uhf(
        L_mat: np.ndarray, C_occ_a: np.ndarray, C_occ_b: np.ndarray, nbf: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    @staticmethod
    def transform_3c_mo(
        L_mat: np.ndarray, C1: np.ndarray, C2: np.ndarray, nbf: int
    ) -> np.ndarray: ...
    @staticmethod
    def semicanonicalize_subspaces(
        C: np.ndarray, F_ao: np.ndarray, n_occ: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    @staticmethod
    def eval_rmp2_df(
        B_ov: np.ndarray, eps_occ: np.ndarray, eps_vir: np.ndarray
    ) -> Tuple[float, float, np.ndarray]: ...
    @staticmethod
    def eval_omp2_1rdm_rhf(
        T2: np.ndarray, n_occ: int, n_vir: int, c_os: float = 1.0, c_ss: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]: ...
    @staticmethod
    def eval_omp2_zvec_rhf_df(
        T2: np.ndarray,
        B_oo: np.ndarray,
        B_ov: np.ndarray,
        B_vv: np.ndarray,
        n_occ: int,
        n_vir: int,
        c_os: float = 1.0,
        c_ss: float = 1.0,
    ) -> np.ndarray: ...
    @staticmethod
    def eval_pp_ladder_rhf_df(
        T2: np.ndarray, B_vv: np.ndarray, n_occ: int, n_vir: int
    ) -> np.ndarray: ...
    @staticmethod
    def eval_rmp3_df(
        T2: np.ndarray,
        B_oo: np.ndarray,
        B_ov: np.ndarray,
        B_vv: np.ndarray,
        eps_occ: np.ndarray,
        eps_vir: np.ndarray,
    ) -> Tuple[float, np.ndarray]: ...
    @staticmethod
    def eval_omp3_1rdm_rhf(
        T2: np.ndarray, L2: np.ndarray, n_occ: int, n_vir: int
    ) -> Tuple[np.ndarray, np.ndarray]: ...
    @staticmethod
    def eval_omp3_zvec_rhf_df(
        T2: np.ndarray,
        L2: np.ndarray,
        B_oo: np.ndarray,
        B_ov: np.ndarray,
        B_vv: np.ndarray,
        n_occ: int,
        n_vir: int,
    ) -> np.ndarray: ...
    @staticmethod
    def apply_mo_df_hessian_rhf(
        kappa_vo: np.ndarray,
        eps_occ: np.ndarray,
        eps_vir: np.ndarray,
        B_oo: np.ndarray,
        B_ov: np.ndarray,
        B_vv: np.ndarray,
    ) -> np.ndarray: ...
    @staticmethod
    def eval_ump2_df(
        B_ov_a: np.ndarray,
        B_ov_b: np.ndarray,
        eps_oa: np.ndarray,
        eps_va: np.ndarray,
        eps_ob: np.ndarray,
        eps_vb: np.ndarray,
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]: ...
    @staticmethod
    def eval_omp2_1rdm_uhf(
        T2_aa: np.ndarray,
        T2_bb: np.ndarray,
        T2_ab: np.ndarray,
        na: int,
        va: int,
        nb: int,
        vb: int,
        c_os: float = 1.0,
        c_ss: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    @staticmethod
    def eval_omp2_zvec_uhf_df(
        T2_aa: np.ndarray,
        T2_bb: np.ndarray,
        T2_ab: np.ndarray,
        B_oo_a: np.ndarray,
        B_ov_a: np.ndarray,
        B_vv_a: np.ndarray,
        B_oo_b: np.ndarray,
        B_ov_b: np.ndarray,
        B_vv_b: np.ndarray,
        na: int,
        va: int,
        nb: int,
        vb: int,
        c_os: float = 1.0,
        c_ss: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]: ...
    @staticmethod
    def eval_ump3_df(
        T2_aa: np.ndarray,
        T2_bb: np.ndarray,
        T2_ab: np.ndarray,
        B_oo_a: np.ndarray,
        B_ov_a: np.ndarray,
        B_vv_a: np.ndarray,
        B_oo_b: np.ndarray,
        B_ov_b: np.ndarray,
        B_vv_b: np.ndarray,
        eps_oa: np.ndarray,
        eps_va: np.ndarray,
        eps_ob: np.ndarray,
        eps_vb: np.ndarray,
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]: ...
    @staticmethod
    def eval_omp3_1rdm_uhf(
        T2_aa: np.ndarray,
        T2_bb: np.ndarray,
        T2_ab: np.ndarray,
        L2_aa: np.ndarray,
        L2_bb: np.ndarray,
        L2_ab: np.ndarray,
        na: int,
        va: int,
        nb: int,
        vb: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    @staticmethod
    def eval_omp3_zvec_uhf_df(
        T2_aa: np.ndarray,
        T2_bb: np.ndarray,
        T2_ab: np.ndarray,
        L2_aa: np.ndarray,
        L2_bb: np.ndarray,
        L2_ab: np.ndarray,
        B_oo_a: np.ndarray,
        B_ov_a: np.ndarray,
        B_vv_a: np.ndarray,
        B_oo_b: np.ndarray,
        B_ov_b: np.ndarray,
        B_vv_b: np.ndarray,
        na: int,
        va: int,
        nb: int,
        vb: int,
    ) -> Tuple[np.ndarray, np.ndarray]: ...
    @staticmethod
    def apply_mo_df_hessian_uhf(
        kappa_vo_a: np.ndarray,
        kappa_vo_b: np.ndarray,
        eps_oa: np.ndarray,
        eps_va: np.ndarray,
        eps_ob: np.ndarray,
        eps_vb: np.ndarray,
        B_oo_a: np.ndarray,
        B_ov_a: np.ndarray,
        B_vv_a: np.ndarray,
        B_oo_b: np.ndarray,
        B_ov_b: np.ndarray,
        B_vv_b: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]: ...
