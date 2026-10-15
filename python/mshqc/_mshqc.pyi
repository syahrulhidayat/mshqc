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

import numpy as np
from typing import List, Optional, Tuple, overload

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
    def build(self) -> None: ...
    def get_unique_pairs(self) -> List[UniqueShellPair]: ...

class BasisSymmetrizer:
    def __init__(self, basis: BasisSet, pg: PointGroup, pl: PetiteList) -> None: ...
    def symmetrize(self, S: np.ndarray) -> np.ndarray: ...

class DIIS:
    def __init__(self, max_vectors: int = 8) -> None: ...
    def clear(self) -> None: ...
    def add_iteration(self, F: np.ndarray, err: np.ndarray, P: np.ndarray) -> None: ...
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

class ROHF:
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

class UMP2:
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

class OMP2:
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

class UMP3:
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

def bohr_to_angstrom(bohr: float) -> float: ...
def angstrom_to_bohr(angstrom: float) -> float: ...
def hartree_to_ev(hartree: float) -> float: ...
def hartree_to_kcal(hartree: float) -> float: ...
