# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""Type stubs for compiled C++20 Thin-Core module mshqc._mshqc (v2.0.0)."""

from typing import List, Tuple, overload
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
    def l(self) -> int: ...
    def n_functions(self) -> int: ...


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
    def reconstruct(self, i: int, j: int, k: int, l: int) -> float: ...
    def reconstruct_full(self) -> ERITensor: ...
    def is_decomposed(self) -> bool: ...
    def decomposed(self) -> bool: ...


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
    def build_exchange_3c_diag(L_mat: np.ndarray, d_diag: np.ndarray, nbf: int) -> np.ndarray: ...
    @staticmethod
    def build_exchange_3c_sym(L_mat: np.ndarray, D_sym: np.ndarray, nbf: int) -> np.ndarray: ...
    @staticmethod
    def build_jk_3c_rhf(L_mat: np.ndarray, C_occ: np.ndarray, nbf: int) -> Tuple[np.ndarray, np.ndarray]: ...
    @staticmethod
    def build_jk_3c_uhf(
        L_mat: np.ndarray, C_occ_a: np.ndarray, C_occ_b: np.ndarray, nbf: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    @staticmethod
    def transform_3c_mo(L_mat: np.ndarray, C1: np.ndarray, C2: np.ndarray, nbf: int) -> np.ndarray: ...
    @staticmethod
    def transform_3c_mo_blocks(
        L_mat: np.ndarray, C_occ: np.ndarray, C_vir: np.ndarray, nbf: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    @staticmethod
    def transform_3c_mo_blocks_uhf(
        L_mat: np.ndarray,
        Ca_occ: np.ndarray,
        Ca_vir: np.ndarray,
        Cb_occ: np.ndarray,
        Cb_vir: np.ndarray,
        nbf: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
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
    @staticmethod
    def contract_ao_sym(L_mat: np.ndarray, M_sym: np.ndarray, nbf: int) -> np.ndarray: ...
    @staticmethod
    def compute_gamma2c(Gamma_full: np.ndarray, L_mat: np.ndarray, nbf: int) -> np.ndarray: ...
    @staticmethod
    def eval_df_nuclear_gradient(
        mol: Molecule,
        primary_basis: BasisSet,
        aux_basis: BasisSet,
        P_eff_ao: np.ndarray,
        W_eff_ao: np.ndarray,
        Gamma_3c: np.ndarray,
        gamma_2c: np.ndarray,
        cutoff: float = 1e-12,
    ) -> np.ndarray: ...
    @staticmethod
    def eval_lt_sos_intermediates_rhf_df(
        B_ov: np.ndarray,
        d_ov: np.ndarray,
        weights: np.ndarray,
        nodes: np.ndarray,
        c_os: float = 1.3,
    ) -> Tuple[float, np.ndarray, np.ndarray]: ...
    @staticmethod
    def eval_lt_sos_intermediates_uhf_df(
        B_ov_a: np.ndarray,
        B_ov_b: np.ndarray,
        da: np.ndarray,
        db: np.ndarray,
        weights: np.ndarray,
        nodes: np.ndarray,
        c_os: float = 1.3,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    @staticmethod
    def build_jk_4d_rhf(eri_ao: ERITensor, C_occ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...
    @staticmethod
    def build_jk_4d_uhf(
        eri_ao: ERITensor, C_occ_a: np.ndarray, C_occ_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    @staticmethod
    def build_jk_4d_density_uhf(
        eri_ao: ERITensor, P_alpha: np.ndarray, P_beta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    @staticmethod
    def transform_eri_4d(
        eri_ao: ERITensor,
        C1: np.ndarray,
        C2: np.ndarray,
        C3: np.ndarray,
        C4: np.ndarray,
    ) -> np.ndarray: ...
    @staticmethod
    def eval_rmp2_exact(
        eri_ao: ERITensor,
        C_occ: np.ndarray,
        C_vir: np.ndarray,
        eps_occ: np.ndarray,
        eps_vir: np.ndarray,
    ) -> Tuple[float, float]: ...
    @staticmethod
    def eval_ump2_exact(
        eri_ao: ERITensor,
        Ca_occ: np.ndarray,
        Ca_vir: np.ndarray,
        Cb_occ: np.ndarray,
        Cb_vir: np.ndarray,
        eps_oa: np.ndarray,
        eps_va: np.ndarray,
        eps_ob: np.ndarray,
        eps_vb: np.ndarray,
    ) -> Tuple[float, float, float]: ...
    @staticmethod
    def factorize_eri_4d_exact(eri_ao: ERITensor, threshold: float = 1e-12) -> np.ndarray: ...
    @staticmethod
    def eval_mo_gcorr_rhf_df(
        B_oo: np.ndarray,
        B_ov: np.ndarray,
        B_vv: np.ndarray,
        g_oo: np.ndarray,
        g_vv: np.ndarray,
        n_occ: int,
        n_vir: int,
    ) -> np.ndarray: ...
    @staticmethod
    def eval_mo_gcorr_uhf_df(
        B_oo_a: np.ndarray,
        B_ov_a: np.ndarray,
        B_vv_a: np.ndarray,
        B_oo_b: np.ndarray,
        B_ov_b: np.ndarray,
        B_vv_b: np.ndarray,
        g_oo_a: np.ndarray,
        g_vv_a: np.ndarray,
        g_oo_b: np.ndarray,
        g_vv_b: np.ndarray,
        na: int,
        va: int,
        nb: int,
        vb: int,
    ) -> Tuple[np.ndarray, np.ndarray]: ...
