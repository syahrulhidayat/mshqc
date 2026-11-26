# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""High-level Pythonic interface untuk mshqc v2.0 Pure Thin-Core Engine."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import mshqc
from .geom_opt import ANGSTROM_TO_BOHR, GeomOptResult, UniversalGeomOptimizer
from .molecules import SYMBOL_TO_Z as _SYMBOL_TO_Z, molecule as _build_mol_spec


def build_molecule(
    atoms: Union[mshqc.Molecule, str, int, List[Tuple]],
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
    units: str = "bohr",
) -> mshqc.Molecule:
    """Bangun objek mshqc.Molecule dari nama atom/preset, file .xyz, string XYZ, atau list tuple."""
    return _build_mol_spec(atoms, charge=charge, multiplicity=multiplicity, units=units)


class MSHQCCalculator:
    """
    Orkestrator tingkat tinggi mshqc v2.0 (Pure Thin-Core):
    RHF/UHF/ROHF, MP2/MP3/ROMP2/ROMP3, OMP2/SCS-OMP2/SOS-OMP2/LT-SOS-OMP2/OMP2.5/OMP3/kappa-OMP2.
    """

    def __init__(
        self,
        molecule: Union[mshqc.Molecule, str, List[Tuple]],
        basis_name: str = "cc-pVTZ",
        eri_method: Optional[str] = None,
        aux_basis_name: Optional[str] = None,
        charge: Optional[int] = None,
        multiplicity: Optional[int] = None,
        units: str = "bohr",
        reference: Optional[str] = None,
        threshold: float = 1e-9,
        scf_type: str = "incore",
        print_level: int = 0,
        basis_dir: Optional[str] = None,
        freeze_core: bool = False,
        n_frozen: int = 0,
        memory_limit_gb: float = 4.0,
        use_symmetry: bool = False,
        level_shift: float = 0.0,
    ):
        self.molecule = build_molecule(molecule, charge=charge, multiplicity=multiplicity, units=units)
        self.eri_method = "df" if eri_method is None else eri_method.lower()
        if self.eri_method not in ("df", "cholesky", "exact"):
            raise ValueError(
                f"eri_method harus 'exact', 'cholesky', atau 'df' (diterima: {self.eri_method!r})."
            )

        self.use_symmetry = bool(use_symmetry)
        self.pg = None
        self.pl = None
        self.basis_name = basis_name
        self.aux_basis_name = aux_basis_name or f"{basis_name}-RI"
        self.threshold = float(threshold)
        self.scf_type = scf_type
        self.print_level = int(print_level)
        self.freeze_core = bool(freeze_core)
        self.n_frozen = (
            int(n_frozen)
            if n_frozen > 0
            else (int(mshqc.standard_frozen_core(self.molecule)) if freeze_core else 0)
        )
        self.memory_limit_gb = float(memory_limit_gb)
        self.level_shift = float(level_shift)

        self.basis_dir = basis_dir or os.environ.get("MSHQC_BASIS_PATH", "data/basis")
        self.basis = mshqc.BasisSet(basis_name, self.molecule, self.basis_dir)
        self.integrals = mshqc.IntegralEngine(self.molecule, self.basis)

        n_elec = int(self.molecule.total_nuclear_charge()) - int(self.molecule.charge())
        mult = int(self.molecule.multiplicity())
        self.n_alpha = (n_elec + mult - 1) // 2
        self.n_beta = n_elec - self.n_alpha
        self.reference = ("rhf" if mult == 1 else "uhf") if reference is None else reference.lower()

        self.scf_res = None
        self.mp2_res = None
        self._ao_factor_cache: Optional[np.ndarray] = None
        self._eri4_cache = None
      
        self._h_core_cache: Optional[np.ndarray] = None
        self._s_ovlp_cache: Optional[np.ndarray] = None
        self._salc_blocks = None
        if self.use_symmetry:
            from .methods.symmetry import detect_abelian_symmetry
            self.pg = detect_abelian_symmetry(self.molecule)
            if self.pg.point_group != "C1" and hasattr(self.integrals, "set_abelian_symmetry"):
                self.integrals.set_abelian_symmetry(True)

    def _get_h_core(self) -> np.ndarray:
        if self._h_core_cache is None:
            self._h_core_cache = np.ascontiguousarray(
                self.integrals.compute_kinetic() + self.integrals.compute_nuclear(),
                dtype=np.float64,
            )
        return self._h_core_cache

    def _get_s_ovlp(self) -> np.ndarray:
        if self._s_ovlp_cache is None:
            self._s_ovlp_cache = np.ascontiguousarray(
                self.integrals.compute_overlap(),
                dtype=np.float64,
            )
        return self._s_ovlp_cache

    def _get_salc_blocks(self):
        if not self.use_symmetry or self.pg is None or self.pg.point_group == "C1":
            return None
        if self._salc_blocks is None:
            from .methods.symmetry import build_salc_orthogonalizer
            self._salc_blocks = build_salc_orthogonalizer(
                self.molecule, self.basis, self._get_s_ovlp(), self.pg
            )
        return self._salc_blocks

    def _get_eri4(self):
        if self._eri4_cache is None:
            self._eri4_cache = self.integrals.compute_eri()
        return self._eri4_cache

    def _get_ao_3index_factor(self, scf_res: Optional[Any] = None) -> np.ndarray:
        if scf_res is not None and hasattr(scf_res, "L_mat") and scf_res.L_mat.shape[0] > 0 and scf_res.L_mat.shape[1] > 0:
            return scf_res.L_mat
        if self._ao_factor_cache is not None:
            return self._ao_factor_cache
        if self.eri_method == "df":
            aux_basis = mshqc.BasisSet(self.aux_basis_name, self.molecule, self.basis_dir)
            self._ao_factor_cache = mshqc.kernels.compute_df_ao_factor(
                self.molecule, self.basis, aux_basis, self.threshold, False, self.print_level > 0
            )
            return self._ao_factor_cache
        if self.eri_method == "exact":
            self._ao_factor_cache = mshqc.kernels.factorize_eri_4d_exact(
                self._get_eri4(), min(self.threshold * 0.1, 1.0e-10)
            )
            return self._ao_factor_cache
        chol = mshqc.CholeskyERI(self.basis, self.integrals)
        chol.set_threshold(self.threshold)
        chol.set_print_level(self.print_level)
        chol.compute()
        self._ao_factor_cache = chol.get_L_mat().copy()
        return self._ao_factor_cache

    def run_scf(self, ref: Optional[str] = None, use_hybrid: Optional[bool] = None):
        del use_hybrid
        from .methods.scf_hybrid import build_sad_guess, run_df_rhf, run_df_rohf, run_df_uhf

        ref_mode = (ref or self.reference).lower()
        P_sad = build_sad_guess(self.molecule, self.basis)
        eri4_arg = self._get_eri4() if self.eri_method == "exact" else None
        B_ao = None if eri4_arg is not None else self._get_ao_3index_factor(None)
        H_core = self._get_h_core()
        S_ovlp = self._get_s_ovlp()
        salc_blocks = self._get_salc_blocks()
        v_nn = float(self.molecule.nuclear_repulsion_energy())

        if ref_mode == "rhf":
            self.scf_res = run_df_rhf(
                H_core, S_ovlp, v_nn, B_ao, self.n_alpha,
                energy_tol=self.threshold, density_tol=self.threshold,
                level_shift=self.level_shift, P_init=P_sad, eri4=eri4_arg,
                salc_blocks=salc_blocks,
            )
        elif ref_mode == "rohf":
            self.scf_res = run_df_rohf(
                H_core, S_ovlp, v_nn, B_ao, self.n_alpha, self.n_beta,
                energy_tol=self.threshold, density_tol=self.threshold,
                P_init=P_sad, eri4=eri4_arg, salc_blocks=salc_blocks,
            )
        elif ref_mode == "uhf":
            self.scf_res = run_df_uhf(
                H_core, S_ovlp, v_nn, B_ao, self.n_alpha, self.n_beta,
                energy_tol=self.threshold, density_tol=self.threshold,
                level_shift=self.level_shift, P_init=P_sad, eri4=eri4_arg,
                salc_blocks=salc_blocks,
            )
        else:
            raise ValueError(f"Referensi SCF tidak dikenali: {ref_mode}")
        return self.scf_res

    def _dispatch_correlated(
        self,
        method_name: str,
        scf_res: Optional[Any] = None,
        grad_thresh: float = 1e-5,
        max_iter: int = 50,
        kappa: float = 1.45,
    ):
        from .methods.interop import run_from_arrays

        if scf_res is None:
            scf_res = self.scf_res or self.run_scf()
        B_ao = self._get_ao_3index_factor(scf_res)
        eri4_arg = self._get_eri4() if self.eri_method == "exact" else None
        H_core = self._get_h_core()
        S_ovlp = self._get_s_ovlp()
        v_nn = float(self.molecule.nuclear_repulsion_energy())

        salc_blocks = self._get_salc_blocks()
        if self.reference == "rhf" and method_name not in ("romp2", "romp3"):
            return run_from_arrays(
                H_core, S_ovlp, v_nn, B_ao, scf_res.C_alpha, scf_res.n_occ_alpha,
                method=method_name, grad_tol=grad_thresh, energy_tol=self.threshold,
                max_iter=max_iter, n_frozen=self.n_frozen, kappa=kappa, eri4=eri4_arg,
                salc_blocks=salc_blocks,
            )
        return run_from_arrays(
            H_core, S_ovlp, v_nn, B_ao, scf_res.C_alpha, scf_res.n_occ_alpha,
            C_beta=scf_res.C_beta, n_occ_beta=scf_res.n_occ_beta,
            method=method_name, grad_tol=grad_thresh, energy_tol=self.threshold,
            max_iter=max_iter, n_frozen=self.n_frozen, kappa=kappa, eri4=eri4_arg,
            salc_blocks=salc_blocks,
        )

    def run_mp2(self, scf_res: Optional[Any] = None, use_hybrid: Optional[bool] = None):
        del use_hybrid
        if self.eri_method == "exact" and self.reference != "rohf":
            from .methods.mp_hybrid import run_exact_mp2_rhf, run_exact_mp2_uhf
            if scf_res is None:
                scf_res = self.scf_res or self.run_scf()
            H_core = self._get_h_core()
            v_nn = float(self.molecule.nuclear_repulsion_energy())
            eri4 = self._get_eri4()
            if self.reference == "rhf":
                self.mp2_res = run_exact_mp2_rhf(
                    H_core, v_nn, eri4, scf_res.C_alpha, scf_res.n_occ_alpha, n_frozen=self.n_frozen
                )
            else:
                self.mp2_res = run_exact_mp2_uhf(
                    H_core, v_nn, eri4, scf_res.C_alpha, scf_res.C_beta,
                    scf_res.n_occ_alpha, scf_res.n_occ_beta, n_frozen=self.n_frozen
                )
            return self.mp2_res
        key = "romp2" if self.reference == "rohf" else "mp2"
        self.mp2_res = self._dispatch_correlated(key, scf_res=scf_res)
        return self.mp2_res

    def run_mp3(self, scf_res: Optional[Any] = None, mp2_res: Optional[Any] = None, use_hybrid: Optional[bool] = None):
        del mp2_res, use_hybrid
        key = "romp3" if self.reference == "rohf" else "mp3"
        return self._dispatch_correlated(key, scf_res=scf_res)

    def run_omp2(self, scf_res: Optional[Any] = None, opt_method: str = "soscf", grad_thresh: float = 1e-5, max_iter: int = 50, use_hybrid: Optional[bool] = None):
        del opt_method, use_hybrid
        return self._dispatch_correlated("omp2", scf_res=scf_res, grad_thresh=grad_thresh, max_iter=max_iter)

    def run_omp3(self, scf_res: Optional[Any] = None, opt_method: str = "soscf", grad_thresh: float = 1e-5, max_iter: int = 50, use_hybrid: Optional[bool] = None):
        del opt_method, use_hybrid
        return self._dispatch_correlated("omp3", scf_res=scf_res, grad_thresh=grad_thresh, max_iter=max_iter)

    def run_scs_omp2(self, scf_res: Optional[Any] = None, grad_thresh: float = 1e-5, max_iter: int = 50):
        return self._dispatch_correlated("scs-omp2", scf_res=scf_res, grad_thresh=grad_thresh, max_iter=max_iter)

    def run_sos_omp2(self, scf_res: Optional[Any] = None, grad_thresh: float = 1e-5, max_iter: int = 50):
        return self._dispatch_correlated("sos-omp2", scf_res=scf_res, grad_thresh=grad_thresh, max_iter=max_iter)

    def run_lt_sos_omp2(self, scf_res: Optional[Any] = None, grad_thresh: float = 1e-5, max_iter: int = 50):
        return self._dispatch_correlated("lt-sos-omp2", scf_res=scf_res, grad_thresh=grad_thresh, max_iter=max_iter)

    def run_omp25(self, scf_res: Optional[Any] = None, grad_thresh: float = 1e-5, max_iter: int = 50):
        return self._dispatch_correlated("omp2.5", scf_res=scf_res, grad_thresh=grad_thresh, max_iter=max_iter)

    def run_kappa_omp2(self, scf_res: Optional[Any] = None, kappa: float = 1.45, grad_thresh: float = 1e-5, max_iter: int = 50):
        return self._dispatch_correlated("kappa-omp2", scf_res=scf_res, grad_thresh=grad_thresh, max_iter=max_iter, kappa=kappa)


def run(
    molecule: Union[mshqc.Molecule, str, List[Tuple]],
    method: Union[str, List[str]] = "omp2",
    basis: str = "cc-pVTZ",
    eri_method: str = "df",
    aux_basis: Optional[str] = None,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
    units: str = "bohr",
    reference: Optional[str] = None,
    threshold: float = 1e-9,
    grad_thresh: float = 1e-5,
    opt_method: str = "soscf",
    print_level: int = 0,
    freeze_core: bool = False,
    n_frozen: int = 0,
    memory_limit_gb: float = 4.0,
    use_symmetry: bool = False,
    scf_type: str = "incore",
    level_shift: float = 0.0,
    kappa: float = 1.45,
) -> Union[Any, Dict[str, Any]]:
    """One-liner untuk menjalankan satu atau beberapa metode mshqc v2.0 sekaligus."""
    calc = MSHQCCalculator(
        molecule=molecule,
        basis_name=basis,
        eri_method=eri_method,
        aux_basis_name=aux_basis,
        charge=charge,
        multiplicity=multiplicity,
        units=units,
        reference=reference,
        threshold=threshold,
        print_level=print_level,
        freeze_core=freeze_core,
        n_frozen=n_frozen,
        memory_limit_gb=memory_limit_gb,
        use_symmetry=use_symmetry,
        scf_type=scf_type,
        level_shift=level_shift,
    )

    methods = [method] if isinstance(method, str) else list(method)
    results: Dict[str, Any] = {}

    for m in methods:
        m_low = m.lower().strip()
        t0 = time.perf_counter()
        if m_low in ("scf", "hf", "rhf", "uhf", "rohf"):
            res = calc.run_scf(ref=m_low if m_low in ("rhf", "uhf", "rohf") else None)
        elif m_low in ("mp2", "rmp2", "ump2", "romp2"):
            res = calc.run_mp2()
        elif m_low in ("mp3", "rmp3", "ump3", "romp3"):
            res = calc.run_mp3()
        elif m_low == "omp2":
            res = calc.run_omp2(opt_method=opt_method, grad_thresh=grad_thresh)
        elif m_low == "omp3":
            res = calc.run_omp3(opt_method=opt_method, grad_thresh=grad_thresh)
        elif m_low in ("scs-omp2", "scs_omp2"):
            res = calc.run_scs_omp2(grad_thresh=grad_thresh)
        elif m_low in ("sos-omp2", "sos_omp2"):
            res = calc.run_sos_omp2(grad_thresh=grad_thresh)
        elif m_low in ("lt-sos-omp2", "lt_sos_omp2"):
            res = calc.run_lt_sos_omp2(grad_thresh=grad_thresh)
        elif m_low in ("omp2.5", "omp25", "omp2_5"):
            res = calc.run_omp25(grad_thresh=grad_thresh)
        elif m_low in ("kappa-omp2", "kappa_omp2", "k-omp2", "kappa-oomp2"):
            res = calc.run_kappa_omp2(kappa=kappa, grad_thresh=grad_thresh)
        else:
            raise ValueError(f"Metode tidak dikenali: {m}")
        elapsed = time.perf_counter() - t0
        results[m] = {
            "result": res,
            "energy": getattr(res, "energy_total", getattr(res, "e_total", None)),
            "time": elapsed,
        }

    if isinstance(method, str):
        return results[method]["result"]
    return results


def optimize_geometry_universal(
    atoms: list,
    coords: np.ndarray,
    basis: str = "cc-pVDZ",
    method: str = "omp2",
    charge: int = 0,
    multiplicity: int = 1,
    units: str = "bohr",
    integral_mode: str = "df",
    single_point_runner=None,
    max_iter: int = 50,
    grad_rms_tol: float = 1.0e-5,
    grad_max_tol: float = 3.0e-5,
    fd_step_bohr: float = 1.0e-4,
    **kwargs,
) -> GeomOptResult:
    """Optimisasi geometri berbasis UniversalGeomOptimizer dengan runner kustom atau bawaan."""
    coords_arr = np.asarray(coords, dtype=np.float64)
    coords_bohr = coords_arr * ANGSTROM_TO_BOHR if units.strip().lower().startswith("ang") else coords_arr.copy()

    def _default_energy_eval(c_bohr: np.ndarray) -> float:
        if single_point_runner is not None:
            return single_point_runner(c_bohr)
        mol_tuples = [(atoms[i], float(c_bohr[i, 0]), float(c_bohr[i, 1]), float(c_bohr[i, 2])) for i in range(len(atoms))]
        res = run(
            mol_tuples, method=method, basis=basis, eri_method=integral_mode,
            charge=charge, multiplicity=multiplicity, units="bohr", **kwargs,
        )
        return float(getattr(res, "energy_total", getattr(res, "e_total")))

    opt = UniversalGeomOptimizer(
        energy_fn=_default_energy_eval,
        method=method,
        integral_mode=integral_mode,
        fd_step_bohr=fd_step_bohr,
        max_iter=max_iter,
        grad_rms_tol=grad_rms_tol,
        grad_max_tol=grad_max_tol,
    )
    return opt.run(coords_bohr)


def recommend_hpc_settings(
    molecule: Union[mshqc.Molecule, str, List[Tuple]],
    basis_name: str = "cc-pVTZ",
    method: str = "omp2",
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
    units: str = "angstrom",
    memory_limit_gb: float = 8.0,
    basis_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Auto-profiler HPC untuk estimasi N_bf, kebutuhan RAM, dan rekomendasi metode mshqc v2.0."""
    mol = build_molecule(molecule, charge=charge, multiplicity=multiplicity, units=units)
    bdir = basis_dir or os.environ.get("MSHQC_BASIS_PATH", "data/basis")
    basis = mshqc.BasisSet(basis_name, mol, bdir)
    nbf = int(basis.n_basis_functions())
    n_elec = int(mol.total_nuclear_charge()) - int(mol.charge())
    mult = int(mol.multiplicity())
    na = (n_elec + mult - 1) // 2
    va = max(0, nbf - na)
    n_aux_est = int(nbf * 3.0)

    max_z = max((int(mol.atom(i).atomic_number) for i in range(mol.n_atoms())), default=0)
    n_frozen_std = int(mshqc.standard_frozen_core(mol))

    mem_df_packed_gb = (nbf * (nbf + 1) / 2.0) * n_aux_est * 8.0 / 1.0e9
    mem_bvv_gb = (va ** 2) * n_aux_est * 8.0 / 1.0e9
    mem_t2_gb = (max(1, na - n_frozen_std) ** 2) * (va ** 2) * 8.0 / 1.0e9

    tier = "light" if nbf <= 60 else ("medium" if nbf <= 250 else "heavy")
    fc_rec = nbf > 60
    warnings: List[str] = []
    if method.lower().replace("_", "-") in ("lt-sos-omp2", "lt-sos-oomp2") and nbf < 60:
        warnings.append(
            "LT-SOS-OMP2 berskala O(n_q * O * V * N_aux^2); paling efisien pada mode DF "
            "dengan sistem menengah-besar (N_bf >= 60). Hindari mode exact/cholesky ber-rank besar."
        )
    if max_z > 36:
        warnings.append(
            f"Terdeteksi atom berat Z={max_z} (>36). mshqc all-electron non-relativistik tanpa ECP/X2C."
        )
    if method.lower() in ("omp2", "omp3", "kappa-omp2") and (mem_df_packed_gb + mem_bvv_gb + mem_t2_gb) > memory_limit_gb:
        warnings.append(
            f"Estimasi puncak RAM ({mem_df_packed_gb + mem_bvv_gb + mem_t2_gb:.2f} GB) melebihi "
            f"memory_limit_gb={memory_limit_gb:.2f} GB; pertimbangkan method='lt-sos-omp2' (zero-T2 O(N^4))."
        )

    return {
        "tier": tier,
        "nbf": nbf,
        "n_electrons": n_elec,
        "max_Z": max_z,
        "eri_method": "df",
        "freeze_core": fc_rec,
        "n_frozen_standard": n_frozen_std,
        "est_ram_gb": {
            "df_packed_3c": round(mem_df_packed_gb, 3),
            "omp_bvv_3c": round(mem_bvv_gb, 3),
            "t2_active_4d": round(mem_t2_gb, 3),
        },
        "warnings": warnings,
    }
