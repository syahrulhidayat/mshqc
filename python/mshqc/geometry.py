"""
mshqc.geometry
==============
High-Performance Geometry Optimization, Harmonic Vibrational Frequencies,
Thermochemistry (ZPE, H, G), PES Scanning, and Open-Shell Radical Property
Analysis for mshqc (RHF/UHF/ROHF, MP2, MP3, OMP2, OMP3).
"""

from __future__ import annotations
import os
import time
import math
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
HARTREE_TO_KCAL = 627.5094740631
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_J = 4.3597447222071e-18
BOHR_TO_M = 5.29177210903e-11
AMU_TO_KG = 1.66053906660e-27
SPEED_OF_LIGHT_CMS = 2.99792458e10
PLANCK_SI = 6.62607015e-34
BOLTZMANN_SI = 1.380649e-23

ATOMIC_MASSES_AMU: Dict[int, float] = {
    1: 1.007825032, 2: 4.002603254, 3: 7.016003437, 4: 9.01218307,
    5: 11.00930536, 6: 12.00000000, 7: 14.003074004, 8: 15.994914619,
    9: 18.998403163, 10: 19.99244018, 11: 22.98976928, 12: 23.98504170,
    13: 26.98153853, 14: 27.976926535, 15: 30.973761998, 16: 31.972071174,
    17: 34.96885268, 18: 39.96238312, 19: 38.96370649, 20: 39.96259086,
    21: 44.955908, 22: 47.947942, 23: 50.943957, 24: 51.940506,
    25: 54.938043, 26: 55.934936, 27: 58.933194, 28: 57.935342,
    29: 62.929597, 30: 63.929142, 35: 78.918338, 53: 126.904472,
}

SYMBOL_TO_Z: Dict[str, int] = {
    "H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "NE": 10, "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15,
    "S": 16, "CL": 17, "AR": 18, "K": 19, "CA": 20, "SC": 21, "TI": 22,
    "V": 23, "CR": 24, "MN": 25, "FE": 26, "CO": 27, "NI": 28, "CU": 29,
    "ZN": 30, "BR": 35, "I": 53,
}
Z_TO_SYMBOL: Dict[int, str] = {v: k.capitalize() for k, v in SYMBOL_TO_Z.items()}


@dataclass
class GeometryOptimizationResult:
    converged: bool
    iterations: int
    method: str
    eri_mode: str
    basis: str
    initial_energy: float
    final_energy: float
    final_g_rms: float
    final_g_max: float
    coordinates_bohr: np.ndarray
    coordinates_angstrom: np.ndarray
    gradient_ha_bohr: np.ndarray
    energy_history: List[float] = field(default_factory=list)
    g_rms_history: List[float] = field(default_factory=list)
    bond_lengths_angstrom: Dict[str, float] = field(default_factory=dict)
    wall_time_sec: float = 0.0
    xyz_trajectory_path: Optional[str] = None

    def summary(self) -> str:
        status = "CONVERGED" if self.converged else "MAX_ITER REACHED"
        lines = [
            "=" * 72,
            f"  MSHQC GEOMETRY OPTIMIZATION SUMMARY ({self.method.upper()} / {self.eri_mode.upper()})",
            "=" * 72,
            f"  Status             : {status} in {self.iterations} steps ({self.wall_time_sec:.2f} s)",
            f"  Basis Set          : {self.basis}",
            f"  Initial Energy     : {self.initial_energy:18.10f} Ha",
            f"  Final Opt Energy   : {self.final_energy:18.10f} Ha",
            f"  Relaxation Lowering: {(self.final_energy - self.initial_energy)*HARTREE_TO_KCAL:14.4f} kcal/mol",
            f"  Final RMS Gradient : {self.final_g_rms:18.3e} Ha/Bohr",
            f"  Final MAX Gradient : {self.final_g_max:18.3e} Ha/Bohr",
        ]
        if self.bond_lengths_angstrom:
            lines.append("-" * 72)
            lines.append("  Key Interatomic Distances (Angstrom):")
            for pair, dist in self.bond_lengths_angstrom.items():
                lines.append(f"    {pair:<18}: {dist:10.5f} A")
        lines.append("=" * 72)
        return "\n".join(lines)


@dataclass
class FrequencyThermochemResult:
    frequencies_cm1: np.ndarray
    normal_modes: np.ndarray
    zpe_hartree: float
    zpe_kcal_mol: float
    electronic_energy_ha: float
    thermal_energy_ha: float
    enthalpy_ha: float
    gibbs_free_energy_ha: float
    temperature_k: float
    n_imaginary: int
    hessian_ha_bohr2: np.ndarray


def _extract_molecule_metadata(mol: Any) -> Tuple[np.ndarray, List[str], List[int], int, int, str]:
    """Mengekstrak koordinat (Bohr), simbol atom, nomor atom, charge, mult, dan basis dari objek Molecule mshqc."""
    coords_bohr = None
    symbols: List[str] = []
    atomic_nums: List[int] = []

    # 1. Coba atribut / metode standar pada Molecule mshqc
    if hasattr(mol, "atoms") and callable(getattr(mol, "atoms")):
        atoms_list = mol.atoms()
        coords = []
        for a in atoms_list:
            z = int(getattr(a, "atomic_number", getattr(a, "Z", getattr(a, "z", 6))))
            atomic_nums.append(z)
            symbols.append(Z_TO_SYMBOL.get(z, f"X{z}"))
            if hasattr(a, "position"):
                pos = a.position() if callable(a.position) else a.position
                coords.append([float(pos[0]), float(pos[1]), float(pos[2])])
            else:
                coords.append([float(getattr(a, "x", 0.0)), float(getattr(a, "y", 0.0)), float(getattr(a, "z", 0.0))])
        coords_bohr = np.asarray(coords, dtype=np.float64)
    elif hasattr(mol, "atoms") and isinstance(getattr(mol, "atoms"), (list, tuple)):
        atoms_list = mol.atoms
        coords = []
        for a in atoms_list:
            if isinstance(a, dict):
                z = int(a.get("Z", a.get("atomic_number", SYMBOL_TO_Z.get(str(a.get("symbol", "C")).upper(), 6))))
                atomic_nums.append(z)
                symbols.append(Z_TO_SYMBOL.get(z, "C"))
                coords.append([float(a["x"]), float(a["y"]), float(a["z"])])
            else:
                z = int(getattr(a, "atomic_number", getattr(a, "Z", getattr(a, "z", 6))))
                atomic_nums.append(z)
                symbols.append(Z_TO_SYMBOL.get(z, "C"))
                coords.append([float(getattr(a, "x", 0.0)), float(getattr(a, "y", 0.0)), float(getattr(a, "z", 0.0))])
        coords_bohr = np.asarray(coords, dtype=np.float64)

    if coords_bohr is None and hasattr(mol, "coordinates"):
        c = mol.coordinates() if callable(mol.coordinates) else mol.coordinates
        coords_bohr = np.asarray(c, dtype=np.float64).reshape(-1, 3)

    if coords_bohr is None and hasattr(mol, "geometry"):
        c = mol.geometry() if callable(mol.geometry) else mol.geometry
        coords_bohr = np.asarray(c, dtype=np.float64).reshape(-1, 3)

    if not atomic_nums and hasattr(mol, "atomic_numbers"):
        an = mol.atomic_numbers() if callable(mol.atomic_numbers) else mol.atomic_numbers
        atomic_nums = [int(z) for z in an]
        symbols = [Z_TO_SYMBOL.get(z, f"X{z}") for z in atomic_nums]

    if not atomic_nums and hasattr(mol, "n_atoms"):
        n_at = int(mol.n_atoms() if callable(mol.n_atoms) else mol.n_atoms)
        if coords_bohr is None:
            coords = []
            for i in range(n_at):
                atom_i = mol.atom(i) if hasattr(mol, "atom") else None
                if atom_i is not None:
                    z = int(getattr(atom_i, "atomic_number", getattr(atom_i, "Z", 6)))
                    atomic_nums.append(z)
                    symbols.append(Z_TO_SYMBOL.get(z, "C"))
                    coords.append([float(atom_i.x), float(atom_i.y), float(atom_i.z)])
            if coords:
                coords_bohr = np.asarray(coords, dtype=np.float64)

    if coords_bohr is None:
        raise RuntimeError("Tidak dapat mengekstrak koordinat 3D dari objek Molecule.")

    charge = int(mol.charge() if hasattr(mol, "charge") and callable(mol.charge) else getattr(mol, "charge", 0))
    mult = int(mol.multiplicity() if hasattr(mol, "multiplicity") and callable(mol.multiplicity) else getattr(mol, "multiplicity", 1))
    basis = str(getattr(mol, "basis_name", getattr(mol, "basis", "cc-pVDZ")))

    return coords_bohr, symbols, atomic_nums, charge, mult, basis


def _clone_molecule_with_bohr_coords(mol: Any, new_coords_bohr: np.ndarray) -> Any:
    """Membangun atau memperbarui objek Molecule pada koordinat Bohr baru."""
    coords_3d = np.asarray(new_coords_bohr, dtype=np.float64).reshape(-1, 3)
    _, symbols, atomic_nums, charge, mult, _ = _extract_molecule_metadata(mol)

    # Jika kelas Molecule memiliki setter set_geometry / set_coordinates langsung
    mol_cls = type(mol)
    try:
        # Coba konstruktor standar mshqc.Molecule
        new_mol = mol_cls(charge, mult) if _accepts_args(mol_cls.__init__, 3) else mol_cls()
        if hasattr(new_mol, "set_charge"):
            new_mol.set_charge(charge)
        if hasattr(new_mol, "set_multiplicity"):
            new_mol.set_multiplicity(mult)

        added = False
        if hasattr(new_mol, "add_atom"):
            for idx, (z, sym, xyz_b) in enumerate(zip(atomic_nums, symbols, coords_3d)):
                try:
                    new_mol.add_atom(z, float(xyz_b[0]), float(xyz_b[1]), float(xyz_b[2]))
                    added = True
                except TypeError:
                    new_mol.add_atom(sym, float(xyz_b[0]), float(xyz_b[1]), float(xyz_b[2]))
                    added = True
        if added:
            for attr in ("basis_name", "basis", "_basis", "_reference"):
                if hasattr(mol, attr):
                    try:
                        setattr(new_mol, attr, getattr(mol, attr))
                    except Exception:
                        pass
            return new_mol
    except Exception:
        pass

    # Fallback: gunakan from_xyz / string XYZ (dalam Angstrom atau Bohr sesuai konvensi)
    xyz_lines = [f"{len(symbols)}", f"{charge} {mult}"]
    for sym, xyz_b in zip(symbols, coords_3d):
        xyz_a = xyz_b * BOHR_TO_ANGSTROM
        xyz_lines.append(f"{sym} {xyz_a[0]:.12f} {xyz_a[1]:.12f} {xyz_a[2]:.12f}")
    xyz_str = "\n".join(xyz_lines)

    if hasattr(mol_cls, "from_xyz_string"):
        return mol_cls.from_xyz_string(xyz_str, charge=charge, multiplicity=mult)
    if hasattr(mol_cls, "from_string"):
        return mol_cls.from_string(xyz_str)

    # Jika objek mendukung mutasi in-place pada atom
    if hasattr(mol, "set_coordinates"):
        mol.set_coordinates(coords_3d)
        return mol
    if hasattr(mol, "set_geometry"):
        mol.set_geometry(coords_3d)
        return mol

    raise RuntimeError("Tidak dapat memperbarui koordinat pada objek Molecule.")


def _accepts_args(func: Callable, n_args: int) -> bool:
    try:
        sig = inspect.signature(func)
        return len(sig.parameters) >= n_args
    except (ValueError, TypeError):
        return False


def _evaluate_single_point_energy(
    mol: Any,
    method: str = "omp2",
    basis: Optional[str] = None,
    eri_mode: str = "df",
    reference: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[float, Optional[np.ndarray]]:
    """
    Mengevaluasi energi (dan gradien analitik jika tersedia di C++ backend)
    untuk Molecule pada metode dan eri_mode yang diminta.
    """
    import mshqc

    method_low = method.lower().strip()
    eri_low = eri_mode.lower().strip()
    _, _, _, _, mult, default_basis = _extract_molecule_metadata(mol)
    basis_use = basis or default_basis or "cc-pVDZ"
    ref_use = (reference or ("uhf" if mult > 1 else "rhf")).lower()

    # 1. Cek jika C++ binding menyediakan gradien analitik langsung
    if hasattr(mshqc, "compute_omp_gradient") and method_low in ("omp2", "omp3"):
        try:
            res = mshqc.compute_omp_gradient(mol, basis_use, method=method_low, eri_mode=eri_low)
            if isinstance(res, tuple) and len(res) == 2:
                return float(res[0]), np.asarray(res[1], dtype=np.float64).ravel()
            if hasattr(res, "energy") and hasattr(res, "gradient"):
                return float(res.energy), np.asarray(res.gradient, dtype=np.float64).ravel()
        except Exception:
            pass

    # 2. Gunakan high-level API di mshqc.calculators / mshqc.__init__
    fn_candidates = [
        f"run_{method_low}",
        method_low,
        "calculate",
        "single_point",
    ]
    for fn_name in fn_candidates:
        fn = getattr(mshqc, fn_name, None)
        if callable(fn):
            try:
                sig = inspect.signature(fn)
                call_kwargs = {}
                if "basis" in sig.parameters:
                    call_kwargs["basis"] = basis_use
                if "eri_mode" in sig.parameters:
                    call_kwargs["eri_mode"] = eri_low
                elif "integral_mode" in sig.parameters:
                    call_kwargs["integral_mode"] = eri_low
                if "reference" in sig.parameters:
                    call_kwargs["reference"] = ref_use
                if "method" in sig.parameters and fn_name in ("calculate", "single_point"):
                    call_kwargs["method"] = method_low
                for k, v in kwargs.items():
                    if k in sig.parameters:
                        call_kwargs[k] = v

                out = fn(mol, **call_kwargs)
                return _extract_energy_from_result(out), _extract_grad_from_result(out)
            except Exception:
                continue

    # 3. Gunakan kelas Calculator (OMP2Calculator, OMP3Calculator, MP2Calculator, SCFCalculator)
    calc_map = {
        "omp2": ["OMP2Calculator", "OMP2"],
        "omp3": ["OMP3Calculator", "OMP3"],
        "mp2":  ["MP2Calculator", "MP2"],
        "mp3":  ["MP3Calculator", "MP3"],
        "hf":   ["SCFCalculator", "HFCalculator"],
        "rhf":  ["SCFCalculator", "RHFCalculator"],
        "uhf":  ["SCFCalculator", "UHFCalculator"],
        "rohf": ["SCFCalculator", "ROHFCalculator"],
    }
    for cls_name in calc_map.get(method_low, []):
        cls_obj = getattr(mshqc, cls_name, None)
        if cls_obj is None and hasattr(mshqc, "calculators"):
            cls_obj = getattr(mshqc.calculators, cls_name, None)
        if cls_obj is not None:
            try:
                calc = cls_obj(mol, basis_use, eri_mode=eri_low, reference=ref_use)
            except TypeError:
                try:
                    calc = cls_obj(mol, basis_use, eri_low)
                except TypeError:
                    calc = cls_obj(mol, basis_use)
            out = calc.run() if hasattr(calc, "run") else calc.compute()
            return _extract_energy_from_result(out), _extract_grad_from_result(out)

    # 4. Fallback ke Session API jika ada
    if hasattr(mshqc, "Session"):
        sess = mshqc.Session(mol, basis=basis_use, eri_mode=eri_low, reference=ref_use)
        out = sess.run(method_low)
        return _extract_energy_from_result(out), _extract_grad_from_result(out)

    raise RuntimeError(
        f"Tidak dapat menemukan runner untuk method='{method}' (eri_mode='{eri_mode}') di mshqc."
    )


def _extract_energy_from_result(res: Any) -> float:
    if isinstance(res, (float, int, np.floating)):
        return float(res)
    if isinstance(res, dict):
        for k in ("total_energy", "e_tot", "energy", "e_omp2", "e_omp3", "e_mp2", "e_mp3", "e_scf"):
            if k in res:
                return float(res[k])
    for attr in ("total_energy", "e_tot", "energy", "final_energy", "e_omp2", "e_omp3"):
        if hasattr(res, attr):
            val = getattr(res, attr)
            return float(val() if callable(val) else val)
    raise ValueError(f"Tidak dapat mengekstrak skalar energi dari objek hasil: {type(res)}")


def _extract_grad_from_result(res: Any) -> Optional[np.ndarray]:
    if isinstance(res, dict) and "gradient" in res and res["gradient"] is not None:
        return np.asarray(res["gradient"], dtype=np.float64).ravel()
    if hasattr(res, "gradient"):
        g = getattr(res, "gradient")
        g_val = g() if callable(g) else g
        if g_val is not None:
            arr = np.asarray(g_val, dtype=np.float64).ravel()
            if arr.size > 0 and np.any(np.abs(arr) > 0.0):
                return arr
    return None


def _project_out_trans_rot_py(
    coords_bohr: np.ndarray,
    vec: np.ndarray,
    masses_amu: np.ndarray,
) -> np.ndarray:
    """Memproyeksikan keluar 6 mode translasi & rotasi secara eksak menggunakan SVD."""
    x = coords_bohr.reshape(-1, 3)
    natoms = x.shape[0]
    ndof = 3 * natoms
    if natoms <= 1:
        return np.zeros_like(vec)

    m = np.maximum(masses_amu, 1.0e-6)
    com = np.sum(x * m[:, None], axis=0) / np.sum(m)
    r = x - com[None, :]

    D = np.zeros((ndof, 6), dtype=np.float64)
    for i in range(natoms):
        D[3 * i + 0, 0] = 1.0
        D[3 * i + 1, 1] = 1.0
        D[3 * i + 2, 2] = 1.0

        D[3 * i + 1, 3] = -r[i, 2]
        D[3 * i + 2, 3] =  r[i, 1]
        D[3 * i + 0, 4] =  r[i, 2]
        D[3 * i + 2, 4] = -r[i, 0]
        D[3 * i + 0, 5] = -r[i, 1]
        D[3 * i + 1, 5] =  r[i, 0]

    U, s, _ = np.linalg.svd(D, full_matrices=False)
    v_proj = vec.ravel().copy()
    for k in range(len(s)):
        if s[k] > 1.0e-7:
            uk = U[:, k]
            v_proj -= np.dot(uk, v_proj) * uk
    return v_proj


def compute_nuclear_gradient(
    mol: Any,
    method: str = "omp2",
    basis: Optional[str] = None,
    eri_mode: str = "df",
    reference: Optional[str] = None,
    fd_step_bohr: float = 5.0e-4,
    stencil: int = 2,
    project_trans_rot: bool = True,
    **kwargs: Any,
) -> Tuple[float, np.ndarray]:
    """
    Menghitung energi total dan gradien gaya nuklir dE/dR (Ha/Bohr).
    Otomatis menggunakan gradien analitik Hellmann-Feynman jika tersedia,
    atau central finite-difference yang mengenali invarian translasi atom terakhir
    (menghemat 3*2 = 6 evaluasi titik tunggal per step!).
    """
    coords_bohr, _, atomic_nums, _, _, _ = _extract_molecule_metadata(mol)
    natoms = coords_bohr.shape[0]
    ndof = 3 * natoms
    masses = np.array([ATOMIC_MASSES_AMU.get(z, float(2 * z)) for z in atomic_nums], dtype=np.float64)

    e0, grad_analytic = _evaluate_single_point_energy(
        mol, method=method, basis=basis, eri_mode=eri_mode, reference=reference, **kwargs
    )
    if grad_analytic is not None and grad_analytic.size == ndof:
        if project_trans_rot:
            grad_analytic = _project_out_trans_rot_py(coords_bohr, grad_analytic, masses)
        return e0, grad_analytic

    # Invarian translasi: sum_A grad_A = 0 -> kita hanya perlu menghitung (N - 1) atom pertama!
    # Untuk molekul diatomik (N=2), ini memangkas biaya evaluasi gradien tepat 50%!
    grad = np.zeros((natoms, 3), dtype=np.float64)
    x0 = coords_bohr.copy()
    h = float(fd_step_bohr)

    for a in range(max(0, natoms - 1)):
        for xyz in range(3):
            # Cek apakah koordinat ini secara simetri identik nol (planar/linier)
            if np.all(np.abs(x0[:, xyz]) < 1.0e-12):
                grad[a, xyz] = 0.0
                continue

            xp = x0.copy()
            xm = x0.copy()
            xp[a, xyz] += h
            xm[a, xyz] -= h

            ep, _ = _evaluate_single_point_energy(
                _clone_molecule_with_bohr_coords(mol, xp),
                method=method, basis=basis, eri_mode=eri_mode, reference=reference, **kwargs
            )
            em, _ = _evaluate_single_point_energy(
                _clone_molecule_with_bohr_coords(mol, xm),
                method=method, basis=basis, eri_mode=eri_mode, reference=reference, **kwargs
            )

            if stencil == 4:
                xp2 = x0.copy()
                xm2 = x0.copy()
                xp2[a, xyz] += 2.0 * h
                xm2[a, xyz] -= 2.0 * h
                ep2, _ = _evaluate_single_point_energy(
                    _clone_molecule_with_bohr_coords(mol, xp2),
                    method=method, basis=basis, eri_mode=eri_mode, reference=reference, **kwargs
                )
                em2, _ = _evaluate_single_point_energy(
                    _clone_molecule_with_bohr_coords(mol, xm2),
                    method=method, basis=basis, eri_mode=eri_mode, reference=reference, **kwargs
                )
                grad[a, xyz] = (-ep2 + 8.0 * ep - 8.0 * em + em2) / (12.0 * h)
            else:
                grad[a, xyz] = (ep - em) / (2.0 * h)

    # Gunakan Hukum III Newton / Invarian Translasi untuk atom terakhir: F_last = -sum_{a=0}^{N-2} F_a
    if natoms > 1:
        grad[natoms - 1, :] = -np.sum(grad[: natoms - 1, :], axis=0)

    grad_flat = grad.ravel()
    if project_trans_rot:
        grad_flat = _project_out_trans_rot_py(coords_bohr, grad_flat, masses)

    return e0, grad_flat


def optimize_geometry(
    mol: Any,
    method: str = "omp2",
    basis: Optional[str] = None,
    eri_mode: str = "df",
    reference: Optional[str] = None,
    max_iter: int = 40,
    g_rms_tol: float = 1.0e-5,
    g_max_tol: float = 3.0e-5,
    dx_rms_tol: float = 4.0e-4,
    dx_max_tol: float = 1.2e-3,
    trust_radius: float = 0.25,
    fd_step_bohr: float = 5.0e-4,
    xyz_trajectory: Optional[str] = "mshqc_opt_traj.xyz",
    update_molecule: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> GeometryOptimizationResult:
    """
    Mengoptimasi geometri 3D molekul menuju titik stasioner minimum energi (dE/dR -> 0)
    menggunakan algoritma Trust-Radius Damped-BFGS dengan proyeksi rotasi-translasi SVD.
    """
    t0 = time.perf_counter()
    coords_bohr, symbols, atomic_nums, _, _, default_basis = _extract_molecule_metadata(mol)
    basis_use = basis or default_basis or "cc-pVDZ"
    natoms = coords_bohr.shape[0]
    ndof = 3 * natoms
    masses = np.array([ATOMIC_MASSES_AMU.get(z, float(2 * z)) for z in atomic_nums], dtype=np.float64)

    x = coords_bohr.ravel().copy()

    # Bangun model Hessian awal berbasis konstanta pegas empiris Lindh-Schlegel
    H = np.eye(ndof, dtype=np.float64) * 0.35
    x3 = x.reshape(natoms, 3)
    for i in range(natoms):
        for j in range(i + 1, natoms):
            rij_vec = x3[i] - x3[j]
            rij = float(np.linalg.norm(rij_vec))
            if 1.0e-3 < rij < 5.5:
                u_ij = rij_vec / rij
                k_stretch = 0.45 * math.exp(-0.45 * (rij - 1.8))
                P_ij = k_stretch * np.outer(u_ij, u_ij)
                H[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += P_ij
                H[3 * j : 3 * j + 3, 3 * j : 3 * j + 3] += P_ij
                H[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] -= P_ij
                H[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] -= P_ij

    if xyz_trajectory:
        with open(xyz_trajectory, "w", encoding="utf-8") as f:
            pass

    energy_hist: List[float] = []
    g_rms_hist: List[float] = []
    x_prev = x.copy()
    g_prev = np.zeros_like(x)
    e_prev = 0.0
    tr = float(trust_radius)
    converged = False

    if verbose:
        print("=" * 76)
        print(f"  MSHQC HPC GEOMETRY OPTIMIZER | Method: {method.upper()} ({eri_mode.upper()}) | Basis: {basis_use}")
        print("-" * 76)
        print(f"  {'Iter':>4} | {'Total Energy (Ha)':>18} | {'dE (Ha)':>12} | {'RMS Grad':>10} | {'MAX Grad':>10} | {'Step RMS':>10}")
        print("-" * 76)

    for it in range(max_iter):
        cur_mol = _clone_molecule_with_bohr_coords(mol, x)
        energy, g = compute_nuclear_gradient(
            cur_mol,
            method=method,
            basis=basis_use,
            eri_mode=eri_mode,
            reference=reference,
            fd_step_bohr=fd_step_bohr,
            project_trans_rot=True,
            **kwargs,
        )

        # Jika langkah menyebabkan energi naik drastis (overshoot), lakukan backtracking
        if it > 0 and energy > e_prev + 1.0e-4:
            tr = max(1.0e-3, 0.4 * np.linalg.norm(x - x_prev))
            x = 0.5 * (x + x_prev)
            cur_mol = _clone_molecule_with_bohr_coords(mol, x)
            energy, g = compute_nuclear_gradient(
                cur_mol,
                method=method,
                basis=basis_use,
                eri_mode=eri_mode,
                reference=reference,
                fd_step_bohr=fd_step_bohr,
                project_trans_rot=True,
                **kwargs,
            )

        g_rms = float(np.sqrt(np.mean(g**2)))
        g_max = float(np.max(np.abs(g)))
        de = 0.0 if it == 0 else (energy - e_prev)

        energy_hist.append(energy)
        g_rms_hist.append(g_rms)

        if xyz_trajectory:
            coords_ang = x.reshape(natoms, 3) * BOHR_TO_ANGSTROM
            with open(xyz_trajectory, "a", encoding="utf-8") as f:
                f.write(f"{natoms}\n")
                f.write(f"Step {it} E={energy:.10f} Ha g_rms={g_rms:.3e}\n")
                for sym, ca in zip(symbols, coords_ang):
                    f.write(f"{sym:<4} {ca[0]:15.8f} {ca[1]:15.8f} {ca[2]:15.8f}\n")

        # Update Damped-BFGS Hessian
        if it > 0:
            s = x - x_prev
            y = g - g_prev
            sy = float(np.dot(s, y))
            Hs = H @ s
            sHs = float(np.dot(s, Hs))

            theta = 1.0
            if sy < 0.2 * sHs and sHs > 1.0e-14:
                theta = (0.8 * sHs) / (sHs - sy)
            r_vec = theta * y + (1.0 - theta) * Hs
            sr = float(np.dot(s, r_vec))

            if sr > 1.0e-12 and sHs > 1.0e-12:
                H = H + np.outer(r_vec, r_vec) / sr - np.outer(Hs, Hs) / sHs

            pred_de = float(np.dot(g_prev, s) + 0.5 * sHs)
            if abs(pred_de) > 1.0e-12:
                ratio = de / pred_de
                if ratio > 0.75 and np.linalg.norm(s) >= 0.85 * tr:
                    tr = min(0.50, 1.4 * tr)
                elif ratio < 0.25:
                    tr = max(2.0e-3, 0.5 * tr)

        # Hitung langkah kuasi-Newton di sub-ruang bebas rotasi-translasi
        evals, evecs = np.linalg.eigh(0.5 * (H + H.T))
        g_mode = evecs.T @ g
        shift = max(0.0, -float(np.min(evals)) + 1.0e-3) if np.min(evals) < 1.0e-3 else 0.0
        dx_mode = -g_mode / np.maximum(evals + shift, 1.0e-3)
        dx = _project_out_trans_rot_py(x, evecs @ dx_mode, masses)

        step_norm = float(np.linalg.norm(dx))
        if step_norm > tr:
            dx *= tr / step_norm

        dx_rms = float(np.sqrt(np.mean(dx**2)))
        dx_max = float(np.max(np.abs(dx)))

        if verbose:
            print(f"  {it:4d} | {energy:18.10f} | {de:+12.3e} | {g_rms:10.3e} | {g_max:10.3e} | {dx_rms:10.3e}")

        grad_ok = (g_rms < g_rms_tol) and (g_max < g_max_tol)
        step_ok = (dx_rms < dx_rms_tol) and (dx_max < dx_max_tol)
        tight_grad = (g_rms < 0.25 * g_rms_tol) and (g_max < 0.25 * g_max_tol)

        if (grad_ok and step_ok) or tight_grad or natoms <= 1:
            converged = True
            break

        x_prev = x.copy()
        g_prev = g.copy()
        e_prev = energy
        x = x + dx

    final_coords_bohr = x.reshape(natoms, 3)
    final_coords_ang = final_coords_bohr * BOHR_TO_ANGSTROM

    # Hitung daftar jarak ikatan antar-atom
    bond_lengths: Dict[str, float] = {}
    for i in range(natoms):
        for j in range(i + 1, natoms):
            dist_a = float(np.linalg.norm(final_coords_ang[i] - final_coords_ang[j]))
            if dist_a < 3.0 or natoms <= 4:
                key = f"{symbols[i]}{i+1}-{symbols[j]}{j+1}"
                bond_lengths[key] = dist_a

    if update_molecule:
        try:
            updated_mol = _clone_molecule_with_bohr_coords(mol, final_coords_bohr)
            if hasattr(mol, "__dict__") and hasattr(updated_mol, "__dict__"):
                mol.__dict__.update(updated_mol.__dict__)
        except Exception:
            pass

    wall_t = time.perf_counter() - t0
    result = GeometryOptimizationResult(
        converged=converged,
        iterations=len(energy_hist),
        method=method,
        eri_mode=eri_mode,
        basis=basis_use,
        initial_energy=energy_hist[0],
        final_energy=energy_hist[-1],
        final_g_rms=g_rms_hist[-1],
        final_g_max=float(np.max(np.abs(g))),
        coordinates_bohr=final_coords_bohr,
        coordinates_angstrom=final_coords_ang,
        gradient_ha_bohr=g.reshape(natoms, 3),
        energy_history=energy_hist,
        g_rms_history=g_rms_hist,
        bond_lengths_angstrom=bond_lengths,
        wall_time_sec=wall_t,
        xyz_trajectory_path=xyz_trajectory,
    )
    if verbose:
        print(result.summary())
    return result


def compute_frequencies(
    mol: Any,
    method: str = "omp2",
    basis: Optional[str] = None,
    eri_mode: str = "df",
    reference: Optional[str] = None,
    temperature_k: float = 298.15,
    fd_step_bohr: float = 2.0e-3,
    verbose: bool = True,
    **kwargs: Any,
) -> FrequencyThermochemResult:
    """
    Menghitung frekuensi vibrasi harmonik (cm^-1), Zero-Point Energy (ZPE),
    Entalpi (H), dan Energi Bebas Gibbs (G) pada geometri saat ini.
    """
    coords_bohr, symbols, atomic_nums, _, mult, default_basis = _extract_molecule_metadata(mol)
    basis_use = basis or default_basis or "cc-pVDZ"
    natoms = coords_bohr.shape[0]
    ndof = 3 * natoms
    masses = np.array([ATOMIC_MASSES_AMU.get(z, float(2 * z)) for z in atomic_nums], dtype=np.float64)

    e0, _ = _evaluate_single_point_energy(
        mol, method=method, basis=basis_use, eri_mode=eri_mode, reference=reference, **kwargs
    )

    x0 = coords_bohr.ravel().copy()
    hess = np.zeros((ndof, ndof), dtype=np.float64)
    h = float(fd_step_bohr)

    # Evaluasi matriks Hessian melalui diferensiasi hingga gradien
    for k in range(ndof):
        xp = x0.copy()
        xm = x0.copy()
        xp[k] += h
        xm[k] -= h
        _, gp = compute_nuclear_gradient(
            _clone_molecule_with_bohr_coords(mol, xp),
            method=method, basis=basis_use, eri_mode=eri_mode, reference=reference,
            project_trans_rot=False, **kwargs
        )
        _, gm = compute_nuclear_gradient(
            _clone_molecule_with_bohr_coords(mol, xm),
            method=method, basis=basis_use, eri_mode=eri_mode, reference=reference,
            project_trans_rot=False, **kwargs
        )
        hess[:, k] = (gp - gm) / (2.0 * h)

    hess = 0.5 * (hess + hess.T)

    # Proyeksikan mode translasi & rotasi di ruang terbobot-massa
    x3 = coords_bohr.reshape(natoms, 3)
    com = np.sum(x3 * masses[:, None], axis=0) / np.sum(masses)
    r = x3 - com[None, :]

    D = np.zeros((ndof, 6), dtype=np.float64)
    for i in range(natoms):
        sm = math.sqrt(masses[i])
        D[3 * i + 0, 0] = sm
        D[3 * i + 1, 1] = sm
        D[3 * i + 2, 2] = sm
        D[3 * i + 1, 3] = -sm * r[i, 2]
        D[3 * i + 2, 3] =  sm * r[i, 1]
        D[3 * i + 0, 4] =  sm * r[i, 2]
        D[3 * i + 2, 4] = -sm * r[i, 0]
        D[3 * i + 0, 5] = -sm * r[i, 1]
        D[3 * i + 1, 5] =  sm * r[i, 0]

    U, s, _ = np.linalg.svd(D, full_matrices=True)
    n_tr = int(np.sum(s > 1.0e-6))
    V_int = U[:, n_tr:]

    inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)
    F_mw = hess * np.outer(inv_sqrt_m, inv_sqrt_m)
    F_int = V_int.T @ F_mw @ V_int

    evals, evecs = np.linalg.eigh(0.5 * (F_int + F_int.T))
    normal_modes = V_int @ evecs

    force_conv = HARTREE_TO_J / (BOHR_TO_M**2 * AMU_TO_KG)
    to_cm1 = 1.0 / (2.0 * math.pi * SPEED_OF_LIGHT_CMS)

    freqs_cm1 = np.zeros(len(evals), dtype=np.float64)
    n_imag = 0
    zpe_j = 0.0
    u_vib_j = 0.0
    s_vib_jk = 0.0

    for idx, lam in enumerate(evals):
        if lam < 0.0:
            freqs_cm1[idx] = -math.sqrt(-lam * force_conv) * to_cm1
            n_imag += 1
        else:
            f_cm = math.sqrt(lam * force_conv) * to_cm1
            freqs_cm1[idx] = f_cm
            hv = PLANCK_SI * f_cm * SPEED_OF_LIGHT_CMS
            zpe_j += 0.5 * hv
            if temperature_k > 1.0e-3 and f_cm > 1.0:
                x_val = hv / (BOLTZMANN_SI * temperature_k)
                if x_val < 100.0:
                    u_vib_j += hv / (math.exp(x_val) - 1.0)
                    s_vib_jk += BOLTZMANN_SI * (x_val / (math.exp(x_val) - 1.0) - math.log(1.0 - math.exp(-x_val)))

    zpe_ha = zpe_j / HARTREE_TO_J
    kt_ha = (BOLTZMANN_SI * temperature_k) / HARTREE_TO_J
    is_linear = (n_tr == 5)
    u_tr_ha = (2.5 if is_linear else 3.0) * kt_ha

    e_therm_ha = e0 + zpe_ha + u_tr_ha + (u_vib_j / HARTREE_TO_J)
    h_ha = e_therm_ha + kt_ha
    s_elec_jk = BOLTZMANN_SI * math.log(max(1, mult))
    s_tr_jk = BOLTZMANN_SI * (18.5 if is_linear else 21.0)
    g_ha = h_ha - temperature_k * (s_tr_jk + s_vib_jk + s_elec_jk) / HARTREE_TO_J

    res = FrequencyThermochemResult(
        frequencies_cm1=freqs_cm1,
        normal_modes=normal_modes,
        zpe_hartree=zpe_ha,
        zpe_kcal_mol=zpe_ha * HARTREE_TO_KCAL,
        electronic_energy_ha=e0,
        thermal_energy_ha=e_therm_ha,
        enthalpy_ha=h_ha,
        gibbs_free_energy_ha=g_ha,
        temperature_k=temperature_k,
        n_imaginary=n_imag,
        hessian_ha_bohr2=hess,
    )

    if verbose:
        print("=" * 72)
        print(f"  MSHQC HARMONIC VIBRATIONAL & THERMOCHEMISTRY ANALYSIS ({method.upper()})")
        print("-" * 72)
        for i, f_val in enumerate(freqs_cm1):
            tag = " (IMAGINARY / TS)" if f_val < 0 else ""
            print(f"    Mode {i+1:2d}: {f_val:10.2f} cm^-1{tag}")
        print("-" * 72)
        print(f"  Zero-Point Energy (ZPE) : {zpe_ha:14.8f} Ha ({res.zpe_kcal_mol:8.3f} kcal/mol)")
        print(f"  Thermal Enthalpy H(T)   : {h_ha:14.8f} Ha (at {temperature_k:.2f} K)")
        print(f"  Gibbs Free Energy G(T)  : {g_ha:14.8f} Ha (at {temperature_k:.2f} K)")
        print("=" * 72)

    return res


def scan_pes(
    mol: Any,
    atom_i: int,
    atom_j: int,
    distances_angstrom: List[float],
    method: str = "omp2",
    basis: Optional[str] = None,
    eri_mode: str = "df",
    reference: Optional[str] = None,
    verbose: bool = True,
    **kwargs: Any,
) -> List[Dict[str, float]]:
    """
    Memindai kurva Permukaan Energi Potensial (PES) sepanjang koordinat ikatan
    antara atom_i dan atom_j (sangat berguna untuk riset pemutusan ikatan homolitik OMP2/OMP3).
    """
    coords_bohr, symbols, _, _, _, default_basis = _extract_molecule_metadata(mol)
    basis_use = basis or default_basis or "cc-pVDZ"
    x0 = coords_bohr.reshape(-1, 3).copy()
    v_ij = x0[atom_j] - x0[atom_i]
    r0 = float(np.linalg.norm(v_ij))
    u_ij = v_ij / max(r0, 1.0e-8)

    results: List[Dict[str, float]] = []
    if verbose:
        print("=" * 60)
        print(f"  MSHQC PES SCAN: {symbols[atom_i]}{atom_i+1}-{symbols[atom_j]}{atom_j+1} ({method.upper()}/{eri_mode.upper()})")
        print("-" * 60)
        print(f"  {'R (Angstrom)':>14} | {'R (Bohr)':>12} | {'Energy (Ha)':>18}")
        print("-" * 60)

    for r_ang in distances_angstrom:
        r_bohr = float(r_ang) * ANGSTROM_TO_BOHR
        delta = r_bohr - r0
        x_scan = x0.copy()
        x_scan[atom_j] = x0[atom_j] + delta * u_ij
        scan_mol = _clone_molecule_with_bohr_coords(mol, x_scan)
        e_val, _ = _evaluate_single_point_energy(
            scan_mol, method=method, basis=basis_use, eri_mode=eri_mode, reference=reference, **kwargs
        )
        results.append({"r_angstrom": float(r_ang), "r_bohr": r_bohr, "energy_ha": e_val})
        if verbose:
            print(f"  {r_ang:14.4f} | {r_bohr:12.4f} | {e_val:18.10f}")

    if verbose:
        print("=" * 60)
    return results


def attach_geometry_methods_to_molecule(mol_cls: Any) -> None:
    """Menyuntikkan metode .optimize_geometry(), .frequencies(), .gradient(), dan .scan_pes() ke kelas Molecule."""
    if mol_cls is None:
        return
    try:
        setattr(mol_cls, "optimize_geometry", optimize_geometry)
        setattr(mol_cls, "frequencies", compute_frequencies)
        setattr(mol_cls, "gradient", compute_nuclear_gradient)
        setattr(mol_cls, "scan_pes", scan_pes)
    except TypeError:
        # Jika mol_cls adalah C++ nanobind/pybind11 extension type yang immutable secara langsung,
        # kita bungkus atau ekspos melalui mshqc.__init__
        pass
