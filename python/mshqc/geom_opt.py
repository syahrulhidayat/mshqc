"""
python/mshqc/geom_opt.py
Mesin Optimasi Geometri Molekuler & Evaluasi Gradien Nuklir Multi-Metode mshqc.

Mendukung metode:
  1. SCF Variasional       : RHF, UHF, ROHF
  2. MP Kanonik            : RMP2, UMP2, ROMP2, RMP3, UMP3, ROMP3
  3. Orbital-Optimized MP2 : OMP2 (RHF-OMP2), UHF-OMP2 (Exact / DF / Cholesky)
  4. Orbital-Optimized MP3 : OMP3 (RHF-OMP3), UHF-OMP3 (Exact / DF / Cholesky)
  5. Hibrida Dual-Level    : MP3@OMP2, UMP3@UHF-OMP2
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM


@dataclass
class GeomOptResult:
    """Struktur hasil akhir optimasi geometri untuk seluruh metode mshqc."""
    converged: bool
    method: str
    reference: str
    integral_mode: str
    iterations: int
    initial_energy: float
    final_energy: float
    final_coords_bohr: np.ndarray
    final_coords_angstrom: np.ndarray
    final_gradient_bohr: np.ndarray
    grad_rms: float
    grad_max: float
    step_rms: float
    step_max: float
    energy_history: List[float] = field(default_factory=list)
    grad_rms_history: List[float] = field(default_factory=list)
    trajectory_bohr: List[np.ndarray] = field(default_factory=list)


def project_translation_rotation(
    coords_bohr: np.ndarray,
    vec_flat: np.ndarray,
    masses: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Memproyeksikan keluar 6 mode translasi & rotasi eksternal (5 untuk molekul linear)
    dari vektor gradien atau langkah perpindahan Kartesian 3N (Proyeksi Eckart).
    """
    natoms = coords_bohr.shape[0]
    if natoms <= 1:
        return np.zeros_like(vec_flat)

    if masses is None:
        masses = np.ones(natoms, dtype=np.float64)

    total_mass = np.sum(masses)
    com = np.sum(coords_bohr * masses[:, None], axis=0) / total_mass
    rel = coords_bohr - com

    basis_tr = []
    # 3 Mode Translasi Kaku
    for ax in range(3):
        t_vec = np.zeros((natoms, 3), dtype=np.float64)
        t_vec[:, ax] = np.sqrt(masses)
        basis_tr.append(t_vec.reshape(-1))

    # 3 Mode Rotasi Kaku Infinitesimal
    eye3 = np.eye(3, dtype=np.float64)
    for ax in range(3):
        r_vec = np.cross(rel, eye3[ax]) * np.sqrt(masses)[:, None]
        basis_tr.append(r_vec.reshape(-1))

    B = np.column_stack(basis_tr)
    U, s, _ = np.linalg.svd(B, full_matrices=False)
    U_tr = U[:, s > 1e-8]

    return vec_flat - U_tr @ (U_tr.T @ vec_flat)


def symmetrize_cartesian_vector(coords_bohr: np.ndarray, vec_flat: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """
    Menjaga simetri refleksi bidang Kartesian (x -> -x, y -> -y, z -> -z) yang ada pada
    geometri molekul agar floating-point noise dari OpenMP/TBLIS tidak merusak grup titik.
    """
    natoms = coords_bohr.shape[0]
    vec_3d = vec_flat.reshape(natoms, 3).copy()

    for axis in range(3):
        refl_coords = coords_bohr.copy()
        refl_coords[:, axis] *= -1.0

        perm = np.full(natoms, -1, dtype=int)
        for i in range(natoms):
            dists = np.linalg.norm(coords_bohr - refl_coords[i], axis=1)
            j = int(np.argmin(dists))
            if dists[j] < tol:
                perm[i] = j

        if np.all(perm >= 0) and len(set(perm)) == natoms:
            refl_vec = vec_3d[perm].copy()
            refl_vec[:, axis] *= -1.0
            vec_3d = 0.5 * (vec_3d + refl_vec)

    return vec_3d.reshape(-1)


def extract_energy_scalar(res: Any, method_key: str = "") -> float:
    """Mengekstrak energi total (Hartree) dari nilai balik fungsi/objek C++ mshqc."""
    if isinstance(res, (float, int, np.floating)):
        return float(res)

    if isinstance(res, dict):
        priority_keys = (
            "e_total", "total_energy", "energy", "final_energy",
            "e_omp3", "e_omp2", "e_mp3", "e_mp2", "e_scf",
        )
        for k in priority_keys:
            if k in res and res[k] is not None:
                return float(res[k])
        if "e_ref" in res and "e_corr" in res:
            return float(res["e_ref"]) + float(res["e_corr"])

    for attr in ("e_total", "total_energy", "energy", "final_energy", "e_omp3", "e_omp2"):
        if hasattr(res, attr):
            val = getattr(res, attr)
            return float(val() if callable(val) else val)

    if hasattr(res, "e_ref") and hasattr(res, "e_corr"):
        return float(res.e_ref) + float(res.e_corr)

    raise TypeError(f"Tidak dapat mengekstrak skalar energi total dari objek {type(res)} (method={method_key}).")


class UniversalGeomOptimizer:
    """
    Optimizer Geometri Molekuler untuk seluruh metode yang terimplementasi di mshqc.
    """

    METHOD_REGISTRY: Dict[str, Tuple[str, str]] = {
        # Variational Mean-Field SCF
        "rhf": ("RHF", "SCF"),
        "uhf": ("UHF", "SCF"),
        "rohf": ("ROHF", "SCF"),
        # Canonical Møller-Plesset
        "mp2": ("RHF", "MP2"),
        "rmp2": ("RHF", "MP2"),
        "ump2": ("UHF", "MP2"),
        "romp2": ("ROHF", "ROMP2"),
        "mp3": ("RHF", "MP3"),
        "rmp3": ("RHF", "MP3"),
        "ump3": ("UHF", "MP3"),
        "romp3": ("ROHF", "ROMP3"),
        # Orbital-Optimized Møller-Plesset (Closed-Shell RHF & Open-Shell UHF)
        "omp2": ("RHF", "OMP2"),
        "rhf-omp2": ("RHF", "OMP2"),
        "uhf-omp2": ("UHF", "OMP2"),
        "omp3": ("RHF", "OMP3"),
        "rhf-omp3": ("RHF", "OMP3"),
        "uhf-omp3": ("UHF", "OMP3"),
        # Dual-Level Hybrid
        "mp3@omp2": ("RHF", "MP3@OMP2"),
        "ump3@uhf-omp2": ("UHF", "MP3@OMP2"),
    }

    def __init__(
        self,
        energy_fn: Callable[[np.ndarray], Any],
        grad_fn: Optional[Callable[[np.ndarray], Tuple[float, np.ndarray]]] = None,
        method: str = "omp2",
        integral_mode: str = "df",
        fd_step_bohr: float = 1.0e-4,
        max_iter: int = 50,
        grad_rms_tol: float = 1.0e-5,
        grad_max_tol: float = 3.0e-5,
        step_rms_tol: float = 4.0e-5,
        step_max_tol: float = 1.2e-4,
        energy_tol: float = 1.0e-7,
        trust_radius_init: float = 0.25,
        trust_radius_max: float = 0.50,
        trust_radius_min: float = 1.0e-3,
        enforce_symmetry: bool = True,
    ):
        norm = method.strip().lower().replace("_", "-")
        for prefix in ("df-", "cd-", "cholesky-", "exact-"):
            if norm.startswith(prefix):
                integral_mode = prefix.rstrip("-")
                norm = norm[len(prefix):]

        if norm in ("rohf-omp2", "rohf-omp3"):
            raise NotImplementedError(
                f"Metode '{method}' tidak diimplementasikan di mshqc. "
                f"Untuk sistem open-shell gunakan 'uhf-{norm.split('-')[1]}' (Orbital-Optimized) "
                f"atau 'ro{norm.split('-')[1]}' (Semicanonical ROHF Perturbation)."
            )

        if norm not in self.METHOD_REGISTRY:
            raise ValueError(
                f"Metode '{method}' tidak valid. Pilihan yang tersedia: "
                f"{sorted(self.METHOD_REGISTRY.keys())}"
            )

        self.method_key = norm
        self.reference, self.corr_type = self.METHOD_REGISTRY[norm]
        self.integral_mode = integral_mode.upper()
        self.energy_fn = energy_fn
        self.grad_fn = grad_fn
        self.fd_step_bohr = fd_step_bohr
        self.max_iter = max_iter
        self.grad_rms_tol = grad_rms_tol
        self.grad_max_tol = grad_max_tol
        self.step_rms_tol = step_rms_tol
        self.step_max_tol = step_max_tol
        self.energy_tol = energy_tol
        self.trust_radius_init = trust_radius_init
        self.trust_radius_max = trust_radius_max
        self.trust_radius_min = trust_radius_min
        self.enforce_symmetry = enforce_symmetry

    def evaluate_energy_and_gradient(self, coords_bohr: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Mengevaluasi energi (Ha) dan gradien nuklir (Ha/Bohr).
        Jika grad_fn analitik tersedia, gunakan langsung; jika tidak, gunakan
        Central Finite-Difference O(h^2) dengan penghematan translasi pusat massa.
        """
        coords_bohr = np.asarray(coords_bohr, dtype=np.float64)
        natoms = coords_bohr.shape[0]

        if self.grad_fn is not None:
            e_val, g_raw = self.grad_fn(coords_bohr)
            g_flat = np.asarray(g_raw, dtype=np.float64).reshape(-1)
            g_flat = project_translation_rotation(coords_bohr, g_flat)
            if self.enforce_symmetry:
                g_flat = symmetrize_cartesian_vector(coords_bohr, g_flat)
            return float(e_val), g_flat

        e0 = extract_energy_scalar(self.energy_fn(coords_bohr), self.method_key)
        if natoms <= 1:
            return e0, np.zeros(3 * natoms, dtype=np.float64)

        grad_mat = np.zeros((natoms, 3), dtype=np.float64)
        h = self.fd_step_bohr

        # Gunakan sifat sum_A (dE/dR_A) = 0: cukup hitung N-1 atom pertama,
        # atom terakhir ditentukan secara eksak dari invarian translasi.
        for atom_idx in range(natoms - 1):
            for cart in range(3):
                c_plus = coords_bohr.copy()
                c_minus = coords_bohr.copy()
                c_plus[atom_idx, cart] += h
                c_minus[atom_idx, cart] -= h

                # Jika pergeseran membuat semua koordinat di sumbu cart bernilai 0 pada titik asal,
                # refleksi simetris E(+h) == E(-h) membuat gradien sumbu tersebut 0 secara identitas.
                if self.enforce_symmetry and abs(coords_bohr[atom_idx, cart]) < 1e-12:
                    refl_check = coords_bohr.copy()
                    refl_check[:, cart] *= -1.0
                    if np.allclose(coords_bohr, refl_check, atol=1e-7):
                        grad_mat[atom_idx, cart] = 0.0
                        continue

                e_p = extract_energy_scalar(self.energy_fn(c_plus), self.method_key)
                e_m = extract_energy_scalar(self.energy_fn(c_minus), self.method_key)
                grad_mat[atom_idx, cart] = (e_p - e_m) / (2.0 * h)

        grad_mat[-1, :] = -np.sum(grad_mat[:-1, :], axis=0)
        g_flat = project_translation_rotation(coords_bohr, grad_mat.reshape(-1))
        if self.enforce_symmetry:
            g_flat = symmetrize_cartesian_vector(coords_bohr, g_flat)

        return e0, g_flat

    def run(self, initial_coords_bohr: np.ndarray) -> GeomOptResult:
        """Menjalankan siklus optimasi geometri Trust-Radius Damped BFGS."""
        coords = np.asarray(initial_coords_bohr, dtype=np.float64).copy()
        natoms = coords.shape[0]
        ndim = 3 * natoms

        B_hess = np.eye(ndim, dtype=np.float64) * 0.5
        H_inv = np.eye(ndim, dtype=np.float64) * 2.0
        trust_radius = self.trust_radius_init

        e_curr, g_curr = self.evaluate_energy_and_gradient(coords)
        e_init = e_curr

        energy_hist = [e_curr]
        g_rms = float(np.sqrt(np.mean(g_curr ** 2)))
        g_max = float(np.max(np.abs(g_curr)))
        grad_rms_hist = [g_rms]
        traj_hist = [coords.copy()]

        if g_rms < self.grad_rms_tol and g_max < self.grad_max_tol:
            return GeomOptResult(
                converged=True,
                method=self.method_key.upper(),
                reference=self.reference,
                integral_mode=self.integral_mode,
                iterations=0,
                initial_energy=e_init,
                final_energy=e_curr,
                final_coords_bohr=coords,
                final_coords_angstrom=coords * BOHR_TO_ANGSTROM,
                final_gradient_bohr=g_curr.reshape(natoms, 3),
                grad_rms=g_rms,
                grad_max=g_max,
                step_rms=0.0,
                step_max=0.0,
                energy_history=energy_hist,
                grad_rms_history=grad_rms_hist,
                trajectory_bohr=traj_hist,
            )

        converged = False
        dx_rms = 0.0
        dx_max = 0.0

        for _ in range(1, self.max_iter + 1):
            dx = -H_inv @ g_curr
            dx = project_translation_rotation(coords, dx)
            if self.enforce_symmetry:
                dx = symmetrize_cartesian_vector(coords, dx)

            step_norm = float(np.linalg.norm(dx))
            if step_norm > trust_radius:
                dx = dx * (trust_radius / step_norm)

            dx_rms = float(np.sqrt(np.mean(dx ** 2)))
            dx_max = float(np.max(np.abs(dx)))

            de_pred = float(np.dot(g_curr, dx) + 0.5 * np.dot(dx, B_hess @ dx))
            coords_new = coords + dx.reshape(natoms, 3)
            e_new, g_new = self.evaluate_energy_and_gradient(coords_new)
            de_actual = e_new - e_curr

            # Backtracking line-search jika langkah menaikkan energi
            if de_actual > 1e-5 and step_norm > self.trust_radius_min:
                dx = 0.5 * dx
                dx_rms = float(np.sqrt(np.mean(dx ** 2)))
                dx_max = float(np.max(np.abs(dx)))
                coords_new = coords + dx.reshape(natoms, 3)
                e_new, g_new = self.evaluate_energy_and_gradient(coords_new)
                de_actual = e_new - e_curr

            rho = de_actual / de_pred if abs(de_pred) > 1e-14 else 1.0
            if rho < 0.25:
                trust_radius = max(self.trust_radius_min, 0.5 * step_norm)
            elif rho > 0.75 and abs(step_norm - trust_radius) < 1e-5:
                trust_radius = min(self.trust_radius_max, 1.5 * trust_radius)

            # Damped BFGS Update (Powell Damping)
            s_vec = dx
            y_vec = g_new - g_curr
            sy = float(np.dot(s_vec, y_vec))
            Bs = B_hess @ s_vec
            sBs = float(np.dot(s_vec, Bs))

            if sy >= 0.2 * sBs and sy > 1e-12:
                theta = 1.0
            elif sBs > 1e-12:
                theta = (0.8 * sBs) / (sBs - sy)
            else:
                theta = 1.0

            r_vec = theta * y_vec + (1.0 - theta) * Bs
            sr = float(np.dot(s_vec, r_vec))
            if sr > 1e-12 and sBs > 1e-12:
                B_hess = B_hess - np.outer(Bs, Bs) / sBs + np.outer(r_vec, r_vec) / sr
                eigvals, eigvecs = np.linalg.eigh(0.5 * (B_hess + B_hess.T))
                eigvals_safe = np.clip(eigvals, 1.0e-2, 50.0)
                H_inv = (eigvecs / eigvals_safe) @ eigvecs.T

            coords = coords_new
            e_curr = e_new
            g_curr = g_new

            g_rms = float(np.sqrt(np.mean(g_curr ** 2)))
            g_max = float(np.max(np.abs(g_curr)))

            energy_hist.append(e_curr)
            grad_rms_hist.append(g_rms)
            traj_hist.append(coords.copy())

            if (g_rms < self.grad_rms_tol and g_max < self.grad_max_tol) and (
                (dx_rms < self.step_rms_tol and dx_max < self.step_max_tol)
                or abs(de_actual) < self.energy_tol
            ):
                converged = True
                break

        return GeomOptResult(
            converged=converged,
            method=self.method_key.upper(),
            reference=self.reference,
            integral_mode=self.integral_mode,
            iterations=len(energy_hist) - 1,
            initial_energy=e_init,
            final_energy=e_curr,
            final_coords_bohr=coords,
            final_coords_angstrom=coords * BOHR_TO_ANGSTROM,
            final_gradient_bohr=g_curr.reshape(natoms, 3),
            grad_rms=g_rms,
            grad_max=g_max,
            step_rms=dx_rms,
            step_max=dx_max,
            energy_history=energy_hist,
            grad_rms_history=grad_rms_hist,
            trajectory_bohr=traj_hist,
        )