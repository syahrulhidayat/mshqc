"""
python/mshqc/geom_opt.py
Mesin Optimasi Geometri Molekuler Berbasis Matrix-Free Trust-Region SOSCF (Steihaug-Toint TCG)
di Subruang Vibrasi Internal Eckart & Evaluasi Gradien Nuklir Multi-Metode mshqc.

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
from typing import Any, Callable, Dict, List, Optional, Tuple


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


def build_internal_vibrational_subspace(
    coords_bohr: np.ndarray,
    masses: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Membangun basis ortonormal subruang vibrasi internal V_int (3N x N_vib)
    yang bebas dari 5 atau 6 mode translasi & rotasi eksternal (Proyeksi Eckart).
    """
    natoms = coords_bohr.shape[0]
    ndof = 3 * natoms
    if natoms <= 1:
        return np.zeros((ndof, 0), dtype=np.float64)

    if masses is None:
        masses = np.ones(natoms, dtype=np.float64)

    total_mass = float(np.sum(masses))
    com = np.sum(coords_bohr * masses[:, None], axis=0) / total_mass
    rel = coords_bohr - com

    basis_tr = []
    for ax in range(3):
        t_vec = np.zeros((natoms, 3), dtype=np.float64)
        t_vec[:, ax] = np.sqrt(masses)
        basis_tr.append(t_vec.reshape(-1))

    eye3 = np.eye(3, dtype=np.float64)
    for ax in range(3):
        r_vec = np.cross(rel, eye3[ax]) * np.sqrt(masses)[:, None]
        basis_tr.append(r_vec.reshape(-1))

    B = np.column_stack(basis_tr)
    U, s, _ = np.linalg.svd(B, full_matrices=True)
    n_tr = int(np.sum(s > 1e-8))
    return U[:, n_tr:]


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

    V_int = build_internal_vibrational_subspace(coords_bohr, masses)
    if V_int.shape[1] == 0:
        return np.zeros_like(vec_flat)
    return V_int @ (V_int.T @ vec_flat)


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


def solve_trust_region_soscf_step(
    gradient: np.ndarray,
    hessian: np.ndarray,
    trust_radius: float,
    max_micro_iter: int = 40,
    micro_thresh: float = 1e-10,
    precond_shift: float = 1e-3,
) -> Tuple[np.ndarray, float, bool]:
    """
    Solver submasalah kuadratik Steihaug-Toint Preconditioned Truncated Conjugate Gradient (TCG):
        min_{||p||_2 <= R} g^T p + 0.5 p^T H p
    Identik secara matematis dengan mshqc::gradient::TrustRegionSOSCF di C++.
    """
    n = gradient.size
    z = np.zeros(n, dtype=np.float64)
    Hz = np.zeros(n, dtype=np.float64)
    r = gradient.copy()

    diag_h = np.maximum(np.diag(hessian), precond_shift)
    M_inv = 1.0 / diag_h
    p = -M_inv * r

    if np.linalg.norm(r) < micro_thresh:
        return z, 0.0, False

    def boundary_tau(z_vec: np.ndarray, p_vec: np.ndarray, rad: float) -> float:
        a = float(np.dot(p_vec, p_vec))
        b = 2.0 * float(np.dot(z_vec, p_vec))
        c = float(np.dot(z_vec, z_vec) - rad * rad)
        disc = b * b - 4.0 * a * c
        if disc < 0.0 or a <= 0.0:
            return 0.0
        return (-b + float(np.sqrt(disc))) / (2.0 * a)

    r_M_r_old = float(np.dot(r, -p))

    for _ in range(max_micro_iter):
        Hp = hessian @ p
        kappa = float(np.dot(p, Hp))
        if kappa <= 0.0:
            tau = boundary_tau(z, p, trust_radius)
            step = z + tau * p
            H_step = Hz + tau * Hp
            m_energy = float(np.dot(gradient, step) + 0.5 * np.dot(step, H_step))
            return step, m_energy, True

        alpha = r_M_r_old / kappa
        z_next = z + alpha * p
        if float(np.linalg.norm(z_next)) >= trust_radius:
            tau = boundary_tau(z, p, trust_radius)
            step = z + tau * p
            H_step = Hz + tau * Hp
            m_energy = float(np.dot(gradient, step) + 0.5 * np.dot(step, H_step))
            return step, m_energy, True

        z = z_next
        Hz = Hz + alpha * Hp
        r = r + alpha * Hp

        if float(np.linalg.norm(r)) < micro_thresh:
            m_energy = float(np.dot(gradient, z) + 0.5 * np.dot(z, Hz))
            return z, m_energy, False

        z_M_inv = M_inv * r
        r_M_r_new = float(np.dot(r, z_M_inv))
        beta = r_M_r_new / max(r_M_r_old, 1e-30)
        p = -z_M_inv + beta * p
        r_M_r_old = r_M_r_new

    final_m = float(np.dot(gradient, z) + 0.5 * np.dot(z, Hz))
    return z, final_m, False


def build_lindh_model_hessian_py(coords_bohr: np.ndarray) -> np.ndarray:
    """Membangun matriks Hessian awal model Lindh-Schlegel (ikatan + sudut valensi)."""
    natoms = coords_bohr.shape[0]
    ndof = 3 * natoms
    H = np.eye(ndof, dtype=np.float64) * 0.05

    for i in range(natoms):
        for j in range(i + 1, natoms):
            rij_vec = coords_bohr[i] - coords_bohr[j]
            rij = float(np.linalg.norm(rij_vec))
            if 1e-3 < rij < 5.5:
                u_ij = rij_vec / rij
                k_stretch = 0.45 * np.exp(-0.45 * (rij - 1.8))
                P_ij = k_stretch * np.outer(u_ij, u_ij)
                H[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += P_ij
                H[3 * j : 3 * j + 3, 3 * j : 3 * j + 3] += P_ij
                H[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] -= P_ij
                H[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] -= P_ij

    for j in range(natoms):
        for i in range(natoms):
            if i == j:
                continue
            u_ji = coords_bohr[i] - coords_bohr[j]
            r_ji = float(np.linalg.norm(u_ji))
            if r_ji < 1e-3 or r_ji > 4.5:
                continue
            u_ji = u_ji / r_ji
            for k in range(i + 1, natoms):
                if k == j:
                    continue
                u_jk = coords_bohr[k] - coords_bohr[j]
                r_jk = float(np.linalg.norm(u_jk))
                if r_jk < 1e-3 or r_jk > 4.5:
                    continue
                u_jk = u_jk / r_jk
                cos_t = float(np.clip(np.dot(u_ji, u_jk), -0.999999, 0.999999))
                sin_t = float(np.sqrt(max(1e-8, 1.0 - cos_t * cos_t)))
                if sin_t < 1e-3:
                    continue
                b_i = (cos_t * u_ji - u_jk) / (r_ji * sin_t)
                b_k = (cos_t * u_jk - u_ji) / (r_jk * sin_t)
                b_j = -(b_i + b_k)
                b_vec = np.zeros(ndof, dtype=np.float64)
                b_vec[3 * i : 3 * i + 3] = b_i
                b_vec[3 * j : 3 * j + 3] = b_j
                b_vec[3 * k : 3 * k + 3] = b_k
                k_bend = 0.16 * np.exp(-0.4 * ((r_ji - 1.8) + (r_jk - 1.8)))
                H += k_bend * np.outer(b_vec, b_vec)

    return 0.5 * (H + H.T)


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
    Optimizer Geometri Molekuler berbasis Matrix-Free Trust-Region SOSCF (Steihaug-Toint TCG)
    untuk seluruh metode yang terimplementasi di mshqc.
    """

    METHOD_REGISTRY: Dict[str, Tuple[str, str]] = {
        "rhf": ("RHF", "SCF"),
        "uhf": ("UHF", "SCF"),
        "rohf": ("ROHF", "SCF"),
        "mp2": ("RHF", "MP2"),
        "rmp2": ("RHF", "MP2"),
        "ump2": ("UHF", "MP2"),
        "romp2": ("ROHF", "ROMP2"),
        "mp3": ("RHF", "MP3"),
        "rmp3": ("RHF", "MP3"),
        "ump3": ("UHF", "MP3"),
        "romp3": ("ROHF", "ROMP3"),
        "omp2": ("RHF", "OMP2"),
        "rhf-omp2": ("RHF", "OMP2"),
        "uhf-omp2": ("UHF", "OMP2"),
        "omp3": ("RHF", "OMP3"),
        "rhf-omp3": ("RHF", "OMP3"),
        "uhf-omp3": ("UHF", "OMP3"),
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
        self._last_diag_curv: Optional[np.ndarray] = None

    def evaluate_energy_and_gradient(self, coords_bohr: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Mengevaluasi energi (Ha), gradien nuklir (Ha/Bohr), serta menyimpan kurvatur
        turunan kedua lokal d^2E/dx_i^2 dari stensil Central Finite-Difference O(h^2).
        """
        coords_bohr = np.asarray(coords_bohr, dtype=np.float64)
        natoms = coords_bohr.shape[0]
        self._last_diag_curv = np.zeros((natoms, 3), dtype=np.float64)

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

        for atom_idx in range(natoms - 1):
            for cart in range(3):
                c_plus = coords_bohr.copy()
                c_minus = coords_bohr.copy()
                c_plus[atom_idx, cart] += h
                c_minus[atom_idx, cart] -= h

                if self.enforce_symmetry and abs(coords_bohr[atom_idx, cart]) < 1e-12:
                    refl_check = coords_bohr.copy()
                    refl_check[:, cart] *= -1.0
                    if np.allclose(coords_bohr, refl_check, atol=1e-7):
                        grad_mat[atom_idx, cart] = 0.0
                        continue

                e_p = extract_energy_scalar(self.energy_fn(c_plus), self.method_key)
                e_m = extract_energy_scalar(self.energy_fn(c_minus), self.method_key)
                grad_mat[atom_idx, cart] = (e_p - e_m) / (2.0 * h)
                curv = (e_p - 2.0 * e0 + e_m) / (h * h)
                if np.isfinite(curv) and 1e-2 < curv < 20.0:
                    self._last_diag_curv[atom_idx, cart] = curv

        grad_mat[-1, :] = -np.sum(grad_mat[:-1, :], axis=0)
        g_flat = project_translation_rotation(coords_bohr, grad_mat.reshape(-1))
        if self.enforce_symmetry:
            g_flat = symmetrize_cartesian_vector(coords_bohr, g_flat)

        return e0, g_flat

    def run(self, initial_coords_bohr: np.ndarray) -> GeomOptResult:
        """Menjalankan siklus optimasi geometri Matrix-Free Trust-Region SOSCF (Steihaug-Toint TCG)."""
        coords = np.asarray(initial_coords_bohr, dtype=np.float64).copy()
        natoms = coords.shape[0]
        ndim = 3 * natoms

        H_hess = build_lindh_model_hessian_py(coords)
        trust_radius = self.trust_radius_init

        e_curr, g_curr = self.evaluate_energy_and_gradient(coords)
        e_init = e_curr

        if self._last_diag_curv is not None:
            if natoms == 2:
                for xyz in range(3):
                    k_ex = float(self._last_diag_curv[0, xyz])
                    if k_ex > 1e-2:
                        H_hess[xyz, xyz] = k_ex
                        H_hess[3 + xyz, 3 + xyz] = k_ex
                        H_hess[xyz, 3 + xyz] = -k_ex
                        H_hess[3 + xyz, xyz] = -k_ex
            else:
                for a in range(natoms - 1):
                    for xyz in range(3):
                        k_ex = float(self._last_diag_curv[a, xyz])
                        if k_ex > 1e-2:
                            idx = 3 * a + xyz
                            H_hess[idx, idx] = 0.7 * k_ex + 0.3 * H_hess[idx, idx]

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
            V_int = build_internal_vibrational_subspace(coords)
            if V_int.shape[1] > 0:
                g_int = V_int.T @ g_curr
                H_int = 0.5 * (V_int.T @ H_hess @ V_int + (V_int.T @ H_hess @ V_int).T)
                eigvals, eigvecs = np.linalg.eigh(H_int)
                eigvals_safe = np.clip(eigvals, 2.0e-2, 20.0)
                H_int = (eigvecs * eigvals_safe) @ eigvecs.T

                p_int, de_pred, hit_boundary = solve_trust_region_soscf_step(
                    g_int, H_int, trust_radius
                )
                dx = V_int @ p_int
            else:
                dx = np.zeros(ndim, dtype=np.float64)
                de_pred = 0.0
                hit_boundary = False

            dx = project_translation_rotation(coords, dx)
            if self.enforce_symmetry:
                dx = symmetrize_cartesian_vector(coords, dx)

            step_norm = float(np.linalg.norm(dx))
            dx_rms = float(np.sqrt(np.mean(dx ** 2)))
            dx_max = float(np.max(np.abs(dx)))

            coords_new = coords + dx.reshape(natoms, 3)
            e_new, g_new = self.evaluate_energy_and_gradient(coords_new)
            de_actual = e_new - e_curr

            # Step rejection / backtracking jika langkah menaikkan energi
            if de_actual > 1e-7 and step_norm > self.trust_radius_min:
                trust_radius = max(self.trust_radius_min, 0.4 * step_norm)
                dx = 0.5 * dx
                step_norm = float(np.linalg.norm(dx))
                dx_rms = float(np.sqrt(np.mean(dx ** 2)))
                dx_max = float(np.max(np.abs(dx)))
                coords_new = coords + dx.reshape(natoms, 3)
                e_new, g_new = self.evaluate_energy_and_gradient(coords_new)
                de_actual = e_new - e_curr

            rho = de_actual / de_pred if abs(de_pred) > 1e-14 else 1.0
            if rho < 0.25:
                trust_radius = max(self.trust_radius_min, 0.5 * step_norm)
            elif rho > 0.75 and hit_boundary:
                trust_radius = min(self.trust_radius_max, 1.5 * trust_radius)

            # Injeksi kurvatur eksak diatomik atau Update Hessian Bofill (SR1 + PSB) untuk poliatomik
            if natoms == 2 and self._last_diag_curv is not None:
                for xyz in range(3):
                    k_ex = float(self._last_diag_curv[0, xyz])
                    if k_ex > 1e-2:
                        H_hess[xyz, xyz] = k_ex
                        H_hess[3 + xyz, 3 + xyz] = k_ex
                        H_hess[xyz, 3 + xyz] = -k_ex
                        H_hess[3 + xyz, xyz] = -k_ex
            else:
                s_vec = dx
                y_vec = g_new - g_curr
                s_norm = float(np.linalg.norm(s_vec))
                y_norm = float(np.linalg.norm(y_vec))
                if s_norm > 1e-4 and y_norm > 1e-6:
                    r_vec = y_vec - H_hess @ s_vec
                    s2 = float(np.dot(s_vec, s_vec))
                    r2 = float(np.dot(r_vec, r_vec))
                    sr = float(np.dot(s_vec, r_vec))
                    if s2 > 1e-12 and r2 > 1e-16:
                        phi = float(np.clip((sr * sr) / (r2 * s2), 0.0, 1.0))
                        dH_psb = (
                            (np.outer(r_vec, s_vec) + np.outer(s_vec, r_vec)) / s2
                            - (sr / (s2 * s2)) * np.outer(s_vec, s_vec)
                        )
                        dH = (1.0 - phi) * dH_psb
                        if abs(sr) > 1e-8 * np.sqrt(s2 * r2):
                            dH = dH + phi * (np.outer(r_vec, r_vec) / sr)
                        H_hess = 0.5 * (H_hess + H_hess.T + dH + dH.T)

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

try:
    from ._mshqc import HPCGeometryOptimizer, GeomOptConfig, VibThermoResult, PESScanPoint
except ImportError:
    HPCGeometryOptimizer = None  # type: ignore
