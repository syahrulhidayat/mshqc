# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
Orkestrasi SCF Python tipis (DF / Cholesky RHF & UHF) di atas kernel C++20 `mshqc.kernels`.
Mencakup Pure-Python SAD Guess (Z=1..118 + anomali d/f-block), faktorisasi low-rank Iter-1,
Pulay DIIS (SVD-equilibrated), dan orthogonalizer Löwdin tahan ketergantungan linear.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np
from scipy.linalg import eigh

from .._mshqc import kernels as K


@dataclass
class HybridSCFResult:
    reference: str
    energy_total: float
    converged: bool
    iterations: int
    C_alpha: np.ndarray
    C_beta: np.ndarray
    orbital_energies_alpha: np.ndarray
    orbital_energies_beta: np.ndarray
    n_occ_alpha: int
    n_occ_beta: int
    s2: float
    L_mat: np.ndarray

    @property
    def e_total(self) -> float:
        return self.energy_total


_SAD_ANOMALIES: Dict[int, Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]] = {
    24: ((2, 2, 2, 1), (6, 6), (5,), ()),
    29: ((2, 2, 2, 1), (6, 6), (10,), ()),
    41: ((2, 2, 2, 2, 1), (6, 6, 6), (10, 4), ()),
    42: ((2, 2, 2, 2, 1), (6, 6, 6), (10, 5), ()),
    44: ((2, 2, 2, 2, 1), (6, 6, 6), (10, 7), ()),
    45: ((2, 2, 2, 2, 1), (6, 6, 6), (10, 8), ()),
    46: ((2, 2, 2, 2), (6, 6, 6), (10, 10), ()),
    47: ((2, 2, 2, 2, 1), (6, 6, 6), (10, 10), ()),
    57: ((2, 2, 2, 2, 2, 2), (6, 6, 6, 6), (10, 10, 1), ()),
    58: ((2, 2, 2, 2, 2, 2), (6, 6, 6, 6), (10, 10, 1), (1,)),
    64: ((2, 2, 2, 2, 2, 2), (6, 6, 6, 6), (10, 10, 1), (7,)),
    78: ((2, 2, 2, 2, 2, 1), (6, 6, 6, 6), (10, 10, 9), (14,)),
    79: ((2, 2, 2, 2, 2, 1), (6, 6, 6, 6), (10, 10, 10), (14,)),
    89: ((2, 2, 2, 2, 2, 2, 2), (6, 6, 6, 6, 6), (10, 10, 10, 1), (14,)),
    90: ((2, 2, 2, 2, 2, 2, 2), (6, 6, 6, 6, 6), (10, 10, 10, 2), (14,)),
    91: ((2, 2, 2, 2, 2, 2, 2), (6, 6, 6, 6, 6), (10, 10, 10, 1), (14, 2)),
    92: ((2, 2, 2, 2, 2, 2, 2), (6, 6, 6, 6, 6), (10, 10, 10, 1), (14, 3)),
    93: ((2, 2, 2, 2, 2, 2, 2), (6, 6, 6, 6, 6), (10, 10, 10, 1), (14, 4)),
    96: ((2, 2, 2, 2, 2, 2, 2), (6, 6, 6, 6, 6), (10, 10, 10, 1), (14, 7)),
}


def _subshell_occupations(z: int) -> Tuple[List[float], List[float], List[float], List[float]]:
    if z in _SAD_ANOMALIES:
        s, p, d, f = _SAD_ANOMALIES[z]
        return list(s), list(p), list(d), list(f)

    s_occ: List[float] = []
    p_occ: List[float] = []
    d_occ: List[float] = []
    f_occ: List[float] = []

    def take(rem: float, cap: float, target: List[float]) -> float:
        val = min(cap, max(0.0, rem))
        if val > 0.0:
            target.append(val)
        return rem - val

    rem = float(z)
    for caps in (
        ((2.0, s_occ),),
        ((2.0, s_occ), (6.0, p_occ)),
        ((2.0, s_occ), (6.0, p_occ)),
        ((2.0, s_occ), (10.0, d_occ), (6.0, p_occ)),
        ((2.0, s_occ), (10.0, d_occ), (6.0, p_occ)),
        ((2.0, s_occ), (14.0, f_occ), (10.0, d_occ), (6.0, p_occ)),
        ((2.0, s_occ), (14.0, f_occ), (10.0, d_occ), (6.0, p_occ)),
        ((2.0, s_occ),),
    ):
        for cap, target in caps:
            if rem <= 0.0:
                return s_occ, p_occ, d_occ, f_occ
            rem = take(rem, cap, target)
    return s_occ, p_occ, d_occ, f_occ


def _shell_l_and_nf(shell) -> Tuple[int, int]:
    l_val = int(shell.l()) if hasattr(shell, "l") else int(shell.angular_momentum.value)
    nf = int(shell.n_functions()) if hasattr(shell, "n_functions") else (2 * l_val + 1)
    return l_val, nf


def build_sad_guess(mol, basis) -> np.ndarray:
    """Pure-Python SAD guess (identik secara matematis dengan src/scf/sad_guess.cc)."""
    nbf = int(basis.n_basis_functions())
    p_diag = np.zeros(nbf, dtype=np.float64)
    bname = str(basis.name()).lower()
    allow_split = not any(tok in bname for tok in ("sto", "min", "3-21g"))

    n_shells = int(basis.n_shells())
    sh_idx = 0
    bf_off = 0

    for atom_i in range(int(mol.n_atoms())):
        z = int(mol.atom(atom_i).atomic_number)
        groups: List[List[Tuple[int, int]]] = [[], [], [], []]
        atom_bf = 0

        while sh_idx < n_shells:
            sh = basis.shell(sh_idx)
            if int(sh.center) != atom_i:
                break
            l_val, nf = _shell_l_and_nf(sh)
            if 0 <= l_val <= 3:
                groups[l_val].append((bf_off + atom_bf, nf))
            atom_bf += nf
            sh_idx += 1

        occ_groups = _subshell_occupations(z)
        for occ_list, sh_list in zip(occ_groups, groups):
            n_sub = len(occ_list)
            for k, occ in enumerate(occ_list):
                if k >= len(sh_list):
                    break
                off1, deg = sh_list[k]
                avg = occ / float(deg)
                if (k + 1 == n_sub) and allow_split and (k + 1 < len(sh_list)):
                    off2, _ = sh_list[k + 1]
                    p_diag[off1 : off1 + deg] = 0.8 * avg
                    p_diag[off2 : off2 + deg] = 0.2 * avg
                else:
                    p_diag[off1 : off1 + deg] = avg

        bf_off += atom_bf

    return np.diag(p_diag)


def _factorize_psd_density(P: np.ndarray, tol: float = 1.0e-12) -> np.ndarray:
    """Ekstrak faktor low-rank C_fac (P = C_fac @ C_fac.T) agar Iter-1 memakai batched DGEMM."""
    n = P.shape[0]
    diag = np.diag(P)
    off_norm = float(np.max(np.abs(P - np.diag(diag)))) if n > 0 else 0.0
    if off_norm < 1.0e-13:
        nz = np.flatnonzero(diag > tol)
        C_fac = np.zeros((n, nz.size), dtype=np.float64)
        C_fac[nz, np.arange(nz.size)] = np.sqrt(diag[nz])
        return np.ascontiguousarray(C_fac)

    evals, evecs = eigh(0.5 * (P + P.T))
    keep = evals > tol
    return np.ascontiguousarray(evecs[:, keep] * np.sqrt(evals[keep])[None, :])


class PulayDIIS:
    """Incremental SVD-equilibrated Pulay DIIS (setara matematis dengan src/scf/diis.cc)."""

    def __init__(self, max_vectors: int = 8) -> None:
        if max_vectors < 2:
            raise ValueError("max_vectors harus >= 2")
        self.max_vectors = int(max_vectors)
        self._focks: Deque[Tuple[np.ndarray, ...]] = deque()
        self._errors: Deque[np.ndarray] = deque()
        self._B = np.zeros((self.max_vectors, self.max_vectors), dtype=np.float64)

    def clear(self) -> None:
        self._focks.clear()
        self._errors.clear()
        self._B.fill(0.0)

    def add(self, fock_tuple: Tuple[np.ndarray, ...], err_vec: np.ndarray) -> None:
        err_flat = np.asarray(err_vec, dtype=np.float64).ravel()
        if len(self._focks) >= self.max_vectors:
            self._focks.popleft()
            self._errors.popleft()
            m = self.max_vectors - 1
            self._B[:m, :m] = self._B[1 : m + 1, 1 : m + 1]

        self._focks.append(tuple(np.array(f, dtype=np.float64, copy=True) for f in fock_tuple))
        self._errors.append(err_flat.copy())

        idx = len(self._errors) - 1
        for i in range(idx + 1):
            val = float(np.dot(self._errors[i], err_flat))
            self._B[i, idx] = val
            self._B[idx, i] = val

    def extrapolate(self) -> Tuple[np.ndarray, ...]:
        n = len(self._focks)
        if n < 2:
            return self._focks[-1]

        B_sub = self._B[:n, :n]
        b_scale = float(np.max(np.abs(B_sub)))
        if b_scale <= 0.0:
            return self._focks[-1]

        A = np.zeros((n + 1, n + 1), dtype=np.float64)
        A[:n, :n] = B_sub / b_scale
        A[:n, n] = -1.0
        A[n, :n] = -1.0

        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        tol = 1.0e-12 * max(1.0, float(s[0]))
        s_inv = np.where(s > tol, 1.0 / s, 0.0)
        rhs = np.zeros(n + 1, dtype=np.float64)
        rhs[n] = -1.0
        coeffs = (Vt.T * s_inv) @ (U.T @ rhs)

        n_mats = len(self._focks[0])
        out = []
        for m_idx in range(n_mats):
            acc = np.zeros_like(self._focks[0][m_idx])
            for k in range(n):
                acc += coeffs[k] * self._focks[k][m_idx]
            out.append(acc)
        return tuple(out)


def _symmetric_orthogonalizer(S: np.ndarray, lindep_tol: float = 1.0e-8) -> np.ndarray:
    evals, evecs = eigh(S)
    keep = evals > lindep_tol
    if not np.any(keep):
        raise ValueError("Matriks overlap S singular atau tidak positif definit.")
    inv_sqrt = 1.0 / np.sqrt(evals[keep])
    U = evecs[:, keep]
    return (U * inv_sqrt[None, :]) @ U.T


def _diagonalize_fock(
    F_ao: np.ndarray,
    X: np.ndarray,
    C_prev: Optional[np.ndarray] = None,
    n_occ: int = 0,
    level_shift: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if level_shift > 0.0 and C_prev is not None and 0 < n_occ < F_ao.shape[0]:
        F_mo = C_prev.T @ F_ao @ C_prev
        idx_v = np.arange(n_occ, F_ao.shape[0])
        F_mo[idx_v, idx_v] += level_shift
        eps_s, U = eigh(F_mo)
        C_new = C_prev @ U
        eps = eps_s.copy()
        eps[n_occ:] -= level_shift
        return eps, C_new

    eps, C_orth = eigh(X.T @ F_ao @ X)
    return eps, X @ C_orth


def run_df_rhf(
    H_core: np.ndarray,
    S_ovlp: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    n_occ: int,
    energy_tol: float = 1.0e-9,
    density_tol: float = 1.0e-9,
    max_iter: int = 100,
    diis_space: int = 8,
    level_shift: float = 0.0,
    C_init: Optional[np.ndarray] = None,
    P_init: Optional[np.ndarray] = None,
) -> HybridSCFResult:
    """Menjalankan DF/Cholesky-RHF di Python menggunakan kernel C++20 `build_jk_3c_rhf`."""
    H = np.ascontiguousarray(H_core, dtype=np.float64)
    S = np.ascontiguousarray(S_ovlp, dtype=np.float64)
    B = np.ascontiguousarray(B_ao, dtype=np.float64)
    nbf = H.shape[0]
    X = _symmetric_orthogonalizer(S)

    if C_init is not None:
        C = np.array(C_init, dtype=np.float64, copy=True)
        eps = np.zeros(nbf, dtype=np.float64)
    elif P_init is not None:
        P0 = np.ascontiguousarray(P_init, dtype=np.float64)
        C_sad = _factorize_psd_density(0.5 * P0)
        J0 = K.build_coulomb_3c(B, P0, nbf)
        K0 = K.build_exchange_3c_occ(B, C_sad, nbf)
        eps, C = _diagonalize_fock(H + J0 - K0, X)
    else:
        eps, C = _diagonalize_fock(H, X)

    diis = PulayDIIS(max_vectors=diis_space)
    e_prev = 0.0
    P_prev = np.zeros((nbf, nbf), dtype=np.float64)
    converged = False
    e_tot = 0.0
    it_used = 0

    for it in range(1, max_iter + 1):
        it_used = it
        C_occ = np.ascontiguousarray(C[:, :n_occ])
        J, K_mat = K.build_jk_3c_rhf(B, C_occ, nbf)
        F = H + J - K_mat
        P = 2.0 * (C_occ @ C_occ.T)
        e_tot = float(0.5 * np.sum(P * (H + F)) + v_nn)

        grad_ao = F @ P @ S - S @ P @ F
        grad_orth = X.T @ grad_ao @ X
        grad_norm = float(np.linalg.norm(grad_orth))
        dE = abs(e_tot - e_prev)
        dP_rms = float(np.sqrt(np.mean((P - P_prev) ** 2)))

        if it > 1 and dE <= energy_tol and dP_rms <= density_tol and grad_norm <= max(1.0e-6, density_tol * 1.0e3):
            eps, C = _diagonalize_fock(F, X)
            converged = True
            break

        diis.add((F,), grad_orth)
        (F_ext,) = diis.extrapolate()
        eps, C = _diagonalize_fock(F_ext, X, C_prev=C, n_occ=n_occ, level_shift=level_shift)
        e_prev = e_tot
        P_prev = P

    return HybridSCFResult(
        reference="rhf",
        energy_total=e_tot,
        converged=converged,
        iterations=it_used,
        C_alpha=C,
        C_beta=C.copy(),
        orbital_energies_alpha=eps,
        orbital_energies_beta=eps.copy(),
        n_occ_alpha=n_occ,
        n_occ_beta=n_occ,
        s2=0.0,
        L_mat=B,
    )


def run_df_uhf(
    H_core: np.ndarray,
    S_ovlp: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    na: int,
    nb: int,
    energy_tol: float = 1.0e-9,
    density_tol: float = 1.0e-9,
    max_iter: int = 120,
    diis_space: int = 8,
    level_shift: float = 0.0,
    C_init_a: Optional[np.ndarray] = None,
    C_init_b: Optional[np.ndarray] = None,
    P_init: Optional[np.ndarray] = None,
    mix_homo_lumo: bool = False,
) -> HybridSCFResult:
    """Menjalankan DF/Cholesky-UHF di Python menggunakan kernel C++20 `build_jk_3c_uhf`."""
    H = np.ascontiguousarray(H_core, dtype=np.float64)
    S = np.ascontiguousarray(S_ovlp, dtype=np.float64)
    B = np.ascontiguousarray(B_ao, dtype=np.float64)
    nbf = H.shape[0]
    X = _symmetric_orthogonalizer(S)

    if C_init_a is not None and C_init_b is not None:
        Ca = np.array(C_init_a, dtype=np.float64, copy=True)
        Cb = np.array(C_init_b, dtype=np.float64, copy=True)
        eps_a = np.zeros(nbf, dtype=np.float64)
        eps_b = np.zeros(nbf, dtype=np.float64)
    elif P_init is not None:
        P0 = np.ascontiguousarray(P_init, dtype=np.float64)
        n_tot = max(1, na + nb)
        fa_scale = float(na) / float(n_tot)
        fb_scale = float(nb) / float(n_tot)
        J0 = K.build_coulomb_3c(B, P0, nbf)
        Ca_fac = _factorize_psd_density(fa_scale * P0)
        Cb_fac = _factorize_psd_density(fb_scale * P0)
        Ka0 = K.build_exchange_3c_occ(B, Ca_fac, nbf)
        Kb0 = K.build_exchange_3c_occ(B, Cb_fac, nbf) if nb > 0 else np.zeros((nbf, nbf), dtype=np.float64)
        eps_a, Ca = _diagonalize_fock(H + J0 - Ka0, X)
        eps_b, Cb = _diagonalize_fock(H + J0 - Kb0, X)
    else:
        eps_a, Ca = _diagonalize_fock(H, X)
        eps_b, Cb = eps_a.copy(), Ca.copy()
        if mix_homo_lumo and na == nb and 0 < na < nbf:
            theta = 0.25 * np.pi
            c, s_val = np.cos(theta), np.sin(theta)
            h_col = Ca[:, na - 1].copy()
            l_col = Ca[:, na].copy()
            Ca[:, na - 1] = c * h_col + s_val * l_col
            Ca[:, na] = -s_val * h_col + c * l_col

    diis = PulayDIIS(max_vectors=diis_space)
    e_prev = 0.0
    Pa_prev = np.zeros((nbf, nbf), dtype=np.float64)
    Pb_prev = np.zeros((nbf, nbf), dtype=np.float64)
    converged = False
    e_tot = 0.0
    it_used = 0

    for it in range(1, max_iter + 1):
        it_used = it
        Ca_o = np.ascontiguousarray(Ca[:, :na])
        Cb_o = np.ascontiguousarray(Cb[:, :nb])
        J, Ka, Kb = K.build_jk_3c_uhf(B, Ca_o, Cb_o, nbf)
        Fa = H + J - Ka
        Fb = H + J - Kb

        Pa = Ca_o @ Ca_o.T if na > 0 else np.zeros((nbf, nbf), dtype=np.float64)
        Pb = Cb_o @ Cb_o.T if nb > 0 else np.zeros((nbf, nbf), dtype=np.float64)
        e_tot = float(0.5 * (np.sum(Pa * (H + Fa)) + np.sum(Pb * (H + Fb))) + v_nn)

        ga_orth = X.T @ (Fa @ Pa @ S - S @ Pa @ Fa) @ X
        gb_orth = X.T @ (Fb @ Pb @ S - S @ Pb @ Fb) @ X
        err_stack = np.concatenate((ga_orth.ravel(), gb_orth.ravel()))
        grad_norm = float(np.linalg.norm(err_stack))

        dE = abs(e_tot - e_prev)
        dP_rms = float(
            np.sqrt(0.5 * (np.mean((Pa - Pa_prev) ** 2) + np.mean((Pb - Pb_prev) ** 2)))
        )

        if it > 1 and dE <= energy_tol and dP_rms <= density_tol and grad_norm <= max(1.0e-6, density_tol * 1.0e3):
            eps_a, Ca = _diagonalize_fock(Fa, X)
            eps_b, Cb = _diagonalize_fock(Fb, X)
            converged = True
            break

        diis.add((Fa, Fb), err_stack)
        Fa_ext, Fb_ext = diis.extrapolate()
        eps_a, Ca = _diagonalize_fock(Fa_ext, X, C_prev=Ca, n_occ=na, level_shift=level_shift)
        eps_b, Cb = _diagonalize_fock(Fb_ext, X, C_prev=Cb, n_occ=nb, level_shift=level_shift)
        e_prev = e_tot
        Pa_prev, Pb_prev = Pa, Pb

    sz = 0.5 * (na - nb)
    if nb > 0 and na > 0:
        ovlp = Ca[:, :na].T @ S @ Cb[:, :nb]
        s2 = float(sz * (sz + 1.0) + nb - np.sum(ovlp ** 2))
    else:
        s2 = float(sz * (sz + 1.0))

    return HybridSCFResult(
        reference="uhf",
        energy_total=e_tot,
        converged=converged,
        iterations=it_used,
        C_alpha=Ca,
        C_beta=Cb,
        orbital_energies_alpha=eps_a,
        orbital_energies_beta=eps_b,
        n_occ_alpha=na,
        n_occ_beta=nb,
        s2=s2,
        L_mat=B,
    )
