# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
Keadaan tereksitasi elektron tunggal CIS & TDHF/RPA (RHF singlet & triplet) di atas
faktor 3-indeks Density Fitting `mshqc.kernels` (0 baris C++ baru).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.linalg import eigh

from ..backends import active_kernels as K  
from .omp_hybrid import semicanonicalize_subspaces_frozen

_HARTREE_TO_EV = 27.211386245988


@dataclass
class ExcitedStateResult:
    method: str
    spin: str
    n_states: int
    excitation_energies_ha: np.ndarray
    excitation_energies_ev: np.ndarray
    amplitudes_vo: np.ndarray
    converged: bool
    iterations: int


def _build_cis_and_b_blocks(
    B_oo: np.ndarray,
    B_ov: np.ndarray,
    B_vv: np.ndarray,
    eps_o: np.ndarray,
    eps_v: np.ndarray,
    triplet: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bangun blok matriks CIS (A) dan kopling de-eksitasi (B) berdimensi (nv*no x nv*no)."""
    no, nv = eps_o.size, eps_v.size
    n_aux = B_ov.shape[1]
    dim = nv * no

    # B_ov: (no*nv, n_aux) -> ubah urutan ke (nv, no, n_aux) agar cocok dengan vektor (nv, no)
    B_vo = B_ov.reshape(no, nv, n_aux).transpose(1, 0, 2).reshape(dim, n_aux)
    B_oo_3d = B_oo.reshape(no, no, n_aux)
    B_vv_3d = B_vv.reshape(nv, nv, n_aux)

    # Suku Coulomb singlet: 2 * (ai|bj) = 2 * sum_P B^P_{ai} B^P_{bj}
    J_ai_bj = (B_vo @ B_vo.T) if not triplet else np.zeros((dim, dim), dtype=np.float64)

    # Suku Exchange (ab|ji) = sum_P B^P_{ab} B^P_{ji}
    K_ab_ji = np.einsum("abP,jiP->aibj", B_vv_3d, B_oo_3d, optimize=True).reshape(dim, dim)

    delta_ai = (eps_v[:, None] - eps_o[None, :]).reshape(dim)
    A = np.diag(delta_ai) + 2.0 * J_ai_bj - K_ab_ji

    # Blok B_{ai,bj} = 2*(ai|jb) - (aj|bi)
    K_aj_bi = np.einsum("ajP,biP->aibj", B_ov.reshape(no, nv, n_aux).transpose(1, 0, 2),
                        B_ov.reshape(no, nv, n_aux).transpose(1, 0, 2), optimize=True).reshape(dim, dim)
    B = 2.0 * J_ai_bj - K_aj_bi
    return 0.5 * (A + A.T), 0.5 * (B + B.T)


def _davidson_hermitian(
    matvec,
    diag: np.ndarray,
    n_states: int,
    tol: float = 1.0e-7,
    max_iter: int = 60,
) -> Tuple[np.ndarray, np.ndarray, bool, int]:
    """Solver block-Davidson untuk k nilai eigen terendah dari operator simetris."""
    dim = diag.size
    k = min(int(n_states), dim)
    if k <= 0:
        return np.zeros(0), np.zeros((dim, 0)), True, 0

    # Jika dimensi ruang kecil, diagonalisasi langsung secara eksak
    if dim <= max(160, 6 * k):
        M_full = np.column_stack([matvec(np.eye(1, dim, j).ravel()) for j in range(dim)])
        evals, evecs = eigh(0.5 * (M_full + M_full.T))
        return evals[:k], evecs[:, :k], True, 1

    init_idx = np.argsort(diag)[: max(k + 2, min(dim, 2 * k))]
    V = np.zeros((dim, init_idx.size), dtype=np.float64)
    for col, idx in enumerate(init_idx):
        V[idx, col] = 1.0

    max_sub = min(dim, max(24, 8 * k))
    for it in range(1, max_iter + 1):
        V, _ = np.linalg.qr(V, mode="reduced")
        AV = np.column_stack([matvec(V[:, j]) for j in range(V.shape[1])])
        H_sub = 0.5 * (V.T @ AV + (V.T @ AV).T)
        evals, u_sub = eigh(H_sub)
        ritz_e = evals[:k]
        ritz_v = V @ u_sub[:, :k]
        res = AV @ u_sub[:, :k] - ritz_v * ritz_e[None, :]
        max_res = float(np.max(np.linalg.norm(res, axis=0)))
        if max_res < tol:
            return ritz_e, ritz_v, True, it

        new_vecs = []
        for j in range(k):
            if float(np.linalg.norm(res[:, j])) >= tol * 0.5:
                denom = diag - ritz_e[j]
                denom = np.where(np.abs(denom) < 1.0e-4, np.sign(denom + 1e-12) * 1.0e-4, denom)
                t = -res[:, j] / denom
                t -= V @ (V.T @ t)
                nrm = float(np.linalg.norm(t))
                if nrm > 1.0e-10:
                    new_vecs.append(t / nrm)

        if not new_vecs:
            return ritz_e, ritz_v, True, it
        if V.shape[1] + len(new_vecs) > max_sub:
            V = np.column_stack([ritz_v] + new_vecs)
        else:
            V = np.column_stack([V] + new_vecs)

    return ritz_e, ritz_v, False, max_iter


def run_df_cis_rhf(
    H_core: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    C0: np.ndarray,
    n_occ: int,
    n_states: int = 5,
    triplet: bool = False,
    n_frozen: int = 0,
    tol: float = 1.0e-7,
) -> ExcitedStateResult:
    """Hitung n_states eksitasi terendah DF-CIS (singlet atau triplet) dari referensi RHF."""
    del v_nn
    nbf = H_core.shape[0]
    nf = max(0, min(int(n_frozen), int(n_occ)))
    C_occ_all = np.ascontiguousarray(C0[:, :n_occ])
    J, K_mat = K.build_jk_3c_rhf(B_ao, C_occ_all, nbf)
    F_ao = H_core + J - K_mat
    C_sc, eps, _ = semicanonicalize_subspaces_frozen(C0, F_ao, n_occ, nf)

    C_act = np.ascontiguousarray(C_sc[:, nf:n_occ])
    C_vir = np.ascontiguousarray(C_sc[:, n_occ:])
    eps_o, eps_v = eps[nf:n_occ], eps[n_occ:]
    no_act, nv = eps_o.size, eps_v.size

    B_oo, B_ov, B_vv = K.transform_3c_mo_blocks(B_ao, C_act, C_vir, nbf)
    A, _ = _build_cis_and_b_blocks(B_oo, B_ov, B_vv, eps_o, eps_v, triplet=triplet)
    evals, evecs, conv, iters = _davidson_hermitian(lambda x: A @ x, np.diag(A), n_states, tol=tol)
    amps = evecs.T.reshape(-1, nv, no_act)
    return ExcitedStateResult(
        method="DF-CIS",
        spin="triplet" if triplet else "singlet",
        n_states=int(evals.size),
        excitation_energies_ha=evals,
        excitation_energies_ev=evals * _HARTREE_TO_EV,
        amplitudes_vo=amps,
        converged=conv,
        iterations=iters,
    )


def run_df_tdhf_rhf(
    H_core: np.ndarray,
    v_nn: float,
    B_ao: np.ndarray,
    C0: np.ndarray,
    n_occ: int,
    n_states: int = 5,
    triplet: bool = False,
    n_frozen: int = 0,
) -> ExcitedStateResult:
    """Hitung n_states eksitasi terendah DF-TDHF / RPA (singlet atau triplet) dari referensi RHF."""
    del v_nn
    nbf = H_core.shape[0]
    nf = max(0, min(int(n_frozen), int(n_occ)))
    C_occ_all = np.ascontiguousarray(C0[:, :n_occ])
    J, K_mat = K.build_jk_3c_rhf(B_ao, C_occ_all, nbf)
    F_ao = H_core + J - K_mat
    C_sc, eps, _ = semicanonicalize_subspaces_frozen(C0, F_ao, n_occ, nf)

    C_act = np.ascontiguousarray(C_sc[:, nf:n_occ])
    C_vir = np.ascontiguousarray(C_sc[:, n_occ:])
    eps_o, eps_v = eps[nf:n_occ], eps[n_occ:]
    no_act, nv = eps_o.size, eps_v.size

    B_oo, B_ov, B_vv = K.transform_3c_mo_blocks(B_ao, C_act, C_vir, nbf)
    A, B = _build_cis_and_b_blocks(B_oo, B_ov, B_vv, eps_o, eps_v, triplet=triplet)

    # Reduksi simetris M = (A - B)^{1/2} (A + B) (A - B)^{1/2}, nilai eigen = omega^2
    amb = A - B
    apb = A + B
    ev_m, U_m = eigh(amb)
    if float(np.min(ev_m)) <= 1.0e-10:
        raise RuntimeError("Referensi RHF tidak stabil terhadap instabilitas triplet/singlet (A - B tidak PD).")
    amb_half = (U_m * np.sqrt(ev_m)[None, :]) @ U_m.T
    amb_inv_half = (U_m * (1.0 / np.sqrt(ev_m))[None, :]) @ U_m.T

    M_sym = amb_half @ apb @ amb_half
    w2, Z = eigh(0.5 * (M_sym + M_sym.T))
    k = min(int(n_states), w2.size)
    omega = np.sqrt(np.maximum(w2[:k], 0.0))
    X_plus_Y = amb_half @ Z[:, :k]
    X_minus_Y = (apb @ X_plus_Y) / np.maximum(omega[None, :], 1.0e-12)
    X_vecs = 0.5 * (X_plus_Y + X_minus_Y)
    # Normalisasi RPA: <X|X> - <Y|Y> = 1
    Y_vecs = 0.5 * (X_plus_Y - X_minus_Y)
    norms = np.sqrt(np.maximum(np.sum(X_vecs ** 2 - Y_vecs ** 2, axis=0), 1.0e-14))
    X_vecs = X_vecs / norms[None, :]
    del amb_inv_half

    return ExcitedStateResult(
        method="DF-TDHF",
        spin="triplet" if triplet else "singlet",
        n_states=k,
        excitation_energies_ha=omega,
        excitation_energies_ev=omega * _HARTREE_TO_EV,
        amplitudes_vo=X_vecs.T.reshape(k, nv, no_act),
        converged=True,
        iterations=1,
    )
