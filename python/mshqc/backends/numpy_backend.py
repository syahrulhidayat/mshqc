# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""Backend referensi murni NumPy/SciPy yang mencerminkan kontrak `mshqc.kernels`."""

from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy.linalg import eigh


def _unpack_ao(L_mat: np.ndarray, nbf: int) -> np.ndarray:
    """Konversi L_mat packed (nbf*(nbf+1)//2, n_aux) atau full (nbf^2, n_aux) -> (n_aux, nbf, nbf)."""
    n_aux = L_mat.shape[1]
    if L_mat.shape[0] == nbf * nbf:
        return L_mat.T.reshape(n_aux, nbf, nbf)
    out = np.zeros((n_aux, nbf, nbf), dtype=np.float64)
    idx = 0
    for p in range(nbf):
        for q in range(p):
            col = L_mat[idx, :]
            out[:, p, q] = col
            out[:, q, p] = col
            idx += 1
        out[:, p, p] = L_mat[idx, :]
        idx += 1
    return out


class NumpyKernelBackend:
    """Implementasi referensi murni NumPy untuk kontrak primitif inti `mshqc.kernels`."""

    @staticmethod
    def is_packed_ao(L_mat: np.ndarray, nbf: int) -> bool:
        return nbf > 1 and L_mat.shape[0] == (nbf * (nbf + 1)) // 2

    @staticmethod
    def build_coulomb_3c(L_mat: np.ndarray, P_tot: np.ndarray, nbf: int) -> np.ndarray:
        B3 = _unpack_ao(L_mat, nbf)
        c = np.einsum("Pmn,mn->P", B3, P_tot, optimize=True)
        return np.einsum("Pmn,P->mn", B3, c, optimize=True)

    @staticmethod
    def build_exchange_3c_occ(L_mat: np.ndarray, C_occ: np.ndarray, nbf: int) -> np.ndarray:
        B3 = _unpack_ao(L_mat, nbf)
        T = np.einsum("Pmn,ni->Pmi", B3, C_occ, optimize=True)
        return np.einsum("Pmi,Pni->mn", T, T, optimize=True)

    @staticmethod
    def build_exchange_3c_diag(L_mat: np.ndarray, d_diag: np.ndarray, nbf: int) -> np.ndarray:
        B3 = _unpack_ao(L_mat, nbf)
        return np.einsum("Pmk,k,Pnk->mn", B3, d_diag, B3, optimize=True)

    @staticmethod
    def build_exchange_3c_sym(L_mat: np.ndarray, D_sym: np.ndarray, nbf: int) -> np.ndarray:
        B3 = _unpack_ao(L_mat, nbf)
        return np.einsum("Pmk,kl,Pnl->mn", B3, D_sym, B3, optimize=True)

    @classmethod
    def build_jk_3c_rhf(cls, L_mat: np.ndarray, C_occ: np.ndarray, nbf: int) -> Tuple[np.ndarray, np.ndarray]:
        P_tot = 2.0 * (C_occ @ C_occ.T)
        return cls.build_coulomb_3c(L_mat, P_tot, nbf), cls.build_exchange_3c_occ(L_mat, C_occ, nbf)

    @staticmethod
    def transform_3c_mo(L_mat: np.ndarray, C1: np.ndarray, C2: np.ndarray, nbf: int) -> np.ndarray:
        B3 = _unpack_ao(L_mat, nbf)
        B_mo = np.einsum("Pmn,mi,nj->ijP", B3, C1, C2, optimize=True)
        return np.ascontiguousarray(B_mo.reshape(C1.shape[1] * C2.shape[1], B3.shape[0]))

    @classmethod
    def transform_3c_mo_blocks(
        cls, L_mat: np.ndarray, C_occ: np.ndarray, C_vir: np.ndarray, nbf: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            cls.transform_3c_mo(L_mat, C_occ, C_occ, nbf),
            cls.transform_3c_mo(L_mat, C_occ, C_vir, nbf),
            cls.transform_3c_mo(L_mat, C_vir, C_vir, nbf),
        )

    @staticmethod
    def semicanonicalize_subspaces(
        C: np.ndarray, F_ao: np.ndarray, n_occ: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nbf = C.shape[0]
        C_o, C_v = C[:, :n_occ], C[:, n_occ:]
        eo, Uo = eigh(C_o.T @ F_ao @ C_o)
        ev, Uv = eigh(C_v.T @ F_ao @ C_v)
        C_sc = np.column_stack((C_o @ Uo, C_v @ Uv))
        eps = np.concatenate((eo, ev))
        F_vo = C_sc[:, n_occ:].T @ F_ao @ C_sc[:, :n_occ]
        return C_sc, eps, F_vo

    @staticmethod
    def eval_rmp2_df(
        B_ov: np.ndarray, eps_occ: np.ndarray, eps_vir: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        no, nv = eps_occ.size, eps_vir.size
        V = (B_ov @ B_ov.T).reshape(no, nv, no, nv).transpose(0, 2, 1, 3)
        D = eps_occ[:, None, None, None] + eps_occ[None, :, None, None] - eps_vir[None, None, :, None] - eps_vir[None, None, None, :]
        T2 = V / D
        e_os = float(np.sum(T2 * V))
        e_ss = float(np.sum(T2 * (V - V.transpose(0, 1, 3, 2))))
        T2_mat = np.ascontiguousarray(T2.transpose(3, 2, 0, 1).reshape(nv * nv, no * no))
        return e_os, e_ss, T2_mat
