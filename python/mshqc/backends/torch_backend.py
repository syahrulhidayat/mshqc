# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
Backend akselerasi GPU CUDA (FP64) berbasis PyTorch untuk mshqc v2.0 (0 baris CUDA C++).
Fitur inti:
  - FP64 ketat (torch.float64) untuk menjaga presisi kimia kuantum (< 1e-11 Ha vs C++20)
  - VRAM Cache otomatis untuk faktor 3-indeks B_ao agar tidak diunggah ulang di tiap iterasi
  - Akselerasi penuh RHF/UHF JK, MO 3-indeks, RMP2, O(N^4) LT-SOS-OMP2, dan OMP3 VVVV ladder/Z-vec
  - Auto-fallback transparan ke kernel C++20 (_fallback_to_cpp=True) untuk kernel kecil/khusus
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Tuple
import numpy as np

from .._mshqc import kernels as _cpp_kernels


class TorchKernelBackend:
    """Backend akselerasi PyTorch (CUDA FP64 / CPU) untuk primitif GEMM inti mshqc.kernels."""

    def __init__(
        self,
        device: str = "auto",
        fallback_to_cpp: bool = True,
        max_cache_items: int = 4,
        max_chunk_mb: int = 256,
    ) -> None:
        import torch

        self._torch = torch
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self._fallback_to_cpp = bool(fallback_to_cpp)
        self._max_cache_items = max(1, int(max_cache_items))
        self._max_chunk_bytes = max(32, int(max_chunk_mb)) * 1024 * 1024
        self._b3_cache: OrderedDict[Tuple[int, Tuple[int, ...], int], object] = OrderedDict()
        self._tri_idx_cache: dict[int, Tuple[object, object]] = {}

    def clear_cache(self) -> None:
        self._b3_cache.clear()
        if self.device.type == "cuda":
            self._torch.cuda.empty_cache()

    def _to_t(self, arr: np.ndarray):
        return self._torch.as_tensor(
            np.ascontiguousarray(arr, dtype=np.float64),
            dtype=self._torch.float64,
            device=self.device,
        )

    def _to_np(self, tensor) -> np.ndarray:
        return np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float64)

    def _get_b3_gpu(self, L_mat: np.ndarray, nbf: int):
        """Unpack & cache B3 berdimensi (n_aux, nbf, nbf) di VRAM berdasarkan pointer+shape."""
        torch = self._torch
        arr = np.asarray(L_mat, dtype=np.float64)
        key = (int(arr.__array_interface__["data"][0]), arr.shape, int(nbf))
        if key in self._b3_cache:
            self._b3_cache.move_to_end(key)
            return self._b3_cache[key]

        n_aux = arr.shape[1]
        L_t = self._to_t(arr)
        if arr.shape[0] == nbf * nbf:
            B3 = L_t.T.reshape(n_aux, nbf, nbf).contiguous()
        else:
            if nbf not in self._tri_idx_cache:
                rows, cols = np.tril_indices(nbf)
                self._tri_idx_cache[nbf] = (
                    torch.as_tensor(rows, dtype=torch.long, device=self.device),
                    torch.as_tensor(cols, dtype=torch.long, device=self.device),
                )
            rows_t, cols_t = self._tri_idx_cache[nbf]
            B3 = torch.zeros((n_aux, nbf, nbf), dtype=torch.float64, device=self.device)
            Lt = L_t.T  # (n_aux, n_tri)
            B3[:, rows_t, cols_t] = Lt
            B3[:, cols_t, rows_t] = Lt

        self._b3_cache[key] = B3
        while len(self._b3_cache) > self._max_cache_items:
            self._b3_cache.popitem(last=False)
        return B3

    @staticmethod
    def is_packed_ao(L_mat: np.ndarray, nbf: int) -> bool:
        return nbf > 1 and L_mat.shape[0] == (nbf * (nbf + 1)) // 2

    def build_coulomb_3c(self, L_mat: np.ndarray, P_tot: np.ndarray, nbf: int) -> np.ndarray:
        B3 = self._get_b3_gpu(L_mat, nbf)
        P_t = self._to_t(P_tot).reshape(nbf * nbf)
        B_flat = B3.reshape(B3.shape[0], nbf * nbf)
        c = B_flat @ P_t
        J = (c @ B_flat).reshape(nbf, nbf)
        return self._to_np(0.5 * (J + J.T))

    def build_exchange_3c_occ(self, L_mat: np.ndarray, C_occ: np.ndarray, nbf: int) -> np.ndarray:
        torch = self._torch
        if C_occ.shape[1] == 0:
            return np.zeros((nbf, nbf), dtype=np.float64)
        B3 = self._get_b3_gpu(L_mat, nbf)
        C_t = self._to_t(C_occ)
        T = torch.matmul(B3, C_t)
        T_mat = T.permute(1, 0, 2).reshape(nbf, -1)
        K = T_mat @ T_mat.T
        return self._to_np(0.5 * (K + K.T))

    def build_jk_3c_rhf(self, L_mat: np.ndarray, C_occ: np.ndarray, nbf: int) -> Tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        B3 = self._get_b3_gpu(L_mat, nbf)
        C_t = self._to_t(C_occ)
        P_t = 2.0 * (C_t @ C_t.T)
        B_flat = B3.reshape(B3.shape[0], nbf * nbf)
        J = ((B_flat @ P_t.reshape(-1)) @ B_flat).reshape(nbf, nbf)
        T = torch.matmul(B3, C_t).permute(1, 0, 2).reshape(nbf, -1)
        K = T @ T.T
        return self._to_np(0.5 * (J + J.T)), self._to_np(0.5 * (K + K.T))

    def build_jk_3c_uhf(
        self, L_mat: np.ndarray, C_occ_a: np.ndarray, C_occ_b: np.ndarray, nbf: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        torch = self._torch
        B3 = self._get_b3_gpu(L_mat, nbf)
        Ca = self._to_t(C_occ_a)
        Cb = self._to_t(C_occ_b)
        P_tot = torch.zeros((nbf, nbf), dtype=torch.float64, device=self.device)
        if Ca.shape[1] > 0:
            P_tot = P_tot + Ca @ Ca.T
        if Cb.shape[1] > 0:
            P_tot = P_tot + Cb @ Cb.T
        B_flat = B3.reshape(B3.shape[0], nbf * nbf)
        J = ((B_flat @ P_tot.reshape(-1)) @ B_flat).reshape(nbf, nbf)
        if Ca.shape[1] > 0:
            Ta = torch.matmul(B3, Ca).permute(1, 0, 2).reshape(nbf, -1)
            Ka = Ta @ Ta.T
        else:
            Ka = torch.zeros((nbf, nbf), dtype=torch.float64, device=self.device)
        if Cb.shape[1] > 0:
            Tb = torch.matmul(B3, Cb).permute(1, 0, 2).reshape(nbf, -1)
            Kb = Tb @ Tb.T
        else:
            Kb = torch.zeros((nbf, nbf), dtype=torch.float64, device=self.device)
        return self._to_np(0.5 * (J + J.T)), self._to_np(0.5 * (Ka + Ka.T)), self._to_np(0.5 * (Kb + Kb.T))

    def transform_3c_mo(self, L_mat: np.ndarray, C1: np.ndarray, C2: np.ndarray, nbf: int) -> np.ndarray:
        torch = self._torch
        B3 = self._get_b3_gpu(L_mat, nbf)
        C1_t = self._to_t(C1)
        C2_t = self._to_t(C2)
        half = torch.matmul(B3, C2_t)
        B_mo = torch.matmul(C1_t.T.unsqueeze(0), half).permute(1, 2, 0).reshape(
            C1.shape[1] * C2.shape[1], B3.shape[0]
        )
        return self._to_np(B_mo)

    def transform_3c_mo_blocks(
        self, L_mat: np.ndarray, C_occ: np.ndarray, C_vir: np.ndarray, nbf: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.transform_3c_mo(L_mat, C_occ, C_occ, nbf),
            self.transform_3c_mo(L_mat, C_occ, C_vir, nbf),
            self.transform_3c_mo(L_mat, C_vir, C_vir, nbf),
        )

    def eval_rmp2_df(
        self, B_ov: np.ndarray, eps_occ: np.ndarray, eps_vir: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        torch = self._torch
        no, nv = eps_occ.size, eps_vir.size
        B_t = self._to_t(B_ov)
        eo = self._to_t(eps_occ)
        ev = self._to_t(eps_vir)
        V = (B_t @ B_t.T).reshape(no, nv, no, nv).permute(0, 2, 1, 3)
        D = (
            eo[:, None, None, None]
            + eo[None, :, None, None]
            - ev[None, None, :, None]
            - ev[None, None, None, :]
        )
        T2 = V / D
        e_os = float(torch.sum(T2 * V).item())
        e_ss = float(torch.sum(T2 * (V - V.permute(0, 1, 3, 2))).item())
        T2_mat = T2.permute(3, 2, 0, 1).reshape(nv * nv, no * no)
        return e_os, e_ss, self._to_np(T2_mat)

    def eval_pp_ladder_rhf_df(
        self, T2: np.ndarray, B_vv: np.ndarray, n_occ: int, n_vir: int
    ) -> np.ndarray:
        """
        Adaptive 2-path O(V^4)-memory-free particle-particle ladder W_pp di GPU FP64:
        W_{ij}^{ab} = sum_{P, e, f} B^P_{ae} T_{ij}^{ef} B^P_{bf} (batas VRAM <= max_chunk_mb).
        """
        torch = self._torch
        no, nv = int(n_occ), int(n_vir)
        n_aux = int(B_vv.shape[1])
        if no <= 0 or nv <= 0 or n_aux <= 0:
            return np.zeros((nv * nv, no * no), dtype=np.float64)

        T2_t = self._to_t(T2)
        B_vv_t = self._to_t(B_vv)  # (nv*nv, n_aux), row = a*nv + e
        T4 = T2_t.reshape(nv, nv, no, no).permute(2, 3, 1, 0).contiguous()  # (no, no, nv, nv)

        idx_i, idx_j = torch.triu_indices(no, no, device=self.device)
        n_pairs = int(idx_i.numel())
        T_pairs = T4[idx_i, idx_j].contiguous()  # (n_pairs, nv, nv)
        W_pairs = torch.zeros((n_pairs, nv, nv), dtype=torch.float64, device=self.device)

        # Pilih jalur FLOPs terkecil antara Batched-A cuBLAS DGEMM vs P-loop cuBLAS DGEMM
        if nv * (n_aux + n_pairs) < 2 * n_aux * n_pairs:
            B_ae = B_vv_t.reshape(nv, nv, n_aux)
            T_pack = T_pairs.reshape(n_pairs, nv * nv).T.contiguous()  # (nv*nv, n_pairs)
            slice_bytes = max(1, nv * nv * nv * 8)
            ba = max(1, min(nv, int(self._max_chunk_bytes // slice_bytes)))

            for a0 in range(0, nv, ba):
                a1 = min(nv, a0 + ba)
                cur_ba = a1 - a0
                V_chunk = (B_ae[a0:a1].reshape(cur_ba * nv, n_aux) @ B_vv_t.T).reshape(cur_ba, nv, nv, nv)
                V_ab_ef = V_chunk.permute(0, 2, 1, 3).reshape(cur_ba * nv, nv * nv)
                W_chunk = (V_ab_ef @ T_pack).reshape(cur_ba, nv, n_pairs)
                W_pairs[:, a0:a1, :] = W_chunk.permute(2, 0, 1)
        else:
            B3 = B_vv_t.T.reshape(n_aux, nv, nv).contiguous()  # (n_aux, a, e)
            T_flat = T_pairs.reshape(n_pairs * nv, nv)
            for p_idx in range(n_aux):
                BP = B3[p_idx]  # (nv, nv)
                X = (T_flat @ BP.T).reshape(n_pairs, nv, nv)  # (k, e, b)
                W_step = (BP @ X.permute(1, 0, 2).reshape(nv, n_pairs * nv)).reshape(nv, n_pairs, nv)
                W_pairs.add_(W_step.permute(1, 0, 2))

        W4 = torch.zeros((no, no, nv, nv), dtype=torch.float64, device=self.device)
        W4[idx_i, idx_j] = W_pairs
        off = idx_i != idx_j
        if bool(torch.any(off)):
            W4[idx_j[off], idx_i[off]] = W_pairs[off].transpose(-1, -2)

        W_mat = W4.permute(3, 2, 0, 1).reshape(nv * nv, no * no)
        return self._to_np(W_mat)

    def eval_rmp3_df(
        self,
        T2: np.ndarray,
        B_oo: np.ndarray,
        B_ov: np.ndarray,
        B_vv: np.ndarray,
        eps_occ: np.ndarray,
        eps_vir: np.ndarray,
        W_pp_override: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """Hitung PP-ladder O(O^2 V^4) di GPU FP64 lalu teruskan ke C++ untuk suku ringan."""
        if W_pp_override is None or np.asarray(W_pp_override).size == 0:
            W_pp = self.eval_pp_ladder_rhf_df(T2, B_vv, int(eps_occ.size), int(eps_vir.size))
        else:
            W_pp = np.ascontiguousarray(W_pp_override, dtype=np.float64)
        return _cpp_kernels.eval_rmp3_df(T2, B_oo, B_ov, B_vv, eps_occ, eps_vir, W_pp)

    def eval_omp3_zvec_rhf_df(
        self,
        T2: np.ndarray,
        L2: np.ndarray,
        B_oo: np.ndarray,
        B_ov: np.ndarray,
        B_vv: np.ndarray,
        n_occ: int,
        n_vir: int,
        include_vvvv: bool = True,
    ) -> np.ndarray:
        """
        Hitung kontribusi dominan VVVV Z-vector OMP3 di GPU FP64 (chunked <= max_chunk_mb)
        dan gabungkan dengan suku OOOO/OVOV/OOVV dari C++20.
        """
        no, nv = int(n_occ), int(n_vir)
        Z_base = _cpp_kernels.eval_omp3_zvec_rhf_df(T2, L2, B_oo, B_ov, B_vv, no, nv, False)
        if not include_vvvv or no <= 0 or nv <= 0 or B_ov.shape[1] == 0:
            return Z_base

        torch = self._torch
        T2_t = self._to_t(T2)
        B_ov_t = self._to_t(B_ov)  # (no*nv, n_aux)
        B_vv_t = self._to_t(B_vv)  # (nv*nv, n_aux)

        T4 = T2_t.reshape(nv, nv, no, no).permute(2, 3, 1, 0).contiguous()  # (i, j, a, b)
        T2t = (2.0 * T4 - T4.transpose(2, 3)).reshape(no * no, nv * nv).contiguous()
        T2_x = T4.permute(2, 0, 1, 3).reshape(nv, no * no * nv).contiguous()

        slice_bytes = max(1, nv * nv * nv * 8)
        mb_max = max(1, min(no, int(self._max_chunk_bytes // slice_bytes)))
        Z_vvvv = torch.zeros((nv, no), dtype=torch.float64, device=self.device)

        for m0 in range(0, no, mb_max):
            mb = min(mb_max, no - m0)
            B_chunk = B_ov_t[m0 * nv : (m0 + mb) * nv]
            V_chunk = (B_chunk @ B_vv_t.T).reshape(mb, nv, nv, nv)  # (m, a, f, b)
            V_ab_mf = V_chunk.permute(1, 3, 0, 2).reshape(nv * nv, mb * nv)
            Y_ij_mf = (T2t @ V_ab_mf).reshape(no * no, mb, nv)
            Y_ijf_m = Y_ij_mf.permute(0, 2, 1).reshape(no * no * nv, mb)
            Z_vvvv[:, m0 : m0 + mb] = -(T2_x @ Y_ijf_m)

        return np.ascontiguousarray(Z_base + self._to_np(Z_vvvv), dtype=np.float64)

    def eval_lt_sos_intermediates_rhf_df(
        self,
        B_ov: np.ndarray,
        d_ov: np.ndarray,
        weights: np.ndarray,
        nodes: np.ndarray,
        c_os: float = 1.3,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """O(N^4) zero-T2 Laplace-Transformed SOS-OMP2 RHF sepenuhnya di GPU (FP64 cuBLAS)."""
        torch = self._torch
        B_t = self._to_t(B_ov)
        d_t = self._to_t(d_ov)
        w_arr = np.asarray(weights, dtype=np.float64)
        t_arr = np.asarray(nodes, dtype=np.float64)

        e_sum = 0.0
        X_conn = torch.zeros_like(B_t)
        X_deriv = torch.zeros_like(B_t)

        for wq, tq in zip(w_arr, t_arr):
            eh = torch.exp((-0.5 * float(tq)) * d_t).unsqueeze(1)
            Bw = B_t * eh
            Xq = Bw.T @ Bw
            e_sum += float(wq) * float(torch.sum(Xq * Xq).item())
            Yq = (Bw @ Xq) * eh
            X_conn.add_(Yq, alpha=-float(c_os * wq))
            X_deriv.add_(Yq, alpha=-float(c_os * wq * tq))

        return -float(c_os) * e_sum, self._to_np(X_conn), self._to_np(X_deriv)

    def eval_lt_sos_intermediates_uhf_df(
        self,
        B_ov_a: np.ndarray,
        B_ov_b: np.ndarray,
        da: np.ndarray,
        db: np.ndarray,
        weights: np.ndarray,
        nodes: np.ndarray,
        c_os: float = 1.3,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """O(N^4) zero-T2 Laplace-Transformed SOS-OMP2 UHF sepenuhnya di GPU (FP64 cuBLAS)."""
        torch = self._torch
        Ba = self._to_t(B_ov_a)
        Bb = self._to_t(B_ov_b)
        da_t = self._to_t(da)
        db_t = self._to_t(db)
        w_arr = np.asarray(weights, dtype=np.float64)
        t_arr = np.asarray(nodes, dtype=np.float64)

        e_sum = 0.0
        Xca = torch.zeros_like(Ba)
        Xda = torch.zeros_like(Ba)
        Xcb = torch.zeros_like(Bb)
        Xdb = torch.zeros_like(Bb)

        for wq, tq in zip(w_arr, t_arr):
            ea = torch.exp((-0.5 * float(tq)) * da_t).unsqueeze(1)
            eb = torch.exp((-0.5 * float(tq)) * db_t).unsqueeze(1)
            Baw = Ba * ea
            Bbw = Bb * eb
            Xa = Baw.T @ Baw
            Xb = Bbw.T @ Bbw
            e_sum += float(wq) * float(torch.sum(Xa * Xb).item())
            Ya = (Baw @ Xb) * ea
            Yb = (Bbw @ Xa) * eb
            Xca.add_(Ya, alpha=-float(c_os * wq))
            Xda.add_(Ya, alpha=-float(c_os * wq * tq))
            Xcb.add_(Yb, alpha=-float(c_os * wq))
            Xdb.add_(Yb, alpha=-float(c_os * wq * tq))

        return (
            -float(c_os) * e_sum,
            self._to_np(Xca),
            self._to_np(Xda),
            self._to_np(Xcb),
            self._to_np(Xdb),
        )
