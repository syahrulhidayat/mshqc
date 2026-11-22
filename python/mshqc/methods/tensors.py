# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""Konverter tata letak tensor 4D/3D (ramah np.einsum & PySCF) <-> matriks kernel C++20."""

import numpy as np


def t2_to_4d(T2_mat: np.ndarray, no1: int, no2: int, nv1: int, nv2: int) -> np.ndarray:
    """
    Konversi matriks amplitudo kernel (nv1*nv2, no1*no2) ke tensor 4D standar (no1, no2, nv1, nv2).
    Layout kernel: col = i*no2 + j, row = a + b*nv1 (column-major pada (a, b)).
    """
    arr = np.asarray(T2_mat, dtype=np.float64)
    if arr.shape != (nv1 * nv2, no1 * no2):
        raise ValueError(f"Bentuk T2_mat {arr.shape} tidak cocok dengan ({nv1*nv2}, {no1*no2})")
    # Reshape (nv2, nv1, no1, no2) karena row = a + b*nv1 -> sumbu b lebih lambat dari a
    return arr.reshape(nv2, nv1, no1, no2).transpose(2, 3, 1, 0)


def t2_from_4d(T2_4d: np.ndarray) -> np.ndarray:
    """Konversi tensor 4D standar (no1, no2, nv1, nv2) ke matriks kernel (nv1*nv2, no1*no2)."""
    t = np.asarray(T2_4d, dtype=np.float64)
    if t.ndim != 4:
        raise ValueError(f"Diharapkan tensor 4D (no1, no2, nv1, nv2), diperoleh ndim={t.ndim}")
    no1, no2, nv1, nv2 = t.shape
    return np.ascontiguousarray(t.transpose(3, 2, 0, 1).reshape(nv1 * nv2, no1 * no2))


def b3c_to_3d(B_mat: np.ndarray, d1: int, d2: int) -> np.ndarray:
    """Konversi faktor 3-indeks kernel (d1*d2, n_aux) ke tensor 3D (n_aux, d1, d2) gaya PySCF."""
    b = np.asarray(B_mat, dtype=np.float64)
    n_aux = b.shape[1]
    return b.reshape(d1, d2, n_aux).transpose(2, 0, 1)


def b3c_from_3d(B_3d: np.ndarray) -> np.ndarray:
    """Konversi tensor 3D (n_aux, d1, d2) ke matriks kernel (d1*d2, n_aux)."""
    b = np.asarray(B_3d, dtype=np.float64)
    n_aux, d1, d2 = b.shape
    return np.ascontiguousarray(b.transpose(1, 2, 0).reshape(d1 * d2, n_aux))
