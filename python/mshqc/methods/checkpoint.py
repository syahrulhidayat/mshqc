# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
Checkpoint orbital ringan berbasis NumPy `.npz` (zero-HDF5 dependency) untuk
warm-start antar metode (SCF -> OMP2 -> kappa-OOMP2 -> OMP3) maupun antar sesi.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union
import numpy as np


def save_orbital_checkpoint(path: Union[str, Path], result: Any) -> Path:
    """Simpan matriks orbital (C_alpha, C_beta), energi orbital, dan metadata ke berkas `.npz`."""
    out_path = Path(path)
    if out_path.suffix != ".npz":
        out_path = out_path.with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    C_a = np.asarray(result.C_alpha, dtype=np.float64)
    C_b_raw = getattr(result, "C_beta", None)
    C_b = np.asarray(C_b_raw, dtype=np.float64) if (C_b_raw is not None and np.asarray(C_b_raw).size > 0) else C_a

    eps_a_raw = getattr(result, "orbital_energies_alpha", None)
    eps_b_raw = getattr(result, "orbital_energies_beta", None)
    eps_a = np.asarray(eps_a_raw, dtype=np.float64) if eps_a_raw is not None else np.zeros(C_a.shape[1])
    eps_b = np.asarray(eps_b_raw, dtype=np.float64) if eps_b_raw is not None else eps_a

    payload = {
        "C_alpha": C_a,
        "C_beta": C_b,
        "orbital_energies_alpha": eps_a,
        "orbital_energies_beta": eps_b,
        "n_occ_alpha": np.int64(getattr(result, "n_occ_alpha", 0)),
        "n_occ_beta": np.int64(getattr(result, "n_occ_beta", 0)),
        "energy_total": np.float64(getattr(result, "energy_total", getattr(result, "e_total", 0.0))),
        "reference": np.array(str(getattr(result, "reference", "rhf"))),
        "method": np.array(str(getattr(result, "method", getattr(result, "reference", "scf")))),
    }
    np.savez(out_path, **payload)
    return out_path


def load_orbital_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
    """Muat kembali checkpoint orbital `.npz` menjadi dictionary siap pakai untuk `C_init`."""
    in_path = Path(path)
    with np.load(in_path, allow_pickle=False) as data:
        return {
            "C_alpha": np.ascontiguousarray(data["C_alpha"], dtype=np.float64),
            "C_beta": np.ascontiguousarray(data["C_beta"], dtype=np.float64),
            "orbital_energies_alpha": np.asarray(data["orbital_energies_alpha"], dtype=np.float64),
            "orbital_energies_beta": np.asarray(data["orbital_energies_beta"], dtype=np.float64),
            "n_occ_alpha": int(data["n_occ_alpha"]),
            "n_occ_beta": int(data["n_occ_beta"]),
            "energy_total": float(data["energy_total"]),
            "reference": str(data["reference"].item()),
            "method": str(data["method"].item()),
        }
