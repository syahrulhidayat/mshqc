# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
Plugin hook generik (RHF & UHF) untuk merakit metode Orbital-Optimized kustom
di atas C++20 `mshqc.kernels` + TR-SOSCF Steihaug-Toint.
(Didelegasikan langsung ke driver terpadu di `omp_hybrid.py` tanpa duplikasi loop.)
"""

from .omp_hybrid import (
    CorrCallbackRHF,
    CorrCallbackUHF,
    optimize_orbitals_rhf,
    optimize_orbitals_uhf,
)

__all__ = [
    "CorrCallbackRHF",
    "CorrCallbackUHF",
    "optimize_orbitals_rhf",
    "optimize_orbitals_uhf",
]
