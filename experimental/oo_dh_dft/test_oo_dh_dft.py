#!/usr/bin/env python3
"""
Tes eksperimental OO-Double-Hybrid DFT (dikarantina dari rilis CPC v2.0.0).
Aktifkan kembali setelah suku respons kernel f_xc[P_corr] diimplementasikan
dan lulus uji beda hingga (finite difference <= 1e-7 Ha/Bohr).
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oo_dh_dft


def test_oo_double_hybrid_dft_if_pyscf_installed() -> None:
    pyscf = pytest.importorskip("pyscf")
    mol = pyscf.gto.M(
        atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
        basis="cc-pVDZ",
        verbose=0,
    )
    res_b2 = oo_dh_dft.run_oo_double_hybrid_rhf(
        mol, aux_basis="cc-pVDZ-RI", functional="oo-b2plyp", grad_tol=1e-5
    )
    res_dsd = oo_dh_dft.run_oo_double_hybrid_rhf(
        mol, aux_basis="cc-pVDZ-RI", functional="oo-dsd-pbep86", grad_tol=1e-5
    )
    assert res_b2.converged and res_dsd.converged
