# Copyright 2026 Muhamad Syahrul Hidayat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Automated pytest validation suite for mshqc (JOSS submission).
Tests both the High-Level API (mshqc.run, MSHQCSession, quick_calculation)
and the Low-Level C++20 API across exact, Cholesky, and Density Fitting modes.
"""

import pytest
import numpy as np
import mshqc

H2O_XYZ = """
O  0.0000000000   0.0000000000   0.1173000000
H  0.0000000000   0.7572000000  -0.4692000000
H  0.0000000000  -0.7572000000  -0.4692000000
"""


def test_unit_conversions_and_molecule_builder():
    """Verify Bohr/Angstrom conversions and Molecule construction."""
    bohr = mshqc.angstrom_to_bohr(1.0)
    assert np.isclose(bohr, 1.8897261246, atol=1e-8)
    assert np.isclose(mshqc.bohr_to_angstrom(bohr), 1.0, atol=1e-10)

    mol1 = mshqc.build_molecule(H2O_XYZ, charge=0, multiplicity=1, units="angstrom")
    mol2 = mshqc.Molecule(charge=0, multiplicity=1)
    mol2.add_atom("O", 0.0,  0.0000,  0.1173, units="angstrom")
    mol2.add_atom("H", 0.0,  0.7572, -0.4692, units="angstrom")
    mol2.add_atom("H", 0.0, -0.7572, -0.4692, units="angstrom")

    assert mol1.n_atoms() == 3
    assert mol1.n_electrons() == 10
    assert np.isclose(mol1.nuclear_repulsion_energy(), mol2.nuclear_repulsion_energy(), atol=1e-12)


@pytest.mark.parametrize("eri_mode,expected", [
    ("exact",    (-128.53186164, -128.80915324, -128.81022310, -128.81027068)),
    ("cholesky", (-128.53186164, -128.80915324, -128.81022310, -128.81027068)),
    ("df",       (-128.53187686, -128.80915351, -128.81022440, -128.81042564)),
])
def test_neon_closed_shell_modes(eri_mode, expected):
    """Validate Neon (1S) RHF, RMP2, OMP2, and OMP3 against Psi4 reference values."""
    res = mshqc.run(
        "Ne 0.0 0.0 0.0",
        method=["RHF", "RMP2", "OMP2", "OMP3"],
        basis="cc-pVTZ",
        eri_method=eri_mode,
        charge=0,
        multiplicity=1,
        threshold=1e-9,
    )
    e_rhf, e_rmp2, e_omp2, e_omp3 = expected
    assert np.isclose(res["RHF"]["energy"],  e_rhf,  atol=1e-6)
    assert np.isclose(res["RMP2"]["energy"], e_rmp2, atol=1e-6)
    assert np.isclose(res["OMP2"]["energy"], e_omp2, atol=1e-6)
    assert np.isclose(res["OMP3"]["energy"], e_omp3, atol=1e-6)


@pytest.mark.parametrize("eri_mode,expected", [
    ("exact",    (-76.05712742, -76.33224440, -76.33406054, -76.33711056)),
    ("cholesky", (-76.05712737, -76.33224431, -76.33406044, -76.33711183)),
    ("df",       (-76.05711972, -76.33220605, -76.33402331, -76.33725847)),
])
def test_h2o_closed_shell_modes(eri_mode, expected):
    """Validate H2O (1A1) RHF, RMP2, OMP2, and OMP3 against Psi4 reference values."""
    res = mshqc.run(
        H2O_XYZ,
        method=["RHF", "RMP2", "OMP2", "OMP3"],
        basis="cc-pVTZ",
        eri_method=eri_mode,
        units="angstrom",
        charge=0,
        multiplicity=1,
        threshold=1e-9,
    )
    e_rhf, e_rmp2, e_omp2, e_omp3 = expected
    assert np.isclose(res["RHF"]["energy"],  e_rhf,  atol=5e-6)
    assert np.isclose(res["RMP2"]["energy"], e_rmp2, atol=5e-6)
    assert np.isclose(res["OMP2"]["energy"], e_omp2, atol=5e-6)
    assert np.isclose(res["OMP3"]["energy"], e_omp3, atol=5e-6)


@pytest.mark.parametrize("eri_mode,expected", [
    ("exact",    (-74.81175662, -74.96597192, -74.96657683, -74.98104071)),
    ("cholesky", (-74.81175662, -74.96597192, -74.96657683, -74.98104071)),
    ("df",       (-74.81174765, -74.96595721, -74.96656224, -74.98111725)),
])
def test_oxygen_open_shell_uhf_modes(eri_mode, expected):
    """Validate open-shell triplet Oxygen (3P, mult=3) UHF, UMP2, OMP2, and OMP3."""
    res = mshqc.run(
        "O 0.0 0.0 0.0",
        method=["UHF", "UMP2", "OMP2", "OMP3"],
        basis="cc-pVTZ",
        eri_method=eri_mode,
        charge=0,
        multiplicity=3,
        reference="uhf",
        threshold=1e-9,
    )
    e_uhf, e_ump2, e_omp2, e_omp3 = expected
    assert np.isclose(res["UHF"]["energy"],  e_uhf,  atol=5e-6)
    assert np.isclose(res["UMP2"]["energy"], e_ump2, atol=5e-6)
    assert np.isclose(res["OMP2"]["energy"], e_omp2, atol=5e-6)
    assert np.isclose(res["OMP3"]["energy"], e_omp3, atol=5e-6)


def test_low_level_cpp_bindings_and_cholesky():
    """Verify direct instantiation of low-level C++20 classes (Molecule, RHF, RMP2, OMP2, OMP3, CholeskyERI)."""
    raw_mol = mshqc.Molecule(charge=0, multiplicity=1)
    raw_mol.add_atom("O", 0.0,  0.0000,  0.1173, units="angstrom")
    raw_mol.add_atom("H", 0.0,  0.7572, -0.4692, units="angstrom")
    raw_mol.add_atom("H", 0.0, -0.7572, -0.4692, units="angstrom")

    pg = mshqc.PointGroup(raw_mol)
    pg.detect()
    mol = pg.get_aligned_molecule()

    basis = mshqc.BasisSet("cc-pVDZ", mol)
    ints  = mshqc.IntegralEngine(mol, basis)
    pl    = mshqc.PetiteList(basis, pg)
    pl.build()

    chol = mshqc.CholeskyERI(mol, basis, threshold=1e-8)
    assert chol.n_vectors() > 0

    scf_conf = mshqc.SCFConfig(eri_method="df", threshold=1e-9)
    scf_res  = mshqc.RHF(mol, basis, ints, scf_conf, pg=pg, pl=pl).compute()
    assert scf_res.converged

    mp2_conf = mshqc.MP2Config(eri_method="df", threshold=1e-9, gradient_threshold=1e-5)
    mp2_res  = mshqc.RMP2(mol, basis, ints, scf_res, mp2_conf, pg=pg, pl=pl).compute()
    omp2_res = mshqc.OMP2(mol, basis, ints, scf_res, mp2_conf, pg=pg, pl=pl).compute()
    omp3_res = mshqc.OMP3(mol, basis, ints, scf_res, mp2_conf, pg=pg, pl=pl).compute()

    assert mp2_res.energy_mp2_corr < -0.15
    assert omp2_res.converged
    assert omp3_res.converged
    assert omp3_res.energy_total < omp2_res.energy_total
