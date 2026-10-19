#!/usr/bin/env python3
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

import pytest
import mshqc
import atexit

# Global list untuk menyimpan semua hasil pengujian
_TEST_RESULTS = []

def print_final_table():
    """Fungsi ini akan otomatis dipanggil di akhir eksekusi untuk mencetak tabel."""
    if not _TEST_RESULTS:
        return
        
    print("\n\n=======================================================================")
    print("                     FINAL TEST RESULTS SUMMARY")
    print("=======================================================================")
    print("-" * 65)
    print(f"| {'System':<8} | {'ERI Mode':<10} | {'Method':<8} | {'Total Energy (Ha)':<20} |")
    print("-" * 65)
    for r in _TEST_RESULTS:
        print(f"| {r[0]:<8} | {r[1]:<10} | {r[2]:<8} | {r[3]:<20.10f} |")
    print("-" * 65)

# Daftarkan fungsi print_final_table agar jalan otomatis saat script/pytest selesai
atexit.register(print_final_table)


@pytest.mark.parametrize("eri_mode", ["exact", "cholesky", "df"])
def test_closed_shell_h2o(eri_mode: str):
    """Test closed-shell H2O (RHF -> RMP2 -> OMP2 -> OMP3) with simplified bindings."""
    # 1. Initialize molecule
    mol = mshqc.Molecule(charge=0, multiplicity=1)
    mol.add_atom("O", 0.0000,  0.0000,  0.1173, units="angstrom")
    mol.add_atom("H", 0.0000,  0.7572, -0.4692, units="angstrom")
    mol.add_atom("H", 0.0000, -0.7572, -0.4692, units="angstrom")

    # 2. Load basis set and analytical integral engine
    basis     = mshqc.BasisSet("cc-pVTZ", mol)
    integrals = mshqc.IntegralEngine(mol, basis)

    # 3. Run RHF
    scf_conf = mshqc.SCFConfig(eri_method=eri_mode, threshold=1e-9)
    scf_res  = mshqc.RHF(mol, basis, integrals, scf_conf).compute()

    # 4. Run RMP2, OMP2, and OMP3 on the converged SCF reference
    mp2_conf = mshqc.MP2Config(eri_method=eri_mode, threshold=1e-9, gradient_threshold=1e-5)

    rmp2_res = mshqc.RMP2(mol, basis, integrals, scf_res, mp2_conf).compute()
    omp2_res = mshqc.OMP2(mol, basis, integrals, scf_res, mp2_conf).compute()
    omp3_res = mshqc.OMP3(mol, basis, integrals, scf_res, mp2_conf).compute()

    # Simpan hasil ke global list
    _TEST_RESULTS.append(("H2O", eri_mode.upper(), "RHF", scf_res.energy_total))
    _TEST_RESULTS.append(("H2O", eri_mode.upper(), "RMP2", rmp2_res.energy_total))
    _TEST_RESULTS.append(("H2O", eri_mode.upper(), "OMP2", omp2_res.energy_total))
    _TEST_RESULTS.append(("H2O", eri_mode.upper(), "OMP3", omp3_res.energy_total))

    print(f"\n[H2O | {eri_mode.upper()}]")
    print(f"RHF  Total Energy : {scf_res.energy_total:.10f} Ha")
    print(f"RMP2 Total Energy : {rmp2_res.energy_total:.10f} Ha (Corr: {rmp2_res.energy_mp2_corr:.10f} Ha)")
    print(f"OMP2 Total Energy : {omp2_res.energy_total:.10f} Ha")
    print(f"OMP3 Total Energy : {omp3_res.energy_total:.10f} Ha")

    assert scf_res.converged
    assert omp2_res.converged
    assert omp3_res.converged
    assert rmp2_res.energy_total < scf_res.energy_total
    assert omp2_res.energy_total < scf_res.energy_total
    assert omp3_res.energy_total < omp2_res.energy_total


@pytest.mark.parametrize("eri_mode", ["exact", "cholesky", "df"])
def test_open_shell_oxygen(eri_mode: str):
    """Test open-shell Oxygen (UHF -> UMP2 -> OMP2 -> OMP3) with simplified bindings."""
    # 1. Initialize triplet Oxygen atom (charge=0, multiplicity=3)
    mol = mshqc.Molecule(charge=0, multiplicity=3)
    mol.add_atom("O", 0.0000, 0.0000, 0.0000, units="angstrom")

    # 2. Load basis set and analytical integral engine
    basis     = mshqc.BasisSet("cc-pVTZ", mol)
    integrals = mshqc.IntegralEngine(mol, basis)

    # 3. Run UHF
    scf_conf = mshqc.SCFConfig(eri_method=eri_mode, threshold=1e-9)
    scf_res  = mshqc.UHF(mol, basis, integrals, scf_conf).compute()

    # 4. Run UMP2, OMP2, and OMP3 on the converged UHF reference
    mp2_conf = mshqc.MP2Config(eri_method=eri_mode, threshold=1e-9, gradient_threshold=1e-5)

    ump2_res = mshqc.UMP2(mol, basis, integrals, scf_res, mp2_conf).compute()
    omp2_res = mshqc.OMP2(mol, basis, integrals, scf_res, mp2_conf).compute()
    omp3_res = mshqc.OMP3(mol, basis, integrals, scf_res, mp2_conf).compute()

    # Simpan hasil ke global list
    _TEST_RESULTS.append(("Oxygen", eri_mode.upper(), "UHF", scf_res.energy_total))
    _TEST_RESULTS.append(("Oxygen", eri_mode.upper(), "UMP2", ump2_res.energy_total))
    _TEST_RESULTS.append(("Oxygen", eri_mode.upper(), "OMP2", omp2_res.energy_total))
    _TEST_RESULTS.append(("Oxygen", eri_mode.upper(), "OMP3", omp3_res.energy_total))

    print(f"\n[Oxygen | {eri_mode.upper()}]")
    print(f"UHF  Total Energy : {scf_res.energy_total:.10f} Ha")
    print(f"UMP2 Total Energy : {ump2_res.energy_total:.10f} Ha (Corr: {ump2_res.energy_mp2_corr:.10f} Ha)")
    print(f"OMP2 Total Energy : {omp2_res.energy_total:.10f} Ha")
    print(f"OMP3 Total Energy : {omp3_res.energy_total:.10f} Ha")

    assert scf_res.converged
    assert omp2_res.converged
    assert omp3_res.converged
    assert ump2_res.energy_total < scf_res.energy_total
    assert omp2_res.energy_total < scf_res.energy_total
    assert omp3_res.energy_total < omp2_res.energy_total


if __name__ == "__main__":
    for mode in ["exact", "cholesky", "df"]:
        test_closed_shell_h2o(mode)
        test_open_shell_oxygen(mode)