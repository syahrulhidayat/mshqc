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
Utility Functions for MSHQC
File: python/mshqc/utils.py
"""

import time
from typing import List, Optional, Dict, Any, Union, Tuple
import mshqc
from .calculators import MSHQCCalculator, build_molecule
from .session import MSHQCSession


def _parse_method_and_eri(method: str, default_eri: str = "exact") -> Tuple[str, str]:
    """Deteksi awalan df-/cholesky- pada nama metode (misal 'df-omp3' atau 'chomp2')."""
    m = method.lower().strip()
    if m.startswith("df-") or m.startswith("df_"):
        return m[3:], "df"
    if m.startswith("ch-") or m.startswith("ch_"):
        return m[3:], "cholesky"
    if m.startswith("cholesky-") or m.startswith("cholesky_"):
        return m[9:], "cholesky"
    if m in ("chomp2", "chomp3"):
        return m[2:], "cholesky"
    return m, default_eri


def quick_calculation(element_or_mol: Union[str, int, List[Tuple], mshqc.Molecule],
                      basis: str = "cc-pVTZ",
                      method: str = "omp2",
                      eri_method: str = "exact",
                      aux_basis: Optional[str] = None,
                      charge: int = 0,
                      multiplicity: Optional[int] = None,
                      units: str = "bohr",
                      threshold: float = 1e-9,
                      basis_dir: Optional[str] = None,
                      print_level: int = 0) -> Any:
    """
    Perhitungan cepat untuk atom tunggal maupun molekul.

    Contoh:
        >>> res = quick_calculation("Ne", basis="cc-pVTZ", method="df-omp3")
        >>> print(res.energy_total)
    """
    clean_method, resolved_eri = _parse_method_and_eri(method, default_eri=eri_method)

    if isinstance(element_or_mol, int) or (
        isinstance(element_or_mol, str) and len(element_or_mol.split()) == 1
    ):
        mol_input = [(element_or_mol, 0.0, 0.0, 0.0)]
    else:
        mol_input = element_or_mol

    mol = build_molecule(mol_input, charge=charge, multiplicity=multiplicity, units=units)
    mult = mol.multiplicity()

    if clean_method in ("uhf", "ump2", "ump3"):
        reference = "uhf"
    elif clean_method == "rohf":
        reference = "rohf"
    elif clean_method in ("rhf", "rmp2", "rmp3"):
        reference = "rhf"
    else:
        reference = "rhf" if mult == 1 else "uhf"

    calc = MSHQCCalculator(
        molecule=mol,
        basis_name=basis,
        eri_method=resolved_eri,
        aux_basis_name=aux_basis,
        charge=charge,
        multiplicity=mult,
        units=units,
        reference=reference,
        threshold=threshold,
        print_level=print_level,
        basis_dir=basis_dir,
    )

    scf_res = calc.run_scf()
    if clean_method in ("scf", "hf", "rhf", "uhf", "rohf"):
        return scf_res
    if clean_method in ("mp2", "rmp2", "ump2"):
        return calc.run_mp2(scf_res)
    if clean_method in ("mp3", "rmp3", "ump3"):
        return calc.run_mp3(scf_res)
    if clean_method == "omp2":
        return calc.run_omp2(scf_res)
    if clean_method == "omp3":
        return calc.run_omp3(scf_res)

    raise ValueError(f"Metode tidak dikenali: {method}")


def benchmark_basis_sets(element_or_mol: Union[str, int, List[Tuple], mshqc.Molecule],
                         basis_list: List[str],
                         method: str = "omp2",
                         eri_method: str = "exact",
                         charge: int = 0,
                         multiplicity: Optional[int] = None) -> Dict[str, Any]:
    """Benchmark satu metode pada berbagai basis set."""
    results: Dict[str, Any] = {}

    for basis in basis_list:
        t0 = time.perf_counter()
        try:
            res = quick_calculation(
                element_or_mol,
                basis=basis,
                method=method,
                eri_method=eri_method,
                charge=charge,
                multiplicity=multiplicity,
            )
            elapsed = time.perf_counter() - t0
            energy = getattr(res, "energy_total", getattr(res, "e_total", None))
            results[basis] = {
                "result": res,
                "energy": energy,
                "time": elapsed,
                "success": True,
            }
        except Exception as e:
            results[basis] = {
                "success": False,
                "error": str(e),
            }

    return results


def compare_methods(molecule: Union[mshqc.Molecule, str, List[Tuple]],
                    basis: str = "cc-pVTZ",
                    methods: Optional[List[str]] = None,
                    eri_method: str = "exact",
                    aux_basis: Optional[str] = None,
                    charge: int = 0,
                    multiplicity: Optional[int] = None,
                    units: str = "bohr") -> Dict[str, Any]:
    """Bandingkan beberapa metode pada molekul dan basis set yang sama."""
    if methods is None:
        methods = ["scf", "mp2", "omp2", "omp3"]

    session = MSHQCSession(
        molecule=molecule,
        basis_name=basis,
        eri_method=eri_method,
        aux_basis_name=aux_basis,
        charge=charge,
        multiplicity=multiplicity,
        units=units,
    )
    return session.run_complete_analysis(methods)