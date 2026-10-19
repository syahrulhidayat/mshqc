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
MSHQC Session - Complete Workflow Manager
File: python/mshqc/session.py
"""

import time
from typing import List, Optional, Dict, Any, Union, Tuple
import mshqc
from .calculators import MSHQCCalculator


class MSHQCSession:
    """Manajer alur kerja lengkap untuk perhitungan multi-metode MSHQC."""

    def __init__(self,
                 molecule: Union[mshqc.Molecule, str, List[Tuple]],
                 basis_name: str = "cc-pVTZ",
                 eri_method: str = "exact",
                 aux_basis_name: Optional[str] = None,
                 charge: int = 0,
                 multiplicity: Optional[int] = None,
                 units: str = "bohr",
                 reference: Optional[str] = None,
                 threshold: float = 1e-9,
                 print_level: int = 0):
        self.basis_name = basis_name
        self.eri_method = eri_method.lower()
        self.aux_basis_name = aux_basis_name or f"{basis_name}-RI"
        self.threshold = threshold
        self.print_level = print_level

        self.calc = MSHQCCalculator(
            molecule=molecule,
            basis_name=basis_name,
            eri_method=self.eri_method,
            aux_basis_name=self.aux_basis_name,
            charge=charge,
            multiplicity=multiplicity,
            units=units,
            reference=reference,
            threshold=threshold,
            print_level=print_level,
        )
        self.molecule = self.calc.molecule
        self.results: Dict[str, Any] = {}
        self.timings: Dict[str, float] = {}

    def run_complete_analysis(self,
                              methods: Optional[List[str]] = None,
                              opt_method: str = "soscf",
                              grad_thresh: float = 1e-5) -> Dict[str, Any]:
        """Jalankan rangkaian metode secara berurutan dan cetak ringkasan energi."""
        if methods is None:
            methods = ["scf", "mp2", "omp2", "omp3"]

        norm_methods = [m.lower() for m in methods]
        if "all" in norm_methods:
            norm_methods = ["scf", "mp2", "mp3", "omp2", "omp3"]

        scf_res = None
        mp2_res = None

        for m in norm_methods:
            t0 = time.perf_counter()
            if m in ("scf", "hf", "rhf", "uhf", "rohf"):
                ref_override = m if m in ("rhf", "uhf", "rohf") else None
                scf_res = self.calc.run_scf(ref=ref_override)
                key = (ref_override or self.calc.reference).upper()
                self.results[key] = scf_res
                self.timings[key] = time.perf_counter() - t0

            elif m in ("mp2", "rmp2", "ump2"):
                if scf_res is None:
                    t_s = time.perf_counter()
                    scf_res = self.calc.run_scf()
                    ref_key = self.calc.reference.upper()
                    self.results[ref_key] = scf_res
                    self.timings[ref_key] = time.perf_counter() - t_s
                t0 = time.perf_counter()
                mp2_res = self.calc.run_mp2(scf_res)
                key = "RMP2" if self.calc.reference == "rhf" else "UMP2"
                self.results[key] = mp2_res
                self.timings[key] = time.perf_counter() - t0

            elif m in ("mp3", "rmp3", "ump3"):
                if scf_res is None:
                    scf_res = self.calc.run_scf()
                if mp2_res is None:
                    mp2_res = self.calc.run_mp2(scf_res)
                t0 = time.perf_counter()
                mp3_res = self.calc.run_mp3(scf_res, mp2_res)
                key = "RMP3" if self.calc.reference == "rhf" else "UMP3"
                self.results[key] = mp3_res
                self.timings[key] = time.perf_counter() - t0

            elif m == "omp2":
                if scf_res is None:
                    scf_res = self.calc.run_scf()
                t0 = time.perf_counter()
                omp2_res = self.calc.run_omp2(scf_res, opt_method=opt_method, grad_thresh=grad_thresh)
                self.results["OMP2"] = omp2_res
                self.timings["OMP2"] = time.perf_counter() - t0

            elif m == "omp3":
                if scf_res is None:
                    scf_res = self.calc.run_scf()
                t0 = time.perf_counter()
                omp3_res = self.calc.run_omp3(scf_res, opt_method=opt_method, grad_thresh=grad_thresh)
                self.results["OMP3"] = omp3_res
                self.timings["OMP3"] = time.perf_counter() - t0

            else:
                raise ValueError(f"Metode tidak dikenali dalam MSHQCSession: {m}")

        if self.print_level > 0:
            self._print_summary()

        return self.results

    def run_cholesky_workflow(self,
                              threshold: float = 1e-9,
                              scf_type: Optional[str] = None,
                              methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """Jalankan pipeline Cholesky terpadu (SCF -> MP2 -> OMP2 -> OMP3)."""
        self.eri_method = "cholesky"
        self.calc.eri_method = "cholesky"
        self.calc.threshold = threshold
        if scf_type:
            self.calc.reference = scf_type.lower()
        return self.run_complete_analysis(methods=methods or ["scf", "mp2", "omp2", "omp3"])

    def run_df_workflow(self,
                        aux_basis_name: Optional[str] = None,
                        threshold: float = 1e-9,
                        scf_type: Optional[str] = None,
                        methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """Jalankan pipeline Density Fitting terpadu (SCF -> MP2 -> OMP2 -> OMP3)."""
        self.eri_method = "df"
        self.calc.eri_method = "df"
        self.calc.threshold = threshold
        if aux_basis_name:
            self.calc.aux_basis_name = aux_basis_name
        if scf_type:
            self.calc.reference = scf_type.lower()
        return self.run_complete_analysis(methods=methods or ["scf", "mp2", "omp2", "omp3"])

    def _print_summary(self) -> None:
        print("\n" + "=" * 72)
        print(f"MSHQC SESSION SUMMARY | Basis: {self.basis_name} | ERI: {self.eri_method.upper()}")
        print("=" * 72)
        print(f"{'Method':<12} {'Total Energy (Ha)':<22} {'Corr Energy (Ha)':<20} {'Time (s)':<10}")
        print("-" * 72)

        ref_e = None
        for name, res in self.results.items():
            e_tot = getattr(res, "energy_total", getattr(res, "e_total", 0.0))
            if ref_e is None:
                ref_e = e_tot
                e_corr_str = "-"
            else:
                e_corr = getattr(res, "energy_mp2_corr", getattr(res, "e_corr_total", e_tot - ref_e))
                e_corr_str = f"{e_corr:.10f}"
            t_val = self.timings.get(name, 0.0)
            print(f"{name:<12} {e_tot:<22.10f} {e_corr_str:<20} {t_val:<10.4f}")
        print("=" * 72)