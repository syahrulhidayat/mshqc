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



# File: python/mshqc/calculators.py
import os
import time
from typing import List, Optional, Tuple, Union, Dict, Any
import mshqc

_SYMBOL_TO_Z = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
    'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54
}

def build_molecule(atoms: Union[mshqc.Molecule, str, List[Tuple]],
                   charge: int = 0,
                   multiplicity: Optional[int] = None,
                   units: str = "bohr") -> mshqc.Molecule:
    """Bangun objek mshqc.Molecule dari list tuple [(Z/sym, x, y, z)] atau string XYZ."""
    if isinstance(atoms, mshqc.Molecule):
        return atoms

    scale = mshqc.angstrom_to_bohr(1.0) if units.lower().startswith("ang") else 1.0
    mol = mshqc.Molecule()

    if isinstance(atoms, str):
        for line in atoms.strip().splitlines():
            parts = line.split()
            if len(parts) == 4:
                sym, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
                z_num = _SYMBOL_TO_Z[sym.capitalize()] if not sym.isdigit() else int(sym)
                mol.add_atom(z_num, x * scale, y * scale, z * scale)
    else:
        for item in atoms:
            sym, x, y, z = item[0], float(item[1]), float(item[2]), float(item[3])
            z_num = _SYMBOL_TO_Z[sym.capitalize()] if isinstance(sym, str) else int(sym)
            mol.add_atom(z_num, x * scale, y * scale, z * scale)

    mol.set_charge(charge)
    n_elec = int(mol.total_nuclear_charge()) - charge
    if multiplicity is None:
        multiplicity = 2 if (n_elec % 2 == 1) else 1
    mol.set_multiplicity(multiplicity)
    return mol


class MSHQCCalculator:
    """High-level Pythonic interface untuk Unified Engine MSHQC (RHF/UHF/ROHF, MP2/MP3, OMP2/OMP3)."""

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
                 scf_type: str = "incore",
                 print_level: int = 0,
                 basis_dir: Optional[str] = None):

        raw_mol = build_molecule(molecule, charge=charge, multiplicity=multiplicity, units=units)
        self.pg = mshqc.PointGroup(raw_mol)
        self.pg.detect()
        self.molecule = self.pg.get_aligned_molecule()

        self.basis_name = basis_name
        self.eri_method = eri_method.lower()
        self.aux_basis_name = aux_basis_name or f"{basis_name}-RI"
        self.threshold = threshold
        self.scf_type = scf_type
        self.print_level = print_level

        if basis_dir is None:
            basis_dir = os.environ.get("MSHQC_BASIS_PATH", "data/basis")
        self.basis_dir = basis_dir

        self.basis = mshqc.BasisSet(basis_name, self.molecule, self.basis_dir)
        self.integrals = mshqc.IntegralEngine(self.molecule, self.basis)
        self.pl = mshqc.PetiteList(self.basis, self.pg)
        self.pl.build()

        n_elec = int(self.molecule.total_nuclear_charge()) - self.molecule.charge()
        mult = self.molecule.multiplicity()
        self.n_alpha = (n_elec + mult - 1) // 2
        self.n_beta = n_elec - self.n_alpha

        if reference is None:
            self.reference = "rhf" if mult == 1 else "uhf"
        else:
            self.reference = reference.lower()

        self.scf_res = None
        self.mp2_res = None

    def _make_scf_config(self) -> mshqc.SCFConfig:
        conf = mshqc.SCFConfig()
        conf.scf_type = self.scf_type
        conf.energy_threshold = self.threshold
        conf.density_threshold = self.threshold
        conf.print_level = self.print_level
        conf.eri_method = self.eri_method
        if self.eri_method == "df":
            conf.use_df = True
            conf.aux_basis_name = self.aux_basis_name
            conf.df_threshold = self.threshold
        else:
            conf.use_df = False
            conf.cholesky_threshold = self.threshold
        return conf

    def _make_mp2_config(self, opt_method: str = "soscf", grad_thresh: float = 1e-5, max_iter: int = 50) -> mshqc.MP2Config:
        conf = mshqc.MP2Config()
        conf.scf_type = self.scf_type
        conf.energy_threshold = self.threshold
        conf.gradient_threshold = grad_thresh
        conf.max_iterations = max_iter
        conf.opt_method = opt_method
        conf.print_level = self.print_level
        conf.eri_method = self.eri_method
        if self.eri_method == "df":
            conf.use_df = True
            conf.aux_basis_name = self.aux_basis_name
            conf.df_threshold = self.threshold
        else:
            conf.use_df = False
            conf.cholesky_threshold = self.threshold
        return conf

    def run_scf(self, ref: Optional[str] = None) -> mshqc.SCFResult:
        ref_mode = (ref or self.reference).lower()
        conf = self._make_scf_config()
        if ref_mode == "rhf":
            solver = mshqc.RHF(self.molecule, self.basis, self.integrals, self.pg, self.pl, conf)
        elif ref_mode == "rohf":
            solver = mshqc.ROHF(self.molecule, self.basis, self.integrals, self.pg, self.pl, self.n_alpha, self.n_beta, conf)
        else:
            solver = mshqc.UHF(self.molecule, self.basis, self.integrals, self.pg, self.pl, self.n_alpha, self.n_beta, conf)
        self.scf_res = solver.compute()
        return self.scf_res

    def run_mp2(self, scf_res: Optional[mshqc.SCFResult] = None) -> mshqc.MP2Result:
        if scf_res is None:
            scf_res = self.scf_res or self.run_scf()
        conf = self._make_mp2_config()
        if self.reference == "rhf":
            solver = mshqc.RMP2(self.molecule, self.basis, self.integrals, scf_res, conf, self.pg, self.pl)
        else:
            solver = mshqc.UMP2(self.molecule, self.basis, self.integrals, scf_res, conf, self.pg, self.pl)
        self.mp2_res = solver.compute()
        return self.mp2_res

    def run_mp3(self, scf_res: Optional[mshqc.SCFResult] = None, mp2_res: Optional[mshqc.MP2Result] = None) -> mshqc.MP3Result:
        if scf_res is None:
            scf_res = self.scf_res or self.run_scf()
        if mp2_res is None:
            mp2_res = self.mp2_res or self.run_mp2(scf_res)
        conf = self._make_mp2_config()
        if self.reference == "rhf":
            solver = mshqc.RMP3(scf_res, mp2_res, conf, self.integrals)
        else:
            solver = mshqc.UMP3(scf_res, mp2_res, conf, self.integrals)
        return solver.compute()

    def run_omp2(self, scf_res: Optional[mshqc.SCFResult] = None, opt_method: str = "soscf", grad_thresh: float = 1e-5, max_iter: int = 50) -> mshqc.MP2Result:
        if scf_res is None:
            scf_res = self.scf_res or self.run_scf()
        conf = self._make_mp2_config(opt_method=opt_method, grad_thresh=grad_thresh, max_iter=max_iter)
        solver = mshqc.OMP2(self.molecule, self.basis, self.integrals, scf_res, conf, self.pg, self.pl)
        return solver.compute()

    def run_omp3(self, scf_res: Optional[mshqc.SCFResult] = None, opt_method: str = "soscf", grad_thresh: float = 1e-5, max_iter: int = 50) -> mshqc.MP3Result:
        if scf_res is None:
            scf_res = self.scf_res or self.run_scf()
        conf = self._make_mp2_config(opt_method=opt_method, grad_thresh=grad_thresh, max_iter=max_iter)
        solver = mshqc.OMP3(self.molecule, self.basis, self.integrals, scf_res, conf, self.pg, self.pl)
        return solver.compute()


def run(molecule: Union[mshqc.Molecule, str, List[Tuple]],
        method: Union[str, List[str]] = "omp2",
        basis: str = "cc-pVTZ",
        eri_method: str = "df",
        aux_basis: Optional[str] = None,
        charge: int = 0,
        multiplicity: Optional[int] = None,
        units: str = "bohr",
        reference: Optional[str] = None,
        threshold: float = 1e-9,
        grad_thresh: float = 1e-5,
        opt_method: str = "soscf",
        print_level: int = 0) -> Union[Any, Dict[str, Any]]:
    """
    Fungsi satu baris (one-liner) untuk menjalankan satu atau beberapa metode sekaligus.
    Contoh:
        res = mshqc.run("O 0 0 0.1173\\nH 0 0.7572 -0.4692\\nH 0 -0.7572 -0.4692",
                        method=["scf", "mp2", "omp2", "omp3"],
                        basis="cc-pVTZ", eri_method="df", units="angstrom")
    """
    calc = MSHQCCalculator(
        molecule=molecule, basis_name=basis, eri_method=eri_method,
        aux_basis_name=aux_basis, charge=charge, multiplicity=multiplicity,
        units=units, reference=reference, threshold=threshold, print_level=print_level
    )

    methods = [method] if isinstance(method, str) else list(method)
    results: Dict[str, Any] = {}

    for m in methods:
        m_low = m.lower()
        t0 = time.perf_counter()
        if m_low in ("scf", "hf", "rhf", "uhf", "rohf"):
            res = calc.run_scf(ref=m_low if m_low in ("rhf", "uhf", "rohf") else None)
        elif m_low in ("mp2", "rmp2", "ump2"):
            res = calc.run_mp2()
        elif m_low in ("mp3", "rmp3", "ump3"):
            res = calc.run_mp3()
        elif m_low == "omp2":
            res = calc.run_omp2(opt_method=opt_method, grad_thresh=grad_thresh)
        elif m_low == "omp3":
            res = calc.run_omp3(opt_method=opt_method, grad_thresh=grad_thresh)
        else:
            raise ValueError(f"Metode tidak dikenali: {m}")
        elapsed = time.perf_counter() - t0
        results[m] = {"result": res, "energy": getattr(res, "energy_total", getattr(res, "e_total", None)), "time": elapsed}

    if isinstance(method, str):
        return results[method]["result"]
    return results