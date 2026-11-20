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
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
    'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
    'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
    'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94,
    'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102,
    'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
    'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
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

    # [INJECTED_HPC_V3_CALCULATORS]
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
                 basis_dir: Optional[str] = None,
                 freeze_core: bool = False,
                 n_frozen: int = 0,
                 memory_limit_gb: float = 4.0,
                 use_symmetry: bool = True):

        raw_mol = build_molecule(molecule, charge=charge, multiplicity=multiplicity, units=units)
        self.eri_method = eri_method.lower()
        self.use_symmetry = use_symmetry

        if use_symmetry:
            self.pg = mshqc.PointGroup(raw_mol)
            self.pg.detect()
            self.molecule = self.pg.get_aligned_molecule()
        else:
            self.pg = None
            self.molecule = raw_mol

        self.basis_name = basis_name
        self.aux_basis_name = aux_basis_name or f"{basis_name}-RI"
        self.threshold = threshold
        self.scf_type = scf_type
        self.print_level = print_level
        self.freeze_core = freeze_core
        self.n_frozen = n_frozen if n_frozen > 0 else (
            mshqc.standard_frozen_core(self.molecule) if freeze_core else 0
        )
        self.memory_limit_gb = memory_limit_gb

        if basis_dir is None:
            basis_dir = os.environ.get("MSHQC_BASIS_PATH", "data/basis")
        self.basis_dir = basis_dir

        self.basis = mshqc.BasisSet(basis_name, self.molecule, self.basis_dir)
        self.integrals = mshqc.IntegralEngine(self.molecule, self.basis)
        # Jangan pernah bangun PetiteList 4D O(N^4) saat mode DF/Cholesky!
        if self.pg is not None:
            self.pl = mshqc.PetiteList(self.basis, self.pg)
            if self.eri_method == "exact":
                self.pl.build()
        else:
            self.pl = None

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
        if hasattr(conf, 'freeze_core'): conf.freeze_core = self.freeze_core
        if hasattr(conf, 'n_frozen'): conf.n_frozen = self.n_frozen
        if hasattr(conf, 'memory_limit_gb'): conf.memory_limit_gb = self.memory_limit_gb
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
        if hasattr(conf, 'freeze_core'): conf.freeze_core = self.freeze_core
        if hasattr(conf, 'n_frozen'): conf.n_frozen = self.n_frozen
        if hasattr(conf, 'memory_limit_gb'): conf.memory_limit_gb = self.memory_limit_gb
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
        use_sym = (self.pg is not None and self.pl is not None)
        if ref_mode == "rhf":
            if use_sym:
                solver = mshqc.RHF(self.molecule, self.basis, self.integrals, self.pg, self.pl, conf)
            else:
                solver = mshqc.RHF(self.molecule, self.basis, self.integrals, conf)
        elif ref_mode == "rohf":
            if use_sym:
                solver = mshqc.ROHF(self.molecule, self.basis, self.integrals, self.pg, self.pl, self.n_alpha, self.n_beta, conf)
            else:
                solver = mshqc.ROHF(self.molecule, self.basis, self.integrals, self.n_alpha, self.n_beta, conf)
        else:
            if use_sym:
                solver = mshqc.UHF(self.molecule, self.basis, self.integrals, self.pg, self.pl, self.n_alpha, self.n_beta, conf)
            else:
                solver = mshqc.UHF(self.molecule, self.basis, self.integrals, self.n_alpha, self.n_beta, conf)
        self.scf_res = solver.compute()
        return self.scf_res

    def run_mp2(self, scf_res: Optional[mshqc.SCFResult] = None) -> mshqc.MP2Result:
        if scf_res is None:
            scf_res = self.scf_res or self.run_scf()
        conf = self._make_mp2_config()
        sym_args = (self.pg, self.pl) if (self.pg is not None and self.pl is not None) else ()
        if self.reference == "rhf":
            solver = mshqc.RMP2(self.molecule, self.basis, self.integrals, scf_res, conf, *sym_args)
        elif self.reference == "rohf":
            solver = mshqc.ROMP2(self.molecule, self.basis, self.integrals, scf_res, conf, *sym_args)
        else:
            solver = mshqc.UMP2(self.molecule, self.basis, self.integrals, scf_res, conf, *sym_args)
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
        elif self.reference == "rohf":
            solver = mshqc.ROMP3(scf_res, mp2_res, conf, self.integrals)
        else:
            solver = mshqc.UMP3(scf_res, mp2_res, conf, self.integrals)
        return solver.compute()

    def run_omp2(self, scf_res: Optional[mshqc.SCFResult] = None, opt_method: str = "soscf", grad_thresh: float = 1e-5, max_iter: int = 50) -> mshqc.MP2Result:
        if scf_res is None:
            scf_res = self.scf_res or self.run_scf()
        conf = self._make_mp2_config(opt_method=opt_method, grad_thresh=grad_thresh, max_iter=max_iter)
        sym_args = (self.pg, self.pl) if (self.pg is not None and self.pl is not None) else ()
        solver = mshqc.OMP2(self.molecule, self.basis, self.integrals, scf_res, conf, *sym_args)
        return solver.compute()

    def run_omp3(self, scf_res: Optional[mshqc.SCFResult] = None, opt_method: str = "soscf", grad_thresh: float = 1e-5, max_iter: int = 50) -> mshqc.MP3Result:
        if scf_res is None:
            scf_res = self.scf_res or self.run_scf()
        conf = self._make_mp2_config(opt_method=opt_method, grad_thresh=grad_thresh, max_iter=max_iter)
        sym_args = (self.pg, self.pl) if (self.pg is not None and self.pl is not None) else ()
        solver = mshqc.OMP3(self.molecule, self.basis, self.integrals, scf_res, conf, *sym_args)
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
        print_level: int = 0,
        freeze_core: bool = False,
        n_frozen: int = 0,
        memory_limit_gb: float = 4.0,
        use_symmetry: bool = True) -> Union[Any, Dict[str, Any]]:
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
        units=units, reference=reference, threshold=threshold, print_level=print_level,
        freeze_core=freeze_core, n_frozen=n_frozen, memory_limit_gb=memory_limit_gb, use_symmetry=use_symmetry
    )

    methods = [method] if isinstance(method, str) else list(method)
    results: Dict[str, Any] = {}

    for m in methods:
        m_low = m.lower()
        t0 = time.perf_counter()
        if m_low in ("scf", "hf", "rhf", "uhf", "rohf"):
            res = calc.run_scf(ref=m_low if m_low in ("rhf", "uhf", "rohf") else None)
        elif m_low in ("mp2", "rmp2", "ump2", "romp2"):
            res = calc.run_mp2()
        elif m_low in ("mp3", "rmp3", "ump3", "romp3"):
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

from .geom_opt import UniversalGeomOptimizer, GeomOptResult, ANGSTROM_TO_BOHR


def optimize_geometry_universal(
    atoms: list,
    coords: "np.ndarray",
    basis: str = "cc-pVDZ",
    method: str = "omp2",
    charge: int = 0,
    multiplicity: int = 1,
    units: str = "bohr",
    integral_mode: str = "df",
    single_point_runner=None,
    max_iter: int = 50,
    grad_rms_tol: float = 1.0e-5,
    grad_max_tol: float = 3.0e-5,
    fd_step_bohr: float = 1.0e-4,
    **kwargs,
) -> GeomOptResult:
    """
    Menjalankan optimasi geometri untuk metode apa pun yang tersedia di mshqc:
      - 'rhf', 'uhf', 'rohf'
      - 'rmp2', 'ump2', 'romp2', 'rmp3', 'ump3', 'romp3'
      - 'omp2' ('rhf-omp2'), 'uhf-omp2'
      - 'omp3' ('rhf-omp3'), 'uhf-omp3'
      - 'mp3@omp2', 'ump3@uhf-omp2'
    """
    import numpy as np

    coords_arr = np.asarray(coords, dtype=np.float64)
    if units.strip().lower().startswith("ang"):
        coords_bohr = coords_arr * ANGSTROM_TO_BOHR
    else:
        coords_bohr = coords_arr.copy()

    def _default_energy_eval(c_bohr: np.ndarray) -> float:
        if single_point_runner is not None:
            return single_point_runner(c_bohr)

        # Resolusi otomatis fungsi kalkulator yang ada di modul calculators.py
        g_dict = globals()
        m_low = method.strip().lower().replace("-", "_")

        # 1. Cek apakah ada fungsi spesifik seperti run_omp3, run_omp2, dll.
        for candidate in (f"run_{m_low}", m_low, "calculate", "single_point", "run"):
            if candidate in g_dict and callable(g_dict[candidate]):
                fn = g_dict[candidate]
                try:
                    return fn(
                        atoms=atoms,
                        coords=c_bohr,
                        basis=basis,
                        method=method,
                        charge=charge,
                        multiplicity=multiplicity,
                        units="bohr",
                        **kwargs,
                    )
                except TypeError:
                    pass

        # 2. Cek apakah ada kelas Calculator di calculators.py
        if "Calculator" in g_dict:
            calc_obj = g_dict["Calculator"](
                atoms=atoms,
                coords=c_bohr,
                basis=basis,
                charge=charge,
                multiplicity=multiplicity,
                units="bohr",
                **kwargs,
            )
            if hasattr(calc_obj, "run"):
                return calc_obj.run(method=method)
            if hasattr(calc_obj, m_low):
                return getattr(calc_obj, m_low)()

        raise RuntimeError(
            "Berikan argumen single_point_runner=lambda coords_bohr: <energi> "
            "sesuai pemanggilan single-point mshqc Anda."
        )

    opt = UniversalGeomOptimizer(
        energy_fn=_default_energy_eval,
        method=method,
        integral_mode=integral_mode,
        fd_step_bohr=fd_step_bohr,
        max_iter=max_iter,
        grad_rms_tol=grad_rms_tol,
        grad_max_tol=grad_max_tol,
    )
    return opt.run(coords_bohr)



def recommend_hpc_settings(
    molecule: Union[mshqc.Molecule, str, List[Tuple]],
    basis_name: str = "cc-pVTZ",
    method: str = "omp2",
    charge: int = 0,
    multiplicity: Optional[int] = None,
    units: str = "angstrom",
    memory_limit_gb: float = 8.0,
    basis_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Auto-profiler HPC untuk rentang molekul ringan -> berat dan atom berat.
    Menghitung estimasi N_bf, N_aux, kebutuhan RAM, rekomendasi ERI/SCF/freeze_core,
    serta peringatan fisika kuantum jika terdapat unsur Z > 36 (memerlukan ECP/relativistik).
    """
    mol = build_molecule(molecule, charge=charge, multiplicity=multiplicity, units=units)
    bdir = basis_dir or os.environ.get("MSHQC_BASIS_PATH", "data/basis")
    basis = mshqc.BasisSet(basis_name, mol, bdir)
    nbf = int(basis.n_basis_functions())
    n_elec = int(mol.total_nuclear_charge()) - int(mol.charge())
    mult = int(mol.multiplicity())
    na = (n_elec + mult - 1) // 2
    nb = n_elec - na
    va = max(0, nbf - na)
    n_aux_est = int(nbf * 3.0)

    max_z = max((int(mol.atom(i).atomic_number) for i in range(mol.n_atoms())), default=0)
    n_frozen_std = int(mshqc.standard_frozen_core(mol)) if hasattr(mshqc, "standard_frozen_core") else 0

    mem_exact_gb = (nbf ** 4) * 8.0 / 1.0e9
    mem_df_packed_gb = (nbf * (nbf + 1) / 2.0) * n_aux_est * 8.0 / 1.0e9
    mem_bvv_gb = (va ** 2) * n_aux_est * 8.0 / 1.0e9
    mem_t2_gb = (max(1, na - n_frozen_std) ** 2) * (va ** 2) * 8.0 / 1.0e9

    if nbf <= 60:
        tier = "light"
        eri_rec = "df" if method.lower() in ("omp2", "omp3") else "exact"
        scf_rec = "incore"
        fc_rec = False
    elif nbf <= 250:
        tier = "medium"
        eri_rec = "df"
        scf_rec = "incore"
        fc_rec = True
    else:
        tier = "heavy"
        eri_rec = "df"
        scf_rec = "ooc" if mem_df_packed_gb > memory_limit_gb * 0.55 else "incore"
        fc_rec = True

    warnings: List[str] = []
    if max_z > 36:
        warnings.append(
            f"Terdeteksi atom berat Z={max_z} (>36). mshqc-sync saat ini all-electron non-relativistik "
            "tanpa ECP/X2C; gunakan dengan kehati-hatian karena efek relativistik signifikan."
        )
    elif max_z >= 21 and max_z <= 30 and mult > 1:
        warnings.append(
            f"Terdeteksi logam transisi 3d open-shell (Z={max_z}, mult={mult}). "
            "Disarankan max_iterations>=200 dan verifikasi spin <S^2> serta kestabilan orbital."
        )
    if method.lower() in ("omp2", "omp3") and (mem_df_packed_gb + mem_bvv_gb + mem_t2_gb) > memory_limit_gb:
        warnings.append(
            f"Estimasi puncak RAM OMP2/OMP3 ({mem_df_packed_gb + mem_bvv_gb + mem_t2_gb:.2f} GB) "
            f"melebihi memory_limit_gb={memory_limit_gb:.2f} GB; pertimbangkan DF-MP2 streaming atau basis lebih kecil."
        )

    return {
        "tier": tier,
        "nbf": nbf,
        "n_electrons": n_elec,
        "max_Z": max_z,
        "eri_method": eri_rec,
        "scf_type": scf_rec,
        "freeze_core": fc_rec,
        "n_frozen_standard": n_frozen_std,
        "use_symmetry": (eri_rec == "exact"),
        "est_ram_gb": {
            "exact_eri_4d": round(mem_exact_gb, 3),
            "df_packed_3c": round(mem_df_packed_gb, 3),
            "omp_bvv_3c": round(mem_bvv_gb, 3),
            "t2_active_4d": round(mem_t2_gb, 3),
        },
        "warnings": warnings,
    }
