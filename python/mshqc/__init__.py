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

# File: python/mshqc/__init__.py
import os
import multiprocessing

def _configure_environment():
    try:
        n_cores = multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        n_cores = 4

    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(n_cores)
    if "OPENBLAS_NUM_THREADS" not in os.environ:
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
    if "MKL_NUM_THREADS" not in os.environ:
        os.environ["MKL_NUM_THREADS"] = "1"
    if "VECLIB_MAXIMUM_THREADS" not in os.environ:
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    if "NUMEXPR_NUM_THREADS" not in os.environ:
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    if "MSHQC_BASIS_PATH" not in os.environ:
        pkg_basis = os.path.join(os.path.dirname(__file__), "basis")
        repo_basis = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "basis"))
        if os.path.isdir(pkg_basis):
            os.environ["MSHQC_BASIS_PATH"] = pkg_basis
        elif os.path.isdir(repo_basis):
            os.environ["MSHQC_BASIS_PATH"] = repo_basis

_configure_environment()

from ._mshqc import *
from .calculators import MSHQCCalculator, build_molecule, run
from .session import MSHQCSession
from .utils import quick_calculation, benchmark_basis_sets, compare_methods

__version__ = "1.0.3"

__all__ = [
    "Atom", "Molecule", "GaussianPrimitive", "Shell", "AngularMomentum",
    "BasisSet", "IntegralEngine", "ERITensor", "Screening", "ShellPair",
    "CholeskyERI", "CholeskyDecompositionResult",
    "PointGroup", "SymmetryOperation", "SymOpType", "PetiteList", "UniqueShellPair", "BasisSymmetrizer",
    "DIIS", "SCFConfig", "SCFResult", "RHF", "UHF", "ROHF",
    "MP2Config", "MP2Result", "RMP2", "UMP2", "OMP2",
    "MP3Result", "RMP3", "UMP3", "OMP3",
    "bohr_to_angstrom", "angstrom_to_bohr", "hartree_to_ev", "hartree_to_kcal",
    "MSHQCCalculator", "MSHQCSession", "build_molecule", "run",
    "quick_calculation", "benchmark_basis_sets", "compare_methods",
]
