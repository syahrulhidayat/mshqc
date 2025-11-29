import os
import sys
import glob
import subprocess
from setuptools import setup, Extension
import numpy
from pathlib import Path
import sysconfig

try:
    import pybind11
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11>=2.12"])
    import pybind11

# Base directory
BASE_DIR = Path(__file__).parent.absolute()
print(f"Base directory: {BASE_DIR}")

# Source files
source_files = [str(BASE_DIR / "python" / "bindings.cc")]

# Collect all C++ source files
src_patterns = [
    "src/ci/*.cc",
    "src/core/*.cc",
    "src/foundation/*.cc",
    "src/gradient/*.cc",
    "src/integrals/*.cc",
    "src/integration/*.cc",
    "src/mcscf/*.cc",
    "src/mp/*.cc",
    "src/mp2/*.cc",
    "src/mp3/*.cc",
    "src/scf/*.cc",
    "src/validation/*.cc",
]

for pattern in src_patterns:
    files = list(BASE_DIR.glob(pattern))
    files = [str(f) for f in files if not str(f).endswith(('.backup', '.new'))]
    source_files.extend(files)

print(f"Total source files: {len(source_files)}")

# Include directories
include_dirs = [
    str(BASE_DIR / "include"),
    str(BASE_DIR / "src"),
    numpy.get_include(),
    pybind11.get_include(),
    sysconfig.get_path('include'), 
]

# Find Eigen3 - FEDORA SPECIFIC
print("\n=== Searching for Eigen3 ===")
eigen_paths = [
    "/usr/include/eigen3",
    "/usr/include/Eigen3",
    "/usr/include",
    "/usr/local/include/eigen3",
]

eigen_found = False
for path in eigen_paths:
    eigen_header = os.path.join(path, "Eigen", "Dense")
    if os.path.exists(eigen_header):
        include_dirs.append(path)
        print(f"✓ Found Eigen3 at: {path}")
        eigen_found = True
        break
        
# Cari dulu lokasi Eigen3 yang benar
eigen_location = subprocess.run(
    ["find", "/usr/include", "-name", "Dense", "-path", "*/Eigen/*"],
    capture_output=True, text=True
).stdout.strip()

if eigen_location:
    # Extract parent directory
    eigen_dir = os.path.dirname(os.path.dirname(eigen_location))
    include_dirs.append(eigen_dir)
    print(f"✓ Eigen3 found: {eigen_dir}")

# Try pkg-config (Fedora way)
if not eigen_found:
    try:
        result = subprocess.run(
            ["pkg-config", "--cflags-only-I", "eigen3"],
            capture_output=True, text=True, check=True
        )
        eigen_path = result.stdout.strip().replace("-I", "").strip()
        if eigen_path and os.path.exists(eigen_path):
            include_dirs.append(eigen_path)
            print(f"✓ Found Eigen3 via pkg-config: {eigen_path}")
            eigen_found = True
    except:
        pass

if not eigen_found:
    print("⚠ WARNING: Eigen3 not found!")
    print("Install with: sudo dnf install eigen3-devel")

# Compile flags
extra_compile_args = [
    "-std=c++17",
    "-O3",
    "-fPIC",
    "-Wall",
    "-Wno-unused-variable",
    "-Wno-unused-function",
]

extra_link_args = []

# Libraries
libraries = ["m"]

# Extension
ext_modules = [
    Extension(
        "mshqc._core",
        sources=source_files,
        include_dirs=include_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]

setup(
    name="mshqc",
    version="0.1.0",
    author="Muhamad Sahrul Hidayat",
    description="Quantum Chemistry Library",
    packages=["mshqc"],
    package_dir={"mshqc": "python"},
    ext_modules=ext_modules,
    python_requires=">=3.8",
    install_requires=["numpy>=1.22", "pybind11>=2.12"],
    zip_safe=False,
)
