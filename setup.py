import os
import sys
import subprocess
from setuptools import setup, Extension, find_packages
import numpy
from pathlib import Path
import sysconfig

try:
    import pybind11
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11>=2.12"])
    import pybind11

BASE_DIR = Path(__file__).parent.absolute()
print(f"Base directory: {BASE_DIR}")

# Files to EXCLUDE (incompatible or missing dependencies)
EXCLUDE_FILES = [
    "libcint_wrapper.cc",  # Requires libcint (not available)
    "mp_density.cc",       # Struct mismatch with UMP3Result
    "ump4.cc",             # UMP3Result missing required fields
]

# Source files - START WITH bindings.cc
source_files = ["python/bindings.cc"]

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
    for f in files:
        rel_path = str(f.relative_to(BASE_DIR))
        
        if any(exclude in rel_path for exclude in EXCLUDE_FILES):
            print(f"⚠️  Excluding: {rel_path}")
            continue
        
        if any(rel_path.endswith(ex) for ex in ['.backup', '.new']):
            continue
        
        source_files.append(rel_path)

print(f"Total source files: {len(source_files)}")

# Include directories
include_dirs = [
    str(BASE_DIR / "include"),
    str(BASE_DIR / "src"),
    numpy.get_include(),
    pybind11.get_include(),
    sysconfig.get_path('include'),
]

# Find libint2
print("\n=== Searching for libint2 ===")
libint_paths = [
    "/usr/include",
    "/usr/local/include",
    str(Path.home() / "miniconda" / "envs" / "mshqc" / "include"),
]

libint_found = False
for path in libint_paths:
    if os.path.exists(os.path.join(path, "libint2.hpp")):
        if path not in include_dirs:
            include_dirs.append(path)
        print(f"✓ Found libint2 at: {path}")
        libint_found = True
        break

if not libint_found:
    print("⚠️  WARNING: libint2 not found!")

# Find Eigen3
print("\n=== Searching for Eigen3 ===")
eigen_paths = [
    "/usr/include/eigen3",
    "/usr/include/Eigen3",
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
    print("⚠️  WARNING: Eigen3 not found!")

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
libraries = ["m"]

# Check for libint2 library
libint_lib_paths = [
    "/usr/lib64/libint2.so",
    "/usr/lib/libint2.so",
    "/usr/local/lib/libint2.so",
]

for lib_path in libint_lib_paths:
    if os.path.exists(lib_path):
        libraries.append("int2")
        print("✓ Will link against libint2 (int2)")
        break

print(f"\n=== Configuration Summary ===")
print(f"Source files: {len(source_files)}")
print(f"Include dirs: {len(include_dirs)}")
print(f"Libraries: {libraries}")

# Extension module
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

# Setup
setup(
    name="mshqc",
    version="0.1.0",
    author="Muhamad Sahrul Hidayat",
    description="Modern Quantum Chemistry Library",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    install_requires=[
        "numpy>=1.22",
        "pybind11>=2.12",
        "scipy>=1.9.0"
    ],
    python_requires=">=3.8",
    zip_safe=False,
)
