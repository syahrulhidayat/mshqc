import os
import sys
import glob
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy
from pathlib import Path

try:
    import pybind11
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11>=2.12"])
    import pybind11

# ============================================================================
# CONFIGURATION
# ============================================================================

# Detect platform
is_windows = sys.platform.startswith('win')
is_macos = sys.platform == 'darwin'
is_linux = sys.platform.startswith('linux')

# Base directories
base_dir = Path(__file__).parent.absolute()
src_dir = base_dir / "src"
include_dir = base_dir / "include"
python_dir = base_dir / "python"

# ============================================================================
# SOURCE FILES - Kumpulkan SEMUA file .cc dari src/
# ============================================================================

source_files = [str(python_dir / "bindings.cc")]  # Main bindings

# Tambahkan SEMUA file .cc dari subdirektori src/
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
    files = glob.glob(str(base_dir / pattern))
    # Filter out backup files
    files = [f for f in files if not f.endswith('.backup') and not f.endswith('.new')]
    source_files.extend(files)

print(f"Found {len(source_files)} source files")
print(f"Bindings file: {source_files[0]}")

# ============================================================================
# INCLUDE DIRECTORIES
# ============================================================================

include_dirs = [
    str(include_dir),
    str(src_dir),
    numpy.get_include(),
    pybind11.get_include(),
]

# Eigen3 - coba berbagai lokasi
eigen_paths = [
    "/usr/include/eigen3",
    "/usr/local/include/eigen3",
    "/opt/homebrew/include/eigen3",  # macOS ARM
    "C:/eigen3/include",  # Windows
    str(Path.home() / "eigen3"),
]

eigen_found = False
for eigen_path in eigen_paths:
    if os.path.exists(eigen_path):
        include_dirs.append(eigen_path)
        print(f"Found Eigen3 at: {eigen_path}")
        eigen_found = True
        break

if not eigen_found:
    print("WARNING: Eigen3 not found in standard locations!")
    print("Please install Eigen3 or set EIGEN3_INCLUDE_DIR environment variable")

# Tambahkan subdirektori include
include_dirs.extend([
    str(include_dir / "mshqc"),
    str(include_dir / "mshqc" / "ci"),
    str(include_dir / "mshqc" / "foundation"),
    str(include_dir / "mshqc" / "gradient"),
    str(include_dir / "mshqc" / "integrals"),
    str(include_dir / "mshqc" / "integration"),
    str(include_dir / "mshqc" / "mcscf"),
    str(include_dir / "mshqc" / "mp"),
    str(include_dir / "mshqc" / "validation"),
])

# ============================================================================
# COMPILE FLAGS
# ============================================================================

extra_compile_args = ["-std=c++17", "-O3"]
extra_link_args = []

if is_linux or is_macos:
    extra_compile_args.extend([
        "-fPIC",
        "-Wall",
        "-Wno-unused-variable",
        "-Wno-unused-function",
    ])
    # OpenMP support (optional)
    if os.environ.get("MSHQC_WITH_OPENMP", "OFF") == "ON":
        extra_compile_args.append("-fopenmp")
        extra_link_args.append("-fopenmp")
        
elif is_windows:
    extra_compile_args.extend([
        "/EHsc",  # Exception handling
        "/MD",    # Runtime library
        "/O2",    # Optimization
    ])

# ============================================================================
# LIBRARIES
# ============================================================================

libraries = []

if is_linux or is_macos:
    libraries.append("m")  # Math library
    
# Optional: Libint2 dan Libcint
if os.environ.get("MSHQC_WITH_LIBINT2", "OFF") == "ON":
    libraries.append("int2")
    
if os.environ.get("MSHQC_WITH_LIBCINT", "OFF") == "ON":
    libraries.append("cint")

# ============================================================================
# EXTENSION MODULE
# ============================================================================

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

# ============================================================================
# SETUP
# ============================================================================

setup(
    name="mshqc",
    version="0.1.0",
    description="Python bindings for MSH-QC quantum mechanics library",
    long_description=Path("README.md").read_text(encoding="utf-8") if Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    author="Muhamad Sahrul Hidayat",
    license="MIT",
    url="https://github.com/syahrulhidayat/mshqc",
    packages=["mshqc"],
    package_dir={"mshqc": "python"},
    package_data={"mshqc": ["py.typed", "*.pyi"]},
    include_package_data=True,
    ext_modules=ext_modules,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.22",
        "pybind11>=2.12",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "mypy>=0.950",
        ],
    },
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
