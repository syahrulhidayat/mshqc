import os
import sys
import glob
import subprocess
from setuptools import setup, Extension, find_packages
import numpy
from pathlib import Path
import sysconfig
from setuptools.command.build_ext import build_ext
import pybind11


try:
    import pybind11
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11>=2.12"])
    import pybind11

# Base directory
BASE_DIR = Path(__file__).parent.absolute()
print(f"Base directory: {BASE_DIR}")

# Source files - GUNAKAN PATH RELATIF
source_files = ["python/bindings.cc"]  # ← INI YANG DIPERBAIKI

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
    # Konversi ke path relatif terhadap BASE_DIR
    files = [str(f.relative_to(BASE_DIR)) for f in files if not str(f).endswith(('.backup', '.new'))]
    source_files.extend(files)

print(f"Total source files: {len(source_files)}")

# Include directories - tetap gunakan absolute path untuk include_dirs (ini diperbolehkan)
include_dirs = [
    str(BASE_DIR / "include"),
    str(BASE_DIR / "src"),
    numpy.get_include(),
    pybind11.get_include(),
    sysconfig.get_path('include'),
]

# Tambah include path untuk libint2 kalau ada (mis. Fedora: /usr/include/libint2.hpp)
libint_include_candidates = [
    "/usr/include",
    "/usr/local/include",
]
for path in libint_include_candidates:
    if os.path.exists(os.path.join(path, "libint2.hpp")) and path not in include_dirs:
        include_dirs.append(path)
        print(f"✓ Added libint2 include dir: {path}")
        break

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
try:
    eigen_location = subprocess.run(
        ["find", "/usr/include", "-name", "Dense", "-path", "*/Eigen/*"],
        capture_output=True, text=True, timeout=5
    ).stdout.strip()

    if eigen_location:
        # Extract parent directory
        eigen_dir = os.path.dirname(os.path.dirname(eigen_location))
        if eigen_dir not in include_dirs:
            include_dirs.append(eigen_dir)
            print(f"✓ Eigen3 found: {eigen_dir}")
            eigen_found = True
except Exception as e:
    print(f"Warning: Could not search for Eigen3: {e}")

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
# Tambahkan libint2 jika tersedia (Fedora: libint2.so → -lint2)
libraries = ["m"]
if os.path.exists("/usr/lib64/libint2.so") or os.path.exists("/usr/lib/libint2.so"):
    libraries.append("int2")
    print("✓ Will link against libint2 (int2)")

# Extension
ext_modules = [
    Extension(
        "mshqc._core",
        sources=source_files,  # Sudah menggunakan path relatif
        include_dirs=include_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]

# Satu-satunya setup() yang dipakai: layout paket Python di bawah python/
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
    install_requires=["numpy>=1.22", "pybind11>=2.12", "scipy>=1.9.0"],
    python_requires=">=3.8",
    zip_safe=False,
)

# ---- Di bawah ini adalah kode CMake lama yang tidak lagi dipakai langsung ----
def check_libint2():
    """Check if libint2 is available"""
    search_paths = [
        "/usr/include",
        "/usr/local/include",
        os.path.expanduser("~/miniconda/include"),
        os.path.expanduser("~/.local/include"),
    ]
    
    for path in search_paths:
        if os.path.exists(os.path.join(path, "libint2.hpp")):
            print(f"✓ Found libint2 at: {path}")
            return path
    
    print("✗ libint2 not found!")
    print("\nPlease install libint2:")
    print("  Ubuntu/Debian: sudo apt-get install libint2-dev")
    print("  Conda: conda install -c conda-forge libint")
    print("  macOS: brew install libint")
    sys.exit(1)

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        # Check dependencies
        libint_path = check_libint2()
        
        # ... rest of your CMake build code ...

# Get all source files
def get_source_files():
    base_dir = Path(__file__).parent
    source_dirs = [
        base_dir / "src" / "ci",
        base_dir / "src" / "core",
        base_dir / "src" / "foundation",
        base_dir / "src" / "gradient",
        base_dir / "src" / "integrals",
        base_dir / "src" / "integration",
        base_dir / "src" / "mcscf",
        base_dir / "src" / "mp",
        base_dir / "src" / "mp2",
        base_dir / "src" / "mp3",
        base_dir / "src" / "scf",
        base_dir / "src" / "validation",
    ]
    
    sources = [str(base_dir / "python" / "bindings.cc")]
    for src_dir in source_dirs:
        if src_dir.exists():
            sources.extend([str(f) for f in src_dir.glob("*.cc")])
    
    return sources

setup(
    name="mshqc",
    version="0.1.0",
    author="Syahrul Hidayat",
    description="Modern Quantum Chemistry Library",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[CMakeExtension("mshqc._core")],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.9.0",
    ],
    python_requires=">=3.8",
    zip_safe=False,
)
