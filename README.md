# MSH-QC: Advanced Quantum Chemistry Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

MSH-QC adalah library quantum mechanics berbasis C++ dengan interface Python yang dirancang untuk simulasi molekuler akurat. Library ini menyediakan implementasi lengkap metode struktur elektronik dari Hartree-Fock dasar hingga pendekatan multi-reference lanjutan.

## ✨ Fitur Utama

### Metode Self-Consistent Field (SCF)
- **RHF** (Restricted Hartree-Fock) - untuk sistem closed-shell
- **ROHF** (Restricted Open-shell Hartree-Fock) - untuk sistem open-shell dengan spin tertentu
- **UHF** (Unrestricted Hartree-Fock) - untuk sistem dengan spin tidak terbatas

### Teori Perturbasi Møller-Plesset
- **MP2** (RMP2, UMP2, OMP2) - korelasi orde kedua
- **MP3** (RMP3, UMP3, OMP3) - korelasi orde ketiga
- **MP4/MP5** (UMP4, UMP5) - teori perturbasi orde tinggi
- **DF-MP2** - Density-Fitting MP2 untuk efisiensi komputasi

### Metode Configuration Interaction (CI)
- **CIS** - Configuration Interaction Singles untuk excited states
- **CISD** - Singles dan Doubles
- **CISDT** - Singles, Doubles, dan Triples
- **FCI** - Full Configuration Interaction (exact solution)
- **MRCI** - Multi-Reference CI untuk sistem kompleks
- **CIPSI** - Selected CI dengan seleksi perturbatif

### Metode Multi-Reference
- **CASSCF** - Complete Active Space SCF
- **SA-CASSCF** - State-Averaged CASSCF untuk multiple states
- **CASPT2** - Complete Active Space Perturbation Theory orde 2
- **DF-CASPT2** - Density-Fitting CASPT2
- **MRMP2** - Multi-Reference Møller-Plesset

### Analisis & Properti
- Natural Orbital Analysis
- Analisis fungsi gelombang
- One- dan Two-Particle Density Matrices
- Gradien numerik
- Optimisasi geometri molekul

## 📋 Persyaratan Sistem

### Software Wajib
- **Python**: 3.8 atau lebih baru
- **C++ Compiler**: 
  - Linux: GCC 9+ atau Clang 8+
  - Windows: Visual Studio 2019+ atau MinGW-w64
- **CMake**: 3.14 atau lebih baru

### Library Python
- NumPy >= 1.22
- pybind11 >= 2.12
- setuptools >= 45

### Library C++ (Opsional)
- **Eigen** >= 3.3 (header-only, wajib)
- **Libint2** >= 2.7 (untuk integral repulsi elektron)
- **Libcint** (untuk density fitting)
- **OpenMP** (untuk paralelisasi)

## 🚀 Instalasi

### Instalasi Cepat (Semua Platform)

**Instalasi Minimal** - Tanpa dependency eksternal (cocok untuk testing):
```bash
# Clone repository
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc

# Install tanpa libint2 dan libcint
pip install .
```

**Instalasi via pip** (langsung dari GitHub):
```bash
pip install git+https://github.com/syahrulhidayat/mshqc.git
```

### Instalasi Lengkap di Linux

#### Ubuntu/Debian
```bash
# Install dependencies sistem
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libeigen3-dev \
    libint2-dev \
    python3-dev \
    python3-pip

# Clone dan install
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc
pip install .
```

#### Fedora/RHEL/CentOS
```bash
# Install dependencies
sudo dnf install -y \
    gcc-c++ \
    cmake \
    eigen3-devel \
    python3-devel \
    python3-pip

# Clone dan install
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc
pip install .
```

#### Arch Linux
```bash
# Install dependencies
sudo pacman -S --needed \
    base-devel \
    cmake \
    eigen \
    python \
    python-pip

# Clone dan install
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc
pip install .
```

### Instalasi Lengkap di Windows

#### Metode 1: Menggunakan Visual Studio

```powershell
# Install Visual Studio 2019/2022 dengan C++ development tools
# Download dari: https://visualstudio.microsoft.com/

# Install CMake
# Download dari: https://cmake.org/download/

# Install Python 3.8+
# Download dari: https://www.python.org/downloads/

# Clone repository
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc

# Install dengan pip
pip install .
```

#### Metode 2: Menggunakan MinGW-w64

```powershell
# Install MinGW-w64
# Download dari: https://sourceforge.net/projects/mingw-w64/

# Tambahkan MinGW ke PATH
$env:PATH = "C:\mingw-w64\bin;$env:PATH"

# Install CMake dan Python seperti Metode 1

# Clone dan install
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc
pip install .
```

#### Metode 3: Menggunakan MSYS2 (Direkomendasikan)

```bash
# Install MSYS2 dari https://www.msys2.org/

# Di MSYS2 terminal:
pacman -S --needed \
    mingw-w64-x86_64-gcc \
    mingw-w64-x86_64-cmake \
    mingw-w64-x86_64-eigen3 \
    mingw-w64-x86_64-python \
    mingw-w64-x86_64-python-pip

# Clone dan install
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc
pip install .
```

### Instalasi dengan Conda (Cross-Platform)

```bash
# Buat environment baru
conda create -n mshqc python=3.10
conda activate mshqc

# Install dependencies
conda install -c conda-forge \
    cmake \
    compilers \
    eigen \
    numpy \
    pybind11

# Clone dan install
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc
pip install .
```

### Instalasi Mode Development

Untuk pengembangan dan testing:

```bash
# Clone repository
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc

# Install dalam mode editable
pip install -e .

# Atau dengan dependencies development
pip install -e ".[dev]"
```

## ⚙️ Konfigurasi Build

### Environment Variables

Anda dapat mengontrol fitur yang di-build menggunakan environment variables:

#### Linux/macOS:
```bash
# Aktifkan/nonaktifkan libint2
export MSHQC_WITH_LIBINT2=ON   # atau OFF

# Aktifkan/nonaktifkan libcint
export MSHQC_WITH_LIBCINT=ON   # atau OFF

# Tentukan path custom untuk Eigen
export EIGEN3_INCLUDE_DIR=/path/to/eigen3

# Kemudian install
pip install .
```

#### Windows (PowerShell):
```powershell
# Aktifkan/nonaktifkan fitur
$env:MSHQC_WITH_LIBINT2="ON"  # atau "OFF"
$env:MSHQC_WITH_LIBCINT="ON"  # atau "OFF"

# Install
pip install .
```

#### Windows (Command Prompt):
```cmd
set MSHQC_WITH_LIBINT2=ON
set MSHQC_WITH_LIBCINT=ON
pip install .
```

### Build Manual dengan CMake

Untuk kontrol penuh atas proses build:

```bash
# Buat direktori build
mkdir build && cd build

# Configure dengan CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMSHQC_WITH_LIBINT2=ON \
    -DMSHQC_WITH_LIBCINT=ON \
    -DCMAKE_INSTALL_PREFIX=/path/to/install

# Build
cmake --build . --config Release

# Install
cmake --install .
```

## 📖 Cara Penggunaan

### Contoh Dasar: Kalkulasi RHF

```python
import mshqc

# Buat molekul H2
mol = mshqc.create_h2_molecule()

# Jalankan kalkulasi RHF dengan basis set STO-3G
result = mshqc.quick_rhf(mol, "sto-3g")

# Tampilkan hasil
print(f"RHF Energy: {result.energy:.6f} Hartree")
print(f"Iterations: {result.iterations}")
```

### Contoh: Kalkulasi MP2

```python
import mshqc

# Buat molekul air
mol = mshqc.Molecule()
mol.add_atom("O", 0.0, 0.0, 0.0)
mol.add_atom("H", 0.0, 0.757, 0.587)
mol.add_atom("H", 0.0, -0.757, 0.587)
mol.set_charge(0)
mol.set_multiplicity(1)

# Kalkulasi RHF terlebih dahulu
rhf_result = mshqc.quick_rhf(mol, "cc-pvdz")
print(f"RHF Energy: {rhf_result.energy:.6f} Hartree")

# Kalkulasi MP2
mp2_result = mshqc.quick_mp2(mol, "cc-pvdz", rhf_result)
print(f"MP2 Energy: {mp2_result.energy:.6f} Hartree")
print(f"Correlation Energy: {mp2_result.correlation_energy:.6f} Hartree")
```

### Contoh: CASSCF

```python
import mshqc

# Buat molekul
mol = mshqc.create_molecule("N2", bond_length=1.098)

# Setup active space: (6 elektron, 6 orbital)
active_space = mshqc.ActiveSpace(
    n_electrons=6,
    n_orbitals=6
)

# Jalankan CASSCF
casscf_result = mshqc.quick_casscf(
    mol, 
    basis="cc-pvdz",
    active_space=active_space
)

print(f"CASSCF Energy: {casscf_result.energy:.6f} Hartree")
```

### Contoh: Optimisasi Geometri

```python
import mshqc

# Buat molekul dengan geometri awal
mol = mshqc.create_h2o_molecule()

# Optimisasi dengan RHF
optimizer = mshqc.GeometryOptimizer(
    method="rhf",
    basis="6-31g"
)

optimized_mol = optimizer.optimize(mol)
print(f"Optimized Energy: {optimizer.final_energy:.6f} Hartree")
print("Optimized Geometry:")
print(optimized_mol.get_xyz())
```

## 🐛 Troubleshooting

### Masalah Umum dan Solusi

#### 1. Error: "Cannot find Eigen3"
```bash
# Linux
sudo apt-get install libeigen3-dev

# macOS
brew install eigen

# Windows (dengan vcpkg)
vcpkg install eigen3
```

#### 2. Error: "MSVC compiler not found" (Windows)
- Install Visual Studio dengan C++ development tools
- Atau gunakan MinGW-w64/MSYS2
- Pastikan compiler ada di PATH

#### 3. Build gagal pada libint2
```bash
# Install tanpa libint2
export MSHQC_WITH_LIBINT2=OFF  # Linux/macOS
$env:MSHQC_WITH_LIBINT2="OFF"  # Windows PowerShell

pip install .
```

#### 4. Import error setelah instalasi
```python
# Cek instalasi
python -c "import mshqc; print(mshqc.__version__)"

# Reinstall jika perlu
pip uninstall mshqc
pip install --no-cache-dir .
```

#### 5. "Permission denied" saat instalasi
```bash
# Gunakan virtual environment (direkomendasikan)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

pip install .
```

#### 6. Metode tertentu tidak tersedia
Beberapa metode memerlukan library eksternal:
- **DF-MP2, DF-CASPT2**: Memerlukan libcint
- **Integral presisi tinggi**: Memerlukan libint2

Install dengan fitur lengkap:
```bash
# Linux
sudo apt-get install libint2-dev
export MSHQC_WITH_LIBINT2=ON
export MSHQC_WITH_LIBCINT=ON
pip install .
```

## 🔧 Konfigurasi Performa

### Parallelization dengan OpenMP

```python
import mshqc

# Set jumlah thread
mshqc.set_num_threads(4)

# Atau gunakan semua core yang tersedia
mshqc.set_num_threads(mshqc.get_num_cores())
```

### Memory Management

```python
# Set maksimal memory (dalam MB)
mshqc.set_memory(4096)  # 4 GB

# Untuk kalkulasi besar
mshqc.set_memory(16384)  # 16 GB
```

## 📊 Performa

MSH-QC dioptimalkan untuk berbagai ukuran sistem:
- **Molekul kecil** (< 10 atom): Single-thread efisien
- **Molekul medium** (10-50 atom): Paralelisasi OpenMP
- **Molekul besar** (> 50 atom): Density-fitting methods

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan:
1. Fork repository
2. Buat branch fitur (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📝 Lisensi

Didistribusikan di bawah MIT License. Lihat `LICENSE` untuk informasi lebih lanjut.

## 📚 Sitasi

Jika Anda menggunakan MSH-QC dalam penelitian, silakan sitasi:

```bibtex
@software{mshqc2024,
  title = {MSH-QC: Quantum Chemistry Library for Computational Chemistry},
  author = {Syahrul Hidayat},
  year = {2024},
  url = {https://github.com/syahrulhidayat/mshqc}
}
```

## 📧 Kontak

Syahrul Hidayat - [@syahrulhidayat](https://github.com/syahrulhidayat)
gmail: hidayatsyahrul53@gmail.com
Project Link: [https://github.com/syahrulhidayat/mshqc](https://github.com/syahrulhidayat/mshqc)

## 🙏 Acknowledgments

- Eigen library untuk linear algebra
- Libint2 untuk integral calculation
- pybind11 untuk Python bindings
- Komunitas quantum chemistry open source
