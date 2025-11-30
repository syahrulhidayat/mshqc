import mshqc

# Test 1: Cek apakah modul bisa diimport
print(f"mshqc version: {mshqc.__version__}")

# Test 2: Cek class yang tersedia
print("\nAvailable classes:")
for attr in dir(mshqc):
    if not attr.startswith('_'):
        print(f"  - {attr}")

# Test 3: Cek signature BasisSet
import inspect
print("\nBasisSet constructor signature:")
print(inspect.signature(mshqc.BasisSet.__init__))
