# Quick Start Guide

## Instalasi dalam 5 Menit

```bash
# Clone dan install
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc
pip install .

# Test instalasi
python -c "import mshqc; print('✓ MSH-QC ready!')"
```

## Contoh Pertama: RHF Calculation

```python
import mshqc

# Buat molekul H2
mol = mshqc.create_h2_molecule()

# Run RHF
result = mshqc.quick_rhf(mol, "sto-3g")
print(f"Energy: {result.energy:.6f} Hartree")
```

## Contoh Kedua: MP2 Calculation

```python
import mshqc

# Buat molekul air
mol = mshqc.Molecule()
mol.add_atom("O", 0.0, 0.0, 0.0)
mol.add_atom("H", 0.0, 0.757, 0.587)
mol.add_atom("H", 0.0, -0.757, 0.587)

# RHF calculation
rhf = mshqc.quick_rhf(mol, "cc-pvdz")

# MP2 calculation
mp2 = mshqc.quick_mp2(mol, "cc-pvdz", rhf)
print(f"MP2 Energy: {mp2.energy:.6f} Hartree")
```

## Next Steps

- [More Examples](examples/)
- [API Reference](API.md)
- [Full Documentation](installation/README.md)
