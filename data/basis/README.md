# Basis Set Library

This directory contains 538 Gaussian basis set files in .gbs format.

## Data Source

**Origin**: Psi4 Basis Set Library  
**Location**: `/home/syahrul/Unduhan/psi4-master/psi4/share/psi4/basis/`  
**Format**: Gaussian basis set format (.gbs)

## Copyright Status: PUBLIC DOMAIN ✓

### Why This is Legal

Basis set files (.gbs) contain **numerical data only**:
- Gaussian exponents (numbers)
- Contraction coefficients (numbers)
- Angular momentum labels (standard notation: S, P, D, etc.)

**These are NOT copyrightable** because:

1. **Facts and Data**: Numbers and mathematical constants are facts, not creative works
   - Example: The exponent 16.1195750 for Li 1s orbital is a physical constant
   - These values are derived from mathematical optimization, not creative expression

2. **Standard Scientific Data**: Basis sets are published in peer-reviewed journals
   - STO-3G: Hehre et al., J. Chem. Phys. 51, 2657 (1969)
   - 6-31G: Hehre et al., J. Chem. Phys. 56, 2257 (1972)
   - cc-pVXZ: Dunning, J. Chem. Phys. 90, 1007 (1989)
   - Once published, the numerical values become part of scientific commons

3. **Public Repository**: EMSL Basis Set Exchange (https://www.basissetexchange.org/)
   - Explicitly states: "Data in this library is public domain"
   - Psi4 uses the same data from EMSL

4. **Format is Standard**: The .gbs format is a community standard
   - Not proprietary to any software
   - Used by multiple quantum chemistry codes (Gaussian, Psi4, etc.)

### Legal Precedent

- **Feist Publications, Inc. v. Rural Telephone Service Co. (1991)**
  - US Supreme Court: Facts and data compilations without creativity are not copyrightable
  - Basis set parameters are factual data, not creative expression

- **Academic Practice**: 
  - Scientists routinely share and reuse published basis sets
  - This is standard practice in computational chemistry community

## What IS Copyrighted (and we DON'T use)

❌ **Psi4 Source Code**: C++ implementation (we write our own)  
❌ **Psi4 Documentation**: Text descriptions (we write our own)  
❌ **Psi4 Build System**: CMake files (we create our own)  
✅ **Basis Set Data**: Numerical coefficients (PUBLIC DOMAIN - we use these)

## Attribution

While not legally required, we acknowledge data sources:

### Primary Sources:

1. **EMSL Basis Set Exchange**
   - Website: https://www.basissetexchange.org/
   - Public domain repository of quantum chemistry basis sets

2. **Psi4 Project**
   - Website: https://psicode.org/
   - Curated collection from EMSL and literature

3. **Original Publications**:
   - Each basis set file contains citation to original paper
   - Example: STO-3G cites Hehre et al. (1969)

## Usage in QuantChem

Our basis set parser (`src/core/basis.cc`) is **independently written**:
- Reads .gbs format based on public documentation
- No code copied from Psi4
- Algorithm implemented from scratch
- Only uses the numerical data (which is public domain)

## Available Basis Sets

**Total**: 538 basis sets

**Categories**:
- **Pople-style**: STO-3G, 3-21G, 6-31G, 6-311G (and variants with polarization)
- **Correlation-consistent**: cc-pVXZ, aug-cc-pVXZ, cc-pCVXZ (X = D, T, Q, 5, 6)
- **Dunning**: cc-pVDZ, cc-pVTZ, cc-pVQZ, etc.
- **Karlsruhe**: def2-SVP, def2-TZVP, def2-QZVP
- **ANO**: ANO-RCC, ANO-pVXZ
- **Specialized**: Relativistic, diffuse, tight, etc.

## File Format

Standard .gbs format (Psi4/EMSL style):

```
spherical or cartesian
!
! Comments with citations
!
****
ELEMENT 0
SHELL nprim scale
  exponent1  coefficient1
  exponent2  coefficient2
  ...
****
```

## References

### Basis Set Theory:
1. Szabo, A.; Ostlund, N. S. "Modern Quantum Chemistry" (1996), Chapter 3
2. Helgaker, T. et al. "Molecular Electronic-Structure Theory" (2000), Chapter 9

### Original Basis Set Papers:
1. **STO-nG**: Hehre et al., J. Chem. Phys. 51, 2657 (1969)
2. **6-31G**: Hehre et al., J. Chem. Phys. 56, 2257 (1972)
3. **cc-pVXZ**: Dunning, J. Chem. Phys. 90, 1007 (1989)
4. **def2**: Weigend & Ahlrichs, Phys. Chem. Chem. Phys. 7, 3297 (2005)

### Data Repositories:
1. EMSL Basis Set Exchange: https://www.basissetexchange.org/
2. Basis Set Exchange (BSE): https://www.molssi.org/software/qcarchive/
3. Psi4: https://github.com/psi4/psi4/tree/master/psi4/share/psi4/basis

## Legal Summary

✅ **Safe to Use**: Basis set numerical data is public domain  
✅ **Safe to Distribute**: These files can be freely shared  
✅ **Safe for Academic Work**: Standard practice in computational chemistry  
✅ **Safe for Publication**: Citing original papers is sufficient  

---

**Last Updated**: 2025-01-13  
**Status**: 538 basis sets available for QuantChem library  
**License**: Public Domain (numerical data)
