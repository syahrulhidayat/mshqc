# Gradient Test Programs

**Author**: Muhamad Syahrul Hidayat  
**Date**: 2025-11-17  
**Module**: Numerical Gradients

---

## Overview

Test programs for validating numerical gradient implementation via finite differences.

**Method**: Central difference  
**Formula**: `∂E/∂x ≈ [E(x+δ) - E(x-δ)] / (2δ)`  
**Step size**: δ = 10⁻⁵ au (optimal for double precision)

---

## Test Programs

### 1. test_gradient_h2.cc
**System**: H₂ molecule (homonuclear diatomic)  
**Method**: RHF/STO-3G  
**Geometry**: R = 1.4 Bohr (slightly stretched)

**Tests**:
- ✓ Cylindrical symmetry (x,y gradients ~0)
- ✓ Translational invariance (Σ∇E = 0)
- ✓ Newton's 3rd law (equal/opposite forces)

**Expected Results**:
- H₁: ∂E/∂z < 0 (attractive force toward H₂)
- H₂: ∂E/∂z > 0 (attractive force toward H₁)
- Magnitude: ~0.02-0.05 Ha/bohr

**Usage**:
```bash
cd build
./test_gradient_h2
```

---

### 2. test_gradient_li.cc
**System**: Li atom (open-shell)  
**Method**: UHF (doublet, S=1/2)  
**Basis**: STO-3G and cc-pVDZ

**Tests**:
- ✓ Spherical symmetry (all gradients ~0)
- ✓ Atomic invariance (no forces on single atom)
- ✓ Basis set independence
- ✓ Open-shell UHF functionality

**Expected Results**:
- All components: ~0 (within numerical precision)
- ||∇E|| < 10⁻⁶ Ha/bohr

**Usage**:
```bash
cd build
./test_gradient_li
```

**Physical Interpretation**:
- Single atom at origin has spherical symmetry
- No preferred direction → gradient must be zero
- Non-zero gradient indicates only numerical error

---

### 3. test_gradient_lih.cc
**System**: LiH molecule (heteronuclear diatomic)  
**Method**: RHF/STO-3G  
**Geometry**: R = 3.5 Bohr (stretched from R_eq ≈ 3.02 Bohr)

**Tests**:
- ✓ Cylindrical symmetry (x,y gradients ~0)
- ✓ Translational invariance
- ✓ Newton's 3rd law
- ✓ Heteronuclear forces

**Expected Results**:
- Stretched bond → attractive forces
- Li: ∂E/∂z > 0 (pull toward H)
- H:  ∂E/∂z < 0 (pull toward Li)
- Magnitude: ~0.01-0.05 Ha/bohr

**Usage**:
```bash
cd build
./test_gradient_lih
```

---

### 4. test_gradient_h2o_pvdz.cc
**System**: H₂O molecule (polyatomic)  
**Method**: RHF/cc-pVDZ  
**Geometry**: R(O-H) = 1.9 Bohr, θ(H-O-H) = 104.5°

**Tests**:
- ✓ C2v point group symmetry
- ✓ Planar symmetry (molecule in xz-plane)
- ✓ Translational invariance
- ✓ Equivalent H atoms forces

**Expected Results**:
- 3 atoms → 9 gradient components
- Larger basis set (25 basis functions)
- H₁ and H₂ symmetric by reflection
- Y-components ~0 (planar molecule)

**Usage**:
```bash
cd build
./test_gradient_h2o_pvdz
```

---

## Test Coverage

| Feature | H₂ | Li | LiH | H₂O |
|---------|----|----|-----|-----|
| Homonuclear | ✓ | - | - | - |
| Heteronuclear | - | - | ✓ | - |
| Polyatomic | - | - | - | ✓ |
| Closed-shell (RHF) | ✓ | - | ✓ | ✓ |
| Open-shell (UHF) | - | ✓ | - | - |
| Atomic system | - | ✓ | - | - |
| Diatomic molecule | ✓ | - | ✓ | - |
| Polyatomic molecule | - | - | - | ✓ |
| Symmetry tests | ✓ | ✓ | ✓ | ✓ |
| Conservation laws | ✓ | ✓ | ✓ | ✓ |
| STO-3G basis | ✓ | ✓ | ✓ | - |
| cc-pVDZ basis | - | ✓ | - | ✓ |
| Point group symmetry | - | - | - | ✓ (C2v) |

**Total Coverage**: 4 systems, 2 methods (RHF/UHF), 2 basis sets

---

## Validation Criteria

### 1. Symmetry Tests

**Cylindrical symmetry** (linear molecules):
- Molecules along z-axis
- Expect: ∂E/∂x ≈ 0, ∂E/∂y ≈ 0
- Tolerance: < 10⁻⁶ Ha/bohr

**Spherical symmetry** (atoms):
- Single atom at origin
- Expect: ∂E/∂x = ∂E/∂y = ∂E/∂z ≈ 0
- Tolerance: < 10⁻⁶ Ha/bohr

### 2. Conservation Laws

**Translational invariance**:
```
Σ_A ∂E/∂R_A = 0
```
- Total force must be zero
- Tolerance: ||Σ∇E|| < 10⁻⁶ Ha/bohr

**Newton's 3rd law** (two-body):
```
||∇E_A|| ≈ ||∇E_B||
```
- Equal magnitude, opposite direction
- Tolerance: ratio within 10%

### 3. Physical Expectations

**Stretched bonds**:
- Attractive forces (pulling atoms together)
- Gradients point toward equilibrium

**Compressed bonds**:
- Repulsive forces (pushing atoms apart)
- Gradients point away from compression

**Equilibrium geometry**:
- All gradients ~0
- Stable minimum energy

---

## Compilation

### Using CMake (recommended):
```bash
cd /home/shared/project-mshqc
mkdir -p build && cd build
cmake ..
make test_gradient_h2
make test_gradient_li
make test_gradient_lih
```

### Manual compilation:
```bash
g++ -std=c++17 -O2 \
    -I../include \
    -L../lib \
    test_gradient_h2.cc \
    -o test_gradient_h2 \
    -lmshqc -leigen3 -llibint2
```

---

## Expected Output

### Successful Test:
```
========================================
  H2 Numerical Gradient Test
  RHF/STO-3G
========================================

...

Test results:
  X symmetry:    PASS ✓
  Y symmetry:    PASS ✓
  Translation:   PASS ✓
========================================

All tests passed! ✓
```

### Failed Test:
```
Test results:
  X symmetry:    FAIL ✗
  Y symmetry:    PASS ✓
  Translation:   FAIL ✗
========================================

Some tests failed! ✗
```

**Troubleshooting**:
1. Check SCF convergence
2. Verify step size (try δ = 1e-4 or 1e-6)
3. Increase SCF precision
4. Check for numerical instabilities

---

## Theory References

**Numerical Gradients**:
1. W. H. Press et al., "Numerical Recipes" (2007), Section 5.7
2. J. Baker, J. Comput. Chem. **7**, 385 (1986)

**Analytical Gradients** (future):
1. P. Pulay, Mol. Phys. **17**, 197 (1969)
2. T. Helgaker et al., "Molecular Electronic-Structure Theory" (2000)

---

## Files

```
examples/gradient/
├── README.md                  # This file
├── test_gradient_h2.cc        # H₂ test (171 lines)
├── test_gradient_li.cc        # Li test (198 lines)
├── test_gradient_lih.cc       # LiH test (223 lines)
└── test_gradient_h2o_pvdz.cc  # H₂O/cc-pVDZ test (254 lines)
```

**Total**: 846 lines of test code

---

## Next Steps

1. **Week 2**: Compile and run all tests
2. **Week 3**: Add more molecules (H₂O, NH₃)
3. **Week 4**: Implement geometry optimizer

---

**Status**: Ready for compilation ✅  
**Compliance**: AI_RULES.md ✅  
**Author**: Muhamad Syahrul Hidayat  
**Date**: 2025-11-17
