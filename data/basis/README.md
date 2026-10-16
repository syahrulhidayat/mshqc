# Basis Set Library

This directory contains 538 Gaussian basis set files in standard `.gbs` format bundled with `mshqc`.

## Data Source & Attribution

1. **Basis Set Exchange (BSE / EMSL)** — https://www.basissetexchange.org/
2. **Psi4 Basis Set Library** — https://psicode.org/
3. **Original Peer-Reviewed Publications** — Cited in the header comments of each `.gbs` file (e.g., Dunning `cc-pVXZ`, Weigend `-RI` / `-JKFIT` auxiliary basis sets, Pople `6-31G`/`6-311G`, and Jensen `pcseg` families).

## Copyright & Public Domain Status

Numerical basis set parameters (Gaussian exponents and contraction coefficients) constitute factual scientific data in the public domain. The `.gbs` parser in `mshqc` (`src/core/basis.cc`) is an independent C++20 implementation licensed under the Apache License 2.0.

## Automatic Path Resolution in `mshqc`

When `mshqc` is imported in Python, `MSHQC_BASIS_PATH` is resolved automatically to the installed package's `basis/` directory or the repository's `data/basis/` directory. You can also specify a custom directory explicitly:

```python
import mshqc

mol = mshqc.build_molecule("Ne 0.0 0.0 0.0")
basis = mshqc.BasisSet("cc-pVTZ", mol, basis_dir="data/basis")
```
