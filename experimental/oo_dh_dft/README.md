# Experimental Modules (Excluded from CPC v2.0.0 Release)

## `oo_dh_dft.py`
Quarantined from `python/mshqc/methods/` during Phase 0 feature freeze for the
*Computer Physics Communications* (CPC) v2.0.0 manuscript.

**Re-entry criteria:**
1. Implement the full exchange-correlation kernel response term `f_xc[P_corr]`
   in the orbital gradient.
2. Verify analytic orbital and nuclear gradients against central finite
   differences within `<= 1e-7` Ha/Bohr.
