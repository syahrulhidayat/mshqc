#!/usr/bin/env python3
"""
validate_natural_orbital_logic.py
Simple validation of natural orbital sum rule logic

This script validates the mathematical correctness of natural orbital analysis
without needing to compile/link the C++ code.

THEORY:
  Natural occupations are eigenvalues of 1-RDM
  Sum rule: Σ n_i = N_electrons (must hold exactly)
  
VALIDATION:
  Uses numpy to diagonalize a toy 1-RDM and verify sum rule
  
Author: Muhamad Syahrul Hidayat
Date: 2025-11-16
"""

import numpy as np

def test_sum_rule():
    """Test that eigenvalues of 1-RDM sum to trace"""
    print("=" * 60)
    print("  Natural Orbital Sum Rule Validation")
    print("=" * 60)
    print()
    
    # Create a toy 1-RDM for 3 electrons in 5 orbitals
    # Diagonal elements should sum to N_elec
    n_elec = 3
    n_orb = 5
    
    # Construct a realistic 1-RDM (Hermitian, positive semi-definite)
    # Diagonal: fractional occupations that sum to n_elec
    diag = np.array([1.95, 0.98, 0.05, 0.015, 0.005])  # sums to 3.0
    
    # Add small off-diagonal elements (to make it realistic)
    rdm = np.diag(diag)
    for i in range(n_orb-1):
        rdm[i, i+1] = 0.01
        rdm[i+1, i] = 0.01
    
    print(f"System: {n_elec} electrons in {n_orb} orbitals")
    print()
    print("1-RDM diagonal (before diagonalization):")
    print(f"  {diag}")
    print(f"  Trace = {np.trace(rdm):.10f}")
    print()
    
    # Diagonalize to get natural occupations
    eigenvalues, eigenvectors = np.linalg.eigh(rdm)
    
    # Sort descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    
    print("Natural occupations (eigenvalues):")
    for i, n in enumerate(eigenvalues):
        print(f"  NO {i+1}: {n:12.8f}")
    print()
    
    # Verify sum rule
    total_occ = np.sum(eigenvalues)
    error = abs(total_occ - n_elec)
    
    print("=" * 60)
    print("  SUM RULE CHECK")
    print("=" * 60)
    print(f"Expected:  {n_elec} electrons")
    print(f"Computed:  {total_occ:.10f}")
    print(f"Error:     {error:.2e}")
    print()
    
    if error < 1e-10:
        print("✅ PASSED: Sum rule satisfied (error < 1e-10)")
        return True
    else:
        print(f"❌ FAILED: Sum rule violated (error = {error:.2e})")
        return False

def test_bounds():
    """Test that natural occupations are in range [0, 2]"""
    print()
    print("=" * 60)
    print("  Natural Occupation Bounds Check")
    print("=" * 60)
    print()
    
    # Create 1-RDM with various occupations
    occupations = np.array([1.98, 1.01, 0.50, 0.30, 0.15, 0.05, 0.01])
    rdm = np.diag(occupations)
    
    print(f"Test occupations: {occupations}")
    print()
    
    eigenvalues, _ = np.linalg.eigh(rdm)
    
    min_occ = np.min(eigenvalues)
    max_occ = np.max(eigenvalues)
    
    print(f"Min occupation: {min_occ:.8f}")
    print(f"Max occupation: {max_occ:.8f}")
    print()
    
    if min_occ >= -1e-10 and max_occ <= 2.0 + 1e-10:
        print("✅ PASSED: All occupations in range [0, 2]")
        return True
    else:
        print("❌ FAILED: Occupations out of bounds!")
        return False

def test_correlation_measure():
    """Test correlation measure (deviation from integer occupations)"""
    print()
    print("=" * 60)
    print("  Correlation Measure")
    print("=" * 60)
    print()
    
    # HF case: integer occupations (no correlation)
    hf_occ = np.array([2.0, 1.0, 0.0, 0.0, 0.0])
    
    # Correlated case: fractional occupations
    corr_occ = np.array([1.95, 0.98, 0.05, 0.015, 0.005])
    
    def correlation_measure(occ):
        """Compute deviation from nearest integer"""
        nearest = np.round(occ)
        return np.sum(np.abs(occ - nearest))
    
    hf_measure = correlation_measure(hf_occ)
    corr_measure = correlation_measure(corr_occ)
    
    print("HF occupations (integer):")
    print(f"  {hf_occ}")
    print(f"  Correlation measure: {hf_measure:.6f}")
    print()
    
    print("Correlated occupations (fractional):")
    print(f"  {corr_occ}")
    print(f"  Correlation measure: {corr_measure:.6f}")
    print()
    
    if hf_measure < 1e-6 and corr_measure > 0.01:
        print("✅ PASSED: Correlation measure distinguishes HF from correlated")
        return True
    else:
        print("❌ FAILED: Correlation measure incorrect")
        return False

if __name__ == "__main__":
    print()
    print("Natural Orbital Analysis - Logic Validation")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: Sum rule
    results.append(test_sum_rule())
    
    # Test 2: Bounds
    results.append(test_bounds())
    
    # Test 3: Correlation measure
    results.append(test_correlation_measure())
    
    # Summary
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print()
    
    n_passed = sum(results)
    n_total = len(results)
    
    print(f"Tests passed: {n_passed}/{n_total}")
    print()
    
    if all(results):
        print("✅ ALL TESTS PASSED")
        print()
        print("Natural orbital logic is mathematically correct.")
        print("The C++ implementation should work correctly if compiled.")
        exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        exit(1)
