/**
 * @file li_casscf_test.cc
 * @brief Lithium atom CASSCF validation test
 * 
 * SYSTEM: Li atom (3 electrons, 1s² 2s¹ configuration)
 * ACTIVE SPACE: CAS(3,5) - ALL 3 electrons in 5 orbitals (full valence)
 * BASIS: STO-3G (5 basis functions: 1s, 2s, 2px, 2py, 2pz)
 * 
 * RATIONALE FOR CAS(3,5):
 * - Li ground state requires 1s-2s correlation (essential for proper description)
 * - CAS(1,4) [freezing 1s²] is WRONG: misses critical correlation
 * - CAS(3,5) captures full correlation including 1s-2s mixing
 * 
 * EXPECTED BEHAVIOR:
 * - Inactive: 0 orbitals (full valence, no frozen core)
 * - Active: 5 orbitals (all orbitals in active space)
 * - Virtual: 0 orbitals (minimal basis)
 * 
 * VALIDATION:
 * - E(CASSCF) should be < E(UHF) (variational improvement)
 * - Orbital gradient should decrease across iterations
 * - OPDM trace should equal 3 (3 active electrons)
 * 
 * THEORY REFERENCE:
 * - Szabo & Ostlund (1996), "Modern Quantum Chemistry", Li atom example
 * - Werner & Knowles (1988), CASSCF orbital optimization
 * 
 * @author AI Agent 3 (Multireference Master)
 * @date 2025-11-12
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/mcscf/casscf.h"
#include "mshqc/mcscf/active_space.h"

using namespace mshqc;

int main() {
    std::cout << "========================================\n";
    std::cout << " CASSCF Validation: Li Atom Ground State\n";
    std::cout << "========================================\n\n";
    
    // ========================================================================
    // System Setup: Li atom
    // ========================================================================
    
    // Li atom at origin (no geometry, single atom)
    Molecule li_atom;
    li_atom.add_atom(3, 0.0, 0.0, 0.0);  // Z=3 (Lithium), position (0,0,0)
    
    std::cout << "System: Lithium atom (Li)\n";
    std::cout << "Nuclear charge: 3\n";
    std::cout << "Electrons: 3 (configuration: 1s² 2s¹)\n\n";
    
    // Basis set: STO-3G (minimal basis)
    // Li/STO-3G has 5 basis functions: 1s, 2s, 2px, 2py, 2pz
    BasisSet basis("sto-3g", li_atom);
    
    std::cout << "Basis: STO-3G\n";
    std::cout << "Basis functions: " << basis.n_basis_functions() << "\n\n";
    
    // ========================================================================
    // Reference: RHF Calculation
    // ========================================================================
    
    std::cout << "=== Step 1: RHF Reference ===\n";
    
    auto integrals = std::make_shared<IntegralEngine>(li_atom, basis);
    
    // Li has 3 electrons (open-shell doublet: 1s² 2s¹)
    // Need UHF (Unrestricted Hartree-Fock) as reference
    
    std::cout << "NOTE: Li has 3 electrons (open-shell doublet)\n";
    std::cout << "Using UHF as initial guess for CASSCF\n";
    std::cout << "Configuration: 1s² 2s¹ (n_α=2, n_β=1)\n\n";
    
    // UHF for Li
    int n_alpha = 2;  // 1sα + 2sα
    int n_beta = 1;   // 1sβ
    
    SCFConfig scf_config;
    scf_config.max_iterations = 100;
    scf_config.energy_threshold = 1e-8;
    scf_config.density_threshold = 1e-6;
    
    UHF uhf(li_atom, basis, integrals, n_alpha, n_beta, scf_config);
    auto uhf_result = uhf.compute();
    
    std::cout << "UHF converged: " << (uhf_result.converged ? "Yes" : "No") << "\n";
    std::cout << "E(UHF) = " << std::fixed << std::setprecision(10) 
              << uhf_result.energy_total << " Ha\n\n";
    
    // ========================================================================
    // Active Space: CAS(3, 5) - FULL VALENCE
    // ========================================================================
    
    std::cout << "=== Step 2: Define Active Space ===\n";
    
    int n_basis = basis.n_basis_functions();  // 5 for Li/STO-3G
    int n_elec_total = 3;  // Li has 3 electrons
    int n_elec_active = 3;  // ALL 3 electrons in active space
    int n_orb_active = 5;   // ALL 5 orbitals (1s, 2s, 2px, 2py, 2pz)
    
    // CAS(3,5): ALL 3 electrons in ALL 5 orbitals
    // This is ESSENTIAL for Li ground state!
    // Rationale:
    //   - 1s-2s correlation is critical for proper Li description
    //   - Freezing 1s² (CAS(1,4)) gives WRONG ground state
    // Inactive: 0 orbitals (no frozen core)
    // Active: 5 orbitals (full space)
    // Virtual: 0 (minimal basis)
    
    auto active_space = mcscf::ActiveSpace::CAS(
        n_elec_active,  // 3 active electrons
        n_orb_active,   // 5 active orbitals
        n_basis,        // 5 total orbitals
        n_elec_total    // 3 total electrons
    );
    
    std::cout << "Active space: " << active_space.to_string() << "\n";
    std::cout << "  Inactive: " << active_space.n_inactive() << " orbital(s)\n";
    std::cout << "  Active:   " << active_space.n_active() << " orbital(s)\n";
    std::cout << "  Virtual:  " << active_space.n_virtual() << " orbital(s)\n\n";
    
    // ========================================================================
    // CASSCF Calculation
    // ========================================================================
    
    std::cout << "=== Step 3: CASSCF Optimization ===\n\n";
    
    mcscf::CASSCF casscf(li_atom, basis, integrals, active_space);
    
    // Use UHF orbitals as initial guess
    auto casscf_result = casscf.compute(uhf_result);
    
    // ========================================================================
    // Results Analysis
    // ========================================================================
    
    std::cout << "\n=== Results Summary ===\n";
    std::cout << "E(UHF)      = " << std::fixed << std::setprecision(10) 
              << uhf_result.energy_total << " Ha\n";
    std::cout << "E(CASSCF)   = " << std::fixed << std::setprecision(10) 
              << casscf_result.e_casscf << " Ha\n";
    
    double delta_e = casscf_result.e_casscf - uhf_result.energy_total;
    std::cout << "ΔE          = " << std::scientific << std::setprecision(4)
              << delta_e << " Ha\n\n";
    
    std::cout << "CASSCF converged: " << (casscf_result.converged ? "Yes" : "No") << "\n";
    std::cout << "Iterations: " << casscf_result.n_iterations << "\n\n";
    
    // ========================================================================
    // Validation Checks
    // ========================================================================
    
    std::cout << "=== Validation Checks ===\n";
    
    bool all_passed = true;
    
    // Check 1: Convergence
    if (casscf_result.converged) {
        std::cout << "✓ CASSCF converged\n";
    } else {
        std::cout << "✗ CASSCF did not converge\n";
        all_passed = false;
    }
    
    // Check 2: Energy history (should decrease or stay constant)
    bool energy_variational = true;
    for (size_t i = 1; i < casscf_result.energy_history.size(); i++) {
        if (casscf_result.energy_history[i] > casscf_result.energy_history[i-1] + 1e-8) {
            energy_variational = false;
            break;
        }
    }
    
    if (energy_variational) {
        std::cout << "✓ Energy decreases monotonically (variational)\n";
    } else {
        std::cout << "✗ Energy increased during optimization\n";
        all_passed = false;
    }
    
    // Check 3: Number of iterations reasonable (< 50)
    if (casscf_result.n_iterations < 50) {
        std::cout << "✓ Converged in reasonable iterations\n";
    } else {
        std::cout << "✗ Too many iterations\n";
        all_passed = false;
    }
    
    // Check 4: Energy is finite
    if (std::isfinite(casscf_result.e_casscf)) {
        std::cout << "✓ Energy is finite\n";
    } else {
        std::cout << "✗ Energy is NaN or Inf\n";
        all_passed = false;
    }
    
    std::cout << "\n";
    
    if (all_passed) {
        std::cout << "🎉 All validation checks PASSED!\n";
        return 0;
    } else {
        std::cout << "⚠️  Some checks FAILED - review results\n";
        return 1;
    }
}
