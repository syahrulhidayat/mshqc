/**
 * @file li_caspt2_vs_mrmp2_ccpvtz.cc
 * @brief Compare CASPT2 vs MRMP2 for Li ground state with cc-pVTZ
 * 
 * Test objective:
 * 1. Run UHF → CASSCF → CASPT2
 * 2. Run UHF → CASSCF → MRMP2
 * 3. Compare denominators and final energies
 * 4. Detect potential bugs in CASPT2 implementation
 * 
 * Expected differences:
 * - CASPT2: D = E_CASSCF - E_K
 * - MRMP2: D = Σ_occ ε_i - Σ_virt ε_a
 * - Small energy difference (~0.1-0.5 mHa)
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/mcscf/casscf.h"
#include "mshqc/mcscf/caspt2.h"
#include "mshqc/mcscf/mrmp2.h"
#include <iostream>
#include <iomanip>

using namespace mshqc;

int main() {
    std::cout << "======================================================================\n";
    std::cout << "Li Ground State: CASPT2 vs MRMP2 Comparison (cc-pVTZ)\n";
    std::cout << "======================================================================\n\n";

    // ========================================================================
    // System Setup
    // ========================================================================
    
    std::cout << "Setting up Lithium atom...\n";
    Molecule mol;
    mol.add_atom(3, 0.0, 0.0, 0.0);  // Li (Z=3) at origin
    
    int n_elec = mol.n_electrons();  // 3 electrons
    int n_alpha = 2;  // 1s² 2s¹ → 2 α, 1 β
    int n_beta = 1;
    
    std::cout << "  Atom: Li (Z=3)\n";
    std::cout << "  Electrons: 3 (α=2, β=1)\n";
    std::cout << "  Ground state: ²S (doublet)\n\n";

    std::cout << "Setting up cc-pVTZ basis...\n";
    BasisSet basis("cc-pvtz", mol);
    int nbf = basis.n_basis_functions();
    std::cout << "  Basis: cc-pVTZ\n";
    std::cout << "  Basis functions: " << nbf << "\n\n";

    std::cout << "Computing integrals...\n";
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    std::cout << "  Integrals computed\n\n";

    // ========================================================================
    // UHF Calculation
    // ========================================================================
    
    std::cout << "Running UHF (Unrestricted Hartree-Fock)...\n\n";
    
    SCFConfig config;
    config.max_iterations = 100;
    config.energy_threshold = 1e-8;
    config.density_threshold = 1e-6;
    
    UHF uhf(mol, basis, integrals, n_alpha, n_beta, config);
    auto uhf_result = uhf.compute();
    
    std::cout << "\nUHF Results:\n";
    std::cout << "  E(UHF)     = " << std::fixed << std::setprecision(10) 
              << uhf_result.energy_total << " Ha\n";
    std::cout << "  Iterations = " << uhf_result.iterations << "\n";
    std::cout << "  Converged  = " << (uhf_result.converged ? "Yes" : "No") << "\n\n";
    
    double s2 = uhf.compute_s_squared(uhf_result);
    double s2_exact = 0.75;  // S(S+1) for doublet
    std::cout << "  ⟨S²⟩       = " << std::setprecision(6) << s2 << "\n";
    std::cout << "  ⟨S²⟩_exact = " << s2_exact << " (doublet)\n";
    std::cout << "  Spin contamination = " << std::setprecision(4) 
              << (s2 - s2_exact) << "\n\n";

    // ========================================================================
    // CASSCF Calculation
    // ========================================================================
    
    std::cout << "Defining active space CAS(3,5)...\n";
    
    mcscf::ActiveSpace active_space = mcscf::ActiveSpace::CAS(
        3,     // ALL 3 electrons in active space
        5,     // 5 orbitals (1s, 2s, 2px, 2py, 2pz)
        nbf,   // Total number of orbitals
        n_elec // Total electrons (3)
    );
    
    std::cout << "  " << active_space.to_string() << "\n";
    std::cout << "  Inactive (core): " << active_space.n_inactive() 
              << " orbitals (none - full valence)\n";
    std::cout << "  Active:          " << active_space.n_active() 
              << " orbitals (1s, 2s, 2p)\n";
    std::cout << "  Virtual:         " << active_space.n_virtual() 
              << " orbitals\n\n";

    std::cout << "Running CASSCF...\n\n";
    
    mcscf::CASSCF casscf(mol, basis, integrals, active_space);
    casscf.set_max_iterations(50);
    casscf.set_energy_threshold(1e-8);
    casscf.set_gradient_threshold(1e-6);
    
    auto casscf_result = casscf.compute(uhf_result);
    
    std::cout << "\nCASSCF Results:\n";
    std::cout << "  E(CASSCF)  = " << std::fixed << std::setprecision(10) 
              << casscf_result.e_casscf << " Ha\n";
    std::cout << "  Iterations = " << casscf_result.n_iterations << "\n";
    std::cout << "  Determinants = " << casscf_result.n_determinants << "\n";
    std::cout << "  Converged  = " << (casscf_result.converged ? "Yes" : "No") << "\n\n";
    
    double correlation_casscf = casscf_result.e_casscf - uhf_result.energy_total;
    std::cout << "  Correlation (CASSCF-UHF) = " << std::scientific 
              << std::setprecision(6) << correlation_casscf << " Ha\n\n";

    // ========================================================================
    // CASPT2 Calculation
    // ========================================================================
    
    std::cout << "Running CASPT2...\n";
    
    mcscf::CASPT2 caspt2(mol, basis, integrals, casscf_result);
    auto caspt2_result = caspt2.compute();
    
    std::cout << "\nCASSCF Results (for comparison):\n";
    std::cout << "  E(CASSCF)  = " << std::fixed << std::setprecision(10) 
              << caspt2_result.e_casscf << " Ha\n";
    std::cout << "  E(PT2)     = " << std::setprecision(10) 
              << caspt2_result.e_pt2 << " Ha\n";
    std::cout << "  E(CASPT2)  = " << std::setprecision(10) 
              << caspt2_result.e_total << " Ha\n\n";

    // ========================================================================
    // MRMP2 Calculation
    // ========================================================================
    
    std::cout << "Running MRMP2...\n";
    
    mcscf::MRMP2 mrmp2(mol, basis, integrals, casscf_result);
    auto mrmp2_result = mrmp2.compute();
    
    std::cout << "\nCASSCF Results (for comparison):\n";
    std::cout << "  E(CASSCF)  = " << std::fixed << std::setprecision(10) 
              << mrmp2_result.e_casscf << " Ha\n";
    std::cout << "  E(MRMP2)   = " << std::setprecision(10) 
              << mrmp2_result.e_mrmp2_correction << " Ha\n";
    std::cout << "  E(total)   = " << std::setprecision(10) 
              << mrmp2_result.e_total << " Ha\n\n";

    // ========================================================================
    // Comparison & Analysis
    // ========================================================================
    
    std::cout << "\n======================================================================\n";
    std::cout << "CASPT2 vs MRMP2 Comparison\n";
    std::cout << "======================================================================\n\n";
    
    std::cout << "Energy Summary:\n";
    std::cout << "  E(UHF)     = " << std::fixed << std::setprecision(10) 
              << uhf_result.energy_total << " Ha\n";
    std::cout << "  E(CASSCF)  = " << std::setprecision(10) 
              << casscf_result.e_casscf << " Ha\n";
    std::cout << "  E(CASPT2)  = " << std::setprecision(10) 
              << caspt2_result.e_total << " Ha\n";
    std::cout << "  E(MRMP2)   = " << std::setprecision(10) 
              << mrmp2_result.e_total << " Ha\n\n";
    
    std::cout << "Perturbation Corrections:\n";
    std::cout << "  CASPT2:    " << std::setprecision(10) 
              << caspt2_result.e_pt2 << " Ha\n";
    std::cout << "  MRMP2:     " << std::setprecision(10) 
              << mrmp2_result.e_mrmp2_correction << " Ha\n";
    std::cout << "  Difference: " << std::setprecision(10) 
              << (mrmp2_result.e_mrmp2_correction - caspt2_result.e_pt2) << " Ha\n";
    std::cout << "              " << std::setprecision(3) 
              << (mrmp2_result.e_mrmp2_correction - caspt2_result.e_pt2) * 1000.0 
              << " mHa\n\n";
    
    std::cout << "Total Energy Difference:\n";
    std::cout << "  E(MRMP2) - E(CASPT2) = " << std::setprecision(10) 
              << (mrmp2_result.e_total - caspt2_result.e_total) << " Ha\n";
    std::cout << "                        = " << std::setprecision(3) 
              << (mrmp2_result.e_total - caspt2_result.e_total) * 1000.0 << " mHa\n\n";
    
    // Validation
    std::cout << "Validation:\n";
    bool uhf_ok = uhf_result.converged;
    bool casscf_ok = casscf_result.converged;
    bool caspt2_ok = caspt2_result.converged;
    bool mrmp2_ok = mrmp2_result.converged;
    bool variational = casscf_result.e_casscf <= uhf_result.energy_total + 1e-8;
    bool caspt2_lowers = caspt2_result.e_pt2 < 0.0;
    bool mrmp2_lowers = mrmp2_result.e_mrmp2_correction < 0.0;
    bool reasonable_diff = std::abs(mrmp2_result.e_total - caspt2_result.e_total) < 0.001;
    
    std::cout << "  " << (uhf_ok ? "✓" : "✗") << " UHF converged\n";
    std::cout << "  " << (casscf_ok ? "✓" : "✗") << " CASSCF converged\n";
    std::cout << "  " << (caspt2_ok ? "✓" : "✗") << " CASPT2 converged\n";
    std::cout << "  " << (mrmp2_ok ? "✓" : "✗") << " MRMP2 converged\n";
    std::cout << "  " << (variational ? "✓" : "✗") 
              << " E(CASSCF) <= E(UHF) (variational)\n";
    std::cout << "  " << (caspt2_lowers ? "✓" : "✗") 
              << " CASPT2 lowers energy (E_PT2 < 0)\n";
    std::cout << "  " << (mrmp2_lowers ? "✓" : "✗") 
              << " MRMP2 lowers energy (E_MRMP2 < 0)\n";
    std::cout << "  " << (reasonable_diff ? "✓" : "⚠") 
              << " Reasonable difference (<1 mHa)\n\n";
    
    std::cout << "Expected Literature Values:\n";
    std::cout << "  E(Li, cc-pVTZ) ~ -7.432776 Ha (CCSD(T))\n";
    std::cout << "  Our E(CASPT2)  = " << std::setprecision(6) 
              << caspt2_result.e_total << " Ha\n";
    std::cout << "  Our E(MRMP2)   = " << std::setprecision(6) 
              << mrmp2_result.e_total << " Ha\n\n";
    
    bool all_ok = uhf_ok && casscf_ok && caspt2_ok && mrmp2_ok && 
                  variational && caspt2_lowers && mrmp2_lowers;
    
    if (all_ok) {
        std::cout << "======================================================================\n";
        std::cout << "TEST COMPLETED SUCCESSFULLY ✓\n";
        std::cout << "======================================================================\n";
        return 0;
    } else {
        std::cout << "======================================================================\n";
        std::cout << "TEST COMPLETED WITH WARNINGS ⚠\n";
        std::cout << "======================================================================\n";
        return 1;
    }
}
