/**
 * @file test_sacas_li.cc
 * @brief Test State-Averaged CASSCF for Li atom excited states
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 * 
 * Test system: Li atom (3 electrons)
 * - Ground state: 1s² 2s¹ (²S)
 * - Excited state: 1s² 2p¹ (²P)
 * 
 * Active space: CAS(1,5) = 1 electron in 5 orbitals (2s, 2p_x, 2p_y, 2p_z, 3s)
 * States: 2 (ground + first excited)
 * Basis: STO-3G
 * 
 * Expected transition: 2s → 2p (~1.85 eV experimental)
 * 
 * REFERENCE:
 * - NIST Atomic Spectra Database: Li I 2s-2p transition at 670.8 nm (1.848 eV)
 * - Werner & Meyer (1981) - State-averaged CASSCF
 * 
 * ORIGINALITY:
 * Test implementation created specifically for MSH-QC package to validate
 * SA-CASSCF implementation for atomic excited states.
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/mcscf/sa_casscf.h"
#include <iostream>
#include <iomanip>
#include <memory>

using namespace mshqc;
using namespace mshqc::mcscf;

int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  SA-CASSCF Test: Li Atom\n";
    std::cout << "========================================\n";
    std::cout << "System: Li atom (3 electrons)\n";
    std::cout << "Ground state: 1s² 2s¹ (²S)\n";
    std::cout << "Excited state: 1s² 2p¹ (²P)\n";
    std::cout << "Basis: STO-3G\n";
    std::cout << "Active space: CAS(1,5)\n";
    std::cout << "States: 2 (equal weights)\n";
    std::cout << "========================================\n\n";
    
    try {
        // Create Li atom at origin
        Molecule mol;
        mol.set_charge(0);
        mol.set_multiplicity(2);  // Doublet: 2S+1 = 2 for S=1/2
        mol.add_atom(3, 0.0, 0.0, 0.0);  // Li at origin (Z=3)
        
        std::cout << "Molecule information:\n";
        std::cout << "  Number of atoms: " << mol.n_atoms() << "\n";
        std::cout << "  Total charge: " << mol.charge() << "\n";
        std::cout << "  Multiplicity: " << mol.multiplicity() << "\n";
        std::cout << "  Number of electrons: " << mol.n_electrons() << "\n";
        std::cout << "\n";
        
        // Build STO-3G basis
        std::string basis_name = "sto-3g";
        BasisSet basis(basis_name, mol);
        int nbasis = basis.n_basis_functions();
        
        std::cout << "Basis set information:\n";
        std::cout << "  Type: " << basis_name << "\n";
        std::cout << "  Number of basis functions: " << nbasis << "\n";
        std::cout << "\n";
        
        // Setup integral engine
        auto integrals = std::make_shared<IntegralEngine>(mol, basis);
        
        // Configure SA-CASSCF
        SACASConfig config;
        config.n_active_electrons = 1;   // One electron in active space (2s/2p)
        config.n_active_orbitals = 5;    // 2s, 3×2p, 3s orbitals
        config.n_states = 2;              // Ground + first excited
        config.set_equal_weights();       // w = [0.5, 0.5]
        config.max_iterations = 50;
        config.energy_thresh = 1.0e-8;
        config.gradient_thresh = 1.0e-6;
        config.print_level = 2;
        
        std::cout << "SA-CASSCF configuration:\n";
        std::cout << "  Active electrons: " << config.n_active_electrons << "\n";
        std::cout << "  Active orbitals: " << config.n_active_orbitals << "\n";
        std::cout << "  Number of states: " << config.n_states << "\n";
        std::cout << "  State weights: [";
        for (size_t i = 0; i < config.state_weights.size(); ++i) {
            std::cout << config.state_weights[i];
            if (i < config.state_weights.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "  Energy threshold: " << config.energy_thresh << "\n";
        std::cout << "  Gradient threshold: " << config.gradient_thresh << "\n";
        std::cout << "\n";
        
        // Run SA-CASSCF
        SACASSCF sacas(mol, basis, integrals, config);
        auto result = sacas.compute();
        
        // Analyze results
        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << "  Analysis\n";
        std::cout << "========================================\n";
        
        if (result.n_states >= 2) {
            double exc_energy_ha = result.state_energies[1] - result.state_energies[0];
            double exc_energy_ev = exc_energy_ha * 27.211386245988;
            double wavelength_nm = 1.0e7 / (exc_energy_ha * 219474.6313632);
            
            std::cout << "Excitation energy (S0 → S1):\n";
            std::cout << "  ΔE = " << std::fixed << std::setprecision(8) 
                      << exc_energy_ha << " Ha\n";
            std::cout << "  ΔE = " << std::setprecision(4) 
                      << exc_energy_ev << " eV\n";
            std::cout << "  λ  = " << std::setprecision(2) 
                      << wavelength_nm << " nm\n";
            std::cout << "\n";
            
            std::cout << "Comparison with experiment:\n";
            std::cout << "  Experimental (2s → 2p): 1.848 eV (670.8 nm)\n";
            std::cout << "  Computed: " << std::setprecision(3) 
                      << exc_energy_ev << " eV (" 
                      << std::setprecision(1) << wavelength_nm << " nm)\n";
            
            double error_ev = std::abs(exc_energy_ev - 1.848);
            double error_pct = 100.0 * error_ev / 1.848;
            std::cout << "  Error: " << std::setprecision(3) 
                      << error_ev << " eV (" 
                      << std::setprecision(1) << error_pct << "%)\n";
            std::cout << "\n";
        }
        
        // Transition properties
        std::cout << "Computing transition properties...\n";
        auto trans_01 = sacas.compute_transition_properties(result, 0, 1);
        
        std::cout << "\n";
        std::cout << "Transition 0 → 1 properties:\n";
        std::cout << "  Energy: " << std::fixed << std::setprecision(4) 
                  << trans_01.energy_diff * 27.211386245988 << " eV\n";
        std::cout << "  Frequency: " << std::setprecision(2) 
                  << trans_01.frequency << " cm⁻¹\n";
        std::cout << "  Wavelength: " << std::setprecision(2) 
                  << trans_01.wavelength << " nm\n";
        std::cout << "  Transition dipole: [" 
                  << trans_01.transition_dipole(0) << ", "
                  << trans_01.transition_dipole(1) << ", "
                  << trans_01.transition_dipole(2) << "] au\n";
        std::cout << "  Dipole strength: " << std::scientific << std::setprecision(4)
                  << trans_01.dipole_strength << " au²\n";
        std::cout << "  Oscillator strength: " << trans_01.oscillator_strength << "\n";
        std::cout << "  Einstein A: " << trans_01.einstein_A << " s⁻¹\n";
        
        // State properties
        std::cout << "\n";
        std::cout << "Computing state properties...\n";
        
        for (int i = 0; i < result.n_states; ++i) {
            auto state_props = sacas.compute_state_properties(result, i);
            
            std::cout << "\nState " << i << " properties:\n";
            std::cout << "  Energy: " << std::fixed << std::setprecision(10)
                      << state_props.energy << " Ha\n";
            std::cout << "  Dipole moment: [" << std::setprecision(6)
                      << state_props.dipole_moment(0) << ", "
                      << state_props.dipole_moment(1) << ", "
                      << state_props.dipole_moment(2) << "] au\n";
            
            double dipole_mag = state_props.dipole_moment.norm();
            std::cout << "  |μ| = " << dipole_mag << " au = " 
                      << dipole_mag * 2.5417464739297717 << " Debye\n";
        }
        
        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << "  Test completed successfully ✓\n";
        std::cout << "========================================\n\n";
        
        std::cout << "Notes:\n";
        std::cout << "- This is a skeleton test with placeholder CI solver\n";
        std::cout << "- Full implementation requires multi-root FCI/CASCI\n";
        std::cout << "- Transition dipoles need CI vector integration\n";
        std::cout << "- Results shown are for structure validation only\n";
        std::cout << "\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
