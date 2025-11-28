/**
 * Minimal test to debug DFCASPT2 constructor/destructor issue
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/mcscf/casscf.h"
#include "mshqc/mcscf/df_caspt2.h"
#include <iostream>

using namespace mshqc;

int main() {
    try {
        Molecule mol;
        mol.add_atom(3, 0.0, 0.0, 0.0);
        mol.set_multiplicity(2);
        
        BasisSet basis("cc-pvdz", mol, "data/basis");
        BasisSet aux_basis("cc-pvdz-ri", mol, "data/basis");
        
        auto integrals = std::make_shared<IntegralEngine>(mol, basis);
        
        SCFConfig scf_cfg;
        scf_cfg.max_iterations = 10;
        scf_cfg.energy_threshold = 1e-6;
        UHF uhf(mol, basis, integrals, 2, 1, scf_cfg);
        auto uhf_result = uhf.compute();
        
        int n_total_orb = uhf_result.C_alpha.cols();
        int n_total_elec = mol.n_electrons();
        mcscf::ActiveSpace active_space = mcscf::ActiveSpace::CAS(3, 5, n_total_orb, n_total_elec);
        
        mcscf::CASSCF casscf(mol, basis, integrals, active_space);
        casscf.set_max_iterations(5);
        casscf.set_energy_threshold(1e-6);
        auto cas_result = casscf.compute(uhf_result);
        
        std::cout << "\n=== CREATING DFCASPT2 OBJECT ===\n";
        auto cas_result_ptr = std::make_shared<mcscf::CASResult>(cas_result);
        {
            std::cout << "Before constructor\n" << std::flush;
            mcscf::DFCASPT2 df_caspt2(mol, basis, aux_basis, integrals, cas_result_ptr);
            std::cout << "After constructor, before block exit\n" << std::flush;
        }
        std::cout << "After block exit (destructor should have run)\n" << std::flush;
        
        std::cout << "\n=== SUCCESS! ===\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
