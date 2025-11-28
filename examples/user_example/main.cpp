#include <iostream>
#include <mshqc/scf.h>
#include <mshqc/basis.h>
#include <mshqc/molecule.h>

int main() {
    std::cout << "MSH-QC Library Example\n";
    std::cout << "=====================\n";
    
    // Create a simple H2 molecule
    mshqc::Molecule molecule;
    molecule.add_atom(1, 0.0, 0.0, -0.7);  // H atom
    molecule.add_atom(1, 0.0, 0.0, 0.7);   // H atom
    
    std::cout << "Created H2 molecule with " << molecule.natoms() << " atoms\n";
    
    // Create a minimal basis set
    mshqc::BasisSet basis;
    basis.load_minimal("sto-3g");
    
    std::cout << "Loaded STO-3G basis set\n";
    
    // Perform RHF calculation
    mshqc::RHF rhf(molecule, basis);
    auto result = rhf.solve();
    
    std::cout << "RHF calculation completed\n";
    std::cout << "Energy: " << result.energy << " Hartree\n";
    
    // You can now use other methods like MP2, CI, etc.
    // using the rhf result as reference
    
    return 0;
}