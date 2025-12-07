#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/foundation/rmp2.h"
#include "mshqc/foundation/rmp3.h"
#include "mshqc/foundation/rmp4.h" // Tambahkan header RMP4

#include <iostream>
#include <memory>
#include <string>

using namespace mshqc;

/**
 * @brief Helper function to run full analysis for a single atom
 */
void run_analysis(int atomic_number, const std::string& atom_name, const std::string& basis_name) {
    std::cout << "\n==========================================================\n";
    std::cout << " PROCESSING: " << atom_name << " (Z=" << atomic_number << ")\n";
    std::cout << " BASIS SET:  " << basis_name << "\n";
    std::cout << "==========================================================\n";

    // 1. Define Molecule
    Molecule mol;
    mol.add_atom(atomic_number, 0.0, 0.0, 0.0);
    mol.set_charge(0);
    mol.set_multiplicity(1); // Closed shell singlet
    
    // 2. Define Basis Set
    // Note: Pastikan file basis (misal: cc-pVQZ.gbs) tersedia di folder data/basis/
    BasisSet basis(basis_name, mol);
    
    std::cout << "System: " << atom_name << " atom (" << mol.n_electrons() << " electrons)\n";
    std::cout << "Basis functions: " << basis.n_basis_functions() << "\n\n";

    // 3. Integrals Engine
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);

    // 4. Run RHF
    SCFConfig scf_config;
    scf_config.energy_threshold = 1e-10; // Perketat sedikit untuk MP4
    scf_config.max_iterations = 100;
    
    std::cout << "--- Step 1: Hartree-Fock (RHF) ---\n";
    RHF rhf(mol, basis, integrals, scf_config);
    auto rhf_res = rhf.compute();
    
    if (!rhf_res.converged) {
        std::cout << "RHF did not converge! Skipping post-HF methods.\n";
        return;
    }
    
    // 5. Run RMP2
    // RMP2 menangkap korelasi pair dasar
    foundation::RMP2 rmp2(rhf_res, basis, integrals);
    auto rmp2_res = rmp2.compute();
    
    // 6. Run RMP3
    // RMP3 mengoreksi over-korelasi MP2 (penting untuk Be)
    foundation::RMP3 rmp3(rhf_res, rmp2_res, basis, integrals);
    auto rmp3_res = rmp3.compute();
    
    // 7. Run RMP4(SDQ)
    // RMP4 menambahkan efek Singles (relaksasi) dan Quadruples (renormalisasi)
    foundation::RMP4 rmp4(rhf_res, rmp2_res, rmp3_res, basis, integrals);
    auto rmp4_res = rmp4.compute();
    
    std::cout << "\n--- Final Summary for " << atom_name << " ---\n";
    std::cout << "E(RHF)      : " << rhf_res.energy_total << " Ha\n";
    std::cout << "E(MP2)      : " << rmp2_res.e_total << " Ha\n";
    std::cout << "E(MP3)      : " << rmp3_res.e_total << " Ha\n";
    std::cout << "E(MP4-SDQ)  : " << rmp4_res.e_total << " Ha\n";
    std::cout << "----------------------------------------------------------\n";
}

int main() {
    try {
        // --- Kasus 1: Beryllium (Be) ---
        // Be dikenal sulit karena degenerasi 2s-2p.
        // MP3 akan positif, MP4(S) akan sangat berpengaruh.
        run_analysis(4, "Beryllium", "cc-pVQZ");
        
        // --- Kasus 2: Neon (Ne) ---
        // Ne adalah atom tertutup cangkang, korelasi lebih kecil.
        run_analysis(10, "Neon", "cc-pVQZ");
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\nCRITICAL ERROR: " << e.what() << "\n";
        return 1;
    }
}