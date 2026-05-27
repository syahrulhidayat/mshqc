/**
 * @file src/scf/sad_guess.cc
 * @brief Implementasi logika pembangun SAD Guess.
 * @details 
 * - Mendeteksi tipe basis set sekali saja (di luar loop).
 * - Mengambil blok densitas atomik dan menyusunnya ke matriks molekul.
 */

#include "mshqc/sad.h"
#include <iostream>
#include <algorithm>
#include <string>
#ifdef I
#undef I
#endif

namespace mshqc {

Eigen::MatrixXd SADGuess::build(const Molecule& mol, const BasisSet& basis) {
    int nbf = basis.n_basis_functions();
    Eigen::MatrixXd P_sad = Eigen::MatrixXd::Zero(nbf, nbf);
    
    // -------------------------------------------------------------------------
    // OPTIMASI 1: Deteksi Basis Set (Sekali Saja)
    // -------------------------------------------------------------------------
    std::string bname = basis.name();
    std::transform(bname.begin(), bname.end(), bname.begin(), ::tolower);
    
    SadBasisType btype = SadBasisType::UNKNOWN;
    
    // Deteksi Heuristik Sederhana
    if (bname.find("sto") != std::string::npos || 
        bname.find("min") != std::string::npos ||
        bname.find("3-21g") != std::string::npos) {
        btype = SadBasisType::MINIMAL;
    } 
    else if (bname.find("dz") != std::string::npos || 
             bname.find("6-31g") != std::string::npos ||
             bname.find("def2-sv") != std::string::npos) {
        btype = SadBasisType::DOUBLE_ZETA;
    }
    // Tambahkan 'else if' lain untuk Triple Zeta dsb jika database Anda berkembang.
    
    // -------------------------------------------------------------------------
    // CORE LOOP: Iterasi Atom
    // -------------------------------------------------------------------------
    int current_shell_idx = 0;
    int current_bf_offset = 0;
    bool missing_data = false;
    int atoms_found = 0;

    for (int i = 0; i < mol.n_atoms(); ++i) {
        int Z = mol.atom(i).atomic_number;
        
        // 1. Hitung jumlah fungsi basis untuk atom ini
        // Kita iterasi shell basis set secara berurutan.
        // Asumsi: BasisSet diurutkan sesuai urutan atom di Molecule (Standar Libint/GBS).
        int n_atom_bf = 0;
        int shells_for_this_atom = 0;

        for (int s = current_shell_idx; s < basis.n_shells(); ++s) {
            const auto& shell = basis.shell(s);
            // Fungsi center_index() harus sudah ada di class Shell (basis.h)
            if (shell.center_index() == i) {
                n_atom_bf += shell.n_functions();
                shells_for_this_atom++;
            } else {
                // Sudah masuk ke shell milik atom berikutnya, stop.
                break;
            }
        }

        // Safety check: Jika atom tidak punya basis (misal dummy atom), skip logic density
        if (n_atom_bf == 0) {
            current_shell_idx += shells_for_this_atom;
            continue; 
        }

        // 2. Ambil Densitas Atomik dari Database (sad_data.cc)
        // Menggunakan tipe basis yang sudah dideteksi di awal.
        Eigen::MatrixXd D_atom = get_atomic_density(Z, btype, n_atom_bf);

        // 3. Validasi & Copy
        if (D_atom.rows() == 0 || D_atom.rows() != n_atom_bf) {
            missing_data = true;
            // Jika data hilang, biarkan blok ini 0.0. 
            // Nanti GWH fallback atau noise akan menanganinya jika total densitas terlalu kecil.
        } else {
            // Copy blok diagonal (Block Diagonal Approximation)
            // P_sad(offset : offset+dim, offset : offset+dim) = D_atom
            P_sad.block(current_bf_offset, current_bf_offset, n_atom_bf, n_atom_bf) = D_atom;
            atoms_found++;
        }

        // Update tracking index untuk iterasi berikutnya
        current_bf_offset += n_atom_bf;
        current_shell_idx += shells_for_this_atom;
    }

    // -------------------------------------------------------------------------
    // FINAL CHECK
    // -------------------------------------------------------------------------
    // Jika densitas kosong atau hampir nol (misal semua atom tidak ada di DB),
    // kembalikan matriks kosong agar UHF/RHF menggunakan GWH Fallback.
    if (P_sad.norm() < 1e-6) {
        return Eigen::MatrixXd::Zero(0,0); 
    }

    return P_sad;
}

} // namespace mshqc