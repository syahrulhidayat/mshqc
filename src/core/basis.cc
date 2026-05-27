/**
 * @file src/core/basis.cc
 * @brief Basis Set Handling (Raw Import - No Manual Normalization)
 * @details 
 * 1. File ini hanya membaca koefisien dan eksponen mentah dari file .gbs.
 * 2. Normalisasi diserahkan sepenuhnya ke Libint (di integrals.cc).
 * 3. Tabel periodik lengkap (Z=0-118).
 */

#include "mshqc/basis.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cctype>
#include <vector>
#ifdef I
#undef I
#endif

namespace mshqc {
    
constexpr double PI = 3.14159265358979323846;

// ============================================================================
// HELPERS
// ============================================================================

// Helper: Mengubah 'D'/'d' menjadi 'E' untuk notasi ilmiah (Format Fortran lama)
// Contoh: 0.123D-04 -> 0.123E-04
std::string sanitize_number(std::string str) {
    for (char &c : str) {
        if (c == 'D' || c == 'd') c = 'E';
    }
    return str;
}

// ============================================================================
// Shell Implementation
// ============================================================================

Shell::Shell(AngularMomentum am, int center, const std::array<double, 3>& center_pos)
    : am_(am), center_(center), position_(center_pos), spherical_(true) {
}

void Shell::add_primitive(double exponent, double coeff) {
    primitives_.emplace_back(exponent, coeff);
}

int Shell::n_functions() const {
    if (spherical_) {
        return n_spherical_functions(am_);
    } else {
        return n_cartesian_functions(am_);
    }
}

void Shell::normalize() {
    // [EMPTY BY DESIGN]
    // Kita SENGAJA tidak melakukan apa-apa di sini.
    // Kita membiarkan Libint melakukan normalisasi otomatis saat Engine dibuat.
    // Ini mencegah "Double Normalization" yang menyebabkan Overlap > 1.0.
}

// ============================================================================
// BasisSet Implementation (Constructors)
// ============================================================================

BasisSet::BasisSet() : name_("unknown"), spherical_(true) { }

BasisSet::BasisSet(const std::string& basis_name, const Molecule& mol, const std::string& basis_dir)
    : name_(basis_name), spherical_(true) {
    
    std::string basis_lower = basis_name;
    std::transform(basis_lower.begin(), basis_lower.end(), basis_lower.begin(), ::tolower);
    std::string basis_file = basis_dir + "/" + basis_lower + ".gbs";
    
    if (!read_gbs(basis_file, mol)) {
        throw std::runtime_error("Failed to read basis set: " + basis_file);
    }
}

// ============================================================================
// GBS Parsing Implementation
// ============================================================================

bool BasisSet::read_gbs(const std::string& basis_file, const Molecule& mol) {
    std::ifstream f(basis_file);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot open basis file: " << basis_file << std::endl;
        return false;
    }
    
    std::string line;
    std::getline(f, line); // Baca header file
    
    // Deteksi default spherical/cartesian dari header file GBS jika ada
    if (line.find("spherical") != std::string::npos) spherical_ = true;
    else if (line.find("cartesian") != std::string::npos) spherical_ = false;
    
    shells_.clear();
    
    // Loop untuk setiap atom dalam molekul
    for (size_t i = 0; i < mol.n_atoms(); i++) {
        const auto& atom = mol.atom(i);
        std::string sym = get_element_symbol(atom.atomic_number);
        std::array<double, 3> pos = {atom.x, atom.y, atom.z};
        
        // Reset file stream ke awal untuk setiap atom (Inefisen tapi aman)
        f.clear();
        f.seekg(0);
        std::getline(f, line); // Skip header lagi
        
        bool found = false;
        while (std::getline(f, line)) {
            // Skip komentar/baris kosong
            if (line.empty() || line[0] == '!') continue;

            std::istringstream iss(line);
            std::string elem;
            int dummy; 
            
            if (iss >> elem >> dummy) {
                // Header atom di GBS biasanya format: "Li 0"
                if (elem == sym && dummy == 0) {
                    found = true;
                    parse_atom_basis(f, sym, i, pos);
                    break;
                }
            }
        }
        if (!found) {
            std::cerr << "Warning: Basis not found for element " << sym 
                      << " (Z=" << atom.atomic_number << ")" << std::endl;
        }
    }
    
    return true;
}

int BasisSet::parse_atom_basis(std::ifstream& file,
                               const std::string& atom_symbol,
                               int atom_index,
                               const std::array<double, 3>& atom_pos) {
    std::string line;
    int n_added = 0;
    
    while (std::getline(file, line)) {
        if (line.find("****") != std::string::npos) break; // Penanda akhir blok atom
        if (line.empty() || line[0] == '!') continue;
        
        std::istringstream iss(line);
        std::string stype;
        int nprim = 0;
        double scale = 1.0; 
        
        if (!(iss >> stype)) continue;
        std::transform(stype.begin(), stype.end(), stype.begin(), ::toupper);

        // Parsing baris tipe shell: "S 3 1.00" atau "P 2 1.00"
        if (!(iss >> nprim)) {
             // Handle kasus format non-standar
             continue; 
        }
        iss >> scale; // Scale factor (biasanya 1.0)
        
        if (stype == "SP") {
            // Shell gabungan SP (S dan P share exponent yang sama)
            // Umum di STO-3G, 6-31G, dll.
            Shell s_sh(AngularMomentum::S, atom_index, atom_pos);
            Shell p_sh(AngularMomentum::P, atom_index, atom_pos);
            s_sh.set_spherical(spherical_);
            p_sh.set_spherical(spherical_);
            
            for (int i = 0; i < nprim; i++) {
                if (!std::getline(file, line)) break;
                
                // [CRITICAL] Sanitasi input D -> E
                line = sanitize_number(line); 
                
                std::istringstream piss(line);
                double exp, sc, pc;
                // Format: exponent S-coeff P-coeff
                if (piss >> exp >> sc >> pc) {
                    s_sh.add_primitive(exp, sc);
                    p_sh.add_primitive(exp, pc);
                }
            }
            add_shell(s_sh);
            add_shell(p_sh);
            n_added += 2;
            
        } else {
            // Shell tunggal (S, P, D, F, ...)
            AngularMomentum am = char_to_am(stype[0]);
            
            Shell sh(am, atom_index, atom_pos);
            sh.set_spherical(spherical_);
            
            for (int i = 0; i < nprim; i++) {
                if (!std::getline(file, line)) break;
                
                // [CRITICAL] Sanitasi input
                line = sanitize_number(line); 
                
                std::istringstream piss(line);
                double exp, c;
                if (piss >> exp >> c) sh.add_primitive(exp, c);
            }
            add_shell(sh);
            n_added++;
        }
    }
    return n_added;
}

void BasisSet::add_shell(const Shell& shell) { shells_.push_back(shell); }

size_t BasisSet::n_basis_functions() const {
    size_t total = 0;
    for (const auto& shell : shells_) total += shell.n_functions();
    return total;
}

void BasisSet::set_spherical(bool sph) {
    spherical_ = sph;
    for (auto& s : shells_) { 
        s.set_spherical(sph);
        s.set_cartesian(!sph); 
    }
}

void BasisSet::print() const {
    std::cout << "\nBasis set: " << name_ << " (" << (spherical_ ? "Spherical" : "Cartesian") << ")\n";
    std::cout << "Shells: " << n_shells() << " | Functions: " << n_basis_functions() << "\n";
}

void BasisSet::append(const BasisSet& other) {
    // Gabungkan list shell
    this->shells_.insert(this->shells_.end(), other.shells_.begin(), other.shells_.end());

}

int BasisSet::max_angular_momentum() const {
    int max_l = 0;
    for (const auto& shell : shells_) max_l = std::max(max_l, shell.l());
    return max_l;
}

// ============================================================================
// Utility Functions (Mappers & Periodic Table)
// ============================================================================

std::vector<int> BasisSet::shell_to_basis_function_map() const {
    std::vector<int> map;
    int bf_index = 0;
    for (const auto& shell : shells_) {
        map.push_back(bf_index);
        bf_index += shell.n_functions();
    }
    return map;
}

AngularMomentum char_to_am(char c) {
    switch (std::toupper(c)) {
        case 'S': return AngularMomentum::S;
        case 'P': return AngularMomentum::P;
        case 'D': return AngularMomentum::D;
        case 'F': return AngularMomentum::F;
        case 'G': return AngularMomentum::G;
        case 'H': return AngularMomentum::H;
        default: return AngularMomentum::S; // Fallback safe
    }
}

std::string am_to_string(AngularMomentum am) {
    switch (am) {
        case AngularMomentum::S: return "s";
        case AngularMomentum::P: return "p";
        case AngularMomentum::D: return "d";
        case AngularMomentum::F: return "f";
        case AngularMomentum::G: return "g";
        case AngularMomentum::H: return "h";
        default: return "?";
    }
}

std::string get_element_symbol(int Z) {
    // Daftar Lengkap Unsur (Z=0 s/d 118)
    static const char* symbols[] = {
        "X", // 0
        "H", "He", // 1-2
        "Li", "Be", "B", "C", "N", "O", "F", "Ne", // 3-10
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", // 11-18
        "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", // 19-36
        "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", // 37-54
        "Cs", "Ba", // 55-56
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", // 57-71 (Lanthanides)
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", // 72-86
        "Fr", "Ra", // 87-88
        "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", // 89-103 (Actinides)
        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og" // 104-118
    };
    
    // Bounds checking
    if (Z < 0 || Z >= static_cast<int>(sizeof(symbols)/sizeof(char*))) {
        return "?";
    }
    
    return symbols[Z];
}

} // namespace mshqc