/**
 * @file src/core/basis.cc
 * @brief Basis Set Handling (HPC Edition - Exact Contraction Normalization & Dynamic Path)
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
#include <cstdlib> // Diperlukan untuk std::getenv

#ifdef I
#undef I
#endif

namespace mshqc {
    
constexpr double PI = 3.14159265358979323846;

// ============================================================================
// HELPERS
// ============================================================================

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

int Shell::l() const {
    return static_cast<int>(am_);
}

void Shell::normalize() {
    double self_overlap = 0.0;
    int l_val = this->l();
    
    for (size_t i = 0; i < primitives_.size(); ++i) {
        for (size_t j = 0; j < primitives_.size(); ++j) {
            double alpha_i = primitives_[i].exponent;
            double alpha_j = primitives_[j].exponent;
            double c_i = primitives_[i].coefficient;
            double c_j = primitives_[j].coefficient;
            double term = (2.0 * std::sqrt(alpha_i * alpha_j)) / (alpha_i + alpha_j);
            double S_ij = std::pow(term, l_val + 1.5);
            
            self_overlap += c_i * c_j * S_ij;
        }
    }
    
    if (self_overlap > 1e-12) {
        double scale = 1.0 / std::sqrt(self_overlap);
        for (auto& prim : primitives_) {
            prim.coefficient *= scale;
        }
    }
}

// ============================================================================
// BasisSet Implementation (Constructors)
// ============================================================================

BasisSet::BasisSet() : name_("unknown"), spherical_(true) { }

BasisSet::BasisSet(const std::string& basis_name, const Molecule& mol, const std::string& basis_dir)
    : name_(basis_name), spherical_(true) {
    
    std::string basis_lower = basis_name;
    std::transform(basis_lower.begin(), basis_lower.end(), basis_lower.begin(), ::tolower);
    std::string resolved_dir = basis_dir;
    const char* env_path = std::getenv("MSHQC_BASIS_PATH");
    
    if (env_path != nullptr) {
        resolved_dir = std::string(env_path);
    } else if (resolved_dir.empty()) {
        resolved_dir = "./basis"; 
    }
    
    std::string basis_file = resolved_dir + "/" + basis_lower + ".gbs";
    
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
    std::getline(f, line); 
    
    if (line.find("spherical") != std::string::npos) spherical_ = true;
    else if (line.find("cartesian") != std::string::npos) spherical_ = false;
    
    shells_.clear();
    
    for (size_t i = 0; i < mol.n_atoms(); i++) {
        const auto& atom = mol.atom(i);
        std::string sym = get_element_symbol(atom.atomic_number);
        std::array<double, 3> pos = {atom.x, atom.y, atom.z};
        
        f.clear();
        f.seekg(0);
        std::getline(f, line); 
        
        bool found = false;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '!') continue;

            std::istringstream iss(line);
            std::string elem;
            int dummy; 
            
            if (iss >> elem >> dummy) {
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
        if (line.find("****") != std::string::npos) break; 
        if (line.empty() || line[0] == '!') continue;
        
        std::istringstream iss(line);
        std::string stype;
        int nprim = 0;
        double scale = 1.0; 
        
        if (!(iss >> stype)) continue;
        std::transform(stype.begin(), stype.end(), stype.begin(), ::toupper);

        if (!(iss >> nprim)) continue; 
        iss >> scale; 
        
        if (stype == "SP") {
            Shell s_sh(AngularMomentum::S, atom_index, atom_pos);
            Shell p_sh(AngularMomentum::P, atom_index, atom_pos);
            s_sh.set_spherical(spherical_);
            p_sh.set_spherical(spherical_);
            
            for (int i = 0; i < nprim; i++) {
                if (!std::getline(file, line)) break;
                line = sanitize_number(line); 
                
                std::istringstream piss(line);
                double exp, sc, pc;
                if (piss >> exp >> sc >> pc) {
                    s_sh.add_primitive(exp, sc);
                    p_sh.add_primitive(exp, pc);
                }
            }
            
            // Eksekusi normalisasi kontraksi sebelum dimasukkan ke list paket
            s_sh.normalize();
            p_sh.normalize();
            
            add_shell(s_sh);
            add_shell(p_sh);
            n_added += 2;
            
        } else {
            AngularMomentum am = char_to_am(stype[0]);
            Shell sh(am, atom_index, atom_pos);
            sh.set_spherical(spherical_);
            
            for (int i = 0; i < nprim; i++) {
                if (!std::getline(file, line)) break;
                line = sanitize_number(line); 
                
                std::istringstream piss(line);
                double exp, c;
                if (piss >> exp >> c) sh.add_primitive(exp, c);
            }
            sh.normalize();
            
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
    this->shells_.insert(this->shells_.end(), other.shells_.begin(), other.shells_.end());
}

int BasisSet::max_angular_momentum() const {
    int max_l = 0;
    for (const auto& shell : shells_) max_l = std::max(max_l, shell.l());
    return max_l;
}

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
        default: return AngularMomentum::S; 
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
    static const char* symbols[] = {
        "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", 
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", 
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", 
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", 
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", 
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", 
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", 
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", 
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", 
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", 
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    };
    if (Z < 0 || Z >= static_cast<int>(sizeof(symbols)/sizeof(char*))) {
        return "?";
    }
    return symbols[Z];
}

} // namespace mshqc