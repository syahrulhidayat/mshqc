/**
 * @file basis.cc
 * @brief Basis set handling and Gaussian basis function management
 * 
 * Implementation of BasisSet class for loading and managing Gaussian basis
 * functions from external .gbs files (cc-pVTZ, STO-3G, etc.).
 * 
 * Theory References:
 *   - S. F. Boys, Proc. R. Soc. London A 200, 542 (1950)
 *     [Gaussian-type orbitals (GTOs) foundation]
 *   - T. H. Dunning Jr., J. Chem. Phys. 90, 1007 (1989)
 *     [Correlation-consistent basis sets (cc-pVXZ)]
 *   - T. Helgaker et al., "Molecular Electronic-Structure Theory" (2000)
 *     [Eq. (9.2.8), p. 320: Gaussian normalization]
 *   - Standard .gbs format (EMSL Basis Set Exchange, public domain)
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-01-29
 * @license MIT License (see LICENSE file in project root)
 * 
 * @note Original implementation for basis set management.
 *       Parses standard .gbs format (industry standard, public domain).
 *       No code copied from existing quantum chemistry software.
 */

/**
 * @file basis.cc
 * @brief Basis set handling and Gaussian basis function management
 * (FINAL PRODUCTION VERSION: Fixes Atom vs Shell Ambiguity)
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

namespace mshqc {
    
constexpr double PI = 3.14159265358979323846;

// ============================================================================
// MATH HELPERS
// ============================================================================

double df(int n) {
    if (n <= 1) return 1.0;
    return n * df(n - 2);
}

double compute_prim_norm(int L, double alpha) {
    double term1 = std::pow(2.0 * alpha / PI, 0.75); 
    double term2 = std::pow(4.0 * alpha, (double)L / 2.0);
    double term3 = std::sqrt(1.0 / df(2 * L - 1));
    return term1 * term2 * term3;
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
    int L = static_cast<int>(am_);
    for (auto& p : primitives_) {
        double N = compute_prim_norm(L, p.exponent);
        p.coefficient *= N;
    }

    double sum = 0.0;
    double power = (double)L + 1.5; 

    for (const auto& pi : primitives_) {
        for (const auto& pj : primitives_) {
            double ratio = 2.0 * std::sqrt(pi.exponent * pj.exponent) / (pi.exponent + pj.exponent);
            double overlap = std::pow(ratio, power);
            sum += pi.coefficient * pj.coefficient * overlap;
        }
    }

    if (sum > 1e-14) { 
        double contraction_norm = 1.0 / std::sqrt(sum);
        for (auto& p : primitives_) {
            p.coefficient *= contraction_norm;
        }
    }
}

// ============================================================================
// BasisSet Implementation
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
            std::istringstream iss(line);
            std::string elem;
            int dummy; // Ini biasanya 0 untuk header atom
            
            if (iss >> elem >> dummy) {
                // [FIX UTAMA] Cek dummy == 0 untuk memastikan ini ATOM, bukan SHELL
                if (elem == sym && dummy == 0) {
                    found = true;
                    parse_atom_basis(f, sym, i, pos);
                    break;
                }
            }
        }
        if (!found) std::cerr << "Warning: Basis not found for element " << sym << std::endl;
    }
    
    for (auto& s : shells_) s.normalize();
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
        
        // Auto Uppercase (Robustness)
        std::transform(stype.begin(), stype.end(), stype.begin(), ::toupper);

        // Smart NPrim Read
        if (!(iss >> nprim)) {
             if (std::string("SPDFGHI").find(stype) != std::string::npos) {
                nprim = 1; 
            } else {
                continue; 
            }
        }
        iss >> scale;
        
        if (stype == "SP") {
            Shell s_sh(AngularMomentum::S, atom_index, atom_pos);
            Shell p_sh(AngularMomentum::P, atom_index, atom_pos);
            s_sh.set_spherical(spherical_);
            p_sh.set_spherical(spherical_);
            
            for (int i = 0; i < nprim; i++) {
                if (!std::getline(file, line)) break;
                std::istringstream piss(line);
                double exp, sc, pc;
                if (piss >> exp >> sc >> pc) {
                    s_sh.add_primitive(exp, sc);
                    p_sh.add_primitive(exp, pc);
                }
            }
            add_shell(s_sh);
            add_shell(p_sh);
            n_added += 2;
            
        } else {
            AngularMomentum am;
            if (stype == "S") am = AngularMomentum::S;
            else if (stype == "P") am = AngularMomentum::P;
            else if (stype == "D") am = AngularMomentum::D;
            else if (stype == "F") am = AngularMomentum::F;
            else if (stype == "G") am = AngularMomentum::G;
            else if (stype == "H") am = AngularMomentum::H;
            else {
                // Skip silently
                continue;
            }
            
            Shell sh(am, atom_index, atom_pos);
            sh.set_spherical(spherical_);
            
            for (int i = 0; i < nprim; i++) {
                if (!std::getline(file, line)) break;
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
    for (auto& shell : shells_) shell.set_spherical(sph);
}

void BasisSet::print() const {
    std::cout << "\nBasis set: " << name_ << " (" << (spherical_ ? "Spherical" : "Cartesian") << ")\n";
    std::cout << "Shells: " << n_shells() << " | Functions: " << n_basis_functions() << "\n";
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
        default: throw std::runtime_error("Unknown angular momentum");
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

double gaussian_normalization_s(double alpha) { return std::pow(2.0 * alpha / PI, 0.75); }

std::string get_element_symbol(int Z) {
    // Array statis berisi simbol atom dari Z=0 sampai Z=118
    static const char* symbols[] = {
        "X", // 0 (Placeholder/Ghost Atom)
        
        // Period 1
        "H",  "He",
        
        // Period 2
        "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
        
        // Period 3
        "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
        
        // Period 4
        "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni",
        "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
        
        // Period 5
        "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
        "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",
        
        // Period 6 
        "Cs", "Ba", 
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
        "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn",
        
        // Period 7 
        "Fr", "Ra", 
        "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", 
        "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    };
    
    // Total ada 119 elemen (termasuk indeks 0)
    if (Z < 0 || Z >= static_cast<int>(sizeof(symbols)/sizeof(char*))) {
        return "?";
    }
    
    return symbols[Z];
}

} // namespace mshqc