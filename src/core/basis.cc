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

#include "mshqc/basis.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

namespace mshqc {

// ============================================================================
// Constants
// ============================================================================

constexpr double PI = 3.14159265358979323846;

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
    // REFERENCE: Helgaker et al. (2000), Eq. (9.2.8), p. 320
    // REFERENCE: Szabo & Ostlund (1996), Problem 3.25, p. 217
    // Normalize contracted Gaussian: ⟨φ|φ⟩ = 1 where φ = Σ_i d_i N_i exp(-α_i r²)
    
    if (am_ != AngularMomentum::S) {
        return; // p,d,f normalization more complex, implement when needed
    }
    
    double S_self = 0.0;
    
    for (size_t i = 0; i < primitives_.size(); i++) {
        double ai = primitives_[i].exponent;
        double di = primitives_[i].coefficient;
        double Ni = gaussian_normalization_s(ai);
        
        for (size_t j = 0; j < primitives_.size(); j++) {
            double aj = primitives_[j].exponent;
            double dj = primitives_[j].coefficient;
            double Nj = gaussian_normalization_s(aj);
            
            double Sij = primitive_overlap_s(ai, aj, position_, position_);
            S_self += di * dj * Ni * Nj * Sij;
        }
    }
    
    if (S_self > 0.0) {
        double norm = 1.0 / std::sqrt(S_self);
        for (auto& p : primitives_) {
            p.coefficient *= norm;
        }
    }
}

// ============================================================================
// BasisSet Implementation
// ============================================================================

BasisSet::BasisSet() : name_("unknown"), spherical_(true) {
}

BasisSet::BasisSet(const std::string& basis_name, 
                   const Molecule& mol,
                   const std::string& basis_dir)
    : name_(basis_name), spherical_(true) {
    
    // Convert basis name to lowercase and construct file path
    std::string basis_lower = basis_name;
    std::transform(basis_lower.begin(), basis_lower.end(), 
                  basis_lower.begin(), ::tolower);
    
    std::string basis_file = basis_dir + "/" + basis_lower + ".gbs";
    
    if (!read_gbs(basis_file, mol)) {
        throw std::runtime_error("Failed to read basis set: " + basis_file);
    }
}

bool BasisSet::read_gbs(const std::string& basis_file, const Molecule& mol) {
    // Parse .gbs format (Psi4/EMSL)
    // DATA SOURCE: EMSL Basis Set Exchange (public domain)
    // Format: spherical/cartesian → **** → ELEMENT 0 → SHELL nprim scale → exponents/coeffs
    
    std::ifstream f(basis_file);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot open basis file: " << basis_file << std::endl;
        return false;
    }
    
    std::string line;
    
    std::getline(f, line);
    if (line.find("spherical") != std::string::npos) {
        spherical_ = true;
    } else if (line.find("cartesian") != std::string::npos) {
        spherical_ = false;
    }
    
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
            int dummy;
            
            if (iss >> elem >> dummy) {
                if (elem == sym) {
                    found = true;
                    parse_atom_basis(f, sym, i, pos);
                    break;
                }
            }
        }
        
        if (!found) {
            std::cerr << "Warning: Basis not found for element " << sym << std::endl;
        }
    }
    
    for (auto& s : shells_) {
        s.normalize();
    }
    
    return true;
}

int BasisSet::parse_atom_basis(std::ifstream& file,
                               const std::string& atom_symbol,
                               int atom_index,
                               const std::array<double, 3>& atom_pos) {
    // Parse shells for single atom until "****" delimiter
    
    std::string line;
    int n_added = 0;
    
    while (std::getline(file, line)) {
        if (line.find("****") != std::string::npos) break;
        if (line.empty() || line[0] == '!') continue;
        
        std::istringstream iss(line);
        std::string stype;
        int nprim;
        double scale;
        
        if (!(iss >> stype >> nprim >> scale)) continue;
        
        if (stype == "SP") {
            Shell s_sh(AngularMomentum::S, atom_index, atom_pos);
            Shell p_sh(AngularMomentum::P, atom_index, atom_pos);
            
            for (int i = 0; i < nprim; i++) {
                if (!std::getline(file, line)) break;
                
                std::istringstream piss(line);
                double exp, sc, pc;
                
                if (piss >> exp >> sc >> pc) {
                    s_sh.add_primitive(exp, sc);
                    p_sh.add_primitive(exp, pc);
                }
            }
            
            s_sh.set_spherical(spherical_);
            p_sh.set_spherical(spherical_);
            
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
                std::cerr << "Unknown shell type: " << stype << std::endl;
                continue;
            }
            
            Shell sh(am, atom_index, atom_pos);
            
            for (int i = 0; i < nprim; i++) {
                if (!std::getline(file, line)) break;
                
                std::istringstream piss(line);
                double exp, c;
                
                if (piss >> exp >> c) {
                    sh.add_primitive(exp, c);
                }
            }
            
            sh.set_spherical(spherical_);
            add_shell(sh);
            n_added++;
        }
    }
    
    return n_added;
}

void BasisSet::add_shell(const Shell& shell) {
    shells_.push_back(shell);
}

size_t BasisSet::n_basis_functions() const {
    size_t total = 0;
    for (const auto& shell : shells_) {
        total += shell.n_functions();
    }
    return total;
}

void BasisSet::set_spherical(bool sph) {
    spherical_ = sph;
    for (auto& shell : shells_) {
        shell.set_spherical(sph);
    }
}

void BasisSet::print() const {
    std::cout << "\n";
    std::cout << "============================================\n";
    std::cout << "          BASIS SET INFORMATION\n";
    std::cout << "============================================\n";
    
    std::cout << "\nBasis set: " << name_ << "\n";
    std::cout << "Type: " << (spherical_ ? "Spherical" : "Cartesian") << "\n";
    std::cout << "Number of shells: " << n_shells() << "\n";
    std::cout << "Number of basis functions: " << n_basis_functions() << "\n";
    std::cout << "Maximum angular momentum: " << max_angular_momentum() << "\n\n";
    
    std::cout << "Shell Details:\n";
    std::cout << "  Shell  Atom   Type  NPrim  NFuncs\n";
    std::cout << "  -----  ----   ----  -----  ------\n";
    
    for (size_t i = 0; i < shells_.size(); i++) {
        const auto& shell = shells_[i];
        std::cout << "  " << std::setw(5) << i
                  << "  " << std::setw(4) << shell.center()
                  << "   " << std::setw(4) << am_to_string(shell.angular_momentum())
                  << "  " << std::setw(5) << shell.n_primitives()
                  << "  " << std::setw(6) << shell.n_functions() << "\n";
    }
    
    std::cout << "\n============================================\n\n";
}

int BasisSet::max_angular_momentum() const {
    int max_l = 0;
    for (const auto& shell : shells_) {
        max_l = std::max(max_l, shell.l());
    }
    return max_l;
}

std::vector<int> BasisSet::shell_to_basis_function_map() const {
    std::vector<int> map;
    map.reserve(shells_.size());
    
    int bf_index = 0;
    for (const auto& shell : shells_) {
        map.push_back(bf_index);
        bf_index += shell.n_functions();
    }
    
    return map;
}

// ============================================================================
// Utility Functions
// ============================================================================

AngularMomentum char_to_am(char c) {
    switch (std::toupper(c)) {
        case 'S': return AngularMomentum::S;
        case 'P': return AngularMomentum::P;
        case 'D': return AngularMomentum::D;
        case 'F': return AngularMomentum::F;
        case 'G': return AngularMomentum::G;
        case 'H': return AngularMomentum::H;
        default:
            throw std::runtime_error("Unknown angular momentum: " + std::string(1, c));
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

double gaussian_normalization_s(double alpha) {
    /**
     * Normalization constant for s-type primitive Gaussian
     * 
     * For an unnormalized s-type Gaussian:
     *   g(r) = exp(-α × r²)
     * 
     * The normalization condition ⟨g|g⟩ = 1 requires:
     * 
     *   ∫∫∫ exp(-2α × r²) dr = π^(3/2) / (2α)^(3/2)
     * 
     * Therefore, the normalization constant is:
     * 
     *   N = [(2α/π)^(3/2)]^(1/2) = (2α/π)^(3/4)
     * 
     * DERIVATION:
     * Using the Gaussian integral:
     *   ∫_{-∞}^{∞} exp(-a x²) dx = √(π/a)
     * 
     * In 3D:
     *   ∫∫∫ exp(-α r²) dr = ∫∫∫ exp(-α(x²+y²+z²)) dx dy dz
     *                     = [√(π/α)]³ = π^(3/2) / α^(3/2)
     * 
     * REFERENCE:
     * Szabo & Ostlund (1996), Eq. (3.203), p. 153
     * "The normalization constant for the Gaussian function..."
     * 
     * @param alpha Gaussian exponent
     * @return Normalization constant
     */
    
    return std::pow(2.0 * alpha / PI, 0.75);
}

double primitive_overlap_s(double alpha_a, double alpha_b,
                          const std::array<double, 3>& Ra,
                          const std::array<double, 3>& Rb) {
    /**
     * Overlap integral between two s-type primitive Gaussians
     * 
     * For two unnormalized s-type Gaussians:
     *   g_A(r) = exp(-α_A × |r - R_A|²)
     *   g_B(r) = exp(-α_B × |r - R_B|²)
     * 
     * The overlap integral is:
     * 
     *   S = ⟨g_A|g_B⟩ = (π/(α_A + α_B))^(3/2) × exp(-ξ × R_AB²)
     * 
     * where:
     *   ξ = (α_A × α_B) / (α_A + α_B)  (reduced exponent)
     *   R_AB = |R_A - R_B|              (distance between centers)
     * 
     * DERIVATION:
     * Using Gaussian product theorem:
     *   exp(-α_A r_A²) × exp(-α_B r_B²) 
     *     = K × exp(-(α_A + α_B) × r_P²)
     * 
     * where K = exp(-ξ × R_AB²) and P is the "Gaussian product center"
     * 
     * REFERENCES:
     * [1] Szabo & Ostlund (1996), Eq. (A.9), Appendix A, p. 411
     *     "Gaussian Product Theorem"
     * 
     * [2] Helgaker et al. (2000), Eq. (9.2.15), p. 322
     *     "Overlap between primitive Gaussians"
     * 
     * @param alpha_a Exponent of first Gaussian
     * @param alpha_b Exponent of second Gaussian
     * @param Ra Position of first Gaussian (Bohr)
     * @param Rb Position of second Gaussian (Bohr)
     * @return Overlap integral (unnormalized)
     */
    
    // Compute distance squared
    double dx = Ra[0] - Rb[0];
    double dy = Ra[1] - Rb[1];
    double dz = Ra[2] - Rb[2];
    double R_AB_sq = dx*dx + dy*dy + dz*dz;
    
    // Reduced exponent
    double alpha_sum = alpha_a + alpha_b;
    double xi = (alpha_a * alpha_b) / alpha_sum;
    
    // Compute overlap
    double prefactor = std::pow(PI / alpha_sum, 1.5);
    double exponential = std::exp(-xi * R_AB_sq);
    
    return prefactor * exponential;
}

// Helper function for BasisSet (get element symbol - needs to be accessible)
std::string get_element_symbol(int Z) {
    static const char* symbols[] = {
        "X",   // Z=0 (placeholder)
        "H",   "He",  // Z=1-2
        "Li",  "Be",  "B",   "C",   "N",   "O",   "F",   "Ne",  // Z=3-10
        "Na",  "Mg",  "Al",  "Si",  "P",   "S",   "Cl",  "Ar",  // Z=11-18
        "K",   "Ca",  "Sc",  "Ti",  "V",   "Cr",  "Mn",  "Fe",  // Z=19-26
        "Co",  "Ni",  "Cu",  "Zn",  "Ga",  "Ge",  "As",  "Se"   // Z=27-34
    };
    
    if (Z < 0 || Z >= static_cast<int>(sizeof(symbols)/sizeof(char*))) {
        return "?";
    }
    
    return symbols[Z];
}

} // namespace mshqc
