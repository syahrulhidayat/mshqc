#ifndef MSHQC_BASIS_H
#define MSHQC_BASIS_H

#include "mshqc/molecule.h"
#include <vector>
#include <string>
#include <array>
#include <memory>

/**
 * @file basis.h
 * @brief Gaussian basis set representation
 * 
 * REFERENCES:
 * Szabo & Ostlund (1996), Chapter 3, Appendix A
 * Helgaker et al. (2000), Chapter 9, pp. 315-362
 * Hehre et al., J. Chem. Phys. 51, 2657 (1969) - STO-3G
 * 
 * BASIS DATA: EMSL Basis Set Exchange (public domain)
 */

namespace mshqc {

/**
 * @brief Angular momentum quantum number and shell types
 * 
 * Standard spectroscopic notation:
 * - s: l = 0 (1 function)
 * - p: l = 1 (3 functions: px, py, pz)
 * - d: l = 2 (5 or 6 functions, spherical or cartesian)
 * - f: l = 3 (7 or 10 functions)
 * 
 * REFERENCE:
 * Szabo & Ostlund (1996), Table 3.1, p. 157
 */
enum class AngularMomentum {
    S = 0,  ///< s orbital (l=0)
    P = 1,  ///< p orbital (l=1)
    D = 2,  ///< d orbital (l=2)
    F = 3,  ///< f orbital (l=3)
    G = 4,  ///< g orbital (l=4)
    H = 5   ///< h orbital (l=5)
};

/**
 * @brief Convert angular momentum enum to integer
 */
inline int am_to_int(AngularMomentum am) {
    return static_cast<int>(am);
}

/**
 * @brief Number of cartesian basis functions for given angular momentum
 * 
 * REFERENCE: Helgaker et al. (2000), Eq. (9.2.2), p. 316
 * n_cart(l) = (l+1)(l+2)/2 (e.g., s:1, p:3, d:6, f:10)
 */
inline int n_cartesian_functions(AngularMomentum am) {
    int l = am_to_int(am);
    return (l + 1) * (l + 2) / 2;
}

/**
 * @brief Number of spherical basis functions for given angular momentum
 * 
 * REFERENCE: Helgaker et al. (2000), Section 9.2, p. 317
 * n_sph(l) = 2l + 1 (e.g., s:1, p:3, d:5, f:7)
 */
inline int n_spherical_functions(AngularMomentum am) {
    int l = am_to_int(am);
    return 2 * l + 1;
}

// ============================================================================
// Gaussian Primitive
// ============================================================================

/**
 * @brief A single Gaussian primitive function
 * 
 * REFERENCES:
 * Szabo & Ostlund (1996), Eq. (3.203), p. 153
 * Helgaker et al. (2000), Eq. (9.2.5), p. 318
 * 
 * Form: g(r; α, l, m, n) = N × x^l y^m z^n exp(-α r²)
 * Normalization (s-type): N = (2α/π)^(3/4)
 */
struct GaussianPrimitive {
    double exponent;   ///< Gaussian exponent α (or zeta)
    double coefficient; ///< Contraction coeff d_i
    
    GaussianPrimitive(double exp, double coef)
        : exponent(exp), coefficient(coef) {}
};

// ============================================================================
// Contracted Gaussian (Shell)
// ============================================================================

/**
 * @brief A contracted Gaussian basis function (shell)
 * 
 * REFERENCES:
 * Szabo & Ostlund (1996), Section 3.5, p. 178
 * Helgaker et al. (2000), Section 9.2.2, p. 319
 * 
 * Contracted: φ(r) = Σ_i d_i × g_i(r; α_i)
 * Shell types: S, P, D, F, SP (s+p combined in Pople sets)
 */
class Shell {
public:
    /**
     * @brief Construct shell with given angular momentum
     * @param am Angular momentum (S, P, D, etc.)
     * @param center Atom index in molecule (which atom this shell is on)
     * @param center_pos Cartesian position of shell center (Bohr)
     */
    Shell(AngularMomentum am, int center, const std::array<double, 3>& center_pos);
    
    /**
     * @brief Add primitive Gaussian to this shell
     * @param exponent Gaussian exponent α
     * @param coeff Contraction coefficient
     */
    void add_primitive(double exponent, double coeff);
    
    /// Get angular momentum
    AngularMomentum angular_momentum() const { return am_; }
    
    /// Get angular momentum as integer
    int l() const { return am_to_int(am_); }
    
    /// Get center atom index
    int center() const { return center_; }
    
    /// Get shell center position
    const std::array<double, 3>& position() const { return position_; }
    
    /// Get number of primitives in this shell
    size_t n_primitives() const { return primitives_.size(); }
    
    /// Get primitive by index
    const GaussianPrimitive& primitive(size_t i) const { return primitives_[i]; }
    
    /// Get all primitives
    const std::vector<GaussianPrimitive>& primitives() const { return primitives_; }
    
    /**
     * @brief Get number of basis functions in this shell
     * (spherical: 2l+1, cartesian: (l+1)(l+2)/2)
     */
    int n_functions() const;
    
    /**
     * @brief Normalize shell (ensures ⟨φ|φ⟩ = 1)
     * 
     * REFERENCE: Helgaker et al. (2000), Eq. (9.2.8), p. 320
     */
    void normalize();
    
    /// Check if using spherical harmonics (vs cartesian)
    bool is_spherical() const { return spherical_; }
    
    /// Set spherical/cartesian mode
    void set_spherical(bool sph) { spherical_ = sph; }
    
private:
    AngularMomentum am_;                      ///< Angular momentum
    int center_;                              ///< Atom index
    std::array<double, 3> position_;          ///< Shell center (Bohr)
    std::vector<GaussianPrimitive> primitives_; ///< List of primitives
    bool spherical_;                          ///< Use spherical harmonics?
};

// ============================================================================
// Basis Set
// ============================================================================

/**
 * @brief Complete basis set for a molecule
 * 
 * REFERENCES:
 * Hehre et al., J. Chem. Phys. 51, 2657 (1969) - STO-3G
 * Hehre et al., J. Chem. Phys. 56, 2257 (1972) - 6-31G
 * Dunning, J. Chem. Phys. 90, 1007 (1989) - cc-pVXZ
 * 
 * Reads .gbs format (EMSL/Psi4). Common sets: STO-3G, 6-31G, cc-pVDZ/TZ
 */
class BasisSet {
public:
    /**
     * @brief Construct empty basis set
     */
    BasisSet();
    
    /**
     * @brief Construct basis set for molecule from library
     * @param basis_name Name of basis (e.g., "sto-3g", "6-31g")
     * @param mol Molecule to build basis for
     * @param basis_dir Directory containing .gbs files
     */
    BasisSet(const std::string& basis_name, 
             const Molecule& mol,
             const std::string& basis_dir = "../data/basis");
    
    /**
     * @brief Build basis set from .gbs file
     * @param basis_file Path to .gbs file
     * @param mol Molecule
     * @return true if successful
     */
    bool read_gbs(const std::string& basis_file, const Molecule& mol);
    
    /**
     * @brief Add shell to basis set
     * @param shell Shell to add
     */
    void add_shell(const Shell& shell);
    
    /// Get total number of shells
    size_t n_shells() const { return shells_.size(); }
    
    /// Get shell by index
    const Shell& shell(size_t i) const { return shells_[i]; }
    
    /// Get all shells
    const std::vector<Shell>& shells() const { return shells_; }
    
    /**
     * @brief Get total number of basis functions (Σ_i n_functions(shell_i))
     */
    size_t n_basis_functions() const;
    
    /// Get basis set name
    const std::string& name() const { return name_; }
    
    /// Set basis set name
    void set_name(const std::string& name) { name_ = name; }
    
    /// Check if using spherical harmonics
    bool is_spherical() const { return spherical_; }
    
    /// Set spherical/cartesian mode for all shells
    void set_spherical(bool sph);
    
    /**
     * @brief Print basis set information (shells, angular momentum, primitives)
     */
    void print() const;
    
    /**
     * @brief Get maximum angular momentum in basis set
     */
    int max_angular_momentum() const;
    
    /**
     * @brief Get shell-to-basis-function map
     * (element i = starting basis function index for shell i)
     */
    std::vector<int> shell_to_basis_function_map() const;
    
private:
    std::string name_;              ///< Basis set name
    std::vector<Shell> shells_;     ///< List of shells
    bool spherical_;                ///< Use spherical harmonics?
    
    /**
     * @brief Parse single atom's basis from .gbs file stream
     * @param file Input file stream
     * @param atom_symbol Element symbol to match
     * @param atom_index Index of atom in molecule
     * @param atom_pos Position of atom
     * @return Number of shells added
     */
    int parse_atom_basis(std::ifstream& file, 
                        const std::string& atom_symbol,
                        int atom_index,
                        const std::array<double, 3>& atom_pos);
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Convert angular momentum character to enum
 * @param c Character ('S', 'P', 'D', 'F', etc.)
 * @return AngularMomentum enum
 */
AngularMomentum char_to_am(char c);

/**
 * @brief Convert angular momentum to spectroscopic notation
 * @param am Angular momentum
 * @return String ("s", "p", "d", "f", etc.)
 */
std::string am_to_string(AngularMomentum am);

/**
 * @brief Compute normalization constant for s-type primitive Gaussian
 * 
 * REFERENCE: Szabo & Ostlund (1996), Eq. (3.203), p. 153
 * N = (2α/π)^(3/4) ensures ⟨g|g⟩ = 1
 */
double gaussian_normalization_s(double alpha);

/**
 * @brief Compute overlap integral between two s-type primitive Gaussians
 * 
 * REFERENCE: Szabo & Ostlund (1996), Eq. (A.9), Appendix A, p. 411
 * S = (π/(α_A + α_B))^(3/2) × exp(-α_A α_B R²/(α_A + α_B))
 */
double primitive_overlap_s(double alpha_a, double alpha_b,
                          const std::array<double, 3>& Ra,
                          const std::array<double, 3>& Rb);

/**
 * @brief Get element symbol from atomic number
 * @param Z Atomic number
 * @return Element symbol (e.g., "H", "Li", "C")
 */
std::string get_element_symbol(int Z);

} // namespace mshqc

#endif // MSHQC_BASIS_H
