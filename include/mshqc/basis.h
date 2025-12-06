#ifndef MSHQC_BASIS_H
#define MSHQC_BASIS_H

#include "mshqc/molecule.h"
#include <vector>
#include <string>
#include <array>
#include <memory>
#include <fstream> // Tambahkan fstream karena dipakai di parse_atom_basis

namespace mshqc {

enum class AngularMomentum {
    S = 0, P = 1, D = 2, F = 3, G = 4, H = 5
};

inline int am_to_int(AngularMomentum am) {
    return static_cast<int>(am);
}

inline int n_cartesian_functions(AngularMomentum am) {
    int l = am_to_int(am);
    return (l + 1) * (l + 2) / 2;
}

inline int n_spherical_functions(AngularMomentum am) {
    int l = am_to_int(am);
    return 2 * l + 1;
}

struct GaussianPrimitive {
    double exponent;
    double coefficient;
    GaussianPrimitive(double exp, double coef) : exponent(exp), coefficient(coef) {}
};

// ============================================================================
// Shell Class
// ============================================================================
class Shell {
public:
    Shell(AngularMomentum am, int center, const std::array<double, 3>& center_pos);
    
    void add_primitive(double exponent, double coeff);
    
    AngularMomentum angular_momentum() const { return am_; }
    int l() const { return am_to_int(am_); }
    int center() const { return center_; }
    const std::array<double, 3>& position() const { return position_; }
    
    size_t n_primitives() const { return primitives_.size(); }
    const GaussianPrimitive& primitive(size_t i) const { return primitives_[i]; }
    const std::vector<GaussianPrimitive>& primitives() const { return primitives_; }
    
    int n_functions() const;
    void normalize();
    
    bool is_spherical() const { return spherical_; }
    void set_spherical(bool sph) { spherical_ = sph; }

private:
    AngularMomentum am_;
    int center_;
    std::array<double, 3> position_;
    std::vector<GaussianPrimitive> primitives_;
    bool spherical_;
};

// ============================================================================
// BasisSet Class
// ============================================================================
class BasisSet {
public:
    BasisSet();
    BasisSet(const std::string& basis_name, 
             const Molecule& mol,
             const std::string& basis_dir = "data/basis");
    
    // Core Functions (Hanya satu deklarasi!)
    bool read_gbs(const std::string& basis_file, const Molecule& mol);
    void add_shell(const Shell& shell);
    
    // Accessors
    size_t n_shells() const { return shells_.size(); }
    const Shell& shell(size_t i) const { return shells_[i]; }
    const std::vector<Shell>& shells() const { return shells_; }
    
    // Total basis functions count
    size_t n_basis_functions() const;
    
    const std::string& name() const { return name_; }
    void set_name(const std::string& name) { name_ = name; }
    
    bool is_spherical() const { return spherical_; }
    void set_spherical(bool sph);
    
    void print() const;
    int max_angular_momentum() const;
    std::vector<int> shell_to_basis_function_map() const;
    
private:
    std::string name_;
    std::vector<Shell> shells_;
    bool spherical_;
    size_t n_basis_ = 0;

    // Helper function (private)
    int parse_atom_basis(std::ifstream& file, 
                        const std::string& atom_symbol,
                        int atom_index,
                        const std::array<double, 3>& atom_pos);
};

// ============================================================================
// Utility Functions
// ============================================================================
AngularMomentum char_to_am(char c);
std::string am_to_string(AngularMomentum am);
double gaussian_normalization_s(double alpha);
double primitive_overlap_s(double alpha_a, double alpha_b,
                          const std::array<double, 3>& Ra,
                          const std::array<double, 3>& Rb);
std::string get_element_symbol(int Z);

} // namespace mshqc

#endif // MSHQC_BASIS_H