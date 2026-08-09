 // ==============================================================================
 // Copyright (c) 2026 Syahrul and mshqc contributors
 //
 // Licensed under the Apache License, Version 2.0 (the "License");
 // you may not use this file except in compliance with the License.
 // You may obtain a copy of the License at
 //
 //     http://www.apache.org/licenses/LICENSE-2.0
 //
 // Unless required by applicable law or agreed to in writing, software
 // distributed under the License is distributed on an "AS IS" BASIS,
 // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 // See the License for the specific language governing permissions and
 // limitations under the License.
 // ==============================================================================

#ifndef MSHQC_BASIS_H
#define MSHQC_BASIS_H

#include "mshqc/core/molecule.h"
#include <vector>
#include <string>
#include <array>
#include <memory>
#include <fstream>

#ifdef I
#undef I
#endif

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

class Shell {
public:
    Shell(AngularMomentum am, int center, const std::array<double, 3>& center_pos);

    void add_primitive(double exponent, double coeff);

    AngularMomentum angular_momentum() const { return am_; }
    int l() const { return am_to_int(am_); }
    int center() const { return center_; }
    const std::array<double, 3>& position() const { return position_; }

    std::vector<double> origin() const {
        return {position_[0], position_[1], position_[2]};
    }

    size_t n_primitives() const { return primitives_.size(); }
    const GaussianPrimitive& primitive(size_t i) const { return primitives_[i]; }
    const std::vector<GaussianPrimitive>& primitives() const { return primitives_; }

    int center_index() const { return center_; }
    int n_functions() const;
    void normalize();

    bool is_spherical() const { return spherical_; }
    void set_spherical(bool sph) { spherical_ = sph; }
    void set_cartesian(bool cart) { spherical_ = !cart; }

private:
    AngularMomentum am_;
    int center_;
    std::array<double, 3> position_;
    std::vector<GaussianPrimitive> primitives_;
    bool spherical_;
};

class BasisSet {
public:
    BasisSet();
    BasisSet(const std::string& basis_name,
             const Molecule& mol,
             const std::string& basis_dir = "data/basis");

    bool read_gbs(const std::string& basis_file, const Molecule& mol);
    void add_shell(const Shell& shell);

    size_t n_shells() const { return shells_.size(); }
    const Shell& shell(size_t i) const { return shells_[i]; }
    const std::vector<Shell>& shells() const { return shells_; }

    size_t n_basis_functions() const;

    const std::string& name() const { return name_; }
    void set_name(const std::string& name) { name_ = name; }

    bool is_spherical() const { return spherical_; }
    void set_spherical(bool sph);
    void set_cartesian(bool cart) { spherical_ = !cart; }

    void print() const;
    int max_angular_momentum() const;
    std::vector<int> shell_to_basis_function_map() const;
    void append(const BasisSet& other);

private:
    std::string name_;
    std::vector<Shell> shells_;
    bool spherical_;
    size_t n_basis_ = 0;

    int parse_atom_basis(std::ifstream& file,
                        const std::string& atom_symbol,
                        int atom_index,
                        const std::array<double, 3>& atom_pos);
};

AngularMomentum char_to_am(char c);
std::string am_to_string(AngularMomentum am);
double gaussian_normalization_s(double alpha);
double primitive_overlap_s(double alpha_a, double alpha_b,
                          const std::array<double, 3>& Ra,
                          const std::array<double, 3>& Rb);
std::string get_element_symbol(int Z);

}

#endif
