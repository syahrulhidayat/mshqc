 // ==============================================================================
 // Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
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

#ifndef MSHQC_MOLECULE_H
#define MSHQC_MOLECULE_H

#include <vector>
#include <string>
#include <array>
#ifdef I
#undef I
#endif

namespace mshqc
{

    struct Atom
    {
        int atomic_number;

        double x, y, z;

        Atom(int Z, double x_pos, double y_pos, double z_pos)
            : atomic_number(Z), x(x_pos), y(y_pos), z(z_pos) {}

        std::array<double, 3> position() const { return {x, y, z}; }
    };

    class Molecule
    {
    public:
        Molecule() : atoms_(), charge_(0), multiplicity_(1) {}

        Molecule(int charge, int multiplicity)
            : atoms_(), charge_(charge), multiplicity_(multiplicity) {}

        void add_atom(int Z, double x, double y, double z);

        void add_atom(const Atom &atom);

        size_t n_atoms() const { return atoms_.size(); }

        const Atom &atom(size_t i) const { return atoms_[i]; }

        const std::vector<Atom> &atoms() const { return atoms_; }

        int total_nuclear_charge() const;

        int n_electrons() const { return total_nuclear_charge() - charge_; }

        int charge() const { return charge_; }

        void set_charge(int q) { charge_ = q; }

        int multiplicity() const { return multiplicity_; }

        void set_multiplicity(int m) { multiplicity_ = m; }

        double nuclear_repulsion_energy() const;

        std::array<double, 3> center_of_mass() const;

        double total_mass() const;

        void translate(double dx, double dy, double dz);

        void move_to_com();

        bool read_xyz(const std::string &filename);

        void print() const;

    private:
        std::vector<Atom> atoms_;

        int charge_;

        int multiplicity_;

        double get_atomic_mass(int Z) const;

        std::string get_element_symbol(int Z) const;
    };

    constexpr double ANGSTROM_TO_BOHR = 1.88972612457;

    constexpr double BOHR_TO_ANGSTROM = 0.529177210903;

}

#endif
