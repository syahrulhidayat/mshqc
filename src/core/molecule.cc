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

















#include "mshqc/core/molecule.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdexcept>
#ifdef I
#undef I
#endif

namespace mshqc {

void Molecule::add_atom(int Z, double x, double y, double z) {
    atoms_.emplace_back(Z, x, y, z);
}

void Molecule::add_atom(const Atom& atom) {
    atoms_.push_back(atom);
}

int Molecule::total_nuclear_charge() const {
    int total = 0;
    for (const auto& atom : atoms_) {
        total += atom.atomic_number;
    }
    return total;
}

double Molecule::nuclear_repulsion_energy() const {

    double e_nuc = 0.0;

    size_t n = atoms_.size();
    for (size_t A = 0; A < n; A++) {
        for (size_t B = A + 1; B < n; B++) {
            double dx = atoms_[A].x - atoms_[B].x;
            double dy = atoms_[A].y - atoms_[B].y;
            double dz = atoms_[A].z - atoms_[B].z;
            double r_AB = std::sqrt(dx*dx + dy*dy + dz*dz);

            double ZA = atoms_[A].atomic_number;
            double ZB = atoms_[B].atomic_number;
            e_nuc += ZA * ZB / r_AB;
        }
    }

    return e_nuc;
}

std::array<double, 3> Molecule::center_of_mass() const {

    double total_m = 0.0;
    double com_x = 0.0, com_y = 0.0, com_z = 0.0;

    for (const auto& atom : atoms_) {
        double mass = get_atomic_mass(atom.atomic_number);
        com_x += mass * atom.x;
        com_y += mass * atom.y;
        com_z += mass * atom.z;
        total_m += mass;
    }

    if (total_m > 0.0) {
        com_x /= total_m;
        com_y /= total_m;
        com_z /= total_m;
    }

    return {com_x, com_y, com_z};
}

double Molecule::total_mass() const {
    double mass = 0.0;
    for (const auto& atom : atoms_) {
        mass += get_atomic_mass(atom.atomic_number);
    }
    return mass;
}

void Molecule::translate(double dx, double dy, double dz) {
    for (auto& atom : atoms_) {
        atom.x += dx;
        atom.y += dy;
        atom.z += dz;
    }
}

void Molecule::move_to_com() {
    auto com = center_of_mass();
    translate(-com[0], -com[1], -com[2]);
}

bool Molecule::read_xyz(const std::string& filename) {

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    int natom;
    std::string line;

    if (!(file >> natom)) {
        std::cerr << "Error: Cannot read number of atoms" << std::endl;
        return false;
    }
    std::getline(file, line);

    std::getline(file, line);

    atoms_.clear();

    for (int i = 0; i < natom; i++) {
        std::string symbol;
        double x, y, z;

        if (!(file >> symbol >> x >> y >> z)) {
            std::cerr << "Error: Cannot read atom " << i+1 << std::endl;
            return false;
        }

        x *= ANGSTROM_TO_BOHR;
        y *= ANGSTROM_TO_BOHR;
        z *= ANGSTROM_TO_BOHR;

        int Z = 0;
        if(symbol == "H") Z = 1;
        else if(symbol == "He") Z = 2;
        else if(symbol == "Li") Z = 3;
        else if(symbol == "Be") Z = 4;
        else if(symbol == "B") Z = 5;
        else if(symbol == "C") Z = 6;
        else if(symbol == "N") Z = 7;
        else if(symbol == "O") Z = 8;
        else if(symbol == "F") Z = 9;
        else if(symbol == "Ne") Z = 10;
        else {
            std::cerr << "Error: Unknown element symbol " << symbol << std::endl;
            return false;
        }

        add_atom(Z, x, y, z);
    }
    move_to_com();

    return true;
}

void Molecule::print() const {
    std::cout << "\n";
    std::cout << "============================================\n";
    std::cout << "           MOLECULAR GEOMETRY\n";
    std::cout << "============================================\n";
    std::cout << std::fixed << std::setprecision(8);

    std::cout << "\nNumber of atoms: " << n_atoms() << "\n";
    std::cout << "Charge: " << charge_ << "\n";
    std::cout << "Multiplicity: " << multiplicity_ << " (";
    int n_unpaired = multiplicity_ - 1;
    std::cout << n_unpaired << " unpaired electron";
    if (n_unpaired != 1) std::cout << "s";
    std::cout << ")\n";
    std::cout << "Number of electrons: " << n_electrons() << "\n\n";

    std::cout << "Coordinates (Bohr):\n";
    std::cout << "  Atom       X              Y              Z\n";
    std::cout << "  ----  -----------    -----------    -----------\n";
    for (const auto& atom : atoms_) {
        std::cout << "  " << std::setw(2) << get_element_symbol(atom.atomic_number)
                  << "    " << std::setw(12) << atom.x
                  << "   " << std::setw(12) << atom.y
                  << "   " << std::setw(12) << atom.z << "\n";
    }

    std::cout << "\nCoordinates (Angstrom):\n";
    std::cout << "  Atom       X              Y              Z\n";
    std::cout << "  ----  -----------    -----------    -----------\n";
    for (const auto& atom : atoms_) {
        std::cout << "  " << std::setw(2) << get_element_symbol(atom.atomic_number)
                  << "    " << std::setw(12) << atom.x * BOHR_TO_ANGSTROM
                  << "   " << std::setw(12) << atom.y * BOHR_TO_ANGSTROM
                  << "   " << std::setw(12) << atom.z * BOHR_TO_ANGSTROM << "\n";
    }

    std::cout << std::setprecision(10);
    std::cout << "\nNuclear repulsion energy: " << nuclear_repulsion_energy()
              << " Hartree\n";
    std::cout << "Total mass: " << total_mass() << " amu\n";

    auto com = center_of_mass();
    std::cout << "\nCenter of mass (Bohr): ("
              << com[0] << ", " << com[1] << ", " << com[2] << ")\n";

    std::cout << "============================================\n\n";
}

double Molecule::get_atomic_mass(int Z) const {

    static const double atomic_masses[] = {
        0.0,
        1.008,
        4.0026,
        6.94,
        9.0122,
        10.81,
        12.011,
        14.007,
        15.999,
        18.998,
        20.180,
        22.990,
        24.305,
        26.982,
        28.085,
        30.974,
        32.06,
        35.45,
        39.948
    };

    if (Z < 0 || Z >= static_cast<int>(sizeof(atomic_masses)/sizeof(double))) {
        throw std::runtime_error("Atomic number out of range: " + std::to_string(Z));
    }

    return atomic_masses[Z];
}

std::string Molecule::get_element_symbol(int Z) const {

    static const char* symbols[] = {
        "X",
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar"
    };

    if (Z < 0 || Z >= static_cast<int>(sizeof(symbols)/sizeof(char*))) {
        return "?";
    }

    return symbols[Z];
}

}
