/**
 * @file molecule.cc
 * @brief Molecular system representation and geometry management
 * 
 * Implementation of Molecule class for storing atomic coordinates,
 * computing nuclear repulsion energy, and handling molecular I/O.
 * 
 * Theory References:
 *   - A. Szabo & N. S. Ostlund, "Modern Quantum Chemistry" (1996)
 *     [Eq. (1.45), p. 41: Nuclear repulsion energy formula]
 *   - Classical mechanics references (center of mass, molecular mass)
 *   - XYZ file format (standard molecular geometry format)
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-01-29
 * @license MIT License (see LICENSE file in project root)
 * 
 * @note This is an original implementation derived from standard theory.
 *       Nuclear repulsion: E_nuc = \u03a3_(A<B) Z_A Z_B / R_AB (atomic units).
 *       XYZ reader converts Angstrom to Bohr automatically.
 */

#include "mshqc/molecule.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mshqc {

// ============================================================================
// Atom management
// ============================================================================

void Molecule::add_atom(int Z, double x, double y, double z) {
    atoms_.emplace_back(Z, x, y, z);
}

void Molecule::add_atom(const Atom& atom) {
    atoms_.push_back(atom);
}

// ============================================================================
// Electronic properties
// ============================================================================

int Molecule::total_nuclear_charge() const {
    int total = 0;
    for (const auto& atom : atoms_) {
        total += atom.atomic_number;
    }
    return total;
}

// ============================================================================
// Nuclear properties
// ============================================================================

double Molecule::nuclear_repulsion_energy() const {
    // REFERENCE: Szabo & Ostlund (1996), Eq. (1.45), p. 41
    // Nuclear repulsion: E_nuc = Σ_(A<B) Z_A Z_B / R_AB
    // Classical Coulomb repulsion in atomic units
    
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
    // Center of mass: R_cm = Σ(m_i R_i) / Σ(m_i)
    // Standard classical mechanics (not QC-specific)
    
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

// ============================================================================
// I/O operations
// ============================================================================

bool Molecule::read_xyz(const std::string& filename) {
    // XYZ format: line 1 = natom, line 2 = comment, lines 3+ = Symbol X Y Z
    // Converts Angstrom → Bohr automatically
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    
    int natom;
    std::string line;
    
    // Read number of atoms
    if (!(file >> natom)) {
        std::cerr << "Error: Cannot read number of atoms" << std::endl;
        return false;
    }
    std::getline(file, line); // consume rest of line
    
    // Skip comment line
    std::getline(file, line);
    
    // Clear existing atoms
    atoms_.clear();
    
    // Read atoms
    for (int i = 0; i < natom; i++) {
        std::string symbol;
        double x, y, z;
        
        if (!(file >> symbol >> x >> y >> z)) {
            std::cerr << "Error: Cannot read atom " << i+1 << std::endl;
            return false;
        }
        
        // Angstrom → Bohr
        x *= ANGSTROM_TO_BOHR;
        y *= ANGSTROM_TO_BOHR;
        z *= ANGSTROM_TO_BOHR;
        
        // Symbol → atomic number
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

// ============================================================================
// Private helper functions
// ============================================================================

double Molecule::get_atomic_mass(int Z) const {
    // Atomic masses from NIST (2021 standard atomic weights)
    // https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions
    static const double atomic_masses[] = {
        0.0,        // Z=0 (placeholder)
        1.008,      // H
        4.0026,     // He
        6.94,       // Li
        9.0122,     // Be
        10.81,      // B
        12.011,     // C
        14.007,     // N
        15.999,     // O
        18.998,     // F
        20.180,     // Ne
        22.990,     // Na
        24.305,     // Mg
        26.982,     // Al
        28.085,     // Si
        30.974,     // P
        32.06,      // S
        35.45,      // Cl
        39.948      // Ar
    };
    
    if (Z < 0 || Z >= static_cast<int>(sizeof(atomic_masses)/sizeof(double))) {
        throw std::runtime_error("Atomic number out of range: " + std::to_string(Z));
    }
    
    return atomic_masses[Z];
}

std::string Molecule::get_element_symbol(int Z) const {
    /**
     * Periodic table element symbols
     * 
     * Standard IUPAC element symbols.
     */
    
    static const char* symbols[] = {
        "X",   // Z=0 (placeholder)
        "H",   // Z=1
        "He",  // Z=2
        "Li",  // Z=3
        "Be",  // Z=4
        "B",   // Z=5
        "C",   // Z=6
        "N",   // Z=7
        "O",   // Z=8
        "F",   // Z=9
        "Ne",  // Z=10
        "Na",  // Z=11
        "Mg",  // Z=12
        "Al",  // Z=13
        "Si",  // Z=14
        "P",   // Z=15
        "S",   // Z=16
        "Cl",  // Z=17
        "Ar"   // Z=18
    };
    
    if (Z < 0 || Z >= static_cast<int>(sizeof(symbols)/sizeof(char*))) {
        return "?";
    }
    
    return symbols[Z];
}

} // namespace mshqc
