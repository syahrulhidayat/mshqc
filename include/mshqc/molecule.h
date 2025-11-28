#ifndef MSHQC_MOLECULE_H
#define MSHQC_MOLECULE_H

#include <vector>
#include <string>
#include <array>

/**
 * @file molecule.h
 * @brief Molecular geometry and nuclear properties
 * 
 * REFERENCES:
 * Szabo & Ostlund (1996) - Modern Quantum Chemistry
 * Helgaker et al. (2000) - Molecular Electronic-Structure Theory
 */

namespace mshqc {

/**
 * @brief Single atom: atomic number + position
 * Units: Bohr (atomic units)
 */
struct Atom {
    int atomic_number;  // Nuclear charge Z
    double x, y, z;     // Cartesian coords (Bohr)
    
    Atom(int Z, double x_pos, double y_pos, double z_pos)
        : atomic_number(Z), x(x_pos), y(y_pos), z(z_pos) {}
    
    std::array<double, 3> position() const { return {x, y, z}; }
};

/**
 * @brief Molecule: collection of atoms + electronic properties
 */
class Molecule {
public:
    Molecule() : atoms_(), charge_(0), multiplicity_(1) {}
    
    Molecule(int charge, int multiplicity) 
        : atoms_(), charge_(charge), multiplicity_(multiplicity) {}
    
    // ========================================================================
    // Atom management
    // ========================================================================
    
    /**
     * @brief Add atom to molecule
     * @param Z Atomic number
     * @param x X coordinate in Bohr
     * @param y Y coordinate in Bohr
     * @param z Z coordinate in Bohr
     */
    void add_atom(int Z, double x, double y, double z);
    
    /**
     * @brief Add atom to molecule
     * @param atom Atom object to add
     */
    void add_atom(const Atom& atom);
    
    /// Get number of atoms
    size_t n_atoms() const { return atoms_.size(); }
    
    /// Get atom by index
    const Atom& atom(size_t i) const { return atoms_[i]; }
    
    /// Get all atoms
    const std::vector<Atom>& atoms() const { return atoms_; }
    
    // ========================================================================
    // Electronic properties
    // ========================================================================
    
    /// Get total nuclear charge
    int total_nuclear_charge() const;
    
    /// Get number of electrons (nuclear charge - molecular charge)
    int n_electrons() const { return total_nuclear_charge() - charge_; }
    
    /// Get molecular charge
    int charge() const { return charge_; }
    
    /// Set molecular charge
    void set_charge(int q) { charge_ = q; }
    
    /// Get spin multiplicity (2S+1)
    int multiplicity() const { return multiplicity_; }
    
    /// Set spin multiplicity
    void set_multiplicity(int m) { multiplicity_ = m; }
    
    // ========================================================================
    // Nuclear properties
    // ========================================================================
    
    /**
     * @brief Nuclear repulsion energy
     * 
     * REFERENCE: Szabo & Ostlund (1996), Eq. (1.45), p. 41
     * E_nuc = Σ_(A<B) Z_A Z_B / R_AB
     * 
     * @return Energy in Hartree
     */
    double nuclear_repulsion_energy() const;
    
    /**
     * @brief Center of mass
     * R_cm = Σ(m_i R_i) / Σ(m_i)
     * Standard classical mechanics
     */
    std::array<double, 3> center_of_mass() const;
    
    /**
     * @brief Compute total mass
     * 
     * Uses standard atomic masses from NIST Atomic Weights.
     * 
     * @return Total molecular mass in atomic mass units (amu)
     */
    double total_mass() const;
    
    /**
     * @brief Translate molecule by given vector
     * @param dx Translation in x direction (Bohr)
     * @param dy Translation in y direction (Bohr)
     * @param dz Translation in z direction (Bohr)
     */
    void translate(double dx, double dy, double dz);
    
    /**
     * @brief Move molecule so center of mass is at origin
     */
    void move_to_com();
    
    // ========================================================================
    // I/O operations
    // ========================================================================
    
    /**
     * @brief Read molecule from XYZ file format
     * 
     * XYZ format:
     *   Line 1: Number of atoms
     *   Line 2: Comment line
     *   Lines 3+: Symbol X Y Z (coordinates in Angstrom)
     * 
     * Automatically converts Angstrom → Bohr (multiply by 1.88972612457)
     * 
     * @param filename Path to XYZ file
     * @return true if successful, false otherwise
     */
    bool read_xyz(const std::string& filename);
    
    /**
     * @brief Print molecule information
     * 
     * Prints atomic coordinates, charge, multiplicity, and nuclear properties.
     */
    void print() const;
    
private:
    std::vector<Atom> atoms_;    ///< List of atoms in molecule
    int charge_;                 ///< Total molecular charge
    int multiplicity_;           ///< Spin multiplicity (2S+1)
    
    /**
     * @brief Get atomic mass from atomic number
     * 
     * Uses standard atomic masses from NIST.
     * Data source: NIST Atomic Weights and Isotopic Compositions
     * https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions
     * 
     * @param Z Atomic number
     * @return Atomic mass in amu (atomic mass units)
     */
    double get_atomic_mass(int Z) const;
    
    /**
     * @brief Get element symbol from atomic number
     * @param Z Atomic number
     * @return Element symbol (e.g., "H", "Li", "C")
     */
    std::string get_element_symbol(int Z) const;
};

// ============================================================================
// Constants
// ============================================================================

/// Conversion factor: Angstrom to Bohr
constexpr double ANGSTROM_TO_BOHR = 1.88972612457;

/// Conversion factor: Bohr to Angstrom
constexpr double BOHR_TO_ANGSTROM = 0.529177210903;

} // namespace mshqc

#endif // MSHQC_MOLECULE_H
