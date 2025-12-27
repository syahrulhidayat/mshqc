

#/**
 * @file active_space.h
 * @brief Active space definition for CASSCF calculations
 * * Represents CAS(n,m) - n electrons in m orbitals
 * Divides MOs into: inactive (doubly-occ) | active (partial) | virtual (empty)
 * * THEORY REFERENCES:
 * - B. O. Roos et al., Chem. Phys. 48, 157 (1980)
 * - P. E. M. Siegbahn et al., Physica Scripta 21, 323 (1980)
 * - T. Helgaker et al., "Molecular Electronic Structure Theory" (2000), Ch. 14
 * * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 * @license MIT
 * * @note Original implementation from published CASSCF theory.
 * No code copied from existing quantum chemistry software.
 */

#ifndef MSHQC_MCSCF_ACTIVE_SPACE_H
#define MSHQC_MCSCF_ACTIVE_SPACE_H

#include <vector>
#include <string>
#include <stdexcept>

namespace mshqc {
namespace mcscf {

/**
 * @brief Active space specification for multi-configurational methods
 * * Example: H2O with CAS(8,6)
 * - Inactive: 1 orbital (O 1s core, always doubly-occupied)
 * - Active: 6 orbitals (O 2s, 2p, H bonding/antibonding)
 * - Virtual: rest (unoccupied)
 * - Active electrons: 8 (from 10 valence - 2 in inactive)
 */
class ActiveSpace {
public:
    // Constructors
    ActiveSpace() : n_inact_(0), n_act_(0), n_virt_(0), n_elec_act_(0) {}
    
    /**
     * @brief Construct active space from orbital counts
     * @param n_inactive Number of inactive (frozen core) orbitals
     * @param n_active Number of active orbitals
     * @param n_virtual Number of virtual (unoccupied) orbitals
     * @param n_elec_active Number of electrons in active space
     */
    ActiveSpace(int n_inactive, int n_active, int n_virtual, int n_elec_active);
    
    /**
     * @brief Standard CAS Construction: CAS(n,m)
     * @details Create Active Space by specifying ACTIVE electrons and ACTIVE orbitals.
     * Inactive orbitals are calculated automatically: (TotalElec - ActiveElec)/2
     * * @param n_elec Number of electrons in active space (e.g., 8 for water)
     * @param n_orb Number of active orbitals (e.g., 6)
     * @param n_total_orb Total number of molecular orbitals
     * @param n_total_elec Total number of electrons
     */
    static ActiveSpace CAS(int n_elec, int n_orb, 
                           int n_total_orb, int n_total_elec);

    /**
     * @brief Frozen Core Construction
     * @details Create Active Space by specifying FROZEN (Inactive) orbitals.
     * Active electrons are calculated automatically: TotalElec - 2*FrozenOrb.
     * Useful for inputs where Frozen Core is the primary parameter.
     * * @param n_frozen_orb Number of frozen/inactive orbitals (e.g., 0 for Li full)
     * @param n_active_orb Number of active orbitals
     * @param n_total_orb Total number of molecular orbitals
     * @param n_total_elec Total number of electrons
     */
    static ActiveSpace CAS_Frozen(int n_frozen_orb, int n_active_orb,
                                  int n_total_orb, int n_total_elec);
    
    // Accessors
    int n_inactive() const { return n_inact_; }
    int n_active() const { return n_act_; }
    int n_virtual() const { return n_virt_; }
    int n_elec_active() const { return n_elec_act_; }
    int n_total_orb() const { return n_inact_ + n_act_ + n_virt_; }
    
    /**
     * @brief Get global MO indices of inactive orbitals
     * @return Vector of indices [0, 1, ..., n_inactive-1]
     */
    std::vector<int> inactive_indices() const;
    
    /**
     * @brief Get global MO indices of active orbitals
     * @return Vector of indices [n_inactive, n_inactive+1, ..., n_inactive+n_active-1]
     */
    std::vector<int> active_indices() const;
    
    /**
     * @brief Get global MO indices of virtual orbitals
     * @return Vector of indices [n_inactive+n_active, ..., n_total_orb-1]
     */
    std::vector<int> virtual_indices() const;
    
    /**
     * @brief String representation: "CAS(8,6)"
     */
    std::string to_string() const;
    
    /**
     * @brief Check if active space is valid
     * Verifies: n_elec_active <= 2*n_active (Pauli principle)
     */
    bool is_valid() const;
    
    // Manual selection (for advanced users)
    void set_active_indices(const std::vector<int>& indices, int n_elec);
    
private:
    int n_inact_;      // Inactive (core) orbitals
    int n_act_;        // Active orbitals
    int n_virt_;       // Virtual orbitals
    int n_elec_act_;   // Electrons in active space
    
    void validate() const;
};

} // namespace mcscf
} // namespace mshqc

#endif // MSHQC_MCSCF_ACTIVE_SPACE_H