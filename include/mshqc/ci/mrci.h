/**
 * @file mrci.h
 * @brief Multi-Reference Configuration Interaction (MRCI)
 * 
 * THEORY REFERENCES:
 *   - Werner & Knowles (1988), J. Chem. Phys. 89, 5803
 *   - Szalay et al. (2012), Chem. Rev. 112, 108
 *   - Roos (1987), Adv. Chem. Phys. 69, 399
 * 
 * MRCI starts from multiple reference determinants (from CASSCF)
 * and generates singles and doubles from each reference.
 * 
 * Used for:
 * - Multi-reference systems (bond breaking, transition states)
 * - Post-CASSCF correlation (CASPT alternative)
 * - Size-consistent correlation treatment
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 */

#ifndef MSHQC_CI_MRCI_H
#define MSHQC_CI_MRCI_H

#include "mshqc/ci/determinant.h"
#include "mshqc/ci/slater_condon.h"
#include "mshqc/ci/davidson.h"
#include <Eigen/Dense>
#include <vector>
#include <set>

namespace mshqc {
namespace ci {

/**
 * MRCI result structure
 */
struct MRCIResult {
    double e_mrci;             // MRCI energy
    double e_ref;              // Reference space energy (usually CASSCF)
    double e_corr;             // Correlation energy (MRCI - ref)
    int n_determinants;        // Total determinants (ref + singles + doubles)
    int n_references;          // Number of reference determinants
    int n_singles;             // Number of singles from all references
    int n_doubles;             // Number of doubles from all references
    bool converged;            // Solver convergence
    int iterations;            // Iterations
    
    // Wavefunction data
    std::vector<Determinant> determinants;
    Eigen::VectorXd coefficients;
    
    // Analysis
    double ref_weight;         // Total weight of reference determinants
    double singles_weight;     // Singles contribution
    double doubles_weight;     // Doubles contribution
};

/**
 * Multi-Reference Configuration Interaction
 * 
 * Generates singles and doubles excitations from multiple reference
 * determinants (typically from CAS wavefunction).
 * 
 * USAGE:
 *   // Get references from CASSCF
 *   std::vector<Determinant> refs = casscf.get_references();
 *   
 *   MRCI mrci(ints, refs, n_core, n_active, n_virtual);
 *   auto result = mrci.compute();
 * 
 * KEY FEATURES:
 * - Multi-reference starting point (fixes single-ref failures)
 * - Size-extensive (unlike truncated CI)
 * - Can handle near-degeneracies
 * 
 * TYPICAL USE CASES:
 * - Bond dissociation (H2 → 2H)
 * - Transition states
 * - Excited states with multiple configurations
 * - Systems with static correlation
 */
class MRCI {
public:
    /**
     * Constructor
     * 
     * @param ints MO integrals (full space, not just active)
     * @param references Reference determinants (from CASSCF or manual)
     * @param n_core Number of core (frozen) orbitals
     * @param n_active Number of active orbitals (references use these)
     * @param n_virtual Number of virtual orbitals
     */
    MRCI(const CIIntegrals& ints,
         const std::vector<Determinant>& references,
         int n_core,
         int n_active,
         int n_virtual);
    
    /**
     * Compute MRCI ground state
     * 
     * Algorithm:
     * 1. Start with reference determinants
     * 2. Generate singles from each reference
     * 3. Generate doubles from each reference
     * 4. Remove duplicates (important!)
     * 5. Solve with Davidson
     * 6. Analyze wavefunction
     * 
     * @return MRCIResult with energy and wavefunction
     */
    MRCIResult compute();
    
    /**
     * Compute with comparison to CASSCF energy
     * 
     * @param casscf_energy CASSCF energy for comparison
     * @return MRCIResult with comparison data
     */
    MRCIResult compute_with_comparison(double casscf_energy);
    
    /**
     * Get all determinants (without solving)
     * Useful for analysis
     */
    std::vector<Determinant> get_determinants() const;
    
    /**
     * Estimate number of determinants
     * Based on number of references and excitation space
     */
    size_t estimate_n_determinants() const;
    
private:
    const CIIntegrals& ints_;
    std::vector<Determinant> references_;
    int n_core_;
    int n_active_;
    int n_virtual_;
    
    /**
     * Generate all singles from references
     * 
     * For each reference determinant, generate all single excitations
     * from occupied to virtual orbitals.
     * 
     * IMPORTANT: Remove duplicates (multiple refs may give same single)
     */
    std::vector<Determinant> generate_singles_from_references();
    
    /**
     * Generate all doubles from references
     * 
     * For each reference, generate all double excitations.
     * This is the most expensive step (O(N^4) per reference).
     * 
     * IMPORTANT: Remove duplicates
     */
    std::vector<Determinant> generate_doubles_from_references();
    
    /**
     * Remove duplicate determinants
     * Use std::set for automatic uniqueness
     * 
     * @param dets Input determinant list (may have duplicates)
     * @return Unique determinants only
     */
    std::vector<Determinant> remove_duplicates(const std::vector<Determinant>& dets);
    
    /**
     * Check if determinant is in reference space
     * 
     * @param det Determinant to check
     * @return true if det is one of the references
     */
    bool is_reference(const Determinant& det) const;
    
    /**
     * Analyze MRCI wavefunction
     * Breakdown by reference vs external space
     */
    void analyze_wavefunction(const Eigen::VectorXd& c,
                             const std::vector<Determinant>& dets,
                             MRCIResult& result);
    
    /**
     * Generate singles from a single determinant
     * (Helper for generate_singles_from_references)
     */
    std::vector<Determinant> generate_singles_from_det(const Determinant& det);
    
    /**
     * Generate doubles from a single determinant
     * (Helper for generate_doubles_from_references)
     */
    std::vector<Determinant> generate_doubles_from_det(const Determinant& det);
};

/**
 * Helper: Create MRCI references from CASSCF coefficients
 * 
 * Selects dominant determinants from CAS wavefunction as references.
 * 
 * @param casscf_dets CASSCF determinants
 * @param casscf_coeffs CASSCF CI coefficients
 * @param threshold Minimum weight to include (default 0.01 = 1%)
 * @return Reference determinants
 */
std::vector<Determinant> mrci_references_from_casscf(
    const std::vector<Determinant>& casscf_dets,
    const Eigen::VectorXd& casscf_coeffs,
    double threshold = 0.01
);

/**
 * Helper: Estimate MRCI size
 * 
 * @param n_refs Number of reference determinants
 * @param n_occ_core Number of core electrons
 * @param n_occ_active Number of active electrons
 * @param n_virt Number of virtual orbitals
 * @return Estimated MRCI determinant count
 */
size_t estimate_mrci_size(int n_refs, int n_occ_core, 
                          int n_occ_active, int n_virt);

} // namespace ci
} // namespace mshqc

#endif // MSHQC_CI_MRCI_H
