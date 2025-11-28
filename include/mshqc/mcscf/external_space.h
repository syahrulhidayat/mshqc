/**
 * @file external_space.h
 * @brief External space generation for CASPT2
 * 
 * Generates single excitations from CASSCF reference to external space:
 * - Semi-Internal (SI): active → virtual
 * - Semi-External (SE): core → active  
 * - Doubly-External (DE): core → virtual
 * 
 * THEORY REFERENCES:
 *   - Andersson et al., J. Phys. Chem. 94, 5483 (1990), Eq. 10-12
 *   - Helgaker et al., "Molecular Electronic Structure Theory" (2000), Ch. 14
 * 
 * @author AI Agent 3 (Multireference Master)
 * @date 2025-11-12
 * @license MIT
 * 
 * @note Original implementation from published CASPT2 theory.
 *       No code copied from existing quantum chemistry software.
 */

#ifndef MSHQC_MCSCF_EXTERNAL_SPACE_H
#define MSHQC_MCSCF_EXTERNAL_SPACE_H

#include "mshqc/mcscf/active_space.h"
#include "mshqc/ci/determinant.h"
#include <vector>
#include <set>

namespace mshqc {
namespace mcscf {

/**
 * @brief External space configuration types
 * 
 * REFERENCE: Andersson et al. (1990), Eq. 10-12
 */
enum class ExcitationType {
    SEMI_INTERNAL,    ///< SI: active → virtual
    SEMI_EXTERNAL,    ///< SE: core → active
    DOUBLY_EXTERNAL   ///< DE: core → virtual
};

/**
 * @brief External space generator for CASPT2
 * 
 * Generates all single excitations from CASSCF reference that are:
 * 1. NOT in CAS space
 * 2. Connected to |Ψ₀⟩ by Hamiltonian
 * 3. Accessible through SI/SE/DE excitations
 * 
 * EXAMPLE - H2O CAS(8,6):
 *   CAS determinants: ~20
 *   External (SI):    ~20 × 4 × 10 = 800
 *   External (SE):    ~20 × 1 × 2 = 40  
 *   External (DE):    ~1 × 10 = 10
 *   Total external:   ~850 (after duplicate removal)
 */
class ExternalSpaceGenerator {
public:
    /**
     * @brief Construct external space generator
     * @param active_space Active space definition
     * @param n_total_orbitals Total number of molecular orbitals
     */
    ExternalSpaceGenerator(const ActiveSpace& active_space,
                           int n_total_orbitals);
    
    /**
     * @brief Generate external space from CAS reference
     * @param cas_determinants CASSCF reference determinants
     * @return Vector of external determinants (duplicates removed)
     * 
     * REFERENCE: Andersson et al. (1990), Eq. 10-12
     * 
     * Complexity: O(N_CAS × (N_act × N_virt + N_core × N_act) + N_core × N_virt)
     * Typical:    < 1 second for H2O, ~10 seconds for C2H4
     */
    std::vector<ci::Determinant> generate(
        const std::vector<ci::Determinant>& cas_determinants) const;
    
    /**
     * @brief Get external space statistics
     * @param cas_determinants CASSCF reference determinants
     * @return Struct with counts by excitation type
     */
    struct ExternalSpaceStats {
        int n_semi_internal;    ///< Count of SI excitations
        int n_semi_external;    ///< Count of SE excitations
        int n_doubly_external;  ///< Count of DE excitations
        int n_total;            ///< Total (after duplicate removal)
        int n_duplicates;       ///< Number of duplicates removed
    };
    
    ExternalSpaceStats get_statistics(
        const std::vector<ci::Determinant>& cas_determinants) const;
    
    /**
     * @brief Classify external determinant by excitation type
     * @param ext_det External determinant
     * @param ref_det Reference (CAS) determinant
     * @return Excitation type (SI, SE, or DE)
     * 
     * Used for analysis and debugging
     */
    ExcitationType classify_excitation(
        const ci::Determinant& ext_det,
        const ci::Determinant& ref_det) const;
    
private:
    ActiveSpace active_space_;
    int n_total_orbitals_;
    
    /**
     * @brief Generate semi-internal excitations (active → virtual)
     * REFERENCE: Andersson et al. (1990), Eq. 10
     */
    void generate_semi_internal(
        const ci::Determinant& ref_det,
        std::vector<ci::Determinant>& external) const;
    
    /**
     * @brief Generate semi-external excitations (core → active)
     * REFERENCE: Andersson et al. (1990), Eq. 11
     */
    void generate_semi_external(
        const ci::Determinant& ref_det,
        std::vector<ci::Determinant>& external) const;
    
    /**
     * @brief Generate doubly-external excitations (core → virtual)
     * REFERENCE: Andersson et al. (1990), Eq. 12
     * 
     * NOTE: DE excitations are independent of CAS determinants,
     *       generated only once from HF-like reference
     */
    void generate_doubly_external(
        const ci::Determinant& ref_det,
        std::vector<ci::Determinant>& external) const;
    
    /**
     * @brief Remove duplicate determinants (from multiple CAS references)
     * @param determinants Vector to deduplicate (modified in-place)
     * @return Number of duplicates removed
     * 
     * Uses sorting + std::unique for O(N log N) complexity
     */
    int remove_duplicates(std::vector<ci::Determinant>& determinants) const;
    
    /**
     * @brief Get occupied active orbitals in determinant
     * @param det Determinant to analyze
     * @return Vector of occupied active orbital indices
     */
    std::vector<int> get_occupied_active(const ci::Determinant& det) const;
    
    /**
     * @brief Get unoccupied active orbitals in determinant
     * @param det Determinant to analyze
     * @return Vector of unoccupied active orbital indices
     */
    std::vector<int> get_unoccupied_active(const ci::Determinant& det) const;
};

} // namespace mcscf
} // namespace mshqc

#endif // MSHQC_MCSCF_EXTERNAL_SPACE_H
