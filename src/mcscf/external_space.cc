// ============================================================================
// External Space Generation for CASPT2
// Complete Active Space Second-Order Perturbation Theory
// ============================================================================
// Author: AI  (Multireference Master)
// Date: 2025-11-12
// License: MIT
//
// THEORY REFERENCES:
// - Andersson, K. et al., J. Phys. Chem. 94, 5483-5488 (1990)
//   "Second-order perturbation theory with a CASSCF reference function"
//   Equations 10-12: External space definition (SI, SE, DE)
//
// - Helgaker, T. et al., "Molecular Electronic-Structure Theory" (2000)
//   Chapter 14: Multireference Perturbation Theory
//
// DEPENDENCIES:
// - : Determinant class for bit-string manipulations
// - CASSCF: ActiveSpace for orbital partitioning
// ============================================================================

#include "mshqc/mcscf/external_space.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace mshqc {
namespace mcscf {

// ============================================================================
// Constructor
// ============================================================================

ExternalSpaceGenerator::ExternalSpaceGenerator(
    const ActiveSpace& active_space,
    int n_total_orbitals)
    : active_space_(active_space), n_total_orbitals_(n_total_orbitals) {
    
    // Validation
    if (n_total_orbitals != active_space.n_total_orb()) {
        throw std::runtime_error(
            "Inconsistent orbital count in ExternalSpaceGenerator");
    }
}

// ============================================================================
// Main External Space Generation
// ============================================================================

std::vector<ci::Determinant> ExternalSpaceGenerator::generate(
    const std::vector<ci::Determinant>& cas_determinants) const {
    
    std::vector<ci::Determinant> external;
    
    // Estimate capacity to avoid reallocations
    // Rough estimate: N_CAS × (N_act × N_virt + N_core × N_act) + N_core × N_virt
    int n_cas = cas_determinants.size();
    int n_act = active_space_.n_active();
    int n_virt = active_space_.n_virtual();
    int n_core = active_space_.n_inactive();
    
    int estimated_size = n_cas * (n_act * n_virt + n_core * n_act) + n_core * n_virt;
    external.reserve(estimated_size * 2);  // × 2 for alpha + beta
    
    // Generate excitations from each CAS determinant
    for (const auto& cas_det : cas_determinants) {
        // Semi-Internal (SI): active → virtual
        generate_semi_internal(cas_det, external);
        
        // Semi-External (SE): core → active
        generate_semi_external(cas_det, external);
    }
    
    // Doubly-External (DE): core → virtual
    // Only need to generate once (independent of CAS configurations)
    if (!cas_determinants.empty()) {
        generate_doubly_external(cas_determinants[0], external);
    }
    
    // Remove duplicates
    int n_duplicates = remove_duplicates(external);
    
    // Report
    std::cout << "External space generated:\n";
    std::cout << "  Before deduplication: " << external.size() + n_duplicates << "\n";
    std::cout << "  Duplicates removed:   " << n_duplicates << "\n";
    std::cout << "  Final count:          " << external.size() << "\n";
    
    return external;
}

// ============================================================================
// Semi-Internal Excitations (SI): active → virtual
// ============================================================================
// THEORY: Andersson et al. (1990), Eq. 10
// |Φ_tv^SI⟩ = a†_v a_t |Ψ₀⟩
// Creates particle in virtual space from occupied active orbital
// ============================================================================

void ExternalSpaceGenerator::generate_semi_internal(
    const ci::Determinant& ref_det,
    std::vector<ci::Determinant>& external) const {
    
    auto active_idx = active_space_.active_indices();
    auto virtual_idx = active_space_.virtual_indices();
    
    // Loop over occupied active orbitals
    for (int t : active_idx) {
        // Check if occupied in reference determinant (alpha or beta)
        bool occ_alpha = ref_det.is_occupied(t, true);
        bool occ_beta = ref_det.is_occupied(t, false);
        
        // Loop over virtual orbitals
        for (int v : virtual_idx) {
            // Alpha excitation: t(α) → v(α)
            if (occ_alpha) {
                try {
                    auto ext_det = ref_det.single_excite(t, v, true);
                    external.push_back(ext_det);
                } catch (...) {
                    // Excitation not allowed (e.g., v already occupied)
                }
            }
            
            // Beta excitation: t(β) → v(β)
            if (occ_beta) {
                try {
                    auto ext_det = ref_det.single_excite(t, v, false);
                    external.push_back(ext_det);
                } catch (...) {
                    // Excitation not allowed
                }
            }
        }
    }
}

// ============================================================================
// Semi-External Excitations (SE): core → active
// ============================================================================
// THEORY: Andersson et al. (1990), Eq. 11
// |Φ_it^SE⟩ = a†_t a_i |Ψ₀⟩
// Creates hole in core, promotes electron to active space
// ============================================================================

void ExternalSpaceGenerator::generate_semi_external(
    const ci::Determinant& ref_det,
    std::vector<ci::Determinant>& external) const {
    
    auto inactive_idx = active_space_.inactive_indices();
    auto active_idx = active_space_.active_indices();
    
    // Loop over core (inactive) orbitals (always occupied)
    for (int i : inactive_idx) {
        // Loop over active orbitals
        for (int t : active_idx) {
            // Check if t is unoccupied (hole available)
            bool unocc_alpha = !ref_det.is_occupied(t, true);
            bool unocc_beta = !ref_det.is_occupied(t, false);
            
            // Alpha excitation: i(α) → t(α)
            if (unocc_alpha) {
                try {
                    auto ext_det = ref_det.single_excite(i, t, true);
                    external.push_back(ext_det);
                } catch (...) {
                    // Excitation not allowed
                }
            }
            
            // Beta excitation: i(β) → t(β)
            if (unocc_beta) {
                try {
                    auto ext_det = ref_det.single_excite(i, t, false);
                    external.push_back(ext_det);
                } catch (...) {
                    // Excitation not allowed
                }
            }
        }
    }
}

// ============================================================================
// Doubly-External Excitations (DE): core → virtual
// ============================================================================
// THEORY: Andersson et al. (1990), Eq. 12
// |Φ_iv^DE⟩ = a†_v a_i |Ψ₀⟩
// Bypasses active space completely (core to virtual)
// Independent of CAS configuration - generate only once
// ============================================================================

void ExternalSpaceGenerator::generate_doubly_external(
    const ci::Determinant& ref_det,
    std::vector<ci::Determinant>& external) const {
    
    auto inactive_idx = active_space_.inactive_indices();
    auto virtual_idx = active_space_.virtual_indices();
    
    // Loop over core orbitals
    for (int i : inactive_idx) {
        // Loop over virtual orbitals
        for (int v : virtual_idx) {
            // Alpha excitation: i(α) → v(α)
            try {
                auto ext_det = ref_det.single_excite(i, v, true);
                external.push_back(ext_det);
            } catch (...) {
                // Excitation not allowed
            }
            
            // Beta excitation: i(β) → v(β)
            try {
                auto ext_det = ref_det.single_excite(i, v, false);
                external.push_back(ext_det);
            } catch (...) {
                // Excitation not allowed
            }
        }
    }
}

// ============================================================================
// Duplicate Removal
// ============================================================================
// Multiple CAS reference determinants can generate identical external
// configurations through different excitation paths.
// Use sort + unique for O(N log N) complexity.
// ============================================================================

int ExternalSpaceGenerator::remove_duplicates(
    std::vector<ci::Determinant>& determinants) const {
    
    if (determinants.empty()) return 0;
    
    // Sort determinants (Determinant must have operator<)
    std::sort(determinants.begin(), determinants.end());
    
    // Count original size
    int original_size = determinants.size();
    
    // Remove duplicates
    auto last = std::unique(determinants.begin(), determinants.end());
    determinants.erase(last, determinants.end());
    
    // Return number of duplicates removed
    return original_size - determinants.size();
}

// ============================================================================
// Statistics
// ============================================================================

ExternalSpaceGenerator::ExternalSpaceStats 
ExternalSpaceGenerator::get_statistics(
    const std::vector<ci::Determinant>& cas_determinants) const {
    
    ExternalSpaceStats stats;
    
    // Generate each type separately to count
    std::vector<ci::Determinant> si_dets, se_dets, de_dets;
    
    for (const auto& cas_det : cas_determinants) {
        generate_semi_internal(cas_det, si_dets);
        generate_semi_external(cas_det, se_dets);
    }
    
    if (!cas_determinants.empty()) {
        generate_doubly_external(cas_determinants[0], de_dets);
    }
    
    stats.n_semi_internal = si_dets.size();
    stats.n_semi_external = se_dets.size();
    stats.n_doubly_external = de_dets.size();
    
    // Combine and count after deduplication
    std::vector<ci::Determinant> all_external;
    all_external.insert(all_external.end(), si_dets.begin(), si_dets.end());
    all_external.insert(all_external.end(), se_dets.begin(), se_dets.end());
    all_external.insert(all_external.end(), de_dets.begin(), de_dets.end());
    
    int before_dedup = all_external.size();
    stats.n_duplicates = remove_duplicates(all_external);
    stats.n_total = all_external.size();
    
    return stats;
}

// ============================================================================
// Excitation Classification
// ============================================================================

ExcitationType ExternalSpaceGenerator::classify_excitation(
    const ci::Determinant& ext_det,
    const ci::Determinant& ref_det) const {
    
    // Get excitation level between determinants
    auto exc = ext_det.excitation_level(ref_det);
    int total_exc = exc.first + exc.second;
    
    if (total_exc != 1) {
        throw std::runtime_error(
            "classify_excitation: not a single excitation");
    }
    
    // Find which orbital changed
    // This is simplified - full implementation would use excitation operators
    
    auto inactive_idx = active_space_.inactive_indices();
    auto active_idx = active_space_.active_indices();
    auto virtual_idx = active_space_.virtual_indices();
    
    // Check orbital space membership (simplified heuristic)
    // In production, would extract actual excitation indices
    
    // For now, return based on orbital counts
    // This is a placeholder - proper classification requires
    // extracting the actual i→a excitation indices
    
    return ExcitationType::SEMI_INTERNAL;  // Placeholder
}

// ============================================================================
// Helper Methods
// ============================================================================

std::vector<int> ExternalSpaceGenerator::get_occupied_active(
    const ci::Determinant& det) const {
    
    std::vector<int> occupied;
    auto active_idx = active_space_.active_indices();
    
    for (int t : active_idx) {
        if (det.is_occupied(t, true) || det.is_occupied(t, false)) {
            occupied.push_back(t);
        }
    }
    
    return occupied;
}

std::vector<int> ExternalSpaceGenerator::get_unoccupied_active(
    const ci::Determinant& det) const {
    
    std::vector<int> unoccupied;
    auto active_idx = active_space_.active_indices();
    
    for (int t : active_idx) {
        if (!det.is_occupied(t, true) && !det.is_occupied(t, false)) {
            unoccupied.push_back(t);
        }
    }
    
    return unoccupied;
}

} // namespace mcscf
} // namespace mshqc
