/**
 * @file active_space.cc
 * @brief Implementation of active space for CASSCF
 * 
 * THEORY REFERENCES:
 *   - B. O. Roos et al., Chem. Phys. 48, 157 (1980)
 *   - T. Helgaker et al., "Molecular Electronic Structure Theory" (2000), Ch. 14
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 * @license MIT
 */

#include "mshqc/mcscf/active_space.h"
#include <sstream>

namespace mshqc {
namespace mcscf {

ActiveSpace::ActiveSpace(int n_inactive, int n_active, 
                         int n_virtual, int n_elec_active)
    : n_inact_(n_inactive), n_act_(n_active), 
      n_virt_(n_virtual), n_elec_act_(n_elec_active) {
    validate();
}

ActiveSpace ActiveSpace::CAS(int n_elec, int n_orb, 
                             int n_total_orb, int n_total_elec) {
    // REFERENCE: Roos et al. (1980), Chem. Phys. 48, 157
    // CAS(n,m): n electrons in m orbitals
    // Inactive orbitals = (total_elec - active_elec) / 2
    
    if (n_elec > n_total_elec) {
        throw std::runtime_error("CAS electrons > total electrons");
    }
    
    if (n_orb > n_total_orb) {
        throw std::runtime_error("CAS orbitals > total orbitals");
    }
    
    // Compute inactive (doubly-occupied core)
    int n_core_elec = n_total_elec - n_elec;
    if (n_core_elec % 2 != 0) {
        throw std::runtime_error("CAS: core electrons must be even (closed-shell core)");
    }
    
    int n_inactive = n_core_elec / 2;
    int n_virtual = n_total_orb - n_inactive - n_orb;
    
    if (n_virtual < 0) {
        throw std::runtime_error("CAS: not enough orbitals for requested active space");
    }
    
    return ActiveSpace(n_inactive, n_orb, n_virtual, n_elec);
}

std::vector<int> ActiveSpace::inactive_indices() const {
    // Inactive orbitals are indexed: [0, 1, ..., n_inact-1]
    std::vector<int> indices;
    indices.reserve(n_inact_);
    
    for(int i = 0; i < n_inact_; i++) {
        indices.push_back(i);
    }
    
    return indices;
}

std::vector<int> ActiveSpace::active_indices() const {
    // Active orbitals are indexed: [n_inact, n_inact+1, ..., n_inact+n_act-1]
    std::vector<int> indices;
    indices.reserve(n_act_);
    
    for(int i = 0; i < n_act_; i++) {
        indices.push_back(n_inact_ + i);
    }
    
    return indices;
}

std::vector<int> ActiveSpace::virtual_indices() const {
    // Virtual orbitals are indexed: [n_inact+n_act, ..., n_inact+n_act+n_virt-1]
    std::vector<int> indices;
    indices.reserve(n_virt_);
    
    int start = n_inact_ + n_act_;
    for(int i = 0; i < n_virt_; i++) {
        indices.push_back(start + i);
    }
    
    return indices;
}

std::string ActiveSpace::to_string() const {
    std::ostringstream oss;
    oss << "CAS(" << n_elec_act_ << "," << n_act_ << ")";
    return oss.str();
}

bool ActiveSpace::is_valid() const {
    // Check Pauli principle: max 2 electrons per orbital
    if (n_elec_act_ > 2 * n_act_) return false;
    
    // Check non-negative
    if (n_inact_ < 0 || n_act_ < 0 || n_virt_ < 0 || n_elec_act_ < 0) 
        return false;
    
    return true;
}

void ActiveSpace::set_active_indices(const std::vector<int>& indices, int n_elec) {
    // Manual selection for advanced users
    // Example: select orbitals [3,4,5,6,7,8] with 8 electrons
    
    if(indices.empty()) {
        throw std::runtime_error("Active indices cannot be empty");
    }
    
    n_act_ = static_cast<int>(indices.size());
    n_elec_act_ = n_elec;
    
    // Assume indices are contiguous (simplification for now)
    // In production, would handle arbitrary selection
    n_inact_ = indices[0];
    
    validate();
}

void ActiveSpace::validate() const {
    if (!is_valid()) {
        std::ostringstream oss;
        oss << "Invalid active space: " << to_string() 
            << " (n_inact=" << n_inact_ << ", n_virt=" << n_virt_ << ")";
        throw std::runtime_error(oss.str());
    }
}

} // namespace mcscf
} // namespace mshqc
