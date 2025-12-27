#ifndef MSHQC_CI_UTILS_H
#define MSHQC_CI_UTILS_H

#include "mshqc/ci/determinant.h"
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "mshqc/ci/slater_condon.h"

/**
 * @file ci_utils.h
 * @brief CI utility functions and data structures
 * * Contains:
 * 1. CIIntegrals struct (Container for active space integrals)
 * 2. Hash-based determinant indexing for O(1) lookup
 * 3. Tensor helper functions
 * * @author Muhamad Syahrul Hidayat
 * @date 2025-11-14
 */

namespace mshqc {
namespace ci {

// ============================================================================
// 1. CI Integrals Structure (YANG HILANG SEBELUMNYA)
// ============================================================================

/**
 * @brief Container for integrals transformed to active space
 * Used to pass integrals from CASSCF to CI Solvers
 */


// ============================================================================
// 2. Determinant Hashing Utilities
// ============================================================================

/**
 * @brief Build determinant-to-index hash map for O(1) lookup
 */
inline std::unordered_map<Determinant, int> 
build_determinant_index_map(const std::vector<Determinant>& dets) {
    std::unordered_map<Determinant, int> det_to_idx;
    det_to_idx.reserve(dets.size());
    
    for (size_t i = 0; i < dets.size(); i++) {
        det_to_idx[dets[i]] = static_cast<int>(i);
    }
    
    return det_to_idx;
}

/**
 * @brief Find determinant index using hash map (O(1))
 */
inline int find_determinant_index(
    const std::unordered_map<Determinant, int>& det_map,
    const Determinant& det) {
    
    auto it = det_map.find(det);
    if (it != det_map.end()) {
        return it->second;
    }
    return -1;  // Not found
}

// ============================================================================
// 3. Tensor Utilities (ERI Mapping)
// ============================================================================

using Tensor4 = Eigen::Tensor<double, 4>;

// Build same-spin antisymmetrized tensor from chemist ERI (pq|rs):
// out(p,q,r,s) = (pr|qs) - (ps|qr)
inline void build_same_spin_antisym_from_chemist(const Tensor4& ERI_pqrs,
                                                 Tensor4& out) {
    const int n = static_cast<int>(ERI_pqrs.dimension(0));
    out.resize(n,n,n,n);
    for (int p = 0; p < n; ++p)
      for (int q = 0; q < n; ++q)
        for (int r = 0; r < n; ++r)
          for (int s = 0; s < n; ++s)
            out(p,q,r,s) = ERI_pqrs(p,r,q,s) - ERI_pqrs(p,s,q,r);
}

// Build mixed-spin αβ tensor from chemist ERI (no antisym): out = (pq|rs)
inline void build_alpha_beta_from_chemist(const Tensor4& ERI_pqrs,
                                          Tensor4& out) {
    const int n = static_cast<int>(ERI_pqrs.dimension(0));
    out.resize(n,n,n,n);
    for (int p = 0; p < n; ++p)
      for (int q = 0; q < n; ++q)
        for (int r = 0; r < n; ++r)
          for (int s = 0; s < n; ++s)
            out(p,q,r,s) = ERI_pqrs(p,q,r,s);
}

} // namespace ci
} // namespace mshqc

#endif // MSHQC_CI_UTILS_H