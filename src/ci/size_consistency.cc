// Author: Muhamad Syahrul Hidayat
// Date: 2025-11-16
//
// Implementation of size-consistency corrections for CI methods
//
// ============================================================================
// ORIGINAL IMPLEMENTATION - NO CODE COPIED FROM PYSCF/PSI4
// ============================================================================

#include "mshqc/ci/size_consistency.h"
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace mshqc {
namespace ci {

double SizeConsistencyCorrection::davidson_q_correction(
    double e_cisd, 
    double e_hf, 
    double c0
) {
    // Validate input
    if (std::abs(c0) < 1e-10) {
        throw std::runtime_error(
            "SizeConsistencyCorrection::davidson_q_correction: "
            "HF coefficient c0 is too small (nearly zero). "
            "Davidson +Q is not applicable for highly multi-reference systems."
        );
    }

    if (std::abs(c0) > 1.0 + 1e-6) {
        throw std::runtime_error(
            "SizeConsistencyCorrection::davidson_q_correction: "
            "HF coefficient |c0| > 1.0, which is unphysical. "
            "Check wavefunction normalization."
        );
    }

    // Correlation energy: ΔE_corr = E(CISD) - E(HF)
    double delta_e_corr = e_cisd - e_hf;

    // Davidson +Q correction: ΔE_Q = (1 - c₀²) * ΔE_corr
    // Physical interpretation:
    //   - Accounts for quadruple excitations via perturbation theory
    //   - If c₀ ≈ 1: correction is small (dominated by HF reference)
    //   - If c₀ << 1: correction is large (multi-reference character)
    double c0_squared = c0 * c0;
    double delta_e_q = (1.0 - c0_squared) * delta_e_corr;

    return delta_e_q;
}

double SizeConsistencyCorrection::cisd_plus_q(
    double e_cisd, 
    double e_hf, 
    double c0
) {
    double delta_e_q = davidson_q_correction(e_cisd, e_hf, c0);
    return e_cisd + delta_e_q;
}

double SizeConsistencyCorrection::size_consistency_error(
    double e_single, 
    double e_multiple, 
    int n
) {
    if (n <= 0) {
        throw std::invalid_argument(
            "SizeConsistencyCorrection::size_consistency_error: "
            "Number of copies n must be positive"
        );
    }

    // Size-consistency error: Error = E(n*A) - n*E(A)
    // For size-consistent methods (FCI, CC): Error → 0
    // For CISD: Error ≠ 0 and scales as O(n)
    double error = e_multiple - n * e_single;
    return error;
}

double SizeConsistencyCorrection::renormalized_q_correction(
    double e_mrcisd, 
    double e_cas,
    const std::vector<double>& ref_weights
) {
    if (ref_weights.empty()) {
        throw std::invalid_argument(
            "SizeConsistencyCorrection::renormalized_q_correction: "
            "Reference weights vector is empty"
        );
    }

    // Compute sum of squared reference weights: Σᵢ c²ᵢ
    double sum_c_squared = 0.0;
    for (double c : ref_weights) {
        sum_c_squared += c * c;
    }

    // Validate normalization (should be ≤ 1.0)
    if (sum_c_squared > 1.0 + 1e-6) {
        throw std::runtime_error(
            "SizeConsistencyCorrection::renormalized_q_correction: "
            "Sum of squared reference weights > 1.0, check normalization"
        );
    }

    // Correlation energy relative to CAS
    double delta_e_corr = e_mrcisd - e_cas;

    // Renormalized Davidson correction:
    //   ΔE_Q = (1 - Σᵢc²ᵢ) * ΔE_corr
    // Accounts for multi-reference character in CAS space
    double delta_e_q = (1.0 - sum_c_squared) * delta_e_corr;

    return delta_e_q;
}

double SizeConsistencyCorrection::qcisd_approximation(
    double e_cisd, 
    double e_hf, 
    double c0,
    double k
) {
    // Validate input
    if (std::abs(c0) < 1e-10 || std::abs(c0) > 1.0 + 1e-6) {
        throw std::runtime_error(
            "SizeConsistencyCorrection::qcisd_approximation: "
            "Invalid HF coefficient c0"
        );
    }

    if (k <= 0.0 || k > 2.0) {
        throw std::invalid_argument(
            "SizeConsistencyCorrection::qcisd_approximation: "
            "Scaling factor k should be in range (0, 2]. Typical value: 0.75"
        );
    }

    // Correlation energy
    double delta_e_corr = e_cisd - e_hf;

    // QCISD approximation: E ≈ E(CISD) + k * (1 - c₀²)² * ΔE_corr
    // More accurate than Davidson +Q for connected triples
    // Reference: Pople et al., J. Chem. Phys. 87, 5968 (1987)
    double c0_squared = c0 * c0;
    double correction_factor = 1.0 - c0_squared;
    double delta_e_qci = k * correction_factor * correction_factor * delta_e_corr;

    return e_cisd + delta_e_qci;
}

bool SizeConsistencyCorrection::is_davidson_q_reliable(double c0) {
    double c0_squared = c0 * c0;

    // Davidson +Q is reliable when:
    //   0.90 < c₀² < 0.99
    // 
    // This corresponds to weakly correlated, single-reference systems
    // where perturbative correction for higher excitations is valid
    
    bool reliable = (c0_squared > 0.90) && (c0_squared < 0.99);
    return reliable;
}

std::string SizeConsistencyCorrection::get_diagnostic_message(double c0) {
    double c0_squared = c0 * c0;
    std::ostringstream msg;

    msg << "Davidson +Q Diagnostic:\n";
    msg << "  HF coefficient: c₀ = " << c0 << "\n";
    msg << "  HF weight: c₀² = " << c0_squared << "\n\n";

    if (c0_squared > 0.99) {
        msg << "  Status: ⚠️  WARNING - Almost pure HF reference\n";
        msg << "  Recommendation: Davidson +Q correction is negligible.\n";
        msg << "                  System is weakly correlated, CISD should be sufficient.\n";
    } else if (c0_squared > 0.90) {
        msg << "  Status: ✅ RELIABLE - Single-reference character\n";
        msg << "  Recommendation: Davidson +Q correction is appropriate.\n";
        msg << "                  Expect good agreement with higher-order methods.\n";
    } else if (c0_squared > 0.80) {
        msg << "  Status: ⚠️  CAUTION - Moderate multi-reference character\n";
        msg << "  Recommendation: Davidson +Q may be qualitatively useful,\n";
        msg << "                  but consider MRCI or CASPT2 for accuracy.\n";
    } else {
        msg << "  Status: ❌ UNRELIABLE - Strong multi-reference character\n";
        msg << "  Recommendation: DO NOT use Davidson +Q correction.\n";
        msg << "                  Use MRCI, CASPT2, or DMRG instead.\n";
    }

    return msg.str();
}

} // namespace ci
} // namespace mshqc
