/**
 * @file wavefunction.cc
 * @brief Implementation of universal wavefunction container
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 * @license MIT
 */

#include "mshqc/foundation/wavefunction.h"
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <stdexcept>

namespace mshqc {
namespace foundation {

// ============================================================================
// Excitation struct
// ============================================================================

std::string Excitation::to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << std::setw(8) << amplitude << "  |Φ_{" << i << "," << j 
        << "}^{" << a << "," << b << "}⟩(" << spin << ")";
    return oss.str();
}

// ============================================================================
// Wavefunction class
// ============================================================================

Wavefunction::Wavefunction(int nocc_a, int nocc_b, int nvirt_a, int nvirt_b)
    : nocc_a_(nocc_a), nocc_b_(nocc_b), nvirt_a_(nvirt_a), nvirt_b_(nvirt_b) {
    
    // Initialize empty tensors (size 0)
    // Will allocate when set_t2_order_X() is called
}

void Wavefunction::set_t2_order_1(const Eigen::Tensor<double, 4>& t2_aa,
                                   const Eigen::Tensor<double, 4>& t2_bb,
                                   const Eigen::Tensor<double, 4>& t2_ab) {
    t2_1_aa_ = t2_aa;
    t2_1_bb_ = t2_bb;
    t2_1_ab_ = t2_ab;
}

void Wavefunction::set_t2_order_2(const Eigen::Tensor<double, 4>& t2_aa,
                                   const Eigen::Tensor<double, 4>& t2_bb,
                                   const Eigen::Tensor<double, 4>& t2_ab) {
    t2_2_aa_ = t2_aa;
    t2_2_bb_ = t2_bb;
    t2_2_ab_ = t2_ab;
}

void Wavefunction::set_t2_order_3(const Eigen::Tensor<double, 4>& t2_aa,
                                   const Eigen::Tensor<double, 4>& t2_bb,
                                   const Eigen::Tensor<double, 4>& t2_ab) {
    t2_3_aa_ = t2_aa;
    t2_3_bb_ = t2_bb;
    t2_3_ab_ = t2_ab;
}

const Eigen::Tensor<double, 4>& Wavefunction::get_t2(int order, 
                                                       const std::string& spin) const {
    if (order == 1) {
        if (spin == "aa") return t2_1_aa_;
        else if (spin == "bb") return t2_1_bb_;
        else if (spin == "ab") return t2_1_ab_;
    }
    else if (order == 2) {
        if (spin == "aa") return t2_2_aa_;
        else if (spin == "bb") return t2_2_bb_;
        else if (spin == "ab") return t2_2_ab_;
    }
    else if (order == 3) {
        if (spin == "aa") return t2_3_aa_;
        else if (spin == "bb") return t2_3_bb_;
        else if (spin == "ab") return t2_3_ab_;
    }
    
    throw std::runtime_error("Invalid order or spin in get_t2");
}

bool Wavefunction::has_amplitudes(int order) const {
    if (order == 1) {
        return t2_1_aa_.size() > 0 || t2_1_bb_.size() > 0 || t2_1_ab_.size() > 0;
    }
    else if (order == 2) {
        return t2_2_aa_.size() > 0 || t2_2_bb_.size() > 0 || t2_2_ab_.size() > 0;
    }
    else if (order == 3) {
        return t2_3_aa_.size() > 0 || t2_3_bb_.size() > 0 || t2_3_ab_.size() > 0;
    }
    return false;
}

std::vector<Excitation> Wavefunction::extract_from_tensor(
    const Eigen::Tensor<double, 4>& t2,
    const std::string& spin,
    double threshold) const {
    
    std::vector<Excitation> excitations;
    
    if (t2.size() == 0) return excitations;  // Empty tensor
    
    // Determine dimensions based on spin
    int ni, nj, na, nb;
    if (spin == "aa") {
        ni = nj = nocc_a_;
        na = nb = nvirt_a_;
    } else if (spin == "bb") {
        ni = nj = nocc_b_;
        na = nb = nvirt_b_;
    } else { // ab
        ni = nocc_a_;
        nj = nocc_b_;
        na = nvirt_a_;
        nb = nvirt_b_;
    }
    
    // Extract all amplitudes above threshold
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            for (int a = 0; a < na; a++) {
                for (int b = 0; b < nb; b++) {
                    double amp = t2(i, j, a, b);
                    
                    if (std::abs(amp) > threshold) {
                        Excitation exc;
                        exc.i = i;
                        exc.j = j;
                        exc.a = a;
                        exc.b = b;
                        exc.amplitude = amp;
                        exc.spin = spin;
                        excitations.push_back(exc);
                    }
                }
            }
        }
    }
    
    return excitations;
}

std::vector<Excitation> Wavefunction::dominant_amplitudes(
    double threshold, int max_excitations) const {
    
    std::vector<Excitation> all_excitations;
    
    // Extract from all spin cases (order 1 only for now)
    if (has_amplitudes(1)) {
        if (t2_1_aa_.size() > 0) {
            auto exc_aa = extract_from_tensor(t2_1_aa_, "aa", threshold);
            all_excitations.insert(all_excitations.end(), exc_aa.begin(), exc_aa.end());
        }
        
        if (t2_1_bb_.size() > 0) {
            auto exc_bb = extract_from_tensor(t2_1_bb_, "bb", threshold);
            all_excitations.insert(all_excitations.end(), exc_bb.begin(), exc_bb.end());
        }
        
        if (t2_1_ab_.size() > 0) {
            auto exc_ab = extract_from_tensor(t2_1_ab_, "ab", threshold);
            all_excitations.insert(all_excitations.end(), exc_ab.begin(), exc_ab.end());
        }
    }
    
    // Sort by absolute amplitude (largest first)
    std::sort(all_excitations.begin(), all_excitations.end());
    
    // Limit to max_excitations
    if (all_excitations.size() > static_cast<size_t>(max_excitations)) {
        all_excitations.resize(max_excitations);
    }
    
    return all_excitations;
}

double Wavefunction::norm_squared(int order) const {
    // Wavefunction norm: ⟨Ψ|Ψ⟩ = 1 + Σ |t_ij^ab|²
    
    double norm2 = 1.0;  // HF reference contribution
    
    // Add contributions from each order
    for (int ord = 1; ord <= order; ord++) {
        if (!has_amplitudes(ord)) continue;
        
        // Get tensors for this order
        const Eigen::Tensor<double, 4>* t2_aa = nullptr;
        const Eigen::Tensor<double, 4>* t2_bb = nullptr;
        const Eigen::Tensor<double, 4>* t2_ab = nullptr;
        
        if (ord == 1) {
            t2_aa = &t2_1_aa_;
            t2_bb = &t2_1_bb_;
            t2_ab = &t2_1_ab_;
        } else if (ord == 2) {
            t2_aa = &t2_2_aa_;
            t2_bb = &t2_2_bb_;
            t2_ab = &t2_2_ab_;
        } else if (ord == 3) {
            t2_aa = &t2_3_aa_;
            t2_bb = &t2_3_bb_;
            t2_ab = &t2_3_ab_;
        }
        
        // Sum |t|² for each spin case
        if (t2_aa && t2_aa->size() > 0) {
            for (int i = 0; i < nocc_a_; i++)
                for (int j = 0; j < nocc_a_; j++)
                    for (int a = 0; a < nvirt_a_; a++)
                        for (int b = 0; b < nvirt_a_; b++)
                            norm2 += (*t2_aa)(i,j,a,b) * (*t2_aa)(i,j,a,b);
        }
        
        if (t2_bb && t2_bb->size() > 0) {
            for (int i = 0; i < nocc_b_; i++)
                for (int j = 0; j < nocc_b_; j++)
                    for (int a = 0; a < nvirt_b_; a++)
                        for (int b = 0; b < nvirt_b_; b++)
                            norm2 += (*t2_bb)(i,j,a,b) * (*t2_bb)(i,j,a,b);
        }
        
        if (t2_ab && t2_ab->size() > 0) {
            for (int i = 0; i < nocc_a_; i++)
                for (int j = 0; j < nocc_b_; j++)
                    for (int a = 0; a < nvirt_a_; a++)
                        for (int b = 0; b < nvirt_b_; b++)
                            norm2 += (*t2_ab)(i,j,a,b) * (*t2_ab)(i,j,a,b);
        }
    }
    
    return norm2;
}

void Wavefunction::print_summary(std::ostream& os) const {
    os << "\n=== Wavefunction Summary ===\n";
    os << "Occupied α: " << nocc_a_ << ", Virtual α: " << nvirt_a_ << "\n";
    os << "Occupied β: " << nocc_b_ << ", Virtual β: " << nvirt_b_ << "\n";
    
    // Check what's stored
    os << "\nAmplitudes stored:\n";
    if (has_amplitudes(1)) os << "  ✓ First-order (MP2)\n";
    if (has_amplitudes(2)) os << "  ✓ Second-order (MP3)\n";
    if (has_amplitudes(3)) os << "  ✓ Third-order (MP4)\n";
    
    // Wavefunction norm
    if (has_amplitudes(1)) {
        double norm2 = norm_squared(1);
        os << "\nWavefunction norm² (MP2): " << std::fixed << std::setprecision(6) 
           << norm2 << "\n";
        os << "Correlation contribution: " << (norm2 - 1.0) << "\n";
    }
    
    // Dominant excitations
    if (has_amplitudes(1)) {
        auto dominant = dominant_amplitudes(0.01, 10);  // Top 10, threshold 0.01
        
        os << "\nDominant excitations (|t| > 0.01):\n";
        os << "  Amplitude   Excitation\n";
        os << "  ---------   ----------\n";
        for (const auto& exc : dominant) {
            os << "  " << exc.to_string() << "\n";
        }
        
        if (dominant.empty()) {
            os << "  (None above threshold)\n";
        }
    }
}

void Wavefunction::print_latex(std::ostream& os, int max_terms) const {
    if (!has_amplitudes(1)) {
        os << "|Ψ⟩ = |Ψ_HF⟩  (no correlation)\n";
        return;
    }
    
    auto dominant = dominant_amplitudes(0.0, max_terms);  // Get top N
    
    os << "|Ψ⟩ = |Ψ_HF⟩";
    
    for (const auto& exc : dominant) {
        // Format sign
        if (exc.amplitude >= 0) {
            os << " + ";
        } else {
            os << " - ";
        }
        
        // Format amplitude and excitation
        os << std::fixed << std::setprecision(3) << std::abs(exc.amplitude);
        os << " |Φ_{" << exc.i << "," << exc.j << "}^{" 
           << exc.a << "," << exc.b << "}⟩(" << exc.spin << ")";
    }
    
    os << " + ...\n";
}

// ============================================================================
// Helper functions (OPDM, natural orbitals) - stub for now
// ============================================================================

Eigen::MatrixXd compute_opdm(const Wavefunction& wfn) {
    // TODO: Implement OPDM calculation (Month 4)
    // For now, return placeholder
    int ntot = wfn.nocc_alpha() + wfn.nvirt_alpha();
    return Eigen::MatrixXd::Identity(ntot, ntot);
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> 
compute_natural_orbitals(const Wavefunction& wfn) {
    // TODO: Implement natural orbital analysis (Month 4)
    // For now, return placeholder
    int ntot = wfn.nocc_alpha() + wfn.nvirt_alpha();
    Eigen::MatrixXd U = Eigen::MatrixXd::Identity(ntot, ntot);
    Eigen::VectorXd n = Eigen::VectorXd::Ones(ntot);
    return {U, n};
}

} // namespace foundation
} // namespace mshqc
