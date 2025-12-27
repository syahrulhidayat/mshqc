/**
 * @file cholesky_sa_caspt2.h
 * @brief State-Averaged CASPT2 with TBLIS Tensor Contractions
 * @author Muhamad Syahrul Hidayat
 * @date 2025-12-16
 */

#ifndef MSHQC_CHOLESKY_SA_CASPT2_H
#define MSHQC_CHOLESKY_SA_CASPT2_H

#include "mshqc/mcscf/cholesky_sa_casscf.h"

#include <vector>
#include <memory>
#include <Eigen/Dense>

namespace mshqc {
namespace mcscf {

// ============================================================================
// STRUCT DEFINITIONS (Updated for Full CASPT2)
// ============================================================================

// Placeholder jika MOIntegrals didefinisikan di tempat lain, 
// tapi jika belum ada, kita definisikan dummy agar compile.
// (Sebaiknya MOIntegrals didefinisikan lengkap jika digunakan TBLIS)
struct MOIntegrals {
    // Definisi tensor integral MO (kosongkan jika hanya pakai loop Cholesky)
    // Implementasi TBLIS memerlukan ini terisi.
};

// [PERBAIKAN UTAMA] Update struct amplitudo dengan Semi-Internal vectors
struct PT2Amplitudes {
    // Standard Amplitudes
    std::vector<double> t2_core;   // (i,j -> a,b)
    std::vector<double> t2_active; // (t,u -> a,b)
    
    // [NEW] Semi-Internal Amplitudes (Required for Full CASPT2)
    std::vector<double> t2_semi1;  // Class C (i,t -> a,b)
    std::vector<double> t2_semi2;  // Class D (i,j -> t,a)

    int n_in, n_act, n_vir;

    // Resize function
    void resize(int ni, int na, int nv) {
        n_in = ni; n_act = na; n_vir = nv;
        long n_virt_sq = (long)nv * nv;
        long n_in_sq = (long)ni * ni;

        t2_core.assign(n_in_sq * n_virt_sq, 0.0);
        t2_active.assign((long)na * na * n_virt_sq, 0.0);
        
        // Resize Semi-Internal vectors [NEW]
        t2_semi1.assign((long)ni * na * n_virt_sq, 0.0);
        t2_semi2.assign(n_in_sq * na * nv, 0.0);
    }

    // Indexing Helpers
    long idx_core(int i, int j, int a, int b) const {
        return ((i * n_in + j) * n_vir + a) * n_vir + b;
    }
    
    long idx_active(int t, int u, int a, int b) const {
        return ((t * n_act + u) * n_vir + a) * n_vir + b;
    }

    // [NEW] Helper for Semi-Internal 1 (i, t -> a, b)
    long idx_semi1(int i, int t, int a, int b) const {
        return ((i * n_act + t) * n_vir + a) * n_vir + b;
    }
};

struct CASPT2Config {
    double shift = 0;
    double zero_thresh = 1e-8;
    int print_level = 1;
    bool use_tblis = true;           // Enable TBLIS contractions
    bool export_amplitudes = true;   // Store T2 for PT3
};

struct CASPT2Result {
    std::vector<double> e_cas;
    std::vector<double> e_pt2;
    std::vector<double> e_total;
    
    // Amplitude export for PT3
    std::vector<PT2Amplitudes> amplitudes;  // Per-state amplitudes
    
    // Shared MO integrals (compute once, use in PT3)
    std::shared_ptr<MOIntegrals> mo_ints;
};

// ============================================================================
// CLASS DECLARATION
// ============================================================================

class CholeskySACASPT2 {
public:
    CholeskySACASPT2(const SACASResult& result,
                     const std::vector<Eigen::VectorXd>& vecs,
                     int n_basis,
                     const ActiveSpace& active_space,
                     const CASPT2Config& config);

    /**
     * @brief Main compute with optional MO integral reuse
     * @param mo_ints Pre-computed integrals (nullptr = compute new)
     */
    CASPT2Result compute(std::shared_ptr<MOIntegrals> mo_ints = nullptr);

private:
    SACASResult cas_res_;
    std::vector<Eigen::VectorXd> L_ao_;
    int nbasis_;
    ActiveSpace active_space_;
    CASPT2Config config_;

    int n_inact_, n_act_, n_virt_;
    
    // === Core Functions ===
    
    /**
     * @brief Transform Cholesky vectors to full MO integrals
     * Only called if mo_ints not provided
     */
    std::shared_ptr<MOIntegrals> compute_mo_integrals(const Eigen::MatrixXd& C_mo);
    
    /**
     * @brief Compute PT2 energy for single state (TBLIS version)
     */
    double compute_state_pt2_tblis(int state_idx,
                                    const MOIntegrals& mo_ints,
                                    const Eigen::VectorXd& eps,
                                    PT2Amplitudes* amps = nullptr);
    
    /**
     * @brief Legacy loop-based PT2 (fallback if TBLIS unavailable)
     */
    double compute_state_pt2_loops(int state_idx,
                                    const std::vector<Eigen::MatrixXd>& L_mo,
                                    const Eigen::VectorXd& eps,
                                    PT2Amplitudes* amps = nullptr);
    
    // === TBLIS Helpers ===
    
    double compute_core_term_tblis(const MOIntegrals& mo_ints,
                                    const Eigen::VectorXd& eps,
                                    PT2Amplitudes& amps);
    
    double compute_active_term_tblis(int state_idx,
                                      const MOIntegrals& mo_ints,
                                      const Eigen::VectorXd& eps,
                                      PT2Amplitudes& amps);
    
    // === Utility ===
    std::vector<Eigen::MatrixXd> transform_cholesky_to_mo(const Eigen::MatrixXd& C_mo) const;
    
};

} // namespace mcscf
} // namespace mshqc

#endif