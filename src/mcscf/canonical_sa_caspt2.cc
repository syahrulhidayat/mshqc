/**
 * @file src/mcscf/canonical_sa_caspt2.cc
 * @brief Implementation of Canonical SA-CASPT2 (High Performance O(N^5))
 * @details 
 * OPTIMIZATION:
 * - Replaced Naive O(N^8) transformation with Stepwise O(N^5) algorithm.
 * - Uses ~3x N^4 memory buffer but is orders of magnitude faster.
 */

#include "mshqc/mcscf/canonical_sa_caspt2.h"
#include "mshqc/mcscf/cholesky_sa_caspt2.h" // Reuse Config & Result structs
#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <vector>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace mcscf {

// ============================================================================
// CONSTRUCTOR
// ============================================================================

CanonicalSACASPT2::CanonicalSACASPT2(const SACASResult& result,
                                     std::shared_ptr<IntegralEngine> integrals,
                                     const BasisSet& basis,
                                     const ActiveSpace& active_space,
                                     const CASPT2Config& config)
    : cas_res_(result), integrals_(integrals), basis_(basis),
      active_space_(active_space), config_(config)
{
    n_inact_ = active_space.n_inactive();
    n_act_   = active_space.n_active();
    n_virt_  = active_space.n_virtual();
    nbasis_  = basis.n_basis_functions();
}

// ============================================================================
// INTEGRAL TRANSFORMATION (O(N^5) STEPWISE) - SANGAT CEPAT
// ============================================================================

Eigen::Tensor<double, 4> CanonicalSACASPT2::transform_integrals_to_mo() const {
    if (config_.print_level > 0) 
        std::cout << "  [Canonical PT2] Transforming AO Integrals to MO (Stepwise O(N^5))...\n";

    // 1. Get AO Integrals
    Eigen::Tensor<double, 4> ao_eri = integrals_->compute_eri();
    const Eigen::MatrixXd& C = cas_res_.C_mo;
    int n = nbasis_;
    
    // Total size N^4
    long size = (long)n * n * n * n;
    
    // Buffer Tensors (Flattened for easier OpenMP access)
    // Kita gunakan std::vector untuk manajemen memori otomatis
    std::vector<double> T1(size, 0.0); // (mu, nu, lam, s)
    
    // --- STEP 1: Transform index 4 (sig -> s) ---
    // Target: (mu, nu, lam, s) = Sum_sig (mu, nu, lam, sig) * C(sig, s)
    #pragma omp parallel for collapse(3) schedule(static)
    for(int mu=0; mu<n; ++mu) {
        for(int nu=0; nu<n; ++nu) {
            for(int lam=0; lam<n; ++lam) {
                for(int s=0; s<n; ++s) {
                    double val = 0.0;
                    for(int sig=0; sig<n; ++sig) {
                        val += ao_eri(mu, nu, lam, sig) * C(sig, s);
                    }
                    T1[((mu*n + nu)*n + lam)*n + s] = val;
                }
            }
        }
    }

    // --- STEP 2: Transform index 3 (lam -> r) ---
    // Target: (mu, nu, r, s) = Sum_lam (mu, nu, lam, s) * C(lam, r)
    // Reuse T1 buffer? No, we need a second buffer to avoid overwrite race
    std::vector<double> T2(size, 0.0);
    
    #pragma omp parallel for collapse(3) schedule(static)
    for(int mu=0; mu<n; ++mu) {
        for(int nu=0; nu<n; ++nu) {
            for(int r=0; r<n; ++r) {
                for(int s=0; s<n; ++s) {
                    double val = 0.0;
                    for(int lam=0; lam<n; ++lam) {
                        val += T1[((mu*n + nu)*n + lam)*n + s] * C(lam, r);
                    }
                    T2[((mu*n + nu)*n + r)*n + s] = val;
                }
            }
        }
    }
    // Free T1
    std::vector<double>().swap(T1);

    // --- STEP 3: Transform index 2 (nu -> q) ---
    // Target: (mu, q, r, s) = Sum_nu (mu, nu, r, s) * C(nu, q)
    std::vector<double> T3(size, 0.0);
    
    #pragma omp parallel for collapse(3) schedule(static)
    for(int mu=0; mu<n; ++mu) {
        for(int q=0; q<n; ++q) {
            for(int r=0; r<n; ++r) {
                for(int s=0; s<n; ++s) {
                    double val = 0.0;
                    for(int nu=0; nu<n; ++nu) {
                        val += T2[((mu*n + nu)*n + r)*n + s] * C(nu, q);
                    }
                    T3[((mu*n + q)*n + r)*n + s] = val;
                }
            }
        }
    }
    // Free T2
    std::vector<double>().swap(T2);

    // --- STEP 4: Transform index 1 (mu -> p) ---
    // Target: (p, q, r, s) = Sum_mu (mu, q, r, s) * C(mu, p)
    Eigen::Tensor<double, 4> mo_eri(n, n, n, n);
    
    #pragma omp parallel for collapse(4) schedule(static)
    for(int p=0; p<n; ++p) {
        for(int q=0; q<n; ++q) {
            for(int r=0; r<n; ++r) {
                for(int s=0; s<n; ++s) {
                    double val = 0.0;
                    for(int mu=0; mu<n; ++mu) {
                        val += T3[((mu*n + q)*n + r)*n + s] * C(mu, p);
                    }
                    mo_eri(p, q, r, s) = val;
                }
            }
        }
    }
    
    return mo_eri;
}

// ============================================================================
// COMPUTE ENGINE (Sama seperti sebelumnya)
// ============================================================================

CASPT2Result CanonicalSACASPT2::compute() {
    CASPT2Result res;
    int n_states = cas_res_.state_energies.size();
    
    if (config_.print_level > 0) {
        std::cout << "\n" << std::string(70, '-') << "\n";
        std::cout << "  Canonical SA-CASPT2 Calculation (O(N^5) Optimized)\n";
        std::cout << "  Shift: " << std::fixed << std::setprecision(10) << config_.shift << " Ha\n";
        std::cout << std::string(70, '-') << "\n";
    }

    // Call Optimized Transform
    Eigen::Tensor<double, 4> mo_eri = transform_integrals_to_mo();
    
    if (cas_res_.orbital_energies.empty()) {
        std::cerr << "Error: Orbital energies missing in SACASResult.\n";
        return res;
    }

    Eigen::VectorXd eps = Eigen::Map<const Eigen::VectorXd>(
        cas_res_.orbital_energies.data(), nbasis_);

    if (config_.export_amplitudes) {
        res.amplitudes.resize(n_states);
    }

    for (int s = 0; s < n_states; ++s) {
        PT2Amplitudes* amp_ptr = config_.export_amplitudes ? &res.amplitudes[s] : nullptr;

        double E2 = compute_state_pt2(s, mo_eri, eps, amp_ptr);
        
        res.e_cas.push_back(cas_res_.state_energies[s]);
        res.e_pt2.push_back(E2);
        res.e_total.push_back(cas_res_.state_energies[s] + E2);
        
        if (config_.print_level > 0) {
            std::cout << "  > State " << s 
                      << ": E_PT2 = " << std::fixed << std::setprecision(8) << E2 
                      << " | E_Total = " << res.e_total[s] << "\n";
        }
    }
    
    return res;
}

// ============================================================================
// STATE KERNEL (Sama seperti sebelumnya)
// ============================================================================

double CanonicalSACASPT2::compute_state_pt2(int state_idx, 
                                            const Eigen::Tensor<double, 4>& mo_eri,
                                            const Eigen::VectorXd& eps,
                                            PT2Amplitudes* amps) 
{
    if (amps) amps->resize(n_inact_, n_act_, n_virt_);

    auto inact = active_space_.inactive_indices();
    auto act = active_space_.active_indices();
    auto virt = active_space_.virtual_indices();
    
    double energy = 0.0;
    
    Eigen::MatrixXd rdm1 = Eigen::MatrixXd::Identity(n_act_, n_act_);
    if (state_idx < (int)cas_res_.rdm1_states.size()) {
        rdm1 = cas_res_.rdm1_states[state_idx];
    }

    // ------------------------------------------------------------------------
    // CLASS A: CLOSED-SHELL (Core-Core -> Virt-Virt)
    // ------------------------------------------------------------------------
    double e2_closed = 0.0;
    
    #pragma omp parallel for reduction(+:e2_closed)
    for(int i_idx=0; i_idx<n_inact_; ++i_idx) {
        int i = inact[i_idx];
        for(int j_idx=0; j_idx<n_inact_; ++j_idx) {
            int j = inact[j_idx];
            for(int a_idx=0; a_idx<n_virt_; ++a_idx) {
                int a = virt[a_idx];
                for(int b_idx=0; b_idx<n_virt_; ++b_idx) {
                    int b = virt[b_idx];
                    
                    double val_iajb = mo_eri(i, a, j, b);
                    double val_ibja = mo_eri(i, b, j, a);
                    
                    double denom = eps(i) + eps(j) - eps(a) - eps(b) + config_.shift;
                    if(std::abs(denom) < 1e-12) denom = (denom >= 0 ? 1e-12 : -1e-12);
                    
                    double t = val_iajb / denom;
                    e2_closed += t * (2.0 * val_iajb - val_ibja);

                    if (amps) amps->t2_core[amps->idx_core(i_idx, j_idx, a_idx, b_idx)] = t;
                }
            }
        }
    }
    energy += e2_closed;

    // ------------------------------------------------------------------------
    // CLASS B: ACTIVE-ACTIVE (Act-Act -> Virt-Virt)
    // ------------------------------------------------------------------------
    double e2_act = 0.0;
    #pragma omp parallel for reduction(+:e2_act)
    for(int t_idx=0; t_idx<n_act_; ++t_idx) {
        int t = act[t_idx];
        for(int u_idx=0; u_idx<n_act_; ++u_idx) {
            int u = act[u_idx];
            
            double dens = rdm1(t_idx, u_idx);
            if (std::abs(dens) < 1e-9) continue;
            
            for(int a_idx=0; a_idx<n_virt_; ++a_idx) {
                int a = virt[a_idx];
                for(int b_idx=0; b_idx<n_virt_; ++b_idx) {
                    int b = virt[b_idx];
                    
                    double val_taub = mo_eri(t, a, u, b);
                    double val_tbua = mo_eri(t, b, u, a);
                    
                    double denom = eps(t) + eps(u) - eps(a) - eps(b) + config_.shift;
                    if(std::abs(denom) < 1e-12) denom = (denom >= 0 ? 1e-12 : -1e-12);
                    
                    double numerator = 0.0;
                    if (dens > 1.5) numerator = dens * val_taub * (2.0 * val_taub - val_tbua);
                    else numerator = dens * (val_taub * val_taub - val_taub * val_tbua);
                    
                    double t_val = val_taub / denom;
                    e2_act += numerator / denom;

                    if (amps) amps->t2_active[amps->idx_active(t_idx, u_idx, a_idx, b_idx)] = t_val;
                }
            }
        }
    }
    energy += e2_act;

    // ------------------------------------------------------------------------
    // CLASS C: SEMI-INTERNAL 1 (Core-Act -> Virt-Virt)
    // ------------------------------------------------------------------------
    double e2_semi1 = 0.0;
    #pragma omp parallel for reduction(+:e2_semi1)
    for(int i_idx=0; i_idx<n_inact_; ++i_idx) {
        int i = inact[i_idx];
        for(int t_idx=0; t_idx<n_act_; ++t_idx) {
            int t = act[t_idx];
            double occ_t = rdm1(t_idx, t_idx);
            if (occ_t < 1e-9) continue;
            
            for(int a_idx=0; a_idx<n_virt_; ++a_idx) {
                int a = virt[a_idx];
                for(int b_idx=0; b_idx<n_virt_; ++b_idx) {
                    int b = virt[b_idx];
                    
                    double val_iatb = mo_eri(i, a, t, b);
                    double val_ibta = mo_eri(i, b, t, a);
                    
                    double denom = eps(i) + eps(t) - eps(a) - eps(b) + config_.shift;
                    if(std::abs(denom) < 1e-12) denom = (denom >= 0 ? 1e-12 : -1e-12);
                    
                    double spin_factor = (occ_t > 1.5) ? 2.0 : 1.0;
                    double term = (spin_factor * val_iatb - val_ibta);
                    if (occ_t < 1.5) term = (val_iatb - 0.5 * val_ibta);
                    
                    double t_val = val_iatb / denom;
                    e2_semi1 += occ_t * t_val * term;

                    if (amps) amps->t2_semi1[amps->idx_semi1(i_idx, t_idx, a_idx, b_idx)] = t_val;
                }
            }
        }
    }
    energy += e2_semi1;

    // ------------------------------------------------------------------------
    // CLASS D: SEMI-INTERNAL 2 (Core-Core -> Act-Virt)
    // ------------------------------------------------------------------------
    double e2_semi2 = 0.0;
    #pragma omp parallel for reduction(+:e2_semi2)
    for(int i_idx=0; i_idx<n_inact_; ++i_idx) {
        int i = inact[i_idx];
        for(int j_idx=0; j_idx<n_inact_; ++j_idx) {
            int j = inact[j_idx];
            for(int t_idx=0; t_idx<n_act_; ++t_idx) {
                int t = act[t_idx];
                double hole_t = 2.0 - rdm1(t_idx, t_idx);
                if (hole_t < 1e-9) continue;
                
                for(int a_idx=0; a_idx<n_virt_; ++a_idx) {
                    int a = virt[a_idx];
                    double val_iajt = mo_eri(i, a, j, t);
                    double denom = eps(i) + eps(j) - eps(t) - eps(a) + config_.shift;
                    if(std::abs(denom) < 1e-12) denom = (denom >= 0 ? 1e-12 : -1e-12);
                    
                    e2_semi2 += hole_t * (2.0 * val_iajt * val_iajt) / denom;
                }
            }
        }
    }
    energy += e2_semi2;

    return energy;
}

} // namespace mcscf
} // namespace mshqc