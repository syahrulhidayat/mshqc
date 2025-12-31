/**
 * @file src/mcscf/cholesky_sa_casscf.cc
 * @brief Implementation of Cholesky State-Averaged CASSCF
 * @details Optimized with GEMM + Energy Diagnostics + Full Gradient Logic
 */

#include "mshqc/mcscf/cholesky_sa_casscf.h"
#include "mshqc/ci/slater_condon.h"
#include "mshqc/ci/determinant.h"
#include "mshqc/ci/ci_utils.h"
#include "mshqc/integrals/cholesky_eri.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unsupported/Eigen/MatrixFunctions> 

namespace mshqc {
namespace mcscf {

// ============================================================================
// CONSTRUCTORS
// ============================================================================

CholeskySACASSCF::CholeskySACASSCF(const Molecule& mol, const BasisSet& basis,
                                   std::shared_ptr<IntegralEngine> integrals,
                                   const ActiveSpace& active_space, const SACASConfig& config)
    : mol_(mol), basis_(basis), integrals_(integrals), 
      active_space_(active_space), config_(config), vectors_provided_(false) {}

CholeskySACASSCF::CholeskySACASSCF(const Molecule& mol, const BasisSet& basis,
                                   std::shared_ptr<IntegralEngine> integrals,
                                   const ActiveSpace& active_space, const SACASConfig& config,
                                   const std::vector<Eigen::VectorXd>& L_vectors)
    : mol_(mol), basis_(basis), integrals_(integrals), 
      active_space_(active_space), config_(config), 
      L_ao_vectors_(L_vectors), vectors_provided_(true) 
{
    if(config.print_level > 0) std::cout << "  [SA-CASSCF] Vectors reused.\n";
}

// ============================================================================
// HELPERS
// ============================================================================

void CholeskySACASSCF::ensure_cholesky_vectors() {
    if (vectors_provided_ && !L_ao_vectors_.empty()) return;

    if (config_.print_level > 0) 
        std::cout << "  [SA-CASSCF] Decomposing Integrals...\n";
    
    mshqc::integrals::CholeskyERI chol(config_.cholesky_thresh);
    auto eri_ao = integrals_->compute_eri();
    chol.decompose(eri_ao);
    L_ao_vectors_ = chol.get_L_vectors();
    vectors_provided_ = true;
}

std::vector<Eigen::MatrixXd> CholeskySACASSCF::transform_cholesky_to_mo(
    const Eigen::MatrixXd& C_mo) const 
{
    int nbf = C_mo.rows();
    int n_chol = L_ao_vectors_.size();
    std::vector<Eigen::MatrixXd> L_mo(n_chol);
    
    #pragma omp parallel for schedule(dynamic)
    for(int k = 0; k < n_chol; ++k) {
        Eigen::Map<const Eigen::MatrixXd> L_ao(L_ao_vectors_[k].data(), nbf, nbf);
        L_mo[k] = C_mo.transpose() * L_ao * C_mo;
    }
    return L_mo;
}

// ============================================================================
// COMPUTE ENGINE
// ============================================================================

SACASResult CholeskySACASSCF::compute(const SCFResult& initial_guess) {
    return compute(initial_guess.C_alpha);
}

SACASResult CholeskySACASSCF::compute(const Eigen::MatrixXd& initial_orbitals) {
    ensure_cholesky_vectors();
    
    int n_basis = initial_orbitals.rows();
    Eigen::MatrixXd C_mo = initial_orbitals;
    
    // --- 1. Generate Determinants ---
    auto generate_dets = [&](int n_orb, int n_elec) {
        int n_alpha = (n_elec + mol_.multiplicity() - 1) / 2;
        int n_beta = n_elec - n_alpha;
        
        auto combos = [](int n, int k) {
            std::vector<std::vector<int>> r;
            if (k==0) { r.push_back({}); return r; }
            std::string bm(k, 1); bm.resize(n, 0);
            do {
                std::vector<int> c; 
                for(int i=0; i<n; ++i) if(bm[i]) c.push_back(i);
                r.push_back(c);
            } while(std::prev_permutation(bm.begin(), bm.end()));
            return r;
        };
        auto a_s = combos(n_orb, n_alpha); 
        auto b_s = combos(n_orb, n_beta);
        std::vector<ci::Determinant> d;
        for(auto& a : a_s) for(auto& b : b_s) d.emplace_back(a, b);
        return d;
    };
    
    int n_act_elec = active_space_.n_elec_active();
    int n_act_orb = active_space_.n_active();
    auto determinants = generate_dets(n_act_orb, n_act_elec);
    
    // --- [FIX: Safety Check for N_States] ---
    // Mencegah crash jika user meminta state lebih banyak dari determinan yang ada
    if (config_.n_states > (int)determinants.size()) {
        if (config_.print_level >= 0) {
            std::cout << "\n  [WARNING] Requested " << config_.n_states << " states, but Active Space only supports " 
                      << determinants.size() << " determinants.\n"
                      << "            Reducing n_states to " << determinants.size() << ".\n";
        }
        config_.n_states = (int)determinants.size();
        
        // Resize weights & Renormalize
        if ((int)config_.weights.size() > config_.n_states) {
            config_.weights.resize(config_.n_states);
            double w_sum = 0.0;
            for(double w : config_.weights) w_sum += w;
            if(std::abs(w_sum) > 1e-12) {
                for(double& w : config_.weights) w /= w_sum;
            }
        }
    }
    // ----------------------------------------

    SACASResult res;
    res.state_energies.resize(config_.n_states);
    res.ci_vectors.resize(config_.n_states);
    
    double e_avg_prev = 0.0;
    double damping = config_.rotation_damping;

    std::vector<Eigen::MatrixXd> L_mo;        
    Eigen::MatrixXd rdm1_avg;                 
    Eigen::MatrixXd F_gen;                    
    bool is_converged = false;

    // --- MAIN OPTIMIZATION LOOP ---
    for (int iter = 0; iter < config_.max_iter; ++iter) {
        
        L_mo = transform_cholesky_to_mo(C_mo);
        
        // B. Build Active Space Integrals (Optimized)
        ci::CIIntegrals ci_ints;
        auto act_idx = active_space_.active_indices();
        auto inact_idx = active_space_.inactive_indices();
        int n_chol = L_mo.size();
        
        // 1-body integrals construction
        {
            Eigen::MatrixXd h_core = integrals_->compute_kinetic() + integrals_->compute_nuclear();
            Eigen::MatrixXd h_mo = C_mo.transpose() * h_core * C_mo;
            ci_ints.h_alpha = Eigen::MatrixXd::Zero(n_act_orb, n_act_orb);
            
            #pragma omp parallel for schedule(dynamic)
            for(int p=0; p<n_act_orb; ++p) {
                for(int q=0; q<n_act_orb; ++q) {
                    int P = act_idx[p]; int Q = act_idx[q];
                    double val = h_mo(P, Q);
                    for(const auto& L : L_mo) {
                         double core_J = 0.0;
                         double core_K = 0.0;
                         for(int i : inact_idx) {
                             core_J += L(i,i);
                             core_K += L(i,P) * L(i,Q);
                         }
                         val += 2.0 * core_J * L(P,Q) - core_K;
                    }
                    ci_ints.h_alpha(p,q) = val;
                }
            }
            ci_ints.h_beta = ci_ints.h_alpha;
        }
        
        // 2-body integrals (GEMM Optimized)
        {
            int dim_act2 = n_act_orb * n_act_orb;
            Eigen::MatrixXd L_flat(dim_act2, n_chol);

            #pragma omp parallel for
            for(int k=0; k<n_chol; ++k) {
                for(int t=0; t<n_act_orb; ++t) {
                    for(int u=0; u<n_act_orb; ++u) {
                        L_flat(t*n_act_orb + u, k) = L_mo[k](act_idx[t], act_idx[u]);
                    }
                }
            }

            // ERI_flat(X, Y) menyimpan integral Mulliken (t u | v w)
            // dimana X = t*n + u, Y = v*n + w
            Eigen::MatrixXd ERI_flat = L_flat * L_flat.transpose();

            ci_ints.eri_aaaa = Eigen::Tensor<double, 4>(n_act_orb, n_act_orb, n_act_orb, n_act_orb);
            ci_ints.eri_bbbb = Eigen::Tensor<double, 4>(n_act_orb, n_act_orb, n_act_orb, n_act_orb);
            ci_ints.eri_aabb = Eigen::Tensor<double, 4>(n_act_orb, n_act_orb, n_act_orb, n_act_orb);

            #pragma omp parallel for collapse(4)
            for(int p=0; p<n_act_orb; ++p) {
                for(int q=0; q<n_act_orb; ++q) {
                    for(int r=0; r<n_act_orb; ++r) {
                        for(int s=0; s<n_act_orb; ++s) {
                            // ---------------------------------------------------------
                            // CORRECTION: MAPPING PHYSICIST <pq|rs> -> MULLIKEN (pr|qs)
                            // ---------------------------------------------------------
                            // Target: Integrals.eri(p,q,r,s) harus berisi <pq|rs>
                            // Sumber: ERI_flat(X, Y) berisi Mulliken (tu|vw)
                            // Hubungan: <pq|rs> = (pr|qs)
                            // Maka kita butuh indeks Mulliken: t=p, u=r, v=q, w=s
                            
                            // Akses baris (p*n + r) dan kolom (q*n + s)
                            double val_phys = ERI_flat(p*n_act_orb + r, q*n_act_orb + s);
                            
                            ci_ints.eri_aaaa(p,q,r,s) = val_phys; 
                            ci_ints.eri_bbbb(p,q,r,s) = val_phys;
                            
                            // Untuk aabb (Alpha-Beta), integral spasialnya sama: <pq|rs>
                            ci_ints.eri_aabb(p,q,r,s) = val_phys;
                        }
                    }
                }
            }
        }
        
        ci_ints.e_nuc = mol_.nuclear_repulsion_energy();

        // C. Solve CI
        Eigen::MatrixXd H_ci = ci::build_hamiltonian(determinants, ci_ints);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H_ci);
        
        // Core energy calculation
        double e_core = 0.0;
        {
             Eigen::MatrixXd h_core = integrals_->compute_kinetic() + integrals_->compute_nuclear();
             Eigen::MatrixXd h_mo = C_mo.transpose() * h_core * C_mo;
             for(int i : inact_idx) {
                 e_core += 2.0 * h_mo(i,i);
                 for(int j : inact_idx) {
                     double J=0, K=0;
                     for(const auto& L : L_mo) { 
                        J += L(i,i)*L(j,j); K += L(i,j)*L(i,j); 
                     }
                     e_core += (2.0*J - K);
                 }
             }
        }

        // DIAGNOSTIC PRINT (Hanya iterasi 1)
        if (iter == 0 && config_.print_level == 0) { 
            std::cout << "  [DIAGNOSTIC] E_Core: " << e_core 
                      << " | E_Nuc: " << ci_ints.e_nuc 
                      << " | E_CI[0]: " << es.eigenvalues()(0) 
                      << " | Total: " << e_core + ci_ints.e_nuc + es.eigenvalues()(0) << "\n";
        }

        double e_avg_curr = 0.0;
        rdm1_avg = Eigen::MatrixXd::Zero(n_act_orb, n_act_orb);
        
        for(int s=0; s<config_.n_states; ++s) {
            res.state_energies[s] = es.eigenvalues()(s) + e_core + mol_.nuclear_repulsion_energy();
            res.ci_vectors[s] = es.eigenvectors().col(s);
            e_avg_curr += config_.weights[s] * res.state_energies[s];
            
            for(int I=0; I<determinants.size(); ++I) {
                double cI = res.ci_vectors[s](I);
                if(std::abs(cI) < 1e-9) continue;
                for(int p=0; p<n_act_orb; ++p) {
                    double occ = 0.0;
                    if(determinants[I].is_occupied(p, true)) occ += 1.0;
                    if(determinants[I].is_occupied(p, false)) occ += 1.0;
                    rdm1_avg(p,p) += config_.weights[s] * cI * cI * occ;
                }
            }
        }

        double dE = e_avg_curr - e_avg_prev;
        if (config_.print_level > 0) {
            std::cout << "  Iter " << std::setw(2) << iter + 1
                      << " E_avg: " << std::fixed << std::setprecision(8) << e_avg_curr
                      << " dE: " << std::scientific << dE << "\n";
        }

        is_converged = (iter > 0 && std::abs(dE) < config_.e_thresh);
        
        res.converged = is_converged;
        res.n_iterations = iter + 1;
        res.e_avg = e_avg_curr;
        res.C_mo = C_mo;

        F_gen = compute_generalized_fock({rdm1_avg}, L_mo, C_mo);

        if (is_converged) break;

        auto gradient = compute_orbital_gradient(F_gen, C_mo);
        Eigen::VectorXd kappa = -1.0 * damping * gradient;
        double max_step = 0.2;
        if(kappa.norm() > max_step) kappa *= (max_step / kappa.norm());
        
        C_mo = apply_rotation(C_mo, kappa);
        e_avg_prev = e_avg_curr;
        
        if (dE > 0) damping *= 0.5; 
        else damping = std::min(1.0, damping * 1.1);
    }

    // ========================================================================
    // FINALIZATION: CANONICALIZATION & SORTING
    // ========================================================================

    if (config_.print_level > 0) 
        std::cout << "  [SA-CASSCF] Finalizing & Canonicalizing orbitals...\n";

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_fock(F_gen);
    Eigen::VectorXd eps_all = es_fock.eigenvalues();
    Eigen::MatrixXd U = es_fock.eigenvectors();
    res.C_mo = res.C_mo * U;

    std::vector<std::pair<double, int>> sorted_orbs;
    for(int i=0; i<eps_all.size(); ++i) sorted_orbs.push_back({eps_all(i), i});
    std::sort(sorted_orbs.begin(), sorted_orbs.end());

    auto inact = active_space_.inactive_indices();
    auto act = active_space_.active_indices();
    auto virt = active_space_.virtual_indices();
    int n_inact = inact.size();
    int n_act = act.size();

    res.orbital_energies.resize(n_basis);
    for(int i=0; i<n_inact; ++i) res.orbital_energies[inact[i]] = sorted_orbs[i].first;
    for(int i=0; i<n_act; ++i) res.orbital_energies[act[i]] = sorted_orbs[n_inact + i].first;
    for(int i=0; i<virt.size(); ++i) res.orbital_energies[virt[i]] = sorted_orbs[n_inact + n_act + i].first;

    // Store State-Specific RDMs for PT2
    res.rdm1_states.resize(config_.n_states);
    for (int s = 0; s < config_.n_states; ++s) {
        res.rdm1_states[s] = Eigen::MatrixXd::Zero(n_act_orb, n_act_orb);
        for(int I=0; I<determinants.size(); ++I) {
            double cI = res.ci_vectors[s](I);
            if(std::abs(cI) < 1e-12) continue;
            for(int p=0; p<n_act_orb; ++p) {
                double occ = 0.0;
                if(determinants[I].is_occupied(p, true)) occ += 1.0;
                if(determinants[I].is_occupied(p, false)) occ += 1.0;
                res.rdm1_states[s](p,p) += cI * cI * occ;
            }
        }
    }

    return res;
}

// ============================================================================
// GRADIENT & ROTATION LOGIC (IMPLEMENTASI LENGKAP)
// ============================================================================

Eigen::MatrixXd CholeskySACASSCF::compute_generalized_fock(
    const std::vector<Eigen::MatrixXd>& rdm1_states,
    const std::vector<Eigen::MatrixXd>& L_mo,
    const Eigen::MatrixXd& C_mo
) const {
    int nbasis = C_mo.rows();
    Eigen::MatrixXd h_core = integrals_->compute_kinetic() + integrals_->compute_nuclear();
    Eigen::MatrixXd F = C_mo.transpose() * h_core * C_mo;
    
    const auto& P_avg_act = rdm1_states[0];
    auto act_idx = active_space_.active_indices();
    auto inact_idx = active_space_.inactive_indices();
    int n_chol = L_mo.size();

    // 1. Calculate Density Scalars B[k]
    Eigen::VectorXd B = Eigen::VectorXd::Zero(n_chol);
    #pragma omp parallel for
    for(int k=0; k<n_chol; ++k) {
        double val = 0.0;
        for(int i : inact_idx) val += 2.0 * L_mo[k](i,i);
        for(int t=0; t<act_idx.size(); ++t) {
            for(int u=0; u<act_idx.size(); ++u) {
                 val += P_avg_act(t, u) * L_mo[k](act_idx[t], act_idx[u]);
            }
        }
        B(k) = val;
    }

    // 2. Add Coulomb (J)
    #pragma omp parallel for collapse(2)
    for(int p=0; p<nbasis; ++p) {
        for(int q=0; q<nbasis; ++q) {
            double val_J = 0.0;
            for(int k=0; k<n_chol; ++k) val_J += B(k) * L_mo[k](p,q);
            F(p,q) += val_J;
        }
    }

    // 3. Add Exchange (K)
    #pragma omp parallel for collapse(2)
    for(int p=0; p<nbasis; ++p) {
        for(int q=0; q<nbasis; ++q) {
             double val_K = 0.0;
             for(int k=0; k<n_chol; ++k) {
                 for(int i : inact_idx) val_K += L_mo[k](i,p) * L_mo[k](i,q);
                 for(int t=0; t<act_idx.size(); ++t) {
                     for(int u=0; u<act_idx.size(); ++u) {
                         double gamma = P_avg_act(t,u);
                         if(std::abs(gamma) > 1e-9)
                             val_K += 0.5 * gamma * L_mo[k](act_idx[t], p) * L_mo[k](act_idx[u], q);
                     }
                 }
             }
             F(p,q) -= val_K;
        }
    }
    return F;
}

Eigen::VectorXd CholeskySACASSCF::compute_orbital_gradient(
    const Eigen::MatrixXd& F_gen, const Eigen::MatrixXd& C_mo
) const {
    std::vector<double> g_vals;
    auto inact = active_space_.inactive_indices();
    auto act = active_space_.active_indices();
    auto virt = active_space_.virtual_indices();
    
    // Gradient: 2(F_ij - F_ji)
    for(int i : inact) for(int t : act) g_vals.push_back(2.0 * (F_gen(i,t) - F_gen(t,i)));
    for(int t : act) for(int a : virt) g_vals.push_back(2.0 * (F_gen(t,a) - F_gen(a,t)));
    for(int i : inact) for(int a : virt) g_vals.push_back(2.0 * (F_gen(i,a) - F_gen(a,i)));
    
    Eigen::VectorXd g(g_vals.size());
    for(size_t i=0; i<g_vals.size(); ++i) g(i) = g_vals[i];
    return g;
}

Eigen::MatrixXd CholeskySACASSCF::apply_rotation(const Eigen::MatrixXd& C, const Eigen::VectorXd& kappa) const {
    int n = C.rows();
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n, n);
    auto inact = active_space_.inactive_indices();
    auto act = active_space_.active_indices();
    auto virt = active_space_.virtual_indices();
    
    int idx = 0;
    for(int i : inact) for(int t : act) { K(i,t) = kappa(idx); K(t,i) = -kappa(idx); idx++; }
    for(int t : act) for(int a : virt)  { K(t,a) = kappa(idx); K(a,t) = -kappa(idx); idx++; }
    for(int i : inact) for(int a : virt){ K(i,a) = kappa(idx); K(a,i) = -kappa(idx); idx++; }
    
    return C * K.exp();
}

} // namespace mcscf
} // namespace mshqc