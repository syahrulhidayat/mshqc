/**
 * @file cholesky_casscf.cc
 * @brief Cholesky-CASSCF Implementation (FULL)
 */

#include "mshqc/mcscf/cholesky_casscf.h"
#include "mshqc/ci/slater_condon.h"
#include "mshqc/ci/davidson.h"
#include "mshqc/ci/determinant.h"
#include "mshqc/ci/ci_utils.h"
#include "mshqc/scf.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

namespace mshqc {
namespace mcscf {

// ============================================================================
// CONSTRUCTORS
// ============================================================================

CholeskyCASSCF::CholeskyCASSCF(const Molecule& mol,
                               const BasisSet& basis,
                               std::shared_ptr<IntegralEngine> integrals,
                               const ActiveSpace& active_space)
    : mol_(mol), basis_(basis), integrals_(integrals), 
      active_space_(active_space), vectors_provided_(false)
{
}

CholeskyCASSCF::CholeskyCASSCF(const Molecule& mol,
                               const BasisSet& basis,
                               std::shared_ptr<IntegralEngine> integrals,
                               const ActiveSpace& active_space,
                               const std::vector<Eigen::VectorXd>& L_vectors)
    : mol_(mol), basis_(basis), integrals_(integrals), 
      active_space_(active_space), L_ao_vectors_(L_vectors), vectors_provided_(true)
{
    std::cout << "  [Cholesky-CASSCF] Initialized with " << L_ao_vectors_.size() 
              << " reused Cholesky vectors.\n";
}

// ============================================================================
// CHOLESKY HELPERS
// ============================================================================

void CholeskyCASSCF::ensure_cholesky_vectors() {
    if (vectors_provided_ && !L_ao_vectors_.empty()) return;

    std::cout << "  [Cholesky-CASSCF] No vectors provided. Running decomposition...\n";
    
    // Run decomposition on-the-fly
    integrals::CholeskyERI chol(cholesky_thresh_);
    auto eri_ao = integrals_->compute_eri();
    chol.decompose(eri_ao);
    
    L_ao_vectors_ = chol.get_L_vectors();
    vectors_provided_ = true;
    
    std::cout << "  [Cholesky-CASSCF] Decomposition complete. " 
              << L_ao_vectors_.size() << " vectors generated.\n";
}

std::vector<Eigen::MatrixXd> CholeskyCASSCF::transform_cholesky_to_mo(
    const Eigen::MatrixXd& C_mo) const 
{
    std::vector<Eigen::MatrixXd> L_mo;
    L_mo.reserve(L_ao_vectors_.size());
    
    int nbf = C_mo.rows();
    int n_chol = L_ao_vectors_.size();
    
    // Resize vector holder
    L_mo.resize(n_chol);
    
    #pragma omp parallel for schedule(dynamic)
    for(int k = 0; k < n_chol; ++k) {
        // Map Vector -> Matrix AO
        const Eigen::VectorXd& vec = L_ao_vectors_[k];
        Eigen::Map<const Eigen::MatrixXd> L_ao_mat(vec.data(), nbf, nbf);
        
        // Transform: C^T * L_AO * C
        L_mo[k] = C_mo.transpose() * L_ao_mat * C_mo;
    }
    
    return L_mo;
}

// ============================================================================
// INTEGRAL CONSTRUCTION (ACTIVE SPACE)
// ============================================================================

ci::CIIntegrals CholeskyCASSCF::construct_active_integrals(
    const std::vector<Eigen::MatrixXd>& L_mo, 
    const Eigen::MatrixXd& C_mo) const 
{
    ci::CIIntegrals ints;
    int n_active = active_space_.n_active();
    auto active_idx = active_space_.active_indices();
    
    // 1. One-electron integrals (Standard)
    Eigen::MatrixXd h_ao = integrals_->compute_kinetic() + integrals_->compute_nuclear();
    Eigen::MatrixXd h_full_mo = C_mo.transpose() * h_ao * C_mo;
    
    ints.h_alpha = Eigen::MatrixXd(n_active, n_active);
    for(int i=0; i<n_active; ++i)
        for(int j=0; j<n_active; ++j)
            ints.h_alpha(i,j) = h_full_mo(active_idx[i], active_idx[j]);
            
    // Add Core Fock Contribution using Cholesky
    // F_act = h_act + sum_inact (2 J_inact - K_inact)
    int n_inact = active_space_.n_inactive();
    auto inact_idx = active_space_.inactive_indices();
    
    if (n_inact > 0) {
        for(int p=0; p<n_active; ++p) {
            for(int q=0; q<n_active; ++q) {
                double val = 0.0;
                for(int i_idx=0; i_idx<n_inact; ++i_idx) {
                    int i = inact_idx[i_idx];
                    int p_abs = active_idx[p];
                    int q_abs = active_idx[q];
                    
                    // Reconstruct using Cholesky vectors
                    for(const auto& L : L_mo) {
                        double L_ii = L(i, i);
                        double L_pq = L(p_abs, q_abs);
                        double L_ip = L(i, p_abs);
                        double L_iq = L(i, q_abs);
                        
                        // 2 (ii|pq) - (ip|iq)
                        val += 2.0 * L_ii * L_pq - L_ip * L_iq;
                    }
                }
                ints.h_alpha(p,q) += val;
            }
        }
    }
    ints.h_beta = ints.h_alpha; // RHF assumption
    
    // 2. Two-electron integrals (Active Only) from Cholesky
    ints.eri_aaaa = Eigen::Tensor<double, 4>(n_active, n_active, n_active, n_active);
    ints.eri_bbbb = Eigen::Tensor<double, 4>(n_active, n_active, n_active, n_active);
    ints.eri_aabb = Eigen::Tensor<double, 4>(n_active, n_active, n_active, n_active);
    ints.eri_aaaa.setZero();
    ints.eri_bbbb.setZero();
    ints.eri_aabb.setZero();
    
    for(int t=0; t<n_active; ++t) {
        for(int u=0; u<n_active; ++u) {
            for(int v=0; v<n_active; ++v) {
                for(int w=0; w<n_active; ++w) {
                    int T = active_idx[t]; int U = active_idx[u];
                    int V = active_idx[v]; int W = active_idx[w];
                    
                    // Chemist Notation Reconstruction
                    // (tu|vw) = sum_K L_tu^K * L_vw^K
                    double coul = 0.0; // (tu|vw)_chem -> (pr|qs) in slater code context
                    double exch = 0.0; // (tw|vu)_chem -> (ps|qr) in slater code context
                    
                    for(const auto& L : L_mo) {
                        coul += L(T, V) * L(U, W); // (TV|UW) -> (pr|qs) mapping
                        exch += L(T, W) * L(U, V); // (TW|UV) -> (ps|qr) mapping
                    }
                    
                    // Map to Physicist <tu||vw> = (tv|uw) - (tw|uv)
                    // Note: Indices here are tricky.
                    // Slater Condon expects: eri(p,q,r,s) for <pq||rs>
                    // <pq||rs> = <pq|rs> - <pq|sr>
                    // <pq|rs>_phys = (pr|qs)_chem
                    
                    // So we computed:
                    // coul = (TV|UW) = <TU|VW>_phys_dir
                    // exch = (TW|UV) = <TU|WV>_phys_exc
                    
                    ints.eri_aaaa(t, u, v, w) = coul - exch;
                    ints.eri_bbbb(t, u, v, w) = coul - exch;
                    ints.eri_aabb(t, u, v, w) = coul;
                }
            }
        }
    }
    
    ints.e_nuc = mol_.nuclear_repulsion_energy();
    return ints;
}

// ============================================================================
// FOCK BUILD (CHOLESKY OPTIMIZED)
// ============================================================================

Eigen::MatrixXd CholeskyCASSCF::compute_fock_cholesky(
    const Eigen::MatrixXd& opdm,
    const std::vector<Eigen::MatrixXd>& L_mo,
    const Eigen::MatrixXd& C_mo) const
{
    int n_mo = C_mo.cols();
    Eigen::MatrixXd F = C_mo.transpose() * (integrals_->compute_kinetic() + integrals_->compute_nuclear()) * C_mo;
    
    auto inact = active_space_.inactive_indices();
    auto act = active_space_.active_indices();
    int n_act = active_space_.n_active();
    
    for(int p=0; p<n_mo; ++p) {
        for(int q=0; q<n_mo; ++q) {
            
            // Core Contribution (2J - K)
            for(int i : inact) {
                double J = 0.0; double K = 0.0;
                for(const auto& L : L_mo) {
                    J += L(i,i) * L(p,q);     // (ii|pq)
                    K += L(i,p) * L(i,q);     // (ip|iq)
                }
                F(p,q) += 2.0 * J - K;
            }
            
            // Active Contribution
            for(int t_idx=0; t_idx<n_act; ++t_idx) {
                for(int u_idx=0; u_idx<n_act; ++u_idx) {
                    int t = act[t_idx];
                    int u = act[u_idx];
                    double gamma = opdm(t_idx, u_idx);
                    
                    double J = 0.0; double K = 0.0;
                    for(const auto& L : L_mo) {
                        J += L(t,u) * L(p,q); // (tu|pq)
                        K += L(t,p) * L(u,q); // (tp|uq)
                    }
                    F(p,q) += gamma * (J - 0.5 * K);
                }
            }
        }
    }
    return F;
}

// ============================================================================
// ENERGY COMPUTATION
// ============================================================================

double CholeskyCASSCF::compute_total_energy(
    const Eigen::MatrixXd& C_mo, 
    double e_ci,
    const std::vector<Eigen::MatrixXd>& L_mo) const 
{
    double e_core = 0.0;
    int n_inact = active_space_.n_inactive();
    auto inact_idx = active_space_.inactive_indices();
    
    if (n_inact > 0) {
        Eigen::MatrixXd h_mo = C_mo.transpose() * (integrals_->compute_kinetic() + integrals_->compute_nuclear()) * C_mo;
        
        for(int i_idx=0; i_idx<n_inact; ++i_idx) {
            int i = inact_idx[i_idx];
            e_core += 2.0 * h_mo(i,i);
            
            for(int j_idx=0; j_idx<n_inact; ++j_idx) {
                int j = inact_idx[j_idx];
                double J = 0.0; double K = 0.0;
                for(const auto& L : L_mo) {
                    J += L(i,i) * L(j,j);
                    K += L(i,j) * L(i,j);
                }
                e_core += 2.0 * J - K;
            }
        }
    }
    return e_core + e_ci + mol_.nuclear_repulsion_energy();
}

// ============================================================================
// RE-IMPLEMENTED HELPERS (Copied from casscf.cc to avoid linker errors)
// ============================================================================

std::vector<ci::Determinant> CholeskyCASSCF::generate_determinants() const {
    int n_elec = active_space_.n_elec_active();
    int n_orb = active_space_.n_active();
    int n_alpha = (n_elec + 1) / 2;
    int n_beta = n_elec / 2;
    
    std::cout << "Generating determinants for CAS(" << n_elec << "," << n_orb << ")\n";
    
    auto generate_combinations = [](int n, int k) -> std::vector<std::vector<int>> {
        std::vector<std::vector<int>> result;
        if (k == 0) { result.push_back({}); return result; }
        if (k > n) return result;
        std::vector<int> combo(k);
        for (int i=0; i<k; i++) combo[i] = i;
        result.push_back(combo);
        while (true) {
            int i = k - 1;
            while (i >= 0 && combo[i] == n - k + i) i--;
            if (i < 0) break;
            combo[i]++;
            for (int j=i+1; j<k; j++) combo[j] = combo[j-1] + 1;
            result.push_back(combo);
        }
        return result;
    };
    
    auto a_str = generate_combinations(n_orb, n_alpha);
    auto b_str = generate_combinations(n_orb, n_beta);
    std::vector<ci::Determinant> dets;
    for(const auto& a : a_str) 
        for(const auto& b : b_str) 
            dets.emplace_back(a, b);
            
    return dets;
}

std::pair<double, std::vector<double>> CholeskyCASSCF::solve_ci_problem(
    const std::vector<ci::Determinant>& dets, 
    const ci::CIIntegrals& ints) 
{
    int n_det = dets.size();
    if (n_det <= 1000) {
        // Dense
        Eigen::MatrixXd H = ci::build_hamiltonian(dets, ints);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
        
        std::vector<double> c(n_det);
        Eigen::VectorXd ev = es.eigenvectors().col(0);
        for(int i=0; i<n_det; ++i) c[i] = ev(i);
        
        return {es.eigenvalues()(0), c};
    } else {
        // Davidson
        ci::DavidsonOptions opts;
        opts.max_iter = 100;
        ci::DavidsonSolver solver(opts);
        Eigen::VectorXd guess = Eigen::VectorXd::Zero(n_det); guess(0) = 1.0;
        auto res = solver.solve(dets, ints, guess);
        
        std::vector<double> c(n_det);
        for(int i=0; i<n_det; ++i) c[i] = res.eigenvector(i);
        return {res.energy, c};
    }
}

Eigen::MatrixXd CholeskyCASSCF::compute_opdm(
    const std::vector<double>& ci_coeffs,
    const std::vector<ci::Determinant>& determinants) const 
{
    int n_act = active_space_.n_active();
    Eigen::MatrixXd opdm = Eigen::MatrixXd::Zero(n_act, n_act);
    int n_det = determinants.size();
    
    for(int I=0; I<n_det; ++I) {
        for(int J=0; J<n_det; ++J) {
            if(std::abs(ci_coeffs[I]*ci_coeffs[J]) < 1e-12) continue;
            auto exc = determinants[I].excitation_level(determinants[J]);
            if(exc.first + exc.second > 1) continue;
            
            double w = ci_coeffs[I] * ci_coeffs[J];
            for(int p=0; p<n_act; ++p) {
                for(int q=0; q<n_act; ++q) {
                    double val = 0.0;
                    if(I==J && p==q) {
                        if(determinants[J].is_occupied(p, true)) val += 1.0;
                        if(determinants[J].is_occupied(p, false)) val += 1.0;
                    }
                    // Alpha transition
                    if(determinants[J].is_occupied(q, true)) {
                        try {
                            if(determinants[J].single_excite(q,p,true) == determinants[I])
                                val += determinants[J].phase(q,p,true);
                        } catch(...) {}
                    }
                    // Beta transition
                    if(determinants[J].is_occupied(q, false)) {
                        try {
                            if(determinants[J].single_excite(q,p,false) == determinants[I])
                                val += determinants[J].phase(q,p,false);
                        } catch(...) {}
                    }
                    opdm(p,q) += w * val;
                }
            }
        }
    }
    return opdm;
}

Eigen::VectorXd CholeskyCASSCF::compute_orbital_gradient(
    const Eigen::MatrixXd& fock,
    const Eigen::MatrixXd& C_mo) const
{
    auto inact = active_space_.inactive_indices();
    auto act = active_space_.active_indices();
    auto vir = active_space_.virtual_indices();
    
    std::vector<double> g;
    // Inactive-Active
    for(int i : inact) for(int t : act) g.push_back(2.0 * (fock(i,t) - fock(t,i)));
    // Active-Virtual
    for(int t : act) for(int a : vir) g.push_back(2.0 * (fock(t,a) - fock(a,t)));
    
    Eigen::VectorXd grad(g.size());
    for(size_t i=0; i<g.size(); ++i) grad(i) = g[i];
    return grad;
}

Eigen::VectorXd CholeskyCASSCF::compute_orbital_step_newton(
    const Eigen::VectorXd& gradient,
    const Eigen::VectorXd& orbital_energies) const
{
    auto inact = active_space_.inactive_indices();
    auto act = active_space_.active_indices();
    auto vir = active_space_.virtual_indices();
    
    Eigen::VectorXd kappa(gradient.size());
    int idx = 0;
    
    for(int i : inact) for(int t : act) {
        double denom = orbital_energies(i) - orbital_energies(t);
        if(std::abs(denom) < 1e-6) denom = (denom > 0) ? 1e-6 : -1e-6;
        kappa(idx) = -gradient(idx) / denom;
        idx++;
    }
    for(int t : act) for(int a : vir) {
        double denom = orbital_energies(t) - orbital_energies(a);
        if(std::abs(denom) < 1e-6) denom = (denom > 0) ? 1e-6 : -1e-6;
        kappa(idx) = -gradient(idx) / denom;
        idx++;
    }
    return kappa;
}

Eigen::MatrixXd CholeskyCASSCF::apply_orbital_rotation(
    const Eigen::MatrixXd& C_mo,
    const Eigen::VectorXd& kappa,
    double damping) const
{
    if(kappa.norm() < 1e-10) return C_mo;
    int n = C_mo.cols();
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n, n);
    
    auto inact = active_space_.inactive_indices();
    auto act = active_space_.active_indices();
    auto vir = active_space_.virtual_indices();
    
    Eigen::VectorXd k = damping * kappa;
    int idx = 0;
    
    for(int i : inact) for(int t : act) { K(i,t) = k(idx); K(t,i) = -k(idx); idx++; }
    for(int t : act) for(int a : vir) { K(t,a) = k(idx); K(a,t) = -k(idx); idx++; }
    
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n,n);
    Eigen::MatrixXd U = (I - 0.5*K).inverse() * (I + 0.5*K);
    return C_mo * U;
}

bool CholeskyCASSCF::check_convergence(double delta_e, double grad_norm) const {
    return std::abs(delta_e) < e_thresh_ && grad_norm < grad_thresh_;
}

CASResult CholeskyCASSCF::compute(const SCFResult& initial_guess) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Cholesky-CASSCF Calculation Started\n";
    std::cout << std::string(70, '=') << "\n";
    
    ensure_cholesky_vectors();
    
    Eigen::MatrixXd C_mo = initial_guess.C_alpha;
    double e_prev = 0.0;
    double damping = 0.5;
    
    CASResult result;
    result.e_nuclear = mol_.nuclear_repulsion_energy();
    result.converged = false;
    result.active_space = active_space_;
    
    auto determinants = generate_determinants();
    result.n_determinants = static_cast<int>(determinants.size());
    
    for (int iter = 0; iter < max_iter_; iter++) {
        std::cout << "\n--- Iteration " << (iter + 1) << " ---\n";
        
        auto L_mo = transform_cholesky_to_mo(C_mo);
        auto ints = construct_active_integrals(L_mo, C_mo);
        auto ci_res = solve_ci_problem(determinants, ints);
        double e_ci = ci_res.first;
        
        auto opdm = compute_opdm(ci_res.second, determinants);
        double e_casscf = compute_total_energy(C_mo, e_ci, L_mo);
        double delta_e = e_casscf - e_prev;
        
        auto fock = compute_fock_cholesky(opdm, L_mo, C_mo);
        auto gradient = compute_orbital_gradient(fock, C_mo);
        double grad_norm = gradient.norm();
        
        std::cout << "E(CASSCF) = " << std::fixed << std::setprecision(10) << e_casscf << " Ha\n";
        std::cout << "Delta E   = " << std::scientific << delta_e << "\n";
        std::cout << "|Grad|    = " << grad_norm << "\n";
        
        result.energy_history.push_back(e_casscf);
        
        if (iter > 0 && check_convergence(delta_e, grad_norm)) {
            result.converged = true;
            result.e_casscf = e_casscf;
            result.C_mo = C_mo;
            result.ci_coeffs = ci_res.second;
            result.determinants = determinants;
            result.n_iterations = iter + 1;
            result.orbital_energies = fock.diagonal();
            return result;
        }
        
        if (iter < max_iter_ - 1) {
            Eigen::VectorXd kappa;
            if (orbital_opt_ == "newton") {
                kappa = compute_orbital_step_newton(gradient, fock.diagonal());
            } else {
                kappa = -gradient;
            }
            
            if (iter > 0) {
                if (delta_e < 0.0) damping = std::min(1.0, damping * 1.2);
                else damping = std::max(0.1, damping * 0.5);
            }
            
            C_mo = apply_orbital_rotation(C_mo, kappa, damping);
        }
        e_prev = e_casscf;
    }
    
    result.e_casscf = e_prev;
    result.C_mo = C_mo;
    result.n_iterations = max_iter_;
    return result;
}

} // namespace mcscf
} // namespace mshqc