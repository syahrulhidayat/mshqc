
/**
 * @file uhf.cc
 * @brief Unrestricted Hartree-Fock (UHF)
 * @details FIXED: Robust Orthogonalization & Electron Count Validation
 * for Large/Diffuse Basis Sets (cc-pV5Z, etc.)
 */




#include "mshqc/scf.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

namespace mshqc {

UHF::UHF(const Molecule& mol, 
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals,
         int n_alpha,
         int n_beta,
         const SCFConfig& config)
    : mol_(mol), basis_(basis), integrals_(integrals), config_(config),
      n_alpha_(n_alpha), n_beta_(n_beta) {
    
    nbasis_ = basis.n_basis_functions();
    
    if (n_alpha_ < n_beta_) {
        throw std::runtime_error("UHF: n_alpha must be >= n_beta");
    }
}

void UHF::init_integrals() {
    // 1. Compute Raw Integrals
    S_ = integrals_->compute_overlap();
    Eigen::MatrixXd T = integrals_->compute_kinetic();
    Eigen::MatrixXd V = integrals_->compute_nuclear();
    H_ = T + V;

    // 2. DIAGNOSA LINEAR DEPENDENCE
    // Untuk basis besar (5Z/6Z) atau diffuse (aug-), matriks S sering singular.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S_);
    Eigen::VectorXd s_vals = es.eigenvalues();
    Eigen::MatrixXd s_vecs = es.eigenvectors();
    
    // Threshold ketat untuk basis besar (1e-5 s/d 1e-6)
    double threshold = 1.0e-5;
    
    // Hitung statistik kondisi matriks
    double smallest_eig = s_vals(0);
    double largest_eig = s_vals(nbasis_ - 1);
    double condition_number = largest_eig / smallest_eig;

    if (config_.print_level > 0) {
        std::cout << "  Overlap Matrix Analysis:\n";
        std::cout << "    Smallest Eigenvalue: " << smallest_eig << "\n";
        std::cout << "    Condition Number:    " << std::scientific << condition_number << "\n" << std::defaultfloat;
    }

    // 3. CANONICAL ORTHOGONALIZATION
    // Membangun matriks X = U * s^-1/2 * U^T hanya untuk eigenvalue > threshold
    int n_kept = 0;
    X_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    
    for (int i = 0; i < nbasis_; i++) {
        if (s_vals(i) >= threshold) {
            n_kept++;
            double inv_sqrt_s = 1.0 / std::sqrt(s_vals(i));
            X_ += s_vecs.col(i) * s_vecs.col(i).transpose() * inv_sqrt_s;
        }
    }

    // Report jika ada vektor yang dibuang
    if (n_kept < nbasis_) {
        std::cout << "  [WARNING] Linear dependence detected!\n";
        std::cout << "            Basis functions: " << nbasis_ << "\n";
        std::cout << "            Removed vectors: " << (nbasis_ - n_kept) << " (Threshold: " << threshold << ")\n";
    }
}

void UHF::initial_guess() {
    // Core Hamiltonian guess
    // PROBLEM: For atoms like Li, 2s and 2p are degenerate in H_core.
    // Solver might pick 2p (excited state) instead of 2s (ground state).
    // FIX: Apply small bias to lower indices (s-orbitals) to break degeneracy.
    
    Eigen::MatrixXd H_bias = H_;
    
    // Bias: Turunkan energi diagonal fungsi basis awal (biasanya s-type)
    // H_mm -= 0.5 * exp(-mu)
    // Ini memastikan orbital s dipilih lebih dulu daripada p/d saat degenerate.
    for (size_t i = 0; i < nbasis_; i++) {
        H_bias(i, i) -= 0.1 * std::exp(-0.1 * i);
    }

    // Diagonalize Biased H for Guess
    solve_fock(H_bias, C_a_, eps_a_);
    solve_fock(H_bias, C_b_, eps_b_);
    
    // Build initial densities from this biased guess
    P_a_ = build_density(C_a_, n_alpha_);
    P_b_ = build_density(C_b_, n_beta_);
    
    energy_ = 0.0;
    energy_old_ = 0.0;
    
    if (config_.print_level > 0) {
        std::cout << "  Initial Guess: Core Hamiltonian + S-bias applied.\n";
    }
}

void UHF::build_fock() {
    F_a_ = H_;
    F_b_ = H_;
    
    auto eri = integrals_->compute_eri();
    Eigen::MatrixXd P_tot = P_a_ + P_b_;
    
    // O(N^4) Fock Build
    // Loop explicit untuk akurasi maksimal
    for (size_t mu = 0; mu < nbasis_; mu++) {
        for (size_t nu = 0; nu < nbasis_; nu++) {
            double g_a = 0.0, g_b = 0.0;
            
            for (size_t lam = 0; lam < nbasis_; lam++) {
                for (size_t sig = 0; sig < nbasis_; sig++) {
                    // Integral ERI (mu, nu, lambda, sigma)
                    double val = eri(mu, nu, lam, sig);
                    
                    // Coulomb (Direct): J = (mu nu | lam sig) * P_tot(lam, sig)
                    double j_term = val * P_tot(lam, sig);
                    g_a += j_term;
                    g_b += j_term;
                    
                    // Exchange (K): K = (mu lam | nu sig) * P(lam, sig)
                    // Perhatikan indeks val harus ditukar untuk exchange jika pakai tensor simetris
                    double val_ex = eri(mu, lam, nu, sig);
                    g_a -= val_ex * P_a_(lam, sig);
                    g_b -= val_ex * P_b_(lam, sig);
                }
            }
            F_a_(mu, nu) += g_a;
            F_b_(mu, nu) += g_b;
        }
    }
}

Eigen::MatrixXd UHF::build_density(const Eigen::MatrixXd& C, int n) {
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    // Density = C_occ * C_occ^T
    for (int i = 0; i < n; i++) {
        P += C.col(i) * C.col(i).transpose();
    }
    return P;
}

double UHF::compute_energy() {
    // E = 0.5 * Tr[ P_a(H + F_a) + P_b(H + F_b) ]
    double e_a = 0.5 * (P_a_ * (H_ + F_a_)).trace();
    double e_b = 0.5 * (P_b_ * (H_ + F_b_)).trace();
    return e_a + e_b;
}

void UHF::solve_fock(const Eigen::MatrixXd& F,
                     Eigen::MatrixXd& C,
                     Eigen::VectorXd& eps) {
    // F' = X^T * F * X
    Eigen::MatrixXd Fp = X_.transpose() * F * X_;
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Fp);
    eps = es.eigenvalues();
    Eigen::MatrixXd Cp = es.eigenvectors();
    
    // C = X * C'
    C = X_ * Cp;
}

bool UHF::check_convergence() {
    double de = std::abs(energy_ - energy_old_);
    return de < config_.energy_threshold;
}

void UHF::print_iter(int iter, double de, double dp) {
    if (config_.print_level < 1) return;
    std::cout << std::setw(4) << iter
              << std::setw(18) << std::fixed << std::setprecision(10) << energy_
              << std::setw(14) << std::scientific << std::setprecision(4) << de
              << std::setw(14) << dp << "\n";
}

void UHF::print_final(const SCFResult& r) {
    if (config_.print_level < 1) return;
    
    std::cout << "\n=== UHF Results ===\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Nuclear repulsion: " << r.energy_nuclear << " Ha\n";
    std::cout << "Electronic energy: " << r.energy_electronic << " Ha\n";
    std::cout << "Total energy:      " << r.energy_total << " Ha\n";
    std::cout << "Iterations:        " << r.iterations << "\n";
    
    double s2 = compute_s_squared(r);
    double s2_exact = 0.25 * (n_alpha_ - n_beta_) * (n_alpha_ - n_beta_ + 2);
    std::cout << "<S²>:              " << s2 << " (exact: " << s2_exact << ")\n";
    std::cout << "Spin contam:       " << (s2 - s2_exact) << "\n";
    
    // VALIDASI JUMLAH ELEKTRON (PENTING untuk Debugging Basis Set)
    double n_a_calc = (P_a_ * S_).trace();
    double n_b_calc = (P_b_ * S_).trace();
    std::cout << "Electron count:    α=" << n_a_calc << " (target " << n_alpha_ << "), "
              << "β=" << n_b_calc << " (target " << n_beta_ << ")\n";
              
    if (std::abs(n_a_calc - n_alpha_) > 1e-4 || std::abs(n_b_calc - n_beta_) > 1e-4) {
        std::cout << "Warning: Electron count mismatch! Basis set or Integral problem likely.\n";
    }
}

double UHF::compute_s_squared(const SCFResult& result) {
    int na = result.n_occ_alpha;
    int nb = result.n_occ_beta;
    double s_exact = 0.25 * (na - nb) * (na - nb + 2);
    
    // Overlap dalam MO basis: S_ij = C_a^T * S * C_b
    // Ambil blok occupied
    Eigen::MatrixXd C_a_occ = result.C_alpha.leftCols(na);
    Eigen::MatrixXd C_b_occ = result.C_beta.leftCols(nb);
    Eigen::MatrixXd S_ab = C_a_occ.transpose() * S_ * C_b_occ;
    
    double overlap_sum = S_ab.array().abs2().sum();
    return s_exact + nb - overlap_sum;
}

SCFResult UHF::compute() {
    std::cout << "\n====================================\n";
    std::cout << "  Unrestricted Hartree-Fock (UHF)\n";
    std::cout << "====================================\n";
    std::cout << "Basis functions: " << nbasis_ << "\n";
    std::cout << "Alpha electrons: " << n_alpha_ << "\n";
    std::cout << "Beta electrons:  " << n_beta_ << "\n";
    
    // Inisialisasi Integrals & Orthogonalizer
    init_integrals();
    
    // Tebakan Awal
    initial_guess();
    
    // DIIS Setup
    DIIS diis_a(config_.diis_max_vectors);
    DIIS diis_b(config_.diis_max_vectors);
    
    if (config_.print_level > 0) {
        std::cout << "====================================\n";
        std::cout << "\nSCF Iterations:\n";
        std::cout << std::setw(4) << "Iter"
                  << std::setw(18) << "Energy (Ha)"
                  << std::setw(14) << "ΔE"
                  << std::setw(14) << "||ΔP||\n";
        std::cout << std::string(50, '-') << "\n";
    }
    
    bool converged = false;
    int iter;
    
    for (iter = 1; iter <= config_.max_iterations; iter++) {
        energy_old_ = energy_;
        
        // 1. Build Fock
        build_fock();
        
        // 2. DIIS Extrapolation
        // Error vector: e = F*P*S - S*P*F (Commutator)
        Eigen::MatrixXd e_a = F_a_ * P_a_ * S_ - S_ * P_a_ * F_a_;
        Eigen::MatrixXd e_b = F_b_ * P_b_ * S_ - S_ * P_b_ * F_b_;
        double err_norm = std::max(e_a.norm(), e_b.norm());
        
        if (err_norm < config_.diis_threshold) {
            diis_a.add_iteration(F_a_, e_a);
            diis_b.add_iteration(F_b_, e_b);
            
            if (diis_a.can_extrapolate()) {
                F_a_ = diis_a.extrapolate();
                F_b_ = diis_b.extrapolate();
            }
        }
        
        // 3. Diagonalize
        solve_fock(F_a_, C_a_, eps_a_);
        solve_fock(F_b_, C_b_, eps_b_);
        
        // 4. Update Density
        Eigen::MatrixXd P_a_old = P_a_;
        Eigen::MatrixXd P_b_old = P_b_;
        P_a_ = build_density(C_a_, n_alpha_);
        P_b_ = build_density(C_b_, n_beta_);
        
        // 5. Compute Energy
        energy_ = compute_energy();
        
        // Check Convergence
        double de = std::abs(energy_ - energy_old_);
        double dp = std::max((P_a_ - P_a_old).norm(), (P_b_ - P_b_old).norm());
        
        print_iter(iter, de, dp);
        
        if (check_convergence() && dp < config_.density_threshold) {
            converged = true;
            break;
        }
    }
    
    SCFResult result;
    result.energy_nuclear = mol_.nuclear_repulsion_energy();
    result.energy_electronic = energy_;
    result.energy_total = energy_ + result.energy_nuclear;
    result.orbital_energies_alpha = eps_a_;
    result.orbital_energies_beta = eps_b_;
    result.C_alpha = C_a_;
    result.C_beta = C_b_;
    result.P_alpha = P_a_;
    result.P_beta = P_b_;
    result.F_alpha = F_a_;
    result.F_beta = F_b_;
    result.iterations = iter;
    result.converged = converged;
    result.n_occ_alpha = n_alpha_;
    result.n_occ_beta = n_beta_;
    
    print_final(result);
    
    return result;
}

} // namespace mshqc