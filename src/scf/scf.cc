// Copyright 2026 Muhamad Syahrul Hidayat
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file src/scf/scf.cc
 * @brief Unified SCF Engine (BaseSCF, RHF, UHF, ROHF) in a single file.
 * @details Fully embeds Cholesky L_mat_ and SIMD In-Core for all methods.
 */

#include "mshqc/scf/scf.h"
#include "mshqc/scf/diis.h"
#include "mshqc/scf/sad.h"
#include "mshqc/symmetry/salc_builder.h"
#include "mshqc/integrals/df_eri.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <omp.h>
#include "mshqc/utils/hdf5_io.h" 
#include <tblis/tblis.h>         
#include <memory>                
#include <vector>
#ifdef I
#undef I
#endif

namespace mshqc {




BaseSCF::BaseSCF(const Molecule& mol, const BasisSet& basis,
                 std::shared_ptr<IntegralEngine> integrals,
                 std::shared_ptr<PointGroup> pg,
                 std::shared_ptr<PetiteList> pl,
                 int n_alpha, int n_beta,
                 const SCFConfig& config)
    : mol_(mol), basis_(basis), integrals_(integrals),
      pg_(pg), pl_(pl), config_(config),
      nbasis_(basis.n_basis_functions()),
      n_alpha_(n_alpha), n_beta_(n_beta) 
{
    if (pg_ && pl_) symmetrizer_ = std::make_unique<BasisSymmetrizer>(basis_, *pg_, *pl_);

    int nshells = basis_.n_shells();
    shell_starts_.resize(nshells);
    shell_sizes_.resize(nshells);
    int offset = 0;
    for(int i = 0; i < nshells; ++i) {
        shell_starts_[i] = offset;
        shell_sizes_[i] = basis_.shell(i).n_functions();
        offset += shell_sizes_[i];
    }

    if (config_.use_df) {
        BasisSet aux_basis(config_.aux_basis_name, mol_); 
        BasisSet combined_basis = basis_;
        combined_basis.append(aux_basis);
        auto df_integrals = std::make_shared<IntegralEngine>(mol_, combined_basis);
        auto df_engine = std::make_shared<integrals::DensityFittingERI>(basis_, aux_basis, df_integrals, config_.df_threshold);
        df_engine->compute();
        
        
        
        this->L_mat_.resize(0, aux_basis.n_basis_functions()); 
    }
}

void BaseSCF::init_integrals() {
    auto t_start = std::chrono::high_resolution_clock::now();

    S_ = integrals_->compute_overlap();
    if (S_.rows() == 0) throw std::runtime_error("CRITICAL: Empty Overlap Matrix.");
    H_ = integrals_->compute_kinetic() + integrals_->compute_nuclear();

    if (config_.print_level > 0) std::cout << "  [SCF] Computing Orthogonalizer (SALC)... ";
    if (symmetrizer_) {
        SalcBuilder salc(symmetrizer_.get());
        auto salc_data = salc.build_salc(S_);
        X_ = salc_data.first; salc_irreps_ = salc_data.second; 
    } else {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S_);
        X_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
        for (int i = 0; i < nbasis_; i++) {
            if (es.eigenvalues()(i) >= 1e-6) {
                double val = 1.0 / std::sqrt(es.eigenvalues()(i));
                X_ += val * es.eigenvectors().col(i) * es.eigenvectors().col(i).transpose();
            }
        }
        salc_irreps_.assign(nbasis_, 0); 
    }
    if (config_.print_level > 0) std::cout << "Done.\n";

    schwarz_ = precompute_shell_schwarz();

    if (config_.eri_method == "cholesky" && !config_.use_df) init_integrals_cholesky();                   
    else if (config_.scf_type == "incore") init_integrals_incore();

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t_end - t_start;
    if (config_.print_level > 0) {
        std::string mode_str = (config_.scf_type == "incore") ? "In-Core" : "Direct";
        std::cout << "  [SCF] Initialization Done. Mode: " << mode_str << " (" 
                  << std::fixed << std::setprecision(3) << elapsed.count() << "s)\n";
    }
}

Eigen::MatrixXd BaseSCF::precompute_shell_schwarz() {
    int nshells = basis_.n_shells();
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(nshells, nshells);
    
    for (int M = 0; M < nshells; ++M) {
        for (int N = 0; N <= M; ++N) {
            auto shell_ints = integrals_->compute_shell_block(M, N, M, N);
            if (shell_ints.empty()) continue;
            double max_val = 0.0;
            for (double val : shell_ints) if (std::abs(val) > max_val) max_val = std::abs(val);
            Q(M, N) = std::sqrt(max_val); Q(N, M) = Q(M, N);
        }
    }
    return Q;
}

void BaseSCF::init_integrals_incore() {
    if (config_.print_level > 0) std::cout << "  [SCF] Building Optimized PK-InCore Integrals (4D Symmetry)... " << std::flush;

    J_val_.clear(); J_ind_.clear(); K_ind_.clear();
    double shell_cutoff = 1e-12;
    const double sparse_threshold = 1e-12;

    if (pl_ && pl_->get_unique_quartets().empty()) pl_->build();

    size_t est_ints = 0;
    if (pl_) {
        for (const auto& [M, N, P, Q, w] : pl_->get_unique_quartets()) {
            if (schwarz_(M, N) * schwarz_(P, Q) < shell_cutoff) continue;
            est_ints += shell_sizes_[M] * shell_sizes_[N] * shell_sizes_[P] * shell_sizes_[Q];
        }
    } else {
        int nshells = basis_.n_shells();
        for (int M = 0; M < nshells; ++M) {
            for (int N = 0; N <= M; ++N) {
                for (int P = 0; P <= M; ++P) {
                    int Q_max = (M == P) ? N : P;
                    for (int Q = 0; Q <= Q_max; ++Q) {
                        if (schwarz_(M, N) * schwarz_(P, Q) < shell_cutoff) continue;
                        est_ints += shell_sizes_[M] * shell_sizes_[N] * shell_sizes_[P] * shell_sizes_[Q];
                    }
                }
            }
        }
    }

    J_val_.assign(est_ints, 0.0);
    J_ind_.assign(est_ints, 0);
    K_ind_.assign(est_ints, 0);
    size_t idx = 0;

    auto process_quartet = [&](int M, int N, int P, int Q, double weight) {
        if (schwarz_(M, N) * schwarz_(P, Q) < shell_cutoff) return;
        auto buf = integrals_->compute_shell_block(M, N, P, Q);
        
        int dimM = shell_sizes_[M]; int dimN = shell_sizes_[N];
        int dimP = shell_sizes_[P]; int dimQ = shell_sizes_[Q];
        int stM = shell_starts_[M]; int stN = shell_starts_[N];
        int stP = shell_starts_[P]; int stQ = shell_starts_[Q];

        // Faktor simetri dihitung per-shell, seperti di fock_builder.cc
        double fac = weight;
        if (M == N) fac *= 0.5;
        if (P == Q) fac *= 0.5;
        if (M == P && N == Q) fac *= 0.5;

        for (int m = 0; m < dimM; ++m) {
            for (int n = 0; n < dimN; ++n) {
                int mu = stM + m; int nu = stN + n;
                int packed_mn = (mu << 16) | nu;
                
                for (int p = 0; p < dimP; ++p) {
                    for (int q = 0; q < dimQ; ++q) {
                        int lam = stP + p; int sig = stQ + q;
                        int packed_ls = (lam << 16) | sig;
                        
                        double val = buf[m + dimM * (n + dimN * (p + dimP * q))] * fac;
                        if (std::abs(val) > sparse_threshold) {
                            J_val_[idx] = val;
                            J_ind_[idx] = packed_mn;
                            K_ind_[idx] = packed_ls;
                            idx++;
                        }
                    }
                }
            }
        }
    };

    if (pl_) {
        for (const auto& [M, N, P, Q, w] : pl_->get_unique_quartets()) process_quartet(M, N, P, Q, w);
    } else {
        int nshells = basis_.n_shells();
        for (int M = 0; M < nshells; ++M) {
            for (int N = 0; N <= M; ++N) {
                for (int P = 0; P <= M; ++P) {
                    int Q_max = (M == P) ? N : P;
                    for (int Q = 0; Q <= Q_max; ++Q) process_quartet(M, N, P, Q, 1.0);
                }
            }
        }
    }

    J_val_.resize(idx); J_ind_.resize(idx); K_ind_.resize(idx);
    K_val_.clear(); J_ptr_.clear(); K_ptr_.clear(); row_map_.clear(); 
    if (config_.print_level > 0) std::cout << "Done. (Stored PK Integrals: " << idx << ")\n";
}
void BaseSCF::init_integrals_cholesky() {
    if (config_.print_level > 0) std::cout << "  [SCF] Decomposing Integrals (Cholesky)... " << std::flush;
    internal_cholesky_ = std::make_unique<integrals::CholeskyERI>(basis_, integrals_);
    internal_cholesky_->set_threshold(config_.cholesky_threshold);
    internal_cholesky_->compute(); 
    
    
    L_mat_ = internal_cholesky_->get_L_mat();
    
    if (config_.print_level > 0) std::cout << " Done. Rank: " << L_vecs_.size() << "\n";
}

Eigen::VectorXd BaseSCF::smear_electrons(const Eigen::VectorXd& eps, int target_electrons) {
    Eigen::VectorXd occ = Eigen::VectorXd::Zero(nbasis_);
    int electrons_left = target_electrons; double degen_tol = 1e-4; int i = 0;
    while (i < nbasis_ && electrons_left > 0) {
        int degen_count = 1; double current_e = eps(i);
        while (i + degen_count < nbasis_ && std::abs(eps(i + degen_count) - current_e) < degen_tol) degen_count++;
        int electrons_to_fill = std::min(electrons_left, degen_count); 
        double fraction = (double)electrons_to_fill / (double)degen_count;
        for (int k = 0; k < degen_count; ++k) occ(i + k) = fraction;
        electrons_left -= electrons_to_fill; i += degen_count;
    }
    return occ;
}

void BaseSCF::solve_fock(const Eigen::MatrixXd& F, Eigen::MatrixXd& C, Eigen::VectorXd& eps) {
    Eigen::MatrixXd Fp = X_.transpose() * F * X_;
    if (symmetrizer_ && !salc_irreps_.empty()) {
        Eigen::MatrixXd C_salc = Eigen::MatrixXd::Zero(nbasis_, nbasis_); eps.resize(nbasis_);
        int start_idx = 0;
        while (start_idx < nbasis_) {
            int current_irrep = salc_irreps_[start_idx]; int size = 0;
            while (start_idx + size < nbasis_ && salc_irreps_[start_idx + size] == current_irrep) size++;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Fp.block(start_idx, start_idx, size, size));
            C_salc.block(start_idx, start_idx, size, size) = es.eigenvectors(); eps.segment(start_idx, size) = es.eigenvalues();
            start_idx += size;
        }
        std::vector<std::pair<double, int>> sort_data(nbasis_);
        for (int i = 0; i < nbasis_; ++i) sort_data[i] = {eps(i), i};
        std::sort(sort_data.begin(), sort_data.end());
        Eigen::MatrixXd C_sorted(nbasis_, nbasis_); Eigen::VectorXd eps_sorted(nbasis_);
        for (int i = 0; i < nbasis_; ++i) { C_sorted.col(i) = C_salc.col(sort_data[i].second); eps_sorted(i) = sort_data[i].first; }
        C.noalias() = X_ * C_sorted; eps = eps_sorted;
    } else {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Fp); eps = es.eigenvalues(); C.noalias() = X_ * es.eigenvectors();
    }
}

void BaseSCF::print_final(const SCFResult& r) {
    if (config_.print_level < 1) return;
    std::cout << "\n=== SCF Final Results ===\n";
    std::cout << "  Total Energy      : " << std::fixed << std::setprecision(10) << r.energy_total << " Ha\n";
    std::cout << "  Iterations        : " << r.iterations << "\n";
    std::cout << "  Converged         : " << (r.converged ? "Yes" : "No") << "\n";
    std::cout << "======================================\n";
}

SCFResult BaseSCF::compute() {
    init_integrals(); initial_guess(); 
    DIIS diis_a(config_.diis_max_vectors); DIIS diis_b(config_.diis_max_vectors);
    
    if (config_.scf_type == "direct") {
        fock_engine_ = std::make_unique<FockBuilder>(integrals_, basis_, H_, schwarz_);
        if (pl_) fock_engine_->set_petite_list(pl_.get());
    }
    
    bool converged = false; double thresh = 1e-5; bool is_uhf = (n_alpha_ != n_beta_);
    if (config_.print_level > 0) printf("\n Iter       Energy (Ha)        Delta E    Delta P\n");
    
    for (iter_scf_ = 1; iter_scf_ <= config_.max_iterations; iter_scf_++) {
        energy_old_ = energy_; Eigen::MatrixXd P_tot_snapshot = P_alpha_; if (is_uhf) P_tot_snapshot += P_beta_;
        if (iter_scf_ > 2) { double de = std::abs(energy_ - energy_old_); if (de < 1e-3) thresh = 1e-9; if (de < 1e-6) thresh = 1e-13; }
        if (symmetrizer_) { symmetrizer_->symmetrize(P_alpha_); if (is_uhf) symmetrizer_->symmetrize(P_beta_); }

        build_fock_matrix(); 

        if (symmetrizer_) { symmetrizer_->symmetrize(F_alpha_); if (is_uhf) symmetrizer_->symmetrize(F_beta_); }
        energy_ = compute_energy(); 
        
        Eigen::MatrixXd F_eff_a, F_eff_b;
        Eigen::MatrixXd FPS_a = F_alpha_ * P_alpha_ * S_; Eigen::MatrixXd err_a = FPS_a - FPS_a.transpose(); diis_a.add_iteration(F_alpha_, err_a, P_alpha_);
        F_eff_a = (iter_scf_ <= 1) ? (0.7 * F_alpha_ + 0.3 * H_) : diis_a.extrapolate();

        if (is_uhf) {
            Eigen::MatrixXd FPS_b = F_beta_ * P_beta_ * S_; Eigen::MatrixXd err_b = FPS_b - FPS_b.transpose(); diis_b.add_iteration(F_beta_, err_b, P_beta_);
            F_eff_b = (iter_scf_ <= 1) ? (0.7 * F_beta_ + 0.3 * H_) : diis_b.extrapolate();
        }

        solve_fock(F_eff_a, C_alpha_, eps_alpha_); occ_numbers_alpha_ = smear_electrons(eps_alpha_, n_alpha_);
        if (is_uhf) { solve_fock(F_eff_b, C_beta_, eps_beta_); occ_numbers_beta_ = smear_electrons(eps_beta_, n_beta_); } 
        else { C_beta_ = C_alpha_; eps_beta_ = eps_alpha_; occ_numbers_beta_ = occ_numbers_alpha_; }

        update_densities(); 
        
        double dE = std::abs(energy_ - energy_old_); Eigen::MatrixXd P_tot_new = P_alpha_; if (is_uhf) P_tot_new += P_beta_;
        double dP = (P_tot_new - P_tot_snapshot).norm() / nbasis_;
        
        if (config_.print_level > 0) printf(" %4d %18.10f %10.2e %10.2e\n", iter_scf_, energy() , dE, dP);
        if (dE < config_.energy_threshold && dP < config_.density_threshold) { converged = true; break; }
    }
    
    std::vector<int> irreps_a(nbasis_, 0), irreps_b(nbasis_, 0);
    if (symmetrizer_) {
        if (config_.print_level > 0) std::cout << "\n  [Symmetry] Sorting Orbitals by Irrep...\n";
        irreps_a = symmetrizer_->assign_mo_irreps(C_alpha_);
        irreps_b = is_uhf ? symmetrizer_->assign_mo_irreps(C_beta_) : irreps_a;
    }

    if (config_.use_df && L_mat_.rows() == 0) {
        if (config_.print_level > 0) std::cout << "\n  [SCF] Reloading df_tensor.h5 to RAM for MP2 compatibility...\n";
        int n_aux = L_mat_.cols();
        L_mat_.resize(nbasis_ * nbasis_, n_aux);
        utils::HDF5TensorIO io("df_tensor.h5", utils::HDF5TensorIO::Mode::READ_ONLY);
        io.read_slice_4d("df_tensor", {0, 0, 0, 0}, {(long)n_aux, (long)nbasis_, (long)nbasis_, 1}, L_mat_.data());
    }

   
    
    SCFResult r; r.energy_total = energy(); r.iterations = iter_scf_; r.converged = converged;
    r.C_alpha = C_alpha_; r.C_beta = C_beta_; r.P_alpha = P_alpha_; r.P_beta = P_beta_;
    r.F_alpha = F_alpha_; r.F_beta = F_beta_; r.orbital_energies_alpha = eps_alpha_; r.orbital_energies_beta = eps_beta_;
    r.irreps_alpha = irreps_a; r.irreps_beta = irreps_b; r.n_occ_alpha = n_alpha_; r.n_occ_beta = n_beta_;
    r.L_mat = L_mat_;
    print_final(r); return r;

}




RHF::RHF(const Molecule& mol, const BasisSet& basis, std::shared_ptr<IntegralEngine> integrals, std::shared_ptr<PointGroup> pg, std::shared_ptr<PetiteList> pl, const SCFConfig& config)
    : BaseSCF(mol, basis, integrals, pg, pl, mol.n_electrons() / 2, mol.n_electrons() / 2, config) {
    if (mol.n_electrons() % 2 != 0) throw std::runtime_error("CRITICAL: RHF requires an even number of electrons.");
    G_J_accum_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_); 
    G_accum_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_); 
    P_old_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
}

void RHF::initial_guess() {
    if (config_.print_level > 0) std::cout << "  [RHF] Generating Initial Guess... ";
    Eigen::MatrixXd P_sad = SADGuess::build(mol_, basis_);
    if (P_sad.rows() > 0 && P_sad.norm() > 1e-6) {
        if (config_.print_level > 0) std::cout << "Using SAD.\n";
        P_alpha_ = 0.5 * P_sad; 
        P_beta_ = P_alpha_;
        
        
        
        
        solve_fock(H_, C_alpha_, eps_alpha_);
        C_beta_ = C_alpha_; 
        eps_beta_ = eps_alpha_;
        occ_numbers_alpha_ = smear_electrons(eps_alpha_, n_alpha_);
        occ_numbers_beta_  = occ_numbers_alpha_;
        

        energy_ = 0.0; return; 
    }
    if (config_.print_level > 0) std::cout << "Using GWH.\n";
    Eigen::MatrixXd F_gwh = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    for (int i = 0; i < nbasis_; ++i) for (int j = 0; j < nbasis_; ++j) F_gwh(i, j) = (i == j) ? H_(i, j) : 0.5 * 1.75 * S_(i, j) * (H_(i, i) + H_(j, j));
    solve_fock(F_gwh, C_alpha_, eps_alpha_); 
    occ_numbers_alpha_ = smear_electrons(eps_alpha_, n_alpha_); 
    update_densities(); 
    energy_ = 0.0;
}

void RHF::update_densities() {
    P_alpha_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    if (occ_numbers_alpha_.size() == nbasis_) {
        for (int i = 0; i < nbasis_; ++i) if (occ_numbers_alpha_(i) > 1e-12) P_alpha_ += occ_numbers_alpha_(i) * C_alpha_.col(i) * C_alpha_.col(i).transpose();
    } else {
        if (n_alpha_ > 0) P_alpha_ = C_alpha_.leftCols(n_alpha_) * C_alpha_.leftCols(n_alpha_).transpose();
    }
    P_beta_ = P_alpha_; 
}

void RHF::build_fock_matrix() {
    if (config_.eri_method == "cholesky" || config_.use_df) {
        Eigen::MatrixXd dP = (iter_scf_ == 1) ? P_alpha_ : (P_alpha_ - P_old_); 
        if (dP.cwiseAbs().maxCoeff() < 1e-11) {
            if (C_alpha_.rows() != nbasis_) F_alpha_ = H_ + G_accum_;
        } else {
            bool use_mo_alg = (iter_scf_ > 1) && (C_alpha_.rows() == nbasis_) && (C_alpha_.cols() == nbasis_);
            Eigen::MatrixXd dJ_mat = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
            Eigen::MatrixXd dK_acc = Eigen::MatrixXd::Zero(nbasis_, nbasis_); 
            
            int n_chol = L_mat_.cols(); 
            bool is_ooc = (L_mat_.rows() == 0 && n_chol > 0);
            
            std::unique_ptr<utils::HDF5TensorIO> io = nullptr;
            if (is_ooc) io = std::make_unique<utils::HDF5TensorIO>("df_tensor.h5", utils::HDF5TensorIO::Mode::READ_ONLY);
            
            int chunk_size = is_ooc ? 128 : n_chol; 
            
            // [FAST BLAS RE-USE VECTOR] Memetakan densitas menjadi vektor 1D
            Eigen::Map<const Eigen::VectorXd> dP_flat(dP.data(), nbasis_ * nbasis_);

            // Jika In-Core, jalankan operasi BLAS sekaligus untuk efisiensi L3 Cache maksimum
            if (!is_ooc) {
                Eigen::VectorXd X_J = L_mat_.transpose() * dP_flat; 
                Eigen::VectorXd J_flat = L_mat_ * X_J;                 
                dJ_mat = Eigen::Map<Eigen::MatrixXd>(J_flat.data(), nbasis_, nbasis_);
            }
            
            for (int K_start = 0; K_start < n_chol; K_start += chunk_size) {
                int K_end = std::min(n_chol, K_start + chunk_size);
                int k_size = K_end - K_start;
                
                Eigen::MatrixXd L_chunk;
                if (is_ooc) {
                    L_chunk.resize(nbasis_ * nbasis_, k_size);
                    io->read_slice_4d("df_tensor", {K_start, 0, 0, 0}, {k_size, (long)nbasis_, (long)nbasis_, 1}, L_chunk.data());
                    
                    // Operasi BLAS secara chunking untuk Out-Of-Core, memanfaatkan bandwidth lokal
                    Eigen::VectorXd X_J_chunk = L_chunk.transpose() * dP_flat;
                    Eigen::VectorXd J_flat_chunk = L_chunk * X_J_chunk;
                    dJ_mat += Eigen::Map<Eigen::MatrixXd>(J_flat_chunk.data(), nbasis_, nbasis_);
                }
                
                if (use_mo_alg && n_alpha_ > 0) {
                    // Algoritma basis-MO menggunakan TBLIS untuk matriks Exchange (K)
                    using tblis::len_type; using tblis::stride_type; using tblis::varray_view;
                    std::vector<len_type> len_L = { (len_type)nbasis_, (len_type)nbasis_, (len_type)k_size };
                    std::vector<stride_type> str_L = { 1, (stride_type)nbasis_, (stride_type)(nbasis_ * nbasis_) };
                    varray_view<double> t_L(len_L, is_ooc ? L_chunk.data() : const_cast<double*>(L_mat_.col(K_start).data()), str_L);

                    Eigen::MatrixXd Ca_occ = C_alpha_.leftCols(n_alpha_);
                    std::vector<len_type> len_Ca = { (len_type)nbasis_, (len_type)n_alpha_ };
                    std::vector<stride_type> str_Ca = { 1, (stride_type)nbasis_ };
                    varray_view<double> t_Ca(len_Ca, Ca_occ.data(), str_Ca);

                    std::vector<double> Ta_buf(nbasis_ * n_alpha_ * k_size, 0.0);
                    std::vector<len_type> len_Ta = { (len_type)nbasis_, (len_type)n_alpha_, (len_type)k_size };
                    std::vector<stride_type> str_Ta = { 1, (stride_type)nbasis_, (stride_type)(nbasis_ * n_alpha_) };
                    varray_view<double> t_Ta(len_Ta, Ta_buf.data(), str_Ta);

                    tblis::mult<double>(1.0, t_L, "mnP", t_Ca, "ni", 0.0, t_Ta, "miP");

                    std::vector<len_type> len_K = { (len_type)nbasis_, (len_type)nbasis_ };
                    std::vector<stride_type> str_K = { 1, (stride_type)nbasis_ };
                    varray_view<double> t_K(len_K, dK_acc.data(), str_K);

                    tblis::mult<double>(1.0, t_Ta, "miP", t_Ta, "niP", 1.0, t_K, "mn");
                } else if (!use_mo_alg) {
                    // Fallback basis AO dengan loop OpenMP konvensional
                    #pragma omp parallel
                    {
                        Eigen::MatrixXd K_priv = Eigen::MatrixXd::Zero(nbasis_, nbasis_); 
                        Eigen::MatrixXd T_buf(nbasis_, nbasis_);
                        #pragma omp for schedule(dynamic)
                        for (int k = 0; k < k_size; ++k) {
                            int K_global = K_start + k;
                            Eigen::Map<const Eigen::MatrixXd> L_K(is_ooc ? L_chunk.col(k).data() : L_mat_.col(K_global).data(), nbasis_, nbasis_);
                            T_buf.noalias() = L_K * dP; 
                            K_priv.noalias() += T_buf * L_K;
                        }
                        #pragma omp critical
                        { dK_acc += K_priv; }
                    }
                }
            } 

            if (use_mo_alg) {
                G_J_accum_ += 2.0 * dJ_mat; F_alpha_ = H_ + G_J_accum_ - dK_acc;
            } else {
                G_J_accum_ += 2.0 * dJ_mat; G_accum_ += (2.0 * dJ_mat - dK_acc); 
                F_alpha_ = H_ + G_accum_;
            }
        }
    } else if (config_.scf_type == "incore") {
        // [Optimasi Cache-Locality dari Kode 2]
        Eigen::MatrixXd dP = (iter_scf_ == 1) ? P_alpha_ : (P_alpha_ - P_old_);
        double max_dP = dP.cwiseAbs().maxCoeff(); 
        if (max_dP < 1e-11) {
            F_alpha_ = H_ + G_accum_;
        } else {
            const double* val = J_val_.data();
            const int* ind1 = J_ind_.data();
            const int* ind2 = K_ind_.data();
            size_t n_ints = J_val_.size();

            Eigen::MatrixXd dG = Eigen::MatrixXd::Zero(nbasis_, nbasis_);

            #pragma omp parallel
            {
                Eigen::MatrixXd G_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                #pragma omp for schedule(dynamic, 2048)
                for (size_t k = 0; k < n_ints; ++k) {
                    int p_mn = ind1[k]; int mu = (p_mn >> 16) & 0xFFFF; int nu = p_mn & 0xFFFF;
                    int p_ls = ind2[k]; int lam = (p_ls >> 16) & 0xFFFF; int sig = p_ls & 0xFFFF;
                    
                    double v = val[k];
                    double vJ = 4.0 * v; 
                    double vK = 2.0 * v;

                    double pt_ls = dP(lam, sig); 
                    double pt_mn = dP(mu, nu);
                    
                    // Coulomb
                    G_local(mu, nu) += vJ * pt_ls;
                    if (p_mn != p_ls) { 
                        G_local(lam, sig) += vJ * pt_mn; 
                    }
                    
                    // Exchange
                    G_local(mu, lam) -= vK * dP(nu, sig); 
                    G_local(mu, sig) -= vK * dP(nu, lam);
                    G_local(nu, lam) -= vK * dP(mu, sig); 
                    G_local(nu, sig) -= vK * dP(mu, lam);
                }
                #pragma omp critical
                { dG += G_local; }
            }
            dG = dG + dG.transpose().eval();
            for (int i = 0; i < nbasis_; ++i) dG(i, i) *= 0.5;
            
            G_accum_ += dG;
            F_alpha_ = H_ + G_accum_;
        }
    } else {
        Eigen::MatrixXd P_empty, F_empty; 
        fock_engine_->compute(P_alpha_, P_empty, F_alpha_, F_empty);
    }
    P_old_ = P_alpha_; 
    F_beta_ = F_alpha_;
}


void UHF::build_fock_matrix() { 
    if (config_.eri_method == "cholesky" || config_.use_df) {
        Eigen::MatrixXd dPa = (iter_scf_ == 1) ? P_alpha_ : (P_alpha_ - P_alpha_old_);
        Eigen::MatrixXd dPb = (iter_scf_ == 1) ? P_beta_  : (P_beta_ - P_beta_old_);
        Eigen::MatrixXd dP_tot = dPa + dPb;

        if (dP_tot.cwiseAbs().maxCoeff() < 1e-11) {
            if (C_alpha_.rows() != nbasis_) { F_alpha_ = H_ + G_accum_a_; F_beta_  = H_ + G_accum_b_; }
        } else {
            bool use_mo_alg = (iter_scf_ > 1) && (C_alpha_.rows() == nbasis_) && (C_beta_.rows() == nbasis_);
            Eigen::MatrixXd dJ_mat = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
            Eigen::MatrixXd dKa_acc = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
            Eigen::MatrixXd dKb_acc = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
            
            int n_chol = L_mat_.cols();
            bool is_ooc = (L_mat_.rows() == 0 && n_chol > 0);
            std::unique_ptr<utils::HDF5TensorIO> io = nullptr;
            if (is_ooc) io = std::make_unique<utils::HDF5TensorIO>("df_tensor.h5", utils::HDF5TensorIO::Mode::READ_ONLY);
            int chunk_size = is_ooc ? 128 : n_chol;
            
            // [FAST BLAS RE-USE VECTOR] Memetakan densitas menjadi vektor 1D
            Eigen::Map<const Eigen::VectorXd> dP_tot_flat(dP_tot.data(), nbasis_ * nbasis_);

            // Jika In-Core, jalankan operasi BLAS sekaligus untuk efisiensi L3 Cache maksimum
            if (!is_ooc) {
                Eigen::VectorXd X_J = L_mat_.transpose() * dP_tot_flat; 
                Eigen::VectorXd J_flat = L_mat_ * X_J;                 
                dJ_mat = Eigen::Map<Eigen::MatrixXd>(J_flat.data(), nbasis_, nbasis_);
            }
            
            for (int K_start = 0; K_start < n_chol; K_start += chunk_size) {
                int K_end = std::min(n_chol, K_start + chunk_size);
                int k_size = K_end - K_start;
                
                Eigen::MatrixXd L_chunk;
                if (is_ooc) {
                    L_chunk.resize(nbasis_ * nbasis_, k_size);
                    io->read_slice_4d("df_tensor", {K_start, 0, 0, 0}, {k_size, (long)nbasis_, (long)nbasis_, 1}, L_chunk.data());
                    
                    // Operasi BLAS secara chunking untuk Out-Of-Core, memanfaatkan bandwidth lokal
                    Eigen::VectorXd X_J_chunk = L_chunk.transpose() * dP_tot_flat;
                    Eigen::VectorXd J_flat_chunk = L_chunk * X_J_chunk;
                    dJ_mat += Eigen::Map<Eigen::MatrixXd>(J_flat_chunk.data(), nbasis_, nbasis_);
                }

                if (use_mo_alg) {
                    // Algoritma basis-MO menggunakan TBLIS untuk kompleksitas waktu lebih rendah
                    using tblis::len_type; using tblis::stride_type; using tblis::varray_view;
                    std::vector<len_type> len_L = { (len_type)nbasis_, (len_type)nbasis_, (len_type)k_size };
                    std::vector<stride_type> str_L = { 1, (stride_type)nbasis_, (stride_type)(nbasis_ * nbasis_) };
                    varray_view<double> t_L(len_L, is_ooc ? L_chunk.data() : const_cast<double*>(L_mat_.col(K_start).data()), str_L);

                    if (n_alpha_ > 0) {
                        Eigen::MatrixXd Ca_occ = C_alpha_.leftCols(n_alpha_);
                        std::vector<len_type> len_Ca = { (len_type)nbasis_, (len_type)n_alpha_ };
                        std::vector<stride_type> str_Ca = { 1, (stride_type)nbasis_ };
                        varray_view<double> t_Ca(len_Ca, Ca_occ.data(), str_Ca);

                        std::vector<double> Ta_buf(nbasis_ * n_alpha_ * k_size, 0.0);
                        std::vector<len_type> len_Ta = { (len_type)nbasis_, (len_type)n_alpha_, (len_type)k_size };
                        std::vector<stride_type> str_Ta = { 1, (stride_type)nbasis_, (stride_type)(nbasis_ * n_alpha_) };
                        varray_view<double> t_Ta(len_Ta, Ta_buf.data(), str_Ta);

                        tblis::mult<double>(1.0, t_L, "mnP", t_Ca, "ni", 0.0, t_Ta, "miP");
                        
                        std::vector<len_type> len_Ka = { (len_type)nbasis_, (len_type)nbasis_ };
                        std::vector<stride_type> str_Ka = { 1, (stride_type)nbasis_ };
                        varray_view<double> t_Ka(len_Ka, dKa_acc.data(), str_Ka);
                        tblis::mult<double>(1.0, t_Ta, "miP", t_Ta, "niP", 1.0, t_Ka, "mn");
                    }

                    if (n_beta_ > 0) {
                        Eigen::MatrixXd Cb_occ = C_beta_.leftCols(n_beta_);
                        std::vector<len_type> len_Cb = { (len_type)nbasis_, (len_type)n_beta_ };
                        std::vector<stride_type> str_Cb = { 1, (stride_type)nbasis_ };
                        varray_view<double> t_Cb(len_Cb, Cb_occ.data(), str_Cb);

                        std::vector<double> Tb_buf(nbasis_ * n_beta_ * k_size, 0.0);
                        std::vector<len_type> len_Tb = { (len_type)nbasis_, (len_type)n_beta_, (len_type)k_size };
                        std::vector<stride_type> str_Tb = { 1, (stride_type)nbasis_, (stride_type)(nbasis_ * n_beta_) };
                        varray_view<double> t_Tb(len_Tb, Tb_buf.data(), str_Tb);

                        tblis::mult<double>(1.0, t_L, "mnP", t_Cb, "ni", 0.0, t_Tb, "miP");
                        
                        std::vector<len_type> len_Kb = { (len_type)nbasis_, (len_type)nbasis_ };
                        std::vector<stride_type> str_Kb = { 1, (stride_type)nbasis_ };
                        varray_view<double> t_Kb(len_Kb, dKb_acc.data(), str_Kb);
                        tblis::mult<double>(1.0, t_Tb, "miP", t_Tb, "niP", 1.0, t_Kb, "mn");
                    }
                } else {
                    // Fallback basis AO dengan loop OpenMP konvensional
                    #pragma omp parallel
                    {
                        Eigen::MatrixXd Ka_priv = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                        Eigen::MatrixXd Kb_priv = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                        Eigen::MatrixXd Ta_buf(nbasis_, nbasis_);
                        Eigen::MatrixXd Tb_buf(nbasis_, nbasis_);
                        #pragma omp for schedule(dynamic)
                        for (int k = 0; k < k_size; ++k) {
                            int K_global = K_start + k;
                            Eigen::Map<const Eigen::MatrixXd> L_K(is_ooc ? L_chunk.col(k).data() : L_mat_.col(K_global).data(), nbasis_, nbasis_);
                            Ta_buf.noalias() = L_K * dPa; Ka_priv.noalias() += Ta_buf * L_K;
                            Tb_buf.noalias() = L_K * dPb; Kb_priv.noalias() += Tb_buf * L_K;
                        }
                        #pragma omp critical
                        { dKa_acc += Ka_priv; dKb_acc += Kb_priv; }
                    }
                }
            } 
            
            if (use_mo_alg) {
                G_J_accum_a_ += dJ_mat; G_J_accum_b_ += dJ_mat;
                F_alpha_ = H_ + G_J_accum_a_ - dKa_acc; F_beta_  = H_ + G_J_accum_b_ - dKb_acc;
            } else {
                G_J_accum_a_ += dJ_mat; G_J_accum_b_ += dJ_mat; 
                G_accum_a_ += (dJ_mat - dKa_acc); G_accum_b_ += (dJ_mat - dKb_acc);
                F_alpha_ = H_ + G_accum_a_; F_beta_  = H_ + G_accum_b_;
            }
        }
    } else if (config_.scf_type == "incore") {
        // [Kode 1 Dipertahankan: Cache-Locality Optimisation pada penulisan mu,nu/lam,sig]
        Eigen::MatrixXd dPa = (iter_scf_ == 1) ? P_alpha_ : (P_alpha_ - P_alpha_old_);
        Eigen::MatrixXd dPb = (iter_scf_ == 1) ? P_beta_ : (P_beta_ - P_beta_old_);
        Eigen::MatrixXd dP_tot = dPa + dPb; 
        double max_dP = dP_tot.cwiseAbs().maxCoeff(); 

        if (max_dP < 1e-11) {
            F_alpha_ = H_ + G_accum_a_;
            F_beta_  = H_ + G_accum_b_;
        } else {
            const double* val = J_val_.data();
            const int* ind1 = J_ind_.data();
            const int* ind2 = K_ind_.data();
            size_t n_ints = J_val_.size();

            Eigen::MatrixXd dGa = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
            Eigen::MatrixXd dGb = Eigen::MatrixXd::Zero(nbasis_, nbasis_);

            #pragma omp parallel
            {
                Eigen::MatrixXd Ga_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                Eigen::MatrixXd Gb_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                
                #pragma omp for schedule(dynamic, 2048)
                for (size_t k = 0; k < n_ints; ++k) {
                    int p_mn = ind1[k]; int mu = (p_mn >> 16) & 0xFFFF; int nu = p_mn & 0xFFFF;
                    int p_ls = ind2[k]; int lam = (p_ls >> 16) & 0xFFFF; int sig = p_ls & 0xFFFF;
                    double v = val[k];

                    double pt_ls = dP_tot(lam, sig);
                    double pt_mn = dP_tot(mu, nu);
                    
                    // Perbaikan: vJ harus 2.0, bukan 4.0 (menghindari double Coulomb)
                    double vJ = 2.0 * v; 
                    double vK = 1.0 * v;

                    double J_mn = vJ * pt_ls;
                    double J_ls = vJ * pt_mn;

                    Ga_local(mu, nu) += J_mn;
                    Gb_local(mu, nu) += J_mn;
                    if (p_mn != p_ls) {
                        Ga_local(lam, sig) += J_ls;
                        Gb_local(lam, sig) += J_ls;
                    }

                    Ga_local(mu, lam) -= vK * dPa(nu, sig);
                    Ga_local(mu, sig) -= vK * dPa(nu, lam);
                    Ga_local(nu, lam) -= vK * dPa(mu, sig);
                    Ga_local(nu, sig) -= vK * dPa(mu, lam);

                    Gb_local(mu, lam) -= vK * dPb(nu, sig);
                    Gb_local(mu, sig) -= vK * dPb(nu, lam);
                    Gb_local(nu, lam) -= vK * dPb(mu, sig);
                    Gb_local(nu, sig) -= vK * dPb(mu, lam);
                }
                #pragma omp critical
                { dGa += Ga_local; dGb += Gb_local; }
            }
            
            dGa = dGa + dGa.transpose().eval();
            dGb = dGb + dGb.transpose().eval();
            for (int i = 0; i < nbasis_; ++i) {
                dGa(i, i) *= 0.5;
                dGb(i, i) *= 0.5;
            }
            
            G_accum_a_ += dGa;
            G_accum_b_ += dGb;
            F_alpha_ = H_ + G_accum_a_; 
            F_beta_  = H_ + G_accum_b_;
        }
    } else {
        fock_engine_->compute(P_alpha_, P_beta_, F_alpha_, F_beta_); 
    }
    P_alpha_old_ = P_alpha_; P_beta_old_  = P_beta_; 
}

double RHF::compute_energy() { return P_alpha_.cwiseProduct(H_ + F_alpha_).sum(); }




UHF::UHF(const Molecule& mol, const BasisSet& basis, std::shared_ptr<IntegralEngine> integrals, std::shared_ptr<PointGroup> pg, std::shared_ptr<PetiteList> pl, int n_alpha, int n_beta, const SCFConfig& config)
    : BaseSCF(mol, basis, integrals, pg, pl, n_alpha, n_beta, config) {
    if (n_alpha < n_beta) throw std::runtime_error("UHF Error: High spin assumption violated.");
    
    G_accum_a_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_); 
    G_accum_b_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    
    
    G_J_accum_a_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    G_J_accum_b_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    
    
    P_alpha_old_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_); 
    P_beta_old_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
}

void UHF::initial_guess() {
    if (config_.print_level > 0) std::cout << "  [UHF] Generating Initial Guess... ";
    Eigen::MatrixXd P_sad = SADGuess::build(mol_, basis_);
    if (P_sad.rows() > 0 && P_sad.norm() > 1e-6) {
        P_alpha_ = 0.5 * P_sad; 
        P_beta_  = 0.5 * P_sad; 
        if (n_alpha_ != n_beta_) {
            for (int i=0; i<nbasis_; ++i) if (i % 2 == 0) P_alpha_(i,i) *= 1.1; 
        }
        
        
        solve_fock(H_, C_alpha_, eps_alpha_);
        solve_fock(H_, C_beta_, eps_beta_); 
        occ_numbers_alpha_ = smear_electrons(eps_alpha_, n_alpha_);
        occ_numbers_beta_  = smear_electrons(eps_beta_, n_beta_);
        
        energy_ = 0.0; return; 
    }
    if (config_.print_level > 0) std::cout << "Using GWH.\n";
    Eigen::MatrixXd F_gwh = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    for (int i = 0; i < nbasis_; ++i) for (int j = 0; j < nbasis_; ++j) F_gwh(i, j) = (i == j) ? H_(i, j) : 0.5 * 1.75 * S_(i, j) * (H_(i, i) + H_(j, j));
    solve_fock(F_gwh, C_alpha_, eps_alpha_); C_beta_ = C_alpha_; eps_beta_ = eps_alpha_;
    occ_numbers_alpha_ = smear_electrons(eps_alpha_, n_alpha_); occ_numbers_beta_  = smear_electrons(eps_beta_, n_beta_); update_densities(); energy_ = 0.0;
}

void UHF::update_densities() {
    
    
    P_alpha_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_); 
    P_beta_  = Eigen::MatrixXd::Zero(nbasis_, nbasis_);

    
    if (n_alpha_ > 0) {
        P_alpha_.noalias() = C_alpha_.leftCols(n_alpha_) * C_alpha_.leftCols(n_alpha_).transpose();
    }
    if (n_beta_ > 0) {
        P_beta_.noalias() = C_beta_.leftCols(n_beta_) * C_beta_.leftCols(n_beta_).transpose();
    }
}


double UHF::compute_energy() { return 0.5 * (P_alpha_.cwiseProduct(H_ + F_alpha_).sum() + P_beta_.cwiseProduct(H_ + F_beta_).sum()); }

double UHF::compute_s2(const Eigen::MatrixXd& Ca, const Eigen::MatrixXd& Cb, const Eigen::MatrixXd& S) {
    double sz = 0.5 * (n_alpha_ - n_beta_);
    if(n_alpha_ == 0 || n_beta_ == 0) return sz * (sz + 1.0);
    return sz * (sz + 1.0) + n_beta_ - (Ca.leftCols(n_alpha_).transpose() * S * Cb.leftCols(n_beta_)).array().square().sum();
}




ROHF::ROHF(const Molecule& mol, const BasisSet& basis, std::shared_ptr<IntegralEngine> integrals, std::shared_ptr<PointGroup> pg, std::shared_ptr<PetiteList> pl, int n_alpha, int n_beta, const SCFConfig& config)
    : BaseSCF(mol, basis, integrals, pg, pl, n_alpha, n_beta, config) {
    if (n_alpha < n_beta) throw std::runtime_error("ROHF Error: Alpha must be >= Beta.");
    G_accum_a_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_); 
    G_accum_b_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_); 
    P_alpha_old_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_); 
    P_beta_old_  = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    G_J_accum_a_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    G_J_accum_b_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
}

void ROHF::initial_guess() {
    if (config_.print_level > 0) std::cout << "  [ROHF] Generating Initial Guess... ";
    Eigen::MatrixXd P_sad = SADGuess::build(mol_, basis_);
    if (P_sad.rows() > 0 && P_sad.norm() > 1e-6) {
        if (config_.print_level > 0) std::cout << "Using SAD.\n";
        P_alpha_ = 0.5 * P_sad; 
        P_beta_  = 0.5 * P_sad; 
        
        solve_fock(H_, C_alpha_, eps_alpha_);
        C_beta_ = C_alpha_; 
        eps_beta_ = eps_alpha_;
        occ_numbers_alpha_ = smear_electrons(eps_alpha_, n_alpha_); 
        occ_numbers_beta_  = smear_electrons(eps_beta_, n_beta_);
        energy_ = 0.0; return; 
    }
    Eigen::MatrixXd F_gwh = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    for (int i = 0; i < nbasis_; ++i) for (int j = 0; j < nbasis_; ++j) F_gwh(i, j) = (i == j) ? H_(i, j) : 0.5 * 1.75 * S_(i, j) * (H_(i, i) + H_(j, j));
    solve_fock(F_gwh, C_alpha_, eps_alpha_); C_beta_ = C_alpha_; eps_beta_ = eps_alpha_;
    occ_numbers_alpha_ = smear_electrons(eps_alpha_, n_alpha_); occ_numbers_beta_  = smear_electrons(eps_beta_, n_beta_); update_densities(); energy_ = 0.0;
}

void ROHF::update_densities() {
    P_alpha_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_); P_beta_  = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    if (occ_numbers_alpha_.size() == nbasis_) {
        for (int i = 0; i < nbasis_; ++i) {
            if (occ_numbers_alpha_(i) > 1e-12) P_alpha_ += occ_numbers_alpha_(i) * C_alpha_.col(i) * C_alpha_.col(i).transpose();
            if (occ_numbers_beta_(i) > 1e-12)  P_beta_  += occ_numbers_beta_(i) * C_alpha_.col(i) * C_alpha_.col(i).transpose();
        }
    } else {
        if (n_alpha_ > 0) P_alpha_ = C_alpha_.leftCols(n_alpha_) * C_alpha_.leftCols(n_alpha_).transpose();
        if (n_beta_ > 0)  P_beta_  = C_alpha_.leftCols(n_beta_) * C_alpha_.leftCols(n_beta_).transpose();
    }
}



void ROHF::build_fock_matrix() { 
    if (config_.eri_method == "cholesky" || config_.use_df) {
        Eigen::MatrixXd dPa = (iter_scf_ == 1) ? P_alpha_ : (P_alpha_ - P_alpha_old_);
        Eigen::MatrixXd dPb = (iter_scf_ == 1) ? P_beta_  : (P_beta_ - P_beta_old_);
        Eigen::MatrixXd dP_tot = dPa + dPb;

        if (dP_tot.cwiseAbs().maxCoeff() < 1e-11) {
            if (C_alpha_.rows() != nbasis_) { F_alpha_ = H_ + G_accum_a_; F_beta_  = H_ + G_accum_b_; }
        } else {
            bool use_mo_alg = (iter_scf_ > 1) && (C_alpha_.rows() == nbasis_);
            Eigen::MatrixXd dJ_mat = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
            Eigen::MatrixXd dKa_acc = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
            Eigen::MatrixXd dKb_acc = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
            
            int n_chol = L_mat_.cols();
            bool is_ooc = (L_mat_.rows() == 0 && n_chol > 0);
            std::unique_ptr<utils::HDF5TensorIO> io = nullptr;
            if (is_ooc) io = std::make_unique<utils::HDF5TensorIO>("df_tensor.h5", utils::HDF5TensorIO::Mode::READ_ONLY);
            int chunk_size = is_ooc ? 128 : n_chol;
            
            // [FAST BLAS RE-USE VECTOR] Memetakan densitas menjadi vektor 1D
            Eigen::Map<const Eigen::VectorXd> dP_tot_flat(dP_tot.data(), nbasis_ * nbasis_);

            // Jika In-Core, jalankan operasi BLAS sekaligus untuk efisiensi L3 Cache maksimum
            if (!is_ooc) {
                Eigen::VectorXd X_J = L_mat_.transpose() * dP_tot_flat; 
                Eigen::VectorXd J_flat = L_mat_ * X_J;                 
                dJ_mat = Eigen::Map<Eigen::MatrixXd>(J_flat.data(), nbasis_, nbasis_);
            }
            
            for (int K_start = 0; K_start < n_chol; K_start += chunk_size) {
                int K_end = std::min(n_chol, K_start + chunk_size);
                int k_size = K_end - K_start;
                
                Eigen::MatrixXd L_chunk;
                if (is_ooc) {
                    L_chunk.resize(nbasis_ * nbasis_, k_size);
                    io->read_slice_4d("df_tensor", {K_start, 0, 0, 0}, {k_size, (long)nbasis_, (long)nbasis_, 1}, L_chunk.data());
                    
                    // Operasi BLAS secara chunking untuk Out-Of-Core, memanfaatkan bandwidth lokal
                    Eigen::VectorXd X_J_chunk = L_chunk.transpose() * dP_tot_flat;
                    Eigen::VectorXd J_flat_chunk = L_chunk * X_J_chunk;
                    dJ_mat += Eigen::Map<Eigen::MatrixXd>(J_flat_chunk.data(), nbasis_, nbasis_);
                }

                if (use_mo_alg) {
                    using tblis::len_type; using tblis::stride_type; using tblis::varray_view;
                    std::vector<len_type> len_L = { (len_type)nbasis_, (len_type)nbasis_, (len_type)k_size };
                    std::vector<stride_type> str_L = { 1, (stride_type)nbasis_, (stride_type)(nbasis_ * nbasis_) };
                    varray_view<double> t_L(len_L, is_ooc ? L_chunk.data() : const_cast<double*>(L_mat_.col(K_start).data()), str_L);

                    if (n_alpha_ > 0) {
                        Eigen::MatrixXd Ca_occ = C_alpha_.leftCols(n_alpha_);
                        std::vector<len_type> len_Ca = { (len_type)nbasis_, (len_type)n_alpha_ };
                        std::vector<stride_type> str_Ca = { 1, (stride_type)nbasis_ };
                        varray_view<double> t_Ca(len_Ca, Ca_occ.data(), str_Ca);

                        std::vector<double> Ta_buf(nbasis_ * n_alpha_ * k_size, 0.0);
                        std::vector<len_type> len_Ta = { (len_type)nbasis_, (len_type)n_alpha_, (len_type)k_size };
                        std::vector<stride_type> str_Ta = { 1, (stride_type)nbasis_, (stride_type)(nbasis_ * n_alpha_) };
                        varray_view<double> t_Ta(len_Ta, Ta_buf.data(), str_Ta);

                        tblis::mult<double>(1.0, t_L, "mnP", t_Ca, "ni", 0.0, t_Ta, "miP");
                        
                        std::vector<len_type> len_Ka = { (len_type)nbasis_, (len_type)nbasis_ };
                        std::vector<stride_type> str_Ka = { 1, (stride_type)nbasis_ };
                        varray_view<double> t_Ka(len_Ka, dKa_acc.data(), str_Ka);
                        tblis::mult<double>(1.0, t_Ta, "miP", t_Ta, "niP", 1.0, t_Ka, "mn");
                    }

                    if (n_beta_ > 0) {
                        Eigen::MatrixXd Cb_occ = C_alpha_.leftCols(n_beta_); 
                        std::vector<len_type> len_Cb = { (len_type)nbasis_, (len_type)n_beta_ };
                        std::vector<stride_type> str_Cb = { 1, (stride_type)nbasis_ };
                        varray_view<double> t_Cb(len_Cb, Cb_occ.data(), str_Cb);

                        std::vector<double> Tb_buf(nbasis_ * n_beta_ * k_size, 0.0);
                        std::vector<len_type> len_Tb = { (len_type)nbasis_, (len_type)n_beta_, (len_type)k_size };
                        std::vector<stride_type> str_Tb = { 1, (stride_type)nbasis_, (stride_type)(nbasis_ * n_beta_) };
                        varray_view<double> t_Tb(len_Tb, Tb_buf.data(), str_Tb);

                        tblis::mult<double>(1.0, t_L, "mnP", t_Cb, "ni", 0.0, t_Tb, "miP");
                        
                        std::vector<len_type> len_Kb = { (len_type)nbasis_, (len_type)nbasis_ };
                        std::vector<stride_type> str_Kb = { 1, (stride_type)nbasis_ };
                        varray_view<double> t_Kb(len_Kb, dKb_acc.data(), str_Kb);
                        tblis::mult<double>(1.0, t_Tb, "miP", t_Tb, "niP", 1.0, t_Kb, "mn");
                    }
                } else {
                    // Fallback basis AO dengan loop OpenMP konvensional
                    #pragma omp parallel
                    {
                        Eigen::MatrixXd Ka_priv = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                        Eigen::MatrixXd Kb_priv = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                        Eigen::MatrixXd Ta_buf(nbasis_, nbasis_);
                        Eigen::MatrixXd Tb_buf(nbasis_, nbasis_);
                        #pragma omp for schedule(dynamic)
                        for (int k = 0; k < k_size; ++k) {
                            int K_global = K_start + k;
                            Eigen::Map<const Eigen::MatrixXd> L_K(is_ooc ? L_chunk.col(k).data() : L_mat_.col(K_global).data(), nbasis_, nbasis_);
                            Ta_buf.noalias() = L_K * dPa; Ka_priv.noalias() += Ta_buf * L_K;
                            Tb_buf.noalias() = L_K * dPb; Kb_priv.noalias() += Tb_buf * L_K;
                        }
                        #pragma omp critical
                        { dKa_acc += Ka_priv; dKb_acc += Kb_priv; }
                    }
                }
            } 
            
            if (use_mo_alg) {
                G_J_accum_a_ += dJ_mat; G_J_accum_b_ += dJ_mat;
                F_alpha_ = H_ + G_J_accum_a_ - dKa_acc; F_beta_  = H_ + G_J_accum_b_ - dKb_acc;
            } else {
                G_J_accum_a_ += dJ_mat; G_J_accum_b_ += dJ_mat; 
                G_accum_a_ += (dJ_mat - dKa_acc); G_accum_b_ += (dJ_mat - dKb_acc);
                F_alpha_ = H_ + G_accum_a_; F_beta_  = H_ + G_accum_b_;
            }
        }
    } else if (config_.scf_type == "incore") {
        // [Optimasi Cache-Locality pada penulisan mu,nu/lam,sig]
        Eigen::MatrixXd dPa = (iter_scf_ == 1) ? P_alpha_ : (P_alpha_ - P_alpha_old_);
        Eigen::MatrixXd dPb = (iter_scf_ == 1) ? P_beta_ : (P_beta_ - P_beta_old_);
        Eigen::MatrixXd dP_tot = dPa + dPb; 
        double max_dP = dP_tot.cwiseAbs().maxCoeff(); 

        if (max_dP < 1e-11) {
            F_alpha_ = H_ + G_accum_a_;
            F_beta_  = H_ + G_accum_b_;
        } else {
            const double* val = J_val_.data();
            const int* ind1 = J_ind_.data();
            const int* ind2 = K_ind_.data();
            size_t n_ints = J_val_.size();

            Eigen::MatrixXd dGa = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
            Eigen::MatrixXd dGb = Eigen::MatrixXd::Zero(nbasis_, nbasis_);

            #pragma omp parallel
            {
                Eigen::MatrixXd Ga_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                Eigen::MatrixXd Gb_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                
                #pragma omp for schedule(dynamic, 2048)
                for (size_t k = 0; k < n_ints; ++k) {
                    int p_mn = ind1[k]; int mu = (p_mn >> 16) & 0xFFFF; int nu = p_mn & 0xFFFF;
                    int p_ls = ind2[k]; int lam = (p_ls >> 16) & 0xFFFF; int sig = p_ls & 0xFFFF;
                    double v = val[k];

                    double pt_ls = dP_tot(lam, sig);
                    double pt_mn = dP_tot(mu, nu);
                    
                    // Perbaikan: vJ harus 2.0, bukan 4.0 (menghindari double Coulomb)
                    double vJ = 2.0 * v; 
                    double vK = 1.0 * v;

                    double J_mn = vJ * pt_ls;
                    double J_ls = vJ * pt_mn;

                    Ga_local(mu, nu) += J_mn;
                    Gb_local(mu, nu) += J_mn;
                    if (p_mn != p_ls) {
                        Ga_local(lam, sig) += J_ls;
                        Gb_local(lam, sig) += J_ls;
                    }

                    Ga_local(mu, lam) -= vK * dPa(nu, sig);
                    Ga_local(mu, sig) -= vK * dPa(nu, lam);
                    Ga_local(nu, lam) -= vK * dPa(mu, sig);
                    Ga_local(nu, sig) -= vK * dPa(mu, lam);

                    Gb_local(mu, lam) -= vK * dPb(nu, sig);
                    Gb_local(mu, sig) -= vK * dPb(nu, lam);
                    Gb_local(nu, lam) -= vK * dPb(mu, sig);
                    Gb_local(nu, sig) -= vK * dPb(mu, lam);
                }
                #pragma omp critical
                { dGa += Ga_local; dGb += Gb_local; }
            }
            
            dGa = dGa + dGa.transpose().eval();
            dGb = dGb + dGb.transpose().eval();
            for (int i = 0; i < nbasis_; ++i) {
                dGa(i, i) *= 0.5;
                dGb(i, i) *= 0.5;
            }
            
            G_accum_a_ += dGa;
            G_accum_b_ += dGb;
            F_alpha_ = H_ + G_accum_a_; 
            F_beta_  = H_ + G_accum_b_;
        }
    } else {
        fock_engine_->compute(P_alpha_, P_beta_, F_alpha_, F_beta_); 
    }
    P_alpha_old_ = P_alpha_; P_beta_old_  = P_beta_; 
}
double ROHF::compute_energy() { return 0.5 * (P_alpha_.cwiseProduct(H_ + F_alpha_).sum() + P_beta_.cwiseProduct(H_ + F_beta_).sum()); }

SCFResult ROHF::compute() {
    if (n_alpha_ == n_beta_) { RHF rhf_engine(mol_, basis_, integrals_, pg_, pl_, config_); return rhf_engine.compute(); }
    init_integrals(); initial_guess(); 
    if (config_.scf_type == "direct") { fock_engine_ = std::make_unique<FockBuilder>(integrals_, basis_, H_, schwarz_); if (pl_) fock_engine_->set_petite_list(pl_.get()); }

    DIIS diis(config_.diis_max_vectors); bool converged = false; double thresh = 1e-6;
    if (config_.print_level > 0) printf("\n Iter        Energy (Ha)        Delta E    Delta P\n");

    for (iter_scf_ = 1; iter_scf_ <= config_.max_iterations; iter_scf_++) {
        energy_old_ = energy_; Eigen::MatrixXd P_tot_old = P_alpha_ + P_beta_;
        if (iter_scf_ > 2) { double de = std::abs(energy_ - energy_old_); if (de < 1e-3) thresh = 1e-9; if (de < 1e-6) thresh = 1e-12; }
        if (symmetrizer_) { symmetrizer_->symmetrize(P_alpha_); symmetrizer_->symmetrize(P_beta_); }

        build_fock_matrix();
        if (symmetrizer_) { symmetrizer_->symmetrize(F_alpha_); symmetrizer_->symmetrize(F_beta_); }
        energy_ = compute_energy();
        
        Eigen::MatrixXd Fa_mo = C_alpha_.transpose() * F_alpha_ * C_alpha_; Eigen::MatrixXd Fb_mo = C_alpha_.transpose() * F_beta_  * C_alpha_;
        Eigen::MatrixXd F_uni = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
        int n_c = n_beta_; int n_o = n_alpha_ - n_beta_; int n_v = nbasis_ - n_alpha_; double shift_o = 0.5; double shift_v = 1.0; 

        if (n_c > 0) F_uni.topLeftCorner(n_c, n_c) = 0.5 * (Fa_mo.topLeftCorner(n_c, n_c) + Fb_mo.topLeftCorner(n_c, n_c));
        if (n_o > 0) { F_uni.block(n_c, n_c, n_o, n_o) = Fa_mo.block(n_c, n_c, n_o, n_o); for(int i=0; i<n_o; ++i) F_uni(n_c+i, n_c+i) += shift_o; }
        if (n_v > 0) { F_uni.bottomRightCorner(n_v, n_v) = 0.5 * (Fa_mo.bottomRightCorner(n_v, n_v) + Fb_mo.bottomRightCorner(n_v, n_v)); for(int i=0; i<n_v; ++i) F_uni(n_alpha_+i, n_alpha_+i) += shift_v; }
        if (n_c > 0 && n_o > 0) { Eigen::MatrixXd blk = Fb_mo.block(0, n_c, n_c, n_o); F_uni.block(0, n_c, n_c, n_o) = blk; F_uni.block(n_c, 0, n_o, n_c) = blk.transpose(); }
        if (n_c > 0 && n_v > 0) { Eigen::MatrixXd blk = 0.5 * (Fa_mo.block(0, n_alpha_, n_c, n_v) + Fb_mo.block(0, n_alpha_, n_c, n_v)); F_uni.block(0, n_alpha_, n_c, n_v) = blk; F_uni.block(n_alpha_, 0, n_v, n_c) = blk.transpose(); }
        if (n_o > 0 && n_v > 0) { Eigen::MatrixXd blk = Fa_mo.block(n_c, n_alpha_, n_o, n_v); F_uni.block(n_c, n_alpha_, n_o, n_v) = blk; F_uni.block(n_alpha_, n_c, n_v, n_o) = blk.transpose(); }

        Eigen::MatrixXd SC = S_ * C_alpha_; Eigen::MatrixXd F_eff_ao = SC * F_uni * SC.transpose(); Eigen::MatrixXd FPS = F_eff_ao * P_alpha_ * S_; 
        diis.add_iteration(F_eff_ao, FPS - FPS.transpose(), P_alpha_);
        solve_fock((iter_scf_ <= 1) ? 0.7 * F_eff_ao + 0.3 * H_ : diis.extrapolate(), C_alpha_, eps_alpha_); C_beta_ = C_alpha_; eps_beta_ = eps_alpha_;
        occ_numbers_alpha_ = smear_electrons(eps_alpha_, n_alpha_); occ_numbers_beta_  = smear_electrons(eps_beta_, n_beta_); update_densities();
        
        double dE = std::abs(energy_ - energy_old_); double dP = ((P_alpha_ + P_beta_) - P_tot_old).norm() / nbasis_;
        if (config_.print_level > 0) printf(" %4d %18.10f %10.2e %10.2e\n", iter_scf_, energy(), dE, dP);
        if (dE < config_.energy_threshold && dP < config_.density_threshold) { converged = true; break; }
    }
    
    
    if (config_.use_df && L_mat_.rows() == 0) {
        if (config_.print_level > 0) std::cout << "\n  [SCF] Reloading df_tensor.h5 to RAM for MP2 compatibility...\n";
        int n_aux = L_mat_.cols();
        L_mat_.resize(nbasis_ * nbasis_, n_aux);
        utils::HDF5TensorIO io("df_tensor.h5", utils::HDF5TensorIO::Mode::READ_ONLY);
        io.read_slice_4d("df_tensor", {0, 0, 0, 0}, {(long)n_aux, (long)nbasis_, (long)nbasis_, 1}, L_mat_.data());
    }
    
    SCFResult r; r.energy_total = energy(); r.iterations = iter_scf_; r.converged = converged;
    r.C_alpha = C_alpha_; r.C_beta = C_beta_; r.P_alpha = P_alpha_; r.P_beta = P_beta_; r.F_alpha = F_alpha_; r.F_beta = F_beta_;
    r.orbital_energies_alpha = eps_alpha_; r.orbital_energies_beta = eps_beta_; r.n_occ_alpha = n_alpha_; r.n_occ_beta = n_beta_; 
    r.L_mat = L_mat_; 
    
    print_final(r); return r;
}

} 
