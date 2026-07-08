/**
 * @file src/scf/scf.cc
 * @brief Unified SCF Engine (BaseSCF, RHF, UHF, ROHF) in a single file.
 * @details Fully embeds Cholesky L_mat_ and SIMD In-Core for all methods.
 */

#include "mshqc/scf.h"
#include "mshqc/diis.h"
#include "mshqc/sad.h"
#include "mshqc/symmetry/salc_builder.h"
#include "mshqc/integrals/df_eri.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <omp.h>
#include <tblis/tblis.h>
#include <vector>
#ifdef I
#undef I
#endif

namespace mshqc {



 ============================================================================
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
    if (config_.print_level > 0) std::cout << "  [SCF] Building Optimized CRS Integrals... " << std::flush;

    J_val_.clear(); J_ind_.clear(); J_ptr_.clear();
    K_val_.clear(); K_ind_.clear(); K_ptr_.clear(); row_map_.clear();
    
    long long est_nnz = (long long)(std::pow(nbasis_, 4) * 0.15); 
    if (est_nnz > 0) { J_val_.reserve(est_nnz); J_ind_.reserve(est_nnz); K_val_.reserve(est_nnz); K_ind_.reserve(est_nnz); }
    row_map_.reserve(nbasis_ * nbasis_ / 2); schwarz_basis_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    J_ptr_.push_back(0); K_ptr_.push_back(0);

    const auto& ERI = integrals_->compute_eri();
    const double sparse_threshold = 1e-12; int nshells = basis_.n_shells(); double shell_cutoff = 1e-12;

    struct SymPair { int M; int N; double w; };
    std::vector<SymPair> shell_pairs;
    if (pl_ && !pl_->get_unique_pairs().empty()) {
        for (const auto& u : pl_->get_unique_pairs()) shell_pairs.push_back({u.p, u.q, u.weight}); 
    } else {
        for(int i = 0; i < nshells; ++i) for(int j = 0; j <= i; ++j) shell_pairs.push_back({i, j, 1.0});
    }

    for (const auto& pair_mn : shell_pairs) {
        int M = pair_mn.M; int N = pair_mn.N; double sym_weight = pair_mn.w; double Q_MN = schwarz_(M, N); 
        if (Q_MN * schwarz_.maxCoeff() < shell_cutoff) continue;

        for (int m = 0; m < shell_sizes_[M]; ++m) {
            for (int n = 0; n < shell_sizes_[N]; ++n) {
                int mu = shell_starts_[M] + m; int nu = shell_starts_[N] + n;
                if (M == N && nu > mu) continue;
                row_map_.push_back({mu, nu}); double max_val_row = 0.0;
                
                for (int L = 0; L < nshells; ++L) {
                    for (int S = 0; S < nshells; ++S) {
                        if (Q_MN * schwarz_(L, S) < shell_cutoff) continue;
                        for (int s_idx = 0; s_idx < shell_sizes_[S]; ++s_idx) {
                            for (int l_idx = 0; l_idx < shell_sizes_[L]; ++l_idx) {
                                int sig = shell_starts_[S] + s_idx; int lam = shell_starts_[L] + l_idx; int density_idx = lam + sig * nbasis_;
                                double vJ = ERI(mu, nu, lam, sig);
                                if (std::abs(vJ) > sparse_threshold) { J_val_.push_back(vJ * sym_weight); J_ind_.push_back(density_idx); max_val_row = std::max(max_val_row, std::abs(vJ)); }
                                double vK = ERI(mu, lam, nu, sig);
                                if (std::abs(vK) > sparse_threshold) { K_val_.push_back(vK * sym_weight); K_ind_.push_back(density_idx); }
                            }
                        }
                    }
                }
                J_ptr_.push_back(J_val_.size()); K_ptr_.push_back(K_val_.size());
                schwarz_basis_(mu, nu) = std::sqrt(max_val_row); schwarz_basis_(nu, mu) = std::sqrt(max_val_row);
            }
        }
    }
    if (config_.print_level > 0) std::cout << "Done. (Mapped Rows: " << row_map_.size() << ")\n";
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
            
            for (int K_start = 0; K_start < n_chol; K_start += chunk_size) {
                int K_end = std::min(n_chol, K_start + chunk_size);
                int k_size = K_end - K_start;
                
                Eigen::MatrixXd L_chunk;
                if (is_ooc) {
                    L_chunk.resize(nbasis_ * nbasis_, k_size);
                    io->read_slice_4d("df_tensor", {K_start, 0, 0, 0}, {k_size, (long)nbasis_, (long)nbasis_, 1}, L_chunk.data());
                }

                #pragma omp parallel
                {
                    Eigen::MatrixXd J_priv  = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                    #pragma omp for schedule(dynamic)
                    for (int k = 0; k < k_size; ++k) {
                        int K_global = K_start + k;
                        Eigen::Map<const Eigen::MatrixXd> L_K(is_ooc ? L_chunk.col(k).data() : L_mat_.col(K_global).data(), nbasis_, nbasis_);
                        double val_J = (L_K.cwiseProduct(dP_tot)).sum();
                        J_priv += val_J * L_K;
                    }
                    #pragma omp critical
                    { dJ_mat += J_priv; }
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
                    #pragma omp parallel
                    {
                        Eigen::MatrixXd Ka_priv = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                        Eigen::MatrixXd Kb_priv = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                        Eigen::MatrixXd Ta_buf(nbasis_, nbasis_), Tb_buf(nbasis_, nbasis_);
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
        Eigen::MatrixXd dP = (iter_scf_ == 1) ? P_alpha_ : (P_alpha_ - P_old_);
        double max_dP = dP.cwiseAbs().maxCoeff(); 
        if (max_dP < 1e-11) {
            F_alpha_ = H_ + G_accum_;
        } else {
            const double* __restrict__ p_d = dP.data();
            const double* __restrict__ Jv = J_val_.data(); const int* __restrict__ Ji = J_ind_.data(); const size_t* __restrict__ Jp = J_ptr_.data();
            const double* __restrict__ Kv = K_val_.data(); const int* __restrict__ Ki = K_ind_.data(); const size_t* __restrict__ Kp = K_ptr_.data();

            Eigen::MatrixXd dG = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
            size_t n_rows = row_map_.size();
            
            #pragma omp parallel
            {
                Eigen::MatrixXd G_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                #pragma omp for schedule(dynamic, 32)
                for (size_t r = 0; r < n_rows; ++r) {
                    int mu = row_map_[r].first; int nu = row_map_[r].second;
                    if (schwarz_basis_(mu, nu) * max_dP < 1e-12) continue;

                    size_t js = Jp[r]; size_t je = Jp[r+1];
                    size_t ks = Kp[r]; size_t ke = Kp[r+1];
                    double vj = 0.0, vk = 0.0;
                    
                    #pragma omp simd reduction(+:vj)
                    for (size_t k = js; k < je; ++k) vj += Jv[k] * p_d[Ji[k]];
                    
                    #pragma omp simd reduction(+:vk)
                    for (size_t k = ks; k < ke; ++k) vk += Kv[k] * p_d[Ki[k]];
                    
                    G_local(mu, nu) += 2.0 * vj - vk;
                }
                #pragma omp critical
                { dG += G_local; }
            } 
            for (int i = 0; i < nbasis_; ++i) {
                for (int j = 0; j < i; ++j) dG(j, i) = dG(i, j); 
            }
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
void UHF::build_fock_matrix() { 
    if (config_.eri_method == "cholesky" || config_.use_df) {
        Eigen::MatrixXd dPa = (iter_scf_ == 1) ? P_alpha_ : (P_alpha_ - P_alpha_old_);
        Eigen::MatrixXd dPb = (iter_scf_ == 1) ? P_beta_  : (P_beta_ - P_beta_old_);
        Eigen::MatrixXd dP_tot = dPa + dPb;

        if (dP_tot.cwiseAbs().maxCoeff() < 1e-11) {
            if (C_alpha_.rows() != nbasis_) {
                F_alpha_ = H_ + G_accum_a_;
                F_beta_  = H_ + G_accum_b_;
            }
        } else {
            bool use_mo_alg = (iter_scf_ > 1) && (C_alpha_.rows() == nbasis_) && (C_alpha_.cols() == nbasis_) &&
                  (C_beta_.rows() == nbasis_)  && (C_beta_.cols() == nbasis_);
            
            Eigen::MatrixXd dJ_mat = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
            Eigen::MatrixXd dKa_acc = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
            Eigen::MatrixXd dKb_acc = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
            int n_chol = L_mat_.cols();

            #pragma omp parallel
            {
                Eigen::MatrixXd J_priv  = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                #pragma omp for schedule(dynamic)
                for (int K = 0; K < n_chol; ++K) {
                    Eigen::Map<const Eigen::MatrixXd> L_K(L_mat_.col(K).data(), nbasis_, nbasis_);
                    double val_J = (L_K.cwiseProduct(dP_tot)).sum();
                    J_priv += val_J * L_K;
                }
                #pragma omp critical
                { dJ_mat += J_priv; }
            }

            if (use_mo_alg) {
                using tblis::len_type;
                using tblis::stride_type;
                using tblis::varray_view;

                std::vector<len_type> len_L = { (len_type)nbasis_, (len_type)nbasis_, (len_type)n_chol };
                std::vector<stride_type> str_L = { 1, (stride_type)nbasis_, (stride_type)(nbasis_ * nbasis_) };
                varray_view<double> t_L(len_L, L_mat_.data(), str_L);

                
                if (n_alpha_ > 0) {
                    Eigen::MatrixXd Ca_occ = C_alpha_.leftCols(n_alpha_);
                    std::vector<len_type> len_Ca = { (len_type)nbasis_, (len_type)n_alpha_ };
                    std::vector<stride_type> str_Ca = { 1, (stride_type)nbasis_ };
                    varray_view<double> t_Ca(len_Ca, Ca_occ.data(), str_Ca);

                    std::vector<double> Ta_buf(nbasis_ * n_alpha_ * n_chol, 0.0);
                    std::vector<len_type> len_Ta = { (len_type)nbasis_, (len_type)n_alpha_, (len_type)n_chol };
                    std::vector<stride_type> str_Ta = { 1, (stride_type)nbasis_, (stride_type)(nbasis_ * n_alpha_) };
                    varray_view<double> t_Ta(len_Ta, Ta_buf.data(), str_Ta);

                    tblis::mult<double>(1.0, t_L, "mnP", t_Ca, "ni", 0.0, t_Ta, "miP");

                    std::vector<len_type> len_Ka = { (len_type)nbasis_, (len_type)nbasis_ };
                    std::vector<stride_type> str_Ka = { 1, (stride_type)nbasis_ };
                    varray_view<double> t_Ka(len_Ka, dKa_acc.data(), str_Ka);
                    tblis::mult<double>(1.0, t_Ta, "miP", t_Ta, "niP", 0.0, t_Ka, "mn");
                }

                
                if (n_beta_ > 0) {
                    Eigen::MatrixXd Cb_occ = C_beta_.leftCols(n_beta_);
                    std::vector<len_type> len_Cb = { (len_type)nbasis_, (len_type)n_beta_ };
                    std::vector<stride_type> str_Cb = { 1, (stride_type)nbasis_ };
                    varray_view<double> t_Cb(len_Cb, Cb_occ.data(), str_Cb);

                    std::vector<double> Tb_buf(nbasis_ * n_beta_ * n_chol, 0.0);
                    std::vector<len_type> len_Tb = { (len_type)nbasis_, (len_type)n_beta_, (len_type)n_chol };
                    std::vector<stride_type> str_Tb = { 1, (stride_type)nbasis_, (stride_type)(nbasis_ * n_beta_) };
                    varray_view<double> t_Tb(len_Tb, Tb_buf.data(), str_Tb);

                    tblis::mult<double>(1.0, t_L, "mnP", t_Cb, "ni", 0.0, t_Tb, "miP");

                    std::vector<len_type> len_Kb = { (len_type)nbasis_, (len_type)nbasis_ };
                    std::vector<stride_type> str_Kb = { 1, (stride_type)nbasis_ };
                    varray_view<double> t_Kb(len_Kb, dKb_acc.data(), str_Kb);
                    tblis::mult<double>(1.0, t_Tb, "miP", t_Tb, "niP", 0.0, t_Kb, "mn");
                }

                G_J_accum_a_ += dJ_mat; G_J_accum_b_ += dJ_mat;
                F_alpha_ = H_ + G_J_accum_a_ - dKa_acc;
                F_beta_  = H_ + G_J_accum_b_ - dKb_acc;

            } else {
                #pragma omp parallel
                {
                    Eigen::MatrixXd Ka_priv = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                    Eigen::MatrixXd Kb_priv = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                    Eigen::MatrixXd Ta_buf(nbasis_, nbasis_);
                    Eigen::MatrixXd Tb_buf(nbasis_, nbasis_);
                    #pragma omp for schedule(dynamic)
                    for (int K = 0; K < n_chol; ++K) {
                        Eigen::Map<const Eigen::MatrixXd> L_K(L_mat_.col(K).data(), nbasis_, nbasis_);
                        Ta_buf.noalias() = L_K * dPa; Ka_priv.noalias() += Ta_buf * L_K;
                        Tb_buf.noalias() = L_K * dPb; Kb_priv.noalias() += Tb_buf * L_K;
                    }
                    #pragma omp critical
                    { dKa_acc += Ka_priv; dKb_acc += Kb_priv; }
                }
                
                G_J_accum_a_ += dJ_mat; G_J_accum_b_ += dJ_mat; 
                G_accum_a_ += (dJ_mat - dKa_acc); G_accum_b_ += (dJ_mat - dKb_acc);
                F_alpha_ = H_ + G_accum_a_; F_beta_  = H_ + G_accum_b_;
            }
        }
    } else if (config_.scf_type == "incore") {
        Eigen::MatrixXd dPa = (iter_scf_ == 1) ? P_alpha_ : (P_alpha_ - P_alpha_old_);
        Eigen::MatrixXd dPb = (iter_scf_ == 1) ? P_beta_ : (P_beta_ - P_beta_old_);
        Eigen::MatrixXd dP_tot = dPa + dPb; 
        double max_dP = dP_tot.cwiseAbs().maxCoeff(); 

        if (max_dP < 1e-11) {
            F_alpha_ = H_ + G_accum_a_;
            F_beta_  = H_ + G_accum_b_;
        } else {
            const double* __restrict__ p_dtot = dP_tot.data();
            const double* __restrict__ p_da   = dPa.data();
            const double* __restrict__ p_db   = dPb.data();
            const double* __restrict__ Jv = J_val_.data(); const int* __restrict__ Ji = J_ind_.data(); const size_t* __restrict__ Jp = J_ptr_.data();
            const double* __restrict__ Kv = K_val_.data(); const int* __restrict__ Ki = K_ind_.data(); const size_t* __restrict__ Kp = K_ptr_.data();

            Eigen::MatrixXd dGa = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
            Eigen::MatrixXd dGb = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
            size_t n_rows = row_map_.size();

            #pragma omp parallel
            {
                Eigen::MatrixXd Ga_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                Eigen::MatrixXd Gb_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);

                #pragma omp for schedule(dynamic, 32)
                for (size_t r = 0; r < n_rows; ++r) {
                    int mu = row_map_[r].first; int nu = row_map_[r].second;
                    if (schwarz_basis_(mu, nu) * max_dP < 1e-12) continue;

                    size_t js = Jp[r]; size_t je = Jp[r+1];
                    size_t ks = Kp[r]; size_t ke = Kp[r+1];

                    double vj = 0.0, ka = 0.0, kb = 0.0;
                    #pragma omp simd reduction(+:vj)
                    for (size_t k = js; k < je; ++k) vj += Jv[k] * p_dtot[Ji[k]];

                    #pragma omp simd reduction(+:ka, kb)
                    for (size_t k = ks; k < ke; ++k) {
                        double v = Kv[k]; int idx = Ki[k];
                        ka += v * p_da[idx]; kb += v * p_db[idx];
                    }
                    Ga_local(mu, nu) += vj - ka; Gb_local(mu, nu) += vj - kb;
                }
                #pragma omp critical
                { dGa += Ga_local; dGb += Gb_local; }
            }
            
            for (int i = 0; i < nbasis_; ++i) {
                for (int j = 0; j < i; ++j) { dGa(j, i) = dGa(i, j); dGb(j, i) = dGb(i, j); }
            }
            G_accum_a_ += dGa; G_accum_b_ += dGb;
            F_alpha_ = H_ + G_accum_a_; 
            F_beta_  = H_ + G_accum_b_;
        }
    } else {
        fock_engine_->compute(P_alpha_, P_beta_, F_alpha_, F_beta_); 
    }
    P_alpha_old_ = P_alpha_; 
    P_beta_old_  = P_beta_; 
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
            if (C_alpha_.rows() != nbasis_) { 
                F_alpha_ = H_ + G_accum_a_; 
                F_beta_  = H_ + G_accum_b_; 
            }
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
            
            for (int K_start = 0; K_start < n_chol; K_start += chunk_size) {
                int K_end = std::min(n_chol, K_start + chunk_size);
                int k_size = K_end - K_start;
                
                Eigen::MatrixXd L_chunk;
                if (is_ooc) {
                    L_chunk.resize(nbasis_ * nbasis_, k_size);
                    io->read_slice_4d("df_tensor", {K_start, 0, 0, 0}, {k_size, (long)nbasis_, (long)nbasis_, 1}, L_chunk.data());
                }

                #pragma omp parallel
                {
                    Eigen::MatrixXd J_priv  = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                    #pragma omp for schedule(dynamic)
                    for (int k = 0; k < k_size; ++k) {
                        int K_global = K_start + k;
                        Eigen::Map<const Eigen::MatrixXd> L_K(is_ooc ? L_chunk.col(k).data() : L_mat_.col(K_global).data(), nbasis_, nbasis_);
                        double val_J = (L_K.cwiseProduct(dP_tot)).sum();
                        J_priv += val_J * L_K;
                    }
                    #pragma omp critical
                    { dJ_mat += J_priv; }
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
                    #pragma omp parallel
                    {
                        Eigen::MatrixXd Ka_priv = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                        Eigen::MatrixXd Kb_priv = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                        Eigen::MatrixXd Ta_buf(nbasis_, nbasis_), Tb_buf(nbasis_, nbasis_);
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
        
        Eigen::MatrixXd dPa = (iter_scf_ == 1) ? P_alpha_ : (P_alpha_ - P_alpha_old_);
        Eigen::MatrixXd dPb = (iter_scf_ == 1) ? P_beta_ : (P_beta_ - P_beta_old_);
        Eigen::MatrixXd dP_tot = dPa + dPb; 
        double max_dP = dP_tot.cwiseAbs().maxCoeff(); 

        if (max_dP < 1e-11) {
            F_alpha_ = H_ + G_accum_a_;
            F_beta_  = H_ + G_accum_b_;
        } else {
            const double* __restrict__ p_dtot = dP_tot.data();
            const double* __restrict__ p_da   = dPa.data();
            const double* __restrict__ p_db   = dPb.data();
            const double* __restrict__ Jv = J_val_.data(); const int* __restrict__ Ji = J_ind_.data(); const size_t* __restrict__ Jp = J_ptr_.data();
            const double* __restrict__ Kv = K_val_.data(); const int* __restrict__ Ki = K_ind_.data(); const size_t* __restrict__ Kp = K_ptr_.data();

            Eigen::MatrixXd dGa = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
            Eigen::MatrixXd dGb = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
            size_t n_rows = row_map_.size();

            #pragma omp parallel
            {
                Eigen::MatrixXd Ga_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
                Eigen::MatrixXd Gb_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);

                #pragma omp for schedule(dynamic, 32)
                for (size_t r = 0; r < n_rows; ++r) {
                    int mu = row_map_[r].first; int nu = row_map_[r].second;
                    if (schwarz_basis_(mu, nu) * max_dP < 1e-12) continue;

                    size_t js = Jp[r]; size_t je = Jp[r+1];
                    size_t ks = Kp[r]; size_t ke = Kp[r+1];

                    double vj = 0.0, ka = 0.0, kb = 0.0;
                    #pragma omp simd reduction(+:vj)
                    for (size_t k = js; k < je; ++k) vj += Jv[k] * p_dtot[Ji[k]];

                    #pragma omp simd reduction(+:ka, kb)
                    for (size_t k = ks; k < ke; ++k) {
                        double v = Kv[k]; int idx = Ki[k];
                        ka += v * p_da[idx]; kb += v * p_db[idx];
                    }
                    Ga_local(mu, nu) += vj - ka; Gb_local(mu, nu) += vj - kb;
                }
                #pragma omp critical
                { dGa += Ga_local; dGb += Gb_local; }
            }
            
            for (int i = 0; i < nbasis_; ++i) {
                for (int j = 0; j < i; ++j) { dGa(j, i) = dGa(i, j); dGb(j, i) = dGb(i, j); }
            }
            G_accum_a_ += dGa; G_accum_b_ += dGb;
            F_alpha_ = H_ + G_accum_a_; 
            F_beta_  = H_ + G_accum_b_;
        }
    } else {
        fock_engine_->compute(P_alpha_, P_beta_, F_alpha_, F_beta_); 
    }
    P_alpha_old_ = P_alpha_; 
    P_beta_old_  = P_beta_; 
}

double ROHF::compute_energy() { return 0.5 * (P_alpha_.cwiseProduct(H_ + F_alpha_).sum() + P_beta_.cwiseProduct(H_ + F_beta_).sum()); }

SCFResult ROHF::compute() {
    if (n_alpha_ == n_beta_) { RHF rhf_engine(mol_, basis_, integrals_, pg_, pl_, config_); return rhf_engine.compute(); }
    init_integrals(); initial_guess(); 
    if (config_.scf_type == "direct") { fock_engine_ = std::make_unique<FockBuilder>(integrals_, basis_, H_, schwarz_); if (pl_) fock_engine_->set_petite_list(pl_.get()); }

    DIIS diis(config_.diis_max_vectors); bool converged = false; double thresh = 1e-6;
    if (config_.print_level > 0) printf("\n Iter       Energy (Ha)        Delta E    Delta P\n");

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
    
    SCFResult r; r.energy_total = energy(); r.iterations = iter_scf_; r.converged = converged;
    r.C_alpha = C_alpha_; r.C_beta = C_beta_; r.P_alpha = P_alpha_; r.P_beta = P_beta_; r.F_alpha = F_alpha_; r.F_beta = F_beta_;
    r.orbital_energies_alpha = eps_alpha_; r.orbital_energies_beta = eps_beta_; r.n_occ_alpha = n_alpha_; r.n_occ_beta = n_beta_; 
    print_final(r); return r;
}

} 
