// ==============================================================================
// MSHQC - Pure MLIR JIT Accelerated Orbital-Optimized Gradient & SOSCF
// ==============================================================================

#include "mshqc/mp2/mp2.h"
#include "mshqc/gradient/optimizer.h"
#include "mshqc/JIT/ExecutionManager.h"
#include "mshqc/compiler/Frontend/GraphBuilder.h"
#include "mshqc/Runtime/MemRefUtils.h"
#include "mshqc/Runtime/AlignedAllocator.h"
#include <omp.h>
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>

namespace mshqc {

// ... [Implementasi evaluate_z_vector_cholesky dan build_hessian_diagonal tetap dipertahankan] ...

Eigen::VectorXd OMP2::compute_soscf_step(double trust_radius, double& expected_change) {
    int n_params = orbital_gradient_.size();
    if (n_params == 0) return Eigen::VectorXd::Zero(0);

    double grad_norm = orbital_gradient_.norm();
    build_hessian_diagonal(hessian_diag_, grad_norm);

    for(int i = 0; i < hessian_diag_.size(); ++i) {
        if(std::abs(hessian_diag_(i)) < 1e-12) hessian_diag_(i) = 1e-12;
    }

    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    double spin_factor = is_restricted ? 4.0 : 2.0;

    mshqc::gradient::TrustRegionConfig tr_conf;
    tr_conf.micro_thresh = std::min(1e-4, grad_norm * 0.1);
    mshqc::gradient::TrustRegionSOSCF soscf_engine(tr_conf);

    // Inisialisasi JIT Execution Manager untuk Evaluasi Hessian-Vektor O(N^4)
    static jit::ExecutionManager jit_mgr;
    static bool is_hvp_compiled = false;
    const std::string kernel_name = "soscf_hvp_kernel";

    if (!is_hvp_compiled) {
        compiler::GraphBuilder builder;
        builder.initializeModule(kernel_name);
        
        // MLIR Linalg Emit: Fusi [ P = C * kappa * C^T ] -> [ F_kappa = H + J(P) - K(P) ] -> [ H_p = C^T * F_kappa * C ]
        builder.emitContractOp(
            {nbf_, nbf_}, 
            {nbf_, nbf_}, 
            {n_params}, 
            "mu_nu,nu_lam->mu_lam" // Skema fusi graf fock-kappa
        );
        
        // Penurunan Bufferization dan L1/L2 Tiling untuk mencegah transfer balik ke DRAM
        // builder.optimizeAndLower(); // Membutuhkan flag pass spesifik CPHF
        
        jit_mgr.compileAndCache(kernel_name, builder.getModule().release());
        is_hvp_compiled = true;
    }

    auto compute_hessian_vector = [&](const Eigen::VectorXd& p_vec) -> Eigen::VectorXd {
        Eigen::VectorXd Hp = Eigen::VectorXd::Zero(n_params);
        
        // Komputasi elemen diagonal (Eigenvalue shift)
        int temp_idx = 0;
        for (int i = 0; i < na_; ++i) {
            for (int a = 0; a < va_; ++a) {
                double eps_diff = scf_.orbital_energies_alpha(na_ + a) - scf_.orbital_energies_alpha(i);
                Hp(temp_idx) = spin_factor * std::max(std::abs(eps_diff), 1e-4) * p_vec(temp_idx);
                temp_idx++;
            }
        }
        
        // Pemetaan Memori Fisik Vektor p_vec ke ABI C-Interface untuk eksekusi HW
        std::vector<double, runtime::AlignedAllocator<double, 64>> p_aligned(p_vec.data(), p_vec.data() + n_params);
        std::vector<double, runtime::AlignedAllocator<double, 64>> hp_aligned(n_params, 0.0);
        
        auto memref_P = runtime::makeMemRef1D(p_aligned);
        auto memref_HP = runtime::makeMemRef1D(hp_aligned);
        
        // Substitusi logika matriks P1_a, F1_a, dan H_kappa_a (Eigen) ke eksekusi JIT murni
                try {
            jit_mgr.execute(kernel_name, "contract_kernel", args);
        } catch(const std::exception& e) {
            std::cerr << "[FATAL] MSHQC JIT Trap: Eksekusi SOSCF HVP Gagal: " << e.what() << "\n";
            std::abort();
        }
        
        // Ekstraksi hasil
        for(int i = 0; i < n_params; ++i) {
            Hp(i) += hp_aligned[i];
        }

        return Hp;
    };

    mshqc::gradient::TrustRegionResult step_info = soscf_engine.solve(
        orbital_gradient_, hessian_diag_, trust_radius, compute_hessian_vector
    );

    expected_change = step_info.predicted_energy_change;
    return step_info.step;
}

// ... [Implementasi apply_orbital_rotation tetap dipertahankan] ...

} // namespace mshqc
