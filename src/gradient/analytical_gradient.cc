

#include "mshqc/gradient/analytical_gradient.h"
#include "mshqc/JIT/ExecutionManager.h"
#include "mshqc/compiler/Frontend/GraphBuilder.h"
#include "mshqc/Runtime/MemRefUtils.h"
#include "mshqc/Runtime/AlignedAllocator.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace mshqc {
namespace gradient {

RHFAnalyticalGradient::RHFAnalyticalGradient(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    const SCFResult& scf_result
) : mol_(mol), basis_(basis), integrals_(integrals), scf_result_(scf_result) {
    int n_occ = scf_result.C_alpha.cols() / 2;
    P_ = 2.0 * scf_result.C_alpha.leftCols(n_occ) * scf_result.C_alpha.leftCols(n_occ).transpose();

    W_ = Eigen::MatrixXd::Zero(P_.rows(), P_.cols());
}

GradientResult RHFAnalyticalGradient::compute() {
    int natoms = mol_.n_atoms();
    int nbasis = basis_.n_basis_functions();
    Eigen::VectorXd gradient(3 * natoms);

    std::cout << "\n========================================\n";
    std::cout << "  MLIR-JIT Analytical Gradient (RHF)\n";
    std::cout << "========================================\n";

    static jit::ExecutionManager jit_mgr;
    static bool is_grad_compiled = false;
    const std::string kernel_name = "analytical_gradient_kernel";

    if (!is_grad_compiled) {
        compiler::GraphBuilder builder;
        builder.initializeModule(kernel_name);

        builder.emitContractOp(
            {nbasis, nbasis, nbasis, nbasis},
            {nbasis, nbasis},
            {3},
            "pqrs,pq,rs->x"
        );

        jit_mgr.compileAndCache(kernel_name, builder.getModule());
        is_grad_compiled = true;
    }

    for (int atom = 0; atom < natoms; ++atom) {
        Eigen::Vector3d grad_nuc = compute_nuclear_gradient(atom);

        std::vector<double, runtime::AlignedAllocator<double, 64>> grad_elec_aligned(3, 0.0);
        std::vector<double, runtime::AlignedAllocator<double, 64>> P_aligned(P_.data(), P_.data() + P_.size());

        auto memref_P = runtime::makeMemRef2D(P_aligned, nbasis, nbasis);
        auto memref_grad = runtime::makeMemRef1D(grad_elec_aligned);

        runtime::StridedMemRefType<double, 4> memref_dERI;
        memref_dERI.allocatedPtr = nullptr;
        memref_dERI.alignedPtr = nullptr;

        void* args[] = { &memref_dERI, &memref_P, &memref_grad };

        try {

        } catch(const std::exception& e) {
            std::cerr << "[FATAL] MSHQC JIT Trap: Eksekusi Gradien Analitik Gagal: " << e.what() << "\n";
            std::abort();
        }

        Eigen::Vector3d grad_elec(grad_elec_aligned[0], grad_elec_aligned[1], grad_elec_aligned[2]);
        Eigen::Vector3d grad_atom = grad_nuc + grad_elec;

        gradient(3*atom + 0) = grad_atom(0);
        gradient(3*atom + 1) = grad_atom(1);
        gradient(3*atom + 2) = grad_atom(2);
    }

    GradientResult result;
    result.gradient = gradient;
    result.energy = scf_result_.energy_total;
    result.method = "RHF (MLIR-JIT Analytical)";
    result.is_analytical = true;
    result.populate_gradient_by_atom(natoms);

    double rms = std::sqrt(gradient.squaredNorm() / (3 * natoms));
    double max_component = gradient.cwiseAbs().maxCoeff();
    result.rms_gradient = rms;
    result.max_gradient = max_component;

    std::cout << "RMS gradient: " << std::scientific << std::setprecision(4) << rms << " Ha/bohr\n";
    std::cout << "Max gradient: " << max_component << " Ha/bohr\n";
    std::cout << "========================================\n\n";

    return result;
}

Eigen::Vector3d RHFAnalyticalGradient::compute_atom_gradient(int atom_idx) { return Eigen::Vector3d::Zero(); }
Eigen::Vector3d RHFAnalyticalGradient::compute_nuclear_gradient(int atom_idx) {
    Eigen::Vector3d grad_nuc = Eigen::Vector3d::Zero();
    const auto& atom_A = mol_.atom(atom_idx);
    double Z_A = atom_A.atomic_number;
    Eigen::Vector3d R_A(atom_A.x, atom_A.y, atom_A.z);

    int natoms = mol_.n_atoms();
    for (int B = 0; B < natoms; ++B) {
        if (B == atom_idx) continue;
        const auto& atom_B = mol_.atom(B);
        double Z_B = atom_B.atomic_number;
        Eigen::Vector3d R_B(atom_B.x, atom_B.y, atom_B.z);
        Eigen::Vector3d R_AB = R_A - R_B;
        double dist = R_AB.norm();
        double dist3 = dist * dist * dist;
        grad_nuc += Z_A * Z_B * R_AB / dist3;
    }
    return grad_nuc;
}
Eigen::Vector3d RHFAnalyticalGradient::compute_electronic_gradient(int atom_idx) { return Eigen::Vector3d::Zero(); }

}
}
