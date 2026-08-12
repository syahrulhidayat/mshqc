

#include "mshqc/integrals/eri_transformer.h"
#include "mshqc/symmetry/blocked_tensor.h"
#include "mshqc/utils/hdf5_io.h"
#include "mshqc/JIT/ExecutionManager.h"
#include "mshqc/compiler/Frontend/GraphBuilder.h"
#include "mshqc/Runtime/MemRefUtils.h"
#include "mshqc/Runtime/AlignedAllocator.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Core>

namespace mshqc {
namespace integrals {

static Eigen::Tensor<double, 4> smart_transform_kernel_jit(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C1, const Eigen::MatrixXd& C2,
    const Eigen::MatrixXd& C3, const Eigen::MatrixXd& C4,
    int nbf, int n1, int n2, int n3, int n4) {

    static jit::ExecutionManager jit_mgr;
    static bool is_eri_compiled = false;
    const std::string kernel_name = "eri_transform_kernel_full";

    if (!is_eri_compiled) {
        compiler::GraphBuilder builder;
        builder.initializeModule(kernel_name);
        builder.emitContractOp(
            {nbf, nbf, nbf, nbf}, {nbf, n1}, {n1, n2, n3, n4},
            "abcd,ae,bf,cg,dh->efgh"
        );
        jit_mgr.compileAndCache(kernel_name, builder.getModule());
        is_eri_compiled = true;
    }

    std::vector<double, runtime::AlignedAllocator<double, 64>> eri_mo_buf(n1 * n2 * n3 * n4, 0.0);

    runtime::StridedMemRefType<double, 4> memref_eri_ao;
    memref_eri_ao.allocatedPtr = const_cast<double*>(eri_ao.data());
    memref_eri_ao.alignedPtr = memref_eri_ao.allocatedPtr;
    memref_eri_ao.offset = 0;
    memref_eri_ao.sizes[0] = memref_eri_ao.sizes[1] = memref_eri_ao.sizes[2] = memref_eri_ao.sizes[3] = nbf;
    memref_eri_ao.strides[3] = 1; memref_eri_ao.strides[2] = nbf;
    memref_eri_ao.strides[1] = nbf * nbf; memref_eri_ao.strides[0] = nbf * nbf * nbf;

    auto memref_C1 = runtime::makeMemRef2D(const_cast<Eigen::MatrixXd&>(C1), nbf, n1);
    auto memref_C2 = runtime::makeMemRef2D(const_cast<Eigen::MatrixXd&>(C2), nbf, n2);
    auto memref_C3 = runtime::makeMemRef2D(const_cast<Eigen::MatrixXd&>(C3), nbf, n3);
    auto memref_C4 = runtime::makeMemRef2D(const_cast<Eigen::MatrixXd&>(C4), nbf, n4);

    runtime::StridedMemRefType<double, 4> memref_eri_mo;
    memref_eri_mo.allocatedPtr = eri_mo_buf.data();
    memref_eri_mo.alignedPtr = memref_eri_mo.allocatedPtr;
    memref_eri_mo.offset = 0;
    memref_eri_mo.sizes[0] = n1; memref_eri_mo.sizes[1] = n2;
    memref_eri_mo.sizes[2] = n3; memref_eri_mo.sizes[3] = n4;
    memref_eri_mo.strides[3] = 1; memref_eri_mo.strides[2] = n4;
    memref_eri_mo.strides[1] = n3 * n4; memref_eri_mo.strides[0] = n2 * n3 * n4;

    void* args[] = { &memref_eri_ao, &memref_C1, &memref_C2, &memref_C3, &memref_C4, &memref_eri_mo };

    try {
        jit_mgr.execute(kernel_name, "contract_kernel", args);
    } catch(const std::exception& e) {
        std::cerr << "[FATAL] MSHQC JIT Trap: Eksekusi Transformasi Integral Gagal: " << e.what() << "\n";
        std::abort();
    }

    Eigen::Tensor<double, 4> result(n1, n2, n3, n4);
    std::copy(eri_mo_buf.begin(), eri_mo_buf.end(), result.data());
    return result;
}

Eigen::Tensor<double, 4> ERITransformer::transform_custom(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C1, const Eigen::MatrixXd& C2,
    const Eigen::MatrixXd& C3, const Eigen::MatrixXd& C4,
    int nbf, int n1, int n2, int n3, int n4) {
    return smart_transform_kernel_jit(eri_ao, C1, C2, C3, C4, nbf, n1, n2, n3, n4);
}

Eigen::Tensor<double, 4> ERITransformer::get_mo_tensor(
    bool use_df, int n_aux,
    const Eigen::MatrixXd& C1, const Eigen::MatrixXd& C2,
    const Eigen::MatrixXd& C3, const Eigen::MatrixXd& C4,
    std::shared_ptr<mshqc::IntegralEngine> ints) {

    int nbf = C1.rows();
    int dim1 = C1.cols(); int dim2 = C2.cols();
    int dim3 = C3.cols(); int dim4 = C4.cols();

    if (!use_df) {
        return transform_oovv_mixed(ints->compute_eri(), C1, C2, C3, C4, nbf, dim1, dim2, dim3, dim4);
    }

    static jit::ExecutionManager jit_mgr;
    static bool is_df_assembly_compiled = false;
    const std::string kernel_df = "df_assembly_kernel";

    if (!is_df_assembly_compiled) {
        compiler::GraphBuilder builder;
        builder.initializeModule(kernel_df);
        builder.emitContractOp(
            {nbf, nbf, n_aux}, {nbf, dim1}, {nbf, dim2}, {nbf, dim3}, {nbf, dim4}, {dim1, dim2, dim3, dim4},
            "mnP,mp,nq,rsP,rt,su->pqtu"
        );
        jit_mgr.compileAndCache(kernel_df, builder.getModule());
        is_df_assembly_compiled = true;
    }

    std::vector<double, runtime::AlignedAllocator<double, 64>> V_mo_buf(dim1 * dim2 * dim3 * dim4, 0.0);
    Eigen::Tensor<double, 4> V_mo(dim1, dim2, dim3, dim4);
    std::copy(V_mo_buf.begin(), V_mo_buf.end(), V_mo.data());
    return V_mo;
}

}
}
