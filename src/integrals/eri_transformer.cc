

#include "mshqc/integrals/eri_transformer.h"
#include "mshqc/JIT/ExecutionManager.h"
#include "mshqc/compiler/Frontend/GraphBuilder.h"
#include "mshqc/Runtime/MemRefUtils.h"
#include "mshqc/Runtime/AlignedAllocator.h"
#include <iostream>

namespace mshqc {
namespace integrals {

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
            {nbf, nbf, n_aux},
            {nbf, dim1},
            {nbf, dim2},
            {nbf, dim3},
            {nbf, dim4},
            {dim1, dim2, dim3, dim4},
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
