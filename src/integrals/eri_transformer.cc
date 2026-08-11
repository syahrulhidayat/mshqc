// ==============================================================================
// Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
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
// ==============================================================================


#include "mshqc/integrals/eri_transformer.h"
#include "mshqc/symmetry/blocked_tensor.h"
#include "mshqc/utils/hdf5_io.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <tblis/tblis.h>

namespace mshqc {
namespace integrals {

static Eigen::Tensor<double, 4> smart_transform_kernel(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C1,
    const Eigen::MatrixXd& C2,
    const Eigen::MatrixXd& C3,
    const Eigen::MatrixXd& C4,
    int nbf, int n1, int n2, int n3, int n4
) {
    using tblis::len_type;
    using tblis::stride_type;
    using tblis::varray_view;

    auto make_view = [](const Eigen::MatrixXd& C) {
        std::vector<len_type> len = { (len_type)C.rows(), (len_type)C.cols() };
        std::vector<stride_type> str = { 1, (stride_type)C.rows() };
        return varray_view<double>(len, const_cast<double*>(C.data()), str);
    };

    auto t_C1 = make_view(C1);
    auto t_C2 = make_view(C2);
    auto t_C3 = make_view(C3);
    auto t_C4 = make_view(C4);

    std::vector<len_type> len_eri = { (len_type)nbf, (len_type)nbf, (len_type)nbf, (len_type)nbf };
    std::vector<stride_type> str_eri = { 1, (stride_type)nbf, (stride_type)(nbf*nbf), (stride_type)(nbf*nbf*nbf) };
    varray_view<double> t_eri(len_eri, const_cast<double*>(eri_ao.data()), str_eri);

    Eigen::Tensor<double, 4> T1(n1, nbf, nbf, nbf);
    T1.setZero();
    std::vector<len_type> len_T1 = { (len_type)n1, (len_type)nbf, (len_type)nbf, (len_type)nbf };
    std::vector<stride_type> str_T1 = { 1, (stride_type)n1, (stride_type)(n1*nbf), (stride_type)(n1*nbf*nbf) };
    varray_view<double> t_T1_view(len_T1, T1.data(), str_T1);
    tblis::mult<double>(1.0, t_eri, "abcd", t_C1, "ae", 0.0, t_T1_view, "ebcd");

    Eigen::Tensor<double, 4> T2(n1, n2, nbf, nbf);
    T2.setZero();
    std::vector<len_type> len_T2 = { (len_type)n1, (len_type)n2, (len_type)nbf, (len_type)nbf };
    std::vector<stride_type> str_T2 = { 1, (stride_type)n1, (stride_type)(n1*n2), (stride_type)(n1*n2*nbf) };
    varray_view<double> t_T2_view(len_T2, T2.data(), str_T2);
    tblis::mult<double>(1.0, t_T1_view, "ebcd", t_C2, "bf", 0.0, t_T2_view, "efcd");

    Eigen::Tensor<double, 4> T3(n1, n2, n3, nbf);
    T3.setZero();
    std::vector<len_type> len_T3 = { (len_type)n1, (len_type)n2, (len_type)n3, (len_type)nbf };
    std::vector<stride_type> str_T3 = { 1, (stride_type)n1, (stride_type)(n1*n2), (stride_type)(n1*n2*n3) };
    varray_view<double> t_T3_view(len_T3, T3.data(), str_T3);
    tblis::mult<double>(1.0, t_T2_view, "efcd", t_C3, "cg", 0.0, t_T3_view, "efgd");

    Eigen::Tensor<double, 4> result(n1, n2, n3, n4);
    result.setZero();
    std::vector<len_type> len_out = { (len_type)n1, (len_type)n2, (len_type)n3, (len_type)n4 };
    std::vector<stride_type> str_out = { 1, (stride_type)n1, (stride_type)(n1*n2), (stride_type)(n1*n2*n3) };
    varray_view<double> t_out_view(len_out, result.data(), str_out);
    tblis::mult<double>(1.0, t_T3_view, "efgd", t_C4, "dh", 0.0, t_out_view, "efgh");

    return result;
}

static Eigen::Tensor<double, 4> optimal_oovv_kernel(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& Co, const Eigen::MatrixXd& Cv,
    int nbf, int no, int nv
) {
    using tblis::len_type;
    using tblis::stride_type;
    using tblis::varray_view;

    auto make_view = [](const Eigen::MatrixXd& C) {
        std::vector<len_type> len = { (len_type)C.rows(), (len_type)C.cols() };
        std::vector<stride_type> str = { 1, (stride_type)C.rows() };
        return varray_view<double>(len, const_cast<double*>(C.data()), str);
    };

    auto t_Co = make_view(Co);
    auto t_Cv = make_view(Cv);

    std::vector<len_type> len_eri = { (len_type)nbf, (len_type)nbf, (len_type)nbf, (len_type)nbf };
    std::vector<stride_type> str_eri = { 1, (stride_type)nbf, (stride_type)(nbf*nbf), (stride_type)(nbf*nbf*nbf) };
    varray_view<double> t_eri(len_eri, const_cast<double*>(eri_ao.data()), str_eri);

    Eigen::Tensor<double, 4> T1(no, nbf, nbf, nbf);
    T1.setZero();
    std::vector<len_type> len_T1 = { (len_type)no, (len_type)nbf, (len_type)nbf, (len_type)nbf };
    std::vector<stride_type> str_T1 = { 1, (stride_type)no, (stride_type)(no*nbf), (stride_type)(no*nbf*nbf) };
    varray_view<double> t_T1_view(len_T1, T1.data(), str_T1);
    tblis::mult<double>(1.0, t_eri, "abcd", t_Co, "ae", 0.0, t_T1_view, "ebcd");

    Eigen::Tensor<double, 4> T2(no, nbf, no, nbf);
    T2.setZero();
    std::vector<len_type> len_T2 = { (len_type)no, (len_type)nbf, (len_type)no, (len_type)nbf };
    std::vector<stride_type> str_T2 = { 1, (stride_type)no, (stride_type)(no*nbf), (stride_type)(no*nbf*no) };
    varray_view<double> t_T2_view(len_T2, T2.data(), str_T2);
    tblis::mult<double>(1.0, t_T1_view, "ebcd", t_Co, "cf", 0.0, t_T2_view, "ebfd");

    Eigen::Tensor<double, 4> T3(no, nv, no, nbf);
    T3.setZero();
    std::vector<len_type> len_T3 = { (len_type)no, (len_type)nv, (len_type)no, (len_type)nbf };
    std::vector<stride_type> str_T3 = { 1, (stride_type)no, (stride_type)(no*nv), (stride_type)(no*nv*no) };
    varray_view<double> t_T3_view(len_T3, T3.data(), str_T3);
    tblis::mult<double>(1.0, t_T2_view, "ebfd", t_Cv, "bg", 0.0, t_T3_view, "egfd");

    Eigen::Tensor<double, 4> result(no, nv, no, nv);
    result.setZero();
    std::vector<len_type> len_out = { (len_type)no, (len_type)nv, (len_type)no, (len_type)nv };
    std::vector<stride_type> str_out = { 1, (stride_type)no, (stride_type)(no*nv), (stride_type)(no*nv*no) };
    varray_view<double> t_out_view(len_out, result.data(), str_out);
    tblis::mult<double>(1.0, t_T3_view, "egfd", t_Cv, "dh", 0.0, t_out_view, "egfh");

    return result;
}
BlockedTensor4D ERITransformer::transform_oovv_blocked(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& Co, const Eigen::MatrixXd& Cv,
    const std::vector<IrrepSpace>& occ_spaces,
    const std::vector<IrrepSpace>& virt_spaces,
    int nbf)
{
    BlockedTensor4D result;
    int no = Co.cols();
    int nv = Cv.cols();

    Eigen::Tensor<double, 4> dense_oovv = optimal_oovv_kernel(eri_ao, Co, Cv, nbf, no, nv);

    for (const auto& o1 : occ_spaces) {
        if (o1.size == 0) continue;
        for (const auto& v1 : virt_spaces) {
            if (v1.size == 0) continue;
            for (const auto& o2 : occ_spaces) {
                if (o2.size == 0) continue;
                for (const auto& v2 : virt_spaces) {
                    if (v2.size == 0) continue;

                    if ((o1.id ^ v1.id ^ o2.id ^ v2.id) == 0) {
                        Eigen::Tensor<double, 4> block(o1.size, v1.size, o2.size, v2.size);
                        block.setZero();

                        #pragma omp parallel for collapse(2) schedule(static)
                        for (int i = 0; i < o1.size; ++i) {
                            for (int a = 0; a < v1.size; ++a) {
                                for (int j = 0; j < o2.size; ++j) {
                                    for (int b = 0; b < v2.size; ++b) {

                                        block(i, a, j, b) = dense_oovv(o1.offset + i, v1.offset + a, o2.offset + j, v2.offset + b);
                                    }
                                }
                            }
                        }
                        result.blocks[pack_irreps(o1.id, v1.id, o2.id, v2.id)] = std::move(block);
                    }
                }
            }
        }
    }

    size_t total_elements = 0;
    for (const auto& item : result.blocks) {
        total_elements += item.second.size();
    }

    double mb_used = (total_elements * 8.0) / (1024.0 * 1024.0);
    double mb_dense = (no * nv * no * nv * 8.0) / (1024.0 * 1024.0);
    double saved_percent = 100.0 * (1.0 - (mb_used / mb_dense));

    std::cout << "  [TBLIS] Dense-to-Slice OOVV Selesai! (" << result.blocks.size() << " blok non-zero)\n";
    std::cout << "  [Memory] OOVV Blocked RAM : " << std::fixed << std::setprecision(2)
              << mb_used << " MB (Hemat " << saved_percent << "% vs Dense " << mb_dense << " MB)\n";

    return result;
}
BlockedTensor4D ERITransformer::transform_ovvv_blocked(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& Co, const Eigen::MatrixXd& Cv,
    const std::vector<IrrepSpace>& occ_spaces,
    const std::vector<IrrepSpace>& virt_spaces,
    int nbf)
{
    BlockedTensor4D result;
    int no = Co.cols();
    int nv = Cv.cols();

    Eigen::Tensor<double, 4> dense_ovvv = smart_transform_kernel(eri_ao, Co, Cv, Cv, Cv, nbf, no, nv, nv, nv);

    for (const auto& o1 : occ_spaces) {
        if (o1.size == 0) continue;
        for (const auto& v1 : virt_spaces) {
            if (v1.size == 0) continue;
            for (const auto& v2 : virt_spaces) {
                if (v2.size == 0) continue;
                for (const auto& v3 : virt_spaces) {
                    if (v3.size == 0) continue;

                    if ((o1.id ^ v1.id ^ v2.id ^ v3.id) == 0) {
                        Eigen::Tensor<double, 4> block(o1.size, v1.size, v2.size, v3.size);
                        block.setZero();
                        for (int i = 0; i < o1.size; ++i) {
                            for (int a = 0; a < v1.size; ++a) {
                                for (int b = 0; b < v2.size; ++b) {
                                    for (int c = 0; c < v3.size; ++c) {
                                        block(i, a, b, c) = dense_ovvv(o1.offset + i, v1.offset + a, v2.offset + b, v3.offset + c);
                                    }
                                }
                            }
                        }
                        result.blocks[pack_irreps(o1.id, v1.id, v2.id, v3.id)] = std::move(block);
                    }
                }
            }
        }
    }
    return result;
}

BlockedTensor4D ERITransformer::transform_ooov_blocked(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& Co, const Eigen::MatrixXd& Cv,
    const std::vector<IrrepSpace>& occ_spaces,
    const std::vector<IrrepSpace>& virt_spaces,
    int nbf)
{
    BlockedTensor4D result;
    int no = Co.cols();
    int nv = Cv.cols();

    Eigen::Tensor<double, 4> dense_ooov = smart_transform_kernel(eri_ao, Co, Co, Co, Cv, nbf, no, no, no, nv);

    for (const auto& o1 : occ_spaces) {
        if (o1.size == 0) continue;
        for (const auto& o2 : occ_spaces) {
            if (o2.size == 0) continue;
            for (const auto& o3 : occ_spaces) {
                if (o3.size == 0) continue;
                for (const auto& v1 : virt_spaces) {
                    if (v1.size == 0) continue;

                    if ((o1.id ^ o2.id ^ o3.id ^ v1.id) == 0) {
                        Eigen::Tensor<double, 4> block(o1.size, o2.size, o3.size, v1.size);
                        block.setZero();
                        for (int i = 0; i < o1.size; ++i) {
                            for (int j = 0; j < o2.size; ++j) {
                                for (int k = 0; k < o3.size; ++k) {
                                    for (int a = 0; a < v1.size; ++a) {
                                        block(i, j, k, a) = dense_ooov(o1.offset + i, o2.offset + j, o3.offset + k, v1.offset + a);
                                    }
                                }
                            }
                        }
                        result.blocks[pack_irreps(o1.id, o2.id, o3.id, v1.id)] = std::move(block);
                    }
                }
            }
        }
    }
    return result;
}

Eigen::Tensor<double, 4> ERITransformer::transform_oovv(
    const Eigen::Tensor<double, 4>& eri, const Eigen::MatrixXd& Co, const Eigen::MatrixXd& Cv,
    int nbf, int no, int nv, bool use_disk, const std::string& hdf5_filename) {

    Eigen::Tensor<double, 4> result = smart_transform_kernel(eri, Co, Cv, Co, Cv, nbf, no, nv, no, nv);

    if (use_disk) {
        utils::HDF5TensorIO io(hdf5_filename, utils::HDF5TensorIO::Mode::WRITE_TRUNCATE);

        std::array<long, 4> dims = {no, nv, no, nv};
        std::array<long, 4> chunk_dims = {1, nv, no, nv};

        std::string dataset_name = "oovv";
        io.create_dataset_4d(dataset_name, dims, chunk_dims);
        io.write_tensor_4d(dataset_name, result);

        std::cout << "[HDF5] Transformed OOVV tensor saved to " << hdf5_filename << "\n";

        return Eigen::Tensor<double, 4>();
    }

    return result;
}

Eigen::Tensor<double, 4> ERITransformer::transform_oovv_quarter(
    const Eigen::Tensor<double, 4>& eri, const Eigen::MatrixXd& Co, const Eigen::MatrixXd& Cv, int nbf, int no, int nv) {
    return smart_transform_kernel(eri, Co, Cv, Co, Cv, nbf, no, nv, no, nv);
}

Eigen::Tensor<double, 4> ERITransformer::transform_oovv_mixed(
    const Eigen::Tensor<double, 4>& eri, const Eigen::MatrixXd& Ca, const Eigen::MatrixXd& Cb, const Eigen::MatrixXd& Va, const Eigen::MatrixXd& Vb, int nbf, int oa, int ob, int va, int vb) {
    return smart_transform_kernel(eri, Ca, Va, Cb, Vb, nbf, oa, va, ob, vb);
}

Eigen::Tensor<double, 4> ERITransformer::transform_oo_vv(
    const Eigen::Tensor<double, 4>& eri, const Eigen::MatrixXd& Co, const Eigen::MatrixXd& Cv,
    int nbf, int no, int nv, bool use_disk, const std::string& hdf5_filename) {

    Eigen::Tensor<double, 4> result = smart_transform_kernel(eri, Co, Co, Cv, Cv, nbf, no, no, nv, nv);

    if (use_disk) {
        utils::HDF5TensorIO io(hdf5_filename, utils::HDF5TensorIO::Mode::WRITE_TRUNCATE);

        std::array<long, 4> dims = {no, no, nv, nv};
        std::array<long, 4> chunk_dims = {1, no, nv, nv};

        std::string dataset_name = "oo_vv";
        io.create_dataset_4d(dataset_name, dims, chunk_dims);
        io.write_tensor_4d(dataset_name, result);

        std::cout << "[HDF5] Transformed OO-VV tensor saved to " << hdf5_filename << "\n";
        return Eigen::Tensor<double, 4>();
    }
    return result;
}

Eigen::Tensor<double, 4> ERITransformer::transform_oo_vv_mixed(
    const Eigen::Tensor<double, 4>& eri, const Eigen::MatrixXd& C1, const Eigen::MatrixXd& C2, const Eigen::MatrixXd& V1, const Eigen::MatrixXd& V2, int nbf, int o1, int o2, int v1, int v2) {
    return smart_transform_kernel(eri, C1, C2, V1, V2, nbf, o1, o2, v1, v2);
}

Eigen::Tensor<double, 4> ERITransformer::transform_vvov(
    const Eigen::Tensor<double, 4>& eri, const Eigen::MatrixXd& Co, const Eigen::MatrixXd& Cv, int nbf, int no, int nv) {
    return smart_transform_kernel(eri, Cv, Cv, Co, Cv, nbf, nv, nv, no, nv);
}

Eigen::Tensor<double, 4> ERITransformer::transform_oooo(
    const Eigen::Tensor<double, 4>& eri, const Eigen::MatrixXd& C, int nbf, int n) {
     return smart_transform_kernel(eri, C, C, C, C, nbf, n, n, n, n);
}

Eigen::Tensor<double, 4> ERITransformer::transform_oooo_mixed(
    const Eigen::Tensor<double, 4>& eri, const Eigen::MatrixXd& Ca, const Eigen::MatrixXd& Cb, int nbf, int na, int nb) {
    return smart_transform_kernel(eri, Ca, Ca, Cb, Cb, nbf, na, na, nb, nb);
}

Eigen::Tensor<double, 4> ERITransformer::transform_vvvv(
    const Eigen::Tensor<double, 4>& eri, const Eigen::MatrixXd& C, int nbf, int n,
    bool use_disk, const std::string& hdf5_filename) {

    Eigen::Tensor<double, 4> result = smart_transform_kernel(eri, C, C, C, C, nbf, n, n, n, n);

    if (use_disk) {
        utils::HDF5TensorIO io(hdf5_filename, utils::HDF5TensorIO::Mode::WRITE_TRUNCATE);

        std::array<long, 4> dims = {n, n, n, n};

        std::array<long, 4> chunk_dims = {1, n, n, n};

        std::string dataset_name = "vvvv";
        io.create_dataset_4d(dataset_name, dims, chunk_dims);
        io.write_tensor_4d(dataset_name, result);

        std::cout << "[HDF5] Transformed VVVV tensor saved to " << hdf5_filename << "\n";

        return Eigen::Tensor<double, 4>();
    }

    return result;
}

Eigen::Tensor<double, 4> ERITransformer::transform_vvvv_mixed(
    const Eigen::Tensor<double, 4>& eri, const Eigen::MatrixXd& Va, const Eigen::MatrixXd& Vb,
    int nbf, int na, int nb, bool use_disk, const std::string& hdf5_filename) {

    Eigen::Tensor<double, 4> result = smart_transform_kernel(eri, Va, Va, Vb, Vb, nbf, na, na, nb, nb);

    if (use_disk) {
        utils::HDF5TensorIO io(hdf5_filename, utils::HDF5TensorIO::Mode::WRITE_TRUNCATE);

        std::array<long, 4> dims = {na, na, nb, nb};
        std::array<long, 4> chunk_dims = {1, na, nb, nb};

        std::string dataset_name = "vvvv_mixed";
        io.create_dataset_4d(dataset_name, dims, chunk_dims);
        io.write_tensor_4d(dataset_name, result);

        std::cout << "[HDF5] Transformed VVVV_Mixed tensor saved to " << hdf5_filename << "\n";
        return Eigen::Tensor<double, 4>();
    }
    return result;
}
Eigen::Tensor<double, 4> ERITransformer::transform_ovov(
    const Eigen::Tensor<double, 4>& eri, const Eigen::MatrixXd& Co, const Eigen::MatrixXd& Cv,
    int nbf, int no, int nv, bool use_disk, const std::string& hdf5_filename) {

    Eigen::Tensor<double, 4> result = smart_transform_kernel(eri, Co, Cv, Co, Cv, nbf, no, nv, no, nv);

    if (use_disk) {
        utils::HDF5TensorIO io(hdf5_filename, utils::HDF5TensorIO::Mode::WRITE_TRUNCATE);

        std::array<long, 4> dims = {no, nv, no, nv};
        std::array<long, 4> chunk_dims = {1, nv, no, nv};

        std::string dataset_name = "ovov";
        io.create_dataset_4d(dataset_name, dims, chunk_dims);
        io.write_tensor_4d(dataset_name, result);

        std::cout << "[HDF5] Transformed OVOV tensor saved to " << hdf5_filename << "\n";
        return Eigen::Tensor<double, 4>();
    }
    return result;
}

Eigen::Tensor<double, 4> ERITransformer::transform_ovov_mixed(
    const Eigen::Tensor<double, 4>& eri, const Eigen::MatrixXd& Ca, const Eigen::MatrixXd& Vb, int nbf, int oa, int vb) {
    return smart_transform_kernel(eri, Ca, Vb, Ca, Vb, nbf, oa, vb, oa, vb);
}
Eigen::Tensor<double, 4> ERITransformer::transform_vvvo(
    const Eigen::Tensor<double, 4>& eri, const Eigen::MatrixXd& Co, const Eigen::MatrixXd& Cv,
    int nbf, int no, int nv, bool use_disk, const std::string& hdf5_filename) {

    Eigen::Tensor<double, 4> result = smart_transform_kernel(eri, Cv, Cv, Cv, Co, nbf, nv, nv, nv, no);

    if (use_disk) {
        utils::HDF5TensorIO io(hdf5_filename, utils::HDF5TensorIO::Mode::WRITE_TRUNCATE);

        std::array<long, 4> dims = {nv, nv, nv, no};
        std::array<long, 4> chunk_dims = {1, nv, nv, no};

        std::string dataset_name = "vvvo";
        io.create_dataset_4d(dataset_name, dims, chunk_dims);
        io.write_tensor_4d(dataset_name, result);

        std::cout << "[HDF5] Transformed VVVO tensor saved to " << hdf5_filename << "\n";
        return Eigen::Tensor<double, 4>();
    }
    return result;
}

Eigen::Tensor<double, 4> ERITransformer::transform_vvvo_mixed(
    const Eigen::Tensor<double, 4>& eri, const Eigen::MatrixXd& K, const Eigen::MatrixXd& AC, const Eigen::MatrixXd& B, int nbf, int nK, int nAC, int nB) {
    return smart_transform_kernel(eri, AC, B, AC, K, nbf, nAC, nB, nAC, nK);
}

Eigen::Tensor<double, 4> ERITransformer::transform_ooov(
    const Eigen::Tensor<double, 4>& eri, const Eigen::MatrixXd& Co, const Eigen::MatrixXd& Cv, int nbf, int no, int nv) {
    return smart_transform_kernel(eri, Co, Co, Co, Cv, nbf, no, no, no, nv);
}
Eigen::Tensor<double, 4> ERITransformer::transform_ovvv(
    const Eigen::Tensor<double, 4>& eri, const Eigen::MatrixXd& Co, const Eigen::MatrixXd& Cv,
    int nbf, int no, int nv, bool use_disk, const std::string& hdf5_filename) {

    Eigen::Tensor<double, 4> result = smart_transform_kernel(eri, Co, Cv, Cv, Cv, nbf, no, nv, nv, nv);

    if (use_disk) {
        utils::HDF5TensorIO io(hdf5_filename, utils::HDF5TensorIO::Mode::WRITE_TRUNCATE);

        std::array<long, 4> dims = {no, nv, nv, nv};
        std::array<long, 4> chunk_dims = {1, nv, nv, nv};

        std::string dataset_name = "ovvv";
        io.create_dataset_4d(dataset_name, dims, chunk_dims);
        io.write_tensor_4d(dataset_name, result);

        std::cout << "[HDF5] Transformed OVVV tensor saved to " << hdf5_filename << "\n";
        return Eigen::Tensor<double, 4>();
    }
    return result;
}

Eigen::Tensor<double, 4> ERITransformer::transform_custom(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C1, const Eigen::MatrixXd& C2,
    const Eigen::MatrixXd& C3, const Eigen::MatrixXd& C4,
    int nbf, int n1, int n2, int n3, int n4) {
    return smart_transform_kernel(eri_ao, C1, C2, C3, C4, nbf, n1, n2, n3, n4);
}

Eigen::Tensor<double, 4> ERITransformer::transform_oovv_parallel(
    const Eigen::Tensor<double, 4>& e, const Eigen::MatrixXd& o, const Eigen::MatrixXd& v, int n, int no, int nv, int) {
    return transform_oovv(e, o, v, n, no, nv);
}

Eigen::Tensor<double, 4> ERITransformer::transform_vvvv_parallel(
    const Eigen::Tensor<double, 4>& e, const Eigen::MatrixXd& v, int n, int nv, int) {
    return transform_vvvv(e, v, n, nv);
}

Eigen::Tensor<double, 4> ERITransformer::transform_oooo_parallel(
    const Eigen::Tensor<double, 4>& e, const Eigen::MatrixXd& o, int n, int no, int) {
    return transform_oooo(e, o, n, no);
}
Eigen::Tensor<double, 4> ERITransformer::get_mo_tensor(
    bool use_df, int n_aux,
    const Eigen::MatrixXd& C1, const Eigen::MatrixXd& C2,
    const Eigen::MatrixXd& C3, const Eigen::MatrixXd& C4,
    std::shared_ptr<IntegralEngine> ints)
{

    int nbf = C1.rows();
    int dim1 = C1.cols();
    int dim2 = C2.cols();
    int dim3 = C3.cols();
    int dim4 = C4.cols();

    if (!use_df) {
        if (!ints) throw std::runtime_error("Exact integrals engine missing!");
        return transform_oovv_mixed(ints->compute_eri(), C1, C2, C3, C4, nbf, dim1, dim2, dim3, dim4);
    }

    Eigen::Tensor<double, 4> V_mo(dim1, dim2, dim3, dim4);
    V_mo.setZero();
    if (n_aux <= 0) return V_mo;

    mshqc::utils::HDF5TensorIO io("df_tensor.h5", mshqc::utils::HDF5TensorIO::Mode::READ_ONLY);
    int chunk_size = std::min(128, n_aux);

    Eigen::MatrixXd L_ao(nbf * nbf, chunk_size);
    std::vector<double> L_left_buf(dim1 * dim2 * chunk_size, 0.0);
    std::vector<double> L_right_buf(dim3 * dim4 * chunk_size, 0.0);

    using tblis::len_type; using tblis::stride_type; using tblis::varray_view;

    for (int P_start = 0; P_start < n_aux; P_start += chunk_size) {
        int P_end = std::min(n_aux, P_start + chunk_size);
        int P_size = P_end - P_start;

        L_ao.resize(nbf * nbf, P_size);
        io.read_slice_4d("df_tensor", {P_start, 0, 0, 0}, {P_size, (long)nbf, (long)nbf, 1}, L_ao.data());

        std::vector<len_type> len_L_ao = { (len_type)nbf, (len_type)nbf, (len_type)P_size };
        std::vector<stride_type> str_L_ao = { 1, (stride_type)nbf, (stride_type)(nbf * nbf) };
        varray_view<double> t_L_ao(len_L_ao, L_ao.data(), str_L_ao);

        std::vector<len_type> len_C1 = { (len_type)nbf, (len_type)dim1 };
        std::vector<stride_type> str_C1 = { 1, (stride_type)nbf };
        varray_view<double> t_C1(len_C1, const_cast<double*>(C1.data()), str_C1);

        std::vector<len_type> len_C2 = { (len_type)nbf, (len_type)dim2 };
        std::vector<stride_type> str_C2 = { 1, (stride_type)nbf };
        varray_view<double> t_C2(len_C2, const_cast<double*>(C2.data()), str_C2);

        std::vector<double> tmp_half(nbf * dim2 * P_size, 0.0);
        varray_view<double> t_tmp_half({(len_type)nbf, (len_type)dim2, (len_type)P_size}, tmp_half.data(), {1, (stride_type)nbf, (stride_type)(nbf*dim2)});
        varray_view<double> t_L_left({(len_type)dim1, (len_type)dim2, (len_type)P_size}, L_left_buf.data(), {1, (stride_type)dim1, (stride_type)(dim1*dim2)});

        tblis::mult<double>(1.0, t_L_ao, "mnP", t_C2, "nq", 0.0, t_tmp_half, "mqP");
        tblis::mult<double>(1.0, t_tmp_half, "mqP", t_C1, "mp", 0.0, t_L_left, "pqP");

        varray_view<double> t_C3({(len_type)nbf, (len_type)dim3}, const_cast<double*>(C3.data()), {1, (stride_type)nbf});
        varray_view<double> t_C4({(len_type)nbf, (len_type)dim4}, const_cast<double*>(C4.data()), {1, (stride_type)nbf});

        std::vector<double> tmp_half2(nbf * dim4 * P_size, 0.0);
        varray_view<double> t_tmp_half2({(len_type)nbf, (len_type)dim4, (len_type)P_size}, tmp_half2.data(), {1, (stride_type)nbf, (stride_type)(nbf*dim4)});
        varray_view<double> t_L_right({(len_type)dim3, (len_type)dim4, (len_type)P_size}, L_right_buf.data(), {1, (stride_type)dim3, (stride_type)(dim3*dim4)});

        tblis::mult<double>(1.0, t_L_ao, "mnP", t_C4, "ns", 0.0, t_tmp_half2, "msP");
        tblis::mult<double>(1.0, t_tmp_half2, "msP", t_C3, "mr", 0.0, t_L_right, "rsP");

        varray_view<double> t_V_mo({(len_type)dim1, (len_type)dim2, (len_type)dim3, (len_type)dim4}, V_mo.data(),
            {1, (stride_type)dim1, (stride_type)(dim1*dim2), (stride_type)(dim1*dim2*dim3)});

        tblis::mult<double>(1.0, t_L_left, "pqP", t_L_right, "rsP", 1.0, t_V_mo, "pqrs");
    }

    return V_mo;
}

void ERITransformer::print_transform_info(const char*, int, int, int, int, double) {}
void ERITransformer::antisymmetrize_vvvv(Eigen::Tensor<double, 4>&, int) {}
void ERITransformer::antisymmetrize_oooo(Eigen::Tensor<double, 4>&, int) {}
void ERITransformer::antisymmetrize_oovv(Eigen::Tensor<double, 4>&, int, int) {}
void ERITransformer::antisymmetrize_ovov(Eigen::Tensor<double, 4>&, int, int) {}

}
}
