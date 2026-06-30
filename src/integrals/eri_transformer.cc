/**
 * @file src/integrals/eri_transformer.cc
 * @brief TBLIS-Powered ERI Transformation (Pure C++ API)
 * @details 
 * Menggunakan pustaka TBLIS versi C++ (tblis::tensor, tblis::mult) 
 * untuk melakukan kontraksi tensor multidimensi dengan aman dan super cepat.
 */

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

// ============================================================================
// SMART TRANSFORM KERNEL (Powered by TBLIS Native C++ API)
// ============================================================================
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

    // --- STEP 1: mu -> i ---
    Eigen::Tensor<double, 4> T1(n1, nbf, nbf, nbf); 
    // HAPUS .setZero() karena tblis::mult dengan beta=0.0 akan menimpanya!
    std::vector<len_type> len_T1 = { (len_type)n1, (len_type)nbf, (len_type)nbf, (len_type)nbf };
    std::vector<stride_type> str_T1 = { 1, (stride_type)n1, (stride_type)(n1*nbf), (stride_type)(n1*nbf*nbf) };
    varray_view<double> t_T1_view(len_T1, T1.data(), str_T1);
    tblis::mult<double>(1.0, t_eri, "abcd", t_C1, "ae", 0.0, t_T1_view, "ebcd");

    // --- STEP 2: nu -> a ---
    Eigen::Tensor<double, 4> T2(n1, n2, nbf, nbf); 
    std::vector<len_type> len_T2 = { (len_type)n1, (len_type)n2, (len_type)nbf, (len_type)nbf };
    std::vector<stride_type> str_T2 = { 1, (stride_type)n1, (stride_type)(n1*n2), (stride_type)(n1*n2*nbf) };
    varray_view<double> t_T2_view(len_T2, T2.data(), str_T2);
    tblis::mult<double>(1.0, t_T1_view, "ebcd", t_C2, "bf", 0.0, t_T2_view, "efcd");

    // --- STEP 3: lam -> j ---
    Eigen::Tensor<double, 4> T3(n1, n2, n3, nbf); 
    std::vector<len_type> len_T3 = { (len_type)n1, (len_type)n2, (len_type)n3, (len_type)nbf };
    std::vector<stride_type> str_T3 = { 1, (stride_type)n1, (stride_type)(n1*n2), (stride_type)(n1*n2*n3) };
    varray_view<double> t_T3_view(len_T3, T3.data(), str_T3);
    tblis::mult<double>(1.0, t_T2_view, "efcd", t_C3, "cg", 0.0, t_T3_view, "efgd");

    // --- STEP 4: sig -> b ---
    Eigen::Tensor<double, 4> result(n1, n2, n3, n4); 
    std::vector<len_type> len_out = { (len_type)n1, (len_type)n2, (len_type)n3, (len_type)n4 };
    std::vector<stride_type> str_out = { 1, (stride_type)n1, (stride_type)(n1*n2), (stride_type)(n1*n2*n3) };
    varray_view<double> t_out_view(len_out, result.data(), str_out);
    tblis::mult<double>(1.0, t_T3_view, "efgd", t_C4, "dh", 0.0, t_out_view, "efgh");

    return result;
}
// ============================================================================
// THE "O-O-V-V" OPTIMAL KERNEL (Trik Rahasia HPC Kimia Kuantum)
// ============================================================================
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

    // --- STEP 1: mu -> i (Menggunakan Occupied Co) ---
    // (a,b,c,d) * (a,e) -> (e,b,c,d)
    Eigen::Tensor<double, 4> T1(no, nbf, nbf, nbf); 
    std::vector<len_type> len_T1 = { (len_type)no, (len_type)nbf, (len_type)nbf, (len_type)nbf };
    std::vector<stride_type> str_T1 = { 1, (stride_type)no, (stride_type)(no*nbf), (stride_type)(no*nbf*nbf) };
    varray_view<double> t_T1_view(len_T1, T1.data(), str_T1);
    tblis::mult<double>(1.0, t_eri, "abcd", t_Co, "ae", 0.0, t_T1_view, "ebcd");

    // --- STEP 2: lam -> j (Menggunakan Occupied Co) ---
    // KEAJAIBAN TBLIS: Kita lewati indeks ke-2 (b) dan langsung sikat indeks ke-3 (c)!
    // (e,b,c,d) * (c,f) -> (e,b,f,d)
    Eigen::Tensor<double, 4> T2(no, nbf, no, nbf); 
    std::vector<len_type> len_T2 = { (len_type)no, (len_type)nbf, (len_type)no, (len_type)nbf };
    std::vector<stride_type> str_T2 = { 1, (stride_type)no, (stride_type)(no*nbf), (stride_type)(no*nbf*no) };
    varray_view<double> t_T2_view(len_T2, T2.data(), str_T2);
    tblis::mult<double>(1.0, t_T1_view, "ebcd", t_Co, "cf", 0.0, t_T2_view, "ebfd");

    // --- STEP 3: nu -> a (Menggunakan Virtual Cv) ---
    // (e,b,f,d) * (b,g) -> (e,g,f,d)
    Eigen::Tensor<double, 4> T3(no, nv, no, nbf); 
    std::vector<len_type> len_T3 = { (len_type)no, (len_type)nv, (len_type)no, (len_type)nbf };
    std::vector<stride_type> str_T3 = { 1, (stride_type)no, (stride_type)(no*nv), (stride_type)(no*nv*no) };
    varray_view<double> t_T3_view(len_T3, T3.data(), str_T3);
    tblis::mult<double>(1.0, t_T2_view, "ebfd", t_Cv, "bg", 0.0, t_T3_view, "egfd");

    // --- STEP 4: sig -> b (Menggunakan Virtual Cv) ---
    // (e,g,f,d) * (d,h) -> (e,g,f,h)
    Eigen::Tensor<double, 4> result(no, nv, no, nv); 
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

    // 1. PANGGIL OPTIMAL KERNEL O-O-V-V (Kecepatan 11x Lipat)
    Eigen::Tensor<double, 4> dense_oovv = optimal_oovv_kernel(eri_ao, Co, Cv, nbf, no, nv);

    // 2. SLICING MEMORY MULTI-THREADED
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
                        
                        #pragma omp parallel for collapse(2) schedule(static)
                        for (int i = 0; i < o1.size; ++i) {
                            for (int a = 0; a < v1.size; ++a) {
                                for (int j = 0; j < o2.size; ++j) {
                                    for (int b = 0; b < v2.size; ++b) {
                                        // Karena output Kernel sudah berindeks (i, a, j, b), slicingnya tetap natural!
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
    
    // =========================================================================
    // TAHAP 6: MEMORY PROFILING (Untuk membuktikan penghematan RAM ke User)
    // =========================================================================
    size_t total_elements = 0;
    for (const auto& item : result.blocks) {
        total_elements += item.second.size();
    }
    
    // Asumsi 1 double = 8 bytes
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

    // 1. DENSE TRANSFORMATION (1 Kali Saja)
    Eigen::Tensor<double, 4> dense_ovvv = smart_transform_kernel(eri_ao, Co, Cv, Cv, Cv, nbf, no, nv, nv, nv);

    // 2. SLICING MEMORY
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

    // 1. DENSE TRANSFORMATION (1 Kali Saja)
    Eigen::Tensor<double, 4> dense_ooov = smart_transform_kernel(eri_ao, Co, Co, Co, Cv, nbf, no, no, no, nv);

    // 2. SLICING MEMORY
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


// WRAPPER IMPLEMENTATIONS
// ============================================================================

// ============================================================================
// WRAPPER IMPLEMENTATIONS
// ============================================================================

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
        
        // Perhatikan urutannya: Occupied, Occupied, Virtual, Virtual
        std::array<long, 4> dims = {no, no, nv, nv};
        std::array<long, 4> chunk_dims = {1, no, nv, nv}; // Chunk per 1 indeks occupied
        
        std::string dataset_name = "oo_vv";
        io.create_dataset_4d(dataset_name, dims, chunk_dims);
        io.write_tensor_4d(dataset_name, result);
        
        std::cout << "[HDF5] Transformed OO-VV tensor saved to " << hdf5_filename << "\n";
        return Eigen::Tensor<double, 4>(); // Bebaskan memori
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
    
    // Panggil kernel transformasi
    Eigen::Tensor<double, 4> result = smart_transform_kernel(eri, C, C, C, C, nbf, n, n, n, n);

    // Tambahkan logika penyimpanan HDF5
    if (use_disk) {
        utils::HDF5TensorIO io(hdf5_filename, utils::HDF5TensorIO::Mode::WRITE_TRUNCATE);
        
        // Sesuaikan dimensi dengan output (n, n, n, n)
        std::array<long, 4> dims = {n, n, n, n};
        // Atur chunking, misalnya per 1 elemen di dimensi pertama
        std::array<long, 4> chunk_dims = {1, n, n, n}; 
        
        std::string dataset_name = "vvvv";
        io.create_dataset_4d(dataset_name, dims, chunk_dims);
        io.write_tensor_4d(dataset_name, result);
        
        std::cout << "[HDF5] Transformed VVVV tensor saved to " << hdf5_filename << "\n";
        
        // Kembalikan tensor kosong untuk membebaskan RAM
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
        
        // Perhatikan urutannya: Occupied, Virtual, Occupied, Virtual
        std::array<long, 4> dims = {no, nv, no, nv};
        std::array<long, 4> chunk_dims = {1, nv, no, nv}; // Chunk per 1 indeks occupied
        
        std::string dataset_name = "ovov";
        io.create_dataset_4d(dataset_name, dims, chunk_dims);
        io.write_tensor_4d(dataset_name, result);
        
        std::cout << "[HDF5] Transformed OVOV tensor saved to " << hdf5_filename << "\n";
        return Eigen::Tensor<double, 4>(); // Bebaskan memori
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
        std::array<long, 4> chunk_dims = {1, nv, nv, no}; // Chunking per 1 virtual index di awal
        
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
        std::array<long, 4> chunk_dims = {1, nv, nv, nv}; // Chunking di dimensi pertama (occupied)
        
        std::string dataset_name = "ovvv";
        io.create_dataset_4d(dataset_name, dims, chunk_dims);
        io.write_tensor_4d(dataset_name, result);
        
        std::cout << "[HDF5] Transformed OVVV tensor saved to " << hdf5_filename << "\n";
        return Eigen::Tensor<double, 4>(); // Kembalikan tensor kosong
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

// Parallel Wrappers 
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

// Stubs for specific manipulations
void ERITransformer::print_transform_info(const char*, int, int, int, int, double) {}
void ERITransformer::antisymmetrize_vvvv(Eigen::Tensor<double, 4>&, int) {}
void ERITransformer::antisymmetrize_oooo(Eigen::Tensor<double, 4>&, int) {}
void ERITransformer::antisymmetrize_oovv(Eigen::Tensor<double, 4>&, int, int) {}
void ERITransformer::antisymmetrize_ovov(Eigen::Tensor<double, 4>&, int, int) {}

} // namespace integrals
} // namespace mshqc
