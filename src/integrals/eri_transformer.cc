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
#include <map>
#include <utility>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <tblis/tblis.h>

namespace mshqc {
namespace integrals {

// ============================================================================
// PURE EIGEN SMART TRANSFORM KERNEL (100% TBLIS-FREE)
// ============================================================================
static Eigen::Tensor<double, 4> smart_transform_kernel(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& C1, const Eigen::MatrixXd& C2, 
    const Eigen::MatrixXd& C3, const Eigen::MatrixXd& C4, 
    int nbf, int n1, int n2, int n3, int n4 
) {
    Eigen::Tensor<double, 4> T1(n1, nbf, nbf, nbf); 
    Eigen::Map<const Eigen::MatrixXd> eri_mat(eri_ao.data(), nbf, nbf * nbf * nbf);
    Eigen::Map<Eigen::MatrixXd> T1_mat(T1.data(), n1, nbf * nbf * nbf);
    T1_mat.noalias() = C1.transpose() * eri_mat; 

    Eigen::Tensor<double, 4> T2(n1, n2, nbf, nbf); 
    #pragma omp parallel for collapse(2) schedule(static)
    for(int lam = 0; lam < nbf; ++lam) {
        for(int sig = 0; sig < nbf; ++sig) {
            Eigen::Map<const Eigen::MatrixXd> T1_slice(T1.data() + (lam + sig * nbf) * n1 * nbf, n1, nbf);
            Eigen::Map<Eigen::MatrixXd> T2_slice(T2.data() + (lam + sig * nbf) * n1 * n2, n1, n2);
            T2_slice.noalias() = T1_slice * C2; 
        }
    }

    Eigen::Tensor<double, 4> T3(n1, n2, n3, nbf); 
    #pragma omp parallel for schedule(static)
    for(int sig = 0; sig < nbf; ++sig) {
        Eigen::Map<const Eigen::MatrixXd> T2_slice(T2.data() + sig * n1 * n2 * nbf, n1 * n2, nbf);
        Eigen::Map<Eigen::MatrixXd> T3_slice(T3.data() + sig * n1 * n2 * n3, n1 * n2, n3);
        T3_slice.noalias() = T2_slice * C3; 
    }

    Eigen::Tensor<double, 4> result(n1, n2, n3, n4); 
    Eigen::Map<const Eigen::MatrixXd> T3_mat(T3.data(), n1 * n2 * n3, nbf);
    Eigen::Map<Eigen::MatrixXd> res_mat(result.data(), n1 * n2 * n3, n4);
    res_mat.noalias() = T3_mat * C4; 

    return result;
}
// ============================================================================
// PURE EIGEN OPTIMAL O-O-V-V KERNEL (100% TBLIS-FREE)
// ============================================================================
static Eigen::Tensor<double, 4> optimal_oovv_kernel(
    const Eigen::Tensor<double, 4>& eri_ao,
    const Eigen::MatrixXd& Co, const Eigen::MatrixXd& Cv, 
    int nbf, int no, int nv 
) {
    Eigen::Tensor<double, 4> T1(no, nbf, nbf, nbf); 
    Eigen::Map<const Eigen::MatrixXd> eri_mat(eri_ao.data(), nbf, nbf * nbf * nbf);
    Eigen::Map<Eigen::MatrixXd> T1_mat(T1.data(), no, nbf * nbf * nbf);
    T1_mat.noalias() = Co.transpose() * eri_mat;

    Eigen::Tensor<double, 4> T2(no, nbf, no, nbf); 
    #pragma omp parallel for schedule(static)
    for (int sig = 0; sig < nbf; ++sig) {
        Eigen::Map<const Eigen::MatrixXd> T1_slice(T1.data() + sig * no * nbf * nbf, no * nbf, nbf);
        Eigen::Map<Eigen::MatrixXd> T2_slice(T2.data() + sig * no * nbf * no, no * nbf, no);
        T2_slice.noalias() = T1_slice * Co;
    }

    Eigen::Tensor<double, 4> T3(no, nv, no, nbf); 
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < no; ++j) {
        for (int sig = 0; sig < nbf; ++sig) {
            Eigen::Map<const Eigen::MatrixXd> T2_slice(T2.data() + (j + sig * no) * no * nbf, no, nbf);
            Eigen::Map<Eigen::MatrixXd> T3_slice(T3.data() + (j + sig * no) * no * nv, no, nv);
            T3_slice.noalias() = T2_slice * Cv;
        }
    }

    Eigen::Tensor<double, 4> result(no, nv, no, nv); 
    Eigen::Map<const Eigen::MatrixXd> T3_mat(T3.data(), no * nv * no, nbf);
    Eigen::Map<Eigen::MatrixXd> res_mat(result.data(), no * nv * no, nv);
    res_mat.noalias() = T3_mat * Cv;

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
// ============================================================================
// BLOCK-SPARSE TRANSFORMATIONS WITH JIT MICRO-GEMM
// ============================================================================
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

    // 1. GLOBAL HALF-TRANSFORMATION (Pure Eigen DGEMM - Cache Optimized)
    Eigen::Tensor<double, 4> T1(no, nbf, nbf, nbf);
    Eigen::Map<const Eigen::MatrixXd> eri_mat(eri_ao.data(), nbf, nbf * nbf * nbf);
    Eigen::Map<Eigen::MatrixXd> T1_mat(T1.data(), no, nbf * nbf * nbf);
    T1_mat.noalias() = Co.transpose() * eri_mat;

    Eigen::Tensor<double, 4> T2(no, nv, nbf, nbf);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int lam = 0; lam < nbf; ++lam) {
        for (int sig = 0; sig < nbf; ++sig) {
            Eigen::Map<const Eigen::MatrixXd> T1_slice(T1.data() + (lam + sig * nbf) * no * nbf, no, nbf);
            Eigen::Map<Eigen::MatrixXd> T2_slice(T2.data() + (lam + sig * nbf) * no * nv, no, nv);
            T2_slice.noalias() = T1_slice * Cv;
        }
    }

    // 2. PRE-COMPUTE C23 KERNEL CACHE (Menghindari Redundansi)
    std::map<std::pair<int, int>, Eigen::MatrixXd> C23_cache;
    for (const auto& v2 : virt_spaces) {
        if (v2.size == 0) continue;
        for (const auto& v3 : virt_spaces) {
            if (v3.size == 0) continue;
            Eigen::MatrixXd C23_mat(nbf * nbf, v2.size * v3.size);
            #pragma omp parallel for collapse(2)
            for (int lam = 0; lam < nbf; ++lam) {
                for (int sig = 0; sig < nbf; ++sig) {
                    for (int b = 0; b < v2.size; ++b) {
                        for (int c = 0; c < v3.size; ++c) {
                            C23_mat(lam + sig * nbf, b + c * v2.size) = Cv(lam, v2.offset + b) * Cv(sig, v3.offset + c);
                        }
                    }
                }
            }
            C23_cache[{v2.id, v3.id}] = std::move(C23_mat);
        }
    }

    // 3. JIT MICRO-GEMM OVER IRREP BLOCKS (The HPC Magic)
    for (const auto& o1 : occ_spaces) {
        if (o1.size == 0) continue;
        for (const auto& v1 : virt_spaces) {
            if (v1.size == 0) continue;

            Eigen::MatrixXd T2_mat(o1.size * v1.size, nbf * nbf);
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < o1.size; ++i) {
                for (int a = 0; a < v1.size; ++a) {
                    for (int lam = 0; lam < nbf; ++lam) {
                        for (int sig = 0; sig < nbf; ++sig) {
                            T2_mat(i + a * o1.size, lam + sig * nbf) = T2(o1.offset + i, v1.offset + a, lam, sig);
                        }
                    }
                }
            }

            for (const auto& v2 : virt_spaces) {
                if (v2.size == 0) continue;
                for (const auto& v3 : virt_spaces) {
                    if (v3.size == 0) continue;

                    // EKSEKUSI HANYA JIKA SIMETRI COCOK!
                    if ((o1.id ^ v1.id ^ v2.id ^ v3.id) == 0) {
                        Eigen::MatrixXd Out_mat = T2_mat * C23_cache[{v2.id, v3.id}];
                        Eigen::Tensor<double, 4> block(o1.size, v1.size, v2.size, v3.size);
                        
                        #pragma omp parallel for collapse(2)
                        for (int b = 0; b < v2.size; ++b) {
                            for (int c = 0; c < v3.size; ++c) {
                                for (int i = 0; i < o1.size; ++i) {
                                    for (int a = 0; a < v1.size; ++a) {
                                        block(i, a, b, c) = Out_mat(i + a * o1.size, b + c * v2.size);
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

    // 1. GLOBAL HALF-TRANSFORMATION
    Eigen::Tensor<double, 4> T1(no, nbf, nbf, nbf);
    Eigen::Map<const Eigen::MatrixXd> eri_mat(eri_ao.data(), nbf, nbf * nbf * nbf);
    Eigen::Map<Eigen::MatrixXd> T1_mat(T1.data(), no, nbf * nbf * nbf);
    T1_mat.noalias() = Co.transpose() * eri_mat;

    Eigen::Tensor<double, 4> T2(no, no, nbf, nbf);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int lam = 0; lam < nbf; ++lam) {
        for (int sig = 0; sig < nbf; ++sig) {
            Eigen::Map<const Eigen::MatrixXd> T1_slice(T1.data() + (lam + sig * nbf) * no * nbf, no, nbf);
            Eigen::Map<Eigen::MatrixXd> T2_slice(T2.data() + (lam + sig * nbf) * no * no, no, no);
            T2_slice.noalias() = T1_slice * Co;
        }
    }

    // 2. PRE-COMPUTE C34 KERNEL CACHE
    std::map<std::pair<int, int>, Eigen::MatrixXd> C34_cache;
    for (const auto& o3 : occ_spaces) {
        if (o3.size == 0) continue;
        for (const auto& v1 : virt_spaces) {
            if (v1.size == 0) continue;
            Eigen::MatrixXd C34_mat(nbf * nbf, o3.size * v1.size);
            #pragma omp parallel for collapse(2)
            for (int lam = 0; lam < nbf; ++lam) {
                for (int sig = 0; sig < nbf; ++sig) {
                    for (int k = 0; k < o3.size; ++k) {
                        for (int a = 0; a < v1.size; ++a) {
                            C34_mat(lam + sig * nbf, k + a * o3.size) = Co(lam, o3.offset + k) * Cv(sig, v1.offset + a);
                        }
                    }
                }
            }
            C34_cache[{o3.id, v1.id}] = std::move(C34_mat);
        }
    }

    // 3. JIT MICRO-GEMM OVER IRREP BLOCKS
    for (const auto& o1 : occ_spaces) {
        if (o1.size == 0) continue;
        for (const auto& o2 : occ_spaces) {
            if (o2.size == 0) continue;

            Eigen::MatrixXd T2_mat(o1.size * o2.size, nbf * nbf);
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < o1.size; ++i) {
                for (int j = 0; j < o2.size; ++j) {
                    for (int lam = 0; lam < nbf; ++lam) {
                        for (int sig = 0; sig < nbf; ++sig) {
                            T2_mat(i + j * o1.size, lam + sig * nbf) = T2(o1.offset + i, o2.offset + j, lam, sig);
                        }
                    }
                }
            }

            for (const auto& o3 : occ_spaces) {
                if (o3.size == 0) continue;
                for (const auto& v1 : virt_spaces) {
                    if (v1.size == 0) continue;

                    // EKSEKUSI HANYA JIKA SIMETRI COCOK!
                    if ((o1.id ^ o2.id ^ o3.id ^ v1.id) == 0) {
                        Eigen::MatrixXd Out_mat = T2_mat * C34_cache[{o3.id, v1.id}];
                        Eigen::Tensor<double, 4> block(o1.size, o2.size, o3.size, v1.size);
                        
                        #pragma omp parallel for collapse(2)
                        for (int k = 0; k < o3.size; ++k) {
                            for (int a = 0; a < v1.size; ++a) {
                                for (int i = 0; i < o1.size; ++i) {
                                    for (int j = 0; j < o2.size; ++j) {
                                        block(i, j, k, a) = Out_mat(i + j * o1.size, k + a * o3.size);
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
