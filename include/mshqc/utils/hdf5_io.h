/**
 * @file include/mshqc/utils/hdf5_io.h
 * @brief HDF5 Out-of-Core Tensor Storage Engine
 * @details Modul ini menangani penyimpanan dan pembacaan tensor ERI raksasa 
 * ke/dari SSD menggunakan format biner HDF5 untuk mencegah kehabisan RAM.
 */

#pragma once

#include <string>
#include <vector>
#include <array>
#include <memory>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>



#include <H5Cpp.h>

namespace mshqc {
namespace utils {

/**
 * @brief Kelas pengelola input/output HDF5 untuk Tensor Eigen.
 */
class HDF5TensorIO {
public:
    /**
     * @brief Mode akses file HDF5.
     */
    enum class Mode {
        READ_ONLY,
        WRITE_TRUNCATE, 

        READ_WRITE      

    };

    /**
     * @brief Konstruktor untuk membuka atau membuat file HDF5.
     * @param filename Path file HDF5 (misal: "eri_cache.h5").
     * @param mode Mode akses file.
     */
    HDF5TensorIO(const std::string& filename, Mode mode);

    /**
     * @brief Destruktor. Akan menutup file HDF5 dengan aman.
     */
    ~HDF5TensorIO();

    /**
     * @brief Membuat dataset 4D baru di dalam file HDF5 dengan Chunking.
     * @param dataset_name Nama dataset (misal: "eri_tensor").
     * @param dims Dimensi penuh dari tensor [n1, n2, n3, n4].
     * @param chunk_dims Dimensi chunk untuk optimasi I/O (misal: [1, 1, n3, n4]).
     */
    void create_dataset_4d(const std::string& dataset_name, 
                           const std::array<long, 4>& dims,
                           const std::array<long, 4>& chunk_dims);

    /**
     * @brief Menulis seluruh tensor 4D ke dalam dataset (Hanya jika muat di RAM).
     * @param dataset_name Nama dataset tujuan.
     * @param tensor Tensor 4D dari Eigen yang akan ditulis.
     */
    void write_tensor_4d(const std::string& dataset_name, 
                         const Eigen::Tensor<double, 4>& tensor);

    /**
     * @brief Membaca seluruh tensor 4D dari dataset ke RAM.
     * @param dataset_name Nama dataset sumber.
     * @return Tensor 4D yang dibaca dari disk.
     */
    Eigen::Tensor<double, 4> read_tensor_4d(const std::string& dataset_name);

    /**
     * @brief Menulis sepotong (*slice*) dari tensor 4D ke disk.
     * Sangat berguna untuk operasi Out-of-Core (misal: menyimpan satu kolom ERI).
     * @param dataset_name Nama dataset.
     * @param offset Titik awal penulisan [i, j, k, l].
     * @param slice_dims Ukuran potongan yang ditulis.
     * @param data_ptr Pointer mentah ke array data.
     */
    void write_slice_4d(const std::string& dataset_name,
                        const std::array<long, 4>& offset,
                        const std::array<long, 4>& slice_dims,
                        const double* data_ptr);

    /**
     * @brief Membaca sepotong (*slice*) dari tensor 4D di disk ke RAM.
     * @param dataset_name Nama dataset.
     * @param offset Titik awal pembacaan [i, j, k, l].
     * @param slice_dims Ukuran potongan yang dibaca.
     * @param data_ptr Pointer mentah untuk menyimpan hasil bacaan.
     */
    void read_slice_4d(const std::string& dataset_name,
                       const std::array<long, 4>& offset,
                       const std::array<long, 4>& slice_dims,
                       double* data_ptr);

private:
    std::string filename_;
    std::unique_ptr<H5::H5File> file_;

    

    std::vector<hsize_t> to_hsize(const std::array<long, 4>& dims) const;
};

} 

} 
