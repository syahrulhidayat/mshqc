/**
 * @file include/mshqc/ump3_memory.h
 * @brief Memory Manager for UMP3 Tensors
 * @details Mengelola alokasi dan dealokasi dinamis tensor integral 
 * untuk mencegah penggunaan RAM berlebih.
 */

#ifndef MSHQC_UMP3_MEMORY_H
#define MSHQC_UMP3_MEMORY_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <map>
#include <string>
#include <iostream>
#ifdef I
#undef I
#endif

namespace mshqc {

class UMP3Workspace {
public:
    // Tipe Tensor 4D standar
    using Tensor4D = Eigen::Tensor<double, 4>;

    UMP3Workspace() = default;
    ~UMP3Workspace() { clear_all(); }

    // Alokasi Tensor baru (atau reset yang sudah ada)
    // Mengembalikan referensi agar bisa langsung diisi data
    Tensor4D& allocate(const std::string& key, long d1, long d2, long d3, long d4);

    // Mengambil tensor yang sudah ada (Throw error jika tidak ada)
    Tensor4D& get(const std::string& key);

    // Menghapus tensor dari memori segera (PENTING untuk hemat RAM)
    void free(const std::string& key);

    // Menghapus semua
    void clear_all();

    // Cek penggunaan memori (Estimasi MB)
    double get_memory_usage_mb() const;

private:
    std::map<std::string, Tensor4D> storage_;
};

} // namespace mshqc

#endif // MSHQC_UMP3_MEMORY_H