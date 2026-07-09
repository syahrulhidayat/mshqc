/**
 * @file include/mshqc/mp3/ump3_memory.h
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
    

    using Tensor4D = Eigen::Tensor<double, 4>;

    UMP3Workspace() = default;
    ~UMP3Workspace() { clear_all(); }

    

    

    Tensor4D& allocate(const std::string& key, long d1, long d2, long d3, long d4);

    

    Tensor4D& get(const std::string& key);

    

    void free(const std::string& key);

    

    void clear_all();

    

    double get_memory_usage_mb() const;

private:
    std::map<std::string, Tensor4D> storage_;
};

} 


#endif 
