/**
 * @file src/mp3/ump3_memory.cc
 * @brief Implementation of UMP3 Workspace
 */

#include "mshqc/ump3_memory.h"
#include <stdexcept>
#ifdef I
#undef I
#endif

namespace mshqc {

UMP3Workspace::Tensor4D& UMP3Workspace::allocate(const std::string& key, long d1, long d2, long d3, long d4) {
    // Jika key sudah ada, resize ulang (reuse memory jika kapasitas cukup)
    // Eigen Tensor resize biasanya efisien
    Tensor4D& t = storage_[key];
    t.resize(d1, d2, d3, d4);
    
    // Optional: Print alokasi untuk debug
    // double size_mb = (double)(d1*d2*d3*d4) * 8.0 / 1024.0 / 1024.0;
    // std::cout << "  [MEM] Allocating " << key << ": " << size_mb << " MB\n";
    
    return t;
}

UMP3Workspace::Tensor4D& UMP3Workspace::get(const std::string& key) {
    auto it = storage_.find(key);
    if (it == storage_.end()) {
        throw std::runtime_error("UMP3Workspace: Tensor '" + key + "' not found (maybe freed?).");
    }
    return it->second;
}

void UMP3Workspace::free(const std::string& key) {
    auto it = storage_.find(key);
    if (it != storage_.end()) {
        // Cara paling ampuh membebaskan memori Eigen Tensor:
        // Resize ke 0 dan shrink_to_fit (atau biarkan destructor map bekerja saat erase)
        it->second.resize(0, 0, 0, 0); 
        storage_.erase(it);
        
        // std::cout << "  [MEM] Freed " << key << "\n";
    }
}

void UMP3Workspace::clear_all() {
    storage_.clear();
}

double UMP3Workspace::get_memory_usage_mb() const {
    double total_bytes = 0.0;
    for (const auto& pair : storage_) {
        total_bytes += pair.second.size() * sizeof(double);
    }
    return total_bytes / 1024.0 / 1024.0;
}

} // namespace mshqc