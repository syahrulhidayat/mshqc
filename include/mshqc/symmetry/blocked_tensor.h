#ifndef MSHQC_SYMMETRY_BLOCKED_TENSOR_H
#define MSHQC_SYMMETRY_BLOCKED_TENSOR_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unordered_map>
#include <vector>
#include <iostream>

namespace mshqc {

struct IrrepSpace {
    int id;
    int offset;
    int size;
};

inline int pack_irreps(int i, int a, int j, int b) {
    return (i << 9) | (a << 6) | (j << 3) | b;
}

inline void unpack_irreps(int packed, int& i, int& a, int& j, int& b) {
    b = packed & 7;
    j = (packed >> 3) & 7;
    a = (packed >> 6) & 7;
    i = (packed >> 9) & 7;
}







class BlockedTensor2D {
public:
    

    std::unordered_map<int, Eigen::MatrixXd> blocks;

    Eigen::MatrixXd* get_block(int irrep) {
        auto it = blocks.find(irrep);
        return (it != blocks.end()) ? &(it->second) : nullptr;
    }

    const Eigen::MatrixXd* get_block(int irrep) const {
        auto it = blocks.find(irrep);
        return (it != blocks.end()) ? &(it->second) : nullptr;
    }

    void allocate_block(int irrep, int d1, int d2) {
        blocks[irrep] = Eigen::MatrixXd::Zero(d1, d2);
    }
};







class BlockedTensor4D {
public:
    std::unordered_map<int, Eigen::Tensor<double, 4>> blocks;

    Eigen::Tensor<double, 4>* get_block(int i, int a, int j, int b) {
        auto it = blocks.find(pack_irreps(i, a, j, b));
        return (it != blocks.end()) ? &(it->second) : nullptr;
    }

    const Eigen::Tensor<double, 4>* get_block(int i, int a, int j, int b) const {
        auto it = blocks.find(pack_irreps(i, a, j, b));
        return (it != blocks.end()) ? &(it->second) : nullptr;
    }

    void allocate_block(int i, int a, int j, int b, int d1, int d2, int d3, int d4) {
        blocks[pack_irreps(i, a, j, b)] = Eigen::Tensor<double, 4>(d1, d2, d3, d4);
    }

    

    void clear() {
        blocks.clear();
    }
};



inline std::vector<IrrepSpace> get_irrep_spaces(const std::vector<int>& sorted_irreps, int start_idx, int total_size) {
    std::vector<IrrepSpace> spaces;
    if (total_size == 0) return spaces;

    int current_id = sorted_irreps[start_idx];
    int current_offset = 0;
    int current_size = 0;

    for (int i = 0; i < total_size; ++i) {
        int id = sorted_irreps[start_idx + i];
        if (id == current_id) {
            current_size++;
        } else {
            spaces.push_back({current_id, current_offset, current_size});
            current_id = id;
            current_offset += current_size;
            current_size = 1;
        }
    }
    spaces.push_back({current_id, current_offset, current_size});
    return spaces;
}

} 

#endif