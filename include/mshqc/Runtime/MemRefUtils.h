

#pragma once

#include <cstdint>
#include <vector>
#include "AlignedAllocator.h"

namespace mshqc {
namespace runtime {

template <typename T, size_t N>
struct StridedMemRefType {
    T *allocatedPtr;
    T *alignedPtr;
    int64_t offset;
    int64_t sizes[N];
    int64_t strides[N];
};

template <typename T>
struct StridedMemRefType<T, 1> {
    T *allocatedPtr;
    T *alignedPtr;
    int64_t offset;
    int64_t sizes[1];
    int64_t strides[1];
};

template <typename T>
struct StridedMemRefType<T, 2> {
    T *allocatedPtr;
    T *alignedPtr;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
};

template <typename T>
StridedMemRefType<T, 1> makeMemRef1D(std::vector<T, AlignedAllocator<T>>& vec) {
    StridedMemRefType<T, 1> memref;
    memref.allocatedPtr = vec.data();
    memref.alignedPtr = vec.data();
    memref.offset = 0;
    memref.sizes[0] = vec.size();
    memref.strides[0] = 1;
    return memref;
}

template <typename T>
StridedMemRefType<T, 2> makeMemRef2D(std::vector<T, AlignedAllocator<T>>& vec, int64_t rows, int64_t cols) {
    StridedMemRefType<T, 2> memref;
    memref.allocatedPtr = vec.data();
    memref.alignedPtr = vec.data();
    memref.offset = 0;
    memref.sizes[0] = rows;
    memref.sizes[1] = cols;
    memref.strides[0] = cols;
    memref.strides[1] = 1;
    return memref;
}

}
}
