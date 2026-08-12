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
