// ==============================================================================
// Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
// MSHQC - MLIR JIT Runtime Components
// 64-byte Aligned Allocator (C++11 Standard Compliant for AVX-512)
// ==============================================================================

#pragma once

#include <cstdlib>
#include <new>
#include <limits>
#include <cstddef>
#include <type_traits>

namespace mshqc {
namespace runtime {

template <typename T, std::size_t Alignment = 64>
struct AlignedAllocator {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    // MENGEMBALIKAN DEFAULT CONSTRUCTOR
    AlignedAllocator() noexcept = default;
    AlignedAllocator(const AlignedAllocator&) noexcept = default;
    
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        
        std::size_t size = n * sizeof(T);
        std::size_t aligned_size = (size + Alignment - 1) & ~(Alignment - 1);
        
        void* ptr = nullptr;
#if defined(_WIN32)
        ptr = _aligned_malloc(aligned_size, Alignment);
#else
        if (posix_memalign(&ptr, Alignment, aligned_size) != 0) {
            throw std::bad_alloc();
        }
#endif
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
#if defined(_WIN32)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }
};

template <typename T, typename U, std::size_t Alignment>
inline bool operator==(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&) noexcept { return true; }

template <typename T, typename U, std::size_t Alignment>
inline bool operator!=(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&) noexcept { return false; }

} // namespace runtime
} // namespace mshqc
