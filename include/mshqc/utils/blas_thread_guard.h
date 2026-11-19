// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0
// [INJECTED_HPC_V2_BLAS_GUARD]
#pragma once

#include <Eigen/Core>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace mshqc {
namespace utils {

class ScopedBlasThreads {
 public:
  // Naikkan thread MKL/OpenBLAS/Eigen hanya di luar wilayah paralel OpenMP
  // dan hanya jika ukuran matriks >= threshold agar sistem kecil tanpa overhead.
  explicit ScopedBlasThreads(int problem_dim, int min_dim = 96) : active_(false) {
#ifdef _OPENMP
    if (omp_in_parallel()) return;
    const int target = omp_get_max_threads();
    if (target <= 1 || problem_dim < min_dim) return;
    set_backend_threads(target);
    active_ = true;
#else
    (void)problem_dim;
    (void)min_dim;
#endif
  }

  ~ScopedBlasThreads() {
    if (active_) {
      set_backend_threads(1);
    }
  }

  ScopedBlasThreads(const ScopedBlasThreads&) = delete;
  ScopedBlasThreads& operator=(const ScopedBlasThreads&) = delete;

 private:
  bool active_;

  using SetThreadsFn = void (*)(int);

  struct BackendHandles {
    SetThreadsFn mkl_set = nullptr;
    SetThreadsFn openblas_set = nullptr;

    BackendHandles() {
#if defined(_WIN32)
      const char* mkl_dlls[] = {"mkl_rt.2.dll", "mkl_rt.3.dll", "mkl_rt.dll"};
      for (const char* dll : mkl_dlls) {
        if (HMODULE h = GetModuleHandleA(dll)) {
          mkl_set = reinterpret_cast<SetThreadsFn>(GetProcAddress(h, "MKL_Set_Num_Threads"));
          if (mkl_set) break;
        }
      }
      const char* ob_dlls[] = {"libopenblas.dll", "openblas.dll"};
      for (const char* dll : ob_dlls) {
        if (HMODULE h = GetModuleHandleA(dll)) {
          openblas_set = reinterpret_cast<SetThreadsFn>(GetProcAddress(h, "openblas_set_num_threads"));
          if (openblas_set) break;
        }
      }
#elif defined(__linux__) || defined(__APPLE__)
      mkl_set = reinterpret_cast<SetThreadsFn>(dlsym(RTLD_DEFAULT, "MKL_Set_Num_Threads"));
      openblas_set = reinterpret_cast<SetThreadsFn>(dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
#endif
    }
  };

  static void set_backend_threads(int nt) {
    static const BackendHandles handles;
    if (handles.mkl_set) handles.mkl_set(nt);
    if (handles.openblas_set) handles.openblas_set(nt);
    Eigen::setNbThreads(nt);
  }
};

}  // namespace utils
}  // namespace mshqc
