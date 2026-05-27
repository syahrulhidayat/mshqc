#include <iostream>
#include <thread>
#include <omp.h>
#include <algorithm>

// =========================================================================
// [TRIK HPC KELAS DEWA]: WEAK LINKING C-API
// Kita mendeklarasikan fungsi internal BLAS/MKL sebagai "weak".
// Jika saat runtime program menemukan OpenBLAS/MKL, fungsi ini akan aktif.
// Jika tidak, pointer-nya bernilai null dan aman dilewati (Tidak akan Crash!).
// =========================================================================
extern "C" {
    void openblas_set_num_threads(int num_threads) __attribute__((weak));
    void mkl_set_num_threads(int num_threads) __attribute__((weak));
}

namespace mshqc {
namespace utils {

void auto_tune_hardware(int n_basis_functions, bool is_open_shell) {
    int logical_threads = std::thread::hardware_concurrency();
    int physical_cores = logical_threads / 2; // Ryzen 5 5500U: 12 Logis -> 6 Fisik
    
    int optimal_threads = physical_cores; 

    // Logika Pintar: Molekul kecil tidak butuh banyak pekerja
    if (n_basis_functions < 20) {
        optimal_threads = 1;
    } else if (n_basis_functions < 50) {
        optimal_threads = std::max(2, physical_cores - 2);
    } else {
        optimal_threads = physical_cores;
    }

    // =========================================================================
    // 1. KONTROL OPENMP (Internal C++)
    // =========================================================================
    omp_set_dynamic(0); // Matikan auto-dynamic agar tidak liar
    omp_set_num_threads(optimal_threads);

    // =========================================================================
    // 2. KONTROL BLAS / MKL SECARA NATIVE (Menggantikan setenv!)
    // Ini langsung mencekik jumlah thread BLAS secara real-time di dalam RAM
    // =========================================================================
    if (openblas_set_num_threads) {
        openblas_set_num_threads(optimal_threads);
    }
    
    if (mkl_set_num_threads) {
        mkl_set_num_threads(optimal_threads);
    }

    // Report ke Terminal
    std::cout << "  [HPC Monitor] C++ Native Auto-Tuning Activated:\n"
              << "    - Target Physical Cores   : " << optimal_threads << " Threads\n"
              << "    - OpenMP & BLAS Sync      : Natively Locked (No Thread Explosion)\n"
              << "=========================================================================\n";
}

} // namespace utils
} // namespace mshqc