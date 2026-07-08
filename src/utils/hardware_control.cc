#include <iostream>
#include <thread>
#include <omp.h>
#include <algorithm>
#include <cstdlib>

// =========================================================================
// HEADER KHUSUS SISTEM OPERASI
// =========================================================================
#if defined(_WIN32)
    #include <windows.h>
    #include <vector>
#elif defined(__APPLE__)
    #include <sys/types.h>
    #include <sys/sysctl.h>
#elif defined(__linux__)
    #include <fstream>
    #include <string>
    #include <unordered_set>
#endif

// =========================================================================
// WEAK LINKING API (Aman dilewati jika pustaka tidak ada)
// =========================================================================
extern "C" {
    void openblas_set_num_threads(int num_threads) __attribute__((weak));
    void mkl_set_num_threads(int num_threads) __attribute__((weak));
}

namespace mshqc {
namespace utils {

// =========================================================================
// FUNGSI 1: KONTROL ENVIRONMENT LINTAS PLATFORM
// =========================================================================
void set_env_safe(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value); // Windows API
#else
    setenv(name, value, 1); // POSIX (Linux/macOS)
#endif
}

// =========================================================================
// FUNGSI 2: DETEKSI INTI FISIK (PHYSICAL CORES) MULTI-OS
// =========================================================================
int get_physical_cores() {
#if defined(_WIN32)
    // Mode Windows: Menggunakan GetLogicalProcessorInformation
    DWORD length = 0;
    GetLogicalProcessorInformation(nullptr, &length);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
    }
    
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (!GetLogicalProcessorInformation(buffer.data(), &length)) {
        return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
    }
    
    int physical_cores = 0;
    for (const auto& info : buffer) {
        if (info.Relationship == RelationProcessorCore) {
            physical_cores++;
        }
    }
    return physical_cores > 0 ? physical_cores : std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);

#elif defined(__APPLE__)
    // Mode macOS: Menggunakan Sysctl
    int count = 0;
    size_t size = sizeof(count);
    if (sysctlbyname("hw.physicalcpu", &count, &size, nullptr, 0) == 0) {
        return count;
    }
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);

#elif defined(__linux__)
    // Mode Linux: Membaca Topologi /proc/cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    std::unordered_set<std::string> unique_cores;
    std::string current_phys_id = "0";
    std::string current_core_id = "0";

    if (cpuinfo.is_open()) {
        while (std::getline(cpuinfo, line)) {
            if (line.find("physical id") == 0) {
                current_phys_id = line.substr(line.find(":") + 1);
            } else if (line.find("core id") == 0) {
                current_core_id = line.substr(line.find(":") + 1);
                unique_cores.insert(current_phys_id + "_" + current_core_id);
            }
        }
        if (!unique_cores.empty()) {
            return unique_cores.size();
        }
    }
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);

#else
    // Mode Fallback OS Lainnya
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
#endif
}

// =========================================================================
// FUNGSI 3: AUTO-TUNING HARDWARE UTAMA
// =========================================================================
void auto_tune_hardware(int n_basis_functions, bool is_open_shell) {
    int physical_cores = get_physical_cores();
    int optimal_omp_threads = physical_cores; 

    // Logika Pintar: Matikan paralelisme untuk sistem molekul sangat kecil
    if (n_basis_functions < 20) {
        optimal_omp_threads = 1;
    } else if (n_basis_functions < 50) {
        optimal_omp_threads = std::max(2, physical_cores - 2);
    } else {
        optimal_omp_threads = physical_cores;
    }

    // 1. Kunci OpenMP (Master Loop)
    omp_set_dynamic(0); 
    omp_set_num_threads(optimal_omp_threads);

    // 2. Kunci BLAS/MKL ke-1 (Anti Tabrakan Thread)
    if (openblas_set_num_threads) {
        openblas_set_num_threads(1);
    }
    if (mkl_set_num_threads) {
        mkl_set_num_threads(1);
    }
    
    // 3. Kunci TBLIS & OpenMP Internal via Environment Cross-Platform
    set_env_safe("TBLIS_NUM_THREADS", "1");
    set_env_safe("OMP_MAX_ACTIVE_LEVELS", "1"); 

    // 4. Report ke Terminal
    std::cout << "\n  [MSHQC Auto-Tuning] Multi-OS Hardware Detected:\n"
              << "    -> Physical Cores   : " << physical_cores << "\n"
              << "    -> OpenMP Threads   : " << optimal_omp_threads << " (Master Loop)\n"
              << "    -> BLAS/MKL Threads : 1 (Native Weak-Linked)\n"
              << "    -> TBLIS Threads    : 1 (Env Locked)\n"
              << "  ---------------------------------------------------\n";
}

} // namespace utils
} // namespace mshqc
