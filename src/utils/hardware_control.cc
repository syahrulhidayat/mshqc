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

















#include <iostream>
#include <thread>
#include <omp.h>
#include <algorithm>
#include <cstdlib>

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

extern "C" {
    void openblas_set_num_threads(int num_threads) __attribute__((weak));
    void mkl_set_num_threads(int num_threads) __attribute__((weak));
}

namespace mshqc {
namespace utils {

void set_env_safe(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

int get_physical_cores() {
#if defined(_WIN32)

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

    int count = 0;
    size_t size = sizeof(count);
    if (sysctlbyname("hw.physicalcpu", &count, &size, nullptr, 0) == 0) {
        return count;
    }
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);

#elif defined(__linux__)

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

    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
#endif
}

void auto_tune_hardware(int n_basis_functions, bool is_open_shell) {
    int physical_cores = get_physical_cores();
    int optimal_omp_threads = physical_cores;

    if (n_basis_functions < 20) {
        optimal_omp_threads = 1;
    } else if (n_basis_functions < 50) {
        optimal_omp_threads = std::max(2, physical_cores - 2);
    } else {
        optimal_omp_threads = physical_cores;
    }

    omp_set_dynamic(0);
    omp_set_num_threads(optimal_omp_threads);

    if (openblas_set_num_threads) {
        openblas_set_num_threads(1);
    }
    if (mkl_set_num_threads) {
        mkl_set_num_threads(1);
    }

    set_env_safe("TBLIS_NUM_THREADS", "1");
    set_env_safe("OMP_MAX_ACTIVE_LEVELS", "1");

    std::cout << "\n  [MSHQC Auto-Tuning] Multi-OS Hardware Detected:\n"
              << "    -> Physical Cores   : " << physical_cores << "\n"
              << "    -> OpenMP Threads   : " << optimal_omp_threads << " (Master Loop)\n"
              << "    -> BLAS/MKL Threads : 1 (Native Weak-Linked)\n"
              << "    -> TBLIS Threads    : 1 (Env Locked)\n"
              << "  ---------------------------------------------------\n";
}

}
}
