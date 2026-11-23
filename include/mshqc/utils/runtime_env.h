// Copyright 2026 Muhamad Syahrul Hidayat
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
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#if defined(_WIN32)
  #include <windows.h>
#elif defined(__APPLE__)
  #include <dlfcn.h>
  #include <sys/sysctl.h>
  #include <sys/types.h>
#elif defined(__linux__)
  #include <dlfcn.h>
  #include <fstream>
  #include <sched.h>
  #include <unordered_set>
#endif

#include "mshqc/core/molecule.h"

namespace mshqc_auto_tune {

inline void set_env_if_empty(const char* name, const char* value) {
    if (std::getenv(name) == nullptr) {
#if defined(_WIN32)
        _putenv_s(name, value);
#else
        setenv(name, value, 0);
#endif
    }
}

inline int get_physical_cores() {
#if defined(_WIN32)
    DWORD length = 0;
    GetLogicalProcessorInformation(nullptr, &length);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
    }
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(
        length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (!GetLogicalProcessorInformation(buffer.data(), &length)) {
        return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
    }
    int physical_cores = 0;
    for (const auto& info : buffer) {
        if (info.Relationship == RelationProcessorCore) physical_cores++;
    }
    return physical_cores > 0
        ? physical_cores
        : std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
#elif defined(__APPLE__)
    int count = 0;
    size_t size = sizeof(count);
    if (sysctlbyname("hw.physicalcpu", &count, &size, nullptr, 0) == 0) return count;
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
#elif defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line, current_phys_id = "0", current_core_id = "0";
    std::unordered_set<std::string> unique_cores;
    if (cpuinfo.is_open()) {
        while (std::getline(cpuinfo, line)) {
            if (line.find("physical id") == 0) current_phys_id = line.substr(line.find(":") + 1);
            else if (line.find("core id") == 0) {
                current_core_id = line.substr(line.find(":") + 1);
                unique_cores.insert(current_phys_id + "_" + current_core_id);
            }
        }
        if (!unique_cores.empty()) return static_cast<int>(unique_cores.size());
    }
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
#else
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
#endif
}

inline void initialize_hpc_environment() {
#if defined(__linux__)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    unsigned int logical_cpus = std::max(1u, std::thread::hardware_concurrency());
    for (unsigned int i = 0; i < logical_cpus; ++i) CPU_SET(i, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
#endif

    int num_threads = get_physical_cores();
    if (const char* env_omp = std::getenv("OMP_NUM_THREADS")) {
        int parsed = std::atoi(env_omp);
        if (parsed > 0) num_threads = parsed;
    }

    std::string threads_str = std::to_string(num_threads);
    set_env_if_empty("OMP_NUM_THREADS", threads_str.c_str());
#if defined(__APPLE__) || defined(_WIN32)
  #if defined(_WIN32)
    _putenv_s("TBLIS_NUM_THREADS", "1");
    _putenv_s("MKL_NUM_THREADS", "1");
    _putenv_s("OPENBLAS_NUM_THREADS", "1");
  #else
    set_env_if_empty("TBLIS_NUM_THREADS", "1");
  #endif
    set_env_if_empty("OMP_WAIT_POLICY", "PASSIVE");
    set_env_if_empty("KMP_BLOCKTIME", "0");
#else
    set_env_if_empty("TBLIS_NUM_THREADS", threads_str.c_str());
#endif
    set_env_if_empty("OPENBLAS_NUM_THREADS", "1");
    set_env_if_empty("MKL_NUM_THREADS", "1");
    set_env_if_empty("VECLIB_MAXIMUM_THREADS", "1");
    set_env_if_empty("NUMEXPR_NUM_THREADS", "1");
    set_env_if_empty("OMP_MAX_ACTIVE_LEVELS", "1");
    set_env_if_empty("HDF5_USE_FILE_LOCKING", "FALSE");

    Eigen::setNbThreads(1);
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    omp_set_max_active_levels(1);

#if defined(_WIN32)
    using set_threads_fn = void (*)(int);
    for (const char* dll : {"libtblis.dll", "tblis.dll"}) {
        if (HMODULE h = GetModuleHandleA(dll)) {
            if (auto fn = reinterpret_cast<set_threads_fn>(GetProcAddress(h, "tblis_set_num_threads"))) fn(1);
        }
    }
    for (const char* dll : {"mkl_rt.2.dll", "mkl_rt.3.dll", "mkl_rt.dll"}) {
        if (HMODULE h = GetModuleHandleA(dll)) {
            if (auto fn = reinterpret_cast<set_threads_fn>(GetProcAddress(h, "MKL_Set_Num_Threads"))) fn(1);
        }
    }
#elif defined(__linux__) || defined(__APPLE__)
    using set_threads_fn = void (*)(int);
    if (auto fn = reinterpret_cast<set_threads_fn>(dlsym(RTLD_DEFAULT, "openblas_set_num_threads"))) fn(1);
    if (auto fn = reinterpret_cast<set_threads_fn>(dlsym(RTLD_DEFAULT, "MKL_Set_Num_Threads"))) fn(1);
    if (auto fn = reinterpret_cast<set_threads_fn>(dlsym(RTLD_DEFAULT, "tblis_set_num_threads"))) {
  #if defined(__APPLE__)
        fn(1);
  #else
        fn(num_threads);
  #endif
    }
#endif
}

inline int symbol_to_z(const std::string& sym) {
    static const std::unordered_map<std::string, int> table = {
        {"H",1},{"He",2},{"Li",3},{"Be",4},{"B",5},{"C",6},{"N",7},{"O",8},{"F",9},{"Ne",10},
        {"Na",11},{"Mg",12},{"Al",13},{"Si",14},{"P",15},{"S",16},{"Cl",17},{"Ar",18},
        {"K",19},{"Ca",20},{"Sc",21},{"Ti",22},{"V",23},{"Cr",24},{"Mn",25},{"Fe",26},
        {"Co",27},{"Ni",28},{"Cu",29},{"Zn",30},{"Ga",31},{"Ge",32},{"As",33},{"Se",34},
        {"Br",35},{"Kr",36},{"Rb",37},{"Sr",38},{"Y",39},{"Zr",40},{"Nb",41},{"Mo",42},
        {"Tc",43},{"Ru",44},{"Rh",45},{"Pd",46},{"Ag",47},{"Cd",48},{"In",49},{"Sn",50},
        {"Sb",51},{"Te",52},{"I",53},{"Xe",54},
        {"Cs",55},{"Ba",56},{"La",57},{"Ce",58},{"Pr",59},{"Nd",60},{"Pm",61},{"Sm",62},{"Eu",63},{"Gd",64},
        {"Tb",65},{"Dy",66},{"Ho",67},{"Er",68},{"Tm",69},{"Yb",70},{"Lu",71},
        {"Hf",72},{"Ta",73},{"W",74},{"Re",75},{"Os",76},{"Ir",77},{"Pt",78},{"Au",79},{"Hg",80},
        {"Tl",81},{"Pb",82},{"Bi",83},{"Po",84},{"At",85},{"Rn",86},
        {"Fr",87},{"Ra",88},{"Ac",89},{"Th",90},{"Pa",91},{"U",92},{"Np",93},{"Pu",94},
        {"Am",95},{"Cm",96},{"Bk",97},{"Cf",98},{"Es",99},{"Fm",100},{"Md",101},{"No",102},
        {"Lr",103},{"Rf",104},{"Db",105},{"Sg",106},{"Bh",107},{"Hs",108},{"Mt",109},{"Ds",110},
        {"Rg",111},{"Cn",112},{"Nh",113},{"Fl",114},{"Mc",115},{"Lv",116},{"Ts",117},{"Og",118}
    };
    if (sym.empty()) throw std::invalid_argument("Empty atom symbol");
    std::string key = sym;
    key[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(key[0])));
    for (size_t i = 1; i < key.size(); ++i) {
        key[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(key[i])));
    }
    auto it = table.find(key);
    if (it == table.end()) throw std::invalid_argument("Unknown element symbol: " + sym);
    return it->second;
}

inline double unit_scale(const std::string& units) {
    if (!units.empty() && (units[0] == 'a' || units[0] == 'A')) {
        return 1.0 / 0.529177210903;
    }
    return 1.0;
}

inline int standard_frozen_core_count(const mshqc::Molecule& mol) {
    int nf = 0;
    for (size_t i = 0; i < mol.n_atoms(); ++i) {
        const int Z = mol.atom(i).atomic_number;
        if (Z <= 2) nf += 0;
        else if (Z <= 10) nf += 1;
        else if (Z <= 18) nf += 5;
        else if (Z <= 36) nf += 9;
        else if (Z <= 54) nf += 18;
        else if (Z <= 71) nf += 27;
        else if (Z <= 86) nf += 34;
        else nf += 43;
    }
    return nf;
}

} // namespace mshqc_auto_tune
