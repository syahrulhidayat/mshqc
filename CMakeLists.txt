cmake_minimum_required(VERSION 3.14)

# ==============================================================================
# 1. INFORMASI PROYEK
# ==============================================================================
project(mshqc LANGUAGES CXX C)

# Jika dikompilasi lewat pip/setuptools, ambil versi dari environment variable
if(DEFINED ENV{SETUPTOOLS_SCM_PRETEND_VERSION})
    set(MSHQC_VERSION "$ENV{SETUPTOOLS_SCM_PRETEND_VERSION}")
else()
    set(MSHQC_VERSION "1.0.0-dev") # Fallback lokal jika tidak lewat Git
endif()

message(STATUS ">> Membangun MSHQC Versi: ${MSHQC_VERSION}")
# ==============================================================================
# [OPTIMASI KOMPILER CERDAS]
# ==============================================================================
include(CheckCXXCompilerFlag)

if(DEFINED ENV{GITHUB_ACTIONS} OR DEFINED ENV{CIBUILDWHEEL})
    # [PUBLIC RELEASE]: Biner Universal Berkekuatan Tinggi (manylinux safe)
    message(STATUS ">> CI/CD Server Detected: Menyiapkan optimasi Universal HPC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -funroll-loops -ftree-vectorize -mtune=generic -DNDEBUG")
    
    # Deteksi aman untuk injeksi AVX2/FMA agar tidak crash di CPU lawas
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
    
    if(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
        message(STATUS ">> Menginjeksi instruksi Vektor AVX2 & FMA secara aman.")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx2 -mfma")
    endif()
else()
    # [PRIVATE/LOCAL]: Super Turbo (Maksimal untuk mesin saat ini)
    message(STATUS ">> Local Environment Detected: Mengaktifkan Super Turbo (-march=native)")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -funroll-loops -ftree-vectorize -DNDEBUG")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Mengembalikan Interprocedural Optimization (IPO/LTO) yang terbukti aman!
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if(ipo_supported)
    message(STATUS ">> IPO / LTO Didukung dan Diaktifkan! (Super Turbo)")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ==============================================================================
# 2. KONFIGURASI LINGKUNGAN CONDA & PYTHON (PORTABEL UNTUK CODESPACES)
# ==============================================================================
if(DEFINED ENV{CONDA_PREFIX})
    set(Python_ROOT_DIR "$ENV{CONDA_PREFIX}")
    set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
    set(Python3_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
    
    include_directories("$ENV{CONDA_PREFIX}/include")
    link_directories("$ENV{CONDA_PREFIX}/lib")
elseif(EXISTS "/opt/conda")
    set(Python_ROOT_DIR "/opt/conda")
    include_directories("/opt/conda/include")
    link_directories("/opt/conda/lib")
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# ==============================================================================
# 3. PENCARIAN PUSTAKA "ULTIMATE STACK" 
# ==============================================================================
find_package(OpenMP REQUIRED)
# --- BLAS & LAPACK (SMART DYNAMIC DISPATCH) ---
if(DEFINED ENV{GITHUB_ACTIONS} OR DEFINED ENV{CIBUILDWHEEL})
    message(STATUS ">> CI/CD Server Detected: Memaksa OpenBLAS untuk PyPI Wheel (Portabilitas & Ukuran Ideal)")
    
    find_library(OPENBLAS_LIBRARY NAMES openblas openblas_pthread HINTS "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib" "/usr/lib")
    if(OPENBLAS_LIBRARY)
        message(STATUS ">> BLAS/LAPACK: OpenBLAS Ditemukan untuk Biner Publik!")
        set(BLAS_LIBRARIES ${OPENBLAS_LIBRARY})
        set(LAPACK_LIBRARIES ${OPENBLAS_LIBRARY})
    else()
        message(FATAL_ERROR "CRITICAL: OpenBLAS tidak ditemukan di CI!")
    endif()

else()
    message(STATUS ">> Local Build Detected: Mencari Intel MKL (Skala Prioritas 1)...")
    
    find_library(MKL_RT_LIBRARY NAMES mkl_rt HINTS "$ENV{CONDA_PREFIX}/lib" "/opt/conda/lib" "/usr/local/lib" "/usr/lib")
    if(MKL_RT_LIBRARY)
        message(STATUS ">> BLAS/LAPACK: Intel MKL (mkl_rt) Ditemukan! Mengaktifkan optimasi Intel dengan GNU OpenMP.")
        set(BLAS_LIBRARIES ${MKL_RT_LIBRARY})
        set(LAPACK_LIBRARIES ${MKL_RT_LIBRARY})
        
        # INJEKSI KRUSIAL: Paksa MKL memakai libgomp agar tidak Segfault
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_SET_INTERFACE_GNU")
    else()
        message(STATUS ">> MKL tidak ditemukan. Fallback ke OpenBLAS (Prioritas 2)...")
        find_library(OPENBLAS_LIBRARY NAMES openblas openblas_pthread HINTS "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib" "/usr/lib")
        if(OPENBLAS_LIBRARY)
            message(STATUS ">> BLAS/LAPACK: OpenBLAS Ditemukan!")
            set(BLAS_LIBRARIES ${OPENBLAS_LIBRARY})
            set(LAPACK_LIBRARIES ${OPENBLAS_LIBRARY})
        else()
            message(FATAL_ERROR "CRITICAL: MKL dan OpenBLAS tidak ditemukan! Silakan install salah satu.")
        endif()
    endif()
endif()
# --- LIBCINT ---
find_path(CINT_INCLUDE_DIR "cint.h" HINTS "$ENV{CONDA_PREFIX}/include" "/opt/conda/include" "/usr/local/include" "/usr/include")
find_library(CINT_LIBRARY cint HINTS "$ENV{CONDA_PREFIX}/lib" "/opt/conda/lib" "/usr/local/lib" "/usr/lib")

if(CINT_LIBRARY AND CINT_INCLUDE_DIR)
    message(STATUS ">> Integral Engine: libcint ditemukan di ${CINT_INCLUDE_DIR}!")
  
    include_directories(${CINT_INCLUDE_DIR}) 
    
else()
    message(FATAL_ERROR "CRITICAL: libcint (library atau header) tidak ditemukan!")
endif()
# --- TBLIS ---
find_library(TBLIS_LIBRARY tblis HINTS "$ENV{CONDA_PREFIX}/lib" "/opt/conda/lib" "/usr/local/lib" "/usr/lib")
if(TBLIS_LIBRARY)
    message(STATUS ">> Tensor Engine: TBLIS ditemukan!")
else()
    message(FATAL_ERROR "CRITICAL: TBLIS tidak ditemukan!")
endif()

# --- [PERBAIKAN FINAL]: HDF5 FLEKSIBEL (CONDA FIRST, LALU SISTEM) ---
# HINTS akan mencari di Conda terlebih dahulu. Jika gagal, akan mencari di standar sistem operasi.
find_path(HDF5_INCLUDE_DIRS "H5Cpp.h" 
    HINTS "$ENV{CONDA_PREFIX}/include" "/opt/conda/include" 
    PATHS "/usr/include/hdf5/serial" "/usr/local/include"
)
find_library(HDF5_C_LIB hdf5 
    HINTS "$ENV{CONDA_PREFIX}/lib" "/opt/conda/lib" 
    PATHS "/usr/lib/x86_64-linux-gnu/hdf5/serial" "/usr/local/lib" "/usr/lib64"
)
find_library(HDF5_CXX_LIB hdf5_cpp 
    HINTS "$ENV{CONDA_PREFIX}/lib" "/opt/conda/lib" 
    PATHS "/usr/lib/x86_64-linux-gnu/hdf5/serial" "/usr/local/lib" "/usr/lib64"
)

if(HDF5_INCLUDE_DIRS AND HDF5_C_LIB AND HDF5_CXX_LIB)
    message(STATUS ">> Out-of-Core Engine: HDF5 C++ ditemukan di ${HDF5_INCLUDE_DIRS}")
    include_directories(${HDF5_INCLUDE_DIRS})
    set(HDF5_LIBRARIES ${HDF5_CXX_LIB} ${HDF5_C_LIB})
else()
    # Jaring Pengaman Terakhir: Gunakan modul bawaan CMake jika pencarian manual di atas meleset
    message(STATUS ">> HDF5 manual tidak ditemukan. Beralih ke modul FindHDF5 bawaan CMake...")
    find_package(HDF5 COMPONENTS CXX REQUIRED)
    if(HDF5_FOUND)
        message(STATUS ">> HDF5 ditemukan via FindHDF5.")
        include_directories(${HDF5_INCLUDE_DIRS})
        # HDF5_LIBRARIES sudah otomatis terisi oleh modul ini
    else()
        message(FATAL_ERROR "CRITICAL: Pustaka HDF5 C++ sama sekali tidak ditemukan di sistem ini!")
    endif()
endif()

# --- JEMALLOC ---

if(JEMALLOC_LIB)
    message(STATUS ">> Memori Allocator: Jemalloc ditemukan!")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
endif()

# ==============================================================================
# 4. KOLEKSI SOURCE CODE
# ==============================================================================
file(GLOB_RECURSE SRC_CORE CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/core/*.cc")
file(GLOB_RECURSE SRC_SCF CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/scf/*.cc")
file(GLOB_RECURSE SRC_MP CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/mp/*.cc" "${CMAKE_SOURCE_DIR}/src/mp2/*.cc" "${CMAKE_SOURCE_DIR}/src/mp3/*.cc")
file(GLOB_RECURSE SRC_MCSCF CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/mcscf/*.cc")
file(GLOB_RECURSE SRC_INTEGRALS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/integrals/*.cc" "${CMAKE_SOURCE_DIR}/src/ints/*.cc")
file(GLOB_RECURSE SRC_FOUNDATION CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/foundation/*.cc")
file(GLOB_RECURSE SRC_GRADIENT CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/gradient/*.cc")
file(GLOB_RECURSE SRC_INTEGRATION CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/integration/*.cc")
file(GLOB_RECURSE SRC_CI CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/ci/*.cc")
file(GLOB_RECURSE SRC_SYMMETRY CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/symmetry/*.cc")
file(GLOB_RECURSE SRC_UTILS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/utils/*.cc")

set(ALL_SOURCES
    ${SRC_CORE} ${SRC_SCF} ${SRC_MP} ${SRC_MCSCF}
    ${SRC_INTEGRALS} ${SRC_FOUNDATION} ${SRC_GRADIENT}
    ${SRC_INTEGRATION} ${SRC_CI} ${SRC_SYMMETRY}
    ${SRC_UTILS}
)

# ==============================================================================
# 5. TARGET: LIBRARY C++ UTAMA (mshqc)
# ==============================================================================
add_library(mshqc SHARED ${ALL_SOURCES})

set_target_properties(mshqc PROPERTIES 
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../../.."
    BUILD_WITH_INSTALL_RPATH TRUE
)

target_compile_definitions(mshqc PUBLIC 
    EIGEN_USE_BLAS 
    EIGEN_USE_LAPACKE
    EIGEN_NO_DEBUG 
    EIGEN_MPL2_ONLY
)

# [PENTING] HDF5_LIBRARIES ditambahkan di sini agar simbol C++ terhubung ke biner
target_link_libraries(mshqc PUBLIC 
    Eigen3::Eigen
    ${CINT_LIBRARY}
    ${TBLIS_LIBRARY}
    ${HDF5_LIBRARIES} 
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    OpenMP::OpenMP_CXX   
)


target_include_directories(mshqc PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:$ENV{CONDA_PREFIX}/include>   
    $<INSTALL_INTERFACE:include>
)
# ==============================================================================
# 6. TARGET: PYTHON MODULE (_mshqc)
# ==============================================================================
nanobind_add_module(_mshqc "python/bindings.cc")

target_link_libraries(_mshqc PRIVATE mshqc ${CINT_LIBRARY} ${TBLIS_LIBRARY})

set_target_properties(_mshqc PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/mshqc"
    PREFIX "" 
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../../.."
    BUILD_WITH_INSTALL_RPATH TRUE
)

if(ipo_supported)
    set_target_properties(mshqc PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_target_properties(_mshqc PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ==============================================================================
# 7. INSTALASI
# ==============================================================================
install(TARGETS mshqc _mshqc
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

if(EXISTS "${CMAKE_SOURCE_DIR}/include/mshqc")
    install(DIRECTORY include/mshqc DESTINATION include)
endif()
