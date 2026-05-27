cmake_minimum_required(VERSION 3.14)

# ==============================================================================
# 1. INFORMASI PROYEK
# ==============================================================================
project(mshqc VERSION 1.0.0 LANGUAGES CXX C)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# [OPTIMASI KOMPILER TURBO]
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -funroll-loops -ftree-vectorize -DNDEBUG")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if(ipo_supported)
    message(STATUS ">> IPO / LTO Didukung dan Diaktifkan! (Super Turbo)")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ==============================================================================
# 2. KONFIGURASI LINGKUNGAN CONDA & PYTHON
# ==============================================================================
if(DEFINED ENV{CONDA_PREFIX})
    set(Python_ROOT_DIR "$ENV{CONDA_PREFIX}")
    set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
    set(Python3_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
    
    include_directories("$ENV{CONDA_PREFIX}/include")
    link_directories("$ENV{CONDA_PREFIX}/lib")
else()
    include_directories("/home/syahrul/miniconda3/include")
    link_directories("/home/syahrul/miniconda3/lib")
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# ==============================================================================
# 3. PENCARIAN PUSTAKA "ULTIMATE STACK" 
# ==============================================================================
find_package(OpenMP REQUIRED)

# --- [MODIFIKASI TAHAP 1]: AMD AOCL (BLIS & FLAME) ---
# Memaksa CMake mencari AOCL sebelum melakukan fallback
find_library(BLIS_LIBRARY NAMES blis-mt blis HINTS "/usr/lib64" "/usr/lib64/blis" "/usr/lib")
find_library(FLAME_LIBRARY NAMES flame HINTS "/usr/lib64" "/usr/lib64/flame" "/usr/lib")

# Pisahkan logika BLAS dan LAPACK agar lebih fleksibel
if(BLIS_LIBRARY)
    message(STATUS ">> BLAS: BLIS ditemukan! (Performa Maksimal untuk AMD)")
    set(BLAS_LIBRARIES ${BLIS_LIBRARY})
else()
    message(STATUS ">> BLIS tidak ditemukan. Fallback ke default BLAS.")
    find_package(BLAS REQUIRED)
endif()

if(FLAME_LIBRARY)
    message(STATUS ">> LAPACK: libflame ditemukan!")
    set(LAPACK_LIBRARIES ${FLAME_LIBRARY})
else()
    message(STATUS ">> libflame tidak ditemukan. Fallback ke default LAPACK.")
    find_package(LAPACK REQUIRED)
endif()

# ------------------------------------------------------------------------------
# [PERBAIKAN EIGEN 3.4.0]: Prioritas Direktori Lokal
# ------------------------------------------------------------------------------
set(EIGEN_LOCAL_PATH "/home/syahrul/mshqc/include/eigen-3.4.0")

if(EXISTS "${EIGEN_LOCAL_PATH}/Eigen/Dense")
    message(STATUS ">> Linear Algebra: Menggunakan Eigen 3.4.0 Lokal dari ${EIGEN_LOCAL_PATH}")
    include_directories("${EIGEN_LOCAL_PATH}")
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE "${EIGEN_LOCAL_PATH}")
else()
    message(STATUS ">> Eigen lokal tidak ditemukan. Mencari Eigen 3.4.0 di Conda...")
    find_package(Eigen3 3.4.0 EXACT CONFIG HINTS "$ENV{CONDA_PREFIX}/share/eigen3/cmake" "/home/syahrul/miniconda3/share/eigen3/cmake")
    if(Eigen3_FOUND)
        message(STATUS ">> Linear Algebra: Eigen ${Eigen3_VERSION} ditemukan dari Conda!")
    else()
        find_package(Eigen3 3.4 REQUIRED NO_MODULE)
        message(STATUS ">> Linear Algebra: Eigen ${Eigen3_VERSION} ditemukan sebagai fallback.")
    endif()
endif()

# libcint
find_library(CINT_LIBRARY cint HINTS "$ENV{CONDA_PREFIX}/lib" "/home/syahrul/miniconda3/lib" "/usr/lib64" "/usr/lib")
if(CINT_LIBRARY)
    message(STATUS ">> Integral Engine: libcint ditemukan!")
else()
    message(FATAL_ERROR "CRITICAL: libcint tidak ditemukan!")
endif()

# TBLIS
find_library(TBLIS_LIBRARY tblis HINTS "$ENV{CONDA_PREFIX}/lib" "/home/syahrul/miniconda3/lib" "/usr/lib64" "/usr/lib")
if(TBLIS_LIBRARY)
    message(STATUS ">> Tensor Engine: TBLIS ditemukan!")
else()
    message(FATAL_ERROR "CRITICAL: TBLIS tidak ditemukan!")
endif()

# --- [MODIFIKASI TAHAP 1]: HDF5 ---
find_package(HDF5 REQUIRED COMPONENTS CXX)
if(HDF5_FOUND)
    message(STATUS ">> Out-of-Core Engine: HDF5 (v${HDF5_VERSION}) ditemukan!")
    include_directories(${HDF5_INCLUDE_DIRS})
endif()

# --- [MODIFIKASI TAHAP 1]: Jemalloc ---
find_library(JEMALLOC_LIB jemalloc HINTS "$ENV{CONDA_PREFIX}/lib" "/home/syahrul/miniconda3/lib" "/usr/lib64" "/usr/lib")
if(JEMALLOC_LIB)
    message(STATUS ">> Memori Allocator: Jemalloc ditemukan!")
    # Menonaktifkan fungsi malloc bawaan agar glibc menggunakan Jemalloc sepenuhnya
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
endif()

# ==============================================================================
# 4. KOLEKSI SOURCE CODE (DENGAN AUTO-REFRESH KETIKA ADA FILE BARU)
# ==============================================================================
file(GLOB_RECURSE SRC_CORE CONFIGURE_DEPENDS "src/core/*.cc")
file(GLOB_RECURSE SRC_SCF CONFIGURE_DEPENDS "src/scf/*.cc")
file(GLOB_RECURSE SRC_MP CONFIGURE_DEPENDS "src/mp/*.cc" "src/mp2/*.cc" "src/mp3/*.cc")
file(GLOB_RECURSE SRC_MCSCF CONFIGURE_DEPENDS "src/mcscf/*.cc")
file(GLOB_RECURSE SRC_INTEGRALS CONFIGURE_DEPENDS "src/integrals/*.cc" "src/ints/*.cc")
file(GLOB_RECURSE SRC_FOUNDATION CONFIGURE_DEPENDS "src/foundation/*.cc")
file(GLOB_RECURSE SRC_GRADIENT CONFIGURE_DEPENDS "src/gradient/*.cc")
file(GLOB_RECURSE SRC_INTEGRATION CONFIGURE_DEPENDS "src/integration/*.cc")
file(GLOB_RECURSE SRC_CI CONFIGURE_DEPENDS "src/ci/*.cc")
file(GLOB_RECURSE SRC_SYMMETRY CONFIGURE_DEPENDS "src/symmetry/*.cc")
file(GLOB_RECURSE SRC_UTILS CONFIGURE_DEPENDS "src/utils/*.cc")

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

target_compile_definitions(mshqc PUBLIC 
    EIGEN_USE_BLAS 
    EIGEN_USE_LAPACKE
    EIGEN_NO_DEBUG 
    EIGEN_MPL2_ONLY
)

target_link_libraries(mshqc PUBLIC 
    Eigen3::Eigen
    ${CINT_LIBRARY}
    ${TBLIS_LIBRARY}
    ${HDF5_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    OpenMP::OpenMP_CXX   
)

if(JEMALLOC_LIB)
    target_link_libraries(mshqc PUBLIC ${JEMALLOC_LIB})
endif()

target_include_directories(mshqc PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${HDF5_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:include>
)

# ==============================================================================
# 6. TARGET: PYTHON MODULE (_mshqc)
# ==============================================================================
nanobind_add_module(_mshqc "python/bindings.cc")

# Pastikan module Python mengikat Jemalloc jika tersedia
if(JEMALLOC_LIB)
    target_link_libraries(_mshqc PRIVATE mshqc ${CINT_LIBRARY} ${TBLIS_LIBRARY} ${JEMALLOC_LIB})
else()
    target_link_libraries(_mshqc PRIVATE mshqc ${CINT_LIBRARY} ${TBLIS_LIBRARY})
endif()

set_target_properties(_mshqc PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/mshqc"
    PREFIX "" 
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)


if(ipo_supported)
    set_target_properties(mshqc PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_target_properties(_mshqc PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
# -----------------------

# ==============================================================================
# 7. INSTALASI
# ==============================================================================
install(TARGETS mshqc _mshqc
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/mshqc DESTINATION include)