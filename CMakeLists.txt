# ==============================================================================
# Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
cmake_minimum_required(VERSION 3.18)

project(mshqc VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Standardize Symbol Visibility for Optimized IPO/LTO
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# ==============================================================================
# BUILD OPTIONS
# ==============================================================================
option(MSHQC_ENABLE_NATIVE_OPT "Enable hardware-specific optimization (-march=native)" OFF)
option(MSHQC_ENABLE_IPO "Enable Interprocedural Optimization (LTO)" ON)

if(DEFINED ENV{GITHUB_ACTIONS})
    set(MSHQC_ENABLE_NATIVE_OPT OFF CACHE BOOL "" FORCE)
endif()

# ==============================================================================
# DEPENDENCY DISCOVERY (HERMETIC & TARGET-BASED)
# ==============================================================================
# Python & Nanobind
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# BLAS & LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# LLVM & MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Restorasi inklusi direktori secara global untuk utilitas TableGen (mlir-tblgen)
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
# Domain Libraries: Eigen3, CINT, TBLIS, HDF5
# Mekanisme fallback hermetis untuk melewati anomali versi registri Conda
set(EIGEN_LOCAL_PATH "${CMAKE_SOURCE_DIR}/include/eigen-3.4.0")
if(EXISTS "${EIGEN_LOCAL_PATH}/Eigen/Dense")
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE "${EIGEN_LOCAL_PATH}")
else()
    find_package(Eigen3 CONFIG REQUIRED)
endif()

# ==============================================================================
# Domain Libraries: CINT, TBLIS (Dengan Hermetic HPC Path Resolution)
# ==============================================================================
find_path(CINT_INCLUDE_DIR NAMES cint.h 
    HINTS "$ENV{CONDA_PREFIX}/include" "$ENV{HOME}/libcint_install/include" "/usr/local/include" "/usr/include" REQUIRED)
find_library(CINT_LIBRARY NAMES cint 
    HINTS "$ENV{CONDA_PREFIX}/lib" "$ENV{HOME}/libcint_install/lib64" "$ENV{HOME}/libcint_install/lib" "/usr/local/lib64" "/usr/local/lib" "/usr/lib64" "/usr/lib" REQUIRED)

find_path(TBLIS_INCLUDE_DIR NAMES tblis/tblis.h 
    HINTS "$ENV{TBLIS_DIR}/include" "$ENV{CONDA_PREFIX}/include" "$ENV{HOME}/tblis_install/include" "/usr/local/include" "/usr/include" REQUIRED)
find_library(TBLIS_LIBRARY NAMES tblis libtblis.a 
    HINTS "$ENV{TBLIS_DIR}/lib64" "$ENV{TBLIS_DIR}/lib" "$ENV{CONDA_PREFIX}/lib" "$ENV{HOME}/tblis_install/lib64" "$ENV{HOME}/tblis_install/lib" "/usr/lib64" "/usr/lib" REQUIRED)
find_library(TCI_LIBRARY NAMES tci libtci.a 
    HINTS "$ENV{TBLIS_DIR}/lib64" "$ENV{TBLIS_DIR}/lib" "$ENV{CONDA_PREFIX}/lib" "$ENV{HOME}/tblis_install/lib64" "$ENV{HOME}/tblis_install/lib" "/usr/lib64" "/usr/lib")

find_package(HDF5 COMPONENTS C CXX REQUIRED)

# ==============================================================================
# COMPILER / SUBDIRECTORIES
# ==============================================================================
llvm_map_components_to_libnames(llvm_libs support core orcjit native)

add_subdirectory(src/compiler)
add_subdirectory(tools)

# ==============================================================================
# TARGET DEFINITION: mshqc STATIC LIBRARY
# ==============================================================================
file(GLOB_RECURSE MSHQC_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/core/*.cc"
    "${CMAKE_SOURCE_DIR}/src/scf/*.cc"
    "${CMAKE_SOURCE_DIR}/src/mp/*.cc"
    "${CMAKE_SOURCE_DIR}/src/mp2/*.cc"
    "${CMAKE_SOURCE_DIR}/src/mp3/*.cc"
    "${CMAKE_SOURCE_DIR}/src/integrals/*.cc"
    "${CMAKE_SOURCE_DIR}/src/ints/*.cc"
    "${CMAKE_SOURCE_DIR}/src/gradient/*.cc"
    "${CMAKE_SOURCE_DIR}/src/symmetry/*.cc"
    "${CMAKE_SOURCE_DIR}/src/utils/*.cc"
)

add_library(mshqc STATIC ${MSHQC_SOURCES})

# Target Compile Options (Compiler Agnostic & Scoped)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mshqc PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Release>:-O3 -funroll-loops -ftree-vectorize -fno-math-errno>
    )
    if(MSHQC_ENABLE_NATIVE_OPT)
        target_compile_options(mshqc PRIVATE $<$<CONFIG:Release>:-march=native>)
    else()
        target_compile_options(mshqc PRIVATE $<$<CONFIG:Release>:-mavx2 -mfma -mno-avx512f -mtune=generic>)
    endif()
endif()

target_compile_definitions(mshqc PUBLIC
    EIGEN_USE_MKL_ALL
    MSHQC_ENABLE_MLIR
    EIGEN_NO_DEBUG
    EIGEN_MPL2_ONLY
)

target_include_directories(mshqc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${CINT_INCLUDE_DIR}
    ${TBLIS_INCLUDE_DIR}
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(mshqc PUBLIC
    Eigen3::Eigen
    MLIRIR
    MLIRSupport
    ${llvm_libs}
    ${CINT_LIBRARY}
    ${TBLIS_LIBRARY}
    hdf5::hdf5 hdf5::hdf5_cpp
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    OpenMP::OpenMP_CXX
    ${CMAKE_DL_LIBS}
    atomic
)

if(TCI_LIBRARY)
    target_link_libraries(mshqc PUBLIC ${TCI_LIBRARY})
endif()

# Whole-archive handling for mshqc_compiler layer (Platform Aware)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_libraries(mshqc PUBLIC -Wl,--whole-archive mshqc_compiler -Wl,--no-whole-archive)
elseif(MSVC)
    target_link_libraries(mshqc PUBLIC /WHOLEARCHIVE:mshqc_compiler)
endif()

# ==============================================================================
# TARGET DEFINITION: PYTHON BINDINGS
# ==============================================================================
nanobind_add_module(_mshqc "python/bindings.cc")

target_link_libraries(_mshqc PRIVATE mshqc)

set_target_properties(mshqc _mshqc PROPERTIES
    # LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/mshqc" // Didelegasikan ke setuptools
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../../.."
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Interprocedural Optimization Configuration
if(MSHQC_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if(ipo_supported)
        set_target_properties(mshqc _mshqc PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()

# ==============================================================================
# INSTALLATION SPECIFICATION
# ==============================================================================
include(GNUInstallDirs)

install(TARGETS mshqc _mshqc
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(EXISTS "${CMAKE_SOURCE_DIR}/include/mshqc")
    install(DIRECTORY include/mshqc DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()