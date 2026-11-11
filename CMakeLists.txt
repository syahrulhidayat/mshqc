cmake_minimum_required(VERSION 3.14)

project(mshqc VERSION 1.0.4 LANGUAGES CXX C)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ==============================================================================
# 0. OS & COMPILER SPECIFIC FLAGS
# ==============================================================================
if(MSVC)
    # Konfigurasi Windows (MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    # /openmp:llvm mengizinkan tipe unsigned (size_t) pada loop OpenMP seperti di GCC
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /Oi /Ot /arch:AVX2 /fp:fast /openmp:llvm /Zc:__cplusplus /permissive- /DNDEBUG /EHsc /bigobj")
    add_compile_definitions(_USE_MATH_DEFINES NOMINMAX __restrict__=__restrict H5_BUILT_AS_DYNAMIC_LIB)
elseif(APPLE)
    # Konfigurasi macOS (Apple Silicon ARM64 & Intel x86_64)
    if(DEFINED ENV{GITHUB_ACTIONS} OR DEFINED ENV{CIBUILDWHEEL})
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g -fno-omit-frame-pointer -funroll-loops -ftree-vectorize -DNDEBUG -fno-math-errno")
        else()
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g -fno-omit-frame-pointer -mavx2 -mfma -funroll-loops -ftree-vectorize -mtune=generic -DNDEBUG -fno-math-errno")
        endif()
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g -fno-omit-frame-pointer -mcpu=native -funroll-loops -ftree-vectorize -DNDEBUG -fno-math-errno")
    endif()
else()
    # Konfigurasi Linux (TETAP UTUH 100% SESUAI ASLINYA)
    if(DEFINED ENV{GITHUB_ACTIONS} OR DEFINED ENV{CIBUILDWHEEL})
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g -fno-omit-frame-pointer -mavx2 -mfma -mno-avx512f -funroll-loops -ftree-vectorize -mtune=generic -DNDEBUG -fno-math-errno")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g -fno-omit-frame-pointer -march=native -funroll-loops -ftree-vectorize -DNDEBUG -fno-math-errno")
    endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 0)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)

# ==============================================================================
# 1. PYTHON & NANOBIND DISCOVERY
# ==============================================================================
# Wajib agar CMake di Windows/macOS memilih Python milik cibuildwheel (bukan Python 3.15 sistem)
set(Python_FIND_STRATEGY LOCATION)
set(Python_FIND_VIRTUALENV FIRST)

if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    set(Python_ROOT_DIR "$ENV{CONDA_PREFIX}")
    if(WIN32)
        set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/python.exe")
        set(Python3_EXECUTABLE "$ENV{CONDA_PREFIX}/python.exe")
        link_directories("$ENV{CONDA_PREFIX}/Library/lib" "$ENV{CONDA_PREFIX}/libs")
    else()
        set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
        set(Python3_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
        link_directories("$ENV{CONDA_PREFIX}/lib")
    endif()
elseif(EXISTS "/opt/conda")
    set(Python_ROOT_DIR "/opt/conda")
    link_directories("/opt/conda/lib")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

if(NOT nanobind_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
        OUTPUT_VARIABLE nanobind_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

find_package(nanobind CONFIG REQUIRED HINTS "${nanobind_DIR}" "$ENV{CONDA_PREFIX}/lib/python3.14/site-packages/nanobind/cmake")

if(APPLE)
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
    elseif(EXISTS "/usr/local/opt/libomp")
        set(OpenMP_ROOT "/usr/local/opt/libomp")
    endif()
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${OpenMP_ROOT}/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${OpenMP_ROOT}/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${OpenMP_ROOT}/lib/libomp.dylib")
endif()

find_package(OpenMP REQUIRED)

# ==============================================================================
# 2. BLAS / LAPACK / MKL DISCOVERY
# ==============================================================================
if(UNIX AND NOT APPLE)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".so.2" ".a")
elseif(APPLE)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".dylib" ".a")
elseif(WIN32)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll.a" ".dll" ".a")
endif()

if(NOT MKL_LIBRARY)
    find_library(MKL_LIBRARY
        NAMES mkl_rt mkl_rt.lib libmkl_rt.so libmkl_rt.so.2 libmkl_rt.dylib
        HINTS
            "${Python_ROOT_DIR}/lib"
            "${Python_ROOT_DIR}/Library/lib"
            "${Python3_ROOT_DIR}/lib"
            "${Python3_ROOT_DIR}/Library/lib"
            "$ENV{CONDA_PREFIX}/lib"
            "$ENV{CONDA_PREFIX}/Library/lib"
            "C:/deps/Library/lib"
            "/opt/conda/lib"
            "/usr/local/lib"
            "/usr/local/lib64"
            "/usr/lib64"
            "/usr/lib/x86_64-linux-gnu"
            "/usr/lib"
    )
endif()

if(NOT MKL_INCLUDE_DIR)
    find_path(MKL_INCLUDE_DIR
        NAMES mkl.h
        HINTS
            "${Python_ROOT_DIR}/include"
            "${Python_ROOT_DIR}/Library/include"
            "${Python3_ROOT_DIR}/include"
            "${Python3_ROOT_DIR}/Library/include"
            "$ENV{CONDA_PREFIX}/include"
            "$ENV{CONDA_PREFIX}/Library/include"
            "$ENV{CONDA_PREFIX}/include/mkl"
            "C:/deps/Library/include"
            "/opt/conda/include"
            "/usr/local/include"
            "/usr/include"
    )
endif()

if(MKL_LIBRARY AND MKL_INCLUDE_DIR)
    message(STATUS "Found MKL: ${MKL_LIBRARY} (include: ${MKL_INCLUDE_DIR})")
    set(BLAS_LIBRARIES ${MKL_LIBRARY})
    set(LAPACK_LIBRARIES ${MKL_LIBRARY})
else()
    find_library(BLIS_LIBRARY NAMES blis-mt blis HINTS "$ENV{CONDA_PREFIX}/lib" "/opt/homebrew/lib" "/opt/homebrew/opt/openblas/lib" "/usr/local/lib" "/usr/local/lib64" "/usr/lib64" "/usr/lib/x86_64-linux-gnu" "/usr/lib")
    find_library(FLAME_LIBRARY NAMES flame HINTS "$ENV{CONDA_PREFIX}/lib" "/opt/homebrew/lib" "/usr/local/lib" "/usr/local/lib64" "/usr/lib64" "/usr/lib/x86_64-linux-gnu" "/usr/lib")

    if(BLIS_LIBRARY)
        set(BLAS_LIBRARIES ${BLIS_LIBRARY})
    else()
        find_package(BLAS REQUIRED)
    endif()

    if(FLAME_LIBRARY)
        set(LAPACK_LIBRARIES ${FLAME_LIBRARY})
    else()
        find_package(LAPACK REQUIRED)
        find_library(LAPACKE_LIBRARY NAMES lapacke HINTS "$ENV{CONDA_PREFIX}/lib" "/opt/conda/lib" "/opt/homebrew/opt/lapack/lib" "/opt/homebrew/opt/openblas/lib" "/usr/local/opt/lapack/lib" "/usr/lib64" "/usr/lib/x86_64-linux-gnu" "/usr/lib")
        if(LAPACKE_LIBRARY)
            list(APPEND LAPACK_LIBRARIES ${LAPACKE_LIBRARY})
        elseif(NOT APPLE)
            message(FATAL_ERROR "CRITICAL: LAPACKE tidak ditemukan di sistem!")
        endif()
    endif()
endif()

# ==============================================================================
# 3. EIGEN DISCOVERY
# ==============================================================================
find_path(EIGEN3_INCLUDE_DIR
    NAMES Eigen/Dense
    HINTS
        "${CMAKE_SOURCE_DIR}/include/eigen-3.4.0"
        "$ENV{CONDA_PREFIX}/include/eigen3"
        "$ENV{CONDA_PREFIX}/include"
        "$ENV{CONDA_PREFIX}/Library/include/eigen3"
        "$ENV{CONDA_PREFIX}/Library/include"
        "C:/deps/Library/include/eigen3"
        "C:/deps/Library/include"
        "/opt/homebrew/include/eigen3"
        "/opt/homebrew/opt/eigen/include/eigen3"
        "/usr/local/include/eigen3"
        "/usr/local/include"
        "/usr/include/eigen3"
        "/usr/include"
)
if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "CRITICAL: Eigen/Dense header tidak ditemukan!")
endif()
if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE "${EIGEN3_INCLUDE_DIR}")
endif()

# ==============================================================================
# 4. LIBCINT & TBLIS DISCOVERY
# ==============================================================================
find_library(CINT_LIBRARY
    NAMES cint libcint.a cint.lib libcint.dll.a
    HINTS "$ENV{CONDA_PREFIX}/lib" "$ENV{CONDA_PREFIX}/Library/lib" "C:/deps/Library/lib" "$ENV{HOME}/libcint_install/lib64" "$ENV{HOME}/libcint_install/lib" "/opt/homebrew/lib" "/usr/local/lib64" "/usr/local/lib" "/usr/lib64" "/usr/lib"
)
find_path(CINT_INCLUDE_DIR
    NAMES cint.h
    HINTS "$ENV{CONDA_PREFIX}/include" "$ENV{CONDA_PREFIX}/Library/include" "C:/deps/Library/include" "$ENV{HOME}/libcint_install/include" "/opt/homebrew/include" "/usr/local/include" "/usr/include"
)

if(NOT CINT_LIBRARY OR NOT CINT_INCLUDE_DIR)
    message(FATAL_ERROR "CRITICAL: libcint library atau cint.h header tidak ditemukan!")
endif()

find_library(TBLIS_LIBRARY
    NAMES tblis libtblis.a tblis.lib libtblis.dll.a
    HINTS "$ENV{TBLIS_DIR}/lib64" "$ENV{TBLIS_DIR}/lib" "$ENV{CONDA_PREFIX}/lib" "$ENV{CONDA_PREFIX}/Library/lib" "C:/deps/Library/lib" "$ENV{HOME}/tblis_install/lib64" "$ENV{HOME}/tblis_install/lib" "/opt/homebrew/lib" "/usr/local/lib64" "/usr/local/lib" "/usr/lib64" "/usr/lib"
)
find_library(TCI_LIBRARY
    NAMES tci libtci.a tci.lib libtci.dll.a
    HINTS "$ENV{TBLIS_DIR}/lib64" "$ENV{TBLIS_DIR}/lib" "$ENV{CONDA_PREFIX}/lib" "$ENV{CONDA_PREFIX}/Library/lib" "C:/deps/Library/lib" "$ENV{HOME}/tblis_install/lib64" "$ENV{HOME}/tblis_install/lib" "/opt/homebrew/lib" "/usr/local/lib64" "/usr/local/lib" "/usr/lib64" "/usr/lib"
)
find_path(TBLIS_INCLUDE_DIR
    NAMES tblis/tblis.h
    HINTS "$ENV{TBLIS_DIR}/include" "$ENV{CONDA_PREFIX}/include" "$ENV{CONDA_PREFIX}/Library/include" "C:/deps/Library/include" "$ENV{HOME}/tblis_install/include" "/opt/homebrew/include" "/usr/local/include" "/usr/include"
)

if(NOT TBLIS_LIBRARY OR NOT TBLIS_INCLUDE_DIR)
    message(FATAL_ERROR "CRITICAL: TBLIS library atau header tidak ditemukan!")
endif()

# ==============================================================================
# 5. HDF5 DISCOVERY
# ==============================================================================
find_library(HDF5_C_LIB
    NAMES hdf5 libhdf5 hdf5.lib libhdf5.lib
    HINTS "$ENV{CONDA_PREFIX}/lib" "$ENV{CONDA_PREFIX}/Library/lib" "C:/deps/Library/lib" "/opt/homebrew/opt/hdf5/lib" "/opt/homebrew/lib" "/usr/local/lib64" "/usr/local/lib" "/usr/lib64" "/usr/lib/x86_64-linux-gnu" "/usr/lib"
)
find_library(HDF5_CXX_LIB
    NAMES hdf5_cpp libhdf5_cpp hdf5_cpp.lib libhdf5_cpp.lib
    HINTS "$ENV{CONDA_PREFIX}/lib" "$ENV{CONDA_PREFIX}/Library/lib" "C:/deps/Library/lib" "/opt/homebrew/opt/hdf5/lib" "/opt/homebrew/lib" "/usr/local/lib64" "/usr/local/lib" "/usr/lib64" "/usr/lib/x86_64-linux-gnu" "/usr/lib"
)
find_path(HDF5_INCLUDE_DIR
    NAMES H5Cpp.h
    HINTS "$ENV{CONDA_PREFIX}/include" "$ENV{CONDA_PREFIX}/Library/include" "C:/deps/Library/include" "/opt/homebrew/opt/hdf5/include" "/opt/homebrew/include" "/usr/local/include" "/usr/include" "/usr/include/hdf5/serial"
)

if(HDF5_C_LIB AND HDF5_CXX_LIB)
    set(HDF5_LIBRARIES ${HDF5_CXX_LIB} ${HDF5_C_LIB})
else()
    message(FATAL_ERROR "CRITICAL: HDF5 C/C++ libraries tidak ditemukan!")
endif()

# ==============================================================================
# 6. SOURCE FILES & TARGET CONFIGURATION
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

add_library(mshqc STATIC ${ALL_SOURCES})

if(APPLE)
    set(MSHQC_RPATH "@loader_path;@loader_path/../../..")
else()
    set(MSHQC_RPATH "$ORIGIN:$ORIGIN/../../..")
endif()

set_target_properties(mshqc PROPERTIES 
    INSTALL_RPATH "${MSHQC_RPATH}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

if(MKL_LIBRARY AND MKL_INCLUDE_DIR)
    target_compile_definitions(mshqc PUBLIC EIGEN_USE_MKL_ALL)
endif()
target_compile_definitions(mshqc PUBLIC EIGEN_NO_DEBUG EIGEN_MPL2_ONLY)

set(MSHQC_LINK_LIBS 
    Eigen3::Eigen
    ${CINT_LIBRARY}
    ${TBLIS_LIBRARY}
    ${HDF5_LIBRARIES} 
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    OpenMP::OpenMP_CXX
)

if(UNIX AND NOT APPLE)
    list(APPEND MSHQC_LINK_LIBS quadmath atomic)
endif()

if(TCI_LIBRARY)
    list(APPEND MSHQC_LINK_LIBS ${TCI_LIBRARY})
endif()

target_link_libraries(mshqc PUBLIC ${MSHQC_LINK_LIBS})

# Kumpulkan candidate include directories secara aman per-OS agar C:/ atau NOTFOUND tidak bocor ke Linux/macOS
set(MSHQC_BUILD_INCLUDES
    "${CMAKE_SOURCE_DIR}/include"
    "${EIGEN3_INCLUDE_DIR}"
    "${CINT_INCLUDE_DIR}"
    "${TBLIS_INCLUDE_DIR}"
    "${HDF5_INCLUDE_DIR}"
    "${Python_ROOT_DIR}/include"
)

if(MKL_INCLUDE_DIR AND EXISTS "${MKL_INCLUDE_DIR}")
    list(APPEND MSHQC_BUILD_INCLUDES "${MKL_INCLUDE_DIR}")
endif()

if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    list(APPEND MSHQC_BUILD_INCLUDES
        "$ENV{CONDA_PREFIX}/include"
        "$ENV{CONDA_PREFIX}/Library/include"
    )
endif()

if(WIN32)
    list(APPEND MSHQC_BUILD_INCLUDES "C:/deps/Library/include")
endif()

target_include_directories(mshqc PUBLIC $<INSTALL_INTERFACE:include>)
foreach(_inc IN LISTS MSHQC_BUILD_INCLUDES)
    if(_inc AND IS_ABSOLUTE "${_inc}" AND EXISTS "${_inc}")
        target_include_directories(mshqc PUBLIC $<BUILD_INTERFACE:${_inc}>)
    endif()
endforeach()

nanobind_add_module(_mshqc "python/bindings.cc")

foreach(_inc IN LISTS MSHQC_BUILD_INCLUDES)
    if(_inc AND IS_ABSOLUTE "${_inc}" AND EXISTS "${_inc}")
        target_include_directories(_mshqc PRIVATE "${_inc}")
    endif()
endforeach()

set(MSHQC_MODULE_LIBS mshqc ${CINT_LIBRARY} ${TBLIS_LIBRARY} ${TCI_LIBRARY} ${BLAS_LIBRARIES})
if(UNIX AND NOT APPLE)
    list(APPEND MSHQC_MODULE_LIBS quadmath atomic)
endif()
target_link_libraries(_mshqc PRIVATE ${MSHQC_MODULE_LIBS})

# Di Windows (MSVC), RUNTIME_OUTPUT_DIRECTORY_RELEASE wajib disetel agar file .pyd tidak masuk ke subfolder Release/
set_target_properties(_mshqc PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/mshqc"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/mshqc"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/mshqc"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/mshqc"
    PREFIX "" 
    INSTALL_RPATH "${MSHQC_RPATH}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

if(ipo_supported AND NOT DEFINED ENV{CIBUILDWHEEL} AND NOT DEFINED ENV{GITHUB_ACTIONS})
    set_target_properties(mshqc PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_target_properties(_mshqc PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

install(TARGETS mshqc _mshqc
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

if(EXISTS "${CMAKE_SOURCE_DIR}/include/mshqc")
    install(DIRECTORY include/mshqc DESTINATION include)
endif()
