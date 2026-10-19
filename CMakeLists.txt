cmake_minimum_required(VERSION 3.14)

project(mshqc VERSION 1.0.4 LANGUAGES CXX C)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(DEFINED ENV{GITHUB_ACTIONS} OR DEFINED ENV{CIBUILDWHEEL})
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mavx2 -mfma -mno-avx512f -funroll-loops -ftree-vectorize -mtune=generic -DNDEBUG -fno-math-errno")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -funroll-loops -ftree-vectorize -DNDEBUG -fno-math-errno")
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
if(DEFINED ENV{CONDA_PREFIX})
    set(Python_ROOT_DIR "$ENV{CONDA_PREFIX}")
    set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
    set(Python3_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
    link_directories("$ENV{CONDA_PREFIX}/lib")
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
find_package(OpenMP REQUIRED)

# ==============================================================================
# 2. BLAS / LAPACK / MKL DISCOVERY
# ==============================================================================
set(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".so.2" ".a")

if(NOT MKL_LIBRARY)
    find_library(MKL_LIBRARY
        NAMES mkl_rt libmkl_rt.so libmkl_rt.so.2
        HINTS
            "${Python_ROOT_DIR}/lib"
            "${Python3_ROOT_DIR}/lib"
            "$ENV{CONDA_PREFIX}/lib"
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
            "${Python3_ROOT_DIR}/include"
            "$ENV{CONDA_PREFIX}/include"
            "$ENV{CONDA_PREFIX}/include/mkl"
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
    find_library(BLIS_LIBRARY NAMES blis-mt blis HINTS "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib" "/usr/local/lib64" "/usr/lib64" "/usr/lib/x86_64-linux-gnu" "/usr/lib")
    find_library(FLAME_LIBRARY NAMES flame HINTS "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib" "/usr/local/lib64" "/usr/lib64" "/usr/lib/x86_64-linux-gnu" "/usr/lib")

    if(BLIS_LIBRARY)
        set(BLAS_LIBRARIES ${BLIS_LIBRARY})
    else()
        find_package(BLAS REQUIRED)
    endif()

    if(FLAME_LIBRARY)
        set(LAPACK_LIBRARIES ${FLAME_LIBRARY})
    else()
        find_package(LAPACK REQUIRED)
        find_library(LAPACKE_LIBRARY NAMES lapacke HINTS "$ENV{CONDA_PREFIX}/lib" "/opt/conda/lib" "/usr/lib64" "/usr/lib/x86_64-linux-gnu" "/usr/lib")
        if(LAPACKE_LIBRARY)
            list(APPEND LAPACK_LIBRARIES ${LAPACKE_LIBRARY})
        else()
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
    NAMES cint libcint.a
    HINTS "$ENV{CONDA_PREFIX}/lib" "$ENV{HOME}/libcint_install/lib64" "$ENV{HOME}/libcint_install/lib" "/usr/local/lib64" "/usr/local/lib" "/usr/lib64" "/usr/lib"
)
find_path(CINT_INCLUDE_DIR
    NAMES cint.h
    HINTS "$ENV{CONDA_PREFIX}/include" "$ENV{HOME}/libcint_install/include" "/usr/local/include" "/usr/include"
)

if(NOT CINT_LIBRARY OR NOT CINT_INCLUDE_DIR)
    message(FATAL_ERROR "CRITICAL: libcint library atau cint.h header tidak ditemukan!")
endif()

find_library(TBLIS_LIBRARY
    NAMES tblis libtblis.a
    HINTS "$ENV{TBLIS_DIR}/lib64" "$ENV{TBLIS_DIR}/lib" "$ENV{CONDA_PREFIX}/lib" "$ENV{HOME}/tblis_install/lib64" "$ENV{HOME}/tblis_install/lib" "/usr/local/lib64" "/usr/local/lib" "/usr/lib64" "/usr/lib"
)
find_library(TCI_LIBRARY
    NAMES tci libtci.a
    HINTS "$ENV{TBLIS_DIR}/lib64" "$ENV{TBLIS_DIR}/lib" "$ENV{CONDA_PREFIX}/lib" "$ENV{HOME}/tblis_install/lib64" "$ENV{HOME}/tblis_install/lib" "/usr/local/lib64" "/usr/local/lib" "/usr/lib64" "/usr/lib"
)
find_path(TBLIS_INCLUDE_DIR
    NAMES tblis/tblis.h
    HINTS "$ENV{TBLIS_DIR}/include" "$ENV{CONDA_PREFIX}/include" "$ENV{HOME}/tblis_install/include" "/usr/local/include" "/usr/include"
)

if(NOT TBLIS_LIBRARY OR NOT TBLIS_INCLUDE_DIR)
    message(FATAL_ERROR "CRITICAL: TBLIS library atau header tidak ditemukan!")
endif()

# ==============================================================================
# 5. HDF5 DISCOVERY
# ==============================================================================
find_library(HDF5_C_LIB
    NAMES hdf5
    HINTS "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib64" "/usr/local/lib" "/usr/lib64" "/usr/lib/x86_64-linux-gnu" "/usr/lib"
)
find_library(HDF5_CXX_LIB
    NAMES hdf5_cpp
    HINTS "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib64" "/usr/local/lib" "/usr/lib64" "/usr/lib/x86_64-linux-gnu" "/usr/lib"
)
find_path(HDF5_INCLUDE_DIR
    NAMES H5Cpp.h
    HINTS "$ENV{CONDA_PREFIX}/include" "/usr/local/include" "/usr/include" "/usr/include/hdf5/serial"
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

set_target_properties(mshqc PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/mshqc"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../../.."
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
    quadmath
    atomic
)
if(TCI_LIBRARY)
    list(APPEND MSHQC_LINK_LIBS ${TCI_LIBRARY})
endif()

target_link_libraries(mshqc PUBLIC ${MSHQC_LINK_LIBS})

target_include_directories(mshqc PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CINT_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${TBLIS_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${HDF5_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${MKL_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${Python_ROOT_DIR}/include>
    $<BUILD_INTERFACE:$ENV{CONDA_PREFIX}/include>
    $<INSTALL_INTERFACE:include>
)

nanobind_add_module(_mshqc "python/bindings.cc")

target_include_directories(_mshqc PRIVATE "${EIGEN3_INCLUDE_DIR}" "${MKL_INCLUDE_DIR}")
target_link_libraries(_mshqc PRIVATE mshqc ${CINT_LIBRARY} ${TBLIS_LIBRARY} ${TCI_LIBRARY} ${BLAS_LIBRARIES} quadmath atomic)

set_target_properties(_mshqc PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/mshqc"
    PREFIX "" 
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../../.."
    BUILD_WITH_INSTALL_RPATH TRUE
)

if(ipo_supported AND NOT DEFINED ENV{CIBUILDWHEEL})
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