cmake_minimum_required(VERSION 3.14)




project(mshqc VERSION 1.0.0 LANGUAGES CXX C)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()




if(DEFINED ENV{GITHUB_ACTIONS})
    
    
    
    message(STATUS ">> CI/CD Server Detected: Menggunakan optimasi Biner Universal (AVX2/FMA Baseline)")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mavx2 -mfma -mno-avx512f -funroll-loops -ftree-vectorize -mtune=generic -DNDEBUG -fno-math-errno")
else()
    
    message(STATUS ">> Local Environment Detected: Mengaktifkan Super Turbo (-march=native)")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -funroll-loops -ftree-vectorize -DNDEBUG -fno-math-errno")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 0)


include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)

    
    





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




find_package(OpenMP REQUIRED)



find_library(MKL_LIBRARY NAMES mkl_rt HINTS "$ENV{CONDA_PREFIX}/lib" "/opt/conda/lib" "/usr/local/lib" "/usr/lib")

if(MKL_LIBRARY)
    message(STATUS ">> Linear Algebra Engine: Intel MKL ditemukan! (${MKL_LIBRARY})")
    
    set(BLAS_LIBRARIES ${MKL_LIBRARY})
    set(LAPACK_LIBRARIES ${MKL_LIBRARY})
else()
    message(STATUS ">> Intel MKL tidak ditemukan. Jatuh ke fallback (BLIS/FLAME/Standar)...")
    find_library(BLIS_LIBRARY NAMES blis-mt blis HINTS "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib" "/usr/lib64" "/usr/lib")
    find_library(FLAME_LIBRARY NAMES flame HINTS "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib" "/usr/lib64" "/usr/lib")

    if(BLIS_LIBRARY)
        message(STATUS ">> BLAS: BLIS ditemukan!")
        set(BLAS_LIBRARIES ${BLIS_LIBRARY})
    else()
        find_package(BLAS REQUIRED)
    endif()

    if(FLAME_LIBRARY)
        message(STATUS ">> LAPACK: libflame ditemukan!")
        set(LAPACK_LIBRARIES ${FLAME_LIBRARY})
    else()
        find_package(LAPACK REQUIRED)
        
        find_library(LAPACKE_LIBRARY NAMES lapacke HINTS "$ENV{CONDA_PREFIX}/lib" "/opt/conda/lib" "/usr/lib/x86_64-linux-gnu" "/usr/lib")
        if(LAPACKE_LIBRARY)
            message(STATUS ">> LAPACKE: Antarmuka C LAPACKE ditemukan! (${LAPACKE_LIBRARY})")
            list(APPEND LAPACK_LIBRARIES ${LAPACKE_LIBRARY})
        else()
            message(FATAL_ERROR "CRITICAL: LAPACKE (lapacke) tidak ditemukan di sistem!")
        endif()
    endif()
endif()

set(EIGEN_LOCAL_PATH "${CMAKE_SOURCE_DIR}/include/eigen-3.4.0")
if(EXISTS "${EIGEN_LOCAL_PATH}/Eigen/Dense")
    message(STATUS ">> Linear Algebra: Menggunakan Eigen Lokal")
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE "${EIGEN_LOCAL_PATH}")
else()
    message(STATUS ">> Eigen lokal tidak ditemukan. Mencari versi terbaru di Conda/Sistem...")
    find_package(Eigen3 CONFIG HINTS "$ENV{CONDA_PREFIX}/share/eigen3/cmake" "/opt/conda/share/eigen3/cmake")
    if(Eigen3_FOUND)
        message(STATUS ">> Linear Algebra: Eigen ${Eigen3_VERSION} ditemukan dari Conda!")
    else()
        find_package(Eigen3 REQUIRED NO_MODULE)
        message(STATUS ">> Linear Algebra: Eigen ${Eigen3_VERSION} ditemukan sebagai fallback.")
    endif()
endif()


find_library(CINT_LIBRARY cint HINTS "$ENV{CONDA_PREFIX}/lib" "/opt/conda/lib" "/usr/local/lib" "/usr/lib")
if(CINT_LIBRARY)
    message(STATUS ">> Integral Engine: libcint ditemukan!")
else()
    message(FATAL_ERROR "CRITICAL: libcint tidak ditemukan!")
endif()


find_library(TBLIS_LIBRARY NAMES libtblis.a tblis HINTS "$ENV{TBLIS_DIR}/lib" "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib" "/usr/lib")
find_library(TCI_LIBRARY NAMES libtci.a tci HINTS "$ENV{TBLIS_DIR}/lib" "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib" "/usr/lib")

if(TBLIS_LIBRARY AND TCI_LIBRARY)
    message(STATUS ">> Tensor Engine: TBLIS & TCI ditemukan!")
    include_directories("$ENV{TBLIS_DIR}/include") 
else()
    message(FATAL_ERROR "CRITICAL: TBLIS atau TCI tidak ditemukan!")
endif()


find_path(HDF5_INCLUDE_DIRS "H5Cpp.h" HINTS "$ENV{CONDA_PREFIX}/include" NO_DEFAULT_PATH)
find_library(HDF5_C_LIB hdf5 HINTS "$ENV{CONDA_PREFIX}/lib" NO_DEFAULT_PATH)
find_library(HDF5_CXX_LIB hdf5_cpp HINTS "$ENV{CONDA_PREFIX}/lib" NO_DEFAULT_PATH)

if(HDF5_INCLUDE_DIRS AND HDF5_C_LIB AND HDF5_CXX_LIB)
    message(STATUS ">> Out-of-Core Engine: HDF5 C++ Conda ditemukan secara manual!")
    include_directories(${HDF5_INCLUDE_DIRS})
    set(HDF5_LIBRARIES ${HDF5_CXX_LIB} ${HDF5_C_LIB})
else()
    message(FATAL_ERROR "CRITICAL: HDF5 C++ tidak ditemukan di Conda mshqcenv!")
endif()


if(JEMALLOC_LIB)
    message(STATUS ">> Memori Allocator: Jemalloc ditemukan!")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
endif()




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


target_compile_definitions(mshqc PUBLIC 
    EIGEN_USE_MKL_ALL
    EIGEN_NO_DEBUG 
    EIGEN_MPL2_ONLY
)

target_link_libraries(mshqc PUBLIC 
    Eigen3::Eigen
    ${CINT_LIBRARY}
    ${TBLIS_LIBRARY}
    ${TCI_LIBRARY}         
    ${HDF5_LIBRARIES} 
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    OpenMP::OpenMP_CXX   
    atomic
)

if(JEMALLOC_LIB)
    target_link_libraries(mshqc PUBLIC ${JEMALLOC_LIB})
endif()

target_include_directories(mshqc PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${HDF5_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:include>
)




nanobind_add_module(_mshqc "python/bindings.cc")

if(JEMALLOC_LIB)
    target_link_libraries(_mshqc PRIVATE mshqc ${CINT_LIBRARY} ${TBLIS_LIBRARY} ${JEMALLOC_LIB} atomic) 
else()
    target_link_libraries(_mshqc PRIVATE mshqc ${CINT_LIBRARY} ${TBLIS_LIBRARY} atomic)
endif()

set_target_properties(_mshqc PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/mshqc"
    PREFIX "" 
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../../.."
    BUILD_WITH_INSTALL_RPATH TRUE
)

if(ipo_supported)
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
