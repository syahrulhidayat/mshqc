cmake_minimum_required(VERSION 3.14)

# Informasi Proyek
project(mshqc VERSION 1.0.0 LANGUAGES CXX)

# Standar C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) 

# ==============================================================================
# 0. SUPPRESS WARNINGS
# ==============================================================================
if(MSVC)
    add_compile_options(/wd4838)
else()
    add_compile_options(-Wno-narrowing) 
endif()

# ==============================================================================
# 1. PENGATURAN DEPENDENSI
# ==============================================================================
find_package(pybind11 REQUIRED)

include_directories(
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/include/eigen-3.4.0"
    "${CMAKE_SOURCE_DIR}/include/libint2"
    "/usr/include/eigen3"
    "/usr/include/libint2"
)

find_library(LIBINT2_LIBRARY NAMES int2 libint2 
    HINTS "${CMAKE_SOURCE_DIR}/lib" "/usr/local/lib" "/usr/lib")

if(NOT LIBINT2_LIBRARY)
    message(WARNING "Library Libint2 tidak ditemukan.")
endif()

# ==============================================================================
# 2. MENGUMPULKAN SOURCE CODE
# ==============================================================================
file(GLOB_RECURSE SRC_CORE "src/core/*.cc")
file(GLOB_RECURSE SRC_SCF "src/scf/*.cc")
# Note: GLOB_RECURSE src/mp/*.cc sudah mencakup subfolder mp2/mp3 jika ada di dalamnya
file(GLOB_RECURSE SRC_MP "src/mp/*.cc" "src/mp2/*.cc" "src/mp3/*.cc")
file(GLOB_RECURSE SRC_MCSCF "src/mcscf/*.cc")
file(GLOB_RECURSE SRC_INTEGRALS "src/integrals/*.cc")
file(GLOB_RECURSE SRC_FOUNDATION "src/foundation/*.cc")
file(GLOB_RECURSE SRC_GRADIENT "src/gradient/*.cc")
file(GLOB_RECURSE SRC_INTEGRATION "src/integration/*.cc")
file(GLOB_RECURSE SRC_CI "src/ci/*.cc")

set(ALL_SOURCES
    ${SRC_CORE} ${SRC_SCF} ${SRC_MP} ${SRC_MCSCF}
    ${SRC_INTEGRALS} ${SRC_FOUNDATION} ${SRC_GRADIENT}
    ${SRC_INTEGRATION} ${SRC_CI}
)

# ==============================================================================
# 3. PENGATURAN FITUR (Exclude mp_density & CI berat)
# ==============================================================================
option(ENABLE_FCI "Aktifkan modul FCI, CISD, dan metode berat lainnya" OFF)

set(FCI_SOURCES
    "${CMAKE_SOURCE_DIR}/src/ci/cisd.cc"
    "${CMAKE_SOURCE_DIR}/src/ci/cipsi.cc"
    "${CMAKE_SOURCE_DIR}/src/ci/cis.cc"
    "${CMAKE_SOURCE_DIR}/src/ci/cisdt.cc"
    "${CMAKE_SOURCE_DIR}/src/ci/mrci.cc"
    "${CMAKE_SOURCE_DIR}/src/mp/mp_density.cc"
)

if(ENABLE_FCI)
    message(STATUS ">> Build Mode: HEAVY (FCI Enabled)")
else()
    list(REMOVE_ITEM ALL_SOURCES ${FCI_SOURCES})
    message(STATUS ">> Build Mode: LIGHT (FCI & mp_density EXCLUDED)")
endif()

# ==============================================================================
# 4. MEMBUAT LIBRARY UTAMA (C++) - Target: mshqc
# ==============================================================================
add_library(mshqc SHARED ${ALL_SOURCES})

if(LIBINT2_LIBRARY)
    target_link_libraries(mshqc PUBLIC ${LIBINT2_LIBRARY})
endif()

target_include_directories(mshqc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/eigen-3.4.0>
    $<INSTALL_INTERFACE:include>
)

# ==============================================================================
# 5. MEMBUAT PYTHON MODULE - Target: _mshqc
# ==============================================================================
# Penting: Nama di sini (_mshqc) harus BEDA dengan nama library di atas (mshqc)
pybind11_add_module(_mshqc "python/bindings.cc")

# Link ke library utama agar fungsi C++ bisa dipanggil Python
target_link_libraries(_mshqc PRIVATE mshqc)

if(LIBINT2_LIBRARY)
    target_link_libraries(_mshqc PRIVATE ${LIBINT2_LIBRARY})
endif()

# ==============================================================================
# [PENTING] Output Directory
# ==============================================================================
# Memastikan hasil compile (_mshqc.so) masuk ke folder python/mshqc/
set_target_properties(_mshqc PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/mshqc"
    PREFIX "" # Agar nama file jadi _mshqc.so, BUKAN lib_mshqc.so
)

# ==============================================================================
# 6. INSTALASI (Opsional untuk pip)
# ==============================================================================
# Koreksi: Menghapus duplikasi nama target
install(TARGETS mshqc _mshqc
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/mshqc DESTINATION include)