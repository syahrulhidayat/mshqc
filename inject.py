#!/usr/bin/env python3
"""
Skrip perbaikan menyeluruh untuk:
1. CMakeLists.txt
2. setup.py
3. .github/workflows/release_public.yml (Fix MSYS2 pacman + libcint/TBLIS DLL/LIB untuk MSVC)
4. src/core/basis.cc (Auto-resolve ke repo mshqc-basis)
5. python/mshqc/__init__.py (Auto-set MSHQC_BASIS_PATH ke mshqc-basis jika lokal)
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ==============================================================================
# 1. CMakeLists.txt
# ==============================================================================
CMAKELISTS_CONTENT = r'''cmake_minimum_required(VERSION 3.14)

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
    NAMES cint.lib libcint.dll.a cint libcint.a
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
    NAMES tblis.lib libtblis.dll.a tblis libtblis.a
    HINTS "$ENV{TBLIS_DIR}/lib64" "$ENV{TBLIS_DIR}/lib" "$ENV{CONDA_PREFIX}/lib" "$ENV{CONDA_PREFIX}/Library/lib" "C:/deps/Library/lib" "$ENV{HOME}/tblis_install/lib64" "$ENV{HOME}/tblis_install/lib" "/opt/homebrew/lib" "/usr/local/lib64" "/usr/local/lib" "/usr/lib64" "/usr/lib"
)
find_library(TCI_LIBRARY
    NAMES tci.lib libtci.dll.a tci libtci.a
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

# Validasi ketat direktori include agar tidak ada relative path lintas OS
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
'''

# ==============================================================================
# 2. setup.py
# ==============================================================================
SETUP_PY_CONTENT = r'''import os
import sys
import glob
import shlex
import shutil
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self) -> None:
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed and accessible in PATH to build extensions.")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: CMakeExtension) -> None:
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        if not extdir.endswith("mshqc"):
            extdir = os.path.join(extdir, "mshqc")
        os.makedirs(extdir, exist_ok=True)

        import nanobind
        nanobind_cmake_path = nanobind.cmake_dir()

        real_exe_root = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))
        exe_root = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
        candidate_roots = [
            sys.prefix,
            sys.base_prefix,
            exe_root,
            real_exe_root,
            os.environ.get("MKLROOT", ""),
            os.environ.get("CONDA_PREFIX", ""),
            "C:/deps/Library" if sys.platform == "win32" else "/usr/local",
            "C:/deps" if sys.platform == "win32" else "",
        ]

        mkl_lib_path = ""
        mkl_inc_path = ""
        for root in candidate_roots:
            if not root:
                continue
            lib_dirs = [os.path.join(root, "lib"), os.path.join(root, "Library", "lib")]
            inc_dirs = [os.path.join(root, "include"), os.path.join(root, "Library", "include")]

            if not mkl_lib_path:
                for lib_dir in lib_dirs:
                    if not os.path.isdir(lib_dir):
                        continue
                    if sys.platform == "win32":
                        matches = glob.glob(os.path.join(lib_dir, "mkl_rt*.lib"))
                        if matches:
                            mkl_lib_path = matches[0]
                            break
                    elif sys.platform == "darwin":
                        matches = glob.glob(os.path.join(lib_dir, "libmkl_rt*.dylib"))
                        if matches:
                            mkl_lib_path = matches[0]
                            break
                    else:
                        matches = glob.glob(os.path.join(lib_dir, "libmkl_rt.so*"))
                        if matches:
                            so_link = os.path.join(lib_dir, "libmkl_rt.so")
                            if not os.path.exists(so_link):
                                try:
                                    os.symlink(os.path.basename(matches[0]), so_link)
                                    mkl_lib_path = so_link
                                except OSError:
                                    mkl_lib_path = matches[0]
                            else:
                                mkl_lib_path = so_link
                            break

            if not mkl_inc_path:
                for inc_dir in inc_dirs:
                    if os.path.isfile(os.path.join(inc_dir, "mkl.h")):
                        mkl_inc_path = inc_dir
                        break

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={extdir}",
            f"-DPython_ROOT_DIR={real_exe_root}",
            f"-DPython3_ROOT_DIR={real_exe_root}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-Dnanobind_DIR={nanobind_cmake_path}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        if mkl_lib_path:
            cmake_args.append(f"-DMKL_LIBRARY={mkl_lib_path}")
        if mkl_inc_path:
            cmake_args.append(f"-DMKL_INCLUDE_DIR={mkl_inc_path}")

        env_cmake_args = os.environ.get("CMAKE_ARGS")
        if env_cmake_args:
            cmake_args.extend(shlex.split(env_cmake_args))

        if sys.platform != "win32" and shutil.which("ninja"):
            cmake_args.extend(["-G", "Ninja"])

        parallel_level = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 4))
        build_args = ["--config", "Release", "-j", parallel_level]

        os.makedirs(self.build_temp, exist_ok=True)

        env = os.environ.copy()
        if sys.platform != "win32":
            env["CXXFLAGS"] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

        for ext_pat in ("**/*.so", "**/*.pyd", "**/*.dylib"):
            for filepath in glob.glob(os.path.join(self.build_temp, ext_pat), recursive=True):
                if "_mshqc" in os.path.basename(filepath):
                    shutil.copy2(filepath, extdir)

        for ext_pat in ("*.so", "*.pyd", "*.dylib"):
            for filepath in glob.glob(os.path.join(ext.sourcedir, "python", "mshqc", ext_pat)):
                if "_mshqc" in os.path.basename(filepath):
                    shutil.copy2(filepath, extdir)

        basis_candidates = [
            os.path.join(ext.sourcedir, "data", "basis"),
            os.path.join(os.path.dirname(ext.sourcedir), "mshqc-basis", "data", "basis"),
            os.path.join(os.path.expanduser("~"), "mshqc-basis", "data", "basis"),
        ]
        src_basis_dir = ""
        for cand in basis_candidates:
            if os.path.isdir(cand) and glob.glob(os.path.join(cand, "*.gbs")):
                src_basis_dir = cand
                break

        if src_basis_dir:
            pkg_src_basis_dir = os.path.join(ext.sourcedir, "python", "mshqc", "basis")
            dst_wheel_basis_dir = os.path.join(extdir, "basis")

            os.makedirs(pkg_src_basis_dir, exist_ok=True)
            os.makedirs(dst_wheel_basis_dir, exist_ok=True)

            for gbs_file in glob.glob(os.path.join(src_basis_dir, "*.gbs")):
                shutil.copy2(gbs_file, pkg_src_basis_dir)
                if os.path.abspath(pkg_src_basis_dir) != os.path.abspath(dst_wheel_basis_dir):
                    shutil.copy2(gbs_file, dst_wheel_basis_dir)


setup(
    ext_modules=[CMakeExtension("mshqc._mshqc")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
'''

# ==============================================================================
# 3. .github/workflows/release_public.yml
# ==============================================================================
RELEASE_PUBLIC_YML_CONTENT = r'''name: Compile and Release Wheel to Public Repo

on:
  push:
    branches: [ "master", "benchmark" ]
    tags: [ "v*" ]
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build_wheels:
    name: Build wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-14, windows-latest]

    steps:
      - name: Checkout Source Code
        uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - name: Fetch Basis Sets from mshqc-basis
        shell: bash
        run: |
          git clone --depth 1 https://github.com/syahrulhidayat/mshqc-basis.git /tmp/mshqc-basis
          mkdir -p data/basis
          cp -rf /tmp/mshqc-basis/data/basis/* data/basis/
          rm -rf /tmp/mshqc-basis

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Cache Static Dependencies (Linux)
        if: runner.os == 'Linux'
        uses: actions/cache@v4
        with:
          path: .cache_deps
          key: manylinux_2_28-static-deps-v6.1.2-tblis-v1.2.0

      # ========================================================================
      # STEP KHUSUS MACOS: Kompilasi libcint & TBLIS langsung di Host macOS
      # ========================================================================
      - name: Build C/C++ Dependencies (macOS)
        if: runner.os == 'macOS'
        shell: bash
        run: |
          set -e
          brew install --quiet cmake ninja eigen libomp hdf5 openblas lapack || true
          sudo mkdir -p /usr/local/include /usr/local/lib

          git clone --depth 1 -b v6.1.2 https://github.com/sunqm/libcint.git /tmp/libcint
          cmake -S /tmp/libcint -B /tmp/libcint/build \
                -DCMAKE_INSTALL_PREFIX=/usr/local \
                -DCMAKE_PREFIX_PATH="/opt/homebrew;/opt/homebrew/opt/openblas;/opt/homebrew/opt/lapack" \
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
                -DCMAKE_C_FLAGS="-O3 -fPIC -Wno-incompatible-pointer-types -Wno-implicit-function-declaration" \
                -DENABLE_EXAMPLE=0 -DENABLE_TEST=0 -DBUILD_SHARED_LIBS=OFF
          cmake --build /tmp/libcint/build -j$(sysctl -n hw.ncpu)
          sudo cmake --install /tmp/libcint/build

          git clone --depth 1 -b v1.2.0 https://github.com/devinamatthews/tblis.git /tmp/tblis
          cd /tmp/tblis
          git submodule update --init --recursive
          perl -pi -e 's/SIGSTKSZ/8192/g' src/external/catch/catch.hpp
          perl -pi -e 'print "#include <limits>\n" if $. == 1' src/external/marray/include/short_vector.hpp
          perl -pi -e 'print "#include <limits>\n" if $. == 1' src/external/marray/include/range.hpp
          perl -pi -e 'print "#include <limits>\n" if $. == 1' src/util/thread.h
          ./configure --prefix=/usr/local --enable-static --disable-shared --with-pic --without-hwloc \
                      --enable-config=reference \
                      CFLAGS="-O3 -fPIC" CXXFLAGS="-O3 -fPIC -std=c++17"
          make -j$(sysctl -n hw.ncpu)
          sudo make install

      # ========================================================================
      # STEP KHUSUS WINDOWS: Instalasi HDF5/MKL (Conda) & libcint/TBLIS (MSYS2)
      # ========================================================================
      - name: Install MSVC Libraries via Conda (Windows)
        if: runner.os == 'Windows'
        shell: cmd
        run: |
          call %CONDA%\condabin\conda.bat create -p C:\deps -c conda-forge hdf5 eigen mkl-devel -y

      - name: Build libcint & TBLIS via MSYS2 MinGW64 (Windows)
        if: runner.os == 'Windows'
        shell: C:\msys64\usr\bin\bash.exe --noprofile --norc -eo pipefail "{0}"
        env:
          MSYSTEM: MINGW64
          MSYS2_PATH_TYPE: inherit
        run: |
          export PATH="/mingw64/bin:/usr/bin:$PATH"
          pacman -Sy --noconfirm --needed \
            make \
            diffutils \
            mingw-w64-x86_64-gcc \
            mingw-w64-x86_64-gcc-fortran \
            mingw-w64-x86_64-make \
            mingw-w64-x86_64-cmake

          mkdir -p /c/deps/Library/include /c/deps/Library/lib /c/deps/Library/bin
          cp -f /mingw64/include/pthread*.h /mingw64/include/sched.h /c/deps/Library/include/ 2>/dev/null || true

          # Buat header jembatan unistd.h agar MSVC mengenali tipe POSIX ssize_t saat membaca tblis.h
          cat << 'EOF' > /c/deps/Library/include/unistd.h
          #ifndef _UNISTD_H
          #define _UNISTD_H
          #include <io.h>
          #include <process.h>
          #include <BaseTsd.h>
          typedef SSIZE_T ssize_t;
          #endif
          EOF

          # 1. Build libcint (Shared DLL + Import Library untuk MSVC)
          rm -rf /tmp/libcint
          git clone --depth 1 -b v6.1.2 https://github.com/sunqm/libcint.git /tmp/libcint
          /mingw64/bin/cmake.exe -S /tmp/libcint -B /tmp/libcint/build -G "MinGW Makefiles" \
                -DCMAKE_INSTALL_PREFIX=/c/deps/Library \
                -DCMAKE_C_COMPILER=/mingw64/bin/gcc.exe \
                -DCMAKE_MAKE_PROGRAM=/mingw64/bin/mingw32-make.exe \
                -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--export-all-symbols" \
                -DCMAKE_C_FLAGS="-O3 -mavx2 -mfma -mno-avx512f -funroll-loops -fno-math-errno -Wno-incompatible-pointer-types -std=gnu17" \
                -DENABLE_EXAMPLE=0 -DENABLE_TEST=0 -DBUILD_SHARED_LIBS=ON
          /mingw64/bin/cmake.exe --build /tmp/libcint/build -j4 --target install
          cp -f /c/deps/Library/lib/*.dll /c/deps/Library/bin/ 2>/dev/null || true
          cp -f /c/deps/Library/lib/libcint.dll.a /c/deps/Library/lib/cint.lib

          # 2. Build TBLIS (Kompilasi objek lalu konversi ke DLL + Import Library untuk MSVC)
          rm -rf /tmp/tblis
          git clone --depth 1 -b v1.2.0 https://github.com/devinamatthews/tblis.git /tmp/tblis
          cd /tmp/tblis
          git submodule update --init --recursive
          sed -i 's/SIGSTKSZ/8192/g' src/external/catch/catch.hpp
          sed -i '1i #include <limits>' src/external/marray/include/short_vector.hpp
          sed -i '1i #include <limits>' src/external/marray/include/range.hpp
          sed -i '1i #include <limits>' src/util/thread.h
          export CC=/mingw64/bin/gcc.exe
          export CXX=/mingw64/bin/g++.exe
          export CFLAGS="-O3 -mno-avx512f -mno-avx512vl -mno-avx512dq -mno-avx512bw -mno-avx512cd -Wno-incompatible-pointer-types -std=gnu17"
          export CXXFLAGS="-O3 -mno-avx512f -mno-avx512vl -mno-avx512dq -mno-avx512bw -mno-avx512cd -std=gnu++17"
          ./configure --prefix=/c/deps/Library --enable-static --disable-shared --without-hwloc \
                      --enable-config=haswell,zen,core2,sandybridge,reference \
                      CFLAGS="$CFLAGS" CXXFLAGS="$CXXFLAGS" LIBS="-latomic -lpthread"
          make -j4
          make install

          # Buat libtci.dll + tci.lib dan libtblis.dll + tblis.lib dengan semua simbol terekspor
          /mingw64/bin/g++.exe -shared -o /c/deps/Library/bin/libtci.dll \
              -Wl,--out-implib,/c/deps/Library/lib/libtci.dll.a \
              -Wl,--export-all-symbols \
              -Wl,--whole-archive /c/deps/Library/lib/libtci.a -Wl,--no-whole-archive \
              -lpthread -latomic
          cp -f /c/deps/Library/lib/libtci.dll.a /c/deps/Library/lib/tci.lib

          /mingw64/bin/g++.exe -shared -o /c/deps/Library/bin/libtblis.dll \
              -Wl,--out-implib,/c/deps/Library/lib/libtblis.dll.a \
              -Wl,--export-all-symbols \
              -Wl,--whole-archive /c/deps/Library/lib/libtblis.a -Wl,--no-whole-archive \
              /c/deps/Library/lib/libtci.dll.a -lpthread -latomic
          cp -f /c/deps/Library/lib/libtblis.dll.a /c/deps/Library/lib/tblis.lib

          # Hapus archive .a statis MinGW agar MSVC hanya memilih import library .lib / .dll.a
          rm -f /c/deps/Library/lib/libtblis.a /c/deps/Library/lib/libtci.a

      - name: Install cibuildwheel
        run: python -m pip install --upgrade cibuildwheel

      - name: Build wheels
        run: python -m cibuildwheel --output-dir wheelhouse
        env:
          CIBW_BUILD_VERBOSITY: "1"
          CIBW_MANYLINUX_X86_64_IMAGE: manylinux_2_28
          CIBW_BUILD: "cp310-* cp311-* cp312-* cp313-*"
          CIBW_SKIP: "*-win32 *-manylinux_i686 *-musllinux_*"
          CIBW_BUILD_FRONTEND: "pip; args: --no-build-isolation"
          CIBW_TEST_SKIP: "*"
          CIBW_TEST_COMMAND: ""

          # ====================================================================
          # 1. KONFIGURASI LINUX (TETAP UTUH 100%)
          # ====================================================================
          CIBW_BEFORE_ALL_LINUX: >
            dnf install -y dnf-plugins-core epel-release &&
            dnf config-manager --set-enabled powertools &&
            dnf install -y git wget hdf5-devel eigen3-devel ninja-build patchelf libatomic gcc-toolset-12-libatomic-devel &&
            if [ -f "/project/.cache_deps/lib/libtblis.a" ] || [ -f "/project/.cache_deps/lib64/libtblis.a" ]; then
              echo "Restoring cached libcint and TBLIS to /usr/local..." &&
              cp -rn /project/.cache_deps/* /usr/local/ &&
              ldconfig
            else
              mkdir -p /project/.cache_deps &&
              git clone --depth 1 -b v6.1.2 https://github.com/sunqm/libcint.git /tmp/libcint &&
              cd /tmp/libcint && mkdir build && cd build &&
              export CFLAGS="-O3 -fPIC -mavx2 -mfma -mno-avx512f -funroll-loops -fno-math-errno -Wno-incompatible-pointer-types -std=gnu17" &&
              cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
                    -DCMAKE_C_FLAGS="$CFLAGS" \
                    -DENABLE_EXAMPLE=0 -DENABLE_TEST=0 -DBUILD_SHARED_LIBS=OFF .. &&
              make -j$(nproc) && make install &&
              git clone --depth 1 -b v1.2.0 https://github.com/devinamatthews/tblis.git /tmp/tblis &&
              cd /tmp/tblis && git submodule update --init --recursive &&
              sed -i 's/SIGSTKSZ/8192/g' src/external/catch/catch.hpp &&
              sed -i '1i #include <limits>' src/external/marray/include/short_vector.hpp &&
              sed -i '1i #include <limits>' src/external/marray/include/range.hpp &&
              sed -i '1i #include <limits>' src/util/thread.h &&
              export CFLAGS="-O3 -fPIC -mno-avx512f -mno-avx512vl -mno-avx512dq -mno-avx512bw -mno-avx512cd -Wno-incompatible-pointer-types -std=gnu17" &&
              export CXXFLAGS="-O3 -fPIC -mno-avx512f -mno-avx512vl -mno-avx512dq -mno-avx512bw -mno-avx512cd -Wno-incompatible-pointer-types -std=gnu++17" &&
              ./configure --prefix=/usr/local --enable-static --disable-shared --with-pic --without-hwloc \
                          --enable-config=haswell,zen,core2,sandybridge,bulldozer,piledriver,excavator,reference \
                          CFLAGS="$CFLAGS" CXXFLAGS="$CXXFLAGS" LIBS="-latomic" &&
              make -j$(nproc) && make install &&
              cp -r /usr/local/include /usr/local/lib* /project/.cache_deps/ 2>/dev/null || true
            fi

          CIBW_BEFORE_BUILD_LINUX: >
            rm -rf /project/build /project/python/mshqc/*.so /project/python/mshqc/*.a &&
            pip install --upgrade pip setuptools wheel nanobind cmake ninja mkl mkl-devel mkl-include numpy scipy

          CIBW_ENVIRONMENT_LINUX: >
            GITHUB_ACTIONS=true
            CIBUILDWHEEL=1
            CXXFLAGS="-std=c++20 -fopenmp -O3 -DNDEBUG -mavx2 -mfma -mno-avx512f -funroll-loops -ftree-vectorize -mtune=generic -fno-math-errno -DEIGEN_USE_MKL_ALL -I/usr/local/include -I/usr/include/eigen3"
            CFLAGS="-O3 -fPIC -DNDEBUG"
            CMAKE_BUILD_TYPE=Release
            LDFLAGS="-L/usr/local/lib -L/usr/local/lib64 -lquadmath -latomic"
            CMAKE_PREFIX_PATH="/usr/local"
            TBLIS_DIR="/usr/local"
            CMAKE_BUILD_PARALLEL_LEVEL=4

          CIBW_REPAIR_WHEEL_COMMAND_LINUX: >
            auditwheel repair --exclude libmkl_rt.so.3 --exclude libmkl_rt.so.2 --exclude libmkl_rt.so --exclude libiomp5.so --exclude libtbb.so.12 -w {dest_dir} {wheel}

          # ====================================================================
          # 2. KONFIGURASI MACOS
          # ====================================================================
          CIBW_BEFORE_BUILD_MACOS: >
            rm -rf build python/mshqc/*.so python/mshqc/*.dylib python/mshqc/*.a &&
            pip install --upgrade pip setuptools wheel nanobind cmake ninja numpy scipy

          CIBW_ENVIRONMENT_MACOS: >
            GITHUB_ACTIONS=true
            CIBUILDWHEEL=1
            CMAKE_BUILD_TYPE=Release
            CMAKE_PREFIX_PATH="/usr/local;/opt/homebrew;/opt/homebrew/opt/libomp;/opt/homebrew/opt/openblas;/opt/homebrew/opt/lapack;/opt/homebrew/opt/hdf5"
            OpenMP_ROOT="/opt/homebrew/opt/libomp"
            TBLIS_DIR="/usr/local"
            CMAKE_BUILD_PARALLEL_LEVEL=4

          # ====================================================================
          # 3. KONFIGURASI WINDOWS
          # ====================================================================
          CIBW_BEFORE_BUILD_WINDOWS: >
            if exist build rmdir /s /q build &&
            pip install --upgrade pip setuptools wheel nanobind cmake ninja mkl mkl-devel mkl-include numpy scipy delvewheel

          CIBW_ENVIRONMENT_WINDOWS: >
            GITHUB_ACTIONS=true
            CIBUILDWHEEL=1
            CMAKE_BUILD_TYPE=Release
            CMAKE_PREFIX_PATH="C:/deps/Library;C:/deps"
            TBLIS_DIR="C:/deps/Library"
            CMAKE_BUILD_PARALLEL_LEVEL=4

          CIBW_REPAIR_WHEEL_COMMAND_WINDOWS: >
            delvewheel repair --add-path C:/deps/Library/bin;C:/msys64/mingw64/bin --no-dll mkl_rt.2.dll;mkl_rt.dll -w {dest_dir} -v {wheel}

      - name: Upload Wheels as Artifacts
        uses: actions/upload-artifact@v4
        with:
          name: cibw-wheels-${{ matrix.os }}
          path: ./wheelhouse/*.whl

  build_sdist:
    name: Build source distribution
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Fetch Basis Sets from mshqc-basis
        run: |
          git clone --depth 1 https://github.com/syahrulhidayat/mshqc-basis.git /tmp/mshqc-basis
          mkdir -p data/basis
          cp -rf /tmp/mshqc-basis/data/basis/* data/basis/
          rm -rf /tmp/mshqc-basis
      - name: Remove local cache dir if exists
        run: rm -rf .cache_deps
      - name: Build sdist
        run: pipx run build --sdist
      - uses: actions/upload-artifact@v4
        with:
          name: cibw-sdist
          path: dist/*.tar.gz

  upload_pypi:
    name: Upload to PyPI
    needs: [build_wheels, build_sdist]
    runs-on: ubuntu-latest
    if: (github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')) || github.event_name == 'workflow_dispatch'

    steps:
      - name: Download artifacts
        uses: actions/download-artifact@v4
        with:
          pattern: cibw-*
          path: dist
          merge-multiple: true

      - name: Publish package to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
'''

# ==============================================================================
# 4. src/core/basis.cc
# ==============================================================================
BASIS_CC_CONTENT = r'''// Copyright 2026 Muhamad Syahrul Hidayat
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

/**
 * @file src/core/basis.cc
 * @brief Basis Set Handling (Exact Contraction Normalization & Dynamic Path)
 */

#include "mshqc/basis.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cctype>
#include <vector>
#include <cstdlib>

#ifdef I
#undef I
#endif

namespace mshqc {
    
constexpr double PI = 3.14159265358979323846;

std::string sanitize_number(std::string str) {
    for (char &c : str) {
        if (c == 'D' || c == 'd') c = 'E';
    }
    return str;
}

Shell::Shell(AngularMomentum am, int center, const std::array<double, 3>& center_pos)
    : am_(am), center_(center), position_(center_pos), spherical_(true) {
}

void Shell::add_primitive(double exponent, double coeff) {
    primitives_.emplace_back(exponent, coeff);
}

int Shell::n_functions() const {
    if (spherical_) {
        return n_spherical_functions(am_);
    } else {
        return n_cartesian_functions(am_);
    }
}

void Shell::normalize() {
    double self_overlap = 0.0;
    int l_val = this->l();
    
    for (size_t i = 0; i < primitives_.size(); ++i) {
        for (size_t j = 0; j < primitives_.size(); ++j) {
            double alpha_i = primitives_[i].exponent;
            double alpha_j = primitives_[j].exponent;
            double c_i = primitives_[i].coefficient;
            double c_j = primitives_[j].coefficient;
            double term = (2.0 * std::sqrt(alpha_i * alpha_j)) / (alpha_i + alpha_j);
            double S_ij = std::pow(term, l_val + 1.5);
            
            self_overlap += c_i * c_j * S_ij;
        }
    }
    
    if (self_overlap > 1e-12) {
        double scale = 1.0 / std::sqrt(self_overlap);
        for (auto& prim : primitives_) {
            prim.coefficient *= scale;
        }
    }
}

BasisSet::BasisSet() : name_("unknown"), spherical_(true) { }

BasisSet::BasisSet(const std::string& basis_name, const Molecule& mol, const std::string& basis_dir)
    : name_(basis_name), spherical_(true) {
    
    std::string basis_lower = basis_name;
    std::transform(basis_lower.begin(), basis_lower.end(), basis_lower.begin(), ::tolower);
    
    std::string gbs_name = basis_lower;
    if (gbs_name.size() < 4 || gbs_name.substr(gbs_name.size() - 4) != ".gbs") {
        gbs_name += ".gbs";
    }

    std::vector<std::string> search_dirs;
    if (const char* env_path = std::getenv("MSHQC_BASIS_PATH")) {
        search_dirs.emplace_back(env_path);
    }
    if (const char* env_dir = std::getenv("MSHQC_BASIS_DIR")) {
        search_dirs.emplace_back(env_dir);
    }
    if (!basis_dir.empty()) {
        search_dirs.push_back(basis_dir);
    }
    search_dirs.push_back("data/basis");
    search_dirs.push_back("../mshqc-basis/data/basis");
    search_dirs.push_back("../../mshqc-basis/data/basis");
    if (const char* home = std::getenv("HOME")) {
        search_dirs.push_back(std::string(home) + "/mshqc-basis/data/basis");
    }
    if (const char* userprofile = std::getenv("USERPROFILE")) {
        search_dirs.push_back(std::string(userprofile) + "/mshqc-basis/data/basis");
    }
    search_dirs.push_back("./basis");
    search_dirs.push_back("python/mshqc/basis");

    std::string basis_file = (basis_dir.empty() ? "./basis" : basis_dir) + "/" + gbs_name;
    for (const auto& dir : search_dirs) {
        std::string cand = dir + "/" + gbs_name;
        std::ifstream f_check(cand);
        if (f_check.good()) {
            basis_file = cand;
            break;
        }
    }
    
    if (!read_gbs(basis_file, mol)) {
        throw std::runtime_error("Failed to read basis set: " + basis_file);
    }
}

bool BasisSet::read_gbs(const std::string& basis_file, const Molecule& mol) {
    std::ifstream f(basis_file);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot open basis file: " << basis_file << std::endl;
        return false;
    }
    
    std::string line;
    std::getline(f, line); 
    
    if (line.find("spherical") != std::string::npos) spherical_ = true;
    else if (line.find("cartesian") != std::string::npos) spherical_ = false;
    
    shells_.clear();
    
    for (size_t i = 0; i < mol.n_atoms(); i++) {
        const auto& atom = mol.atom(i);
        std::string sym = get_element_symbol(atom.atomic_number);
        std::array<double, 3> pos = {atom.x, atom.y, atom.z};
        
        f.clear();
        f.seekg(0);
        std::getline(f, line); 
        
        bool found = false;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '!') continue;

            std::istringstream iss(line);
            std::string elem;
            int dummy; 
            
            if (iss >> elem >> dummy) {
                if (elem == sym && dummy == 0) {
                    found = true;
                    parse_atom_basis(f, sym, i, pos);
                    break;
                }
            }
        }
        if (!found) {
            std::cerr << "Warning: Basis not found for element " << sym 
                      << " (Z=" << atom.atomic_number << ")" << std::endl;
        }
    }
    
    return true;
}

int BasisSet::parse_atom_basis(std::ifstream& file,
                               const std::string& atom_symbol,
                               int atom_index,
                               const std::array<double, 3>& atom_pos) {
    std::string line;
    int n_added = 0;
    
    while (std::getline(file, line)) {
        if (line.find("****") != std::string::npos) break; 
        if (line.empty() || line[0] == '!') continue;
        
        std::istringstream iss(line);
        std::string stype;
        int nprim = 0;
        double scale = 1.0; 
        
        if (!(iss >> stype)) continue;
        std::transform(stype.begin(), stype.end(), stype.begin(), ::toupper);

        if (!(iss >> nprim)) continue; 
        iss >> scale; 
        
        if (stype == "SP") {
            Shell s_sh(AngularMomentum::S, atom_index, atom_pos);
            Shell p_sh(AngularMomentum::P, atom_index, atom_pos);
            s_sh.set_spherical(spherical_);
            p_sh.set_spherical(spherical_);
            
            for (int i = 0; i < nprim; i++) {
                if (!std::getline(file, line)) break;
                line = sanitize_number(line); 
                
                std::istringstream piss(line);
                double exp, sc, pc;
                if (piss >> exp >> sc >> pc) {
                    s_sh.add_primitive(exp, sc);
                    p_sh.add_primitive(exp, pc);
                }
            }
            
            s_sh.normalize();
            p_sh.normalize();
            
            add_shell(s_sh);
            add_shell(p_sh);
            n_added += 2;
            
        } else {
            AngularMomentum am = char_to_am(stype[0]);
            Shell sh(am, atom_index, atom_pos);
            sh.set_spherical(spherical_);
            
            for (int i = 0; i < nprim; i++) {
                if (!std::getline(file, line)) break;
                line = sanitize_number(line); 
                
                std::istringstream piss(line);
                double exp, c;
                if (piss >> exp >> c) sh.add_primitive(exp, c);
            }
            sh.normalize();
            
            add_shell(sh);
            n_added++;
        }
    }
    return n_added;
}

void BasisSet::add_shell(const Shell& shell) { shells_.push_back(shell); }

size_t BasisSet::n_basis_functions() const {
    size_t total = 0;
    for (const auto& shell : shells_) total += shell.n_functions();
    return total;
}

void BasisSet::set_spherical(bool sph) {
    spherical_ = sph;
    for (auto& s : shells_) { 
        s.set_spherical(sph);
        s.set_cartesian(!sph); 
    }
}

void BasisSet::print() const {
    std::cout << "\nBasis set: " << name_ << " (" << (spherical_ ? "Spherical" : "Cartesian") << ")\n";
    std::cout << "Shells: " << n_shells() << " | Functions: " << n_basis_functions() << "\n";
}

void BasisSet::append(const BasisSet& other) {
    this->shells_.insert(this->shells_.end(), other.shells_.begin(), other.shells_.end());
}

int BasisSet::max_angular_momentum() const {
    int max_l = 0;
    for (const auto& shell : shells_) max_l = std::max(max_l, shell.l());
    return max_l;
}

std::vector<int> BasisSet::shell_to_basis_function_map() const {
    std::vector<int> map;
    int bf_index = 0;
    for (const auto& shell : shells_) {
        map.push_back(bf_index);
        bf_index += shell.n_functions();
    }
    return map;
}

AngularMomentum char_to_am(char c) {
    switch (std::toupper(c)) {
        case 'S': return AngularMomentum::S;
        case 'P': return AngularMomentum::P;
        case 'D': return AngularMomentum::D;
        case 'F': return AngularMomentum::F;
        case 'G': return AngularMomentum::G;
        case 'H': return AngularMomentum::H;
        default: return AngularMomentum::S; 
    }
}

std::string am_to_string(AngularMomentum am) {
    switch (am) {
        case AngularMomentum::S: return "s";
        case AngularMomentum::P: return "p";
        case AngularMomentum::D: return "d";
        case AngularMomentum::F: return "f";
        case AngularMomentum::G: return "g";
        case AngularMomentum::H: return "h";
        default: return "?";
    }
}

std::string get_element_symbol(int Z) {
    static const char* symbols[] = {
        "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", 
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", 
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", 
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", 
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", 
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", 
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", 
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", 
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", 
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", 
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    };
    if (Z < 0 || Z >= static_cast<int>(sizeof(symbols)/sizeof(char*))) {
        return "?";
    }
    return symbols[Z];
}

} // namespace mshqc
'''

# ==============================================================================
# 5. Patch Tambahan untuk python/mshqc/__init__.py
# ==============================================================================
INIT_PY_BASIS_HOOK = '''
# --- Auto-resolve MSHQC_BASIS_PATH (mendukung repo terpisah mshqc-basis) ---
def _resolve_mshqc_basis_dir() -> None:
    import os as _os
    from pathlib import Path as _Path
    _pkg_dir = _Path(__file__).resolve().parent
    _candidates = [
        _pkg_dir / "basis",
        _pkg_dir.parent.parent / "data" / "basis",
        _pkg_dir.parent.parent.parent / "mshqc-basis" / "data" / "basis",
        _Path.home() / "mshqc-basis" / "data" / "basis",
    ]
    _cur = _os.environ.get("MSHQC_BASIS_PATH", "")
    if _cur and _Path(_cur).is_dir() and any(_Path(_cur).glob("*.gbs")):
        return
    for _cand in _candidates:
        if _cand.is_dir() and any(_cand.glob("*.gbs")):
            _os.environ["MSHQC_BASIS_PATH"] = str(_cand)
            break

_resolve_mshqc_basis_dir()
'''


def main() -> None:
    targets = {
        ROOT / "CMakeLists.txt": CMAKELISTS_CONTENT,
        ROOT / "setup.py": SETUP_PY_CONTENT,
        ROOT / ".github" / "workflows" / "release_public.yml": RELEASE_PUBLIC_YML_CONTENT,
        ROOT / "src" / "core" / "basis.cc": BASIS_CC_CONTENT,
    }

    for path, content in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"[OK] Berhasil menimpa: {path.relative_to(ROOT)}")

    init_py = ROOT / "python" / "mshqc" / "__init__.py"
    if init_py.exists():
        text = init_py.read_text(encoding="utf-8")
        if "_resolve_mshqc_basis_dir" not in text:
            init_py.write_text(text.rstrip() + "\n" + INIT_PY_BASIS_HOOK, encoding="utf-8")
            print(f"[OK] Berhasil menambahkan auto-resolver mshqc-basis ke: {init_py.relative_to(ROOT)}")
        else:
            print(f"[OK] {init_py.relative_to(ROOT)} sudah memiliki auto-resolver mshqc-basis")

    print("\nSelesai! Silakan jalankan git status & git diff untuk memeriksa hasil.")


if __name__ == "__main__":
    main()