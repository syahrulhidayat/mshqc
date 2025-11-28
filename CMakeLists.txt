cmake_minimum_required(VERSION 3.15)
project(MSH-QC VERSION 0.1.0 LANGUAGES CXX)

# Allow custom modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -Wall -Wextra")

# Dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Try libint2 via pkg-config or manual
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(Libint2 libint2)
endif()
if(NOT Libint2_FOUND)
  find_path(Libint2_INCLUDE_DIR libint2.hpp PATHS /usr/include /usr/local/include)
  find_library(Libint2_LIBRARY NAMES int2 libint2 PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64)
  if(Libint2_INCLUDE_DIR AND Libint2_LIBRARY)
    set(Libint2_FOUND ON)
    set(Libint2_INCLUDE_DIRS ${Libint2_INCLUDE_DIR})
    set(Libint2_LIBRARIES ${Libint2_LIBRARY})
  endif()
endif()

# Libcint (optional)
find_package(Libcint QUIET)

include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${Libint2_INCLUDE_DIRS}
  ${Libcint_INCLUDE_DIRS}
)

# Sources
file(GLOB_RECURSE MSHQC_SOURCES CONFIGURE_DEPENDS
  src/core/*.cc
  src/scf/*.cc
  src/mp2/*.cc
  src/mp3/*.cc
  src/mp/*.cc
  src/mp/*.cc
  src/ci/*.cc
  src/mcscf/*.cc
  src/integrals/*.cc
  src/integration/*.cc
  src/validation/*.cc
  src/gradient/*.cc
)

add_library(mshqc STATIC ${MSHQC_SOURCES})

# Threading
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(mshqc PUBLIC OpenMP::OpenMP_CXX)
endif()

# Link deps
if(Libint2_FOUND)
  target_link_libraries(mshqc PUBLIC ${Libint2_LIBRARIES})
endif()
if(Libcint_FOUND)
  target_link_libraries(mshqc PUBLIC ${Libcint_LIBRARIES})
  target_compile_definitions(mshqc PUBLIC LIBCINT_FOUND)
endif()

target_link_libraries(mshqc PUBLIC Eigen3::Eigen)

# Install
include(GNUInstallDirs)

install(TARGETS mshqc
  EXPORT MSHQCTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/mshqc DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY data/basis DESTINATION ${CMAKE_INSTALL_DATADIR}/mshqc)

# Install CMake config files
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindLibcint.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MSHQC
)

# Generate CMake config files
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MSHQCConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/MSHQCConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MSHQC
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/MSHQCConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

# Install CMake config files
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/MSHQCConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/MSHQCConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MSHQC
)

# Install exported targets
install(EXPORT MSHQCTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MSHQC
  NAMESPACE MSHQC::
)

# Examples and tests (optional, only if files exist)
file(GLOB EXAMPLES examples/*.cc examples/*/*.cc)
foreach(ex ${EXAMPLES})
  get_filename_component(name ${ex} NAME_WE)
  add_executable(${name} ${ex})
  target_link_libraries(${name} mshqc Eigen3::Eigen ${Libint2_LIBRARIES})
endforeach()
