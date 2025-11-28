# Example of how to use MSH-QC in your own project

cmake_minimum_required(VERSION 3.15)
project(MyQuantumProject CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MSH-QC
# Note: After installing MSH-QC, you should set CMAKE_PREFIX_PATH to installation directory
# or use MSHQC_DIR environment variable to point to installation
find_package(MSHQC REQUIRED)

# Create an executable that uses MSH-QC
add_executable(my_qm_app main.cpp)

# Link against MSH-QC
target_link_libraries(my_qm_app PRIVATE mshqc)

# If MSH-QC was built with density fitting
if(MSHQC_HAS_DENSITY_FITTING)
  target_compile_definitions(my_qm_app PRIVATE HAS_DENSITY_FITTING)
endif()

# Example alternative: If you have MSH-QC installed in a custom location
# find_package(MSHQC REQUIRED PATHS /path/to/mshqc/install)
# or
# set(MSHQC_DIR /path/to/mshqc/install/lib/cmake/MSHQC)
# find_package(MSHQC REQUIRED)