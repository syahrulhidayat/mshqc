# Try to find Libcint
# Once done this will define:
#  Libcint_FOUND - System has Libcint
#  Libcint_INCLUDE_DIRS - The Libcint include directories
#  Libcint_LIBRARIES - The libraries needed to use Libcint

find_path(Libcint_INCLUDE_DIR
  NAMES cint.h
  PATHS
    /usr/include
    /usr/local/include
    /opt/homebrew/include
    /opt/local/include
    ${LIBCINT_ROOT}/include
    $ENV{LIBCINT_ROOT}/include
  DOC "Libcint include directory"
)

find_library(Libcint_LIBRARY
  NAMES cint
  PATHS
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    /opt/homebrew/lib
    /opt/local/lib
    ${LIBCINT_ROOT}/lib
    ${LIBCINT_ROOT}/lib64
    $ENV{LIBCINT_ROOT}/lib
    $ENV{LIBCINT_ROOT}/lib64
  DOC "Libcint library"
)

set(Libcint_LIBRARIES ${Libcint_LIBRARY})
set(Libcint_INCLUDE_DIRS ${Libcint_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Libcint
  REQUIRED_VARS Libcint_LIBRARY Libcint_INCLUDE_DIR
)

mark_as_advanced(Libcint_INCLUDE_DIR Libcint_LIBRARY)
