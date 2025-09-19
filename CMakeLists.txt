cmake_minimum_required(VERSION 3.23)

project(Hoominer LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

option(HOOMINER_DISABLE_OPENCL "Disable OpenCL support" OFF)
option(HOOMINER_DISABLE_CUDA "Disable CUDA support" OFF)

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS -DWIN32_LEAN_AND_MEAN)
endif()

set(HOOMINER_SRC
  src/api.c
  src/config.c
  src/datatypes.c
  src/hoominer.c
  src/miner-hoohash.c
  src/opencl-host.c
  src/overclock.c
  src/reporting.c
  src/stratum.c
  src/target.c
  src/platform_compat.c
)

if(NOT HOOMINER_DISABLE_CUDA)
  list(APPEND HOOMINER_SRC src/cuda-host.cu)
endif()

# hoohash static library
add_library(hoohash STATIC
  algorithms/hoohash/hoohash.c
  algorithms/hoohash/bigint.c
)
target_include_directories(hoohash PUBLIC algorithms/hoohash)

# Find installed BLAKE3 (e.g., via vcpkg: blake3)
find_path(BLAKE3_INCLUDE_DIR blake3.h)
find_library(BLAKE3_LIBRARY NAMES blake3)
if(NOT BLAKE3_INCLUDE_DIR OR NOT BLAKE3_LIBRARY)
  message(FATAL_ERROR "BLAKE3 not found. Install via vcpkg: 'vcpkg install blake3:x64-windows'")
endif()
target_include_directories(hoohash PRIVATE ${BLAKE3_INCLUDE_DIR})

add_executable(hoominer ${HOOMINER_SRC})

target_include_directories(hoominer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# OpenSSL
find_package(OpenSSL REQUIRED)
target_link_libraries(hoominer PRIVATE OpenSSL::SSL OpenSSL::Crypto)

# JSON-C (prefer CMake config from vcpkg)
find_package(json-c CONFIG QUIET)
if(json-c_FOUND)
  target_link_libraries(hoominer PRIVATE json-c::json-c)
else()
  find_path(JSONC_INCLUDE_DIR json-c/json.h)
  find_library(JSONC_LIBRARY NAMES json-c json)
  if(JSONC_INCLUDE_DIR AND JSONC_LIBRARY)
    target_include_directories(hoominer PRIVATE ${JSONC_INCLUDE_DIR})
    target_link_libraries(hoominer PRIVATE ${JSONC_LIBRARY})
  else()
    message(FATAL_ERROR "json-c not found. Install via vcpkg: 'vcpkg install json-c:x64-windows'")
  endif()
endif()

# libmicrohttpd (handle multiple vcpkg export patterns)
set(_mhd_found FALSE)
find_package(unofficial-libmicrohttpd CONFIG QUIET)
if(TARGET unofficial::libmicrohttpd::libmicrohttpd)
  target_link_libraries(hoominer PRIVATE unofficial::libmicrohttpd::libmicrohttpd)
  set(_mhd_found TRUE)
elseif(TARGET unofficial::libmicrohttpd)
  target_link_libraries(hoominer PRIVATE unofficial::libmicrohttpd)
  set(_mhd_found TRUE)
endif()
if(NOT _mhd_found)
  find_package(libmicrohttpd CONFIG QUIET)
  if(TARGET libmicrohttpd::libmicrohttpd)
    target_link_libraries(hoominer PRIVATE libmicrohttpd::libmicrohttpd)
    set(_mhd_found TRUE)
  elseif(TARGET microhttpd::microhttpd)
    target_link_libraries(hoominer PRIVATE microhttpd::microhttpd)
    set(_mhd_found TRUE)
  endif()
endif()
if(NOT _mhd_found)
  # Try pkg-config
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(MHD QUIET libmicrohttpd)
    if(MHD_FOUND)
      target_include_directories(hoominer PRIVATE ${MHD_INCLUDE_DIRS})
      target_link_libraries(hoominer PRIVATE ${MHD_LINK_LIBRARIES})
      set(_mhd_found TRUE)
    endif()
  endif()
endif()
if(NOT _mhd_found)
  # Fallback manual search; check typical vcpkg locations and CMAKE_PREFIX_PATH entries
  if(DEFINED VCPKG_TARGET_TRIPLET AND DEFINED VCPKG_ROOT)
    list(APPEND _MHD_HINTS "${VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}")
  endif()
  list(APPEND _MHD_HINTS ${CMAKE_PREFIX_PATH})
  find_path(MICROHTTPD_INCLUDE_DIR microhttpd.h HINTS ${_MHD_HINTS} PATH_SUFFIXES include)
  find_library(MICROHTTPD_LIBRARY NAMES microhttpd libmicrohttpd libmicrohttpd-dll HINTS ${_MHD_HINTS} PATH_SUFFIXES lib debug/lib)
  if(MICROHTTPD_INCLUDE_DIR AND MICROHTTPD_LIBRARY)
    target_include_directories(hoominer PRIVATE ${MICROHTTPD_INCLUDE_DIR})
    target_link_libraries(hoominer PRIVATE ${MICROHTTPD_LIBRARY})
    set(_mhd_found TRUE)
  endif()
endif()
if(NOT _mhd_found)
  message(FATAL_ERROR "libmicrohttpd not found. Install via vcpkg: 'vcpkg install libmicrohttpd:x64-windows'")
endif()

# OpenCL (optional via option)
if(NOT HOOMINER_DISABLE_OPENCL)
  find_package(OpenCL REQUIRED)
  target_link_libraries(hoominer PRIVATE OpenCL::OpenCL)
  target_compile_definitions(hoominer PRIVATE CL_TARGET_OPENCL_VERSION=200)
endif()

# Threads (pthreads on Unix; not used on Windows for winsock)
find_package(Threads REQUIRED)
target_link_libraries(hoominer PRIVATE Threads::Threads)

# pthreads (vcpkg) for Windows builds
if(WIN32)
  find_package(pthreads CONFIG QUIET)
  if(TARGET pthreads::pthreads)
    target_link_libraries(hoominer PRIVATE pthreads::pthreads)
  else()
    # Fallback manual search for common pthreads4w library names
    if(DEFINED VCPKG_ROOT)
      list(APPEND _PTHREAD_HINTS "${VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}")
    endif()
    find_library(PTHREADS_LIBRARY NAMES pthread pthreadVC3 pthreadVC2 HINTS ${_PTHREAD_HINTS} PATH_SUFFIXES lib debug/lib)
    if(PTHREADS_LIBRARY)
      target_link_libraries(hoominer PRIVATE ${PTHREADS_LIBRARY})
    else()
      message(FATAL_ERROR "pthreads not found. Install via vcpkg: 'vcpkg install pthreads:x64-windows'")
    endif()
  endif()
endif()

# CUDA runtime (we use CUDA runtime + driver; kernels loaded from cubins)
if(NOT HOOMINER_DISABLE_CUDA)
  enable_language(CUDA)
  # Let NVCC use MSVC on Windows automatically
  set_target_properties(hoominer PROPERTIES CUDA_ARCHITECTURES native)
  # Ensure separable compilation is off; we load cubins at runtime
  set_target_properties(hoominer PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(hoominer PRIVATE CUDA::cudart)
endif()

# Link order: external libs then project libs
target_link_libraries(hoominer PRIVATE ${BLAKE3_LIBRARY} hoohash)

# GMP (big integers)
find_package(GMP CONFIG QUIET)
if(TARGET gmp::gmp)
  target_link_libraries(hoominer PRIVATE gmp::gmp)
else()
  if(DEFINED VCPKG_ROOT)
    list(APPEND _GMP_HINTS "${VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}")
  endif()
  find_path(GMP_INCLUDE_DIR gmp.h HINTS ${_GMP_HINTS} PATH_SUFFIXES include)
  find_library(GMP_LIBRARY NAMES gmp libgmp HINTS ${_GMP_HINTS} PATH_SUFFIXES lib debug/lib)
  if(GMP_INCLUDE_DIR AND GMP_LIBRARY)
    target_include_directories(hoominer PRIVATE ${GMP_INCLUDE_DIR})
    target_link_libraries(hoominer PRIVATE ${GMP_LIBRARY})
  else()
    message(FATAL_ERROR "GMP not found. Install via vcpkg: 'vcpkg install gmp:x64-windows'")
  endif()
endif()

# Windows-specific libs
if(WIN32)
  target_link_libraries(hoominer PRIVATE ws2_32)
  target_compile_definitions(hoominer PRIVATE _WIN32)
endif()

install(TARGETS hoominer RUNTIME DESTINATION bin)

