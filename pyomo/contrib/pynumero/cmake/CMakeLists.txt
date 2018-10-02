
cmake_minimum_required(VERSION 3.2)

PROJECT( Pynumero )

set(CMAKE_BUILD_TYPE release)

##################### Checks for compiler #####################
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)

if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#elseif(COMPILER_SUPPORTS_CXX0X)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.1)
      message(FATAL_ERROR "CLANG version must be at least 3.1!")
    endif()
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
      message(FATAL_ERROR "GCC version must be at least 4.7!")
    endif()
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12.0)
      message(FATAL_ERROR "ICC version must be at least 12.0!")
    endif()
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12.0)
       message(FATAL_ERROR "MSVC version must be at least 12.0!")
    endif()
  endif()
endif()

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

# to find our dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

if(WIN32)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -static-libstdc++")
endif()

# check fpr windows
if(MSVC OR MSYS OR MINGW)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -static-libstdc++")
# check for apple
elseif(APPLE)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O3 -fpermissive")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -static-libstdc++ -fpermissive")
#set(CMAKE_CXX_FLAGS_RELEASE "-O2 -static-libstdc++ -fpermissive -static-libgcc")
# check for linux
else()
set(CMAKE_CXX_FLAGS_DEBUG "-g -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -static-libstdc++")
endif()


option(BUILD_ASL "BUILD_ASL" ON)
option(BUILD_HSL "BUILD_HSL" OFF)
option(BUILD_SPARSE "BUILD_SPARSE" ON)

if(${BUILD_ASL})
add_subdirectory(asl_interface)
endif()

if(${BUILD_SPARSE})
add_subdirectory(sparse_utils)
endif()

if(${BUILD_HSL})
add_subdirectory(hsl_interface)
endif()


