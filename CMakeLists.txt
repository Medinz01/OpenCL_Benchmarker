cmake_minimum_required(VERSION 3.10)
project(OpenCL_Benchmarker)

set(CMAKE_CXX_STANDARD 17)

# 1. Fix Deprecation Warnings
# We tell the compiler we are targeting OpenCL 1.2 (compatible with everything)
add_definitions(-DCL_TARGET_OPENCL_VERSION=120)

find_package(OpenCL REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCL_INCLUDE_DIRS})

add_executable(Benchmarker src/main.cpp)
target_link_libraries(Benchmarker ${OpenCL_LIBRARIES})

# 2. FIX THE COPY PATH
# Old: Copied kernels -> build/kernels (resulted in build/kernels/kernels/filters.cl)
# New: Copies kernels -> build (results in build/kernels/filters.cl)
file(COPY ${CMAKE_SOURCE_DIR}/kernels DESTINATION ${CMAKE_BINARY_DIR})