cmake_minimum_required(VERSION 3.10)
project(OpenCL_Benchmarker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCL
find_package(OpenCL REQUIRED)

# Include directories
include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Original benchmark executable
add_executable(Benchmarker 
    src/main.cpp
)

target_link_libraries(Benchmarker ${OpenCL_LIBRARIES})

# Comprehensive benchmark executable (NEW)
add_executable(ComprehensiveBenchmark
    src/comprehensive_benchmark.cpp
)

target_link_libraries(ComprehensiveBenchmark ${OpenCL_LIBRARIES})

# Copy kernel files to build directory
file(COPY ${CMAKE_SOURCE_DIR}/kernels DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/test_images DESTINATION ${CMAKE_BINARY_DIR})

# Set output directory
set_target_properties(Benchmarker ComprehensiveBenchmark
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release
)

# MSVC specific settings
if(MSVC)
    target_compile_options(Benchmarker PRIVATE /W3)
    target_compile_options(ComprehensiveBenchmark PRIVATE /W3)
endif()