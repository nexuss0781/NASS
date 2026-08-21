cmake_minimum_required(VERSION 3.20)
project(NASS-X VERSION 1.0.0 LANGUAGES CXX)

# C++20 Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Position independent code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Warnings as errors option
option(WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
if(WARNINGS_AS_ERRORS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

# AddressSanitizer option
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Find packages
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavformat libavcodec libavutil libswresample)

# Optional: Intel MKL
option(USE_MKL "Use Intel MKL for FFT" OFF)
if(USE_MKL)
    find_package(MKL REQUIRED)
    add_definitions(-DNASS_USE_MKL)
endif()

# Source files
set(IO_SOURCES
    src/io/audio_reader.cpp
)

set(THREADING_SOURCES
)

set(MEMORY_SOURCES
    src/memory/arena.cpp
    src/memory/buffer.cpp
)

set(MATH_SOURCES
    src/math/fft_simd.cpp
    src/math/stft.cpp
    src/math/istft.cpp
)

set(TENSOR_SOURCES
    src/tensor/tensor.cpp
    src/tensor/ops.cpp
)

set(PIPELINE_SOURCES
    src/pipeline/engine.cpp
    src/pipeline/worker.cpp
)

set(CORE_SOURCES
    src/core/types.cpp
    src/core/config.cpp
    src/core/status.cpp
)

# Main library
add_library(nass_x STATIC
    ${IO_SOURCES}
    ${THREADING_SOURCES}
    ${MEMORY_SOURCES}
    ${MATH_SOURCES}
    ${TENSOR_SOURCES}
    ${PIPELINE_SOURCES}
    ${CORE_SOURCES}
)

target_include_directories(nass_x PUBLIC ${FFMPEG_INCLUDE_DIRS})
target_link_libraries(nass_x PUBLIC Threads::Threads ${FFMPEG_LIBRARIES})

# Executable for testing
add_executable(nass_test tests/main.cpp)
target_link_libraries(nass_test PRIVATE nass_x)

# Benchmarks
option(BUILD_BENCHMARKS "Build benchmarks" ON)
if(BUILD_BENCHMARKS)
    add_executable(nass_bench benchmarks/bench_stft.cpp)
    target_link_libraries(nass_bench PRIVATE nass_x)
endif()

# Install targets
install(TARGETS nass_x
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/nass
    DESTINATION include
)

# Print configuration
message(STATUS "NASS-X Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Use MKL: ${USE_MKL}")
message(STATUS "  Build Benchmarks: ${BUILD_BENCHMARKS}")
