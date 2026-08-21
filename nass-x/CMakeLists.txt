cmake_minimum_required(VERSION 3.20)

# Check for CUDA first before enabling it
set(USE_CUDA_INIT OFF)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    set(USE_CUDA_INIT ON)
endif()

project(NASS-X VERSION 1.0.0 LANGUAGES CXX)

# Optionally add CUDA if available
if(USE_CUDA_INIT)
    enable_language(CUDA)
endif()

# C++20 Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA Standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

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

# Optional: CUDA/cuFFT (Phase 3)
option(USE_CUDA "Use CUDA for GPU acceleration" ON)
if(USE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
        add_definitions(-DNASS_USE_CUDA)
        
        # Set CUDA architectures
        set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86" CACHE STRING "CUDA architectures")
    else()
        message(WARNING "CUDA not found, GPU acceleration disabled")
        set(USE_CUDA OFF)
    endif()
endif()

# Optional: TensorRT (Phase 4 - AGI)
option(USE_TENSORRT "Use NVIDIA TensorRT for AI inference" OFF)
if(USE_TENSORRT)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS /usr/include /usr/local/include)
    find_library(TENSORRT_LIBRARY nvinfer PATHS /usr/lib /usr/local/lib)
    find_library(TENSORRT_PARSER_LIBRARY nvonnxparser PATHS /usr/lib /usr/local/lib)
    
    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
        message(STATUS "TensorRT found")
        add_definitions(-DNASS_X_USE_TENSORRT)
        include_directories(${TENSORRT_INCLUDE_DIR})
    else()
        message(WARNING "TensorRT not found, AI inference limited to ONNX Runtime")
        set(USE_TENSORRT OFF)
    endif()
endif()

# Optional: ONNX Runtime (Phase 4 - AGI)
option(USE_ONNX "Use ONNX Runtime for AI inference" OFF)
if(USE_ONNX)
    find_path(ONNX_INCLUDE_DIR onnxruntime_cxx_api.h PATHS /usr/include/onnxruntime /usr/local/include/onnxruntime)
    find_library(ONNX_LIBRARY onnxruntime PATHS /usr/lib /usr/local/lib)
    
    if(ONNX_INCLUDE_DIR AND ONNX_LIBRARY)
        message(STATUS "ONNX Runtime found")
        add_definitions(-DNASS_X_USE_ONNX)
        include_directories(${ONNX_INCLUDE_DIR})
    else()
        # Try pkg-config
        pkg_check_modules(ONNXRuntime QUIET onnxruntime)
        if(ONNXRuntime_FOUND)
            message(STATUS "ONNX Runtime found via pkg-config")
            add_definitions(-DNASS_X_USE_ONNX)
            include_directories(${ONNXRuntime_INCLUDE_DIRS})
        else()
            message(WARNING "ONNX Runtime not found, AI features disabled")
            set(USE_ONNX OFF)
        endif()
    endif()
endif()

# Source files
set(IO_SOURCES
    src/io/audio_reader.cpp
    src/io/audio_writer.cpp
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

# GPU sources (Phase 3 - CUDA)
set(GPU_SOURCES
)
if(USE_CUDA)
    set(GPU_SOURCES
        src/gpu/cuda_backend.cpp
        src/gpu/fft_cuda.cu
        src/gpu/gpu_engine.cpp
    )
endif()

# AI sources (Phase 4 - AGI Neural Integration)
set(AI_SOURCES
    src/ai/neural_engine.cpp
    src/ai/audio_processor.cpp
)
if(USE_TENSORRT)
    list(APPEND AI_SOURCES src/ai/tensorrt_engine.cpp)
endif()
if(USE_ONNX)
    list(APPEND AI_SOURCES src/ai/onnx_engine.cpp)
endif()

# Main library
add_library(nass_x STATIC
    ${IO_SOURCES}
    ${THREADING_SOURCES}
    ${MEMORY_SOURCES}
    ${MATH_SOURCES}
    ${TENSOR_SOURCES}
    ${PIPELINE_SOURCES}
    ${CORE_SOURCES}
    ${GPU_SOURCES}
    ${AI_SOURCES}
)

target_include_directories(nass_x PUBLIC ${FFMPEG_INCLUDE_DIRS})
target_link_libraries(nass_x PUBLIC Threads::Threads ${FFMPEG_LIBRARIES})

# CUDA linkage (Phase 3)
if(USE_CUDA AND CUDAToolkit_FOUND)
    target_link_libraries(nass_x PUBLIC CUDA::cudart CUDA::cufft)
    set_target_properties(nass_x PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# TensorRT linkage (Phase 4)
if(USE_TENSORRT AND TENSORRT_LIBRARY)
    target_link_libraries(nass_x PRIVATE ${TENSORRT_LIBRARY} ${TENSORRT_PARSER_LIBRARY})
endif()

# ONNX Runtime linkage (Phase 4)
if(USE_ONNX AND ONNX_LIBRARY)
    target_link_libraries(nass_x PRIVATE ${ONNX_LIBRARY})
elseif(USE_ONNX AND ONNXRuntime_FOUND)
    target_link_libraries(nass_x PRIVATE ${ONNXRuntime_LIBRARIES})
endif()

# Executable for testing
add_executable(nass_test tests/main.cpp)
target_link_libraries(nass_test PRIVATE nass_x)

# Benchmarks
option(BUILD_BENCHMARKS "Build benchmarks" ON)
if(BUILD_BENCHMARKS)
    add_executable(nass_bench benchmarks/bench_stft.cpp)
    target_link_libraries(nass_bench PRIVATE nass_x)
    
    # GPU benchmark (Phase 3)
    if(USE_CUDA AND CUDAToolkit_FOUND)
        add_executable(nass_gpu_bench benchmarks/bench_gpu_stft.cpp)
        target_link_libraries(nass_gpu_bench PRIVATE nass_x)
        set_target_properties(nass_gpu_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()
endif()

# Examples
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_executable(ai_denoiser examples/ai_denoiser.cpp)
    target_link_libraries(ai_denoiser PRIVATE nass_x)
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
message(STATUS "  Use CUDA: ${USE_CUDA}")
if(USE_CUDA AND CUDAToolkit_FOUND)
    message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
    message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  Use TensorRT: ${USE_TENSORRT}")
message(STATUS "  Use ONNX Runtime: ${USE_ONNX}")
message(STATUS "  Build Benchmarks: ${BUILD_BENCHMARKS}")
