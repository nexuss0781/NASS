#pragma once

#include <cuda_runtime.h>
#include <cufftXt.h>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>

namespace nass_x::gpu {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + \
                cudaGetErrorString(err) + " at " + __FILE__ + ":" + \
                std::to_string(__LINE__)); \
        } \
    } while(0)

// cuFFT error checking macro
#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            throw std::runtime_error(std::string("cuFFT error: " + \
                std::to_string(err) + " at " + __FILE__ + ":" + \
                std::to_string(__LINE__))); \
        } \
    } while(0)

/**
 * @brief GPU Device Manager - handles initialization, streams, and memory pools
 */
class CudaBackend {
public:
    explicit CudaBackend(int device_id = 0);
    ~CudaBackend();

    // Non-copyable, movable
    CudaBackend(const CudaBackend&) = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;
    CudaBackend(CudaBackend&& other) noexcept;
    CudaBackend& operator=(CudaBackend&& other) noexcept;

    // Device info
    int get_device_id() const { return device_id_; }
    cudaDeviceProp get_device_prop() const { return device_prop_; }
    size_t get_total_memory() const { return total_memory_; }
    size_t get_free_memory() const;
    
    // Stream management
    cudaStream_t create_stream();
    void destroy_stream(cudaStream_t stream);
    void sync_stream(cudaStream_t stream);
    void sync_device();

    // Memory management (pinned host memory + device memory)
    void* allocate_host(size_t size, bool pinned = true);
    void free_host(void* ptr);
    
    void* allocate_device(size_t size);
    void free_device(void* ptr);
    
    // Async memory transfers
    void memcpy_host_to_device(void* dst, const void* src, size_t size, cudaStream_t stream = 0);
    void memcpy_device_to_host(void* dst, const void* src, size_t size, cudaStream_t stream = 0);
    void memcpy_device_to_device(void* dst, const void* src, size_t size, cudaStream_t stream = 0);

    // Check if GPU is available
    static bool is_available();
    static int get_device_count();

private:
    int device_id_;
    cudaDeviceProp device_prop_;
    size_t total_memory_;
    std::vector<cudaStream_t> streams_;
    
    void initialize();
};

/**
 * @brief RAII wrapper for CUDA streams
 */
class CudaStream {
public:
    explicit CudaStream(CudaBackend& backend);
    ~CudaStream();
    
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    cudaStream_t get() const { return stream_; }
    void sync() { backend_.sync_stream(stream_); }

private:
    CudaBackend& backend_;
    cudaStream_t stream_;
};

} // namespace nass_x::tensor_x::gpu
