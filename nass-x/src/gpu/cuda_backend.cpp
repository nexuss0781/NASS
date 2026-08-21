#include "nass_x/gpu/cuda_backend.hpp"

namespace nass_x::gpu {

bool CudaBackend::is_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

int CudaBackend::get_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return 0;
    return count;
}

CudaBackend::CudaBackend(int device_id) 
    : device_id_(device_id), total_memory_(0) {
    initialize();
}

CudaBackend::~CudaBackend() {
    // Destroy all streams
    for (auto stream : streams_) {
        cudaStreamDestroy(stream);
    }
}

CudaBackend::CudaBackend(CudaBackend&& other) noexcept
    : device_id_(other.device_id_),
      device_prop_(other.device_prop_),
      total_memory_(other.total_memory_),
      streams_(std::move(other.streams_)) {
    other.device_id_ = -1;
    other.total_memory_ = 0;
}

CudaBackend& CudaBackend::operator=(CudaBackend&& other) noexcept {
    if (this != &other) {
        // Clean up current resources
        for (auto stream : streams_) {
            cudaStreamDestroy(stream);
        }
        
        device_id_ = other.device_id_;
        device_prop_ = other.device_prop_;
        total_memory_ = other.total_memory_;
        streams_ = std::move(other.streams_);
        
        other.device_id_ = -1;
        other.total_memory_ = 0;
    }
    return *this;
}

void CudaBackend::initialize() {
    CUDA_CHECK(cudaSetDevice(device_id_));
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop_, device_id_));
    
    size_t free_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_memory_));
    
    std::cout << "CUDA Device: " << device_prop_.name << std::endl;
    std::cout << "  Compute Capability: " << device_prop_.major << "." << device_prop_.minor << std::endl;
    std::cout << "  Total Memory: " << (total_memory_ / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Multiprocessors: " << device_prop_.multiProcessorCount << std::endl;
}

size_t CudaBackend::get_free_memory() const {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return free_mem;
}

cudaStream_t CudaBackend::create_stream() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    streams_.push_back(stream);
    return stream;
}

void CudaBackend::destroy_stream(cudaStream_t stream) {
    auto it = std::find(streams_.begin(), streams_.end(), stream);
    if (it != streams_.end()) {
        CUDA_CHECK(cudaStreamDestroy(stream));
        streams_.erase(it);
    }
}

void CudaBackend::sync_stream(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void CudaBackend::sync_device() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

void* CudaBackend::allocate_host(size_t size, bool pinned) {
    void* ptr;
    if (pinned) {
        CUDA_CHECK(cudaMallocHost(&ptr, size));
    } else {
        ptr = malloc(size);
    }
    return ptr;
}

void CudaBackend::free_host(void* ptr) {
    // Check if it's pinned memory by attempting cudaFreeHost
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaSuccess) {
        free(ptr);
    }
}

void* CudaBackend::allocate_device(size_t size) {
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void CudaBackend::free_device(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

void CudaBackend::memcpy_host_to_device(void* dst, const void* src, size_t size, cudaStream_t stream) {
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    }
}

void CudaBackend::memcpy_device_to_host(void* dst, const void* src, size_t size, cudaStream_t stream) {
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    }
}

void CudaBackend::memcpy_device_to_device(void* dst, const void* src, size_t size, cudaStream_t stream) {
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    }
}

// CudaStream implementation
CudaStream::CudaStream(CudaBackend& backend) 
    : backend_(backend), stream_(backend.create_stream()) {}

CudaStream::~CudaStream() {
    backend_.destroy_stream(stream_);
}

} // namespace nass_x::tensor_x::gpu
