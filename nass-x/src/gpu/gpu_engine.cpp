#include "nass_x/gpu/gpu_engine.hpp"
#include <chrono>
#include <cstring>
#include <iostream>

namespace nass_x::gpu {

GpuEngine::GpuEngine(int device_id) 
    : stats_{0, 0.0, 0.0, 0} {
    
    if (!CudaBackend::is_available()) {
        std::cerr << "Warning: No CUDA devices available. GPU engine will not be functional." << std::endl;
        return;
    }
    
    try {
        backend_ = std::make_unique<CudaBackend>(device_id);
        fft_engine_ = std::make_unique<FftCuda>(*backend_);
        main_stream_ = backend_->create_stream();
        
        std::cout << "GPU Engine initialized on device " << device_id << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize GPU engine: " << e.what() << std::endl;
    }
}

GpuEngine::~GpuEngine() {
    free_buffers();
    if (main_stream_) {
        backend_->destroy_stream(main_stream_);
    }
}

void GpuEngine::allocate_buffers(size_t max_batch_size, int fft_size) {
    // Allocate device buffers for processing
    // Simplified allocation - in production would use memory pools
    
    size_t audio_buffer_size = max_batch_size * fft_size * sizeof(float);
    size_t spectrum_buffer_size = max_batch_size * (fft_size / 2 + 1) * sizeof(std::complex<float>);
    
    void* d_audio = backend_->allocate_device(audio_buffer_size);
    void* d_spectrum = backend_->allocate_device(spectrum_buffer_size);
    
    device_buffers_.push_back(d_audio);
    device_buffers_.push_back(d_spectrum);
    
    // Allocate pinned host buffers for faster transfers
    void* h_audio = backend_->allocate_host(audio_buffer_size, true);
    void* h_spectrum = backend_->allocate_host(spectrum_buffer_size, true);
    
    host_pinned_buffers_.push_back(h_audio);
    host_pinned_buffers_.push_back(h_spectrum);
}

void GpuEngine::free_buffers() {
    for (void* buf : device_buffers_) {
        if (buf) backend_->free_device(buf);
    }
    device_buffers_.clear();
    
    for (void* buf : host_pinned_buffers_) {
        if (buf) backend_->free_host(buf);
    }
    host_pinned_buffers_.clear();
}

double GpuEngine::execute_stft(const float* audio_batch, std::complex<float>* spectrogram_batch,
                               int batch_size, int fft_size, int hop_size) {
    if (!backend_ || !fft_engine_) {
        throw std::runtime_error("GPU engine not initialized");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Calculate buffer sizes
    size_t samples_per_frame = fft_size; // Simplified: assume frame size = fft_size
    size_t total_samples = batch_size * samples_per_frame;
    size_t freq_bins = fft_size / 2 + 1;
    
    size_t audio_bytes = total_samples * sizeof(float);
    size_t spectrum_bytes = batch_size * freq_bins * sizeof(std::complex<float>);
    
    // Allocate temporary device buffers
    float* d_audio = static_cast<float*>(backend_->allocate_device(audio_bytes));
    cufftComplex* d_spectrum = reinterpret_cast<cufftComplex*>(
        backend_->allocate_device(spectrum_bytes));
    
    // Copy input to device (async)
    backend_->memcpy_host_to_device(d_audio, audio_batch, audio_bytes, main_stream_);
    
    // Apply Hann window (optional - could be pre-applied)
    // fft_engine_->apply_hann_window(d_audio, batch_size, fft_size, main_stream_);
    
    // Execute batched STFT
    fft_engine_->stft_forward(d_audio, d_spectrum, batch_size, fft_size, main_stream_);
    
    // Copy result back to host (async)
    backend_->memcpy_device_to_host(spectrogram_batch, d_spectrum, spectrum_bytes, main_stream_);
    
    // Synchronize to ensure completion
    backend_->sync_stream(main_stream_);
    
    // Free temporary buffers
    backend_->free_device(d_audio);
    backend_->free_device(d_spectrum);
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Update statistics
    stats_.total_operations++;
    stats_.bytes_transferred += audio_bytes + spectrum_bytes;
    
    // Simple moving average for latency
    stats_.avg_latency_ms = (stats_.avg_latency_ms * (stats_.total_operations - 1) + elapsed_ms) 
                            / stats_.total_operations;
    
    // Estimate GFLOPS (simplified: FFT is O(N log N))
    double flops = batch_size * (5.0 * fft_size * std::log2(fft_size)); // Approximate FFT operations
    stats_.peak_throughput_gflops = (flops / (elapsed_ms * 1e-3)) / 1e9;
    
    return elapsed_ms;
}

double GpuEngine::execute_istft(const std::complex<float>* spectrogram_batch, float* audio_batch,
                                int batch_size, int fft_size, int hop_size) {
    if (!backend_ || !fft_engine_) {
        throw std::runtime_error("GPU engine not initialized");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    size_t freq_bins = fft_size / 2 + 1;
    size_t total_samples = batch_size * fft_size;
    
    size_t spectrum_bytes = batch_size * freq_bins * sizeof(std::complex<float>);
    size_t audio_bytes = total_samples * sizeof(float);
    
    // Allocate temporary device buffers
    cufftComplex* d_spectrum = reinterpret_cast<cufftComplex*>(
        backend_->allocate_device(spectrum_bytes));
    float* d_audio = static_cast<float*>(backend_->allocate_device(audio_bytes));
    
    // Copy input to device
    backend_->memcpy_host_to_device(d_spectrum, spectrogram_batch, spectrum_bytes, main_stream_);
    
    // Execute batched iSTFT
    fft_engine_->istft_inverse(d_spectrum, d_audio, batch_size, fft_size, main_stream_);
    
    // Apply overlap-add
    // fft_engine_->overlap_add(d_audio, d_audio, batch_size, hop_size, fft_size, total_samples, main_stream_);
    
    // Copy result back to host
    backend_->memcpy_device_to_host(audio_batch, d_audio, audio_bytes, main_stream_);
    
    // Synchronize
    backend_->sync_stream(main_stream_);
    
    // Free temporary buffers
    backend_->free_device(d_spectrum);
    backend_->free_device(d_audio);
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Update statistics
    stats_.total_operations++;
    stats_.bytes_transferred += spectrum_bytes + audio_bytes;
    stats_.avg_latency_ms = (stats_.avg_latency_ms * (stats_.total_operations - 1) + elapsed_ms) 
                            / stats_.total_operations;
    
    return elapsed_ms;
}

void GpuEngine::execute_stft_async(const float* audio_batch,
                                   std::function<void(std::complex<float>*, double)> callback,
                                   int batch_size, int fft_size, int hop_size) {
    // In a full implementation, this would:
    // 1. Allocate persistent device buffers
    // 2. Launch async memcpy and FFT kernels
    // 3. Register a CUDA callback that triggers the user callback when complete
    // 4. Return immediately without blocking
    
    // For Phase 3, we'll do a simple async simulation
    std::thread([=, this]() {
        try {
            std::complex<float>* result = new std::complex<float>[batch_size * (fft_size / 2 + 1)];
            double latency = execute_stft(audio_batch, result, batch_size, fft_size, hop_size);
            callback(result, latency);
            
            // Note: Caller is responsible for freeing result in this simple implementation
            // In production, would use smart pointers or a memory pool
        } catch (const std::exception& e) {
            std::cerr << "Async STFT failed: " << e.what() << std::endl;
        }
    }).detach();
}

} // namespace nass_x::gpu
