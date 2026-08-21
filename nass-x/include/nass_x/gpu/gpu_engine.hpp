#pragma once

#include "nass_x/gpu/cuda_backend.hpp"
#include "nass_x/gpu/fft_cuda.hpp"
#include "nass_x/tensor/tensor.hpp"
#include <vector>
#include <memory>
#include <functional>

namespace nass_x::gpu {

/**
 * @brief GPU Pipeline Engine - orchestrates async STFT/iSTFT processing
 * Manages kernel launches, stream synchronization, and batch optimization
 */
class GpuEngine {
public:
    explicit GpuEngine(int device_id = 0);
    ~GpuEngine();

    GpuEngine(const GpuEngine&) = delete;
    GpuEngine& operator=(const GpuEngine&) = delete;

    /**
     * @brief Execute batched STFT on GPU
     * @param audio_batch Host pointer to audio samples [batch_size × samples]
     * @param spectrogram_batch Host pointer to output spectrum [batch_size × freq_bins × complex]
     * @param batch_size Number of audio segments to process
     * @param fft_size FFT size (512, 1024, 2048, etc.)
     * @param hop_size Hop size for framing
     * @return Processing time in milliseconds
     */
    double execute_stft(const float* audio_batch, std::complex<float>* spectrogram_batch,
                        int batch_size, int fft_size, int hop_size);

    /**
     * @brief Execute batched iSTFT on GPU
     * @param spectrogram_batch Host pointer to input spectrum
     * @param audio_batch Host pointer to reconstructed audio
     * @param batch_size Number of spectrograms to process
     * @param fft_size FFT size
     * @param hop_size Hop size for overlap-add
     * @return Processing time in milliseconds
     */
    double execute_istft(const std::complex<float>* spectrogram_batch, float* audio_batch,
                         int batch_size, int fft_size, int hop_size);

    /**
     * @brief Async STFT with callback
     * @param audio_batch Host pointer to audio
     * @param callback Function called when complete with result pointer
     * @param batch_size Number of segments
     * @param fft_size FFT size
     * @param hop_size Hop size
     */
    void execute_stft_async(const float* audio_batch,
                            std::function<void(std::complex<float>*, double)> callback,
                            int batch_size, int fft_size, int hop_size);

    /**
     * @brief Get performance statistics
     */
    struct Stats {
        size_t total_operations;
        double avg_latency_ms;
        double peak_throughput_gflops;
        size_t bytes_transferred;
    };
    
    Stats get_stats() const { return stats_; }

    /**
     * @brief Check if GPU is available and initialized
     */
    bool is_ready() const { return backend_ != nullptr; }

    /**
     * @brief Get CUDA backend reference
     */
    CudaBackend& get_backend() { return *backend_; }

private:
    std::unique_ptr<CudaBackend> backend_;
    std::unique_ptr<FftCuda> fft_engine_;
    cudaStream_t main_stream_;
    
    Stats stats_;
    
    // Device memory pools (simplified for Phase 3)
    std::vector<void*> device_buffers_;
    std::vector<void*> host_pinned_buffers_;
    
    void allocate_buffers(size_t max_batch_size, int fft_size);
    void free_buffers();
};

} // namespace nass_x::tensor_x::gpu
