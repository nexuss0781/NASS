#pragma once

#include "nass_x/gpu/cuda_backend.hpp"
#include <cufft.h>
#include <complex>
#include <vector>
#include <memory>

namespace nass_x::gpu {

/**
 * @brief GPU-accelerated FFT engine using cuFFT
 * Supports batched operations for massive parallelism
 */
class FftCuda {
public:
    explicit FftCuda(CudaBackend& backend);
    ~FftCuda();

    FftCuda(const FftCuda&) = delete;
    FftCuda& operator=(const FftCuda&) = delete;

    /**
     * @brief Execute batched forward STFT on GPU
     * @param input Device pointer to input audio (batch_size × samples)
     * @param output Device pointer to output spectrum (batch_size × fft_size/2+1 × complex)
     * @param batch_size Number of independent FFTs to compute
     * @param fft_size FFT size (e.g., 512, 1024, 2048)
     * @param stream CUDA stream for async execution
     */
    void stft_forward(const float* input, cufftComplex* output, 
                      int batch_size, int fft_size, cudaStream_t stream = 0);

    /**
     * @brief Execute batched inverse iSTFT on GPU
     * @param input Device pointer to input spectrum (batch_size × fft_size/2+1 × complex)
     * @param output Device pointer to output audio (batch_size × samples)
     * @param batch_size Number of independent IFFTs to compute
     * @param fft_size FFT size
     * @param stream CUDA stream for async execution
     */
    void istft_inverse(const cufftComplex* input, float* output,
                       int batch_size, int fft_size, cudaStream_t stream = 0);

    /**
     * @brief Execute batched complex-to-complex forward FFT
     */
    void fft_complex_forward(const cufftComplex* input, cufftComplex* output,
                             int batch_size, int fft_size, cudaStream_t stream = 0);

    /**
     * @brief Execute batched complex-to-complex inverse FFT
     */
    void fft_complex_inverse(const cufftComplex* input, cufftComplex* output,
                             int batch_size, int fft_size, cudaStream_t stream = 0);

    /**
     * @brief Apply Hann window on GPU
     * @param data Device pointer to audio data
     * @param batch_size Number of frames
     * @param frame_size Size of each frame
     * @param stream CUDA stream
     */
    void apply_hann_window(float* data, int batch_size, int frame_size, cudaStream_t stream = 0);

    /**
     * @brief Overlap-add operation on GPU
     * @param input Device pointer to framed spectra
     * @param output Device pointer to reconstructed audio
     * @param batch_size Number of frames
     * @param hop_size Hop size for overlap
     * @param fft_size FFT size
     * @param output_size Size of output buffer
     * @param stream CUDA stream
     */
    void overlap_add(const float* input, float* output,
                     int batch_size, int hop_size, int fft_size, int output_size,
                     cudaStream_t stream = 0);

private:
    CudaBackend& backend_;
    
    // cuFFT plans (cached for reuse)
    cufftHandle plan_r2c_1d_;
    cufftHandle plan_c2r_1d_;
    cufftHandle plan_c2c_1d_fwd_;
    cufftHandle plan_c2c_1d_inv_;
    
    // Pre-computed Hann window on device
    float* d_hann_window_;
    size_t hann_window_size_;

    void create_plans(int fft_size);
    void destroy_plans();
    void upload_hann_window(int size);
};

/**
 * @brief RAII wrapper for cuFFT plan execution
 */
class FftPlanGuard {
public:
    explicit FftPlanGuard(FftCuda& fft, int batch_size, int fft_size, bool is_real);
    ~FftPlanGuard();
    
    cufftHandle get_plan() const { return plan_; }
    
private:
    FftCuda& fft_;
    cufftHandle plan_;
};

} // namespace nass_x::gpu
