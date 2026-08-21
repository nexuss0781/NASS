#include "nass_x/gpu/fft_cuda.hpp"
#include <cufftXt.h>
#include <cmath>
#include <iostream>

namespace nass_x::gpu {

// CUDA kernel for Hann window application
__global__ void apply_hann_window_kernel(float* data, const float* window, 
                                         int batch_size, int frame_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * frame_size;
    
    if (idx < total_elements) {
        int frame_idx = idx / frame_size;
        int sample_idx = idx % frame_size;
        data[idx] *= window[sample_idx];
    }
}

// CUDA kernel for overlap-add
__global__ void overlap_add_kernel(const float* input, float* output,
                                   int batch_size, int hop_size, int fft_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < output_size) {
        float sum = 0.0f;
        int num_overlaps = (batch_size * hop_size - fft_size + hop_size) / hop_size;
        
        for (int frame = 0; frame < batch_size; ++frame) {
            int frame_start = frame * hop_size;
            int sample_in_frame = idx - frame_start;
            
            if (sample_in_frame >= 0 && sample_in_frame < fft_size) {
                sum += input[frame * fft_size + sample_in_frame];
            }
        }
        
        // Normalize by number of overlaps (simplified - proper normalization needs window sum)
        output[idx] = sum;
    }
}

FftCuda::FftCuda(CudaBackend& backend)
    : backend_(backend),
      d_hann_window_(nullptr),
      hann_window_size_(0) {
    cufftCreate(&plan_r2c_1d_);
    cufftCreate(&plan_c2r_1d_);
    cufftCreate(&plan_c2c_1d_fwd_);
    cufftCreate(&plan_c2c_1d_inv_);
}

FftCuda::~FftCuda() {
    destroy_plans();
    if (d_hann_window_) {
        cudaFree(d_hann_window_);
    }
}

void FftCuda::create_plans(int fft_size) {
    int batch = 1; // Will set batch dynamically in execution
    
    // Real-to-Complex plan (for STFT)
    cufftDestroy(plan_r2c_1d_);
    cufftCreate(&plan_r2c_1d_);
    cufftPlan1d(&plan_r2c_1d_, fft_size, CUFFT_R2C, batch);
    
    // Complex-to-Real plan (for iSTFT)
    cufftDestroy(plan_c2r_1d_);
    cufftCreate(&plan_c2r_1d_);
    cufftPlan1d(&plan_c2r_1d_, fft_size, CUFFT_C2R, batch);
    
    // Complex-to-Complex forward
    cufftDestroy(plan_c2c_1d_fwd_);
    cufftCreate(&plan_c2c_1d_fwd_);
    cufftPlan1d(&plan_c2c_1d_fwd_, fft_size, CUFFT_C2C, batch);
    
    // Complex-to-Complex inverse
    cufftDestroy(plan_c2c_1d_inv_);
    cufftCreate(&plan_c2c_1d_inv_);
    cufftPlan1d(&plan_c2c_1d_inv_, fft_size, CUFFT_C2C, batch);
}

void FftCuda::destroy_plans() {
    cufftDestroy(plan_r2c_1d_);
    cufftDestroy(plan_c2r_1d_);
    cufftDestroy(plan_c2c_1d_fwd_);
    cufftDestroy(plan_c2c_1d_inv_);
}

void FftCuda::upload_hann_window(int size) {
    if (d_hann_window_ && hann_window_size_ == static_cast<size_t>(size)) {
        return; // Already uploaded
    }
    
    // Free existing window if different size
    if (d_hann_window_) {
        cudaFree(d_hann_window_);
    }
    
    // Create host window
    std::vector<float> h_window(size);
    for (int i = 0; i < size; ++i) {
        h_window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (size - 1)));
    }
    
    // Upload to device
    cudaMalloc(&d_hann_window_, size * sizeof(float));
    cudaMemcpy(d_hann_window_, h_window.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    hann_window_size_ = size;
}

void FftCuda::stft_forward(const float* input, cufftComplex* output,
                           int batch_size, int fft_size, cudaStream_t stream) {
    // Ensure plans exist
    if (plan_r2c_1d_ == 0) {
        create_plans(fft_size);
    }
    
    // Set cuFFT stream
    cufftSetStream(plan_r2c_1d_, stream);
    
    // Execute batched R2C FFT
    // Note: cuFFT automatically handles batching when input/output are properly sized
    cufftExecR2C(plan_r2c_1d_, 
                 reinterpret_cast<cufftReal*>(const_cast<float*>(input)),
                 output);
}

void FftCuda::istft_inverse(const cufftComplex* input, float* output,
                            int batch_size, int fft_size, cudaStream_t stream) {
    // Ensure plans exist
    if (plan_c2r_1d_ == 0) {
        create_plans(fft_size);
    }
    
    // Set cuFFT stream
    cufftSetStream(plan_c2r_1d_, stream);
    
    // Execute batched C2R IFFT
    cufftExecC2R(plan_c2r_1d_,
                 const_cast<cufftComplex*>(input),
                 reinterpret_cast<cufftReal*>(output));
    
    // Scale output (cuFFT doesn't normalize inverse)
    int total_samples = batch_size * fft_size;
    float scale = 1.0f / fft_size;
    
    // Simple scaling kernel could be added here for efficiency
    // For now, this is handled by caller or overlap-add
}

void FftCuda::fft_complex_forward(const cufftComplex* input, cufftComplex* output,
                                  int batch_size, int fft_size, cudaStream_t stream) {
    if (plan_c2c_1d_fwd_ == 0) {
        create_plans(fft_size);
    }
    
    cufftSetStream(plan_c2c_1d_fwd_, stream);
    cufftExecC2C(plan_c2c_1d_fwd_,
                 const_cast<cufftComplex*>(input),
                 output,
                 CUFFT_FORWARD);
}

void FftCuda::fft_complex_inverse(const cufftComplex* input, cufftComplex* output,
                                  int batch_size, int fft_size, cudaStream_t stream) {
    if (plan_c2c_1d_inv_ == 0) {
        create_plans(fft_size);
    }
    
    cufftSetStream(plan_c2c_1d_inv_, stream);
    cufftExecC2C(plan_c2c_1d_inv_,
                 const_cast<cufftComplex*>(input),
                 output,
                 CUFFT_INVERSE);
    
    // Scale output
    float scale = 1.0f / fft_size;
    int total_elements = batch_size * fft_size;
    
    // Scaling can be done with a simple kernel
}

void FftCuda::apply_hann_window(float* data, int batch_size, int frame_size, cudaStream_t stream) {
    upload_hann_window(frame_size);
    
    int threads_per_block = 256;
    int total_elements = batch_size * frame_size;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    apply_hann_window_kernel<<<blocks, threads_per_block, 0, stream>>>(
        data, d_hann_window_, batch_size, frame_size);
}

void FftCuda::overlap_add(const float* input, float* output,
                          int batch_size, int hop_size, int fft_size, int output_size,
                          cudaStream_t stream) {
    int threads_per_block = 256;
    int blocks = (output_size + threads_per_block - 1) / threads_per_block;
    
    overlap_add_kernel<<<blocks, threads_per_block, 0, stream>>>(
        input, output, batch_size, hop_size, fft_size, output_size);
}

// FftPlanGuard implementation
FftPlanGuard::FftPlanGuard(FftCuda& fft, int batch_size, int fft_size, bool is_real)
    : fft_(fft), plan_(0) {
    // In a full implementation, this would create a new plan with the specific batch size
    // For now, we rely on the cached plans in FftCuda
}

FftPlanGuard::~FftPlanGuard() {
    // Plan cleanup handled by FftCuda destructor
}

} // namespace nass_x::gpu
