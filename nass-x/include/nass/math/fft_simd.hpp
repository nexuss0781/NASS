#ifndef NASS_MATH_FFT_SIMD_HPP
#define NASS_MATH_FFT_SIMD_HPP

#include <cstddef>
#include <complex>
#include "nass/core/types.hpp"
#include "nass/memory/buffer.hpp"

namespace nass_x::tensor {

// SIMD level detection
enum class SIMDLevel {
    NONE = 0,
    SSE = 1,
    AVX2 = 2,
    AVX512 = 3
};

// Runtime SIMD level detection
SIMDLevel detect_simd_level();

// Get current SIMD level as string
const char* simd_level_to_string(SIMDLevel level);

// FFT implementation with SIMD optimization
class FFTSimd {
public:
    explicit FFTSimd(size_t fft_size);
    ~FFTSimd();
    
    // Non-copyable
    FFTSimd(const FFTSimd&) = delete;
    FFTSimd& operator=(const FFTSimd&) = delete;
    
    // Forward FFT (time -> frequency)
    void forward(const Float32* input, ComplexFloat* output);
    void forward(const Float64* input, ComplexDouble* output);
    
    // Inverse FFT (frequency -> time)
    void inverse(const ComplexFloat* input, Float32* output);
    void inverse(const ComplexDouble* input, Float64* output);
    
    // Get FFT size
    size_t size() const { return fft_size_; }
    
    // Check if initialization succeeded
    bool is_valid() const { return valid_; }
    
private:
    size_t fft_size_;
    bool valid_;
    void* impl_data_;  // Opaque pointer to backend-specific data
    
    // Backend implementations
    void init_kissfft();
    void forward_kissfft(const Float32* input, ComplexFloat* output);
    void inverse_kissfft(const ComplexFloat* input, Float32* output);
    
#ifdef NASS_USE_MKL
    void init_mkl();
    void forward_mkl(const Float32* input, ComplexFloat* output);
    void inverse_mkl(const ComplexFloat* input, Float32* output);
#endif
};

// Window functions
class WindowFunction {
public:
    enum class Type {
        HANN,
        HAMMING,
        BLACKMAN,
        RECTANGULAR
    };
    
    // Generate window
    static Buffer<Float32> generate(Type type, size_t size, bool normalized = true);
    
    // Apply window to buffer (in-place)
    static void apply(Float32* data, const Float32* window, size_t size);
    
    // Pre-computed Hann window for STFT (most common)
    static const Float32* hann_window(size_t size);
    
private:
    static Buffer<Float32> generate_hann(size_t size, bool normalized);
    static Buffer<Float32> generate_hamming(size_t size, bool normalized);
    static Buffer<Float32> generate_blackman(size_t size, bool normalized);
};

// Overlap-Add helper for iSTFT
class OverlapAdd {
public:
    explicit OverlapAdd(size_t hop_size, size_t win_size);
    
    // Process one frame and add to output buffer
    void add_frame(const Float32* frame_data, Float32* output, size_t frame_idx);
    
    // Finalize (handle edge cases)
    void finalize(Float32* output, size_t total_samples);
    
    // Reset internal state
    void reset();
    
private:
    size_t hop_size_;
    size_t win_size_;
    size_t overlap_size_;
    Buffer<Float32> normalization_buffer_;
};

} // namespace nass_x::tensor

#endif // NASS_MATH_FFT_SIMD_HPP
