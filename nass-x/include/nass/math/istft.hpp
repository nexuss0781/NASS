#ifndef NASS_MATH_ISTFT_HPP
#define NASS_MATH_ISTFT_HPP

#include <cstddef>
#include <memory>
#include "nass/core/types.hpp"
#include "nass/core/status.hpp"
#include "nass/memory/buffer.hpp"
#include "nass/math/fft_simd.hpp"
#include "nass/math/stft.hpp"

namespace nass_x::tensor {

// Inverse Short-Time Fourier Transform engine
class ISTFT {
public:
    explicit ISTFT(const STFTConfig& config);
    ~ISTFT();
    
    // Non-copyable
    ISTFT(const ISTFT&) = delete;
    ISTFT& operator=(const ISTFT&) = delete;
    
    // Reconstruct audio from spectrogram
    Result<FloatBuffer> inverse(const ComplexFloat* spectrogram, size_t num_frames);
    
    // Inverse with pre-allocated output buffer
    Status inverse(const ComplexFloat* spectrogram, size_t num_frames,
                   Float32* output, size_t output_size);
    
    // Get configuration
    const STFTConfig& config() const { return config_; }
    
    // Check if initialized correctly
    bool is_valid() const { return valid_; }
    
    // Get reconstruction error (for testing perfect reconstruction)
    Float32 reconstruction_error() const { return reconstruction_error_; }
    
private:
    STFTConfig config_;
    bool valid_;
    Float32 reconstruction_error_;
    
    std::unique_ptr<FFTSimd> fft_;
    Buffer<Float32> window_;
    Buffer<ComplexFloat> ifft_buffer_;
    Buffer<Float32> overlap_buffer_;
    std::unique_ptr<OverlapAdd> overlap_add_;
    
    // Internal processing
    void apply_window(ComplexFloat* spectrum);
    size_t calculate_output_size(size_t num_frames) const;
};

// Perfect reconstruction validator
class ReconstructionValidator {
public:
    explicit ReconstructionValidator(const STFTConfig& config);
    
    // Test round-trip: STFT -> iSTFT == original
    Float32 validate_round_trip(const Float32* audio, size_t num_samples);
    
    // Check if error is within tolerance
    bool is_perfect(Float32 error, Float32 tolerance = 1e-6) const;
    
private:
    STFTConfig config_;
    std::unique_ptr<STFT> stft_;
    std::unique_ptr<ISTFT> istft_;
    Buffer<Float32> temp_buffer_;
};

} // namespace nass_x::tensor

#endif // NASS_MATH_ISTFT_HPP
