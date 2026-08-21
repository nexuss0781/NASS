#ifndef NASS_MATH_STFT_HPP
#define NASS_MATH_STFT_HPP

#include <cstddef>
#include <memory>
#include <vector>
#include "nass/core/types.hpp"
#include "nass/core/status.hpp"
#include "nass/memory/buffer.hpp"
#include "nass/math/fft_simd.hpp"

namespace nass_x::tensor {

// Short-Time Fourier Transform engine
class STFT {
public:
    explicit STFT(const STFTConfig& config);
    ~STFT();
    
    // Non-copyable
    STFT(const STFT&) = delete;
    STFT& operator=(const STFT&) = delete;
    
    // Process audio buffer to spectrogram
    Result<Spectrogram> transform(const Float32* audio, size_t num_samples);
    
    // Transform with pre-allocated output buffer
    Status transform(const Float32* audio, size_t num_samples, 
                     ComplexFloat* spectrogram, size_t num_frames);
    
    // Get configuration
    const STFTConfig& config() const { return config_; }
    
    // Get number of frequency bins
    size_t num_bins() const { return config_.num_freq_bins(); }
    
    // Get hop size
    size_t hop_size() const { return config_.hop_size; }
    
    // Get FFT size
    size_t fft_size() const { return config_.fft_size; }
    
    // Check if initialized correctly
    bool is_valid() const { return valid_; }
    
private:
    STFTConfig config_;
    bool valid_;
    
    std::unique_ptr<FFTSimd> fft_;
    Buffer<Float32> window_;
    Buffer<ComplexFloat> fft_buffer_;
    Buffer<Float32> frame_buffer_;
    
    // Internal processing
    void apply_window(Float32* frame);
    size_t calculate_num_frames(size_t num_samples) const;
};

// Batch STFT for parallel processing
class BatchSTFT {
public:
    explicit BatchSTFT(const STFTConfig& config, size_t batch_size = 1);
    ~BatchSTFT();
    
    // Process multiple audio chunks in parallel
    Result<std::vector<Spectrogram>> transform_batch(
        const std::vector<const Float32*>& audio_chunks,
        const std::vector<size_t>& sample_counts);
    
    size_t batch_size() const { return batch_size_; }
    
private:
    STFTConfig config_;
    size_t batch_size_;
    std::vector<std::unique_ptr<STFT>> engines_;
};

} // namespace nass_x::tensor

#endif // NASS_MATH_STFT_HPP
